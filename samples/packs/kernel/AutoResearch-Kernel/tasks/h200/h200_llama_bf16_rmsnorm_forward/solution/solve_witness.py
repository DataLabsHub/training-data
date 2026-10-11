import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CPP = r"""
#include <torch/extension.h>
torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor w, int64_t kind);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm_cuda);
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void rmsnorm_packed(const __nv_bfloat16* __restrict__ x,
                    const __nv_bfloat16* __restrict__ w,
                    __nv_bfloat16* __restrict__ y, int H, float inv_H) {
  constexpr int PAIRS = 8;
  constexpr int WARPS = BLOCK / 32;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_base = blockIdx.x * H;
  __nv_bfloat162 xv[PAIRS];
  float ss = 0.0f;

  #pragma unroll
  for (int i = 0; i < PAIRS; ++i) {
    int j = 2 * (tid + i * BLOCK);
    __nv_bfloat162 v = __float2bfloat162_rn(0.0f);
    if (j + 1 < H)
      v = *reinterpret_cast<const __nv_bfloat162*>(x + row_base + j);
    xv[i] = v;
    float2 f = __bfloat1622float2(v);
    ss = fmaf(f.x, f.x, ss);
    ss = fmaf(f.y, f.y, ss);
  }

  #pragma unroll
  for (int d = 16; d; d >>= 1)
    ss += __shfl_down_sync(0xffffffff, ss, d);
  __shared__ float sums[WARPS];
  if (lane == 0) sums[warp] = ss;
  __syncthreads();

  float total = lane < WARPS ? sums[lane] : 0.0f;
  #pragma unroll
  for (int d = 16; d; d >>= 1)
    total += __shfl_down_sync(0xffffffff, total, d);
  float rstd = 0.0f;
  if (lane == 0) rstd = rsqrtf(total * inv_H + 1.0e-6f);
  rstd = __shfl_sync(0xffffffff, rstd, 0);

  #pragma unroll
  for (int i = 0; i < PAIRS; ++i) {
    int j = 2 * (tid + i * BLOCK);
    if (j + 1 < H) {
      float2 f = __bfloat1622float2(xv[i]);
      __nv_bfloat162 n = __floats2bfloat162_rn(f.x * rstd, f.y * rstd);
      __nv_bfloat162 g = *reinterpret_cast<const __nv_bfloat162*>(w + j);
      *reinterpret_cast<__nv_bfloat162*>(y + row_base + j) = __hmul2(n, g);
    }
  }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor w, int64_t kind) {
  torch::Tensor y = torch::empty_like(x);
  const int T = x.size(0), H = x.size(1);
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
  auto yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float inv_H = 1.0f / float(H);
  if (kind == 0) rmsnorm_packed<128><<<T, 128, 0, stream>>>(xp, wp, yp, H, inv_H);
  else if (kind == 1) rmsnorm_packed<256><<<T, 256, 0, stream>>>(xp, wp, yp, H, inv_H);
  else rmsnorm_packed<512><<<T, 512, 0, stream>>>(xp, wp, yp, H, inv_H);
  return y;
}
"""

_ext = load_inline(
    name="kf_h200_rmsnorm_packed_v2",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=None,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        hidden = x.shape[1]
        kind = 0 if hidden <= 2048 else (1 if hidden <= 4096 else 2)
        return _ext.rmsnorm(x, weight, kind)

