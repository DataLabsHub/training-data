import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

struct WelfordData { float mean, m2; int n; float nf; };

__device__ __forceinline__ WelfordData update(WelfordData a, float x) {
  int new_n = a.n + 1;
  float new_nf = static_cast<float>(new_n);
  float delta = x - a.mean;
  float new_mean = a.mean + delta / new_nf;
  float new_delta = x - new_mean;
  return {new_mean, a.m2 + delta * new_delta, new_n, new_nf};
}

__device__ __forceinline__ WelfordData combine(WelfordData a, WelfordData b) {
  if (a.nf == 0.0f) return b;
  if (b.nf == 0.0f) return a;
  float delta = b.mean - a.mean;
  float new_count = a.nf + b.nf;
  float nb_over_n = b.nf / new_count;
  return {a.mean + delta * nb_over_n,
          a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
          -1, new_count};
}

__device__ __forceinline__ WelfordData warp_reduce(WelfordData value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    WelfordData other{__shfl_down_sync(0xffffffffu, value.mean, offset),
                      __shfl_down_sync(0xffffffffu, value.m2, offset),
                      __shfl_down_sync(0xffffffffu, value.n, offset),
                      __shfl_down_sync(0xffffffffu, value.nf, offset)};
    value = combine(value, other);
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ float select_quarter_sum(
    float q0, float q1, float q2, float q3, int source, int quarter) {
  float r0 = __shfl_sync(0xffffffffu, q0, source);
  float r1 = __shfl_sync(0xffffffffu, q1, source);
  float r2 = __shfl_sync(0xffffffffu, q2, source);
  float r3 = __shfl_sync(0xffffffffu, q3, source);
  return quarter == 0 ? r0 : (quarter == 1 ? r1 : (quarter == 2 ? r2 : r3));
}

__global__ __launch_bounds__(512, 4)
void adaln_kernel(const __nv_bfloat16* __restrict__ x,
                  const __nv_bfloat16* __restrict__ scale,
                  const __nv_bfloat16* __restrict__ shift,
                  __nv_bfloat16* __restrict__ out,
                  float* __restrict__ row_means,
                  float* __restrict__ row_variances,
                  int rows, int rows_per_batch, int dim,
                  int parameter_batch_stride, float inv_dim) {
  int lane = threadIdx.x;
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  const __nv_bfloat16* row_x = x + static_cast<long long>(row) * dim;

  float qsum0 = 0.0f, qsum1 = 0.0f, qsum2 = 0.0f, qsum3 = 0.0f;
  WelfordData w0{0.0f, 0.0f, 0, 0.0f}, w1{0.0f, 0.0f, 0, 0.0f};

  // Welford uses two interleaved scalar streams per lane.  The four loads also
  // form four contiguous warp quarters; retaining their partial sums lets us
  // reconstruct MeanOps' float4 accumulation tree without loading x twice.
  for (int base = 0; base < dim; base += 128) {
    int wc = base + lane;
    float v0 = wc < dim ? __bfloat162float(row_x[wc]) : 0.0f;
    float v1 = wc + 32 < dim ? __bfloat162float(row_x[wc + 32]) : 0.0f;
    float v2 = wc + 64 < dim ? __bfloat162float(row_x[wc + 64]) : 0.0f;
    float v3 = wc + 96 < dim ? __bfloat162float(row_x[wc + 96]) : 0.0f;
    qsum0 += v0; qsum1 += v1; qsum2 += v2; qsum3 += v3;
    if (wc < dim) w0 = update(w0, v0);
    if (wc + 32 < dim) w1 = update(w1, v1);
    if (wc + 64 < dim) w0 = update(w0, v2);
    if (wc + 96 < dim) w1 = update(w1, v3);
  }

  int quarter = lane >> 3;
  int source = (lane & 7) << 2;
  float sum0 = select_quarter_sum(qsum0, qsum1, qsum2, qsum3,
                                  source, quarter);
  float sum1 = select_quarter_sum(qsum0, qsum1, qsum2, qsum3,
                                  source + 1, quarter);
  float sum2 = select_quarter_sum(qsum0, qsum1, qsum2, qsum3,
                                  source + 2, quarter);
  float sum3 = select_quarter_sum(qsum0, qsum1, qsum2, qsum3,
                                  source + 3, quarter);
  float mean = warp_sum(((sum0 + sum1) + sum2) + sum3) * inv_dim;
  WelfordData moments = combine(w0, w1);
  moments = warp_reduce(moments);
  float variance = moments.m2 / moments.nf;
  mean = __shfl_sync(0xffffffffu, mean, 0);
  variance = __shfl_sync(0xffffffffu, variance, 0);
  if (lane == 0 && row_means != nullptr) {
    row_means[row] = mean;
    row_variances[row] = variance;
  }
  float std = sqrtf(variance + 1.0e-5f);

  int batch = row / rows_per_batch;
  const __nv_bfloat16* row_scale = scale +
      static_cast<long long>(batch) * parameter_batch_stride;
  const __nv_bfloat16* row_shift = shift +
      static_cast<long long>(batch) * parameter_batch_stride;
  __nv_bfloat16* row_out = out + static_cast<long long>(row) * dim;
  for (int col = lane; col < dim; col += 32) {
    float xv = __bfloat162float(row_x[col]);
    __nv_bfloat16 normalized = __float2bfloat16_rn((xv - mean) / std);
    float sv = __bfloat162float(row_scale[col]);
    __nv_bfloat16 modulated =
        __float2bfloat16_rn(__bfloat162float(normalized) * (1.0f + sv));
    float hv = __bfloat162float(row_shift[col]);
    row_out[col] = __float2bfloat16_rn(__bfloat162float(modulated) + hv);
  }
}

torch::Tensor adaln_forward(torch::Tensor x, torch::Tensor scale,
                            torch::Tensor shift) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16,
              "x must be CUDA BF16");
  TORCH_CHECK(x.is_contiguous() && scale.is_contiguous() && shift.is_contiguous(),
              "inputs must be contiguous");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  int batch = static_cast<int>(x.size(0));
  int rows_per_batch = static_cast<int>(x.size(1));
  int dim = static_cast<int>(x.size(2));
  int rows = batch * rows_per_batch;
  int parameter_batch_stride = scale.size(0) == batch ? dim : 0;
  dim3 block(32, 16);
  dim3 grid((rows + block.y - 1) / block.y);
  adaln_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      nullptr, nullptr,
      rows, rows_per_batch, dim, parameter_batch_stride,
      1.0f / static_cast<float>(dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> debug_forward(torch::Tensor x, torch::Tensor scale,
                                         torch::Tensor shift) {
  c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  int batch = static_cast<int>(x.size(0));
  int rows_per_batch = static_cast<int>(x.size(1));
  int dim = static_cast<int>(x.size(2));
  int rows = batch * rows_per_batch;
  auto opts = x.options().dtype(torch::kFloat32);
  auto means = torch::empty({rows}, opts);
  auto variances = torch::empty({rows}, opts);
  int parameter_batch_stride = scale.size(0) == batch ? dim : 0;
  adaln_kernel<<<(rows + 15) / 16, dim3(32, 16), 0,
                 at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      means.data_ptr<float>(), variances.data_ptr<float>(),
      rows, rows_per_batch, dim, parameter_batch_stride,
      1.0f / static_cast<float>(dim));
  return {out, means, variances};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &adaln_forward, "adaptive LayerNorm modulation (CUDA)");
  m.def("debug_forward", &debug_forward, "debug adaptive LayerNorm statistics");
}
"""


_cuda_module = load_inline(
    name="a100_adaln_modulation_cuda_v9",
    cpp_sources="",
    cuda_sources=_CUDA_SOURCE,
    extra_cuda_cflags=["-O3", "--prec-div=true", "--prec-sqrt=true"],
    functions=None,
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scale, shift):
        return _cuda_module.forward(x, scale, shift)
