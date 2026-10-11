import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["PATH"] = "/home/user/github/autoresearch-runtime/a100-20260824/site/bin:" + os.environ.get("PATH", "")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"


_cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template<int BM, int BN, int BK, int TM = 8, int TN = 8>
__global__ void nt_sgemm_part(const half* __restrict__ A,
                              const half* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K, int k0, int k1, bool add) {
  constexpr int NTX = BN / TN;
  constexpr int NTY = BM / TM;
  constexpr int NTHREAD = NTX * NTY;
  __shared__ float As[BK][BM + 1];
  __shared__ float Bs[BK][BN + 1];

  const int tid = threadIdx.x;
  const int tx = tid % NTX;
  const int ty = tid / NTX;
  const int m0 = blockIdx.y * BM + ty * TM;
  const int n0 = blockIdx.x * BN + tx;
  float acc[TM][TN] = {0.0f};

  if (gridDim.z > 1) {
    const int part_size = (K + gridDim.z - 1) / gridDim.z;
    k0 = blockIdx.z * part_size;
    k1 = min(K, k0 + part_size);
  }

  int first_k = k0;
  if (first_k & 1) {
    float av[TM];
    float bv[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) av[i] = __half2float(A[(m0 + i) * K + first_k]);
    #pragma unroll
    for (int j = 0; j < TN; ++j) bv[j] = __half2float(B[(n0 + j * NTX) * K + first_k]);
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    ++first_k;
  }

  for (int kb = first_k; kb < k1; kb += BK) {
    #pragma unroll
    for (int x = tid; x < BM * (BK / 2); x += NTHREAD) {
      const int m = x / (BK / 2);
      const int kk = (x - m * (BK / 2)) * 2;
      const int gm = blockIdx.y * BM + m;
      const half2 hv = (kb + kk + 1 < K)
          ? *reinterpret_cast<const half2*>(A + gm * K + kb + kk)
          : __float2half2_rn(0.0f);
      const float2 fv = __half22float2(hv);
      As[kk][m] = (kb + kk < k1) ? fv.x : 0.0f;
      As[kk + 1][m] = (kb + kk + 1 < k1) ? fv.y : 0.0f;
    }
    #pragma unroll
    for (int x = tid; x < BN * (BK / 2); x += NTHREAD) {
      const int n = x / (BK / 2);
      const int kk = (x - n * (BK / 2)) * 2;
      const int gn = blockIdx.x * BN + n;
      const half2 hv = (kb + kk + 1 < K)
          ? *reinterpret_cast<const half2*>(B + gn * K + kb + kk)
          : __float2half2_rn(0.0f);
      const float2 fv = __half22float2(hv);
      Bs[kk][n] = (kb + kk < k1) ? fv.x : 0.0f;
      Bs[kk + 1][n] = (kb + kk + 1 < k1) ? fv.y : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      if (kb + kk < k1) {
        float av[TM];
        float bv[TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i) av[i] = As[kk][ty * TM + i];
        #pragma unroll
        for (int j = 0; j < TN; ++j) bv[j] = Bs[kk][tx + j * NTX];
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
          #pragma unroll
          for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int m = m0 + i;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + j * NTX;
      const int o = m * N + n;
      if (gridDim.z > 1)
        atomicAdd(C + o, acc[i][j]);
      else
        C[o] = add ? C[o] + acc[i][j] : acc[i][j];
    }
  }
}

void launch_nt_part(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                    int64_t k0, int64_t k1, int64_t bm, int64_t bn, bool add) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  const dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const half* ap = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
  const half* bp = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
  float* cp = c.data_ptr<float>();
  if (bm == 128 && bn == 128)
    nt_sgemm_part<128, 128, 16><<<grid, 256, 0, stream>>>(ap, bp, cp, M, N, K, k0, k1, add);
  else if (bm == 128 && bn == 64)
    nt_sgemm_part<128, 64, 16><<<grid, 128, 0, stream>>>(ap, bp, cp, M, N, K, k0, k1, add);
  else if (bm == 64 && bn == 128)
    nt_sgemm_part<64, 128, 16><<<grid, 128, 0, stream>>>(ap, bp, cp, M, N, K, k0, k1, add);
  else if (bm == 128 && bn == 32)
    nt_sgemm_part<128, 32, 16, 4, 4><<<grid, 256, 0, stream>>>(ap, bp, cp, M, N, K, k0, k1, add);
  else if (bm == 32 && bn == 128)
    nt_sgemm_part<32, 128, 16><<<grid, 64, 0, stream>>>(ap, bp, cp, M, N, K, k0, k1, add);
  else
    nt_sgemm_part<64, 64, 16><<<grid, 64, 0, stream>>>(ap, bp, cp, M, N, K, k0, k1, add);
}

void launch_nt_split(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                     int64_t parts, int64_t bm, int64_t bn) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  const dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm, parts);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(c.data_ptr<float>(), 0, M * N * sizeof(float), stream);
  const half* ap = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
  const half* bp = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
  float* cp = c.data_ptr<float>();
  if (bm == 128 && bn == 128)
    nt_sgemm_part<128, 128, 16><<<grid, 256, 0, stream>>>(ap, bp, cp, M, N, K, 0, K, false);
  else if (bm == 128 && bn == 64)
    nt_sgemm_part<128, 64, 16><<<grid, 128, 0, stream>>>(ap, bp, cp, M, N, K, 0, K, false);
  else if (bm == 64 && bn == 128)
    nt_sgemm_part<64, 128, 16><<<grid, 128, 0, stream>>>(ap, bp, cp, M, N, K, 0, K, false);
  else if (bm == 32 && bn == 128)
    nt_sgemm_part<32, 128, 16><<<grid, 64, 0, stream>>>(ap, bp, cp, M, N, K, 0, K, false);
  else
    nt_sgemm_part<64, 64, 16><<<grid, 64, 0, stream>>>(ap, bp, cp, M, N, K, 0, K, false);
}
'''

_cuda_mod = load_inline(
    name="kf_nt_sgemm_exact_v11",
    cpp_sources="void launch_nt_part(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, bool); void launch_nt_split(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
    cuda_sources=_cuda_src,
    functions=["launch_nt_part", "launch_nt_split"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Fused FP16 NT GEMM with IEEE FP32 accumulation and FP32 output."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        assert a.dim() == 2 and b.dim() == 2
        assert a.shape[1] == b.shape[1]
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()

        M, K = a.shape
        N = b.shape[0]

        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        if M == 0 or N == 0:
            return c
        if K == 0:
            c.zero_()
            return c


        if K == 2048 and N == 1024:
            reduction_parts = 5 if M % 1024 == 0 else 2
        elif K >= 4096:
            reduction_parts = 2
        elif K > 2048:
            reduction_parts = 5
        else:
            reduction_parts = 1

        if reduction_parts == 1 and M <= 512:
            block_m, block_n = 128, 32
        elif reduction_parts == 1 and K == 1024 and N <= 1536:
            block_m, block_n = 128, 128
        else:
            block_m, block_n = 128, 64

        if reduction_parts > 1:
            _cuda_mod.launch_nt_split(
                a, b, c, reduction_parts, block_m, block_n
            )
        else:
            _cuda_mod.launch_nt_part(
                a, b, c, 0, K, block_m, block_n, False
            )
        return c
