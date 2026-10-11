import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(
    a_ptr, b_ptr, out_ptr,
    T, F,
    BLOCK_T: tl.constexpr, BLOCK_F: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)

    mask = (offs_t[:, None] < T) & (offs_f[None, :] < F)

    a = tl.load(a_ptr + offs_t[:, None] * F + offs_f[None, :], mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_t[:, None] * F + offs_f[None, :], mask=mask, other=0.0)

    a32 = a.to(tl.float32)

    # sigmoid in FP32 using stable identity: sig = 1/(1+exp(-x))
    # sig = exp(x) / (exp(x) + 1) == 1 / (1 + exp(-x))
    # Use erf-based? No — use exp path in FP32.
    neg = -a32
    e = tl.exp(neg)
    sig = 1.0 / (1.0 + e)

    gate32 = a32 * sig
    gate16 = gate32.to(tl.bfloat16)      # ROUNDING BOUNDARY #1

    # Multiply in FP32 then cast to BF16 (boundary #2).
    b32 = b.to(tl.float32)
    g32 = gate16.to(tl.float32)
    prod32 = g32 * b32
    out = prod32.to(tl.bfloat16)         # ROUNDING BOUNDARY #2

    tl.store(out_ptr + offs_t[:, None] * F + offs_f[None, :], out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        T, F = a.shape
        out = torch.empty_like(a)

        # Tiled 2D grid. Choose block sizes tuned for H200 (BF16 memory bound).
        BLOCK_T = 8
        BLOCK_F = 128

        grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(F, BLOCK_F))
        _silu_mul_kernel[grid](a, b, out, T, F, BLOCK_T, BLOCK_F)
        return out
