import torch
import torch.nn as nn

import triton
import triton.language as tl

EPS = 1e-5


@triton.jit
def _bn_stats_kernel(
    X,                      # *fp16, (T, H) contiguous
    SUM,                    # *fp32, (NSPLIT, H)
    SUMSQ,                  # *fp32, (NSPLIT, H)
    T,                      # i32
    H,                      # i32
    SPLIT_ROWS,             # i32, multiple of BLOCK_T
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Partial per-column sum / sum-of-squares over a row slab (fp32 accum)."""
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    row_start = pid_s * SPLIT_ROWS
    row_end = tl.minimum(row_start + SPLIT_ROWS, T)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H], dtype=tl.float32)

    for r0 in range(row_start, row_end, BLOCK_T):
        offs_t = r0 + tl.arange(0, BLOCK_T)
        m = (offs_t[:, None] < row_end) & mask_h[None, :]
        v = tl.load(X + offs_t[:, None] * H + offs_h[None, :], mask=m, other=0.0)
        vf = v.to(tl.float32)
        acc += tl.sum(vf, axis=0)
        acc2 += tl.sum(vf * vf, axis=0)

    base = pid_s * H + offs_h
    tl.store(SUM + base, acc, mask=mask_h)
    tl.store(SUMSQ + base, acc2, mask=mask_h)


@triton.jit
def _bn_finalize_kernel(
    SUM, SUMSQ,             # *fp32, (NSPLIT, H)
    W, B,                   # *fp32, (H,)
    MEAN, SCALE, SHIFT,     # *fp32, (H,)
    H,                      # i32
    NSPLIT,                 # i32
    T_F,                    # fp32 (float(T))
    eps,                    # fp32
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Cross-slab reduction -> mean, rstd*weight, bias (all fp32)."""
    pid_h = tl.program_id(0)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    s = tl.zeros([BLOCK_H], dtype=tl.float32)
    s2 = tl.zeros([BLOCK_H], dtype=tl.float32)
    for n0 in range(0, NSPLIT, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        m = (offs_n[:, None] < NSPLIT) & mask_h[None, :]
        p = offs_n[:, None] * H + offs_h[None, :]
        s += tl.sum(tl.load(SUM + p, mask=m, other=0.0), axis=0)
        s2 += tl.sum(tl.load(SUMSQ + p, mask=m, other=0.0), axis=0)

    mean = s / T_F
    var = s2 / T_F - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + offs_h, mask=mask_h, other=0.0)
    b = tl.load(B + offs_h, mask=mask_h, other=0.0)

    tl.store(MEAN + offs_h, mean, mask=mask_h)
    tl.store(SCALE + offs_h, rstd * w, mask=mask_h)
    tl.store(SHIFT + offs_h, b, mask=mask_h)


@triton.jit
def _bn_apply_kernel(
    X, Y,                   # *fp16, (T, H)
    MEAN, SCALE, SHIFT,     # *fp32, (H,)
    T, H,                   # i32
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """y = (x - mean) * (rstd*w) + b, fp32 math, single fp16 rounding on store."""
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    mean = tl.load(MEAN + offs_h, mask=mask_h, other=0.0)
    scale = tl.load(SCALE + offs_h, mask=mask_h, other=0.0)
    shift = tl.load(SHIFT + offs_h, mask=mask_h, other=0.0)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    m = (offs_t[:, None] < T) & mask_h[None, :]
    off = offs_t[:, None] * H + offs_h[None, :]

    v = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    res = (v - mean[None, :]) * scale[None, :] + shift[None, :]
    tl.store(Y + off, res.to(Y.dtype.element_ty), mask=m)


def _pick_blocks(H):
    if H >= 2048:
        return 256, 32, 8
    if H >= 512:
        return 128, 64, 8
    return 128, 32, 4


class ModelNew(nn.Module):
    """Fused mixed-precision forward BatchNorm over (T, H): stats along T."""

    def __init__(self, eps=EPS):
        super().__init__()
        self.eps = float(eps)
        self._TARGET_PROGRAMS = 864          # ~8 waves on 108 SMs
        self._MIN_ROWS_PER_PROGRAM = 64

    def forward(self, x, weight, bias):
        assert x.dim() == 2, "expected a 2-D activation matrix (T, H)"
        if not x.is_contiguous():
            x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        T, H = x.shape
        dev = x.device

        BLOCK_H, BLOCK_T, num_warps = _pick_blocks(H)
        col_blocks = triton.cdiv(H, BLOCK_H)

        # Row-slab split: enough programs to fill the GPU, but keep each
        # program's inner loop long enough to amortize its launch/reduce cost.
        want = max(1, self._TARGET_PROGRAMS // col_blocks)
        max_split = max(1, T // self._MIN_ROWS_PER_PROGRAM)
        nsplit = min(want, max_split)
        split_rows = triton.cdiv(triton.cdiv(T, nsplit), BLOCK_T) * BLOCK_T
        nsplit = triton.cdiv(T, split_rows)

        psum = torch.empty((nsplit, H), device=dev, dtype=torch.float32)
        psumsq = torch.empty((nsplit, H), device=dev, dtype=torch.float32)

        _bn_stats_kernel[(col_blocks, nsplit)](
            x, psum, psumsq,
            T, H, split_rows,
            BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=3,
        )

        mean = torch.empty((H,), device=dev, dtype=torch.float32)
        scale = torch.empty((H,), device=dev, dtype=torch.float32)
        shift = torch.empty((H,), device=dev, dtype=torch.float32)

        BLOCK_N = min(128, max(8, triton.next_power_of_2(nsplit)))
        _bn_finalize_kernel[(col_blocks,)](
            psum, psumsq, weight, bias,
            mean, scale, shift,
            H, nsplit, float(T), self.eps,
            BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        y = torch.empty_like(x)
        A_BLOCK_T = max(8192 // BLOCK_H, 8)
        _bn_apply_kernel[(col_blocks, triton.cdiv(T, A_BLOCK_T))](
            x, y, mean, scale, shift,
            T, H,
            BLOCK_T=A_BLOCK_T, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=3,
        )
        return y
