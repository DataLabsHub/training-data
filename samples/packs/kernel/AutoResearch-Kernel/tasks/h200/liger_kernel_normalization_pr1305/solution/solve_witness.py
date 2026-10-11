# ModelNew: fused single-pass RMSNorm (Llama casting) in Triton
# target: NVIDIA H200 / sm_90, bf16 in-out, fp32 accumulation
#
# Strategy
#   * one program handles ROWS consecutive rows; the gain vector is loaded once
#     into registers and reused across those rows (weight traffic per row would
#     otherwise equal activation traffic per row)
#   * x is loaded exactly once: kept in registers between the sum-of-squares
#     reduction and the normalize+scale store, so HBM traffic is the ideal
#     T*H*2 read + T*H*2 write
#   * H that is not a power of two is covered by a two-tile decomposition
#     (largest pow2 <= H, plus a masked tail tile) instead of rounding the whole
#     row up to the next power of two
#   * Llama semantics are reproduced exactly: rstd in fp32, x*rstd rounded DOWN
#     to bf16 first, then multiplied by the bf16 gain and rounded to bf16

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_pow2(
    X, W, Y,
    stride_x, stride_y,
    Hf, eps,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """H is exactly BLOCK (power of two): no masking anywhere."""
    row0 = tl.program_id(0) * ROWS
    off = tl.arange(0, BLOCK)

    w = tl.load(W + off, eviction_policy="evict_last").to(tl.float32)

    for i in tl.static_range(ROWS):
        r = row0 + i
        x = tl.load(X + r * stride_x + off, eviction_policy="evict_first").to(tl.float32)
        ms = tl.sum(x * x, axis=0) / Hf
        rstd = tl.rsqrt(ms + eps)
        # Llama cast-before-gain: round to bf16, then apply the gain in fp32
        normed = (x * rstd).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + r * stride_y + off, (normed * w).to(tl.bfloat16))


@triton.jit
def _rmsnorm_fwd_tail(
    X, W, Y,
    stride_x, stride_y,
    Hf, eps, H,
    B1: tl.constexpr,
    B2: tl.constexpr,
    ROWS: tl.constexpr,
):
    """H = B1 (largest pow2 <= H) + remainder covered by a masked B2 tile."""
    row0 = tl.program_id(0) * ROWS
    off1 = tl.arange(0, B1)
    off2 = B1 + tl.arange(0, B2)
    m2 = off2 < H

    w1 = tl.load(W + off1, eviction_policy="evict_last").to(tl.float32)
    w2 = tl.load(W + off2, mask=m2, other=0.0, eviction_policy="evict_last").to(tl.float32)

    for i in tl.static_range(ROWS):
        r = row0 + i
        xp = X + r * stride_x
        x1 = tl.load(xp + off1, eviction_policy="evict_first").to(tl.float32)
        x2 = tl.load(xp + off2, mask=m2, other=0.0, eviction_policy="evict_first").to(tl.float32)

        acc = tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)
        rstd = tl.rsqrt(acc / Hf + eps)

        yp = Y + r * stride_y
        n1 = (x1 * rstd).to(tl.bfloat16).to(tl.float32)
        tl.store(yp + off1, (n1 * w1).to(tl.bfloat16))
        n2 = (x2 * rstd).to(tl.bfloat16).to(tl.float32)
        tl.store(yp + off2, (n2 * w2).to(tl.bfloat16), mask=m2)


@triton.jit
def _rmsnorm_fwd_big(
    X, W, Y,
    stride_x, stride_y,
    Hf, eps, H,
    BLOCK: tl.constexpr,
):
    """Fallback for rows too wide to hold in registers: two streaming passes."""
    r = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, H, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        x = tl.load(X + r * stride_x + idx, mask=idx < H, other=0.0).to(tl.float32)
        acc += x * x
    rstd = tl.rsqrt(tl.sum(acc, axis=0) / Hf + eps)

    for start in range(0, H, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        m = idx < H
        x = tl.load(X + r * stride_x + idx, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + idx, mask=m, other=0.0).to(tl.float32)
        n = (x * rstd).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + r * stride_y + idx, (n * w).to(tl.bfloat16), mask=m)


def _prev_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _num_warps(total):
    if total <= 2048:
        return 4
    if total <= 4096:
        return 8
    return 16


def _pick_rows(T, total):
    if total <= 2048:
        cand = 8
    elif total <= 4096:
        cand = 4
    else:
        cand = 2
    while cand > 1 and (T % cand != 0 or (T // cand) < 1024):
        cand //= 2
    return cand


class ModelNew(nn.Module):
    """RMSNorm, Llama casting mode, eps = 1e-6, offset = 0.0. Inputs untouched."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self._plans = {}

    def _plan(self, T, H):
        key = (T, H)
        plan = self._plans.get(key)
        if plan is not None:
            return plan

        if H > 16384:
            block = 2048
            plan = ("big", block, 0, 1, _num_warps(block))
        else:
            b1 = _prev_pow2(H)
            if b1 == H:
                rows = _pick_rows(T, b1)
                plan = ("pow2", b1, 0, rows, _num_warps(b1))
            else:
                b2 = triton.next_power_of_2(H - b1)
                total = b1 + b2
                rows = _pick_rows(T, total)
                plan = ("tail", b1, b2, rows, _num_warps(total))

        self._plans[key] = plan
        return plan

    def forward(self, x, weight):
        H = x.shape[-1]
        x2 = x if x.dim() == 2 else x.reshape(-1, H)
        T = x2.shape[0]

        out = torch.empty_like(x2)
        if T == 0 or H == 0:
            return out if x.dim() == 2 else out.view(x.shape)

        kind, b1, b2, rows, warps = self._plan(T, H)
        sx, sy = x2.stride(0), out.stride(0)
        Hf = float(H)

        if kind == "pow2":
            _rmsnorm_fwd_pow2[(T // rows,)](
                x2, weight, out, sx, sy, Hf, self.eps,
                BLOCK=b1, ROWS=rows, num_warps=warps, num_stages=1,
            )
        elif kind == "tail":
            _rmsnorm_fwd_tail[(T // rows,)](
                x2, weight, out, sx, sy, Hf, self.eps, H,
                B1=b1, B2=b2, ROWS=rows, num_warps=warps, num_stages=1,
            )
        else:
            _rmsnorm_fwd_big[(T,)](
                x2, weight, out, sx, sy, Hf, self.eps, H,
                BLOCK=b1, num_warps=warps, num_stages=2,
            )

        return out if x.dim() == 2 else out.view(x.shape)
