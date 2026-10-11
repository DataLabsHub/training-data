"""
Fused forward LayerNorm (row-per-CTA, fp32 accumulate, fp16 in/out) in Triton.

Contract reproduced exactly:
    mu   = mean_h(x)                       (fp32)
    var  = mean_h((x - mu)^2)              (BIASED, /H, fp32, two-pass)
    y    = (x - mu) * rsqrt(var + 1e-5) * weight + bias
    single cast to fp16 at the very end.

Strategy: one program per row.  The whole row is loaded ONCE into registers,
both reductions (mean, then centered variance) run out of registers/shared, and
the affine + fp16 store happen without ever re-reading global memory.  Total
DRAM traffic is therefore the floor: read T*H fp16 + write T*H fp16.
"""

import torch
import torch.nn as nn

import triton
import triton.language as tl

EPS = 1e-5


# --------------------------------------------------------------------------- #
# Fast path: whole row lives in registers (H <= 8192, the legal domain max).
# --------------------------------------------------------------------------- #
@triton.jit
def _ln_fwd_persistent(
    X, W, B, Y,
    stride_m,
    H,
    h_f,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    xp = X + row * stride_m
    yp = Y + row * stride_m

    cols = tl.arange(0, BLOCK)
    mask = cols < H

    # Single global read of the activation row.
    x = tl.load(xp + cols, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)

    # Pass 1 (registers): mean.
    mu = tl.sum(x, axis=0) / h_f

    # Pass 2 (registers): biased variance of the centered row.
    xc = tl.where(mask, x - mu, 0.0)
    var = tl.sum(xc * xc, axis=0) / h_f
    rstd = 1.0 / tl.sqrt(var + eps)

    # Per-hidden affine (weight/bias stay resident in L1/L2 across rows).
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * w + b
    tl.store(yp + cols, y.to(tl.float16), mask=mask)


# --------------------------------------------------------------------------- #
# Generic path: streaming loop (used only if H exceeds the register budget).
# --------------------------------------------------------------------------- #
@triton.jit
def _ln_fwd_loop(
    X, W, B, Y,
    stride_m,
    H,
    h_f,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    xp = X + row * stride_m
    yp = Y + row * stride_m

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < H
        x = tl.load(xp + cols, mask=m, other=0.0).to(tl.float32)
        acc += x
        acc2 += x * x

    mu = tl.sum(acc, axis=0) / h_f
    var = tl.sum(acc2, axis=0) / h_f - mu * mu
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < H
        x = tl.load(xp + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=m, other=0.0).to(tl.float32)
        y = (x - mu) * rstd * w + b
        tl.store(yp + cols, y.to(tl.float16), mask=m)


def _plan(H: int):
    """Pick BLOCK / num_warps so each thread owns ~16 elements (2x 16B loads)."""
    block = triton.next_power_of_2(H)
    if block <= 8192:
        if block <= 512:
            warps = 2
        elif block <= 2048:
            warps = 4
        elif block <= 4096:
            warps = 8
        else:
            warps = 16
        return ("persistent", block, warps, 1)
    # Fallback streaming path (outside the declared legal domain).
    return ("loop", 4096, 8, 3)


class ModelNew(nn.Module):
    """Fused row-wise LayerNorm: one Triton CTA per row, fp32 accumulation."""

    def __init__(self):
        super().__init__()
        self.eps = EPS
        self._plans = {}

    def forward(self, x, weight, bias):
        H = x.shape[-1]
        x2 = x if x.dim() == 2 else x.reshape(-1, H)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        T = x2.shape[0]

        y = torch.empty_like(x2)
        if T == 0 or H == 0:
            return y.reshape(x.shape)

        w = weight.contiguous()
        b = bias.contiguous()

        plan = self._plans.get(H)
        if plan is None:
            plan = _plan(H)
            self._plans[H] = plan
        kind, BLOCK, num_warps, num_stages = plan

        h_f = float(H)
        grid = (T,)
        if kind == "persistent":
            _ln_fwd_persistent[grid](
                x2, w, b, y,
                x2.stride(0),
                H, h_f, self.eps,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            _ln_fwd_loop[grid](
                x2, w, b, y,
                x2.stride(0),
                H, h_f, self.eps,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        return y if y.shape == x.shape else y.reshape(x.shape)
