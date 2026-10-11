"""
Fused Llama/Megatron RMSNorm forward (bf16 in / fp32 accumulate / bf16 out)
for NVIDIA H200 (SM90).

Semantics reproduced exactly (cast-before-weight):
    ms   = mean(x.float()**2, -1)
    rstd = rsqrt(ms + eps)
    xhat = (x.float() * rstd).to(bf16)      # mandatory intermediate rounding
    y    = (xhat * weight).to(bf16)         # bf16 channel gain

One pass over [T, H]: the row is loaded once into registers, reduced in fp32,
normalized, re-cast to bf16, gained, and stored. Nothing is materialized.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Single-block kernel: whole row lives in registers (H <= 16384)
# --------------------------------------------------------------------------- #

def _rows_configs():
    cfgs = []
    for rows in (1, 2, 4, 8):
        for warps in (2, 4, 8, 16):
            cfgs.append(triton.Config({"ROWS": rows}, num_warps=warps, num_stages=1))
    return cfgs


@triton.autotune(configs=_rows_configs(), key=["N", "BLOCK"])
@triton.jit
def _rmsnorm_fwd_reg(
    X_ptr,          # *bf16 [T, N] contiguous
    W_ptr,          # *bf16 [N]
    Y_ptr,          # *bf16 [T, N] contiguous
    T,              # int32 rows
    N,              # int32 hidden
    eps,            # fp32
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Channel gain hoisted out of the row loop (amortized over ROWS rows).
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    n_f = N.to(tl.float32)
    base = pid * ROWS

    for i in tl.static_range(ROWS):
        row = base + i
        if row < T:
            off = row.to(tl.int64) * N + cols

            x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)

            # fp32 row reduction -> mean square -> reciprocal std
            ms = tl.sum(x * x, axis=0) / n_f
            rstd = tl.math.rsqrt(ms + eps)

            # Llama casting mode: round the normalized activation to bf16
            # BEFORE the channel gain is applied.
            xhat = (x * rstd).to(tl.bfloat16).to(tl.float32)
            y = (xhat * w).to(tl.bfloat16)

            tl.store(Y_ptr + off, y, mask=mask)


# --------------------------------------------------------------------------- #
# Streaming fallback for very wide rows (H > 16384): two passes over L2
# --------------------------------------------------------------------------- #

@triton.jit
def _rmsnorm_fwd_stream(
    X_ptr, W_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    Xr = X_ptr + row.to(tl.int64) * N
    Yr = Y_ptr + row.to(tl.int64) * N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(Xr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x

    ms = tl.sum(acc, axis=0) / N.to(tl.float32)
    rstd = tl.math.rsqrt(ms + eps)

    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(Xr + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=m, other=0.0).to(tl.float32)
        xhat = (x * rstd).to(tl.bfloat16).to(tl.float32)
        tl.store(Yr + cols, (xhat * w).to(tl.bfloat16), mask=m)


# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #

EPS = 1e-6


class ModelNew(nn.Module):
    """Fused RMSNorm forward. Same signature as the reference Model."""

    def __init__(self):
        super().__init__()
        self.eps = EPS

    def forward(self, x, weight):
        x_in = x if x.is_contiguous() else x.contiguous()
        w_in = weight if weight.is_contiguous() else weight.contiguous()

        orig_shape = x_in.shape
        H = orig_shape[-1]
        x2d = x_in.reshape(-1, H)
        T = x2d.shape[0]

        y = torch.empty_like(x2d)

        if T == 0 or H == 0:
            return y.reshape(orig_shape)

        BLOCK = triton.next_power_of_2(H)

        if BLOCK <= 16384:
            grid = lambda meta: (triton.cdiv(T, meta["ROWS"]),)  # noqa: E731
            _rmsnorm_fwd_reg[grid](
                x2d, w_in, y,
                T, H, float(self.eps),
                BLOCK=BLOCK,
            )
        else:
            num_warps = 8
            _rmsnorm_fwd_stream[(T,)](
                x2d, w_in, y,
                H, float(self.eps),
                BLOCK=4096,
                num_warps=num_warps,
                num_stages=2,
            )

        return y.reshape(orig_shape)
