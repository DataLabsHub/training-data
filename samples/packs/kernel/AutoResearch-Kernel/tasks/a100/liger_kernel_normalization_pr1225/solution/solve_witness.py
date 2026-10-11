import torch
import torch.nn as nn

import triton
import triton.language as tl

EPS = 1e-6


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _autotune_configs():
    cfgs = []
    for rows in (1, 2, 4):
        for warps in (4, 8, 16):
            cfgs.append(
                triton.Config({"ROWS_PER_PROG": rows}, num_warps=warps, num_stages=1)
            )
    return cfgs


@triton.autotune(configs=_autotune_configs(), key=["H", "BLOCK_H"])
@triton.jit
def _mod_rmsnorm_fwd(
    X,            # *bf16 (T, H)
    W,            # *bf16 (H,)
    S,            # *bf16 (H,)
    B,            # *bf16 (H,)
    Y,            # *bf16 (T, H)
    stride_xm,
    stride_ym,
    T,
    H,
    eps,
    BLOCK_H: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """Fused modulated RMSNorm: y = (x * rsqrt(mean(x^2)+eps)) * w * (1+s) + b.

    One program owns ROWS_PER_PROG whole rows; the hidden axis is covered by a
    single BLOCK_H tile (BLOCK_H = next_pow2(H)) so the reduction and the
    modulation share one fp32 register copy of the row -- x is read from DRAM
    exactly once and there is a single bf16 rounding, at the store.
    """
    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    # Per-hidden modulation vectors: promoted to fp32 once, reused for every
    # row handled by this program.  gain = weight * (1 + scale) in fp32.
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    gain = w * (1.0 + s)

    hf = H.to(tl.float32)
    row0 = pid * ROWS_PER_PROG

    for i in tl.static_range(ROWS_PER_PROG):
        row = row0 + i
        if row < T:
            x = tl.load(
                X + row * stride_xm + cols,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # fp32 mean-square reduction over H (masked lanes contribute 0).
            mean_sq = tl.sum(x * x, axis=0) / hf
            rstd = tl.math.rsqrt(mean_sq + eps)

            y = x * rstd * gain + b

            tl.store(
                Y + row * stride_ym + cols,
                y.to(tl.bfloat16),
                mask=mask,
                eviction_policy="evict_first",
            )


class ModelNew(nn.Module):
    """Forward-only modulated RMSNorm (FiLM / AdaRMSNorm) as one Triton kernel."""

    def __init__(self):
        super().__init__()
        self.eps = EPS

    def forward(self, x, weight, scale, shift):
        orig_shape = x.shape
        H = orig_shape[-1]
        x2 = x if x.dim() == 2 else x.reshape(-1, H)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        T = x2.shape[0]

        y = torch.empty((T, H), dtype=torch.bfloat16, device=x2.device)

        BLOCK_H = _next_pow2(H)

        grid = lambda meta: (triton.cdiv(T, meta["ROWS_PER_PROG"]),)
        _mod_rmsnorm_fwd[grid](
            x2,
            weight,
            scale,
            shift,
            y,
            x2.stride(0),
            y.stride(0),
            T,
            H,
            self.eps,
            BLOCK_H=BLOCK_H,
        )

        return y if x.dim() == 2 else y.reshape(orig_shape)
