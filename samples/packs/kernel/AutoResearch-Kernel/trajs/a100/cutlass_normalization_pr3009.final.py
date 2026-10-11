"""Single-pass fused fp16 LayerNorm with fp32 reductions."""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layer_norm(x, weight, bias, y, n_cols: tl.constexpr,
                BLOCK: tl.constexpr, STREAM_X: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    # The affine vectors are shared by every row.  Bypassing L1 for the
    # streaming activation/output keeps those vectors resident on narrower
    # rows; for the widest rows normal caching is marginally faster.
    if STREAM_X:
        x_f32 = tl.load(x + offsets, mask=mask, other=0.0,
                        cache_modifier=".cg").to(tl.float32)
    else:
        x_f32 = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x_f32, axis=0) / n_cols
    centered = tl.where(mask, x_f32 - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(var + 1.0e-5)

    w = tl.load(weight + cols, mask=mask, other=0.0,
                cache_modifier=".ca").to(tl.float32)
    b = tl.load(bias + cols, mask=mask, other=0.0,
                cache_modifier=".ca").to(tl.float32)
    out = centered * rstd * w + b
    if STREAM_X:
        tl.store(y + offsets, out, mask=mask, cache_modifier=".cg")
    else:
        tl.store(y + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, weight, bias):
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        block = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x)

        # Four warps give enough occupancy with less reduction overhead for
        # 2K and smaller rows. Wider rows benefit from eight-way reduction.
        num_warps = 4 if block <= 2048 else 8
        _layer_norm[(n_rows,)](
            x, weight, bias, y,
            n_cols=n_cols,
            BLOCK=block,
            STREAM_X=block <= 4096,
            num_warps=num_warps,
        )
        return y
