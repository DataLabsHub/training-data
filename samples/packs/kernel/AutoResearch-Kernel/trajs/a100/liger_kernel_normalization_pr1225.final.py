import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rms_fwd(x_ptr, w_ptr, s_ptr, b_ptr, out_ptr,
             n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(s_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    gain = weight * (1.0 + scale)
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) * (1.0 / n_cols)
    rstd = tl.rsqrt(mean_square + 1.0e-6)
    y = (x * rstd) * gain + shift
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


@triton.jit
def _rms_fwd_4rows(x_ptr, w_ptr, s_ptr, b_ptr, out_ptr,
                   n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row0 = tl.program_id(0) * 4
    rows = row0 + tl.arange(0, 4)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    mask = cols < n_cols
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0,
                     eviction_policy="evict_last").to(tl.float32)
    scale = tl.load(s_ptr + cols, mask=mask, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)
    shift = tl.load(b_ptr + cols, mask=mask, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)
    gain = weight * (1.0 + scale)
    x = tl.load(x_ptr + rows * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1) * (1.0 / n_cols)
    rstd = tl.rsqrt(mean_square + 1.0e-6)
    y = (x * rstd[:, None]) * gain + shift
    tl.store(out_ptr + rows * n_cols + cols, y, mask=mask,
             eviction_policy="evict_first")


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, scale, shift):
        rows, cols = x.shape
        out = torch.empty_like(x)
        if cols <= 2048:
            _rms_fwd_4rows[(rows // 4,)](
                x, weight, scale, shift, out, cols,
                BLOCK=2048, num_warps=4,
            )
        elif cols <= 4096:
            _rms_fwd[(rows,)](
                x, weight, scale, shift, out, cols,
                BLOCK=4096, num_warps=16,
            )
        else:
            _rms_fwd[(rows,)](
                x, weight, scale, shift, out, cols,
                BLOCK=8192, num_warps=32,
            )
        return out
