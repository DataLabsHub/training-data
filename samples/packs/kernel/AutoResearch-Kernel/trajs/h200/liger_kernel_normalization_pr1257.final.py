"""Single-pass Llama-style RMS normalization for contiguous BF16 tensors."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rms_norm(x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    xf = tl.load(x_ptr + row * n_cols + cols, mask=mask,
                 other=0.0).to(tl.float32)
    square_sum = tl.sum(xf * xf, axis=0)
    rstd = tl.rsqrt(square_sum / n_cols + 1.0e-6)

    # Llama casting mode requires this rounding before the BF16 gain.
    xhat = (xf * rstd).to(tl.bfloat16)
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, xhat * weight, mask=mask)


@triton.jit
def _rms_norm_exact(x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, n_cols)
    xf = tl.load(x_ptr + row * n_cols + cols).to(tl.float32)
    square_sum = tl.sum(xf * xf, axis=0)
    rstd = tl.rsqrt(square_sum / n_cols + 1.0e-6)
    xhat = (xf * rstd).to(tl.bfloat16)
    weight = tl.load(w_ptr + cols)
    tl.store(y_ptr + row * n_cols + cols, xhat * weight)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        rows, cols = x.shape
        out = torch.empty_like(x)

        if cols <= 2048:
            warps = 4
        elif cols >= 7168:
            warps = 16
        else:
            warps = 8

        if cols & (cols - 1) == 0:
            _rms_norm_exact[(rows,)](x, weight, out, n_cols=cols,
                                     num_warps=warps)
        else:
            block = triton.next_power_of_2(cols)
            _rms_norm[(rows,)](x, weight, out, n_cols=cols, BLOCK=block,
                               num_warps=warps)
        return out
