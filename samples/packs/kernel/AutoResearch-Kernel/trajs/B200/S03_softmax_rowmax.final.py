"""Fused row-wise bf16 softmax for contiguous two-dimensional inputs."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask,
                other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    numer = tl.exp(x)
    denom = tl.sum(numer, axis=0)
    tl.store(y_ptr + row * n_cols + cols, numer / denom, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        block = triton.next_power_of_2(n_cols)
        _softmax_kernel[(n_rows,)](
            x, out, n_cols=n_cols, BLOCK=block,
            num_warps=8, num_stages=1,
        )
        return out
