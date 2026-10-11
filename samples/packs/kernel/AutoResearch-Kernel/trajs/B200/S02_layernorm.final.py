import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(x_ptr, gamma_ptr, beta_ptr, y_ptr, hidden, eps,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden
    offset = row * hidden + cols

    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden
    rstd = tl.rsqrt(variance + eps)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * gamma + beta
    tl.store(y_ptr + offset, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta, eps):
        tokens, hidden = x.shape
        y = torch.empty_like(x)
        block = triton.next_power_of_2(hidden)
        num_warps = 4 if block <= 1024 else 8
        _layer_norm_kernel[(tokens,)](
            x, gamma, beta, y, hidden, eps,
            BLOCK=block, num_warps=num_warps, num_stages=1,
        )
        return y
