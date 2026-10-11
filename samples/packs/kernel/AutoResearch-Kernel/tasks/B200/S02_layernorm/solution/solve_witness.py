import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd(
    X, Y, G, B,
    stride_x, stride_y,
    T, H,
    inv_h, eps,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cmask = cols < H

    # gamma / beta are shared by every row this program handles -> load once.
    g = tl.load(G + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)

    for i in tl.static_range(ROWS):
        row = pid * ROWS + i
        rmask = cmask & (row < T)
        xp = X + row.to(tl.int64) * stride_x + cols
        x = tl.load(xp, mask=rmask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) * inv_h
        xc = tl.where(rmask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) * inv_h
        rstd = tl.rsqrt(var + eps)

        y = xc * rstd * g + b
        tl.store(Y + row.to(tl.int64) * stride_y + cols, y.to(tl.bfloat16), mask=rmask)


def _cfg(H):
    BLOCK = triton.next_power_of_2(int(H))
    if BLOCK < 128:
        BLOCK = 128
    if BLOCK <= 1024:
        num_warps = 4
        rows = 4
    elif BLOCK <= 2048:
        num_warps = 8
        rows = 2
    elif BLOCK <= 4096:
        num_warps = 8
        rows = 1
    else:
        num_warps = 16
        rows = 1
    return BLOCK, rows, num_warps


class ModelNew(nn.Module):
    """Fused bf16 LayerNorm forward: one HBM read + one HBM write, fp32 accumulation."""

    def __init__(self):
        super().__init__()
        self._cfg_cache = {}

    def forward(self, x, gamma, beta, eps):
        assert x.dim() == 2, "expected [T, H]"
        if not x.is_contiguous():
            x = x.contiguous()
        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()

        T, H = x.shape
        y = torch.empty_like(x, dtype=torch.bfloat16)

        cfg = self._cfg_cache.get(H)
        if cfg is None:
            cfg = _cfg(H)
            self._cfg_cache[H] = cfg
        BLOCK, ROWS, num_warps = cfg

        grid = (triton.cdiv(T, ROWS),)
        _layernorm_fwd[grid](
            x, y, gamma, beta,
            x.stride(0), y.stride(0),
            T, H,
            1.0 / float(H), float(eps),
            BLOCK=BLOCK,
            ROWS=ROWS,
            num_warps=num_warps,
            num_stages=1,
        )
        return y
