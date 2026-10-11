import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _ragged_swiglu_kernel(
    X,            # *bf16  [E, Tmax, 2H]
    CNT,          # *int32 [E]
    Y,            # *bf16  [E, Tmax, H]
    Tmax,
    H,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_h = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.program_id(2)

    c = tl.load(CNT + e).to(tl.int32)

    offs = pid_h * BLOCK + tl.arange(0, BLOCK)

    row = e.to(tl.int64) * Tmax + t.to(tl.int64)
    y_ptr = Y + row * H + offs

    if t < c:
        xbase = X + row * (2 * H)
        if EVEN:
            g = tl.load(xbase + offs).to(tl.float32)
            u = tl.load(xbase + H + offs).to(tl.float32)
            # SwiGLU in fp32: gate * sigmoid(gate) * up
            act = (g * u) / (1.0 + tl.exp(-g))
            tl.store(y_ptr, act.to(tl.bfloat16))
        else:
            m = offs < H
            g = tl.load(xbase + offs, mask=m, other=0.0).to(tl.float32)
            u = tl.load(xbase + H + offs, mask=m, other=0.0).to(tl.float32)
            act = (g * u) / (1.0 + tl.exp(-g))
            tl.store(y_ptr, act.to(tl.bfloat16), mask=m)
    else:
        # padded row -> exact +0.0
        z = tl.zeros([BLOCK], dtype=tl.bfloat16)
        if EVEN:
            tl.store(y_ptr, z)
        else:
            tl.store(y_ptr, z, mask=offs < H)


def _pick_block(H):
    # largest power-of-two tile that divides H (fast, no masking), capped at 2048
    for b in (2048, 1024, 512, 256, 128, 64, 32):
        if H % b == 0:
            return b, True
    b = 1
    while b < H:
        b *= 2
    return min(b, 2048), False


class ModelNew(nn.Module):
    """Fused ragged expert-major SwiGLU (BF16 in / BF16 out) with per-expert counts."""

    def __init__(self):
        super().__init__()
        self._cfg_cache = {}

    def forward(self, x, counts):
        E, Tmax, two_h = x.shape
        H = two_h // 2

        if not x.is_contiguous():
            x = x.contiguous()
        if counts.dtype != torch.int32:
            counts = counts.to(torch.int32)
        if not counts.is_contiguous():
            counts = counts.contiguous()

        y = torch.empty((E, Tmax, H), dtype=torch.bfloat16, device=x.device)

        if E == 0 or Tmax == 0 or H == 0:
            return y

        cfg = self._cfg_cache.get(H)
        if cfg is None:
            BLOCK, EVEN = _pick_block(H)
            nw = max(1, min(8, BLOCK // 256))
            cfg = (BLOCK, EVEN, nw)
            self._cfg_cache[H] = cfg
        BLOCK, EVEN, num_warps = cfg

        grid = (triton.cdiv(H, BLOCK), Tmax, E)

        _ragged_swiglu_kernel[grid](
            x, counts, y,
            Tmax, H,
            BLOCK=BLOCK,
            EVEN=EVEN,
            num_warps=num_warps,
            num_stages=1,
        )
        return y
