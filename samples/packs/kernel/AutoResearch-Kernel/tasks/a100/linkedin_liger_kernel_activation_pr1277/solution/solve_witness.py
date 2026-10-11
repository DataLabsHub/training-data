import torch
import torch.nn as nn
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=1),
    ]


@triton.autotune(configs=_configs(), key=['n'])
@triton.jit
def _swiglu_fwd(A, B, O, n, mult, BLOCK: tl.constexpr, INT64: tl.constexpr):
    """out = silu(a*mult) * b  computed in fp32, single bf16 round at the end.

    Pure streaming elementwise kernel: 2 x bf16 loads + 1 x bf16 store per
    element (6 B/elem).  Each thread handles BLOCK/(32*num_warps) contiguous
    bf16 values so the vectorizer emits 128-bit ld.global/st.global.
    """
    pid = tl.program_id(0)
    if INT64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)

    # Uniform (per-block) branch: the interior blocks run fully unmasked, only
    # the final ragged block pays for predication.
    if base + BLOCK <= n:
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
        a = tl.load(A + offs)
        b = tl.load(B + offs)
        x = a.to(tl.float32) * mult
        y = x * tl.sigmoid(x) * b.to(tl.float32)
        tl.store(O + offs, y.to(tl.bfloat16))
    else:
        m = offs < n
        a = tl.load(A + offs, mask=m, other=0.0)
        b = tl.load(B + offs, mask=m, other=0.0)
        x = a.to(tl.float32) * mult
        y = x * tl.sigmoid(x) * b.to(tl.float32)
        tl.store(O + offs, y.to(tl.bfloat16), mask=m)


class ModelNew(nn.Module):
    """Fused SiLU-gated multiply: silu(a * gate_multiplier) * b -> bf16."""

    def __init__(self, gate_multiplier=1.702):
        super().__init__()
        self.gate_multiplier = float(gate_multiplier)

    def forward(self, a, b):
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()

        out = torch.empty(a.shape, device=a.device, dtype=torch.bfloat16)
        n = a.numel()
        if n == 0:
            return out

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        _swiglu_fwd[grid](
            a, b, out, n, self.gate_multiplier,
            INT64=(n >= 2 ** 31),
        )
        return out
