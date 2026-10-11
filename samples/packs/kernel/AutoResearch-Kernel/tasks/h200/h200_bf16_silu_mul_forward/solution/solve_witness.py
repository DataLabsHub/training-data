"""Fused BF16 SiLU-multiply for Hopper."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # Loads are BF16, while the nonlinear stage is explicitly evaluated in
    # FP32.  Keep the conversion here: this is the first mandated rounding.
    x = tl.load(a_ptr + offsets, cache_modifier=".cg").to(tl.float32)
    y = tl.load(b_ptr + offsets, cache_modifier=".cg")
    gate = (x * tl.sigmoid(x)).to(tl.bfloat16)

    # Perform stage 2 directly as packed BF16 arithmetic.  Besides making the
    # second rounding explicit, this avoids unpacking each pair to FP32.
    result = tl.inline_asm_elementwise(
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        [gate, y],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )
    tl.store(out_ptr + offsets, result, cache_modifier=".cs")


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        out = torch.empty_like(a)
        n = a.numel()
        _swiglu_kernel[(n // 512,)](
            a, b, out, BLOCK=512, num_warps=1,
        )
        return out

