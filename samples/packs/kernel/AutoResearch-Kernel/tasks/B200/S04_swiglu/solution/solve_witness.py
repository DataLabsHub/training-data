import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    G_ptr, U_ptr, O_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * (BLOCK * VEC)

    for i in tl.static_range(VEC):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        g = tl.load(G_ptr + offs, mask=mask, other=0).to(tl.float32)
        u = tl.load(U_ptr + offs, mask=mask, other=0).to(tl.float32)

        # fp32 nonlinear math: silu(g) * u, single bf16 round at the store
        s = g * tl.sigmoid(g)
        o = s * u

        tl.store(O_ptr + offs, o.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    """Fused SwiGLU activation epilogue: one streaming pass over bf16 operands.

    Loads gate/up in bf16, does sigmoid + both multiplies in fp32, and issues
    exactly one bf16 store — 4 bytes moved per element instead of the naive
    reference's six HBM passes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
        g = gate_proj.contiguous()
        u = up_proj.contiguous()
        out = torch.empty_like(g, dtype=torch.bfloat16)

        n = g.numel()
        if n == 0:
            return out

        BLOCK = 1024
        VEC = 4
        per_prog = BLOCK * VEC
        grid = (triton.cdiv(n, per_prog),)

        _swiglu_kernel[grid](
            g, u, out,
            n,
            BLOCK=BLOCK,
            VEC=VEC,
            num_warps=4,
            num_stages=1,
        )
        return out
