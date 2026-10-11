import torch
import torch.nn as nn

TASK_MODE = "tensor"
TASK_RTOL = 1e-2
TASK_ATOL = 1e-2

CASES = {
    0: (2048, 4096),
    1: (4096, 11008),
    2: (8192, 8192),
    3: (16384, 4096),
    4: (2048, 32768),
    5: (24320, 4096),
    6: (12160, 8192),
    7: (6912, 14336),
    8: (4864, 18432),
    9: (3456, 28672),
    10: (2944, 32768),
    11: (6272, 13824),
}

WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]


class Model(nn.Module):
    """Naive correct SwiGLU/SiLU-multiply activation: silu(a*gate_multiplier) * b."""

    def __init__(self, gate_multiplier=1.702):
        super().__init__()
        self.gate_multiplier = gate_multiplier

    def forward(self, a, b):
        x = a.float() * self.gate_multiplier
        y = x * torch.sigmoid(x) * b.float()
        return y.to(torch.bfloat16)


def make_inputs(shape, device, seed=0):
    case_id = shape[0]
    T, H = CASES[case_id]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Deterministic, seed-varying, finite, bounded contiguous bf16 tensors.
    a = torch.rand(T, H, generator=g, dtype=torch.float32) * 2.0 - 1.0
    b = torch.rand(T, H, generator=g, dtype=torch.float32) * 2.0 - 1.0
    # Inject exact-zero and near-zero sentinels so hidden-code paths are exercised.
    a[:, 0] = 0.0
    b[:, 1] = 0.0
    a[0, 1] = 1e-6
    b[1, 0] = -1e-6
    a = a.contiguous().to(torch.bfloat16)
    b = b.contiguous().to(torch.bfloat16)
    return [a, b]


def get_init_inputs():
    return [1.702]


OUTPUT_INDEX = 0
GRAD_INPUTS = []

BASELINE_CONTRACT = {
    "input_dtype": "bf16",
    "output_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "allow_tf32": False,
    "note": "fp32 composite x=a.float()*1.702, silu(x)*b.float(), single bf16 cast.",
}


def reduction_length(shape):
    return 1


def sample_shapes(rng, n):
    """Unseen case-id tuples drawn from the held-out half of the domain."""
    return [(int(rng.choice([5, 6, 7, 8, 9, 10, 11])),) for _ in range(n)]


def invariants(out, ins, shape):
    a, b = ins[0], ins[1]
    T, H = CASES[shape[0]]
    checks = []

    checks.append(("shape_matches", tuple(out.shape) == (T, H),
                   f"out {tuple(out.shape)} vs expected {(T, H)}"))

    checks.append(("bf16_dtype", out.dtype == torch.bfloat16,
                   f"dtype {out.dtype}"))

    fin = torch.isfinite(out.float()).all().item()
    checks.append(("finite", bool(fin), "all output values finite"))

    # Gate multiplier and sigmoid are strictly positive, so the true fp32 result
    # has the sign of a*b. bf16 rounding keeps that sign except for values
    # rounding to zero, hence a nonnegativity check on out*a*b up to slack.
    slack = 4.0 * torch.finfo(torch.bfloat16).eps
    prod = out.float() * a.float() * b.float()
    ok = bool((prod >= -slack).all().item())
    checks.append(("sign_consistency", ok,
                   "out*a*b >= -slack for all elements (gate>0, sigmoid>0)"))

    return checks


def independent_oracle(ins, shape):
    """Independent closed-form recompute for the swiglu composite."""
    a, b = ins[0], ins[1]
    return (a.float() * 1.702 * torch.sigmoid(a.float() * 1.702) * b.float()).to(torch.bfloat16)

MEM_WEIGHT = 0.0
