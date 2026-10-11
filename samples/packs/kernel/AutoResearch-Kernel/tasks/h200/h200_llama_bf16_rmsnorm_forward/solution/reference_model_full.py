# reference_model.py
#
# Task: h200_llama_bf16_rmsnorm_forward
#   domain    : normalization
#   task_type : fusion
#   task_mode : tensor
#   track     : fusion_and_layout
#
# Generic forward-only RMSNorm workload for contiguous BF16 transformer
# activations on an NVIDIA H200 (sm_90, Hopper). A row of BF16 activations is
# reduced in FP32, scaled by rsqrt(mean_square + eps), cast back to BF16, and
# multiplied by a BF16 per-channel gain. The optimization opportunity is fusing
# the reduction, the reciprocal-root scaling, the BF16 cast, and the gain
# multiply into a single pass so that no full-size intermediate tensor is ever
# written to HBM.
#
# `Model` below is the NAIVE, obviously-correct plain-PyTorch reference: it is
# both the correctness Oracle and the timing denominator. It is intentionally
# unoptimized (materializes FP32 intermediates). `baseline(...)` calls the
# pinned third-party kernel, which is the official performance baseline for
# this task.

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_ID = "h200_llama_bf16_rmsnorm_forward"

TASK_MODE = "tensor"

# BF16 storage of both the normalized activation and the gain product means the
# achievable agreement is ~1 BF16 ulp; 2e-2 relative/absolute is the honest band.
TASK_RTOL = 2e-2
TASK_ATOL = 2e-2

GRAD_INPUTS = []

OUTPUT_INDEX = 0

MEM_WEIGHT = 0.5

EPS = 1e-6
OFFSET = 0.0
CASTING_MODE = "llama"

WORKLOAD = [
    (4096, 2048),
    (8192, 4096),
    (16384, 4096),
    (16384, 8192),
]

# Declared legal shape domain (axes are independent choices).
T_CHOICES = (4096, 8192, 12288, 16384)
H_CHOICES = (2048, 3072, 4096, 5120, 6144, 7168, 8192)

BASELINE_CONTRACT = {
    "input_dtype": "bf16",
    "output_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "allow_tf32": False,
    "note": (
        "Llama casting semantics: the row mean-square is accumulated in FP32 over "
        "the H axis, rstd = rsqrt(mean_square + 1e-6) is computed in FP32, the "
        "normalized activation is cast to BF16 FIRST, and only then multiplied by "
        "the BF16 per-channel gain (offset=0.0, casting_mode='llama'). No mean "
        "subtraction, no TF32 paths, no input mutation, and the output is a fresh "
        "contiguous BF16 [T, H] tensor that does not alias input storage."
    ),
}

# Reserved rows used by the metamorphic invariants:
#   row 1 = 2 * row 0   (positive-scale invariance of RMSNorm)
#   row 2 = -row 0      (odd symmetry of RMSNorm)
_RESERVED_ROWS = 3

# A row must carry enough signal that rstd is well-conditioned.
_MIN_MEAN_SQUARE = 0.25


# ---------------------------------------------------------------------------
# Naive reference (correctness Oracle + timing denominator)
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """Naive plain-PyTorch Llama-style RMSNorm forward.

    forward(x, weight):
        x      : contiguous BF16 [T, H]
        weight : contiguous BF16 [H]
        returns: contiguous BF16 [T, H]

    Deliberately unfused: the FP32 scaled activation is materialized before the
    BF16 cast and the gain multiply.
    """

    def __init__(self):
        super().__init__()
        self.eps = EPS

    def forward(self, x, weight):
        x_fp32 = x.float()
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_square + self.eps)
        normalized = (x_fp32 * rstd).to(torch.bfloat16)
        return (normalized * weight).contiguous()


def get_init_inputs():
    return []


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def _alternating_row(H, device):
    """Deterministic nonconstant fallback row with mean-square exactly 1."""
    row = torch.ones(H, device=device, dtype=torch.float32)
    row[1::2] = -1.0
    return row


def make_inputs(shape, device, seed=0):
    """Deterministic per-seed inputs; both x and weight change with the seed.

    Guarantees:
      * contiguous BF16 x [T, H] and BF16 weight [H]
      * finite, nonconstant rows whose POST-CAST mean-square is >= 0.25
        (a deficient row is replaced by a deterministic +/-1 alternating row)
      * weight is positive and channel-varying in [0.75, 1.25]
      * reserved rows: x[1] == 2 * x[0], x[2] == -x[0]
    """
    T, H = int(shape[0]), int(shape[1])
    device = torch.device(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 9176 + 12345)

    x = torch.randn(T, H, generator=gen, device=device, dtype=torch.float32)
    # Mild channel-dependent tilt so the reduction is not perfectly homogeneous.
    tilt = 1.0 + 0.25 * torch.cos(
        torch.arange(H, device=device, dtype=torch.float32) * (3.0 / max(H, 1))
    )
    x = (x * tilt).to(torch.bfloat16)

    # Repair rows whose post-cast mean-square is too small (measured in FP32).
    ms = x.float().pow(2).mean(dim=-1)
    deficient = ~torch.isfinite(ms) | (ms < _MIN_MEAN_SQUARE)
    if bool(deficient.any()):
        fallback = _alternating_row(H, device).to(torch.bfloat16)
        x[deficient] = fallback

    # Reserved rows are derived AFTER repair so their properties hold exactly.
    if T >= _RESERVED_ROWS:
        row0 = x[0].clone()
        x[1] = row0 * 2.0   # exact in BF16 (power-of-two scaling)
        x[2] = -row0

    weight = (
        0.75
        + 0.5 * torch.rand(H, generator=gen, device=device, dtype=torch.float32)
    ).to(torch.bfloat16)

    return [x.contiguous(), weight.contiguous()]


def sample_shapes(rng, n):
    """Unseen (non-WORKLOAD) shapes drawn only from the declared T/H choices."""
    workload = {(int(t), int(h)) for (t, h) in WORKLOAD}
    candidates = [
        (t, h)
        for t in T_CHOICES
        for h in H_CHOICES
        if (t, h) not in workload
    ]
    n = int(n)
    if n <= 0:
        return []
    idx = list(range(len(candidates)))
    # rng may be a random.Random or a numpy Generator; support both.
    try:
        rng.shuffle(idx)
    except TypeError:
        idx = list(rng.permutation(len(candidates)))
    return [candidates[i] for i in idx[:min(n, len(candidates))]]


# ---------------------------------------------------------------------------
# Independent oracle (written straight from the RMSNorm formula)
# ---------------------------------------------------------------------------

def independent_oracle(ins, shape):
    """Recompute the expected output from the definition, in float64.

    Independent of Model: uses an explicit float64 sum of squares and a float64
    square root (not rsqrt), then casts the normalized activation to BF16 before
    a separate gain multiplication and final BF16 cast. Never calls Model or
    baseline.
    """
    x, weight = ins[0], ins[1]
    T, H = int(shape[0]), int(shape[1])

    w64 = weight.double()
    out = torch.empty((T, H), device=x.device, dtype=torch.bfloat16)

    chunk = 2048
    for start in range(0, T, chunk):
        stop = min(start + chunk, T)
        blk = x[start:stop].double()
        sum_sq = blk.pow(2).sum(dim=-1, keepdim=True)
        mean_square = sum_sq / float(H)
        rstd = 1.0 / torch.sqrt(mean_square + float(EPS))
        normalized_bf16 = (blk * rstd).to(torch.bfloat16)          # cast first
        out[start:stop] = (normalized_bf16.double() * w64).to(torch.bfloat16)

    return out.contiguous()


# ---------------------------------------------------------------------------
# Metamorphic invariants (no Model re-run)
# ---------------------------------------------------------------------------

def invariants(out, ins, shape):
    """Algebraic properties a correct RMSNorm output must satisfy.

    Independent of Model's code path: only the candidate output and the raw
    inputs are inspected.
    """
    x, weight = ins[0], ins[1]
    T, H = int(shape[0]), int(shape[1])
    checks = []

    # 1) Shape, dtype, contiguity, finiteness.
    ok_shape = tuple(out.shape) == (T, H)
    ok_dtype = out.dtype == torch.bfloat16
    ok_contig = out.is_contiguous()
    ok_finite = bool(torch.isfinite(out.float()).all())
    checks.append((
        "shape_dtype_finite",
        bool(ok_shape and ok_dtype and ok_contig and ok_finite),
        f"shape={tuple(out.shape)} expected={(T, H)} dtype={out.dtype} "
        f"contiguous={ok_contig} all_finite={ok_finite}",
    ))

    o = out.float()

    # 2) Positive-scale invariance: RMSNorm(2*r) == RMSNorm(r).
    #    Reserved rows guarantee x[1] == 2 * x[0] exactly in BF16.
    if T >= _RESERVED_ROWS:
        ref = o[0]
        scale = float(ref.abs().max().item()) + 1e-6
        d_scale = float((o[1] - ref).abs().max().item())
        passed = d_scale <= 3e-2 * scale
        checks.append((
            "positive_scale_invariance",
            bool(passed),
            f"max|out[1]-out[0]|={d_scale:.4e} tol={3e-2 * scale:.4e} "
            "(x[1] == 2*x[0])",
        ))

        # 3) Odd symmetry: RMSNorm(-r) == -RMSNorm(r).
        d_odd = float((o[2] + ref).abs().max().item())
        passed_odd = d_odd <= 3e-2 * scale
        checks.append((
            "odd_symmetry",
            bool(passed_odd),
            f"max|out[2]+out[0]|={d_odd:.4e} tol={3e-2 * scale:.4e} "
            "(x[2] == -x[0])",
        ))
    else:
        checks.append((
            "positive_scale_invariance",
            False,
            f"T={T} is smaller than the {_RESERVED_ROWS} reserved rows",
        ))
        checks.append((
            "odd_symmetry",
            False,
            f"T={T} is smaller than the {_RESERVED_ROWS} reserved rows",
        ))

    # 4) Unit RMS after removing the gain: z = out / weight must have row
    #    mean-square == 1 (the defining property of RMS normalization).
    z = o / weight.float().clamp_min(1e-3)
    row_ms = z.pow(2).mean(dim=-1)
    max_dev = float((row_ms - 1.0).abs().max().item())
    checks.append((
        "unit_row_mean_square_after_degain",
        bool(math.isfinite(max_dev) and max_dev <= 4e-2),
        f"max|mean(z^2)-1|={max_dev:.4e} tol=4.0e-02 over {T} rows",
    ))

    return checks


# ---------------------------------------------------------------------------
# Strong library baseline (official performance baseline; fails closed)
# ---------------------------------------------------------------------------

def baseline(x, weight, init_args=None):
    """Pinned third-party RMSNorm forward: the official performance baseline.

    Calls exactly:
        LigerRMSNormFunction.apply(x, weight, 1e-6, 0.0, 'llama', False, None)

    No alternate signatures are probed and no fallback path exists: if the
    pinned implementation is missing or incompatible, this raises.
    """
    if init_args not in (None, []):
        raise ValueError(
            f"baseline() takes no constructor arguments; got init_args={init_args!r}"
        )

    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    except Exception as exc:  # missing / broken pin -> fail closed
        raise RuntimeError(
            "pinned baseline liger_kernel.ops.rms_norm.LigerRMSNormFunction is "
            f"unavailable: {exc!r}"
        ) from exc

    try:
        return LigerRMSNormFunction.apply(x, weight, 1e-6, 0.0, "llama", False, None)
    except TypeError as exc:  # incompatible pinned signature -> fail closed
        raise RuntimeError(
            "pinned baseline LigerRMSNormFunction.apply is incompatible with the "
            "required call LigerRMSNormFunction.apply(x, weight, 1e-6, 0.0, "
            f"'llama', False, None): {exc!r}"
        ) from exc
