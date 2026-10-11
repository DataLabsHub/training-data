"""
reference_model.py — task_id: cutlass_normalization_pr3009

Domain      : normalization
Task type   : fusion   (optimization track: fusion_and_layout)
Task mode   : tensor   (forward-only, no gradients)
Precision   : input fp16 / accumulate fp32 / output fp16
Calibration : a100_pcie_80gb_cu124_torch260  (NVIDIA A100 80GB PCIe, sm_80, Ampere)

Provenance
----------
Distilled from NVIDIA/cutlass#3009 into a portable A100 forward-LayerNorm fusion
task.  The PR supplies the *idea* (fuse the row reduction, the normalization and
the per-hidden affine so the (T, H) activation is not repeatedly round-tripped
through global memory) and nothing else: no CUTLASS header is imported here and
CUTLASS is neither a trusted implementation nor the performance denominator.

The hotspot
-----------
For an activation `x` of shape (T, H) a textbook PyTorch LayerNorm walks the
tensor several times: once for the mean, once for the centered residual, once
for the variance, once for the scale, once for the affine.  Each pass streams
T*H fp16 elements through HBM and writes an fp32 temporary that is immediately
re-read.  The whole op is DRAM-bound, so the achievable floor is roughly
"read x once, write y once" -- a fused row-per-block kernel with an fp32
Welford/two-pass reduction in registers/shared memory should approach it.

`Model` below is the NAIVE, obviously-correct reference.  It is deliberately
written as the multi-pass version so that it is both the correctness Oracle and
an honest timing denominator that a real fused kernel can beat.
`baseline(...)` is the strong library frontier (torch.nn.functional.layer_norm).

Math contract (exactly what a submission must reproduce)
--------------------------------------------------------
    accumulate in fp32:
        mu   = mean_h(x)                      # over the last axis, length H
        var  = mean_h((x - mu)^2)             # BIASED: divide by H, not H-1
        y    = (x - mu) * rsqrt(var + 1e-5) * weight + bias
    then cast `y` to fp16 exactly once, at the very end.

eps = 1e-5 lives *inside* the rsqrt.  Only `y` (shape (T, H), dtype fp16) is
returned -- mu/rsigma are not part of the output contract.
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Task constants
# --------------------------------------------------------------------------- #

EPS = 1e-5

TASK_MODE = "tensor"

# Output is fp16.  Normalized values reach ~|4| and `weight` reaches 1.25, so the
# output magnitude reaches ~5 where one fp16 ulp is ~4.9e-3; the absolute
# tolerance has to admit a single-ulp disagreement in the final rounding.
TASK_RTOL = 2e-3
TASK_ATOL = 5e-3

# Forward-only task: nothing is differentiated.
GRAD_INPUTS = []

# Single output tensor.
OUTPUT_INDEX = 0

# Representative PUBLIC shapes (T, H).
WORKLOAD = [
    (4096, 2048),
    (8192, 4096),
    (16384, 4096),
    (32768, 8192),
]

BASELINE_CONTRACT = {
    "input_dtype": "fp16",
    "output_dtype": "fp16",
    "accumulation_dtype": "fp32",
    "allow_tf32": "false",
    "note": (
        "Row-wise LayerNorm over the last axis of an (T, H) fp16 activation. "
        "Reductions and the affine must accumulate in fp32; the result is cast "
        "to fp16 exactly once at the end. Variance is BIASED (divide by H) and "
        "eps=1e-5 is added inside the rsqrt. No TF32 tensor-core paths are "
        "involved (this is a pure memory-bound reduction, no GEMM). The strong "
        "library frontier is torch.nn.functional.layer_norm; `baseline` is a "
        "pure adapter that only supplies normalized_shape=(H,) and eps."
    ),
}

# Legal shape domain, mirrored from the task declaration.
#   T: 4096 .. 32768 step 128
#   H: 2048 ..  8192 step 8
_T_CHOICES = list(range(4096, 32768 + 1, 128))
_H_CHOICES = list(range(2048, 8192 + 1, 8))

# Below this per-row biased variance the inverse-affine invariant loses its
# numerical footing, so make_inputs repairs any offending row.
_MIN_ROW_VAR = 0.25


# --------------------------------------------------------------------------- #
# Naive reference (the correctness Oracle and the timing denominator)
# --------------------------------------------------------------------------- #

class Model(nn.Module):
    """Plain, multi-pass PyTorch LayerNorm. Simple and correct, NOT optimized.

    Every intermediate (`centered`, `var`, `normed`, ...) is materialized as a
    full (T, H) or (T, 1) fp32 tensor, which is precisely the global-memory
    traffic that a fused kernel is expected to eliminate.
    """

    def __init__(self):
        super().__init__()
        self.eps = EPS

    def forward(self, x, weight, bias):
        # Accumulate in fp32 (input is fp16).
        xf = x.float()

        # Pass 1: per-row mean.
        mu = xf.mean(dim=-1, keepdim=True)

        # Pass 2: centered residual (materialized).
        centered = xf - mu

        # Pass 3: per-row BIASED variance (divide by H).
        var = (centered * centered).mean(dim=-1, keepdim=True)

        # Pass 4: reciprocal standard deviation, eps inside the rsqrt.
        rsigma = torch.rsqrt(var + self.eps)

        # Pass 5: normalize (materialized).
        normed = centered * rsigma

        # Pass 6: per-hidden affine.
        y = normed * weight.float() + bias.float()

        # Single, final cast to fp16.
        return y.to(torch.float16)


def get_init_inputs():
    """Arguments forwarded to Model(...). This model is stateless."""
    return []


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def _alternating_row(H, device):
    """Deterministic (-1, +1, -1, +1, ...) row: exact in fp16, mean 0, var 1."""
    idx = torch.arange(H, device=device, dtype=torch.float32)
    return torch.where(idx % 2 == 0, -1.0, 1.0)


def make_inputs(shape, device, seed=0):
    """Build [x, weight, bias] for a (T, H) workload.

    `seed` fully determines the data (and changing it changes the data).

    Guarantees relied upon by `invariants`:
      * weight in [0.75, 1.25]  ->  |weight| >= 0.5, so dividing by it is safe.
      * bias   in [-0.25, 0.25] ->  the inverse affine stays well conditioned.
      * every row of x, measured AFTER the fp16 cast, has biased variance
        >= 0.25.  Any row that violates this is deterministically replaced by an
        alternating -1/+1 pattern (variance exactly 1). With var >= 0.25 the
        eps=1e-5 term perturbs the unit mean-square of the normalized row by at
        most ~4e-5, so the invariant thresholds below are dominated by fp16
        rounding rather than by eps.
    """
    T, H = int(shape[0]), int(shape[1])
    device = torch.device(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 100003 + 12345)

    # Non-constant, unit-ish rows with a mild per-row shift/gain so the mean and
    # variance reductions are not trivially zero/one before normalization.
    x32 = torch.randn(T, H, generator=gen, device=device, dtype=torch.float32)
    row_gain = 0.75 + 0.5 * torch.rand(T, 1, generator=gen, device=device, dtype=torch.float32)
    row_shift = -1.0 + 2.0 * torch.rand(T, 1, generator=gen, device=device, dtype=torch.float32)
    x32 = x32 * row_gain + row_shift

    x = x32.to(torch.float16)

    # Enforce the row-variance floor on the POST-CAST data.
    xf = x.float()
    row_mu = xf.mean(dim=-1, keepdim=True)
    row_var = ((xf - row_mu) ** 2).mean(dim=-1)
    bad = row_var < _MIN_ROW_VAR
    if bool(bad.any()):
        alt = _alternating_row(H, device).to(torch.float16)
        x[bad] = alt
        # Cheap re-assertion: the replacement row has biased variance 1 exactly.
        xf = x.float()
        row_mu = xf.mean(dim=-1, keepdim=True)
        row_var = ((xf - row_mu) ** 2).mean(dim=-1)
        assert bool((row_var >= _MIN_ROW_VAR * 0.99).all()), "row variance floor not met"

    # weight in [0.75, 1.25], bias in [-0.25, 0.25].
    weight = (0.75 + 0.5 * torch.rand(H, generator=gen, device=device,
                                      dtype=torch.float32)).to(torch.float16)
    bias = (-0.25 + 0.5 * torch.rand(H, generator=gen, device=device,
                                     dtype=torch.float32)).to(torch.float16)

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    return [x, weight, bias]


def sample_shapes(rng, n):
    """UNSEEN shapes drawn from the legal domain (hidden-shape tripwire).

    Never returns a shape that appears in WORKLOAD, and never repeats.
    """
    public = set(WORKLOAD)
    seen = set()
    out = []
    guard = 0
    while len(out) < int(n) and guard < 4000:
        guard += 1
        T = _T_CHOICES[rng.randrange(len(_T_CHOICES))]
        H = _H_CHOICES[rng.randrange(len(_H_CHOICES))]
        cand = (T, H)
        if cand in public or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


# --------------------------------------------------------------------------- #
# Independent oracle (different numerical path from Model)
# --------------------------------------------------------------------------- #

def independent_oracle(ins, shape):
    """float64 recompute via the sum / sum-of-squares identity.

    Deliberately NOT Model's two-pass fp32 formulation: this uses
    var = E[x^2] - E[x]^2 accumulated in double precision, so agreement is real
    evidence rather than a re-run of the same code.

    Called by the harness as independent_oracle(ins, shape) where `ins` is the
    complete list returned by make_inputs.
    """
    x, weight, bias = ins[0], ins[1], ins[2]
    H = int(x.shape[-1])

    xd = x.double()
    s1 = xd.sum(dim=-1, keepdim=True)
    s2 = (xd * xd).sum(dim=-1, keepdim=True)
    mu = s1 / H
    var = s2 / H - mu * mu
    var = torch.clamp(var, min=0.0)

    inv = 1.0 / torch.sqrt(var + EPS)
    y = (xd - mu) * inv * weight.double() + bias.double()
    return y.to(torch.float16)


# --------------------------------------------------------------------------- #
# Metamorphic invariants (independent of Model's code path)
# --------------------------------------------------------------------------- #

def invariants(out, ins, shape):
    """Algebraic properties any correct output must satisfy.

    None of these re-run `Model`. The affine is *undone* analytically:

        z = (y - bias) / weight

    which, for a correct LayerNorm, is the normalized row. A normalized row must
    have mean 0 and mean-square 1 up to (a) the eps=1e-5 inside the rsqrt and
    (b) fp16 rounding of `y` plus the inverse-affine division.

    Because make_inputs guarantees per-row biased variance >= 0.25, the eps term
    shrinks the mean-square by at most eps/(var+eps) ~ 4e-5 -- three orders of
    magnitude under the 2e-2 budget below, which is therefore spent almost
    entirely on fp16 output rounding and the inverse-affine error.
    """
    checks = []

    y = out[OUTPUT_INDEX] if isinstance(out, (list, tuple)) else out
    x, weight, bias = ins[0], ins[1], ins[2]
    T, H = int(shape[0]), int(shape[1])

    # 1) Shape + dtype contract.
    ok_shape = tuple(y.shape) == (T, H)
    ok_dtype = (y.dtype == torch.float16)
    checks.append((
        "shape_dtype",
        bool(ok_shape and ok_dtype),
        f"shape={tuple(y.shape)} (want {(T, H)}), dtype={y.dtype} (want torch.float16)",
    ))
    if not ok_shape:
        # Everything downstream would be meaningless.
        checks.append(("finite", False, "skipped: wrong shape"))
        checks.append(("row_mean_zero", False, "skipped: wrong shape"))
        checks.append(("row_meansquare_one", False, "skipped: wrong shape"))
        return checks

    # 2) Finiteness: no NaN/Inf may leak out of the reduction or the rsqrt.
    yf = y.float()
    n_bad = int((~torch.isfinite(yf)).sum().item())
    checks.append((
        "finite",
        n_bad == 0,
        f"non-finite outputs: {n_bad} / {T * H}",
    ))

    # Undo the affine analytically. |weight| >= 0.5 by construction.
    w = weight.float()
    b = bias.float()
    z = (yf - b) / w

    # 3) Every row of the normalized activation has (near) zero mean.
    row_mean = z.mean(dim=-1)
    max_abs_mean = float(row_mean.abs().max().item())
    checks.append((
        "row_mean_zero",
        max_abs_mean <= 5e-3,
        f"max_row |mean(z)| = {max_abs_mean:.3e} (limit 5e-3)",
    ))

    # 4) Every row of the normalized activation has (near) unit mean-square.
    row_ms = (z * z).mean(dim=-1)
    max_abs_ms_err = float((row_ms - 1.0).abs().max().item())
    checks.append((
        "row_meansquare_one",
        max_abs_ms_err <= 2e-2,
        f"max_row |mean(z*z) - 1| = {max_abs_ms_err:.3e} (limit 2e-2)",
    ))

    return checks


# --------------------------------------------------------------------------- #
# Strong library baseline (the honest frontier)
# --------------------------------------------------------------------------- #

def baseline(x, weight, bias, init_args=None):
    """torch.nn.functional.layer_norm — the library frontier to beat.

    Pure adapter: `normalized_shape` and `eps` are not make_inputs arguments, so
    this wrapper exists solely to supply them. No authored math, no fallback.
    `init_args` is ignored (Model is stateless).
    """
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps=EPS)
