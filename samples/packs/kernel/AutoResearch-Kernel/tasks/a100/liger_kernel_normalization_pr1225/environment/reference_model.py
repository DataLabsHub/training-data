# reference_model.py
#
# Task: liger_kernel_normalization_pr1225
#   domain      : normalization
#   task_type   : fusion            (task_mode = 'tensor', forward-only)
#   track       : fusion_and_layout
#   frontier    : torch.compile
#   calibration : a100_pcie_80gb_cu124_torch260 (NVIDIA A100 80GB PCIe, sm_80, Ampere)
#
# Modulated RMSNorm (FiLM / AdaRMSNorm style), forward only:
#
#     rstd = rsqrt( mean_h(x*x) + eps )                       (fp32)
#     y    = (x * rstd) * weight * (1 + scale) + shift        (fp32)
#     out  = y.to(bfloat16)                                   (single, final cast)
#
# weight / scale / shift are MANDATORY per-hidden vectors of shape (H,), broadcast
# globally over rows.  eps is fixed at 1e-6.  All reduction and modulation
# arithmetic happens in fp32; there is exactly one narrowing cast, at the very end.
#
# Provenance note: the fusion pattern is distilled from linkedin/Liger-Kernel#1225.
# This file deliberately does NOT import or depend on liger_kernel -- the upstream PR
# is provenance only, and is not a trusted implementation, baseline, performance
# result, or certification.  Optional weight/shift, non-global modulation broadcasts,
# alternate casting modes, and the backward pass are out of scope for this task.

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

EPS = 1e-6

TASK_MODE = "tensor"

# bf16 output: half-ulp is ~2^-9 (~0.2%) relative, so allow a couple of ulps.
TASK_RTOL = 2e-2
TASK_ATOL = 1e-2

GRAD_INPUTS = []          # forward-only task
OUTPUT_INDEX = 0          # single output: y, shape (T, H), bf16

# Public (visible) workloads.
WORKLOAD = [
    (4096, 2048),
    (8192, 4096),
    (16384, 4096),
    (8192, 8192),
]

# Legal shape domain (mirrors the task axes spec).
T_CHOICES = list(range(4096, 16384 + 1, 256))   # 4096 .. 16384 step 256
H_CHOICES = list(range(2048, 8192 + 1, 8))      # 2048 .. 8192  step 8

BASELINE_CONTRACT = {
    "input_dtype": "bf16",
    "output_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "allow_tf32": False,
    "note": (
        "Forward-only modulated RMSNorm. Inputs: contiguous bf16 x of shape (T,H) and "
        "contiguous bf16 weight/scale/shift vectors of shape (H,). eps is fixed at 1e-6. "
        "The mean-square reduction over H and ALL modulation arithmetic "
        "((x*rstd)*weight*(1+scale)+shift) must be performed in fp32; the result is cast "
        "to bf16 exactly once, at the end. No intermediate bf16 rounding is permitted. "
        "Only y is returned, shape (T,H), dtype bf16. No matmuls are involved, so TF32 "
        "must not be used anywhere. No gradients are required or verified."
    ),
}


# ---------------------------------------------------------------------------
# NAIVE reference model -- the correctness Oracle and the timing denominator.
# Written for obvious correctness, not for speed: every step is a separate,
# fully materialized plain-PyTorch op.
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """Naive plain-PyTorch modulated RMSNorm (FiLM / AdaRMSNorm), forward only."""

    def __init__(self):
        super().__init__()
        self.eps = EPS

    def forward(self, x, weight, scale, shift):
        # Promote everything to fp32 -- all math below is fp32.
        x_f = x.float()
        w_f = weight.float()
        s_f = scale.float()
        b_f = shift.float()

        # RMS reduction over the hidden axis.
        sq = x_f * x_f                                   # (T, H)
        mean_sq = sq.mean(dim=-1, keepdim=True)          # (T, 1)
        rstd = torch.rsqrt(mean_sq + self.eps)           # (T, 1)

        # Normalize, then apply the modulation (gain then shift).
        x_hat = x_f * rstd                               # (T, H)
        gain = w_f * (1.0 + s_f)                         # (H,)
        y = x_hat * gain                                 # (T, H)
        y = y + b_f                                      # (T, H)

        # Exactly one narrowing cast.
        return y.to(torch.bfloat16)


def get_init_inputs():
    return []


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def make_inputs(shape, device, seed=0):
    """Deterministic inputs for shape == (T, H).

    Returns [x, weight, scale, shift]:
      x      : (T, H) bf16, contiguous, row-varying magnitudes, every row has
               mean(x.float()**2) >= 0.25 AFTER the bf16 cast.
      weight : (H,) bf16 in [0.75, 1.25]
      scale  : (H,) bf16 in [-0.25, 0.25]
      shift  : (H,) bf16 in [-0.25, 0.25]  (nontrivial, not all-zero)

    Hence |weight * (1 + scale)| >= 0.75 * 0.75 = 0.5625 elementwise, which keeps
    the invariant checks (which divide by that gain) well conditioned.
    """
    T, H = int(shape[0]), int(shape[1])
    device = torch.device(device)

    gen = torch.Generator(device=device)
    gen.manual_seed((int(seed) * 1_000_003 + T * 9176 + H * 31 + 12345) & 0x7FFFFFFF)

    # --- modulation vectors -------------------------------------------------
    w32 = 0.75 + 0.50 * torch.rand(H, generator=gen, device=device, dtype=torch.float32)
    s32 = -0.25 + 0.50 * torch.rand(H, generator=gen, device=device, dtype=torch.float32)
    b32 = -0.25 + 0.50 * torch.rand(H, generator=gen, device=device, dtype=torch.float32)

    weight = w32.to(torch.bfloat16)
    scale = s32.to(torch.bfloat16)
    shift = b32.to(torch.bfloat16)

    # bf16 rounding is exact at 0.75 / -0.25, but clamp defensively so the
    # gain bound |w*(1+s)| >= 0.5625 holds on the *rounded* values.
    weight = torch.where(weight.float() < 0.75,
                         torch.full_like(weight, 0.75), weight)
    scale = torch.where(scale.float() < -0.25,
                        torch.full_like(scale, -0.25), scale)
    scale = torch.where(scale.float() > 0.25,
                        torch.full_like(scale, 0.25), scale)

    # --- activations with row-varying magnitude -----------------------------
    base = torch.randn(T, H, generator=gen, device=device, dtype=torch.float32)
    row_scale = 0.75 + 1.75 * torch.rand(T, 1, generator=gen, device=device,
                                         dtype=torch.float32)
    x = (base * row_scale).to(torch.bfloat16)

    # Guarantee mean(x.float()**2) >= 0.25 per row AFTER the cast.
    xf = x.float()
    row_ms = (xf * xf).mean(dim=-1)                                # (T,)
    bad = row_ms < 0.25
    if bool(bad.any()):
        idx = torch.arange(H, device=device)
        alt = (1.0 - 2.0 * (idx % 2).to(torch.float32)).to(torch.bfloat16)  # +1,-1,...
        x = torch.where(bad.unsqueeze(1), alt.unsqueeze(0).expand(T, H), x)

    x = x.contiguous()
    return [x, weight.contiguous(), scale.contiguous(), shift.contiguous()]


def sample_shapes(rng, n):
    """Unseen shapes drawn from the legal domain (hidden-shape tripwire)."""
    public = set(WORKLOAD)
    out = []
    seen = set()
    guard = 0
    while len(out) < n and guard < 4000:
        guard += 1
        s = (rng.choice(T_CHOICES), rng.choice(H_CHOICES))
        if s in public or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Independent float64 oracle -- does NOT call Model.
# Uses linalg.vector_norm plus explicit sqrt/divide (no rsqrt, no mean),
# and the fused gain weight*(1+scale).
# ---------------------------------------------------------------------------

def independent_oracle(ins, shape):
    x, weight, scale, shift = ins[0], ins[1], ins[2], ins[3]
    H = int(shape[1])

    xd = x.double()
    # ||row||_2, then mean-square = ||row||^2 / H, then explicit sqrt + divide.
    nrm = torch.linalg.vector_norm(xd, ord=2, dim=-1, keepdim=True)   # (T, 1)
    mean_sq = (nrm * nrm) / float(H)                                  # (T, 1)
    denom = torch.sqrt(mean_sq + EPS)                                 # (T, 1)
    x_hat = xd / denom                                                # (T, H)

    gain = weight.double() * (1.0 + scale.double())                   # (H,)
    y = x_hat * gain + shift.double()
    return y.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Metamorphic invariants -- strong, independent evidence, no Model rerun.
#
# The modulation is invertible elementwise because |weight*(1+scale)| >= 0.5625,
# so we can undo it:  z = (y - shift) / (weight * (1 + scale)).
# For a correct kernel z must equal x scaled by a single POSITIVE per-row
# constant rstd, and must have unit mean-square per row.
# ---------------------------------------------------------------------------

def invariants(out, ins, shape):
    y = out[OUTPUT_INDEX] if isinstance(out, (tuple, list)) else out
    x, weight, scale, shift = ins[0], ins[1], ins[2], ins[3]
    T, H = int(shape[0]), int(shape[1])
    checks = []

    # ---- 1. structural: shape, dtype, finiteness ---------------------------
    shape_ok = tuple(y.shape) == (T, H)
    dtype_ok = (y.dtype == torch.bfloat16)
    finite_ok = bool(torch.isfinite(y.float()).all())
    checks.append((
        "shape_dtype_finite",
        bool(shape_ok and dtype_ok and finite_ok),
        f"shape={tuple(y.shape)} (want {(T, H)}), dtype={y.dtype} (want torch.bfloat16), "
        f"all_finite={finite_ok}",
    ))
    if not (shape_ok and dtype_ok and finite_ok):
        # Remaining algebraic checks are meaningless without valid structure.
        checks.append(("unit_row_mean_square", False, "skipped: structural check failed"))
        checks.append(("positive_row_scale", False, "skipped: structural check failed"))
        checks.append(("row_proportional_to_input", False, "skipped: structural check failed"))
        return checks

    yf = y.float()
    xf = x.float()
    gain = weight.float() * (1.0 + scale.float())          # (H,), |gain| >= 0.5625
    z = (yf - shift.float()) / gain                        # (T, H) ~= x * rstd

    # ---- 2. de-modulated rows have unit mean-square ------------------------
    z_ms = (z * z).sum(dim=-1, dtype=torch.float64) / float(H)     # (T,)
    ms_err = (z_ms - 1.0).abs()
    ms_ok = bool((ms_err <= 3e-2).all())
    checks.append((
        "unit_row_mean_square",
        ms_ok,
        f"max |mean(z*z) - 1| = {float(ms_err.max()):.3e} (tol 3e-2); "
        f"z = (y - shift) / (weight*(1+scale))",
    ))

    # ---- 3./4. z must be a POSITIVE per-row multiple of x ------------------
    num = (z * xf).sum(dim=-1, dtype=torch.float64)                # (T,)
    den = (xf * xf).sum(dim=-1, dtype=torch.float64)               # (T,)
    alpha = num / den.clamp_min(1e-30)                             # least-squares row scale
    alpha_ok = bool((alpha > 0).all())
    checks.append((
        "positive_row_scale",
        alpha_ok,
        f"min alpha = {float(alpha.min()):.6e} (must be > 0); "
        f"alpha = sum(z*x)/sum(x*x) per row",
    ))

    resid = z - alpha.to(torch.float32).unsqueeze(1) * xf
    r_rms = torch.sqrt((resid * resid).sum(dim=-1, dtype=torch.float64) / float(H))
    z_rms = torch.sqrt(z_ms)
    ratio = r_rms / (z_rms + 1e-6)
    prop_ok = bool((ratio <= 2e-2).all())
    checks.append((
        "row_proportional_to_input",
        prop_ok,
        f"max rms(z - alpha*x)/(rms(z)+1e-6) = {float(ratio.max()):.3e} (tol 2e-2)",
    ))

    return checks


# ---------------------------------------------------------------------------
# Strong-library baseline: torch.compile (the honest frontier).
#
# Private adapter: ignores init_args, caches a single torch.compile(dynamic=False)
# wrapper over the exact single-expression PyTorch implementation, and performs no
# custom-kernel / library dispatch and no eager fallback.  Compilation + warmup are
# forced on the first call for a given shape/device/dtype key, so timed steady-state
# calls never pay compile cost.
# ---------------------------------------------------------------------------

def _modulated_rmsnorm_expr(x, weight, scale, shift):
    # Single expression; fp32 throughout, one final bf16 cast.
    xf = x.float()
    return (
        xf
        * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + EPS)
        * weight.float()
        * (1.0 + scale.float())
        + shift.float()
    ).to(torch.bfloat16)

MEM_WEIGHT = 0.0
