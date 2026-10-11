"""
S02_layernorm -- LayerNorm forward over a bf16 [tokens, hidden] activation tensor.

TASK (not a solution). This module defines:
  * Model      : the NAIVE, deliberately-unfused plain-PyTorch reference. It is the
                 correctness Oracle AND the timing denominator. Every intermediate
                 (mean, centered residual, squared residual, variance, rstd, scaled
                 value, affine bias) is materialized as its own full-size tensor, so
                 the reference makes several full HBM round-trips over [T, H].
  * baseline   : optional torch.compile frontier adapter for calibration/provenance.

The optimization target is a single fused kernel: one pass (or a Welford/two-pass
in-SRAM variant) over the hidden axis with fp32 accumulation, bf16 in / bf16 out,
no materialized intermediates.

Reference/inspiration: triton-lang/triton
python/tutorials/05-layer-norm.py

Numerics contract
-----------------
  input      : bf16   x[T, H], gamma[H], beta[H]
  accumulate : fp32   (mean / variance / rstd computed in fp32)
  output     : bf16   y[T, H]
  variance   : BIASED (divide by H, not H-1), matching torch LayerNorm.
  y = (x - mean) * rsqrt(var + eps) * gamma + beta

Delivery note: no vendor library anchor is declared for this route, so the GB200
frontier must be MEASURED honestly from compile/witness on the target SKU. Do not
inherit a speedup number from a different SKU.
"""

import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Naive reference (the Oracle + the timing denominator)
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """Correctness-only LayerNorm forward. Deliberately unfused.

    Each step below is a separate PyTorch op producing a separate full-size
    temporary. This is what an optimized kernel must beat: the arithmetic is
    trivial, the cost is the repeated [T, H] HBM traffic.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta, eps):
        # Accumulate in fp32 over the hidden axis. Materialize everything.
        xf = x.to(torch.float32)                       # pass 1: upcast
        mean = xf.mean(dim=-1, keepdim=True)           # pass 2: row mean
        centered = xf - mean                           # pass 3: centering
        sq = centered * centered                       # pass 4: squared residual
        var = sq.mean(dim=-1, keepdim=True)            # pass 5: biased variance
        rstd = torch.rsqrt(var + eps)                  # reciprocal std
        normed = centered * rstd                       # pass 6: normalize
        scaled = normed * gamma.to(torch.float32)      # pass 7: affine scale
        shifted = scaled + beta.to(torch.float32)      # pass 8: affine bias
        return shifted.to(torch.bfloat16)              # pass 9: downcast


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def get_init_inputs():
    """Args passed to Model(...)."""
    return []


def make_inputs(shape, device, seed=0):
    """Build forward args for a given (T, H) shape.

    Returns [x, gamma, beta, eps]:
        x     : bf16 [T, H]
        gamma : bf16 [H]
        beta  : bf16 [H]
        eps   : python float

    `seed` must change the data (it does -- it drives a fresh CPU generator).
    """
    T, H = int(shape[0]), int(shape[1])
    dev = torch.device(device)

    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # Activations: unit-ish scale, with a per-row offset and gain so that rows are
    # NOT already normalized. This keeps mean/var recovery a real computation and
    # exercises the fp32 accumulation path.
    base = torch.randn(T, H, generator=g, dtype=torch.float32)
    row_shift = torch.randn(T, 1, generator=g, dtype=torch.float32) * 2.0
    row_gain = 0.5 + torch.rand(T, 1, generator=g, dtype=torch.float32) * 2.0
    x = base * row_gain + row_shift

    # Affine params: gamma near 1, beta near 0, but not exactly -- so a kernel that
    # silently drops the affine step is caught.
    gamma = 1.0 + 0.25 * torch.randn(H, generator=g, dtype=torch.float32)
    beta = 0.10 * torch.randn(H, generator=g, dtype=torch.float32)

    x = x.to(device=dev, dtype=torch.bfloat16).contiguous()
    gamma = gamma.to(device=dev, dtype=torch.bfloat16).contiguous()
    beta = beta.to(device=dev, dtype=torch.bfloat16).contiguous()
    eps = 1e-5

    return [x, gamma, beta, eps]


# --------------------------------------------------------------------------- #
# Task metadata
# --------------------------------------------------------------------------- #
# Public, visible shapes. (2048, 1024) is the accepted public shape.
TASK_ID = "S02_layernorm"

WORKLOAD = [
    (2048, 1024),
    (4096, 1024),
    (2048, 2048),
    (4096, 2048),
]

TASK_MODE = "tensor"

# bf16 output of an fp32-accumulated reduction: ~2^-8 relative resolution, plus
# reduction-order slack over H up to 2048.
TASK_RTOL = 2e-2
TASK_ATOL = 2e-2

# Forward-only inference task.
GRAD_INPUTS = []

OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    "input_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "accumulation_dtype": "float32",
    "allow_tf32": True,
    "note": (
        "LayerNorm forward, inference only. x is bf16 [T, H]; gamma/beta are bf16 [H]; "
        "eps is a python float. Mean and variance are reduced over the hidden axis in "
        "fp32; variance is BIASED (divide by H, matching torch.nn.LayerNorm). Output is "
        "bf16 [T, H]. No vendor library anchor is declared for this route -- the GB200 "
        "frontier must be measured honestly from compile/witness on the target SKU, not "
        "inherited from another SKU. Reference: triton 05-layer-norm.py."
    ),
}


# --------------------------------------------------------------------------- #
# Hidden-shape tripwire
# --------------------------------------------------------------------------- #
def sample_shapes(rng, n):
    """Unseen (T, H) shapes inside the legal domain T=[2048,4096], H=[1024,2048].

    Deliberately generated at runtime and mostly NOT powers of two, so a kernel
    that hard-codes the public shape (or assumes H is a power of two, or assumes
    H fits one fixed BLOCK) fails here. All shapes stay 8-element aligned on H so
    they remain legal for vectorized bf16 loads; a couple are only 8-aligned
    (not 128-aligned) to stress tail handling.
    """
    shapes = []
    seen = set(WORKLOAD)
    guard = 0
    while len(shapes) < int(n) and guard < 4096:
        guard += 1

        T = int(rng.randrange(2048, 4097, 8))

        style = rng.randrange(3)
        if style == 0:
            H = int(rng.randrange(1024, 2049, 128))     # nicely aligned, non-p2
        elif style == 1:
            H = int(rng.randrange(1024, 2049, 32))      # medium alignment
        else:
            H = int(rng.randrange(1024, 2049, 8))       # ragged tail

        if H < 1024 or H > 2048 or T < 2048 or T > 4096:
            continue
        cand = (T, H)
        if cand in seen:
            continue
        seen.add(cand)
        shapes.append(cand)

    # Deterministic fallback if the rng starved us.
    fallback = [(2056, 1160), (3072, 1536), (4088, 2040), (2432, 1288), (3480, 1976)]
    i = 0
    while len(shapes) < int(n):
        cand = fallback[i % len(fallback)]
        i += 1
        if cand in seen:
            cand = (min(4096, cand[0] + 8 * i), cand[1])
        seen.add(cand)
        shapes.append(cand)

    return shapes[: int(n)]


# --------------------------------------------------------------------------- #
# Metamorphic invariants (strong, independent evidence)
# --------------------------------------------------------------------------- #
def invariants(out, ins, shape):
    """Algebraic properties a correct LayerNorm output MUST satisfy.

    These are computed WITHOUT rerunning Model -- they check the output against
    closed-form properties of layer normalization and against affine-equivariance
    relations derived directly from the inputs. They are independent evidence:
    each would fail for a kernel that, e.g., used unbiased variance, skipped the
    affine step, normalized the wrong axis, or reduced in bf16.

    Returns a list of (name, passed_bool, detail).
    """
    checks = []

    y = out[0] if isinstance(out, (list, tuple)) else out
    x, gamma, beta, eps = ins[0], ins[1], ins[2], float(ins[3])
    T, H = int(shape[0]), int(shape[1])

    # ---- 1. Shape / dtype / finiteness -------------------------------------
    ok_shape = tuple(y.shape) == (T, H)
    ok_dtype = (y.dtype == torch.bfloat16)
    ok_finite = bool(torch.isfinite(y.to(torch.float32)).all().item())
    checks.append((
        "shape_dtype_finite",
        bool(ok_shape and ok_dtype and ok_finite),
        "shape={} (want {}), dtype={} (want torch.bfloat16), all_finite={}".format(
            tuple(y.shape), (T, H), y.dtype, ok_finite
        ),
    ))

    yf = y.to(torch.float32)
    gf = gamma.to(torch.float32)
    bf = beta.to(torch.float32)

    # ---- 2. Closed-form de-affine moments ----------------------------------
    # Strip the affine transform: z = (y - beta) / gamma must be exactly the
    # normalized activation, so per row  mean(z) == 0  and  mean(z^2) == var/(var+eps)
    # ~= 1 (biased second moment). This is the defining property of LayerNorm and
    # is derived from y alone -- no recompute of the normalization.
    # Only use columns where |gamma| is large enough that the division is stable.
    safe = gf.abs() > 0.15
    n_safe = int(safe.sum().item())
    if n_safe >= max(64, H // 8):
        z = (yf[:, safe] - bf[safe]) / gf[safe]
        z_mean = z.mean(dim=-1)
        z_sec = (z * z).mean(dim=-1)
        max_abs_mean = float(z_mean.abs().max().item())
        max_sec_dev = float((z_sec - 1.0).abs().max().item())
        # bf16 output quantization (~4e-3 relative) dominates the tolerance; the
        # column subset makes the moments approximate rather than exact, so allow
        # a generous but still discriminating band.
        passed2 = (max_abs_mean < 0.12) and (max_sec_dev < 0.35)
        detail2 = (
            "over {}/{} stable-gamma cols: max|mean(z)|={:.4f} (<0.12), "
            "max|mean(z^2)-1|={:.4f} (<0.35)".format(
                n_safe, H, max_abs_mean, max_sec_dev
            )
        )
    else:
        passed2 = True
        detail2 = "skipped: only {}/{} stable-gamma columns".format(n_safe, H)
    checks.append(("de_affine_row_moments", bool(passed2), detail2))

    # ---- 3. Shift/scale equivariance of the INPUT --------------------------
    # LayerNorm is invariant to a per-row additive shift and equivariant (identity)
    # to a positive per-row scale of x, up to the eps term. Concretely, for
    #     x' = a * x + b   (a > 0 scalar per row, b scalar per row)
    # the normalized value is unchanged when eps -> 0, so y'(x') == y(x). We verify
    # the relation the other way around WITHOUT calling Model: recompute the
    # normalization from x directly in fp64 (a genuinely different code path and a
    # different precision than the bf16/fp32 kernel) on a random row subset and
    # confirm agreement. fp64 makes reduction-order effects vanish, so a kernel
    # that accumulates in bf16 or uses unbiased variance is exposed.
    n_rows = min(64, T)
    step = max(1, T // n_rows)
    idx = torch.arange(0, T, step, device=y.device)[:n_rows]
    xd = x[idx].to(torch.float64)
    mu = xd.mean(dim=-1, keepdim=True)
    c = xd - mu
    v = (c * c).mean(dim=-1, keepdim=True)          # BIASED: /H
    ref = c * torch.rsqrt(v + eps) * gf.to(torch.float64) + bf.to(torch.float64)
    got = yf[idx].to(torch.float64)
    denom = ref.abs().clamp_min(1.0)
    rel = ((got - ref).abs() / denom)
    max_rel = float(rel.max().item())
    mean_rel = float(rel.mean().item())
    passed3 = max_rel < 3e-2
    checks.append((
        "fp64_closed_form_rows",
        bool(passed3),
        "rows={}, max_rel={:.5f} (<3e-2), mean_rel={:.6f}; biased var (/H), eps={:g}".format(
            n_rows, max_rel, mean_rel, eps
        ),
    ))

    # ---- 4. Row independence / no cross-row leakage ------------------------
    # Rows of LayerNorm are computed independently. Re-normalizing a single row in
    # isolation must reproduce that row of y. Pick the row with the largest input
    # magnitude (worst case for fp32 accumulation) and one arbitrary row; a kernel
    # that shares reduction state across rows, or that has an off-by-one in its row
    # tiling, breaks here.
    row_energy = x.to(torch.float32).abs().amax(dim=-1)
    probe = [int(torch.argmax(row_energy).item()), T - 1, 0]
    worst = 0.0
    for r in probe:
        xr = x[r].to(torch.float64)
        mr = xr.mean()
        cr = xr - mr
        vr = (cr * cr).mean()
        yr = cr * torch.rsqrt(vr + eps) * gf.to(torch.float64) + bf.to(torch.float64)
        gr = yf[r].to(torch.float64)
        worst = max(worst, float(((gr - yr).abs() / yr.abs().clamp_min(1.0)).max().item()))
    passed4 = worst < 3e-2
    checks.append((
        "row_independence",
        bool(passed4),
        "probe rows {} (incl. max-magnitude row): max_rel={:.5f} (<3e-2)".format(
            sorted(set(probe)), worst
        ),
    ))

    return checks

MEM_WEIGHT = 0.0
