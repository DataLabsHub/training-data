"""
S03_softmax_rowmax — Numerically stable row-wise softmax over a bf16 [rows, cols] tensor.

Reference (Oracle) semantics
----------------------------
Given x of shape [R, C] in bfloat16, produce y of shape [R, C] in bfloat16 with

    m_r   = max_c x[r, c]                      (fp32)
    z_r,c = exp(x[r, c] - m_r)                 (fp32)
    s_r   = sum_c z[r, c]                      (fp32)
    y_r,c = bf16( z[r, c] / s_r )

The naive Model below implements exactly this, but *intentionally multi-pass*:
each of max / subtract / exp / sum / divide is a separate full-tensor pass, so
the [R, C] matrix is streamed through HBM five-plus times.  That is the honest
memory-bound denominator a fused kernel must beat (a fused kernel touches the
matrix once for read and once for write, keeping the row reductions in
registers/SRAM in fp32).

Optimization target: a single fused softmax kernel (one row per program /
persistent-tile style, à la triton-lang/triton `02-fused-softmax.py`) that reads
bf16, reduces in fp32, and writes bf16.

Inspiration: https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive, obviously-correct, multi-pass numerically-stable row softmax.

    Deliberately NOT optimized: every stage materializes a full [R, C]
    intermediate in fp32 and re-reads it from HBM.  This is the correctness
    Oracle and the timing denominator.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Pass 0: widen to fp32 for the accumulation domain.
        xf = x.float()

        # Pass 1: row maxima (full read of xf).
        row_max = torch.max(xf, dim=-1, keepdim=True).values

        # Pass 2: subtract the row max (full read + full write).
        shifted = xf - row_max

        # Pass 3: exponentiate (full read + full write).
        num = torch.exp(shifted)

        # Pass 4: row sums (full read).
        row_sum = torch.sum(num, dim=-1, keepdim=True)

        # Pass 5: normalize (full read + full write).
        prob = num / row_sum

        # Pass 6: narrow back to bf16.
        return prob.to(x.dtype).contiguous()


def make_inputs(shape, device, seed=0):
    """Build forward args: [x] with x contiguous bf16 [R, C].

    The data intentionally carries large per-row offsets in roughly
    [-100, 100].  Any implementation that evaluates exp(x) without first
    subtracting the row max will overflow for large positive row offsets; after
    subtracting the row max, the tail may underflow to 0 safely and the row sum
    remains numerically meaningful — bf16 output of
    such an implementation will contain NaN/Inf or all-zero rows and fail.
    """
    R, C = int(shape[0]), int(shape[1])
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # Base per-element variation: mixture of a moderate normal and a wider tail
    # so within-row dynamic range is nontrivial.
    base = torch.randn(R, C, generator=g, dtype=torch.float32) * 4.0
    tail = torch.randn(R, C, generator=g, dtype=torch.float32) * 12.0
    mask = (torch.rand(R, C, generator=g, dtype=torch.float32) < 0.05).float()
    vals = base + mask * tail

    # Per-row offsets spanning [-100, 100]: this is the overflow tripwire.
    offsets = (torch.rand(R, 1, generator=g, dtype=torch.float32) * 200.0) - 100.0
    # Pin a few rows to the extremes so the tripwire fires deterministically.
    if R >= 4:
        offsets[0, 0] = 100.0
        offsets[1, 0] = -100.0
        offsets[2, 0] = 88.0
        offsets[3, 0] = -88.0
    vals = vals + offsets

    x = vals.to(device=device, dtype=torch.bfloat16).contiguous()
    return [x]


def get_init_inputs():
    return []


TASK_ID = "S03_softmax_rowmax"

WORKLOAD = [
    (4096, 1024),
    (8192, 2048),
]

TASK_MODE = "tensor"

# bf16 output tolerance: values are probabilities in [0, 1]; bf16 has ~3 decimal
# digits, so a relative tolerance dominates with a small absolute floor.
TASK_RTOL = 2e-2
TASK_ATOL = 2e-3

GRAD_INPUTS = []

OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    "input_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "accumulation_dtype": "float32",
    "allow_tf32": False,
    "note": (
        "Forward-only inference. Row-wise numerically stable softmax over the "
        "last axis of a contiguous bf16 [R, C] tensor. Row max and row sum MUST "
        "be accumulated in fp32; the output is rounded to bf16 exactly once, at "
        "the end. No vendor library anchor is declared for this task — the "
        "GB200 frontier must come from real compile/witness measurement of a "
        "fused kernel against the multi-pass naive Model. baseline() below is "
        "a torch.compile frontier over the naive Model, provided for "
        "calibration/provenance only, not as the scored anchor."
    ),
}


def sample_shapes(rng, n):
    """Unseen shapes inside the legal domain R=[4096, 8192], C=[1024, 2048].

    R is a multiple of 256, C a multiple of 128; anything colliding with a
    public WORKLOAD entry is rejected.  Hidden-shape tripwire for kernels that
    hard-code the public tile geometry.
    """
    public = {(int(r), int(c)) for (r, c) in WORKLOAD}
    r_choices = list(range(4096, 8192 + 1, 256))
    c_choices = list(range(1024, 2048 + 1, 128))

    out = []
    seen = set()
    guard = 0
    while len(out) < int(n) and guard < 4096:
        guard += 1
        R = r_choices[rng.randrange(len(r_choices))]
        C = c_choices[rng.randrange(len(c_choices))]
        cand = (R, C)
        if cand in public or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)

    # Deterministic fallback fill if the rng kept colliding.
    i = 0
    while len(out) < int(n):
        R = r_choices[i % len(r_choices)]
        C = c_choices[(i * 3 + 1) % len(c_choices)]
        cand = (R, C)
        i += 1
        if cand in public or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)

    return out


def invariants(out, ins, shape):
    """Metamorphic / algebraic checks on the candidate output alone.

    None of these re-runs Model.  They are properties any correct softmax
    output must satisfy, derived from the definition rather than from the
    reference implementation's code path:

      1. structure   — shape, dtype, contiguity, finiteness, non-negativity.
      2. partition   — every row sums to 1 (the defining normalization).
      3. shift       — softmax is invariant to adding a per-row constant, so
                       recomputing from a *differently shifted* input (shift by
                       the row MIN rather than the row max) must reproduce the
                       same distribution.  This exercises the stability claim.
      4. ratio law   — y[r,i] / y[r,j] must equal exp(x[r,i] - x[r,j]); checked
                       in log space on the argmax column against a random
                       partner column.  Purely local, no reduction reused.
    """
    checks = []

    y = out[OUTPUT_INDEX] if isinstance(out, (list, tuple)) else out
    x = ins[0]
    R, C = int(shape[0]), int(shape[1])

    # ---- 1. structure -----------------------------------------------------
    ok_shape = tuple(y.shape) == (R, C)
    ok_dtype = y.dtype == torch.bfloat16
    ok_contig = y.is_contiguous()
    yf = y.float()
    ok_finite = bool(torch.isfinite(yf).all().item())
    ok_range = bool(((yf >= 0.0) & (yf <= 1.0 + 1e-3)).all().item())
    passed = ok_shape and ok_dtype and ok_contig and ok_finite and ok_range
    checks.append((
        "structure_finite_probability",
        bool(passed),
        (f"shape={tuple(y.shape)} expected={(R, C)} dtype={y.dtype} "
         f"contiguous={ok_contig} finite={ok_finite} in_unit_range={ok_range}"),
    ))

    # ---- 2. partition of unity -------------------------------------------
    row_sums = yf.sum(dim=-1)
    max_dev = float((row_sums - 1.0).abs().max().item())
    # bf16 elementwise rounding accumulates over C terms; ~C * 2^-9 * mean is
    # the honest bound, but errors are signed and cancel, so allow ~1%.
    sum_tol = 1.5e-2
    checks.append((
        "rows_sum_to_one",
        bool(max_dev <= sum_tol),
        f"max |row_sum - 1| = {max_dev:.5e} (tol {sum_tol:.1e}) over {R} rows",
    ))

    # ---- 3. shift invariance (independent stabilization path) -------------
    # softmax(x + c) == softmax(x) for any per-row constant c.  Recompute the
    # distribution after shifting by the row MINIMUM instead of the row max.
    # This never touches Model; it re-derives the answer from the definition
    # using a different (and differently conditioned) shift.
    xf = x.float()
    row_min = xf.min(dim=-1, keepdim=True).values
    # Guard the fp32 exponential range: min-shifted values can be large, so
    # evaluate log-domain and renormalize by the log-sum-exp.
    t = xf - row_min
    t_max = t.max(dim=-1, keepdim=True).values
    logsumexp = t_max + torch.log(torch.exp(t - t_max).sum(dim=-1, keepdim=True))
    ref_shift = torch.exp(t - logsumexp)
    diff = (yf - ref_shift).abs()
    denom = ref_shift.abs().clamp_min(1e-6)
    rel = (diff / denom)
    # Only judge entries that carry real mass; bf16 denormal tails are noise.
    mass = ref_shift > 1e-4
    if bool(mass.any().item()):
        worst_rel = float(rel[mass].max().item())
    else:
        worst_rel = 0.0
    shift_tol = 6e-2
    checks.append((
        "shift_invariance_min_anchored",
        bool(worst_rel <= shift_tol),
        (f"max rel-err vs min-anchored logsumexp recompute = {worst_rel:.5e} "
         f"(tol {shift_tol:.1e}) on {int(mass.sum().item())} significant entries"),
    ))

    # ---- 4. pairwise ratio law -------------------------------------------
    # y[r,i]/y[r,j] == exp(x[r,i] - x[r,j]).  Take i = argmax column (largest
    # probability, best conditioned in bf16) and j = the runner-up column, so
    # both operands carry real mass.  Fully local: no row reduction of y reused.
    top2 = torch.topk(xf, k=2, dim=-1)
    xi = top2.values[:, 0]
    xj = top2.values[:, 1]
    idx_i = top2.indices[:, 0]
    idx_j = top2.indices[:, 1]
    rows = torch.arange(R, device=yf.device)
    yi = yf[rows, idx_i]
    yj = yf[rows, idx_j]
    good = (yj > 1e-4) & (yi > 1e-4)
    if bool(good.any().item()):
        lhs = torch.log(yi[good]) - torch.log(yj[good])
        rhs = xi[good] - xj[good]
        # bf16 stores ~8 mantissa bits => relative error ~2^-9 => log error
        # ~2e-3 per operand; allow a few multiples plus the bf16 input rounding.
        ratio_err = float((lhs - rhs).abs().max().item())
        n_checked = int(good.sum().item())
    else:
        ratio_err = 0.0
        n_checked = 0
    ratio_tol = 6e-2
    checks.append((
        "pairwise_log_ratio_law",
        bool(ratio_err <= ratio_tol),
        (f"max |log(y_i/y_j) - (x_i - x_j)| = {ratio_err:.5e} "
         f"(tol {ratio_tol:.1e}) on {n_checked}/{R} top-2 column pairs"),
    ))

    return checks


_COMPILED_SOFTMAX = None


def baseline(x):
    """torch.compile frontier over the naive stable-softmax Model.

    No vendor library anchor is declared for this route. This helper exists for
    calibration/provenance only; reward remains anchored to the naive Model.
    """
    global _COMPILED_SOFTMAX
    model = Model()

    def _run(x):
        return model(x)

    if _COMPILED_SOFTMAX is None:
        try:
            _COMPILED_SOFTMAX = torch.compile(_run, dynamic=False)
        except Exception:
            _COMPILED_SOFTMAX = _run
    return _COMPILED_SOFTMAX(x)
