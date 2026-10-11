# reference_model.py
# ---------------------------------------------------------------------------
# task_id : liger_kernel_normalization_pr1305
# domain  : normalization      task_type : fusion       task_mode : tensor
# track   : fusion_and_layout
# target  : h200_sm90_cuda129_torch280_nv2505 (NVIDIA H200, sm_90 / Hopper)
# precision: input bf16, accumulation fp32
#
# WHAT THIS TASK IS
#   A forward-only RMSNorm (Llama casting, offset = 0.0, eps = 1e-6) over a
#   contiguous BF16 activation tensor x [T, H] and a contiguous BF16 gain
#   vector weight [H].  The naive `Model` below is the correctness oracle and
#   the timing denominator; it is deliberately unfused and unoptimized.
#
# PROVENANCE AND SCOPE DISCLOSURE (read this before trusting any speedup)
#   This task is *derived from* linkedin/Liger-Kernel PR #1305, but it does not
#   reproduce it.  PR #1305 changes only tests: it exercises an in-place
#   BACKWARD hazard in which a shared gradient buffer is clobbered when the
#   same tensor feeds multiple consumers.  It modifies no production kernel and
#   makes no performance claim.
#     * This task is FORWARD-ONLY (GRAD_INPUTS = []).  The backward
#       shared-gradient-buffer regression from PR #1305 is OUT OF SCOPE here.
#       It is NOT fixed, NOT exercised, and NOT tested by anything in this file.
#     * Consequently there is NO source performance claim attached to this
#       task.  Any speedup measured here is a property of the forward RMSNorm
#       kernel written against this reference, not of PR #1305.
#   What this task *does* inherit from the PR is the underlying concern about
#   buffer reuse: a submitted kernel must not mutate its inputs.  The scored
#   forward must leave x and weight bitwise unchanged.
#
# VERIFIER LIMITATION (stated honestly, do not overclaim)
#   The current verifier snapshots the non-mutable inputs before the scored
#   call and rejects any input mutation.  That is the extent of the guarantee.
#   It does NOT test output storage aliasing, it does NOT test whether the
#   output shares memory with a workspace reused across calls, and the
#   metamorphic `invariants` below do NOT prove purity either -- they are
#   algebraic checks on the returned values only.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Task metadata
# --------------------------------------------------------------------------- #

TASK_MODE = "tensor"

TASK_RTOL = 2e-2
TASK_ATOL = 1e-2

GRAD_INPUTS = []          # forward-only task
OUTPUT_INDEX = 0          # single output tensor
MEM_WEIGHT = 0.5

# Public, representative shapes (T, H).
WORKLOAD = [
    (4096, 2048),
    (8192, 4096),
    (16384, 8192),
]

# Declared legal shape domain (axes are choice sets, not ranges).
T_CHOICES = (4096, 8192, 12288, 16384)
H_CHOICES = (2048, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 7168, 8192)

EPS = 1e-6
OFFSET = 0.0
CASTING_MODE = "llama"

BASELINE_CONTRACT = {
    "input_dtype": "bf16",
    "output_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "allow_tf32": False,
    "note": (
        "Llama cast-before-weight semantics: the row mean-square and "
        "rstd = rsqrt(mean_square + 1e-6) are accumulated in fp32 from the bf16 "
        "input, the normalized value x.float() * rstd is cast DOWN to bf16 "
        "FIRST, and only then is it multiplied by the bf16 gain weight; the "
        "product is returned as bf16. offset = 0.0, eps = 1e-6, no mean "
        "subtraction. TF32 is irrelevant here (no matmul) and is declared off. "
        "Scope: forward only. The in-place backward shared-gradient-buffer "
        "hazard from Liger-Kernel PR #1305 is out of scope and is neither "
        "fixed nor tested by this task; the scored forward must, however, "
        "leave its inputs bitwise unchanged."
    ),
}


# --------------------------------------------------------------------------- #
# Naive reference model  (correctness Oracle + timing denominator)
# --------------------------------------------------------------------------- #

class Model(nn.Module):
    """Plain-PyTorch RMSNorm, Llama casting mode, eps = 1e-6, offset = 0.0.

    Deliberately naive: several full-size temporaries, no fusion, no reuse.
    Does not mutate its inputs.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, x, weight):
        # x      : bf16 [T, H], contiguous
        # weight : bf16 [H],    contiguous
        x_f32 = x.float()                                   # up-cast
        mean_square = x_f32.pow(2).mean(dim=-1, keepdim=True)   # fp32 accumulate
        rstd = torch.rsqrt(mean_square + self.eps)              # fp32
        x_normed = (x_f32 * rstd).to(torch.bfloat16)            # Llama: cast FIRST
        out = x_normed * weight                                 # then apply gain
        return out.contiguous()


def get_init_inputs():
    """Constructor arguments for Model(...)."""
    return []


# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #

def _pm_one_row(h, device):
    """Deterministic alternating +1 / -1 bf16 row: finite, non-constant, ms == 1."""
    idx = torch.arange(h, device=device, dtype=torch.float32)
    return ((idx % 2.0) * 2.0 - 1.0).to(torch.bfloat16)


def make_inputs(shape, device, seed=0):
    """Return [x, weight] for the given (T, H) shape.

    Guarantees (all measured AFTER the bf16 cast, i.e. on the tensor actually
    handed to the kernel):
      * x is finite, contiguous bf16 [T, H]; every row is non-constant and has
        row mean-square >= 0.25 (rows failing either test are deterministically
        replaced by an alternating +/-1 row).
      * weight is contiguous bf16 [H], strictly positive, channel-varying, and
        lies in [0.75, 1.25].
      * Both x and weight change with `seed` (rows 1 and 2 are sentinels tied
        to row 0 and are therefore not independently randomized, but row 0
        itself -- and hence rows 1, 2 -- does change with the seed).
      * Reserved metamorphic rows:  x[1] == 2 * x[0]  and  x[2] == -x[0],
        both exact in bf16 (power-of-two scale / sign flip).
    """
    t, h = int(shape[0]), int(shape[1])
    if t < 3:
        raise ValueError("this task reserves rows 0/1/2 as metamorphic sentinels; T must be >= 3")

    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed) * 1000003 + 12345)

    # ---- activations -----------------------------------------------------
    x32 = torch.randn(t, h, generator=gen, device=dev, dtype=torch.float32)
    x = x32.to(torch.bfloat16).contiguous()

    # Post-cast repair pass: finite, non-constant, mean-square >= 0.25.
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1)
    row_max = xf.amax(dim=-1)
    row_min = xf.amin(dim=-1)
    bad = (~torch.isfinite(ms)) | (ms < 0.25) | (row_max == row_min)
    if bool(bad.any()):
        x[bad] = _pm_one_row(h, dev)

    # Reserved metamorphic rows (exact in bf16).
    x[1] = x[0] * 2.0
    x[2] = -x[0]
    x = x.contiguous()

    # ---- gain ------------------------------------------------------------
    u = torch.rand(h, generator=gen, device=dev, dtype=torch.float32)      # [0, 1)
    ramp = torch.arange(h, device=dev, dtype=torch.float32) / float(max(h - 1, 1))
    # seed-dependent + guaranteed channel-varying, mapped into [0.75, 1.25]
    weight = (0.75 + 0.5 * (0.5 * u + 0.5 * ramp)).to(torch.bfloat16).contiguous()

    return [x, weight]


def sample_shapes(rng, n):
    """Unseen (T, H) shapes drawn only from the declared choice sets.

    Never returns a WORKLOAD shape and never repeats a shape.
    """
    def rand01():
        if hasattr(rng, "random"):
            return float(rng.random())
        if hasattr(rng, "random_sample"):
            return float(rng.random_sample())
        return float(rng.uniform(0.0, 1.0))

    workload = {(int(a), int(b)) for a, b in WORKLOAD}
    pool = [(t, h) for t in T_CHOICES for h in H_CHOICES if (t, h) not in workload]

    # Deterministic Fisher-Yates driven by rng.
    for i in range(len(pool) - 1, 0, -1):
        j = int(rand01() * (i + 1))
        if j > i:
            j = i
        pool[i], pool[j] = pool[j], pool[i]

    k = max(0, min(int(n), len(pool)))
    return pool[:k]


# --------------------------------------------------------------------------- #
# Independent oracle (does NOT call Model or baseline)
# --------------------------------------------------------------------------- #

def independent_oracle(ins, shape):
    """RMSNorm recomputed straight from the definition in float64.

    Written from the mathematical definition only:
        rms(x_row) = sqrt( (1/H) * sum_j x_j^2 + eps )
        y = bf16( x / rms )            <- Llama: cast BEFORE the gain
        out = bf16( y * weight )
    Row-chunked to keep the float64 working set bounded.
    """
    x, weight = ins[0], ins[1]
    t, h = int(shape[0]), int(shape[1])

    w64 = weight.to(torch.float64)
    out = torch.empty((t, h), dtype=torch.bfloat16, device=x.device)

    chunk = 2048
    for start in range(0, t, chunk):
        stop = min(start + chunk, t)
        blk = x[start:stop].to(torch.float64)
        sumsq = (blk * blk).sum(dim=-1, keepdim=True)           # float64 sum of squares
        rms = torch.sqrt(sumsq / float(h) + EPS)                # float64 square root
        normed = (blk / rms).to(torch.bfloat16)                 # cast to bf16 first
        out[start:stop] = (normed.to(torch.float64) * w64).to(torch.bfloat16)

    return out.contiguous()


# --------------------------------------------------------------------------- #
# Metamorphic invariants (no re-run of Model)
# --------------------------------------------------------------------------- #

def invariants(out, ins, shape):
    """Algebraic properties any correct RMSNorm output must satisfy.

    Computed purely from `out` and `ins` -- Model is never re-run, so these are
    independent evidence rather than a precision recompute.

    NOTE: these checks say nothing about input mutation or output aliasing.
    Purity is enforced separately by the verifier's input snapshot.
    """
    checks = []
    x, weight = ins[0], ins[1]
    t, h = int(shape[0]), int(shape[1])

    if isinstance(out, (tuple, list)):
        out = out[OUTPUT_INDEX]

    # 1) structure + finiteness
    ok_shape = tuple(out.shape) == (t, h)
    ok_dtype = out.dtype == torch.bfloat16
    ok_finite = bool(torch.isfinite(out.float()).all())
    checks.append((
        "shape_dtype_finite",
        bool(ok_shape and ok_dtype and ok_finite),
        f"shape={tuple(out.shape)} expected={(t, h)} dtype={out.dtype} "
        f"expected=torch.bfloat16 all_finite={ok_finite}",
    ))

    of = out.float()

    # 2) positive-scale invariance: RMSNorm(c*x) == RMSNorm(x) for c > 0.
    #    make_inputs guarantees x[1] == 2 * x[0].
    d_scale = (of[1] - of[0]).abs()
    tol_scale = TASK_ATOL + TASK_RTOL * of[0].abs()
    ok_scale = bool((d_scale <= tol_scale).all())
    checks.append((
        "positive_scale_invariance",
        ok_scale,
        f"x[1] == 2*x[0] so out[1] must equal out[0]; "
        f"max_abs_dev={float(d_scale.max()):.6e} max_allowed={float(tol_scale.max()):.6e}",
    ))

    # 3) odd symmetry: RMSNorm(-x) == -RMSNorm(x).  x[2] == -x[0].
    d_odd = (of[2] + of[0]).abs()
    tol_odd = TASK_ATOL + TASK_RTOL * of[0].abs()
    ok_odd = bool((d_odd <= tol_odd).all())
    checks.append((
        "odd_symmetry",
        ok_odd,
        f"x[2] == -x[0] so out[2] must equal -out[0]; "
        f"max_abs_dev={float(d_odd.max()):.6e} max_allowed={float(tol_odd.max()):.6e}",
    ))

    # 4) unit row mean-square after dividing the gain back out.
    #    weight is guaranteed strictly positive in [0.75, 1.25].
    ratio = of / weight.float().unsqueeze(0)
    row_ms = ratio.pow(2).mean(dim=-1)
    worst = float((row_ms - 1.0).abs().max())
    ok_ms = worst <= 4e-2
    checks.append((
        "unit_row_mean_square",
        bool(ok_ms),
        f"max |mean_square(out/weight) - 1| = {worst:.6e} (tol 4e-2)",
    ))

    return checks
