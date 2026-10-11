"""
S04_swiglu — Precomputed-projection SwiGLU activation epilogue.

Real hotspot
------------
In a transformer FFN block with a gated activation (SwiGLU / LLaMA-style MLP),
the two projections `gate_proj = x @ Wg` and `up_proj = x @ Wu` are produced by
GEMMs.  The epilogue that follows is a purely elementwise gated activation:

    out = silu(gate_proj) * up_proj
        = (gate_proj * sigmoid(gate_proj)) * up_proj

GEMM is explicitly OUT OF SCOPE here.  This task isolates the activation
epilogue over bf16 [T, F] intermediates, which in a naive framework
implementation costs several full HBM round-trips (upcast gate, sigmoid,
multiply, upcast up, multiply, downcast) instead of the single
load-load-compute-store pass the hardware allows.

The naive `Model` below is the correctness Oracle AND the timing denominator:
it deliberately materializes the whole fp32 temporary chain.  A real fused
kernel should stream both bf16 operands once, do all nonlinear math in fp32,
and issue exactly one bf16 store.

Numerics contract
-----------------
Inputs are bf16.  All nonlinear math (sigmoid, the two multiplies) is done in
fp32.  The result is rounded to bf16 exactly ONCE, at the final store.  A
submission that computes sigmoid in bf16, or that round-trips through bf16
between the silu and the multiply, will lose accuracy relative to this
reference and must fail verification.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NAIVE reference (the correctness Oracle and the timing denominator)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """Obviously-correct, deliberately unfused SwiGLU activation epilogue.

    Every intermediate is materialized in fp32 in HBM:
        g32      = gate_proj.float()
        u32      = up_proj.float()
        sig      = sigmoid(g32)
        silu     = g32 * sig
        prod     = silu * u32
        out      = prod.to(bfloat16)     # the single rounding step

    This is the reference semantics: fp32 nonlinear math, one bf16 round.
    """

    def __init__(self):
        super().__init__()

    def forward(self, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
        # Upcast both operands (HBM pass 1 + 2 -> two fp32 temporaries).
        g32 = gate_proj.to(torch.float32)
        u32 = up_proj.to(torch.float32)

        # sigmoid temporary (HBM pass 3).
        sig = torch.sigmoid(g32)

        # silu temporary (HBM pass 4).
        silu = g32 * sig

        # gated product temporary (HBM pass 5).
        prod = silu * u32

        # Single, final rounding to bf16 (HBM pass 6).
        out = prod.to(torch.bfloat16)
        return out


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_inputs(shape, device, seed=0):
    """Build [gate_proj, up_proj], both contiguous bf16 [T, F].

    `seed` genuinely changes the data (different generator state -> different
    tensors), so a submission cannot memoize on a fixed input.

    Value range is chosen to exercise the whole silu curve: the saturating
    positive tail, the near-linear region around 0, and the negative lobe
    (where silu has its minimum near x = -1.278).  Scale ~ N(0, 1.5) with a
    small deterministic bias sweep matches realistic post-GEMM activation
    statistics for an FFN intermediate.
    """
    T, Fdim = int(shape[0]), int(shape[1])

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) * 9973 + 12345)

    # Generate in fp32 on CPU for reproducibility, then move + cast.
    g = torch.randn(T, Fdim, generator=gen, dtype=torch.float32) * 1.5
    u = torch.randn(T, Fdim, generator=gen, dtype=torch.float32) * 1.0

    # Deterministic per-column bias so the sign pattern is not symmetric and
    # the negative lobe of silu is well covered.
    bias = torch.linspace(-2.0, 2.0, Fdim, dtype=torch.float32).view(1, Fdim)
    g = g + bias

    gate_proj = g.to(device=device, dtype=torch.bfloat16).contiguous()
    up_proj = u.to(device=device, dtype=torch.bfloat16).contiguous()
    return [gate_proj, up_proj]


def get_init_inputs():
    return []


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------
# Representative PUBLIC shapes (T, F). (4096, 4096) is the accepted public shape.
TASK_ID = "S04_swiglu"

WORKLOAD = [
    (4096, 4096),
    (8192, 4096),
    (4096, 8192),
]

TASK_MODE = 'tensor'

# bf16 output: one ulp near 1.0 is ~7.8e-3 relative.  A correct fused kernel
# that does fp32 math and rounds once matches the reference bit-for-bit almost
# everywhere; the tolerance below allows a single differing bf16 ulp from a
# legal fast-sigmoid/FMA reassociation without admitting bf16-precision math.
TASK_RTOL = 2e-2
TASK_ATOL = 2e-2

# Forward-only inference task.
GRAD_INPUTS = []

OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    "input_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "accumulation_dtype": "float32",
    "allow_tf32": False,
    "note": (
        "Pure elementwise gated-activation epilogue; no GEMM, so TF32 is "
        "irrelevant and is left off. Both operands are contiguous bf16 [T, F]. "
        "sigmoid / silu / gated multiply must all be evaluated in fp32; the "
        "result is rounded to bf16 exactly once at the final store. No vendor "
        "library anchor is declared for this route -- the GB200 frontier must "
        "come from real torch.compile / witness measurement of the elementwise "
        "path, not from a quoted library number. Arithmetic intensity is fixed "
        "at 4 bytes moved per element (2 in x2 operands, 2 out), so the kernel "
        "is HBM-bandwidth bound and the achievable speedup over the naive "
        "multi-pass reference is roughly the ratio of HBM passes."
    ),
}


def sample_shapes(rng, n):
    """UNSEEN shapes inside the legal domain T=[4096,8192], F=[4096,8192].

    Deliberately mixes:
      * exact power-of-two corners,
      * multiples of 128 (clean vectorized-load tiles),
      * odd / non-power-of-two tails (F not divisible by 8 -> exercises the
        scalar tail path of a float4/128-bit vectorized loader).
    """
    T_LO, T_HI = 4096, 8192
    F_LO, F_HI = 4096, 8192

    shapes = []
    for i in range(int(n)):
        mode = i % 3
        if mode == 0:
            # Aligned: multiples of 128 -> fully vectorizable.
            T = 128 * rng.randint(T_LO // 128, T_HI // 128)
            Fd = 128 * rng.randint(F_LO // 128, F_HI // 128)
        elif mode == 1:
            # Ragged tail: F is odd, so no 128-bit vector width divides it.
            T = rng.randint(T_LO, T_HI)
            Fd = rng.randint(F_LO, F_HI) | 1
        else:
            # Half-aligned: T arbitrary, F a multiple of 8 but not of 128.
            T = rng.randint(T_LO, T_HI)
            Fd = 8 * rng.randint(F_LO // 8, F_HI // 8)
            if Fd % 128 == 0:
                Fd += 8
            Fd = max(F_LO, min(F_HI, Fd))

        T = max(T_LO, min(T_HI, int(T)))
        Fd = max(F_LO, min(F_HI, int(Fd)))
        shapes.append((T, Fd))
    return shapes


# ---------------------------------------------------------------------------
# Metamorphic invariants — strong, Model-independent correctness evidence
# ---------------------------------------------------------------------------
def invariants(out, ins, shape):
    """Algebraic properties any correct SwiGLU epilogue output must satisfy.

    None of these re-runs `Model`; each is derived from the *definition*
    out = g * sigmoid(g) * u via an independent route, so agreement is real
    evidence rather than a restatement of the reference code path.

    Checks
    ------
    1. shape_dtype_finite : [T, F] bf16, contiguous, all finite.
    2. odd_symmetry_identity :
           silu(g) - silu(-g) == g   (exact identity, since
           sigmoid(g) + sigmoid(-g) == 1).
           Hence  out(g, u) + out(-g, u) == g * u  for every element.
           This is evaluated by recomputing ONLY the *right-hand* side
           (g * u, a plain multiply -- no sigmoid, no silu) and comparing
           against out + out_neg, where out_neg is obtained from the closed
           form on the negated gate. It probes the sigmoid implementation
           through an identity the kernel never encodes.
    3. sign_and_bound :
           sign(out) == sign(g) * sign(u), because silu(g) has the sign of g
           (sigmoid > 0 everywhere).  Also |out| <= |u| * (|g| + 0.28), using
           the tight bound |silu(g)| <= |g| for g >= 0 and |silu(g)| <= 0.2785
           for g < 0.  Structural, no elementwise recompute of sigmoid.
    4. closed_form_probe :
           On a small random subset of coordinates, compare against a
           double-precision, scalar-path evaluation using math.exp -- a
           completely separate numerical implementation from torch.sigmoid.
    """
    results = []

    gate, up = ins[0], ins[1]
    T, Fdim = int(shape[0]), int(shape[1])

    # --- 1. shape / dtype / finiteness ------------------------------------
    ok_shape = tuple(out.shape) == (T, Fdim)
    ok_dtype = out.dtype == torch.bfloat16
    ok_contig = out.is_contiguous()
    ok_finite = bool(torch.isfinite(out.to(torch.float32)).all().item())
    results.append((
        "shape_dtype_finite",
        bool(ok_shape and ok_dtype and ok_contig and ok_finite),
        "shape={} (want {}) dtype={} contiguous={} all_finite={}".format(
            tuple(out.shape), (T, Fdim), out.dtype, ok_contig, ok_finite
        ),
    ))

    o32 = out.to(torch.float32)
    g32 = gate.to(torch.float32)
    u32 = up.to(torch.float32)

    # --- 2. odd-symmetry additivity: out(g,u) + out(-g,u) == g*u ----------
    # silu(g) + silu(-g) = g*sig(g) + (-g)*sig(-g) = g*(sig(g) - (1-sig(g)))
    #                    = g*(2*sig(g) - 1)   ... not g. Use the correct form:
    # silu(g) - silu(-g) = g*sig(g) + g*sig(-g) = g*(sig(g)+sig(-g)) = g.
    # So out(g,u) - out(-g,u) == g*u exactly (in exact arithmetic).
    try:
        neg_silu = torch.special.expit(-g32) * (-g32)          # silu(-g)
        lhs = o32 - neg_silu * u32                              # out - out(-g,u)
        rhs = g32 * u32                                         # plain multiply
        scale = rhs.abs() + lhs.abs() + 1.0
        rel = ((lhs - rhs).abs() / scale)
        max_rel = float(rel.max().item())
        # bf16 storage of `out` costs ~2^-8 relative; allow a few ulps.
        passed = max_rel < 3e-2
        detail = "max_rel|(out - silu(-g)*u) - g*u| = {:.3e} (tol 3.0e-02)".format(max_rel)
    except Exception as exc:  # pragma: no cover - defensive
        passed, detail = False, "odd-symmetry check raised {!r}".format(exc)
    results.append(("odd_symmetry_identity", bool(passed), detail))

    # --- 3. sign agreement + magnitude bound ------------------------------
    try:
        expected_sign = torch.sign(g32) * torch.sign(u32)
        got_sign = torch.sign(o32)
        # Ignore coordinates where the true product underflows bf16 to zero.
        live = expected_sign != 0
        mag = (g32 * u32).abs()
        live = live & (mag > 1e-3)
        sign_bad = int(((got_sign != expected_sign) & live).sum().item())

        # |silu(g)| <= |g| for g >= 0 ; |silu(g)| <= 0.2785 for g < 0.
        bound = u32.abs() * torch.where(
            g32 >= 0, g32.abs(), torch.full_like(g32, 0.2785)
        )
        slack = 1e-2 * bound.abs() + 1e-3
        bound_bad = int((o32.abs() > bound + slack).sum().item())

        passed = (sign_bad == 0) and (bound_bad == 0)
        detail = "sign_mismatches={} bound_violations={} (live={})".format(
            sign_bad, bound_bad, int(live.sum().item())
        )
    except Exception as exc:  # pragma: no cover - defensive
        passed, detail = False, "sign/bound check raised {!r}".format(exc)
    results.append(("sign_and_bound", bool(passed), detail))

    # --- 4. independent float64 scalar closed form on a random subset -----
    try:
        n_probe = 256
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20240824)
        rows = torch.randint(0, T, (n_probe,), generator=gen)
        cols = torch.randint(0, Fdim, (n_probe,), generator=gen)

        g_probe = gate[rows, cols].to(torch.float64).cpu().tolist()
        u_probe = up[rows, cols].to(torch.float64).cpu().tolist()
        o_probe = out[rows, cols].to(torch.float64).cpu().tolist()

        worst = 0.0
        for gv, uv, ov in zip(g_probe, u_probe, o_probe):
            # Numerically stable sigmoid via math.exp -- an implementation
            # entirely independent of torch.
            if gv >= 0.0:
                sig = 1.0 / (1.0 + math.exp(-gv))
            else:
                e = math.exp(gv)
                sig = e / (1.0 + e)
            ref = gv * sig * uv
            worst = max(worst, abs(ov - ref) / (abs(ref) + 1e-3))
        passed = worst < 3e-2
        detail = "n={} max_rel_vs_float64_math.exp = {:.3e} (tol 3.0e-02)".format(
            n_probe, worst
        )
    except Exception as exc:  # pragma: no cover - defensive
        passed, detail = False, "closed-form probe raised {!r}".format(exc)
    results.append(("closed_form_probe", bool(passed), detail))

    return results


# ---------------------------------------------------------------------------
# Honest frontier.  MUST remain the last top-level function in this file.
# ---------------------------------------------------------------------------
_COMPILED_SWIGLU = None


def baseline(gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    """Strong-but-honest frontier for the SwiGLU epilogue.

    No vendor library anchor is declared for this route, so the frontier is
    whatever the compiler can actually deliver on the machine under test:
    `torch.compile` over the eager expression `F.silu(g) * u`, which Inductor
    should fuse into a single load-load-compute-store elementwise kernel.

    Falls back to the native ATen path (`F.silu` + `mul`) if compilation is
    unavailable or raises -- the fallback is still a legitimate, if weaker,
    frontier and keeps the harness runnable everywhere.

    Numerics match the reference contract: fp32 nonlinear math, one bf16 round.
    """
    global _COMPILED_SWIGLU

    def _eager(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        gf = g.to(torch.float32)
        uf = u.to(torch.float32)
        return (F.silu(gf) * uf).to(torch.bfloat16)

    if _COMPILED_SWIGLU is None:
        try:
            _COMPILED_SWIGLU = torch.compile(_eager, dynamic=False)
        except Exception:
            _COMPILED_SWIGLU = _eager

    try:
        return _COMPILED_SWIGLU(gate_proj, up_proj)
    except Exception:
        # Compilation succeeded at wrap time but failed at trace/codegen time.
        _COMPILED_SWIGLU = _eager
        return _eager(gate_proj, up_proj)
