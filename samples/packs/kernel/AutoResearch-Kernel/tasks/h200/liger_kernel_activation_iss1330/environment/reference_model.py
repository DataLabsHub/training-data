# reference_model.py
# =============================================================================
# task_id : liger_kernel_activation_iss1330
# domain  : activation      task_type: fusion        task_mode: tensor
# track   : fusion_and_layout
# target  : h200_sm90_cuda129_torch280_nv2505  (NVIDIA H200, sm_90, Hopper)
# strong library to beat: liger_kernel.ops.swiglu.LigerSiLUMulFunction
# =============================================================================
#
# PROVENANCE / SCOPE DISCLAIMER (read this before quoting any number):
#
#   The operator contract pinned below is taken from linkedin/Liger-Kernel
#   issue #1330, whose source report is a *B300 / SM103* forward-kernel
#   regression at the Qwen3 shape (3072, 12288).  This file is a
#   B300-source-to-H200 ADAPTATION: it transfers ONLY the SiLUMul numerical
#   contract (fused SiLU, mandatory BF16 intermediate cast, BF16 multiply) to
#   an H200 / SM90 kernel-optimization task.
#
#   It is NOT a reproduction of the reported B300 regression, NOT evidence of
#   any H200 performance behaviour, and asserts NO source timing, speedup,
#   regression ratio, or optimization result on any architecture.  The only
#   thing carried across is the operator definition and its two BF16 rounding
#   boundaries.
#
# THE CONTRACT (Liger v0.8.1 SiLUMul semantics), for BF16 a, b of shape [T, F]:
#
#     gate_fp32 = a.float() * sigmoid(a.float())      # FP32 math
#     gate_bf16 = gate_fp32.to(bfloat16)              # ROUNDING BOUNDARY #1
#     out_bf16  = gate_bf16 * b                       # BF16 x BF16 -> BF16,
#                                                     # ROUNDING BOUNDARY #2
#
#   Do NOT "improve" this into all-FP32 multiplies with a single final cast:
#   that changes observable numerics and silently breaks the contract.  An
#   optimized kernel must reproduce both rounding boundaries while avoiding
#   full-sized FP32 intermediates in HBM and avoiding redundant HBM traffic.
#
# The Model below is deliberately NAIVE (it materializes a full FP32 gate
# tensor and re-reads it): it is the correctness Oracle and the timing
# denominator, not a performance target.
# =============================================================================

import math

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- 
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "liger_kernel_activation_iss1330"
DOMAIN = "activation"
TASK_TYPE = "fusion"
TASK_MODE = "tensor"

# Public / representative shapes (T, F).
WORKLOAD = [
    (3072, 12288),   # Qwen3 shape named in the source report
    (4096, 4096),
    (8192, 8192),
    (8192, 16384),
]

# Legal shape domain (the only axis values a hidden shape may use).
SHAPE_DOMAIN = {
    "T": [3072, 4096, 6144, 8192],
    "F": [4096, 6144, 8192, 11008, 12288, 14336, 16384],
}

# Forward-only task: no gradients are checked.
GRAD_INPUTS = []

# Single output tensor.
OUTPUT_INDEX = 0

# BF16 output with two rounding boundaries -> BF16-aware tolerances.
TASK_RTOL = 2e-2
TASK_ATOL = 2e-2

# Memory-bound elementwise op: weight bandwidth efficiency substantially.
MEM_WEIGHT = 0.5

BASELINE_CONTRACT = {
    "input_dtype": "bf16",
    "output_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "allow_tf32": False,
    "note": (
        "Liger v0.8.1 SiLUMul contract with TWO BF16 rounding boundaries: "
        "(1) the SiLU gate is computed in FP32 (a.float() * sigmoid(a.float())) "
        "and then ROUNDED TO BF16; (2) that BF16 gate is multiplied by the BF16 "
        "tensor b and the product is ROUNDED TO BF16 again. Accumulation/"
        "transcendental math is FP32; TF32 is irrelevant here and must stay "
        "disabled. Replacing this with all-FP32 multiplies followed by a single "
        "final cast is a contract violation, not an optimization. Inputs must "
        "not be mutated and the output must not alias either input."
    ),
}

# Sentinel row layout used by make_inputs / invariants (requires T >= 4).
_ROW_REFLECT_POS = 0   # a[0], b[0]
_ROW_REFLECT_NEG = 1   # a[1] = -a[0], b[1] = b[0]
_ROW_SCALE_BASE = 2    # a[2], b[2]
_ROW_SCALE_DBL = 3     # a[3] = a[2], b[3] = 2 * b[2]

# Deterministic sentinel patterns (all exactly BF16-representable except the
# 1e-3 / 1e-2 entries, which round once and identically under negation).
# Covers: negative saturation, positive linear regime, exact zero, near zero.
_PAT_A0 = [-20.0, -8.0, -1.0, 0.0, 1.0e-3, -1.0e-3, 4.0, 16.0]
_PAT_B0 = [1.0, 0.0, -1.5, 2.0, -0.25, 0.0, 0.75, -1.0]
_PAT_A2 = [-16.0, -3.0, 0.0, 0.5, 2.0, 9.0, -0.5, 1.0e-2]
_PAT_B2 = [0.5, -1.0, 1.25, 0.0, -2.0, 0.25, 1.0, -0.75]


# -----------------------------------------------------------------------------
# The NAIVE reference (correctness Oracle + timing denominator)
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """Plain-PyTorch SiLUMul, written the obvious unfused way.

    forward(a, b) with a, b contiguous BF16 [T, F]:
        gate_fp32 = a.float() * sigmoid(a.float())
        gate_bf16 = gate_fp32.to(bfloat16)      # rounding boundary #1
        out       = gate_bf16 * b               # rounding boundary #2 (BF16)

    This deliberately materializes full-sized FP32 intermediates and re-reads
    them from HBM. That is the point: it is the honest naive denominator.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        a_fp32 = a.float()                       # full-size FP32 intermediate
        gate_fp32 = a_fp32 * torch.sigmoid(a_fp32)   # second full-size FP32 buf
        gate_bf16 = gate_fp32.to(torch.bfloat16)     # BF16 rounding boundary #1
        out = gate_bf16 * b                          # BF16 rounding boundary #2
        return out.contiguous()


def get_init_inputs():
    """Constructor args for Model(...) -- none."""
    return []


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
def _pattern_row(pattern, F, device):
    """Tile `pattern` (python floats) across F columns, as BF16."""
    pat = torch.tensor(pattern, dtype=torch.float32, device=device)
    idx = torch.arange(F, device=device) % pat.numel()
    return pat[idx].to(torch.bfloat16)


def make_inputs(shape, device, seed=0):
    """Deterministic per-seed BF16 inputs [a, b] of shape (T, F).

    Non-sentinel data depends on `seed`; the reserved sentinel rows 0..3 are
    fixed by construction so that `invariants` has stable algebraic anchors:

        rows 0/1 : a[1] = -a[0],  b[1] = b[0]        (reflection identity)
        rows 2/3 : a[3] =  a[2],  b[3] = 2 * b[2]    (b-homogeneity)

    Exact-zero entries are injected into both a and b. All values are finite
    and bounded (|a| <= 20, |b| <= 4), so no overflow/NaN is possible.
    """
    T, F = int(shape[0]), int(shape[1])
    device = torch.device(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    a32 = torch.empty((T, F), dtype=torch.float32, device=device).uniform_(
        -12.0, 12.0, generator=gen
    )
    b32 = torch.empty((T, F), dtype=torch.float32, device=device).uniform_(
        -2.0, 2.0, generator=gen
    )

    a = a32.to(torch.bfloat16)
    b = b32.to(torch.bfloat16)
    del a32, b32

    # Scatter exact zeros through the seed-dependent bulk as well, on a fixed
    # (seed-independent) lattice so the zero sentinel is always exercised.
    if T > 4:
        bulk = slice(4, T)
        col = torch.arange(F, device=device)
        a_zero_cols = (col % 257) == 3
        b_zero_cols = (col % 263) == 11
        a[bulk, a_zero_cols] = 0.0
        b[bulk, b_zero_cols] = 0.0

    if T >= 4:
        a0 = _pattern_row(_PAT_A0, F, device)
        b0 = _pattern_row(_PAT_B0, F, device)
        a2 = _pattern_row(_PAT_A2, F, device)
        b2 = _pattern_row(_PAT_B2, F, device)

        a[_ROW_REFLECT_POS] = a0
        b[_ROW_REFLECT_POS] = b0
        a[_ROW_REFLECT_NEG] = -a0            # exact in BF16
        b[_ROW_REFLECT_NEG] = b0

        a[_ROW_SCALE_BASE] = a2
        b[_ROW_SCALE_BASE] = b2
        a[_ROW_SCALE_DBL] = a2
        b[_ROW_SCALE_DBL] = b2 * 2.0         # exact in BF16 (power of two)

    return [a.contiguous(), b.contiguous()]


def sample_shapes(rng, n):
    """Unseen (T, F) shapes drawn only from the declared choice sets.

    Never returns a WORKLOAD shape; hidden-shape tripwire.
    """
    def _pick(seq):
        if hasattr(rng, "randrange"):
            return seq[rng.randrange(len(seq))]
        if hasattr(rng, "integers"):
            return seq[int(rng.integers(0, len(seq)))]
        return seq[int(rng.randint(0, len(seq) - 1))]

    banned = set(WORKLOAD)
    pool = [
        (t, f)
        for t in SHAPE_DOMAIN["T"]
        for f in SHAPE_DOMAIN["F"]
        if (t, f) not in banned
    ]
    out, seen = [], set()
    attempts = 0
    while len(out) < int(n) and attempts < 4096 and len(seen) < len(pool):
        t = _pick(SHAPE_DOMAIN["T"])
        f = _pick(SHAPE_DOMAIN["F"])
        attempts += 1
        cand = (t, f)
        if cand in banned or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    # Deterministic top-up if the rng kept colliding.
    for cand in pool:
        if len(out) >= int(n):
            break
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out[: int(n)]


# -----------------------------------------------------------------------------
# Independent oracle: math definition in float64, never touches Model/Liger
# -----------------------------------------------------------------------------
def independent_oracle(ins, shape):
    """Recompute SiLUMul from the mathematical definition, in float64.

    Called by the harness as independent_oracle(ins, shape) where `ins` is the
    exact list returned by make_inputs. Uses the numerically stable identity

        sigmoid(x) = 0.5 * (1 + tanh(x / 2))

    and reproduces BOTH BF16 rounding boundaries explicitly (gate cast, then a
    separate multiply whose result is cast). Chunked over rows to bound the
    float64 working set. Does NOT call Model or the Liger baseline.
    """
    a, b = ins[0], ins[1]
    T, F = int(shape[0]), int(shape[1])
    out = torch.empty((T, F), dtype=torch.bfloat16, device=a.device)

    max_elems = 1 << 24                      # ~128 MiB per float64 buffer
    rows = max(1, min(T, max_elems // max(1, F)))

    for r0 in range(0, T, rows):
        r1 = min(T, r0 + rows)
        a64 = a[r0:r1].to(torch.float64)
        sig = 0.5 * (1.0 + torch.tanh(a64 * 0.5))
        gate64 = a64 * sig
        gate_bf16 = gate64.to(torch.bfloat16)        # rounding boundary #1
        del a64, sig, gate64
        prod64 = gate_bf16.to(torch.float64) * b[r0:r1].to(torch.float64)
        out[r0:r1] = prod64.to(torch.bfloat16)       # rounding boundary #2
        del gate_bf16, prod64

    return out.contiguous()


# -----------------------------------------------------------------------------
# Metamorphic invariants (strong, independent of Model's code path)
# -----------------------------------------------------------------------------
def invariants(out, ins, shape):
    """Algebraic properties any correct SiLUMul output must satisfy.

    None of these re-runs Model or the baseline; they are derived from the
    inputs and from closed-form identities of SiLU. Thresholds are BF16-aware
    but never looser than 3e-2.

    1. structure   : shape / dtype / finiteness.
    2. zero        : a == 0 or b == 0  =>  out == 0 exactly.
    3. reflection  : silu(x) - silu(-x) == x  =>  out[0] - out[1] ~= a[0]*b[0].
    4. homogeneity : f(a, 2b) == 2 f(a, b)    =>  out[3] ~= 2 * out[2].
    """
    TOL = 3e-2
    res = []

    a, b = ins[0], ins[1]
    T, F = int(shape[0]), int(shape[1])

    # --- 1. structure --------------------------------------------------------
    ok_shape = tuple(out.shape) == (T, F)
    ok_dtype = out.dtype == torch.bfloat16
    finite = bool(torch.isfinite(out.to(torch.float32)).all().item())
    res.append((
        "structure_shape_dtype_finite",
        bool(ok_shape and ok_dtype and finite),
        f"shape={tuple(out.shape)} (want {(T, F)}), dtype={out.dtype} "
        f"(want torch.bfloat16), all_finite={finite}",
    ))

    # --- 2. exact-zero sentinels --------------------------------------------
    zero_mask = (a == 0) | (b == 0)
    n_zero = int(zero_mask.sum().item())
    if n_zero == 0:
        res.append((
            "exact_zero_sentinel", False,
            "no exact-zero entries found in the inputs; sentinel not exercised",
        ))
    else:
        bad = int((out[zero_mask] != 0).sum().item())
        res.append((
            "exact_zero_sentinel",
            bad == 0,
            f"{n_zero} sentinel positions with a==0 or b==0; "
            f"{bad} produced a nonzero output (want 0)",
        ))

    if T < 4:
        res.append((
            "paired_row_sentinels", False,
            f"T={T} < 4: reserved sentinel rows unavailable",
        ))
        return res

    eps = 1e-6

    # --- 3. reflection identity: f(a,b) - f(-a,b) == a*b ---------------------
    a0 = a[_ROW_REFLECT_POS].to(torch.float64)
    b0 = b[_ROW_REFLECT_POS].to(torch.float64)
    f0 = out[_ROW_REFLECT_POS].to(torch.float64)
    f1 = out[_ROW_REFLECT_NEG].to(torch.float64)
    expect = a0 * b0                            # exact product of two BF16s
    denom = expect.abs() + f0.abs() + f1.abs() + eps
    rel = ((f0 - f1) - expect).abs() / denom
    max_rel = float(rel.max().item())
    res.append((
        "reflection_identity_rows_0_1",
        max_rel <= TOL,
        f"max relative deviation of (out[0]-out[1]) from a[0]*b[0] = "
        f"{max_rel:.3e} (tol {TOL:.1e}); uses silu(x)-silu(-x)==x",
    ))

    # --- 4. b-homogeneity: f(a, 2b) == 2 f(a, b) ----------------------------
    same_a = bool(torch.equal(a[_ROW_SCALE_DBL], a[_ROW_SCALE_BASE]))
    dbl_b = bool(torch.equal(
        b[_ROW_SCALE_DBL].to(torch.float64),
        b[_ROW_SCALE_BASE].to(torch.float64) * 2.0,
    ))
    f2 = out[_ROW_SCALE_BASE].to(torch.float64)
    f3 = out[_ROW_SCALE_DBL].to(torch.float64)
    denom_h = f3.abs() + (2.0 * f2).abs() + eps
    rel_h = (f3 - 2.0 * f2).abs() / denom_h
    max_rel_h = float(rel_h.max().item())
    res.append((
        "b_homogeneity_rows_2_3",
        bool(same_a and dbl_b and max_rel_h <= TOL),
        f"setup_ok(a3==a2:{same_a}, b3==2*b2:{dbl_b}); max relative deviation "
        f"of out[3] from 2*out[2] = {max_rel_h:.3e} (tol {TOL:.1e})",
    ))

    return res
