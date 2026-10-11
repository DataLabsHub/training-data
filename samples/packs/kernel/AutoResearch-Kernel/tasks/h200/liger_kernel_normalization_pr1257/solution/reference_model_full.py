"""
reference_model.py — task_id: liger_kernel_normalization_pr1257

Domain      : normalization
Task type   : fusion  (optimization track: fusion_and_layout)
Task mode   : tensor (forward-only)

WHAT THIS TASK IS
-----------------
Forward Llama/Megatron-style RMSNorm over large contiguous BF16 activations:

    ms   = mean(x.float()**2, dim=-1)          # FP32 row reduction
    rstd = rsqrt(ms + 1e-6)                    # FP32 reciprocal std
    xhat = (x.float() * rstd).to(bf16)         # Llama "cast-before-weight"
    y    = xhat * weight                       # BF16 channel gain

The kernel opportunity is to fuse the FP32 row reduction, the reciprocal
standard deviation, the normalization, its *required* BF16 cast, and the BF16
channel gain into a single pass over the [T, H] activation, so the tensor is
read once and written once instead of being materialized several times.

PROVENANCE / CALIBRATION NOTE
-----------------------------
The originating measurements come from the Megatron RMSNorm benchmark context
in linkedin/Liger-Kernel PR #1257. *All* source numbers in that PR — timing,
memory, and speedup — are H100-only and DO NOT TRANSFER. Nothing from that
report is reused as an expectation here.

This task is NEWLY TARGETED to:
    target_id          = h200_sm90_cuda129_torch280_nv2505
    gpu_model          = NVIDIA H200
    compute_capability = 9.0 (Hopper / SM90)

Every performance figure for this task must be re-measured on H200. Treat any
H100 datapoint as historical context only.

The Model below is the NAIVE, obviously-correct plain-PyTorch reference: it is
both the correctness Oracle and the timing denominator. It is deliberately not
optimized. `baseline(...)` calls the strong library (LigerRMSNormFunction) and
is the honest frontier to beat.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Task metadata
# --------------------------------------------------------------------------- #

TASK_ID = "liger_kernel_normalization_pr1257"
TASK_DOMAIN = "normalization"
TASK_TYPE = "fusion"
TASK_MODE = "tensor"
OPTIMIZATION_TRACK = "fusion_and_layout"

TARGET_ID = "h200_sm90_cuda129_torch280_nv2505"
TARGET_GPU = "NVIDIA H200"
TARGET_COMPUTE_CAPABILITY = "9.0"
TARGET_ARCH = "hopper"

EPS = 1e-6

# Public shapes (T, H). Hidden shapes are drawn by sample_shapes().
WORKLOAD = [
    (4096, 2048),
    (4096, 4096),
    (4096, 8192),
    (8192, 4096),
    (16384, 8192),
]

# Legal shape domain (axes) for this task.
SHAPE_DOMAIN = {
    "T": [4096, 8192, 12288, 16384],
    "H": [2048, 3072, 4096, 5120, 6144, 7168, 8192],
}

# Single BF16 output tensor.
OUTPUT_INDEX = 0

# Forward-only task: no gradients are checked.
GRAD_INPUTS = []

# BF16 storage with FP32 accumulation, plus the mandatory intermediate BF16
# cast of the normalized activations (Llama casting mode) => two roundings.
TASK_RTOL = 1.6e-2
TASK_ATOL = 1.0e-2

# This is a bandwidth-bound fusion; memory traffic carries real weight.
MEM_WEIGHT = 0.5

BASELINE_CONTRACT = {
    "input_dtype": "bf16",
    "output_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "allow_tf32": False,
    "note": (
        "Llama cast-before-weight semantics: the row mean-square and rsqrt are "
        "accumulated in fp32 from the bf16 activations with eps=1e-6 and "
        "offset=0.0; the normalized activations are cast BACK TO BF16 *before* "
        "the bf16 channel gain is applied, and the product is returned as bf16. "
        "The gain must not be folded into the fp32 normalization. No mean "
        "subtraction, no offset on the weight, inputs are never mutated "
        "(in_place=False), and the output must be a fresh contiguous bf16 "
        "tensor that does not alias x or weight. allow_tf32 is irrelevant to "
        "the math here and is pinned False so no reduction is silently demoted. "
        "Source PR measurements are H100-only; this task is newly targeted to "
        "NVIDIA H200 / SM90."
    ),
}


# --------------------------------------------------------------------------- #
# Naive reference model (correctness anchor + timing denominator)
# --------------------------------------------------------------------------- #

class Model(nn.Module):
    """Naive plain-PyTorch Llama/Megatron RMSNorm forward.

    Unfused on purpose: the fp32 upcast, the row reduction, the rsqrt, the
    normalization, the bf16 re-cast and the bf16 gain are separate eager ops,
    each materializing a full [T, H] temporary. eps is fixed internally.
    """

    def __init__(self):
        super().__init__()
        self.eps = EPS

    def forward(self, x, weight):
        # x      : contiguous bf16 [T, H]
        # weight : contiguous bf16 [H]
        x_fp32 = x.float()                                   # [T, H] fp32
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)   # fp32 reduce
        rstd = torch.rsqrt(mean_square + self.eps)               # fp32
        x_hat = (x_fp32 * rstd).to(torch.bfloat16)               # Llama cast
        out = x_hat * weight                                     # bf16 gain
        return out.contiguous()


def get_init_inputs():
    """Model takes no constructor arguments."""
    return []


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def _mix_seed(seed) -> int:
    """Deterministic, well-spread 63-bit seed so nearby seeds give unlike data."""
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    s ^= s >> 33
    s = (s * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    s ^= s >> 29
    return s & 0x7FFFFFFFFFFFFFFF


def _pm_one_rows(row_idx: torch.Tensor, H: int, device) -> torch.Tensor:
    """Deterministic +/-1 pattern (mean-square exactly 1.0) for given rows."""
    cols = torch.arange(H, device=device, dtype=torch.int64)
    parity = (cols.unsqueeze(0) + row_idx.unsqueeze(1)) % 2
    return (1.0 - 2.0 * parity.to(torch.float32))


def make_inputs(shape, device, seed=0):
    """Return [x, weight] for shape (T, H).

    Deterministic for a given (shape, device, seed); changing `seed` changes
    every non-sentinel row of x and every channel of weight.

    Guarantees:
      * x is contiguous bf16 [T, H], finite, non-constant per row, and after
        the bf16 conversion EVERY row has mean-square >= 0.25 (a deficient row
        is replaced by a deterministic +/-1 pattern with mean-square 1.0).
      * weight is contiguous bf16 [H], strictly positive, channel-varying,
        in [0.75, 1.25].
      * Reserved sentinel rows (present whenever T >= 3), with |x0| <= 1.5:
            row 1 == 2 * row 0   (exact in bf16: multiply by a power of two)
            row 2 == -row 0      (exact in bf16: sign flip)
        These let scale-invariance and odd-symmetry be certified without
        rerunning the reference.
    """
    T, H = int(shape[0]), int(shape[1])
    dev = torch.device(device)

    gen = torch.Generator(device=dev)
    gen.manual_seed(_mix_seed(seed))

    # Row-varying magnitudes so the reduction sees a real dynamic range.
    x32 = torch.randn((T, H), generator=gen, device=dev, dtype=torch.float32)
    row_scale = 0.5 + 1.5 * torch.rand((T, 1), generator=gen, device=dev,
                                       dtype=torch.float32)
    x32 = x32 * row_scale
    x = x32.to(torch.bfloat16)
    del x32

    # Repair any row whose post-bf16 mean-square fell below 0.25.
    ms = x.float().pow(2).mean(dim=-1)
    bad = ms < 0.25
    if bool(bad.any()):
        idx = torch.nonzero(bad, as_tuple=True)[0]
        x[idx] = _pm_one_rows(idx, H, dev).to(torch.bfloat16)

    # Reserved sentinel rows 0/1/2 with bounded row 0.
    if T >= 3:
        r0 = x[0].float().clamp_(-1.5, 1.5)
        if float(r0.pow(2).mean()) < 0.25:
            r0 = _pm_one_rows(torch.zeros(1, dtype=torch.int64, device=dev),
                              H, dev)[0]
        b0 = r0.to(torch.bfloat16)
        x[0] = b0
        x[1] = (b0.float() * 2.0).to(torch.bfloat16)   # exact in bf16
        x[2] = (-b0.float()).to(torch.bfloat16)        # exact in bf16

    # Positive, channel-varying gain in [0.75, 1.25].
    w32 = 0.75 + 0.5 * torch.rand((H,), generator=gen, device=dev,
                                  dtype=torch.float32)
    weight = w32.to(torch.bfloat16)

    return [x.contiguous(), weight.contiguous()]


# --------------------------------------------------------------------------- #
# Hidden-shape tripwire
# --------------------------------------------------------------------------- #

def _rand_below(rng, n: int) -> int:
    """Uniform int in [0, n) from random.Random, np.random.Generator, or RandomState."""
    if n <= 1:
        return 0
    for attr in ("randrange",):
        f = getattr(rng, attr, None)
        if f is not None:
            return int(f(n))
    f = getattr(rng, "integers", None)          # numpy Generator
    if f is not None:
        return int(f(0, n))
    f = getattr(rng, "random", None)            # random.Random / np Generator
    if f is not None:
        return min(n - 1, int(float(f()) * n))
    f = getattr(rng, "random_sample", None)     # legacy RandomState
    if f is not None:
        return min(n - 1, int(float(f()) * n))
    raise TypeError("unsupported rng object passed to sample_shapes")


def sample_shapes(rng, n):
    """Unique UNSEEN (T, H) shapes drawn only from the declared choice sets.

    Never returns a WORKLOAD shape, so a kernel that special-cases the public
    shapes is caught immediately.
    """
    public = set(tuple(s) for s in WORKLOAD)
    pool = [(t, h) for t in SHAPE_DOMAIN["T"] for h in SHAPE_DOMAIN["H"]
            if (t, h) not in public]
    # Fisher-Yates with the harness rng.
    for i in range(len(pool) - 1, 0, -1):
        j = _rand_below(rng, i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    k = max(0, min(int(n), len(pool)))
    return pool[:k]


# --------------------------------------------------------------------------- #
# Independent oracle (float64, formula-first; never calls Model or Liger)
# --------------------------------------------------------------------------- #

def independent_oracle(ins, shape):
    """Recompute RMSNorm straight from the definition in float64.

    Independent of Model's code path: uses an explicit float64 sum-of-squares
    and a *square root with division* (not rsqrt, not mean()), then performs
    the mandated bf16 cast of the normalized activations before a separate
    gain multiplication and the final bf16 cast.
    """
    x, weight = ins[0], ins[1]
    T, H = int(shape[0]), int(shape[1])

    w32 = weight.to(torch.float32)
    out = torch.empty((T, H), device=x.device, dtype=torch.bfloat16)

    rows_per_chunk = max(1, min(T, (1 << 24) // max(H, 1)))
    inv_H = 1.0 / float(H)
    for start in range(0, T, rows_per_chunk):
        stop = min(T, start + rows_per_chunk)
        blk = x[start:stop].to(torch.float64)
        sumsq = (blk * blk).sum(dim=-1, keepdim=True)          # float64
        rms = torch.sqrt(sumsq * inv_H + 1e-6)                 # float64 sqrt
        x_hat = (blk / rms).to(torch.bfloat16)                 # cast, then...
        out[start:stop] = (x_hat.to(torch.float32) * w32).to(torch.bfloat16)
        del blk, sumsq, rms, x_hat

    return out


# --------------------------------------------------------------------------- #
# Metamorphic invariants (no Model rerun)
# --------------------------------------------------------------------------- #

def _as_out(out):
    if isinstance(out, (list, tuple)):
        return out[OUTPUT_INDEX]
    return out


def invariants(out, ins, shape):
    """Algebraic properties any correct RMSNorm output must satisfy.

    None of these rerun Model; they are derived from the definition of RMSNorm
    and from the sentinel structure planted by make_inputs.
    """
    checks = []
    y = _as_out(out)
    x, weight = ins[0], ins[1]
    T, H = int(shape[0]), int(shape[1])

    # 1) Structural: shape, dtype, finiteness, non-aliasing of the input.
    ok_shape = tuple(y.shape) == (T, H)
    ok_dtype = (y.dtype == torch.bfloat16)
    ok_dev = (y.device == x.device)
    n_bad = int((~torch.isfinite(y.to(torch.float32))).sum())
    not_alias = y.data_ptr() != x.data_ptr()
    passed = ok_shape and ok_dtype and ok_dev and (n_bad == 0) and not_alias
    checks.append((
        "shape_dtype_finite_no_alias",
        bool(passed),
        f"shape={tuple(y.shape)} exp={(T, H)} dtype={y.dtype} "
        f"device={y.device} non_finite={n_bad} aliases_x={not not_alias}",
    ))

    yf_head = y[:3].to(torch.float32) if T >= 3 else None

    # 2) Positive-scale invariance: x1 = 2*x0  =>  RMSNorm(x1) == RMSNorm(x0).
    if T >= 3:
        y0, y1 = yf_head[0], yf_head[1]
        slack = (y1 - y0).abs() - (TASK_ATOL + TASK_RTOL * y0.abs())
        worst = float(slack.max())
        max_abs = float((y1 - y0).abs().max())
        checks.append((
            "positive_scale_invariance_row1_eq_row0",
            bool(worst <= 0.0),
            f"x[1]==2*x[0] so outputs must match; max|dy|={max_abs:.4e}, "
            f"worst tolerance slack={worst:.4e} (rtol={TASK_RTOL}, atol={TASK_ATOL})",
        ))
    else:
        checks.append(("positive_scale_invariance_row1_eq_row0", True,
                       f"skipped: T={T} < 3"))

    # 3) Odd symmetry: x2 = -x0 => RMSNorm(x2) == -RMSNorm(x0) (exact in bf16).
    if T >= 3:
        y0, y2 = yf_head[0], yf_head[2]
        r_sym, a_sym = 1.0e-2, 4.0e-3
        slack = (y2 + y0).abs() - (a_sym + r_sym * y0.abs())
        worst = float(slack.max())
        max_abs = float((y2 + y0).abs().max())
        checks.append((
            "odd_symmetry_row2_eq_neg_row0",
            bool(worst <= 0.0),
            f"x[2]==-x[0] so out[2]==-out[0]; max|out2+out0|={max_abs:.4e}, "
            f"worst tolerance slack={worst:.4e} (rtol={r_sym}, atol={a_sym})",
        ))
    else:
        checks.append(("odd_symmetry_row2_eq_neg_row0", True,
                       f"skipped: T={T} < 3"))

    # 4) Unit RMS after removing the gain: z = out / weight must have per-row
    #    mean-square ~= 1. This is the defining property of RMSNorm and also
    #    proves the gain was applied per-channel (and not folded pre-cast).
    w32 = weight.to(torch.float32)
    tol_ms = 4.0e-2
    worst_dev = 0.0
    worst_row = -1
    rows_per_chunk = max(1, min(T, (1 << 24) // max(H, 1)))
    for start in range(0, T, rows_per_chunk):
        stop = min(T, start + rows_per_chunk)
        z = y[start:stop].to(torch.float32) / w32
        ms = (z * z).mean(dim=-1)
        dev = (ms - 1.0).abs()
        m, i = torch.max(dev, dim=0)
        m = float(m)
        if m > worst_dev:
            worst_dev = m
            worst_row = start + int(i)
        del z, ms, dev
    checks.append((
        "unit_row_mean_square_after_gain_removal",
        bool(worst_dev <= tol_ms),
        f"max_row |mean((out/weight)^2) - 1| = {worst_dev:.4e} at row "
        f"{worst_row} (tol={tol_ms})",
    ))

    return checks


# --------------------------------------------------------------------------- #
# Strong-library baseline (the honest frontier)
# --------------------------------------------------------------------------- #

_LIGER_EXPECTED_PARAMS = ("X", "W", "eps", "offset", "casting_mode",
                          "in_place", "row_mode")


def _load_liger_rms_norm_function():
    """Import the pinned Liger entrypoint. Fails closed; never falls back."""
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    except Exception as exc:  # noqa: BLE001 - fail closed, do not substitute
        raise RuntimeError(
            "baseline unavailable: could not import "
            "liger_kernel.ops.rms_norm.LigerRMSNormFunction from the pinned "
            f"release ({type(exc).__name__}: {exc}). No fallback to "
            "LigerRMSNorm module or to PyTorch is permitted."
        ) from exc

    fwd = getattr(LigerRMSNormFunction, "forward", None)
    if fwd is None:
        raise RuntimeError(
            "baseline unavailable: LigerRMSNormFunction has no forward()."
        )
    try:
        params = [p for p in inspect.signature(fwd).parameters]
    except (TypeError, ValueError):
        params = None
    if params is not None:
        tail = [p for p in params if p not in ("ctx", "self")]
        # Only fail when the signature is positively readable AND mismatched.
        if tail and not any(p in ("args", "kwargs") for p in tail):
            if tuple(tail[:7]) != _LIGER_EXPECTED_PARAMS:
                raise RuntimeError(
                    "baseline unavailable: unexpected LigerRMSNormFunction."
                    f"forward signature {tuple(tail)}; expected leading "
                    f"{_LIGER_EXPECTED_PARAMS}. Refusing to probe alternative "
                    "signatures."
                )
    return LigerRMSNormFunction


def baseline(x, weight, init_args=None):
    """Strong library reference: Liger's fused RMSNorm forward.

    Exactly one call, exactly one signature:
        LigerRMSNormFunction.apply(x, weight, 1e-6, 0.0, "llama", False, None)
    i.e. eps=1e-6, offset=0.0, casting_mode="llama" (cast-before-weight),
    in_place=False (inputs are never mutated), row_mode=None (auto).

    Import or signature problems raise; there is deliberately no fallback to
    the LigerRMSNorm nn.Module or to a PyTorch implementation.

    NOTE: any published speedup for this call from PR #1257 is H100-only and
    must be re-measured on H200 / SM90.
    """
    if init_args is None:
        pass
    elif isinstance(init_args, list) and len(init_args) == 0:
        pass
    else:
        raise ValueError(
            f"baseline() takes no init args; expected None or [], got "
            f"{init_args!r}"
        )

    LigerRMSNormFunction = _load_liger_rms_norm_function()

    try:
        return LigerRMSNormFunction.apply(x, weight, 1e-6, 0.0, "llama",
                                          False, None)
    except TypeError as exc:
        raise RuntimeError(
            "baseline unavailable: LigerRMSNormFunction.apply rejected the "
            "pinned call (x, weight, 1e-6, 0.0, 'llama', False, None) -> "
            f"{exc}. Refusing to probe alternative signatures or fall back."
        ) from exc
