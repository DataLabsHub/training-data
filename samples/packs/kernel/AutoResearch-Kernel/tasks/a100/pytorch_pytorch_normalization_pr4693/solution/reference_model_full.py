"""
reference_model.py — task_id: pytorch_pytorch_normalization_pr4693

Mixed-precision forward BatchNorm over a 2-D activation matrix (T, H).

  * x       : fp16, shape (T, H), contiguous   -- T samples, H features
  * weight  : fp32, shape (H,), positive       -- per-feature affine scale
  * bias    : fp32, shape (H,)                 -- per-feature affine shift

Semantics (training-mode batch norm, no running stats):

    mean_h = mean_t(x[:, h])
    var_h  = mean_t((x[:, h] - mean_h)^2)          # BIASED variance
    y[t,h] = (x[t,h] - mean_h) * rsqrt(var_h + eps) * weight[h] + bias[h]

All reduction / normalization arithmetic happens in fp32; the result is
rounded to fp16 exactly once, at the very end.

The real hotspot is the heterogeneous-operand case (fp16 activations with
fp32 affine parameters): a fused kernel should accumulate per-feature
statistics in fp32 and emit the affine chain into an fp16 output without
materializing fp32 intermediates or rejecting the mixed dtypes.

NOTE: despite stale operator metadata elsewhere in the task bundle, this is
BatchNorm (reduction over the T axis, statistics per feature h), not
LayerNorm or GroupNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Shape domain: the public "shape" tuple is (case_id,), mapped to (T, H).
# --------------------------------------------------------------------------
CASES = {
    0: (8192, 2048),
    1: (16384, 1024),
    2: (4096, 8192),
    3: (32768, 512),
    4: (128, 128),
    5: (1536, 16384),
    6: (3072, 8192),
    7: (6144, 4096),
    8: (12288, 2048),
    9: (24576, 1024),
    10: (32768, 768),
    11: (20480, 1224),
}

PUBLIC_CASE_IDS = [0, 1, 2, 3]
HIDDEN_CASE_IDS = [4, 5, 6, 7, 8, 9, 10, 11]

WORKLOAD = [(0,), (1,), (2,), (3,)]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-3
TASK_ATOL = 3e-3
GRAD_INPUTS = []
OUTPUT_INDEX = 0

EPS = 1e-5
VAR_FLOOR = 0.25  # every feature's post-fp16 biased variance is >= this

BASELINE_CONTRACT = {
    'input_dtype': 'mixed',
    'output_dtype': 'fp16',
    'accumulation_dtype': 'fp32',
    'allow_tf32': False,
    'note': (
        'Heterogeneous operands: x is fp16 (T, H) contiguous while weight and '
        'bias are fp32 (H,). Per-feature mean and BIASED variance are reduced '
        'over the T axis and the whole normalize+affine chain is evaluated in '
        'fp32 with eps=1e-5; the fp32 result is rounded to fp16 exactly once '
        'on store. No running statistics, no state mutation, forward only.'
    ),
}


def _resolve(shape):
    """(case_id,) -> (T, H)."""
    if isinstance(shape, (list, tuple)):
        if len(shape) != 1:
            raise ValueError(f"expected a 1-tuple (case_id,), got {shape!r}")
        case_id = int(shape[0])
    else:
        case_id = int(shape)
    if case_id not in CASES:
        raise ValueError(f"unknown case_id {case_id}; valid ids: {sorted(CASES)}")
    return CASES[case_id]


def reduction_length(shape):
    """Number of elements reduced per output statistic (the T axis)."""
    T, _ = _resolve(shape)
    return T


# --------------------------------------------------------------------------
# The NAIVE, obviously-correct reference (correctness Oracle + timing denom).
# --------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x, weight, bias):
        # Promote once; every reduction and the affine chain run in fp32.
        xf = x.float()
        mean = xf.mean(dim=0)                       # (H,)
        var = ((xf - mean) ** 2).mean(dim=0)        # (H,) biased variance
        inv_std = torch.rsqrt(var + self.eps)       # (H,)
        y = (xf - mean) * inv_std * weight.float() + bias.float()
        return y.to(x.dtype)                        # single fp16 rounding


def get_init_inputs():
    return [EPS]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def make_inputs(shape, device, seed=0):
    """Deterministic in `seed`; returns [x fp16 (T,H), weight fp32 (H,), bias fp32 (H,)]."""
    T, H = _resolve(shape)
    dev = torch.device(device)
    g = torch.Generator(device='cpu').manual_seed(int(seed) * 9781 + 12345)

    # Per-feature scale/offset so features are genuinely heterogeneous.
    scale = 0.6 + 1.0 * torch.rand(H, generator=g, dtype=torch.float32)   # [0.6, 1.6)
    offset = -2.0 + 4.0 * torch.rand(H, generator=g, dtype=torch.float32)  # [-2, 2)

    x32 = torch.randn(T, H, generator=g, dtype=torch.float32) * scale + offset
    x = x32.to(torch.float16)

    # Guarantee a non-degenerate biased variance for EVERY feature, measured
    # AFTER the fp16 cast. Repair deficient columns with a deterministic
    # alternating +/-1 pattern (biased variance exactly 1.0 in fp16).
    xf = x.float()
    col_var = ((xf - xf.mean(dim=0)) ** 2).mean(dim=0)
    bad = (col_var < VAR_FLOOR) | ~torch.isfinite(col_var)
    if bool(bad.any()):
        pattern = torch.ones(T, dtype=torch.float16)
        pattern[1::2] = -1.0
        idx = torch.nonzero(bad, as_tuple=False).flatten()
        x[:, idx] = pattern.unsqueeze(1).expand(T, idx.numel()).clone()

    weight = 0.75 + 0.5 * torch.rand(H, generator=g, dtype=torch.float32)   # [0.75, 1.25)
    bias = -0.25 + 0.5 * torch.rand(H, generator=g, dtype=torch.float32)    # [-0.25, 0.25)

    x = x.contiguous().to(dev)
    weight = weight.contiguous().to(dev)
    bias = bias.contiguous().to(dev)
    return [x, weight, bias]


def sample_shapes(rng, n):
    """Unseen (hidden) case ids only: 4..11."""
    pool = list(HIDDEN_CASE_IDS)
    out = []
    while len(out) < n:
        chunk = list(pool)
        rng.shuffle(chunk)
        out.extend(chunk)
    return [(cid,) for cid in out[:n]]


# --------------------------------------------------------------------------
# Independent oracle: different algebraic path (fp64, E[x^2] - E[x]^2).
# --------------------------------------------------------------------------
def independent_oracle(ins, shape):
    x, weight, bias = ins[0], ins[1], ins[2]
    xd = x.double()
    m1 = xd.mean(dim=0)
    m2 = (xd * xd).mean(dim=0)
    var = (m2 - m1 * m1).clamp_min(0.0)             # E[x^2] - E[x]^2
    y = (xd - m1) / torch.sqrt(var + EPS) * weight.double() + bias.double()
    return y.to(torch.float16)


# --------------------------------------------------------------------------
# Metamorphic invariants — never re-runs Model.
# --------------------------------------------------------------------------
def invariants(out, ins, shape):
    T, H = _resolve(shape)
    x, weight, bias = ins[0], ins[1], ins[2]
    checks = []

    # 1) Structure: exact shape, fp16 storage, all-finite.
    ok_shape = tuple(out.shape) == (T, H)
    ok_dtype = out.dtype == torch.float16
    ok_finite = bool(torch.isfinite(out.float()).all())
    checks.append((
        'structure_fp16_finite',
        bool(ok_shape and ok_dtype and ok_finite),
        f"shape={tuple(out.shape)} expected={(T, H)} dtype={out.dtype} "
        f"expected=torch.float16 all_finite={ok_finite}",
    ))

    of = out.float()

    # 2) Centering + shift: a normalized column has zero mean, so after the
    #    affine chain each output column mean must equal bias[h].
    col_mean = of.mean(dim=0)
    mean_err = (col_mean - bias.float()).abs()
    max_mean_err = float(mean_err.max())
    checks.append((
        'column_mean_equals_bias',
        max_mean_err <= 2e-2,
        f"max|mean(out[:,h]) - bias[h]| = {max_mean_err:.3e} (tol 2e-2) "
        f"at h={int(mean_err.argmax())}",
    ))

    # 3) Gain law: Var(out[:,h]) = weight[h]^2 * v_h / (v_h + eps), where v_h is
    #    the biased input variance measured directly from x (not from Model).
    xf = x.float()
    v = ((xf - xf.mean(dim=0)) ** 2).mean(dim=0)
    expect_var = weight.float() ** 2 * v / (v + EPS)
    col_var = ((of - col_mean) ** 2).mean(dim=0)
    var_err = (col_var - expect_var).abs()
    tol = 5e-2 + 5e-2 * expect_var.abs()
    worst = int((var_err - tol).argmax())
    checks.append((
        'column_variance_gain_law',
        bool((var_err <= tol).all()),
        f"max slack at h={worst}: |{float(col_var[worst]):.5f} - "
        f"{float(expect_var[worst]):.5f}| = {float(var_err[worst]):.3e} "
        f"vs tol {float(tol[worst]):.3e} (atol 5e-2, rtol 5e-2); "
        f"min input col var = {float(v.min()):.4f}",
    ))

    return checks


# --------------------------------------------------------------------------
# The strong library baseline — the honest frontier.
# --------------------------------------------------------------------------
def baseline(x, weight, bias, init_args=None):
    if init_args is None:
        eps = EPS
    elif isinstance(init_args, (list, tuple)) and len(init_args) == 1 and float(init_args[0]) == EPS:
        eps = EPS
    else:
        raise ValueError(f"baseline only accepts init_args None or [{EPS}], got {init_args!r}")

    y = F.batch_norm(
        x,
        running_mean=None,
        running_var=None,
        weight=weight,
        bias=bias,
        training=True,
        momentum=0.0,
        eps=eps,
    )
    return y.to(x.dtype)
