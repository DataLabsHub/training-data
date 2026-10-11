# reference_model.py
# Task: a100_adaptive_layernorm_modulation
# Adaptive LayerNorm (adaLN) modulation for diffusion-transformer BF16 activations.
#
#   y = bf16( bf16( bf16( (x - mean) / sqrt(var + eps) ) * (1 + scale) ) + shift )
#
# The three BF16 round points are part of the output contract: statistics are
# accumulated in FP32 (biased variance, eps = 1e-5), the normalized value is
# rounded to BF16, the modulated product is rounded to BF16, and the shifted
# result is rounded to BF16 one final time.

import math

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Physical shapes.  Public shapes are one-tuples (case_id,); CASES maps a
# case_id to the physical (B, L, D) activation extent.
# ----------------------------------------------------------------------------
CASES = {
    0: (1, 128, 3072),
    1: (1, 512, 3072),
    2: (1, 1024, 3072),
    3: (1, 4096, 3072),
    4: (1, 16384, 3072),
    5: (2, 256, 3072),
    6: (2, 1024, 3072),
    7: (1, 4216, 3072),
    8: (1, 8192, 3072),
    9: (2, 4096, 3072),
}

EPS = 1e-5

TASK_MODE = 'tensor'
TASK_RTOL = 1.6e-2
TASK_ATOL = 1.0e-2
GRAD_INPUTS = []
OUTPUT_INDEX = 0

WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]

BASELINE_CONTRACT = {
    'input_dtype': 'bfloat16',
    'output_dtype': 'bfloat16',
    'accumulation_dtype': 'float32',
    'allow_tf32': True,
    'note': (
        'Adaptive LayerNorm modulation over the last (D) axis of a [B, L, D] '
        'BF16 activation with [B, 1, D] BF16 scale/shift. Mean and biased '
        'variance are accumulated in FP32 with eps=1e-5. Staged BF16 rounding '
        'contract: (1) normalized = bf16((x_f32 - mean) * rsqrt(var + eps)); '
        '(2) modulated = bf16(normalized_f32 * (1 + scale_f32)); '
        '(3) out = bf16(modulated_f32 + shift_f32). All three round points must '
        'be reproduced; fusing them into one FP32 expression without the '
        'intermediate BF16 truncations violates the contract. TF32 is '
        'irrelevant here (no matmul) but left enabled to match the reference '
        'runtime configuration.'
    ),
}


def _case_dims(shape):
    """Public one-tuple (case_id,) -> physical (B, L, D)."""
    if isinstance(shape, (tuple, list)) and len(shape) == 1:
        return CASES[int(shape[0])]
    if isinstance(shape, int):
        return CASES[int(shape)]
    raise ValueError(f"expected a (case_id,) one-tuple, got {shape!r}")


class Model(nn.Module):
    """Naive, obviously-correct plain-PyTorch reference (the correctness Oracle).

    Deliberately unfused and unoptimized: every stage is a separate materialized
    tensor so the staged rounding contract is readable line by line.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, scale, shift):
        # --- stage 0: FP32 statistics over the last axis (biased variance) ---
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = xf.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (xf - mean) / torch.sqrt(var + EPS)

        # --- round point 1 ---
        normalized = normalized.to(torch.bfloat16)

        # --- stage 1: per-channel scale, broadcast [B, 1, D] over L ---
        modulated = normalized.float() * (1.0 + scale.float())

        # --- round point 2 ---
        modulated = modulated.to(torch.bfloat16)

        # --- stage 2: per-channel shift ---
        out = modulated.float() + shift.float()

        # --- round point 3 ---
        return out.to(torch.bfloat16)


def get_init_inputs():
    return []


def make_inputs(shape, device, seed=0):
    """Build [x, scale, shift] directly on `device`. `seed` changes the data."""
    B, L, D = _case_dims(shape)
    device = torch.device(device)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) * 9973 + 12345)

    x = torch.randn(B, L, D, generator=g, device=device, dtype=torch.float32)
    # Modest per-sample gain so rows are non-degenerate but not extreme.
    gain = 0.75 + 0.5 * torch.rand(B, 1, 1, generator=g, device=device,
                                   dtype=torch.float32)
    x = (x * gain).to(torch.bfloat16).contiguous()

    # 1 + scale stays comfortably away from zero -> modulation is invertible.
    scale = (0.10 * torch.randn(B, 1, D, generator=g, device=device,
                                dtype=torch.float32)).to(torch.bfloat16).contiguous()
    shift = (0.10 * torch.randn(B, 1, D, generator=g, device=device,
                                dtype=torch.float32)).to(torch.bfloat16).contiguous()
    return [x, scale, shift]


def sample_shapes(rng, n):
    """Unseen public shapes near the domain (hidden-shape tripwire)."""
    unseen = [(5,), (6,), (7,), (8,), (9,)]
    out = []
    for i in range(int(n)):
        out.append(unseen[rng.randrange(len(unseen))] if hasattr(rng, 'randrange')
                   else unseen[i % len(unseen)])
    return out


def reduction_length(shape):
    """Extent of the normalization reduction (the D axis)."""
    _, _, D = _case_dims(shape)
    return D


def invariants(out, ins, shape):
    """Metamorphic checks on `out` alone -- Model is never re-run.

    The modulation is an invertible affine map per channel, so demodulating the
    output must recover a row-standardized signal.  That property is derived
    from the *definition* of adaLN, not from the reference implementation.
    """
    B, L, D = _case_dims(shape)
    x, scale, shift = ins[0], ins[1], ins[2]
    checks = []

    # (1) shape / dtype / finiteness
    ok_shape = tuple(out.shape) == (B, L, D)
    ok_dtype = (out.dtype == torch.bfloat16)
    ok_finite = bool(torch.isfinite(out.float()).all().item())
    checks.append((
        'shape_dtype_finite',
        bool(ok_shape and ok_dtype and ok_finite),
        f"shape={tuple(out.shape)} exp={(B, L, D)} dtype={out.dtype} finite={ok_finite}",
    ))

    # Demodulate on a bounded row subsample (keeps the check cheap on L=16384).
    rows = min(L, 256)
    o = out[:, :rows, :].float()
    s = scale.float()
    h = shift.float()
    demod = (o - h) / (1.0 + s)

    # (2) inverse modulation restores zero mean / unit variance per row
    m = demod.mean(dim=-1)
    v = demod.var(dim=-1, unbiased=False)
    max_abs_mean = float(m.abs().max().item())
    max_var_err = float((v - 1.0).abs().max().item())
    checks.append((
        'demodulated_rows_standardized',
        bool(max_abs_mean < 5e-2 and max_var_err < 1e-1),
        f"max|row_mean|={max_abs_mean:.4e} (<5e-2), max|row_var-1|={max_var_err:.4e} (<1e-1)",
    ))

    # (3) a zero-mean, unit-variance vector of length D is bounded by sqrt(D-1);
    #     and its per-row extremum must sit where the input extremum sits
    #     (the affine map x -> (x-mean)/std is strictly increasing).
    bound = math.sqrt(max(D - 1, 1)) * 1.05 + 0.25
    max_abs = float(demod.abs().max().item())
    ok_bound = max_abs <= bound

    xs = x[:, :rows, :].float()
    idx = xs.argmax(dim=-1, keepdim=True)
    demod_at_xmax = torch.gather(demod, -1, idx).squeeze(-1)
    demod_max = demod.max(dim=-1).values
    max_gap = float((demod_max - demod_at_xmax).abs().max().item())
    ok_mono = max_gap < 5e-2
    checks.append((
        'bounded_and_order_preserving',
        bool(ok_bound and ok_mono),
        f"max|demod|={max_abs:.4e} (<= {bound:.4e}); "
        f"argmax(x) vs max(demod) gap={max_gap:.4e} (<5e-2)",
    ))

    return checks


def independent_oracle(ins, shape):
    """Trusted recompute: FP64 statistics, then the identical BF16 round points.

    Shares no code path with Model -- the statistics are accumulated in double
    precision via explicit sums (not var()/mean()) and chunked over L so the
    largest case stays within memory.
    """
    B, L, D = _case_dims(shape)
    x, scale, shift = ins[0], ins[1], ins[2]

    scale64 = scale.double()
    shift64 = shift.double()
    out = torch.empty((B, L, D), device=x.device, dtype=torch.bfloat16)

    chunk = 2048
    for start in range(0, L, chunk):
        stop = min(start + chunk, L)
        xc = x[:, start:stop, :].double()

        n = float(D)
        s1 = xc.sum(dim=-1, keepdim=True)
        mean = s1 / n
        centered = xc - mean
        var = (centered * centered).sum(dim=-1, keepdim=True) / n

        normalized = centered / torch.sqrt(var + float(EPS))
        # round point 1
        normalized = normalized.to(torch.bfloat16)
        # round point 2
        modulated = (normalized.double() * (1.0 + scale64)).to(torch.bfloat16)
        # round point 3
        out[:, start:stop, :] = (modulated.double() + shift64).to(torch.bfloat16)

    return out


# ----------------------------------------------------------------------------
# Strong library baseline: torch.compile'd adaptive-LayerNorm composite.
# ----------------------------------------------------------------------------
_COMPILED = {}


def _adaln_composite(x, scale, shift):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, unbiased=False, keepdim=True)
    normalized = ((xf - mean) * torch.rsqrt(var + EPS)).to(torch.bfloat16)
    modulated = (normalized.float() * (1.0 + scale.float())).to(torch.bfloat16)
    return (modulated.float() + shift.float()).to(torch.bfloat16)


def baseline(*args, init_args=None):
    x, scale, shift = args[0], args[1], args[2]
    key = (tuple(x.shape), str(x.device), str(x.dtype))
    fn = _COMPILED.get(key)
    if fn is None:
        try:
            fn = torch.compile(_adaln_composite, dynamic=False)
        except Exception:
            fn = _adaln_composite
        _COMPILED[key] = fn
    return fn(x, scale, shift)
