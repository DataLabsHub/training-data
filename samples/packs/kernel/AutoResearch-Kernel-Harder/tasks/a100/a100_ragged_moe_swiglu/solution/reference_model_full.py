# reference_model.py
# Task: a100_ragged_moe_swiglu
# Ragged expert-major SwiGLU activation (BF16) with per-expert valid-token counts.
#
# Real hotspot: MoE dispatch produces a padded, expert-major activation buffer
# x[E, Tmax, 2H] where only counts[e] rows of expert e are real tokens. A naive
# implementation streams (and computes on) the whole padded buffer; a good kernel
# reads/writes only the valid rows and fuses split+SiLU+mul+cast into one pass.
#
# Reference (Model) is deliberately NAIVE: it materializes fp32 gate/up planes for
# every padded row, then masks. It is the correctness Oracle and timing denominator.

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Physical shapes behind each public case_id. Public shapes are ALWAYS (case_id,).
# entries are (E, Tmax, H)
# ---------------------------------------------------------------------------
CASES = {
    0: (8, 128, 512),
    1: (16, 256, 1024),
    2: (32, 512, 2048),
    3: (64, 256, 4096),
    4: (8, 2048, 8192),
    5: (12, 384, 1536),
    6: (24, 768, 3072),
    7: (48, 1024, 1024),
    8: (64, 512, 6144),
    9: (32, 2048, 4096),
}

WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-2
TASK_ATOL = 2e-2
GRAD_INPUTS = []          # inference-only activation fusion; no autograd checked
OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    'input_dtype': 'bfloat16',
    'output_dtype': 'bfloat16',
    'accumulation_dtype': 'float32',
    'allow_tf32': True,
    'note': (
        'Inputs: contiguous BF16 x[E, Tmax, 2H] (gate = x[..., :H], up = x[..., H:]) '
        'plus int32 counts[E] in [0, Tmax]; counts is an integer control tensor and is '
        'exempt from the BF16 input_dtype field. Math: promote gate/up to FP32, compute '
        'gate * sigmoid(gate) * up in FP32, round ONCE to BF16 on store. Output is BF16 '
        '[E, Tmax, H]; rows t >= counts[e] must be EXACT +0.0 regardless of the junk held '
        'in those padded input rows. TF32 is irrelevant (no matmul) but permitted.'
    ),
}


# ---------------------------------------------------------------------------
# Naive reference model (correctness anchor + timing denominator)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """Plain-PyTorch ragged SwiGLU: compute everything, then mask the padding."""

    def __init__(self):
        super().__init__()

    def forward(self, x, counts):
        # x: [E, Tmax, 2H] bf16, counts: [E] int32
        E, Tmax, two_h = x.shape
        H = two_h // 2

        gate = x[..., :H].float()
        up = x[..., H:].float()

        act = gate * torch.sigmoid(gate) * up          # fp32 SwiGLU on ALL rows

        pos = torch.arange(Tmax, device=x.device).unsqueeze(0)          # [1, Tmax]
        valid = pos < counts.to(torch.long).unsqueeze(1)                # [E, Tmax]

        zero = torch.zeros((), device=x.device, dtype=act.dtype)
        act = torch.where(valid.unsqueeze(-1), act, zero)

        return act.to(torch.bfloat16)


def get_init_inputs():
    return []


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _make_counts(E, Tmax, gen):
    """Ragged occupancy: empty / full / singleton / short + a skewed random tail."""
    counts = torch.empty(E, dtype=torch.int64)

    fixed = [0, Tmax, 1, max(1, Tmax // 8)]
    for i in range(min(E, len(fixed))):
        counts[i] = fixed[i]

    if E > len(fixed):
        n_rest = E - len(fixed)
        u = torch.rand(n_rest, generator=gen)
        r = torch.rand(n_rest, generator=gen)
        vals = torch.empty(n_rest, dtype=torch.int64)
        for j in range(n_rest):
            if r[j] < 0.20:                                   # mostly-empty expert
                vals[j] = int(u[j].item() * (Tmax // 16 + 1))
            elif r[j] < 0.40:                                 # nearly-full expert
                vals[j] = Tmax - int(u[j].item() * (Tmax // 16 + 1))
            elif r[j] < 0.50:                                 # exactly full
                vals[j] = Tmax
            else:                                             # uniform
                vals[j] = int(u[j].item() * (Tmax + 1))
        counts[len(fixed):] = vals

    return counts.clamp_(0, Tmax)


def make_inputs(shape, device, seed=0):
    case_id = int(shape[0])
    E, Tmax, H = CASES[case_id]

    dev = torch.device(device)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed) * 9176 + case_id * 131 + 17)

    counts = _make_counts(E, Tmax, gen)

    x = torch.randn(E, Tmax, 2 * H, generator=gen, dtype=torch.float32)
    x.mul_(1.25).clamp_(-8.0, 8.0)

    # Padded (invalid) rows carry loud non-zero junk: any kernel that leaks padding
    # into the output, or that forgets to zero-fill the tail, is caught immediately.
    for e in range(E):
        c = int(counts[e].item())
        if c < Tmax:
            junk = torch.randn(Tmax - c, 2 * H, generator=gen, dtype=torch.float32)
            x[e, c:, :] = junk.mul_(2.0).add_(5.0)

    # Planted sentinels inside VALID rows (checked by invariants(), independent of Model):
    #   row 0 -> gate == 0  => silu(0) == 0 => output row must be exactly 0
    #   row 1 -> up   == 0  => output row must be exactly 0
    for e in range(E):
        c = int(counts[e].item())
        if c >= 1:
            x[e, 0, :H] = 0.0
        if c >= 2:
            x[e, 1, H:] = 0.0

    x_bf16 = x.to(torch.bfloat16).to(dev).contiguous()
    counts_i32 = counts.to(torch.int32).to(dev).contiguous()
    return [x_bf16, counts_i32]


def sample_shapes(rng, n):
    """Unseen case_ids near the declared domain (hidden-shape tripwire)."""
    pool = [5, 6, 7, 8, 9]
    out = []
    for i in range(int(n)):
        if hasattr(rng, 'randrange'):
            k = rng.randrange(len(pool))
        elif hasattr(rng, 'integers'):
            k = int(rng.integers(0, len(pool)))
        else:
            k = i % len(pool)
        out.append((pool[k],))
    return out


# ---------------------------------------------------------------------------
# Metamorphic invariants (independent of Model's code path)
# ---------------------------------------------------------------------------
def invariants(out, ins, shape):
    x, counts = ins[0], ins[1]
    case_id = int(shape[0])
    E, Tmax, H = CASES[case_id]

    if isinstance(out, (tuple, list)):
        y = out[0]
    else:
        y = out

    checks = []

    # ---- 1. shape / dtype / finiteness -------------------------------------
    ok_shape = tuple(y.shape) == (E, Tmax, H)
    ok_dtype = (y.dtype == torch.bfloat16)
    ok_dev = (y.device.type == x.device.type)
    finite = bool(torch.isfinite(y.float()).all().item())
    checks.append((
        'shape_dtype_finite',
        bool(ok_shape and ok_dtype and ok_dev and finite),
        f'shape={tuple(y.shape)} exp={(E, Tmax, H)} dtype={y.dtype} '
        f'dev={y.device.type} finite={finite}',
    ))

    pos = torch.arange(Tmax, device=y.device).unsqueeze(0)
    valid = pos < counts.to(torch.long).unsqueeze(1)              # [E, Tmax]
    rowmax = y.abs().amax(dim=-1).float()                         # [E, Tmax]

    # ---- 2. padded rows are EXACT zeros -------------------------------------
    # Padded input rows hold junk >= ~1.0, so a leak here is loudly non-zero.
    inv_rows = ~valid
    n_inv = int(inv_rows.sum().item())
    if n_inv == 0:
        pad_bad = 0.0
    else:
        pad_bad = float(torch.where(inv_rows, rowmax, torch.zeros_like(rowmax)).max().item())
    checks.append((
        'padding_rows_exact_zero',
        pad_bad == 0.0,
        f'{n_inv} padded rows, max|out| over padding = {pad_bad:.6g} (must be exactly 0)',
    ))

    # ---- 3. zero gate / zero up sentinels ----------------------------------
    # silu(0) = 0 and (anything)*0 = 0, so these valid rows must be exact zeros.
    gate_zero_row = (x[..., :H] == 0).all(dim=-1)                 # [E, Tmax]
    up_zero_row = (x[..., H:] == 0).all(dim=-1)
    sentinel = (gate_zero_row | up_zero_row) & valid
    n_sent = int(sentinel.sum().item())
    if n_sent == 0:
        sent_bad = 0.0
    else:
        sent_bad = float(torch.where(sentinel, rowmax, torch.zeros_like(rowmax)).max().item())
    checks.append((
        'zero_gate_or_up_sentinels',
        (n_sent > 0) and (sent_bad == 0.0),
        f'{n_sent} sentinel rows (gate==0 or up==0), max|out| = {sent_bad:.6g} (must be exactly 0)',
    ))

    # ---- 4. sign consistency + |out| <= |gate*up| ---------------------------
    # out = sigmoid(gate) * (gate*up) with 0 < sigmoid < 1, so out shares the sign
    # of gate*up and is strictly smaller in magnitude. Sub-sampled over rows.
    step = max(1, Tmax // 32)
    sl = slice(0, Tmax, step)
    g = x[:, sl, :H].float()
    u = x[:, sl, H:].float()
    o = y[:, sl, :].float()
    v = valid[:, sl].unsqueeze(-1)

    prod = g * u
    tol = 1e-2 + 0.02 * prod.abs()
    bound_ok = ((o.abs() <= prod.abs() + tol) | ~v)
    sign_ok = ((o * prod >= -1e-4 * (prod.abs() + 1.0)) | ~v)

    n_bound_bad = int((~bound_ok).sum().item())
    n_sign_bad = int((~sign_ok).sum().item())
    checks.append((
        'sign_and_magnitude_vs_gate_times_up',
        (n_bound_bad == 0) and (n_sign_bad == 0),
        f'sampled {o.numel()} elems: {n_bound_bad} violate |out|<=|gate*up|, '
        f'{n_sign_bad} violate sign(out)==sign(gate*up)',
    ))

    return checks


# ---------------------------------------------------------------------------
# Independent oracle: per-expert valid slices only, fp64 math, different formula
# ---------------------------------------------------------------------------
def independent_oracle(ins, shape):
    x, counts = ins[0], ins[1]
    case_id = int(shape[0])
    E, Tmax, H = CASES[case_id]

    out = torch.zeros((E, Tmax, H), dtype=torch.bfloat16, device=x.device)
    counts_cpu = counts.to('cpu').to(torch.int64)

    for e in range(E):
        c = int(counts_cpu[e].item())
        if c <= 0:
            continue
        blk = x[e, :c, :].to(torch.float64)                # valid rows only
        g = blk[:, :H]
        u = blk[:, H:]
        silu = g / (1.0 + torch.exp(-g))                   # SiLU via exp, not sigmoid()
        out[e, :c, :] = (silu * u).to(torch.bfloat16)

    return out


# ---------------------------------------------------------------------------
# Strong library baseline: cached torch.compile of the masked composite
# ---------------------------------------------------------------------------
def _masked_swiglu(x, counts):
    E, Tmax, two_h = x.shape
    H = two_h // 2
    gate = x[..., :H].float()
    up = x[..., H:].float()
    act = gate * torch.sigmoid(gate) * up
    pos = torch.arange(Tmax, device=x.device).unsqueeze(0)
    valid = (pos < counts.to(torch.long).unsqueeze(1)).unsqueeze(-1)
    return torch.where(valid, act, torch.zeros((), device=x.device, dtype=act.dtype)).to(torch.bfloat16)


_COMPILED_CACHE = {}


def baseline(*args, init_args=None):
    x, counts = args[0], args[1]
    key = 'ragged_swiglu'
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        try:
            fn = torch.compile(_masked_swiglu, dynamic=False)
        except Exception:
            fn = _masked_swiglu
        _COMPILED_CACHE[key] = fn
    return fn(x, counts)
