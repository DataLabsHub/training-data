"""Task: a100_bf16_int8_mixed_gemm

Mixed BF16-by-INT8 projection on A100 (Ampere, sm_80).

Real hotspot: a BF16 activation matrix is projected by packed 8-bit weights
(signed int8 or unsigned uint8, row-major [N, K]) with an FP32 per-output bias.
The naive reference below materializes a *full* FP32 copy of the weight tensor
before the matmul, which is exactly what a good kernel must avoid: the 8-bit ->
Tensor-Core-operand conversion belongs inside the mainloop (prologue fusion),
accumulation stays FP32, and the bias is an epilogue fusion.

`Model` is the plain-PyTorch correctness Oracle and the timing denominator.
`baseline` is the honest frontier: a cached `torch.compile` composite that
converts the weights inside the timed region.
"""

import math

import torch
import torch.nn as nn


# case_id -> (M, N, K, weight_kind, out_kind)
CASES = {
    0: (128, 1024, 1024, 'int8', 'bf16'),
    1: (512, 2048, 1024, 'uint8', 'bf16'),
    2: (1024, 4096, 2048, 'int8', 'fp32'),
    3: (3456, 4096, 8192, 'int8', 'bf16'),
    4: (64, 8192, 4096, 'uint8', 'fp32'),
    5: (256, 3072, 2048, 'int8', 'bf16'),
    6: (768, 1536, 4096, 'uint8', 'bf16'),
    7: (2048, 2048, 1024, 'int8', 'fp32'),
    8: (4096, 1024, 2048, 'uint8', 'bf16'),
    9: (8192, 512, 4096, 'int8', 'fp32'),
}

_WEIGHT_DTYPE = {'int8': torch.int8, 'uint8': torch.uint8}
_OUT_DTYPE = {'bf16': torch.bfloat16, 'fp32': torch.float32}
_WEIGHT_RANGE = {'int8': (-128, 127), 'uint8': (0, 255)}


WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-2
TASK_ATOL = 5e-2
GRAD_INPUTS = []
OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    'input_dtype': 'mixed',
    'output_dtype': 'mixed',
    'accumulation_dtype': 'fp32',
    'allow_tf32': False,
    'note': (
        'A is bf16 [M, K]; W is row-major packed int8 or uint8 [N, K] read as '
        'plain integers (no zero point, no dequant scale); bias is fp32 [N]. '
        'Math is out = A.float() @ W.float().T + bias with FP32 accumulation, '
        'then a single cast to the per-case output dtype (bf16 or fp32), which '
        'is why output_dtype is "mixed". The 4th forward argument is a '
        'zero-element tensor whose dtype declares that output dtype. '
        'baseline is a cached torch.compile composite: it converts W to bf16 '
        'inside the timed call (never hoisted or pre-converted) and relies on '
        'bf16 Tensor Cores with FP32 accumulate, so on fp32-output cases it '
        'carries one extra bf16 rounding of the accumulator (~2^-9 relative), '
        'which the task tolerances absorb. A winning kernel must convert the '
        '8-bit operands in-register inside the mainloop and fuse the bias '
        'epilogue, never materializing a converted weight tensor.'
    ),
}


class Model(nn.Module):
    """Naive, obviously-correct reference: widen everything, matmul, add bias."""

    def forward(self, a, w, bias, out_dtype_token):
        a32 = a.to(torch.float32)
        w32 = w.to(torch.float32)          # full converted weight tensor (naive)
        acc = torch.matmul(a32, w32.t())   # FP32 accumulation
        acc = acc + bias                   # per-output FP32 bias
        return acc.to(out_dtype_token.dtype)


def get_init_inputs():
    return []


def make_inputs(shape, device, seed=0):
    case_id = int(shape[0])
    M, N, K, weight_kind, out_kind = CASES[case_id]
    w_dtype = _WEIGHT_DTYPE[weight_kind]
    out_dtype = _OUT_DTYPE[out_kind]
    lo, hi = _WEIGHT_RANGE[weight_kind]

    device = torch.device(device)
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed) * 100003 + case_id * 7919 + 13)

    scale = 0.05 / math.sqrt(K)
    a = torch.randn(M, K, generator=g, dtype=torch.float32) * scale
    w = torch.randint(lo, hi + 1, (N, K), generator=g, dtype=torch.int16).to(w_dtype)
    bias = torch.randn(N, generator=g, dtype=torch.float32) * 0.5

    # Metamorphic sentinels (see invariants).
    a[0].zero_()                      # zero activation row -> output must be bias
    if M > 1:
        a[1].fill_(1.0)               # ones row -> output must be column sum + bias
    w[0].fill_(hi)                    # extremal weight row (max)
    if N > 1:
        w[1].fill_(lo)                # extremal weight row (min)

    a = a.to(device=device, dtype=torch.bfloat16)
    w = w.to(device=device)
    bias = bias.to(device=device)
    out_dtype_token = torch.empty(0, dtype=out_dtype, device=device)
    return [a, w, bias, out_dtype_token]


def sample_shapes(rng, n):
    """Unseen cases (5..9) near the declared domain, as (case_id,) one-tuples."""
    hidden = [(5,), (6,), (7,), (8,), (9,)]
    order = list(range(len(hidden)))
    try:
        rng.shuffle(order)
    except Exception:
        pass
    out = []
    for i in range(int(n)):
        out.append(hidden[order[i % len(order)]])
    return out


def reduction_length(shape):
    """Contraction extent K for the given case."""
    return CASES[int(shape[0])][2]


def invariants(out, ins, shape):
    case_id = int(shape[0])
    M, N, K, weight_kind, out_kind = CASES[case_id]
    lo, hi = _WEIGHT_RANGE[weight_kind]
    a, w, bias, token = ins[0], ins[1], ins[2], ins[3]

    y = out[OUTPUT_INDEX] if isinstance(out, (tuple, list)) else out
    checks = []

    # 1) Structural: exact shape, declared dtype, all finite.
    ok_struct = (
        tuple(y.shape) == (M, N)
        and y.dtype == token.dtype
        and bool(torch.isfinite(y.to(torch.float32)).all())
    )
    checks.append((
        'shape_dtype_finite',
        bool(ok_struct),
        f'got shape={tuple(y.shape)} dtype={y.dtype}; want ({M},{N}) {token.dtype}; '
        f'finite={bool(torch.isfinite(y.to(torch.float32)).all())}',
    ))

    yd = y.to(torch.float64)
    bd = bias.to(torch.float64)

    # 2) Zero activation row: the projection vanishes, only the bias survives.
    row0 = yd[0]
    err0 = (row0 - bd).abs()
    tol0 = 1e-2 + 1e-2 * bd.abs()
    ok0 = bool((err0 <= tol0).all())
    checks.append((
        'zero_row_equals_bias',
        ok0,
        f'max|out[0]-bias|={float(err0.max()):.3e}, max allowed={float(tol0.max()):.3e}',
    ))

    # 3) Extremal 8-bit sentinels: with a ones activation row, the output must be
    #    K * (extremal weight value) + bias -- a closed form that pins down both
    #    the integer interpretation (no zero point) and the accumulator width.
    if M > 1:
        want_hi = float(K) * float(hi) + float(bd[0])
        got_hi = float(yd[1, 0])
        errs = [(abs(got_hi - want_hi), abs(want_hi), 'max')]
        detail = f'max-weight col: got={got_hi:.6e} want={want_hi:.6e}'
        if N > 1:
            want_lo = float(K) * float(lo) + float(bd[1])
            got_lo = float(yd[1, 1])
            errs.append((abs(got_lo - want_lo), abs(want_lo), 'min'))
            detail += f'; min-weight col: got={got_lo:.6e} want={want_lo:.6e}'
        ok_ext = all(e <= 1e-2 + 1e-2 * mag for e, mag, _ in errs)
        checks.append((f'int8_extrema_closed_form[{weight_kind}]', bool(ok_ext), detail))
    else:
        checks.append(('int8_extrema_closed_form', True, 'skipped: M == 1'))

    # 4) Spot dot products recomputed independently in FP64 (element-wise
    #    multiply + sum, no matmul path shared with Model).
    rows = sorted({min(2, M - 1), M // 2, M - 1})
    cols = sorted({min(3, N - 1), N // 2, N - 1})
    worst, worst_at = 0.0, None
    for i in rows:
        ai = a[i].to(torch.float64)
        for j in cols:
            want = float((ai * w[j].to(torch.float64)).sum() + bd[j])
            got = float(yd[i, j])
            rel = abs(got - want) / max(1.0, abs(want))
            if rel > worst:
                worst, worst_at = rel, (i, j, got, want)
    ok_spot = worst <= 3e-2
    checks.append((
        'fp64_spot_dot_products',
        bool(ok_spot),
        f'worst rel err={worst:.3e} at {worst_at}',
    ))

    return checks


def independent_oracle(ins, shape):
    """Trusted FP64 recompute of the full output, cast to the declared dtype."""
    case_id = int(shape[0])
    M, N, K, weight_kind, out_kind = CASES[case_id]
    a, w, bias, token = ins[0], ins[1], ins[2], ins[3]

    ref = torch.empty(M, N, dtype=torch.float64, device=a.device)
    a64 = a.to(torch.float64)
    bias64 = bias.to(torch.float64)
    # Chunk over output columns so the FP64 weight copy stays bounded.
    step = max(1, min(N, (1 << 24) // max(1, K)))
    for j0 in range(0, N, step):
        j1 = min(N, j0 + step)
        w64 = w[j0:j1].to(torch.float64)
        ref[:, j0:j1] = torch.matmul(a64, w64.t()) + bias64[j0:j1]
    return ref.to(token.dtype)


_MIXED_COMPILED = {}


def _mixed_gemm_composite(a, w, bias, out_dtype):
    wb = w.to(torch.bfloat16)                       # 8-bit -> TC operand, in-call
    acc = torch.matmul(a, wb.transpose(0, 1))       # bf16 Tensor Cores, FP32 accum
    acc = acc.to(torch.float32) + bias              # fused FP32 bias epilogue
    return acc.to(out_dtype)


def baseline(a, w, bias, out_dtype_token, init_args=None):
    out_dtype = out_dtype_token.dtype
    key = (a.dtype, w.dtype, out_dtype)
    fn = _MIXED_COMPILED.get(key)
    if fn is None:
        fn = torch.compile(_mixed_gemm_composite, dynamic=False)
        _MIXED_COMPILED[key] = fn
    return fn(a, w, bias, out_dtype)
