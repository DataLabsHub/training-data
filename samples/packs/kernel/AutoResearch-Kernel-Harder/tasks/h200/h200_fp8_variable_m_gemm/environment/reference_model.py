"""
Task: h200_fp8_variable_m_gemm
=============================================================================
Real hotspot
------------
H200 (SM90) blocked FP8 E4M3 projection GEMM with FP32 accumulation and
block scales, driven by schedules whose row count M spans single-token decode
(M=1) through large prefill batches (M=8192).  The activation carries one FP32
scale per (token, 128-K) block and the weight carries one FP32 scale per
(128-N, 128-K) block; the projection output is BF16.

    A_deq[m, k] = float(A[m, k]) * sa[m, k // 128]
    W_deq[n, k] = float(W[n, k]) * sw[n // 128, k // 128]
    Y[m, n]     = bf16( sum_k A_deq[m, k] * W_deq[n, k] )      (FP32 accumulate)

Reference (Oracle) is deliberately NAIVE: a plain Python loop over the K
blocks that materialises each dequantised block and accumulates a small
matmul into an FP32 buffer.  It is obviously correct, and it is honestly
unoptimised: it pays K/128 kernel launches, K/128 full (M x N) FP32
read-modify-writes, and it runs the multiply in FP32 (no tensor cores).

Baseline (honest frontier) is the strong library path a framework actually
ships today: a torch.compile'd, fully vectorised blockwise dequantisation
fused into a single BF16 tensor-core matmul with FP32 accumulation.  One
kernel-fused dequant, one big GEMM, no per-block accumulator traffic.

Optimization track: algorithmic_reformulation.  A candidate should keep the
scale application inside the mainloop (per-128-K rescaled FP32 accumulation)
and pick an M-bucket-aware schedule, without any shape-specific correctness
special cases.
"""

import torch
import torch.nn as nn

# (M, N, K) per case_id.  Public shapes are always one-tuples (case_id,).
CASES = {
    0: (1, 4096, 4096),
    1: (2, 4096, 4096),
    2: (8, 4096, 4096),
    3: (32, 4096, 4096),
    4: (128, 4096, 4096),
    5: (512, 4096, 4096),
    6: (1024, 8192, 4096),
    7: (2048, 4096, 8192),
    8: (4096, 4096, 4096),
    9: (8192, 4096, 4096),
    10: (64, 8192, 8192),
    11: (1536, 6144, 4096),
}

BLOCK = 128


class Model(nn.Module):
    """Naive, obviously-correct blockwise FP8 GEMM reference.

    One Python iteration per 128-wide K block: dequantise the activation
    block, dequantise the weight block, multiply in FP32, accumulate.
    """

    def forward(self, a, w, scale_a, scale_w):
        m, k = a.shape
        n = w.shape[0]
        nkb = k // BLOCK

        acc = torch.zeros(m, n, dtype=torch.float32, device=a.device)
        for kb in range(nkb):
            k0 = kb * BLOCK
            a_blk = a[:, k0:k0 + BLOCK].to(torch.float32) * scale_a[:, kb:kb + 1]
            w_col_scale = scale_w[:, kb].repeat_interleave(BLOCK).unsqueeze(1)
            w_blk = w[:, k0:k0 + BLOCK].to(torch.float32) * w_col_scale
            acc += a_blk @ w_blk.t()

        return acc.to(torch.bfloat16)


def make_inputs(shape, device, seed=0):
    case_id = int(shape[0])
    m, n, k = CASES[case_id]
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234 + 9176 * int(seed) + 31 * case_id)

    # Scale magnitude chosen so dequantised dot products land near unit scale.
    f = float(k) ** -0.25

    a_f = torch.randn(m, k, generator=gen, device=dev, dtype=torch.float32)
    # A few exactly-zero activation rows: real schedules contain padded rows
    # and they give a scale-free metamorphic handle on the output.
    if m >= 8:
        a_f[1].zero_()
        a_f[m - 1].zero_()
    elif m >= 4:
        a_f[m - 1].zero_()
    a = a_f.clamp_(-400.0, 400.0).to(torch.float8_e4m3fn)

    w_f = torch.randn(n, k, generator=gen, device=dev, dtype=torch.float32)
    w = w_f.clamp_(-400.0, 400.0).to(torch.float8_e4m3fn)

    scale_a = (0.5 + torch.rand(m, k // BLOCK, generator=gen, device=dev,
                                dtype=torch.float32)) * f
    scale_w = (0.5 + torch.rand(n // BLOCK, k // BLOCK, generator=gen, device=dev,
                                dtype=torch.float32)) * f

    return [a, w, scale_a, scale_w]


def get_init_inputs():
    return []


WORKLOAD = [(0,), (1,), (2,), (3,), (4,), (5,)]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-2
TASK_ATOL = 2e-2
GRAD_INPUTS = []
OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    'input_dtype': 'float8_e4m3fn',
    'output_dtype': 'bfloat16',
    'accumulation_dtype': 'float32',
    'allow_tf32': False,
    'note': (
        'A (M,K) and W (N,K) are FP8 E4M3 with positive FP32 block scales: '
        'scale_a is (M, K/128) per-token per-128-K, scale_w is (N/128, K/128) '
        'per-128x128 tile. Y = bf16(A_deq @ W_deq^T) with FP32 accumulation; '
        'scales must be applied inside the K reduction. The reference runs a '
        'per-K-block FP32 matmul loop; the baseline is a torch.compile fused '
        'blockwise dequantisation feeding one BF16 tensor-core matmul (FP32 '
        'accumulate). FP32 matmul must not silently fall back to TF32. One '
        'candidate must serve every M bucket (decode M=1 through prefill '
        'M=8192) with no shape-specific correctness path; per-shape autotune '
        'caching is allowed and compile time is scored separately.'
    ),
}


def sample_shapes(rng, n):
    """Unseen case_ids (6..11) - larger N/K and off-bucket M values."""
    pool = [6, 7, 8, 9, 10, 11]
    out = []
    for i in range(int(n)):
        if hasattr(rng, 'integers'):
            idx = int(rng.integers(0, len(pool)))
        elif hasattr(rng, 'randrange'):
            idx = int(rng.randrange(len(pool)))
        else:
            idx = i % len(pool)
        out.append((pool[idx],))
    return out


def invariants(out, ins, shape):
    """Metamorphic checks, computed without rerunning Model."""
    if isinstance(out, (tuple, list)):
        out = out[OUTPUT_INDEX]
    a, w, scale_a, scale_w = ins
    m, n, k = CASES[int(shape[0])]
    nkb = k // BLOCK
    dev = out.device
    res = []

    # 1. structural: shape, dtype, finiteness.
    shape_ok = tuple(out.shape) == (m, n)
    dtype_ok = out.dtype == torch.bfloat16
    finite = bool(torch.isfinite(out.to(torch.float32)).all())
    res.append((
        'shape_dtype_finite',
        bool(shape_ok and dtype_ok and finite),
        f'shape={tuple(out.shape)} expected={(m, n)} dtype={out.dtype} finite={finite}',
    ))

    # 2. exact zero rows: an all-zero activation row must give an all-zero
    #    output row for any scales (homogeneity of the reduction).
    a32 = a.to(torch.float32)
    zero_rows = (a32.abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if zero_rows.numel() == 0:
        res.append(('zero_row_annihilation', True,
                    'no all-zero activation rows in this bucket'))
    else:
        max_abs = float(out.index_select(0, zero_rows).to(torch.float32).abs().max())
        res.append(('zero_row_annihilation', max_abs == 0.0,
                    f'{zero_rows.numel()} zero rows, max|out|={max_abs:.3e}'))

    # 3. selected dot products, recomputed elementwise in float64 (no matmul,
    #    no block loop) - fully independent of the reference code path.
    rows = [int(r) for r in range(m) if int(r) not in set(zero_rows.tolist())]
    if not rows:
        rows = [0]
    pairs = []
    for r in (rows[0], rows[len(rows) // 2], rows[-1]):
        for c in (0, n // 2 + 3, n - 1):
            pairs.append((r, c))
            if len(pairs) >= 4:
                break
        if len(pairs) >= 4:
            break
    worst = 0.0
    detail = ''
    for (i, j) in pairs:
        arow = a[i].to(torch.float64).view(nkb, BLOCK) * scale_a[i].to(torch.float64).unsqueeze(-1)
        wrow = w[j].to(torch.float64).view(nkb, BLOCK) * scale_w[j // BLOCK].to(torch.float64).unsqueeze(-1)
        ref = float((arow * wrow).sum())
        got = float(out[i, j].to(torch.float64))
        err = abs(got - ref) / (5e-2 * abs(ref) + 5e-2)
        if err > worst:
            worst = err
            detail = f'(i={i}, j={j}) got={got:.6e} ref={ref:.6e}'
    res.append(('selected_dots_fp64', worst <= 1.0,
                detail or 'no pairs checked'))

    # 4. Cauchy-Schwarz envelope: |Y[m,n]| <= ||A_deq[m,:]|| * ||W_deq[n,:]||.
    sa2 = scale_a.to(torch.float32).pow(2)
    a_norm = (a32.view(m, nkb, BLOCK).pow(2).sum(-1) * sa2).sum(-1).sqrt().max()
    w_norm_max = torch.zeros((), device=dev, dtype=torch.float32)
    step = max(BLOCK, 1024)
    for n0 in range(0, n, step):
        n1 = min(n0 + step, n)
        wt = w[n0:n1].to(torch.float32).view(n1 - n0, nkb, BLOCK).pow(2).sum(-1)
        sw2 = scale_w[n0 // BLOCK:(n1 + BLOCK - 1) // BLOCK].to(torch.float32).pow(2)
        sw2 = sw2.repeat_interleave(BLOCK, dim=0)[: n1 - n0]
        w_norm_max = torch.maximum(w_norm_max, (wt * sw2).sum(-1).sqrt().max())
    bound = float(a_norm) * float(w_norm_max) * 1.02 + 1e-3
    obs = float(out.to(torch.float32).abs().max())
    res.append(('cauchy_schwarz_bound', obs <= bound,
                f'max|out|={obs:.4e} bound={bound:.4e}'))

    return res


def independent_oracle(ins, shape):
    """Chunked dequantised product, tiled over N with view-based scale
    broadcast and FP32 accumulation - a different reduction order than the
    reference's per-K-block loop."""
    a, w, scale_a, scale_w = ins
    m, n, k = CASES[int(shape[0])]
    nkb = k // BLOCK

    a_deq = (a.to(torch.float32).view(m, nkb, BLOCK) *
             scale_a.to(torch.float32).unsqueeze(-1)).view(m, k)

    y = torch.empty(m, n, dtype=torch.float32, device=a.device)
    tile = 1024
    for n0 in range(0, n, tile):
        n1 = min(n0 + tile, n)
        wt = w[n0:n1].to(torch.float32).view((n1 - n0) // BLOCK, BLOCK, nkb, BLOCK)
        st = scale_w[n0 // BLOCK:n1 // BLOCK].to(torch.float32).view((n1 - n0) // BLOCK, 1, nkb, 1)
        y[:, n0:n1] = a_deq @ (wt * st).reshape(n1 - n0, k).t()

    return y.to(torch.bfloat16)


def reduction_length(shape):
    return CASES[int(shape[0])][2]


def _fp8_blockwise_gemm(a, w, scale_a, scale_w):
    """Vectorised blockwise dequant + single BF16 tensor-core GEMM."""
    m, k = a.shape
    n = w.shape[0]
    nkb = k // BLOCK

    a_deq = (a.view(m, nkb, BLOCK).to(torch.float32) *
             scale_a.unsqueeze(-1)).view(m, k).to(torch.bfloat16)
    w_deq = (w.view(n // BLOCK, BLOCK, nkb, BLOCK).to(torch.float32) *
             scale_w.view(n // BLOCK, 1, nkb, 1)).view(n, k).to(torch.bfloat16)

    return a_deq @ w_deq.t()


_COMPILED = None


def _get_compiled():
    global _COMPILED
    if _COMPILED is None:
        try:
            _COMPILED = torch.compile(_fp8_blockwise_gemm, dynamic=True)
        except Exception:
            _COMPILED = _fp8_blockwise_gemm
    return _COMPILED

MEM_WEIGHT = 0.55
