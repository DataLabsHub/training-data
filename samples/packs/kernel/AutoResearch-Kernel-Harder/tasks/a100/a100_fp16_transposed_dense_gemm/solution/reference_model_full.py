# reference_model.py
# Task: a100_fp16_transposed_dense_gemm
# Domain: gemm | Type: fusion | Mode: tensor
# Track: fusion_and_layout
# Calibration target: a100_pcie_80gb_cu124_torch260 (NVIDIA A100 80GB PCIe, sm_80, CUDA 12.4, torch 2.6.0)
#
# Real hotspot
# -----------
# A dense projection Y = A @ B^T where:
#   * A is FP16, row-major, shape [M, K]
#   * B is FP16, row-major, shape [N, K]  (the *right* operand is stored row-major and is
#     consumed as a logical transpose -- i.e. the reduction axis K is the fast axis of BOTH
#     operands: the classic "NT" GEMM layout emitted by fused QKV / MLP projections)
#   * accumulation must be FP32 and the produced tensor must be FP32 [M, N]
#
# The naive way to express "FP16 operands, FP32 math, FP32 result" in plain PyTorch is to
# up-cast both operands and run an honest FP32 matmul with TF32 disabled. On A100 that path
# runs on the FP32 CUDA cores (~19.5 TFLOP/s) and additionally materializes two FP32 copies of
# the operands. That is exactly what `Model` does, and it is the timing denominator.
#
# The frontier (`baseline`) keeps the operands in FP16 so the mma.sync tensor-core path
# (~312 TFLOP/s) is used with FP32 accumulation, then widens the result -- compiled with
# torch.compile(dynamic=False, fullgraph=True) so no eager fallback is taken. A real kernel
# should beat *that* by writing FP32 straight out of the accumulators (no FP16 round-trip on
# the epilogue) and by consuming B's row-major layout directly instead of materializing a
# transposed copy.

import torch
import torch.nn as nn

# The reference math is true IEEE FP32 (BASELINE_CONTRACT.allow_tf32 == False). TF32 would make
# the *oracle* less accurate than the candidate, which would be dishonest.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# FP32 accumulation is part of the contract: forbid cuBLAS' split-K FP16 reductions.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


# case_id -> (M, N, K).  Public workload is 0..4, hidden tripwire ids are 5..9.
CASES = {
    0: (3456, 1024, 2048),
    1: (1024, 1024, 2048),
    2: (2048, 1536, 1024),
    3: (4096, 1024, 2048),
    4: (512, 4096, 2048),
    5: (1536, 3072, 2048),
    6: (3072, 768, 4096),
    7: (2304, 1280, 3072),
    8: (4096, 2048, 1024),
    9: (8192, 512, 2048),
}

# Sentinel row indices planted into A by make_inputs (see invariants()).
_ZERO_ROW = 0          # A[0, :] == 0                -> output row must be exactly 0
_ONEHOT_ROW = 1        # A[1, :] == e_k              -> output row must equal B[:, k] exactly


def _onehot_col(K: int) -> int:
    return K // 3


class Model(nn.Module):
    """NAIVE reference: up-cast FP16 operands to FP32, materialize the logical transpose,
    run a plain FP32 matmul. Obviously correct, deliberately unoptimized."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # a: [M, K] fp16 (row-major)   b: [N, K] fp16 (row-major, consumed as b^T)
        a32 = a.float()
        b32 = b.float()
        out = torch.matmul(a32, b32.transpose(0, 1))
        return out.contiguous()


def make_inputs(shape, device, seed=0):
    """Return [A, B] with A:[M,K] fp16 contiguous, B:[N,K] fp16 contiguous.

    No pre-transposed copy of B is ever handed to the candidate -- eliminating the layout
    materialization is the point of the task.
    """
    case_id = int(shape[0])
    M, N, K = CASES[case_id]
    dev = torch.device(device)

    g = torch.Generator(device=dev)
    g.manual_seed(int(seed) * 7919 + case_id * 104729 + 12345)

    a = (torch.randn(M, K, generator=g, device=dev, dtype=torch.float32) * 0.5).to(torch.float16)
    b = (torch.randn(N, K, generator=g, device=dev, dtype=torch.float32) * 0.5).to(torch.float16)

    # Deterministic sentinels with closed-form outputs (checked by invariants()).
    a[_ZERO_ROW, :] = 0.0
    a[_ONEHOT_ROW, :] = 0.0
    a[_ONEHOT_ROW, _onehot_col(K)] = 1.0

    return [a.contiguous(), b.contiguous()]


def get_init_inputs():
    return []


WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]

TASK_MODE = 'tensor'

# Tolerance budget: the frontier keeps the epilogue in FP16 before widening, so up to a half
# ULP of FP16 (~5e-4 relative) is expected against the FP32 oracle. FP32-accumulation ordering
# differences across K<=4096 are ~1e-6 relative and negligible here.
TASK_RTOL = 2e-3
TASK_ATOL = 2e-2

GRAD_INPUTS = []

OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    'input_dtype': 'fp16',
    'output_dtype': 'fp32',
    'accumulation_dtype': 'fp32',
    'allow_tf32': False,
    'note': (
        'Y = A @ B^T with A[M,K] fp16 row-major and B[N,K] fp16 row-major consumed as a '
        'logical transpose (K is the fast axis of both operands). Multiply-accumulate must be '
        'FP32 (allow_fp16_reduced_precision_reduction disabled); the returned tensor is '
        'contiguous FP32 [M,N]. TF32 is disabled so the FP32 reference path is true IEEE '
        'single precision. A pre-transposed B is never supplied and must not be materialized '
        'as a separate dense tensor by a compliant implementation.'
    ),
}


def sample_shapes(rng, n):
    """Hidden-shape tripwire: unseen case ids 5..9, returned as one-tuples in axis order
    ['case_id'] (never the expanded (M, N, K))."""
    hidden = [5, 6, 7, 8, 9]
    out = []
    pool = []
    while len(out) < int(n):
        if not pool:
            pool = list(hidden)
            rng.shuffle(pool)
        out.append((pool.pop(),))
    return out


def reduction_length(shape):
    """Length of the contracted axis K for case `shape`."""
    return CASES[int(shape[0])][2]


def invariants(out, ins, shape):
    """Metamorphic certification. None of these re-run Model; each is an algebraic property
    the true product must satisfy, recomputed from the raw inputs in FP64."""
    checks = []
    if isinstance(out, (tuple, list)):
        y = out[0]
    else:
        y = out
    a, b = ins[0], ins[1]
    case_id = int(shape[0])
    M, N, K = CASES[case_id]

    # ---- 1. structural: shape / dtype / finiteness -------------------------------------
    ok_shape = tuple(y.shape) == (M, N)
    ok_dtype = (y.dtype == torch.float32)
    ok_finite = bool(torch.isfinite(y).all().item())
    checks.append((
        'shape_dtype_finite',
        bool(ok_shape and ok_dtype and ok_finite),
        f'shape={tuple(y.shape)} expected={(M, N)} dtype={y.dtype} finite={ok_finite}',
    ))

    # ---- 2. closed form on planted sentinel rows ---------------------------------------
    # A[0] == 0  =>  row 0 of the product is *exactly* zero (bit-exact, any accumulation order).
    # A[1] == e_k =>  row 1 of the product is *exactly* B[:, k] (a single exact FP16 term).
    zero_row = y[_ZERO_ROW]
    zero_err = float(zero_row.abs().max().item())
    k = _onehot_col(K)
    onehot_ref = b[:, k].float()
    onehot_err = float((y[_ONEHOT_ROW] - onehot_ref).abs().max().item())
    checks.append((
        'sentinel_closed_form_rows',
        bool(zero_err == 0.0 and onehot_err <= 1e-3),
        f'max|Y[0,:]|={zero_err:.3e} (must be exactly 0); '
        f'max|Y[1,:]-B[:,{k}]|={onehot_err:.3e} (must be ~0)',
    ))

    # ---- 3. two independently recomputed FP64 dot products ------------------------------
    # Straight-line reductions in double precision -- no matmul kernel, no Model code path.
    probes = [(M // 2, N // 3), (M - 1, N - 1)]
    worst_rel = 0.0
    detail_bits = []
    for (r, c) in probes:
        prod = a[r].double() * b[c].double()
        ref = float(prod.sum().item())
        scale = float(prod.abs().sum().item())
        got = float(y[r, c].item())
        tol = max(1e-2, 5e-5 * scale)
        worst_rel = max(worst_rel, abs(got - ref) / max(tol, 1e-30))
        detail_bits.append(f'Y[{r},{c}]={got:.6f} fp64={ref:.6f} tol={tol:.3e}')
    checks.append((
        'fp64_probe_dot_products',
        bool(worst_rel <= 1.0),
        '; '.join(detail_bits),
    ))

    # ---- 4. linear-functional / additivity identity --------------------------------------
    # (A B^T) 1_N == A (B^T 1_N) == A @ colsum(B).  Contracting the output against the all-ones
    # vector must agree with contracting B first -- an order-of-operations invariant that only
    # holds if every one of the M*N entries is right.
    bsum = b.double().sum(dim=0)                     # [K]
    expect = torch.mv(a.double(), bsum)              # [M]
    got_rs = y.sum(dim=1).double()
    num = float((got_rs - expect).norm().item())
    den = float(expect.norm().item()) + 1e-12
    rel = num / den
    checks.append((
        'row_sum_additivity',
        bool(rel <= 5e-3),
        f'||Y@1 - A@colsum(B)|| / ||A@colsum(B)|| = {rel:.3e} (limit 5e-3)',
    ))

    return checks


def independent_oracle(ins, shape):
    """Trusted FP64 recompute of the full product, chunked over M so the double-precision
    temporaries stay small. Never calls Model."""
    a, b = ins[0], ins[1]
    M, N, K = CASES[int(shape[0])]

    bt64 = b.double().transpose(0, 1).contiguous()   # [K, N]
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)

    rows_per_chunk = max(1, min(M, (1 << 22) // max(1, N)))
    for start in range(0, M, rows_per_chunk):
        stop = min(M, start + rows_per_chunk)
        acc = torch.matmul(a[start:stop].double(), bt64)
        out[start:stop] = acc.float()

    return out.contiguous()


_COMPILED_CACHE = {}


def _fp16_nt_projection(a, b):
    # FP16 tensor-core GEMM against the logical transpose of a row-major B, widened to FP32.
    return torch.matmul(a, b.transpose(0, 1)).to(torch.float32).contiguous()


def baseline(*args, init_args=None):
    """Strong library frontier: torch.compile'd FP16 transposed GEMM.

    Compiled with dynamic=False (shape-specialized) and fullgraph=True (no eager fallback), so
    this is Inductor's best static-shape lowering of the NT projection on sm_80 tensor cores.
    """
    a, b = args[0], args[1]

    key = (a.device.type, a.device.index, tuple(a.shape), tuple(b.shape), a.dtype, b.dtype)
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        fn = torch.compile(_fp16_nt_projection, dynamic=False, fullgraph=True)
        _COMPILED_CACHE[key] = fn
    return fn(a, b)
