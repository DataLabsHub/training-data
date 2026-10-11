# reference_model.py
#
# task_id : a100_indexed_row_softmax
# domain  : reduction        task_type : fusion        task_mode : tensor
# track   : fusion_and_layout
# target  : a100_pcie_80gb_cu124_torch260  (NVIDIA A100 80GB PCIe, sm_80, ampere)
#
# Hotspot: an indexed gather producer feeds a numerically stable row-wise softmax.
# The naive reference below materializes the full gathered BF16 intermediate and then
# runs max / subtract / exp / sum / divide as separate passes over it. A real fused
# kernel can beat this by never writing the gathered tensor to HBM (recompute the
# gather inside the reduction, or keep the row resident in shared memory / registers),
# while still honoring repeated and nonmonotonic index semantics.

import torch
import torch.nn as nn

# (R, C, sourceC) -- index is [R, C] int64 into source [R, sourceC]
CASES = {
    0: (1, 128, 256),
    1: (16, 1024, 2048),
    2: (64, 4096, 8192),
    3: (256, 8192, 16384),
    4: (128, 16384, 32768),
    5: (3, 257, 509),
    6: (33, 1537, 3071),
    7: (512, 2048, 4096),
    8: (1024, 8192, 12289),
    9: (2048, 4096, 65536),
}


class Model(nn.Module):
    """Naive, obviously-correct reference: gather, then stable softmax in FP32.

    Deliberately unoptimized -- every stage is its own pass over a full
    [R, C] intermediate. This is the correctness Oracle and the timing denominator.
    """

    def forward(self, source, index):
        gathered = torch.gather(source, 1, index)          # [R, C] bf16 intermediate
        x = gathered.to(torch.float32)                     # fp32 accumulate domain
        row_max = x.max(dim=1, keepdim=True).values        # pass 1
        shifted = x - row_max                              # pass 2
        e = torch.exp(shifted)                             # pass 3
        row_sum = e.sum(dim=1, keepdim=True)               # pass 4
        probs = e / row_sum                                # pass 5
        return probs.to(torch.bfloat16)


def make_inputs(shape, device, seed=0):
    """shape == (case_id,). Returns [source bf16 [R,sourceC], index int64 [R,C]]."""
    (case_id,) = shape
    R, C, SC = CASES[int(case_id)]

    device = torch.device(device)
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) * 9176 + int(case_id) * 7919 + 12345) % (2**31 - 1))

    # Source rows get wildly different scales and offsets so that omitting the
    # max-subtraction overflows FP32 exp(): stability is part of the contract.
    base = torch.randn((R, SC), generator=g, device=device, dtype=torch.float32)
    row_scale = torch.rand((R, 1), generator=g, device=device, dtype=torch.float32) * 7.0 + 0.5
    row_shift = torch.rand((R, 1), generator=g, device=device, dtype=torch.float32) * 240.0 - 120.0
    source = (base * row_scale + row_shift).to(torch.bfloat16)

    # Index layout: a repeat-heavy block, a strictly descending run, then pure random.
    n_rep = max(1, C // 4)
    n_desc = max(1, C // 4)
    n_rand = max(0, C - n_rep - n_desc)

    distinct = max(1, min(SC, 8))
    seeds = torch.randint(0, SC, (R, distinct), generator=g, device=device, dtype=torch.int64)
    reps = (n_rep + distinct - 1) // distinct
    rep_block = seeds.repeat(1, reps)[:, :n_rep]

    start = torch.randint(0, SC, (R, 1), generator=g, device=device, dtype=torch.int64)
    offs = torch.arange(n_desc, device=device, dtype=torch.int64).unsqueeze(0)
    desc_block = (start - offs) % SC

    rand_block = torch.randint(
        0, SC, (R, n_rand), generator=g, device=device, dtype=torch.int64
    )

    index = torch.cat([rep_block, desc_block, rand_block], dim=1).contiguous()

    # Sentinel pair for the permutation-equivariance invariant: row 1 sees the same
    # source row as row 0 but with a reversed index, so its probabilities must be
    # the reverse of row 0's.
    if R >= 2:
        source[1].copy_(source[0])
        index[1].copy_(torch.flip(index[0], dims=[0]))

    return [source, index]


def get_init_inputs():
    return []


WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-2
TASK_ATOL = 2e-3
GRAD_INPUTS = []
OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    'input_dtype': 'bfloat16',
    'output_dtype': 'bfloat16',
    'accumulation_dtype': 'float32',
    'allow_tf32': True,
    'note': (
        'source is bfloat16 [R, sourceC]; index is int64 [R, C] and may contain '
        'repeats, descending runs and arbitrary nonmonotonic positions. The gather '
        'must be performed by the implementation itself -- baseline never receives a '
        'pre-gathered tensor. max/exp/sum accumulate in float32 with explicit '
        'max-subtraction (row values reach |x| ~ 150, so unshifted exp overflows). '
        'Output is bfloat16 [R, C]. No matmul is involved; allow_tf32 is irrelevant '
        'but left enabled to match the calibration target defaults.'
    ),
}


def sample_shapes(rng, n):
    """Unseen case ids (5..9) that are not in WORKLOAD -- hidden-shape tripwire."""
    pool = [5, 6, 7, 8, 9]

    def pick(hi):
        for name, call in (
            ('randrange', lambda: rng.randrange(hi)),
            ('integers', lambda: rng.integers(0, hi)),
            ('randint', lambda: rng.randint(0, hi - 1)),
        ):
            if hasattr(rng, name):
                return int(call()) % hi
        return 0

    start = pick(len(pool))
    return [(pool[(start + i) % len(pool)],) for i in range(int(n))]


def invariants(out, ins, shape):
    """Metamorphic checks on `out` alone -- Model is never re-run here."""
    (case_id,) = shape
    R, C, SC = CASES[int(case_id)]
    source, index = ins[0], ins[1]
    checks = []

    # 1. structure: shape, dtype, finiteness, and probability range [0, 1].
    ok_shape = tuple(out.shape) == (R, C)
    ok_dtype = out.dtype == torch.bfloat16
    f = out.to(torch.float64)
    ok_finite = bool(torch.isfinite(f).all())
    lo = float(f.min()) if f.numel() else 0.0
    hi = float(f.max()) if f.numel() else 0.0
    ok_range = ok_finite and lo >= 0.0 and hi <= 1.0 + 1e-2
    checks.append((
        'shape_dtype_finite_probability_range',
        bool(ok_shape and ok_dtype and ok_range),
        f'shape={tuple(out.shape)} expected={(R, C)} dtype={out.dtype} '
        f'finite={ok_finite} min={lo:.3e} max={hi:.6f}',
    ))

    # 2. each row is a distribution: sums to 1 (bf16 storage error tolerance).
    row_sums = f.sum(dim=1)
    max_dev = float((row_sums - 1.0).abs().max())
    checks.append((
        'row_sums_unit',
        max_dev <= 3e-2,
        f'max |rowsum-1| = {max_dev:.4e} over {R} rows (tol 3e-2)',
    ))

    # 3. repeated index positions gather identical values, so they must receive
    #    identical probabilities. Positions come from `index`, not from Model.
    idx0 = index[0]
    uniq, inverse, counts = torch.unique(idx0, return_inverse=True, return_counts=True)
    k = int(torch.argmax(counts))
    pos = (inverse == k).nonzero().flatten()
    if pos.numel() >= 2:
        seg = f[0].index_select(0, pos)
        spread = float((seg - seg[0]).abs().max())
        checks.append((
            'repeated_index_identical_probability',
            spread <= 1e-6,
            f'index value {int(uniq[k])} appears {int(pos.numel())}x in row 0; '
            f'max probability spread = {spread:.3e}',
        ))
    else:
        checks.append((
            'repeated_index_identical_probability',
            True,
            'no repeated index value present in row 0; check not applicable',
        ))

    # 4. permutation equivariance on the sentinel pair: row 1 reads the same source
    #    row through a reversed index, so out[1] must be reverse(out[0]).
    if R >= 2:
        same_src = bool(torch.equal(source[0], source[1]))
        flipped = bool(torch.equal(index[1], torch.flip(index[0], dims=[0])))
        if same_src and flipped:
            diff = (f[1] - torch.flip(f[0], dims=[0])).abs()
            ref = torch.flip(f[0], dims=[0]).abs()
            err = float((diff - (TASK_RTOL * ref + TASK_ATOL)).max())
            checks.append((
                'reversed_index_permutation_equivariance',
                err <= 0.0,
                f'max tolerance overshoot on out[1] vs reverse(out[0]) = {err:.3e} '
                f'(max abs diff {float(diff.max()):.3e})',
            ))
        else:
            checks.append((
                'reversed_index_permutation_equivariance',
                False,
                f'sentinel precondition broken: same_source_row={same_src} '
                f'reversed_index={flipped}',
            ))
    else:
        checks.append((
            'reversed_index_permutation_equivariance',
            True,
            'R=1, no sentinel pair available; check not applicable',
        ))

    return checks


def independent_oracle(ins, shape):
    """Row-at-a-time FP64 softmax written from the definition, no gather/softmax fusion."""
    (case_id,) = shape
    R, C, SC = CASES[int(case_id)]
    source, index = ins[0], ins[1]

    out = torch.empty((R, C), device=index.device, dtype=torch.float64)
    for r in range(R):
        row = source[r].to(torch.float64)
        vals = row[index[r]]
        m = vals.max()
        e = torch.exp(vals - m)
        out[r] = e / e.sum()
    return out.to(torch.bfloat16)


def reduction_length(shape):
    """Elements reduced per softmax row."""
    (case_id,) = shape
    return CASES[int(case_id)][1]


_COMPILED = {}


def _gather_softmax(source, index):
    return torch.softmax(torch.gather(source, 1, index).to(torch.float32), dim=1).to(
        torch.bfloat16
    )


def baseline(source, index, init_args=None):
    """Strong library frontier: torch.compile over the gather+softmax composite.

    Receives the raw source and index (never a pre-gathered tensor), so Inductor
    is free to fuse the gather into the softmax reduction itself.
    """
    key = (source.device.type, source.dtype, tuple(source.shape), tuple(index.shape))
    fn = _COMPILED.get(key)
    if fn is None:
        fn = torch.compile(_gather_softmax, dynamic=False)
        _COMPILED[key] = fn
    return fn(source, index)
