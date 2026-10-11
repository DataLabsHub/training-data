"""
Task: vllm_block_aligned_moe_routing
Domain: moe | type: fusion | mode: tensor | track: algorithmic_reformulation

Real hotspot
------------
Before a blocked grouped-GEMM MoE expert kernel can run, the top-k routing
assignments must be reorganized so that every CTA sees a contiguous, block
aligned run of token-slots that all belong to the same expert:

  * flatten the (token, slot) top-k assignments as  flat_id = token*TOPK + slot
  * group the flat ids by expert id, ascending expert
  * inside an expert, keep the original flat-id order (STABLE)
  * pad each expert segment up to ceil(count/BLOCK)*BLOCK with the sentinel
    T*TOPK, padded routing weight = 0
  * emit one expert id per BLOCK of the padded arrays, plus the total padded
    length so the grouped kernel can size its grid

Empty experts contribute zero blocks. This is pure index plumbing: it is
launch-bound and permutation-bound, never FLOP-bound, which is exactly why a
fused kernel (or a single-pass counting-sort reformulation) can beat both the
naive per-expert loop and a torch.compile'd vectorized composite.

Reference (Model) is the NAIVE, obviously-correct per-expert Python loop. It is
the correctness oracle and the timing denominator. `baseline(...)` is the honest
frontier: a cached torch.compile adapter over a deterministic vectorized
plain-PyTorch reference (no vLLM, no custom kernel), failing closed.

Calibration target: a100_pcie_80gb_cu124_torch260 (NVIDIA A100 80GB PCIe, sm_80,
CUDA 12.4, torch 2.6.0).

Forward signature
-----------------
    forward(expert_ids, routing_weights, num_experts, block_size)

`expert_ids` (int32 [T,TOPK], values in [0,E)) and `routing_weights`
(fp32 [T,TOPK]) are the only tensors; `num_experts` / `block_size` are plain
Python ints (static launch configuration, not host metadata tensors) because
get_init_inputs() is case-independent and E/BLOCK are not inferable from the
data (an expert may legally be empty).
"""

import torch
import torch.nn as nn


# case_id -> (T, TOPK, E, BLOCK)
CASES = {
    0: (1, 1, 16, 16),
    1: (8, 2, 64, 16),
    2: (32, 4, 128, 32),
    3: (128, 8, 256, 64),
    4: (512, 2, 64, 32),
    5: (1024, 4, 128, 64),
    6: (2048, 8, 256, 128),
    7: (4096, 2, 512, 64),
    8: (17, 3, 32, 16),
    9: (255, 5, 96, 32),
    10: (1536, 6, 384, 128),
    11: (3072, 8, 512, 256),
}

# Public (visible) workload: case ids 0..5. Hidden sampling uses 6..11 only.
WORKLOAD = [(0,), (1,), (2,), (3,), (4,), (5,)]

TASK_MODE = 'tensor'
TASK_RTOL = 1e-6
TASK_ATOL = 1e-6
GRAD_INPUTS = []
OUTPUT_INDICES = [0, 1, 2, 3]

BASELINE_CONTRACT = {
    'input_dtype': 'mixed',
    'output_dtype': 'mixed',
    'accumulation_dtype': 'fp32',
    'allow_tf32': False,
    'note': (
        'Inputs: expert_ids int32 [T,TOPK] in [0,E) plus routing_weights fp32 [T,TOPK]; '
        'num_experts and block_size are static Python ints. Outputs, all CUDA: '
        '(0) sorted_token_slot_ids int32 [total_padded] with sentinel T*TOPK for padding, '
        '(1) sorted_weights fp32 [total_padded] with exactly 0.0 at padded slots, '
        '(2) expert_id_per_block int32 [total_padded//BLOCK], '
        '(3) num_tokens_post_pad int32 [1]. total_padded = sum_e ceil(count_e/BLOCK)*BLOCK, '
        'so empty experts contribute zero blocks and the output length is data dependent. '
        'No matmul is involved: accumulation_dtype fp32 refers to the fp32 routing-weight '
        'gather, which must be an exact value-preserving copy (bitwise), and TF32 is '
        'irrelevant/disabled. Integer outputs are compared for exact equality. The private '
        'baseline is a cached torch.compile adapter over a deterministic vectorized '
        'plain-PyTorch routing reference; it never calls vLLM or a custom kernel and raises '
        'on compile/run failure instead of falling back to eager.'
    ),
}


class Model(nn.Module):
    """Naive, obviously-correct block-aligned MoE routing preparation.

    One Python iteration per expert: mask, collect ascending flat ids, pad the
    segment, gather weights. Host syncs and E kernel launches per call are
    intentional -- this is the correctness anchor, not a fast path.
    """

    def forward(self, expert_ids, routing_weights, num_experts, block_size):
        num_experts = int(num_experts)
        block_size = int(block_size)
        device = expert_ids.device

        flat_experts = expert_ids.reshape(-1).to(torch.int64)
        flat_weights = routing_weights.reshape(-1).to(torch.float32)
        n = int(flat_experts.numel())
        sentinel = n  # T*TOPK

        id_chunks = []
        weight_chunks = []
        block_chunks = []

        for e in range(num_experts):
            idx = torch.nonzero(flat_experts == e, as_tuple=False).reshape(-1)
            count = int(idx.numel())
            if count == 0:
                continue  # empty expert -> zero blocks
            n_blocks = (count + block_size - 1) // block_size
            padded = n_blocks * block_size

            seg_ids = torch.full((padded,), sentinel, dtype=torch.int32, device=device)
            seg_ids[:count] = idx.to(torch.int32)

            seg_weights = torch.zeros((padded,), dtype=torch.float32, device=device)
            seg_weights[:count] = flat_weights[idx]

            id_chunks.append(seg_ids)
            weight_chunks.append(seg_weights)
            block_chunks.append(
                torch.full((n_blocks,), e, dtype=torch.int32, device=device)
            )

        if id_chunks:
            sorted_ids = torch.cat(id_chunks)
            sorted_weights = torch.cat(weight_chunks)
            expert_id_per_block = torch.cat(block_chunks)
        else:
            sorted_ids = torch.empty((0,), dtype=torch.int32, device=device)
            sorted_weights = torch.empty((0,), dtype=torch.float32, device=device)
            expert_id_per_block = torch.empty((0,), dtype=torch.int32, device=device)

        num_tokens_post_pad = torch.full(
            (1,), int(sorted_ids.numel()), dtype=torch.int32, device=device
        )
        return sorted_ids, sorted_weights, expert_id_per_block, num_tokens_post_pad


def get_init_inputs():
    return []


def make_inputs(shape, device, seed=0):
    """Build (expert_ids int32 [T,TOPK], routing_weights fp32 [T,TOPK], E, BLOCK).

    Data is generated on CPU with a seeded CPU generator (so `seed` fully
    determines the assignment pattern and the result is device independent) and
    moved to `device` before any consumer arithmetic. The distribution
    deliberately contains, for every seed:
      * empty experts (a deterministic stride of expert ids is never drawn),
      * a heavily skewed hot expert (~half the mass plus a solid token run),
      * exact ties: repeated expert ids inside a token and rows of identical
        routing weights, so stable-by-flat-id ordering is observable.
    """
    case_id = int(shape[0])
    T, TOPK, E, BLOCK = CASES[case_id]
    dev = torch.device(device)
    seed = int(seed)

    g = torch.Generator().manual_seed(seed * 9176 + case_id * 101 + 7)
    n = T * TOPK

    # Deterministically silence a stride of experts so empty experts always exist.
    stride = 3 if E >= 3 else 1
    offset = seed % stride
    allowed = [e for e in range(E) if (e + offset) % stride != 0] if E >= 3 else list(range(E))
    if not allowed:
        allowed = list(range(E))

    # Skewed (power-law) mass over the allowed experts, permuted per seed.
    ranks = torch.arange(len(allowed), dtype=torch.float32)
    mass = 1.0 / (ranks + 1.0) ** 1.5
    perm = torch.randperm(len(allowed), generator=g)
    probs = torch.zeros(E, dtype=torch.float32)
    for i, e in enumerate(allowed):
        probs[e] = mass[int(perm[i])]
    probs = probs / probs.sum()

    flat = torch.multinomial(probs, n, replacement=True, generator=g)
    expert_ids = flat.reshape(T, TOPK).to(torch.int32)

    # Hot expert: a contiguous token run routed entirely to one expert -> long
    # segments, many intra-token ties, deep block padding pressure.
    hot = allowed[int(perm[0])]
    run = max(1, T // 8)
    start = (seed * 5) % max(1, T - run + 1)
    expert_ids[start:start + run, :] = int(hot)

    weights = torch.rand((T, TOPK), generator=g, dtype=torch.float32) * 0.9 + 0.1
    # Rows of exactly-equal weights (ties in value, never in flat-id order).
    tie_rows = torch.arange(T)[(torch.arange(T) + seed) % 7 == 0]
    if tie_rows.numel() > 0:
        weights[tie_rows] = 0.25
    weights = weights / weights.sum(dim=1, keepdim=True)  # strictly positive

    return [expert_ids.to(dev), weights.to(dev), int(E), int(BLOCK)]


def sample_shapes(rng, n):
    """Unseen case ids (6..11) only -- hidden-shape tripwire."""
    pool = [6, 7, 8, 9, 10, 11]

    def _pick(hi):
        for attempt in (
            lambda: int(rng.integers(0, hi)),
            lambda: int(rng.randrange(hi)),
            lambda: int(rng.randint(0, hi - 1)),
        ):
            try:
                return attempt()
            except AttributeError:
                continue
        return 0

    out = []
    remaining = list(pool)
    for _ in range(int(n)):
        if not remaining:
            remaining = list(pool)
        j = _pick(len(remaining))
        out.append((remaining.pop(j),))
    return out


def invariants(out, ins, shape):
    """Metamorphic certification of the routing layout.

    Everything below is derived from the INPUTS (a CPU bincount / Python
    grouping) and from the candidate output alone. Model is never re-run, so
    these are independent structural evidence, not a precision recompute.
    """
    checks = []
    try:
        sorted_ids, sorted_weights, block_expert, num_post = out[0], out[1], out[2], out[3]

        case_id = int(shape[0])
        T, TOPK, E, BLOCK = CASES[case_id]
        if len(ins) >= 4:
            E, BLOCK = int(ins[2]), int(ins[3])

        flat_e = ins[0].detach().to('cpu').reshape(-1).to(torch.int64)
        flat_w = ins[1].detach().to('cpu').reshape(-1).to(torch.float32)
        n = int(flat_e.numel())

        ids = sorted_ids.detach().to('cpu').to(torch.int64)
        wts = sorted_weights.detach().to('cpu').to(torch.float32)
        blk = block_expert.detach().to('cpu').to(torch.int64)
        npost = int(num_post.detach().to('cpu').reshape(-1)[0].item())

        counts = torch.bincount(flat_e, minlength=E)[:E]
        nblocks = (counts + BLOCK - 1) // BLOCK
        expected_total = int((nblocks * BLOCK).sum().item())
        expected_blocks = int(nblocks.sum().item())

        # 1) exact padded lengths + dtypes + scalar block count
        len_ok = (
            int(ids.numel()) == expected_total
            and int(wts.numel()) == expected_total
            and int(blk.numel()) == expected_blocks
            and expected_total == expected_blocks * BLOCK
            and npost == expected_total
        )
        dtype_ok = (
            sorted_ids.dtype == torch.int32
            and sorted_weights.dtype == torch.float32
            and block_expert.dtype == torch.int32
            and num_post.dtype == torch.int32
        )
        finite_ok = bool(torch.isfinite(wts).all())
        checks.append((
            'exact_padded_lengths_and_dtypes',
            bool(len_ok and dtype_ok and finite_ok),
            'len={} exp={} blocks={} exp={} num_post={} dtypes_ok={} finite={}'.format(
                int(ids.numel()), expected_total, int(blk.numel()), expected_blocks,
                npost, dtype_ok, finite_ok),
        ))

        nonpad = ids < n
        pad = ~nonpad

        # 2) block metadata: ascending, one entry per BLOCK, and every
        #    non-padding element really belongs to its block's expert
        ref_blocks = torch.repeat_interleave(torch.arange(E, dtype=torch.int64), nblocks)
        blocks_exact = blk.numel() == ref_blocks.numel() and bool(torch.equal(blk, ref_blocks))
        monotone = bool(blk.numel() < 2 or torch.all(blk[1:] >= blk[:-1]))
        elem_ok = False
        if int(blk.numel()) * BLOCK == int(ids.numel()):
            elem_expert = torch.repeat_interleave(blk, BLOCK)
            elem_ok = bool(torch.equal(flat_e[ids[nonpad]], elem_expert[nonpad]))
        checks.append((
            'expert_blocks_monotone_and_consistent',
            bool(blocks_exact and monotone and elem_ok),
            'blocks_exact={} monotone={} elementwise_expert_match={}'.format(
                blocks_exact, monotone, elem_ok),
        ))

        # 3) exact one-to-one coverage of all n assignments, stable ascending
        #    flat-id order inside every expert, padding only at segment tails
        nonpad_ids = ids[nonpad]
        coverage = bool(
            nonpad_ids.numel() == n
            and torch.equal(torch.sort(nonpad_ids).values, torch.arange(n, dtype=torch.int64))
        )
        stable = True
        cursor = 0
        if int(ids.numel()) == expected_total:
            for e in range(E):
                seg_len = int(nblocks[e].item()) * BLOCK
                if seg_len == 0:
                    continue
                seg = ids[cursor:cursor + seg_len]
                c = int(counts[e].item())
                stable = stable and bool(torch.all(seg[:c] < n))
                stable = stable and bool(torch.all(seg[c:] == n))  # tail sentinel only
                if c > 1:
                    stable = stable and bool(torch.all(seg[1:c] > seg[:c - 1]))
                cursor += seg_len
        else:
            stable = False
        checks.append((
            'stable_permutation_and_full_coverage',
            bool(coverage and stable),
            'coverage={} nonpad={} n={} stable_within_expert={}'.format(
                coverage, int(nonpad_ids.numel()), n, stable),
        ))

        # 4) weights: padded slots exactly 0.0, non-padding slots bitwise equal
        #    to the routing weight of the flat id they carry
        pad_zero = bool(torch.all(wts[pad] == 0.0))
        gather_exact = bool(torch.equal(wts[nonpad], flat_w[nonpad_ids]))
        checks.append((
            'weight_preservation_and_zero_padding',
            bool(pad_zero and gather_exact),
            'padded_all_zero={} nonpad_bitwise_exact={} pad_slots={}'.format(
                pad_zero, gather_exact, int(pad.sum().item())),
        ))
    except Exception as exc:  # noqa: BLE001 - report, never mask
        checks.append(('invariants_error', False, repr(exc)))
    return checks


def independent_oracle(ins, shape):
    """Trusted CPU/Python stable grouping. Never touches Model or baseline.

    Buckets flat ids in ascending order per expert with plain Python lists, pads
    each bucket, then moves only the finished results to the input device.
    """
    expert_ids, routing_weights = ins[0], ins[1]
    case_id = int(shape[0])
    T, TOPK, E, BLOCK = CASES[case_id]
    if len(ins) >= 4:
        E, BLOCK = int(ins[2]), int(ins[3])

    device = expert_ids.device
    flat_e = expert_ids.detach().to('cpu').reshape(-1).to(torch.int64).tolist()
    flat_w = routing_weights.detach().to('cpu').reshape(-1).to(torch.float32).tolist()
    n = len(flat_e)
    sentinel = n

    buckets = [[] for _ in range(E)]
    for flat_id, e in enumerate(flat_e):
        buckets[int(e)].append(flat_id)

    ids = []
    weights = []
    blocks = []
    for e in range(E):
        bucket = buckets[e]
        if not bucket:
            continue
        n_blocks = (len(bucket) + BLOCK - 1) // BLOCK
        padding = n_blocks * BLOCK - len(bucket)
        for flat_id in bucket:
            ids.append(flat_id)
            weights.append(flat_w[flat_id])
        ids.extend([sentinel] * padding)
        weights.extend([0.0] * padding)
        blocks.extend([e] * n_blocks)

    return (
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
        torch.tensor(blocks, dtype=torch.int32, device=device),
        torch.tensor([len(ids)], dtype=torch.int32, device=device),
    )


def _stable_block_routing_reference(expert_ids, routing_weights, num_experts, block_size):
    """Deterministic vectorized plain-PyTorch routing prep (compile target).

    Stable argsort by expert gives ascending-expert groups with ascending flat
    ids inside each group; a per-expert padded prefix offset scatters them into
    the block-aligned layout in one pass.
    """
    device = expert_ids.device
    flat_e = expert_ids.reshape(-1).to(torch.int64)
    flat_w = routing_weights.reshape(-1).to(torch.float32)
    n = flat_e.numel()

    counts = torch.bincount(flat_e, minlength=num_experts)[:num_experts]
    nblocks = torch.div(counts + block_size - 1, block_size, rounding_mode='floor')
    padded = nblocks * block_size
    seg_start = torch.cumsum(padded, 0) - padded
    grp_start = torch.cumsum(counts, 0) - counts

    order = torch.argsort(flat_e, stable=True)
    sorted_e = flat_e[order]
    rank = torch.arange(n, device=device, dtype=torch.int64) - grp_start[sorted_e]
    dest = seg_start[sorted_e] + rank

    total = int(padded.sum().item())
    sorted_ids = torch.full((total,), n, dtype=torch.int32, device=device)
    sorted_ids[dest] = order.to(torch.int32)
    sorted_weights = torch.zeros((total,), dtype=torch.float32, device=device)
    sorted_weights[dest] = flat_w[order]

    block_expert = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int32), nblocks
    )
    num_tokens_post_pad = torch.full((1,), total, dtype=torch.int32, device=device)
    return sorted_ids, sorted_weights, block_expert, num_tokens_post_pad

MEM_WEIGHT = 0.6
