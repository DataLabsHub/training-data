# Task: vllm_block_aligned_moe_routing

You are given a working but **unoptimized** implementation of a mid-scale GPU
autoresearch-kernel. Your job is to make it **as fast and memory-efficient as
possible** while producing the **same result**, by editing `/app/solve.py`.

## The pipeline

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

## Setup

| Item | Path / Value |
|------|--------------|
| Editable | `/app/solve.py` (defines `class ModelNew`) |
| Naive oracle (read-only) | `/app/reference_model.py` (defines `class Model` = correctness ground truth) |
| Local check / score | `python /app/run_local.py` (correctness + estimated reward) |
| Quick smoke | `python /app/run_local.py --smoke` |
| GPU | Single NVIDIA A100 80GB PCIe (Ampere, sm_80) |

The starter `ModelNew` in `/app/solve.py` is a copy of the naive oracle: it is
already correct but slow (that is your starting point, not a blank slate). Rewrite
its `forward` to compute the identical function faster / with less peak memory.
You may decide how many kernels to use, how to fuse stages, whether to change the
algorithm (e.g. skip masked blocks, avoid materializing the full score matrix,
tile in shared memory), and how to dispatch by shape.

## Your goal

`ModelNew(*init_args)(*inputs)` must match `Model`'s output (mode: **tensor**;
tolerance rtol=1e-6, atol=1e-6). The forward output tensor is compared elementwise.

Score is anchored to the frozen official frontier calibrated on this GPU:

```
speedup_vs_seed = geomean_shapes[ (mem_naive/mem_cand)^0.6 * (t_naive/t_cand)^(1-0.6) ]
reward = speedup_vs_seed / 3.255 = speedup_vs_frontier
```

The naive starter has `speedup_vs_seed` ≈ 1.0 and therefore reward ≈
**0.307220**, while the frozen official frontier scores **1.0**. A reward
above 1.0 beats that frontier. Wrong output on any shape/seed scores **0.0**.

## Two speedup numbers (only reward counts toward your score)
- `speedup_vs_seed`: raw improvement over the deliberately unfused naive oracle.
- `speedup_vs_frontier` (== reward): improvement over **torch.compile**, calibrated at
  about **3.255×** over naive. **> 1.0 means you beat the frontier.**
- `peak_mem_mb`: your candidate's peak memory.

Timing is **steady-state warm-cache**: one-time setup (JIT/autotune/mask build) is run
as warmup and excluded; timed iters run back-to-back with no L2 flush. naive and your
candidate are measured under the identical regime, interleaved.

## Rules

You MUST write a real GPU kernel. The candidate is rejected (score 0) by a static
check unless it contains an actual kernel — Triton (`@triton.jit` + `tl.*` ops) or
CUDA (`__global__` + `torch.utils.cpp_extension.load_inline`).

- Edit `/app/solve.py` only. Do not modify `/app/reference_model.py` or add files under `/app`.
- **Forbidden (these are the hack we reject):** calling a fused high-level op instead
  of writing the kernel — `F.rms_norm`, `F.scaled_dot_product_attention`,
  `torch.softmax`, `torch.matmul`/`torch.bmm`/`torch.einsum`, `F.linear`, `torch.layer_norm`,
  `F.embedding_bag`, `torch.topk`, etc. Also forbidden: inheriting `Model` and using
  `pass`, `try/except` fallback to torch, importing `torch.compile`, reading `/orig/`,
  `/tests/`, `solution/`, reward files, env vars, or spawning processes.
- **Allowed:** your own Triton/CUDA kernel, plus plain tensor plumbing around it
  (allocation, reshape/view, indexing, `.to()`, cat/stack) to wire inputs and outputs.
  The compute itself must live in your kernel, not in a torch fused op.
- Single GPU. Time budget: 2 hours.
