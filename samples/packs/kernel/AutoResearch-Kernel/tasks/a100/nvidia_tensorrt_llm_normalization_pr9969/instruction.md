# autoresearch-kernel task

You are given a working but **unoptimized** implementation of a mid-scale GPU
autoresearch-kernel. Your job is to make it **as fast and memory-efficient as
possible** while producing the **same result**, by editing `/app/solve.py`.

## The pipeline

BF16 grouped RMSNorm with a fixed group size of 128, a per-channel weight,
and a post-normalization SiLU gate. Reductions and the affine/gate chain are
computed in FP32 and rounded once to BF16; a fused kernel should avoid the
reference model's full-sized intermediates.

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
tolerance rtol=2e-2, atol=1e-2). The forward output tensor is compared elementwise.

Score is anchored to the frozen official frontier calibrated on this GPU:

```
speedup_vs_seed = geomean_shapes[ (mem_naive/mem_cand)^0.0 * (t_naive/t_cand)^(1-0.0) ]
reward = speedup_vs_seed / 7.4249 = speedup_vs_frontier
```

The naive starter has `speedup_vs_seed` ≈ 1.0 and therefore reward ≈
**0.134682**, while the frozen official frontier scores **1.0**. A reward
above 1.0 beats that frontier. Wrong output on any shape/seed scores **0.0**.

## Two speedup numbers (only reward counts toward your score)
- `speedup_vs_seed`: raw improvement over the deliberately unfused naive oracle.
- `speedup_vs_frontier` (== reward): improvement over **torch.compile**, calibrated at
  about **7.4249×** over naive. **> 1.0 means you beat the frontier.**
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
