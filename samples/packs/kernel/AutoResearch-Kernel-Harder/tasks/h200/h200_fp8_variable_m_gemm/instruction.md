# Task: h200_fp8_variable_m_gemm

You are given a working but **unoptimized** implementation of a mid-scale GPU
autoresearch-kernel. Your job is to make it **as fast and memory-efficient as
possible** while producing the **same result**, by editing `/app/solve.py`.

## The pipeline

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

## Setup

| Item | Path / Value |
|------|--------------|
| Editable | `/app/solve.py` (defines `class ModelNew`) |
| Naive oracle (read-only) | `/app/reference_model.py` (defines `class Model` = correctness ground truth) |
| Local check / score | `python /app/run_local.py` (correctness + estimated reward) |
| Quick smoke | `python /app/run_local.py --smoke` |
| GPU | Single NVIDIA H200 (Hopper, sm_90) |

The starter `ModelNew` in `/app/solve.py` is a copy of the naive oracle: it is
already correct but slow (that is your starting point, not a blank slate). Rewrite
its `forward` to compute the identical function faster / with less peak memory.
You may decide how many kernels to use, how to fuse stages, whether to change the
algorithm (e.g. skip masked blocks, avoid materializing the full score matrix,
tile in shared memory), and how to dispatch by shape.

## Your goal

`ModelNew(*init_args)(*inputs)` must match `Model`'s output (mode: **tensor**;
tolerance rtol=2e-2, atol=2e-2). The forward output tensor is compared elementwise.

Score is anchored to the frozen official frontier calibrated on this GPU:

```
speedup_vs_seed = geomean_shapes[ (mem_naive/mem_cand)^0.55 * (t_naive/t_cand)^(1-0.55) ]
reward = speedup_vs_seed / 2.2873 = speedup_vs_frontier
```

The naive starter has `speedup_vs_seed` ≈ 1.0 and therefore reward ≈
**0.437197**, while the frozen official frontier scores **1.0**. A reward
above 1.0 beats that frontier. Wrong output on any shape/seed scores **0.0**.

## Two speedup numbers (only reward counts toward your score)
- `speedup_vs_seed`: raw improvement over the deliberately unfused naive oracle.
- `speedup_vs_frontier` (== reward): improvement over **torch.compile**, calibrated at
  about **2.2873×** over naive. **> 1.0 means you beat the frontier.**
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
