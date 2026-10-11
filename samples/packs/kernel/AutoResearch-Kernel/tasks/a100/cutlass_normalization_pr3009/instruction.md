# reference_model.py — task_id: cutlass_normalization_pr3009

You are given a working but **unoptimized** implementation of a mid-scale GPU
autoresearch-kernel. Your job is to make it **as fast and memory-efficient as
possible** while producing the **same result**, by editing `/app/solve.py`.

## The pipeline

reference_model.py — task_id: cutlass_normalization_pr3009

Domain      : normalization
Task type   : fusion   (optimization track: fusion_and_layout)
Task mode   : tensor   (forward-only, no gradients)
Precision   : input fp16 / accumulate fp32 / output fp16
Calibration : a100_pcie_80gb_cu124_torch260  (NVIDIA A100 80GB PCIe, sm_80, Ampere)

Provenance
----------
Distilled from NVIDIA/cutlass#3009 into a portable A100 forward-LayerNorm fusion
task.  The PR supplies the *idea* (fuse the row reduction, the normalization and
the per-hidden affine so the (T, H) activation is not repeatedly round-tripped
through global memory) and nothing else: no CUTLASS header is imported here and
CUTLASS is neither a trusted implementation nor the performance denominator.

The hotspot
-----------
For an activation `x` of shape (T, H) a textbook PyTorch LayerNorm walks the
tensor several times: once for the mean, once for the centered residual, once
for the variance, once for the scale, once for the affine.  Each pass streams
T*H fp16 elements through HBM and writes an fp32 temporary that is immediately
re-read.  The whole op is DRAM-bound, so the achievable floor is roughly
"read x once, write y once" -- a fused row-per-block kernel with an fp32
Welford/two-pass reduction in registers/shared memory should approach it.

`Model` below is the NAIVE, obviously-correct reference.  It is deliberately
written as the multi-pass version so that it is both the correctness Oracle and
an honest timing denominator that a real fused kernel can beat.
`baseline(...)` is the strong library frontier (torch.nn.functional.layer_norm).

Math contract (exactly what a submission must reproduce)
--------------------------------------------------------
    accumulate in fp32:
        mu   = mean_h(x)                      # over the last axis, length H
        var  = mean_h((x - mu)^2)             # BIASED: divide by H, not H-1
        y    = (x - mu) * rsqrt(var + 1e-5) * weight + bias
    then cast `y` to fp16 exactly once, at the very end.

eps = 1e-5 lives *inside* the rsqrt.  Only `y` (shape (T, H), dtype fp16) is
returned -- mu/rsigma are not part of the output contract.

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
tolerance rtol=2e-3, atol=5e-3). The forward output tensor is compared elementwise.

Score is anchored to the frozen official frontier calibrated on this GPU:

```
speedup_vs_seed = geomean_shapes[ (mem_naive/mem_cand)^0.0 * (t_naive/t_cand)^(1-0.0) ]
reward = speedup_vs_seed / 9.517 = speedup_vs_frontier
```

The naive starter has `speedup_vs_seed` ≈ 1.0 and therefore reward ≈
**0.105075**, while the frozen official frontier scores **1.0**. A reward
above 1.0 beats that frontier. Wrong output on any shape/seed scores **0.0**.

## Two speedup numbers (only reward counts toward your score)
- `speedup_vs_seed`: raw improvement over the deliberately unfused naive oracle.
- `speedup_vs_frontier` (== reward): improvement over **torch.nn.functional.layer_norm**, calibrated at
  about **9.517×** over naive. **> 1.0 means you beat the frontier.**
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
  `pass`, `try/except` fallback to torch, importing `torch.nn.functional.layer_norm`, reading `/orig/`,
  `/tests/`, `solution/`, reward files, env vars, or spawning processes.
- **Allowed:** your own Triton/CUDA kernel, plus plain tensor plumbing around it
  (allocation, reshape/view, indexing, `.to()`, cat/stack) to wire inputs and outputs.
  The compute itself must live in your kernel, not in a torch fused op.
- Single GPU. Time budget: 2 hours.
