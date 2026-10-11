# reference_model.py — task_id: pytorch_pytorch_normalization_pr4693

You are given a working but **unoptimized** implementation of a mid-scale GPU
autoresearch-kernel. Your job is to make it **as fast and memory-efficient as
possible** while producing the **same result**, by editing `/app/solve.py`.

## The pipeline

reference_model.py — task_id: pytorch_pytorch_normalization_pr4693

Mixed-precision forward BatchNorm over a 2-D activation matrix (T, H).

  * x       : fp16, shape (T, H), contiguous   -- T samples, H features
  * weight  : fp32, shape (H,), positive       -- per-feature affine scale
  * bias    : fp32, shape (H,)                 -- per-feature affine shift

Semantics (training-mode batch norm, no running stats):

    mean_h = mean_t(x[:, h])
    var_h  = mean_t((x[:, h] - mean_h)^2)          # BIASED variance
    y[t,h] = (x[t,h] - mean_h) * rsqrt(var_h + eps) * weight[h] + bias[h]

All reduction / normalization arithmetic happens in fp32; the result is
rounded to fp16 exactly once, at the very end.

The real hotspot is the heterogeneous-operand case (fp16 activations with
fp32 affine parameters): a fused kernel should accumulate per-feature
statistics in fp32 and emit the affine chain into an fp16 output without
materializing fp32 intermediates or rejecting the mixed dtypes.

NOTE: despite stale operator metadata elsewhere in the task bundle, this is
BatchNorm (reduction over the T axis, statistics per feature h), not
LayerNorm or GroupNorm.

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
tolerance rtol=2e-3, atol=3e-3). The forward output tensor is compared elementwise.

Score is anchored to the frozen official frontier calibrated on this GPU:

```
speedup_vs_seed = geomean_shapes[ (mem_naive/mem_cand)^0.0 * (t_naive/t_cand)^(1-0.0) ]
reward = speedup_vs_seed / 3.6719 = speedup_vs_frontier
```

The naive starter has `speedup_vs_seed` ≈ 1.0 and therefore reward ≈
**0.272339**, while the frozen official frontier scores **1.0**. A reward
above 1.0 beats that frontier. Wrong output on any shape/seed scores **0.0**.

## Two speedup numbers (only reward counts toward your score)
- `speedup_vs_seed`: raw improvement over the deliberately unfused naive oracle.
- `speedup_vs_frontier` (== reward): improvement over **torch.nn.functional.batch_norm**, calibrated at
  about **3.6719×** over naive. **> 1.0 means you beat the frontier.**
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
  `pass`, `try/except` fallback to torch, importing `torch.nn.functional.batch_norm`, reading `/orig/`,
  `/tests/`, `solution/`, reward files, env vars, or spawning processes.
- **Allowed:** your own Triton/CUDA kernel, plus plain tensor plumbing around it
  (allocation, reshape/view, indexing, `.to()`, cat/stack) to wire inputs and outputs.
  The compute itself must live in your kernel, not in a torch fused op.
- Single GPU. Time budget: 2 hours.
