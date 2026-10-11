# reference_model.py — task_id: liger_kernel_normalization_pr1257

You are given a working but **unoptimized** implementation of a mid-scale GPU
autoresearch-kernel. Your job is to make it **as fast and memory-efficient as
possible** while producing the **same result**, by editing `/app/solve.py`.

## The pipeline

reference_model.py — task_id: liger_kernel_normalization_pr1257

Domain      : normalization
Task type   : fusion  (optimization track: fusion_and_layout)
Task mode   : tensor (forward-only)

WHAT THIS TASK IS
-----------------
Forward Llama/Megatron-style RMSNorm over large contiguous BF16 activations:

    ms   = mean(x.float()**2, dim=-1)          # FP32 row reduction
    rstd = rsqrt(ms + 1e-6)                    # FP32 reciprocal std
    xhat = (x.float() * rstd).to(bf16)         # Llama "cast-before-weight"
    y    = xhat * weight                       # BF16 channel gain

The kernel opportunity is to fuse the FP32 row reduction, the reciprocal
standard deviation, the normalization, its *required* BF16 cast, and the BF16
channel gain into a single pass over the [T, H] activation, so the tensor is
read once and written once instead of being materialized several times.

PROVENANCE / CALIBRATION NOTE
-----------------------------
The originating measurements come from the Megatron RMSNorm benchmark context
in linkedin/Liger-Kernel PR #1257. *All* source numbers in that PR — timing,
memory, and speedup — are H100-only and DO NOT TRANSFER. Nothing from that
report is reused as an expectation here.

This task is NEWLY TARGETED to:
    target_id          = h200_sm90_cuda129_torch280_nv2505
    gpu_model          = NVIDIA H200
    compute_capability = 9.0 (Hopper / SM90)

Every performance figure for this task must be re-measured on H200. Treat any
H100 datapoint as historical context only.

The Model below is the NAIVE, obviously-correct plain-PyTorch reference: it is
both the correctness Oracle and the timing denominator. It is deliberately not
optimized. `baseline(...)` calls the strong library (LigerRMSNormFunction) and
is the honest frontier to beat.

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
tolerance rtol=1.6e-2, atol=1.0e-2). The forward output tensor is compared elementwise.

Score is anchored to the frozen official frontier calibrated on this GPU:

```
speedup_vs_seed = geomean_shapes[ (mem_naive/mem_cand)^0.5 * (t_naive/t_cand)^(1-0.5) ]
reward = speedup_vs_seed / 2.3714 = speedup_vs_frontier
```

The naive starter has `speedup_vs_seed` ≈ 1.0 and therefore reward ≈
**0.421692**, while the frozen official frontier scores **1.0**. A reward
above 1.0 beats that frontier. Wrong output on any shape/seed scores **0.0**.

## Two speedup numbers (only reward counts toward your score)
- `speedup_vs_seed`: raw improvement over the deliberately unfused naive oracle.
- `speedup_vs_frontier` (== reward): improvement over **liger_kernel.ops.rms_norm.LigerRMSNormFunction.apply**, calibrated at
  about **2.3714×** over naive. **> 1.0 means you beat the frontier.**
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
  `pass`, `try/except` fallback to torch, importing `liger_kernel.ops.rms_norm.ligerrmsnormfunction`, reading `/orig/`,
  `/tests/`, `solution/`, reward files, env vars, or spawning processes.
- **Allowed:** your own Triton/CUDA kernel, plus plain tensor plumbing around it
  (allocation, reshape/view, indexing, `.to()`, cat/stack) to wire inputs and outputs.
  The compute itself must live in your kernel, not in a torch fused op.
- Single GPU. Time budget: 2 hours.
