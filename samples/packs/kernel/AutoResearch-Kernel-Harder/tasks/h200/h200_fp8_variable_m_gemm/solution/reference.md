# h200_fp8_variable_m_gemm — Reference

## Route h200-20260824 — torch.compile variable-M FP8 GEMM

## Background
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

## Baseline (the shipped-slow starting point)
`environment/reference_model.py`'s `Model` is the naive plain-torch oracle: it is
the correctness ground truth and the live-timing denominator. It materializes the
full intermediate (dense score matrix / per-expert dense matmul / per-pixel
composite / brute-force scan) and is therefore correct but slow — tensor mode.

## Private calibration and positive-control artifacts
`solution/reference_model_full.py` contains the strong `baseline()` implementation,
**torch.compile variable-M FP8 GEMM**, and `solution/strong_baseline_adapter.py` exposes it only for recalibration.
Separately, `solution/solve_witness.py` is the certified real-kernel positive control installed by
`solution/solve.sh`; it traverses the same candidate verifier path as an agent solution. On the
calibration host (NVIDIA H200 (Hopper, sm_90)), the frozen frontier reaches about **2.2873×** over naive.

## Reward and anchors (calibrated, single NVIDIA H200 (sm_90))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.437197
- official frontier (torch.compile): speedup_vs_seed ≈ 2.2873, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 2.2873
- correctness: tensor, rtol=2e-2, atol=2e-2
