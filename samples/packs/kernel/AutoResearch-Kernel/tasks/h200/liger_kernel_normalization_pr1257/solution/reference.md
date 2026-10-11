# liger_kernel_normalization_pr1257 — Reference

## Route h200-20260824 — liger_kernel.ops.rms_norm.LigerRMSNormFunction

## Background
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

## Baseline (the shipped-slow starting point)
`environment/reference_model.py`'s `Model` is the naive plain-torch oracle: it is
the correctness ground truth and the live-timing denominator. It materializes the
full intermediate (dense score matrix / per-expert dense matmul / per-pixel
composite / brute-force scan) and is therefore correct but slow — tensor mode.

## Private calibration and positive-control artifacts
`solution/reference_model_full.py` contains the strong `baseline()` implementation,
**liger_kernel.ops.rms_norm.LigerRMSNormFunction**, and `solution/strong_baseline_adapter.py` exposes it only for recalibration.
Separately, `solution/solve_witness.py` is the certified real-kernel positive control installed by
`solution/solve.sh`; it traverses the same candidate verifier path as an agent solution. On the
calibration host (NVIDIA H200 (Hopper, sm_90)), the frozen frontier reaches about **2.3714×** over naive.

## Reward and anchors (calibrated, single NVIDIA H200 (sm_90))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.421692
- official frontier (liger_kernel.ops.rms_norm.LigerRMSNormFunction.apply): speedup_vs_seed ≈ 2.3714, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 2.3714
- correctness: tensor, rtol=1.6e-2, atol=1.0e-2
