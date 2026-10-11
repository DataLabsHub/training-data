# h200_llama_bf16_rmsnorm_forward — Reference

## Route generic-h200-depr-20260824 — liger_kernel.ops.rms_norm.LigerRMSNormFunction

## Background


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
calibration host (NVIDIA H200 (Hopper, sm_90)), the frozen frontier reaches about **2.9056×** over naive.

## Reward and anchors (calibrated, single NVIDIA H200 (sm_90))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.344163
- official frontier (liger_kernel.ops.rms_norm.LigerRMSNormFunction.apply): speedup_vs_seed ≈ 2.9056, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 2.9056
- correctness: tensor, rtol=2e-2, atol=2e-2
