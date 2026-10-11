# nvidia_tensorrt_llm_normalization_pr9969 — Reference

## Route a100-20260824 — torch.compile

## Background


## Baseline (the shipped-slow starting point)
`environment/reference_model.py`'s `Model` is the naive plain-torch oracle: it is
the correctness ground truth and the live-timing denominator. It materializes the
full intermediate (dense score matrix / per-expert dense matmul / per-pixel
composite / brute-force scan) and is therefore correct but slow — tensor mode.

## Private calibration and positive-control artifacts
`solution/reference_model_full.py` contains the strong `baseline()` implementation,
**torch.compile**, and `solution/strong_baseline_adapter.py` exposes it only for recalibration.
Separately, `solution/solve_witness.py` is the certified real-kernel positive control installed by
`solution/solve.sh`; it traverses the same candidate verifier path as an agent solution. On the
calibration host (NVIDIA A100 80GB PCIe (Ampere, sm_80)), the frozen frontier reaches about **7.4249×** over naive.

## Reward and anchors (calibrated, single NVIDIA A100 80GB PCIe (sm_80))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.134682
- official frontier (torch.compile): speedup_vs_seed ≈ 7.4249, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 7.4249
- correctness: tensor, rtol=2e-2, atol=1e-2
