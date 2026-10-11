# cutlass_normalization_pr3009 — Reference

## Route a100-20260824 — torch.nn.functional.layer_norm

## Background
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

## Baseline (the shipped-slow starting point)
`environment/reference_model.py`'s `Model` is the naive plain-torch oracle: it is
the correctness ground truth and the live-timing denominator. It materializes the
full intermediate (dense score matrix / per-expert dense matmul / per-pixel
composite / brute-force scan) and is therefore correct but slow — tensor mode.

## Private calibration and positive-control artifacts
`solution/reference_model_full.py` contains the strong `baseline()` implementation,
**torch.nn.functional.layer_norm**, and `solution/strong_baseline_adapter.py` exposes it only for recalibration.
Separately, `solution/solve_witness.py` is the certified real-kernel positive control installed by
`solution/solve.sh`; it traverses the same candidate verifier path as an agent solution. On the
calibration host (NVIDIA A100 80GB PCIe (Ampere, sm_80)), the frozen frontier reaches about **9.517×** over naive.

## Reward and anchors (calibrated, single NVIDIA A100 80GB PCIe (sm_80))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.105075
- official frontier (torch.nn.functional.layer_norm): speedup_vs_seed ≈ 9.517, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 9.517
- correctness: tensor, rtol=2e-3, atol=5e-3
