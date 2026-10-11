# pytorch_pytorch_normalization_pr4693 — Reference

## Route a100-20260824 — torch.nn.functional.batch_norm

## Background
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

## Baseline (the shipped-slow starting point)
`environment/reference_model.py`'s `Model` is the naive plain-torch oracle: it is
the correctness ground truth and the live-timing denominator. It materializes the
full intermediate (dense score matrix / per-expert dense matmul / per-pixel
composite / brute-force scan) and is therefore correct but slow — tensor mode.

## Private calibration and positive-control artifacts
`solution/reference_model_full.py` contains the strong `baseline()` implementation,
**torch.nn.functional.batch_norm**, and `solution/strong_baseline_adapter.py` exposes it only for recalibration.
Separately, `solution/solve_witness.py` is the certified real-kernel positive control installed by
`solution/solve.sh`; it traverses the same candidate verifier path as an agent solution. On the
calibration host (NVIDIA A100 80GB PCIe (Ampere, sm_80)), the frozen frontier reaches about **3.6719×** over naive.

## Reward and anchors (calibrated, single NVIDIA A100 80GB PCIe (sm_80))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.272339
- official frontier (torch.nn.functional.batch_norm): speedup_vs_seed ≈ 3.6719, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 3.6719
- correctness: tensor, rtol=2e-3, atol=3e-3
