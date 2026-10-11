# a100_bf16_int8_mixed_gemm — Reference

## Route a100-20260824 — torch.compile mixed BF16 INT8 GEMM

## Background
Task: a100_bf16_int8_mixed_gemm

Mixed BF16-by-INT8 projection on A100 (Ampere, sm_80).

Real hotspot: a BF16 activation matrix is projected by packed 8-bit weights
(signed int8 or unsigned uint8, row-major [N, K]) with an FP32 per-output bias.
The naive reference below materializes a *full* FP32 copy of the weight tensor
before the matmul, which is exactly what a good kernel must avoid: the 8-bit ->
Tensor-Core-operand conversion belongs inside the mainloop (prologue fusion),
accumulation stays FP32, and the bias is an epilogue fusion.

`Model` is the plain-PyTorch correctness Oracle and the timing denominator.
`baseline` is the honest frontier: a cached `torch.compile` composite that
converts the weights inside the timed region.

## Baseline (the shipped-slow starting point)
`environment/reference_model.py`'s `Model` is the naive plain-torch oracle: it is
the correctness ground truth and the live-timing denominator. It materializes the
full intermediate (dense score matrix / per-expert dense matmul / per-pixel
composite / brute-force scan) and is therefore correct but slow — tensor mode.

## Private calibration and positive-control artifacts
`solution/reference_model_full.py` contains the strong `baseline()` implementation,
**torch.compile mixed BF16 INT8 GEMM**, and `solution/strong_baseline_adapter.py` exposes it only for recalibration.
Separately, `solution/solve_witness.py` is the certified real-kernel positive control installed by
`solution/solve.sh`; it traverses the same candidate verifier path as an agent solution. On the
calibration host (NVIDIA A100 80GB PCIe (Ampere, sm_80)), the frozen frontier reaches about **2.0146×** over naive.

## Reward and anchors (calibrated, single NVIDIA A100 80GB PCIe (sm_80))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.496376
- official frontier (torch.compile): speedup_vs_seed ≈ 2.0146, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 2.0146
- correctness: tensor, rtol=2e-2, atol=5e-2
