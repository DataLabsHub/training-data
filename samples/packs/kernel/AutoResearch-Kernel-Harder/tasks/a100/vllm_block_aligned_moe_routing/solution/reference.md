# vllm_block_aligned_moe_routing — Reference

## Route a100-20260824 — torch.compile composite stable MoE routing preparation

## Background
Task: vllm_block_aligned_moe_routing
Domain: moe | type: fusion | mode: tensor | track: algorithmic_reformulation

Real hotspot
------------
Before a blocked grouped-GEMM MoE expert kernel can run, the top-k routing
assignments must be reorganized so that every CTA sees a contiguous, block
aligned run of token-slots that all belong to the same expert:

  * flatten the (token, slot) top-k assignments as  flat_id = token*TOPK + slot
  * group the flat ids by expert id, ascending expert
  * inside an expert, keep the original flat-id order (STABLE)
  * pad each expert segment up to ceil(count/BLOCK)*BLOCK with the sentinel
    T*TOPK, padded routing weight = 0
  * emit one expert id per BLOCK of the padded arrays, plus the total padded
    length so the grouped kernel can size its grid

Empty experts contribute zero blocks. This is pure index plumbing: it is
launch-bound and permutation-bound, never FLOP-bound, which is exactly why a
fused kernel (or a single-pass counting-sort reformulation) can beat both the
naive per-expert loop and a torch.compile'd vectorized composite.

Reference (Model) is the NAIVE, obviously-correct per-expert Python loop. It is
the correctness oracle and the timing denominator. `baseline(...)` is the honest
frontier: a cached torch.compile adapter over a deterministic vectorized
plain-PyTorch reference (no vLLM, no custom kernel), failing closed.

Calibration target: a100_pcie_80gb_cu124_torch260 (NVIDIA A100 80GB PCIe, sm_80,
CUDA 12.4, torch 2.6.0).

Forward signature
-----------------
    forward(expert_ids, routing_weights, num_experts, block_size)

`expert_ids` (int32 [T,TOPK], values in [0,E)) and `routing_weights`
(fp32 [T,TOPK]) are the only tensors; `num_experts` / `block_size` are plain
Python ints (static launch configuration, not host metadata tensors) because
get_init_inputs() is case-independent and E/BLOCK are not inferable from the
data (an expert may legally be empty).

## Baseline (the shipped-slow starting point)
`environment/reference_model.py`'s `Model` is the naive plain-torch oracle: it is
the correctness ground truth and the live-timing denominator. It materializes the
full intermediate (dense score matrix / per-expert dense matmul / per-pixel
composite / brute-force scan) and is therefore correct but slow — tensor mode.

## Private calibration and positive-control artifacts
`solution/reference_model_full.py` contains the strong `baseline()` implementation,
**torch.compile composite stable MoE routing preparation**, and `solution/strong_baseline_adapter.py` exposes it only for recalibration.
Separately, `solution/solve_witness.py` is the certified real-kernel positive control installed by
`solution/solve.sh`; it traverses the same candidate verifier path as an agent solution. On the
calibration host (NVIDIA A100 80GB PCIe (Ampere, sm_80)), the frozen frontier reaches about **3.255×** over naive.

## Reward and anchors (calibrated, single NVIDIA A100 80GB PCIe (sm_80))
- naive starter: speedup_vs_seed = 1.0, reward ≈ 0.307220
- official frontier (torch.compile): speedup_vs_seed ≈ 3.255, reward = 1.0
- frozen formula: reward = speedup_vs_seed / 3.255
- correctness: tensor, rtol=1e-6, atol=1e-6
