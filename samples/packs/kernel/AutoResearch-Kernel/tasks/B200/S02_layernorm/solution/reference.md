# S02_layernorm — Reference

## Route GB200 / sm_100

Compute y = (x - mean) * rsqrt(var + eps) * gamma + beta for bf16 x[T,H], gamma[H], beta[H]; mean/variance are fp32 over H and variance is biased (/H).

## Naive baseline
`environment/reference_model.py` exposes `TASK_ID = "S02_layernorm"` and `class Model`. The model is the
correctness oracle and live timing denominator.

Naive LayerNorm forward: separate fp32 upcast, row mean, centering, squared residual, biased variance, rsqrt, affine scale/bias, and bf16 downcast; several full [T,H] memory passes.

## Certified witness and frontier
`solution/solve_witness.py` is the certified Triton witness shipped with this task.
`solution/strong_baseline_adapter.py` wraps the naive model with `torch.compile` for recalibration
and therefore matches `frontier_source = "compile"`.

This route declares no vendor-library anchor. The GB200 frontier is the measured torch.compile
frontier on a single NVIDIA GB200 (sm_100): 1.1874x over the naive oracle. Reward is
still anchored to the naive oracle; the frontier is diagnostic and not a cap.

## Anchors
- target_id: `gb200_sm100`
- baseline (naive) speedup: 1.0
- compile frontier speedup: 1.1874x
- witness score from certification run: 2.2242x
- library frontier: not declared
- correctness: tensor, rtol=2e-2, atol=2e-2
