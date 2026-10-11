# S04_swiglu — Reference

## Route GB200 / sm_100

Compute out = silu(gate_proj) * up_proj for contiguous bf16 [T,F] tensors; nonlinear math and multiply are fp32, rounded to bf16 once at final store.

## Naive baseline
`environment/reference_model.py` exposes `TASK_ID = "S04_swiglu"` and `class Model`. The model is the
correctness oracle and live timing denominator.

Naive SwiGLU epilogue: fp32 upcast of both operands, sigmoid/silu, multiply, and bf16 downcast through materialized temporaries.

## Certified witness and frontier
`solution/solve_witness.py` is the certified Triton witness shipped with this task.
`solution/strong_baseline_adapter.py` wraps the naive model with `torch.compile` for recalibration
and therefore matches `frontier_source = "compile"`.

This route declares no vendor-library anchor. The GB200 frontier is the measured torch.compile
frontier on a single NVIDIA GB200 (sm_100): 3.5512x over the naive oracle. Reward is
still anchored to the naive oracle; the frontier is diagnostic and not a cap.

## Anchors
- target_id: `gb200_sm100`
- baseline (naive) speedup: 1.0
- compile frontier speedup: 3.5512x
- witness score from certification run: 1.8386x
- library frontier: not declared
- correctness: tensor, rtol=2e-2, atol=2e-2
