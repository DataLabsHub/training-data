# S03_softmax_rowmax — Reference

## Route GB200 / sm_100

Compute row-wise softmax y[r,c] = exp(x[r,c]-max_c x[r,c]) / sum_c exp(x[r,c]-max_c x[r,c]) with fp32 reductions and bf16 output.

## Naive baseline
`environment/reference_model.py` exposes `TASK_ID = "S03_softmax_rowmax"` and `class Model`. The model is the
correctness oracle and live timing denominator.

Naive stable softmax: fp32 upcast, row max, subtract max, exp, row sum, divide, and bf16 downcast as separate full-tensor passes.

## Certified witness and frontier
`solution/solve_witness.py` is the certified Triton witness shipped with this task.
`solution/strong_baseline_adapter.py` wraps the naive model with `torch.compile` for recalibration
and therefore matches `frontier_source = "compile"`.

This route declares no vendor-library anchor. The GB200 frontier is the measured torch.compile
frontier on a single NVIDIA GB200 (sm_100): 1.2754x over the naive oracle. Reward is
still anchored to the naive oracle; the frontier is diagnostic and not a cap.

## Anchors
- target_id: `gb200_sm100`
- baseline (naive) speedup: 1.0
- compile frontier speedup: 1.2754x
- witness score from certification run: 1.6691x
- library frontier: not declared
- correctness: tensor, rtol=2e-2, atol=2e-3
