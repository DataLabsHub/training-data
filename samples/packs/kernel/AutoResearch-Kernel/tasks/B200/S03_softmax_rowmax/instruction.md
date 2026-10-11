# S03_softmax_rowmax — Numerically stable row-wise softmax over bf16 [rows, cols]

You are given a correct but deliberately unoptimized PyTorch reference. Edit only
`/app/solve.py` and implement `class ModelNew` with a real GPU kernel.

## Task

Compute row-wise softmax y[r,c] = exp(x[r,c]-max_c x[r,c]) / sum_c exp(x[r,c]-max_c x[r,c]) with fp32 reductions and bf16 output.

Public workload shapes: (4096,1024), (8192,2048). Hidden shapes are sampled inside the documented legal domain;
do not hard-code only the public shapes.

## Scoring setup

| Item | Path / Value |
|------|--------------|
| Editable file | `/app/solve.py` |
| Naive oracle | `/app/reference_model.py` (`class Model`) |
| Local full check | `python /app/run_local.py` |
| Local smoke | `python /app/run_local.py --smoke` |
| GPU target | NVIDIA GB200, sm_100 (`target_id = gb200_sm100`) |
| Frontier provenance | torch.compile frontier, not a vendor-library anchor |

Correctness mode is tensor comparison with rtol=2e-2, atol=2e-3. Wrong output on any checked
shape or seed scores 0.0 regardless of speed.

Reward is geomean wall-clock speedup over the naive oracle. Memory is reported, but this GB200
route uses `mem_weight = 0.0`, so reward is timing-only:

```text
reward = geomean_shapes(t_naive / t_candidate)
```

The verifier also reports `speedup_vs_frontier`, using the calibrated GB200 torch.compile frontier
(1.2754x over naive). That frontier is diagnostic only; reward remains anchored to the naive
oracle and is not capped by the frontier.

## Optimization target

Fuse row max, exponentiation, row sum, and normalization into a real Triton/CUDA row kernel; do not compute exp(x) before subtracting the row max.

## Rules

- Use a real Triton (`@triton.jit` with `tl.*`) or CUDA (`__global__`) kernel for the compute.
- Do not replace the task with a high-level PyTorch/fused op such as `torch.softmax`,
  `torch.layer_norm`, matmul/einsum, or a library call.
- Do not read `/orig`, `/tests`, `solution/`, reward files, environment variables, command-line
  arguments, or spawn subprocesses/network requests.
- Plain tensor plumbing around your kernel is fine; the mathematical compute must be in your kernel.
- Single GPU. Time budget: 2 hours.
