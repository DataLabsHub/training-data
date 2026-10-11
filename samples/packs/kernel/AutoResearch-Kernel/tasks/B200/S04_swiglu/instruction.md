# S04_swiglu — SwiGLU activation epilogue over bf16 [T, F]

You are given a correct but deliberately unoptimized PyTorch reference. Edit only
`/app/solve.py` and implement `class ModelNew` with a real GPU kernel.

## Task

Compute out = silu(gate_proj) * up_proj for contiguous bf16 [T,F] tensors; nonlinear math and multiply are fp32, rounded to bf16 once at final store.

Public workload shapes: (4096,4096), (8192,4096), (4096,8192). Hidden shapes are sampled inside the documented legal domain;
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

Correctness mode is tensor comparison with rtol=2e-2, atol=2e-2. Wrong output on any checked
shape or seed scores 0.0 regardless of speed.

Reward is geomean wall-clock speedup over the naive oracle. Memory is reported, but this GB200
route uses `mem_weight = 0.0`, so reward is timing-only:

```text
reward = geomean_shapes(t_naive / t_candidate)
```

The verifier also reports `speedup_vs_frontier`, using the calibrated GB200 torch.compile frontier
(3.5512x over naive). That frontier is diagnostic only; reward remains anchored to the naive
oracle and is not capped by the frontier.

## Optimization target

Fuse the elementwise epilogue into a load-load-compute-store Triton/CUDA kernel; GEMM is out of scope.

## Rules

- Use a real Triton (`@triton.jit` with `tl.*`) or CUDA (`__global__`) kernel for the compute.
- Do not replace the task with a high-level PyTorch/fused op such as `torch.softmax`,
  `torch.layer_norm`, matmul/einsum, or a library call.
- Do not read `/orig`, `/tests`, `solution/`, reward files, environment variables, command-line
  arguments, or spawn subprocesses/network requests.
- Plain tensor plumbing around your kernel is fine; the mathematical compute must be in your kernel.
- Single GPU. Time budget: 2 hours.
