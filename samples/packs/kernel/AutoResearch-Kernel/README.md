# A100/H200/B200 AutoResearch

This bundle contains 13 GPU kernel optimization tasks and their corresponding Codex optimization trajectories across NVIDIA A100, H200, and B200 GPUs.

- `tasks/` contains the self-contained task packages, organized by target GPU and task ID. Each package includes the task instruction, environment, target profile, reference implementation, verifier, and solution artifacts.
- `trajs/` contains one sanitized Codex event-stream trajectory (`<task>.events.jsonl`) and its final optimized kernel (`<task>.final.py`) for each task, using the same GPU and task IDs.
- `speedup-report.md` reports each final solution's speedup over the naive baseline and the frozen calibrated frontier. A frontier ratio above `1.0×` means the solution outperformed that frontier.

The `tasks/` and `trajs/` directories contain 13 matching task/trajectory/final-kernel sets: five A100 tasks, five H200 tasks, and three B200 tasks.
