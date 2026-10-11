# A100/H200 AutoResearch — Batch 2

This bundle contains nine GPU kernel optimization tasks and their corresponding one-hour Codex AutoResearch trajectories across NVIDIA A100 and H200 GPUs.

- `tasks/` contains the self-contained task packages, organized by target GPU and task ID. Each package includes the instruction, environment, local target profile, reference implementation, verifier, and witness artifacts.
- `trajs/` contains one sanitized Codex event stream per task, using the flat `<task_id>.events.jsonl` layout of the reference delivery.
- Each trajectory ends with a `delivery.final_kernel` record containing the selected kernel source, followed by a digest-bound `delivery.final_result` record.
- `speedup-report.md` reports each verified final solution's speedup over the naive baseline and the frozen calibrated frontier. A frontier ratio above `1.0x` means the solution outperformed that frontier.

The packaged witness proves that an optimization opportunity exists; it is not a golden answer or an optimization upper bound. The final kernel is the solution selected from the Codex AutoResearch run.

The bundle contains seven A100 pairs and two H200 pairs. Failed or non-certifiable runs are excluded.
