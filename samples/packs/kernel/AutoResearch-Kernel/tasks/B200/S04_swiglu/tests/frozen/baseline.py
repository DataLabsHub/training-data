"""Correctness reference for the frozen accuracy test (AutoResearch /app layout). The naive
oracle Model is the correctness ground truth; the strong-library perf denominator is
NOT shipped into /app (agent-invisible) — run_task.py owns that. Frozen; agent cannot edit."""
import torch
from reference_model import Model, get_init_inputs
_init = get_init_inputs()
_MODEL = None
def baseline_call(*args):
    global _MODEL
    if _MODEL is None:
        dev = args[0].device if (args and torch.is_tensor(args[0])) else 'cuda'
        _MODEL = Model(*_init).to(dev)
    return _MODEL(*args)
