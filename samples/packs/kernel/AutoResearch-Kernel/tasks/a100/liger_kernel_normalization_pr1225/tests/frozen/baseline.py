"""Official baseline dispatch (Kernel Test Bench: the denominator is an official
version-locked interface, not an author-written kernel). Frozen; agent cannot edit."""
import torch
from environment.reference_model import Model, get_init_inputs

# official library via the reference's official-ops-only wrapper baseline() (kept in
# solution/, agent-invisible). This is the OFFICIAL interface, not a candidate kernel.
from solution.reference_model_full import baseline as _official_wrapper
def baseline_call(*args):
    return _official_wrapper(*args, init_args=get_init_inputs())
