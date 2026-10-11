"""Editable solution file. Optimize `ModelNew` to compute the SAME function as the
naive oracle (reference_model.Model) as fast and memory-efficiently as possible.

FALLBACK starter (no per-task Triton starter was provided for this task). It reuses the
naive oracle, so it is correct but will NOT pass the Triton-only static gate as-is —
replace its forward with your own @triton.jit kernel. You may use Triton or inline custom
CUDA, but NOT a fused high-level torch op and NOT the strong baseline library.
"""
import torch
import torch.nn as nn
from reference_model import Model


class ModelNew(Model):
    # TODO: override forward() with your optimized Triton kernel.
    pass
