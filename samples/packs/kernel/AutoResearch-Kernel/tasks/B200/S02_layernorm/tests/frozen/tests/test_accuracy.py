"""Frozen correctness test (independent test author, Kernel Test Bench §精度测试).
Compares the official baseline against the candidate with torch.testing.assert_close
at its DEFAULT dtype tolerance. Computed separately from any timing. FROZEN before kernel
development; the kernel author cannot edit this file."""
import torch
try:
    import pytest
except ImportError:  # runnable standalone when pytest is absent
    class _P:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda f: f
    pytest = _P()
from reference_model import make_inputs, get_init_inputs, Model
from baseline import baseline_call
from solve import ModelNew

SHAPES = [[2048, 1024], [4096, 1024], [2048, 2048], [4096, 2048]]

@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy(shape):
    dev = "cuda"
    init = get_init_inputs()
    ins_b = [x.to(dev) if torch.is_tensor(x) else x for x in make_inputs(shape, dev, seed=917)]
    ins_s = [x.to(dev) if torch.is_tensor(x) else x for x in make_inputs(shape, dev, seed=917)]
    _cand = ModelNew(*init)
    if hasattr(_cand, 'to'): _cand = _cand.to(dev)
    with torch.no_grad():
        out_b = baseline_call(*ins_b)
        out_s = _cand(*ins_s)
    for oi in [0]:
        ob = out_b[oi] if isinstance(out_b, (tuple, list)) else out_b
        os_ = out_s[oi] if isinstance(out_s, (tuple, list)) else out_s
        # same dtype + assert_close default tolerance (no self-set rtol/atol)
        torch.testing.assert_close(os_, ob.to(os_.dtype))
