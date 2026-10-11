"""Frozen benchmark (independent test author, Kernel Test Bench §性能测试).
Times baseline and solve each with triton.testing.do_bench (warmup + repeated measurement
excludes JIT/autotune/short-kernel effects). Reports baseline_ms/solve_ms/speedup per shape.
FROZEN before kernel development; the kernel author cannot edit this file."""
import torch, triton
try:
    import pytest
except ImportError:  # runnable standalone when pytest is absent
    class _P:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda f: f
    pytest = _P()
from environment.reference_model import make_inputs, get_init_inputs
from baseline import baseline_call
from solve import solve

SHAPES = [[4096, 2048], [8192, 4096], [16384, 4096], [8192, 8192]]

@pytest.mark.parametrize("shape", SHAPES)
def test_benchmark(shape):
    dev = "cuda"
    init = get_init_inputs()
    ins_b = [x.to(dev) if torch.is_tensor(x) else x for x in make_inputs(shape, dev, seed=7)]
    ins_s = [x.to(dev) if torch.is_tensor(x) else x for x in make_inputs(shape, dev, seed=7)]
    def run_b():
        with torch.no_grad(): baseline_call(*ins_b)
    def run_s():
        with torch.no_grad(): solve(*ins_s)
    baseline_ms = triton.testing.do_bench(run_b)
    solve_ms = triton.testing.do_bench(run_s)
    speedup = baseline_ms / solve_ms if solve_ms > 0 else 0.0
    print(f'shape={shape}: baseline={baseline_ms:.4f} ms, solve={solve_ms:.4f} ms, speedup={speedup:.2f}x')
    assert solve_ms > 0
