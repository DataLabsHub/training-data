#!/usr/bin/env python3
"""Local iteration harness for the agent. NOT the private scorer.

    python run_local.py --smoke     # one public-shape correctness check only
    python run_local.py             # full: correctness + estimated speedup on WORKLOAD
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_task

HEADROOM = 1.2754   # measured GB200 compile/frontier speedup over naive
REWARD_ANCHOR = "naive"   # reward == geomean speedup over the naive oracle


def _load_candidate(path):
    sys.modules.pop("candidate_solve", None)
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solve"] = mod
    spec.loader.exec_module(mod)
    return mod


def _clone_arg(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().clone()
    except Exception:
        pass
    return x


def _as_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def _maybe_eval_to(model, device):
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def smoke_correctness(device_index):
    """Fast smoke: one public shape, one seed, no timing/geomean scoring."""
    import torch

    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    ref_mod = run_task._load_ref_module(HERE)
    cand_mod = _load_candidate(os.path.join(HERE, "solve.py"))

    init_args = list(ref_mod.get_init_inputs()) if hasattr(ref_mod, "get_init_inputs") else []
    ref = _maybe_eval_to(ref_mod.Model(*init_args), device)
    cand = _maybe_eval_to(cand_mod.ModelNew(*init_args), device)

    shape = tuple(ref_mod.WORKLOAD[0])
    inputs = ref_mod.make_inputs(shape, device, seed=0)
    with torch.no_grad():
        expected = ref(*[_clone_arg(x) for x in inputs])
        got = cand(*[_clone_arg(x) for x in inputs])

    expected_t = _as_tuple(expected)
    got_t = _as_tuple(got)
    output_indices = list(getattr(ref_mod, "OUTPUT_INDICES", [getattr(ref_mod, "OUTPUT_INDEX", 0)]))
    rtol = float(getattr(ref_mod, "TASK_RTOL", 1e-2))
    atol = float(getattr(ref_mod, "TASK_ATOL", 1e-2))
    checks = []
    for idx in output_indices:
        e = expected_t[idx]
        g = got_t[idx]
        ok = torch.allclose(g, e, rtol=rtol, atol=atol)
        max_abs = float((g.to(torch.float32) - e.to(torch.float32)).abs().max().item())
        checks.append((idx, bool(ok), max_abs))
    all_ok = all(ok for _, ok, _ in checks)
    print(f"smoke shape={shape} checks={checks} rtol={rtol} atol={atol}")
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one public-shape correctness check only")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        return smoke_correctness(a.device)

    reward, correct, detail = run_task.score(
        HERE,
        os.path.join(HERE, "solve.py"),
        HEADROOM,
        a.device,
        reward_anchor=REWARD_ANCHOR,
        compile_headroom=HEADROOM,
    )
    for r in detail.get("per_shape", []):
        print(f"  shape={r.get('shape')} correct={r.get('correct')} "
              f"speedup_vs_naive={r.get('composite_vs_naive')} "
              f"t_cand={r.get('t_cand_ms')}ms mem_cand={r.get('mem_cand_mb')}MB")
    print(f"reward={reward} (anchor={detail.get('reward_anchor')})  correct={correct}  "
          f"vs_seed={detail.get('speedup_vs_seed')}  vs_frontier={detail.get('speedup_vs_frontier')}")
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
