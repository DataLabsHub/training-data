#!/usr/bin/env python3
"""Local iteration harness for the agent. NOT the verifier entry point (that executes the
private /tests/run_task.py). This lets the agent check correctness + see the estimated speedup while
iterating on /app/solve.py. Placeholder 2.7266 is the frozen frontier; reward follows the
generated task's official anchor.

    python run_local.py --smoke     # fast correctness check on the smallest shape
    python run_local.py             # full: correctness + estimated speedup on WORKLOAD
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# run_task lives beside this file only for local iteration. Formal verification executes the
# harness-mounted private /tests copy, never this candidate-writable /app mirror.
import run_task

HEADROOM = 2.7266   # official-baseline speedup over naive (the reward denominator)
REWARD_ANCHOR = "official"   # "official" -> reward = geo/HEADROOM; "naive" -> reward = geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="smallest-shape correctness only")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    reward, correct, detail = run_task.score(HERE, os.path.join(HERE, "solve.py"),
                                             HEADROOM, a.device, reward_anchor=REWARD_ANCHOR)
    if a.smoke:
        # correctness is enough for the smoke gate; the harness greps exit code
        ok = detail.get("per_shape", [{}])[0].get("correct", correct)
        print(f"smoke correct={ok}")
        return 0 if ok else 1
    ps = detail.get("per_shape", [])
    for r in ps:
        print(f"  shape={r.get('shape')} correct={r.get('correct')} "
              f"speedup_vs_naive={r.get('composite_vs_naive')} "
              f"t_cand={r.get('t_cand_ms')}ms mem_cand={r.get('mem_cand_mb')}MB")
    print(f"reward={reward} (anchor={detail.get('reward_anchor')})  correct={correct}  "
          f"vs_seed={detail.get('speedup_vs_seed')}  vs_frontier={detail.get('speedup_vs_frontier')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
