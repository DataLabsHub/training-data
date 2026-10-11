#!/usr/bin/env python3
import json
import math
import pathlib
import re
from pathlib import Path

ROOT = pathlib.Path("/tests")
LOG_DIR = pathlib.Path("/logs/verifier")
OUTPUT = LOG_DIR / "test_output.txt"
RE_RESULT = re.compile(r"^RESULT::(.+)::(PASSED|FAILED|SKIPPED|ERROR)$")


def read_lines(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main():
    ftp = read_lines(ROOT / "fail_to_pass.txt")
    p2p = read_lines(ROOT / "pass_to_pass.txt")
    targets = read_lines(ROOT / "test_targets.txt")
    expected = ftp + p2p

    statuses = {}
    if OUTPUT.exists():
        for raw in OUTPUT.read_text(errors="replace").splitlines():
            match = RE_RESULT.match(raw.strip())
            if match:
                statuses[match.group(1)] = match.group(2)

    passed = [tid for tid in expected if statuses.get(tid) == "PASSED"]
    missing_or_bad = [tid for tid in expected if statuses.get(tid) != "PASSED"]
    manifests_ok = (
        bool(ftp)
        and bool(p2p)
        and targets == list(dict.fromkeys(expected))
        and len(set(expected)) == len(expected)
    )
    reward = 1.0 if manifests_ok and not missing_or_bad else 0.0
    if not math.isfinite(reward):
        reward = 0.0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "reward.json").write_text(
        json.dumps(
            {
                "reward": reward,
                "passed": len(passed),
                "total": len(expected),
                "missing_or_bad": missing_or_bad,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
