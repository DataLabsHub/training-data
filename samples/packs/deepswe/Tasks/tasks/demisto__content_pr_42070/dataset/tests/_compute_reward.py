#!/usr/bin/env python3
import json
import os


def read(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    output = os.environ.get("OMNIGRIL_OUTPUT", "/logs/verifier/test_output.txt")
    statuses = {}
    if os.path.exists(output):
        for line in open(output, encoding="utf-8"):
            if line.startswith("RESULT::"):
                tid, status = line.rstrip("\n")[len("RESULT::"):].rsplit("::", 1)
                statuses[tid] = status
    ftp = read("/tests/fail_to_pass.txt")
    p2p = read("/tests/pass_to_pass.txt")
    targets = read("/tests/test_targets.txt")
    expected = set(ftp + p2p)
    ok = bool(ftp) and bool(p2p) and set(targets) == expected and all(statuses.get(t) == "PASSED" for t in expected)
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/reward.json", "w", encoding="utf-8") as f:
        json.dump({"reward": 1.0 if ok else 0.0}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
