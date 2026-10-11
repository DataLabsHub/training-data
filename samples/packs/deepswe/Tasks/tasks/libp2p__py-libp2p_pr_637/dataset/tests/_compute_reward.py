#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT::(.+)::(PASSED|FAILED|SKIPPED|ERROR)\s*$")
EXIT_RE = re.compile(r"^OMNIGRIL_EXIT_CODE=(-?\d+)$")


def read_targets(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return list(
        dict.fromkeys(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )


def ratio(results: dict[str, str], targets: list[str]) -> float:
    if not targets:
        return 1.0
    return sum(results.get(target) == "PASSED" for target in targets) / len(targets)


def parse_output(path: Path, process_rc: int) -> dict[str, str]:
    results: dict[str, str] = {}
    markers: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("RESULT::"):
            match = RESULT_RE.fullmatch(line)
            if match is None:
                raise ValueError(f"malformed RESULT line: {line!r}")
            test_id, status = match.groups()
            if test_id in results:
                raise ValueError(f"duplicate RESULT ID: {test_id}")
            results[test_id] = status
        if line.startswith("OMNIGRIL_EXIT_CODE="):
            match = EXIT_RE.fullmatch(line)
            if match is None:
                raise ValueError(f"malformed OMNIGRIL_EXIT_CODE marker: {line!r}")
            markers.append(int(match.group(1)))
    if len(markers) != 1:
        raise ValueError(
            "test runner must emit exactly one OMNIGRIL_EXIT_CODE marker; "
            f"observed {len(markers)}"
        )
    if markers[0] != process_rc:
        raise ValueError(
            "OMNIGRIL_EXIT_CODE does not match runner process return code: "
            f"marker={markers[0]} process={process_rc}"
        )
    return results


def main() -> int:
    output_path = Path(sys.argv[1])
    reward_path = Path(sys.argv[2])
    process_rc = int(sys.argv[3])
    tests_dir = Path("/tests")
    results = parse_output(output_path, process_rc)

    ftp = read_targets(tests_dir / "fail_to_pass.txt")
    p2p = read_targets(tests_dir / "pass_to_pass.txt")
    targets = list(dict.fromkeys(ftp + p2p))
    if not targets:
        targets = read_targets(tests_dir / "test_targets.txt")
        ftp = targets
    payload = {
        "reward": 1.0
        if targets and all(results.get(target) == "PASSED" for target in targets)
        else 0.0,
        "core": ratio(results, ftp),
        "regression": ratio(results, p2p),
        "results": results,
        "targets": targets,
    }
    reward_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
