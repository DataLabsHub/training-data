#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier /tmp/run

if [ -d /tests/scripts ]; then
  cp -a /tests/scripts/. /tmp/run/
fi

export PATH="/usr/local/go/bin:/testbed/.venv/bin:$PATH"
export GO111MODULE=off
export GOPATH="${GOPATH:-/root/go}"
export GOCACHE="${GOCACHE:-/testbed/.cache/go-build}"
mkdir -p "$GOCACHE" "$GOPATH/src/github.com/wmnsk"
if [ ! -e "$GOPATH/src/github.com/wmnsk/gopcua" ]; then
  ln -s /testbed "$GOPATH/src/github.com/wmnsk/gopcua"
fi

cd /testbed
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  case "$rel" in
    /*|*..*) echo "unsafe test file path: $rel" >&2; exit 2 ;;
  esac
  mkdir -p "$(dirname "$rel")"
  cp "/tests/test_files/$rel" "$rel"
done < /tests/test_files_manifest.txt

mapfile -t TARGETS < /tests/test_targets.txt
set +e
bash /tests/test_run.sh "${TARGETS[@]}" > /logs/verifier/test_output.txt 2>&1
runner_rc=$?
set -e
# OMNIGRIL_NORMALIZE_MISSING_RESULTS_V1
# Make verifier evidence explicit: if the runner/compiler did not emit a verdict
# for a required F2P/P2P target, append RESULT::<target>::ERROR before reward.
python3 - <<'OMNIGRIL_NORMALIZE_RESULTS_PY'
import json
from pathlib import Path

VALID = {"PASSED", "FAILED", "SKIPPED", "ERROR"}
root = Path("/tests")
log_dir = Path("/logs/verifier")
output = log_dir / "test_output.txt"
log_dir.mkdir(parents=True, exist_ok=True)
if not output.exists():
    output.write_text("", encoding="utf-8")

def read_lines(name):
    path = root / name
    try:
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and line not in values:
                values.append(line)
        return values
    except Exception:
        return []

statuses = {}
for line in output.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.startswith("RESULT::"):
        continue
    payload = line[len("RESULT::"):]
    try:
        test_id, status = payload.rsplit("::", 1)
    except ValueError:
        continue
    if status in VALID:
        statuses[test_id] = status
required = []
for name in ("fail_to_pass.txt", "pass_to_pass.txt"):
    for tid in read_lines(name):
        if tid not in required:
            required.append(tid)
if not required:
    required = read_lines("test_targets.txt")
missing = [tid for tid in required if tid not in statuses]
if missing:
    with output.open("a", encoding="utf-8") as stream:
        for tid in missing:
            stream.write(f"RESULT::{tid}::ERROR\n")
(log_dir / "evidence-normalization.json").write_text(json.dumps({
    "schema": "coding-synthesis.verifier-evidence-normalization/v1",
    "required": len(required),
    "observed_before": len(statuses),
    "filled_error": len(missing),
    "valid": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
OMNIGRIL_NORMALIZE_RESULTS_PY

python3 /tests/_compute_reward.py
exit 0
