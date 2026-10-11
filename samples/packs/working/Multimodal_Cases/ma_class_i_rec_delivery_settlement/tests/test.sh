#!/bin/sh
set -eu

mkdir -p /logs/verifier

if ! python3 /tests/grade.py; then
  if [ ! -s /logs/verifier/reward.json ]; then
    python3 - <<'PY'
import json
from pathlib import Path

Path("/logs/verifier/reward.json").write_text(
    json.dumps(
        {
            "reward": 0.0,
            "status": "verifier_failed_closed",
            "error": "grade.py exited before writing a result",
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  fi
fi

test -s /logs/verifier/reward.json
