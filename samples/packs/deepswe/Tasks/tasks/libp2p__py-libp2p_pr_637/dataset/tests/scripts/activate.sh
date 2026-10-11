#!/usr/bin/env bash
set -euo pipefail

cd /testbed

# PART A: activate the persisted Python test environment.
if [ -f /testbed/.venv/bin/activate ]; then
  source /testbed/.venv/bin/activate
fi

# PART B: lightweight repository smoke.
python - <<'PY'
import libp2p
import pytest

assert libp2p is not None
assert pytest is not None
PY
pytest --version >/dev/null

# === add PART A + PART B above this line ===

