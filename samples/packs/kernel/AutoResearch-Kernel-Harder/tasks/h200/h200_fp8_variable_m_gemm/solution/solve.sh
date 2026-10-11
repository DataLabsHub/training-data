#!/usr/bin/env bash
# Install the certified Triton witness (self-contained solve.py) into /app. Copies ONLY solve.py,
# so the /app whitelist is preserved (no strong-library file, no extra imports).
set -euo pipefail
cp /solution/solve_witness.py /app/solve.py
echo "Certified Triton witness installed as /app/solve.py."
