#!/bin/bash
set -uo pipefail

APP_DIR="${APP_DIR:-/app}"
TEST_DIR="${TEST_DIR:-/tests}"
LOG_DIR="${LOG_DIR:-/logs/verifier}"
mkdir -p "$LOG_DIR"

phase_files=(
  test_01_envelope.py test_02_identity.py test_03_deadline.py
  test_04_retry_budget.py test_05_claim.py test_06_lease.py
  test_07_visibility.py test_08_quarantine.py test_09_cancellation.py
  test_10_canvas.py test_11_quorum.py test_12_completion.py
  test_13_result_cas.py test_14_tombstone.py test_15_journal.py
  test_16_replay.py test_17_drain.py test_18_orphan_escrow.py
  test_19_clock_regression.py test_20_observer_regression.py
)
phase_names=(
  canonical_envelope deterministic_identity absolute_deadline retry_budget
  backend_claim delivery_lease visibility_renewal stale_quarantine
  cancellation_scopes canvas_propagation quorum_checkpoint chord_completion
  result_cas result_tombstone event_journal replay_cursor drain_snapshot
  orphan_escrow clock_regression observer_regression
)
weights=(3 3 3 4 4 4 4 4 3 4 4 4 4 4 4 4 5 6 2 2)

run_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$run_dir"
}
trap cleanup EXIT
cp "$TEST_DIR/roadmap_helpers.py" "$run_dir/roadmap_helpers.py"

total=0
earned=0
results_file="$run_dir/results.tsv"
: > "$results_file"

for index in "${!phase_files[@]}"; do
  file="${phase_files[$index]}"
  name="${phase_names[$index]}"
  weight="${weights[$index]}"
  total=$((total + weight))
  phase_path="$run_dir/phase_test.py"
  cp "$TEST_DIR/$file" "$phase_path"
  output_path="$LOG_DIR/phase_$((index + 1)).log"
  if (
    cd "$APP_DIR"
    PYTHONHASHSEED=0 PYTHONPATH="$APP_DIR/components/vine:$APP_DIR/components/kombu:$APP_DIR/components/billiard:$APP_DIR:$run_dir" \
      python -m pytest -q "$phase_path"
  ) >"$output_path" 2>&1; then
    passed=1
    earned=$((earned + weight))
    status=PASS
  else
    passed=0
    status=FAIL
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$weight" "$passed" "$status" >> "$results_file"
  printf 'Phase %02d %-24s %s (weight %s)\n' "$((index + 1))" "$name" "$status" "$weight"
  rm -f "$phase_path"
done

python - "$results_file" "$earned" "$total" "$LOG_DIR" <<'PY'
import json
import pathlib
import sys

results_path, earned, total, log_dir = sys.argv[1:]
earned, total = int(earned), int(total)
phases = []
for line in pathlib.Path(results_path).read_text().splitlines():
    name, weight, passed, status = line.split('\t')
    phases.append({
        'name': name,
        'weight': int(weight),
        'passed': bool(int(passed)),
        'status': status,
    })
reward = earned / total
payload = {
    'reward': reward,
    'earned_weight': earned,
    'total_weight': total,
    'functional_weight': sum(item['weight'] for item in phases[:18]),
    'regression_weight': sum(item['weight'] for item in phases[18:]),
    'phases': phases,
}
directory = pathlib.Path(log_dir)
(directory / 'reward.txt').write_text(f'{reward:.12f}\n')
(directory / 'reward.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(f'Reward: {reward:.12f} ({earned}/{total})')
PY

exit 0
