#!/bin/bash
set -u

repo_dir="${REPO_DIR:-/app}"
tests_dir="${TESTS_DIR:-/tests}"
log_dir="${LOG_DIR:-/logs/verifier}"
mkdir -p "$log_dir"

weights=(2 2 2 4 3 3 3 3 2 3 2 3 4 4 4 3 2 2)
names=(
  ttl_contract memory_ttl file_ttl_recovery subject_markers
  consumer_pause priority_groups pinned_client overflow_priority
  direct_batch direct_multi_get asset_version_gate preferred_leadership
  ingest_budget replication_envelope distributed_trace bounded_expiry_ledger
  public_storage_regression local_extensions_regression
)
passed=()
score=0
total=0

cp "$tests_dir/roadmap_helpers_test.go" "$repo_dir/server/edgebench_roadmap_helpers_test.go"
cleanup() {
  rm -f "$repo_dir/server/edgebench_roadmap_helpers_test.go" "$repo_dir/server/edgebench_phase_test.go"
}
trap cleanup EXIT

for index in $(seq 1 18); do
  phase=$(printf '%02d' "$index")
  source_file=$(find "$tests_dir" -maxdepth 1 -type f -name "test_${phase}_*_test.go" -print -quit)
  weight=${weights[$((index-1))]}
  total=$((total + weight))
  cp "$source_file" "$repo_dir/server/edgebench_phase_test.go"
  phase_log="$log_dir/phase_${phase}.log"
  if (cd "$repo_dir" && go test ./server -run "^TestRoadmapPhase${phase}$" -count=1 -timeout 90s) >"$phase_log" 2>&1; then
    passed+=(true)
    score=$((score + weight))
    printf 'PASS phase %s %-28s weight=%d\n' "$phase" "${names[$((index-1))]}" "$weight"
  else
    passed+=(false)
    printf 'FAIL phase %s %-28s weight=%d\n' "$phase" "${names[$((index-1))]}" "$weight"
    tail -n 12 "$phase_log"
  fi
  rm -f "$repo_dir/server/edgebench_phase_test.go"
done

reward=$(python3 -c 'import sys; print(f"{int(sys.argv[1])/int(sys.argv[2]):.12f}")' "$score" "$total")
printf '%s\n' "$reward" > "$log_dir/reward.txt"

passed_csv=$(IFS=,; echo "${passed[*]}")
python3 - "$log_dir/reward.json" "$reward" "$score" "$total" "$passed_csv" <<'PY'
import json, sys
path, reward, score, total, passed = sys.argv[1:]
names = [
    "ttl_contract", "memory_ttl", "file_ttl_recovery", "subject_markers",
    "consumer_pause", "priority_groups", "pinned_client", "overflow_priority",
    "direct_batch", "direct_multi_get", "asset_version_gate", "preferred_leadership",
    "ingest_budget", "replication_envelope", "distributed_trace", "bounded_expiry_ledger",
    "public_storage_regression", "local_extensions_regression",
]
weights = [2,2,2,4,3,3,3,3,2,3,2,3,4,4,4,3,2,2]
flags = [item == "true" for item in passed.split(",")]
payload = {
    "reward": float(reward), "score": int(score), "total": int(total),
    "phases": [
        {"phase": i + 1, "name": name, "weight": weights[i], "passed": flags[i]}
        for i, name in enumerate(names)
    ],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY

printf 'reward=%s (%d/%d)\n' "$reward" "$score" "$total"
exit 0
