#!/usr/bin/env bash
set -u
mkdir -p /logs/verifier
mkdir -p /testbed/Packs/AggregatedScripts/Scripts/IndicatorEnrichment
while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  mkdir -p "/testbed/$(dirname "$rel")"
  cp "/tests/test_files/$rel" "/testbed/$rel"
done < /tests/test_files_manifest.txt
fallback_conftest=/tests/test_files/Packs/AggregatedScripts/Scripts/IndicatorEnrichment/conftest.py
if [ -f "$fallback_conftest" ]; then
  cp "$fallback_conftest" /testbed/Packs/AggregatedScripts/Scripts/IndicatorEnrichment/conftest.py
fi
mapfile -t targets < /tests/test_targets.txt
set +e
/tests/test_run.sh "${targets[@]}" > /logs/verifier/test_output.txt 2>&1
set -e
python3 /tests/_compute_reward.py
