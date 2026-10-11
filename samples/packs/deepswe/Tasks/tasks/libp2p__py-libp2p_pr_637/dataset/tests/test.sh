#!/usr/bin/env bash
set -euo pipefail

cd /testbed
if ! mkdir -p /logs/verifier 2>/dev/null; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "cannot create /logs/verifier and sudo is unavailable" >&2
    exit 2
  fi
  sudo mkdir -p /logs/verifier
  sudo chown "$(id -u):$(id -g)" /logs /logs/verifier
fi
mkdir -p /tmp/run
if [ -d /tests/scripts ]; then
  cp -a /tests/scripts/. /tmp/run/
fi
if [ -f /tmp/run/activate.sh ]; then
  # shellcheck disable=SC1091
  source /tmp/run/activate.sh
fi

if [ -s /tests/test.patch ]; then
  git -C /testbed apply --binary --whitespace=nowarn /tests/test.patch
fi

while IFS= read -r relpath; do
  [ -n "$relpath" ] || continue
  mkdir -p "$(dirname "$relpath")"
  cp "/tests/test_files/$relpath" "$relpath"
done < /tests/test_files_manifest.txt

mapfile -t TARGETS < <(cat /tests/fail_to_pass.txt /tests/pass_to_pass.txt | awk 'NF && !seen[$0]++')
if [ "${#TARGETS[@]}" -eq 0 ]; then
  mapfile -t TARGETS < <(awk 'NF && !seen[$0]++' /tests/test_targets.txt)
fi

set +e
bash /tests/test_run.sh "${TARGETS[@]}" > /logs/verifier/test_output.txt 2>&1
runner_rc=$?
set -e
python3 /tests/_compute_reward.py \
  /logs/verifier/test_output.txt /logs/verifier/reward.json "$runner_rc"
cat /logs/verifier/reward.json
