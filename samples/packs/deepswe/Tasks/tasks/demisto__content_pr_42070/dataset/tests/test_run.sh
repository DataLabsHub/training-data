#!/usr/bin/env bash
set -u
cd /testbed || exit 2
[ -f /testbed/.venv/bin/activate ] && source /testbed/.venv/bin/activate
export PYTHONPATH="/testbed/Packs/AggregatedScripts/Scripts/IndicatorEnrichment:/testbed/Packs/Base/Scripts/CommonServerPython:/testbed/Packs/ApiModules/Scripts/AggregatedCommandApiModule:/testbed/Packs/ApiModules/Scripts/DemistoClassApiModule:/testbed/Tests/demistomock:/testbed/.local:${PYTHONPATH:-}"
if [ "$#" -eq 0 ]; then echo OMNIGRIL_EXIT_CODE=2; exit 2; fi
plugin=/tests/scripts/omnigril_pytest_result_plugin.py
export PYTHONPATH="/tests/scripts:$PYTHONPATH"
set +e
overall_rc=0
for target in "$@"; do
  results=$(mktemp)
  OMNIGRIL_PYTEST_RESULTS="$results" python -m pytest -q -p omnigril_pytest_result_plugin "$target"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    overall_rc=$rc
  fi
  if [ -f "$results" ]; then
    sed -n 's#^RESULT::#RESULT::#p' "$results"
  fi
  rm -f "$results"
done
set -e
echo "OMNIGRIL_EXIT_CODE=$overall_rc"
exit "$overall_rc"
