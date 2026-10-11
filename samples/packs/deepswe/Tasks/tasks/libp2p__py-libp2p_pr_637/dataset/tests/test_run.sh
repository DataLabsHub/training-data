#!/usr/bin/env bash
set -u

cd /testbed || exit 2

if [ -f /testbed/.venv/bin/activate ]; then
  source /testbed/.venv/bin/activate
fi

export PYTHONPATH="/tmp/run:${PYTHONPATH:-}"

plugin=/tmp/run/omnigril_pytest_result_plugin.py
results=/tmp/run/omnigril_pytest_results.txt
: > "$results"

cat > "$plugin" <<'PY'
import os
from collections import OrderedDict

RESULTS = OrderedDict()
PRIORITY = {
    "PASSED": 1,
    "SKIPPED": 2,
    "FAILED": 3,
    "ERROR": 4,
}


def _record(nodeid, status):
    previous = RESULTS.get(nodeid)
    if previous is None or PRIORITY[status] >= PRIORITY[previous]:
        RESULTS[nodeid] = status


def pytest_runtest_logreport(report):
    nodeid = report.nodeid
    if report.when == "setup":
        if report.failed:
            _record(nodeid, "ERROR")
        elif report.skipped:
            _record(nodeid, "SKIPPED")
        return
    if report.when == "call":
        if report.passed:
            _record(nodeid, "PASSED")
        elif report.failed:
            _record(nodeid, "FAILED")
        elif report.skipped:
            _record(nodeid, "SKIPPED")
        return
    if report.when == "teardown" and report.failed:
        _record(nodeid, "ERROR")


def pytest_collectreport(report):
    if report.failed:
        _record(str(report.nodeid), "ERROR")


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("OMNIGRIL_PYTEST_RESULTS")
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        for nodeid, status in RESULTS.items():
            handle.write(f"RESULT::{nodeid}::{status}\n")
PY

export OMNIGRIL_PYTEST_RESULTS="$results"

if [ "$#" -eq 0 ]; then
  echo "OMNIGRIL_EXIT_CODE=2"
  exit 2
fi

set +e
python -m pytest -q -o addopts='' -p omnigril_pytest_result_plugin "$@"
rc=$?
set -e

if [ -f "$results" ]; then
  cat "$results"
fi

echo "OMNIGRIL_EXIT_CODE=$rc"
exit "$rc"
