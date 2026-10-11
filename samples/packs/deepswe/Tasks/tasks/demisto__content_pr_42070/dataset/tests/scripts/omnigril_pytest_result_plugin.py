import os
from collections import OrderedDict

RESULTS = OrderedDict()
PRIORITY = {"PASSED": 1, "SKIPPED": 2, "FAILED": 3, "ERROR": 4}


def record(nodeid, status):
    old = RESULTS.get(nodeid)
    if old is None or PRIORITY[status] >= PRIORITY[old]:
        RESULTS[nodeid] = status


def pytest_runtest_logreport(report):
    if report.when == "setup":
        if report.failed:
            record(report.nodeid, "ERROR")
        elif report.skipped:
            record(report.nodeid, "SKIPPED")
    elif report.when == "call":
        if getattr(report, "wasxfail", False) or report.skipped:
            record(report.nodeid, "SKIPPED")
        elif report.passed:
            record(report.nodeid, "PASSED")
        elif report.failed:
            record(report.nodeid, "FAILED")
    elif report.when == "teardown" and report.failed:
        record(report.nodeid, "ERROR")


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("OMNIGRIL_PYTEST_RESULTS")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            for nodeid, status in RESULTS.items():
                f.write(f"RESULT::{nodeid}::{status}\n")
