# CANARY: verifier // ml/fused_ce_supervision_carrier // v003 // do not remove
"""Semantic verifier for the fused cross-entropy supervision-carrier task.

Deliverables (harbor stages them at their container paths):

    /app/out/losses.jsonl  one {"id","loss"} per pinned record, 480 lines
    /app/out/grads.jsonl   one {"id","dlogits"} per pinned record, 480 lines
    /app/out/report.json   JSON array of 480 {"id","denominator","offending"}

Reference sourcing.  The verifier image holds no task inputs, so the pinned
corpus is *regenerated in-suite* by the build-time generator the task image
deletes after staging (tests/make_vectors.py, seed 20260916), the three
count_policy carriers are reconstructed from the regenerated eval recordings
the way the instruction demands (fit every member of the carrier rule family
against every recorded eval loss: the winner reproduces all of them, every
rival misses by more than 1e-6 relative), and the contract (C2/C3 terms, C4
divisor, C5 empty carrier) is evaluated in float64 for every pinned record
together with its analytic gradient.  tests/gold_losses.json carries the
Judge-approved Gold's 480 losses, tests/gold_report_sample.json 40 of its
report rows (every empty-carrier fused C4 row plus a spread of the other
shapes) and tests/gold_grads_sample.json the leading rows of two of its
gradient arrays;
those are the frozen anchors the delivered numbers are pinned to.

Tests
-----
test_losses_complete -- 480 lines, {"id","loss"} only, the pinned id set
    exactly once each, JSON numbers, every value finite.
test_grads_complete -- 480 lines, {"id","dlogits"} only, exact id set, every
    array finite and shaped to its own record's (N,V).
test_report_complete -- one object per pinned record, field names and types as
    declared (integer denominator, in-range ascending integer positions,
    path in {eager,fused} once each, clause drawn from the contract's ids).
test_reference_corpus_matches_gold -- the regenerated corpus plus the
    reconstructed carriers reproduce the Gold's 480 losses to 1e-9 relative,
    i.e. the reference really is the build-time corpus.
test_losses_match_contract -- |delivered - oracle| <= 1e-6*max(|oracle|,1e-12)
    for every record, oracle recomputed here from the contract (R1).
test_losses_match_gold -- value anchor: |delivered - Gold| within the same
    bound for all 480 records (R1).
test_grads_match_contract -- max-abs deviation from the analytic contract
    gradient <= 1e-5 for every entry of every record (R2).
test_grads_support_pattern -- rows outside the carrier exactly 0.0, rows at
    ignored positions inside the carrier non-zero whenever label_smoothing is
    positive (R2 pattern requirement).
test_grads_match_gold -- value anchor: the leading rows of two Gold gradient
    arrays reproduced to 1e-5 absolute (R2).
test_grad_rows_sum_to_zero -- arithmetic invariant of the contract's own
    gradient on the delivered numbers: every in-carrier row is a probability
    vector minus its uniform part, so it sums to zero (softmax gauge).
test_report_denominator -- every denominator is the contract divisor for the
    record's reduction and carrier (1 for sum, |L| for mean, 0 for the empty
    carrier under mean) (R3).
test_report_offending -- per record, the offending set of (path, earliest
    contradicted clause, sorted symmetric difference of supports) equals this
    suite's own adjudication of the shipped eager and fused paths, including
    the empty lists where an implementation coincides numerically despite a
    different support (R3).
test_report_sample_matches_gold -- value anchor: 40 frozen Gold report rows
    reproduced field for field (R3).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("VERIFIER_ARTIFACT_DIR", "/app/out"))
LOSSES_PATH = ART / "losses.jsonl"
GRADS_PATH = ART / "grads.jsonl"
REPORT_PATH = ART / "report.json"

N_PINNED = 480
IGNORE_INDEX = -100
REL_TOL = 1e-6          # contract relative tolerance (R1, R3)
GRAD_ATOL = 1e-5        # declared absolute gradient tolerance (R2)
GOLD_GRAD_ATOL = 1e-5   # anchor comparison against the Gold's own arrays
REGEN_RTOL = 1e-9       # regenerated corpus vs the Gold's losses
ROWSUM_ATOL = 1e-4      # row-gauge noise, well below the 1.5x perturbation
CLAUSES = ("C1", "C2", "C3", "C4", "C5")
RULE_NAMES = ("all", "supervised", "span", "prefix", "pre", "suffix")
EXPECTED_POLICY_RULES = {"cp_1": "span", "cp_2": "prefix", "cp_3": "pre"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_json_reference(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


GOLD_LOSSES = _load_json_reference("gold_losses.json")
GOLD_REPORT_SAMPLE = _load_json_reference("gold_report_sample.json")
GOLD_GRADS_SAMPLE = _load_json_reference("gold_grads_sample.json")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value, where):
    if not _is_number(value) or not math.isfinite(float(value)):
        pytest.fail("%s must be a finite JSON number, got %r" % (where, value))
    return float(value)


def _load_lines(path):
    if not path.is_file():
        pytest.fail("declared artifact is missing: %s" % (path,))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - unreadable artifact
        pytest.fail("cannot read %s: %r" % (path, exc))
    return [line for line in text.splitlines() if line.strip()]


def _load_object(path, label):
    if not path.is_file():
        pytest.fail("declared artifact is missing: %s (%s)" % (path, label))
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        pytest.fail("%s is not valid JSON: %r" % (label, exc))


# ---------------------------------------------------------------------------
# the contract, evaluated here in float64
# ---------------------------------------------------------------------------
def _carriers(labels):
    """Every member of the carrier rule family the recordings are fitted to."""
    labels = [int(y) for y in labels]
    n = len(labels)
    supervised = [i for i, y in enumerate(labels) if y != IGNORE_INDEX]
    first_ignored = next((i for i, y in enumerate(labels) if y == IGNORE_INDEX), n)
    return {
        "all": list(range(n)),
        "supervised": supervised,
        "span": list(range(min(supervised), max(supervised) + 1)) if supervised else [],
        "prefix": list(range(max(supervised) + 1)) if supervised else [],
        "pre": list(range(first_ignored)),
        "suffix": list(range(min(supervised), n)) if supervised else [],
    }


def _log_softmax(logits):
    shift = np.max(logits, axis=-1, keepdims=True)
    z = logits - shift
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def _terms(record):
    logits = np.asarray(record["logits"], dtype=np.float64)
    labels = np.asarray(record["labels"], dtype=np.int64)
    eps = float(record["label_smoothing"])
    logp = _log_softmax(logits)
    # C3 at every position, C2 additionally where the position is supervised.
    terms = -eps * np.mean(logp, axis=-1)
    supervised = np.nonzero(labels != IGNORE_INDEX)[0]
    if supervised.size:
        terms[supervised] -= (1.0 - eps) * logp[supervised, labels[supervised]]
    return terms


def _contract_loss(record, carrier):
    if not carrier:
        return 0.0
    total = float(np.sum(_terms(record)[carrier]))
    if record["reduction"] == "sum":
        return total
    return total / len(carrier)


def _contract_gradient(record, carrier):
    logits = np.asarray(record["logits"], dtype=np.float64)
    labels = np.asarray(record["labels"], dtype=np.int64)
    n, v = logits.shape
    gradient = np.zeros((n, v), dtype=np.float64)
    if not carrier:
        return gradient
    eps = float(record["label_smoothing"])
    probabilities = np.exp(_log_softmax(logits))
    block = eps * (probabilities - 1.0 / v)
    supervised = np.nonzero(labels != IGNORE_INDEX)[0]
    block[supervised] += (1.0 - eps) * probabilities[supervised]
    block[supervised, labels[supervised]] -= 1.0 - eps
    rows = block[carrier]
    if record["reduction"] == "mean":
        rows = rows / len(carrier)
    gradient[carrier] = rows
    return gradient


# ---------------------------------------------------------------------------
# the two shipped paths, replayed as shipped
# ---------------------------------------------------------------------------
def _eager_loss(record):
    """loss/eager_ce.py: a term at *every* position, mean divides by N."""
    terms = _terms(record)
    total = float(np.sum(terms))
    if record["reduction"] == "sum":
        return total
    return total / len(record["labels"])


def _fused_loss(record):
    """kernels/fused_ce.cpp: a term only at non-ignored positions, the mean
    divides by their count, so an empty support evaluates 0/0 and is NaN.

    Arithmetic follows the kernel operation for operation in Python doubles.
    """
    logits = np.asarray(record["logits"], dtype=np.float64)
    labels = [int(y) for y in record["labels"]]
    eps = float(record["label_smoothing"])
    n, v = logits.shape
    total = 0.0
    count = 0
    for i in range(n):
        y = labels[i]
        if y < 0:  # ignore_index
            continue
        row = logits[i]
        m = float(row[0])
        for k in range(1, v):
            if float(row[k]) > m:
                m = float(row[k])
        s = 0.0
        for k in range(v):
            s += math.exp(float(row[k]) - m)
        lse = m + math.log(s)
        term = 0.0
        for k in range(v):
            term -= eps * (float(row[k]) - lse) / v
        term -= (1.0 - eps) * (float(row[y]) - lse)
        total += term
        count += 1
    if record["reduction"] == "sum":
        return total
    if count == 0:
        return float("nan")
    return total / count


def _disagrees(actual, expected):
    """The task's own disagreement rule (R1/R3)."""
    if not math.isfinite(actual):
        return True
    return abs(actual - expected) > REL_TOL * max(abs(expected), 1e-12)


def _affected(support, carrier):
    return sorted(set(support).symmetric_difference(carrier))


def _adjudicate(record, carrier, expected):
    n = len(record["labels"])
    supervised = [i for i, y in enumerate(record["labels"]) if int(y) != IGNORE_INDEX]
    entries = []
    for path, actual, support in (
        ("eager", _eager_loss(record), list(range(n))),
        ("fused", _fused_loss(record), supervised),
    ):
        if not _disagrees(actual, expected):
            continue
        affected = _affected(support, carrier)
        # C1 is the carrier clause: it is contradicted exactly when the path
        # forms terms somewhere other than the carrier.  With the supports
        # equal the divisor is equal too, so the only remaining way for these
        # two paths to miss is the empty-carrier quotient C4 forbids.
        entries.append({
            "path": path,
            "clause": "C1" if affected else "C4",
            "affected_positions": affected,
        })
    return entries


# ---------------------------------------------------------------------------
# reference: regenerate the pinned corpus and recover the carriers
# ---------------------------------------------------------------------------
def _recover_policy_rules(evaluation, recorded):
    rules = {}
    for policy in sorted({rec["count_policy"] for rec in evaluation}):
        members = [rec for rec in evaluation if rec["count_policy"] == policy]
        scores = {}
        for name in RULE_NAMES:
            worst = 0.0
            for rec in members:
                expected = float(recorded[rec["id"]])
                gap = abs(_contract_loss(rec, _carriers(rec["labels"])[name]) - expected)
                if expected == 0.0:
                    relative = 0.0 if gap == 0.0 else math.inf
                else:
                    relative = gap / abs(expected)
                worst = max(worst, relative)
            scores[name] = worst
        winner = min(scores, key=scores.get)
        assert scores[winner] <= REGEN_RTOL, (policy, scores)
        rivals = [value for name, value in scores.items() if name != winner]
        assert min(rivals) > REL_TOL, (policy, scores)
        rules[policy] = winner
    return rules


_REFERENCE = None


def _build_reference():
    work = Path(tempfile.mkdtemp(prefix="fused_ce_vectors_"))
    spec = importlib.util.spec_from_file_location("make_vectors_reference",
                                                  HERE / "make_vectors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = sys.argv[:]
    sys.argv = [str(HERE / "make_vectors.py"), str(work)]
    try:
        module.main()
    finally:
        sys.argv = argv

    pinned = [json.loads(line) for line in
              (work / "loss_vectors.jsonl").read_text().splitlines() if line.strip()]
    evaluation = [json.loads(line) for line in
                  (work / "eval_records.jsonl").read_text().splitlines() if line.strip()]
    recorded = json.loads((work / "recorded_metrics.json").read_text())
    assert len(pinned) == N_PINNED

    rules = _recover_policy_rules(evaluation, recorded)
    assert rules == EXPECTED_POLICY_RULES, rules

    order = []
    records = {}
    for rec in pinned:
        carrier = _carriers(rec["labels"])[rules[rec["count_policy"]]]
        loss = _contract_loss(rec, carrier)
        gradient = _contract_gradient(rec, carrier)
        assert math.isfinite(loss) and bool(np.all(np.isfinite(gradient)))
        records[rec["id"]] = {
            "n": len(rec["labels"]),
            "v": len(rec["logits"][0]),
            "carrier": carrier,
            "ignored_in_carrier": [i for i in carrier
                                   if int(rec["labels"][i]) == IGNORE_INDEX],
            "smoothing": float(rec["label_smoothing"]),
            "loss": loss,
            "gradient": gradient,
            "denominator": 1 if rec["reduction"] == "sum" else len(carrier),
            "offending": _adjudicate(rec, carrier, loss),
        }
        order.append(rec["id"])
    return {"order": order, "records": records, "rules": rules}


def reference():
    global _REFERENCE
    if _REFERENCE is None:
        _REFERENCE = _build_reference()
    return _REFERENCE


# ---------------------------------------------------------------------------
# artifact parsers (schema + completeness)
# ---------------------------------------------------------------------------
def parse_losses(ref):
    lines = _load_lines(LOSSES_PATH)
    assert len(lines) == N_PINNED, (
        "losses.jsonl must hold exactly %d lines, found %d" % (N_PINNED, len(lines)))
    table = {}
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError as exc:
            pytest.fail("losses.jsonl line %d is not JSON: %r (%s)"
                        % (index + 1, line[:120], exc))
        assert isinstance(row, dict), "losses.jsonl line %d is not an object" % (index + 1)
        assert set(row) == {"id", "loss"}, (
            "losses.jsonl line %d keys %r" % (index + 1, sorted(row)))
        ident = row["id"]
        assert isinstance(ident, str), "losses.jsonl line %d id is not a string" % (index + 1)
        assert ident not in table, "losses.jsonl repeats id %r" % ident
        table[ident] = _as_number(row["loss"], "losses.jsonl[%s].loss" % ident)
    missing = sorted(set(ref["order"]) - set(table))
    extra = sorted(set(table) - set(ref["order"]))
    assert not missing and not extra, (
        "losses.jsonl id set must be the 480 pinned ids; missing=%s extra=%s"
        % (missing[:8], extra[:8]))
    return table


def parse_grads(ref):
    lines = _load_lines(GRADS_PATH)
    assert len(lines) == N_PINNED, (
        "grads.jsonl must hold exactly %d lines, found %d" % (N_PINNED, len(lines)))
    table = {}
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError as exc:
            pytest.fail("grads.jsonl line %d is not JSON: %r (%s)"
                        % (index + 1, line[:120], exc))
        assert isinstance(row, dict), "grads.jsonl line %d is not an object" % (index + 1)
        assert set(row) == {"id", "dlogits"}, (
            "grads.jsonl line %d keys %r" % (index + 1, sorted(row)))
        ident = row["id"]
        assert isinstance(ident, str), "grads.jsonl line %d id is not a string" % (index + 1)
        assert ident not in table, "grads.jsonl repeats id %r" % ident
        info = ref["records"].get(ident)
        if info is None:
            pytest.fail("grads.jsonl holds %r, which is not a pinned record" % ident)
        block = row["dlogits"]
        assert isinstance(block, list), "grads.jsonl[%s].dlogits is not an array" % ident
        assert len(block) == info["n"], (
            "grads.jsonl[%s].dlogits must have %d rows, found %d"
            % (ident, info["n"], len(block)))
        for i, block_row in enumerate(block):
            assert isinstance(block_row, list), (
                "grads.jsonl[%s].dlogits[%d] is not an array" % (ident, i))
            assert len(block_row) == info["v"], (
                "grads.jsonl[%s].dlogits[%d] must have %d entries, found %d"
                % (ident, i, info["v"], len(block_row)))
            for j, value in enumerate(block_row):
                _as_number(value, "grads.jsonl[%s].dlogits[%d][%d]" % (ident, i, j))
        table[ident] = block
    missing = sorted(set(ref["order"]) - set(table))
    extra = sorted(set(table) - set(ref["order"]))
    assert not missing and not extra, (
        "grads.jsonl id set must be the 480 pinned ids; missing=%s extra=%s"
        % (missing[:8], extra[:8]))
    return table


def parse_report(ref):
    report = _load_object(REPORT_PATH, "report.json")
    assert isinstance(report, list), "report.json must be a JSON array"
    assert len(report) == N_PINNED, (
        "report.json must hold exactly %d objects, found %d" % (N_PINNED, len(report)))
    table = {}
    for index, row in enumerate(report):
        assert isinstance(row, dict), "report.json[%d] is not an object" % index
        assert set(row) == {"id", "denominator", "offending"}, (
            "report.json[%d] keys %r" % (index, sorted(row)))
        ident = row["id"]
        assert isinstance(ident, str), "report.json[%d].id is not a string" % index
        assert ident not in table, "report.json repeats id %r" % ident
        info = ref["records"].get(ident)
        if info is None:
            pytest.fail("report.json holds %r, which is not a pinned record" % ident)
        denominator = row["denominator"]
        assert isinstance(denominator, int) and not isinstance(denominator, bool), (
            "report.json[%s].denominator must be a JSON integer, got %r"
            % (ident, denominator))
        offending = row["offending"]
        assert isinstance(offending, list), (
            "report.json[%s].offending must be an array" % ident)
        paths = []
        for entry in offending:
            assert isinstance(entry, dict), (
                "report.json[%s].offending entries must be objects" % ident)
            assert set(entry) == {"path", "clause", "affected_positions"}, (
                "report.json[%s].offending entry keys %r" % (ident, sorted(entry)))
            assert entry["path"] in ("eager", "fused"), (
                "report.json[%s].path must be eager or fused, got %r"
                % (ident, entry["path"]))
            assert entry["clause"] in CLAUSES, (
                "report.json[%s].clause %r is not a contract clause id"
                % (ident, entry["clause"]))
            affected = entry["affected_positions"]
            assert isinstance(affected, list), (
                "report.json[%s].affected_positions must be an array" % ident)
            assert all(isinstance(x, int) and not isinstance(x, bool) for x in affected), (
                "report.json[%s].affected_positions must hold integers, got %r"
                % (ident, affected))
            assert affected == sorted(affected) and len(set(affected)) == len(affected), (
                "report.json[%s].affected_positions must be sorted and unique: %r"
                % (ident, affected))
            assert all(0 <= x < info["n"] for x in affected), (
                "report.json[%s].affected_positions out of range: %r"
                % (ident, affected))
            paths.append(entry["path"])
        assert len(set(paths)) == len(paths), (
            "report.json[%s] lists an implementation twice" % ident)
        table[ident] = row
    missing = sorted(set(ref["order"]) - set(table))
    extra = sorted(set(table) - set(ref["order"]))
    assert not missing and not extra, (
        "report.json id set must be the 480 pinned ids; missing=%s extra=%s"
        % (missing[:8], extra[:8]))
    return table


def _offending_set(entries):
    return {(entry["path"], entry["clause"], tuple(entry["affected_positions"]))
            for entry in entries}


# ---------------------------------------------------------------------------
# R4 - completeness and well-formedness
# ---------------------------------------------------------------------------
def test_losses_complete():
    ref = reference()
    table = parse_losses(ref)
    assert len(table) == N_PINNED
    assert all(math.isfinite(value) for value in table.values())


def test_grads_complete():
    ref = reference()
    table = parse_grads(ref)
    assert len(table) == N_PINNED


def test_report_complete():
    ref = reference()
    table = parse_report(ref)
    assert len(table) == N_PINNED


# ---------------------------------------------------------------------------
# reference integrity: the regenerated corpus is the build-time corpus
# ---------------------------------------------------------------------------
def test_reference_corpus_matches_gold():
    ref = reference()
    assert ref["rules"] == EXPECTED_POLICY_RULES, (
        "recovered carriers %r" % (ref["rules"],))
    assert len(GOLD_LOSSES) == N_PINNED
    bad = []
    for ident, gold in GOLD_LOSSES.items():
        want = ref["records"][ident]["loss"]
        if abs(want - gold) > REGEN_RTOL * max(abs(gold), 1e-12):
            bad.append((ident, want, gold))
    assert not bad, (
        "the regenerated corpus does not reproduce the pinned recordings: %s"
        % (bad[:5],))


# ---------------------------------------------------------------------------
# R1 - contract losses
# ---------------------------------------------------------------------------
def test_losses_match_contract():
    ref = reference()
    delivered = parse_losses(ref)
    bad = []
    for ident in ref["order"]:
        want = ref["records"][ident]["loss"]
        got = delivered[ident]
        if abs(got - want) > REL_TOL * max(abs(want), 1e-12):
            bad.append((ident, got, want, abs(got - want)))
    assert not bad, (
        "%d of 480 losses disagree with the contract in float64, e.g. %s"
        % (len(bad), bad[:5]))


def test_losses_match_gold():
    ref = reference()
    delivered = parse_losses(ref)
    bad = []
    for ident, gold in GOLD_LOSSES.items():
        got = delivered[ident]
        if abs(got - gold) > REL_TOL * max(abs(gold), 1e-12):
            bad.append((ident, got, gold))
    assert not bad, (
        "%d of 480 losses disagree with the frozen Gold values, e.g. %s"
        % (len(bad), bad[:5]))


# ---------------------------------------------------------------------------
# R2 - contract gradients
# ---------------------------------------------------------------------------
def test_grads_match_contract():
    ref = reference()
    delivered = parse_grads(ref)
    worst = 0.0
    worst_id = None
    bad = []
    for ident in ref["order"]:
        want = ref["records"][ident]["gradient"]
        got = np.asarray(delivered[ident], dtype=np.float64)
        deviation = float(np.max(np.abs(got - want)))
        if deviation > worst:
            worst, worst_id = deviation, ident
        if deviation > GRAD_ATOL:
            bad.append((ident, deviation))
    assert not bad, (
        "%d gradients exceed %g absolute error, worst %s at %s"
        % (len(bad), GRAD_ATOL, worst, worst_id))


def test_grads_support_pattern():
    ref = reference()
    delivered = parse_grads(ref)
    nonzero_violations = []
    zero_violations = []
    for ident in ref["order"]:
        info = ref["records"][ident]
        got = np.asarray(delivered[ident], dtype=np.float64)
        carrier = set(info["carrier"])
        outside = [i for i in range(info["n"]) if i not in carrier]
        if outside:
            block = got[outside]
            if not bool(np.all(block == 0.0)):
                zero_violations.append((ident, outside[:4]))
        if info["smoothing"] > 0:
            for i in info["ignored_in_carrier"]:
                if not bool(np.any(got[i] != 0.0)):
                    nonzero_violations.append((ident, i))
    assert not zero_violations, (
        "rows outside the carrier must be exactly zero: %s" % (zero_violations[:5],))
    assert not nonzero_violations, (
        "ignored positions inside the carrier carry a C3 term, so their rows "
        "must be non-zero under positive label smoothing: %s"
        % (nonzero_violations[:5],))


def test_grads_match_gold():
    ref = reference()
    delivered = parse_grads(ref)
    bad = []
    for ident, gold in GOLD_GRADS_SAMPLE.items():
        # The frozen anchor carries the leading rows of the array; the full
        # (N, V) shape of every record is pinned by parse_grads above.
        want = np.asarray(gold, dtype=np.float64)
        got = np.asarray(delivered[ident], dtype=np.float64)
        assert got.shape[1] == want.shape[1] and got.shape[0] >= want.shape[0], (
            "grads.jsonl[%s] shape %r cannot cover the anchor %r"
            % (ident, got.shape, want.shape))
        deviation = float(np.max(np.abs(got[: want.shape[0]] - want)))
        if deviation > GOLD_GRAD_ATOL:
            bad.append((ident, deviation))
    assert not bad, (
        "delivered gradients disagree with the frozen Gold arrays: %s" % (bad,))


def test_grad_rows_sum_to_zero():
    ref = reference()
    delivered = parse_grads(ref)
    worst = 0.0
    worst_id = None
    for ident in ref["order"]:
        info = ref["records"][ident]
        if not info["carrier"]:
            continue
        got = np.asarray(delivered[ident], dtype=np.float64)
        row_sums = np.abs(got[info["carrier"]].sum(axis=1))
        value = float(np.max(row_sums))
        if value > worst:
            worst, worst_id = value, ident
    assert worst <= ROWSUM_ATOL, (
        "in-carrier gradient rows must be gauge-free (sum to zero); worst |row "
        "sum| = %g at %s" % (worst, worst_id))


# ---------------------------------------------------------------------------
# R3 - adjudication report
# ---------------------------------------------------------------------------
def test_report_denominator():
    ref = reference()
    report = parse_report(ref)
    bad = []
    for ident in ref["order"]:
        want = ref["records"][ident]["denominator"]
        got = report[ident]["denominator"]
        if got != want:
            bad.append((ident, got, want))
    assert not bad, (
        "%d denominators are not the contract's divisor, e.g. %s" % (len(bad), bad[:5]))


def test_report_offending():
    ref = reference()
    report = parse_report(ref)
    bad = []
    for ident in ref["order"]:
        want = _offending_set(ref["records"][ident]["offending"])
        got = _offending_set(report[ident]["offending"])
        if got != want:
            bad.append((ident, sorted(got), sorted(want)))
    assert not bad, (
        "%d records adjudicate the shipped paths differently, e.g. %s"
        % (len(bad), bad[:3]))


def test_report_sample_matches_gold():
    ref = reference()
    report = parse_report(ref)
    assert any(entry["clause"] == "C4"
               for row in GOLD_REPORT_SAMPLE for entry in row["offending"])
    bad = []
    for row in GOLD_REPORT_SAMPLE:
        ident = row["id"]
        got = report[ident]
        if got["denominator"] != row["denominator"]:
            bad.append((ident, "denominator", got["denominator"], row["denominator"]))
        want = _offending_set(row["offending"])
        mine = _offending_set(got["offending"])
        if mine != want:
            bad.append((ident, "offending", sorted(mine), sorted(want)))
    assert not bad, (
        "report rows disagree with the frozen Gold adjudication: %s" % (bad[:5],))
