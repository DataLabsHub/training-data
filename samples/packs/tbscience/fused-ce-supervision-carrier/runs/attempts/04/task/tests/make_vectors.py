#!/usr/bin/env python3
"""Build-time generator for the staged vector files.

Deterministic (seed 20260916).  The image deletes this script after the build:
it is the only artifact that carries the policy designations.

Writes, into the directory given as argv[1] (default ./vectors):

    loss_vectors.jsonl     480 pinned records, no reference values
    eval_records.jsonl     the disjoint eval split (N >= 32)
    recorded_metrics.json  {record id: loss}, the losses the pipeline recorded
                           for the eval split under the true carriers

and asserts that those recordings pin every policy's carrier against every
other member of a broad structural rule family, and that neither shipped
implementation's support is any policy's designation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SEED = 20260916
IGNORE_INDEX = -100
SMOOTH_GRID = (0.0, 0.001, 0.004, 0.05, 0.1, 0.2, 0.4)
REDUCTIONS = ("mean", "sum")
V_RANGE = (6, 12)
PINNED_N_RANGE = (6, 40)
EVAL_N_RANGE = (32, 40)
MIN_STRUCTURAL_N = 8
PINNED_PER_CLASS = 20
EVAL_PER_CLASS = 3

# Opaque registry ids, in the order the records reference them.
POLICY_RULES = (("cp_1", "span"), ("cp_2", "prefix"), ("cp_3", "pre"))

CLASSES = (
    "no_ignored",
    "leading",
    "trailing",
    "both",
    "interior",
    "leading_interior",
    "interior_trailing",
    "all_ignored",
)

# Structural rule family used to check that the recordings pin the carriers.
FAMILY = ("all", "supervised", "span", "prefix", "pre", "suffix")

SEPARATION_TOL = 1e-6      # relative; the grader's own tolerance
REPRODUCE_TOL = 1e-9       # relative; float64 round-trip slack


# --------------------------------------------------------------------------
# carrier rules
# --------------------------------------------------------------------------
def supervised_runs(n, ignored):
    runs = []
    i = 0
    while i < n:
        if i in ignored:
            i += 1
            continue
        j = i
        while j < n and j not in ignored:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def class_holds(cls, n, ignored):
    runs = supervised_runs(n, ignored)
    if cls == "no_ignored":
        return not ignored
    if cls == "all_ignored":
        return len(runs) == 0
    if cls == "leading":
        return bool(ignored) and 0 in ignored and len(runs) == 1 and runs[0][0] > 0
    if cls == "trailing":
        return bool(ignored) and (n - 1) in ignored and len(runs) == 1 and runs[0][1] < n - 1
    if cls == "both":
        return (0 in ignored) and (n - 1) in ignored and len(runs) == 1
    if cls == "interior":
        return bool(ignored) and 0 not in ignored and (n - 1) not in ignored and len(runs) == 2
    if cls == "leading_interior":
        return (0 in ignored) and (n - 1) not in ignored and len(runs) == 2
    if cls == "interior_trailing":
        return (0 not in ignored) and (n - 1) in ignored and len(runs) == 2
    raise ValueError(cls)


def sample_ignored(cls, n, rng):
    for _ in range(400):
        if cls == "no_ignored":
            cand = set()
        elif cls == "all_ignored":
            cand = set(range(n))
        elif cls == "leading":
            cand = set(range(int(rng.integers(1, n))))
        elif cls == "trailing":
            b = int(rng.integers(1, n))
            cand = set(range(n - b, n))
        elif cls == "both":
            a = int(rng.integers(1, n - 1))
            b = int(rng.integers(1, n - 1))
            cand = set(range(a)) | set(range(n - b, n))
        elif cls == "interior":
            k = int(rng.integers(1, n - 1))
            cand = set(int(x) for x in rng.choice(np.arange(1, n - 1), size=k, replace=False))
        elif cls == "leading_interior":
            a = int(rng.integers(1, n - 2))
            pool = np.arange(a + 1, n - 1)
            k = int(rng.integers(1, len(pool) + 1))
            cand = set(range(a)) | set(int(x) for x in rng.choice(pool, size=k, replace=False))
        elif cls == "interior_trailing":
            b = int(rng.integers(1, n - 2))
            pool = np.arange(1, b)
            if len(pool) < 1:
                continue
            k = int(rng.integers(1, len(pool) + 1))
            cand = set(int(x) for x in rng.choice(pool, size=k, replace=False)) | set(range(b + 1, n))
        else:
            raise ValueError(cls)
        if class_holds(cls, n, cand):
            return cand
    raise RuntimeError("could not sample class %r at n=%d" % (cls, n))


def carrier_of(rule, n, ignored):
    sup = [i for i in range(n) if i not in ignored]
    if rule == "all":
        return list(range(n))
    if rule == "supervised":
        return sup
    if rule == "span":
        return list(range(min(sup), max(sup) + 1)) if sup else []
    if rule == "prefix":
        return list(range(0, max(sup) + 1)) if sup else []
    if rule == "pre":
        if not ignored:
            return list(range(n))
        return list(range(0, min(ignored)))
    if rule == "suffix":
        return list(range(min(sup), n)) if sup else []
    raise ValueError(rule)


# --------------------------------------------------------------------------
# the contract evaluated in float64
# --------------------------------------------------------------------------
def log_softmax(z):
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return (z - m) - np.log(np.sum(e, axis=-1, keepdims=True))


def per_position_terms(logits, targets, eps):
    logp = log_softmax(logits)
    terms = -float(eps) * np.mean(logp, axis=-1)
    supervised = targets != IGNORE_INDEX
    if np.any(supervised):
        rows = np.nonzero(supervised)[0]
        terms[rows] -= (1.0 - float(eps)) * logp[rows, targets[rows]]
    return terms


def contract_loss(rec, rule):
    ignored = set(i for i, y in enumerate(rec["labels"]) if y == IGNORE_INDEX)
    carrier = carrier_of(rule, len(rec["labels"]), ignored)
    if not carrier:
        return 0.0
    logits = np.asarray(rec["logits"], dtype=np.float64)
    targets = np.asarray(rec["labels"], dtype=np.int64)
    terms = per_position_terms(logits, targets, rec["label_smoothing"])
    total = float(np.sum(terms[carrier]))
    if rec["reduction"] == "sum":
        return total
    return total / len(carrier)


# --------------------------------------------------------------------------
# record construction
# --------------------------------------------------------------------------
def make_record(pid, cls, n, rng):
    v = int(rng.integers(V_RANGE[0], V_RANGE[1] + 1))
    ignored = sample_ignored(cls, n, rng)
    labels = [IGNORE_INDEX if i in ignored else int(rng.integers(0, v)) for i in range(n)]
    logits = rng.normal(0.0, 1.6, size=(n, v))
    for i in range(n):
        k = int(rng.integers(0, v))
        logits[i, k] += float(rng.uniform(0.0, 6.0))
    logits = np.round(logits, 6)
    eps = float(SMOOTH_GRID[int(rng.integers(0, len(SMOOTH_GRID)))])
    reduction = REDUCTIONS[int(rng.integers(0, len(REDUCTIONS)))]
    rec = {
        "logits": [[float(x) for x in row] for row in logits],
        "labels": [int(y) for y in labels],
        "label_smoothing": eps,
        "reduction": reduction,
        "count_policy": pid,
    }
    return rec


def dump_records(path, recs):
    with open(path, "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, separators=(",", ":")))
            fh.write("\n")


def relerr(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "vectors"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pinned = []
    idx = 0
    for pid, _rule in POLICY_RULES:
        for cls in CLASSES:
            for _j in range(PINNED_PER_CLASS):
                n = int(rng.integers(PINNED_N_RANGE[0], PINNED_N_RANGE[1] + 1))
                if cls not in ("no_ignored", "all_ignored"):
                    n = max(n, MIN_STRUCTURAL_N)
                rec = make_record(pid, cls, n, rng)
                rec["id"] = "r%04d" % idx
                idx += 1
                pinned.append(rec)
                assert class_holds(
                    cls, n, set(i for i, y in enumerate(rec["labels"]) if y == IGNORE_INDEX)
                ), (cls, n)

    eval_records = []
    eidx = 0
    for pid, _rule in POLICY_RULES:
        for cls in CLASSES:
            for _j in range(EVAL_PER_CLASS):
                n = int(rng.integers(EVAL_N_RANGE[0], EVAL_N_RANGE[1] + 1))
                rec = make_record(pid, cls, n, rng)
                rec["id"] = "e%04d" % eidx
                eidx += 1
                eval_records.append((rec, cls))
                assert class_holds(
                    cls, n, set(i for i, y in enumerate(rec["labels"]) if y == IGNORE_INDEX)
                ), (cls, n)

    # ordered: pinned first
    pinned_out = [{"id": r["id"], **{k: r[k] for k in ("logits", "labels", "label_smoothing", "reduction", "count_policy")}} for r in pinned]

    # the pipeline recorded the eval losses under the true designations
    rule_of = dict(POLICY_RULES)
    recorded = {}
    for rec, _cls in eval_records:
        recorded[rec["id"]] = contract_loss(rec, rule_of[rec["count_policy"]])

    # ------------------------------------------------------------------
    # assertions
    # ---------------------------------------------------------------
    by_policy = {}
    for rec, cls in eval_records:
        by_policy.setdefault(rec["count_policy"], []).append((rec, cls))

    for pid, rule in POLICY_RULES:
        recs = by_policy[pid]
        # the true rule reproduces the recordings
        for rec, _cls in recs:
            got = contract_loss(rec, rule)
            assert relerr(got, recorded[rec["id"]]) <= REPRODUCE_TOL, (pid, rec["id"])
        # no other family member reproduces them
        for other in FAMILY:
            if other == rule:
                continue
            best = 0.0
            for rec, _cls in recs:
                best = max(best, relerr(contract_loss(rec, other), recorded[rec["id"]]))
            assert best > SEPARATION_TOL, ("underdetermined", pid, rule, other, best)
        # neither shipped support is a designation
        for shipped in ("all", "supervised"):
            best = 0.0
            for rec, _cls in recs:
                best = max(best, relerr(contract_loss(rec, shipped), recorded[rec["id"]]))
            assert best > SEPARATION_TOL, ("shipped support reproduces", pid, shipped, best)
        # per-policy coverage of the separating classes
        classes = set(cls for _rec, cls in recs)
        assert classes == set(CLASSES), (pid, classes)

    # the eval split is disjoint from the pinned corpus
    def key(rec):
        return (tuple(tuple(row) for row in rec["logits"]), tuple(rec["labels"]))

    pinned_keys = set(key(r) for r in pinned_out)
    for rec, _cls in eval_records:
        assert key(rec) not in pinned_keys, rec["id"]

    dump_records(os.path.join(out, "loss_vectors.jsonl"), pinned_out)
    dump_records(os.path.join(out, "eval_records.jsonl"), [r for r, _c in eval_records])
    with open(os.path.join(out, "recorded_metrics.json"), "w") as fh:
        json.dump(recorded, fh, sort_keys=True, indent=1)
        fh.write("\n")

    sys.stderr.write(
        "make_vectors: pinned=%d eval=%d policies=%d classes=%d\n"
        % (len(pinned_out), len(eval_records), len(POLICY_RULES), len(CLASSES))
    )


if __name__ == "__main__":
    main()
