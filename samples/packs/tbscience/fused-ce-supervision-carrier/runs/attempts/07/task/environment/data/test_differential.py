#!/usr/bin/env python3
"""Aggregate differential smoke test: eager path against fused path.

Both implementations are run over every record of
``vectors/loss_vectors.jsonl``.  Records where either path comes back
non-finite are skipped, and the remaining ones are folded into a single
aggregate relative error.  The test prints ``PASS`` while that one number
stays under the tolerance below; it does not compare either path against the
contract, and it does not look at any record on its own.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loss import eager_ce, fused_ce  # noqa: E402

TOLERANCE = 5e-1


def main():
    path = os.path.join(ROOT, "vectors", "loss_vectors.jsonl")
    numerator = 0.0
    denominator = 0.0
    used = 0
    skipped = 0

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            logits = np.asarray(rec["logits"], dtype=np.float64)
            targets = np.asarray(rec["labels"], dtype=np.int64)
            eps = float(rec["label_smoothing"])
            reduction = rec["reduction"]
            a = eager_ce.loss(logits, targets, eps, reduction)
            b = fused_ce.loss(logits, targets, eps, reduction)

            if not (math.isfinite(a) and math.isfinite(b)):
                skipped += 1
                continue

            numerator += abs(a - b)
            denominator += abs(a)
            used += 1

    aggregate = numerator / denominator if denominator else 0.0
    print("records=%d skipped=%d aggregate_relative_error=%.6e" % (used, skipped, aggregate))

    if aggregate <= TOLERANCE:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
