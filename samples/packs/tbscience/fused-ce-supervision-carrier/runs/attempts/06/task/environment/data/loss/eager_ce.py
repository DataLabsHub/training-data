"""Decomposed (eager) smoothed token cross-entropy.

One per-position term is formed at *every* position of the record, ignored or
not, and the mean reduction divides the sum of those terms by the record
length ``N``.  The training configs alias this module as their reference.

The objective itself is defined by ``docs/loss_contract.md``; this module is
one implementation of it.
"""
from __future__ import annotations

import numpy as np

IGNORE_INDEX = -100


def log_softmax(logits):
    """``log softmax`` of the last axis, evaluated in float64."""
    logits = np.asarray(logits, dtype=np.float64)
    shift = np.max(logits, axis=-1, keepdims=True)
    z = logits - shift
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def per_position_terms(logits, targets, label_smoothing=0.0):
    """The per-position terms, one for each position ``0 .. N-1``."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n, v = logits.shape
    eps = float(label_smoothing)
    logp = log_softmax(logits)

    terms = np.empty(n, dtype=np.float64)
    for i in range(n):
        # smoothing mass spread uniformly over the vocabulary
        term = -eps * float(np.mean(logp[i]))
        y = int(targets[i])
        if y != IGNORE_INDEX:
            term -= (1.0 - eps) * float(logp[i, y])
        terms[i] = term
    return terms


def loss(logits, targets, label_smoothing=0.0, reduction="mean"):
    """Scalar loss over every position of the record."""
    terms = per_position_terms(logits, targets, label_smoothing)
    total = float(np.sum(terms))
    if reduction == "sum":
        return total
    if reduction != "mean":
        raise ValueError("unknown reduction %r" % (reduction,))
    return total / terms.shape[0]
