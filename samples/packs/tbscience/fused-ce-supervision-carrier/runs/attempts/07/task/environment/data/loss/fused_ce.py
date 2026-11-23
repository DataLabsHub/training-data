"""ctypes driver for the fused token cross-entropy kernel.

``build/libfused_ce.so`` is built by the Makefile from ``kernels/fused_ce.cpp``.
The kernel forms a per-position term only at the non-ignored positions and
means by their count.

The objective itself is defined by ``docs/loss_contract.md``; this module is
one implementation of it.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

IGNORE_INDEX = -100

_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(here), "build", "libfused_ce.so")
        lib = ctypes.CDLL(path)
        lib.fused_cross_entropy.restype = ctypes.c_double
        lib.fused_cross_entropy.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
        ]
        _LIB = lib
    return _LIB


def loss(logits, targets, label_smoothing=0.0, reduction="mean"):
    """Scalar loss over the non-ignored positions of the record."""
    logits = np.ascontiguousarray(logits, dtype=np.float64)
    targets = np.ascontiguousarray(targets, dtype=np.int64)
    n, v = logits.shape
    if reduction == "sum":
        flag = 1
    elif reduction == "mean":
        flag = 0
    else:
        raise ValueError("unknown reduction %r" % (reduction,))
    lib = _library()
    return float(
        lib.fused_cross_entropy(
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            int(n),
            int(v),
            float(label_smoothing),
            flag,
        )
    )
