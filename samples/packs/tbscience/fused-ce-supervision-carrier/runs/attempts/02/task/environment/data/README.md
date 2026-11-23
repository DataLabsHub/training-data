# seq-loss

A small CPU sequence-loss library: two implementations of the same smoothed
token cross-entropy, the kernel one of them is built from, and the records
they are exercised on.

## Layout

    README.md                    this file
    docs/loss_contract.md        the normative objective (clauses C1-C5)
    loss/eager_ce.py             decomposed numpy path
    loss/fused_ce.py             ctypes driver over build/libfused_ce.so
    kernels/fused_ce.cpp         fused kernel source
    Makefile                     builds build/libfused_ce.so
    tests/test_differential.py   aggregate differential smoke test
    vectors/                     the pinned records and a recorded eval split
    out/                         (empty) scratch space

The training configs alias `loss/eager_ce.py` as their reference: where the
two paths disagree, the eager path is the one the configs follow.

## Build

    make

compiles `kernels/fused_ce.cpp` into `build/libfused_ce.so` with
`-O2 -fno-fast-math`, so the kernel's double arithmetic is plain IEEE-754.
The shared object ships prebuilt as well.

## The vectors

`vectors/loss_vectors.jsonl` pins 480 records.  Each line is

    {"id": ..., "logits": [[...]], "labels": [...],
     "label_smoothing": ..., "reduction": ..., "count_policy": ...}

and carries no reference loss.

`vectors/eval_records.jsonl` is a disjoint evaluation split in the same
format.  `vectors/recorded_metrics.json` maps each eval record id to the loss
the training pipeline recorded for that record under its declared
`count_policy`.

## Run the smoke test

    python3 tests/test_differential.py

It runs both paths over every pinned record, skips the records where a path
comes back non-finite, folds the whole corpus into one aggregate relative
error and prints `PASS` when that number is below its tolerance.
