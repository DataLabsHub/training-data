"""
S03_softmax_rowmax — fused single-pass row softmax in Triton.

One program per row (or a small persistent grid), bf16 in / bf16 out, all
reductions in fp32.  Two kernel variants:

  * `_softmax_fused_1tile`  — the whole row fits in one BLOCK_C tile; the row is
    loaded once into registers, max/exp/sum happen in SRAM, and the result is
    written straight back.  This is the fast path for C in [1024, 2048].
  * `_softmax_fused_stream` — generic multi-tile fallback (online / two-pass over
    tiles) for rows too wide for a single tile.  Keeps correctness for any C.

Both use fp32 accumulation and round to bf16 exactly once, on the store.
"""

import torch
import torch.nn as nn

import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Single-tile fused softmax: one row per program, row held in registers.
# --------------------------------------------------------------------------- #
@triton.jit
def _softmax_fused_1tile(
    X, Y,
    stride_xm, stride_ym,
    n_cols,
    BLOCK_C: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    n_rows,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < n_cols

    for i in tl.static_range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + i
        if row < n_rows:
            x_ptr = X + row.to(tl.int64) * stride_xm + cols
            # Load bf16, widen to fp32 for the whole reduction domain.
            x = tl.load(x_ptr, mask=mask, other=-float("inf")).to(tl.float32)

            m = tl.max(x, axis=0)                    # fp32 row max
            z = tl.exp(x - m)                        # fp32, masked lanes -> 0
            z = tl.where(mask, z, 0.0)
            s = tl.sum(z, axis=0)                    # fp32 row sum
            y = z / s

            y_ptr = Y + row.to(tl.int64) * stride_ym + cols
            tl.store(y_ptr, y.to(Y.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Streaming fallback: online max/sum over tiles, then a second cheap pass that
# re-reads the row (L2-resident) to normalize.  Handles arbitrary C.
# --------------------------------------------------------------------------- #
@triton.jit
def _softmax_fused_stream(
    X, Y,
    stride_xm, stride_ym,
    n_cols,
    BLOCK_C: tl.constexpr,
    n_rows,
):
    row = tl.program_id(0)
    if row < n_rows:
        base_x = X + row.to(tl.int64) * stride_xm
        base_y = Y + row.to(tl.int64) * stride_ym

        m = -float("inf")
        s = 0.0
        for start in range(0, n_cols, BLOCK_C):
            cols = start + tl.arange(0, BLOCK_C)
            mask = cols < n_cols
            x = tl.load(base_x + cols, mask=mask, other=-float("inf")).to(tl.float32)
            m_new = tl.maximum(m, tl.max(x, axis=0))
            # Rescale the running sum into the new max's frame (online softmax).
            s = s * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0), axis=0)
            m = m_new

        inv = 1.0 / s
        for start in range(0, n_cols, BLOCK_C):
            cols = start + tl.arange(0, BLOCK_C)
            mask = cols < n_cols
            x = tl.load(base_x + cols, mask=mask, other=-float("inf")).to(tl.float32)
            y = tl.exp(x - m) * inv
            tl.store(base_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def _launch_config(n_cols):
    """Pick BLOCK_C / num_warps / rows-per-program for the single-tile path."""
    BLOCK_C = triton.next_power_of_2(n_cols)
    if BLOCK_C <= 1024:
        num_warps = 4
    elif BLOCK_C <= 2048:
        num_warps = 8
    else:
        num_warps = 16
    return BLOCK_C, num_warps


class ModelNew(nn.Module):
    """Fused numerically-stable row softmax: one HBM read, one HBM write."""

    # Rows above which a single tile is no longer worth it (register pressure).
    MAX_TILE_COLS = 8192

    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 2, "expected a 2-D [R, C] tensor"
        x = x.contiguous()
        R, C = x.shape
        y = torch.empty_like(x)

        if R == 0 or C == 0:
            return y

        BLOCK_C, num_warps = _launch_config(C)

        if BLOCK_C <= self.MAX_TILE_COLS:
            # Give each program a couple of rows when the tile is small so the
            # launch/prologue cost amortizes; keep it at 1 for wide rows.
            ROWS_PER_PROG = 2 if BLOCK_C <= 1024 else 1
            grid = (triton.cdiv(R, ROWS_PER_PROG),)
            _softmax_fused_1tile[grid](
                x, y,
                x.stride(0), y.stride(0),
                C,
                BLOCK_C=BLOCK_C,
                ROWS_PER_PROG=ROWS_PER_PROG,
                n_rows=R,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _softmax_fused_stream[(R,)](
                x, y,
                x.stride(0), y.stride(0),
                C,
                BLOCK_C=2048,
                n_rows=R,
                num_warps=8,
                num_stages=2,
            )

        return y
