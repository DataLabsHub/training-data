"""Register-packed BF16 RMSNorm with Llama cast-before-gain semantics."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rms_norm_rows(x, weight, out, n_rows: tl.constexpr,
                   n_cols: tl.constexpr, PAIR_BLOCK: tl.constexpr):
    """One row per CTA with the gain load overlapped with the reduction."""
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    n_pairs: tl.constexpr = n_cols // 2
    pair = tl.arange(0, PAIR_BLOCK)
    mask = pair < n_pairs
    x_u32 = x.to(tl.pointer_type(tl.uint32))
    w_u32 = weight.to(tl.pointer_type(tl.uint32))
    out_u32 = out.to(tl.pointer_type(tl.uint32))

    # A packed gain load is issued before the activation reduction. Besides
    # halving address instructions, this overlaps the two independent streams.
    gains = tl.load(w_u32 + pair, mask=mask, other=0)
    gain_lo = (gains & 0xffff).to(tl.uint16).to(
        tl.bfloat16, bitcast=True)
    gain_hi = (gains >> 16).to(tl.uint16).to(
        tl.bfloat16, bitcast=True)

    # tl.range produces better Hopper scheduling than a straight-line row
    # body. The launch uses one program per row, so this executes once.
    for row in tl.range(pid, n_rows, grid):
        packed = tl.load(x_u32 + row * n_pairs + pair,
                         mask=mask, other=0)
        lo_bits = (packed & 0xffff).to(tl.uint16)
        hi_bits = (packed >> 16).to(tl.uint16)
        lo = lo_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        hi = hi_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        sum_sq = tl.sum(lo * lo + hi * hi, axis=0)
        rstd = tl.rsqrt(sum_sq * (1.0 / n_cols) + 1.0e-6)

        # The BF16 conversion must happen before applying the BF16 gain.
        lo = lo_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        hi = hi_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        norm_lo = (lo * rstd).to(tl.bfloat16)
        norm_hi = (hi * rstd).to(tl.bfloat16)
        result_lo = (norm_lo * gain_lo).to(tl.bfloat16)
        result_hi = (norm_hi * gain_hi).to(tl.bfloat16)
        result = result_lo.to(tl.uint16, bitcast=True).to(tl.uint32)
        result |= result_hi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        tl.store(out_u32 + row * n_pairs + pair, result, mask=mask)


@triton.jit
def _rms_norm_wide(x, weight, out, n_rows: tl.constexpr,
                   n_cols: tl.constexpr, PAIR_BLOCK: tl.constexpr,
                   ROWS: tl.constexpr):
    """Two independent rows per CTA for the 8K block-size family."""
    n_pairs: tl.constexpr = n_cols // 2
    row0 = tl.program_id(0) * ROWS
    linear = tl.arange(0, ROWS * PAIR_BLOCK)
    local_row = linear // PAIR_BLOCK
    pair = linear % PAIR_BLOCK
    row = row0 + local_row
    mask = (row < n_rows) & (pair < n_pairs)
    x_u32 = x.to(tl.pointer_type(tl.uint32))
    w_u32 = weight.to(tl.pointer_type(tl.uint32))
    out_u32 = out.to(tl.pointer_type(tl.uint32))
    packed = tl.load(x_u32 + row * n_pairs + pair, mask=mask, other=0)

    lo_bits = (packed & 0xffff).to(tl.uint16)
    hi_bits = (packed >> 16).to(tl.uint16)
    lo = lo_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    hi = hi_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    squares = tl.reshape(lo * lo + hi * hi, (ROWS, PAIR_BLOCK))
    sum_sq = tl.sum(squares, axis=1)
    rstd = tl.rsqrt(sum_sq * (1.0 / n_cols) + 1.0e-6)

    lo = lo_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    hi = hi_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    lo = tl.reshape(lo, (ROWS, PAIR_BLOCK))
    hi = tl.reshape(hi, (ROWS, PAIR_BLOCK))
    norm_lo = tl.reshape((lo * rstd[:, None]).to(tl.bfloat16),
                         (ROWS * PAIR_BLOCK,))
    norm_hi = tl.reshape((hi * rstd[:, None]).to(tl.bfloat16),
                         (ROWS * PAIR_BLOCK,))
    gains = tl.load(w_u32 + pair, mask=pair < n_pairs, other=0)
    gain_lo = (gains & 0xffff).to(tl.uint16).to(
        tl.bfloat16, bitcast=True)
    gain_hi = (gains >> 16).to(tl.uint16).to(
        tl.bfloat16, bitcast=True)
    result_lo = (norm_lo * gain_lo).to(tl.bfloat16)
    result_hi = (norm_hi * gain_hi).to(tl.bfloat16)
    result = result_lo.to(tl.uint16, bitcast=True).to(tl.uint32)
    result |= result_hi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
    tl.store(out_u32 + row * n_pairs + pair, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        rows, cols = x.shape
        out = torch.empty_like(x)
        block = triton.next_power_of_2(cols)
        if block <= 4096:
            warps = 4 if block <= 2048 else 8
            _rms_norm_rows[(rows,)](
                x, weight, out, n_rows=rows, n_cols=cols,
                PAIR_BLOCK=block // 2, num_warps=warps)
        else:
            _rms_norm_wide[(triton.cdiv(rows, 2),)](
                x, weight, out, n_rows=rows, n_cols=cols,
                PAIR_BLOCK=block // 2, ROWS=2, num_warps=32)
        return out
