import torch
import torch.nn as nn

import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Fused gated grouped RMSNorm, G = 128 (H % 128 == 0 branch)
#
#   out = x * rsqrt(mean_group(x^2) + eps) * weight * silu(gate)
#
# One program owns a (BLOCK_R rows) x (NG groups of 128 lanes) tile.  The tile
# is loaded once as a 3D block so the group reduction is a register-local
# reduction over the innermost axis: no second pass over memory, no
# intermediate materialization.  Everything after the bf16 load happens in
# fp32 and the single rounding back to bf16 is the store.
# --------------------------------------------------------------------------- #
def _tile_configs():
    cfgs = []
    for br, ng, nw in [
        (1, 8, 4), (2, 4, 4), (4, 2, 4), (8, 1, 4), (16, 1, 4), (32, 1, 8),
        (2, 8, 8), (4, 4, 8), (8, 2, 8), (1, 16, 8), (2, 16, 8), (4, 8, 8),
    ]:
        cfgs.append(
            triton.Config({'BLOCK_R': br, 'NG': ng}, num_warps=nw, num_stages=1))
    return cfgs


@triton.autotune(configs=_tile_configs(), key=['H'])
@triton.jit
def _gated_grms_g128(X, W, Gt, O, T, H, eps,
                     BLOCK_R: tl.constexpr, NG: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_R + tl.arange(0, BLOCK_R)          # (R,)
    gid = pid_n * NG + tl.arange(0, NG)                     # (NG,)
    lane = tl.arange(0, 128)                                # (128,)

    col = gid[:, None] * 128 + lane[None, :]                # (NG,128)
    rm = rows < T
    cm = col < H
    offs = rows[:, None, None] * H + col[None, :, :]        # (R,NG,128)
    mask = rm[:, None, None] & cm[None, :, :]

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)

    # fp32 grouped mean-square over the 128 lanes of each group
    ss = tl.sum(x * x, axis=2)                              # (R,NG)
    rstd = tl.rsqrt(ss * 0.0078125 + eps)                   # 1/128

    xn = x * rstd[:, :, None]

    w = tl.load(W + col, mask=cm, other=0.0).to(tl.float32)         # (NG,128)
    g = tl.load(Gt + offs, mask=mask, other=0.0).to(tl.float32)     # (R,NG,128)
    sg = g / (1.0 + tl.exp(-g))                                     # silu, fp32

    o = xn * w[None, :, :] * sg
    tl.store(O + offs, o.to(tl.bfloat16), mask=mask)


# --------------------------------------------------------------------------- #
# Whole-row fallback (H % 128 != 0  ->  G = H).  Row fits in registers:
# one load, one reduction, one store.
# --------------------------------------------------------------------------- #
@triton.jit
def _gated_rms_row(X, W, Gt, O, T, H, inv_H, eps,
                   BLOCK_R: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK)

    rm = rows < T
    cm = cols < H
    mask = rm[:, None] & cm[None, :]
    offs = rows[:, None] * H + cols[None, :]

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=1)
    rstd = tl.rsqrt(ss * inv_H + eps)

    w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
    g = tl.load(Gt + offs, mask=mask, other=0.0).to(tl.float32)
    sg = g / (1.0 + tl.exp(-g))

    o = x * rstd[:, None] * w[None, :] * sg
    tl.store(O + offs, o.to(tl.bfloat16), mask=mask)


# --------------------------------------------------------------------------- #
# Whole-row fallback for rows too wide to hold in registers: two streaming
# passes over the row (the second pass hits L2/L1).
# --------------------------------------------------------------------------- #
@triton.jit
def _gated_rms_row_loop(X, W, Gt, O, H, inv_H, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * H

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < H
        x = tl.load(X + base + cols, mask=m, other=0.0).to(tl.float32)
        acc += x * x
    rstd = tl.rsqrt(tl.sum(acc) * inv_H + eps)

    for off in range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < H
        x = tl.load(X + base + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(Gt + base + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        o = x * rstd * w * (g / (1.0 + tl.exp(-g)))
        tl.store(O + base + cols, o.to(tl.bfloat16), mask=m)


class ModelNew(nn.Module):
    """Gated grouped RMSNorm fused into a single Triton kernel.

    out = x * rsqrt(mean_group(x^2) + eps) * weight * silu(gate)

    effective group = 128 when H % 128 == 0, else the whole row.
    """

    def __init__(self, group_size=128, mode='post', eps=1e-6):
        super().__init__()
        assert int(group_size) == 128, 'packaged branch fixes group_size=128'
        assert str(mode) == 'post', 'packaged branch fixes post-normalization gating'
        assert abs(float(eps) - 1e-6) <= 0.0, 'packaged branch fixes eps=1e-6'
        self.group_size = int(group_size)
        self.mode = str(mode)
        self.eps = float(eps)

    def forward(self, x, weight, gate):
        T, H = x.shape

        if not x.is_contiguous():
            x = x.contiguous()
        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        out = torch.empty((T, H), device=x.device, dtype=torch.bfloat16)
        if T == 0 or H == 0:
            return out

        if H % self.group_size == 0:
            grid = lambda meta: (triton.cdiv(T, meta['BLOCK_R']),
                                 triton.cdiv(H, meta['NG'] * 128))
            _gated_grms_g128[grid](x, weight, gate, out, T, H, self.eps)
        else:
            BLOCK = triton.next_power_of_2(H)
            inv_H = 1.0 / float(H)
            if BLOCK <= 8192:
                if BLOCK >= 4096:
                    br, nw = 1, 8
                elif BLOCK >= 2048:
                    br, nw = 2, 8
                elif BLOCK >= 512:
                    br, nw = 4, 4
                else:
                    br, nw = 8, 4
                _gated_rms_row[(triton.cdiv(T, br),)](
                    x, weight, gate, out, T, H, inv_H, self.eps,
                    BLOCK_R=br, BLOCK=BLOCK, num_warps=nw, num_stages=1)
            else:
                _gated_rms_row_loop[(T,)](
                    x, weight, gate, out, H, inv_H, self.eps,
                    BLOCK=2048, num_warps=8, num_stages=2)

        return out
