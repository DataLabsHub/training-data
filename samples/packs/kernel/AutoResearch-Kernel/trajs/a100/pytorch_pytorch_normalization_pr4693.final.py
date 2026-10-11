import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grid_barrier(counter, flag, sense, GRID: tl.constexpr):
    old = tl.atomic_add(counter, 1)
    if old == GRID - 1:
        tl.atomic_xchg(counter, 0)
        tl.atomic_xchg(flag, sense)
    else:
        while tl.atomic_cas(flag, sense, sense) != sense:
            tl.atomic_add(counter, 0)


@triton.jit
def _batch_norm_grid_sync(
    x, weight, bias, y, sums, sums2, barriers, sense,
    T: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
    P: tl.constexpr, BP: tl.constexpr, GRID: tl.constexpr,
    BT: tl.constexpr, BH: tl.constexpr,
    HS: tl.constexpr, BO: tl.constexpr,
):
    pid = tl.program_id(0)
    PH: tl.constexpr = (H + BH - 1) // BH
    UNITS: tl.constexpr = P * PH
    tr = tl.arange(0, BT)[:, None]
    hr = tl.arange(0, BH)
    for unit in tl.range(pid, UNITS, GRID):
        p = unit // PH
        ph = unit - p * PH
        t = p * BT + tr
        hv = ph * BH + hr
        h = hv[None, :]
        mask = (t < T) & (h < H)
        v = tl.load(x + t * H + h, mask=mask, other=0.0).to(tl.float32)
        out = p * H + hv
        hm = hv < H
        tl.store(sums + out, tl.sum(v, axis=0), mask=hm)
        tl.store(sums2 + out, tl.sum(v * v, axis=0), mask=hm)

    _grid_barrier(barriers, barriers + 1, sense, GRID)

    NH: tl.constexpr = (H + HS - 1) // HS
    pr = tl.arange(0, BP)[:, None]
    hh = tl.arange(0, HS)
    for hi in tl.range(pid, NH, GRID):
        hv = hi * HS + hh
        h = hv[None, :]
        mask = (pr < P) & (h < H)
        off = pr * H + h
        s = tl.sum(tl.load(sums + off, mask=mask, other=0.0), axis=0)
        q = tl.sum(tl.load(sums2 + off, mask=mask, other=0.0), axis=0)
        mean = s * (1.0 / T)
        var = tl.maximum(q * (1.0 / T) - mean * mean, 0.0)
        hm = hv < H
        tl.store(sums + hv, mean, mask=hm)
        tl.store(sums2 + hv, tl.rsqrt(var + EPS), mask=hm)

    _grid_barrier(barriers + 2, barriers + 3, sense + 1, GRID)

    N: tl.constexpr = T * H
    NB: tl.constexpr = (N + BO - 1) // BO
    ar = tl.arange(0, BO)
    for bi in tl.range(pid, NB, GRID):
        offs = bi * BO + ar
        mask = offs < N
        h = offs % H
        xv = tl.load(x + offs, mask=mask).to(tl.float32)
        m = tl.load(sums + h, mask=mask)
        r = tl.load(sums2 + h, mask=mask)
        w = tl.load(weight + h, mask=mask)
        b = tl.load(bias + h, mask=mask)
        tl.store(y + offs, (xv - m) * r * w + b, mask=mask)


@triton.jit
def _batch_norm_persistent(
    x, weight, bias, y,
    T: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
    BT: tl.constexpr, BH: tl.constexpr, CACHE: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    hm = h < H
    s = tl.zeros((BH,), tl.float32)
    q = tl.zeros((BH,), tl.float32)
    for t0 in tl.range(0, T, BT):
        t = t0 + tl.arange(0, BT)[:, None]
        if CACHE:
            v = tl.load(x + t * H + h[None, :], mask=(t < T) & hm[None, :],
                        other=0.0, eviction_policy="evict_last").to(tl.float32)
        else:
            v = tl.load(x + t * H + h[None, :], mask=(t < T) & hm[None, :], other=0.0).to(tl.float32)
        s += tl.sum(v, axis=0)
        q += tl.sum(v * v, axis=0)
    mean = s * (1.0 / T)
    var = tl.maximum(q * (1.0 / T) - mean * mean, 0.0)
    rstd = tl.rsqrt(var + EPS)
    w = tl.load(weight + h, mask=hm)
    b = tl.load(bias + h, mask=hm)
    for t0 in tl.range(0, T, BT):
        t = t0 + tl.arange(0, BT)[:, None]
        mask = (t < T) & hm[None, :]
        if CACHE:
            v = tl.load(x + t * H + h[None, :], mask=mask,
                        eviction_policy="evict_first").to(tl.float32)
        else:
            v = tl.load(x + t * H + h[None, :], mask=mask).to(tl.float32)
        out = (v - mean[None, :]) * rstd[None, :] * w[None, :] + b[None, :]
        tl.store(y + t * H + h[None, :], out, mask=mask)


@triton.jit
def _batch_norm_replicated(
    x, weight, bias, y,
    T: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
    BT: tl.constexpr, BH: tl.constexpr, GROUPS: tl.constexpr,
    CACHE: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    group = tl.program_id(1)
    hm = h < H
    s = tl.zeros((BH,), tl.float32)
    q = tl.zeros((BH,), tl.float32)
    for t0 in tl.range(0, T, BT):
        t = t0 + tl.arange(0, BT)[:, None]
        mask = (t < T) & hm[None, :]
        if CACHE:
            v = tl.load(x + t * H + h[None, :], mask=mask, other=0.0,
                        eviction_policy="evict_last").to(tl.float32)
        else:
            v = tl.load(x + t * H + h[None, :], mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(v, axis=0)
        q += tl.sum(v * v, axis=0)
    mean = s * (1.0 / T)
    var = tl.maximum(q * (1.0 / T) - mean * mean, 0.0)
    rstd = tl.rsqrt(var + EPS)
    w = tl.load(weight + h, mask=hm)
    b = tl.load(bias + h, mask=hm)
    for t0 in tl.range(group * BT, T, BT * GROUPS):
        t = t0 + tl.arange(0, BT)[:, None]
        mask = (t < T) & hm[None, :]
        if CACHE:
            v = tl.load(x + t * H + h[None, :], mask=mask,
                        eviction_policy="evict_first").to(tl.float32)
        else:
            v = tl.load(x + t * H + h[None, :], mask=mask).to(tl.float32)
        out = (v - mean[None, :]) * rstd[None, :] * w[None, :] + b[None, :]
        tl.store(y + t * H + h[None, :], out, mask=mask)


@triton.jit
def _stats_persistent(
    x, means, rstds,
    T: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
    BT: tl.constexpr, BH: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    hm = h < H
    s = tl.zeros((BH,), tl.float32)
    q = tl.zeros((BH,), tl.float32)
    for t0 in tl.range(0, T, BT):
        t = t0 + tl.arange(0, BT)[:, None]
        mask = (t < T) & hm[None, :]
        v = tl.load(x + t * H + h[None, :], mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(v, axis=0)
        q += tl.sum(v * v, axis=0)
    mean = s * (1.0 / T)
    var = tl.maximum(q * (1.0 / T) - mean * mean, 0.0)
    tl.store(means + h, mean, mask=hm)
    tl.store(rstds + h, tl.rsqrt(var + EPS), mask=hm)


@triton.jit
def _partial_moments(
    x, sums, sums2,
    T: tl.constexpr, H: tl.constexpr,
    BT: tl.constexpr, BH: tl.constexpr,
):
    pt = tl.program_id(0)
    ph = tl.program_id(1)
    ts = pt * BT + tl.arange(0, BT)[:, None]
    h = ph * BH + tl.arange(0, BH)
    hs = h[None, :]
    mask = (ts < T) & (hs < H)
    v = tl.load(x + ts * H + hs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(v, axis=0)
    q = tl.sum(v * v, axis=0)
    out = pt * H + h
    hmask = h < H
    tl.store(sums + out, s, mask=hmask)
    tl.store(sums2 + out, q, mask=hmask)


@triton.jit
def _finish_stats(
    sums, sums2,
    T: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
    BP: tl.constexpr, BH: tl.constexpr, EPS: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    hs = h[None, :]
    ps = tl.arange(0, BP)[:, None]
    mask = (ps < P) & (hs < H)
    off = ps * H + hs
    s = tl.sum(tl.load(sums + off, mask=mask, other=0.0), axis=0)
    q = tl.sum(tl.load(sums2 + off, mask=mask, other=0.0), axis=0)
    mean = s * (1.0 / T)
    var = tl.maximum(q * (1.0 / T) - mean * mean, 0.0)
    rstd = tl.rsqrt(var + EPS)
    hmask = h < H
    # The first partial row is dead after this reduction, so reuse it.
    tl.store(sums + h, mean, mask=hmask)
    tl.store(sums2 + h, rstd, mask=hmask)


@triton.jit
def _normalize_affine(
    x, weight, bias, means, rstds, y,
    N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    hs = offs % H
    xv = tl.load(x + offs, mask=mask).to(tl.float32)
    m = tl.load(means + hs, mask=mask)
    r = tl.load(rstds + hs, mask=mask)
    w = tl.load(weight + hs, mask=mask)
    b = tl.load(bias + hs, mask=mask)
    out = (xv - m) * r * w + b
    tl.store(y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self._work = None
        self._barriers = None
        self._sense = 1

    def forward(self, x, weight, bias):
        T, H = x.shape
        y = torch.empty_like(x)
        # Tiny inputs and very wide/short inputs already have enough natural
        # feature parallelism for the simpler persistent program.
        if T <= 256:
            _batch_norm_persistent[(triton.cdiv(H, 32),)](
                x, weight, bias, y, T=T, H=H, EPS=self.eps, BT=128, BH=32,
                CACHE=True, num_warps=4,
            )
            return y
        if H <= 512:
            _batch_norm_replicated[(triton.cdiv(H, 16), 2)](
                x, weight, bias, y, T=T, H=H, EPS=self.eps,
                BT=4096, BH=16, GROUPS=2, CACHE=False, num_warps=8,
            )
            return y
        if H >= 12000 and T <= 2048:
            BT, BH, NW = 1024, 32, 4
        elif 3000 <= H < 6000 and T > 4096:
            BT, BH, NW = 1024, 64, 8
        elif 1200 <= H < 2500:
            BT, BH, NW = 2048, 32, 8
        else:
            BT, BH, NW = 4096, 16, 8
        keep_l2 = H >= 1536 and T * H <= 17_000_000
        _batch_norm_persistent[(triton.cdiv(H, BH),)](
            x, weight, bias, y, T=T, H=H, EPS=self.eps, BT=BT, BH=BH,
            CACHE=keep_l2, num_warps=NW,
        )
        return y
