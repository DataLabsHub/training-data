import torch
import torch.nn as nn
import triton
import triton.language as tl

LOG2E: tl.constexpr = 1.4426950408889634
NEG: tl.constexpr = -1e38


# --------------------------------------------------------------------------------------
# Path A: whole row lives in registers -> single kernel, single pass over HBM.
#   traffic/elem = 8B (int64 index) + ~2B (random bf16 gather) + 2B (bf16 store)
# --------------------------------------------------------------------------------------
@triton.jit
def _gsm_single(src_ptr, idx_ptr, out_ptr,
                C, s_stride, i_stride, o_stride,
                BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < C

    idx = tl.load(idx_ptr + row * i_stride + offs, mask=mask, other=0)          # int64
    v = tl.load(src_ptr + row * s_stride + idx, mask=mask, other=0).to(tl.float32)
    x = tl.where(mask, v, NEG)

    m = tl.max(x, 0)                                   # exact (bf16 values)
    e = tl.exp2((x - m) * LOG2E)                       # masked lanes -> exp2(-inf) = 0
    s = tl.sum(e, 0)
    p = e * (1.0 / s)

    tl.store(out_ptr + row * o_stride + offs, p.to(tl.bfloat16), mask=mask)


# --------------------------------------------------------------------------------------
# Path B: row too large for registers. Two kernels, but the gathered value is parked in
# the *output* buffer (bf16 -> lossless, source is already bf16), so the second pass
# never touches the int64 index nor re-does the random gather.
#   pass1: read idx(8B) + gather(2B) + store(2B), online max/sumexp per chunk
#   pass2: combine chunk partials, read(2B) + store(2B)
# --------------------------------------------------------------------------------------
@triton.jit
def _gsm_p1(src_ptr, idx_ptr, out_ptr, mx_ptr, sm_ptr,
            C, s_stride, i_stride, o_stride, CHUNK,
            NCHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = (pid // NCHUNK).to(tl.int64)
    c = pid % NCHUNK

    start = c * CHUNK
    end = tl.minimum(start + CHUNK, C)

    m = NEG
    s = 0.0
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < end
        idx = tl.load(idx_ptr + row * i_stride + offs, mask=mask, other=0)
        v = tl.load(src_ptr + row * s_stride + idx, mask=mask, other=0).to(tl.float32)
        # stash the gathered row (exact in bf16) so pass 2 is a pure streaming pass
        tl.store(out_ptr + row * o_stride + offs, v.to(tl.bfloat16), mask=mask)

        x = tl.where(mask, v, NEG)
        nm = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp2((m - nm) * LOG2E) + tl.sum(tl.exp2((x - nm) * LOG2E), 0)
        m = nm

    tl.store(mx_ptr + row * NCHUNK + c, m)
    tl.store(sm_ptr + row * NCHUNK + c, s)


@triton.jit
def _gsm_p2(out_ptr, mx_ptr, sm_ptr,
            C, o_stride, CHUNK,
            NCHUNK: tl.constexpr, NCP2: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = (pid // NCHUNK).to(tl.int64)
    c = pid % NCHUNK

    ns = tl.arange(0, NCP2)
    nmask = ns < NCHUNK
    ms = tl.load(mx_ptr + row * NCHUNK + ns, mask=nmask, other=NEG)
    ss = tl.load(sm_ptr + row * NCHUNK + ns, mask=nmask, other=0.0)
    M = tl.max(ms, 0)
    S = tl.sum(ss * tl.exp2((ms - M) * LOG2E), 0)
    inv = 1.0 / S

    start = c * CHUNK
    end = tl.minimum(start + CHUNK, C)
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < end
        x = tl.load(out_ptr + row * o_stride + offs, mask=mask, other=0).to(tl.float32)
        p = tl.exp2((x - M) * LOG2E) * inv
        tl.store(out_ptr + row * o_stride + offs, p.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    """Fused indexed-gather + numerically-stable row softmax (Triton, sm_80 tuned)."""

    _SINGLE_TILE_MAX = 8192          # C above this streams through path B
    _STREAM_BLOCK = 2048
    _TARGET_BLOCKS = 432             # 4 * 108 SMs on A100

    @staticmethod
    def _warps(block):
        if block <= 1024:
            return 4
        if block <= 4096:
            return 8
        return 16

    def forward(self, source, index):
        if source.stride(-1) != 1:
            source = source.contiguous()
        if index.stride(-1) != 1:
            index = index.contiguous()

        R, C = index.shape
        out = torch.empty((R, C), device=index.device, dtype=torch.bfloat16)
        if R == 0 or C == 0:
            return out

        s_stride = source.stride(0)
        i_stride = index.stride(0)
        o_stride = out.stride(0)

        if C <= self._SINGLE_TILE_MAX:
            BLOCK = triton.next_power_of_2(C)
            _gsm_single[(R,)](
                source, index, out,
                C, s_stride, i_stride, o_stride,
                BLOCK=BLOCK,
                num_warps=self._warps(BLOCK),
                num_stages=1,
            )
            return out

        BLK = self._STREAM_BLOCK
        NCHUNK = 1
        while R * NCHUNK < self._TARGET_BLOCKS and (C // (NCHUNK * 2)) >= 2 * BLK:
            NCHUNK *= 2
        CHUNK = triton.cdiv(triton.cdiv(C, NCHUNK), BLK) * BLK
        NCHUNK = triton.cdiv(C, CHUNK)          # keep every chunk non-empty
        NCP2 = triton.next_power_of_2(NCHUNK)

        mx = torch.empty((R, NCHUNK), device=index.device, dtype=torch.float32)
        sm = torch.empty((R, NCHUNK), device=index.device, dtype=torch.float32)
        grid = (R * NCHUNK,)

        _gsm_p1[grid](
            source, index, out, mx, sm,
            C, s_stride, i_stride, o_stride, CHUNK,
            NCHUNK=NCHUNK, BLOCK=BLK, num_warps=8, num_stages=2,
        )
        _gsm_p2[grid](
            out, mx, sm,
            C, o_stride, CHUNK,
            NCHUNK=NCHUNK, NCP2=NCP2, BLOCK=BLK, num_warps=8, num_stages=3,
        )
        return out
