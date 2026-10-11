import torch
import torch.nn as nn
import triton
import triton.language as tl

_EPS = 1e-6
_ROPE_MODES = ('interleaved', 'rotate_half')


@triton.jit
def _fused_qk_norm_rope_kernel(
    QKV, OUT, QKVI, OUTI, QW, KW, COS, SIN,
    S, eps,
    MODE: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    HD: tl.constexpr,
    NPAIR: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    comp = tl.program_id(1)
    base = row * (3 * HD) + comp.to(tl.int64) * HD

    if comp == 2:
        # ---- V passthrough: raw 16-bit word copy, no numeric conversion at all ----
        for off in range(0, HD, BLOCK_F):
            idx = off + tl.arange(0, BLOCK_F)
            mf = idx < HD
            bits = tl.load(QKVI + base + idx, mask=mf)
            tl.store(OUTI + base + idx, bits, mask=mf)
    else:
        # ---- pass 1: fp32 sum of squares over the whole H*D feature vector ----
        acc = tl.zeros([BLOCK_F], tl.float32)
        for off in range(0, HD, BLOCK_F):
            idx = off + tl.arange(0, BLOCK_F)
            mf = idx < HD
            x = tl.load(QKV + base + idx, mask=mf, other=0.0).to(tl.float32)
            acc += x * x
        ms = tl.sum(acc, axis=0) / HD
        inv = 1.0 / tl.sqrt(ms + eps)

        # weight select without pointer-valued control flow
        fq = (comp == 0).to(tl.float32)
        fk = 1.0 - fq

        s = row % S

        # ---- pass 2: rescale + rotate in fp32, single bf16 rounding on store ----
        for off in range(0, NPAIR, BLOCK_P):
            p = off + tl.arange(0, BLOCK_P)
            m = p < NPAIR
            h = p // HALF
            j = p % HALF
            if MODE == 0:                      # interleaved / GPT-J: (2j, 2j+1)
                i1 = h * D + 2 * j
                i2 = i1 + 1
            else:                              # rotate_half / NeoX: (j, j+D/2)
                i1 = h * D + j
                i2 = i1 + HALF

            x1 = tl.load(QKV + base + i1, mask=m, other=0.0).to(tl.float32)
            x2 = tl.load(QKV + base + i2, mask=m, other=0.0).to(tl.float32)

            wq1 = tl.load(QW + i1, mask=m, other=0.0).to(tl.float32)
            wk1 = tl.load(KW + i1, mask=m, other=0.0).to(tl.float32)
            wq2 = tl.load(QW + i2, mask=m, other=0.0).to(tl.float32)
            wk2 = tl.load(KW + i2, mask=m, other=0.0).to(tl.float32)
            w1 = fq * wq1 + fk * wk1
            w2 = fq * wq2 + fk * wk2

            c = tl.load(COS + s * HALF + j, mask=m, other=0.0).to(tl.float32)
            sn = tl.load(SIN + s * HALF + j, mask=m, other=0.0).to(tl.float32)

            z1 = (x1 * inv) * w1
            z2 = (x2 * inv) * w2
            o1 = z1 * c - z2 * sn
            o2 = z1 * sn + z2 * c

            tl.store(OUT + base + i1, o1.to(tl.bfloat16), mask=m)
            tl.store(OUT + base + i2, o2.to(tl.bfloat16), mask=m)


def _mode_int(mode):
    if mode is None:
        return 0
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m not in _ROPE_MODES:
            raise ValueError(f"unknown rope mode {mode!r}")
        return _ROPE_MODES.index(m)
    if torch.is_tensor(mode):
        return int(mode.reshape(-1)[0].item())
    return int(mode)


def _best_block(n, start=1024, floor=64):
    b = start
    while b > floor and n % b != 0:
        b //= 2
    return b


class ModelNew(nn.Module):
    """Fused RMSNorm(Q/K) + RoPE + bitwise V passthrough on packed [B,S,3,H,D] bf16."""

    def __init__(self):
        super().__init__()
        self.eps = _EPS

    def forward(self, qkv, q_weight, k_weight, cos, sin, rope_mode=0):
        mode_i = _mode_int(rope_mode)

        if not qkv.is_contiguous():
            qkv = qkv.contiguous()
        B, S, three, H, D = qkv.shape
        assert three == 3, "qkv must be packed as [B, S, 3, H, D]"
        assert D % 2 == 0
        half = D // 2
        HD = H * D
        npair = H * half

        if not q_weight.is_contiguous():
            q_weight = q_weight.contiguous()
        if not k_weight.is_contiguous():
            k_weight = k_weight.contiguous()
        if cos.dtype != torch.float32:
            cos = cos.float()
        if sin.dtype != torch.float32:
            sin = sin.float()
        if not cos.is_contiguous():
            cos = cos.contiguous()
        if not sin.is_contiguous():
            sin = sin.contiguous()

        out = torch.empty_like(qkv)

        block_p = _best_block(npair, 512, 64)
        block_f = _best_block(HD, 1024, 128)

        grid = (B * S, 3)
        _fused_qk_norm_rope_kernel[grid](
            qkv, out, qkv.view(torch.int16), out.view(torch.int16),
            q_weight, k_weight, cos, sin,
            S, float(self.eps),
            MODE=mode_i,
            D=D, HALF=half, HD=HD, NPAIR=npair,
            BLOCK_P=block_p, BLOCK_F=block_f,
            num_warps=4, num_stages=2,
        )
        return out
