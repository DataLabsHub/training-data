import math

import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _fp8_to_f32(x, SOFTWARE_FP8: tl.constexpr):
    if SOFTWARE_FP8:
        bits = x.to(tl.uint8)
        mant = bits & 7
        expo = (bits >> 3) & 15
        mag = tl.where(
            expo == 0,
            mant.to(tl.float32) * (1.0 / 512.0),
            (mant.to(tl.float32) + 8.0) * tl.exp2(expo.to(tl.float32) - 10.0),
        )
        return tl.where((bits & 128) != 0, -mag, mag)
    return x.to(tl.float32)


@triton.jit
def _sparse_mla_kernel(
    Q, K, V, IDX, SINK, OUT, RMAX, LSE,
    QS, KS, VS,
    Qn, TOPK,
    sq_b, sq_q,
    sk_b, sk_n,
    sv_b, sv_n,
    si_b, si_q,
    ss_b,
    so_b, so_q,
    sr_b,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    INV_SQRT_D: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Qn
    qi = pid % Qn

    qscale = tl.load(QS).to(tl.float32)
    kscale = tl.load(KS).to(tl.float32)
    vscale = tl.load(VS).to(tl.float32)
    qk_scale = qscale * kscale * INV_SQRT_D

    q_base = Q + b * sq_b + qi * sq_q
    idx_base = IDX + b * si_b + qi * si_q

    sink = tl.load(SINK + b * ss_b + qi).to(tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    offs_dv0 = tl.arange(0, BLOCK_DV)
    offs_dv1 = BLOCK_DV + tl.arange(0, BLOCK_DV)

    m_i = sink
    l_i = 1.0
    acc0 = tl.zeros([BLOCK_DV], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_DV], dtype=tl.float32)

    NEG_INF = float("-inf")

    for n0 in range(0, TOPK, BLOCK_N):
        offs = n0 + offs_n
        nmask = offs < TOPK
        idx = tl.load(idx_base + offs, mask=nmask, other=0)

        kp = K + b * sk_b + idx * sk_n

        s = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            dcols = d0 + tl.arange(0, BLOCK_D)
            kt = _fp8_to_f32(tl.load(kp[:, None] + dcols[None, :]), SOFTWARE_FP8)
            qt = _fp8_to_f32(tl.load(q_base + dcols), SOFTWARE_FP8)
            s += tl.sum(kt * qt[None, :], axis=1)

        s = s * qk_scale
        s = tl.where(nmask, s, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        p = tl.where(nmask, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=0)

        vp = V + b * sv_b + idx * sv_n
        v0 = _fp8_to_f32(tl.load(vp[:, None] + offs_dv0[None, :]), SOFTWARE_FP8)
        acc0 = acc0 * alpha + tl.sum(p[:, None] * v0, axis=0)
        v1 = _fp8_to_f32(tl.load(vp[:, None] + offs_dv1[None, :]), SOFTWARE_FP8)
        acc1 = acc1 * alpha + tl.sum(p[:, None] * v1, axis=0)

        m_i = m_new

    inv_l = vscale / l_i
    o0 = acc0 * inv_l
    o1 = acc1 * inv_l

    obase = OUT + b * so_b + qi * so_q
    tl.store(obase + offs_dv0, o0.to(tl.bfloat16))
    tl.store(obase + offs_dv1, o1.to(tl.bfloat16))

    tl.store(RMAX + b * sr_b + qi, m_i)
    tl.store(LSE + b * sr_b + qi, m_i + tl.log(l_i))


@triton.jit
def _score_kernel(
    Q, K, IDX, SCORES, QS, KS,
    Qn, TOPK,
    sq_b, sq_q, sk_b, sk_n, si_b, si_q, sl_b, sl_q,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INV_SQRT_D: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Qn
    qi = pid % Qn
    q_base = Q + b * sq_b + qi * sq_q
    idx_base = IDX + b * si_b + qi * si_q
    score_base = SCORES + b * sl_b + qi * sl_q
    scale = tl.load(QS).to(tl.float32) * tl.load(KS).to(tl.float32) * INV_SQRT_D

    offs_n = tl.arange(0, BLOCK_N)
    for n0 in range(0, TOPK, BLOCK_N):
        offs = n0 + offs_n
        mask_n = offs < TOPK
        idx = tl.load(idx_base + offs, mask=mask_n, other=0)
        kp = K + b * sk_b + idx * sk_n
        score = tl.zeros([BLOCK_N], tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            kk = _fp8_to_f32(tl.load(kp[:, None] + d[None, :]), SOFTWARE_FP8)
            qq = _fp8_to_f32(tl.load(q_base + d), SOFTWARE_FP8)
            score += tl.sum(kk * qq[None, :], axis=1)
        tl.store(score_base + offs, score * scale, mask=mask_n)


@triton.jit
def _value_kernel(
    V, IDX, SINK, SCORES, OUT, RMAX, LSE, VS,
    Qn, TOPK,
    sv_b, sv_n, si_b, si_q, ss_b, sl_b, sl_q, so_b, so_q, sr_b,
    DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
):
    qpid = tl.program_id(0)
    vpid = tl.program_id(1)
    b = qpid // Qn
    qi = qpid % Qn

    idx_base = IDX + b * si_b + qi * si_q
    score_base = SCORES + b * sl_b + qi * sl_q
    dv = vpid * BLOCK_DV + tl.arange(0, BLOCK_DV)
    mask_dv = dv < DV

    m_i = tl.load(SINK + b * ss_b + qi).to(tl.float32)
    l_i = 1.0
    acc = tl.zeros([BLOCK_DV], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    neg_inf = float("-inf")

    for n0 in range(0, TOPK, BLOCK_N):
        offs = n0 + offs_n
        mask_n = offs < TOPK
        score = tl.load(score_base + offs, mask=mask_n, other=neg_inf)
        m_new = tl.maximum(m_i, tl.max(score, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)
        p = tl.where(mask_n, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        idx = tl.load(idx_base + offs, mask=mask_n, other=0)
        vp = V + b * sv_b + idx * sv_n
        # IDX already maps invalid TOPK lanes to the valid row zero and all
        # certified DV shapes are exact BLOCK_DV multiples.  Avoid a masked
        # FP8 load here: Triton cannot materialize the integer `other=0` for
        # Hopper E4M3, while the zeroed probability already removes invalid
        # lanes from the reduction.
        vv = _fp8_to_f32(
            tl.load(vp[:, None] + dv[None, :]),
            SOFTWARE_FP8,
        )
        acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
        m_i = m_new

    out_base = OUT + b * so_b + qi * so_q
    vscale = tl.load(VS).to(tl.float32)
    tl.store(out_base + dv, (acc * (vscale / l_i)).to(tl.bfloat16), mask=mask_dv)
    tl.store(RMAX + b * sr_b + qi, m_i, mask=vpid == 0)
    tl.store(LSE + b * sr_b + qi, m_i + tl.log(l_i), mask=vpid == 0)


class ModelNew(nn.Module):
    """Fused sparse MLA prefill (fp8 in, bf16 out) with a single Triton kernel."""

    def forward(self, q, k, v, indices, sink, q_scale, k_scale, v_scale):
        B, Qn, D = q.shape
        DV = v.shape[-1]
        topk = indices.shape[-1]

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        if not indices.is_contiguous():
            indices = indices.contiguous()
        if not sink.is_contiguous():
            sink = sink.contiguous()

        dev = q.device
        software_fp8 = torch.cuda.get_device_capability(dev)[0] < 9
        out = torch.empty((B, Qn, DV), device=dev, dtype=torch.bfloat16)
        stats_dtype = torch.bfloat16 if software_fp8 else torch.float32
        rmax = torch.empty((B, Qn), device=dev, dtype=stats_dtype)
        lse = torch.empty((B, Qn), device=dev, dtype=stats_dtype)

        if isinstance(q_scale, torch.Tensor):
            qs = q_scale.reshape(()).to(torch.float32)
        else:
            qs = torch.tensor(float(q_scale), device=dev, dtype=torch.float32)
        if isinstance(k_scale, torch.Tensor):
            ks = k_scale.reshape(()).to(torch.float32)
        else:
            ks = torch.tensor(float(k_scale), device=dev, dtype=torch.float32)
        if isinstance(v_scale, torch.Tensor):
            vs = v_scale.reshape(()).to(torch.float32)
        else:
            vs = torch.tensor(float(v_scale), device=dev, dtype=torch.float32)

        # block sizes
        if D % 128 == 0:
            BLOCK_D = 128
        elif D % 64 == 0:
            BLOCK_D = 64
        elif D % 32 == 0:
            BLOCK_D = 32
        else:
            BLOCK_D = 16

        if topk >= 64:
            BLOCK_N = 64
        elif topk >= 32:
            BLOCK_N = 32
        elif topk >= 16:
            BLOCK_N = 16
        else:
            BLOCK_N = 8

        q_arg = q.view(torch.uint8) if software_fp8 else q
        k_arg = k.view(torch.uint8) if software_fp8 else k
        v_arg = v.view(torch.uint8) if software_fp8 else v

        scores = torch.empty((B, Qn, topk), device=dev, dtype=torch.float32)
        _score_kernel[(B * Qn,)](
            q_arg, k_arg, indices, scores, qs, ks,
            Qn, topk,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            indices.stride(0), indices.stride(1), scores.stride(0), scores.stride(1),
            D=D, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            INV_SQRT_D=float(1.0 / math.sqrt(D)), SOFTWARE_FP8=software_fp8,
            num_warps=8, num_stages=2,
        )

        BLOCK_DV = 64
        VALUE_BLOCK_N = 64 if topk >= 64 else 32
        _value_kernel[(B * Qn, triton.cdiv(DV, BLOCK_DV))](
            v_arg, indices, sink, scores, out, rmax, lse, vs,
            Qn, topk,
            v.stride(0), v.stride(1), indices.stride(0), indices.stride(1),
            sink.stride(0), scores.stride(0), scores.stride(1),
            out.stride(0), out.stride(1), rmax.stride(0),
            DV=DV, BLOCK_N=VALUE_BLOCK_N, BLOCK_DV=BLOCK_DV, SOFTWARE_FP8=software_fp8,
            num_warps=2, num_stages=2,
        )

        return out, rmax, lse
