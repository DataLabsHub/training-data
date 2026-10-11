# reference_model.py
# Task: h200_fp8_sparse_mla_prefill
#
# Real hotspot: sparse multi-latent-attention (MLA) prefill on H200. Q/K/V arrive as
# FP8 (e4m3) tensors with positive per-tensor dequant scales, plus an int32 index
# tensor [B, Q, topk] naming, for every query row, the KV positions that survived the
# sparse selector. The kernel must gather those rows, run a numerically stable softmax
# in FP32, and return three things: the BF16 attention output [B, Q, 512], the FP32
# running row max, and the FP32 log-sum-exp. An optional (always supplied here) finite
# sink logit participates in the max and the LSE denominator but contributes nothing to
# the P@V numerator.
#
# The Model below is the NAIVE oracle: full dequant to FP32, materialize the gathered
# [B, Q, topk, D] tensors, batched matmul, softmax, batched matmul. Obviously correct,
# heavily memory bound, and therefore a fair timing denominator.
#
# `baseline(...)` is the honest frontier: a torch.compile'd gather-attention written so
# inductor can fuse the gather into the reduction (no [B, Q, topk, D] materialization),
# BF16 loads, FP32 accumulation, chunked over the query axis.
#
# Optimization track: algorithmic_reformulation.
# Calibration target: h200_sm90_cuda129_torch280_nv2505 (NVIDIA H200, sm90, hopper).

import math

import torch
import torch.nn as nn

# case_id -> (B, Q, KV, topk, d_qk). d_v is always 512 and there is a single KV head.
CASES = {
    0: (1, 64, 512, 32, 512),
    1: (1, 128, 2048, 64, 576),
    2: (2, 256, 4096, 128, 512),
    3: (1, 512, 8192, 256, 576),
    4: (4, 128, 16384, 512, 512),
    5: (3, 96, 3072, 96, 576),
    6: (2, 768, 12288, 384, 512),
    7: (8, 64, 32768, 1024, 576),
    8: (1, 2048, 32768, 2048, 512),
    9: (4, 1024, 16384, 512, 576),
}

D_V = 512
FP8_MAX = 448.0

# Public axis order: ['case_id'] -> one-tuples only.
WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-2
TASK_ATOL = 1e-2
GRAD_INPUTS = []
OUTPUT_INDICES = [0, 1, 2]

BASELINE_CONTRACT = {
    'input_dtype': 'float8_e4m3fn',
    'output_dtype': 'bfloat16',
    'accumulation_dtype': 'float32',
    'allow_tf32': False,
    'note': (
        'Q/K/V are float8_e4m3fn with positive fp32 per-tensor scales; indices are int32 '
        '[B, Q, topk] with no duplicates per row; sink logits are fp32 [B, Q]. Scores and '
        'softmax must accumulate in fp32 with the max subtracted before exp. Outputs: '
        'index 0 = bfloat16 [B, Q, 512] attention values, index 1 = float32 [B, Q] row max '
        '(including the sink logit), index 2 = float32 [B, Q] natural-log LSE (including the '
        'sink term). The sink logit enters row_max and LSE but never the P@V numerator. '
        'TF32 must stay off so the fp32 reference path is not silently degraded.'
    ),
}


class Model(nn.Module):
    """Naive plain-PyTorch sparse MLA prefill. Correctness anchor, not optimized."""

    def forward(self, q, k, v, indices, sink, q_scale, k_scale, v_scale):
        B, Q, D = q.shape
        topk = indices.shape[-1]
        Dv = v.shape[-1]

        qf = q.to(torch.float32) * q_scale
        kf = k.to(torch.float32) * k_scale
        vf = v.to(torch.float32) * v_scale

        idx = indices.to(torch.int64)
        b_ar = torch.arange(B, device=q.device).view(B, 1, 1)

        k_sel = kf[b_ar, idx]                      # [B, Q, topk, D]
        v_sel = vf[b_ar, idx]                      # [B, Q, topk, Dv]

        softmax_scale = 1.0 / math.sqrt(D)
        logits = torch.matmul(k_sel, qf.reshape(B, Q, D, 1)).reshape(B, Q, topk)
        logits = logits * softmax_scale

        sink_f = sink.to(torch.float32)
        row_max = torch.maximum(logits.amax(dim=-1), sink_f)          # [B, Q]
        p = torch.exp(logits - row_max.unsqueeze(-1))                 # [B, Q, topk]
        denom = p.sum(dim=-1) + torch.exp(sink_f - row_max)           # [B, Q]
        lse = row_max + torch.log(denom)

        num = torch.matmul(p.unsqueeze(-2), v_sel).reshape(B, Q, Dv)  # sink excluded here
        out = num / denom.unsqueeze(-1)

        return out.to(torch.bfloat16), row_max.to(torch.float32), lse.to(torch.float32)


def get_init_inputs():
    return []


def _quantize_fp8(x):
    """Symmetric per-tensor fp8 e4m3 quantization. Returns (fp8 tensor, 0-dim fp32 scale)."""
    amax = x.abs().amax().clamp_min(1e-6)
    scale = (amax / FP8_MAX).to(torch.float32)
    q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def make_inputs(shape, device, seed=0):
    case_id = int(shape[0])
    B, Q, KV, topk, D = CASES[case_id]
    dev = torch.device(device)
    if topk > KV:
        raise ValueError('topk must not exceed KV length')

    g = torch.Generator(device=dev).manual_seed(int(seed) * 9973 + 12345 + case_id)

    qf = torch.randn(B, Q, D, generator=g, device=dev, dtype=torch.float32)
    kf = torch.randn(B, KV, D, generator=g, device=dev, dtype=torch.float32)
    vf = torch.randn(B, KV, D_V, generator=g, device=dev, dtype=torch.float32)

    q8, q_scale = _quantize_fp8(qf)
    k8, k_scale = _quantize_fp8(kf)
    v8, v_scale = _quantize_fp8(vf)

    # Duplicate-free selection: one random position drawn from each of topk disjoint blocks.
    stride = KV // topk
    base = (torch.arange(topk, device=dev, dtype=torch.int64) * stride).view(1, 1, topk)
    off = torch.randint(0, stride, (B, Q, topk), generator=g, device=dev, dtype=torch.int64)
    idx = base + off
    if D == 576:  # exercise the unsorted-index path on the 576-dim cases
        perm = torch.argsort(torch.rand(B, Q, topk, generator=g, device=dev), dim=-1)
        idx = torch.gather(idx, -1, perm)
    indices = idx.to(torch.int32)

    sink = torch.randn(B, Q, generator=g, device=dev, dtype=torch.float32) * 0.5

    return [q8, k8, v8, indices, sink, q_scale, k_scale, v_scale]


def sample_shapes(rng, n):
    """Unseen case ids from the declared domain (WORKLOAD covers 0..4)."""
    pool = [(5,), (6,), (7,), (8,), (9,)]
    picks = []
    for _ in range(int(n)):
        try:
            j = int(rng.randrange(len(pool)))
        except AttributeError:
            j = int(rng.integers(len(pool)))
        picks.append(pool[j])
    return picks


def reduction_length(shape):
    """Softmax accumulation depth: the selected keys plus the sink logit."""
    _, _, _, topk, _ = CASES[int(shape[0])]
    return int(topk) + 1


def invariants(out, ins, shape):
    """Metamorphic checks. None of these rerun Model; the fp64 spot check is a fully
    independent closed-form evaluation of a handful of rows."""
    case_id = int(shape[0])
    B, Q, KV, topk, D = CASES[case_id]
    q8, k8, v8, indices, sink, q_scale, k_scale, v_scale = ins[:8]
    o, rmax, lse = out[0], out[1], out[2]
    res = []

    # 1) shapes / dtypes / finiteness
    shape_ok = (
        tuple(o.shape) == (B, Q, D_V)
        and tuple(rmax.shape) == (B, Q)
        and tuple(lse.shape) == (B, Q)
    )
    dtype_ok = (
        o.dtype == torch.bfloat16
        and rmax.dtype == torch.float32
        and lse.dtype == torch.float32
    )
    finite_ok = bool(
        torch.isfinite(o).all().item()
        and torch.isfinite(rmax).all().item()
        and torch.isfinite(lse).all().item()
    )
    res.append((
        'shapes_dtypes_finite',
        bool(shape_ok and dtype_ok and finite_ok),
        'shapes={} dtypes=({},{},{}) finite={}'.format(
            (tuple(o.shape), tuple(rmax.shape), tuple(lse.shape)),
            o.dtype, rmax.dtype, lse.dtype, finite_ok),
    ))

    # 2) index set is legal: in range and duplicate free per query row
    idx = indices.to(torch.int64)
    in_range = bool((idx >= 0).all().item() and (idx < KV).all().item())
    srt, _ = torch.sort(idx, dim=-1)
    uniq = True if topk < 2 else bool((srt[..., 1:] - srt[..., :-1] > 0).all().item())
    res.append((
        'indices_in_range_and_unique',
        bool(in_range and uniq),
        'in_range={} unique_per_row={} KV={} topk={}'.format(in_range, uniq, KV, topk),
    ))

    # 3) algebraic brackets: row_max <= LSE <= row_max + log(topk+1), and the output is a
    #    sub-convex combination of the selected V rows so |out| <= max|V_dequant|.
    gap = (lse - rmax).to(torch.float64)
    gap_lo = float(gap.min().item())
    gap_hi = float(gap.max().item())
    bracket_ok = gap_lo >= -1e-4 and gap_hi <= math.log(topk + 1.0) + 1e-3
    vmax = float((v8.to(torch.float32) * v_scale).abs().amax().item())
    omax = float(o.to(torch.float32).abs().amax().item())
    bound_ok = omax <= vmax * 1.02 + 1e-3
    res.append((
        'max_le_lse_and_convex_value_bound',
        bool(bracket_ok and bound_ok),
        'lse-max in [{:.3e},{:.3e}] limit={:.3e}; |out|max={:.4f} <= |V|max={:.4f}'.format(
            gap_lo, gap_hi, math.log(topk + 1.0), omax, vmax),
    ))

    # 4) fp64 closed form on a few deterministic rows, gathered explicitly
    rows = sorted({(0, 0), (B - 1, Q - 1), (B // 2, Q // 3), (0, Q // 2)})
    inv = 1.0 / math.sqrt(D)
    qs, ks, vs = q_scale.double(), k_scale.double(), v_scale.double()
    worst_o = worst_m = worst_l = 0.0
    for b, qi in rows:
        sel = idx[b, qi]
        kk = k8[b, sel].double() * ks                     # [topk, D]
        vv = v8[b, sel].double() * vs                     # [topk, Dv]
        qq = q8[b, qi].double() * qs                      # [D]
        lg = torch.matmul(kk, qq) * inv
        sk = sink[b, qi].double()
        m = torch.maximum(lg.max(), sk)
        p = torch.exp(lg - m)
        s = p.sum() + torch.exp(sk - m)
        ref_o = torch.matmul(p, vv) / s
        ref_l = m + torch.log(s)
        scale_o = max(1.0, float(ref_o.abs().max().item()))
        scale_m = max(1.0, abs(float(m.item())))
        worst_o = max(worst_o, float((o[b, qi].double() - ref_o).abs().max().item()) / scale_o)
        worst_m = max(worst_m, abs(float(rmax[b, qi].double().item() - m.item())) / scale_m)
        worst_l = max(worst_l, abs(float(lse[b, qi].double().item() - ref_l.item())) / scale_m)
    spot_ok = worst_o <= 3e-2 and worst_m <= 1e-3 and worst_l <= 1e-3
    res.append((
        'fp64_closed_form_spot_rows',
        bool(spot_ok),
        'rows={} rel_err out={:.3e} max={:.3e} lse={:.3e}'.format(
            rows, worst_o, worst_m, worst_l),
    ))

    return res


def independent_oracle(ins, shape):
    """Fully independent FP64 evaluation over explicitly gathered selected rows.
    Chunked over (batch, query) so the fp64 temporaries stay small."""
    q8, k8, v8, indices, sink, q_scale, k_scale, v_scale = ins[:8]
    case_id = int(shape[0])
    B, Q, KV, topk, D = CASES[case_id]
    Dv = v8.shape[-1]
    dev = q8.device
    inv = 1.0 / math.sqrt(D)

    kd = k8.double() * k_scale.double()
    vd = v8.double() * v_scale.double()

    out = torch.empty(B, Q, Dv, device=dev, dtype=torch.float64)
    rmax = torch.empty(B, Q, device=dev, dtype=torch.float64)
    lse = torch.empty(B, Q, device=dev, dtype=torch.float64)

    chunk = max(1, int((1 << 21) // max(1, topk * max(D, Dv))))
    for b in range(B):
        for s in range(0, Q, chunk):
            e = min(Q, s + chunk)
            sel = indices[b, s:e].to(torch.int64)          # [qc, topk]
            k_sel = kd[b][sel]                             # [qc, topk, D]
            v_sel = vd[b][sel]                             # [qc, topk, Dv]
            qq = q8[b, s:e].double() * q_scale.double()    # [qc, D]
            lg = torch.matmul(k_sel, qq.unsqueeze(-1)).squeeze(-1) * inv
            sk = sink[b, s:e].double()
            m = torch.maximum(lg.amax(dim=-1), sk)
            p = torch.exp(lg - m.unsqueeze(-1))
            den = p.sum(dim=-1) + torch.exp(sk - m)
            out[b, s:e] = torch.matmul(p.unsqueeze(-2), v_sel).squeeze(-2) / den.unsqueeze(-1)
            rmax[b, s:e] = m
            lse[b, s:e] = m + torch.log(den)

    return out.to(torch.bfloat16), rmax.to(torch.float32), lse.to(torch.float32)


def _mla_chunk(q_blk, k_all, v_all, idx_blk, sink_blk, softmax_scale):
    """One query chunk, written as gather -> elementwise -> reduction so inductor can fuse
    the gather into the reduction instead of materializing [B, qc, topk, D]."""
    B = q_blk.shape[0]
    b_ar = torch.arange(B, device=q_blk.device).view(B, 1, 1)
    idx = idx_blk.to(torch.int64)

    k_sel = k_all[b_ar, idx]                                        # [B, qc, topk, D]
    logits = (k_sel.to(torch.float32) * q_blk.to(torch.float32).unsqueeze(2)).sum(-1)
    logits = logits * softmax_scale

    m = torch.maximum(logits.amax(dim=-1), sink_blk)
    p = torch.exp(logits - m.unsqueeze(-1))
    den = p.sum(dim=-1) + torch.exp(sink_blk - m)

    v_sel = v_all[b_ar, idx]                                        # [B, qc, topk, Dv]
    num = (p.unsqueeze(-1) * v_sel.to(torch.float32)).sum(dim=-2)
    o = num / den.unsqueeze(-1)

    return o.to(torch.bfloat16), m, m + torch.log(den)


_COMPILED = {}


def _get_compiled():
    fn = _COMPILED.get('chunk')
    if fn is None:
        fn = torch.compile(_mla_chunk, dynamic=False, fullgraph=True)
        _COMPILED['chunk'] = fn
    return fn


def baseline(*args, init_args=None):
    """Strong library frontier: torch.compile'd sparse MLA gather-attention.
    BF16 KV loads, FP32 softmax accumulation, query-chunked to bound the live logits."""
    q, k, v, indices, sink, q_scale, k_scale, v_scale = args[:8]
    B, Q, D = q.shape
    topk = indices.shape[-1]
    softmax_scale = 1.0 / math.sqrt(D)

    q_bf = (q.to(torch.float32) * q_scale).to(torch.bfloat16)
    k_bf = (k.to(torch.float32) * k_scale).to(torch.bfloat16)
    v_bf = (v.to(torch.float32) * v_scale).to(torch.bfloat16)
    sink_f = sink.to(torch.float32)

    fn = _get_compiled()
    qc = max(1, min(Q, (1 << 20) // max(1, topk)))
    if Q % qc:
        qc = Q  # keep chunks uniform so inductor compiles a single shape

    outs, maxs, lses = [], [], []
    for s in range(0, Q, qc):
        e = min(Q, s + qc)
        o, m, l = fn(q_bf[:, s:e], k_bf, v_bf, indices[:, s:e], sink_f[:, s:e], softmax_scale)
        outs.append(o)
        maxs.append(m)
        lses.append(l)

    if len(outs) == 1:
        return outs[0], maxs[0].to(torch.float32), lses[0].to(torch.float32)
    return (
        torch.cat(outs, dim=1),
        torch.cat(maxs, dim=1).to(torch.float32),
        torch.cat(lses, dim=1).to(torch.float32),
    )
