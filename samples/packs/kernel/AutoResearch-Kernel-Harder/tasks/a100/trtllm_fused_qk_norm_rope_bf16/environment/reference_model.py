# reference_model.py
#
# task_id      : trtllm_fused_qk_norm_rope_bf16
# domain       : attention        task_type : fusion        task_mode : tensor
# track        : fusion_and_layout
# calibration  : a100_pcie_80gb_cu124_torch260 (NVIDIA A100 80GB PCIe, sm_80, ampere)
# precision    : inputs bf16, all reductions/rotations accumulated in fp32, single bf16 rounding
#
# WHAT THIS TASK IS
#   Visual-transformer attention preprocessing on a packed QKV activation tensor
#   [B, S, 3, H, D] (bf16, contiguous).  For every (batch, token) row:
#       1. flatten Q over the *complete* H*D feature vector, RMS-normalize it in fp32
#          (eps = 1e-6) and rescale with a per-(h,d) weight  q_weight[H, D];
#       2. same for K with its own weight  k_weight[H, D];
#       3. apply rotary position embedding to the normalized Q and K in fp32, using the
#          per-position table cos/sin of shape [S, D/2];
#       4. copy V through *bit-for-bit* (no numeric conversion at all);
#       5. write one packed contiguous bf16 [B, S, 3, H, D] result.
#   The naive reference below materializes every intermediate in global memory; a real
#   fused kernel keeps the row resident in registers/SMEM between the norm and the RoPE.
#   This is a generic single-GPU task distilled from the NVIDIA/TensorRT-LLM attention
#   seed (PR 13052).  It does NOT reproduce, wrap or package the upstream CUDA code.
#
# NOTE ON THE ARGUMENT LIST
#   Cases 1 and 6 share the exact physical shape (1, 4096, 12, 128) and differ *only* in
#   the RoPE convention, so the convention cannot be recovered from any tensor's shape or
#   from the public one-tuple at call time.  It is therefore carried as a 6th forward
#   argument: a plain Python int (0 = 'interleaved', 1 = 'rotate_half').  It is not a
#   tensor, so no host/device mixing is introduced, and under dynamic=False torch.compile
#   simply specializes on it.  `baseline(qkv, q_weight, k_weight, cos, sin, ...)` keeps
#   the documented 5-tensor prefix and accepts the mode as an optional 6th positional.

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------------------
# public shape domain : the declared axis is exactly (case_id,)
# CASES[case_id] = (B, S, H, D, rope_mode)
# ---------------------------------------------------------------------------------------
CASES = {
    0: (1, 256, 12, 128, 'interleaved'),
    1: (1, 4096, 12, 128, 'interleaved'),
    2: (1, 256, 40, 128, 'interleaved'),
    3: (1, 4096, 40, 128, 'interleaved'),
    4: (1, 16384, 40, 128, 'interleaved'),
    5: (2, 512, 12, 128, 'rotate_half'),
    6: (1, 4096, 12, 128, 'rotate_half'),
    7: (2, 1024, 40, 128, 'rotate_half'),
    8: (1, 4096, 40, 128, 'rotate_half'),
    9: (1, 16384, 40, 128, 'rotate_half'),
}

WORKLOAD = [(0,), (1,), (2,), (3,), (4,)]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-2
TASK_ATOL = 1e-2
GRAD_INPUTS = []
OUTPUT_INDEX = 0

BASELINE_CONTRACT = {
    "input_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "accumulation_dtype": "float32",
    "allow_tf32": False,
    "note": (
        "Packed qkv [B,S,3,H,D] bf16 in, packed bf16 out. RMSNorm (eps=1e-6) is taken over "
        "the whole H*D feature vector of a (batch, token) row for Q and K separately, with "
        "per-(h,d) bf16 weights upcast to fp32; the norm reduction, the weight multiply and "
        "the rotary rotation all accumulate in fp32 and the result is rounded to bf16 exactly "
        "once, at the very end. No matmuls are involved, so TF32 is irrelevant and must stay "
        "off. HARD REQUIREMENT: the V slice out[:, :, 2] must be a bitwise copy of the input "
        "V slice -- no upcast/downcast round trip, no reordering, no flush-to-zero; it is "
        "checked with an integer bit-pattern comparison, not with allclose."
    ),
}

_EPS = 1e-6
_ROPE_MODES = ('interleaved', 'rotate_half')


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _case_id(shape):
    """Accept (case_id,), [case_id] or a bare int."""
    if isinstance(shape, (tuple, list)):
        if len(shape) != 1:
            raise ValueError(f"shape must be a 1-tuple (case_id,), got {shape!r}")
        return int(shape[0])
    return int(shape)


def _case(shape):
    cid = _case_id(shape)
    if cid not in CASES:
        raise KeyError(f"unknown case_id {cid}; legal ids are {sorted(CASES)}")
    return CASES[cid]


def _mode_int(mode):
    """Normalize the rope-mode argument (int / str / 0-d tensor / None) to 0 or 1."""
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


def _rms_norm_rows(x_f32, weight_f32, eps):
    """Naive fp32 RMSNorm over the last (flattened H*D) axis, then per-(h,d) rescale."""
    ms = x_f32.pow(2).mean(dim=-1, keepdim=True)          # [B, S, 1]
    inv = torch.rsqrt(ms + eps)                           # [B, S, 1]
    return (x_f32 * inv) * weight_f32                     # [B, S, H*D]


def _apply_rope(x_f32, cos_f32, sin_f32, mode_i):
    """
    x   : [B, S, H, D] fp32
    cos/sin : [S, D/2] fp32
    mode_i == 0 -> 'interleaved' : pairs are (d, d+1)      (GPT-J style)
    mode_i == 1 -> 'rotate_half' : pairs are (d, d + D/2)  (NeoX / HF style)
    """
    B, S, H, D = x_f32.shape
    half = D // 2
    c = cos_f32.view(1, S, 1, half)
    s = sin_f32.view(1, S, 1, half)
    if mode_i == 0:
        x1 = x_f32[..., 0::2]
        x2 = x_f32[..., 1::2]
        o1 = x1 * c - x2 * s
        o2 = x1 * s + x2 * c
        return torch.stack((o1, o2), dim=-1).reshape(B, S, H, D)
    x1 = x_f32[..., :half]
    x2 = x_f32[..., half:]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.cat((o1, o2), dim=-1)


def _qk_norm_rope_composite(qkv, q_weight, k_weight, cos, sin, mode_i, eps=_EPS):
    """
    The plain-PyTorch composite: RMSNorm(Q) -> RoPE(Q), RMSNorm(K) -> RoPE(K), V passthrough.
    Every intermediate is a full global-memory tensor -- this is the un-fused reference and
    also the body that the strong library (torch.compile) is asked to fuse.
    """
    B, S, three, H, D = qkv.shape
    assert three == 3, "qkv must be packed as [B, S, 3, H, D]"
    HD = H * D

    q = qkv[:, :, 0].float().reshape(B, S, HD)            # [B, S, H*D] fp32
    k = qkv[:, :, 1].float().reshape(B, S, HD)
    v = qkv[:, :, 2]                                      # bf16, untouched

    qw = q_weight.float().reshape(1, 1, HD)
    kw = k_weight.float().reshape(1, 1, HD)

    q = _rms_norm_rows(q, qw, eps).reshape(B, S, H, D)
    k = _rms_norm_rows(k, kw, eps).reshape(B, S, H, D)

    q = _apply_rope(q, cos, sin, mode_i)
    k = _apply_rope(k, cos, sin, mode_i)

    # single rounding to bf16 for Q/K; V is copied bit-for-bit (bf16 -> bf16).
    out = torch.stack((q.to(torch.bfloat16), k.to(torch.bfloat16), v), dim=2)
    return out.contiguous()


# ---------------------------------------------------------------------------------------
# the NAIVE correctness oracle / timing denominator
# ---------------------------------------------------------------------------------------
class Model(nn.Module):
    """Obviously-correct, un-fused reference. eps is fixed; no constructor inputs."""

    def __init__(self):
        super().__init__()
        self.eps = _EPS

    def forward(self, qkv, q_weight, k_weight, cos, sin, rope_mode=0):
        return _qk_norm_rope_composite(
            qkv, q_weight, k_weight, cos, sin, _mode_int(rope_mode), self.eps
        )


def get_init_inputs():
    return []


# ---------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------
def make_inputs(shape, device, seed=0):
    B, S, H, D, mode = _case(shape)
    dev = torch.device(device)
    half = D // 2
    cid = _case_id(shape)
    seed = int(seed)

    gen = torch.Generator(device=dev)
    gen.manual_seed(1234 + 977 * cid + 7919 * seed)

    # packed activations: nonconstant rows, unit-ish scale so bf16 keeps ~3 decimal digits.
    tmp = torch.randn((B, S, 3, H, D), generator=gen, device=dev, dtype=torch.float32)
    qkv = tmp.to(torch.bfloat16).contiguous()
    del tmp

    # strictly positive norm weights in [0.75, 1.25]
    q_weight = (0.75 + 0.5 * torch.rand((H, D), generator=gen, device=dev,
                                        dtype=torch.float32)).to(torch.bfloat16).contiguous()
    k_weight = (0.75 + 0.5 * torch.rand((H, D), generator=gen, device=dev,
                                        dtype=torch.float32)).to(torch.bfloat16).contiguous()

    # exact unit-circle (cos, sin) pairs; the seed shifts the phase so data really changes.
    pos = torch.arange(S, device=dev, dtype=torch.float32)
    inv_freq = 10000.0 ** (-(torch.arange(half, device=dev, dtype=torch.float32) * 2.0 / D))
    phase = 0.017 * float((seed % 13) + 1)
    ang = pos.unsqueeze(1) * inv_freq.unsqueeze(0) + phase        # [S, half]
    cos = torch.cos(ang).contiguous()
    sin = torch.sin(ang).contiguous()

    return [qkv, q_weight, k_weight, cos, sin, _ROPE_MODES.index(mode)]


# ---------------------------------------------------------------------------------------
# hidden-shape tripwire : unseen case ids 5..9
# ---------------------------------------------------------------------------------------
def sample_shapes(rng, n):
    pool = [5, 6, 7, 8, 9]
    try:
        keyed = sorted((float(rng.random()), c) for c in pool)
        order = [c for _, c in keyed]
    except Exception:
        order = pool
    n = max(0, min(int(n), len(order)))
    return [(c,) for c in order[:n]]


def reduction_length(shape, *args, **kwargs):
    """Length of the RMSNorm reduction: the full heads-by-head_dim feature vector."""
    B, S, H, D, _ = _case(shape)
    return H * D


# ---------------------------------------------------------------------------------------
# metamorphic invariants -- computed from (out, ins) only, Model is NEVER re-run
# ---------------------------------------------------------------------------------------
def invariants(out, ins, shape):
    res = []
    if isinstance(out, (tuple, list)):
        out = out[OUTPUT_INDEX]

    qkv, q_weight, k_weight, cos, sin = ins[0], ins[1], ins[2], ins[3], ins[4]
    mode_i = _mode_int(ins[5]) if len(ins) > 5 else 0
    B, S, _three, H, D = qkv.shape
    half = D // 2

    # ---- 1. shape / dtype / finiteness -------------------------------------------------
    ok_shape = tuple(out.shape) == (B, S, 3, H, D)
    ok_dtype = out.dtype == torch.bfloat16
    ok_fin = bool(torch.isfinite(out.float()).all())
    res.append((
        "shape_dtype_finite",
        bool(ok_shape and ok_dtype and ok_fin),
        f"shape={tuple(out.shape)} expected={(B, S, 3, H, D)} dtype={out.dtype} finite={ok_fin}",
    ))

    # ---- 2. V must survive bit-for-bit (integer bit-pattern comparison) -----------------
    v_out_bits = out[:, :, 2].contiguous().view(torch.int16)
    v_in_bits = qkv[:, :, 2].contiguous().view(torch.int16)
    v_same = bool(torch.equal(v_out_bits, v_in_bits))
    n_bad = int((v_out_bits != v_in_bits).sum().item()) if not v_same else 0
    res.append((
        "v_slice_bitwise_identical",
        v_same,
        f"mismatching bf16 words in V: {n_bad} / {v_in_bits.numel()}",
    ))

    # ---- subsample of (b, s) rows, spread over the whole sequence -----------------------
    step = max(1, S // 64)
    s_idx = torch.arange(0, S, step, device=out.device)[:64]
    n = int(s_idx.numel())

    worst_pair = 0.0
    worst_energy = 0.0
    for off, w in ((0, q_weight), (1, k_weight)):
        x = qkv[:, :, off].index_select(1, s_idx).float()      # [B, n, H, D] (exact bf16 vals)
        y = out[:, :, off].index_select(1, s_idx).float()      # [B, n, H, D]
        wf = w.float().view(1, 1, H, D)
        xw = x * wf                                            # weighted, *before* normalization

        if mode_i == 0:
            p_out = y[..., 0::2].pow(2) + y[..., 1::2].pow(2)
            p_in = xw[..., 0::2].pow(2) + xw[..., 1::2].pow(2)
        else:
            p_out = y[..., :half].pow(2) + y[..., half:].pow(2)
            p_in = xw[..., :half].pow(2) + xw[..., half:].pow(2)

        # 3. a rotation is norm-preserving *per pair*, and normalization is a single
        #    per-row scalar => p_out / p_in must be the SAME constant for every (h, j)
        #    of a row.  This uses no knowledge of the norm formula at all.
        p_out_f = p_out.reshape(B, n, -1)
        p_in_f = p_in.reshape(B, n, -1)
        thr = p_in_f.mean(dim=-1, keepdim=True) * 1e-2
        ratio = p_out_f / p_in_f.clamp_min(1e-20)
        masked = torch.where(p_in_f > thr, ratio, torch.full_like(ratio, float("nan")))
        med = torch.nanmedian(masked, dim=-1, keepdim=True).values.clamp_min(1e-20)
        dev = torch.nan_to_num((masked / med - 1.0).abs(), nan=0.0)
        worst_pair = max(worst_pair, float(dev.max().item()))

        # 4. row energy: mean(out^2) over the row must equal mean((w*x)^2)/(mean(x^2)+eps),
        #    i.e. the inverse-weight normalized row has unit mean-square before rotation.
        ms = x.reshape(B, n, -1).pow(2).mean(dim=-1)
        e_out = y.reshape(B, n, -1).pow(2).mean(dim=-1)
        e_pred = xw.reshape(B, n, -1).pow(2).mean(dim=-1) / (ms + _EPS)
        rel = (e_out / e_pred.clamp_min(1e-20) - 1.0).abs()
        worst_energy = max(worst_energy, float(rel.max().item()))

    res.append((
        "rope_pair_norm_preserved",
        worst_pair < 6e-2,
        f"max relative spread of per-pair |out|^2/|w*x|^2 within a row = {worst_pair:.3e} (tol 6e-2)",
    ))
    res.append((
        "rmsnorm_row_energy_unit",
        worst_energy < 2e-2,
        f"max |mean(out^2)*(mean(x^2)+eps)/mean((w*x)^2) - 1| = {worst_energy:.3e} (tol 2e-2)",
    ))
    return res


# ---------------------------------------------------------------------------------------
# independent oracle : same math, deliberately different code path
#   - norm via torch.linalg.vector_norm (fp64 when affordable) instead of pow/mean/rsqrt
#   - rotation via complex multiplication instead of explicit cos/sin cross terms
# ---------------------------------------------------------------------------------------
def independent_oracle(ins, shape):
    qkv, q_weight, k_weight, cos, sin = ins[0], ins[1], ins[2], ins[3], ins[4]
    mode_i = _mode_int(ins[5]) if len(ins) > 5 else 0
    B, S, _three, H, D = qkv.shape
    half = D // 2
    HD = H * D

    hi = torch.float64 if qkv.numel() <= 3 * 10 ** 7 else torch.float32
    rot = torch.complex(cos.to(hi), sin.to(hi)).reshape(1, S, 1, half)

    outs = []
    for off, w in ((0, q_weight), (1, k_weight)):
        x = qkv[:, :, off].to(hi).reshape(B, S, HD)
        nrm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)          # ||x||_2
        # rsqrt(mean(x^2) + eps) == sqrt(HD) / sqrt(||x||^2 + HD*eps)
        scale = math.sqrt(HD) / torch.sqrt(nrm * nrm + HD * _EPS)
        z = (x * scale) * w.to(hi).reshape(1, 1, HD)
        z = z.reshape(B, S, H, D)

        if mode_i == 0:
            zc = torch.view_as_complex(z.reshape(B, S, H, half, 2).contiguous())
            r = torch.view_as_real(zc * rot).reshape(B, S, H, D)
        else:
            zc = torch.complex(z[..., :half].contiguous(), z[..., half:].contiguous())
            zc = zc * rot
            r = torch.cat((zc.real, zc.imag), dim=-1)
        outs.append(r.to(torch.bfloat16))

    return torch.stack((outs[0], outs[1], qkv[:, :, 2]), dim=2).contiguous()

MEM_WEIGHT = 0.7
