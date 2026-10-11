# reference_model.py
#
# task_id  : nvidia_tensorrt_llm_normalization_pr9969
# domain   : normalization      task_type : fusion       task_mode : tensor
# hotspot  : A100 fwd gated grouped-RMSNorm fusion for TensorRT-LLM AutoDeploy.
#            grouped fp32 RMS statistics + learned per-channel scale + post-norm
#            SiLU gate, fused into one operator over bf16 hidden states.
# frontier : torch.compile over the identical fixed composite (see `baseline`).
# track    : algorithmic_reformulation
# target   : a100_pcie_80gb_cu124_torch260 (NVIDIA A100 80GB PCIe, sm_80, ampere)
#
# The packaged forward branch is FIXED: group_size=128, mode='post', eps=1e-6,
# gate always supplied.  effective_G = 128 when H % 128 == 0 else H (the
# non-divisible fallback collapses the grouping to a single whole-row group).

import math

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# shape domain
# --------------------------------------------------------------------------- #
# The public "shape" tuple is (case_id,).  It maps to a (T, H) hidden-state
# problem through CASES below.
CASES = {
    0:  (4096, 4096),
    1:  (8192, 2048),
    2:  (16384, 4096),
    3:  (32768, 1024),
    4:  (128, 128),
    5:  (6144, 16384),
    6:  (12288, 8192),
    7:  (24576, 4096),
    8:  (32768, 3072),
    9:  (20480, 2048),
    10: (6912, 14336),
    11: (24576, 2056),     # H % 128 != 0  -> whole-row fallback group
}

# case 11 is in the public workload so the non-128-divisible fallback is always
# scored; 4..10 are held out for the hidden-shape tripwire.
WORKLOAD = [(0,), (1,), (2,), (3,), (11,)]
_UNSEEN_CASES = [4, 5, 6, 7, 8, 9, 10]

TASK_MODE = 'tensor'
TASK_RTOL = 2e-2
TASK_ATOL = 1e-2
GRAD_INPUTS = []          # forward-only task
OUTPUT_INDEX = 0

GROUP_SIZE = 128
GATE_MODE = 'post'
EPS = 1e-6

BASELINE_CONTRACT = {
    'input_dtype': 'bf16',
    'output_dtype': 'bf16',
    'accumulation_dtype': 'fp32',
    'allow_tf32': False,
    'note': (
        'Hidden states x (T,H), gate (T,H) and per-channel weight (H,) are bf16 '
        'and contiguous. Every reduction (mean of x^2), the rsqrt, the per-channel '
        'weight scaling and the SiLU gate multiply are performed in fp32; the '
        'result is rounded to bf16 exactly once, at the store. Grouping is '
        'effective_G = 128 if H % 128 == 0 else H, applied along the last axis. '
        'Gating is post-normalization: out = x * rsqrt(mean_group(x^2)+1e-6) * '
        'weight * silu(gate). No TF32, no fp16 accumulation, no intermediate '
        'bf16 rounding of the normalized value before the gate.'
    ),
}


def _dims(shape):
    """(case_id,) -> (T, H)."""
    if isinstance(shape, (list, tuple)):
        case_id = int(shape[0])
    else:
        case_id = int(shape)
    if case_id not in CASES:
        raise KeyError(f'unknown case_id {case_id}; legal ids are {sorted(CASES)}')
    return CASES[case_id]


def _effective_group(H, group_size=GROUP_SIZE):
    """Packaged grouping rule: 128 lanes per group, else one group per row."""
    return group_size if (H % group_size == 0) else H


def reduction_length(shape):
    """Length of the innermost reduction actually performed for this shape."""
    _, H = _dims(shape)
    return _effective_group(H)


# --------------------------------------------------------------------------- #
# NAIVE reference  (the correctness Oracle and the timing denominator)
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """Plain-PyTorch gated grouped RMSNorm.  Obviously correct, not optimized.

    out = (x / sqrt(mean_group(x^2) + eps)) * weight * silu(gate)

    Materializes every intermediate; this is intentional -- it is the anchor a
    real fused kernel is supposed to beat.
    """

    def __init__(self, group_size=GROUP_SIZE, mode=GATE_MODE, eps=EPS):
        super().__init__()
        assert int(group_size) == GROUP_SIZE, 'packaged branch fixes group_size=128'
        assert str(mode) == GATE_MODE, "packaged branch fixes post-normalization gating"
        assert abs(float(eps) - EPS) <= 0.0, 'packaged branch fixes eps=1e-6'
        self.group_size = int(group_size)
        self.mode = str(mode)
        self.eps = float(eps)

    def forward(self, x, weight, gate):
        T, H = x.shape
        G = _effective_group(H, self.group_size)
        n_groups = H // G

        xf = x.float()                                   # fp32 accumulation domain
        xg = xf.reshape(T, n_groups, G)
        mean_sq = (xg * xg).mean(dim=-1, keepdim=True)   # fp32 grouped statistic
        xn = xg / torch.sqrt(mean_sq + self.eps)
        xn = xn.reshape(T, H)

        gf = gate.float()
        silu = gf * torch.sigmoid(gf)                    # post-norm gate, fp32
        out = xn * weight.float().reshape(1, H) * silu

        return out.to(torch.bfloat16)                    # single rounding


def get_init_inputs():
    return [128, 'post', 1e-6]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def make_inputs(shape, device, seed=0):
    """Deterministic, contiguous bf16 [x, weight, gate].

    Guarantees (relied upon by `invariants`):
      * weight is positive, channel-varying, in [0.75, 1.25]
      * gate is signed with |gate| in [0.5, 2.0]  ->  |silu(gate)| >= ~0.18 > 0
      * every x group has fp32 mean-square >= 0.25 *after* the bf16 cast
    """
    T, H = _dims(shape)
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 7919 + 12345)

    # --- x -----------------------------------------------------------------
    x = torch.randn(T, H, generator=gen, device=device, dtype=torch.float32)
    # per-row power-of-two scale so RMS varies across rows without introducing
    # any rounding of its own (bf16 * 2^k is exact)
    k = torch.randint(-1, 3, (T, 1), generator=gen, device=device, dtype=torch.int32)
    x = x * torch.pow(torch.tensor(2.0, device=device), k.float())
    xb = x.to(torch.bfloat16)

    G = _effective_group(H)
    n_groups = H // G

    # lift any group whose post-cast fp32 mean-square dipped under the floor.
    # the correction is a power of two, so it is exact in bf16 and one pass
    # provably suffices.
    ms = xb.float().reshape(T, n_groups, G).pow(2).mean(dim=-1, keepdim=True)
    need = (0.3 / ms.clamp(min=1e-12)).clamp(min=1.0)
    expo = torch.ceil(torch.log2(need) * 0.5).clamp(min=0.0)
    scale = torch.pow(torch.tensor(2.0, device=device), expo)
    xb = (xb.float().reshape(T, n_groups, G) * scale).to(torch.bfloat16).reshape(T, H)

    ms_final = xb.float().reshape(T, n_groups, G).pow(2).mean(dim=-1)
    assert float(ms_final.min()) >= 0.25, (
        f'x group mean-square floor violated: {float(ms_final.min())}')

    # --- weight: positive, channel varying, [0.75, 1.25] --------------------
    w = torch.rand(H, generator=gen, device=device, dtype=torch.float32)
    w = 0.75 + 0.5 * w
    wb = w.to(torch.bfloat16)

    # --- gate: signed, |gate| in [0.5, 2.0] ---------------------------------
    mag = 0.5 + 1.5 * torch.rand(T, H, generator=gen, device=device, dtype=torch.float32)
    sgn = torch.where(
        torch.rand(T, H, generator=gen, device=device, dtype=torch.float32) < 0.5,
        torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))
    gb = (mag * sgn).to(torch.bfloat16)

    return [xb.contiguous(), wb.contiguous(), gb.contiguous()]


def sample_shapes(rng, n):
    """Unseen case ids (4..10) near the public domain."""
    out = []
    for _ in range(int(n)):
        out.append((int(_UNSEEN_CASES[rng.randrange(len(_UNSEEN_CASES))]),))
    return out


# --------------------------------------------------------------------------- #
# independent oracle: float64, norm-based, different code path than Model
# --------------------------------------------------------------------------- #
def independent_oracle(ins, shape):
    x, weight, gate = ins[0], ins[1], ins[2]
    T, H = _dims(shape)
    G = _effective_group(H)

    xd = x.double().reshape(T, H // G, G)
    # RMS via vector norm rather than mean-of-squares
    nrm = torch.linalg.vector_norm(xd, ord=2, dim=-1, keepdim=True)
    ms = (nrm * nrm) / float(G)
    xn = (xd / torch.sqrt(ms + EPS)).reshape(T, H)

    gd = gate.double()
    silu = gd / (1.0 + torch.exp(-gd))          # algebraically distinct silu form
    return (xn * weight.double().reshape(1, H) * silu).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# metamorphic invariants  (never call Model)
# --------------------------------------------------------------------------- #
def invariants(out, ins, shape):
    if isinstance(out, (list, tuple)):
        out = out[OUTPUT_INDEX]
    x, weight, gate = ins[0], ins[1], ins[2]
    T, H = _dims(shape)
    G = _effective_group(H)
    checks = []

    # 1) structural: exact shape, bf16 dtype, all finite
    ok_shape = tuple(out.shape) == (T, H)
    ok_dtype = out.dtype == torch.bfloat16
    ok_fin = bool(torch.isfinite(out.float()).all())
    checks.append((
        'shape_dtype_finite',
        bool(ok_shape and ok_dtype and ok_fin),
        f'shape={tuple(out.shape)} exp={(T, H)} dtype={out.dtype} all_finite={ok_fin}',
    ))

    # 2) sign law: normalization is a positive per-group rescale and weight > 0,
    #    so sign(out) must equal sign(x * weight * silu(gate)) wherever the
    #    product is not numerical dust.  Purely algebraic, no norm recompute.
    gf = gate.float()
    gain = weight.float().reshape(1, H) * (gf * torch.sigmoid(gf))
    prod = x.float() * gain
    mask = prod.abs() > 1e-5
    mism = int(((torch.sign(out.float()) != torch.sign(prod)) & mask).sum())
    total = int(mask.sum())
    checks.append((
        'sign_preservation',
        mism == 0,
        f'{mism} sign mismatches over {total} elements with |x*w*silu(g)| > 1e-5',
    ))

    # 3) closed form for the grouped RMS.  weight in [0.75,1.25] and
    #    |silu(gate)| >= 0.18 by construction, so dividing out the gain is safe;
    #    what remains must have per-group RMS exactly sqrt(ms / (ms + eps)).
    recovered = (out.float() / gain).reshape(T, H // G, G)
    rms_got = torch.sqrt(recovered.pow(2).mean(dim=-1))
    ms = x.float().reshape(T, H // G, G).pow(2).mean(dim=-1)
    rms_exp = torch.sqrt(ms / (ms + EPS))
    err = float((rms_got - rms_exp).abs().max())
    checks.append((
        'grouped_rms_closed_form',
        err <= 4e-2,
        f'max |per-group RMS(out/(w*silu(g))) - sqrt(ms/(ms+eps))| = {err:.3e} '
        f'(tol 4e-2, G={G})',
    ))

    return checks


# --------------------------------------------------------------------------- #
# strong library baseline: torch.compile over the exact fixed composite
# --------------------------------------------------------------------------- #
def _fused_gated_group_rmsnorm(x, weight, gate, eps, group_size):
    T, H = x.shape
    G = group_size if (H % group_size == 0) else H
    xf = x.float().view(T, H // G, G)
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = (xf * torch.rsqrt(ms + eps)).view(T, H)
    gf = gate.float()
    return (xn * weight.float().view(1, H) * (gf * torch.sigmoid(gf))).to(torch.bfloat16)


_COMPILED = None
_WARMED = set()


def _get_compiled():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(
            _fused_gated_group_rmsnorm, dynamic=False, fullgraph=True)
    return _COMPILED

MEM_WEIGHT = 0.0
