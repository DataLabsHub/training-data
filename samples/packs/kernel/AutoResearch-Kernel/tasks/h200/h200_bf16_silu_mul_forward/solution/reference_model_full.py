# reference_model.py
# ---------------------------------------------------------------------------
# task_id : h200_bf16_silu_mul_forward
# domain  : activation      task_type : fusion      task_mode : tensor
# track   : fusion_and_layout
# target  : h200_sm90_cuda129_torch280_nv2505  (NVIDIA H200, sm_90, Hopper)
#
# Generic production BF16 SiLUMul forward workload for transformer MLPs.
# Two equal-shaped contiguous BF16 matrices [T, F] arrive from HBM: an
# activation branch `a` and a gate branch `b`. The kernel must evaluate
# SiLU on `a`, round that gate to BF16 (the library's gate-rounding
# boundary), multiply by `b`, and round the product to BF16 -- all without
# materializing FP32 tensors in HBM. This file contains the NAIVE,
# obviously-correct plain-PyTorch reference (the correctness Oracle and the
# timing denominator), an independent float64 oracle, metamorphic
# invariants, and the pinned library baseline that a real kernel must beat.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

TASK_ID = "h200_bf16_silu_mul_forward"
DOMAIN = "activation"
TASK_TYPE = "fusion"
OPT_TRACK = "fusion_and_layout"
TARGET_ID = "h200_sm90_cuda129_torch280_nv2505"

TASK_MODE = "tensor"
OUTPUT_INDEX = 0
GRAD_INPUTS = []
TASK_RTOL = 1e-2
TASK_ATOL = 1e-2
MEM_WEIGHT = 0.5

# Representative PUBLIC shapes (T, F).
WORKLOAD = [
    (4096, 4096),
    (4096, 14336),
    (8192, 8192),
    (8192, 14336),
]

# Legal shape domain (the hidden-shape tripwire draws only from here).
_T_CHOICES = (4096, 6144, 8192)
_F_CHOICES = (4096, 6144, 8192, 11008, 14336, 16384)

BASELINE_CONTRACT = {
    "input_dtype": "bf16",
    "output_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "allow_tf32": False,
    "note": (
        "Elementwise fused SiLU-multiply over contiguous [T, F] rows. Both "
        "inputs are BF16 and the single output is BF16. There are exactly TWO "
        "BF16 rounding stages and they must both be preserved: (1) the SiLU "
        "gate is evaluated with FP32 math on the FP32-widened activation "
        "branch and is ROUNDED TO BF16 before it is used -- this is the "
        "library's gate-rounding boundary; (2) that BF16 gate is multiplied by "
        "the BF16 gate branch with an FP32 intermediate and the product is "
        "ROUNDED TO BF16 to produce the result. An all-FP32 pipeline with a "
        "single final cast is a DIFFERENT numerical contract and is not "
        "accepted. TF32 is irrelevant to this elementwise kernel and is "
        "declared off. No FP32 tensor may be materialized in HBM, inputs must "
        "not be mutated, and the output must be freshly allocated contiguous "
        "storage that does not alias either input."
    ),
}


# ---------------------------------------------------------------------------
# NAIVE reference model (the correctness anchor; deliberately unoptimized).
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """Naive plain-PyTorch BF16 SiLUMul forward.

    Materializes every intermediate as its own tensor -- this is the honest,
    obviously-correct denominator that a fused kernel is expected to beat.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # Stage 0: widen the activation branch to FP32 for the SiLU math.
        a32 = a.to(torch.float32)
        # Stage 1: SiLU on the activation branch, rounded to BF16.
        #          (this BF16 boundary is part of the contract)
        gate = (a32 * torch.sigmoid(a32)).to(torch.bfloat16)
        # Stage 2: multiply by the BF16 gate branch, rounded to BF16.
        #          PyTorch evaluates a BF16 x BF16 product through an FP32
        #          intermediate and rounds once, matching the contract.
        out = (gate * b).to(torch.bfloat16)
        return out.contiguous()


def get_init_inputs():
    return []


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_inputs(shape, device, seed=0):
    """Deterministic per-seed BF16 inputs [T, F] with structural sentinels.

    Data coverage: negative tail (deep saturation of SiLU), the near-zero
    region (where SiLU ~ x/2), and the positive linear region (SiLU ~ x).

    Reserved sentinel rows (T >= 4 for every legal shape):
      row 0 / row 1 : a[1] = -a[0], b[1] = b[0]   (reflection pair)
      row 2 / row 3 : a[3] =  a[2], b[3] = 2*b[2] (gate-branch homogeneity)
    Exact-zero entries are injected into both `a` and `b`.
    """
    T, F = int(shape[0]), int(shape[1])
    if T < 4:
        raise ValueError(f"shape {tuple(shape)} needs T >= 4 for sentinel rows")

    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 9973 + 12345)

    def _u(lo, hi):
        return torch.empty((T, F), device=device, dtype=torch.float32).uniform_(
            lo, hi, generator=gen
        )

    def _r():
        return torch.rand((T, F), device=device, dtype=torch.float32, generator=gen)

    # Activation branch: broad span, then a near-zero band and a deep
    # negative tail carved out of it.
    a = _u(-8.0, 8.0)
    near_zero = _r() < 0.25
    a = torch.where(near_zero, a * (1.0 / 64.0), a)          # |a| <~ 0.125
    deep_neg = _r() < 0.10
    a = torch.where(deep_neg, -(a.abs() + 6.0), a)           # a in [-14, -6]
    a = a.clamp_(-12.0, 12.0)

    # Gate branch: bounded so that 2*b stays comfortably in BF16 range.
    b = _u(-4.0, 4.0)

    # Exact-zero injection (independent masks for the two branches).
    a = torch.where(_r() < 0.01, torch.zeros_like(a), a)
    b = torch.where(_r() < 0.01, torch.zeros_like(b), b)

    a = a.to(torch.bfloat16).contiguous()
    b = b.to(torch.bfloat16).contiguous()

    # Sentinel rows, applied in BF16 so the relations hold exactly:
    # negation and multiplication by 2 are exact in BF16 here.
    a[1] = -a[0]
    b[1] = b[0]
    a[3] = a[2]
    b[3] = b[2] * 2.0

    return [a.contiguous(), b.contiguous()]


def sample_shapes(rng, n):
    """UNSEEN shapes drawn only from the declared T/F choices (no WORKLOAD)."""
    pool = [
        (t, f)
        for t in _T_CHOICES
        for f in _F_CHOICES
        if (t, f) not in set(WORKLOAD)
    ]

    def _randint(hi):
        # Accept random.Random, numpy Generator, or numpy RandomState.
        for name in ("randrange", "integers", "randint"):
            fn = getattr(rng, name, None)
            if fn is None:
                continue
            try:
                if name == "randrange":
                    return int(fn(hi))
                if name == "integers":
                    return int(fn(0, hi))
                return int(fn(0, hi - 1))
            except Exception:
                continue
        return 0

    # Fisher-Yates using only the rng's integer stream.
    for i in range(len(pool) - 1, 0, -1):
        j = _randint(i + 1)
        pool[i], pool[j] = pool[j], pool[i]

    return pool[: max(0, min(int(n), len(pool)))]


# ---------------------------------------------------------------------------
# Independent oracle: float64, numerically stable SiLU, explicit BF16 stages.
# Does NOT call Model or baseline.
# ---------------------------------------------------------------------------
def independent_oracle(ins, shape):
    a, b = ins[0], ins[1]
    T, F = int(shape[0]), int(shape[1])
    out = torch.empty((T, F), device=a.device, dtype=torch.bfloat16)

    chunk = max(1, min(T, 1024))
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        x = a[s:e].to(torch.float64)
        # Stable SiLU: exp is only ever evaluated at -|x| <= 0, so there is no
        # overflow on the negative tail.
        em = torch.exp(-x.abs())
        silu = torch.where(x >= 0, x / (1.0 + em), x * em / (1.0 + em))
        # BF16 rounding stage 1 (the gate-rounding boundary).
        gate = silu.to(torch.bfloat16)
        # Separate gain multiplication, then BF16 rounding stage 2.
        prod = gate.to(torch.float64) * b[s:e].to(torch.float64)
        out[s:e] = prod.to(torch.bfloat16)

    return out.contiguous()


# ---------------------------------------------------------------------------
# Metamorphic invariants (no Model rerun; algebraic properties only).
# ---------------------------------------------------------------------------
def invariants(out, ins, shape):
    checks = []
    a, b = ins[0], ins[1]
    T, F = int(shape[0]), int(shape[1])

    # 1) Structure: shape, dtype, contiguity, finiteness, no aliasing.
    ok_shape = tuple(out.shape) == (T, F)
    ok_dtype = out.dtype == torch.bfloat16
    ok_contig = out.is_contiguous()
    ok_finite = bool(torch.isfinite(out.to(torch.float32)).all().item())
    ok_alias = out.data_ptr() != a.data_ptr() and out.data_ptr() != b.data_ptr()
    ok1 = ok_shape and ok_dtype and ok_contig and ok_finite and ok_alias
    checks.append((
        "shape_dtype_finite",
        bool(ok1),
        f"shape={tuple(out.shape)} want={(T, F)} dtype={out.dtype} "
        f"contig={ok_contig} finite={ok_finite} no_alias={ok_alias}",
    ))

    o = out.to(torch.float32)
    af = a.to(torch.float32)
    bf = b.to(torch.float32)

    # 2) Zero sentinels: silu(0) = 0 and gate*0 = 0, so the output must
    #    vanish wherever either branch holds an exact zero.
    zmask = (af == 0) | (bf == 0)
    nz = int(zmask.sum().item())
    if nz == 0:
        checks.append(("zero_sentinels", False, "no exact-zero entries present in inputs"))
    else:
        zerr = float(o[zmask].abs().max().item())
        checks.append((
            "zero_sentinels",
            zerr <= 1e-3,
            f"max|out| over {nz} zero entries = {zerr:.3e} (tol 1e-3)",
        ))

    # 3) Reflection identity: silu(x) - silu(-x) = x exactly, and rows 0/1
    #    share the same gate branch, so out[0] - out[1] = a[0] * b[0].
    #    Independent of any SiLU implementation.
    diff = o[0] - o[1]
    ref = af[0] * bf[0]
    tol = 3e-2 * (af[0].abs() * bf[0].abs() + 1e-2)
    worst = float(((diff - ref).abs() - tol).max().item())
    idx = int((diff - ref).abs().argmax().item())
    checks.append((
        "paired_reflection",
        worst <= 0.0,
        f"max slack (|out0-out1 - a0*b0| - tol) = {worst:.3e}; worst col {idx}: "
        f"got {float(diff[idx]):.5f} want {float(ref[idx]):.5f}",
    ))

    # 4) Gate-branch homogeneity: rows 2/3 share the activation branch and
    #    b[3] = 2*b[2], so scaling commutes with the BF16 rounding of the
    #    product: out[3] = 2 * out[2].
    lhs = o[3]
    rhs = 2.0 * o[2]
    htol = 1e-2 * rhs.abs() + 1e-2
    hworst = float(((lhs - rhs).abs() - htol).max().item())
    hidx = int((lhs - rhs).abs().argmax().item())
    checks.append((
        "gate_homogeneity",
        hworst <= 0.0,
        f"max slack (|out3 - 2*out2| - tol) = {hworst:.3e}; worst col {hidx}: "
        f"got {float(lhs[hidx]):.5f} want {float(rhs[hidx]):.5f}",
    ))

    return checks


# ---------------------------------------------------------------------------
# Strong library baseline (the honest frontier). Fails closed.
# ---------------------------------------------------------------------------
def baseline(a, b, init_args=None):
    """Official performance baseline: liger_kernel's fused SiLUMul forward."""
    if init_args is not None and list(init_args) != []:
        raise ValueError(
            f"baseline takes no init args; got init_args={init_args!r}"
        )

    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    except Exception as exc:  # pinned implementation missing -> fail closed
        raise RuntimeError(
            "Pinned baseline liger_kernel.ops.swiglu.LigerSiLUMulFunction is "
            "unavailable; refusing to substitute torch.compile, plain PyTorch, "
            "or any alternate Liger interface."
        ) from exc

    if not hasattr(LigerSiLUMulFunction, "apply"):
        raise RuntimeError(
            "liger_kernel.ops.swiglu.LigerSiLUMulFunction is incompatible: no "
            "`.apply` entry point; refusing to fall back."
        )

    return LigerSiLUMulFunction.apply(a, b)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    shp = WORKLOAD[0]
    ins = make_inputs(shp, dev, seed=0)
    y = Model().to(dev)(*ins)
    ora = independent_oracle(ins, shp)
    err = (y.float() - ora.float()).abs().max().item()
    print(f"shape={shp} dtype={y.dtype} max|model-oracle|={err:.3e}")
    for name, ok, detail in invariants(y, ins, shp):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("unseen shapes:", sample_shapes(__import__("random").Random(0), 4))
