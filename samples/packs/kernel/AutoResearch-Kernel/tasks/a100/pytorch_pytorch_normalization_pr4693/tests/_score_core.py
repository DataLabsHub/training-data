"""autoresearch._score_core — the SINGLE shared scoring core (plan §5关卡3,8,9,10,11).

This is the ONE scorer implementation. Both the internal certification path
(autoresearch.verifier, a thin re-export shim over this module) and the DELIVERED Harbor/
AutoResearch path (autoresearch_tasks/_shared/run_task.py, a thin adapter) call into it, so the task
verifier shipped to the agent can never drift from the one used to certify the task. It is
designed to run BOTH in-repo (as autoresearch._score_core) AND as a lone file copied into
a task container that has no autoresearch package installed — so the candidate subprocess is
spawned by __file__ path, never by `-m autoresearch.verifier` (see _run_candidate_subprocess).

It takes its CONTRACT from a spec dict (task_mode, tolerances, mem_weight, grad_inputs,
frontier headroom, allow_ops, cache_regime, resource caps) and its CODE from the spec's
ref_arch_src (Model / make_inputs / get_init_inputs / optional baseline / sample_shapes /
compute_quality). This realizes the schema's core value: contract is separated from code.

Capabilities (all proven):
  * static anti-hack gate (real Triton/CUDA kernel required; fused black-box ops banned
    except the spec's allow_torch_ops) — plan 关卡 8
  * multi hidden-seed correctness — 关卡 3
  * process-role split: candidate runs in a SEPARATE subprocess from the trusted judge — P0#1
  * tensor-mode backward via seeded VJP (decoupled from loss mode) — plan review P0(b)
  * CUDA-event interleaved-median timing on the SAME input instance + CV re-time/flag — 关卡 10
  * timed-invocation output capture (phase-switch resistant) — P1#4
  * process-level peak memory (torch allocator + device free-delta max) — 关卡 11
  * worst-slice regression gate + latency ceiling — 关卡 3
  * hidden-shape tripwire (unseen shapes, correctness-only) — 关卡 3 anti-hardcode
  * THREE-number report: speedup_vs_seed / speedup_vs_frontier / peak_mem — plan §2.2

It imports torch, so it is only imported in the scoring worker/adapter, never at package
import time.
"""
from __future__ import annotations
import os, sys, json, math, gc, argparse, traceback, tempfile, importlib.util, contextlib, signal
import torch

# vendored static checker: in-package when run in-repo, lone-file fallback when copied into a
# container that has no autoresearch package (both files ship beside this one).
try:
    from autoresearch.kernel_static_checker import validate_kernel_static
except Exception:
    try:
        from kernel_static_checker import validate_kernel_static
    except Exception:
        validate_kernel_static = None


# ---- verifier-private defaults (a spec/env may override) ----
CORRECT_SEEDS = [0, 1, 2]
TIME_SEED = 7
N_HIDDEN_SHAPES = 4  # min unseen shapes drawn from the declared domain (spec may raise)


def _harness_seed() -> int:
    """The hidden-shape draw seed. It is NOT a fixed constant: a fixed seed would let a
    candidate predict which unseen shapes the tripwire will use and hardcode them. Priority:
    env KF_HARNESS_SEED (injected by a trusted harness for reproducibility) else random."""
    v = os.environ.get("KF_HARNESS_SEED")
    if v is not None:
        try:
            return int(v)
        except ValueError:
            pass
    import random as _r
    return _r.randrange(1, 2 ** 31)
DEFAULT_REGRESSION_FLOOR = 0.5
MAX_TIMING_CV = 0.20
_SENTINEL = "###KF_VERIFY###"
_TARGET_FROM_ENV = object()

_DTYPE_BY_NAME = {
    "fp32": torch.float32, "float32": torch.float32, "fp64": torch.float64,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2,
}


def _torch_dtype(name):
    if not name:
        return None
    return _DTYPE_BY_NAME.get(str(name).split("(")[0].split("+")[0].strip())


# ============================================================================
#  loading ref + candidate CODE from source strings (schema separates code)
# ============================================================================
def _load_module_from_src(src: str, name: str):
    """Write src to a REAL temp .py file and import it. A real file (not exec-from-string)
    is required because Triton's @triton.jit inspects the defining source file, and
    torch inline-CUDA also reads source; exec(compile(...,'<str>')) breaks both."""
    d = tempfile.mkdtemp(prefix=f"kf_{name}_")
    path = os.path.join(d, f"{name}.py")
    with open(path, "w") as f:
        f.write(src)
    if d not in sys.path:
        sys.path.insert(0, d)   # so `import X` inside the module resolves against its dir
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _scalar(out, idx):
    return out[idx] if isinstance(out, (tuple, list)) else out


def _detach_out(raw):
    """Detach a forward result (tensor or tuple/list of tensors) from the autograd graph, so a
    grad-carrying tensor-mode output can be handed to the judge's value comparison identically to
    the cross-process (CPU round-trip) path."""
    if torch.is_tensor(raw):
        return raw.detach()
    if isinstance(raw, (tuple, list)):
        return type(raw)(_detach_out(v) for v in raw)
    return raw


def resolve_target_frontier(C: dict, spec_dict: dict, target_id) -> dict:
    """HW3: resolve the frontier (and per-target tolerance) for the ACTIVE GPU target, in place
    on the contract dict C. Torch-free so it is unit-testable offline.

    - target_id falsy  -> legacy single-target behavior: keep C['frontier'] from the spec's
      scalar baselines (already set by the caller).
    - target_id set and calibrations[target_id] present -> use THAT target's frontier.
    - target_id set but NOT calibrated -> FAIL CLOSED: frontier=None (vs_frontier will be None
      and a note is emitted). Never reuse another GPU's frontier for an uncalibrated target."""
    C["target_id"] = target_id
    C["frontier_uncalibrated"] = False
    if not target_id:
        return C
    cal = (spec_dict.get("calibrations", {}) or {}).get(target_id)
    identity_ok = (
        isinstance(cal, dict)
        and cal.get("target_id") == target_id
        and isinstance(cal.get("gpu_model"), str) and cal.get("gpu_model").strip()
        and isinstance(cal.get("compute_capability"), str)
        and cal.get("compute_capability").strip()
    )
    frontier = cal.get("frontier_headroom") if identity_ok else None
    if (isinstance(frontier, (int, float)) and not isinstance(frontier, bool)
            and math.isfinite(frontier) and frontier > 0):
        C["frontier"] = frontier
        # Route the complete selected calibration in one operation. A secondary target must never
        # inherit the primary target's component/provenance scalars.
        for key in ("official_headroom", "library_headroom", "compile_headroom",
                    "frontier_source"):
            C[key] = cal.get(key)
    else:
        C["frontier"] = None
        C["frontier_uncalibrated"] = True
    return C


def resolve_reward_anchor(geo: float, frontier, official_declared: bool,
                          frontier_uncalibrated: bool = False) -> dict:
    """Torch-free reward-anchor decision (unit-testable offline). Given the geomean composite vs
    the naive seed (geo) and the frontier headroom, decide the reward:

      * an explicitly targeted but uncalibrated run -> reward = 0 (fail closed; the naive-relative
        speedup remains diagnostic only and cannot certify a hardware-specific score);
      * official baseline declared AND a usable frontier -> reward = speedup_vs_frontier = geo /
        frontier (beat the version-locked official implementation, the honest target);
      * otherwise -> reward = speedup_vs_seed = geo (naive denominator, the dense training signal).

    Returns {reward, speedup_vs_seed, speedup_vs_frontier, reward_anchor, reward_note?}. Both
    speedups are always reported; only which one becomes 'reward' changes."""
    if frontier_uncalibrated:
        return {"reward": 0.0,
                "speedup_vs_seed": round(geo, 4),
                "speedup_vs_frontier": None,
                "reward_anchor": "uncalibrated_target",
                "reward_note": "uncalibrated_target_fail_closed"}

    vs_frontier = round(geo / frontier, 4) if frontier and frontier > 0 else None
    anchor_official = bool(official_declared and vs_frontier is not None)
    reward = vs_frontier if anchor_official else geo
    out = {"reward": round(reward, 4),
           "speedup_vs_seed": round(geo, 4),
           "speedup_vs_frontier": vs_frontier,
           "reward_anchor": ("official_frontier" if anchor_official else "naive_seed")}
    if official_declared and vs_frontier is None:
        out["reward_note"] = "official_anchor_unavailable_fell_back_to_seed"
    return out


def _approx_indices_valid(idx_c, idx_r):
    """P1#15: validate that an approx-mode candidate returned neighbor INDICES, not a
    self-reported quality scalar. The reference's own output (idx_r) defines the expected
    integer index tensor shape/kind. Returns (ok, reason)."""
    if not torch.is_tensor(idx_c):
        return False, f"got {type(idx_c).__name__}, expected an integer index tensor"
    if idx_c.is_floating_point() or idx_c.is_complex():
        return False, f"got {idx_c.dtype} (floating) — indices must be integer, not a recall scalar"
    if torch.is_tensor(idx_r):
        if tuple(idx_c.shape) != tuple(idx_r.shape):
            return False, f"index shape {tuple(idx_c.shape)} != reference {tuple(idx_r.shape)}"
    return True, ""


# ============================================================================
#  static anti-hack gate (关卡 8)
# ============================================================================
def _detect_backend(code: str) -> str:
    if "@triton.jit" in code or "@triton.autotune" in code or "import triton" in code:
        return "triton"
    if "__global__" in code or "load_inline" in code or "cpp_extension" in code:
        return "cuda"
    return "triton"


def static_gate(cand_src: str, allow_ops=None, direct_library_call_allowed=False) -> tuple[bool, str]:
    """(ok, reason). Fail-closed if the checker is unavailable unless KF_VERIFIER_DEV=1.
    direct_library_call_allowed relaxes the kernel-presence requirement for
    integration/system tracks (plan §5关卡8: 'direct library call' legality is per-track)."""
    if direct_library_call_allowed:
        return True, "static gate skipped (track allows direct library call)"
    if validate_kernel_static is None:
        if os.environ.get("KF_VERIFIER_DEV") == "1":
            return True, "static checker unavailable (dev bypass)"
        return False, "static_checker_unavailable (fail-closed; KF_VERIFIER_DEV=1 to bypass in dev)"
    backend = _detect_backend(cand_src)
    forbidden = ["code_bypass", "timing_event_patch", "thread_injection", "lazy_eval",
                 "torch_computation_ops", "stream_injection",
                 "kernel_reachable", "dangerous_api"]
    valid, errors, _warns = validate_kernel_static(
        cand_src, backend=backend, precision="fp16", forbidden=forbidden, allow_ops=allow_ops or [])
    return (True, "") if valid else (False, "static_check_failed: " + "; ".join(errors[:4]))


# ============================================================================
#  tolerance / error metrics / timing (proven helpers)
# ============================================================================
def _assert_close_ok(ref_out, cand_out):
    """Kernel Test Bench doc §精度测试: correctness uses torch.testing.assert_close with its
    DEFAULT, dtype-derived tolerances — the task may NOT self-set rtol/atol or widen tolerance by
    reduction size (a wide self-set tolerance is exactly how a wrong kernel sneaks through). Both
    tensors are compared in the candidate's contract dtype (assert_close derives tolerance from
    that dtype), so ref_out must already be cast to the same dtype by the caller. Returns
    (ok, reason)."""
    try:
        torch.testing.assert_close(cand_out, ref_out)   # default rtol/atol from dtype
        if torch.is_tensor(cand_out) and not bool(torch.isfinite(cand_out.float()).all()):
            return False, "assert_close: candidate has non-finite values"
        return True, ""
    except AssertionError as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else "mismatch"
        return False, f"assert_close: {first[:120]}"
    except Exception as e:
        return False, f"assert_close_error: {type(e).__name__}: {str(e)[:80]}"


def _err_metrics(ref, cand):
    try:
        r = ref.float(); c = cand.float()
        diff = (r - c).abs()
        denom = r.abs().clamp_min(1e-6)
        rel = diff / denom
        near = r.abs() > 1e-4
        relmask = rel[near] if near.any() else diff.new_zeros(1)
        dflat = diff.flatten()
        if dflat.numel() > 1_000_000:
            dflat = dflat[torch.randperm(dflat.numel(), device=dflat.device)[:1_000_000]]
        return {"max_abs": round(diff.max().item(), 6),
                "max_rel": round(relmask.max().item(), 6),
                "rel_l2": round((diff.norm() / r.norm().clamp_min(1e-12)).item(), 6),
                "p99_abs": round(torch.quantile(dflat, 0.99).item(), 6),
                "nan_inf": int((~torch.isfinite(c)).sum().item())}
    except Exception as e:
        return {"metrics_err": f"{type(e).__name__}: {str(e)[:80]}"}


def _fwd_bwd(model, ins, grad_idx, out_idx):
    for i in grad_idx:
        if torch.is_tensor(ins[i]) and ins[i].grad is not None:
            ins[i].grad = None
    loss = _scalar(model(*ins), out_idx)
    loss.backward()
    grads = {i: (ins[i].grad.detach().clone() if torch.is_tensor(ins[i]) and ins[i].grad is not None else None)
             for i in grad_idx}
    return float(loss.detach()), grads


def _cotangent_seed(seed: int) -> int:
    """Derive a deterministic cotangent seed from the correctness seed. The candidate and the judge
    run in SEPARATE processes but both call this with the SAME correctness seed, so both synthesize
    the identical random cotangent for the tensor-mode VJP — the grads are then comparable."""
    return (int(seed) * 2654435761 + 12345) % (2 ** 31)


def _vjp_grads(raw_out, ins, grad_idx, out_indices, cotangent_seed):
    """Tensor-mode backward via a seeded VJP: reduce the (possibly multi-output) forward result to
    a scalar s = Σ_j <cotangent_j, output_j> with a deterministic random cotangent, then backward
    to get grads = Jᵀ·cotangent on the declared grad_inputs. This checks the operator's Jacobian
    WITHOUT importing any loss — a RoPE / layer-scale kernel has a real backward that would
    otherwise go unverified in tensor mode. The cotangent seed is shared across processes so the
    candidate's grads and the judge's recomputed reference grads use the same g and are comparable.
    Returns {i: grad_tensor_or_None} for i in grad_idx."""
    for i in grad_idx:
        if torch.is_tensor(ins[i]) and ins[i].grad is not None:
            ins[i].grad = None
    terms = []
    for k, oi in enumerate(out_indices):
        o = _scalar(raw_out, oi)
        if not torch.is_tensor(o) or not o.is_floating_point():
            continue
        g = torch.Generator(device=o.device).manual_seed(_cotangent_seed(cotangent_seed) + k)
        cot = torch.randn(o.shape, dtype=torch.float32, device=o.device, generator=g)
        terms.append((o.float() * cot).sum())
    if not terms:
        return {i: None for i in grad_idx}
    s = terms[0]
    for t in terms[1:]:
        s = s + t
    s.backward()
    return {i: (ins[i].grad.detach().clone() if torch.is_tensor(ins[i]) and ins[i].grad is not None else None)
            for i in grad_idx}



def _do_bench_ms(fn, warmup=25, rep=100):
    """Time one fixed callable with ``triton.testing.do_bench`` and request its median explicitly.

    Triton 3.6 interprets ``warmup`` / ``rep`` as millisecond budgets, defaults to the *mean*, and
    clears L2 before every measured run.  This helper therefore describes a cold-L2 standalone
    benchmark, not the scorer's ``steady_state_warm`` calibration protocol.  Trusted baseline
    calibration and candidate judging use the CUDA-event interleaved helper below instead.

    Falls back to a CUDA-Event warmup+median loop when triton is unavailable so the schema stays
    usable on a triton-less box. Returns median milliseconds."""
    try:
        import triton.testing as tt
        return float(tt.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))
    except Exception:
        pass
    # fallback: CUDA-Event median (no triton). Warmup absorbs one-time setup, then median of reps.
    n_warm = max(3, warmup // 5)
    n_rep = max(10, rep // 5)
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    return ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])


def _med_cv(xs):
    """Median + a ROBUST coefficient of variation. A plain std/mean CV over a short (~20-100)
    timing sample is dominated by a single outlier launch (a GC pause, a clock ramp, a co-tenant
    kernel), which spuriously trips the CV gate. Before the correctness/performance split that
    over-strict gate was mis-reported as a correctness failure and burned a witness round on a
    kernel that was numerically correct and merely mistimed. Use the median absolute deviation
    (MAD), scaled by 1.4826 so it equals the std for a normal distribution, over the MEDIAN (not
    mean): numerator and denominator are both outlier-resistant, so the reported dispersion
    reflects steady-state jitter rather than one stall — the same robustness triton's do_bench
    gets from taking a median over many reps (G4 uses _do_bench_ms; this aligns G6 to it)."""
    xs = sorted(xs); n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    if med <= 0:
        return med, 0.0
    devs = sorted(abs(x - med) for x in xs)
    mad = devs[n // 2] if n % 2 else 0.5 * (devs[n // 2 - 1] + devs[n // 2])
    return med, (1.4826 * mad / med)


def _timing_attempt_iters(timing: dict) -> list[int]:
    """Return the base measurement and each per-process CV-retry iteration count.

    This never changes the judge's two fresh timing subprocesses; it only controls bounded
    remeasurement inside each process.
    """
    timing = timing or {}
    iters = timing.get("iters", 20)
    max_retries = timing.get("max_retries", 1)
    multiplier = timing.get("retry_iters_multiplier", 2)
    for name, value, minimum in (
        ("iters", iters, 1),
        ("max_retries", max_retries, 0),
        ("retry_iters_multiplier", multiplier, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"timing.{name} must be an integer >= {minimum}")
    return [iters * (multiplier ** retry) for retry in range(max_retries + 1)]


def _time_ms_interleaved(fn_baseline, fn_seed, warmup=5, iters=20):
    """Warm-cache paired CUDA-event timing for trusted baseline calibration.

    Run the baseline and naive seed back-to-back in every warmup and measured iteration, never
    clearing L2.  Return each stream's median and CV so calibration can reject noise on either
    side of the ratio.  ``warmup`` and ``iters`` are iteration counts from ``TimingContract`` —
    unlike Triton's ``do_bench`` millisecond budgets — and therefore exactly match scoring.
    """
    for _ in range(warmup):
        fn_baseline()
        fn_seed()
    torch.cuda.synchronize()

    baseline_ms, seed_ms = [], []
    for _ in range(iters):
        s0 = torch.cuda.Event(enable_timing=True)
        e0 = torch.cuda.Event(enable_timing=True)
        s0.record()
        fn_baseline()
        e0.record()

        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s1.record()
        fn_seed()
        e1.record()

        torch.cuda.synchronize()
        baseline_ms.append(s0.elapsed_time(e0))
        seed_ms.append(s1.elapsed_time(e1))

    baseline_med, baseline_cv = _med_cv(baseline_ms)
    seed_med, seed_cv = _med_cv(seed_ms)
    return baseline_med, seed_med, baseline_cv, seed_cv


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ============================================================================
#  timing-merge + verdict (PURE, torch-free — the correctness-vs-performance split lives here so
#  it can be unit-tested without a GPU; score() calls these after Phase B produces the two judged
#  timing runs).
# ============================================================================
def _is_numeric_fail(msg) -> bool:
    """A timing-phase `fail` string is EITHER a genuine numeric/resource failure (candidate wrong or
    over budget -> correctness=0) OR a pure timing problem (missing/nonpositive samples -> the run is
    unreliable, NOT wrong). Only the former may flip correctness. This predicate is the boundary."""
    m = msg or ""
    return (m.startswith("timed_output_mismatch") or m.startswith("capture_check_failed")
            or m.startswith("memory_fail") or m.startswith("latency_fail"))


def _merge_timing_runs(results, j0, j1, xproc_tol=0.25):
    """Merge the two independent timing processes into each shape's result entry, requiring
    cross-process agreement (P0 §3.3). Mutates `results` in place; returns (functional_ok,
    timing_ok). CV noise / cross-process disagreement must never be reported as correctness=0
    when Phase A + the timed-capture recompute already proved the output right — that false
    failure is what wasted witness authoring rounds (external review #2)."""
    functional_ok = True   # numeric/resource correctness (may set correctness=0)
    timing_ok = True       # timing reliability (only gates performance_valid, never correctness)
    for ent in results:
        shape = tuple(ent["shape"])
        a = j0.get(shape, {}); b = j1.get(shape, {})
        af, bf = a.get("fail"), b.get("fail")
        if af or bf:
            fail = af or bf
            if _is_numeric_fail(fail):
                ent.update(correct=False, timing_fail=fail)
                functional_ok = False
            else:
                # a pure timing failure (e.g. timing_missing_or_nonpositive): correctness stands,
                # performance is inconclusive. composite is None (NOT 0), so it never poses as a
                # real "0x" score nor triggers a spurious cross-process disagreement below.
                ent.update(timing_unreliable=fail, timing_noisy=True, composite_vs_naive=None)
                timing_ok = False
            continue
        ca, cb = a.get("composite"), b.get("composite")
        noisy = bool(a.get("noisy") or b.get("noisy"))
        # cross-process disagreement => timing not reproducible => noisy. Skip when either composite
        # is None/invalid (a None is already-flagged unreliable; don't let a sentinel fake a disagree).
        if ca is not None and cb is not None and ca > 0 and cb > 0:
            if max(ca, cb) / max(min(ca, cb), 1e-9) > 1 + xproc_tol:
                noisy = True
                ent["timing_cross_process_disagree"] = f"composite {ca:.4f} vs {cb:.4f}"
        else:
            noisy = True   # a missing/nonpositive composite from either process => unreliable
        if noisy:
            timing_ok = False
            ent.setdefault("timing_unreliable", f"cv or cross-process noisy (a={ca} b={cb})")
        # conservative: the MINIMUM composite of the two processes; None when unreliable (not 0).
        composite = None if noisy else min(ca, cb)
        ent.update(t_cand_ms=round(a["t_cand"], 3), t_naive_ms=round(a["t_naive"], 3),
                   cv_cand=round(a["cv_cand"], 4), cv_naive=round(a["cv_naive"], 4),
                   timing_noisy=noisy, mem_cand_mb=round(a["mem_cand"], 1),
                   mem_naive_mb=round(a["mem_naive"], 1),
                   timing_retries=max(a.get("timing_retries", 0),
                                      b.get("timing_retries", 0)),
                   effective_iters=max(a.get("effective_iters", 0),
                                       b.get("effective_iters", 0)),
                   composite_vs_naive=(round(composite, 4) if composite is not None else None))
    return functional_ok, timing_ok


def _perf_verdict(functional_ok, timing_ok, geo, worst, reg_floor, results, hidden_results):
    """The three EARLY-RETURN verdicts that precede the success path, split so a timing problem
    never masquerades as a correctness failure. Returns the verdict dict, or None to signal
    'proceed to the success/reward path'."""
    # a genuine numeric/resource failure in the timing phase -> the candidate is WRONG.
    if not functional_ok:
        return {"reward": 0.0, "correctness": 0, "reason": "incorrect",
                "performance": "not_run", "performance_valid": False,
                "speedup_vs_seed": None, "worst_slice": None,
                "per_shape": results, "hidden_shapes": hidden_results}
    # correctness passed (Phase A + timed-capture) but timing is unreliable -> performance is
    # INCONCLUSIVE, retryable with the SAME source. NOT a correctness failure (external review #2).
    if not timing_ok or geo <= 0:
        return {"reward": 0.0, "correctness": 1, "performance": "inconclusive",
                "performance_valid": False, "retryable": True, "reason": "timing_unreliable",
                "speedup_vs_seed": (round(geo, 4) if geo > 0 else None),
                "worst_slice": (round(worst, 4) if worst > 0 else None),
                "per_shape": results, "hidden_shapes": hidden_results}
    if worst < reg_floor:
        # a stable, correct, but genuinely-too-slow candidate: performance IS valid, it just fails
        # the worst-slice floor. correctness stands; reward 0 on the objective.
        return {"reward": 0.0, "correctness": 1, "performance_valid": True,
                "reason": f"worst_slice_regression {worst:.3f} < {reg_floor}",
                "speedup_vs_seed": round(geo, 4), "worst_slice": round(worst, 4),
                "per_shape": results, "hidden_shapes": hidden_results}
    return None


def _time_and_capture(cand, ref, fresh_ins, run_once, base_seed, warmup, iters,
                      grad_idx=()):
    """Interleaved CUDA-Event timing where EACH iteration draws fresh inputs (P1#7): the
    candidate cannot recognise a fixed timing input and switch to a cheap wrong kernel.

    On one randomly chosen iteration the candidate's output is captured FROM THE TIMED
    INVOCATION ITSELF — not from a separate extra call (plan P1#4). A candidate that counts
    its calls and serves garbage during the timed call but correct output in a follow-up call
    would otherwise pass; here the very tensor produced inside the timed region is the one we
    later compare, so serving garbage while timed is caught. The reference truth is computed
    on the SAME inputs after the timed region. The clone happens after e0.record() so it is
    outside the measured window but references the timed call's real output.

    Returns (t_cand, t_naive, cv_cand, cv_naive, capture)."""
    import random
    rng = random.Random(base_seed)
    cap_iter = rng.randrange(iters) if iters > 0 else -1
    for k in range(warmup):
        run_once(cand, fresh_ins(base_seed + 1000 + k))
        run_once(ref, fresh_ins(base_seed + 2000 + k))
    torch.cuda.synchronize()
    tc, tn, cap = [], [], None
    ev = lambda: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for k in range(iters):
        want_cap = (k == cap_iter)
        ins_seed = base_seed + 10000 + k
        ic = fresh_ins(ins_seed)
        # time the candidate; on the capture iter, ALSO retain this invocation's output. The
        # timed region is s0..e0 (just the forward/backward); retain() runs after e0.record().
        s0, e0 = ev(); s0.record()
        got = run_once(cand, ic, capture=want_cap); e0.record()
        # SAME workload instance for the reference: identical seed -> identical values in a
        # separate tensor object (fair same-instance timing; data-dependent kernels are compared
        # on the exact same input the candidate saw). Phase-switch resistance holds: seed varies
        # with k.
        in_ = fresh_ins(ins_seed)
        s1, e1 = ev(); s1.record(); run_once(ref, in_); e1.record()
        torch.cuda.synchronize()
        tc.append(s0.elapsed_time(e0)); tn.append(s1.elapsed_time(e1))
        if want_cap:
            # reference truth on the SAME inputs the candidate was just timed on
            exp = run_once(ref, _clone_inputs(ic, grad_idx), capture=True)
            cap = {"got": got, "exp": exp}
        del ic, in_
    mc, cvc = _med_cv(tc); mn, cvn = _med_cv(tn)
    return mc, mn, cvc, cvn, cap


def _clone_inputs(ins, grad_idx):
    """Fresh copies of the timed inputs so the reference recompute cannot be perturbed by any
    in-place writes the candidate made, and grad inputs get requires_grad again."""
    out = []
    for i, x in enumerate(ins):
        if torch.is_tensor(x):
            y = x.detach().clone()
            if i in grad_idx:
                y.requires_grad_(True)
            out.append(y)
        else:
            out.append(x)
    return out


def _verify_timed_capture(cap, mode, grad_idx, C):
    """Compare the candidate's timed-phase output against the reference truth captured on the
    same inputs. Returns (ok, reason).

    tensor/loss: elementwise / loss+grad allclose vs the reference on the timed inputs.
    approx: the timed candidate output must be a structurally valid index tensor matching the
    reference's index shape/dtype (catches a candidate that serves good neighbors during the
    correctness phase and garbage/zeros/scalars during timing). Full recall is not re-run here
    because compute_quality is seed-coupled to input generation; structural validity on the
    real timed invocation is the phase-switch guard, and correctness-phase recall on hidden
    seeds remains the accuracy gate (plan P1#4)."""
    if cap is None:
        return True, ""
    got, exp = cap.get("got"), cap.get("exp")
    if got is None or exp is None:
        return True, ""
    # the candidate's captured output arrives on CPU (cross-process transport) while the judge's
    # recomputed reference is on GPU. Move to CPU for comparison but PRESERVE the candidate's dtype:
    # assert_close derives its tolerance from the dtype, so upcasting a bf16 candidate to fp32 would
    # wrongly apply fp32-tight tolerance and reject an honest bf16 kernel. _cpu keeps dtype; the
    # reference is cast to the candidate's dtype at the comparison site.
    def _cpu(t):
        return t.detach().to("cpu") if torch.is_tensor(t) else t
    if mode == "approx":
        got = got.to("cpu") if torch.is_tensor(got) else got
        exp = exp.to("cpu") if torch.is_tensor(exp) else exp
        ok, why = _approx_indices_valid(got, exp)
        return (ok, "" if ok else f"timed approx output invalid: {why}")
    if mode == "tensor":
        try:
            ge, gg = _cpu(exp), _cpu(got)
            # assert_close default tolerance in the CANDIDATE's dtype (doc §精度测试): cast the
            # reference to the candidate dtype so a bf16 kernel is judged at bf16 tolerance.
            ok, _why = _assert_close_ok(ge.to(gg.dtype) if torch.is_tensor(gg) else ge, gg)
        except Exception as e:
            return False, f"timed_output_uncomparable: {type(e).__name__}"
        return (ok, "" if ok else "timed candidate output != reference")
    # loss mode: got/exp are (loss, grads)
    loss_g, grads_g = got; loss_e, grads_e = exp
    if not math.isfinite(loss_g):
        return False, f"timed loss non-finite {loss_g}"
    lg_t, le_t = torch.as_tensor(loss_g), torch.as_tensor(loss_e)
    ok, _w = _assert_close_ok(le_t.to(lg_t.dtype), lg_t)
    if not ok:
        return False, f"timed loss mismatch {loss_g} vs {loss_e}"
    for i in grad_idx:
        ge, gg = grads_e.get(i), grads_g.get(i)
        if ge is None:
            continue
        if gg is None:
            return False, f"timed grad[{i}] mismatch"
        ce, cg = _cpu(ge), _cpu(gg)
        gok, _gw = _assert_close_ok(ce.to(cg.dtype), cg)
        if not gok:
            return False, f"timed grad[{i}] mismatch"
    return True, ""


# ============================================================================
#  correctness (one shape,seed) — tensor / loss / approx
# ============================================================================
def _candidate_side(cand, refmod, shape, dev, seed, C):
    """CANDIDATE-SIDE of correctness: run ONLY the candidate on inputs built from `seed`, and
    return the raw data the judge needs. This runs in the (untrusted) candidate process — it
    does NO allclose / verdict / serialization, so a candidate that monkeypatches torch.allclose
    or json.dumps changes nothing the judge relies on (P0#1: the judge re-derives truth itself).
    Returns a dict describing the candidate's output for `mode`."""
    mode, grad_idx, out_idx = C["task_mode"], C["grad_inputs"], C["output_index"]
    mk = refmod.make_inputs
    ins_c = [x.to(dev) if torch.is_tensor(x) else x for x in mk(shape, dev, seed)]
    if mode == "approx":
        with torch.no_grad():
            idx_c = _scalar(cand(*ins_c), out_idx)
        return {"mode": "approx", "idx_c": idx_c}
    if mode == "tensor":
        mutable = set(C.get("mutable_inputs") or [])
        grad_idx = C.get("grad_inputs") or []
        out_indices = C.get("output_indices") or [out_idx]
        for i in grad_idx:
            if i < len(ins_c) and torch.is_tensor(ins_c[i]) and ins_c[i].is_floating_point():
                ins_c[i].requires_grad_(True)
        pre = {i: ins_c[i].detach().clone() for i in range(len(ins_c))
               if torch.is_tensor(ins_c[i]) and i not in mutable}
        # forward under grad only when a backward is required (grad_inputs declared) — otherwise
        # no_grad keeps the forward-only path cheap and matches the inference contract.
        if grad_idx:
            raw_c = cand(*ins_c)
        else:
            with torch.no_grad():
                raw_c = cand(*ins_c)
        # mutation flags computed candidate-side (best-effort contract signal); the judge's
        # value comparison is what actually gates correctness.
        mutated = {i: (not bool(torch.equal(before, ins_c[i].detach() if torch.is_tensor(ins_c[i]) else ins_c[i])))
                   for i, before in pre.items()}
        # P0#4: for DECLARED mutable inputs (e.g. the paged KV cache the candidate must append
        # into), capture their POST-forward state so the judge can verify the update is correct —
        # a candidate that skips the append (leaving the cache as garbage) must be rejected, not
        # silently accepted just because it never touched a forbidden input.
        mut_post = {i: ins_c[i].detach().clone() for i in mutable
                    if i < len(ins_c) and torch.is_tensor(ins_c[i])}
        out = {"mode": "tensor", "raw_c": raw_c, "mutated": mutated, "mut_post": mut_post}
        # tensor-mode backward (VJP): a tensor operator with declared grad_inputs still has a
        # Jacobian to verify. Reduce the output with a seeded cotangent and backward — orthogonal
        # to loss mode (plan review P0(b): backward is decoupled from what the forward returns).
        # Run the VJP BEFORE detaching raw_c (backward needs the graph), then detach the returned
        # outputs so the judge (in-process path) sees the same detached tensors the cross-process
        # path transports via _cpu.
        if grad_idx:
            grads_c = _vjp_grads(raw_c, ins_c, grad_idx, out_indices, seed)
            out["grads_c"] = grads_c
            out["raw_c"] = _detach_out(raw_c)
        return out
    # loss
    loss_c, grads_c = _fwd_bwd(cand, ins_c, grad_idx, out_idx)
    return {"mode": "loss", "loss_c": loss_c, "grads_c": grads_c}


def _judge_side(ref, refmod, shape, dev, seed, C, cand_data):
    """JUDGE-SIDE of correctness: independently recompute the REFERENCE on inputs built from the
    SAME `seed` (deterministic make_inputs => identical inputs), then compare the candidate's
    output (`cand_data`) against it. This runs in the trusted judge process, which NEVER imports
    candidate code, so allclose/isfinite/equal here cannot be monkeypatched by the candidate
    (P0#1). Returns (ok, rtol, atol, metrics).

    `cand_data` may carry live tensors (in-process path) or CPU tensors reloaded from disk
    (process-isolated path). In the latter case it also carries `dev_map` (index -> original
    device type string) so the device-match contract survives the CPU round-trip; the judge
    moves the candidate tensor to `dev` for the value comparison."""
    mode, grad_idx, out_idx = C["task_mode"], C["grad_inputs"], C["output_index"]
    mk = refmod.make_inputs
    ins_r = [x.to(dev) if torch.is_tensor(x) else x for x in mk(shape, dev, seed)]
    dev_map = cand_data.get("dev_map")   # present only on the cross-process path

    def _to_dev(t):
        return t.to(dev) if (torch.is_tensor(t) and t.device.type != torch.device(dev).type) else t

    if mode == "approx":
        idx_c = _to_dev(cand_data["idx_c"])
        with torch.no_grad():
            idx_r = _scalar(ref(*ins_r), out_idx)
        ok_shape, why = _approx_indices_valid(idx_c, idx_r)
        if not ok_shape:
            return False, None, None, {"contract": f"approx output must be indices: {why}"}
        q_r = float(refmod.compute_quality(idx_r, shape, dev, seed))
        q_c = float(refmod.compute_quality(idx_c, shape, dev, seed))
        floor = C.get("quality_floor", 0.0) or 0.0
        slack = C.get("quality_slack", 0.15) or 0.15
        ok = math.isfinite(q_c) and q_c >= floor and q_c >= q_r - slack
        return ok, None, None, {"recall_cand": round(q_c, 4), "recall_naive": round(q_r, 4)}

    if mode == "tensor":
        raw_c = cand_data["raw_c"]
        out_indices = C.get("output_indices") or [out_idx]
        with torch.no_grad():
            raw_r = ref(*ins_r)
        # (1) output count must match the contract for tuple/list returns
        if isinstance(raw_c, (tuple, list)) and isinstance(raw_r, (tuple, list)):
            if len(raw_c) != len(raw_r):
                return False, None, None, {"contract": f"output count {len(raw_c)} != {len(raw_r)}"}
        want_dtype = _torch_dtype(C.get("output_dtype"))
        rtol = atol = None; metrics = {}
        # (2) every declared output index: shape / dtype / device / value must all match
        for j, oi in enumerate(out_indices):
            o_r = _scalar(raw_r, oi); o_c = _scalar(raw_c, oi)
            if not (torch.is_tensor(o_r) and torch.is_tensor(o_c)):
                if o_r != o_c:
                    return False, None, None, {"contract": f"output[{oi}] non-tensor mismatch"}
                continue
            if tuple(o_c.shape) != tuple(o_r.shape):
                return False, None, None, {"contract": f"output[{oi}] shape {tuple(o_c.shape)} != {tuple(o_r.shape)}"}
            # device-match: use the RECORDED original device on the cross-process path (the
            # transported tensor is on CPU), else the live tensor's device.
            c_dev_type = (dev_map or {}).get(oi, o_c.device.type)
            if c_dev_type != o_r.device.type:
                return False, None, None, {"contract": f"output[{oi}] device {c_dev_type} != {o_r.device.type}"}
            if want_dtype is not None and o_c.dtype != want_dtype:
                return False, None, None, {"contract": f"output[{oi}] dtype {o_c.dtype} != contract {want_dtype}"}
            o_c = _to_dev(o_c)
            # correctness = torch.testing.assert_close at its DEFAULT dtype-derived tolerance
            # (Kernel Test Bench doc §精度测试): no self-set rtol/atol, no reduction-size widening.
            # The fp32 naive oracle is cast to the candidate's contract dtype so the tolerance
            # assert_close derives matches the output precision (a bf16 kernel is not held to
            # fp32-tight equality).
            ok, why = _assert_close_ok(o_r.to(o_c.dtype), o_c)
            if not ok:
                return False, None, None, {"contract": f"output[{oi}] {why}", **_err_metrics(o_r, o_c)}
            if j == 0:
                metrics = _err_metrics(o_r, o_c)
        # (3) non-mutable inputs must be untouched (candidate-reported flags)
        for i, was in (cand_data.get("mutated") or {}).items():
            if was:
                return False, rtol, atol, {"contract": f"input[{i}] mutated in place (not in mutable_inputs)"}
        # (4) P0#4: DECLARED mutable inputs must hold the CORRECT post-state. ins_r was mutated
        # in place by the reference's forward above (e.g. paged KV append); compare the
        # candidate's captured post-state against the reference's for every declared mutable
        # index. A candidate that leaves the cache un-appended (garbage) fails here even though
        # its returned tensor might match.
        mut_post = cand_data.get("mut_post") or {}
        for i in (C.get("mutable_inputs") or []):
            if i not in mut_post:
                continue
            post_c = _to_dev(mut_post[i])
            post_r = ins_r[i] if i < len(ins_r) else None
            if not torch.is_tensor(post_r):
                continue
            if tuple(post_c.shape) != tuple(post_r.shape):
                return False, rtol, atol, {"contract": f"mutable input[{i}] post-state shape "
                                           f"{tuple(post_c.shape)} != {tuple(post_r.shape)}"}
            # same assert_close default-tolerance check as the outputs (cast ref to candidate dtype)
            ok, why = _assert_close_ok(post_r.to(post_c.dtype), post_c)
            if not ok:
                return False, rtol, atol, {"contract": f"mutable input[{i}] post-state mismatch "
                                           "(declared mutable but not updated correctly)",
                                           **_err_metrics(post_r, post_c)}
        # (5) tensor-mode backward (VJP): if grad_inputs is declared, the operator has a Jacobian
        # to verify even though the forward returns tensors (not a loss). The judge independently
        # recomputes the reference's VJP grads with the SAME seeded cotangent (deterministic in
        # `seed`) and compares against the candidate's grads_c — the backward is checked in tensor
        # mode too (plan review P0(b)). Reference inputs get requires_grad on a fresh same-seed set.
        if grad_idx and (cand_data.get("grads_c") is not None):
            grads_c = cand_data["grads_c"]
            ins_g = [x.to(dev) if torch.is_tensor(x) else x for x in mk(shape, dev, seed)]
            for i in grad_idx:
                if i < len(ins_g) and torch.is_tensor(ins_g[i]) and ins_g[i].is_floating_point():
                    ins_g[i].requires_grad_(True)
            raw_rg = ref(*ins_g)
            grads_r = _vjp_grads(raw_rg, ins_g, grad_idx, out_indices, seed)
            gdiffs = {}
            for i in grad_idx:
                gr = grads_r.get(i)
                gc_ = grads_c.get(i) if isinstance(grads_c, dict) else None
                gc_ = _to_dev(gc_)
                if gr is None and gc_ is None:
                    continue
                if gr is None or gc_ is None:
                    return False, rtol, atol, {"contract": f"grad[{i}] missing (VJP)", "grads": gdiffs}
                gdiffs[i] = _err_metrics(gr, gc_)
                gok, _gw = _assert_close_ok(gr.to(gc_.dtype), gc_)
                if not gok:
                    return False, rtol, atol, {"contract": f"grad[{i}] mismatch (VJP)", "grads": gdiffs}
            metrics = {**metrics, "grads_vjp": gdiffs}
        return True, rtol, atol, metrics

    # loss mode
    loss_c, grads_c = cand_data["loss_c"], cand_data["grads_c"]
    loss_r, grads_r = _fwd_bwd(ref, ins_r, grad_idx, out_idx)
    # correctness = assert_close default tolerance on the scalar loss AND every declared grad
    # (a training operator is defined by its backward). Compare each in the candidate's dtype.
    if not math.isfinite(loss_c):
        return False, None, None, {"loss_diff": "non-finite candidate loss"}
    lr_t = torch.as_tensor(loss_r); lc_t = torch.as_tensor(loss_c)
    loss_ok, _lw = _assert_close_ok(lr_t.to(lc_t.dtype), lc_t)
    grad_ok, gdiffs = True, {}
    for i in grad_idx:
        gr = grads_r[i]
        gc_ = grads_c.get(i) if isinstance(grads_c, dict) else grads_c[i]
        gc_ = _to_dev(gc_)
        if gr is None and gc_ is None:
            continue
        if gr is None or gc_ is None:
            grad_ok = False; gdiffs[i] = "missing"; continue
        gdiffs[i] = _err_metrics(gr, gc_)
        gok, _gw = _assert_close_ok(gr.to(gc_.dtype), gc_)
        grad_ok = grad_ok and gok
    return bool(loss_ok and grad_ok), None, None, {"loss_diff": round(abs(loss_r - loss_c), 6), "grads": gdiffs}


def _check_correct(ref, cand, refmod, shape, dev, seed, C):
    """In-process correctness (candidate + judge in the SAME process). Kept as the composition
    of _candidate_side + _judge_side so the standalone/offline path and unit tests keep working;
    the process-isolated scoring path (score() -> candidate subprocess -> judge) calls the two
    halves in SEPARATE processes so the candidate cannot tamper with the judge's comparison.
    Returns (ok, rtol, atol, metrics)."""
    cand_data = _candidate_side(cand, refmod, shape, dev, seed, C)
    return _judge_side(ref, refmod, shape, dev, seed, C, cand_data)



def _sample_hidden_shapes(refmod, public_shapes, n, harness_seed, shape_axes=None,
                          constraints=None):
    """Draw hidden (unseen) shapes for the anti-hardcode tripwire.

    PRIMARY (trusted): draw from the declared domain via workload.sample_hidden_shapes —
    the harness owns this, so the untrusted reference cannot shape the hidden distribution.
    FALLBACK (only if no domain declared): the reference's own sample_shapes(rng, k). The
    fallback is a last resort and should be rare; a task with a declared shape_axes always
    uses the trusted path."""
    if shape_axes:
        try:
            from autoresearch.workload import sample_hidden_shapes as _hs
        except Exception:
            from workload import sample_hidden_shapes as _hs
        return _hs(shape_axes, public_shapes, n, harness_seed, constraints=constraints)
    # fallback: untrusted reference sampler (no declared domain to draw from)
    import random
    seen = {tuple(s) for s in public_shapes}
    fn = getattr(refmod, "sample_shapes", None)
    rng = random.Random(harness_seed)
    cand = []
    if callable(fn):
        try:
            cand = [tuple(s) for s in fn(rng, n * 4)]
        except Exception:
            cand = []
    out = []
    for s in cand:
        if s not in seen:
            out.append(s); seen.add(s)
        if len(out) >= n:
            break
    return out


def _dryrun_filter_hidden(refmod, Model, init_args, hidden, dev, C):
    """P0#4: keep only hidden shapes the REFERENCE can actually build + run (small dry-run).

    A declared domain can emit shapes make_inputs/forward cannot handle (wrong dimensionality,
    illegal axis combo, OOM). Scoring a candidate against such a shape would fail an honest
    kernel for the reference's defect. We run the reference on each hidden shape at one seed;
    any shape that raises is dropped. Returns the surviving shapes (order preserved)."""
    keep = []
    for shape in hidden:
        try:
            ins = refmod.make_inputs(shape, dev, 0)
            ins = [x.to(dev) if torch.is_tensor(x) else x for x in ins]
            ref = Model(*init_args).to(dev)
            with torch.no_grad():
                _ = ref(*ins)
            keep.append(shape)
            del ref, ins
        except Exception:
            # reference cannot handle this shape -> not a valid tripwire shape, drop it
            pass
        finally:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return keep


# ============================================================================
#  the score worker
# ============================================================================
def _cpu(x):
    """Move a tensor (or container of tensors) to CPU for cross-process transport, detached."""
    if torch.is_tensor(x):
        return x.detach().to("cpu")
    if isinstance(x, (tuple, list)):
        return type(x)(_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    return x


def _dev_types(raw, out_indices):
    """Record the ORIGINAL device type of each declared output index, so the judge can enforce
    the device-match contract after the candidate output has been moved to CPU for transport."""
    dm = {}
    for oi in out_indices:
        o = _scalar(raw, oi)
        if torch.is_tensor(o):
            dm[oi] = o.device.type
    return dm


def run_candidate_worker(spec_dict: dict, candidate_src: str, device: int,
                         seeds, time_seed, out_dir: str, phase: str = "all") -> dict:
    """CANDIDATE-SIDE worker (untrusted). Loads ONLY the candidate, runs it on every public +
    hidden shape, and writes its raw outputs + timings to `out_dir`. It performs NO allclose,
    NO reward, NO verdict — the trusted judge (a SEPARATE process that never imports this
    candidate) re-derives all reference truth and renders the verdict (P0#1). A candidate that
    monkeypatches torch.allclose / json.dumps therefore corrupts nothing the judge trusts; the
    worst it can do is emit wrong tensors, which the judge's own comparison rejects.

    phase (P0 §3.4 physical correctness/benchmark separation):
      * "correctness": run ONLY correctness (all seeds) on public AND hidden shapes; no timing.
      * "timing":      run ONLY timing on public shapes; correctness outputs are not re-emitted.
      * "all" (default): the legacy single-pass behavior (public correctness+timing, hidden
        correctness) — kept so direct callers / the re-export parity contract are unchanged.
    The judge spawns "correctness" first and only spawns "timing" (twice, for a cross-process
    agreement check — P0 §3.3) AFTER correctness passes, so a wrong kernel is never benchmarked.

    Returns a manifest dict (also written to out_dir/manifest.json) describing what was produced.
    Timing IS measured here (interleaved same-context fairness needs the candidate live); the
    judge applies sanity gates + recomputes the timed-capture reference truth itself and requires
    the two timing processes to agree."""
    dev = f"cuda:{device}"
    torch.cuda.set_device(device)
    interface = spec_dict.get("interface", {})
    tim = spec_dict.get("timing", {})
    wl = spec_dict.get("workload", {})
    C = {
        "task_mode": interface.get("task_mode", "tensor"),
        "grad_inputs": interface.get("grad_inputs", []),
        "output_index": (interface.get("output_indices") or [0])[0],
        "output_indices": list(interface.get("output_indices") or [0]),
        "mutable_inputs": list(interface.get("mutable_inputs") or []),
        # G6 candidate timing is measured over the SAME budget G4 uses for the baseline/official
        # denominators (_do_bench_ms: warmup=25, rep=100). The old warmup=5/iters=20 gave the two
        # gates wildly different noise budgets — G4's 100-rep median was steady while G6's 20-sample
        # window jittered — so a witness could clear G4's frontier yet trip G6's timing gate purely
        # from measurement noise ("G4 green, G6 dead" was a code-allowed systematic outcome). Match
        # them so a timing verdict reflects the kernel, not the sample size.
        "warmup": tim.get("warmup_iters", 25), "iters": tim.get("iters", 100),
    }
    mode, grad_idx, out_idx = C["task_mode"], C["grad_inputs"], C["output_index"]
    entry = interface.get("entry_class", "ModelNew")

    refmod = _load_module_from_src(spec_dict["ref_arch_src"], "kf_ref")   # for make_inputs + naive timing denominator
    candmod = _load_module_from_src(candidate_src, "kf_cand")
    if not hasattr(candmod, entry):
        return {"fatal": f"candidate does not define class {entry}"}
    init_args = refmod.get_init_inputs()
    Model = refmod.Model
    Cand = getattr(candmod, entry)
    public_shapes = [tuple(s) for s in wl.get("public_shapes", [])]

    # the hidden shapes are drawn by the TRUSTED judge and passed in via the manifest request;
    # the candidate worker only runs what it is told (see score() which writes hidden into req).
    hidden_shapes = [tuple(s) for s in (spec_dict.get("_hidden_shapes") or [])]

    manifest = {"public": [], "hidden": []}

    def _run_shape(shape, do_timing, do_correct=True):
        rec = {"shape": list(shape)}
        try:
            cand = Cand(*init_args).to(dev)
        except Exception as e:
            rec["err"] = f"init: {e}"; return rec
        # correctness outputs per seed (skipped in the pure-timing phase, where correctness has
        # already passed in an earlier process — P0 §3.4).
        if do_correct:
            seed_files = {}
            for sd in seeds:
                cd = _candidate_side(cand, refmod, shape, dev, sd, C)
                payload = {"mode": cd["mode"]}
                if cd["mode"] == "approx":
                    payload["idx_c"] = _cpu(cd["idx_c"])
                elif cd["mode"] == "tensor":
                    payload["raw_c"] = _cpu(cd["raw_c"])
                    payload["dev_map"] = _dev_types(cd["raw_c"], C["output_indices"])
                    payload["mutated"] = cd["mutated"]
                    payload["mut_post"] = _cpu(cd.get("mut_post") or {})
                    # tensor-mode backward (VJP): transport the candidate's grads so the judge can
                    # compare them against its own seeded-cotangent recompute (plan review P0(b)).
                    if cd.get("grads_c") is not None:
                        payload["grads_c"] = _cpu(cd["grads_c"])
                else:  # loss
                    payload["loss_c"] = cd["loss_c"]; payload["grads_c"] = _cpu(cd["grads_c"])
                fpath = os.path.join(out_dir, f"cand_{'_'.join(map(str, shape))}_s{sd}.pt")
                torch.save(payload, fpath)
                seed_files[sd] = os.path.basename(fpath)
                gc.collect(); torch.cuda.empty_cache()
            rec["seed_files"] = seed_files
        if not do_timing:
            return rec

        # timing (P1#7 phase-switch resistant) — measured candidate-side; judge recomputes the
        # timed-capture reference truth from cap_seed and applies sanity gates.
        try:
            cand_t = Cand(*init_args).to(dev); ref_t = Model(*init_args).to(dev)
        except Exception as e:
            rec["err"] = f"timing_init: {e}"; return rec

        def fresh_ins(sd):
            xs = [x.to(dev) if torch.is_tensor(x) else x for x in refmod.make_inputs(shape, dev, sd)]
            torch.cuda.synchronize(); return xs

        def run_once(model, ins, capture=False):
            if mode in ("tensor", "approx"):
                with torch.no_grad():
                    o = _scalar(model(*ins), out_idx)
                return (o.detach().clone() if capture else None)
            for i in grad_idx:
                if torch.is_tensor(ins[i]) and ins[i].grad is not None:
                    ins[i].grad = None
            l = _scalar(model(*ins), out_idx); l.backward()
            if capture:
                gr = {i: (ins[i].grad.detach().clone() if torch.is_tensor(ins[i]) and ins[i].grad is not None else None)
                      for i in grad_idx}
                return (float(l.detach()), gr)
            return None

        def peak_of(model, sd):
            ins = fresh_ins(sd)
            for _ in range(3):
                run_once(model, ins)
            for i in grad_idx:
                if torch.is_tensor(ins[i]) and ins[i].grad is not None:
                    ins[i].grad = None
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            fb, _ = torch.cuda.mem_get_info(device)
            run_once(model, ins); torch.cuda.synchronize()
            fa, _ = torch.cuda.mem_get_info(device)
            m = max(torch.cuda.max_memory_allocated(device) / 1e6, max(0.0, fb - fa) / 1e6)
            del ins; return m

        try:
            m_cand = peak_of(cand_t, time_seed + 500); m_naive = peak_of(ref_t, time_seed + 500)
            timing_attempts = []
            for retry_index, measured_iters in enumerate(_timing_attempt_iters(tim)):
                t_cand, t_naive, cvc, cvn, cap, cap_seed = _time_candidate(
                    cand_t, ref_t, fresh_ins, run_once,
                    base_seed=time_seed + retry_index * 9973,
                    warmup=C["warmup"], iters=measured_iters, grad_idx=grad_idx)
                timing_attempts.append({
                    "seed": time_seed + retry_index * 9973,
                    "iters": measured_iters,
                    "candidate_cv": cvc,
                    "naive_cv": cvn,
                })
                if max(cvc, cvn) <= tim.get("cv_threshold", MAX_TIMING_CV):
                    break
        except Exception as e:
            rec["err"] = f"measure: {type(e).__name__}: {e}"; return rec

        rec.update(t_cand=t_cand, t_naive=t_naive, cv_cand=cvc, cv_naive=cvn,
                   mem_cand=m_cand, mem_naive=m_naive, cap_seed=cap_seed,
                   timing_retries=len(timing_attempts) - 1,
                   effective_iters=timing_attempts[-1]["iters"],
                   timing_attempts=timing_attempts)
        if cap is not None and cap.get("got") is not None:
            capf = os.path.join(out_dir, f"cap_{'_'.join(map(str, shape))}.pt")
            torch.save({"got": _cpu(cap["got"])}, capf)
            rec["cap_file"] = os.path.basename(capf)
        del cand_t, ref_t; gc.collect(); torch.cuda.empty_cache()
        return rec

    if phase == "correctness":
        # Phase A: correctness only, on public AND hidden shapes; no timing.
        for shape in public_shapes:
            manifest["public"].append(_run_shape(shape, do_timing=False, do_correct=True))
            gc.collect(); torch.cuda.empty_cache()
        for shape in hidden_shapes:
            manifest["hidden"].append(_run_shape(shape, do_timing=False, do_correct=True))
            gc.collect(); torch.cuda.empty_cache()
    elif phase == "timing":
        # Phase B: timing only, on public shapes (correctness already passed in Phase A). No
        # correctness artifacts are re-emitted; the candidate is freshly reloaded + inputs freshly
        # generated in THIS process, so nothing is reused from the correctness objects (P0 §3.4).
        for shape in public_shapes:
            manifest["public"].append(_run_shape(shape, do_timing=True, do_correct=False))
            gc.collect(); torch.cuda.empty_cache()
    else:  # "all": legacy single-pass (public correctness+timing, hidden correctness)
        for shape in public_shapes:
            manifest["public"].append(_run_shape(shape, do_timing=True, do_correct=True))
            gc.collect(); torch.cuda.empty_cache()
        for shape in hidden_shapes:
            manifest["hidden"].append(_run_shape(shape, do_timing=False, do_correct=True))
            gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


def _time_candidate(cand, ref, fresh_ins, run_once, base_seed, warmup, iters, grad_idx=()):
    """Candidate-side timing: like _time_and_capture but returns the candidate's captured output
    `got` WITHOUT computing the reference truth `exp` (the judge recomputes exp from cap_seed so
    a candidate cannot supply a self-serving reference). Returns
    (t_cand, t_naive, cv_cand, cv_naive, {"got":...}, cap_seed)."""
    import random
    rng = random.Random(base_seed)
    cap_iter = rng.randrange(iters) if iters > 0 else -1
    for k in range(warmup):
        run_once(cand, fresh_ins(base_seed + 1000 + k))
        run_once(ref, fresh_ins(base_seed + 2000 + k))
    torch.cuda.synchronize()
    tc, tn, cap, cap_seed = [], [], None, None
    ev = lambda: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for k in range(iters):
        want_cap = (k == cap_iter)
        # SAME workload instance for candidate and reference at iteration k: make_inputs is
        # deterministic in its seed, so one seed yields identical values in two SEPARATE tensor
        # objects (separate objects so a candidate's in-place write can't corrupt the reference's
        # timing input). This removes the cand-vs-ref input asymmetry the review flagged (a
        # data-dependent kernel — sort/sparse/varlen — could otherwise be timed on an easier or
        # harder instance than the baseline). Phase-switch resistance (P1#7) is unchanged: the
        # seed still changes every iteration k, so the candidate cannot recognise a fixed input.
        ins_seed = base_seed + 10000 + k
        ic = fresh_ins(ins_seed)
        s0, e0 = ev(); s0.record()
        got = run_once(cand, ic, capture=want_cap); e0.record()
        in_ = fresh_ins(ins_seed)   # identical values, separate tensors -> fair same-instance timing
        s1, e1 = ev(); s1.record(); run_once(ref, in_); e1.record()
        torch.cuda.synchronize()
        tc.append(s0.elapsed_time(e0)); tn.append(s1.elapsed_time(e1))
        if want_cap:
            cap = {"got": got}; cap_seed = ins_seed
        del ic, in_
    mc, cvc = _med_cv(tc); mn, cvn = _med_cv(tn)
    return mc, mn, cvc, cvn, cap, cap_seed


# ============================================================================
#  the score worker (JUDGE side)
# ============================================================================
def _judge_recompute_capture(Model, refmod, shape, dev, cap_seed, C, init_args):
    """Judge-side recompute of the timed-capture reference truth on the candidate's cap_seed
    inputs — the judge NEVER trusts a candidate-supplied reference. Mirrors run_once's capture."""
    mode, grad_idx, out_idx = C["task_mode"], C["grad_inputs"], C["output_index"]
    ref = Model(*init_args).to(dev)
    ins = [x.to(dev) if torch.is_tensor(x) else x for x in refmod.make_inputs(shape, dev, cap_seed)]
    ins = _clone_inputs(ins, grad_idx)
    if mode in ("tensor", "approx"):
        with torch.no_grad():
            return _scalar(ref(*ins), out_idx).detach().clone()
    l = _scalar(ref(*ins), out_idx); l.backward()
    gr = {i: (ins[i].grad.detach().clone() if torch.is_tensor(ins[i]) and ins[i].grad is not None else None)
          for i in grad_idx}
    return (float(l.detach()), gr)


def _wait_and_cleanup_process_group(proc, pgid, timeout_s, before_wait=None):
    """Wait for one candidate worker, then kill its saved process group unconditionally.

    A worker can return successfully while a forked descendant keeps running.  Because the worker
    is launched with ``start_new_session=True``, its pid is also the stable process-group id; use
    that saved id even after the leader exits instead of querying a now-gone leader with getpgid.
    ``before_wait`` runs inside this function's cleanup-protected ``try``. The spawn path uses it
    to restore a temporarily blocked termination-signal mask only after ``proc`` and ``pgid`` are
    owned by this ``finally``. Returns whether the leader exceeded the deadline.
    """
    import signal
    import subprocess

    timed_out = False
    try:
        if before_wait is not None:
            before_wait()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if proc.poll() is None:
            proc.wait()
    return timed_out


class _CandidateProcessRegistry:
    """Main-thread termination handshake for candidate sessions spawned by score threads.

    The internal verifier deliberately runs ``score()`` in a background thread to keep its
    private result nonce off the candidate-facing stack. Python delivers signals on the main
    thread, however, so the background thread cannot install cleanup handlers. ``spawn`` holds
    one lock across Popen and PGID registration; a simultaneous main-thread ``terminate_all``
    therefore cannot observe the unsafe Popen-return gap. Once termination starts, future spawns
    fail closed.
    """

    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self._active = {}
        self._terminating = False

    def spawn(self, spawn_fn):
        with self._lock:
            if self._terminating:
                raise RuntimeError("verifier termination already requested")
            proc = spawn_fn()
            pgid = proc.pid  # start_new_session=True makes pid the stable process-group id.
            self._active[pgid] = proc
            return proc, pgid

    def unregister(self, pgid):
        with self._lock:
            self._active.pop(pgid, None)

    def terminate_all(self):
        # Keep the lock through killpg: unregister cannot race us into signalling a reused PGID,
        # and spawn cannot create an unregistered session after the termination snapshot.
        with self._lock:
            self._terminating = True
            for pgid, proc in list(self._active.items()):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.kill()
                    except OSError:
                        pass


_ACTIVE_CANDIDATE_REGISTRY = None


@contextlib.contextmanager
def _worker_candidate_termination_handshake():
    """Install main-thread SIGTERM/SIGINT cleanup for the background score thread."""
    global _ACTIVE_CANDIDATE_REGISTRY
    import threading

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("worker candidate termination handshake requires the main thread")

    registry = _CandidateProcessRegistry()
    previous_registry = _ACTIVE_CANDIDATE_REGISTRY
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    termination_requested = False

    def _terminate(signum, frame):
        nonlocal termination_requested
        termination_requested = True
        registry.terminate_all()
        raise SystemExit(128 + signum)

    _ACTIVE_CANDIDATE_REGISTRY = registry
    try:
        for sig in previous_handlers:
            signal.signal(sig, _terminate)
        yield registry
    finally:
        # Normal completion should have no active candidate, but fail closed if a future score
        # path returns without joining one. On signal unwind this call is idempotent.
        registry.terminate_all()
        if not termination_requested:
            _ACTIVE_CANDIDATE_REGISTRY = previous_registry
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        # On signal unwind the daemon score thread may still be between candidate phases. Keep
        # the terminating registry globally installed until process death so every future spawn
        # attempt fails closed; restoring None here would reopen the cross-thread escape.


def _block_candidate_spawn_signals():
    """Block cleanup signals across Popen -> cleanup ownership; return an idempotent restorer.

    A SIGTERM handler can otherwise unwind from inside ``Popen`` after it creates a new candidate
    session but before the returned process is assigned locally. Main-thread callers close that
    window with a per-thread POSIX mask. Background callers are deliberately a no-op because Python
    installs signal handlers only on the main thread; they retain the internal timeout cleanup.
    """
    import threading

    if (threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "pthread_sigmask")):
        return lambda: None

    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        # Set the guard before unblocking: delivery of a pending signal may raise from inside
        # pthread_sigmask before the call returns, but the OS mask has already been restored.
        restored = True
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    return restore


@contextlib.contextmanager
def _candidate_cleanup_signal_handlers():
    """Turn outer verifier termination into a Python unwind so worker cleanup finally runs."""
    import threading

    if threading.current_thread() is not threading.main_thread():
        # signal.signal is main-thread-only. Background callers still get the hard per-worker
        # timeout and unconditional process-group cleanup from _wait_and_cleanup_process_group.
        yield
        return

    previous = {}

    def _terminate(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _terminate)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _run_candidate_subprocess(spec_dict, candidate_src, device, seeds, time_seed, hidden,
                              timeout_s=1200, candidate_import_dirs=None, phase="all"):
    """Spawn the CANDIDATE worker as its own subprocess so the untrusted candidate code never
    shares a process with the judge (P0#1). The worker writes raw output artifacts to a shared
    dir; here we only pass inputs + collect the manifest. A crash / timeout / missing manifest ->
    fatal (the judge scores reward 0). No nonce is needed: the worker emits only DATA (never a
    verdict), so the worst a hostile candidate can do is emit wrong tensors, which the judge's own
    comparison rejects.

    phase is forwarded to run_candidate_worker (P0 §3.4): "correctness" | "timing" | "all". Each
    call gets its OWN fresh art_dir + fresh subprocess, so a timing process cannot reuse any object
    or state from the correctness process.

    Spawned by THIS FILE's path, not `-m autoresearch.verifier`, so the exact same core works
    when copied into a task container that has no autoresearch package installed. Running the
    core as a script puts its own directory on sys.path[0], so the lone-file
    `from kernel_static_checker import ...` fallback resolves beside it.

    candidate_import_dirs: extra dirs to put on the worker's PYTHONPATH so the candidate's
    `from reference_model import Model` (the delivered AutoResearch solve.py starter) resolves. The
    internal path passes nothing here (witnesses are standalone; ref_arch_src is self-contained)."""
    art_dir = tempfile.mkdtemp(prefix="kf_cand_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    req = dict(spec_dict); req["_hidden_shapes"] = [list(s) for s in hidden]
    req_f = os.path.join(art_dir, "req.json"); cand_f = os.path.join(art_dir, "cand.py")
    with open(req_f, "w") as f:
        json.dump(req, f)
    with open(cand_f, "w") as f:
        f.write(candidate_src or "")
    cmd = [sys.executable, os.path.abspath(__file__), "--candidate",
           "--spec-json", req_f, "--candidate-file", cand_f, "--device", str(device),
           "--art-dir", art_dir, "--seeds", ",".join(str(s) for s in seeds),
           "--time-seed", str(time_seed), "--phase", phase]
    env = dict(os.environ)
    # pkg_parent lets an in-repo worker resolve `from autoresearch...`; this file's own dir lets a
    # lone-file (in-container) worker resolve `from kernel_static_checker import ...`; the
    # candidate_import_dirs let the candidate resolve `from reference_model import Model`.
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self_dir = os.path.dirname(os.path.abspath(__file__))
    path_parts = [pkg_parent, self_dir] + list(candidate_import_dirs or [])
    env["PYTHONPATH"] = os.pathsep.join(path_parts + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    import subprocess
    out_f = os.path.join(art_dir, "cand.out")
    registry = _ACTIVE_CANDIDATE_REGISTRY
    registered_pgid = None
    try:
        with _candidate_cleanup_signal_handlers():
            with open(out_f, "w") as outfh:
                restore_spawn_signals = _block_candidate_spawn_signals()
                try:
                    def _spawn():
                        return subprocess.Popen(
                            cmd, stdout=outfh, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)

                    if registry is not None:
                        proc, pgid = registry.spawn(_spawn)
                        registered_pgid = pgid
                    else:
                        proc = _spawn()
                        pgid = proc.pid
                    if _wait_and_cleanup_process_group(
                            proc, pgid, timeout_s, before_wait=restore_spawn_signals):
                        return {"fatal": f"candidate exceeded {timeout_s}s (killed)",
                                "art_dir": art_dir}
                finally:
                    # Popen failures have no process to reap but must still restore the caller's
                    # original mask. Normal and signalled paths already restored it idempotently.
                    restore_spawn_signals()
        tail = ""
        try:
            with open(out_f) as f:
                tail = f.read()[-400:]
        except OSError:
            pass
        if proc.returncode != 0:
            return {"fatal": f"candidate worker exited {proc.returncode}: {tail}", "art_dir": art_dir}
        mpath = os.path.join(art_dir, "manifest.json")
        if not os.path.exists(mpath):
            return {"fatal": f"candidate produced no manifest: {tail}", "art_dir": art_dir}
        manifest = json.loads(open(mpath).read())
        if manifest.get("fatal"):
            return {"fatal": manifest["fatal"], "art_dir": art_dir}
        return {"art_dir": art_dir, "manifest": manifest}
    except Exception as e:
        return {"fatal": f"candidate spawn failed: {type(e).__name__}: {e}", "art_dir": art_dir}
    finally:
        if registry is not None and registered_pgid is not None:
            registry.unregister(registered_pgid)


def score(spec_dict: dict, candidate_src: str, device: int = 0,
          seeds_override=None, time_seed_override=None, candidate_import_dirs=None,
          target_id=_TARGET_FROM_ENV) -> dict:
    """Judge candidate_src against the spec dict. Returns a flat-ish result dict. This is the
    trusted judge body: it loads ONLY the reference here and spawns the candidate in a separate
    subprocess (P0#1). candidate_import_dirs is forwarded to that subprocess's PYTHONPATH so a
    delivered candidate's `from reference_model import Model` resolves; the internal path leaves
    it empty."""
    dev = f"cuda:{device}"
    torch.cuda.set_device(device)

    # --- contract knobs from the spec ---
    interface = spec_dict.get("interface", {})
    tol = spec_dict.get("tolerance", {})
    res = spec_dict.get("resources", {})
    tim = spec_dict.get("timing", {})
    wl = spec_dict.get("workload", {})
    scope = spec_dict.get("scope", {})
    baselines = spec_dict.get("baselines", {})
    precision_note = spec_dict.get("precision", {})
    C = {
        "task_mode": interface.get("task_mode", "tensor"),
        "grad_inputs": interface.get("grad_inputs", []),
        "output_index": (interface.get("output_indices") or [0])[0],
        "output_indices": list(interface.get("output_indices") or [0]),
        "mutable_inputs": list(interface.get("mutable_inputs") or []),
        "output_dtype": precision_note.get("output_dtype"),
        "entry_class": interface.get("entry_class", "ModelNew"),
        "allow_ops": interface.get("allow_torch_ops", []),
        # correctness is torch.testing.assert_close at its default dtype-derived tolerance
        # (Kernel Test Bench doc §精度测试) — no per-task rtol/atol or policy caps are read here.
        "quality_floor": tol.get("quality_floor"), "quality_slack": tol.get("quality_slack"),
        "mem_weight": res.get("mem_weight", 0.5),
        "reg_floor": wl.get("worst_slice_regression_floor", DEFAULT_REGRESSION_FLOOR),
        "cache_regime": tim.get("cache_regime", "steady_state_warm"),
        # keep in lockstep with the candidate worker's timing budget (see run_candidate_worker):
        # aligned to G4's _do_bench_ms (warmup=25, rep=100) so both gates share a noise budget.
        "warmup": tim.get("warmup_iters", 25), "iters": tim.get("iters", 100),
        "cv_thr": tim.get("cv_threshold", MAX_TIMING_CV),
        "peak_mem_cap": res.get("peak_memory_mb"),
        # optional hard latency ceiling: a correct-but-too-slow candidate (t_cand > ceiling*t_naive
        # on any shape) is rejected. Read from interface.latency_ceiling (the delivered path maps
        # reference_model.LATENCY_CEILING here). None -> no ceiling.
        "lat_ceiling": interface.get("latency_ceiling"),
        "direct_lib": scope.get("direct_library_call_allowed", False),
        "frontier": baselines.get("frontier_headroom") or baselines.get("library_headroom"),
        # P0 §3.1: keep the two frontier components as SEPARATE fields so the report never claims
        # "max(library, compile) is the library headroom". frontier == max(library, compile); the
        # components are surfaced for provenance (which one actually set the frontier).
        "library_headroom": baselines.get("library_headroom"),
        "compile_headroom": baselines.get("compile_headroom"),
        # whether an official STRONG baseline (library / aten) exists — decides speedup_vs_strong.
        # (a bare torch.compile frontier is NOT a strong-library baseline: vs_strong stays null.)
    }
    # reward anchor: when the task declares an OFFICIAL version-locked baseline, reward is anchored
    # to it (speedup_vs_frontier), because "beat the official implementation" is the honest target.
    # When no official baseline exists, reward stays anchored to the naive seed (speedup_vs_seed),
    # the dense training signal. The frontier number itself is the official headroom (measured by
    # baselines.measure_portfolio when official was declared) or the oracle frontier otherwise.
    _ob = (spec_dict.get("official_baseline") or {})
    C["official_declared"] = (_ob.get("kind") or "none") not in ("none", "")
    # HW3: the frontier is a PER-TARGET number. When an active GPU target is declared (env
    # KF_TARGET_ID), read the frontier from spec.calibrations[target_id] — and FAIL CLOSED if
    # that target has no calibration, rather than silently reusing the legacy scalar (which was
    # measured on a DIFFERENT GPU). Reusing H20's 20x frontier to score an H100 run is exactly
    # the cross-hardware transfer bug. No target_id declared -> legacy single-target behavior.
    if target_id is _TARGET_FROM_ENV:
        target_id = os.environ.get("KF_TARGET_ID") or None
    resolve_target_frontier(C, spec_dict, target_id)
    if C.get("frontier_uncalibrated"):
        return {
            "reward": 0.0,
            "correctness": 0,
            "performance": "not_run",
            "reason": f"uncalibrated_target:{target_id}",
            "reward_anchor": "uncalibrated_target",
            "target_id": target_id,
        }
    if target_id:
        calibration = (spec_dict.get("calibrations", {}) or {}).get(target_id) or {}
        try:
            actual_model = torch.cuda.get_device_name(device)
            actual_capability = torch.cuda.get_device_capability(device)
            actual_cc = f"{actual_capability[0]}.{actual_capability[1]}"
        except Exception as ex:
            return {
                "reward": 0.0,
                "correctness": 0,
                "performance": "not_run",
                "reason": f"target_identity_probe_failed:{type(ex).__name__}:{ex}",
                "target_id": target_id,
            }
        expected_model = calibration.get("gpu_model")
        expected_cc = str(calibration.get("compute_capability"))
        if actual_model != expected_model or actual_cc != expected_cc:
            return {
                "reward": 0.0,
                "correctness": 0,
                "performance": "not_run",
                "reason": (
                    "target_identity_mismatch: "
                    f"expected {expected_model} cc={expected_cc}, "
                    f"observed {actual_model} cc={actual_cc}"
                ),
                "target_id": target_id,
            }
    public_shapes = [tuple(s) for s in wl.get("public_shapes", [])]
    entry = C["entry_class"]
    grad_idx, out_idx, mode = C["grad_inputs"], C["output_index"], C["task_mode"]
    mem_w = C["mem_weight"]; reg_floor = C["reg_floor"]
    seeds = list(seeds_override) if seeds_override else CORRECT_SEEDS[:]
    time_seed = time_seed_override if time_seed_override is not None else TIME_SEED

    # --- static anti-hack gate (before running candidate) ---
    ok_gate, gate_reason = static_gate(candidate_src, allow_ops=C["allow_ops"],
                                       direct_library_call_allowed=C["direct_lib"])
    if not ok_gate:
        return {"reward": 0.0, "correctness": 0, "reason": gate_reason}

    # --- load ONLY the reference (oracle) in THIS trusted judge process. The candidate is NEVER
    #     imported here: it runs in a separate subprocess (run_candidate_worker) so it cannot
    #     monkeypatch torch.allclose / json.dumps / seeds in the process that renders the verdict
    #     (P0#1). The judge re-derives ALL reference truth itself. ---
    try:
        refmod = _load_module_from_src(spec_dict["ref_arch_src"], "kf_ref")
    except Exception as e:
        return {"reward": 0.0, "correctness": 0, "reason": f"ref exec failed: {e}",
                "trace": traceback.format_exc()[-500:]}
    init_args = refmod.get_init_inputs()
    Model = refmod.Model

    # hidden-shape tripwire: the domain + count + constraints are the harness's, drawn by the
    # trusted workload.sample_hidden_shapes; the seed is non-constant so the candidate cannot
    # precompute the unseen set (P1#6). Sampled HERE (judge side) and handed to the candidate
    # worker, so the candidate cannot choose its own hidden shapes.
    shape_axes = wl.get("shape_axes") or {}
    n_hidden = int(wl.get("n_hidden_shapes") or N_HIDDEN_SHAPES)
    # P0#4: cross_axis_constraints (e.g. "H % Hkv == 0") is the machine-readable contract the
    # sampler must respect so it never draws an illegal axis combination the reference cannot
    # build. Accept the legacy shape_constraints key too.
    constraints = wl.get("cross_axis_constraints") or wl.get("shape_constraints") or None
    hidden = _sample_hidden_shapes(refmod, public_shapes, n_hidden, _harness_seed(),
                                   shape_axes=shape_axes, constraints=constraints)
    # P0#4 dry-run admission: a drawn hidden shape is only usable if the reference can actually
    # build inputs + run on it. Drop any shape whose make_inputs/forward raises (dimension
    # mismatch, OOM, illegal combo) so a bad domain declaration degrades to fewer hidden shapes
    # instead of scoring an honest candidate 0 on a shape the reference itself cannot handle.
    if hidden:
        hidden = _dryrun_filter_hidden(refmod, Model, init_args, hidden, dev, C)
    if shape_axes and not hidden:
        return {"reward": 0.0, "correctness": 0,
                "reason": "hidden_sampler_empty (declared shape_axes drew no unseen shapes)",
                "per_shape": [], "hidden_shapes": []}

    # ========================================================================
    # PHASE A — correctness only (P0 §3.4). Spawn an isolated candidate subprocess that runs
    # correctness on ALL public + hidden shapes and does NO timing. The judge re-derives every
    # reference truth here. Only if Phase A fully passes do we spawn the benchmark (Phase B), so a
    # wrong / OOM / runaway kernel is never benchmarked and yields no performance number.
    # ========================================================================
    cand_ret = _run_candidate_subprocess(spec_dict, candidate_src, device, seeds, time_seed,
                                         hidden, timeout_s=res.get("eval_timeout_s", 1200),
                                         candidate_import_dirs=candidate_import_dirs,
                                         phase="correctness")
    if cand_ret.get("fatal"):
        return {"reward": 0.0, "correctness": 0, "reason": cand_ret["fatal"],
                "performance": "not_run"}
    artA = cand_ret["art_dir"]
    manifestA = cand_ret["manifest"]

    def _loadA(fn):
        return torch.load(os.path.join(artA, fn), map_location="cpu")

    results, all_ok = [], True
    corr_by_shape = {}   # shape-tuple -> per-shape correctness entry (reused to attach timing)
    for mrec in manifestA.get("public", []):
        shape = tuple(mrec["shape"])
        ent = {"shape": list(shape)}
        if mrec.get("err"):
            ent.update(correct=False, err=mrec["err"]); results.append(ent); all_ok = False
            corr_by_shape[shape] = ent; continue
        try:
            ref = Model(*init_args).to(dev)
        except Exception as e:
            ent.update(correct=False, err=f"ref_init: {e}"); results.append(ent); all_ok = False
            corr_by_shape[shape] = ent; continue
        # correctness: judge recomputes the reference per seed and compares the candidate's
        # transported output.
        try:
            per_seed, ok_all, rtol, atol = {}, True, None, None
            for sd in seeds:
                cand_data = _loadA(mrec["seed_files"][str(sd)] if str(sd) in mrec["seed_files"]
                                   else mrec["seed_files"][sd])
                ok, rtol, atol, metrics = _judge_side(ref, refmod, shape, dev, sd, C, cand_data)
                per_seed[sd] = {"ok": ok, **metrics}; ok_all = ok_all and ok
                del cand_data; gc.collect(); torch.cuda.empty_cache()
            correct = bool(ok_all); ent["seeds"] = per_seed
            if rtol is not None:
                ent["tol"] = [rtol, atol]
        except Exception as e:
            ent.update(correct=False, err=f"correctness: {type(e).__name__}: {e}")
            results.append(ent); all_ok = False; corr_by_shape[shape] = ent
            del ref; continue
        all_ok = all_ok and correct
        ent["correct"] = correct
        del ref
        results.append(ent); corr_by_shape[shape] = ent
        gc.collect(); torch.cuda.empty_cache()

    hidden_results = []
    for mrec in manifestA.get("hidden", []):
        shape = tuple(mrec["shape"])
        h = {"shape": list(shape)}
        if mrec.get("err"):
            h.update(correct=False, err=mrec["err"]); all_ok = False; hidden_results.append(h); continue
        try:
            ref = Model(*init_args).to(dev)
            hs_ok = True
            for sd in seeds:
                cand_data = _loadA(mrec["seed_files"][str(sd)] if str(sd) in mrec["seed_files"]
                                   else mrec["seed_files"][sd])
                ok, *_ = _judge_side(ref, refmod, shape, dev, sd, C, cand_data)
                hs_ok = hs_ok and ok; del cand_data; gc.collect(); torch.cuda.empty_cache()
            h["correct"] = bool(hs_ok); all_ok = all_ok and hs_ok
            del ref
        except Exception as e:
            h.update(correct=False, err=f"hidden_shape: {type(e).__name__}: {e}"); all_ok = False
        hidden_results.append(h); gc.collect(); torch.cuda.empty_cache()
    try:
        import shutil; shutil.rmtree(artA, ignore_errors=True)
    except Exception:
        pass

    # Phase A gate: correctness must fully pass before any benchmark runs. On failure the
    # performance fields are explicitly not_run (P0 §3.4 acceptance: a wrong candidate carries no
    # valid runtime_ms / speedup).
    if not all_ok:
        return {"reward": 0.0, "correctness": 0, "reason": "incorrect", "performance": "not_run",
                "runtime_ms": None, "native_runtime_ms": None, "strong_runtime_ms": None,
                "speedup_vs_native": None, "speedup_vs_strong": None,
                "per_shape": results, "hidden_shapes": hidden_results}

    # ========================================================================
    # PHASE B — timing only (P0 §3.4 + §3.3). Correctness has passed. Spawn the benchmark in a
    # FRESH subprocess that reloads the candidate and regenerates inputs from scratch (no object /
    # state reuse from Phase A). To keep timing out of the candidate's sole control we run TWO
    # independent timing processes and require their per-shape composites to AGREE within a
    # tolerance; a candidate whose timing is not reproducible across processes is flagged noisy
    # (reward 0). The judge still recomputes the timed-capture reference truth + applies sanity
    # gates on each process's numbers.
    # ========================================================================
    tim_runs = []
    for rep in range(2):
        tret = _run_candidate_subprocess(spec_dict, candidate_src, device, seeds,
                                         time_seed + rep * 104729, hidden,
                                         timeout_s=res.get("eval_timeout_s", 1200),
                                         candidate_import_dirs=candidate_import_dirs,
                                         phase="timing")
        if tret.get("fatal"):
            return {"reward": 0.0, "correctness": 0,
                    "reason": f"benchmark_phase_failed: {tret['fatal']}", "performance": "not_run",
                    "per_shape": results, "hidden_shapes": hidden_results}
        tim_runs.append(tret)

    def _judge_timing_run(art_dir, manifest):
        """Apply the timed-capture + sanity gates to one timing process's manifest; return
        {shape-tuple: {t_cand,t_naive,cv_cand,cv_naive,mem_cand,mem_naive,composite,fail?}}."""
        def _load(fn):
            return torch.load(os.path.join(art_dir, fn), map_location="cpu")
        out = {}
        for mrec in manifest.get("public", []):
            shape = tuple(mrec["shape"])
            info = {}
            if mrec.get("err"):
                info["fail"] = mrec["err"]; out[shape] = info; continue
            t_cand = mrec.get("t_cand"); t_naive = mrec.get("t_naive")
            cvc = mrec.get("cv_cand", 1.0); cvn = mrec.get("cv_naive", 1.0)
            m_cand = mrec.get("mem_cand", 0.0); m_naive = mrec.get("mem_naive", 0.0)
            if t_cand is None or t_naive is None or t_cand <= 0 or t_naive <= 0:
                info["fail"] = "timing_missing_or_nonpositive"; out[shape] = info; continue
            # phase-switch tripwire: recompute reference on the candidate's cap_seed inputs.
            if mrec.get("cap_file") and mrec.get("cap_seed") is not None:
                try:
                    ref = Model(*init_args).to(dev)
                    got = _load(mrec["cap_file"])["got"]
                    exp = _judge_recompute_capture(Model, refmod, shape, dev, mrec["cap_seed"],
                                                   C, init_args)
                    tw_ok, tw_reason = _verify_timed_capture({"got": got, "exp": exp},
                                                             mode, grad_idx, C)
                    del ref, got, exp; gc.collect(); torch.cuda.empty_cache()
                    if not tw_ok:
                        info["fail"] = f"timed_output_mismatch: {tw_reason}"; out[shape] = info; continue
                except Exception as e:
                    info["fail"] = f"capture_check_failed: {type(e).__name__}: {e}"
                    out[shape] = info; continue
            if C["peak_mem_cap"] is not None and m_cand > C["peak_mem_cap"]:
                info["fail"] = f"memory_fail:{m_cand:.0f}MB>cap{C['peak_mem_cap']}MB"; out[shape] = info; continue
            if C["lat_ceiling"] is not None and t_naive > 0 and t_cand > C["lat_ceiling"] * t_naive:
                info["fail"] = f"latency_fail:{t_cand:.1f}>{C['lat_ceiling']}x*{t_naive:.1f}"; out[shape] = info; continue
            noisy = max(cvc, cvn) > C["cv_thr"]
            comp = (((m_naive / m_cand) ** mem_w) * ((t_naive / t_cand) ** (1 - mem_w))
                    if (m_cand > 0 and t_cand > 0 and not noisy) else 0.0)
            info.update(t_cand=t_cand, t_naive=t_naive, cv_cand=cvc, cv_naive=cvn,
                        mem_cand=m_cand, mem_naive=m_naive, noisy=noisy, composite=comp,
                        timing_retries=mrec.get("timing_retries", 0),
                        effective_iters=mrec.get("effective_iters", C["iters"]))
            out[shape] = info
        return out

    judged = [_judge_timing_run(t["art_dir"], t["manifest"]) for t in tim_runs]
    for t in tim_runs:
        try:
            import shutil; shutil.rmtree(t["art_dir"], ignore_errors=True)
        except Exception:
            pass

    # Merge the two timing processes into each shape's result entry, requiring cross-process
    # agreement (P0 §3.3: timing is not trusted from a single candidate-controlled run). The
    # correctness-vs-performance split (external review #2) lives in _merge_timing_runs /
    # _perf_verdict, kept torch-free so it is unit-testable without a GPU.
    functional_ok, timing_ok = _merge_timing_runs(results, judged[0], judged[1], xproc_tol=0.25)
    gc.collect(); torch.cuda.empty_cache()
    comps = [r["composite_vs_naive"] for r in results
             if r.get("correct") and r.get("composite_vs_naive")]   # None/0 excluded
    geo = math.exp(sum(math.log(c) for c in comps) / len(comps)) if comps else 0.0
    worst = min(comps) if comps else 0.0

    _early = _perf_verdict(functional_ok, timing_ok, geo, worst, reg_floor, results, hidden_results)
    if _early is not None:
        return _early

    frontier = C["frontier"]
    peak_mems = [r["mem_cand_mb"] for r in results if r.get("correct") and "mem_cand_mb" in r]
    peak_mem = round(max(peak_mems), 1) if peak_mems else None
    # reward anchor (torch-free helper): official baseline declared + usable frontier ->
    # reward = speedup_vs_frontier; otherwise reward = speedup_vs_seed. Both are always reported.
    ra = resolve_reward_anchor(
        geo,
        frontier,
        bool(C.get("official_declared")),
        frontier_uncalibrated=bool(C.get("frontier_uncalibrated")),
    )
    vs_frontier = ra["speedup_vs_frontier"]
    # P0 §3.1 fixed objective fields. runtime_ms is the candidate's steady-state wall-clock
    # (geomean of per-shape median t_cand); native/strong runtimes derive from the measured
    # speedups so the three are mutually consistent.
    cand_ts = [r["t_cand_ms"] for r in results if r.get("correct") and r.get("t_cand_ms")]
    naive_ts = [r["t_naive_ms"] for r in results if r.get("correct") and r.get("t_naive_ms")]
    runtime_ms = round(_geomean(cand_ts), 4) if cand_ts else None
    native_runtime_ms = round(_geomean(naive_ts), 4) if naive_ts else None
    speedup_vs_native = ra["speedup_vs_seed"]           # explicit checklist name (== vs_seed)
    # speedup_vs_strong is null unless a real STRONG library baseline exists. A torch.compile-only
    # frontier is NOT a strong-library baseline, so we do NOT fabricate a vs_strong from it
    # (P0 §3.1: strong absent -> null, never faked as native). library_headroom is the measured
    # strong-library speedup over naive; geo/library_headroom = candidate speedup over the library.
    lib_hr = C.get("library_headroom")
    has_strong = bool(C.get("official_declared")) and bool(lib_hr and lib_hr > 0)
    speedup_vs_strong = round(geo / lib_hr, 4) if has_strong else None
    strong_runtime_ms = (round(native_runtime_ms / lib_hr, 4)
                         if (native_runtime_ms and has_strong) else None)
    out = {"reward": ra["reward"], "correctness": 1, "performance_valid": True,
           "speedup_vs_seed": ra["speedup_vs_seed"], "speedup_vs_frontier": vs_frontier,
           "reward_anchor": ra["reward_anchor"],
           # fixed objective triplet (P0 §3.1): always present, strong is null when absent.
           "runtime_ms": runtime_ms, "native_runtime_ms": native_runtime_ms,
           "strong_runtime_ms": strong_runtime_ms,
           "speedup_vs_native": speedup_vs_native, "speedup_vs_strong": speedup_vs_strong,
           # frontier provenance: components kept separate so nothing mislabels max() as library.
           "frontier_headroom": frontier, "library_headroom": lib_hr,
           "compile_headroom": C.get("compile_headroom"),
           "peak_mem_mb": peak_mem, "worst_slice": round(worst, 4),
           "cache_regime": C["cache_regime"], "frontier_ref": frontier,
           "target_id": C.get("target_id"),
           "per_shape": results, "hidden_shapes": hidden_results}
    if "reward_note" in ra:
        out["reward_note"] = ra["reward_note"]
    if C.get("frontier_uncalibrated"):
        # The target-specific denominator is absent, so this run is diagnostic-only: retain its
        # correctness and naive-relative measurements, but make certification/reward fail closed.
        out["reason"] = "uncalibrated_target"
        out["frontier_note"] = f"no_calibration_for_target:{C.get('target_id')}"
    return out


# ============================================================================
#  subprocess entrypoint
# ============================================================================
def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--candidate", action="store_true",
                    help="CANDIDATE-side mode: load only the candidate, run it, write raw output "
                         "artifacts to --art-dir. No verdict, no reference comparison (P0#1).")
    ap.add_argument("--art-dir", default=None, help="output dir for candidate artifacts")
    ap.add_argument("--phase", default="all", choices=["all", "correctness", "timing"],
                    help="candidate-worker phase (P0 §3.4): correctness-only, timing-only, or "
                         "the legacy single pass.")
    ap.add_argument("--spec-json", required=True)
    ap.add_argument("--candidate-file", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--time-seed", type=int, default=None)
    ap.add_argument("--result-fd", type=int, default=None,
                    help="private inherited fd to write the result JSON to (anti-forgery). "
                         "The candidate never sees this fd and the result is written only "
                         "AFTER the candidate has run, by trusted verifier code.")
    ap.add_argument("--control-fd", type=int, default=None,
                    help="private inherited fd carrying the per-run secret nonce. Read + closed "
                         "here BEFORE any candidate code runs, so the candidate can never learn "
                         "the nonce and cannot forge a nonce-authenticated result write.")
    a = ap.parse_args()

    # CANDIDATE-side mode: run ONLY the candidate, emit artifacts, exit. This process is the
    # untrusted one; it never renders a verdict (the judge in the parent process does), so it
    # needs no nonce/result-fd. A crash here => parent scores reward 0.
    if a.candidate:
        spec_dict = json.loads(open(a.spec_json).read())
        cand_src = open(a.candidate_file).read()
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()] or CORRECT_SEEDS[:]
        ts = a.time_seed if a.time_seed is not None else TIME_SEED
        try:
            run_candidate_worker(spec_dict, cand_src, a.device, seeds, ts, a.art_dir, phase=a.phase)
        except BaseException as e:
            try:
                with open(os.path.join(a.art_dir, "manifest.json"), "w") as f:
                    json.dump({"fatal": f"{type(e).__name__}: {str(e)[:300]}"}, f)
            except Exception:
                pass
            raise
        return

    # Read the secret nonce from the control fd and close it, BEFORE loading/executing the
    # candidate. The nonce then lives only in this local variable; the candidate has no
    # reference to it and the fd is already closed, so a candidate that scans fds/argv cannot
    # forge a result that starts with _SENTINEL+nonce (plan P0#1).
    nonce = ""
    if a.control_fd is not None:
        try:
            parts = []
            while True:
                b = os.read(a.control_fd, 4096)
                if not b:
                    break
                parts.append(b)
            nonce = b"".join(parts).decode("utf-8", errors="replace")
        finally:
            try:
                os.close(a.control_fd)
            except OSError:
                pass

    spec_dict = json.loads(open(a.spec_json).read())
    cand_src = open(a.candidate_file).read()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] or None
    result_fd = a.result_fd

    # Run score() (which loads + executes the untrusted candidate) in a WORKER THREAD, not on
    # _main's stack. This is the structural half of the P0#1 defense: a candidate that walks
    # sys._getframe().f_back... only ever sees frames of its OWN thread, whose base is _run()
    # below — it can never reach _main's frame where `nonce` and `result_fd` live. Combined
    # with the static ban on gc/inspect/io/threading (heap + cross-thread frame access), the
    # trusted secret is unreachable from candidate code. argv is scrubbed as belt-and-braces.
    sys.argv = [sys.argv[0]]

    box = {}
    def _run():
        # BaseException, not Exception: a candidate's top-level `raise SystemExit(0)` /
        # KeyboardInterrupt must NOT bypass the trusted write. Any candidate-triggered exit
        # becomes reward 0, written by trusted code.
        try:
            box["res"] = score(spec_dict, cand_src, device=a.device,
                               seeds_override=seeds, time_seed_override=a.time_seed)
        except BaseException as e:
            box["res"] = {"reward": 0.0, "correctness": 0, "reason": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()[-600:]}
    import threading as _threading
    t = _threading.Thread(target=_run, name="kf_score", daemon=True)
    # Signals are delivered only on this main thread while candidate sessions are created by
    # _run. The registry handshake closes that cross-thread ownership gap before the outer
    # sandbox is allowed to terminate this verifier worker.
    with _worker_candidate_termination_handshake():
        t.start(); t.join()
    res = box.get("res") or {"reward": 0.0, "correctness": 0, "reason": "no result from score thread"}

    payload = json.dumps(res, ensure_ascii=False)
    # Write the result ONLY to the private inherited fd, prefixed with the secret nonce, never
    # to stdout. stdout is shared with the candidate's top-level code, so a candidate could
    # print a forged sentinel there; the private fd requires the nonce (which the candidate
    # cannot read — see thread isolation above) so it cannot be spoofed. Falls back to stdout
    # only in standalone/debug use (no --result-fd), which is never the scoring path.
    if result_fd is not None:
        with os.fdopen(result_fd, "w", closefd=True) as rf:
            rf.write(_SENTINEL + nonce + payload)
            rf.flush()
    else:
        print(_SENTINEL + payload)


if __name__ == "__main__":
    _main()
