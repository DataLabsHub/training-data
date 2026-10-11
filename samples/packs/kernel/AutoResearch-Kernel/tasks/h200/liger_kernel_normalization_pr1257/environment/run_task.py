#!/usr/bin/env python3
"""AutoResearch/Harbor scorer for a mid-scale autoresearch-kernel task — THIN ADAPTER over the shared core.

The scoring engine is autoresearch/_score_core.py (copied into this container beside this file
as `_score_core.py`). This module is a thin adapter so the DELIVERED task verifier is byte-for-
byte the SAME engine used to certify the task internally — no second, weaker implementation. That
means the container gets, for free, everything the core has:

  - process-role split: the candidate runs in a SEPARATE subprocess from the trusted judge, so a
    candidate that monkeypatches torch.allclose / json.dumps cannot forge a verdict;
  - tensor-mode backward via a seeded VJP (a training/fusion kernel's backward is CHECKED, not just
    its forward);
  - timed-invocation output capture (a phase-switching candidate that serves garbage on the timed
    call is caught);
  - candidate and naive timed on the SAME input instance (fair data-dependent timing).

The task is still described the AutoResearch way — a `reference_model.py` module exposing Model /
make_inputs / get_init_inputs plus config attributes (WORKLOAD, TASK_MODE, GRAD_INPUTS, ...). This
adapter reads those attributes and maps them into the core's spec-dict contract, then calls
core.score(). Correctness = torch.testing.assert_close default dtype tolerance; timing =
CUDA-event interleaved median (the protocol calibration froze `headroom` under).

    python run_task.py --task-dir <dir with reference_model.py> --candidate solve.py \
        --headroom <float> --reward-anchor {naive,official}
"""
import os, sys, json, argparse, importlib.util, math, hashlib, traceback

# the shared scoring core ships beside this file in the container (converter copies it in); in-repo
# it is importable as autoresearch._score_core. Try the lone-file name first (container), then
# the package path (in-repo / tests).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import _score_core as _core
except Exception:
    from autoresearch import _score_core as _core


# verifier-private seeds: correctness must hold on ALL of these; timing uses TIME_SEED. Mirrors the
# core defaults so a run without --seeds behaves identically to the internal path.
CORRECT_SEEDS = list(_core.CORRECT_SEEDS)
TIME_SEED = _core.TIME_SEED

# The exporter replaces this marker with a canonical JSON string. Keeping ``None`` in the shared
# source makes the in-repo legacy adapter importable before generation.
_FROZEN_SCORING_CONTRACT_JSON = '{"calibration":{"atol_empirical":0.01,"baseline_comparable":true,"baseline_cv":0.0521,"baseline_fallback":false,"baseline_impl":"liger_kernel.ops.rms_norm.LigerRMSNormFunction.apply","calibration_digest":"909fadc6d318e646bb1409b3cf0cd7421e746544e7c807dd6fc073a650be0c4e","certification_level":"gpu_certified","compile_headroom":2.4528,"compute_capability":"9.0","domain_cap_atol":0.04,"domain_cap_rtol":0.04,"effective_timing":{"backend":"cuda_event_interleaved_median","cache_regime":"steady_state_warm","cv_threshold":0.08,"iters":20,"max_retries":1,"retry_iters_multiplier":2,"warmup_iters":5},"env_profile_digest":"1199fa43e852d9278eeffebea93fd76f7ab5fc78923780ab682e444330440eaf","env_profile_ref":"profiles/h200_nv2505.json","frontier_headroom":2.3714,"frontier_source":"official","frontier_timing_noisy":false,"gpu_model":"NVIDIA H200","library_headroom":null,"official_coverage_complete":true,"official_eligible":true,"official_headroom":2.3714,"official_timing_noisy":false,"official_version_eligible":true,"peak_memory_mb":null,"portfolio":{"16384x8192":{"compile":3.115,"official":3.6276,"seed":0.9298},"4096x2048":{"compile":1.9981,"official":1.3702,"seed":0.9088},"4096x4096":{"compile":2.2086,"official":2.0529,"seed":0.9292},"4096x8192":{"compile":2.6135,"official":2.6251,"seed":0.9314},"8192x4096":{"compile":2.4711,"official":2.7998,"seed":0.9393}},"rtol_empirical":0.016,"seed_cv":0.0058,"target_id":"h200_sm90_cuda129_torch280_nv2505","timing_backend":"cuda_event_interleaved_median","timing_cv":null,"timing_retries":0,"witness_found":true},"official_baseline":{"adapter":true,"call_path":"official interface for liger_kernel_normalization_pr1257","callable":"liger_kernel.ops.rms_norm.LigerRMSNormFunction.apply","kind":"library","overload":"","version_pin":"liger-kernel==0.8.1"},"reward_anchor":"official","target_id":"h200_sm90_cuda129_torch280_nv2505","target_required":true,"task_id":"liger_kernel_normalization_pr1257","timing":{"cache_regime":"steady_state_warm","cv_threshold":0.08,"iters":20,"max_retries":1,"retry_iters_multiplier":2,"warmup_iters":5}}'

_CALIBRATION_DIGEST_FIELDS = (
    "target_id", "gpu_model", "compute_capability", "env_profile_ref",
    "env_profile_digest", "frontier_headroom", "official_headroom",
    "library_headroom", "compile_headroom", "frontier_source",
    "certification_level", "official_eligible", "official_coverage_complete",
    "official_version_eligible", "frontier_timing_noisy", "official_timing_noisy",
    "timing_backend", "baseline_cv", "seed_cv", "timing_retries",
    "effective_timing", "baseline_impl", "baseline_fallback", "baseline_comparable",
)


def _frozen_scoring_contract():
    if _FROZEN_SCORING_CONTRACT_JSON is None:
        return None
    return json.loads(_FROZEN_SCORING_CONTRACT_JSON)


def _calibration_contract_digest(contract):
    calibration = dict(contract.get("calibration") or {})
    payload = {
        "schema": 2,
        "task_id": contract.get("task_id"),
        "target_required": bool(contract.get("target_required")),
        "target_id": contract.get("target_id"),
        "calibration": {
            key: calibration.get(key) for key in _CALIBRATION_DIGEST_FIELDS
        },
        "official_baseline": dict(
            contract.get("official_baseline") or {"kind": "none"}),
        "timing": dict(contract.get("timing") or {}),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_scoring_contract(contract):
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("frozen scoring contract must be an object")
    target_required = bool(contract.get("target_required"))
    target_id = contract.get("target_id")
    calibration = contract.get("calibration")
    targeted_evidence = isinstance(calibration, dict) and any(
        calibration.get(field) for field in (
            "target_id", "gpu_model", "compute_capability", "env_profile_ref",
            "env_profile_digest", "calibration_digest"))
    if not target_required and (target_id or targeted_evidence):
        raise ValueError(
            "target_required=false cannot contain targeted calibration evidence")
    timing = dict(contract.get("timing") or {})
    for name, default, minimum in (
        ("iters", 20, 1),
        ("max_retries", 1, 0),
        ("retry_iters_multiplier", 2, 1),
    ):
        value = timing.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"frozen timing {name} must be an integer >= {minimum}")
    if target_required:
        if not contract.get("task_id"):
            raise ValueError("target-required scoring contract is missing task_id")
        if not target_id:
            raise ValueError("target-required scoring contract is missing target_id")
        if not isinstance(calibration, dict):
            raise ValueError("target-required scoring contract is missing calibration")
        if calibration.get("target_id") != target_id:
            raise ValueError("frozen target/calibration identity mismatch")
        for field in ("gpu_model", "compute_capability", "env_profile_ref"):
            if not calibration.get(field):
                raise ValueError(f"frozen target calibration is missing {field}")
        frontier = calibration.get("frontier_headroom")
        if (not isinstance(frontier, (int, float)) or isinstance(frontier, bool)
                or not math.isfinite(frontier) or frontier <= 0):
            raise ValueError("frozen target calibration has no positive finite frontier")
        if calibration.get("certification_level") not in (
                "gpu_certified", "production_signed"):
            raise ValueError("frozen target calibration is not certified")
        for field in ("env_profile_digest", "calibration_digest"):
            value = calibration.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"frozen target calibration is missing {field}")
        if calibration.get("frontier_timing_noisy") or calibration.get("official_timing_noisy"):
            raise ValueError("frozen target calibration has noisy timing")
        if calibration.get("calibration_digest") != _calibration_contract_digest(contract):
            raise ValueError("frozen target calibration digest mismatch")

        official = dict(contract.get("official_baseline") or {"kind": "none"})
        if (official.get("kind") or "none") not in ("", "none"):
            official_headroom = calibration.get("official_headroom")
            official_ok = (
                calibration.get("frontier_source") == "official"
                and isinstance(official_headroom, (int, float))
                and not isinstance(official_headroom, bool)
                and math.isfinite(official_headroom) and official_headroom > 0
                and calibration.get("official_eligible") is True
                and calibration.get("official_coverage_complete") is True
                and calibration.get("official_version_eligible") is True
                and calibration.get("baseline_fallback") is False
                and calibration.get("baseline_comparable") is True
            )
            if not official_ok:
                raise ValueError("frozen targeted official baseline is not eligible")
    return contract


def _read(path):
    with open(path) as f:
        return f.read()


def _load_ref_module(task_dir):
    """Import reference_model.py ONLY to read its config attributes (WORKLOAD, TASK_MODE, ...).
    The reference CODE itself is passed to the core as a source string; this import never runs the
    candidate."""
    path = os.path.join(task_dir, "reference_model.py")
    sys.modules.pop("reference_model", None)
    spec = importlib.util.spec_from_file_location("reference_model", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["reference_model"] = m
    spec.loader.exec_module(m)
    return m


def _build_spec_dict(ref_mod, ref_src, headroom, reward_anchor, library_headroom=None,
                     compile_headroom=None, scoring_contract=None):
    """Map the reference_model.py module attributes -> the core's spec-dict contract. This is the
    ONE adapter boundary between the AutoResearch task convention and the shared engine.

    headroom is the FRONTIER anchor = max(library, compile). library_headroom / compile_headroom
    are the SEPARATE components (P0 §3.1): the core reports speedup_vs_strong only when a real
    strong library exists, and never mislabels the compile frontier as the library headroom."""
    task_mode = getattr(ref_mod, "TASK_MODE", "loss")
    grad_inputs = list(getattr(ref_mod, "GRAD_INPUTS", []) or [])
    out_idx = getattr(ref_mod, "OUTPUT_INDEX", 0)
    output_indices = list(getattr(ref_mod, "OUTPUT_INDICES", [out_idx]) or [out_idx])
    baseline_contract = getattr(ref_mod, "BASELINE_CONTRACT", {}) or {}
    workload = [list(s) for s in getattr(ref_mod, "WORKLOAD", [])]
    # the reward anchor: when the task declares an official baseline (reward_anchor=="official"),
    # the core anchors reward to the frontier headroom (speedup over the official kernel); else it
    # stays on the naive seed. headroom is the frozen calibration number either way.
    contract = _validate_scoring_contract(scoring_contract)
    official = {"kind": "library" if reward_anchor == "official" else "none"}
    timing = {
        "cache_regime": "steady_state_warm",
        "cv_threshold": getattr(ref_mod, "MAX_TIMING_CV", _core.MAX_TIMING_CV),
    }
    baselines = {"frontier_headroom": headroom,
                 "library_headroom": library_headroom,
                 "compile_headroom": compile_headroom}
    calibrations = {}
    target_id = None
    if contract is not None:
        calibration = contract.get("calibration") or {}
        target_id = contract.get("target_id") if contract.get("target_required") else None
        baselines = {
            "frontier_headroom": calibration.get("frontier_headroom"),
            "official_headroom": calibration.get("official_headroom"),
            "library_headroom": calibration.get("library_headroom"),
            "compile_headroom": calibration.get("compile_headroom"),
            "frontier_source": calibration.get("frontier_source"),
        }
        official = dict(contract.get("official_baseline") or {"kind": "none"})
        timing.update(contract.get("timing") or {})
        if target_id:
            calibrations[target_id] = dict(calibration)
    spec_dict = {
        "task_id": getattr(ref_mod, "TASK_ID", os.path.basename(os.path.dirname(
            os.path.abspath(getattr(ref_mod, "__file__", "task/x"))))),
        "ref_arch_src": ref_src,
        "interface": {
            "task_mode": task_mode,
            "entry_class": "ModelNew",
            "grad_inputs": grad_inputs,
            "output_indices": output_indices,
            "mutable_inputs": list(getattr(ref_mod, "MUTABLE_INPUTS", []) or []),
            "allow_torch_ops": list(getattr(ref_mod, "ALLOW_TORCH_OPS", []) or []),
            "latency_ceiling": getattr(ref_mod, "LATENCY_CEILING", None),
        },
        "precision": {"output_dtype": baseline_contract.get("output_dtype")},
        "tolerance": {
            "quality_floor": getattr(ref_mod, "QUALITY_FLOOR", None),
            "quality_slack": getattr(ref_mod, "QUALITY_SLACK", None),
        },
        "workload": {
            "public_shapes": workload,
            "worst_slice_regression_floor": getattr(ref_mod, "REGRESSION_FLOOR",
                                                    _core.DEFAULT_REGRESSION_FLOOR),
            # opt into the core's trusted hidden-shape sampler when the task declares SHAPE_AXES;
            # otherwise the core falls back to the reference's own sample_shapes(rng, n).
            "shape_axes": getattr(ref_mod, "SHAPE_AXES", {}) or {},
            "cross_axis_constraints": list(getattr(ref_mod, "CROSS_AXIS_CONSTRAINTS", []) or []),
            "n_hidden_shapes": int(getattr(ref_mod, "N_HIDDEN_SHAPES", _core.N_HIDDEN_SHAPES)),
        },
        "timing": timing,
        "resources": {
            "mem_weight": getattr(ref_mod, "MEM_WEIGHT", 0.5),
            "peak_memory_mb": getattr(ref_mod, "PEAK_MEMORY_MB", None),
            "eval_timeout_s": int(getattr(ref_mod, "EVAL_TIMEOUT_S", 1200)),
        },
        "scope": {"direct_library_call_allowed": bool(getattr(ref_mod, "DIRECT_LIBRARY_CALL", False))},
        "baselines": baselines,
        "official_baseline": official,
        "calibrations": calibrations,
        "target_required": bool(contract and contract.get("target_required")),
        "frozen_target_id": target_id,
    }
    return spec_dict


def score(task_dir, cand_path, headroom, device=0, seeds_override=None, time_seed_override=None,
          reward_anchor="naive", library_headroom=None, compile_headroom=None,
          scoring_contract=None):
    """Return (reward, correct, detail). Signature preserved for run_local.py / test.sh callers
    (library_headroom / compile_headroom are optional trailing args, default None).

    headroom is the frozen calibration anchor. reward_anchor: "official" -> reward = geo/headroom
    (beat the official baseline); "naive" (default) -> reward = geo (naive-seed denominator).

    Delegates ALL scoring to the shared core (process-isolated judge + VJP + timed capture). The
    candidate + reference dirs are threaded to the core so the isolated candidate subprocess can
    resolve `from reference_model import Model` (the AutoResearch solve.py starter)."""
    try:
        ref_mod = _load_ref_module(task_dir)
    except Exception as e:
        return 0.0, False, {"reason": f"reference_model load failed: {type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-500:]}
    ref_src = _read(os.path.join(task_dir, "reference_model.py"))
    try:
        cand_src = _read(cand_path)
    except OSError as e:
        return 0.0, False, {"reason": f"cannot read candidate: {e}"}

    try:
        contract = _validate_scoring_contract(
            scoring_contract if scoring_contract is not None else _frozen_scoring_contract())
        spec_dict = _build_spec_dict(ref_mod, ref_src, headroom, reward_anchor,
                                     library_headroom=library_headroom,
                                     compile_headroom=compile_headroom,
                                     scoring_contract=contract)
    except ValueError as e:
        return 0.0, False, {"reason": f"invalid frozen scoring contract: {e}"}
    cand_dir = os.path.dirname(os.path.abspath(cand_path))
    import_dirs = [task_dir, cand_dir]

    core_kwargs = dict(device=device, seeds_override=seeds_override,
                       time_seed_override=time_seed_override,
                       candidate_import_dirs=import_dirs)
    if contract is not None:
        # A generated artifact is routed by its protected frozen contract, never ambient process
        # state. Passing ``None`` explicitly also immunizes an untargeted frozen artifact.
        core_kwargs["target_id"] = spec_dict.get("frozen_target_id")
    res = _core.score(spec_dict, cand_src, **core_kwargs)

    # convert the core's rich result dict -> the (reward, correct, detail) tuple this file's callers
    # expect, preserving every legacy detail key (run_local.py prints them; main() writes reward.json).
    reward = float(res.get("reward", 0.0))
    correct = bool(res.get("correctness"))
    detail = dict(res)
    detail.pop("reward", None); detail.pop("correctness", None)
    # legacy alias: old run_task reported speedup(==speedup_vs_seed) and headroom_ref(==frontier_ref)
    if "speedup" not in detail and detail.get("speedup_vs_seed") is not None:
        detail["speedup"] = detail["speedup_vs_seed"]
    if "headroom_ref" not in detail and detail.get("frontier_ref") is not None:
        detail["headroom_ref"] = detail["frontier_ref"]
    return reward, correct, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True, help="dir holding reference_model.py")
    ap.add_argument("--candidate", required=True, help="path to solve.py (defines ModelNew)")
    ap.add_argument("--headroom", type=float, default=0.0,
                    help="official-baseline speedup over naive from calibration; the reward "
                         "denominator when --reward-anchor=official")
    ap.add_argument("--reward-anchor", choices=["naive", "official"], default="naive",
                    help="official -> reward = geo/headroom (beat the official baseline); naive -> reward = geo")
    ap.add_argument("--library-headroom", type=float, default=None,
                    help="strong-library speedup over naive (P0 §3.1). Separate from --headroom "
                         "(the frontier = max(library, compile)); used to report speedup_vs_strong.")
    ap.add_argument("--compile-headroom", type=float, default=None,
                    help="torch.compile(naive) speedup over naive; reported for frontier provenance.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seeds", default="", help="comma-sep hidden correctness seeds (default 0,1,2)")
    ap.add_argument("--time-seed", type=int, default=None, help="hidden timing seed (default 7)")
    ap.add_argument("--reward-file", default="/logs/verifier/reward.json")
    ap.add_argument("--reward-txt", default="/logs/verifier/reward.txt")
    a = ap.parse_args()
    seeds_override = [int(s) for s in a.seeds.split(",") if s.strip()] or None
    time_seed_used = a.time_seed if a.time_seed is not None else TIME_SEED
    seeds_used = seeds_override if seeds_override else CORRECT_SEEDS[:]
    try:
        reward, correct, detail = score(a.task_dir, a.candidate, a.headroom, a.device,
                                        seeds_override=seeds_override, time_seed_override=a.time_seed,
                                        reward_anchor=a.reward_anchor,
                                        library_headroom=a.library_headroom,
                                        compile_headroom=a.compile_headroom)
    except Exception as e:
        reward, correct, detail = 0.0, False, {"reason": f"{type(e).__name__}: {e}",
                                               "trace": traceback.format_exc()[-600:]}
    # Record the exact hidden seeds so a failing run is reproducible.
    detail.setdefault("correctness_seeds", seeds_used)
    detail.setdefault("timing_seed", time_seed_used)
    # Harbor parses reward.json as a FLAT dict of numbers only; nested lists/strings/bools raise
    # ValidationError. Keep the official reward file numeric; dump rich diagnostics alongside.
    os.makedirs(os.path.dirname(a.reward_file), exist_ok=True)
    flat = {"reward": float(reward), "correctness": 1 if correct else 0}
    for k in ("speedup", "speedup_vs_seed", "speedup_vs_frontier", "peak_mem_mb", "worst_slice",
              "headroom_ref", "runtime_ms", "native_runtime_ms", "strong_runtime_ms",
              "speedup_vs_native", "speedup_vs_strong", "frontier_headroom", "library_headroom",
              "compile_headroom"):
        if isinstance(detail.get(k), (int, float)):
            flat[k] = detail[k]
    with open(a.reward_file, "w") as f:
        json.dump(flat, f, indent=2)
    with open(a.reward_txt, "w") as f:
        f.write(str(reward))
    detail_path = os.path.join(os.path.dirname(a.reward_file), "reward_detail.json")
    with open(detail_path, "w") as f:
        json.dump({"reward": reward, "correctness": correct, **detail}, f, indent=2, ensure_ascii=False)
    print(json.dumps({"reward": reward, "correctness": correct,
                      "speedup_vs_seed": detail.get("speedup_vs_seed"),
                      "speedup_vs_frontier": detail.get("speedup_vs_frontier"),
                      "peak_mem_mb": detail.get("peak_mem_mb"),
                      "worst_slice": detail.get("worst_slice"),
                      "headroom_ref": detail.get("headroom_ref"),
                      "reason": detail.get("reason")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
