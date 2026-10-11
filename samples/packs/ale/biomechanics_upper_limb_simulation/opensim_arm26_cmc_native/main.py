"""ALE task implementation for biomechanics_upper_limb_simulation/opensim_arm26_cmc_native.

The private grader is intentionally inlined in this file. The reference/
directory contains only ground-truth artifacts; grader-only contracts live
under grader_assets/.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig


logger = logging.getLogger(__name__)
from pathlib import Path as _AlePromptPath


def _load_solver_prompt() -> str:
    return (_AlePromptPath(__file__).resolve().parent / "solver_prompt.txt").read_text(encoding="utf-8")

TASK_DIR = Path(__file__).resolve().parent
TASK_NAME = "opensim_arm26_cmc_native"
TASK_CATEGORY = "biomechanics_upper_limb_simulation"


# ---- inline private grader: main.py ----

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


TASK_ID = "opensim_arm26_cmc_native"
REQUIRED_OPENSIM_VERSION_TOKEN = "4.5"

STORAGE_CHECKS = [
    ("ik/arm26_InverseKinematics.mot", "reference/arm26_InverseKinematics.mot", 1e-5, 1e-6),
    ("cmc/arm26_states.sto", "reference/arm26_states.sto", 5e-4, 5e-5),
    ("cmc/arm26_controls.sto", "reference/arm26_controls.sto", 5e-4, 5e-5),
    ("cmc/arm26_Kinematics_q.sto", "reference/arm26_Kinematics_q.sto", 5e-4, 5e-5),
    ("cmc/arm26_pErr.sto", "reference/arm26_pErr.sto", 5e-4, 5e-5),
]

HASHED_OUTPUTS = [
    "setup/arm26_Setup_InverseKinematics.xml",
    "setup/arm26_Setup_ComputedMuscleControl.xml",
    "ik/arm26_InverseKinematics.mot",
    "cmc/arm26_states.sto",
    "cmc/arm26_controls.sto",
    "cmc/arm26_Kinematics_q.sto",
    "cmc/arm26_pErr.sto",
    "run_log.txt",
]

XML_CHECKS = [
    {
        "path": "setup/arm26_Setup_InverseKinematics.xml",
        "tool": "InverseKinematicsTool",
        "must_contain": ["arm26.osim", "arm26_elbow_flex.trc", "arm26_InverseKinematics.mot"],
        "tags": {
            "model_file": "arm26.osim",
            "marker_file": "arm26_elbow_flex.trc",
            "output_motion_file": "arm26_InverseKinematics.mot",
            "time_range": "0.00000000 1.00000000",
        },
    },
    {
        "path": "setup/arm26_Setup_ComputedMuscleControl.xml",
        "tool": "CMCTool",
        "must_contain": [
            "arm26.osim",
            "arm26_Reserve_Actuators.xml",
            "arm26_InverseKinematics.mot",
            "arm26_ComputedMuscleControl_Tasks.xml",
        ],
        "tags": {
            "model_file": "arm26.osim",
            "desired_kinematics_file": "arm26_InverseKinematics.mot",
            "task_set_file": "arm26_ComputedMuscleControl_Tasks.xml",
            "initial_time": "0.00000000",
            "final_time": "1.00000000",
        },
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_input_hashes(task_dir: Path) -> list[str]:
    failures: list[str] = []
    manifest = task_dir / "input" / "checksums.sha256"
    if not manifest.exists():
        return ["input/checksums.sha256 missing"]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, name = line.split(None, 1)
        path = task_dir / "input" / name.strip()
        if not path.exists():
            failures.append(f"input/{name.strip()} missing")
        elif _sha256(path) != expected.lower():
            failures.append(f"input/{name.strip()} hash mismatch")
    return failures


def _parse_storage(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            start = i + 1
            break
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines):
        raise ValueError("missing labels")
    labels = re.split(r"\s+", lines[start].strip())
    rows: list[list[float]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = re.split(r"\s+", stripped)
        if len(parts) != len(labels):
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError("no numeric rows")
    return labels, rows


def _compare_storage(actual: Path, reference: Path, abs_tol: float, rms_tol: float) -> dict[str, object]:
    a_labels, a_rows = _parse_storage(actual)
    r_labels, r_rows = _parse_storage(reference)
    if a_labels != r_labels:
        return {"passed": False, "error": "column labels differ"}
    if len(a_rows) < max(3, int(len(r_rows) * 0.75)):
        return {"passed": False, "error": f"too few rows: {len(a_rows)} < 75% of {len(r_rows)}"}
    compared_rows = a_rows
    mode = "exact_rows"
    if len(a_rows) != len(r_rows):
        try:
            compared_rows = _resample_rows_at_reference_times(a_rows, r_rows)
            mode = "interpolated_to_reference_time"
        except ValueError as exc:
            return {"passed": False, "error": str(exc), "actual_rows": len(a_rows), "reference_rows": len(r_rows)}
    max_abs = 0.0
    sum_sq = 0.0
    count = 0
    for ar, rr in zip(compared_rows, r_rows):
        for av, rv in zip(ar, rr):
            if not (math.isfinite(av) and math.isfinite(rv)):
                return {"passed": False, "error": "non-finite numeric value"}
            diff = abs(av - rv)
            max_abs = max(max_abs, diff)
            sum_sq += diff * diff
            count += 1
    rms = math.sqrt(sum_sq / max(count, 1))
    exact_hash = _sha256(actual) == _sha256(reference)
    result: dict[str, object] = {
        "passed": max_abs <= abs_tol and rms <= rms_tol,
        "max_abs": max_abs,
        "rms": rms,
        "rows": len(a_rows),
        "reference_rows": len(r_rows),
        "columns": len(a_labels),
        "comparison_mode": mode,
        "exact_hash": exact_hash,
    }
    if exact_hash:
        result["passed"] = False
        result["error"] = "output is byte-for-byte identical to the judge-only official reference artifact"
    return result


def _resample_rows_at_reference_times(actual_rows: list[list[float]], reference_rows: list[list[float]]) -> list[list[float]]:
    actual = sorted(actual_rows, key=lambda row: row[0])
    times = [row[0] for row in actual]
    if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
        raise ValueError("actual time column is not strictly increasing")
    ref_times = [row[0] for row in reference_rows]
    if ref_times[0] < times[0] - 1e-9 or ref_times[-1] > times[-1] + 1e-9:
        raise ValueError("actual time range does not cover official reference time range")
    out: list[list[float]] = []
    j = 0
    for t in ref_times:
        while j + 1 < len(actual) and actual[j + 1][0] < t:
            j += 1
        if abs(actual[j][0] - t) <= 1e-12:
            out.append(actual[j])
            continue
        if j + 1 >= len(actual):
            raise ValueError("reference time is beyond actual time range")
        left, right = actual[j], actual[j + 1]
        dt = right[0] - left[0]
        if dt <= 0:
            raise ValueError("actual time column is not strictly increasing")
        alpha = (t - left[0]) / dt
        out.append([t] + [lv + alpha * (rv - lv) for lv, rv in zip(left[1:], right[1:])])
    return out


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _xml_texts(root: ET.Element, tag: str) -> list[str]:
    return [_norm(elem.text) for elem in root.iter() if elem.tag.split("}")[-1] == tag]


def _check_xml(path: Path, spec: dict[str, object]) -> list[str]:
    failures: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    root = ET.fromstring(text)
    tool = str(spec["tool"])
    if not any(elem.tag.split("}")[-1] == tool for elem in root.iter()):
        failures.append(f"{path.name} missing {tool}")
    lowered = text.lower()
    for term in spec.get("must_contain", []):
        if str(term).lower() not in lowered:
            failures.append(f"{path.name} missing starter/result reference {term!r}")
    for tag, expected in dict(spec.get("tags", {})).items():
        texts = _xml_texts(root, tag)
        if not any(str(expected).lower() in value.lower() for value in texts):
            failures.append(f"{path.name} tag <{tag}> does not contain {expected!r}")
    forbidden = ["outputreference", "source_cache", "selected_tasks", "opensim-models", "github.com", "reference\\", "reference/"]
    for term in forbidden:
        if term in lowered:
            failures.append(f"{path.name} appears to reference judge-only path {term!r}")
    return failures


def _verify_provenance(output_dir: Path, task_dir: Path) -> list[str]:
    failures: list[str] = []
    path = output_dir / "provenance.json"
    if not path.exists():
        return ["missing output/provenance.json"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse output/provenance.json: {exc}"]
    text = json.dumps(data, sort_keys=True).lower()
    for term in ["outputreference", "source_cache", "selected_tasks", "opensim-models", "github.com", "reference\\", "reference/"]:
        if term in text:
            failures.append(f"provenance.json appears to reference public/judge-only source {term!r}")
    if data.get("task_id") != TASK_ID:
        failures.append("provenance.json task_id mismatch")
    version = str(data.get("opensim_version", ""))
    if REQUIRED_OPENSIM_VERSION_TOKEN not in version:
        failures.append(f"provenance.json opensim_version must identify OpenSim {REQUIRED_OPENSIM_VERSION_TOKEN}")
    manifest_hash = _sha256(task_dir / "input" / "checksums.sha256")
    if str(data.get("input_manifest_sha256", "")).lower() != manifest_hash:
        failures.append("provenance.json input_manifest_sha256 mismatch")
    commands = data.get("setup_commands")
    if not isinstance(commands, list) or len(commands) < 2:
        failures.append("provenance.json setup_commands must list the IK and CMC executions")
    else:
        commands_text = json.dumps(commands, sort_keys=True).lower()
        for spec in XML_CHECKS:
            setup_name = Path(str(spec["path"])).name.lower()
            if setup_name not in commands_text:
                failures.append(f"provenance.json setup_commands missing {setup_name!r}")
    hashes = data.get("output_file_sha256")
    if not isinstance(hashes, dict):
        failures.append("provenance.json output_file_sha256 must be an object")
        return failures
    for rel in HASHED_OUTPUTS:
        out_file = output_dir / rel
        claimed = str(hashes.get(rel, "")).lower()
        if not claimed:
            failures.append(f"provenance.json missing hash for {rel}")
        elif out_file.exists() and claimed != _sha256(out_file):
            failures.append(f"provenance.json hash mismatch for {rel}")
    return failures


def grade(output_dir: Path, task_dir: Path) -> dict[str, object]:
    failures = _verify_input_hashes(task_dir)
    details: dict[str, object] = {"task_id": TASK_ID, "storage": {}}
    failures.extend(_verify_provenance(output_dir, task_dir))

    for spec in XML_CHECKS:
        rel = Path(str(spec["path"]))
        path = output_dir / rel
        if not path.exists():
            failures.append(f"missing output/{rel.as_posix()}")
            continue
        try:
            failures.extend(_check_xml(path, spec))
        except Exception as exc:
            failures.append(f"could not parse output/{rel.as_posix()}: {exc}")

    for actual_rel, ref_rel, abs_tol, rms_tol in STORAGE_CHECKS:
        actual = output_dir / actual_rel
        ref = task_dir / ref_rel
        if not actual.exists():
            failures.append(f"missing output/{actual_rel}")
            continue
        try:
            result = _compare_storage(actual, ref, abs_tol, rms_tol)
            details["storage"][actual_rel] = result
            if result.get("exact_hash"):
                failures.append(
                    f"output/{actual_rel} is byte-identical to the official public reference; "
                    "submit artifacts regenerated by a local OpenSim run, not copied OutputReference files"
                )
            elif not result.get("passed"):
                failures.append(f"output/{actual_rel} differs from official reference: {result}")
        except Exception as exc:
            failures.append(f"could not compare output/{actual_rel}: {exc}")

    run_log = output_dir / "run_log.txt"
    if not run_log.exists() or run_log.stat().st_size < 80:
        failures.append("missing or too-small output/run_log.txt")
    else:
        log = run_log.read_text(encoding="utf-8", errors="ignore").lower()
        for term in ["opensim", "inverse", "computed", "muscle", "control", "arm26"]:
            if term not in log:
                failures.append(f"run_log.txt missing {term!r}")
        if REQUIRED_OPENSIM_VERSION_TOKEN not in log:
            failures.append(f"run_log.txt must identify OpenSim {REQUIRED_OPENSIM_VERSION_TOKEN}")
        for spec in XML_CHECKS:
            setup_name = Path(str(spec["path"])).name.lower()
            if setup_name not in log:
                failures.append(f"run_log.txt missing setup file {setup_name!r}")
        for term in ["outputreference", "source_cache", "selected_tasks", "opensim-models", "github.com", "reference\\", "reference/"]:
            if term in log:
                failures.append(f"run_log.txt appears to reference public/judge-only source {term!r}")

    copied = [p.name for p in output_dir.rglob("*") if p.is_file() and p.name in {"source_evidence.md", "curator_notes.md", "private_gold_contract.json"}]
    if copied:
        failures.append(f"submission includes judge-only curator files: {copied}")

    return {"passed": not failures, "score": 1.0 if not failures else 0.0, "failures": failures, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    task_dir = Path(__file__).resolve().parent
    result = grade(Path(args.output_dir), task_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1



@dataclass
class TaskConfig(GeneralTaskConfig):
    TASK_TAG: str = TASK_NAME
    TASK_CATEGORY: str = TASK_CATEGORY

    @property
    def task_description(self) -> str:
        return (TASK_DIR / "solver_prompt.txt").read_text(encoding="utf-8")

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update({"task_tag": self.TASK_TAG, "task_category": self.TASK_CATEGORY})
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_load_solver_prompt(),
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    out_dir = task_cfg.metadata["remote_output_dir"]
    try:
        await session.remove_file(out_dir)
    except Exception:
        pass
    await session.makedirs(out_dir)


def _required_output_files() -> list[str]:
    contract_path = TASK_DIR / "input" / "output_contract.json"
    if not contract_path.exists():
        return []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    candidates: list[Any] = []
    for key in ("required_files", "required_outputs", "outputs"):
        if key in contract:
            candidates.append(contract[key])
    if isinstance(contract.get("files"), dict):
        candidates.extend(contract["files"].keys())
    files: list[str] = []
    def add_value(value: Any) -> None:
        if isinstance(value, str):
            files.append(value)
        elif isinstance(value, dict):
            if isinstance(value.get("path"), str):
                files.append(value["path"])
            elif isinstance(value.get("filename"), str):
                files.append(value["filename"])
        elif isinstance(value, list):
            for item in value:
                add_value(item)
    for candidate in candidates:
        if isinstance(candidate, dict):
            for value in candidate.values():
                add_value(value)
        else:
            add_value(candidate)
    cleaned = []
    for name in files:
        normalized = name.replace("\\", "/")
        if normalized.startswith("output/"):
            normalized = normalized[len("output/"):]
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


async def _fetch_outputs(session: cb.DesktopSession, remote_output_dir: str, local_output_dir: Path) -> None:
    for rel_name in _required_output_files():
        local_path = local_output_dir / rel_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            local_path.write_bytes(await session.read_bytes(os.path.join(remote_output_dir, rel_name)))
        except Exception as exc:
            logger.warning("could not fetch output %s: %s", rel_name, exc)


def _run_private_grade(tmp_task: Path) -> dict[str, Any]:
    params = list(inspect.signature(grade).parameters)
    first = params[0] if params else ""
    if len(params) >= 2 and first == "output_dir":
        result = grade(tmp_task / "output", tmp_task)
    elif first == "output_dir":
        result = grade(tmp_task / "output")
    else:
        result = grade(tmp_task)
    if not isinstance(result, dict):
        return {"passed": False, "score": 0.0, "errors": ["grade() returned a non-dict result"]}
    result.setdefault("score", 1.0 if result.get("passed") else 0.0)
    return result


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    remote_out = task_cfg.metadata["remote_output_dir"]
    with tempfile.TemporaryDirectory(prefix=f"{TASK_NAME}_eval_") as tmp:
        tmp_task = Path(tmp) / TASK_NAME
        tmp_task.mkdir(parents=True)
        _copy_tree(TASK_DIR / "input", tmp_task / "input")
        _copy_tree(TASK_DIR / "reference", tmp_task / "reference")
        (tmp_task / "output").mkdir()
        await _fetch_outputs(session, remote_out, tmp_task / "output")
        try:
            report = _run_private_grade(tmp_task)
        except Exception as exc:
            logger.exception("private grading failed: %s", exc)
            report = {"passed": False, "score": 0.0, "errors": [str(exc)]}
        try:
            await session.write_file(os.path.join(remote_out, "autograde_report.json"), json.dumps(report, indent=2, ensure_ascii=False))
        except Exception:
            pass
        return [float(report.get("score", 0.0))]
