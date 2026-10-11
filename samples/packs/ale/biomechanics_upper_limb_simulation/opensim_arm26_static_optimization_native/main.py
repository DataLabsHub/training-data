"""ALE task implementation for biomechanics_upper_limb_simulation/opensim_arm26_static_optimization_native.

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
TASK_NAME = "opensim_arm26_static_optimization_native"
TASK_CATEGORY = "biomechanics_upper_limb_simulation"


# ---- inline private grader: main.py ----

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


FORBIDDEN_PUBLIC_PATH_TERMS = [
    "outputreference",
    "source_cache",
    "reference\\",
    "reference/",
    "official_native_tasks",
    "selected_tasks.zip",
    "selected_tasks",
    "github.com",
    "opensim-models",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_spec(task_dir: Path) -> dict[str, Any]:
    return json.loads((task_dir / "grader_assets" / "grader_spec.json").read_text(encoding="utf-8"))


def _verify_input_hashes(task_dir: Path) -> list[str]:
    failures: list[str] = []
    manifest = task_dir / "input" / "checksums.sha256"
    if not manifest.exists():
        return ["input/checksums.sha256 missing"]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            expected, rel = line.split(None, 1)
        except ValueError:
            failures.append(f"malformed checksum line: {line!r}")
            continue
        rel_parts = rel.strip().replace("\\", "/").split("/")
        path = task_dir / "input" / Path(*rel_parts)
        if not path.exists():
            failures.append(f"input/{rel.strip()} missing")
        elif _sha256(path) != expected.lower():
            failures.append(f"input/{rel.strip()} hash mismatch")
    return failures


def _label_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("#")):
                j += 1
            return j
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" in stripped:
            continue
        parts = re.split(r"\s+", stripped)
        if parts and parts[0].lower() in {"time", "t"}:
            return i
    raise ValueError("missing column labels")


def _parse_storage(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = _label_index(lines)
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
            row = [float(x) for x in parts]
        except ValueError:
            continue
        rows.append(row)
    if not rows:
        raise ValueError("no numeric rows")
    return labels, rows


def _compare_storage(actual: Path, reference: Path, abs_tol: float, rms_tol: float) -> dict[str, Any]:
    a_labels, a_rows = _parse_storage(actual)
    r_labels, r_rows = _parse_storage(reference)
    if a_labels != r_labels:
        return {"passed": False, "error": "column labels differ", "actual_columns": len(a_labels), "reference_columns": len(r_labels)}
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
    result: dict[str, Any] = {
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

def _check_storage_shape(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    labels, rows = _parse_storage(path)
    required = [str(x) for x in spec.get("required_labels", [])]
    missing = [label for label in required if label not in labels]
    return {
        "passed": len(rows) >= int(spec.get("min_rows", 1)) and len(labels) >= int(spec.get("min_columns", 2)) and not missing,
        "rows": len(rows),
        "columns": len(labels),
        "missing_labels": missing,
    }


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _local_name(tag: str) -> str:
    return tag.split("}")[-1]


def _xml_texts(root: ET.Element, tag: str) -> list[str]:
    return [_norm(elem.text) for elem in root.iter() if _local_name(elem.tag) == tag]


def _expected_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _check_setup_xml(path: Path, spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    try:
        root = ET.fromstring(text)
    except Exception as exc:
        return [f"{path.name} is not parseable XML: {exc}"]

    tools = _expected_values(spec.get("tool", []))
    if tools and not any(_local_name(elem.tag) in tools for elem in root.iter()):
        failures.append(f"{path.name} missing expected tool {tools}")

    for term in spec.get("must_contain", []):
        if str(term).lower() not in lowered:
            failures.append(f"{path.name} missing starter/result reference {term!r}")
    for term in spec.get("must_not_contain", []):
        if str(term).lower() in lowered:
            failures.append(f"{path.name} contains forbidden workflow term {term!r}")
    for term in FORBIDDEN_PUBLIC_PATH_TERMS:
        if term in lowered:
            failures.append(f"{path.name} appears to reference judge-only/public path {term!r}")

    for tag, expected in dict(spec.get("tags", {})).items():
        texts = _xml_texts(root, tag)
        expected_values = _expected_values(expected)
        if not any(exp.lower() in value.lower() for exp in expected_values for value in texts):
            failures.append(f"{path.name} tag <{tag}> does not contain one of {expected_values!r}")
    return failures


def _check_text_artifact(actual: Path, reference: Path, spec: dict[str, Any]) -> dict[str, Any]:
    actual_text = actual.read_text(encoding="utf-8", errors="ignore")
    reference_text = reference.read_text(encoding="utf-8", errors="ignore")
    min_ratio = float(spec.get("min_size_ratio", 0.6))
    result: dict[str, Any] = {
        "passed": True,
        "actual_size": actual.stat().st_size,
        "reference_size": reference.stat().st_size,
        "exact_hash": _sha256(actual) == _sha256(reference),
    }
    if result["exact_hash"]:
        result["passed"] = False
        result["error"] = "output is byte-for-byte identical to the judge-only official reference artifact"
        return result
    if actual.stat().st_size < max(40, int(reference.stat().st_size * min_ratio)):
        result["passed"] = False
        result["error"] = "too small compared with official reference"
        return result
    missing = [str(term) for term in spec.get("must_contain", []) if str(term) not in actual_text]
    if missing:
        result["passed"] = False
        result["missing_terms"] = missing
        return result
    if actual.suffix.lower() in {".xml", ".osim"}:
        try:
            a_root = ET.fromstring(actual_text)
            r_root = ET.fromstring(reference_text)
            a_count = sum(1 for _ in a_root.iter())
            r_count = sum(1 for _ in r_root.iter())
            result.update({"actual_root": _local_name(a_root.tag), "reference_root": _local_name(r_root.tag), "actual_elements": a_count, "reference_elements": r_count})
            if _local_name(a_root.tag) != _local_name(r_root.tag) or a_count < int(r_count * min_ratio):
                result["passed"] = False
                result["error"] = "XML structure is not close to official reference"
        except Exception as exc:
            result["passed"] = False
            result["error"] = f"XML parse failed: {exc}"
    return result


def _check_required_file(path: Path, spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not path.exists():
        return [f"missing output/{spec['path']}"]
    if path.stat().st_size < int(spec.get("min_size", 1)):
        failures.append(f"output/{spec['path']} is too small")
    if spec.get("must_contain"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in spec["must_contain"]:
            if str(term) not in text:
                failures.append(f"output/{spec['path']} missing {term!r}")
    return failures


def _expected_hashed_outputs(spec: dict[str, Any]) -> list[str]:
    rels: list[str] = []
    for item in spec.get("setup_checks", []):
        rels.append(str(item["path"]))
    for item in spec.get("required_files", []):
        rels.append(str(item["path"]))
    for item in spec.get("intermediate_storage_checks", []):
        rels.append(str(item["path"]))
    for item in spec.get("storage_checks", []):
        rels.append(str(item["actual"]))
    for item in spec.get("text_checks", []):
        rels.append(str(item["actual"]))
    rels.append("run_log.txt")
    seen: set[str] = set()
    ordered: list[str] = []
    for rel in rels:
        if rel not in seen:
            seen.add(rel)
            ordered.append(rel)
    return ordered


def _verify_provenance(output_dir: Path, task_dir: Path, spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    path = output_dir / "provenance.json"
    if not path.exists():
        return ["missing output/provenance.json"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse output/provenance.json: {exc}"]
    text = json.dumps(data, sort_keys=True).lower()
    for term in FORBIDDEN_PUBLIC_PATH_TERMS:
        if term in text:
            failures.append(f"provenance.json appears to reference judge-only/public path {term!r}")
    if data.get("task_id") != spec.get("task_id"):
        failures.append("provenance.json task_id mismatch")
    version = str(data.get("opensim_version", ""))
    if "4.5" not in version:
        failures.append("provenance.json opensim_version must identify OpenSim 4.5")
    manifest = task_dir / "input" / "checksums.sha256"
    if manifest.exists() and str(data.get("input_manifest_sha256", "")).lower() != _sha256(manifest):
        failures.append("provenance.json input_manifest_sha256 mismatch")
    commands = data.get("setup_commands")
    if not isinstance(commands, list) or len(commands) < max(1, len(spec.get("setup_checks", []))):
        failures.append("provenance.json setup_commands does not list the required setup executions")
    else:
        commands_text = json.dumps(commands, sort_keys=True).lower()
        for item in spec.get("setup_checks", []):
            setup_name = Path(str(item["path"])).name.lower()
            if setup_name not in commands_text:
                failures.append(f"provenance.json setup_commands missing {setup_name!r}")
    hashes = data.get("output_file_sha256")
    if not isinstance(hashes, dict):
        failures.append("provenance.json output_file_sha256 must be an object")
        return failures
    for rel in _expected_hashed_outputs(spec):
        out_file = output_dir / rel
        claimed = str(hashes.get(rel, "")).lower()
        if not claimed:
            failures.append(f"provenance.json missing hash for {rel}")
        elif out_file.exists() and claimed != _sha256(out_file):
            failures.append(f"provenance.json hash mismatch for {rel}")
    return failures

def _check_run_log(output_dir: Path, spec: dict[str, Any]) -> list[str]:
    log_spec = spec.get("run_log", {})
    run_log = output_dir / "run_log.txt"
    if not run_log.exists() or run_log.stat().st_size < int(log_spec.get("min_size", 80)):
        return ["missing or too-small output/run_log.txt"]
    text = run_log.read_text(encoding="utf-8", errors="ignore").lower()
    failures = [f"run_log.txt missing {term!r}" for term in log_spec.get("terms", []) if str(term).lower() not in text]
    if "4.5" not in text:
        failures.append("run_log.txt must identify OpenSim 4.5")
    for item in spec.get("setup_checks", []):
        setup_name = Path(str(item["path"])).name.lower()
        if setup_name not in text:
            failures.append(f"run_log.txt missing setup file {setup_name!r}")
    for term in FORBIDDEN_PUBLIC_PATH_TERMS:
        if term in text:
            failures.append(f"run_log.txt appears to reference judge-only/public path {term!r}")
    return failures


def grade(output_dir: Path, task_dir: Path) -> dict[str, Any]:
    spec = _load_spec(task_dir)
    failures = _verify_input_hashes(task_dir)
    details: dict[str, Any] = {"task_id": spec["task_id"], "storage": {}, "text": {}, "intermediates": {}}
    failures.extend(_verify_provenance(output_dir, task_dir, spec))

    for xml_spec in spec.get("setup_checks", []):
        rel = str(xml_spec["path"])
        path = output_dir / rel
        if not path.exists():
            failures.append(f"missing output/{rel}")
            continue
        failures.extend(_check_setup_xml(path, xml_spec))

    for req in spec.get("required_files", []):
        failures.extend(_check_required_file(output_dir / str(req["path"]), req))

    for item in spec.get("intermediate_storage_checks", []):
        rel = str(item["path"])
        path = output_dir / rel
        if not path.exists():
            failures.append(f"missing output/{rel}")
            continue
        try:
            result = _check_storage_shape(path, item)
            details["intermediates"][rel] = result
            if not result["passed"]:
                failures.append(f"output/{rel} is not a plausible generated storage file: {result}")
        except Exception as exc:
            failures.append(f"could not parse output/{rel}: {exc}")

    for item in spec.get("storage_checks", []):
        actual_rel = str(item["actual"])
        ref_rel = str(item["reference"])
        actual = output_dir / actual_rel
        reference = task_dir / "reference" / ref_rel
        if not actual.exists():
            failures.append(f"missing output/{actual_rel}")
            continue
        try:
            result = _compare_storage(actual, reference, float(item.get("abs_tol", 1e-5)), float(item.get("rms_tol", 1e-6)))
            details["storage"][actual_rel] = result
            if result.get("exact_hash"):
                failures.append(
                    f"output/{actual_rel} is byte-identical to the official public reference; "
                    "submit artifacts regenerated by a local OpenSim run, not copied OutputReference files"
                )
            if not result.get("passed"):
                failures.append(f"output/{actual_rel} differs from official reference: {result}")
        except Exception as exc:
            failures.append(f"could not compare output/{actual_rel}: {exc}")

    for item in spec.get("text_checks", []):
        actual_rel = str(item["actual"])
        ref_rel = str(item["reference"])
        actual = output_dir / actual_rel
        reference = task_dir / "reference" / ref_rel
        if not actual.exists():
            failures.append(f"missing output/{actual_rel}")
            continue
        try:
            result = _check_text_artifact(actual, reference, item)
            details["text"][actual_rel] = result
            if result.get("exact_hash"):
                failures.append(
                    f"output/{actual_rel} is byte-identical to the official public reference; "
                    "submit artifacts regenerated by a local OpenSim run, not copied OutputReference files"
                )
            if not result.get("passed"):
                failures.append(f"output/{actual_rel} failed native text/XML check: {result}")
        except Exception as exc:
            failures.append(f"could not check output/{actual_rel}: {exc}")

    failures.extend(_check_run_log(output_dir, spec))

    copied = [p.name for p in output_dir.rglob("*") if p.is_file() and p.name in {"source_evidence.md", "curator_notes.md", "private_gold_contract.json"}]
    if copied:
        failures.append(f"submission includes judge/task-only files: {copied}")

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
