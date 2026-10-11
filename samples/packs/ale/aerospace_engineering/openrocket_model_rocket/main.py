"""
OpenRocket Model Rocket Task — Judger

Evaluates an AI agent's ability to recreate the official OpenRocket example
"A simple model rocket" in OpenRocket 24.12, run the C6-5 simulation, and
export the required artifacts.
"""

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from utils.evaluation import llm_vision_judge

logger = logging.getLogger(__name__)

from pathlib import Path as _AlePromptPath


def _load_solver_prompt() -> str:
    return (_AlePromptPath(__file__).resolve().parent / "solver_prompt.txt").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig(GeneralTaskConfig):
    TASK_TAG: str = "openrocket_model_rocket"
    TASK_CATEGORY: str = "aerospace_engineering"

    @property
    def task_description(self) -> str:
        return f"""
Goal: Recreate the official OpenRocket example "A simple model rocket" and run
the default C6-5 simulation using OpenRocket 24.12.

Task directory: {self.task_dir}

Required workflow:
1. Open OpenRocket 24.12 on Windows 11 x64 and start a new design.
2. Recreate "A simple model rocket" using input/rocket_spec.csv (ogive nose
   cone, 300 mm body tube, parachute, shock cord, wadding, launch lug, 3-fin
   trapezoidal fin set, centering rings, 75 mm inner tube, engine block).
3. Set motor configuration to Estes C6-5 from the built-in motor database.
4. Set launch conditions: 1.0 m launch rod, launch_into_wind=true, wind 2.0
   m/s, turbulence 0.1, ISA atmosphere.
5. Run the simulation corresponding to "Simulation 4" / C6-5.
6. Export time-series CSV (Time, Altitude, Vertical velocity, Vertical
   acceleration) to output/simulation_data.csv via OpenRocket Plot/Export.
7. Save the .ork file AFTER running the simulation so flightdata is embedded.
8. Export all required deliverables under output/.

Required deliverables (all under {self.remote_output_dir}):
- rocket_design.ork
- results.json  (schema: simulation_name, motor_designation, apogee_m,
                  max_velocity_m_s, max_acceleration_m_s2, time_to_apogee_s,
                  flight_time_s, launchrod_velocity_m_s, deployment_velocity_m_s)
- rocket_design_view.png  (screenshot of the OpenRocket main window)
- simulation_plot.png     (screenshot of the simulation plot view)
- simulation_data.csv
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_tag": self.TASK_TAG,
                "task_category": self.TASK_CATEGORY,
            }
        )
        return metadata


config = TaskConfig()


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_load_solver_prompt(),
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    out_dir = task_cfg.metadata["remote_output_dir"]
    logger.info("[openrocket] start() — clearing %s", out_dir)

    try:
        await session.remove_file(out_dir)
        logger.info("[openrocket] removed existing output dir")
    except Exception:
        logger.info("[openrocket] output dir not present, skipping removal")

    try:
        await session.makedirs(out_dir)
        logger.info("[openrocket] created output dir: %s", out_dir)
    except Exception as e:
        logger.warning("[openrocket] makedirs failed: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_ork_xml(ork_bytes: bytes) -> Optional[ET.Element]:
    """Unzip .ork and parse inner rocket.ork XML; return root element or None."""
    try:
        with zipfile.ZipFile(io.BytesIO(ork_bytes)) as zf:
            names = zf.namelist()
            xml_name = next((n for n in names if n.endswith(".ork")), None)
            if xml_name is None:
                logger.warning("[openrocket] no .ork XML member found inside ZIP")
                return None
            xml_bytes = zf.read(xml_name)
        return ET.fromstring(xml_bytes.decode("utf-8"))
    except Exception as e:
        logger.warning("[openrocket] failed to parse .ork XML: %s", e)
        return None


def _collect_ids(root: ET.Element) -> set:
    """Collect all component/simulation id= attributes and <id>…</id> text values."""
    ids = set()
    for el in root.iter():
        val = el.get("id")
        if val:
            ids.add(val)
        if el.tag == "id" and el.text:
            ids.add(el.text.strip())
    return ids


def _find_sim4(root: ET.Element) -> Optional[ET.Element]:
    """Return the <simulation> element whose <name> is 'Simulation 4'."""
    for sim in root.iter("simulation"):
        name_el = sim.find("name")
        if name_el is not None and name_el.text and name_el.text.strip() == "Simulation 4":
            return sim
    return None


def _float_text(el: Optional[ET.Element], default: Optional[float] = None) -> Optional[float]:
    if el is None or el.text is None:
        return default
    try:
        return float(el.text.strip().split()[0])
    except Exception:
        return default


def _component(root: ET.Element, tag: str, name: Optional[str] = None) -> Optional[ET.Element]:
    for el in root.iter(tag):
        if name is None or (el.findtext("name") or "").strip().lower() == name.lower():
            return el
    return None


def _near(value: Optional[float], target: float, tol: float) -> bool:
    return value is not None and abs(value - target) <= tol


def _check_ork_native_structure(root: ET.Element) -> tuple[bool, dict]:
    """
    Validate OpenRocket-native XML structure for the official simple model
    rocket. This checks component semantics that cannot be inferred from
    results.json alone.
    """
    detail: dict[str, Any] = {"failures": []}
    failures: list[str] = detail["failures"]

    if root.tag != "openrocket":
        failures.append(f"root tag {root.tag!r} is not openrocket")
    creator = root.attrib.get("creator", "")
    detail["creator"] = creator
    if "OpenRocket" not in creator:
        failures.append("creator attribute does not identify OpenRocket")

    rocket = root.find(".//rocket")
    rocket_name = (rocket.findtext("name") if rocket is not None else "") or ""
    detail["rocket_name"] = rocket_name.strip()
    if rocket is None or "simple model rocket" not in rocket_name.lower():
        failures.append("rocket name is not the official simple model rocket")

    expected_counts = {
        "nosecone": 1,
        "bodytube": 1,
        "parachute": 1,
        "shockcord": 1,
        "masscomponent": 1,
        "launchlug": 1,
        "trapezoidfinset": 1,
        "innertube": 1,
        "engineblock": 1,
    }
    counts = {tag: len(list(root.iter(tag))) for tag in expected_counts}
    counts["centeringring"] = len(list(root.iter("centeringring")))
    detail["component_counts"] = counts
    for tag, expected in expected_counts.items():
        if counts[tag] != expected:
            failures.append(f"{tag} count {counts[tag]} != {expected}")
    if counts["centeringring"] < 2:
        failures.append("expected at least two centering rings")

    nose = _component(root, "nosecone", "Nose cone")
    if nose is None:
        failures.append("missing Nose cone")
    else:
        if (nose.findtext("shape") or "").strip().lower() != "ogive":
            failures.append("nose cone shape is not ogive")
        if not _near(_float_text(nose.find("length")), 0.1, 1e-4):
            failures.append("nose cone length is not 0.100 m")
        if not _near(_float_text(nose.find("aftradius")), 0.0125, 1e-5):
            failures.append("nose cone aft radius is not 0.0125 m")

    body = _component(root, "bodytube", "Body tube")
    if body is None:
        failures.append("missing Body tube")
    else:
        if not _near(_float_text(body.find("length")), 0.3, 1e-4):
            failures.append("body tube length is not 0.300 m")
        radius_text = (body.findtext("radius") or "").strip()
        if "0.0125" not in radius_text:
            failures.append("body tube radius is not 0.0125 m / auto 0.0125")

    parachute = _component(root, "parachute", "Parachute")
    if parachute is None or not _near(_float_text(parachute.find("diameter") if parachute is not None else None), 0.3, 1e-4):
        failures.append("missing 0.300 m parachute")

    fins = _component(root, "trapezoidfinset", "Trapezoidal fin set")
    if fins is None:
        failures.append("missing trapezoidal fin set")
    else:
        if (fins.findtext("fincount") or "").strip() != "3":
            failures.append("fin count is not 3")
        if not _near(_float_text(fins.find("rootchord")), 0.0508, 1e-4):
            failures.append("fin root chord is not 0.0508 m")
        if not _near(_float_text(fins.find("height")), 0.03, 1e-4):
            failures.append("fin height is not 0.030 m")

    inner = _component(root, "innertube", "Inner Tube")
    if inner is None:
        failures.append("missing Inner Tube motor mount")
    else:
        if not _near(_float_text(inner.find("length")), 0.075, 1e-4):
            failures.append("inner tube length is not 0.075 m")
        if inner.find("motormount") is None:
            failures.append("inner tube is not marked as motor mount")

    if _component(root, "engineblock", "Engine block") is None:
        failures.append("missing engine block")

    motor_labels = []
    for m in root.iter("motor"):
        designation = (m.findtext("designation") or "").strip()
        delay = _float_text(m.find("delay"))
        label = f"{designation}-{int(delay)}" if designation and delay is not None and delay.is_integer() else designation
        motor_labels.append(label)
    detail["motor_designations"] = motor_labels
    if "C6-5" not in motor_labels:
        failures.append("no Estes C6-5 motor designation found")

    sim4 = _find_sim4(root)
    if sim4 is None:
        failures.append("Simulation 4 not found")
    else:
        cond = sim4.find("conditions")
        if cond is None:
            failures.append("Simulation 4 missing conditions")
        else:
            for tag, target, tol in [
                ("launchrodlength", 1.0, 1e-6),
                ("windaverage", 2.0, 1e-6),
                ("windturbulence", 0.1, 1e-6),
            ]:
                if not _near(_float_text(cond.find(tag)), target, tol):
                    failures.append(f"Simulation 4 {tag} is not {target}")
            if (cond.findtext("launchintowind") or "").strip().lower() != "true":
                failures.append("Simulation 4 launchintowind is not true")
            atmosphere = cond.find("atmosphere")
            if atmosphere is None or atmosphere.attrib.get("model") != "isa":
                failures.append("Simulation 4 atmosphere is not ISA")

    detail["passed"] = not failures
    return not failures, detail


def _extract_ork_reference_facts(root: ET.Element) -> dict:
    """Extract stable OpenRocket facts for GT-vs-submission comparison."""
    def comp_float(tag: str, name: str, child: str) -> Optional[float]:
        el = _component(root, tag, name)
        return _float_text(el.find(child) if el is not None else None)

    def comp_text(tag: str, name: str, child: str) -> str:
        el = _component(root, tag, name)
        return ((el.findtext(child) if el is not None else "") or "").strip()

    sim4 = _find_sim4(root)
    cond = sim4.find("conditions") if sim4 is not None else None
    branches = sim4.findall(".//flightdata/databranch") if sim4 is not None else []

    return {
        "component_counts": {
            tag: len(list(root.iter(tag)))
            for tag in (
                "nosecone", "bodytube", "parachute", "shockcord", "masscomponent",
                "launchlug", "trapezoidfinset", "centeringring", "innertube",
                "engineblock",
            )
        },
        "nose_shape": comp_text("nosecone", "Nose cone", "shape"),
        "nose_length": comp_float("nosecone", "Nose cone", "length"),
        "nose_aftradius": comp_float("nosecone", "Nose cone", "aftradius"),
        "body_length": comp_float("bodytube", "Body tube", "length"),
        "body_radius_text": comp_text("bodytube", "Body tube", "radius"),
        "parachute_diameter": comp_float("parachute", "Parachute", "diameter"),
        "fin_count": comp_text("trapezoidfinset", "Trapezoidal fin set", "fincount"),
        "fin_rootchord": comp_float("trapezoidfinset", "Trapezoidal fin set", "rootchord"),
        "fin_height": comp_float("trapezoidfinset", "Trapezoidal fin set", "height"),
        "inner_length": comp_float("innertube", "Inner Tube", "length"),
        "inner_has_motormount": _component(root, "innertube", "Inner Tube") is not None
                              and _component(root, "innertube", "Inner Tube").find("motormount") is not None,
        "motor_designations": sorted({
            (
                f"{(m.findtext('designation') or '').strip()}-{int(_float_text(m.find('delay')))}"
                if (m.findtext("designation") or "").strip()
                and _float_text(m.find("delay")) is not None
                and _float_text(m.find("delay")).is_integer()
                else (m.findtext("designation") or "").strip()
            )
            for m in root.iter("motor")
        }),
        "sim4_conditions": {
            "launchrodlength": _float_text(cond.find("launchrodlength") if cond is not None else None),
            "launchintowind": ((cond.findtext("launchintowind") if cond is not None else "") or "").strip().lower(),
            "windaverage": _float_text(cond.find("windaverage") if cond is not None else None),
            "windturbulence": _float_text(cond.find("windturbulence") if cond is not None else None),
            "atmosphere_model": (cond.find("atmosphere").attrib.get("model") if cond is not None and cond.find("atmosphere") is not None else ""),
        },
        "sim4_datapoints": len(branches[0].findall("datapoint")) if branches else 0,
    }


def _compare_ork_facts_to_gt(sub_root: ET.Element, ref_root: ET.Element) -> tuple[bool, dict]:
    """
    Compare stable project facts against the GT .ork without requiring
    byte-identical IDs, UUIDs, timestamps, or motor digests.
    """
    sub = _extract_ork_reference_facts(sub_root)
    ref = _extract_ork_reference_facts(ref_root)
    failures: list[str] = []

    if sub["component_counts"] != ref["component_counts"]:
        failures.append(f"component counts differ: {sub['component_counts']} != {ref['component_counts']}")

    for key in ("nose_shape", "body_radius_text", "fin_count", "inner_has_motormount"):
        if sub.get(key) != ref.get(key):
            failures.append(f"{key} differs: {sub.get(key)!r} != {ref.get(key)!r}")

    for key in (
        "nose_length", "nose_aftradius", "body_length", "parachute_diameter",
        "fin_rootchord", "fin_height", "inner_length",
    ):
        sv = sub.get(key)
        rv = ref.get(key)
        if sv is None or rv is None or abs(sv - rv) > 1e-4:
            failures.append(f"{key} differs: {sv!r} != {rv!r}")

    if "C6-5" not in sub.get("motor_designations", []):
        failures.append("C6-5 motor designation missing")

    for key, rv in ref["sim4_conditions"].items():
        sv = sub["sim4_conditions"].get(key)
        if isinstance(rv, float) or isinstance(sv, float):
            if sv is None or rv is None or abs(float(sv) - float(rv)) > 1e-6:
                failures.append(f"Simulation 4 {key} differs: {sv!r} != {rv!r}")
        elif sv != rv:
            failures.append(f"Simulation 4 {key} differs: {sv!r} != {rv!r}")

    if sub["sim4_datapoints"] < 300:
        failures.append(f"Simulation 4 has too few datapoints: {sub['sim4_datapoints']}")

    return not failures, {
        "passed": not failures,
        "failures": failures,
        "submitted_facts": sub,
        "reference_facts": ref,
    }


def _parse_csv_columns(content: str):
    """
    Parse simulation_data.csv; return (rows, t_col, alt_col, vv_col, va_col)
    where each col is a list of floats (rows with parse errors skipped).
    Returns None on fatal parse failure.
    """
    try:
        reader = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames or []
        # Find relevant columns (case-insensitive)
        t_hdr = next((h for h in headers if "time" in h.lower()), None)
        alt_hdr = next((h for h in headers if "altitude" in h.lower()), None)
        vv_hdr = next((h for h in headers if "vertical velocity" in h.lower()), None)
        va_hdr = next((h for h in headers if "vertical acceleration" in h.lower()), None)

        if not all([t_hdr, alt_hdr, vv_hdr, va_hdr]):
            logger.warning(
                "[openrocket] CSV missing required columns; found: %s", headers
            )
            return None

        t_vals, alt_vals, vv_vals, va_vals = [], [], [], []
        for row in reader:
            try:
                tv = float(row[t_hdr])
                av = float(row[alt_hdr])
                vv = float(row[vv_hdr])
                va = float(row[va_hdr])
                t_vals.append(tv)
                alt_vals.append(av)
                vv_vals.append(vv)
                va_vals.append(va)
            except (ValueError, TypeError, KeyError):
                pass  # skip rows with unparseable values

        return t_vals, alt_vals, vv_vals, va_vals
    except Exception as e:
        logger.warning("[openrocket] CSV parse failure: %s", e)
        return None


def _mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def _second_diff_mean(arr):
    """Mean of discrete second differences for interior indices."""
    if len(arr) < 3:
        return 0.0
    diffs = [arr[i + 1] - 2 * arr[i] + arr[i - 1] for i in range(1, len(arr) - 1)]
    return _mean(diffs)


def _pct_err(submitted: float, reference: float) -> float:
    if reference == 0:
        return float("inf")
    return abs(submitted - reference) / abs(reference)


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

W_FILES      = 0.10   # required file existence
W_ODS_STRUCT = 0.20   # .ork structural validity + databranch
W_CSV_PHYS   = 0.20   # simulation_data.csv physics consistency
W_NUMERIC    = 0.50   # numeric agreement with gold


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> List[float]:
    out_dir = task_cfg.metadata["remote_output_dir"]
    ref_dir = task_cfg.metadata["reference_dir"]

    logger.info("[openrocket] evaluate() begin — out_dir=%s", out_dir)

    # Paths on remote machine
    ork_path      = rf"{out_dir}\rocket_design.ork"
    results_path  = rf"{out_dir}\results.json"
    csv_path      = rf"{out_dir}\simulation_data.csv"
    sim_png_path  = rf"{out_dir}\simulation_plot.png"
    view_png_path = rf"{out_dir}\rocket_design_view.png"
    ref_ork_path  = rf"{ref_dir}\rocket_design.ork"
    gold_path     = rf"{task_cfg.metadata["task_dir"]}\grader_assets\private_gold_contract.json"

    report: Dict[str, Any] = {
        "task_tag": "openrocket_model_rocket",
        "gates": {},
        "checks": {},
        "score_breakdown": {},
    }

    # ------------------------------------------------------------------
    # 0. Load gold values
    # ------------------------------------------------------------------
    try:
        gold_raw = await session.read_file(gold_path)
        gold_data = json.loads(gold_raw if isinstance(gold_raw, str) else gold_raw.decode("utf-8"))
        gold = gold_data["gold_values"]
        logger.info("[openrocket] gold loaded OK")
    except Exception as e:
        logger.error("[openrocket] cannot load gold contract: %s", e)
        return [0.0]

    # ------------------------------------------------------------------
    # GUI GATE — checks 9 and 10 (simulation_plot.png / rocket_design_view.png)
    # ------------------------------------------------------------------
    async def _vlm_gui_check(png_path: str, check_name: str, prompt: str) -> bool:
        try:
            png_bytes = await session.read_bytes(png_path)
        except Exception as e:
            logger.warning("[openrocket] %s: cannot read PNG: %s", check_name, e)
            report["gates"][check_name] = f"FAIL — cannot read file: {e}"
            return False

        if not png_bytes or len(png_bytes) < 8192:
            logger.warning("[openrocket] %s: PNG too small (%d bytes)", check_name, len(png_bytes))
            report["gates"][check_name] = "FAIL — file too small or empty"
            return False

        try:
            score = await llm_vision_judge(
                prompt=prompt,
                image_bytes=png_bytes,
                return_binary_score=True,
                max_tokens=16,
            )
            passed = bool(score and score >= 0.5)
            report["gates"][check_name] = "PASS" if passed else "FAIL — VLM did not detect OpenRocket UI tokens"
            logger.info("[openrocket] %s VLM score=%.1f passed=%s", check_name, score, passed)
            return passed
        except Exception as e:
            logger.error("[openrocket] %s VLM error: %s", check_name, e)
            report["gates"][check_name] = f"FAIL — VLM error: {e}"
            return False

    sim_plot_ok = await _vlm_gui_check(
        sim_png_path,
        "check_9_simulation_plot",
        (
            "Is this a screenshot of the OpenRocket 24.12 application showing a "
            "simulation plot view? Answer YES only if you can see OpenRocket window "
            "chrome (menu bar, toolbar, or panel labels such as 'Flight simulations', "
            "'Plot data', 'Plot flight', 'Altitude', 'Vertical velocity', "
            "'Vertical acceleration', 'Simulation 4', 'Configuration', or 'Edit "
            "simulation'). Answer NO if the image is a bare matplotlib/chart export "
            "with no application UI, or blank, or not OpenRocket."
        ),
    )

    design_view_ok = await _vlm_gui_check(
        view_png_path,
        "check_10_design_view",
        (
            "Is this a screenshot of the OpenRocket 24.12 main window showing a "
            "rocket design? Answer YES only if you can see OpenRocket window chrome "
            "such as a component tree panel, toolbar, or text labels like 'OpenRocket', "
            "'Component tree', 'Rocket design', 'Nose cone', 'Body tube', "
            "'Trapezoidal fin set', 'Inner tube', 'Parachute', 'Configuration', "
            "'Mass and CG', 'Stability', or 'Flight simulations'. Answer NO if it "
            "is a bare rocket silhouette, matplotlib figure, blank image, or any "
            "image lacking OpenRocket application UI chrome."
        ),
    )

    if not sim_plot_ok or not design_view_ok:
        logger.warning("[openrocket] GUI gate FAILED — returning 0.0")
        report["final_score"] = 0.0
        report["verdict"] = "GUI gate failed"
        await _save_report(session, out_dir, report)
        return [0.0]

    logger.info("[openrocket] GUI gate PASSED")

    # ------------------------------------------------------------------
    # ANTI-COPY GATE — SHA-256 + component-id comparison
    # ------------------------------------------------------------------
    try:
        submission_ork_bytes = await session.read_bytes(ork_path)
        reference_ork_bytes  = await session.read_bytes(ref_ork_path)
    except Exception as e:
        logger.warning("[openrocket] could not read .ork files for anti-copy: %s", e)
        submission_ork_bytes = None
        reference_ork_bytes  = None

    if submission_ork_bytes is not None and reference_ork_bytes is not None:
        sub_sha = _sha256_bytes(submission_ork_bytes)
        ref_sha = _sha256_bytes(reference_ork_bytes)
        if sub_sha == ref_sha:
            logger.warning("[openrocket] ANTI-COPY gate: SHA-256 match — score=0.0")
            report["gates"]["anti_copy"] = "FAIL — byte-identical to reference"
            report["final_score"] = 0.0
            report["verdict"] = "Anti-copy gate: submission is byte-identical to reference .ork"
            await _save_report(session, out_dir, report)
            return [0.0]

        # Also compare component id multisets
        sub_root = _parse_ork_xml(submission_ork_bytes)
        ref_root = _parse_ork_xml(reference_ork_bytes)
        if sub_root is not None and ref_root is not None:
            sub_ids = _collect_ids(sub_root)
            ref_ids = _collect_ids(ref_root)
            if sub_ids and ref_ids and sub_ids == ref_ids:
                logger.warning("[openrocket] ANTI-COPY gate: component-id multiset matches reference")
                report["gates"]["anti_copy"] = "FAIL — component IDs identical to reference"
                report["final_score"] = 0.0
                report["verdict"] = "Anti-copy gate: component IDs identical to reference"
                await _save_report(session, out_dir, report)
                return [0.0]

        report["gates"]["anti_copy"] = "PASS"
    else:
        # If we cannot read either file, fall through to normal scoring
        report["gates"]["anti_copy"] = "SKIP — could not read one or both .ork files"

    # ------------------------------------------------------------------
    # GRADUATED SCORING
    # ------------------------------------------------------------------
    total_score = 0.0

    # ── 1. Required files exist (weight W_FILES = 0.10) ──────────────
    required_files = [ork_path, results_path, csv_path, sim_png_path, view_png_path]
    files_present = 0
    file_statuses = {}
    for fpath in required_files:
        try:
            exists = await session.exists(fpath)
        except Exception:
            exists = False
        fname = os.path.basename(fpath)
        file_statuses[fname] = "present" if exists else "missing"
        if exists:
            files_present += 1

    files_credit = (files_present / len(required_files)) * W_FILES
    total_score += files_credit
    report["checks"]["files_exist"] = {
        "statuses": file_statuses,
        "credit": round(files_credit, 4),
    }
    logger.info("[openrocket] files credit=%.4f (%d/%d)", files_credit, files_present, len(required_files))

    # ── 2. .ork structural validity + databranch (weight W_ODS_STRUCT = 0.20) ─
    ork_struct_credit = 0.0
    ork_struct_detail = {}

    if submission_ork_bytes is None:
        try:
            submission_ork_bytes = await session.read_bytes(ork_path)
        except Exception as e:
            logger.warning("[openrocket] cannot read .ork for structural check: %s", e)

    if submission_ork_bytes:
        ork_root = _parse_ork_xml(submission_ork_bytes)
        if ork_root is not None:
            ork_struct_detail["xml_parsed"] = True

            # Check root tag
            root_ok = ork_root.tag == "openrocket"
            ork_struct_detail["root_tag_ok"] = root_ok

            # Check rocket name
            rocket_el = ork_root.find(".//rocket")
            rocket_name = rocket_el.findtext("name", "") if rocket_el is not None else ""
            name_ok = "simple model rocket" in rocket_name.lower()
            ork_struct_detail["rocket_name"] = rocket_name
            ork_struct_detail["rocket_name_ok"] = name_ok

            # Databranch presence/count in Simulation 4
            sim4 = _find_sim4(ork_root)
            databranch_ok = False
            databranch_count = 0
            if sim4 is not None:
                flightdata = sim4.find(".//flightdata")
                if flightdata is not None:
                    branches = flightdata.findall("databranch")
                    databranch_count = len(branches)
                    if branches:
                        # Count datapoints in first branch
                        dp_count = len(branches[0].findall("datapoint"))
                        ork_struct_detail["datapoint_count"] = dp_count
                        databranch_ok = dp_count >= 300
                    ork_struct_detail["databranch_count"] = databranch_count
                else:
                    ork_struct_detail["databranch_count"] = 0
            else:
                ork_struct_detail["sim4_found"] = False

            ork_struct_detail["databranch_ok"] = databranch_ok
            native_ok, native_detail = _check_ork_native_structure(ork_root)
            ork_struct_detail["native_structure"] = native_detail

            gt_ok = False
            if reference_ork_bytes is not None:
                ref_for_gt = _parse_ork_xml(reference_ork_bytes)
                if ref_for_gt is not None:
                    gt_ok, gt_detail = _compare_ork_facts_to_gt(ork_root, ref_for_gt)
                    ork_struct_detail["gt_project_fact_match"] = gt_detail
                else:
                    ork_struct_detail["gt_project_fact_match"] = {
                        "passed": False,
                        "failures": ["reference .ork could not be parsed"],
                    }
            else:
                ork_struct_detail["gt_project_fact_match"] = {
                    "passed": False,
                    "failures": ["reference .ork bytes unavailable"],
                }

            # Compute partial credit: xml_parsed(0.3) + root+name(0.2) + databranch≥300(0.5)
            sub_credit = 0.0
            sub_credit += 0.15  # parsed OK
            if root_ok and name_ok:
                sub_credit += 0.10
            if databranch_ok:
                sub_credit += 0.25
            if native_ok:
                sub_credit += 0.20
            if gt_ok:
                sub_credit += 0.30
            ork_struct_credit = sub_credit * W_ODS_STRUCT
        else:
            ork_struct_detail["xml_parsed"] = False
    else:
        ork_struct_detail["file_read"] = False

    total_score += ork_struct_credit
    report["checks"]["ork_structure"] = {**ork_struct_detail, "credit": round(ork_struct_credit, 4)}
    logger.info("[openrocket] ork_struct credit=%.4f", ork_struct_credit)

    # ── 3. simulation_data.csv physics (weight W_CSV_PHYS = 0.20) ────
    csv_credit = 0.0
    csv_detail = {}
    results_json: Dict[str, Any] = {}

    # We need results.json values for cross-checks — load them now
    try:
        results_raw = await session.read_file(results_path)
        results_json = json.loads(results_raw if isinstance(results_raw, str) else results_raw.decode("utf-8"))
        report["checks"]["results_json_loaded"] = True
    except Exception as e:
        logger.warning("[openrocket] cannot load results.json: %s", e)
        report["checks"]["results_json_loaded"] = False

    T = float(results_json.get("time_to_apogee_s", 0) or 0)
    F = float(results_json.get("flight_time_s", 0) or 0)
    R_apogee = float(results_json.get("apogee_m", 0) or 0)
    R_maxv   = float(results_json.get("max_velocity_m_s", 0) or 0)
    R_maxa   = float(results_json.get("max_acceleration_m_s2", 0) or 0)
    R_deploy = float(results_json.get("deployment_velocity_m_s", 0) or 0)

    try:
        csv_raw = await session.read_file(csv_path)
        csv_content = csv_raw if isinstance(csv_raw, str) else csv_raw.decode("utf-8")
        parsed = _parse_csv_columns(csv_content)
    except Exception as e:
        logger.warning("[openrocket] cannot read CSV: %s", e)
        parsed = None

    if parsed is not None:
        t_vals, alt_vals, vv_vals, va_vals = parsed
        n_rows = len(t_vals)
        csv_detail["row_count"] = n_rows

        checks_passed = 0
        total_checks = 0

        def _check(cond, name):
            nonlocal checks_passed, total_checks
            total_checks += 1
            csv_detail[name] = bool(cond)
            if cond:
                checks_passed += 1

        # 7c: ≥300 rows
        _check(n_rows >= 300, "rows_ge_300")

        if n_rows >= 2:
            # 7e: time monotone
            _check(all(t_vals[i] <= t_vals[i + 1] for i in range(n_rows - 1)), "time_monotone")

            # 7f: first t ≤ 0.1
            _check(t_vals[0] <= 0.1, "first_t_le_01")

            # 7g: last t ≥ 0.95 * F
            _check(F > 0 and t_vals[-1] >= 0.95 * F, "last_t_covers_flight")

            # 7h: median time-step ≤ 0.30 s
            diffs = [t_vals[i + 1] - t_vals[i] for i in range(n_rows - 1)]
            diffs_sorted = sorted(diffs)
            median_dt = diffs_sorted[len(diffs_sorted) // 2]
            _check(median_dt <= 0.30, "median_dt_le_030")

            # 7i: max altitude within ±1.5% of R_apogee
            max_alt = max(alt_vals)
            i_apogee = alt_vals.index(max_alt)
            _check(R_apogee > 0 and _pct_err(max_alt, R_apogee) <= 0.015, "max_alt_near_apogee")

            # 7j: t at apogee within ±5% of T
            t_at_apogee = t_vals[i_apogee]
            _check(T > 0 and _pct_err(t_at_apogee, T) <= 0.05, "t_apogee_timing")

            # 7k: |vv at apogee| ≤ 2.0 m/s
            _check(abs(vv_vals[i_apogee]) <= 2.0, "vv_near_zero_at_apogee")

            # 7l: max |vv| within ±5% of R_maxv
            max_abs_vv = max(abs(v) for v in vv_vals)
            _check(R_maxv > 0 and _pct_err(max_abs_vv, R_maxv) <= 0.05, "max_vv_near_gold")

            # 7m: argmax(|va|) time < 2.5 s (boost phase)
            max_abs_va = max(abs(a) for a in va_vals)
            i_amax = next(i for i, a in enumerate(va_vals) if abs(a) == max_abs_va)
            _check(t_vals[i_amax] < 2.5, "amax_in_boost_phase")

            # 7n: max |va| within ±15% of R_maxa
            _check(R_maxa > 0 and _pct_err(max_abs_va, R_maxa) <= 0.15, "max_va_near_gold")

            # 7o: descent window mean |vv| in [0.5, 2 * R_deploy]
            if T > 0 and F > 0:
                descent_rows = [
                    (t_vals[i], vv_vals[i])
                    for i in range(n_rows)
                    if T + 1.0 <= t_vals[i] <= F - 1.0
                ]
                if len(descent_rows) >= 5:
                    mean_desc = _mean([abs(v) for _, v in descent_rows])
                    _check(
                        R_deploy > 0 and 0.5 <= mean_desc <= 2.0 * R_deploy,
                        "descent_velocity_range",
                    )

            # 7p: boost-phase concavity (mean 2nd diff > 0)
            if T > 0:
                bp_rows = [alt_vals[i] for i in range(n_rows) if 0.05 * T <= t_vals[i] <= 0.20 * T]
                if len(bp_rows) >= 5:
                    _check(_second_diff_mean(bp_rows) > 0, "boost_phase_concavity")

            # 7q: coast-phase concavity (mean 2nd diff < 0)
            if T > 0:
                cp_rows = [alt_vals[i] for i in range(n_rows) if 0.40 * T <= t_vals[i] <= 0.85 * T]
                if len(cp_rows) >= 5:
                    _check(_second_diff_mean(cp_rows) < 0, "coast_phase_concavity")

            # 7r: final altitude ≤ 5 m
            _check(alt_vals[-1] <= 5.0, "final_alt_le_5m")

        if total_checks > 0:
            csv_credit = (checks_passed / total_checks) * W_CSV_PHYS
        csv_detail["checks_passed"] = checks_passed
        csv_detail["total_checks"] = total_checks
    else:
        csv_detail["parse_failed"] = True

    total_score += csv_credit
    report["checks"]["csv_physics"] = {**csv_detail, "credit": round(csv_credit, 4)}
    logger.info("[openrocket] csv_physics credit=%.4f", csv_credit)

    # ── 4. Numeric agreement with gold (weight W_NUMERIC = 0.50) ─────
    numeric_credit = 0.0
    numeric_detail = {}

    NUMERIC_FIELDS = [
        ("apogee_m",               "apogee_m"),
        ("max_velocity_m_s",       "max_velocity_m_s"),
        ("max_acceleration_m_s2",  "max_acceleration_m_s2"),
        ("time_to_apogee_s",       "time_to_apogee_s"),
        ("flight_time_s",          "flight_time_s"),
        ("launchrod_velocity_m_s", "launchrod_velocity_m_s"),
        ("deployment_velocity_m_s","deployment_velocity_m_s"),
    ]

    # Check exact-string fields
    string_ok = 0
    for field, expected in [("simulation_name", "Simulation 4"), ("motor_designation", "C6-5")]:
        val = str(results_json.get(field, "")).strip()
        ok = val == expected
        numeric_detail[f"exact_{field}"] = ok
        if ok:
            string_ok += 1

    if results_json:
        field_count = len(NUMERIC_FIELDS)
        per_field_weight = 1.0 / (field_count + 2)  # +2 for exact-string fields

        # Exact string fields contribute to numeric credit too
        numeric_credit += string_ok * per_field_weight

        for result_key, gold_key in NUMERIC_FIELDS:
            gold_entry = gold.get(gold_key, {})
            gold_val   = float(gold_entry.get("value", 0))
            tight_tol  = gold_entry.get("tolerance_pct", 3) / 100.0
            wide_tol   = tight_tol * 2.0

            try:
                sub_val = float(results_json.get(result_key, None))
            except (TypeError, ValueError):
                sub_val = None

            if sub_val is None:
                field_credit = 0.0
                verdict = "missing"
            else:
                pct = _pct_err(sub_val, gold_val)
                if pct <= tight_tol:
                    field_credit = 1.0
                    verdict = f"within tight tol ({pct*100:.2f}% ≤ {tight_tol*100:.1f}%)"
                elif pct <= wide_tol:
                    field_credit = 0.5
                    verdict = f"within 2x tol ({pct*100:.2f}%, half credit)"
                else:
                    field_credit = 0.0
                    verdict = f"out of tolerance ({pct*100:.2f}% > {wide_tol*100:.1f}%)"

            numeric_detail[result_key] = {
                "submitted": sub_val,
                "gold": gold_val,
                "tight_tol_pct": tight_tol * 100,
                "credit": field_credit,
                "verdict": verdict,
            }
            numeric_credit += field_credit * per_field_weight

    numeric_credit = numeric_credit * W_NUMERIC
    total_score += numeric_credit
    report["checks"]["numeric_gold"] = {**numeric_detail, "credit": round(numeric_credit, 4)}
    logger.info("[openrocket] numeric credit=%.4f", numeric_credit)

    # ------------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------------
    final_score = max(0.0, min(1.0, total_score))
    report["final_score"] = round(final_score, 4)
    report["score_breakdown"] = {
        "files":       round(files_credit,       4),
        "ork_struct":  round(ork_struct_credit,  4),
        "csv_physics": round(csv_credit,         4),
        "numeric":     round(numeric_credit,     4),
        "total":       round(final_score,        4),
    }
    logger.info("[openrocket] final_score=%.4f", final_score)

    await _save_report(session, out_dir, report)
    return [final_score]


# ---------------------------------------------------------------------------
# Helper to persist the autograde report
# ---------------------------------------------------------------------------

async def _save_report(session: cb.DesktopSession, out_dir: str, report: dict):
    try:
        await session.makedirs(out_dir)
    except Exception:
        pass
    try:
        report_path = rf"{out_dir}\autograde_report.json"
        await session.write_file(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        logger.info("[openrocket] autograde report saved to %s", report_path)
    except Exception as e:
        logger.warning("[openrocket] failed to save autograde report: %s", e)
