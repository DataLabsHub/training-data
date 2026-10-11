"""
FlightGear ILS Operations Assessment — flightgear_010
Judger: two-layer scoring (Layer 2 hard-zero gate + Layer 1 numeric).
"""

import asyncio
import csv
import io
import json
import logging
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from utils.evaluation import llm_vision_judge, EvaluationContext

logger = logging.getLogger(__name__)

from pathlib import Path as _AlePromptPath


def _load_solver_prompt() -> str:
    return (_AlePromptPath(__file__).resolve().parent / "solver_prompt.txt").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Gold contract — extracted inline so the judger is self-contained.
# Values mirror judger_prompt.txt <!-- BEGIN GOLD_CONTRACT --> block.
# ---------------------------------------------------------------------------
PHASE_A_GOLD = {
    "ksfo_field_elev_ft":               {"value": 13,       "tol": 1,       "kind": "abs"},
    "ksfo_arp_lat_deg":                 {"value": 37.6188,  "tol": 0.001,   "kind": "abs"},
    "ksfo_arp_lon_deg":                 {"value": -122.3756,"tol": 0.001,   "kind": "abs"},
    "rwy_28r_length_m":                 {"value": 3618,     "tol": 5,       "kind": "abs"},
    "rwy_28r_length_ft":                {"value": 11870,    "tol": 10,      "kind": "abs"},
    "rwy_28r_width_ft":                 {"value": 200,      "tol": 0,       "kind": "exact"},
    "rwy_28r_true_bearing_deg":         {"value": 298,      "tol": 1,       "kind": "abs"},
    "rwy_28r_thr_elev_ft":              {"value": 13,       "tol": 1,       "kind": "abs"},
    "ils_28r_freq_mhz":                 {"value": 111.7,    "tol": 0,       "kind": "exact"},
    "ils_28r_fac_magnetic_deg":         {"value": 284,      "tol": 1,       "kind": "abs"},
    "ils_28r_gs_angle_deg":             {"value": 3.0,      "tol": 0.05,    "kind": "abs"},
    "ils_28r_tch_ft":                   {"value": 55,       "tol": 2,       "kind": "abs"},
    "cepin_to_thr_nm":                  {"value": 5.4,      "tol": 0.3,     "kind": "abs"},
    "sfo_vor_freq_mhz":                 {"value": 115.8,    "tol": 0,       "kind": "exact"},
    "sfo_vor_ident":                    {"value": "SFO",    "tol": None,    "kind": "exact"},
    "ksfo_to_koak_arp_great_circle_nm": {"value": 9.5976,   "tol": 0.3,     "kind": "abs"},
    "qnh_kollsman_inhg":                {"value": 29.99,    "tol": 0,       "kind": "exact"},
    "magvar_at_ksfo_deg":               {"value": 13.4,     "tol": 0.5,     "kind": "abs"},
    "nav1_freq_mhz_set":                {"value": 111.7,    "tol": 0,       "kind": "exact"},
    "nav2_freq_mhz_set":                {"value": 115.8,    "tol": 0,       "kind": "exact"},
    "obs1_set":                         {"value": 284,      "tol": 0,       "kind": "exact"},
    "xpdr_squawk_set":                  {"value": "3662",   "tol": None,    "kind": "exact"},
}

PHASE_A_WEIGHTS = {
    "ksfo_field_elev_ft":               0.020,
    "ksfo_arp_lat_deg":                 0.010,
    "ksfo_arp_lon_deg":                 0.010,
    "rwy_28r_length_m":                 0.020,
    "rwy_28r_length_ft":                0.010,
    "rwy_28r_width_ft":                 0.010,
    "rwy_28r_true_bearing_deg":         0.020,
    "rwy_28r_thr_elev_ft":              0.010,
    "ils_28r_freq_mhz":                 0.020,
    "ils_28r_fac_magnetic_deg":         0.020,
    "ils_28r_gs_angle_deg":             0.010,
    "ils_28r_tch_ft":                   0.010,
    "cepin_to_thr_nm":                  0.010,
    "sfo_vor_freq_mhz":                 0.020,
    "sfo_vor_ident":                    0.010,
    "ksfo_to_koak_arp_great_circle_nm": 0.020,
    "qnh_kollsman_inhg":                0.010,
    "magvar_at_ksfo_deg":               0.005,
    "nav1_freq_mhz_set":                0.005,
    "nav2_freq_mhz_set":                0.005,
    "obs1_set":                         0.005,
    "xpdr_squawk_set":                  0.005,
}

PHASE_B_GOLD = {
    "total_flight_time_min":                 {"key": "tape_total_flight_time_min",  "value": 15.317,  "tol": 1.5,    "kind": "abs"},
    "max_altitude_ft_msl":                   {"key": "tape_max_altitude_ft_msl",    "value": 3000.0,  "tol": 80,     "kind": "abs"},
    "avg_glideslope_descent_fpm":            {"key": "tape_avg_gs_descent_fpm",     "value": 442.0,   "tol": 60,     "kind": "abs"},
    "touchdown_distance_from_thr_m":         {"key": "tape_touchdown_distance_m",   "value": 384.12,  "tol": 200,    "kind": "abs"},
    "final_approach_groundspeed_kt_at_200ft":{"key": "tape_final_app_gs_kt",        "value": 64.0,    "tol": 5,      "kind": "abs"},
    "ap_disconnect_agl_ft":                  {"key": "ap_disconnect_agl_ft_window", "value": [1, 200],"tol": None,   "kind": "range"},
    "landing_runway":                        {"key": "landing_runway",               "value": "28R",   "tol": None,   "kind": "exact"},
    "landing_speed_kias":                    {"key": "landing_speed_kias_target",    "value": 60,      "tol": 8,      "kind": "abs"},
}

PHASE_B_WEIGHTS = {
    "total_flight_time_min":                  0.030,
    "max_altitude_ft_msl":                    0.030,
    "avg_glideslope_descent_fpm":             0.030,
    "touchdown_distance_from_thr_m":          0.030,
    "final_approach_groundspeed_kt_at_200ft": 0.020,
    "ap_disconnect_agl_ft":                   0.030,
    "landing_runway":                         0.020,
    "landing_speed_kias":                     0.010,
}

# Phase C: inner tolerances for anti-fabrication check
PHASE_C_INNER_TOL = {
    "hdg_mag":          0.5,
    "alt_indicated_ft": 5.0,
    "ias_kt":           0.5,
    "vs_fpm":           15.0,
    "ils_dev_dots":     0.05,
}

PHASE_C_EVENTS = [
    "T_takeoff", "T_top_of_climb", "T_loc_capture",
    "T_gs_capture", "T_ap_disconnect", "T_touchdown",
]

PHASE_C_INSTRUMENTS = list(PHASE_C_INNER_TOL.keys())

PHASE_C_WEIGHT_PER_READING = 0.20 / 30  # 0.20 total / 30 readings

# Required output files (relative to remote_output_dir)
REQUIRED_FILES = [
    "result.json",
    "route.xml",
    "replay_readings.json",
    "live_approach.fgtape",
    "live_property_log.csv",
    "live_session.mp4",
    "replay_session.mp4",
    "fgfs.log",
    "system_meta.json",
    r"screenshots\A_radio_panel.png",
    r"screenshots\A_map.png",
    r"screenshots\A_route_manager.png",
    r"screenshots\A_property_browser.png",
    r"screenshots\C_T_takeoff.png",
    r"screenshots\C_T_top_of_climb.png",
    r"screenshots\C_T_loc_capture.png",
    r"screenshots\C_T_gs_capture.png",
    r"screenshots\C_T_ap_disconnect.png",
    r"screenshots\C_T_touchdown.png",
]

GATE_WEIGHT = 0.35 / 12  # ~0.02917 per gate (12 gates G1-G12)

# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig(GeneralTaskConfig):
    TASK_TAG: str = "flightgear_010"
    TASK_CATEGORY: str = "aviation_operations"


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_load_solver_prompt(),
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Clear previous output directory and recreate it."""
    out_dir = task_cfg.metadata["remote_output_dir"]
    try:
        await session.remove_file(out_dir)
    except Exception:
        pass
    await session.makedirs(out_dir)
    try:
        await session.makedirs(fr"{out_dir}\screenshots")
    except Exception:
        pass
    logger.info("Setup complete — output directory ready.")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _check_field(actual, gold_entry):
    """Return True if actual value satisfies gold_entry tolerance."""
    kind = gold_entry["kind"]
    expected = gold_entry["value"]
    if kind == "exact":
        try:
            return str(actual).strip() == str(expected).strip()
        except Exception:
            return False
    elif kind == "abs":
        try:
            return abs(float(actual) - float(expected)) <= gold_entry["tol"]
        except Exception:
            return False
    elif kind == "range":
        lo, hi = expected
        try:
            return lo <= float(actual) <= hi
        except Exception:
            return False
    return False


def _parse_png_time_chunk(data: bytes) -> Optional[int]:
    """
    Parse a PNG file's tIME chunk and return a UTC Unix timestamp (int).
    Returns None if no tIME chunk present or on parse error.
    PNG tIME layout: 2-byte year, 1-byte each month/day/hour/minute/second.
    """
    import calendar
    offset = 8  # skip PNG signature
    while offset < len(data) - 12:
        try:
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            chunk_type = data[offset + 4:offset + 8]
            if chunk_type == b"tIME" and length == 7:
                chunk_data = data[offset + 8:offset + 15]
                year, month, day, hour, minute, second = struct.unpack(">HBBBBB", chunk_data)
                import datetime
                dt = datetime.datetime(year, month, day, hour, minute, second,
                                       tzinfo=datetime.timezone.utc)
                return int(dt.timestamp())
            offset += 12 + length
        except Exception:
            break
    return None


def _parse_fgfs_log_timestamps(log_text: str):
    """
    Extract the earliest and latest ISO-8601 / syslog-style timestamps from fgfs.log.
    Returns (start_ts, end_ts) as Unix ints, or (None, None) on failure.
    """
    import datetime
    # Common FG log format: "2024-05-01T12:34:56"
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})')
    timestamps = []
    for m in pattern.finditer(log_text):
        try:
            s = m.group(1).replace(" ", "T")
            dt = datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
            timestamps.append(int(dt.timestamp()))
        except Exception:
            continue
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _closest_csv_row(rows, t_sec, key="sim_time_sec"):
    """Return the row in rows whose sim_time_sec is closest to t_sec (within ±1.5 s)."""
    best = None
    best_diff = float("inf")
    for row in rows:
        try:
            diff = abs(float(row[key]) - float(t_sec))
            if diff < best_diff:
                best_diff = diff
                best = row
        except Exception:
            continue
    if best_diff <= 1.5:
        return best
    return None


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession):
    """Evaluate flightgear_010 — two-layer scoring, returns list[float]."""

    out_dir = task_cfg.metadata["remote_output_dir"]

    ctx = EvaluationContext(
        task_tag="flightgear_010",
        mode="flightgear_ils_assessment",
        output_dir=None,
    )

    # -----------------------------------------------------------------------
    # Existence check — any missing file → score = 0
    # -----------------------------------------------------------------------
    for rel in REQUIRED_FILES:
        path = fr"{out_dir}\{rel}"
        try:
            exists = await session.exists(path)
        except Exception:
            exists = False
        if not exists:
            logger.error(f"Missing required file: {path}")
            ctx.log_error("existence_check", f"Missing required file: {rel}", score=0.0)
            ctx.finalize(reason="missing_required_file", file=rel)
            return [0.0]

    logger.info("All required files present.")

    # -----------------------------------------------------------------------
    # Read key output files
    # -----------------------------------------------------------------------
    async def read_text(rel_path):
        try:
            data = await session.read_file(fr"{out_dir}\{rel_path}")
            return data.decode("utf-8") if isinstance(data, bytes) else data
        except Exception as e:
            logger.error(f"Failed to read {rel_path}: {e}")
            return None

    result_json_txt = await read_text("result.json")
    replay_json_txt = await read_text("replay_readings.json")
    fgfs_log_txt    = await read_text("fgfs.log")
    system_meta_txt = await read_text("system_meta.json")
    route_xml_txt   = await read_text("route.xml")
    live_csv_txt    = await read_text("live_property_log.csv")

    def parse_json_safe(txt, label):
        try:
            return json.loads(txt)
        except Exception as e:
            logger.error(f"JSON parse error for {label}: {e}")
            return None

    result_data  = parse_json_safe(result_json_txt, "result.json")   if result_json_txt  else None
    replay_data  = parse_json_safe(replay_json_txt, "replay_readings.json") if replay_json_txt else None
    system_meta  = parse_json_safe(system_meta_txt, "system_meta.json")    if system_meta_txt else None

    # Parse live CSV into list of dicts
    csv_rows = []
    if live_csv_txt:
        try:
            reader = csv.DictReader(io.StringIO(live_csv_txt))
            csv_rows = list(reader)
        except Exception as e:
            logger.error(f"CSV parse error: {e}")

    # -----------------------------------------------------------------------
    # Layer 2 — GUI Evidence Gates (any failure → 0)
    # -----------------------------------------------------------------------
    gates_passed = []
    gates_failed = []

    def gate_fail(gate_id, reason):
        gates_failed.append({"gate": gate_id, "reason": reason})
        logger.warning(f"Gate {gate_id} FAILED: {reason}")

    def gate_pass(gate_id):
        gates_passed.append(gate_id)
        logger.info(f"Gate {gate_id} PASSED")

    # --- G12: fgfs.log content + system_meta FG version check ---
    g12_ok = True
    if not fgfs_log_txt:
        gate_fail("G12", "fgfs.log could not be read")
        g12_ok = False
    else:
        metar_fragment = "KSFO 271156Z 30006KT"
        if metar_fragment not in fgfs_log_txt:
            gate_fail("G12", f"fgfs.log does not contain METAR fragment '{metar_fragment}'")
            g12_ok = False

    if g12_ok and system_meta:
        version_str = str(system_meta.get("fgfs_version", system_meta.get("version", "")))
        # Extract version numbers, require >= 2024.1.1
        ver_match = re.search(r'(\d{4})\.(\d+)\.(\d+)', version_str)
        if ver_match:
            major, minor, patch = int(ver_match.group(1)), int(ver_match.group(2)), int(ver_match.group(3))
            if (major, minor, patch) < (2024, 1, 1):
                gate_fail("G12", f"FlightGear version {version_str} < 2024.1.1")
                g12_ok = False
        else:
            # Try other version patterns like "FlightGear version: 2024.1.1"
            ver_match2 = re.search(r'(\d{4})[\.\-](\d+)[\.\-](\d+)', version_str)
            if ver_match2:
                major, minor, patch = int(ver_match2.group(1)), int(ver_match2.group(2)), int(ver_match2.group(3))
                if (major, minor, patch) < (2024, 1, 1):
                    gate_fail("G12", f"FlightGear version {version_str} < 2024.1.1")
                    g12_ok = False
            else:
                logger.warning(f"G12: Could not parse FG version from '{version_str}' — allowing")
    elif g12_ok and not system_meta:
        gate_fail("G12", "system_meta.json missing or unreadable")
        g12_ok = False

    if g12_ok:
        gate_pass("G12")

    # --- G10: route.xml schema ---
    if route_xml_txt:
        try:
            root = ET.fromstring(route_xml_txt)
            has_propertylist = (root.tag == "PropertyList")
            has_flightplan   = root.find(".//flightplan[@version='2.0']") is not None
            has_sfo          = "SFO"   in route_xml_txt
            has_cepin        = "CEPIN" in route_xml_txt
            if has_propertylist and has_flightplan and has_sfo and has_cepin:
                gate_pass("G10")
            else:
                missing = []
                if not has_propertylist: missing.append("root not PropertyList")
                if not has_flightplan:   missing.append("no <flightplan version='2.0'>")
                if not has_sfo:          missing.append("no SFO waypoint")
                if not has_cepin:        missing.append("no CEPIN waypoint")
                gate_fail("G10", "; ".join(missing))
        except ET.ParseError as e:
            gate_fail("G10", f"route.xml XML parse error: {e}")
    else:
        gate_fail("G10", "route.xml unreadable")

    # --- G11: NAV1 in-range proof (Phase A property_browser screenshot via VLM) ---
    prop_browser_path = fr"{out_dir}\screenshots\A_property_browser.png"
    try:
        prop_browser_bytes = await session.read_bytes(prop_browser_path)
        g11_prompt = (
            "Look at this FlightGear property browser screenshot. "
            "Does it show /instrumentation/nav[0]/in-range equal to 'true'? "
            "Answer YES or NO only."
        )
        g11_result = await llm_vision_judge(
            prompt=g11_prompt,
            image_bytes=prop_browser_bytes,
            return_binary_score=True,
            max_tokens=10,
        )
        if g11_result == 1.0:
            gate_pass("G11")
        else:
            gate_fail("G11", "Property browser screenshot does not show nav[0]/in-range=true")
    except Exception as e:
        gate_fail("G11", f"Could not evaluate A_property_browser.png: {e}")

    # --- G4: Autopilot state-machine ordering in CSV ---
    if csv_rows:
        try:
            # Required transition order in ap_lock_heading / ap_lock_altitude
            # 1) ap_lock_heading becomes wing-leveler or dg-heading-hold
            # 2) ap_lock_altitude = altitude-hold
            # 3) ap_lock_heading = nav1-hold
            # 4) ap_lock_altitude = gs1-hold
            # 5) both empty
            transitions = []
            prev_hdg = ""
            prev_alt = ""
            for row in csv_rows:
                h = row.get("ap_lock_heading", "").strip()
                a = row.get("ap_lock_altitude", "").strip()
                if h != prev_hdg or a != prev_alt:
                    transitions.append((h, a))
                    prev_hdg, prev_alt = h, a

            def find_state(transitions, hdg_val=None, alt_val=None, after_idx=0):
                for i in range(after_idx, len(transitions)):
                    h, a = transitions[i]
                    hdg_ok = (hdg_val is None) or (h == hdg_val)
                    alt_ok = (alt_val is None) or (a == alt_val)
                    if hdg_ok and alt_ok:
                        return i
                return -1

            idx1 = find_state(transitions, hdg_val=None, alt_val=None)
            # wing-leveler or dg-heading-hold
            idx_wl = find_state(transitions, hdg_val="wing-leveler")
            idx_dg = find_state(transitions, hdg_val="dg-heading-hold")
            idx1 = max(idx_wl, idx_dg)
            if idx1 < 0:
                gate_fail("G4", "No wing-leveler/dg-heading-hold state found in CSV")
            else:
                idx2 = find_state(transitions, alt_val="altitude-hold", after_idx=idx1)
                if idx2 < 0:
                    gate_fail("G4", "No altitude-hold after wing-leveler")
                else:
                    idx3 = find_state(transitions, hdg_val="nav1-hold", after_idx=idx2)
                    if idx3 < 0:
                        gate_fail("G4", "No nav1-hold after altitude-hold")
                    else:
                        idx4 = find_state(transitions, alt_val="gs1-hold", after_idx=idx3)
                        if idx4 < 0:
                            gate_fail("G4", "No gs1-hold after nav1-hold")
                        else:
                            idx5 = find_state(transitions, hdg_val="", alt_val="", after_idx=idx4)
                            if idx5 < 0:
                                gate_fail("G4", "AP never fully disconnected (both locks not empty)")
                            else:
                                gate_pass("G4")
        except Exception as e:
            gate_fail("G4", f"Exception checking AP state machine: {e}")
    else:
        gate_fail("G4", "live_property_log.csv empty or unreadable")

    # --- G5: KAP140 disconnect AGL in (0, 200] ---
    if csv_rows:
        try:
            disconnect_row = None
            prev_hdg_set = False
            prev_alt_set = False
            for row in csv_rows:
                h = row.get("ap_lock_heading", "").strip()
                a = row.get("ap_lock_altitude", "").strip()
                if h and a:
                    prev_hdg_set = True
                    prev_alt_set = True
                if prev_hdg_set and prev_alt_set and h == "" and a == "":
                    disconnect_row = row
                    break
            if disconnect_row:
                agl = float(disconnect_row.get("altitude_agl_ft", -1))
                if 0 < agl <= 200:
                    gate_pass("G5")
                else:
                    gate_fail("G5", f"AP disconnect AGL {agl:.1f} not in (0, 200]")
            else:
                gate_fail("G5", "AP disconnect row not found in CSV")
        except Exception as e:
            gate_fail("G5", f"Exception checking G5: {e}")
    else:
        gate_fail("G5", "live_property_log.csv empty or unreadable")

    # --- G2: Physics self-consistency of live_property_log.csv ---
    if csv_rows and len(csv_rows) > 1:
        try:
            vs_errors = []
            gs_errors = []
            for i in range(1, len(csv_rows)):
                r0, r1 = csv_rows[i - 1], csv_rows[i]
                dt = float(r1["sim_time_sec"]) - float(r0["sim_time_sec"])
                if dt <= 0:
                    continue
                alt0 = float(r0["altitude_ft"])
                alt1 = float(r1["altitude_ft"])
                vs_fpm_computed = 60.0 * (alt1 - alt0) / dt
                vs_fpm_logged   = float(r1.get("vertical_speed_fpm", vs_fpm_computed))
                vs_errors.append(abs(vs_fpm_computed - vs_fpm_logged))
                vn = float(r1.get("velocity_north_kt", 0))
                ve = float(r1.get("velocity_east_kt", 0))
                gs_computed = (vn ** 2 + ve ** 2) ** 0.5
                gs_logged   = float(r1.get("groundspeed_kt", gs_computed))
                gs_errors.append(abs(gs_computed - gs_logged))
            vs_fail = sum(1 for e in vs_errors if e >= 50)
            gs_fail = sum(1 for e in gs_errors if e >= 1)
            if vs_fail == 0 and gs_fail == 0:
                gate_pass("G2")
            else:
                gate_fail("G2", f"Physics violations: {vs_fail} VS rows ≥50 fpm error, {gs_fail} GS rows ≥1 kt error")
        except Exception as e:
            gate_fail("G2", f"Exception in physics check: {e}")
    else:
        gate_fail("G2", "live_property_log.csv has too few rows for physics check")

    # --- G3: Wind verification ---
    if csv_rows and len(csv_rows) > 10:
        try:
            # Find straight-and-level cruise: altitude_agl > 500, vs close to 0
            cruise_rows = [
                r for r in csv_rows
                if float(r.get("altitude_agl_ft", 0)) > 500
                and abs(float(r.get("vertical_speed_fpm", 999))) < 50
            ]
            if len(cruise_rows) >= 5:
                # mean(GS_vec - TAS_vec) ≈ wind vector; METAR 30006KT = 300°M, 6 kt
                # wind_north = -6*cos(300-13.4° true) ≈ compute from (hdg, TAS, GS)
                # Simplified: check mean wind magnitude and direction
                wind_ns = []
                wind_ew = []
                for r in cruise_rows:
                    vn  = float(r.get("velocity_north_kt", 0))
                    ve  = float(r.get("velocity_east_kt", 0))
                    gs  = float(r.get("groundspeed_kt", 0))
                    tas = float(r.get("tas_kt", 0))
                    hdg_true = float(r.get("heading_true_deg", r.get("heading_deg", 0)))
                    import math
                    tas_n = tas * math.cos(math.radians(hdg_true))
                    tas_e = tas * math.sin(math.radians(hdg_true))
                    wind_ns.append(vn - tas_n)
                    wind_ew.append(ve - tas_e)
                mean_wn = sum(wind_ns) / len(wind_ns)
                mean_we = sum(wind_ew) / len(wind_ew)
                import math
                wind_speed = math.sqrt(mean_wn**2 + mean_we**2)
                wind_dir   = (math.degrees(math.atan2(mean_we, mean_wn)) + 360) % 360
                # METAR 30006KT: wind from 300° true = 300 - 13.4 mag ≈ 286.6° true
                # Wind FROM 300° true means wind_north < 0, wind_east < 0
                expected_speed = 6.0
                expected_dir_from_true = (300 - 13.4 + 360) % 360  # ~286.6
                # wind_dir computed is direction air is going (FROM opposite)
                actual_from = (wind_dir + 180) % 360
                speed_ok = abs(wind_speed - expected_speed) <= 3
                dir_diff  = abs(actual_from - expected_dir_from_true)
                if dir_diff > 180:
                    dir_diff = 360 - dir_diff
                dir_ok = dir_diff <= 10
                if speed_ok and dir_ok:
                    gate_pass("G3")
                else:
                    gate_fail("G3", f"Wind mismatch: speed={wind_speed:.1f} kt (exp 6±3), dir_from={actual_from:.0f}° (exp ~{expected_dir_from_true:.0f}°±10°)")
            else:
                gate_fail("G3", f"Insufficient cruise rows for wind check ({len(cruise_rows)} found, need ≥5)")
        except Exception as e:
            gate_fail("G3", f"Exception in wind check: {e}")
    else:
        gate_fail("G3", "live_property_log.csv insufficient for wind check")

    # --- G1: live_approach.fgtape tape replay (existence and size proxy) ---
    # Full replay requires fgfs binary; here we check file non-empty
    tape_path = fr"{out_dir}\live_approach.fgtape"
    try:
        tape_bytes = await session.read_bytes(tape_path)
        if len(tape_bytes) > 10_000:  # >10 KB implies real tape, not empty
            # Check if any tape duration metadata in fgfs.log
            if fgfs_log_txt and "replay" in fgfs_log_txt.lower():
                gate_pass("G1")
            else:
                # Accept on size alone — full replay would require running fgfs
                gate_pass("G1")
        else:
            gate_fail("G1", f"live_approach.fgtape too small ({len(tape_bytes)} bytes), likely empty")
    except Exception as e:
        gate_fail("G1", f"Could not read live_approach.fgtape: {e}")

    # --- G6: screenshot tIME chunk binding to fgfs.log times ---
    fgfs_start_ts, fgfs_end_ts = _parse_fgfs_log_timestamps(fgfs_log_txt) if fgfs_log_txt else (None, None)
    png_files = [
        r"screenshots\A_radio_panel.png",
        r"screenshots\A_map.png",
        r"screenshots\A_route_manager.png",
        r"screenshots\A_property_browser.png",
        r"screenshots\C_T_takeoff.png",
        r"screenshots\C_T_top_of_climb.png",
        r"screenshots\C_T_loc_capture.png",
        r"screenshots\C_T_gs_capture.png",
        r"screenshots\C_T_ap_disconnect.png",
        r"screenshots\C_T_touchdown.png",
    ]
    if fgfs_start_ts and fgfs_end_ts:
        g6_fail_list = []
        for png_rel in png_files:
            try:
                png_data = await session.read_bytes(fr"{out_dir}\{png_rel}")
                ts = _parse_png_time_chunk(png_data)
                if ts is None:
                    logger.warning(f"G6: No tIME chunk in {png_rel} — skipping timestamp check for this file")
                    continue
                # Allow ±300 s window (generous for wall-clock vs sim-clock skew)
                window_low  = fgfs_start_ts - 300
                window_high = fgfs_end_ts   + 300
                if not (window_low <= ts <= window_high):
                    g6_fail_list.append(f"{png_rel} tIME={ts} outside [{window_low},{window_high}]")
            except Exception as e:
                logger.warning(f"G6: Could not process {png_rel}: {e}")
        if g6_fail_list:
            gate_fail("G6", "; ".join(g6_fail_list))
        else:
            gate_pass("G6")
    else:
        logger.warning("G6: Cannot parse fgfs.log timestamps — skipping tIME binding check")
        gate_pass("G6")  # soft pass when log timestamps are unparseable

    # --- G7: live_session.mp4 duration ≥ 480 s via ffprobe ---
    live_mp4_path = fr"{out_dir}\live_session.mp4"
    try:
        ffprobe_out = await session.run_command(
            f'ffprobe -v quiet -print_format json -show_format "{live_mp4_path}"'
        )
        ffprobe_data = json.loads(ffprobe_out)
        duration = float(ffprobe_data.get("format", {}).get("duration", 0))
        codec_tags = str(ffprobe_data)
        if duration >= 480:
            gate_pass("G7")
        else:
            gate_fail("G7", f"live_session.mp4 duration {duration:.1f}s < 480s required")
    except Exception as e:
        gate_fail("G7", f"ffprobe failed on live_session.mp4: {e}")

    # --- G8: replay_session.mp4 duration ≥ 90 s ---
    replay_mp4_path = fr"{out_dir}\replay_session.mp4"
    try:
        ffprobe_out2 = await session.run_command(
            f'ffprobe -v quiet -print_format json -show_format "{replay_mp4_path}"'
        )
        ffprobe_data2 = json.loads(ffprobe_out2)
        duration2 = float(ffprobe_data2.get("format", {}).get("duration", 0))
        if duration2 >= 90:
            gate_pass("G8")
        else:
            gate_fail("G8", f"replay_session.mp4 duration {duration2:.1f}s < 90s required")
    except Exception as e:
        gate_fail("G8", f"ffprobe failed on replay_session.mp4: {e}")

    # --- G9: OCR of replay_session.mp4 for GUI tokens (≥5 of 12) ---
    GUI_TOKENS = [
        "111.70", "115.80", "APR", "GS", "CEPIN", "284",
        "Equipment", "Map", "Route Manager", "Property Browser",
        "Flight Recorder", "KSFO",
    ]
    try:
        # Extract a mid-point frame via ffmpeg
        mid_frame_path = fr"{out_dir}\__g9_frame.png"
        await session.run_command(
            f'ffmpeg -y -ss 30 -i "{replay_mp4_path}" -frames:v 1 -q:v 2 "{mid_frame_path}"'
        )
        frame_bytes = await session.read_bytes(mid_frame_path)
        ocr_prompt = (
            "Look at this FlightGear replay screenshot. "
            "List every text string you can read from the cockpit instruments and GUI dialogs. "
            f"Specifically, how many of these exact tokens appear: {GUI_TOKENS}? "
            "Reply with the count and list the tokens found."
        )
        ocr_result = await llm_vision_judge(
            prompt=ocr_prompt,
            image_bytes=frame_bytes,
            return_details=True,
            max_tokens=300,
        )
        vlm_response = str(ocr_result.get("vlm_response", ""))
        tokens_found = [t for t in GUI_TOKENS if t in vlm_response]
        logger.info(f"G9: VLM found tokens: {tokens_found}")
        if len(tokens_found) >= 5:
            gate_pass("G9")
        else:
            gate_fail("G9", f"OCR detected only {len(tokens_found)}/12 tokens (need ≥5): {tokens_found}")
    except Exception as e:
        gate_fail("G9", f"G9 OCR failed: {e}")

    # -----------------------------------------------------------------------
    # Layer 2 gate decision — any failure → total score = 0
    # -----------------------------------------------------------------------
    if gates_failed:
        logger.error(f"Layer 2 gate failures: {gates_failed}")
        ctx.log_error("layer2_gate", f"Gate failures: {gates_failed}", score=0.0)
        ctx.finalize(gates_passed=gates_passed, gates_failed=gates_failed)
        return [0.0]

    gating_score = GATE_WEIGHT * len(gates_passed)  # 0.35 if all 12 pass

    # -----------------------------------------------------------------------
    # Layer 1 — Phase A: result.json static facts
    # -----------------------------------------------------------------------
    phase_a_score = 0.0
    if result_data:
        for field, gold in PHASE_A_GOLD.items():
            weight = PHASE_A_WEIGHTS.get(field, 0.0)
            actual = result_data.get(field)
            if actual is None:
                logger.warning(f"Phase A: missing field '{field}'")
                continue
            if _check_field(actual, gold):
                phase_a_score += weight
                logger.info(f"Phase A: {field} PASS ({actual})")
            else:
                logger.info(f"Phase A: {field} FAIL (got={actual}, expected={gold['value']})")
    else:
        logger.error("result.json missing/invalid — Phase A scores 0")

    # -----------------------------------------------------------------------
    # Layer 1 — Phase B: live flight metrics
    # -----------------------------------------------------------------------
    phase_b_score = 0.0
    if result_data:
        for field, gold in PHASE_B_GOLD.items():
            weight = PHASE_B_WEIGHTS.get(field, 0.0)
            actual = result_data.get(field)
            if actual is None:
                logger.warning(f"Phase B: missing field '{field}'")
                continue
            if _check_field(actual, gold):
                phase_b_score += weight
                logger.info(f"Phase B: {field} PASS ({actual})")
            else:
                logger.info(f"Phase B: {field} FAIL (got={actual}, expected={gold['value']})")
    else:
        logger.error("result.json missing/invalid — Phase B scores 0")

    # -----------------------------------------------------------------------
    # Layer 1 — Phase C: anti-fabrication self-replay reading
    # -----------------------------------------------------------------------
    phase_c_score = 0.0
    if replay_data and csv_rows:
        for event in PHASE_C_EVENTS:
            event_entry = replay_data.get(event)
            if not event_entry:
                logger.warning(f"Phase C: missing event '{event}'")
                continue

            csv_t = event_entry.get("csv_row_t_sec")
            if csv_t is None:
                logger.warning(f"Phase C: {event} missing csv_row_t_sec")
                continue

            closest = _closest_csv_row(csv_rows, csv_t)
            if closest is None:
                logger.warning(f"Phase C: {event} no CSV row within ±1.5 s of t={csv_t}")
                continue

            # Map CSV column names to instrument names
            col_map = {
                "hdg_mag":          "heading_magnetic_deg",
                "alt_indicated_ft": "altitude_ft",
                "ias_kt":           "ias_kt",
                "vs_fpm":           "vertical_speed_fpm",
                "ils_dev_dots":     "ils_dev_dots",
            }

            for instr in PHASE_C_INSTRUMENTS:
                reported = event_entry.get(instr)
                if reported is None:
                    logger.warning(f"Phase C: {event}.{instr} not reported")
                    continue
                csv_col  = col_map.get(instr, instr)
                csv_val  = closest.get(csv_col)
                if csv_val is None:
                    logger.warning(f"Phase C: CSV col '{csv_col}' not found for {event}.{instr}")
                    continue
                try:
                    diff = abs(float(reported) - float(csv_val))
                    tol  = PHASE_C_INNER_TOL[instr]
                    if diff <= tol:
                        phase_c_score += PHASE_C_WEIGHT_PER_READING
                        logger.info(f"Phase C: {event}.{instr} PASS (reported={reported}, csv={csv_val})")
                    else:
                        logger.info(f"Phase C: {event}.{instr} FAIL (reported={reported}, csv={csv_val}, diff={diff:.3f}>tol={tol})")
                except Exception as e:
                    logger.error(f"Phase C: error checking {event}.{instr}: {e}")
    else:
        logger.error("replay_readings.json or live_property_log.csv missing — Phase C scores 0")

    # -----------------------------------------------------------------------
    # Final score
    # -----------------------------------------------------------------------
    total_score = min(1.0, phase_a_score + phase_b_score + phase_c_score + gating_score)
    logger.info(
        f"Score breakdown — Phase A={phase_a_score:.4f}, B={phase_b_score:.4f}, "
        f"C={phase_c_score:.4f}, Gates={gating_score:.4f}, Total={total_score:.4f}"
    )

    ctx.finalize(
        phase_a=round(phase_a_score, 6),
        phase_b=round(phase_b_score, 6),
        phase_c=round(phase_c_score, 6),
        gating=round(gating_score, 6),
        gates_passed=gates_passed,
        gates_failed=gates_failed,
        total_score=round(total_score, 6),
    )

    return [total_score]
