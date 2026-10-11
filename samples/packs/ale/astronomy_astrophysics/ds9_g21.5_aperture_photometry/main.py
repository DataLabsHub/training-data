"""
Judger for: ds9_g21.5_aperture_photometry
Category  : astronomy_astrophysics

Four evaluation layers (matching reference/grader.py logic):
  1. XPA   — query the agent's live DS9 process titled "ds9_g21" via xpaget.
  2. Artifact — verify output/{sources.reg, session.bck, ds9_view.png,
                               photometry.csv, results.json}.
  3. Round-trip — load output/session.bck into a fresh DS9, verify restored state.
  4. Numeric — RA/Dec and net_counts against grader_assets/private_gold_contract.json.

Hard gates (any failing gate ⇒ score = 0.0, passed = False):
  G1. Live DS9 process evidence (xpaget on agent's XPA title must respond).
  G2. Round-trip restore of session.bck must succeed when DS9 is available.

Pass threshold: 0.50  (informational only; float score is the ranking signal).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    TASK_TAG: str = "ds9_g21.5_aperture_photometry"
    TASK_CATEGORY: str = "astronomy_astrophysics"

    @property
    def input_fits(self) -> str:
        return fr"{self.task_dir}\input\chandra_acis_obs1843.fits"

    @property
    def gold_path(self) -> str:
        return fr"{self.task_dir}\grader_assets\private_gold_contract.json"

    @property
    def task_description(self) -> str:
        return (
            "Open the bundled Chandra ACIS-S image of SNR G21.5-0.9 in SAOImage DS9 "
            "(XPA title: ds9_g21), configure display (log scale, heat colormap), "
            "place a 30\" source circle and 100\"-150\" background annulus on pulsar "
            "PSR J1833-1034 (RA=278.389875°, Dec=-10.568750°), perform aperture "
            "photometry, and save output/sources.reg, output/session.bck, "
            "output/ds9_view.png, output/photometry.csv, and output/results.json. "
            "Do NOT close DS9 before grading."
        )

    def to_metadata(self) -> dict:
        meta = super().to_metadata()
        meta.update({
            "input_fits": self.input_fits,
            "gold_path": self.gold_path,
        })
        return meta


config = TaskConfig()

# ---------------------------------------------------------------------------
# Framework entry points
# ---------------------------------------------------------------------------

@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_load_solver_prompt(),
            metadata=config.to_metadata(),
            computer={"provider": "computer",
                      "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    out_dir = task_cfg.metadata["remote_output_dir"]
    logger.info("[ds9_g21] start(): clearing output dir %s", out_dir)
    try:
        await session.remove_file(out_dir)
    except Exception:
        pass
    try:
        await session.makedirs(out_dir)
    except Exception as e:
        logger.warning("[ds9_g21] makedirs failed: %s", e)


# ---------------------------------------------------------------------------
# Evaluation helpers (mirror reference/grader.py, run on the grader host)
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 0.50
DEFAULT_XPA_TITLE = "ds9_g21"
ROUND_TRIP_TITLE = "ds9_grader_rt"
FITS_NAME = "chandra_acis_obs1843.fits"

WEIGHTS: dict[str, float] = {
    "xpa_file_loaded":        0.05,
    "xpa_scale_log":          0.05,
    "xpa_cmap_set":           0.05,
    "xpa_frame_loaded":       0.05,
    "xpa_circle_near_target": 0.05,
    "xpa_annulus_near_target":0.05,
    "artifact_reg_valid":     0.10,
    "artifact_bck_archive":   0.10,
    "artifact_bck_roundtrip": 0.15,
    "artifact_png_valid":     0.05,
    "numeric_position":       0.10,
    "numeric_net_counts":     0.15,
    "artifact_output_schema": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def _xpaget(title: str, *args: str, timeout: float = 5.0) -> str | None:
    if shutil.which("xpaget") is None:
        return None
    try:
        r = subprocess.run(
            ["xpaget", title, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return r.stdout if r.returncode == 0 else None


def _xpaset_p(title: str, *args: str, timeout: float = 30.0) -> tuple[bool, str]:
    if shutil.which("xpaset") is None:
        return False, "xpaset not on PATH"
    try:
        r = subprocess.run(
            ["xpaset", "-p", title, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def _angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1 = math.radians(ra1), math.radians(dec1)
    r2, d2 = math.radians(ra2), math.radians(dec2)
    cs = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cs)))) * 3600.0


def _parse_sexa(s: str) -> float:
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    parts = re.split(r"[:\s]+", s)
    a = float(parts[0])
    b = float(parts[1]) if len(parts) > 1 else 0.0
    c = float(parts[2]) if len(parts) > 2 else 0.0
    return sign * (a + b / 60.0 + c / 3600.0)


def _to_deg_ra(token: str) -> float | None:
    try:
        return _parse_sexa(token) * 15.0 if ":" in token else float(token)
    except ValueError:
        return None


def _to_deg_dec(token: str) -> float | None:
    try:
        return _parse_sexa(token) if ":" in token else float(token)
    except ValueError:
        return None


def _parse_reg_text(text: str) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    coord_sys = "fk5"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low in {"fk5", "icrs", "j2000", "image", "physical"}:
            coord_sys = low
            continue
        m = re.match(r"(circle|annulus)\(([^)]+)\)", low)
        if not m:
            continue
        kind = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) < 3:
            continue
        ra = _to_deg_ra(args[0])
        dec = _to_deg_dec(args[1])
        if ra is None or dec is None:
            continue
        radii: list[float] = []
        for tok in args[2:]:
            tok = tok.strip()
            try:
                if tok.endswith('"'):
                    radii.append(float(tok[:-1]))
                elif tok.endswith("'"):
                    radii.append(float(tok[:-1]) * 60.0)
                else:
                    radii.append(float(tok))
            except ValueError:
                continue
        regions.append({"kind": kind, "ra": ra, "dec": dec,
                        "radii_arcsec": radii, "system": coord_sys})
    return regions


def _validate_bck_archive(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing session.bck"
    size = path.stat().st_size
    if size < 4096:
        return False, f"size={size} bytes (<4 KB)"
    try:
        with tarfile.open(path, mode="r:*") as tf:
            names = tf.getnames()
            if not names:
                return False, "tar archive is empty"
            tcl_names = [n for n in names
                         if n.endswith(".bck") or n.endswith(".tcl") or "ds9" in n.lower()]
            if not tcl_names:
                return False, f"no DS9 .bck/.tcl members; first 5: {names[:5]}"
            tcl_text = ""
            for nm in tcl_names:
                try:
                    member = tf.extractfile(nm)
                except KeyError:
                    continue
                if member is None:
                    continue
                try:
                    raw = member.read()
                except Exception:
                    continue
                if len(raw) > 4 * 1024 * 1024:
                    raw = raw[: 4 * 1024 * 1024]
                tcl_text += raw.decode("utf-8", errors="replace") + "\n"
            if not tcl_text.strip():
                return False, f"DS9 members readable but empty; members: {tcl_names[:5]}"
            low = tcl_text.lower()
            markers = ["frame", "scale", "cmap", "ds9", "wcs", "regions", "colorbar"]
            present = sorted({m for m in markers if m in low})
            if len(present) < 3:
                return False, f"Tcl lacks DS9 markers; present={present}"
            return True, f"valid DS9 backup; entries={len(names)}, markers={present}"
    except tarfile.ReadError as e:
        return False, f"not a tar archive: {e}"
    except OSError as e:
        return False, f"archive read error: {e}"


def _wait_xpa(title: str, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _xpaget(title, "version", timeout=2.0) is not None:
            return True
        time.sleep(1.0)
    return False


def _kill_ds9(title: str, proc: subprocess.Popen | None) -> None:
    try:
        _xpaset_p(title, "exit", timeout=5.0)
    except Exception:
        pass
    if proc is None:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _round_trip_restore(
    bck_path: Path, gold_ra: float, gold_dec: float
) -> tuple[bool | None, str]:
    """
    Returns (True, detail) on success, (False, detail) on failure,
    (None, reason) when the check must be skipped (no DS9 available).
    """
    if not bck_path.exists():
        return False, "session.bck missing"
    ds9_bin = shutil.which("ds9") or shutil.which("ds9.exe")
    if ds9_bin is None:
        return None, "ds9 binary not on PATH (round-trip skipped)"
    if shutil.which("xpaset") is None or shutil.which("xpaget") is None:
        return None, "xpaset/xpaget not on PATH (round-trip skipped)"

    _xpaset_p(ROUND_TRIP_TITLE, "exit", timeout=5.0)
    time.sleep(0.5)

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            [ds9_bin, "-title", ROUND_TRIP_TITLE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        return None, f"failed to launch ds9: {e} (round-trip skipped)"

    try:
        if not _wait_xpa(ROUND_TRIP_TITLE, timeout_s=30.0):
            return None, "fresh DS9 did not register XPA within 30s (no display? skipped)"

        ok, msg = _xpaset_p(ROUND_TRIP_TITLE, "restore",
                             str(bck_path.resolve()), timeout=60.0)
        if not ok:
            return False, f"DS9 restore rejected .bck: {msg}"

        time.sleep(2.0)

        scale = (_xpaget(ROUND_TRIP_TITLE, "scale") or "").strip().lower()
        cmap  = (_xpaget(ROUND_TRIP_TITLE, "cmap")  or "").strip().lower()
        reg_text = _xpaget(ROUND_TRIP_TITLE, "regions") or ""
        parsed = _parse_reg_text(reg_text)

        scale_ok  = "log" in scale
        cmap_ok   = bool(cmap) and cmap not in {"grey", "gray", ""}
        circle_ok = any(
            r["kind"] == "circle"
            and _angular_sep_arcsec(gold_ra, gold_dec, r["ra"], r["dec"]) <= 2.0
            for r in parsed
        )
        annulus_ok = any(
            r["kind"] == "annulus"
            and _angular_sep_arcsec(gold_ra, gold_dec, r["ra"], r["dec"]) <= 2.0
            for r in parsed
        )
        all_ok = scale_ok and cmap_ok and circle_ok and annulus_ok
        detail = (
            f"scale={scale!r}({scale_ok}) cmap={cmap!r}({cmap_ok}) "
            f"circle@target={circle_ok} annulus@target={annulus_ok} "
            f"({len(parsed)} regions)"
        )
        return all_ok, detail
    finally:
        _kill_ds9(ROUND_TRIP_TITLE, proc)


# ---------------------------------------------------------------------------
# Layer checks
# ---------------------------------------------------------------------------

def _grade_xpa(title: str, gold_ra: float, gold_dec: float) -> dict[str, dict]:
    out: dict[str, dict] = {}

    file_str = _xpaget(title, "file")
    out["xpa_file_loaded"] = {
        "ok": bool(file_str and FITS_NAME in file_str),
        "detail": (file_str or "<no DS9 / xpaget unavailable>").strip(),
    }

    scale_str = _xpaget(title, "scale") or ""
    out["xpa_scale_log"] = {
        "ok": "log" in scale_str.lower(),
        "detail": scale_str.strip(),
    }

    cmap_str = (_xpaget(title, "cmap") or "").strip()
    out["xpa_cmap_set"] = {
        "ok": bool(cmap_str) and cmap_str.lower() not in {"grey", "gray"},
        "detail": cmap_str,
    }

    frame_str = (_xpaget(title, "frame", "all") or "").strip()
    out["xpa_frame_loaded"] = {"ok": bool(frame_str), "detail": frame_str}

    reg_text = _xpaget(title, "regions") or ""
    parsed = _parse_reg_text(reg_text)
    circle_ok = any(
        r["kind"] == "circle"
        and _angular_sep_arcsec(gold_ra, gold_dec, r["ra"], r["dec"]) <= 2.0
        for r in parsed
    )
    annulus_ok = any(
        r["kind"] == "annulus"
        and _angular_sep_arcsec(gold_ra, gold_dec, r["ra"], r["dec"]) <= 2.0
        for r in parsed
    )
    out["xpa_circle_near_target"]  = {"ok": circle_ok,  "detail": f"{len(parsed)} regions via xpaget"}
    out["xpa_annulus_near_target"] = {"ok": annulus_ok, "detail": f"{len(parsed)} regions via xpaget"}
    return out


def _grade_artifacts(
    out_dir: Path, gold_ra: float, gold_dec: float, do_roundtrip: bool
) -> dict[str, dict]:
    out: dict[str, dict] = {}

    # --- sources.reg ---
    reg_path = out_dir / "sources.reg"
    if reg_path.exists():
        text = reg_path.read_text(encoding="utf-8", errors="replace")
        parsed = _parse_reg_text(text)
        has_circle  = any(r["kind"] == "circle"
                          and _angular_sep_arcsec(gold_ra, gold_dec, r["ra"], r["dec"]) <= 2.0
                          for r in parsed)
        has_annulus = any(r["kind"] == "annulus"
                          and _angular_sep_arcsec(gold_ra, gold_dec, r["ra"], r["dec"]) <= 2.0
                          for r in parsed)
        out["artifact_reg_valid"] = {
            "ok": has_circle and has_annulus,
            "detail": f"{len(parsed)} regions, circle@target={has_circle}, annulus@target={has_annulus}",
        }
    else:
        out["artifact_reg_valid"] = {"ok": False, "detail": "missing sources.reg"}

    # --- session.bck ---
    bck_path = out_dir / "session.bck"
    bck_ok, bck_detail = _validate_bck_archive(bck_path)
    out["artifact_bck_archive"] = {
        "ok": bck_ok,
        "detail": bck_detail,
    }

    # --- round-trip ---
    if do_roundtrip and bck_ok:
        rt_ok, rt_detail = _round_trip_restore(bck_path, gold_ra, gold_dec)
        if rt_ok is None:
            out["artifact_bck_roundtrip"] = {
                "ok": False, "detail": f"SKIPPED: {rt_detail}", "skipped": True
            }
        else:
            out["artifact_bck_roundtrip"] = {"ok": rt_ok, "detail": rt_detail}
    elif do_roundtrip:
        out["artifact_bck_roundtrip"] = {
            "ok": False, "detail": ".bck archive invalid; round-trip not attempted"
        }
    else:
        out["artifact_bck_roundtrip"] = {
            "ok": False, "detail": "round-trip disabled", "skipped": True
        }

    # --- ds9_view.png ---
    png_path = out_dir / "ds9_view.png"
    if png_path.exists():
        head = png_path.read_bytes()[:8]
        out["artifact_png_valid"] = {
            "ok": head == b"\x89PNG\r\n\x1a\n",
            "detail": f"size={png_path.stat().st_size}",
        }
    else:
        out["artifact_png_valid"] = {"ok": False, "detail": "missing ds9_view.png"}

    # --- schema: photometry.csv + results.json ---
    schema_ok = True
    schema_msgs: list[str] = []

    csv_path = out_dir / "photometry.csv"
    needed_csv = {"source_id", "ra_deg", "dec_deg", "aperture_radius_arcsec",
                  "gross_counts", "bkg_counts_per_pix", "net_counts"}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
        missing = needed_csv - cols
        if missing:
            schema_ok = False
            schema_msgs.append(f"photometry.csv missing cols: {sorted(missing)}")
    else:
        schema_ok = False
        schema_msgs.append("missing photometry.csv")

    res_path = out_dir / "results.json"
    needed_json = {"obsid", "target", "ds9_xpa_title", "source_ra_deg", "source_dec_deg",
                   "aperture_radius_arcsec", "background_annulus_arcsec",
                   "net_counts", "scale_mode", "cmap"}
    if res_path.exists():
        try:
            res = json.loads(res_path.read_text(encoding="utf-8"))
            missing = needed_json - set(res)
            if missing:
                schema_ok = False
                schema_msgs.append(f"results.json missing keys: {sorted(missing)}")
        except json.JSONDecodeError as e:
            schema_ok = False
            schema_msgs.append(f"results.json invalid JSON: {e}")
    else:
        schema_ok = False
        schema_msgs.append("missing results.json")

    out["artifact_output_schema"] = {
        "ok": schema_ok,
        "detail": "; ".join(schema_msgs) or "OK",
    }
    return out


def _grade_numeric(out_dir: Path, gold: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    res_path = out_dir / "results.json"
    if not res_path.exists():
        out["numeric_position"]   = {"ok": False, "detail": "missing results.json"}
        out["numeric_net_counts"] = {"ok": False, "detail": "missing results.json"}
        return out

    try:
        res = json.loads(res_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        out["numeric_position"]   = {"ok": False, "detail": f"invalid JSON: {e}"}
        out["numeric_net_counts"] = {"ok": False, "detail": f"invalid JSON: {e}"}
        return out

    gv = gold["gold_values"]
    gold_ra   = gv["source_ra_deg"]["value"]
    gold_dec  = gv["source_dec_deg"]["value"]
    gold_net  = gv["net_counts"]["value"]
    tol_as    = gv["source_ra_deg"]["tolerance_arcsec"]   # 2.0
    tol_pct   = gv["net_counts"]["tolerance_pct"]          # 15.0

    try:
        sep = _angular_sep_arcsec(gold_ra, gold_dec,
                                   float(res["source_ra_deg"]),
                                   float(res["source_dec_deg"]))
        out["numeric_position"] = {
            "ok": sep <= tol_as,
            "detail": f"sep={sep:.3f}\" tol={tol_as}\"",
        }
    except (KeyError, TypeError, ValueError) as e:
        out["numeric_position"] = {"ok": False, "detail": str(e)}

    try:
        rep_net = float(res["net_counts"])
        rel = abs(rep_net - gold_net) / max(abs(gold_net), 1e-9) * 100.0
        out["numeric_net_counts"] = {
            "ok": rel <= tol_pct,
            "detail": f"reported={rep_net:.1f} gold={gold_net:.1f} rel={rel:.2f}% tol={tol_pct}%",
        }
    except (KeyError, TypeError, ValueError) as e:
        out["numeric_net_counts"] = {"ok": False, "detail": str(e)}

    return out


async def _vlm_png_check(session: cb.DesktopSession, png_remote_path: str) -> float:
    """
    Use LLM vision to verify the PNG looks like a real DS9 screenshot.
    Returns 1.0 if YES, 0.0 otherwise.  Errors → 0.0 (non-fatal).
    """
    try:
        img_bytes = await session.read_bytes(png_remote_path)
    except Exception as e:
        logger.warning("[ds9_g21] VLM PNG check: could not read %s: %s", png_remote_path, e)
        return 0.0
    prompt = (
        "Does this screenshot show SAOImage DS9 with an X-ray image loaded "
        "and a region overlay (circle or annulus) visible on the image? "
        "Answer YES or NO."
    )
    try:
        result = await llm_vision_judge(
            prompt=prompt,
            image_bytes=img_bytes,
            return_binary_score=True,
        )
        score = float(result) if isinstance(result, (int, float)) else 0.0
        logger.info("[ds9_g21] VLM PNG score=%.1f", score)
        return score
    except Exception as e:
        logger.warning("[ds9_g21] VLM PNG check failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Main evaluate()
# ---------------------------------------------------------------------------

@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """
    Full evaluation.  Returns list[float] with one element in [0, 1].
    """
    meta       = task_cfg.metadata
    remote_out = meta["remote_output_dir"]   # e.g. D:\tasks\...\output
    ref_dir    = meta["reference_dir"]

    logger.info("[ds9_g21] evaluate() begin")

    # ------------------------------------------------------------------
    # Load gold contract from the remote machine
    # ------------------------------------------------------------------
    gold_remote = meta["gold_path"]
    try:
        gold_text = await session.read_file(gold_remote)
        gold = json.loads(gold_text) if isinstance(gold_text, str) else json.loads(gold_text.decode())
    except Exception as e:
        logger.error("[ds9_g21] Cannot read gold contract: %s", e)
        return [0.0]

    gv       = gold["gold_values"]
    gold_ra  = gv["source_ra_deg"]["value"]
    gold_dec = gv["source_dec_deg"]["value"]

    # ------------------------------------------------------------------
    # Copy output files to a local temp dir so grader helpers can work
    # on local Path objects (needed for tarfile, PNG header checks, etc.)
    # ------------------------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ds9_grade_") as tmpdir:
        local_out = Path(tmpdir) / "output"
        local_out.mkdir()

        output_files = [
            "sources.reg", "session.bck", "ds9_view.png",
            "photometry.csv", "results.json",
        ]
        for fname in output_files:
            remote_fpath = os.path.join(remote_out, fname)
            local_fpath  = local_out / fname
            try:
                raw = await session.read_bytes(remote_fpath)
                local_fpath.write_bytes(raw)
                logger.info("[ds9_g21] fetched %s (%d bytes)", fname, len(raw))
            except Exception as e:
                logger.warning("[ds9_g21] could not fetch %s: %s", fname, e)

        # ------------------------------------------------------------------
        # Run grader's local helper logic
        # ------------------------------------------------------------------
        results: dict[str, dict] = {}

        # --- XPA layer (runs on grader host) ---
        if shutil.which("xpaget") is None:
            logger.warning("[ds9_g21] xpaget not on grader PATH — XPA layer skipped")
            for k in ("xpa_file_loaded", "xpa_scale_log", "xpa_cmap_set",
                      "xpa_frame_loaded", "xpa_circle_near_target", "xpa_annulus_near_target"):
                results[k] = {"ok": False, "detail": "xpaget not on grader PATH", "skipped": True}
            xpa_available = False
        else:
            results.update(_grade_xpa(DEFAULT_XPA_TITLE, gold_ra, gold_dec))
            xpa_available = True

        # --- Artifact layer ---
        results.update(_grade_artifacts(local_out, gold_ra, gold_dec, do_roundtrip=True))

        # --- Numeric layer ---
        results.update(_grade_numeric(local_out, gold))

        # ------------------------------------------------------------------
        # Hard gate G1: live DS9 must be responding through XPA.
        # ------------------------------------------------------------------
        live_ds9 = results["xpa_file_loaded"]["ok"]
        if not xpa_available or not live_ds9:
            logger.warning("[ds9_g21] HARD GATE G1 failed: XPA unavailable or no live DS9 process")
            _save_report(session, remote_out, results, 0.0, 0.0,
                         passed=False, reason="xpa_unavailable_or_no_live_ds9",
                         skipped=[k for k in results if results[k].get("skipped")])
            return [0.0]

        # ------------------------------------------------------------------
        # Hard gate G2: saved DS9 session backup must round-trip successfully.
        # ------------------------------------------------------------------
        rt_rec = results["artifact_bck_roundtrip"]
        rt_skipped = rt_rec.get("skipped", False)
        rt_ok      = rt_rec["ok"]
        if rt_skipped or not rt_ok:
            logger.warning("[ds9_g21] HARD GATE G2 failed: round-trip restore failed")
            _save_report(session, remote_out, results, 0.0, 0.0,
                         passed=False, reason="session_bck_roundtrip_failed_or_skipped",
                         skipped=[k for k in results if results[k].get("skipped")])
            return [0.0]

        # ------------------------------------------------------------------
        # VLM PNG check (overrides the simple header check's weight if useful)
        # We award the png weight from VLM if png is present and header is valid.
        # ------------------------------------------------------------------
        png_remote = os.path.join(remote_out, "ds9_view.png")
        if results["artifact_png_valid"]["ok"]:
            vlm_score = await _vlm_png_check(session, png_remote)
            # Use VLM score in place of simple binary; still 0.05 weight
            results["artifact_png_valid"]["vlm_score"] = vlm_score
            results["artifact_png_valid"]["ok"] = vlm_score >= 0.5

        # ------------------------------------------------------------------
        # Compute raw score
        # ------------------------------------------------------------------
        breakdown: dict[str, float] = {}
        raw_score = 0.0
        skipped: list[str] = []
        for k, w in WEIGHTS.items():
            rec = results.get(k, {})
            pts = w if rec.get("ok", False) else 0.0
            breakdown[k] = round(pts, 4)
            raw_score += pts
            if rec.get("skipped"):
                skipped.append(k)

        raw_score = round(raw_score, 4)
        final_score = raw_score if raw_score >= PASS_THRESHOLD else 0.0
        passed = final_score >= PASS_THRESHOLD

        reason = ""
        if final_score == 0.0 and raw_score > 0.0:
            reason = f"below_threshold:{raw_score:.4f}"

        logger.info(
            "[ds9_g21] raw=%.4f final=%.4f passed=%s reason=%r",
            raw_score, final_score, passed, reason,
        )

        _save_report(session, remote_out, results, raw_score, final_score,
                     passed=passed, reason=reason, skipped=skipped,
                     breakdown=breakdown)

        return [final_score]


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

def _save_report(
    session: Any,
    remote_out: str,
    results: dict[str, dict],
    raw_score: float,
    final_score: float,
    *,
    passed: bool,
    reason: str,
    skipped: list[str],
    breakdown: dict[str, float] | None = None,
) -> None:
    notes_parts = [f"{k}: {v.get('detail', '')}" for k, v in results.items()]
    report: dict[str, Any] = {
        "score":          final_score,
        "raw_score":      raw_score,
        "pass_threshold": PASS_THRESHOLD,
        "passed":         passed,
        "reason":         reason,
        "breakdown":      breakdown or {},
        "notes":          " | ".join(notes_parts),
        "skipped":        skipped,
    }
    report_json = json.dumps(report, indent=2)
    logger.info("[ds9_g21] grade report:\n%s", report_json)

    async def _write():
        try:
            await session.write_file(
                os.path.join(remote_out, "autograde_report.json"),
                report_json,
            )
        except Exception as e:
            logger.warning("[ds9_g21] could not write autograde_report.json: %s", e)

    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_write())
        else:
            loop.run_until_complete(_write())
    except Exception:
        pass
