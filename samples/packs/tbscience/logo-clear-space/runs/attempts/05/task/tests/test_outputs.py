# CANARY: apex-logo-clear-space-v002-verifier-canary-4c1f9e73
"""Semantic verifier for media/logo-clear-space v002 (Apex lockup clear space).

The suite reads only the agent's declared artifacts (`/app/out/primary.svg`,
`/app/out/stacked.svg`, `/app/out/icon.svg`, `/app/out/margins.json`) plus the
verifier's own reference data under `/tests/`.  It never touches the task
environment's inputs.

Reference values are sourced two ways, both independent of the agent:

* `/tests/sources/*.svg` are the shipped round-two lockups (verbatim bytes from
  the deterministic build generator).  They pin the artwork, the group
  structure, the per-path source ink boxes and the true drawn-ink union boxes.
* `/tests/reference.json` carries the BrandSans Display OS/2 metrics
  (`unitsPerEm` 1000, `sxHeight` 520, `sCapHeight` 700), the wordmark type sizes
  (100 primary / 140 stacked) and the section 3.3 multipliers (1.20 / 0.75 /
  0.60).  The suite recomputes every expected margin from those numbers and from
  the source paths' true cubic-extremum ink height.

Tests, one per requirement group:

* test_artifacts_present_and_parse - the four artifacts exist, the SVGs parse as
  SVG and contain no unexpected drawing elements, margins.json is a single
  object with exactly the keys primary/stacked/icon and numeric, finite margins
  plus string references, and no key is duplicated anywhere.
* test_reference_values_reconcile - the vendored font metrics, type sizes and
  multipliers reproduce the recorded gold margins and the mark's 114-unit ink
  height; this pins the verifier's own arithmetic before it judges the agent.
* test_margins_match_standard_references - each reported margin equals the
  standard's factor times the named reference length within 0.5 user units
  (R2/R3), and differs beyond tolerance from the surrogate lengths the handoff
  suggests (nominal type size, set-word ink height, mark path-coordinate hull),
  except where a surrogate coincides by construction (stacked all-caps ink).
* test_reference_labels_normalize - each `reference` string normalizes to
  x-height, cap-height or optical-bbox (R3).
* test_svg_ink_placement_matches_standard - the union of drawn ink (cubic
  extrema through composed group transforms) begins at (m, m), measures (W, H),
  and the viewBox is `0 0 (W+2m) (H+2m)` within 0.5 user units (R1).
* test_reported_margins_agree_with_svgs - the margin in margins.json agrees
  with the margin implied by each SVG's ink origin and viewBox (R3).
* test_artwork_preserved_and_uniformly_translated - path-data signatures and
  path count/order match the shipped lockups, and every path's effective ink
  box equals its source box plus one translation shared by all paths in the
  file (R4).
"""

from __future__ import annotations

import functools
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

OUT_DIR = Path("/app/out")
TESTS_DIR = Path("/tests")
SOURCE_DIR = TESTS_DIR / "sources"

LOCKUPS = ("primary", "stacked", "icon")
ALLOWED_TAGS = {"svg", "title", "desc", "g", "path"}

REFERENCE = json.loads((TESTS_DIR / "reference.json").read_text(encoding="utf-8"))
TOL = float(REFERENCE["tolerance"])
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
TOKEN_RE = re.compile(r"[A-Za-z]|" + NUM_RE.pattern)
TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")


def local_name(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


# --------------------------------------------------------------------------- #
# vendored geometry: exact path ink bounds (cubic extrema) and SVG transforms #
# --------------------------------------------------------------------------- #
def _tokenize_path(data: str) -> list[str]:
    return TOKEN_RE.findall(data or "")


def _numbers(text: str) -> list[float]:
    return [float(tok) for tok in NUM_RE.findall(text or "")]


def _cubic_axis(v0: float, v1: float, v2: float, v3: float) -> list[float]:
    """Endpoint values of one cubic plus the values at its interior extrema."""
    a = -v0 + 3.0 * v1 - 3.0 * v2 + v3
    b = 2.0 * (v0 - 2.0 * v1 + v2)
    c = v1 - v0
    values = [v0, v3]
    roots: list[float] = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            roots.append(-c / b)
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            root = math.sqrt(disc)
            roots.append((-b + root) / (2.0 * a))
            roots.append((-b - root) / (2.0 * a))
    for t in roots:
        if 0.0 < t < 1.0:
            mt = 1.0 - t
            values.append(
                mt ** 3 * v0 + 3.0 * mt * mt * t * v1 + 3.0 * mt * t * t * v2 + t ** 3 * v3
            )
    return values


def parse_path(data: str) -> list[tuple]:
    """Parse the SVG path subset used by the lockups into line/cubic segments."""
    tokens = _tokenize_path(data)
    index = 0
    segments: list[tuple] = []
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    command = None

    def take() -> float:
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        if tokens[index][:1].isalpha():
            command = tokens[index]
            index += 1
        if command in ("M", "m"):
            x, y = take(), take()
            if command == "m":
                x += current[0]
                y += current[1]
            current = (x, y)
            start = current
            command = "L" if command == "M" else "l"
        elif command in ("L", "l"):
            x, y = take(), take()
            if command == "l":
                x += current[0]
                y += current[1]
            segments.append(("L", current, (x, y)))
            current = (x, y)
        elif command in ("C", "c"):
            x1, y1, x2, y2, x, y = (take() for _ in range(6))
            if command == "c":
                x1 += current[0]
                y1 += current[1]
                x2 += current[0]
                y2 += current[1]
                x += current[0]
                y += current[1]
            segments.append(("C", current, (x1, y1), (x2, y2), (x, y)))
            current = (x, y)
        elif command in ("H", "h"):
            x = take()
            if command == "h":
                x += current[0]
            segments.append(("L", current, (x, current[1])))
            current = (x, current[1])
        elif command in ("V", "v"):
            y = take()
            if command == "v":
                y += current[1]
            segments.append(("L", current, (current[0], y)))
            current = (current[0], y)
        elif command in ("Z", "z"):
            segments.append(("L", current, start))
            current = start
        else:
            raise ValueError("unsupported path command %r" % (command,))
    return segments


def segment_bbox(segments: list[tuple]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for segment in segments:
        if segment[0] == "L":
            for point in (segment[1], segment[2]):
                xs.append(point[0])
                ys.append(point[1])
        else:
            xs.extend(_cubic_axis(segment[1][0], segment[2][0], segment[3][0], segment[4][0]))
            ys.extend(_cubic_axis(segment[1][1], segment[2][1], segment[3][1], segment[4][1]))
    return (min(xs), min(ys), max(xs), max(ys))


def mat_mul(outer: tuple, inner: tuple) -> tuple:
    """Compose two affine matrices; `inner` is applied to a point first."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def mat_apply(matrix: tuple, point: tuple) -> tuple:
    a, b, c, d, e, f = matrix
    return (a * point[0] + c * point[1] + e, b * point[0] + d * point[1] + f)


def parse_transform(text: str | None) -> tuple:
    matrix = IDENTITY
    if not text:
        return matrix
    for name, args in TRANSFORM_RE.findall(text):
        values = _numbers(args)
        if name == "translate":
            local = (1.0, 0.0, 0.0, 1.0, values[0], values[1] if len(values) > 1 else 0.0)
        elif name == "scale":
            local = (values[0], 0.0, 0.0, values[1] if len(values) > 1 else values[0], 0.0, 0.0)
        elif name == "matrix":
            local = tuple(values[:6])
        else:
            raise ValueError("unsupported transform %r" % (name,))
        matrix = mat_mul(matrix, local)
    return matrix


def transform_bbox(segments: list[tuple], matrix: tuple) -> tuple[float, float, float, float]:
    moved = []
    for segment in segments:
        moved.append((segment[0],) + tuple(mat_apply(matrix, p) for p in segment[1:]))
    return segment_bbox(moved)


def normalize_signature(data: str) -> tuple:
    signature = []
    for token in _tokenize_path(data):
        if token[:1].isalpha():
            signature.append(token)
        else:
            signature.append(round(float(token), 6))
    return tuple(signature)


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(got: float, expected: float, tol: float = TOL) -> bool:
    return is_finite_number(got) and abs(got - expected) <= tol + 1e-9


# --------------------------------------------------------------------------- #
# artifact / reference loading                                               #
# --------------------------------------------------------------------------- #
def parse_svg_text(text: str, label: str) -> dict:
    """Parse one SVG and collect its composed-geometry view of the artwork."""
    root = ET.fromstring(text)
    if local_name(root.tag) != "svg":
        pytest.fail("%s: root element is %r, expected an SVG <svg>" % (label, root.tag))

    bad_tags = sorted({local_name(el.tag) for el in root.iter() if local_name(el.tag) not in ALLOWED_TAGS})
    if bad_tags:
        pytest.fail("%s: unexpected drawing elements present: %s" % (label, bad_tags))

    group_ids: list[str] = []
    paths: list[tuple[str, tuple, str | None]] = []

    def walk(element, matrix, group_id):
        if local_name(element.tag) == "g":
            if element.get("id"):
                group_ids.append(element.get("id"))
                group_id = element.get("id")
            matrix = mat_mul(matrix, parse_transform(element.get("transform")))
        if local_name(element.tag) == "path":
            paths.append((element.get("d") or "", matrix, group_id))
        for child in element:
            walk(child, matrix, group_id)

    walk(root, IDENTITY, None)

    boxes = []
    signatures = []
    for data, matrix, _group in paths:
        boxes.append(transform_bbox(parse_path(data), matrix))
        signatures.append(normalize_signature(data))

    union = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    return {
        "root": root,
        "group_ids": group_ids,
        "paths": paths,
        "boxes": boxes,
        "signatures": signatures,
        "union": union,
        "view_box": root.get("viewBox"),
    }


@functools.lru_cache(maxsize=None)
def source_geometry() -> dict:
    geometry = {}
    for lockup in LOCKUPS:
        path = SOURCE_DIR / ("%s.svg" % lockup)
        geometry[lockup] = parse_svg_text(path.read_text(encoding="utf-8"), "source %s" % lockup)
    return geometry


def read_required(path: Path) -> str:
    if not path.is_file():
        pytest.fail("missing declared artifact: %s" % path)
    return path.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=None)
def output_svg(lockup: str) -> dict:
    text = read_required(OUT_DIR / ("%s.svg" % lockup))
    try:
        return parse_svg_text(text, "output %s" % lockup)
    except pytest.fail.Exception:
        raise
    except Exception as exc:  # malformed XML / unsupported geometry
        pytest.fail("output %s.svg is not parseable SVG geometry: %s" % (lockup, exc))


class DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs):
    seen = set()
    result = {}
    for key, value in pairs:
        if key in seen:
            raise DuplicateKeyError(key)
        seen.add(key)
        result[key] = value
    return result


@functools.lru_cache(maxsize=None)
def reported_margins() -> dict:
    text = read_required(OUT_DIR / "margins.json")
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except DuplicateKeyError as exc:
        pytest.fail("margins.json repeats the key %r" % (exc.args[0],))
    except json.JSONDecodeError as exc:
        pytest.fail("margins.json is not valid JSON: %s" % exc)


# --------------------------------------------------------------------------- #
# reference arithmetic                                                       #
# --------------------------------------------------------------------------- #
def reference_length(lockup: str) -> float:
    """The named reference length in SVG user units, recomputed from source."""
    font = REFERENCE["font"]
    if lockup == "primary":
        return font["x_height"] * REFERENCE["type_sizes"]["primary"] / font["units_per_em"]
    if lockup == "stacked":
        return font["cap_height"] * REFERENCE["type_sizes"]["stacked"] / font["units_per_em"]
    icon_union = source_geometry()["icon"]["union"]
    return icon_union[3] - icon_union[1]


def expected_margin(lockup: str) -> float:
    return REFERENCE["multipliers"][lockup] * reference_length(lockup)


def _group_ink_height(lockup: str, group_id: str) -> float:
    source = source_geometry()[lockup]
    boxes = [box for box, path in zip(source["boxes"], source["paths"]) if path[2] == group_id]
    if not boxes:
        boxes = list(source["boxes"])
    return max(b[3] for b in boxes) - min(b[1] for b in boxes)


def surrogate_margins(lockup: str) -> dict:
    """Margins an unreliable 'tallest measurable height' reading would give."""
    source = source_geometry()[lockup]
    factor = REFERENCE["multipliers"][lockup]
    surrogates = {}
    type_size = REFERENCE["type_sizes"].get(lockup)
    if type_size is not None:
        surrogates["nominal type size"] = factor * type_size
    if lockup in ("primary", "stacked"):
        surrogates["set-word ink height"] = factor * _group_ink_height(lockup, "wordmark")
    else:
        values = _numbers(source["paths"][0][0])
        ys = values[1::2]
        surrogates["mark path-coordinate hull"] = factor * (max(ys) - min(ys))
    return surrogates


def _normalize_reference(text: str) -> str | None:
    """Normalize a reported reference name to one of the three standard names.

    Accepts the standard's own phrasing plus the obvious synonyms a careful
    solver might use (cap/capital height, optical bounding box/bbox, drawn-ink
    height of the mark) so an honest label is never rejected for wording.
    """
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    if "capheight" in compact or "capitalheight" in compact:
        return "cap-height"
    if "xheight" in compact:
        return "x-height"
    if ("boundingbox" in compact or "bbox" in compact) and (
        "optical" in compact or "mark" in compact
    ):
        return "optical-bbox"
    if "optical" in compact and "box" in compact:
        return "optical-bbox"
    if "inkheight" in compact or "drawnheight" in compact:
        return "optical-bbox"
    return None


def _viewbox_numbers(text: str | None, label: str) -> list[float]:
    values = _numbers(text)
    assert len(values) == 4, "%s viewBox must hold four numbers, got %r" % (label, text)
    assert all(math.isfinite(v) for v in values), "%s viewBox holds a non-finite number" % label
    return values


# --------------------------------------------------------------------------- #
# tests                                                                      #
# --------------------------------------------------------------------------- #
def test_artifacts_present_and_parse():
    """All four artifacts exist, parse, and carry the promised top-level shape."""
    for lockup in LOCKUPS:
        out = output_svg(lockup)
        assert len(out["paths"]) >= 1, "%s.svg contains no <path> artwork" % lockup
        assert out["view_box"], "%s.svg has no viewBox attribute" % lockup

    report = reported_margins()
    assert isinstance(report, dict), "margins.json must be a JSON object"
    assert set(report.keys()) == set(LOCKUPS), (
        "margins.json keys must be exactly primary/stacked/icon, got %s" % sorted(report.keys())
    )
    for lockup in LOCKUPS:
        entry = report[lockup]
        assert isinstance(entry, dict), "margins.json[%s] must be an object" % lockup
        assert "margin" in entry, "margins.json[%s] is missing 'margin'" % lockup
        assert "reference" in entry, "margins.json[%s] is missing 'reference'" % lockup
        assert is_finite_number(entry["margin"]), (
            "margins.json[%s].margin must be a finite JSON number" % lockup
        )
        assert isinstance(entry["reference"], str), (
            "margins.json[%s].reference must be a string" % lockup
        )


def test_reference_values_reconcile():
    """The vendored font metrics reproduce the recorded gold margins exactly."""
    for lockup in LOCKUPS:
        assert close(expected_margin(lockup), REFERENCE["expected_margins"][lockup], 1e-6), (
            "verifier reference arithmetic disagrees with the recorded margin for %s" % lockup
        )
    assert close(reference_length("icon"), 114.0, 1e-6), (
        "the mark's true drawn-ink height must be 114 user units (cubic extremum)"
    )


def test_margins_match_standard_references():
    """Each margin equals factor x named reference length, and rejects surrogates."""
    report = reported_margins()
    for lockup in LOCKUPS:
        margin = report[lockup]["margin"]
        want = expected_margin(lockup)
        assert close(margin, want), (
            "%s margin %.4f is not the standard's factor x reference length (%.4f)"
            % (lockup, margin, want)
        )
        for name, surrogate in surrogate_margins(lockup).items():
            if close(surrogate, want):
                continue  # documented coincidence (stacked all-caps ink == cap height)
            assert abs(margin - surrogate) > TOL, (
                "%s margin %.4f matches the surrogate '%s' (%.4f) instead of the "
                "standard's reference length" % (lockup, margin, name, surrogate)
            )


def test_reference_labels_normalize():
    """Each reported reference normalizes to the standard's named reference."""
    report = reported_margins()
    for lockup in LOCKUPS:
        text = report[lockup]["reference"]
        assert isinstance(text, str), "margins.json[%s].reference must be text" % lockup
        got = _normalize_reference(text)
        want = REFERENCE["expected_references"][lockup]
        assert got == want, (
            "%s reference %r normalizes to %r, expected %r" % (lockup, text, got, want)
        )


def test_svg_ink_placement_matches_standard():
    """Drawn ink starts at (m, m) and the viewBox is 0 0 (W+2m) (H+2m)."""
    for lockup in LOCKUPS:
        out = output_svg(lockup)
        source = source_geometry()[lockup]
        margin = expected_margin(lockup)
        min_x, min_y, max_x, max_y = out["union"]
        width = source["union"][2] - source["union"][0]
        height = source["union"][3] - source["union"][1]
        assert all(math.isfinite(v) for v in out["union"]), "%s.svg ink box is non-finite" % lockup

        assert close(min_x, margin) and close(min_y, margin), (
            "%s.svg drawn ink begins at (%.4f, %.4f); the clear space requires (%.4f, %.4f)"
            % (lockup, min_x, min_y, margin, margin)
        )
        assert close(max_x - min_x, width) and close(max_y - min_y, height), (
            "%s.svg drawn ink measures %.4f x %.4f, expected %.4f x %.4f"
            % (lockup, max_x - min_x, max_y - min_y, width, height)
        )

        vb = _viewbox_numbers(out["view_box"], "%s.svg" % lockup)
        assert close(vb[0], 0.0) and close(vb[1], 0.0), (
            "%s.svg viewBox origin must be 0 0, got %s" % (lockup, out["view_box"])
        )
        assert close(vb[2], width + 2.0 * margin), (
            "%s.svg viewBox width %.4f, expected %.4f" % (lockup, vb[2], width + 2.0 * margin)
        )
        assert close(vb[3], height + 2.0 * margin), (
            "%s.svg viewBox height %.4f, expected %.4f" % (lockup, vb[3], height + 2.0 * margin)
        )


def test_reported_margins_agree_with_svgs():
    """The JSON margin agrees with the margin implied by the matching SVG."""
    report = reported_margins()
    for lockup in LOCKUPS:
        out = output_svg(lockup)
        source = source_geometry()[lockup]
        margin = report[lockup]["margin"]
        width = source["union"][2] - source["union"][0]
        height = source["union"][3] - source["union"][1]
        min_x, min_y = out["union"][0], out["union"][1]

        assert close(min_x, margin) and close(min_y, margin), (
            "%s.svg ink origin (%.4f, %.4f) disagrees with the reported margin %.4f"
            % (lockup, min_x, min_y, margin)
        )
        vb = _viewbox_numbers(out["view_box"], "%s.svg" % lockup)
        assert close((vb[2] - width) / 2.0, margin), (
            "%s.svg viewBox width implies margin %.4f, reported %.4f"
            % (lockup, (vb[2] - width) / 2.0, margin)
        )
        assert close((vb[3] - height) / 2.0, margin), (
            "%s.svg viewBox height implies margin %.4f, reported %.4f"
            % (lockup, (vb[3] - height) / 2.0, margin)
        )


def test_artwork_preserved_and_uniformly_translated():
    """Path data and order are untouched; the drawing moves by one translation (R4)."""
    for lockup in LOCKUPS:
        out = output_svg(lockup)
        source = source_geometry()[lockup]
        margin = expected_margin(lockup)

        assert len(out["paths"]) == len(source["paths"]), (
            "%s.svg has %d paths, the shipped lockup has %d"
            % (lockup, len(out["paths"]), len(source["paths"]))
        )
        assert out["signatures"] == source["signatures"], (
            "%s.svg path data differs from the shipped lockup (paths must carry identical "
            "geometry in the same order)" % lockup
        )
        offsets = []
        for (out_data, out_matrix, _g), (src_data, src_matrix, _sg), src_box, out_box in zip(
            out["paths"], source["paths"], source["boxes"], out["boxes"]
        ):
            assert close(out_matrix[0], 1.0, 1e-6) and close(out_matrix[1], 0.0, 1e-6), (
                "%s.svg applies a non-translation transform to the artwork" % lockup
            )
            assert close(out_matrix[2], 0.0, 1e-6) and close(out_matrix[3], 1.0, 1e-6), (
                "%s.svg applies a non-translation transform to the artwork" % lockup
            )
            offsets.append((out_matrix[4] - src_matrix[4], out_matrix[5] - src_matrix[5]))
            for axis, (got, src_value) in enumerate(zip(out_box, src_box)):
                assert close(got, src_value + offsets[-1][axis % 2]), (
                    "%s.svg path %r ink box %s is not the shipped box %s plus the shared "
                    "translation" % (lockup, out_data[:16], out_box, src_box)
                )

        assert all(
            close(dx, offsets[0][0], 1e-6) and close(dy, offsets[0][1], 1e-6)
            for dx, dy in offsets
        ), "%s.svg does not apply one translation shared by all paths" % lockup

        src_min_x, src_min_y = source["union"][0], source["union"][1]
        assert close(offsets[0][0], margin - src_min_x), (
            "%s.svg horizontal translation %.4f does not place the ink at m=%.4f"
            % (lockup, offsets[0][0], margin)
        )
        assert close(offsets[0][1], margin - src_min_y), (
            "%s.svg vertical translation %.4f does not place the ink at m=%.4f"
            % (lockup, offsets[0][1], margin)
        )
