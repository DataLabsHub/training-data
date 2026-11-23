#!/usr/bin/env python3
"""Build-time generator for the Apex round-two asset set.

Everything below is literal, deterministic geometry: no randomness and no
timestamps.  Running it writes

    <root>/fonts/BrandSans-Display.ttf
    <root>/lockups/primary.svg
    <root>/lockups/stacked.svg
    <root>/lockups/icon.svg
    <root>/guidelines/brand-clear-space.md
    <root>/handoff/round2-handoff.md
    <root>/handoff/measure.py

where <root> is /app unless another root is passed on the command line.
"""

import math
import os
import sys

# --------------------------------------------------------------- constants --
UPEM = 1000
X_HEIGHT = 520
CAP_HEIGHT = 700
ASCENDER = 740

# 2026-01-01T00:00:00Z as a TrueType (1904-based) timestamp.
EPOCH_2026 = 3850070400

MARK_D = "M 0,100 L 32,22 C 42,-26 58,-26 68,22 L 100,100 Z"

# Shipped (round-two) canvases: artwork box grown by the supplier's margin.
SHIPPED_MARGIN = {"primary": 88.8, "stacked": 73.5, "icon": 75.6}
SHIPPED_BOX = {
    "primary": (0.0, -14.0, 397.0, 100.0),
    "stacked": (-146.0, -14.0, 246.0, 258.0),
    "icon": (0.0, -14.0, 100.0, 100.0),
}


# ---------------------------------------------------------------- geometry --
def fmt(value):
    """Compact, deterministic number formatting for generated files."""
    text = "%.4f" % value
    text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text


def poly(*points):
    return [tuple(float(c) for c in p) for p in points]


def rect(x0, y0, x1, y1):
    return poly((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def rounded_rect(x0, y0, x1, y1, radius, steps=4):
    """Rectangle with rounded corners; `radius` is one value or (tl,tr,br,bl)."""
    if isinstance(radius, (int, float)):
        radius = (radius, radius, radius, radius)
    rtl, rtr, rbr, rbl = radius
    points = []

    def corner(cx, cy, a0, a1, rad):
        if rad <= 0:
            points.append((cx, cy))
            return
        for i in range(steps + 1):
            a = a0 + (a1 - a0) * i / steps
            points.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))

    corner(x0 + rbl, y0 + rbl, math.pi, 1.5 * math.pi, rbl)
    corner(x1 - rbr, y0 + rbr, 1.5 * math.pi, 2.0 * math.pi, rbr)
    corner(x1 - rtr, y1 - rtr, 0.0, 0.5 * math.pi, rtr)
    corner(x0 + rtl, y1 - rtl, 0.5 * math.pi, math.pi, rtl)
    return points


def ellipse(cx, cy, rx, ry, steps=24):
    return [
        (cx + rx * math.cos(2.0 * math.pi * i / steps),
         cy + ry * math.sin(2.0 * math.pi * i / steps))
        for i in range(steps)
    ]


def signed_area(points):
    total = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def oriented(points, counter_clockwise):
    area = signed_area(points)
    if (area > 0) != counter_clockwise:
        return list(reversed(points))
    return points


def glyph(name, advance, ink_width, contours):
    """`contours` is a list of (points, is_hole) pairs."""
    return {
        "name": name,
        "advance": advance,
        "ink_width": ink_width,
        "contours": contours,
    }


def build_glyphs():
    glyphs = {}

    # ------------------------------------------------------------------ caps --
    glyphs["A"] = glyph("A", 680, 640, [
        (poly((0, 0), (240, CAP_HEIGHT), (400, CAP_HEIGHT), (640, 0)), False),
        (poly((168, 140), (472, 140), (320, 520)), True),
    ])
    glyphs["P"] = glyph("P", 560, 520, [
        (poly((0, 0), (0, CAP_HEIGHT), (520, CAP_HEIGHT), (520, 380),
              (120, 380), (120, 0)), False),
        (rect(120, 500, 400, 600), True),
    ])
    glyphs["E"] = glyph("E", 540, 500, [
        (poly((0, 0), (0, CAP_HEIGHT), (500, CAP_HEIGHT), (500, 580),
              (120, 580), (120, 410), (500, 410), (500, 290), (120, 290),
              (120, 120), (500, 120), (500, 0)), False),
    ])
    glyphs["X"] = glyph("X", 700, 660, [
        (poly((0, 0), (140, 0), (660, CAP_HEIGHT), (520, CAP_HEIGHT)), False),
        (poly((0, CAP_HEIGHT), (140, CAP_HEIGHT), (660, 0), (520, 0)), False),
    ])

    # ------------------------------------------------------------- lowercase --
    glyphs["a"] = glyph("a", 640, 600, [
        (rounded_rect(0, 0, 600, X_HEIGHT, (0, 0, 150, 150)), False),
        (rounded_rect(140, 120, 460, 430, (90, 90, 60, 60)), True),
    ])
    glyphs["p"] = glyph("p", 660, 620, [
        (rounded_rect(0, 0, 620, X_HEIGHT, (0, 0, 150, 150)), False),
        (rounded_rect(140, 120, 500, 430, (90, 90, 60, 60)), True),
        (rect(500, 0, 620, ASCENDER), False),
    ])
    glyphs["e"] = glyph("e", 640, 600, [
        (rounded_rect(0, 0, 600, 530, (120, 120, 120, 120)), False),
        (rounded_rect(140, 300, 460, 440, (70, 70, 70, 70)), True),
        (rounded_rect(140, 90, 460, 230, (70, 70, 70, 70)), True),
    ])
    glyphs["x"] = glyph("x", 640, 600, [
        (poly((0, 0), (120, 0), (600, X_HEIGHT), (480, X_HEIGHT)), False),
        (poly((0, X_HEIGHT), (120, X_HEIGHT), (600, 0), (480, 0)), False),
    ])

    # --------------------------------------------------------------- extras --
    glyphs["bullet"] = glyph("bullet", 280, 280, [
        (ellipse(140, 340, 140, 140), False),
    ])
    glyphs["space"] = glyph("space", 300, 0, [])
    glyphs[".notdef"] = glyph(".notdef", 600, 480, [
        (rect(60, 0, 540, CAP_HEIGHT), False),
        (rect(160, 100, 440, 600), True),
    ])
    return glyphs


# ------------------------------------------------------------------- words --
WORDS = {
    "primary": {
        "glyphs": ["a", "p", "e", "x"],
        "scale": 0.1,
        "tracking": 80.0,
        "type_size": 100,
        "caps": False,
    },
    "stacked": {
        "glyphs": ["A", "P", "E", "X"],
        "scale": 0.14,
        "tracking": 120.0,
        "type_size": 140,
        "caps": True,
    },
}


def set_word(glyphs, spec):
    """Pen positions for a wordmark, in font units."""
    positions = []
    pen = 0.0
    for name in spec["glyphs"]:
        positions.append(pen)
        pen += glyphs[name]["advance"] + spec["tracking"]
    return positions


def word_ink_width(glyphs, spec):
    positions = set_word(glyphs, spec)
    last = positions[-1] + glyphs[spec["glyphs"][-1]]["ink_width"]
    return last  # every glyph starts its ink at x = 0


def contour_to_svg(points, pen_x, scale):
    parts = []
    for i, (x, y) in enumerate(points):
        px = (pen_x + x) * scale
        py = -(y * scale)
        parts.append(("M " if i == 0 else "L ") + fmt(px) + "," + fmt(py))
    return " ".join(parts) + " Z"


def glyph_to_svg(name, glyphs, pen_x, scale):
    """SVG path data for one glyph, baseline on y = 0, y flipped to SVG."""
    chunks = []
    for points, is_hole in glyphs[name]["contours"]:
        ordered = oriented(points, counter_clockwise=not is_hole)
        chunks.append(contour_to_svg(ordered, pen_x, scale))
    return " ".join(chunks)


# ------------------------------------------------------------------- fonts --
def build_font(glyphs, path):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    order = [".notdef", "space", "bullet", "a", "e", "p", "x",
             "A", "E", "P", "X"]
    fb = FontBuilder(UPEM, isTTF=True)
    fb.setupGlyphOrder(order)

    cmap = {32: "space", 0x2022: "bullet", 97: "a", 101: "e", 112: "p",
            120: "x", 65: "A", 69: "E", 80: "P", 88: "X"}
    fb.setupCharacterMap(cmap)

    ttf_glyphs = {}
    for name in order:
        pen = TTGlyphPen(None)
        for points, is_hole in glyphs[name]["contours"]:
            ordered = oriented(points, counter_clockwise=not is_hole)
            pen.moveTo(ordered[0])
            for point in ordered[1:]:
                pen.lineTo(point)
            pen.closePath()
        ttf_glyphs[name] = pen.glyph()
    fb.setupGlyf(ttf_glyphs)

    metrics = {}
    for name in order:
        contours = glyphs[name]["contours"]
        if contours:
            xs = [p[0] for points, _ in contours for p in points]
            lsb = int(round(min(xs)))
        else:
            lsb = 0
        metrics[name] = (int(round(glyphs[name]["advance"])), lsb)
    fb.setupHorizontalMetrics(metrics)

    fb.setupHorizontalHeader(ascent=800, descent=-200, lineGap=0)
    fb.setupNameTable({
        "familyName": "BrandSans Display",
        "styleName": "Regular",
        "uniqueFontIdentifier": "Apex: BrandSans Display Regular",
        "fullName": "BrandSans Display Regular",
        "psName": "BrandSansDisplay-Regular",
        "version": "Version 1.000",
        "copyright": "Licensed to Apex. Display use only.",
        "trademark": "BrandSans is a trademark of the Apex brand group.",
        "designer": "Apex brand group",
        "description": "Geometric display face drawn for the Apex rebrand.",
    })
    fb.setupOS2(
        version=3,
        sTypoAscender=800,
        sTypoDescender=-200,
        sTypoLineGap=0,
        usWinAscent=800,
        usWinDescent=200,
        sxHeight=X_HEIGHT,
        sCapHeight=CAP_HEIGHT,
        achVendID="APEX",
        fsSelection=0x0040,
        usWeightClass=400,
        usWidthClass=5,
    )
    fb.setupPost()
    # Fixed head timestamps so the build is byte-for-byte reproducible.
    fb.updateHead(created=EPOCH_2026, modified=EPOCH_2026)
    fb.save(path)


# -------------------------------------------------------------------- svg ---
SVG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" width="{width}" height="{height}">
  <title>{title}</title>
  <desc>{desc}</desc>
  <g id="mark" transform="translate(0,0)">
    <path d="{mark}"/>
  </g>
{wordmark}</svg>
"""


def svg_document(lockup, glyphs):
    x0, y0, x1, y1 = SHIPPED_BOX[lockup]
    margin = SHIPPED_MARGIN[lockup]
    canvas_x = x0 - margin
    canvas_y = y0 - margin
    canvas_w = (x1 - x0) + 2.0 * margin
    canvas_h = (y1 - y0) + 2.0 * margin
    viewbox = " ".join(fmt(v) for v in (canvas_x, canvas_y, canvas_w, canvas_h))

    if lockup == "icon":
        title = "Apex icon lockup"
        desc = ("Apex icon lockup. Mark artwork only, no wordmark. "
                "Clear space per brand-clear-space.md.")
        wordmark = ""
    else:
        spec = WORDS[lockup]
        positions = set_word(glyphs, spec)
        ink = word_ink_width(glyphs, spec) * spec["scale"]
        if lockup == "primary":
            title = "Apex primary lockup"
            desc = ("Apex primary lockup. Wordmark 'apex' set in BrandSans "
                    "Display, type size 100, lower case, outlined to paths; "
                    "mark at left. Clear space per brand-clear-space.md.")
            wx = 397.0 - ink
            wy = 100.0
        else:
            title = "Apex stacked lockup"
            desc = ("Apex stacked lockup. Wordmark 'APEX' set in BrandSans "
                    "Display, type size 140, all caps, outlined to paths; "
                    "mark above. Clear space per brand-clear-space.md.")
            wx = 50.0 - ink / 2.0
            wy = 258.0

        rows = []
        for name, pen_x in zip(spec["glyphs"], positions):
            data = glyph_to_svg(name, glyphs, pen_x, spec["scale"])
            rows.append('    <path d="%s"/>' % data)
        wordmark = (f'  <g id="wordmark" transform="translate({fmt(wx)},{fmt(wy)})">'
                    + chr(10) + chr(10).join(rows) + chr(10) + '  </g>' + chr(10))

    return SVG_TEMPLATE.format(
        viewbox=viewbox,
        width=fmt(canvas_w),
        height=fmt(canvas_h),
        title=title,
        desc=desc,
        mark=MARK_D,
        wordmark=wordmark,
    )


# ---------------------------------------------------------------- writing ---
GUIDELINES = """# Apex - clear space standard

Brand standards, section 3. Issued for all lockup production.

## 3.1 The clear-space zone

Every Apex lockup is surrounded by a zone of clear space. The zone is part of
the lockup: it travels with the file, it is never trimmed, and nothing - type,
rules, images, page furniture or another mark - is placed inside it.

## 3.2 Reference lengths

A lockup's clear space is expressed as a multiple of a single reference length,
named for that lockup.

| Lockup  | Reference length                              |
|---------|-----------------------------------------------|
| Primary | the wordmark's x-height                       |
| Stacked | the wordmark's cap height                     |
| Icon    | the height of the mark's optical bounding box |

## 3.3 Multipliers

| Lockup  | Clear space         |
|---------|---------------------|
| Primary | 1.20 x the reference |
| Stacked | 0.75 x the reference |
| Icon    | 0.60 x the reference |

## 3.4 Delivering a lockup

A lockup is delivered as one canvas. If the lockup's drawing measures W wide
and H high, the canvas measures (W + 2m) by (H + 2m), where m is the clear
space from 3.3. The origin of the canvas is the top-left corner of the clear
space, so the drawing begins at (m, m) and the file's viewBox reads
`0 0 (W + 2m) (H + 2m)`.

Do not redraw, re-space, re-scale or re-cut a lockup. A lockup is placed.
"""

HANDOFF = """# Round two - clear space pass

Studio production -> Apex brand team
Lockups: `primary.svg`, `stacked.svg`, `icon.svg` in `/app/lockups`

## The pass

Clear space is cut the way it is cut on every pass: take the guideline
multiplier for the lockup from section 3.3 and apply it to the tallest height
measurable on the artwork. That keeps a canvas proportional to something the
artwork itself carries, so a re-export never moves the proportion.

For the two text lockups the measured height is the set word's ink height; for
the icon it is the mark's own path extents. Canvas = artwork box plus the clear
space on all four sides.

| Lockup  | Type size | Measured height | Multiplier | Clear space | Canvas (viewBox)         |
|---------|-----------|-----------------|------------|-------------|--------------------------|
| primary | 100       | 74.0            | 1.20       | 88.8        | -88.8 -102.8 574.6 291.6 |
| stacked | 140       | 98.0            | 0.75       | 73.5        | -219.5 -87.5 539 419     |
| icon    | -         | 126.0           | 0.60       | 75.6        | -75.6 -89.6 251.2 265.2  |

The measured heights are the wordmark ink for `primary` and `stacked`, and the
mark path-coordinate extents for `icon`.

## Checking the numbers

`measure.py` in this folder re-runs the pass straight off the shipped files:

    python3 /app/handoff/measure.py

## Notes

- Mark artwork and wordmarks are outlined; no live type and no embedded fonts.
- The licensed typeface is `BrandSans-Display.ttf` in `/app/fonts` for any
  further copy you set.
- Round three reopens the deck master, not the exported canvases.
"""

MEASURE = '''#!/usr/bin/env python3
"""Round-two clear-space pass - studio production helper.

Recomputes the clear space of the three shipped lockups: the guideline
multiplier applied to the tallest height measurable on the artwork.

    python3 measure.py [lockups-dir]

For the text lockups the measured height is the set word's ink height; for the
icon it is the mark's path-coordinate extents.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

MULTIPLIER = {"primary": 1.20, "stacked": 0.75, "icon": 0.60}
MEASURED_ON = {"primary": "wordmark", "stacked": "wordmark", "icon": "mark"}
SVG_NS = "{http://www.w3.org/2000/svg}"
NUMBER = re.compile("[-+]?(?:[0-9]+[.][0-9]*|[.][0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "lockups")


def numbers(text):
    return [float(t) for t in NUMBER.findall(text or "")]


def path_bounds(data):
    values = numbers(data)
    xs = values[0::2]
    ys = values[1::2]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def group_bounds(root, group_id):
    for group in root.iter(SVG_NS + "g"):
        if group.get("id") == group_id:
            boxes = [b for b in (path_bounds(p.get("d"))
                                 for p in group.iter(SVG_NS + "path")) if b]
            if not boxes:
                return None
            return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes))
    return None


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    print("Apex lockups - clear-space pass")
    print("=" * 62)
    for lockup in ("primary", "stacked", "icon"):
        path = os.path.join(directory, lockup + ".svg")
        root = ET.parse(path).getroot()
        bounds = group_bounds(root, MEASURED_ON[lockup])
        height = bounds[3] - bounds[1]
        margin = MULTIPLIER[lockup] * height
        view_box = " ".join((root.get("viewBox") or "").split())
        print("%-8s %-13s %8.1f x %.2f = %-6.1f  viewBox %s"
              % (lockup, MEASURED_ON[lockup], height, MULTIPLIER[lockup],
                 margin, view_box))
    print()
    print("Measured height: set word ink for the text lockups, mark path")
    print("extents for the icon. Clear space = multiplier x measured height.")


if __name__ == "__main__":
    main()
'''


def write(path, text):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o644)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/app"
    glyphs = build_glyphs()

    font_path = os.path.join(root, "fonts", "BrandSans-Display.ttf")
    if not os.path.isdir(os.path.dirname(font_path)):
        os.makedirs(os.path.dirname(font_path))
    build_font(glyphs, font_path)

    for lockup in ("primary", "stacked", "icon"):
        write(os.path.join(root, "lockups", lockup + ".svg"),
              svg_document(lockup, glyphs))

    write(os.path.join(root, "guidelines", "brand-clear-space.md"), GUIDELINES)
    write(os.path.join(root, "handoff", "round2-handoff.md"), HANDOFF)
    measure_path = os.path.join(root, "handoff", "measure.py")
    write(measure_path, MEASURE)
    os.chmod(measure_path, 0o755)


if __name__ == "__main__":
    main()
