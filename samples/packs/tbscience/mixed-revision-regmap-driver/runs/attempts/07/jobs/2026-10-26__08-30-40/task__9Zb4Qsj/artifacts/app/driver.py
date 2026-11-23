#!/usr/bin/env python3
"""PD-440 register decoder and excitation writer (Python standard library).

Field values in decode output are *codes*: in particular, a phase code of 45
means 90 degrees.  excite accepts physical units and performs the inverse
conversion without rounding.

Revision D correction reconstructed from the site_a experiment:
    0x09: C_GAIN[15:8], D_GAIN[7:0]
    0x0c: no fields
The isolated C/D runs distinguish these placements: their levels track the
two bytes of 0x09, rather than the low bytes of 0x09 and 0x0c.  A and B provide
an independent gain/ADC calibration.  Other fields retain the RM-C placement;
the supplied level measurements do not independently establish frequency or
phase placement.  The ID's REV field is deliberately NOT used to identify a
die, and the driver contains no chip-select-to-revision table.

decode synchronizes the independent clocks using observed output-off edges
and MASTER disable writes.  It checks the affine clock fit, then compares
stable strain levels with both candidate gain maps.  An ambiguous or
inconsistent experiment is an error, not an excuse to guess a revision.
"""

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from collections import defaultdict


CHANNELS = "ABCD"
# (canonical field name, least-significant bit, width)
MAP_C = {
    0x00: (("PART_ID", 12, 4), ("REV", 8, 4), ("DIE_TRIM", 0, 8)),
    0x01: (("FAULT", 8, 8), ("SEQ_STATE", 0, 8)),
    0x0E: tuple(("EN_" + ch, i, 1) for i, ch in enumerate(CHANNELS))
    + (("CLKDIV", 4, 4),),
    0x0F: (("SCRATCH", 0, 16),),
}
for _i, _ch in enumerate(CHANNELS):
    for _j, _kind in enumerate(("FREQ", "GAIN", "PHASE")):
        MAP_C[2 + 3 * _i + _j] = ((_ch + "_" + _kind, 0, 8),)
MAP_D = dict(MAP_C)
MAP_D[0x09] = (("C_GAIN", 8, 8), ("D_GAIN", 0, 8))
MAP_D[0x0C] = ()
MAPS = {"C": MAP_C, "D": MAP_D}


def fields(revision, addr, word):
    return {name: (word >> shift) & ((1 << width) - 1)
            for name, shift, width in MAPS[revision].get(addr, ())}


def read_bus(path):
    result = []
    with open(path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["t_us", "cs", "op", "addr", "data"]:
            raise ValueError("bus CSV must have header t_us,cs,op,addr,data")
        for line, row in enumerate(reader, 2):
            try:
                item = {"t_us": int(row["t_us"]), "cs": int(row["cs"]),
                        "op": row["op"], "addr": int(row["addr"], 16),
                        "data": int(row["data"], 16)}
                if (None in row or item["op"] not in ("R", "W")
                        or not 0 <= item["addr"] <= 255
                        or not 0 <= item["data"] <= 65535
                        or item["cs"] < 0):
                    raise ValueError("invalid transfer")
                if result and item["t_us"] < result[-1]["t_us"]:
                    raise ValueError("bus times are not in capture order")
            except (TypeError, ValueError) as exc:
                raise ValueError("bus CSV line %d: %s" % (line, exc)) from exc
            result.append(item)
    if not result:
        raise ValueError("empty bus capture")
    return result


def read_strain(path):
    times, values = [], []
    with open(path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["t_us", "adc"]:
            raise ValueError("strain CSV must have header t_us,adc")
        for line, row in enumerate(reader, 2):
            try:
                t, value = int(row["t_us"]), int(row["adc"])
                if None in row or (times and t <= times[-1]):
                    raise ValueError("strain times must strictly increase")
            except (TypeError, ValueError) as exc:
                raise ValueError("strain CSV line %d: %s" % (line, exc)) from exc
            times.append(t)
            values.append(value)
    if len(times) < 10:
        raise ValueError("not enough strain samples to identify revisions")
    return times, values


def line_fit(x, y, weights=None):
    """Return slope, intercept, RMS residual for a weighted affine fit."""
    if weights is None:
        weights = [1.0] * len(x)
    total = sum(weights)
    xm = sum(w * v for w, v in zip(weights, x)) / total
    ym = sum(w * v for w, v in zip(weights, y)) / total
    variance = sum(w * (v - xm) ** 2 for w, v in zip(weights, x))
    if variance == 0:
        raise ValueError("insufficient variation for calibration")
    slope = sum(w * (v - xm) * (u - ym)
                for w, v, u in zip(weights, x, y)) / variance
    intercept = ym - slope * xm
    rms = math.sqrt(sum(w * (u - slope * v - intercept) ** 2
                        for w, v, u in zip(weights, x, y)) / total)
    return slope, intercept, rms


def output_off_times(bus):
    """Global enable-to-disabled edges, independent of either gain map."""
    enabled = {}
    result = []
    for access in bus:
        if access["op"] == "W" and access["addr"] == 0x0E:
            before = any(enabled.values())
            enabled[access["cs"]] = access["data"] & 15
            if before and not any(enabled.values()):
                result.append(access["t_us"])
    return result


def strain_off_times(times, values):
    """Locate the separation between the quiet band and driven plateaus.

    No ADC scale or fixed level is assumed.  The largest relative gap between
    populated ADC bands separates quiet from driven output in this experiment.
    Single-sample gaps and short impulses cannot supply synchronization edges.
    """
    levels = sorted(set(values))
    if len(levels) < 3:
        raise ValueError("strain capture has no usable amplitude bands")
    gaps = [b - a for a, b in zip(levels, levels[1:])]
    minimum_gap = max(4 * statistics.median(gaps),
                      (levels[-1] - levels[0]) / 50)
    candidates = [(float(b - a) / (b - levels[0]), a, b)
                  for a, b in zip(levels, levels[1:])
                  if b - a > minimum_gap]
    if not candidates:
        raise ValueError("strain capture has no separated quiet/drive bands")
    _, lower, upper = max(candidates)
    threshold = (lower + upper) / 2
    runs = []
    start = None
    for i, value in enumerate(values):
        if value > threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= 3:
                runs.append((start, i))
            start = None
    if start is not None and len(values) - start >= 3:
        runs.append((start, len(values)))
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(times[end - 1] + times[end]) / 2
            for _, end in merged if end < len(times)]


def synchronize(bus, times, values):
    bus_edges = output_off_times(bus)
    strain_edges = strain_off_times(times, values)
    if len(bus_edges) < 3 or len(strain_edges) < 3:
        raise ValueError("need at least three output-off edges to align clocks")
    sample_period = statistics.median(b - a for a, b in zip(times, times[1:]))
    tolerance = 4 * sample_period

    # The complete bench session has matching edges in order.  Check that
    # hypothesis rather than treating row indices or timestamps as shared.
    if len(bus_edges) == len(strain_edges):
        slope, offset, _ = line_fit(bus_edges, strain_edges)
        if slope > 0 and max(abs(slope * x + offset - y)
                             for x, y in zip(bus_edges, strain_edges)) <= tolerance:
            return slope, offset

    # A truncated log or a muted interval can omit an edge.  Deterministically
    # search affine hypotheses and retain a monotone, one-to-one consensus.
    def matches(slope, offset):
        pairs = []
        previous = -1
        for i, t in enumerate(bus_edges):
            target = slope * t + offset
            j = bisect.bisect_left(strain_edges, target)
            choices = [k for k in (j - 1, j)
                       if previous < k < len(strain_edges)
                       and abs(strain_edges[k] - target) <= tolerance]
            if choices:
                k = min(choices, key=lambda v: (abs(strain_edges[v] - target), v))
                pairs.append((i, k))
                previous = k
        return pairs

    best = None
    for i in range(len(bus_edges) - 1):
        for j in range(i + 1, len(bus_edges)):
            for k in range(len(strain_edges) - 1):
                for l in range(k + 1, len(strain_edges)):
                    slope = ((strain_edges[l] - strain_edges[k])
                             / (bus_edges[j] - bus_edges[i]))
                    offset = strain_edges[k] - slope * bus_edges[i]
                    pairs = matches(slope, offset)
                    if len(pairs) < 3:
                        continue
                    error = sum((slope * bus_edges[p] + offset - strain_edges[q]) ** 2
                                for p, q in pairs)
                    rank = (len(pairs), -error)
                    if best is None or rank > best[0]:
                        best = (rank, pairs)
    required = max(3, math.ceil(0.75 * min(len(bus_edges), len(strain_edges))))
    if best is None or len(best[1]) < required:
        raise ValueError("output-off edges do not establish a common clock mapping")
    x = [bus_edges[i] for i, _ in best[1]]
    y = [strain_edges[j] for _, j in best[1]]
    slope, offset, _ = line_fit(x, y)
    if slope <= 0 or max(abs(slope * a + offset - b) for a, b in zip(x, y)) > tolerance:
        raise ValueError("inconsistent bus/strain clock alignment")
    return slope, offset


def identify_revisions(bus, times, values):
    slope, offset = synchronize(bus, times, values)
    cards = list(dict.fromkeys(row["cs"] for row in bus))
    states = {rev: {cs: {} for cs in cards} for rev in MAPS}
    enabled = {cs: 0 for cs in cards}
    observations = defaultdict(list)

    for index, access in enumerate(bus[:-1]):
        cs, addr = access["cs"], access["addr"]
        if access["op"] == "W":
            if addr >= 2:
                for rev in MAPS:
                    states[rev][cs].update(fields(rev, addr, access["data"]))
            if addr == 0x0E:
                enabled[cs] = access["data"] & 15
        active = [(card, mask) for card, mask in enabled.items() if mask]
        if len(active) != 1:
            continue
        card, mask = active[0]
        if mask not in (1, 2, 4, 8):
            continue  # Only isolated heads provide an unambiguous gain probe.
        channel = CHANNELS[mask.bit_length() - 1]
        predicted = []
        for rev in MAPS:
            state = states[rev][card]
            gain = state.get(channel + "_GAIN")
            freq = state.get(channel + "_FREQ")
            if gain is None or freq is None or gain == 0 or freq == 0:
                break  # Do not assume unobserved power-on state for decode.
            predicted.append(gain)
        if len(predicted) != 2:
            continue
        begin = slope * access["t_us"] + offset
        end = slope * bus[index + 1]["t_us"] + offset
        margin = 0.2 * (end - begin)
        left = bisect.bisect_left(times, begin + margin)
        right = bisect.bisect_right(times, end - margin)
        if right - left < 5:
            continue
        samples = values[left:right]
        level = statistics.median(samples)
        noise = 1.4826 * statistics.median(abs(v - level) for v in samples)
        observations[card].append((predicted[0], predicted[1], level,
                                   len(samples), noise))

    revisions = {}
    for cs in cards:
        obs = observations[cs]
        if not obs or not any(c != d for c, d, _, _, _ in obs):
            raise ValueError("no revision-discriminating strain evidence for cs %d" % cs)
        common = [row for row in obs if row[0] == row[1]]
        calibration = None
        if len({row[0] for row in common}) >= 2:
            calibration = line_fit([row[0] for row in common],
                                   [row[2] for row in common],
                                   [row[3] for row in common])[:2]
        scores = {}
        for column, rev in enumerate(MAPS):
            x = [row[column] for row in obs]
            y = [row[2] for row in obs]
            w = [row[3] for row in obs]
            if calibration is None:
                if len(set(x)) < 3:
                    scores[rev] = math.inf
                    continue
                gain_scale, baseline, _ = line_fit(x, y, w)
            else:
                gain_scale, baseline = calibration
            if gain_scale <= 0:
                scores[rev] = math.inf
            else:
                scores[rev] = math.sqrt(sum(weight * (level - baseline - gain_scale * gain) ** 2
                                            for gain, level, weight in zip(x, y, w)) / sum(w))
        winner, other = sorted(scores, key=lambda rev: (scores[rev], rev))
        noise = max(1.0, statistics.median(row[4] for row in obs))
        span = max(row[2] for row in obs) - min(row[2] for row in obs)
        if (not math.isfinite(scores[winner])
                or scores[winner] > max(5 * noise, 0.05 * span)
                or scores[other] <= scores[winner] + 3 * noise):
            raise ValueError("ambiguous/inconsistent revision evidence for cs %d" % cs)
        revisions[cs] = winner
    return revisions


def decode(args):
    bus = read_bus(args.capture)
    times, values = read_strain(args.strain)
    revisions = identify_revisions(bus, times, values)
    with open(args.out, "w", encoding="utf-8", newline="\n") as target:
        for access in bus:
            revision = revisions[access["cs"]]
            record = {key: access[key] for key in ("t_us", "cs", "op", "addr")}
            record["revision"] = revision
            record["fields"] = fields(revision, access["addr"], access["data"])
            target.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def integer(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = "%d..%d" % (minimum, maximum) if maximum is not None else ">= %d" % minimum
        raise ValueError("%s must be an integer %s" % (name, bounds))
    return value


def excitation_writes(spec):
    if not isinstance(spec, dict) or spec.get("revision") not in MAPS:
        raise ValueError("spec revision must be C or D")
    revision = spec["revision"]
    cs = integer(spec.get("cs"), "cs", 0)
    codes = {"CLKDIV": 0}
    for key, suffix, maximum in (("freq_hz", "FREQ", 255),
                                  ("gain", "GAIN", 63),
                                  ("phase_deg", "PHASE", 510),
                                  ("enable", None, 1)):
        values = spec.get(key)
        if not isinstance(values, dict) or set(values) != set(CHANNELS):
            raise ValueError("%s must contain exactly channels A, B, C, D" % key)
        for channel in CHANNELS:
            value = integer(values[channel], key + "." + channel, 0, maximum)
            if key == "phase_deg":
                if value % 2:
                    raise ValueError("phase_deg.%s must be a multiple of 2 degrees" % channel)
                value //= 2
            name = channel + "_" + suffix if suffix else "EN_" + channel
            codes[name] = value

    # Keep every head muted until all settings are ready.  Assemble entire
    # registers before writing, so the shared C/D gain word is written once.
    writes = [[cs, 0x0E, 0]]
    for addr in sorted(MAPS[revision]):
        if addr < 2 or addr >= 0x0E:
            continue
        definitions = MAPS[revision][addr]
        if definitions:
            word = sum(codes[name] << shift for name, shift, _ in definitions)
            writes.append([cs, addr, word])
    master = sum(codes["EN_" + channel] << i for i, channel in enumerate(CHANNELS))
    writes.append([cs, 0x0E, master])
    return writes


def excite(args):
    with open(args.spec, "r", encoding="utf-8") as source:
        writes = excitation_writes(json.load(source))
    with open(args.out, "w", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps({"writes": writes}, separators=(",", ":"), allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decoder = commands.add_parser("decode", help="decode transfers using measured revision evidence")
    decoder.add_argument("--capture", required=True)
    decoder.add_argument("--strain", required=True)
    decoder.add_argument("--out", required=True)
    decoder.set_defaults(run=decode)
    writer = commands.add_parser("excite", help="generate ordered writes from an excitation spec")
    writer.add_argument("--spec", required=True)
    writer.add_argument("--out", required=True)
    writer.set_defaults(run=excite)
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError, csv.Error) as exc:
        print("driver.py: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
