#!/usr/bin/env python3
"""PD-440 capture decoder and excitation writer (Python standard library).

The ID word deliberately does not identify the revision.  Classification
uses the recorded solo-channel gain sweeps, after registering the independent
clocks using the MASTER disable events.  No chip-select assignment, absolute
timestamp, ID/trim value, or ADC calibration is built into the classifier.

Bench evidence for the D gain layout: after 0x09 <- 0x0A05, the C and D
responses correspond to gains 10 and 5.  The subsequent 0x0C <- 0x0011 does
not change D's response.  Thus D packs C_GAIN[15:8], D_GAIN[7:0] into 0x09;
0x0C has no remaining named fields.  Other fields retain the documented C
placement.  The envelope capture tests gains/enables, not frequency/phase
placement; retaining the latter is a compatibility assumption from the map
and addendum, not an independent measurement of those fields.

Decode fields are raw integer codes: a PHASE value of 45 means 90 degrees.
Excite accepts physical units and converts phase degrees back to codes.
"""

import argparse
import bisect
import csv
import json
import math
import statistics
import sys


CHANNELS = "ABCD"


def register_map(revision):
    """Return addr -> tuples of (canonical name, least bit, width)."""
    result = {
        0x00: (("PART_ID", 12, 4), ("REV", 8, 4), ("DIE_TRIM", 0, 8)),
        0x01: (("FAULT", 8, 8), ("SEQ_STATE", 0, 8)),
        0x0E: tuple(("EN_" + ch, i, 1) for i, ch in enumerate(CHANNELS))
        + (("CLKDIV", 4, 4),),
        0x0F: (("SCRATCH", 0, 16),),
    }
    for i, ch in enumerate(CHANNELS):
        for j, kind in enumerate(("FREQ", "GAIN", "PHASE")):
            result[2 + 3 * i + j] = ((ch + "_" + kind, 0, 8),)
    if revision == "D":
        result[0x09] = (("C_GAIN", 8, 8), ("D_GAIN", 0, 8))
        result[0x0C] = ()
    elif revision != "C":
        raise ValueError("revision must be C or D")
    return result


MAPS = {revision: register_map(revision) for revision in ("C", "D")}


def fields(revision, addr, data):
    return {name: (data >> shift) & ((1 << width) - 1)
            for name, shift, width in MAPS[revision].get(addr, ())}


def read_csv(path, expected_header):
    with open(path, "r", encoding="utf-8", newline="") as source:
        rows = csv.reader(source)
        if next(rows, None) != expected_header:
            raise ValueError("unexpected CSV header in " + str(path))
        for line, row in enumerate(rows, 2):
            if len(row) != len(expected_header):
                raise ValueError("invalid CSV row at %s:%d" % (path, line))
            yield row


def load_bus(path):
    accesses = []
    for t, cs, op, addr, data in read_csv(
            path, ["t_us", "cs", "op", "addr", "data"]):
        t, cs, addr, data = int(t), int(cs), int(addr, 16), int(data, 16)
        if cs < 0 or op not in ("R", "W"):
            raise ValueError("invalid chip select or operation in bus capture")
        if not 0 <= addr <= 255 or not 0 <= data <= 65535:
            raise ValueError("bus address or word is out of range")
        if accesses and t < accesses[-1]["t_us"]:
            raise ValueError("bus timestamps must be in capture order")
        accesses.append(dict(t_us=t, cs=cs, op=op, addr=addr, data=data))
    if not accesses:
        raise ValueError("empty bus capture")
    return accesses


def load_strain(path):
    times, values = [], []
    for t, value in read_csv(path, ["t_us", "adc"]):
        t, value = int(t), int(value)
        if times and t <= times[-1]:
            raise ValueError("strain timestamps must be strictly increasing")
        times.append(t)
        values.append(value)
    if len(times) < 9:
        raise ValueError("not enough strain samples to establish revisions")
    return times, values


def fit_line(xs, ys):
    """Centered least squares; return x center, y center, slope."""
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    variance = math.fsum((x - xm) ** 2 for x in xs)
    if variance == 0:
        raise ValueError("insufficient variation for evidence calibration")
    slope = math.fsum((x - xm) * (y - ym)
                      for x, y in zip(xs, ys)) / variance
    return xm, ym, slope


def align_clocks(accesses, times, values):
    """Fit logger_time = center_y + rate * (bus_time - center_x).

    Pair independently detected falling envelope edges with the ordered bus
    events that disable all channels.  The two files' timestamp origins and
    clock rates are never presumed equal.  A complete solo-sweep session is
    required; unmatched events are rejected rather than silently paired.
    """
    masters = {}
    bus_edges = []
    for access in accesses:
        if access["op"] == "W" and access["addr"] == 0x0E:
            was_active = any(masters.values())
            masters[access["cs"]] = access["data"] & 15
            if was_active and not any(masters.values()):
                bus_edges.append(access["t_us"])

    ordered = sorted(values)
    low = ordered[len(ordered) // 10]
    high = ordered[19 * len(ordered) // 20]
    if high <= low:
        raise ValueError("strain capture has no usable excitation response")
    threshold = low + 0.1 * (high - low)
    logger_edges = []
    run = 0
    # A centered median suppresses isolated ADC outliers without shifting
    # an abrupt sustained transition by an entire window.
    for i in range(2, len(values) - 2):
        active = statistics.median(values[i - 2:i + 3]) > threshold
        if active:
            run += 1
        else:
            if run >= 5:
                logger_edges.append(times[i])
            run = 0

    if len(bus_edges) < 3 or len(logger_edges) != len(bus_edges):
        raise ValueError(
            "cannot align captures: %d bus disable events, %d strain falls; "
            "a complete, distinguishable solo-channel sweep is required"
            % (len(bus_edges), len(logger_edges)))
    xm, ym, rate = fit_line(bus_edges, logger_edges)
    sample_period = statistics.median(
        times[i] - times[i - 1] for i in range(1, len(times)))
    residual = max(abs(y - (ym + rate * (x - xm)))
                   for x, y in zip(bus_edges, logger_edges))
    if rate <= 0 or residual > 3 * sample_period:
        raise ValueError("capture events do not support a consistent clock alignment")
    return xm, ym, rate, sample_period


def gain_code(registers, revision, channel):
    name = CHANNELS[channel] + "_GAIN"
    for addr, definitions in MAPS[revision].items():
        for field, shift, width in definitions:
            if field == name:
                return (registers.get(addr, 0) >> shift) & ((1 << width) - 1)
    raise ValueError("gain field is missing from register map")


def infer_revisions(accesses, times, values):
    xm, ym, rate, sample_period = align_clocks(accesses, times, values)
    cards = sorted({access["cs"] for access in accesses})
    registers = {cs: {} for cs in cards}
    observations = {cs: [] for cs in cards}
    for i, access in enumerate(accesses[:-1]):
        cs, addr, data = access["cs"], access["addr"], access["data"]
        if access["op"] == "W" and 2 <= addr <= 15:
            registers[cs][addr] = data
        active = [(card, reg.get(0x0E, 0) & 15)
                  for card, reg in registers.items() if reg.get(0x0E, 0) & 15]
        if len(active) != 1:
            continue
        cs, enables = active[0]
        if enables not in (1, 2, 4, 8):
            continue
        start = ym + rate * (access["t_us"] - xm)
        end = ym + rate * (accesses[i + 1]["t_us"] - xm)
        # Keep only settled interior samples, away from either transition.
        margin = max(0.2 * (end - start), 3 * sample_period)
        if end - start <= 2 * margin:
            continue
        left = bisect.bisect_left(times, start + margin)
        right = bisect.bisect_right(times, end - margin)
        if right - left < 5:
            continue
        channel = enables.bit_length() - 1
        c_gain = gain_code(registers[cs], "C", channel)
        d_gain = gain_code(registers[cs], "D", channel)
        observations[cs].append((channel, c_gain, d_gain,
                                 statistics.median(values[left:right])))

    revisions = {}
    for cs in cards:
        measured = observations[cs]
        reference = [row for row in measured if row[0] in (0, 1)]
        if len(reference) < 2:
            raise ValueError("card %d lacks A/B gain calibration evidence" % cs)
        gx, gy, slope = fit_line([row[1] for row in reference],
                                 [row[3] for row in reference])
        if slope <= 0:
            raise ValueError("card %d has no positive gain response" % cs)
        noise = math.sqrt(statistics.mean(
            (row[3] - (gy + slope * (row[1] - gx))) ** 2
            for row in reference))
        discriminating = [row for row in measured if row[1] != row[2]]
        if not discriminating:
            raise ValueError("card %d lacks revision-discriminating evidence" % cs)
        errors = {}
        for revision, index in (("C", 1), ("D", 2)):
            errors[revision] = math.sqrt(statistics.mean(
                (row[3] - (gy + slope * (row[index] - gx))) ** 2
                for row in discriminating))
        winner = min(("C", "D"), key=lambda revision: errors[revision])
        loser = "D" if winner == "C" else "C"
        tolerance = max(2.0, 5 * noise, abs(slope) * 0.1)
        if errors[winner] > tolerance or errors[loser] <= max(
                4 * errors[winner], 4 * tolerance):
            raise ValueError("card %d revision is not established by the response" % cs)
        revisions[cs] = winner
    return revisions


def decode(capture, strain, out):
    accesses = load_bus(capture)
    times, values = load_strain(strain)
    revisions = infer_revisions(accesses, times, values)
    with open(out, "w", encoding="utf-8", newline="\n") as destination:
        for access in accesses:
            revision = revisions[access["cs"]]
            record = {key: access[key] for key in ("t_us", "cs", "op", "addr")}
            record["revision"] = revision
            record["fields"] = fields(revision, access["addr"], access["data"])
            destination.write(json.dumps(record, separators=(",", ":")) + "\n")


def integer(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (
            maximum is not None and value > maximum):
        limit = "%d or greater" % minimum if maximum is None else (
            "%d through %d" % (minimum, maximum))
        raise ValueError("%s must be an integer %s" % (name, limit))
    return value


def excitation_writes(spec):
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    revision = spec.get("revision")
    if revision not in ("C", "D"):
        raise ValueError("revision must be C or D")
    cs = integer(spec.get("cs"), "cs", 0)
    codes = {"CLKDIV": 0}
    for key, suffix, maximum in (("freq_hz", "FREQ", 255),
                                  ("gain", "GAIN", 63),
                                  ("phase_deg", "PHASE", 510),
                                  ("enable", None, 1)):
        requested = spec.get(key)
        if not isinstance(requested, dict) or set(requested) != set(CHANNELS):
            raise ValueError(key + " must contain exactly channels A, B, C, D")
        for ch in CHANNELS:
            value = integer(requested[ch], key + "." + ch, 0, maximum)
            if key == "phase_deg":
                if value % 2:
                    raise ValueError("phase_deg." + ch + " must be a multiple of 2")
                value //= 2
            name = "EN_" + ch if suffix is None else ch + "_" + suffix
            codes[name] = value

    # Program complete words with channels disabled; shared fields are packed
    # together, so no write overwrites its neighbor's requested value.  Set
    # CLKDIV to zero and enable channels only after all parameters are loaded.
    writes = [[cs, 0x0E, 0]]
    for addr in sorted(MAPS[revision]):
        definitions = MAPS[revision][addr]
        if addr < 2 or addr >= 0x0E or not definitions:
            continue
        word = sum(codes[name] << shift for name, shift, width in definitions)
        writes.append([cs, addr, word])
    master = sum(codes[name] << shift
                 for name, shift, width in MAPS[revision][0x0E])
    writes.append([cs, 0x0E, master])
    return writes


def excite(spec_path, out):
    with open(spec_path, "r", encoding="utf-8") as source:
        spec = json.load(source)
    writes = excitation_writes(spec)
    with open(out, "w", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps({"writes": writes}, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    decoder = commands.add_parser("decode", help="decode using measured revision evidence")
    decoder.add_argument("--capture", required=True)
    decoder.add_argument("--strain", required=True)
    decoder.add_argument("--out", required=True)
    exciter = commands.add_parser("excite", help="generate excitation register writes")
    exciter.add_argument("--spec", required=True)
    exciter.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "decode":
            decode(args.capture, args.strain, args.out)
        else:
            excite(args.spec, args.out)
    except (OSError, ValueError, csv.Error) as error:
        print("driver.py: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
