#!/usr/bin/env python3
"""PD-440 capture decoder and excitation writer (Python standard library).

The bench evidence supports the following correction to PD440-RM-C on D:
0x09 contains C_GAIN[15:8] and D_GAIN[7:0]; 0x0c has no gain field.
The remaining fields use the documented layout. In particular, frequency,
phase, ID, STATUS, MASTER and SCRATCH placement are taken from the vendor
documentation, not independently established by the scalar strain capture.

Revision identification deliberately does not inspect ID or chip-select
numbers. It matches the ordered mute edges in the two recordings, estimates
their clock relationship, calibrates the strain response using gain settings
common to both layouts, and compares both layouts against each card's actual
response. Unsupported or ambiguous evidence is an error, not a default to C.
Decoded fields are raw integer codes; physical-unit conversion is performed
when compiling an excitation specification (phase code = degrees / 2).
"""

import argparse
import bisect
import csv
import json
import math
import os
import statistics
import sys
import tempfile


CHANNELS = "ABCD"


def register_map(revision):
    """Map address to (canonical field name, least significant bit, width)."""
    if revision not in ("C", "D"):
        raise ValueError("revision must be C or D")
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
    return result


MAPS = {revision: register_map(revision) for revision in ("C", "D")}


def fields(revision, addr, data):
    return {name: (data >> shift) & ((1 << width) - 1)
            for name, shift, width in MAPS[revision].get(addr, ())}


def read_csv(path, header):
    with open(path, "r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != header:
            raise ValueError("{}: expected header {}".format(path, ",".join(header)))
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError("{}:{}: wrong number of columns".format(path, number))
            yield number, row


def read_bus(path):
    accesses = []
    previous = None
    for number, row in read_csv(path, ["t_us", "cs", "op", "addr", "data"]):
        t, cs, op, addr, data = row
        try:
            t, cs, addr, data = int(t, 10), int(cs, 10), int(addr, 16), int(data, 16)
        except ValueError:
            raise ValueError("{}:{}: invalid integer".format(path, number)) from None
        if op not in ("R", "W") or cs < 0 or not 0 <= addr <= 255 or not 0 <= data <= 65535:
            raise ValueError("{}:{}: invalid bus access".format(path, number))
        if previous is not None and t < previous:
            raise ValueError("{}:{}: bus clock runs backwards".format(path, number))
        previous = t
        accesses.append((t, cs, op, addr, data))
    if not accesses:
        raise ValueError("empty bus capture")
    return accesses


def read_strain(path):
    times, values = [], []
    for number, row in read_csv(path, ["t_us", "adc"]):
        try:
            t, value = (int(x, 10) for x in row)
        except ValueError:
            raise ValueError("{}:{}: invalid integer".format(path, number)) from None
        if times and t <= times[-1]:
            raise ValueError("{}:{}: strain times must increase".format(path, number))
        times.append(t)
        values.append(value)
    if len(times) < 64:
        raise ValueError("insufficient strain samples for revision identification")
    return times, values


def linear_fit(points):
    """Return intercept, slope; center coordinates to preserve precision."""
    xm = statistics.mean(x for x, y in points)
    ym = statistics.mean(y for x, y in points)
    denominator = math.fsum((x - xm) ** 2 for x, y in points)
    if denominator == 0:
        raise ValueError("insufficient independent calibration points")
    slope = math.fsum((x - xm) * (y - ym) for x, y in points) / denominator
    return ym - slope * xm, slope


def clock_alignment(accesses, times, values):
    """Fit clocks from completed active bursts, using only their falling edges.

    Rising edges can be delayed until gain is programmed, so they are not
    clock anchors. The bench capture has leading quiescent samples and one
    isolated active channel at a time. Check these conditions instead of
    assuming that a timestamp from one instrument indexes the other.
    """
    enabled = {}
    bus_edges = []
    for t, cs, op, addr, data in accesses:
        if op == "W" and addr == 0x0E:
            was_active = any(enabled.values())
            enabled[cs] = data & 15
            if was_active and not any(enabled.values()):
                bus_edges.append(t)
    if len(bus_edges) < 3:
        raise ValueError("need at least three complete mute transitions to align clocks")

    baseline = values[:32]
    threshold = statistics.median(baseline) + max(1, 4 * (max(baseline) - min(baseline)))
    # A centered median rejects isolated ADC noise without moving a step edge.
    smooth = [statistics.median(values[max(0, i - 4):min(len(values), i + 5)])
              for i in range(len(values))]
    strain_edges = []
    start = None
    for i, value in enumerate(smooth):
        if value > threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= 12:
                strain_edges.append((times[i - 1] + times[i]) / 2)
            start = None
    if len(bus_edges) != len(strain_edges):
        raise ValueError("cannot align clocks: {} bus mute edges, {} strain bursts; "
                         "need a complete isolated-channel bench capture with leading silence"
                         .format(len(bus_edges), len(strain_edges)))
    offset, rate = linear_fit(list(zip(bus_edges, strain_edges)))
    sample_period = statistics.median(b - a for a, b in zip(times, times[1:]))
    residual = max(abs(offset + rate * b - s) for b, s in zip(bus_edges, strain_edges))
    if rate <= 0 or residual > 8 * sample_period:
        raise ValueError("mute transitions do not establish a consistent clock alignment")
    return offset, rate, sample_period


def identify_revisions(accesses, times, values):
    offset, rate, sample_period = clock_alignment(accesses, times, values)
    cards = sorted({row[1] for row in accesses})
    states = {cs: {revision: {} for revision in MAPS} for cs in cards}
    observations = []
    calibration = {}
    for index, (t, cs, op, addr, data) in enumerate(accesses[:-1]):
        if op == "W" and addr not in (0, 1):
            for revision in MAPS:
                states[cs][revision].update(fields(revision, addr, data))
        duration = accesses[index + 1][0] - t
        if duration * rate < 12 * sample_period:
            continue
        active = []
        for card in cards:
            for ch in CHANNELS:
                if states[card]["C"].get("EN_" + ch, 0):
                    active.append((card, ch))
        # A single gauge cannot attribute simultaneous card responses.
        if len(active) != 1:
            continue
        card, ch = active[0]
        lo = bisect.bisect_left(times, offset + rate * (t + 0.2 * duration))
        hi = bisect.bisect_left(times, offset + rate * (t + 0.8 * duration))
        if hi - lo < 7:
            continue
        response = statistics.median(values[lo:hi])
        codes = tuple(states[card][revision].get(ch + "_GAIN", 0)
                      for revision in ("C", "D"))
        observations.append((card, codes, response))
        if codes[0] == codes[1]:
            calibration.setdefault(codes[0], []).append(response)
    if len(calibration) < 2:
        raise ValueError("need distinct gain settings shared by both layouts to calibrate strain")
    intercept, scale = linear_fit([(gain, statistics.median(responses))
                                   for gain, responses in sorted(calibration.items())])
    if scale <= 0:
        raise ValueError("strain response does not establish a positive gain calibration")
    # Validate that the shared-layout controls actually obey the fitted model.
    control_errors = [abs((response - intercept) / scale - codes[0])
                      for card, codes, response in observations if codes[0] == codes[1]]
    if statistics.median(control_errors) > 0.25:
        raise ValueError("shared-layout gain controls do not fit the strain response")
    revisions = {}
    for card in cards:
        evidence = [(codes, (response - intercept) / scale)
                    for cs, codes, response in observations
                    if cs == card and codes[0] != codes[1]]
        if not evidence:
            raise ValueError("card {} has no revision-discriminating gain evidence".format(card))
        errors = [statistics.mean((observed - codes[i]) ** 2
                                  for codes, observed in evidence) for i in (0, 1)]
        winner = 0 if errors[0] < errors[1] else 1
        if errors[winner] > 0.25 or errors[1 - winner] - errors[winner] < 1:
            raise ValueError("card {} has ambiguous or unsupported revision evidence".format(card))
        revisions[card] = ("C", "D")[winner]
    return revisions


def atomic_write(path, write):
    """Do not leave a partial result if validation or serialization fails."""
    destination = os.path.abspath(path)
    fd, temporary = tempfile.mkstemp(prefix=".pd440-", dir=os.path.dirname(destination))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            write(stream)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dump_json(value, stream):
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    stream.write("\n")


def decode(capture, strain, out):
    accesses = read_bus(capture)
    times, values = read_strain(strain)
    revisions = identify_revisions(accesses, times, values)

    def write(stream):
        for t, cs, op, addr, data in accesses:
            revision = revisions[cs]
            dump_json({"t_us": t, "cs": cs, "op": op, "addr": addr,
                       "revision": revision, "fields": fields(revision, addr, data)}, stream)

    atomic_write(out, write)


def integer(value, label, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = "{}..{}".format(minimum, maximum) if maximum is not None else ">={}".format(minimum)
        raise ValueError("{} must be an integer in {}".format(label, bounds))
    return value


def compile_excitation(spec):
    if not isinstance(spec, dict):
        raise ValueError("excitation specification must be an object")
    revision = spec.get("revision")
    if revision not in ("C", "D"):
        raise ValueError("revision must be C or D")
    cs = integer(spec.get("cs"), "cs", 0)
    codes = {"CLKDIV": 0}
    for key, suffix, maximum in (("freq_hz", "FREQ", 255), ("gain", "GAIN", 63),
                                 ("phase_deg", "PHASE", 510), ("enable", None, 1)):
        group = spec.get(key)
        if not isinstance(group, dict) or set(group) != set(CHANNELS):
            raise ValueError("{} must contain exactly A, B, C and D".format(key))
        for ch in CHANNELS:
            value = integer(group[ch], key + "." + ch, 0, maximum)
            if key == "phase_deg":
                if value % 2:
                    raise ValueError("phase_deg.{} is not representable at 2 degrees per code".format(ch))
                value //= 2
            name = "EN_" + ch if suffix is None else ch + "_" + suffix
            codes[name] = value
    words = {}
    for addr, definitions in sorted(MAPS[revision].items()):
        selected = [(name, shift) for name, shift, width in definitions if name in codes]
        if selected:
            words[addr] = sum(codes[name] << shift for name, shift in selected)
    # Set all channel parameters while muted, then enable the requested heads.
    writes = [[cs, 0x0E, 0]]
    writes.extend([cs, addr, word] for addr, word in sorted(words.items()) if addr != 0x0E)
    writes.append([cs, 0x0E, words[0x0E]])
    return {"writes": writes}


def excite(spec_path, out):
    with open(spec_path, "r", encoding="utf-8") as stream:
        spec = json.load(stream)
    result = compile_excitation(spec)
    atomic_write(out, lambda stream: dump_json(result, stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decoder = commands.add_parser("decode", help="decode using each card's measured gain response")
    decoder.add_argument("--capture", required=True)
    decoder.add_argument("--strain", required=True)
    decoder.add_argument("--out", required=True)
    exciter = commands.add_parser("excite", help="compile physical settings to ordered writes")
    exciter.add_argument("--spec", required=True)
    exciter.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "decode":
            decode(args.capture, args.strain, args.out)
        else:
            excite(args.spec, args.out)
    except (OSError, ValueError) as error:
        print("driver.py: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
