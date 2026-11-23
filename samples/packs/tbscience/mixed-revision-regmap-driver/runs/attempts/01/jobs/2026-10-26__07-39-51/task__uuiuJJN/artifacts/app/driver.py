#!/usr/bin/env python3
"""PD-440 driver: verified revision-C excitation only.

This is an INCOMPLETE implementation of the work order.  The supplied evidence
does not identify a complete revision-D map, so decode and revision-D excite
fail explicitly instead of emitting unverified register interpretations.

Why the missing information matters:
* Every card gets only one distinct word per configuration address.  No
  writable register is read back.  ID and STATUS do not distinguish revisions.
* All 18,329 supplied ADC samples admit this witness measurement model, with
  residuals between -6 and +6 counts: shift bus times by 336,000 us; predict 6
  when disabled and 20 + 40 * gain for the enabled channel.  The two observed
  gain patterns are consistent with the documented layout and with packing
  C_GAIN/D_GAIN into the high/low bytes of 0x09, respectively.
* That witness model is not a documented calibration.  It is an example of
  nonuniqueness: it does not depend on frequency or phase.  Keeping the C/D
  frequency addresses unchanged, or exchanging them on revision D, produces
  identical predicted observations but requires different excitation writes.
  The same ambiguity applies to phase.  Clock alignment alone cannot fix it.

A verified revision-D map, or additional measurements with a specified
measurement model that distinguish those alternatives, is required to finish.
"""

import argparse
import csv
import json
import sys


CHANNELS = "ABCD"


class EvidenceError(ValueError):
    """The inputs cannot establish the hardware facts needed for an output."""


def integer(value, name, low, high=None):
    if type(value) is not int:
        raise ValueError(name + " must be an integer")
    if value < low or (high is not None and value > high):
        bounds = str(low) + (".." + str(high) if high is not None else " or greater")
        raise ValueError(name + " must be " + bounds)
    return value


def excitation_writes(spec):
    """Return writes for the documented C map; do not guess a D map."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    required = {"revision", "cs", "freq_hz", "gain", "phase_deg", "enable"}
    if set(spec) != required:
        raise ValueError("spec must contain exactly: " + ", ".join(sorted(required)))
    revision = spec["revision"]
    if revision not in ("C", "D"):
        raise ValueError("revision must be C or D")
    cs = integer(spec["cs"], "cs", 0)
    limits = {"freq_hz": 255, "gain": 63, "phase_deg": 510, "enable": 1}
    for name, maximum in limits.items():
        values = spec[name]
        if not isinstance(values, dict) or set(values) != set(CHANNELS):
            raise ValueError(name + " must contain exactly channels A, B, C, D")
        for channel in CHANNELS:
            value = integer(values[channel], name + "." + channel, 0, maximum)
            if name == "phase_deg" and value % 2:
                raise ValueError("phase_deg." + channel + " must be a multiple of 2")
    if revision == "D":
        raise EvidenceError(
            "revision-D excitation is unresolved: the supplied captures do not "
            "establish its frequency/phase register placement or complete map; "
            "a verified D map or distinguishing measurements are required"
        )
    writes = [[cs, 0x0E, 0]]
    for index, channel in enumerate(CHANNELS):
        base = 0x02 + 3 * index
        writes.extend([
            [cs, base, spec["freq_hz"][channel]],
            [cs, base + 1, spec["gain"][channel]],
            [cs, base + 2, spec["phase_deg"][channel] // 2],
        ])
    enables = sum(spec["enable"][ch] << i for i, ch in enumerate(CHANNELS))
    writes.append([cs, 0x0E, enables])
    return writes


def decode(capture, strain):
    """Validate the inputs, then report the unresolved revision/map evidence."""
    count = 0
    with open(capture, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["t_us", "cs", "op", "addr", "data"]:
            raise ValueError("invalid bus CSV header")
        for row in reader:
            integer(int(row["t_us"], 10), "t_us", 0)
            integer(int(row["cs"], 10), "cs", 0)
            integer(int(row["addr"], 16), "addr", 0, 255)
            integer(int(row["data"], 16), "data", 0, 65535)
            if row["op"] not in ("R", "W"):
                raise ValueError("bus op must be R or W")
            count += 1
    if not count:
        raise ValueError("bus capture is empty")
    samples = 0
    with open(strain, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["t_us", "adc"]:
            raise ValueError("invalid strain CSV header")
        for row in reader:
            int(row["t_us"], 10)
            int(row["adc"], 10)
            samples += 1
    if not samples:
        raise ValueError("strain capture is empty")
    raise EvidenceError(
        "decode is unresolved: no verified revision-D map or strain measurement "
        "model is available. ID cannot establish revision, and response levels "
        "alone do not identify all canonical fields. No output was written"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    decoder = commands.add_parser("decode", help="report unresolved map evidence")
    decoder.add_argument("--capture", required=True)
    decoder.add_argument("--strain", required=True)
    decoder.add_argument("--out", required=True)
    exciter = commands.add_parser("excite", help="generate verified revision-C writes")
    exciter.add_argument("--spec", required=True)
    exciter.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "decode":
            decode(args.capture, args.strain)
        else:
            with open(args.spec, encoding="utf-8") as stream:
                writes = excitation_writes(json.load(stream))
            output = json.dumps({"writes": writes}, separators=(",", ":")) + "\n"
            with open(args.out, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(output)
    except (ValueError, OSError, KeyError, TypeError, csv.Error) as error:
        print("driver.py: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
