#!/usr/bin/env python3
"""PD-440 utility: verified revision-C excitation only.

This is NOT a complete implementation of the work-order contract. The supplied
evidence does not identify revision D's register map or reliably label every
card's revision. Unsupported operations fail before opening the output file.

Evidence limits in the staged bench session:
* The only reads are ID and STATUS; ID intentionally cannot distinguish dies.
* Each channel register receives only one distinct test word, repeated.
* The strain signal has essentially constant plateaus, with no observable
  response to the frequency/phase writes. No sensor transfer model is supplied.
* The instruments have independent clocks and no supplied synchronization.

Even granting alignment and a gain-response model, different frequency/phase
maps can explain these observations while requiring different excitation
writes. A verified D map or additional discriminating experiments are needed.
No probe-ID lookup, fixed chip-select table, or guessed D mapping is used.
"""

import argparse
import json
import os
import sys
import tempfile


CHANNELS = "ABCD"
EVIDENCE_ERROR = (
    "The supplied evidence does not establish the revision-D register map or "
    "each card's revision. ID/STATUS are the only registers read, and the "
    "strain plateaus do not identify frequency/phase field placement. "
    "Provide a verified D map and revision evidence, or additional "
    "discriminating measurements; no output has been written."
)


def integer(value, label, low, high=None):
    if type(value) is not int:
        raise ValueError(label + " must be an integer")
    if value < low or (high is not None and value > high):
        bounds = str(low) + (".." + str(high) if high is not None else " or greater")
        raise ValueError(label + " must be " + bounds)
    return value


def excitation(spec):
    """Return writes proven by PD440-RM-C, or reject an unsupported request."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    revision = spec.get("revision")
    if revision not in ("C", "D"):
        raise ValueError("revision must be C or D")
    if revision == "D":
        raise ValueError(EVIDENCE_ERROR)
    cs = integer(spec.get("cs"), "cs", 0)
    codes = {}
    for name, maximum in (("freq_hz", 255), ("gain", 63),
                          ("phase_deg", 510), ("enable", 1)):
        values = spec.get(name)
        if not isinstance(values, dict) or set(values) != set(CHANNELS):
            raise ValueError(name + " must contain exactly A, B, C, D")
        codes[name] = {}
        for channel in CHANNELS:
            label = name + "." + channel
            value = integer(values[channel], label, 0, maximum)
            if name == "phase_deg":
                if value % 2:
                    raise ValueError(label + " must be a multiple of 2 degrees")
                value //= 2
            codes[name][channel] = value

    # Disable outputs, program all channel fields, then enable the requested
    # channels with CLKDIV=0. No read-only or reserved register is written.
    writes = [[cs, 0x0E, 0]]
    for index, channel in enumerate(CHANNELS):
        base = 0x02 + 3 * index
        for offset, name in enumerate(("freq_hz", "gain", "phase_deg")):
            writes.append([cs, base + offset, codes[name][channel]])
    master = sum(codes["enable"][channel] << index
                 for index, channel in enumerate(CHANNELS))
    writes.append([cs, 0x0E, master])
    return {"writes": writes}


def write_json(path, value):
    """Replace the output only after successful validation and serialization."""
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n"
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".pd440-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decode = commands.add_parser("decode", help="unavailable: revision evidence is insufficient")
    decode.add_argument("--capture", required=True)
    decode.add_argument("--strain", required=True)
    decode.add_argument("--out", required=True)
    excite = commands.add_parser("excite", help="generate verified revision-C writes")
    excite.add_argument("--spec", required=True)
    excite.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "decode":
            raise ValueError(EVIDENCE_ERROR)
        with open(args.spec, encoding="utf-8") as stream:
            result = excitation(json.load(stream))
        write_json(args.out, result)
    except (OSError, ValueError) as error:
        print("driver.py: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
