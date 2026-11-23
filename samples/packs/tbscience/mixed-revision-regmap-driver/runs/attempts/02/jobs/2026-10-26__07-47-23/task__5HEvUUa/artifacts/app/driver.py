#!/usr/bin/env python3
"""PD-440 utility: documented revision-C support only.

The supplied evidence is insufficient to implement the complete requested
driver. In particular, the bus capture reads only ID and STATUS, uses just
one test word per channel configuration address, and does not distinguish
the revisions through ID. No strain-to-channel transfer function or clock
correspondence is documented. A response difference does not uniquely
specify revision D's frequency/phase placement or writable-register masks.

Consequently this program implements revision-C excitation, but rejects
revision-D excitation and revision-dependent capture decoding. These errors
occur before opening the output file. No revision is guessed from a chip
select, probe ID, or an assumed alignment of the instrument clocks.

To complete the utility, supply a verified revision-D map and evidence that
distinguishes each captured card, or a sufficiently informative probing
capture with a documented measurement model.
"""

import argparse
import csv
import json
import sys


CHANNELS = "ABCD"
REV_C_MAP = {
    0x00: (("PART_ID", 12, 4), ("REV", 8, 4), ("DIE_TRIM", 0, 8)),
    0x01: (("FAULT", 8, 8), ("SEQ_STATE", 0, 8)),
    0x0E: tuple(("EN_" + ch, i, 1) for i, ch in enumerate(CHANNELS))
    + (("CLKDIV", 4, 4),),
    0x0F: (("SCRATCH", 0, 16),),
}
for _index, _channel in enumerate(CHANNELS):
    for _offset, _kind in enumerate(("FREQ", "GAIN", "PHASE")):
        REV_C_MAP[2 + 3 * _index + _offset] = (
            (_channel + "_" + _kind, 0, 8),
        )


class DriverError(ValueError):
    """Invalid input or insufficient evidence for a requested operation."""


def integer(value, label, low, high=None):
    if type(value) is not int or value < low or (
        high is not None and value > high
    ):
        interval = str(low) + ".." + (str(high) if high is not None else "unbounded")
        raise DriverError(label + " must be an integer in " + interval)
    return value


def channel_values(spec, name, high, step=1):
    values = spec.get(name)
    if not isinstance(values, dict) or set(values) != set(CHANNELS):
        raise DriverError(name + " must contain exactly A, B, C and D")
    result = {}
    for channel in CHANNELS:
        label = name + "." + channel
        value = integer(values[channel], label, 0, high)
        if value % step:
            raise DriverError(label + " must be a multiple of " + str(step))
        result[channel] = value // step
    return result


def excitation_writes(spec):
    if not isinstance(spec, dict):
        raise DriverError("spec must be a JSON object")
    revision = spec.get("revision")
    if revision not in ("C", "D"):
        raise DriverError("revision must be C or D")
    if revision == "D":
        raise DriverError(
            "revision-D excitation cannot be established from the supplied "
            "evidence: its register/field map is unverified; the compatibility "
            "notice and level measurements do not determine that map"
        )
    cs = integer(spec.get("cs"), "cs", 0)
    frequency = channel_values(spec, "freq_hz", 255)
    gain = channel_values(spec, "gain", 63)
    phase = channel_values(spec, "phase_deg", 510, step=2)
    enable = channel_values(spec, "enable", 1)
    # Disable before configuration; CLKDIV=0 gives the documented code scale.
    writes = [[cs, 0x0E, 0]]
    for index, channel in enumerate(CHANNELS):
        base = 2 + 3 * index
        writes.extend([
            [cs, base, frequency[channel]],
            [cs, base + 1, gain[channel]],
            [cs, base + 2, phase[channel]],
        ])
    master = sum(enable[channel] << index for index, channel in enumerate(CHANNELS))
    writes.append([cs, 0x0E, master])
    return writes


def decode_fields_c(address, word):
    """Decode raw integer field codes once revision C is independently known."""
    return {
        name: (word >> shift) & ((1 << width) - 1)
        for name, shift, width in REV_C_MAP.get(address, ())
    }


def read_capture(path):
    accesses = []
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        if next(reader, None) != ["t_us", "cs", "op", "addr", "data"]:
            raise DriverError("capture header must be t_us,cs,op,addr,data")
        for line, row in enumerate(reader, 2):
            if len(row) != 5:
                raise DriverError("invalid capture row " + str(line))
            timestamp, cs, op, address, word = row
            if op not in ("R", "W"):
                raise DriverError("invalid operation at capture row " + str(line))
            accesses.append({
                "t_us": int(timestamp, 10),
                "cs": integer(int(cs, 10), "capture cs", 0),
                "op": op,
                "addr": integer(int(address, 16), "capture address", 0, 255),
                "data": integer(int(word, 16), "capture word", 0, 65535),
            })
    return accesses


def decode(capture, strain, output):
    accesses = read_capture(capture)
    samples = 0
    with open(strain, newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        if next(reader, None) != ["t_us", "adc"]:
            raise DriverError("strain header must be t_us,adc")
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise DriverError("invalid strain row " + str(line))
            int(row[0], 10)
            int(row[1], 10)
            samples += 1
    if not samples:
        raise DriverError("strain capture contains no samples")
    if not accesses:
        with open(output, "w", encoding="utf-8", newline="\n"):
            pass
        return
    raise DriverError(
        "decode cannot establish each card's revision and revision-D fields "
        "from the supplied evidence. ID does not distinguish the dies; "
        "the configuration map and the strain measurement model are "
        "unverified. No output was written"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decoder = commands.add_parser("decode")
    decoder.add_argument("--capture", required=True)
    decoder.add_argument("--strain", required=True)
    decoder.add_argument("--out", required=True)
    exciter = commands.add_parser("excite")
    exciter.add_argument("--spec", required=True)
    exciter.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "decode":
            decode(args.capture, args.strain, args.out)
        else:
            with open(args.spec, encoding="utf-8") as source:
                writes = excitation_writes(json.load(source))
            with open(args.out, "w", encoding="utf-8", newline="\n") as target:
                json.dump({"writes": writes}, target, separators=(",", ":"))
                target.write("\n")
    except (DriverError, OSError, ValueError) as error:
        print("driver.py: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
