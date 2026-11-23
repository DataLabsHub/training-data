#!/usr/bin/env python3
"""PD-440: the portion of the driver supported by the supplied evidence.

IMPORTANT: This is not a complete implementation of the work order. Rev-C
excitation is implemented. Decode and rev-D excitation deliberately fail
without producing an output file, because the supplied evidence does not
uniquely determine the rev-D register map or establish each card's revision.

Why the missing map cannot be reconstructed from these captures:

* All four cards receive the same sequence of writes. Each configuration
  address is always written with the same word. Only ID and STATUS are read.
  Neither read reveals the writable-register masks or field placement.
* The strain clock can plausibly be aligned to the bus clock using shutdown
  edges (approximately strain_us = .99997407 * bus_us + 339974.78). This is
  an inferred alignment, not a supplied synchronization guarantee.
* After alignment, the four active plateaus for two cards are approximately
  300, 380, 220, 700 counts, and for the other two 300, 380, 420, 220.
  No sensor transfer function or frequency/phase measurement is supplied.
* For example, an enabled-channel response of 20 + 40 * gain, plus noise,
  could explain these levels if D packs C_GAIN in bits 15:8 and D_GAIN in
  bits 7:0 at 0x09 and removes the gain field at 0x0c. This is a possible
  explanation, NOT a verified map. Keeping C/D phase addresses unchanged
  or exchanging them would then give identical recorded observations,
  but different correct decode output and excitation writes. The unknown
  frequency placement admits the same ambiguity. Even the proposed gain
  field widths have not been established by mask/readback tests.

The compatibility notice cannot resolve this: accepting it literally yields
no behavioral distinction with which to identify the swapped dies, while
rejecting it leaves no authoritative D map. A verified D map and a capture
revision discriminator (or controlled distinguishing tests) are required to
finish the two unsupported operations. Do not replace these failures with
an ID-derived revision, a CS lookup table, or an assumed compatible map.

Only Python's standard library is used. No companion files are required.
"""

import argparse
import csv
import json
import sys


CHANNELS = "ABCD"

# (canonical field name, least-significant bit, field width).
# Decode codes are raw integers; phase degrees are twice the phase code.
REV_C_FIELDS = {
    0x00: (("PART_ID", 12, 4), ("REV", 8, 4), ("DIE_TRIM", 0, 8)),
    0x01: (("FAULT", 8, 8), ("SEQ_STATE", 0, 8)),
    0x0E: (("EN_A", 0, 1), ("EN_B", 1, 1), ("EN_C", 2, 1),
           ("EN_D", 3, 1), ("CLKDIV", 4, 4)),
    0x0F: (("SCRATCH", 0, 16),),
}
for _channel_index, _channel in enumerate(CHANNELS):
    for _field_index, _kind in enumerate(("FREQ", "GAIN", "PHASE")):
        REV_C_FIELDS[2 + 3 * _channel_index + _field_index] = (
            (_channel + "_" + _kind, 0, 8),
        )


class EvidenceError(ValueError):
    """An operation needs information not established by the inputs."""


def integer(value, label, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("{} must be an integer in {}..{}".format(
            label, lower, upper))
    return value


def fields_for_known_rev_c(addr, data):
    """Extract fields only when the caller independently knows the die is C."""
    return {name: (data >> shift) & ((1 << width) - 1)
            for name, shift, width in REV_C_FIELDS.get(addr, ())}


def excitation_writes(spec):
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    revision = spec.get("revision")
    if revision not in ("C", "D"):
        raise ValueError('revision must be "C" or "D"')
    cs = spec.get("cs")
    if type(cs) is not int:
        raise ValueError("cs must be an integer")

    limits = {"freq_hz": 255, "gain": 63, "phase_deg": 510, "enable": 1}
    for key, upper in limits.items():
        values = spec.get(key)
        if not isinstance(values, dict) or set(values) != set(CHANNELS):
            raise ValueError(key + " must contain exactly A, B, C and D")
        for channel in CHANNELS:
            value = integer(values[channel], key + "." + channel, 0, upper)
            if key == "phase_deg" and value % 2:
                raise ValueError("phase_deg." + channel +
                                 " must be a multiple of 2 degrees")

    if revision == "D":
        raise EvidenceError(
            "rev-D excitation is unresolved: no verified rev-D field map "
            "was supplied, and the captures cannot uniquely establish its "
            "frequency, phase, gain and enable placement. No writes emitted.")

    # Disable all channels and set divide-by-1 before configuring. Enabling
    # last ensures no channel drives with only part of its requested setup.
    writes = [[cs, 0x0E, 0]]
    for index, channel in enumerate(CHANNELS):
        base = 2 + 3 * index
        writes.extend((
            [cs, base, spec["freq_hz"][channel]],
            [cs, base + 1, spec["gain"][channel]],
            [cs, base + 2, spec["phase_deg"][channel] // 2],
        ))
    master = sum(spec["enable"][channel] << index
                 for index, channel in enumerate(CHANNELS))
    writes.append([cs, 0x0E, master])
    return writes


def read_capture(path):
    accesses = []
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["t_us", "cs", "op", "addr", "data"]:
            raise ValueError("capture header must be t_us,cs,op,addr,data")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong column count")
                t_us, cs = int(row["t_us"], 10), int(row["cs"], 10)
                addr, data = int(row["addr"], 16), int(row["data"], 16)
                integer(addr, "address", 0, 255)
                integer(data, "data", 0, 65535)
                if row["op"] not in ("R", "W"):
                    raise ValueError("op must be R or W")
                accesses.append((t_us, cs, row["op"], addr, data))
            except ValueError as exc:
                raise ValueError("capture line {}: {}".format(line, exc)) from exc
    if not accesses:
        raise ValueError("capture has no accesses")
    return accesses


def read_strain(path):
    samples = []
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["t_us", "adc"]:
            raise ValueError("strain header must be t_us,adc")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong column count")
                samples.append((int(row["t_us"], 10), int(row["adc"], 10)))
            except ValueError as exc:
                raise ValueError("strain line {}: {}".format(line, exc)) from exc
    if not samples:
        raise ValueError("strain capture has no samples")
    return samples


def decode(capture_path, strain_path):
    accesses = read_capture(capture_path)
    read_strain(strain_path)
    read_addresses = sorted({addr for _, _, op, addr, _ in accesses if op == "R"})
    observed = ", ".join("0x{:02X}".format(addr) for addr in read_addresses) or "none"
    raise EvidenceError(
        "decode is unresolved: the rev-D map and an evidence-based card "
        "revision discriminator are missing. Observed read addresses: " +
        observed + ". Strain ADC counts have no documented transfer model "
        "that identifies register fields. No JSONL emitted; see driver.py's "
        "module documentation for the ambiguity.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    decoder = commands.add_parser("decode", help="validate inputs and report unresolved map")
    decoder.add_argument("--capture", required=True)
    decoder.add_argument("--strain", required=True)
    decoder.add_argument("--out", required=True)
    exciter = commands.add_parser("excite", help="generate verified rev-C excitation writes")
    exciter.add_argument("--spec", required=True)
    exciter.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "decode":
            decode(args.capture, args.strain)
        else:
            with open(args.spec, encoding="utf-8") as stream:
                writes = excitation_writes(json.load(stream))
            result = json.dumps({"writes": writes}, separators=(",", ":")) + "\n"
            with open(args.out, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(result)
    except (OSError, ValueError) as exc:
        print("driver.py: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
