#!/usr/bin/env python3
"""PD-440: documented revision-C excitation, with explicit evidence limits.

This is NOT a complete implementation of the requested deliverable.  The
supplied evidence does not identify a unique revision-D register map, so
decode and revision-D excite deliberately fail without creating output.

Why the missing map cannot safely be filled in:

* The only reads in site_a/bus.csv address ID and STATUS.  There are no
  writable-register readbacks.  ID explicitly cannot identify the revision.
* Every sweep writes the same value to each parameter register.  In
  particular, 0x09 always receives 0x0A05 and 0x0D receives 0x005A.
* Consider two possible D maps, keeping A/B and C frequency/phase unchanged:
    X: C_GAIN = 0x09[15:8], D_GAIN = 0x09[7:0],
       D_PHASE = 0x0D[7:0], 0x0C has no fields.
    Y: C_GAIN = 0x09[15:8], D_GAIN = 0x0D[11:4],
       D_PHASE = 0x0C[7:0], 0x09[7:0] is reserved.
  Under both maps, all enabled channels have identical gain codes after
  EVERY captured write.  The differing D_GAIN initialization happens while
  D is disabled.  Both yield D_GAIN=5 when D is enabled.  Their phase codes
  differ, but no strain-to-frequency/phase measurement model was supplied;
  the recorded signal consists of noisy amplitude plateaus.  An amplitude
  measurement insensitive to phase is consistent with the evidence.
* Thus even the apparent gain evidence does not uniquely locate D_GAIN.
  X and Y require different fields in decode and different excite writes.
  Matching plateau timing across the independent clocks cannot resolve it.

A confirmed revision-D map, or additional probes with suitable readbacks
and calibrated physical measurements, is needed for the full contract.
The vendor's blanket compatibility claim cannot supply an evidence-based
revision discriminator or resolve an undocumented map difference.

No third-party modules, network, adjacent documentation, or sidecar files
are needed.  Successful output is deterministic.  Errors exit with code 2;
evidence and input-validation errors leave the output path untouched.
"""

import argparse
import csv
import json
import sys


CHANNELS = "ABCD"


class EvidenceError(ValueError):
    """The requested result would require an unsupported hardware assumption."""


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("{} must be an integer in {}..{}".format(
            name, lower, upper))
    return value


def excitation(spec):
    """Return writes for the documented C map, rejecting unsupported D."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    revision = spec.get("revision")
    if revision not in ("C", "D"):
        raise ValueError("revision must be C or D")
    if revision == "D":
        raise EvidenceError(
            "revision-D excitation is not identifiable from the supplied "
            "evidence: D_GAIN at 0x09[7:0] and at 0x0D[11:4] both fit the "
            "enabled gain observations but require different writes; "
            "a confirmed D map is required")
    cs = spec.get("cs")
    if type(cs) is not int or cs < 0:
        raise ValueError("cs must be a nonnegative integer")

    codes = {}
    for name, maximum in (("freq_hz", 255), ("gain", 63),
                          ("phase_deg", 510), ("enable", 1)):
        values = spec.get(name)
        if not isinstance(values, dict) or set(values) != set(CHANNELS):
            raise ValueError(name + " must contain exactly A, B, C, D")
        codes[name] = {}
        for channel in CHANNELS:
            value = integer(values[channel], name + "." + channel,
                            0, maximum)
            if name == "phase_deg":
                if value % 2:
                    raise ValueError("phase_deg." + channel +
                                     " must be a multiple of 2 degrees")
                value //= 2
            codes[name][channel] = value

    # Disable all channels and explicitly select divide-by-1.  Program
    # all fields, including those of channels that will remain disabled.
    writes = [[cs, 0x0E, 0]]
    for index, channel in enumerate(CHANNELS):
        base = 0x02 + 3 * index
        for offset, name in enumerate(("freq_hz", "gain", "phase_deg")):
            writes.append([cs, base + offset, codes[name][channel]])
    enables = sum(codes["enable"][channel] << index
                  for index, channel in enumerate(CHANNELS))
    writes.append([cs, 0x0E, enables])
    return {"writes": writes}


def check_csv(path, header):
    """Check the requested inputs before reporting the evidence limitation."""
    with open(path, "r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != header:
            raise ValueError("{}: expected header {}".format(
                path, ",".join(header)))
        count = 0
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError("{}:{}: incorrect column count".format(
                    path, line))
            count += 1
        if not count:
            raise ValueError(path + ": capture is empty")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    decode = commands.add_parser(
        "decode", help="validate inputs and report the missing map evidence")
    decode.add_argument("--capture", required=True)
    decode.add_argument("--strain", required=True)
    decode.add_argument("--out", required=True)
    excite = commands.add_parser(
        "excite", help="generate documented revision-C excitation writes")
    excite.add_argument("--spec", required=True)
    excite.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "decode":
            check_csv(args.capture, ["t_us", "cs", "op", "addr", "data"])
            check_csv(args.strain, ["t_us", "adc"])
            raise EvidenceError(
                "decode is unavailable: the supplied documentation and "
                "captures do not establish a unique revision-D map or a "
                "conclusive per-card revision discriminator. No output "
                "was written. See driver.py's module documentation for "
                "two incompatible maps consistent with the gain evidence")
        with open(args.spec, "r", encoding="utf-8") as stream:
            result = excitation(json.load(stream))
        serialized = json.dumps(result, separators=(",", ":")) + "\n"
        with open(args.out, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except (OSError, ValueError) as error:
        print("driver.py: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
