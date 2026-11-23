#!/usr/bin/env python3
"""Deterministic reference analyzer for security/sbom-vs-linkage v002.

Reads the frozen release bundle and the binutils text produced from it:

  build/link.txt                what actually stood on the link line
  service.map                   the archive-inclusion table
  gc-sections.log               the sections --gc-sections removed
  lib/*.a lib/*.so.1            the link inputs and the deployed DSOs
  plugins/*.so                  the deployed plugin directory
  patches/patch_manifest.json   the vendored, vendor-signed patch manifest
  scanner_report.json           the findings to rule on

It derives advisory_id -> {verdict, witness, rejected_claim}.  The same code
runs as the builder-stage self-check (--selfcheck) and describes the witness
rules the verifier applies.  It never ships in the agent image.
"""
import argparse
import json
import pathlib
import re
import shlex
import subprocess
import sys

EXPECTED_VERDICTS = {
    "ADV-2025-0441": "APPLIES",
    "ADV-2025-0442": "GC_DISCARDED",
    "ADV-2025-0443": "NOT_LINKED",
    "ADV-2025-0444": "PATCHED_IN_PLACE",
    "ADV-2025-0445": "APPLIES",
    "ADV-2025-0446": "RUNTIME_BOUND",
}

# Rows whose deciding record is a name the fixture pins exactly.
EXPECTED_WITNESS = {
    "ADV-2025-0441": "libhpack.a(hpack.o)",
    "ADV-2025-0442": ".text._ZN5rotor11rotor_decodeEPKcm",
    "ADV-2025-0443": "libmurmurx.a",
    "ADV-2025-0446": "libcodecext.so",
}

# Rows whose deciding record is the build-id of the DSO the loader binds.
BOUND_SONAME = {
    "ADV-2025-0444": "libvaultcrypto.so.1",
    "ADV-2025-0445": "libvhash.so.1",
}

VERDICTS = ("APPLIES", "NOT_LINKED", "GC_DISCARDED", "PATCHED_IN_PLACE",
            "RUNTIME_BOUND")
CONTRADICTS = ("NOT_LINKED", "GC_DISCARDED", "PATCHED_IN_PLACE")

# librotor ships its decode entry points under the vendor's pre-release
# assembler names; the advisory names them the way a reviewer reads them.
ASSEMBLER_NAMES = {
    "rotor::rotor_decode(char const*, unsigned long)":
        "_ZN5rotor11rotor_decodeEPKcm",
    "rotor::rotor_decode_checked(char const*, unsigned long)":
        "_ZN5rotor18rotor_decode_checkedEPKcm",
}

QUOTES = chr(39) + chr(34)
INCLUSION_RE = re.compile(r"^(\S+\.a)\(([^)]+)\)\s+(\S+)\s+\((.*)\)\s*$")


def read_text(path):
    return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")


def read_bytes(path):
    return pathlib.Path(path).read_bytes()


def run(cmd):
    proc = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    return proc.stdout.decode("utf-8", "replace")


def readelf(path, *flags):
    return run(["readelf"] + list(flags) + [str(path)])


def canonical(token):
    """Canonical spelling of an archive member, section or build-id."""
    text = str(token).strip().strip(QUOTES).strip()
    if text.startswith("./"):
        text = text[2:]
    if text.startswith("lib/") and (".a(" in text or text.endswith(".a")):
        text = text[4:]
    return text


def parse_map_inclusions(map_text):
    """canonical archive -> [(canonical member token, symbol column)]."""
    lines = map_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("Archive member included"):
            start = index + 1
            break
    if start is None:
        return {}
    chunks = []
    for line in lines[start:]:
        if not line.strip():
            if chunks:
                break
            continue
        if line[:1].isspace():
            if chunks:
                chunks[-1] = chunks[-1] + " " + line.strip()
            continue
        chunks.append(line.strip())
    found = {}
    for chunk in chunks:
        match = INCLUSION_RE.match(chunk)
        if not match:
            continue
        archive, member, _referrer, symbol = match.groups()
        key = canonical(archive)
        found.setdefault(key, []).append(
            ("%s(%s)" % (key, member.strip()), symbol.strip()))
    return found


def parse_gc_sections(gc_text):
    """section -> set of input files the gc log blames."""
    marker = "removing unused section "
    separator = " in file "
    found = {}
    for line in gc_text.splitlines():
        low = line.lower()
        start = low.find(marker)
        if start < 0:
            continue
        stop = low.find(separator, start)
        if stop < 0:
            continue
        section = line[start + len(marker):stop].strip().strip(QUOTES)
        blame = line[stop + len(separator):].strip().strip(QUOTES)
        found.setdefault(section, set()).add(canonical(blame))
    return found


def parse_need(service):
    """DT_NEEDED sonames, in readelf -d order."""
    found = []
    for line in readelf(service, "-d").splitlines():
        if "(NEEDED)" in line and "[" in line and "]" in line:
            found.append(line.split("[", 1)[1].split("]", 1)[0].strip())
    return found


def parse_runpath(service):
    found = []
    for line in readelf(service, "-d").splitlines():
        if "(RUNPATH)" in line or "(RPATH)" in line:
            if "[" in line and "]" in line:
                found.append(line.split("[", 1)[1].split("]", 1)[0].strip())
    return found


def parse_verneed(service):
    """version name -> soname the service requires it from."""
    mapping = {}
    current = None
    for line in readelf(service, "-V").splitlines():
        if "File:" in line and "Cnt:" in line:
            current = line.split("File:", 1)[1].split("Cnt:", 1)[0].strip()
        elif current and "Name:" in line and "Flags:" in line:
            name = line.split("Name:", 1)[1].split("Flags:", 1)[0].strip()
            mapping.setdefault(name, current)
    return mapping


def parse_dyn_versions(service):
    """undefined imported symbol -> the version node it is bound to."""
    versions = {}
    for line in readelf(service, "--dyn-syms", "--wide").splitlines():
        if " UND " not in line or "@" not in line:
            continue
        tail = line.split(" UND ", 1)[1].strip()
        symbol, _, rest = tail.partition("@")
        fields = rest.split()
        if symbol and fields:
            versions.setdefault(symbol.strip(), fields[0].strip("()"))
    return versions


def read_build_id(path):
    for line in readelf(path, "-n").splitlines():
        if "Build ID:" in line:
            return line.split("Build ID:", 1)[1].strip().split()[0].lower()
    return ""


def binary_sections(path):
    return set(re.findall(r"\]\s+(\S+)\s+[A-Z]", readelf(path, "-S", "--wide")))


def exported_symbols(path):
    found = set()
    for line in run(["nm", "-D", "--defined-only", str(path)]).splitlines():
        fields = line.split()
        if fields:
            found.add(fields[-1].split("@")[0])
    return found


def archive_members(release, archive):
    """member file name -> set of symbols that member defines."""
    out = run(["nm", "--defined-only", "-A",
               str(pathlib.Path(release) / "lib" / archive)])
    members = {}
    for line in out.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3 or not parts[2].strip():
            continue
        member, symbol = parts[1].strip(), parts[2].split()[-1]
        members.setdefault(member, set()).add(symbol)
    return members


def member_defining(release, archive, symbol):
    for member, defined in archive_members(release, archive).items():
        if symbol in defined:
            return member
    return None


def find_archive(release, component):
    path = pathlib.Path(release) / "lib" / (component + ".a")
    return path.name if path.is_file() else None


def find_dso(release, component):
    for path in sorted((pathlib.Path(release) / "lib").glob("*.so*")):
        if path.name.startswith(component + ".so"):
            return path.name
    return None


def find_plugin(release, component):
    path = pathlib.Path(release) / "plugins" / (component + ".so")
    return path.name if path.is_file() else None


def analyze(release):
    release = pathlib.Path(release)
    service = release / "service"
    findings = json.loads(read_text(release / "scanner_report.json"))["findings"]

    link_tokens = shlex.split(read_text(release / "build" / "link.txt"))
    link_archives = [canonical(t) for t in link_tokens if t.endswith(".a")]
    link_basenames = [pathlib.Path(t).name for t in link_tokens]

    inclusions = parse_map_inclusions(read_text(release / "service.map"))
    gc_sections = parse_gc_sections(read_text(release / "gc-sections.log"))

    needed = parse_need(service)
    runpath = parse_runpath(service)
    verneed = parse_verneed(service)
    dyn_versions = parse_dyn_versions(service)
    sections = binary_sections(service)
    image = read_bytes(service)

    manifest = json.loads(read_text(release / "patches" / "patch_manifest.json"))
    patched_ids = set(str(e.get("build_id", "")).lower()
                      for e in manifest["entries"])

    plugin_dir = release / "plugins"
    plugins = sorted(p.name for p in plugin_dir.glob("*.so"))
    requested = [name for name in plugins if name.encode() in image]

    rulings = {}
    for finding in findings:
        component = finding["component"]
        symbol = finding["affected_symbol"]
        claim = finding.get("claim")
        mangled = ASSEMBLER_NAMES.get(symbol, symbol)
        section = ".text." + mangled

        archive = find_archive(release, component)
        dso = find_dso(release, component)
        plugin = find_plugin(release, component)

        if archive:
            if archive not in link_archives:
                raise SystemExit("%s is not on the link line" % archive)
            included = inclusions.get(archive, [])
            if not included:
                verdict, witness = "NOT_LINKED", archive
            else:
                blamed = gc_sections.get(section, set())
                owner = member_defining(release, archive, mangled)
                if blamed:
                    if owner is None:
                        raise SystemExit("%s lost a section but defines no %s"
                                         % (archive, mangled))
                    if not any(blame.endswith("(%s)" % owner)
                               for blame in blamed):
                        raise SystemExit("%s: blamed on %s, not %s"
                                         % (section, sorted(blamed), owner))
                    if section in sections:
                        raise SystemExit("%s lost %s but the binary kept it"
                                         % (archive, section))
                    verdict, witness = "GC_DISCARDED", section
                else:
                    if not owner:
                        raise SystemExit("%s does not define %s"
                                         % (archive, mangled))
                    if section not in read_text(release / "service.map"):
                        raise SystemExit("%s is not placed by the link"
                                         % section)
                    verdict = "APPLIES"
                    witness = "%s(%s)" % (archive, owner)
                    if witness not in [entry for entry, _ in included]:
                        raise SystemExit("%s is not an inclusion entry of %s"
                                         % (witness, archive))

        elif dso:
            if dso not in needed:
                raise SystemExit("%s is not bound by the service" % dso)
            if symbol not in exported_symbols(release / "lib" / dso):
                raise SystemExit("%s does not export %s" % (dso, symbol))
            version = dyn_versions.get(symbol)
            if version and verneed.get(version) != dso:
                raise SystemExit("%s is bound through %s, not %s"
                                 % (symbol, verneed.get(version), dso))
            build_id = read_build_id(release / "lib" / dso)
            if not build_id:
                raise SystemExit("no build-id on %s" % dso)
            verdict = ("PATCHED_IN_PLACE" if build_id in patched_ids
                       else "APPLIES")
            witness = build_id

        elif plugin:
            if plugin not in requested:
                raise SystemExit("%s is not the plugin the loader requests"
                                 % plugin)
            if not any("$ORIGIN/plugins" in entry for entry in runpath):
                raise SystemExit("plugins/ is not on the loader search path")
            for token in link_basenames:
                if token.startswith(component):
                    raise SystemExit("%s stands on the link line" % component)
            verdict, witness = "RUNTIME_BOUND", plugin

        else:
            raise SystemExit("no deployed artifact carries %s" % component)

        if verdict not in VERDICTS:
            raise SystemExit("bad verdict %r" % verdict)
        rulings[finding["advisory_id"]] = {
            "verdict": verdict,
            "witness": canonical(witness),
            "rejected_claim": claim if verdict in CONTRADICTS else None,
        }
    return rulings


def selfcheck(release, rulings):
    """Assert the six expected rulings and the six expected records."""
    release = pathlib.Path(release)
    problems = []

    actual = dict((key, value["verdict"]) for key, value in rulings.items())
    if actual != EXPECTED_VERDICTS:
        for key in sorted(set(actual) | set(EXPECTED_VERDICTS)):
            if actual.get(key) != EXPECTED_VERDICTS.get(key):
                problems.append("%s: expected %s, analyzer says %s"
                                % (key, EXPECTED_VERDICTS.get(key),
                                   actual.get(key)))

    for key, expected in EXPECTED_WITNESS.items():
        if key in rulings and rulings[key]["witness"] != expected:
            problems.append("%s: expected witness %s, got %s"
                            % (key, expected, rulings[key]["witness"]))

    for key, soname in BOUND_SONAME.items():
        deployed = read_build_id(release / "lib" / soname)
        if key in rulings and rulings[key]["witness"] != deployed:
            problems.append("%s: witness is not the bind of %s" % (key, soname))

    inclusions = parse_map_inclusions(read_text(release / "service.map"))
    gc_sections = parse_gc_sections(read_text(release / "gc-sections.log"))
    link_tokens = shlex.split(read_text(release / "build" / "link.txt"))
    link_archives = [canonical(t) for t in link_tokens if t.endswith(".a")]

    shim = inclusions.get("libshim.a", [])
    if not any(symbol == "murmurx_hash" for _, symbol in shim):
        problems.append("the map does not read libshim.a(shim.o) -> "
                        "murmurx_hash")
    if "libmurmurx.a" in inclusions:
        problems.append("the map carries a libmurmurx.a inclusion")
    if "libshim.a" not in link_archives or "libmurmurx.a" not in link_archives:
        problems.append("the shim and libmurmurx.a are not both on the line")
    elif link_archives.index("libshim.a") > link_archives.index("libmurmurx.a"):
        problems.append("libshim.a does not stand before libmurmurx.a")

    hpack = [entry for entry, _ in inclusions.get("libhpack.a", [])]
    if len(hpack) != 2 or "libhpack.a(hpack.o)" not in hpack:
        problems.append("the map does not carry both libhpack.a members: %s"
                        % hpack)

    if "librotor.a(rotor.o)" not in gc_sections.get(
            ".text._ZN5rotor11rotor_decodeEPKcm", set()):
        problems.append("the gc log does not blame rotor_decode on rotor.o")
    if "obj/service.o" not in gc_sections.get(
            ".text._ZL21hpack_decode_fallbackPKvm", set()):
        problems.append("the gc log does not carry the service's hpack shim")

    service_map = read_text(release / "service.map")
    sections = binary_sections(release / "service")
    if ".text.hpack_decode" not in service_map:
        problems.append("hpack_decode is not placed into the shipped binary")
    if ".text._ZN5rotor11rotor_decodeEPKcm" in sections:
        problems.append("the shipped binary kept rotor_decode")
    if b"librotor 1.4.2" not in read_bytes(release / "service"):
        problems.append("the librotor banner did not survive the strip")

    plugins = sorted(p.name for p in (release / "plugins").glob("*.so"))
    image = read_bytes(release / "service")
    requested = [name for name in plugins if name.encode() in image]
    if len(plugins) != 3 or requested != ["libcodecext.so"]:
        problems.append("plugins %s, requested %s" % (plugins, requested))
    if not any("$ORIGIN/plugins" in entry
               for entry in parse_runpath(release / "service")):
        problems.append("$ORIGIN/plugins is not on the loader search path")

    manifest = json.loads(read_text(release / "patches" / "patch_manifest.json"))
    ids = [str(e.get("build_id", "")).lower() for e in manifest["entries"]]
    vault = read_build_id(release / "lib" / "libvaultcrypto.so.1")
    vhash = read_build_id(release / "lib" / "libvhash.so.1")
    vault_entries = [e for e in manifest["entries"]
                     if e.get("component") == "libvaultcrypto.so.1"]
    vhash_entries = [e for e in manifest["entries"]
                     if e.get("component") == "libvhash.so.1"]
    if len(vault_entries) != 2 or vault not in ids:
        problems.append("the manifest does not carry two libvaultcrypto "
                        "build-ids including the deployed one")
    if len(vhash_entries) != 1 or vhash_entries[0].get("build_id") == vhash:
        problems.append("the manifest's libvhash build-id is the deployed one")

    if problems:
        for problem in problems:
            print("SELF-CHECK FAILED: " + problem, file=sys.stderr)
        return 1
    print("SELF-CHECK OK: six advisories, six expected rulings, six "
          "expected witnesses")
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?", default="/app/release")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    rulings = analyze(args.release)

    if args.json:
        rows = [dict(advisory_id=key, **value)
                for key, value in sorted(rulings.items())]
        print(json.dumps({"advisories": rows}, indent=2))
        return 0

    for key in sorted(rulings):
        row = rulings[key]
        print("%s  %-15s  %s" % (key, row["verdict"], row["witness"]))

    if not args.selfcheck:
        return 0
    return selfcheck(args.release, rulings)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
