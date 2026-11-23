#!/usr/bin/env python3
"""Build the frozen release bundle for security/sbom-vs-linkage v002.

Builder-stage only.  Compiles the pinned fixture, links the service with
--gc-sections/--print-gc-sections/-Map, strips it, stamps the deployed
build-ids into the vendored patch manifest and emits versions.lock,
sbom.spdx.json and scanner_report.json from the same lockfile.  Every byte
is deterministic: SOURCE_DATE_EPOCH is pinned, every translation unit gets a
fixed -frandom-seed, the link uses --build-id=sha1 and no fixture source
touches __DATE__ or __TIME__.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

CXX = os.environ.get("FIXTURE_CXX", "g++")
AR = os.environ.get("FIXTURE_AR", "ar")
STRIP = os.environ.get("FIXTURE_STRIP", "strip")
READELF = os.environ.get("FIXTURE_READELF", "readelf")

CFLAGS = [
    "-O2",
    "-ffunction-sections",
    "-fdata-sections",
    "-fno-asynchronous-unwind-tables",
    "-fPIC",
]

RPATH_ARG = "-Wl,-rpath,$ORIGIN/lib:$ORIGIN/plugins"
RPATH_DISPLAY = "-Wl,-rpath,'$ORIGIN/lib:$ORIGIN/plugins'"

SOURCE_DATE_EPOCH = "1743552000"          # 2025-04-02T00:00:00Z
BUILD_DATE = "2025-04-02T00:00:00Z"

NL = chr(10)

# translation unit -> -frandom-seed and source file
UNITS = [
    ("service", "service.cpp"),
    ("hpack", "hpack.cpp"),
    ("hpackver", "hpackver.cpp"),
    ("rotor", "rotor.cpp"),
    ("rotorversion", "rotorversion.cpp"),
    ("murmurx", "murmurx.cpp"),
    ("shim", "shim.cpp"),
    ("crcutil", "crcutil.cpp"),
    ("vaultcrypto", "vaultcrypto.cpp"),
    ("vhash", "vhash.cpp"),
    ("codecext", "codecext.cpp"),
    ("codecext_dbg", "codecext_dbg.cpp"),
    ("mathplug", "mathplug.cpp"),
]

# archive -> members, in the order the linker will see them
ARCHIVES = [
    ("libhpack.a", ["hpack.o", "hpackver.o"]),
    ("librotor.a", ["rotor.o", "rotorversion.o"]),
    ("libshim.a", ["shim.o"]),
    ("libcrcutil.a", ["crcutil.o"]),
    ("libmurmurx.a", ["murmurx.o"]),
]

# soname -> object, version script
DSOS = [
    ("libvaultcrypto.so.1", "vaultcrypto.o", "vaultcrypto.map"),
    ("libvhash.so.1", "vhash.o", "vhash.map"),
]

PLUGINS = [
    ("libcodecext.so", "codecext.o"),
    ("libcodecext_dbg.so", "codecext_dbg.o"),
    ("libmathplug.so", "mathplug.o"),
]

LOCKFILE = (
    "# acme-service release lockfile" + NL +
    "# frozen vendor bundle for release 3.7.0" + NL +
    "# component | version | supplier" + NL +
    "acme-service | 3.7.0 | Northwind Systems" + NL +
    "libhpack | 1.4.2 | Northwind Components" + NL +
    "librotor | 1.4.2 | Aurelia Labs" + NL +
    "libshim | 1.0.4 | Northwind Systems" + NL +
    "libcrcutil | 1.0.9 | Northwind Components" + NL +
    "libmurmurx | 2.1.0 | Aurelia Labs" + NL +
    "libvaultcrypto | 1.1.0 | Vaultworks" + NL +
    "libvhash | 1.1.0 | Vaultworks" + NL +
    "libcodecext | 0.9.3 | CodecWorks" + NL +
    "libstdc++ | 12.2.0 | FSF" + NL +
    "libgcc | 12.2.0 | FSF" + NL
)

SPDX_PACKAGES = [
    ("acme-service", "3.7.0", "Northwind Systems"),
    ("libhpack", "1.4.2", "Northwind Components"),
    ("librotor", "1.4.2", "Aurelia Labs"),
    ("libshim", "1.0.4", "Northwind Systems"),
    ("libcrcutil", "1.0.9", "Northwind Components"),
    ("libmurmurx", "2.1.0", "Aurelia Labs"),
    ("libvaultcrypto", "1.1.0", "Vaultworks"),
    ("libvhash", "1.1.0", "Vaultworks"),
    ("libcodecext", "0.9.3", "CodecWorks"),
    ("libstdc++", "12.2.0", "FSF"),
    ("libgcc", "12.2.0", "FSF"),
]

# advisory_id, cve, component, version, affected_symbol, severity, fixed_in
FINDINGS = [
    ("ADV-2025-0441", "CVE-2024-38971", "libhpack", "1.4.2",
     "hpack_decode", "high", "1.4.3"),
    ("ADV-2025-0442", "CVE-2023-51764", "librotor", "1.4.2",
     "rotor::rotor_decode(char const*, unsigned long)", "high", "1.4.3"),
    ("ADV-2025-0443", "CVE-2024-22017", "libmurmurx", "2.1.0",
     "murmurx_hash", "medium", "2.1.1"),
    ("ADV-2025-0444", "CVE-2024-33112", "libvaultcrypto", "1.1.0",
     "vaultcrypto_sign", "critical", "1.1.1"),
    ("ADV-2025-0445", "CVE-2024-33119", "libvhash", "1.1.0",
     "vhash_query", "high", "1.1.1"),
    ("ADV-2025-0446", "CVE-2024-40898", "libcodecext", "0.9.3",
     "codec_decode", "high", "0.9.4"),
]

CLAIM = "APPLICABLE"


def run(cmd, cwd=None, stderr_file=None):
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    env["LC_ALL"] = "C"
    if stderr_file is None:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    else:
        with open(stderr_file, "wb") as fh:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                  stderr=fh)
    if proc.returncode != 0:
        sys.stderr.write((proc.stdout or b"").decode("utf-8", "replace"))
        raise SystemExit("command failed (%d): %s"
                         % (proc.returncode, " ".join(cmd)))
    return (proc.stdout or b"").decode("utf-8", "replace")


def read_build_id(path):
    for line in run([READELF, "-n", str(path)]).splitlines():
        if "Build ID:" in line:
            return line.split("Build ID:", 1)[1].strip().split()[0].lower()
    return ""


def stray_build_id(tag):
    """A deterministic build-id belonging to a build that never shipped."""
    return hashlib.sha1(tag.encode("utf-8")).hexdigest()


def emit_metadata(out, deployed):
    """versions.lock, sbom.spdx.json, scanner_report.json, patch manifest."""
    out = pathlib.Path(out)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    (out / "versions.lock").write_text(LOCKFILE, encoding="utf-8")

    lock_hash = hashlib.sha256(LOCKFILE.encode("utf-8")).hexdigest()
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "acme-service-3.7.0",
        "documentNamespace": (
            "https://northwind.example/spdx/acme-service-3.7.0-"
            + lock_hash[:16]),
        "creationInfo": {
            "created": BUILD_DATE,
            "creators": ["Tool: lockfile2spdx 1.4.2",
                         "Organization: Northwind Systems"],
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-" + name.replace("+", "p"),
                "name": name,
                "versionInfo": version,
                "supplier": "Organization: " + supplier,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
            for name, version, supplier in SPDX_PACKAGES
        ],
    }
    (out / "sbom.spdx.json").write_text(json.dumps(sbom, indent=2) + NL,
                                        encoding="utf-8")

    report = {
        "scanner": "fleet-scan 5.2.0",
        "feed": "vendor-vulndb-2025.14.1",
        "scanned_target": "/app/release/service",
        "policy": ("report every component whose installed version is below "
                   "the fixed version"),
        "claim_vocabulary": ["APPLICABLE", "NOT_APPLICABLE", "UNKNOWN"],
        "findings": [
            {
                "advisory_id": advisory_id,
                "cve": cve,
                "component": component,
                "version": version,
                "affected_symbol": symbol,
                "severity": severity,
                "fixed_in": fixed_in,
                "claim": CLAIM,
            }
            for (advisory_id, cve, component, version, symbol, severity,
                 fixed_in) in FINDINGS
        ],
    }
    (out / "scanner_report.json").write_text(
        json.dumps(report, indent=2) + NL, encoding="utf-8")

    entries = [
        {
            "hunk_id": "VC-1.1.0-BACKPORT-0007",
            "component": "libvaultcrypto.so.1",
            "component_version": "1.1.0",
            "kind": "backport",
            "advisory_id": "ADV-2025-0444",
            "cve": "CVE-2024-33112",
            "source_file": "vaultcrypto.c",
            "symbol": "vaultcrypto_sign",
            "upstream_commit": "8c41d7be0f5a2d9c1b0e6f3a4d7c9e21b3a5f608",
            "build_id": deployed["libvaultcrypto.so.1"],
        },
        {
            "hunk_id": "VC-1.1.0-REBUILD-0008",
            "component": "libvaultcrypto.so.1",
            "component_version": "1.1.0",
            "kind": "rebuild",
            "advisory_id": "ADV-2025-0444",
            "cve": "CVE-2024-33112",
            "source_file": "vaultcrypto.c",
            "symbol": "vaultcrypto_sign",
            "upstream_commit": "1f7a0d2c93b45e6087ad4c1e2b9f3107d6c84a52",
            "build_id": stray_build_id("libvaultcrypto.so.1 superseded rebuild"),
        },
        {
            "hunk_id": "VH-1.1.0-BACKPORT-0003",
            "component": "libvhash.so.1",
            "component_version": "1.1.0",
            "kind": "backport",
            "advisory_id": "ADV-2025-0445",
            "cve": "CVE-2024-33119",
            "source_file": "vhash.c",
            "symbol": "vhash_query",
            "upstream_commit": "3b6e9f2c41d0a8579e6c1b3f508d2a4791c6e0fb",
            "build_id": stray_build_id("libvhash.so.1 stale patch record"),
        },
    ]
    manifest_body = {
        "manifest_format": "vendor-patch-manifest/1",
        "vendor": "Northwind Components",
        "release": "acme-service 3.7.0",
        "signed_by": "vendor-psirt-signing-key-3f9c71aa",
        "entries": entries,
    }
    canonical = json.dumps(manifest_body, sort_keys=True, separators=(",", ":"))
    manifest_body["signature"] = "sha256:" + hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    (out / "patches" / "patch_manifest.json").write_text(
        json.dumps(manifest_body, indent=2, sort_keys=True) + NL,
        encoding="utf-8")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/opt/fixture-src")
    ap.add_argument("--work", default="/opt/build")
    ap.add_argument("--out", default="/app/release")
    args = ap.parse_args(argv[1:])

    src = pathlib.Path(args.src)
    work = pathlib.Path(args.work)
    out = pathlib.Path(args.out)

    if work.exists():
        shutil.rmtree(work)
    if out.exists():
        shutil.rmtree(out)
    (work / "obj").mkdir(parents=True)
    (work / "lib").mkdir(parents=True)
    (work / "plugins").mkdir(parents=True)
    (out / "build").mkdir(parents=True)

    # 1. compile every translation unit with the pinned flags.
    for unit, rel in UNITS:
        run([CXX, "-c"] + CFLAGS + ["-frandom-seed=" + unit, "-I", str(src),
                                    "-o", "obj/%s.o" % unit, str(src / rel)],
            cwd=str(work))

    # 2. the static archives that stand on the link line.
    for archive, members in ARCHIVES:
        run([AR, "rcs", "lib/" + archive]
            + ["obj/" + m for m in members], cwd=str(work))

    # 3. the two version-scripted shared objects.
    for soname, obj, vscript in DSOS:
        run([CXX, "-shared", "-o", "lib/" + soname, "obj/" + obj,
             "-Wl,--version-script=" + str(src / vscript),
             "-Wl,-soname," + soname, "-Wl,--build-id=sha1",
             "-Wl,--as-needed", "-static-libgcc"], cwd=str(work))

    # 4. the deployed plugin directory.
    for filename, obj in PLUGINS:
        run([CXX, "-shared", "-o", "plugins/" + filename, "obj/" + obj,
             "-Wl,-soname," + filename, "-Wl,--build-id=sha1",
             "-Wl,--as-needed", "-static-libgcc"], cwd=str(work))

    # Every shipped ELF is stripped, the shared objects included; the
    # build-id note survives the strip, which is the whole point.
    for soname, _, _ in DSOS:
        run([STRIP, "--strip-all", "lib/" + soname], cwd=str(work))
    for filename, _ in PLUGINS:
        run([STRIP, "--strip-all", "plugins/" + filename], cwd=str(work))

    # 5. link the service.  -Llib needs the dev symlinks the vendor's build
    #    tree carried; they are not part of the shipped bundle.
    for soname, _, _ in DSOS:
        link = soname[:soname.index(".so")] + ".so"
        (work / "lib" / link).symlink_to(soname)

    link_cmd = [
        CXX, "-o", "service",
        "-Wl,--gc-sections", "-Wl,--print-gc-sections", "-Wl,-Map=service.map",
        "-Wl,--build-id=sha1", "-Wl,--enable-new-dtags", RPATH_ARG,
        "-static-libstdc++", "-static-libgcc",
        "obj/service.o",
        "lib/libhpack.a", "lib/librotor.a", "lib/libshim.a",
        "lib/libcrcutil.a", "lib/libmurmurx.a",
        "-Llib", "-lvaultcrypto", "-lvhash",
    ]
    run(link_cmd, cwd=str(work), stderr_file=str(work / "gc-sections.log.raw"))

    # 6. keep only the gc-section removal lines, then strip the executable.
    raw = (work / "gc-sections.log.raw").read_text(encoding="utf-8",
                                                   errors="replace")
    gc_lines = [ln for ln in raw.splitlines()
                if "removing unused section" in ln.lower()]
    (work / "gc-sections.log").write_text(NL.join(gc_lines) + NL,
                                          encoding="utf-8")
    (work / "gc-sections.log.raw").unlink()
    run([STRIP, "--strip-all", "service"], cwd=str(work))

    # 7. the deployed build-ids go into the vendored patch manifest.
    deployed = {}
    for soname, _, _ in DSOS:
        deployed[soname] = read_build_id(work / "lib" / soname)
        if not deployed[soname]:
            raise SystemExit("no build-id on " + soname)
    emit_metadata(out, deployed)

    # 8. the release tree itself.
    shutil.copy2(work / "service", out / "service")
    os.chmod(out / "service", 0o755)
    shutil.copy2(work / "service.map", out / "service.map")
    shutil.copy2(work / "gc-sections.log", out / "gc-sections.log")
    (out / "lib").mkdir(parents=True, exist_ok=True)
    for archive, _ in ARCHIVES:
        shutil.copy2(work / "lib" / archive, out / "lib" / archive)
    for soname, _, _ in DSOS:
        shutil.copy2(work / "lib" / soname, out / "lib" / soname)
    (out / "plugins").mkdir(parents=True, exist_ok=True)
    for filename, _ in PLUGINS:
        shutil.copy2(work / "plugins" / filename, out / "plugins" / filename)

    # 9. the exact link command, as it was run.
    display = [RPATH_DISPLAY if a == RPATH_ARG else a for a in link_cmd]
    (out / "build" / "link.txt").write_text(" ".join(display) + NL,
                                            encoding="utf-8")

    print("build_release: wrote %s" % out)
    for soname in sorted(deployed):
        print("  %-22s build-id %s" % (soname, deployed[soname]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
