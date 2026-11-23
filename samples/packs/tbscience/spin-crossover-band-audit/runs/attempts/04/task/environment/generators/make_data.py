#!/usr/bin/env python3
"""Build-time generator for the spin-crossover band-audit task.

Synthesises the sixteen variable-temperature spectra (12 Raman at 532/785/1064 nm
plus 4 IR) and the CSV metadata into /app, using the latent two-state Gaussian
model.  This script carries the latent parameter table, so it is build-only and
must not survive into the finished image.
"""
import csv
import os

import numpy as np

OUTDIR = os.environ.get("OUTDIR", "/app")

# ---- latent model ----------------------------------------------------------
# Overall high-spin fraction implied by the (synthetic) susceptibility.
GAMMA = {100: 0.06, 150: 0.26, 200: 0.55, 300: 0.84}
CHI_A = 0.5
CHI_B = 3.1
TEMPS = [300, 200, 150, 100]

# technique order used by the cross-section tuples below: Raman 532, 785, 1064, IR
TECHNIQUES = [
    ("raman_532nm", 0.45),
    ("raman_785nm", 0.45),
    ("raman_1064nm", 0.90),
    ("ir", 0.45),
]

# id, centre / cm-1, sigma / cm-1, true state, true low-spin-phase activity,
# LS-component cross sections (532, 785, 1064, IR), HS-component cross sections
BANDS = [
    ("B01", 233, 12, "LS", "raman", (150.0, 140.0, 110.0, 2.0), (0.0, 0.0, 0.0, 0.0)),
    ("B02", 257, 12, "HS", "neither", (0.0, 0.0, 0.0, 0.0), (6.0, 8.0, 10.0, 90.0)),
    ("B03", 418, 12, "LS", "ir", (8.0, 10.0, 12.0, 55.0), (0.0, 0.0, 0.0, 0.0)),
    ("B04", 447, 12, "HS", "neither", (0.0, 0.0, 0.0, 0.0), (9.0, 12.0, 14.0, 70.0)),
    ("B05", 507, 13, "LS", "raman", (45.0, 60.0, 55.0, 3.0), (0.0, 0.0, 0.0, 0.0)),
    ("B06", 534, 13, "HS", "neither", (0.0, 0.0, 0.0, 0.0), (7.0, 10.0, 12.0, 80.0)),
    ("B07", 566, 12, "BOTH", "raman", (25.0, 30.0, 28.0, 5.0), (6.0, 9.0, 12.0, 35.0)),
    ("B08", 762, 12, "LS", "raman", (40.0, 45.0, 38.0, 4.0), (0.0, 0.0, 0.0, 0.0)),
    ("B09", 784, 12, "HS", "neither", (0.0, 0.0, 0.0, 0.0), (6.0, 9.0, 11.0, 40.0)),
    ("B10", 1012, 12, "BOTH", "raman", (60.0, 60.0, 50.0, 6.0), (15.0, 22.0, 25.0, 25.0)),
    ("B11", 1291, 12, "LS", "raman", (120.0, 55.0, 35.0, 5.0), (0.0, 0.0, 0.0, 0.0)),
    ("B12", 1312, 12, "HS", "neither", (0.0, 0.0, 0.0, 0.0), (30.0, 12.0, 9.0, 45.0)),
    ("B13", 1472, 13, "LS", "raman", (160.0, 70.0, 40.0, 6.0), (0.0, 0.0, 0.0, 0.0)),
    ("B14", 1501, 13, "HS", "neither", (0.0, 0.0, 0.0, 0.0), (45.0, 18.0, 12.0, 50.0)),
    ("B15", 1601, 13, "BOTH", "raman", (55.0, 35.0, 25.0, 5.0), (20.0, 16.0, 15.0, 30.0)),
    ("B16", 2052, 9, "LS", "ir", (5.0, 6.0, 6.0, 75.0), (0.0, 0.0, 0.0, 0.0)),
    ("B17", 2076, 9, "HS", "neither", (0.0, 0.0, 0.0, 0.0), (5.0, 8.0, 8.0, 60.0)),
    ("B18", 2140, 10, "UNRESOLVABLE", "not_applicable", (1.2, 1.2, 1.2, 1.2), (1.0, 1.0, 1.0, 1.0)),
]

MASTER_SEED = 20260917
GRID = np.arange(150.0, 2201.0, 1.0)


def synth(technique_index, temperature):
    """Return the modelled intensity on GRID for one technique/temperature."""
    rng = np.random.default_rng(MASTER_SEED + 1000 * technique_index + temperature)
    gamma = GAMMA[temperature]
    signal = np.zeros_like(GRID)
    for _bid, centre, sigma, _state, _act, ls_cs, hs_cs in BANDS:
        amp = (1.0 - gamma) * ls_cs[technique_index] + gamma * hs_cs[technique_index]
        if amp == 0.0:
            continue
        signal += amp * np.exp(-0.5 * ((GRID - centre) / sigma) ** 2)
    # weak linear baseline, drawn once per spectrum
    b0 = rng.uniform(0.4, 1.8)
    b1 = rng.uniform(-8.0e-4, 8.0e-4)
    baseline = b0 + b1 * (GRID - GRID[0])
    sigma_noise = TECHNIQUES[technique_index][1]
    noise = rng.normal(0.0, sigma_noise, size=GRID.size)
    return signal + baseline + noise


def write_spectrum(path, intensity):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["wavenumber_cm1", "intensity"])
        for nu, val in zip(GRID, intensity):
            writer.writerow([int(round(nu)), "%.4f" % val])


def write_bands(path):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["band_id", "center_cm1"])
        for bid, centre, _sigma, _state, _act, _ls, _hs in BANDS:
            writer.writerow([bid, centre])


def write_susceptibility(path):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["T_K", "chi_M_T_cm3Kmol", "gamma_HS"])
        for temp in sorted(GAMMA):
            gamma = GAMMA[temp]
            writer.writerow([temp, "%.3f" % (CHI_A + CHI_B * gamma), "%.2f" % gamma])


def write_sheet(path, labels, markers):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["band_id", "center_cm1", "mode_label", "marker"])
        for row in BANDS:
            bid, centre = row[0], row[1]
            writer.writerow([bid, centre, labels[bid], markers.get(bid, "")])


# Mode labels as published on the two (mutually inconsistent) sheets.  They are
# deliberately plausible and, for the disputed bands, wrong.
SHEET_A_LABELS = {
    "B01": "delta(Fe-N-C) bending / lattice",
    "B02": "lattice mode",
    "B03": "nu(Fe-N) bending",
    "B04": "nu(Fe-N) stretch (HS)",
    "B05": "nu(Fe-N) stretch",
    "B06": "nu_sym(Fe-N) symmetric breathing",
    "B07": "ligand ring deformation",
    "B08": "pyridine ring breathing",
    "B09": "pyridine ring breathing (HS)",
    "B10": "ring breathing",
    "B11": "nu(ring) symmetric",
    "B12": "nu(ring)",
    "B13": "nu(C=C) / nu(C=N) ring stretch",
    "B14": "nu(C=C) ring stretch",
    "B15": "nu(C=N) ring stretch",
    "B16": "nu(C#N) nitrile stretch",
    "B17": "nu(C#N) nitrile stretch",
    "B18": "weak combination band",
}
SHEET_A_MARKERS = {"B01": "LS", "B07": "LS", "B10": "LS", "B11": "LS", "B13": "LS", "B16": "LS"}

SHEET_B_LABELS = {
    "B01": "lattice / bending mode",
    "B02": "nu(Fe-N) high-spin stretch",
    "B03": "delta(Fe-N-C) bending",
    "B04": "nu(Fe-N) stretch",
    "B05": "nu(Fe-N) bending",
    "B06": "nu_sym(Fe-N) stretch",
    "B07": "ligand ring deformation",
    "B08": "pyridine ring breathing",
    "B09": "pyridine ring breathing",
    "B10": "ring breathing",
    "B11": "nu(ring)",
    "B12": "nu(ring)",
    "B13": "nu(C=C) ring stretch",
    "B14": "nu(C=C) ring stretch",
    "B15": "nu(C=N) ring stretch",
    "B16": "nu(C#N) nitrile stretch",
    "B17": "nu(C#N) nitrile stretch",
    "B18": "weak combination band",
}
SHEET_B_MARKERS = {
    "B02": "HS", "B04": "HS", "B05": "HS", "B09": "HS",
    "B12": "HS", "B14": "HS", "B15": "HS", "B17": "HS",
}


def main():
    spectra_dir = os.path.join(OUTDIR, "spectra")
    sheets_dir = os.path.join(OUTDIR, "sheets")
    os.makedirs(spectra_dir, exist_ok=True)
    os.makedirs(sheets_dir, exist_ok=True)

    for j, (name, _sigma) in enumerate(TECHNIQUES):
        for temp in TEMPS:
            write_spectrum(
                os.path.join(spectra_dir, "%s_%dK.csv" % (name, temp)),
                synth(j, temp),
            )

    write_bands(os.path.join(OUTDIR, "bands.csv"))
    write_susceptibility(os.path.join(OUTDIR, "susceptibility.csv"))
    write_sheet(os.path.join(sheets_dir, "sheet_A.csv"), SHEET_A_LABELS, SHEET_A_MARKERS)
    write_sheet(os.path.join(sheets_dir, "sheet_B.csv"), SHEET_B_LABELS, SHEET_B_MARKERS)


if __name__ == "__main__":
    main()
