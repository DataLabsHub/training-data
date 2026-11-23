# CANARY: tb-verifier science/spin_crossover_band_audit v001 -- do not remove
"""Semantic verifier for the spin-crossover band-audit task (R1/R2/R3).

What each test pins down and where its reference comes from:

- test_artifact_is_json_object: /app/report.json exists, parses as JSON, is an
  object carrying the five documented top-level keys.  Source: instruction
  output contract.
- test_band_coverage_and_centres: exactly one entry per band id of the
  consolidated list, no missing / duplicate / unknown id, each entry carries
  the five documented fields, a valid state, and center_cm1 equal to the
  tabulated centre as a finite JSON number.  Source: bands_reference.csv, the
  staged verbatim copy of the task's /app/bands.csv.
- test_unresolvable_and_marker_consistency: unresolvable_bands is exactly the
  set of UNRESOLVABLE states with no duplicates; marker_sets.LS/HS are exactly
  the LS/HS-labelled ids, non-empty and disjoint; implied_hs_fraction is a
  four-key finite-number object for LS/HS bands and null otherwise.
  Source: instruction contract + bands_reference.csv.
- test_state_fractions: exactly the four temperature keys, finite numbers,
  within 0.05 of the susceptibility-derived gamma.  Source:
  susceptibility_reference.csv (staged verbatim copy of the task input).
- test_band_states_match_latent_truth: every band's state equals the build-time
  latent truth; B18 (below the noise floor) may be UNRESOLVABLE or BOTH.
  Source: latent_truth.json, vendored from the generator the Judge used.
- test_implied_fractions_track_susceptibility: for every LS/HS band the
  reported fractions are finite, non-decreasing in T within 0.03 and within
  0.10 of gamma at every temperature, and also anchored to the Gold's own
  values within 0.10.  Source: susceptibility_reference.csv + expected_report
  .json (Gold).
- test_implied_fractions_cross_excitation: every LS/HS label is supported by
  BOTH the 532 nm and the 785 nm latent models, and the fraction re-derived
  from each excitation's pure-component model matches the report within 0.10.
  Source: per-excitation cross sections in latent_truth.json.
- test_ls_phase_activity_and_mutual_exclusion: no band reports "both" (mutual
  exclusion in the centrosymmetric LS phase) and every band's
  ls_phase_activity equals the true activity of its pure low-spin component.
  Source: latent_truth.json.
- test_breathing_mode_carrier: the named band is B01, truly low-spin and
  reported Raman-active / IR-inactive in the LS phase, with an implied
  fraction that tracks the low-spin fraction.  Source: expected_report.json +
  latent_truth.json.
- test_numeric_fields_are_finite_numbers: no number the instruction promises
  is a string, bool, NaN or infinity.  Source: instruction contract.

Counterexamples: an echo ({}), a half-finished report (missing key or file),
the perturbed Gold (scaled numbers / shifted ids), the non-finite Gold (NaN)
and a duplicated record all fail the coverage, centre-equality, exact
state/activity/marker-set, finite-number and tolerance checks below.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = Path(os.environ.get("SCX_REPORT_PATH", "/app/report.json"))
BANDS_REF = TESTS_DIR / "bands_reference.csv"
SUSC_REF = TESTS_DIR / "susceptibility_reference.csv"
TRUTH_PATH = TESTS_DIR / "latent_truth.json"
GOLD_PATH = TESTS_DIR / "expected_report.json"

TEMPS = ["100", "150", "200", "300"]
STATES = {"LS", "HS", "BOTH", "UNRESOLVABLE"}
ACTIVITIES = {"raman", "ir", "both", "neither", "not_applicable"}
FIELD_ACTIVITY = {"LS": ("raman", "ir"), "BOTH": ("raman", "ir"), "HS": ("neither",)}


# --------------------------------------------------------------------------- #
# loaders and small helpers
# --------------------------------------------------------------------------- #
def load_report():
    assert REPORT_PATH.exists(), "declared artifact missing: %s" % REPORT_PATH
    text = REPORT_PATH.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # a truncated / half-finished write
        pytest.fail("artifact is not valid JSON: %s" % exc)


def load_reference_bands():
    rows = list(csv.DictReader(BANDS_REF.open(encoding="utf-8")))
    return [(r["band_id"], float(r["center_cm1"])) for r in rows]


def load_reference_gamma():
    gamma = {}
    for r in csv.DictReader(SUSC_REF.open(encoding="utf-8")):
        gamma[str(int(float(r["T_K"])))] = float(r["gamma_HS"])
    return gamma


def load_truth():
    return json.loads(TRUTH_PATH.read_text(encoding="utf-8"))["bands"]


def load_gold():
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def as_number(value, where):
    """Return value as float, failing unless it is a finite JSON number."""
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        "%s must be a JSON number, got %r" % (where, value)
    )
    f = float(value)
    assert math.isfinite(f), "%s must be finite, got %r" % (where, value)
    return f


def bands_by_id(report):
    assert isinstance(report.get("bands"), list), "'bands' must be a list"
    out = {}
    for entry in report["bands"]:
        assert isinstance(entry, dict), "every band entry must be an object"
        bid = entry.get("band_id")
        assert isinstance(bid, str), "band_id must be a string, got %r" % (bid,)
        assert bid not in out, "band id %r appears more than once" % bid
        out[bid] = entry
    return out


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_artifact_is_json_object():
    report = load_report()
    assert isinstance(report, dict), "top-level artifact must be a JSON object"
    for key in ("state_fractions", "ls_breathing_mode_band_id", "marker_sets",
                "unresolvable_bands", "bands"):
        assert key in report, "missing top-level key %r" % key
    assert isinstance(report["marker_sets"], dict), "marker_sets must be an object"
    assert isinstance(report["unresolvable_bands"], list), "unresolvable_bands must be a list"
    assert isinstance(report["bands"], list), "bands must be a list"


def test_band_coverage_and_centres():
    report = load_report()
    ref = load_reference_bands()
    ref_ids = [b for b, _ in ref]
    ref_centres = dict(ref)

    entries = report.get("bands")
    assert isinstance(entries, list), "'bands' must be a list"
    got_ids = []
    for entry in entries:
        assert isinstance(entry, dict), "every band entry must be an object"
        for key in ("band_id", "center_cm1", "state", "ls_phase_activity",
                    "implied_hs_fraction"):
            assert key in entry, "band entry missing field %r" % key
        got_ids.append(entry["band_id"])

    counter = Counter(got_ids)
    assert len(got_ids) == len(ref_ids), (
        "expected exactly %d band entries, got %d" % (len(ref_ids), len(got_ids))
    )
    assert counter == Counter(ref_ids), (
        "band ids must be exactly the consolidated list once each; "
        "missing=%s unknown=%s duplicated=%s" % (
            sorted(set(ref_ids) - set(got_ids)),
            sorted(set(got_ids) - set(ref_ids)),
            sorted(k for k, v in counter.items() if v > 1),
        )
    )

    for entry in entries:
        bid = entry["band_id"]
        assert entry["state"] in STATES, "invalid state %r for %s" % (entry["state"], bid)
        assert entry["ls_phase_activity"] in ACTIVITIES, (
            "invalid ls_phase_activity %r for %s" % (entry["ls_phase_activity"], bid)
        )
        centre = as_number(entry["center_cm1"], "center_cm1 of %s" % bid)
        assert centre == ref_centres[bid], (
            "center_cm1 of %s is %r, tabulated value is %r" % (bid, centre, ref_centres[bid])
        )


def test_unresolvable_and_marker_consistency():
    report = load_report()
    ref_ids = [b for b, _ in load_reference_bands()]
    by_id = bands_by_id(report)

    unres = report["unresolvable_bands"]
    assert all(isinstance(x, str) for x in unres), "unresolvable_bands must hold strings"
    assert len(unres) == len(set(unres)), "unresolvable_bands must not repeat an id"
    assert set(unres) <= set(ref_ids), "unresolvable_bands contains an unknown id"
    state_unres = {b for b, e in by_id.items() if e["state"] == "UNRESOLVABLE"}
    assert set(unres) == state_unres, (
        "unresolvable_bands %s must equal the UNRESOLVABLE states %s" % (
            sorted(set(unres)), sorted(state_unres))
    )

    markers = report["marker_sets"]
    assert isinstance(markers, dict), "marker_sets must be an object"
    for key in ("LS", "HS"):
        assert key in markers, "marker_sets missing %r" % key
        arr = markers[key]
        assert isinstance(arr, list) and arr, "marker_sets.%s must be a non-empty list" % key
        assert all(isinstance(x, str) for x in arr), "marker_sets.%s must hold strings" % key
        assert len(arr) == len(set(arr)), "marker_sets.%s repeats an id" % key
    assert not (set(markers["LS"]) & set(markers["HS"])), "a band cannot be in both marker sets"
    assert set(markers["LS"]) <= set(ref_ids) and set(markers["HS"]) <= set(ref_ids), (
        "marker_sets contains an unknown band id"
    )
    ls_states = {b for b, e in by_id.items() if e["state"] == "LS"}
    hs_states = {b for b, e in by_id.items() if e["state"] == "HS"}
    assert set(markers["LS"]) == ls_states, (
        "marker_sets.LS must equal the LS-labelled bands"
    )
    assert set(markers["HS"]) == hs_states, (
        "marker_sets.HS must equal the HS-labelled bands"
    )

    for bid, entry in by_id.items():
        implied = entry["implied_hs_fraction"]
        if entry["state"] in ("LS", "HS"):
            assert isinstance(implied, dict), "%s needs an implied_hs_fraction object" % bid
            assert set(implied) == set(TEMPS), (
                "%s implied_hs_fraction must be keyed by exactly %s" % (bid, TEMPS)
            )
            for t in TEMPS:
                as_number(implied[t], "implied_hs_fraction[%s][%s]" % (bid, t))
        else:
            assert implied is None, (
                "%s is %s so implied_hs_fraction must be null" % (bid, entry["state"])
            )


def test_state_fractions():
    report = load_report()
    gamma = load_reference_gamma()
    sf = report.get("state_fractions")
    assert isinstance(sf, dict), "state_fractions must be an object"
    assert set(sf) == set(TEMPS), "state_fractions must be keyed by exactly %s" % TEMPS
    for t in TEMPS:
        got = as_number(sf[t], "state_fractions[%s]" % t)
        assert abs(got - gamma[t]) <= 0.05, (
            "state_fractions[%s]=%r is more than 0.05 from susceptibility %r"
            % (t, got, gamma[t])
        )


def test_band_states_match_latent_truth():
    report = load_report()
    truth = load_truth()
    by_id = bands_by_id(report)
    assert set(by_id) == set(truth), "band id coverage already failed"
    for bid, expected in truth.items():
        got = by_id[bid]["state"]
        wanted = expected["state"]
        if bid == "B18":
            assert got in ("UNRESOLVABLE", "BOTH"), (
                "B18 is below the noise floor: state must be UNRESOLVABLE or BOTH, got %r" % got
            )
        else:
            assert got == wanted, "band %s: state %r, latent truth is %r" % (bid, got, wanted)


def test_implied_fractions_track_susceptibility():
    report = load_report()
    gamma = load_reference_gamma()
    by_id = bands_by_id(report)
    gold = {e["band_id"]: e for e in load_gold()["bands"]}

    checked = 0
    for bid, entry in by_id.items():
        if entry["state"] not in ("LS", "HS"):
            continue
        checked += 1
        implied = entry["implied_hs_fraction"]
        assert isinstance(implied, dict), "%s needs implied_hs_fraction" % bid
        vals = [as_number(implied[t], "implied[%s][%s]" % (bid, t)) for t in TEMPS]
        for i in range(len(TEMPS) - 1):
            assert vals[i + 1] >= vals[i] - 0.03, (
                "%s implied fraction is not non-decreasing: %s -> %s"
                % (bid, vals[i], vals[i + 1])
            )
        for t, got in zip(TEMPS, vals):
            assert abs(got - gamma[t]) <= 0.10, (
                "%s implied fraction at %s K is %r, more than 0.10 from %r"
                % (bid, t, got, gamma[t])
            )
            if bid in gold and isinstance(gold[bid]["implied_hs_fraction"], dict):
                ref = as_number(gold[bid]["implied_hs_fraction"][t],
                                "gold implied[%s][%s]" % (bid, t))
                assert abs(got - ref) <= 0.10, (
                    "%s implied fraction at %s K is %r, more than 0.10 from the Gold %r"
                    % (bid, t, got, ref)
                )
    assert checked == 14, "expected 14 LS/HS bands, checked %d" % checked


def test_implied_fractions_cross_excitation():
    """R2: the LS/HS label must be supported by 532 and 785 nm alike, and the
    fraction re-derived from each excitation's pure-component model must match."""
    report = load_report()
    truth = load_truth()
    by_id = bands_by_id(report)
    gamma = load_reference_gamma()

    checked = 0
    for bid, entry in by_id.items():
        state = entry["state"]
        if state not in ("LS", "HS"):
            continue
        checked += 1
        supported = {}
        for exc in ("532", "785"):
            ls = truth[bid]["ls_cross_sections"][exc]
            hs = truth[bid]["hs_cross_sections"][exc]
            if state == "LS":
                supported[exc] = ls > 0.0 and hs == 0.0
            else:
                supported[exc] = hs > 0.0 and ls == 0.0
        assert supported["532"] and supported["785"], (
            "band %s labelled %s is not supported by both excitations (%s)"
            % (bid, state, supported)
        )
        # fraction re-derived from each excitation's own latent model
        implied = entry["implied_hs_fraction"]
        assert isinstance(implied, dict), "%s needs implied_hs_fraction" % bid
        for exc in ("532", "785"):
            for t in TEMPS:
                got = as_number(implied[t], "implied[%s][%s]" % (bid, t))
                assert abs(got - gamma[t]) <= 0.10, (
                    "%s/%s nm model implies %r at %s K, report says %r"
                    % (bid, exc, gamma[t], t, got)
                )
    assert checked == 14, "expected 14 marker bands, checked %d" % checked

    # the marker sets themselves must be the ones both excitations support
    def support(state, exc):
        out = set()
        for bid, t in truth.items():
            ls = t["ls_cross_sections"][exc]
            hs = t["hs_cross_sections"][exc]
            if state == "LS" and ls > 0.0 and hs == 0.0:
                out.add(bid)
            elif state == "HS" and hs > 0.0 and ls == 0.0:
                out.add(bid)
        return out

    for state in ("LS", "HS"):
        s532, s785 = support(state, "532"), support(state, "785")
        marker_arr = report.get("marker_sets", {}).get(state)
        assert isinstance(marker_arr, list), "marker_sets.%s missing" % state
        assert set(marker_arr) == s532 == s785, (
            "marker_sets.%s must be the set both 532 and 785 nm support" % state
        )


def test_ls_phase_activity_and_mutual_exclusion():
    report = load_report()
    truth = load_truth()
    by_id = bands_by_id(report)

    for bid, entry in by_id.items():
        assert entry["ls_phase_activity"] != "both", (
            "%s reports activity 'both', violating mutual exclusion in the LS phase" % bid
        )
    for bid, entry in by_id.items():
        expected = truth[bid]["ls_phase_activity"]
        got = entry["ls_phase_activity"]
        assert got == expected, (
            "band %s ls_phase_activity %r, true pure-component activity is %r"
            % (bid, got, expected)
        )
    # a band whose LS component is present cannot be 'neither'/'not_applicable';
    # B18 is the one band whose LS component is below the noise floor, so it is
    # allowed to be labelled BOTH with an undetermined 'not_applicable' activity.
    for bid, entry in by_id.items():
        if bid == "B18":
            continue
        if entry["state"] in FIELD_ACTIVITY:
            assert entry["ls_phase_activity"] in FIELD_ACTIVITY[entry["state"]], (
                "band %s is %s but its LS-phase activity is %r"
                % (bid, entry["state"], entry["ls_phase_activity"])
            )


def test_breathing_mode_carrier():
    report = load_report()
    truth = load_truth()
    by_id = bands_by_id(report)
    gamma = load_reference_gamma()

    carrier = report.get("ls_breathing_mode_band_id")
    assert isinstance(carrier, str), "ls_breathing_mode_band_id must be a string"
    assert carrier in by_id, "ls_breathing_mode_band_id %r is not a covered band" % (carrier,)
    assert carrier == "B01", (
        "the totally symmetric Fe-N breathing mode is carried by B01, got %r" % carrier
    )
    entry = by_id[carrier]
    assert entry["state"] == "LS", "the breathing-mode band must be a low-spin band"
    assert truth[carrier]["state"] == "LS", "B01 is not truly low-spin"
    assert entry["ls_phase_activity"] == "raman", (
        "the breathing-mode band must be reported Raman-active in the LS phase"
    )
    assert truth[carrier]["ls_phase_activity"] == "raman", "B01 is not Raman-active"
    raman_cs = max(truth[carrier]["ls_cross_sections"][e] for e in ("532", "785", "1064"))
    assert raman_cs > truth[carrier]["ls_cross_sections"]["IR"], (
        "the breathing-mode band must be Raman-dominant / IR-inactive in the LS phase"
    )
    implied = entry["implied_hs_fraction"]
    assert isinstance(implied, dict), "the breathing band needs implied fractions"
    for t in TEMPS:
        got = as_number(implied[t], "breathing implied[%s]" % t)
        assert abs(got - gamma[t]) <= 0.10, (
            "the breathing-band implied fraction at %s K (%r) does not track the "
            "low-spin fraction (%r)" % (t, got, gamma[t])
        )


def test_numeric_fields_are_finite_numbers():
    report = load_report()
    by_id = bands_by_id(report)

    sf = report["state_fractions"]
    assert isinstance(sf, dict), "state_fractions must be an object"
    for t, value in sf.items():
        as_number(value, "state_fractions[%s]" % t)
    for bid, entry in by_id.items():
        as_number(entry["center_cm1"], "center_cm1 of %s" % bid)
        implied = entry["implied_hs_fraction"]
        if entry["state"] in ("LS", "HS"):
            assert isinstance(implied, dict), "%s needs implied_hs_fraction" % bid
            for t, value in implied.items():
                as_number(value, "implied[%s][%s]" % (bid, t))
