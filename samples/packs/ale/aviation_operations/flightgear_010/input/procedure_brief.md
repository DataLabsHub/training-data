# ILS Z RWY 28R at KSFO — Procedure Brief

This brief gives you what a real pilot would carry into the cockpit: the
narrative of the procedure, chart-published values that don't appear in
any FG dialog, and aircraft V-speeds. Airport / runway / navaid facts
that **are** discoverable from FlightGear's own GUI dialogs (Equipment →
Map, Equipment → Route Manager, Debug → Browse Internal Properties) are
**deliberately omitted** — Phase A grading checks that you can extract
those via the FG GUI, not by copying from this brief.

## What you're flying

- Aircraft: Cessna 172P (`c172p`).
- Airport: San Francisco International (KSFO).
- Departure runway: 10R (you'll take off here).
- Approach: ILS Z RWY 28R (you'll fly the procedure to the opposite end of the same physical runway).
- Final approach fix (FAF): named **CEPIN**.
- Localizer ident: **I-GWQ**.
- Alternate airport: KOAK (Oakland), measured later in Phase A.

## Chart-published values not visible in FG dialogs

These come from the FAA d-TPP IAP plate `00375IZ28R`. There is no FG GUI
dialog that displays them, so the brief gives them directly.

| Value | Basis |
|---|---|
| Glideslope angle: **3.00°** | FAA d-TPP plate IZ28R |
| Threshold crossing height (TCH): **55 ft** | FAA d-TPP plate IZ28R |
| FAF (CEPIN) to runway threshold: **5.4 NM** | FAA d-TPP plate IZ28R |

## What you must look up via FG GUI (NOT given here)

These are graded in Phase A as GUI-extraction tests; the brief intentionally
does not give the values. Use the indicated FG dialog:

- KSFO field elevation (ft MSL) — `Debug → Browse Internal Properties → /position/altitude-ft` while parked.
- KSFO ARP latitude / longitude (decimal degrees) — Property Browser at `/position/latitude-deg` and `/position/longitude-deg`.
- RWY 28R length (m and ft), width (ft), true bearing (°), threshold elevation (ft) — Equipment → Map (zoom to KSFO, click runway info) or Equipment → Route Manager runway data.
- Magnetic variation at KSFO — Property Browser at `/environment/magnetic-variation-deg`.
- Great-circle distance from KSFO ARP to KOAK ARP (NM) — Equipment → Map's ruler tool.

## Operational values (also in INPUT_RADIO_SETUP)

You set these via the cockpit panel; the values are pinned in
`task_description.txt` under `<!-- BEGIN INPUT_RADIO_SETUP -->`. They
appear here only because real pilots carry them on a kneeboard:

| Item | Value |
|---|---|
| NAV1 (ILS LOC/GS) | 111.70 MHz |
| NAV2 (SFO VOR-DME) | 115.80 MHz |
| NAV1 OBS (ILS final approach course, magnetic) | 284° |
| Transponder | 3662, mode ALT |
| Altimeter QNH | 29.99 inHg (from METAR A2999) |

## Aircraft V-speeds (Cessna 172P POH)

| Item | Value | Basis |
|---|---|---|
| Rotation speed (Vr) | 55 KIAS | Cessna POH; FlightGear wiki Cessna_172P |
| Landing speed (full flaps) | 60 KIAS | Cessna POH; FlightGear wiki |
| Cruise speed (typical) | 100–110 KIAS | Cessna POH |

## KAP140 autopilot limits (Bendix/King)

| Item | Value | Basis |
|---|---|---|
| Minimum AGL for use | 200 ft | KAP140 Pilot Guide; FlightGear wiki |
| IAS envelope | 80–160 KIAS | KAP140 Pilot Guide; FlightGear wiki |

## Weather (pinned)

```
KSFO 271156Z 30006KT 10SM FEW015 SCT023 SCT070 12/08 A2999 RMK AO2 SLP157 T01220078 10133 20122 55000 $
```
- Wind: 300° at 6 kt.
- Visibility: 10 SM. Few at 1500 ft; scattered at 2300 ft and 7000 ft.
- Temperature 12°C / Dewpoint 8°C.
- QNH: 29.99 inHg → set Kollsman to 29.99.

The METAR string is pinned for this benchmark run; do not replace it with live weather.
