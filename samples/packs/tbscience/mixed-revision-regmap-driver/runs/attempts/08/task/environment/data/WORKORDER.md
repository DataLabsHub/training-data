# WORK ORDER - PD-440 driver set, line 2 vibration rack

## Situation

Line 2 of the mill runs a four-head shaker rack driven by a bank of
discontinued PD-440 piezo-driver cards on a shared SPI-like service bus, one
chip select per card. Two positions in the rack were field-swapped to
later-revision dies pulled from a spare chassis. The only paperwork that came
with those dies is a short vendor addendum, and it claims the register map did
not change between the two revisions. The only map document left on the
machine covers the earlier revision. We have run out of options for keeping
these cards alive and need a driver we can put on the box ourselves.

## What survived

Two logs came back from a single post-swap bench session on the rack, plus a
hand-written example of the excitation request the process engineers hand us.
`captures/site_a/bus.csv` is the bus tool's trace of every transfer on the
service bus during that session. `captures/site_a/strain.csv` is the strain
gauge channel recorded on the same rig over the same session. Each instrument
ran on its own free-running clock; there is no common time base between the
two files and no per-access index anywhere. That is all the provenance that
came back with the files.

## Deliverable

Exactly one file, `/app/driver.py`: Python 3, standard library only, no
third-party packages, no network dependency, deterministic (two runs over the
same inputs must produce byte-identical output). It exposes two subcommands.

### decode

```
python3 /app/driver.py decode --capture <bus.csv> --strain <strain.csv> --out <decode.jsonl>
```

Writes one JSON object per bus access, one object per line, in capture order.
Each object carries:

    {"t_us": <int>, "cs": <int>, "op": "R"|"W", "addr": <int>,
     "revision": "C"|"D", "fields": {"<FIELD_NAME>": <int>, ...}}

`t_us`, `cs`, `op` and `addr` are that access's own values from the capture.
`revision` is the die revision of that access's card. `fields` maps every
canonical field name of the addressed register, as named in the map document,
to that field's integer code as it sits in the transfer word (apply the map's
code scales for physical units). A register that has no defined fields on that
revision, and any reserved address, yields `{}`. Each card's revision must be
established from the evidence in the capture itself, not from the probe ID and
not from a fixed card-number table.

### excite

```
python3 /app/driver.py excite --spec <spec.json> --out <writes.json>
```

The excitation spec is:

    {"revision": "C"|"D", "cs": <int>,
     "freq_hz":  {"A": <int>, "B": <int>, "C": <int>, "D": <int>},
     "gain":     {"A": <int>, "B": <int>, "C": <int>, "D": <int>},
     "phase_deg":{"A": <int>, "B": <int>, "C": <int>, "D": <int>},
     "enable":   {"A": <int>, "B": <int>, "C": <int>, "D": <int>}}

Writes:

    {"writes": [[<cs>, <addr>, <data>], ...]}

with integer `cs`, `addr` and `data`, in the order the writes must be applied.
Replaying those writes in order against a die of the revision named in the
spec, starting from its power-on defaults, must leave every channel's
frequency in Hz, drive level, phase in degrees and enable flag at exactly the
value requested. Use that revision's own register and field placement and the
code scales given in the map document, and never write a read-only register.

## Log column conventions

Every CSV has exactly one header line. Columns are comma separated with no
quoting and no spaces.

`captures/site_a/bus.csv` - header `t_us,cs,op,addr,data`

| Column | Convention | Meaning                                              |
|--------|------------|------------------------------------------------------|
| t_us   | decimal    | microseconds on the bus tool's own clock             |
| cs     | decimal    | chip select / card number                            |
| op     | R or W     | R = read transfer, W = write transfer                |
| addr   | hex, 0xNN  | register address                                     |
| data   | hex, 0xNNNN| transfer word: value returned on R, value driven on W|

`captures/site_a/strain.csv` - header `t_us,adc`

| Column | Convention | Meaning                                              |
|--------|------------|------------------------------------------------------|
| t_us   | decimal    | microseconds on the strain logger's own clock        |
| adc    | decimal    | strain ADC counts for that sample                    |

## Files staged

    WORKORDER.md                     this order
    doc/pd440_revC_regmap.md         surviving vendor map document (rev C)
    doc/revD_addendum.txt            vendor addendum for the later die
    captures/site_a/bus.csv          post-swap bench bus trace
    captures/site_a/strain.csv       concurrently recorded strain log
    excitation_spec.json             example excitation request

There is no driver, decoder or analysis script on the box, only `python3` and
coreutils. Nothing you create under `/app` other than `/app/driver.py` is kept.
