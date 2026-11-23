#!/usr/bin/env python3
"""Build-time generator for the site A post-swap bench capture.

Deterministic: fixed seeds, no wall-clock, no network, no third-party
imports.  Emits the two surviving instrument logs

    <out_dir>/bus.csv
    <out_dir>/strain.csv

from the true PD-440 device model (both die revisions), the bench sweep
schedule and the strain amplitude model.  This script is build-time only:
it never ships inside the solver image.
"""

import os
import sys

# --------------------------------------------------------------------------
# PD-440 device model
# --------------------------------------------------------------------------
ADDR_SPACE = 0x80
MASK16 = 0xFFFF

REG_ID = 0x00
REG_STATUS = 0x01
REG_MASTER = 0x0E
REG_SCRATCH = 0x0F

PART_ID = 0x5          # PD-440 family, bits[15:12] of the ID word
REV_CODE = 0x3         # die revision code the rev C document calls "C"

# One 16-bit word is written to every one of these registers on every pass.
SWEEP_WORDS = {
    0x02: 0x0078,      # A_FREQ  (120 Hz, 1 Hz/LSB)
    0x03: 0x0007,      # A_GAIN  (drive code 7)
    0x04: 0x002D,      # A_PHASE (90 deg, 2 deg/LSB)
    0x05: 0x00C8,      # B_FREQ  (200 Hz)
    0x06: 0x0009,      # B_GAIN  (drive code 9)
    0x07: 0x003C,      # B_PHASE (120 deg)
    0x08: 0x00F0,      # C_FREQ  (240 Hz)
    0x09: 0x0A05,      # C_GAIN / D_GAIN pack (placement is revision dependent)
    0x0A: 0x004B,      # C_PHASE (150 deg)
    0x0B: 0x00FA,      # D_FREQ  (250 Hz)
    0x0C: 0x0011,      # D_GAIN  (rev C only; reserved on rev D)
    0x0D: 0x005A,      # D_PHASE (180 deg)
}
SWEEP_REGS = tuple(range(0x02, 0x0E))

PASS_MASKS = (0x01, 0x02, 0x04, 0x08)   # master enable bits for A, B, C, D

# Revision of each card in the shipped capture: positions 1 and 3 came off a
# later-revision spare and were field-swapped into the rack.
CARD_REV = {0: "C", 1: "D", 2: "C", 3: "D"}

# Per-die trim, drawn independently of revision.
DIE_TRIM = {0: 0x5A, 1: 0x2E, 2: 0xC1, 3: 0x07}

# Bench timing, microseconds.
XFER_US = 8
SLOT_HOLD_US = 320000
BLANK_HOLD_US = 320000
INTERCARD_US = 1600000

# Strain logger: its own free-running clock, 250 Hz.
STRAIN_PERIOD_US = 4000          # 250 Hz
STRAIN_OFFSET_US = 337009        # strain_t = bus_t + offset (not slot aligned)

IDLE_ADC = 6
CH_ADC_BASE = 14
CH_ADC_PER_CODE = 40
NOISE_ADC = 6
LCG_SEED = 0x5EED4402

CH_ORDER = ("A", "B", "C", "D")
CH_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def field_mask(rev, addr):
    """Bits of `addr` that are implemented on `rev`; 0 means reserved."""
    if addr == REG_ID or addr == REG_STATUS or addr == REG_SCRATCH:
        return MASK16
    if addr == REG_MASTER:
        return 0x00FF                      # EN_A..EN_D [3:0], CLKDIV [7:4]
    if rev == "C":
        if 0x02 <= addr <= 0x0D:
            return 0x00FF                  # every channel field is [7:0]
        return 0x0000
    # rev D
    if addr == 0x09:
        return MASK16                      # C_GAIN [15:8], D_GAIN [7:0]
    if 0x02 <= addr <= 0x08 or addr in (0x0A, 0x0B, 0x0D):
        return 0x00FF
    return 0x0000                          # 0x0C is reserved on rev D


def writable(rev, addr):
    if addr < 0x02 or addr >= ADDR_SPACE:
        return False
    return field_mask(rev, addr) != 0x0000


class Card(object):
    """True device model for one die."""

    def __init__(self, rev, trim):
        self.rev = rev
        self.trim = trim & 0xFF
        self.regs = [0x0000] * ADDR_SPACE
        self.status = 0x000F               # FAULT=0, SEQ_STATE=0x0F

    def id_word(self):
        return (PART_ID << 12) | (REV_CODE << 8) | self.trim

    def write(self, addr, data):
        if not writable(self.rev, addr):
            return
        self.regs[addr] = data & field_mask(self.rev, addr)

    def read(self, addr):
        if addr == REG_ID:
            return self.id_word()
        if addr == REG_STATUS:
            return self.status
        if addr >= ADDR_SPACE:
            return 0x0000
        return self.regs[addr] & field_mask(self.rev, addr)

    def gain(self, ch):
        if ch == "A":
            return self.regs[0x03] & 0xFF
        if ch == "B":
            return self.regs[0x06] & 0xFF
        if ch == "C":
            if self.rev == "D":
                return (self.regs[0x09] >> 8) & 0xFF
            return self.regs[0x09] & 0xFF
        if self.rev == "D":
            return self.regs[0x09] & 0xFF
        return self.regs[0x0C] & 0xFF

    def enabled(self, ch):
        return (self.regs[REG_MASTER] >> CH_INDEX[ch]) & 0x1

    def drive_contrib(self):
        return sum(CH_ADC_BASE + CH_ADC_PER_CODE * self.gain(ch)
                   for ch in CH_ORDER if self.enabled(ch))


def build_bus():
    cards = {cs: Card(CARD_REV[cs], DIE_TRIM[cs]) for cs in range(4)}
    rows = []
    t = 0

    def read(cs, addr):
        nonlocal t
        rows.append((t, cs, "R", addr, cards[cs].read(addr)))
        t += XFER_US

    def write(cs, addr, data):
        nonlocal t
        cards[cs].write(addr, data)
        rows.append((t, cs, "W", addr, data & 0xFFFF))
        t += XFER_US

    for cs in range(4):
        read(cs, REG_ID)                                   # inventory probe
        write(cs, REG_MASTER, 0x0000)                      # all channels off
        for mask in PASS_MASKS:
            write(cs, REG_MASTER, mask)
            for reg in SWEEP_REGS:
                write(cs, reg, SWEEP_WORDS[reg])
                t += SLOT_HOLD_US
            write(cs, REG_MASTER, 0x0000)
            t += BLANK_HOLD_US
        read(cs, REG_STATUS)                               # closing status
        t += INTERCARD_US
    return cards, rows, t


def build_strain(rows, end_t):
    state = [Card(CARD_REV[cs], DIE_TRIM[cs]) for cs in range(4)]
    samples = []
    ri = 0
    n = len(rows)
    seed = LCG_SEED
    last = end_t + STRAIN_OFFSET_US + 4 * STRAIN_PERIOD_US
    k = 0
    while True:
        ts = k * STRAIN_PERIOD_US
        if ts > last:
            break
        tb = ts - STRAIN_OFFSET_US
        while ri < n and rows[ri][0] <= tb:
            _, cs, op, addr, data = rows[ri]
            if op == "W":
                state[cs].write(addr, data)
            ri += 1
        contrib = 0
        for card in state:
            contrib += card.drive_contrib()
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        noise = (seed % (2 * NOISE_ADC + 1)) - NOISE_ADC
        samples.append((ts, IDLE_ADC + contrib + noise))
        k += 1
    return samples


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/captures/site_a"
    os.makedirs(out_dir, exist_ok=True)

    _cards, rows, end_t = build_bus()

    with open(os.path.join(out_dir, "bus.csv"), "w") as fh:
        fh.write("t_us,cs,op,addr,data\n")
        for (t, cs, op, addr, data) in rows:
            fh.write("%d,%d,%s,0x%02X,0x%04X\n" % (t, cs, op, addr, data))

    samples = build_strain(rows, end_t)
    with open(os.path.join(out_dir, "strain.csv"), "w") as fh:
        fh.write("t_us,adc\n")
        for (ts, adc) in samples:
            fh.write("%d,%d\n" % (ts, adc))


if __name__ == "__main__":
    main()
