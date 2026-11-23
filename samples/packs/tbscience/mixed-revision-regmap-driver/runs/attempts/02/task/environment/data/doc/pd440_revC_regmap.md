# PD-440 four-channel piezo shaker driver - register map

Part: PD-440
Document: PD440-RM-C
Applies to: PD-440 revision C die

## 1. Bus and registers

The PD-440 sits on a shared service bus with one chip select per card. A
transfer is an 8-bit register address followed by a 16-bit data word,
most-significant byte first. A read returns the addressed register; a write
stores the data word into the addressed register, masked to the bits the
register actually implements. Bit positions are numbered upward from 0 (least
significant bit) inside the 16-bit word.

Addresses 0x10-0x7F are not decoded: they read 0x0000 and ignore writes.
Reserved bits inside a decoded register read 0 and are not stored. Registers
0x00 and 0x01 are read-only.

## 2. Register summary

| Addr | Name     | Access | Fields                                              |
|------|----------|--------|-----------------------------------------------------|
| 0x00 | ID       | R      | PART_ID[15:12], REV[11:8], DIE_TRIM[7:0]            |
| 0x01 | STATUS   | R      | FAULT[15:8], SEQ_STATE[7:0]                         |
| 0x02 | A_FREQ   | RW     | A_FREQ[7:0]                                         |
| 0x03 | A_GAIN   | RW     | A_GAIN[7:0]                                         |
| 0x04 | A_PHASE  | RW     | A_PHASE[7:0]                                        |
| 0x05 | B_FREQ   | RW     | B_FREQ[7:0]                                         |
| 0x06 | B_GAIN   | RW     | B_GAIN[7:0]                                         |
| 0x07 | B_PHASE  | RW     | B_PHASE[7:0]                                        |
| 0x08 | C_FREQ   | RW     | C_FREQ[7:0]                                         |
| 0x09 | C_GAIN   | RW     | C_GAIN[7:0]                                         |
| 0x0A | C_PHASE  | RW     | C_PHASE[7:0]                                        |
| 0x0B | D_FREQ   | RW     | D_FREQ[7:0]                                         |
| 0x0C | D_GAIN   | RW     | D_GAIN[7:0]                                         |
| 0x0D | D_PHASE  | RW     | D_PHASE[7:0]                                        |
| 0x0E | MASTER   | RW     | EN_A[0], EN_B[1], EN_C[2], EN_D[3], CLKDIV[7:4]    |
| 0x0F | SCRATCH  | RW     | SCRATCH[15:0]                                       |
| 0x10-0x7F | -   | -      | reserved (read 0x0000, writes ignored)              |

## 3. Field code scales

| Field     | Width | Scale                                                       |
|-----------|-------|-------------------------------------------------------------|
| *_FREQ    | 8 bit | 1 Hz per LSB. Channel frequency in Hz equals the code.      |
| *_PHASE   | 8 bit | 2 degrees per LSB. Channel phase in degrees equals code x 2.|
| *_GAIN    | 8 bit | Drive level is the raw code, valid range 1-63. 0 mutes.     |
| EN_x      | 1 bit | 1 = channel enabled, 0 = channel muted.                     |
| CLKDIV    | 4 bit | Master clock divider, 0 = divide-by-1.                      |
| SCRATCH   | 16 bit| Free scratch word, no hardware effect.                      |

## 4. ID word

| Field    | Bits  | Meaning                                  |
|----------|-------|------------------------------------------|
| PART_ID  | 15:12 | Part family. 0x5 = PD-440.               |
| REV      | 11:8  | Die revision. 0x3 = revision C.          |
| DIE_TRIM | 7:0   | Per-die trim, varies from part to part.  |

## 5. STATUS word

FAULT[15:8] is a latched fault code, 0 = no fault. SEQ_STATE[7:0] is the
hardware sequencer state and is not software settable.

## 6. Power-on state

Every writable register powers up as 0x0000, so MASTER reads 0x0000 (all
channels disabled) and every field code is 0.
