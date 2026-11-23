# CANARY-VERIFIER: security/cms-trust-adjudicator/v001 -- semantic suite (do not reuse).
'''Semantic verifier for the CMS trust-store acceptance adjudicator (R1-R4).

The graded deliverable is a program (/app/adjudicate.py), so this suite executes
it: only that one file is copied into a scratch tree, the graded corpus is
rebuilt from the task's own pinned build-time generator (vendored verbatim in
this file, zlib compressed, sha256 pinned to generators/generate_corpus.py), and
the program is run as 'nobody' against a scratch corpus so the artifact can
neither read the expected answers under /tests nor write anywhere the reward
depends on. Reference values come from that generator (its per-object canonical
DER, its independent model of the appliance for verdict/offset, and its label
list), plus the Gold-frozen reference NDJSON digests pinned below.

Every test, what it pins down, and where the reference comes from:

* test_r1_program_is_single_self_contained_stdlib_artifact -- one file of
  parseable Python, importing no process/network module and calling no
  os-level process spawner [instruction R1].
* test_r1_exits_zero_and_emits_ndjson_for_every_split -- exit 0 and stdout that
  is exactly one JSON object per line, for both graded splits [R1].
* test_r1_one_record_per_input_file_in_ascending_order -- the object list equals
  the sorted *.der basenames: exact count, exact order, no duplicates [R1].
* test_r1_record_schema_and_types -- the four documented keys with the
  documented types, first_rejecting_element an int (not bool) or null [R1].
* test_r1_non_der_files_ignored_and_empty_corpus_clean -- only *.der inputs are
  consumed (decoy ignored) and an empty corpus prints nothing [R1].
* test_r2_canonical_der_re_encodes_the_same_abstract_value -- canonical_der is
  padded standard-alphabet base64 in canonical form whose bytes equal the
  generator's reference DER byte for byte, over all 111 objects [R2; generator].
* test_r3_verdict_matches_the_appliance_reference -- accept/reject equals the
  generator's independent model of the appliance, with the two not-a-blanket-rule
  anchors asserted on the reference itself (unsorted SET OF rejected inside a
  Certificate, tolerated in the CMS envelope) and both verdict classes present
  [R3; generator model()].
* test_r4_first_rejecting_element_matches_reference -- the offset equals the
  reference for every rejected object, is null exactly when accepted, and every
  reported offset is the tag octet of a real element of the input [R4].
* test_no_duplicate_records_and_exact_counts -- one record per input, no repeat
  [anti-cheat: duplicated agent].
* test_output_is_deterministic_across_runs -- a second run reproduces the
  artifact's own stdout [R1].
* test_reference_corpus_matches_pinned_gold_digests -- the vendored generator
  still emits the Gold's frozen reference NDJSON for both splits [anchor].
'''
from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
# The single declared artifact.  ADJUDICATE_ARTIFACT is a dry-run hook for the
# verifier author only; the pipeline never sets it.
ARTIFACT = Path(os.environ.get('ADJUDICATE_ARTIFACT', '/app/adjudicate.py'))
RUN_TIMEOUT = 300  # measured ~0.3 s per corpus; 300 s is ~1000x headroom

# The task's own build-time corpus generator, vendored verbatim but stored
# zlib-compressed so the suite stays compact.  Its sha256 is the digest the task
# published for generators/generate_corpus.py, so the reference data cannot
# drift silently, and the corpus it emits is byte-identical to the graded one.
_GENERATOR_ZLIB_B64 = (
    'eNq1fWl34zay6Hf9Co7m9BuyW1JL3jrxjXKO263O+Exvp+3kzFxfPYUSYZsjiVSTlJeXmf/+asFK'
    'grI9ye3EtkQUCoVCoRagAP75T6+3ZfF6nmavRXYbbB6qmzzb73S73bfbdJX0q3QtgkVebLZlcC0y'
    'UcRVXgRX8FPdiOD043lQFduy6pfwWATxYiE2VZwtRFDF5XLQ6VzcpGVQLop0UwXwSdyLxbYSSZBm'
    'ZZoIQvJyji2J4mVQVvG1CPIregzkpEWerUUGNddQMOicEY5M3IoCaNqkhKfKCRwADGBnAiAPQT7/'
    'p1hQnU2RJ9sFwFd3KRCXQ/FxpxMEL4O4DGLog6B2T84/DUZBlieiDBZxUTyk2XUv2EBz+KwXxFnQ'
    'FdkiT+A5UPuwEl1oPK4AUxAkYpFixbsbAQQxg7Aatn9XpFUlMkCa5Vm6iFcroK4A6oM8Uz2WSFbp'
    'HLksAAJg+7oGMn1dElbkLWKN5/m2QqoS1RUsfTv5+vrd5Gswf6gEEFkgrUgkV8WeUheA0iqnERJB'
    'Ia5EIXDYgDFJuqhe51dXpahooKHThpGLfL2hAZw/QK3+Ji5Kxi86LxltQg2XL4O7tLrBymmWiI2A'
    'XzA+K2DDSgRrgFupgY43m1WKMtOTQzt/6JBQScTQziLdCO6DGcgbsR4EwYUu/0sZ5HcZSNgGSI2r'
    'NIeWy86iyMuyv7gRiyVUiq9jELyKUTER8IHkL0BJJ15Sm9jPdAVVfmWZLn/twByJQaiQCV6k87y6'
    '6QUli2N1g4yGwcvyKkiK9KqCfsZFBQz/uQQZPabRljNKzHiGDTYPQb8PYxq8O/saXPb7JXCmCm7T'
    'Mp2vxL9uxCqBwn+B9EypuucfVJIEB683cXXzuspfywdTwiiA4k9TnOCdTrre5AUQVlzjOAr1fR6X'
    '4uhAfftnmWfqc16qTwUwKl+rb+V2DgOzEKUuLx/KTud8MnkXjIPh/eH+weHB6cHR4d7B4eF+EPw5'
    'uErvRXIcdM8nH95/nZwDMT9/OvuFgIfDzunF3+njd8NO58/Akj/gH8/qTqezWMVlGXwKWaYjHghg'
    'x0mAkgzCefHhFwIedKhomSYlDvonnKk0H0AI03VapbcoL0G4AWGA8XkIxHpTPUQg46WcOCBFIBjF'
    'dgETZkDI3tcfB59PLyYXr9+eXQTnF1/PPv0kZRPEH+cXTymc2VabSlfoR4iwIvVXBkuBmjYLfsXS'
    'Xweqd9yX2axc5VU5mwF3w+5iVXZ7QTfbrvEPEoV/b+MV/sFu41/QAvkVfkiuVlW315S7LjaEACtW'
    'iPAJlAdhRFRRR+q1K2g9zdJqNgtLsbrqBdA8zPjtukf8GL+PVyV0DuqMkdU9Yrz8SERIiCYFSJkE'
    'RFrkRyZn3F2nGRCzUHjlgOM/JGMAVAAz4Lf7GOiCx/C7Bg2UIjj8cQsAPTyH3+5jEp4xdcUtoB5B'
    'Cf11i7A7UIJ/3ALsHBTgH7eAuwpF/MEthMGAEux6rStMMv6BWYHjA9Z3HtKoYP9obCS35syi4F9B'
    'OLzfGwbpFYEEAgYkGEYEA8+QaT8E+yPD4kJU2yJjixBezgEBwEy5Auq5cUAPh/ej96zSEnzEH+9u'
    'QAEjvEGXDMBYgCkJsaX/A9XevI90IT778cdx8IYRDQo0JqUIGQIn5D1OjOTyuD+aGpxAhsJ6T6R8'
    'N9T0qYLkEqrwU6dHABJJ5q1ENg+J/cA45qBkHjBGjs846KIxvOqa1kEXCVCXzHAuxUkiEN812E+w'
    'Ypa+6TYY2/0foLfbbGcTJ1YrS+DqOr4PR70gzAZzmISMPoyCV8GbKHj9OvgOP45aBg6ZArxZThEo'
    'G1T5jAuWQPM8ve4aCYDxR+A2CcgkF3GYLTytRDn4vTRBnTCJiLAEhuLd5P3k61cyOyBGPDRga2YJ'
    'ejiFSEKj8H9Czcm+HuncFX6XfqnFc/ArYIpsyVdNCyAiRs8FvaNygIpVCVeagS3vBTc5+rIoaIoU'
    'wwtgEBfzVEZfFrwDnJrHjlZDdZ5mWzNhb5ICOkTzUyKgaao+k/DILzhpdb0FmnKoKctQFbwijgHC'
    'SH4kwZUAKynABMKPoO+R01Jk0Jc34NeEsuPUVqScWN37y2M1DAwNcwOKq1iOQ8ZEjfmhFiNSnIYp'
    'yN8l8rReYKhYKryyNbCc2BYRpYf8XBRpDPZZBL9mv/bQccwL8uVFvLghWfgLetVlvgIfF/ktR1oP'
    'dCFWRkFpQpV7YOia58kDiTgqWYKjD2bAWXXOu2ZGu+aQKvOjAT3MBtroYFXLim1RO+OkgIAlfgij'
    'p3ENeqK0G442YLEGVuGFcUE+Am+HUb1vPA2x3n/QBUugawaHpXIOGsEWRmwzklrVkpsxjW9zMJrT'
    'ingBI47s+H/pJmRYFIFV1CJONHec6VIYPadtrkenSz2FfXzF3HpFino4xJ9uxwsjxRbi2Rl4LOGt'
    'JbQQwm2gQrLNkhhjKBGTzObgP4Kbf1Xkawxh7/K/QEwCYctKUMxMSksLbor+tGVWsVe3UdCHgh+D'
    'kaOibi/TKfYLPXGKh+AB2oZpQ7czYhASYzcgfLUxvH/vYvhx/BQUZU0fzkHpLm22AT6tVihADjPD'
    'L4x9letMfJJh5hx87nghGYMKgEL8VIAUgHDQpATa8zVG6olhHA72ojZ7DXnK8x5LqKfOBfJlO5pj'
    '2Q6LYDfh+H7PbcKvqmz0T9dWzBb0nccBRW44zBm7zeNgr2nPuAkl3vJJ9Lu1H8bUpVHJj6o9tIYs'
    'MUtX3UGHRLtF1k0pnSkilxXkx7t1GL6EaNjSyYYVwM3BP3PgBcH9J5yghilasIcKB0K3MpYQDeWk'
    '4wH5fZcqliEUqgw1blIVA4R80vkjAnW5/lZur+NCOQ3iW/gSh1HOb0V+iELXC0ZHMnq8KLYqZMT4'
    'O6QqyheA7zYO9IYNED/icVqKhzFJR+Rt7E2zsaUKTvGZam9RPcDDkMbrd7V7evF3O0R+vN173Wid'
    'X+2oLuH3VKHI0yRs4fURB+cAMUODFRr+Wo+45gZnJMz18D4ygRfMBVzOCruDbjS1IkDjulxuLofT'
    '4GVwMATB2lyOplMTu90iis3l3vHUNVZeo2SFbrfRbte6rJau+mD7eFtzCqqlRtgIPPHfrRV6qgpu'
    'AOoEoVBcC0N3hqK1Qqz9aEA6z3PwFMB1apk5PJrol1xddZGVc6XlyUtRaLLtahX6URwqFBqYXJK2'
    'Fg8kuCYwrcAih/NesM22pUjGQ3+9fVmPwz6GpXhPYwKrmFXxfCXaRHf0PeMoB7ygFmrR3VZX37XW'
    '2muvtcDl4raKe/ttFddLnBW3PTAOCa9nqTgoUC4Qyx/OTyMcVfFQ84hwscnEz5mMk0EppNcZsJK1'
    'gt+HIjN4j0vCwWcQ0KtVfjcpirxwW8iMUwayAeRawQ2uHkoxIcHBWcvT0IjQ+/ddGiMvf2qSQK50'
    'yFssuCpvFk7wm2c9g5Zn9aoXFRpHR9ep++W1WlwsSYjLmbUqK2lZ3Gyz5fh741p+FbiR4ywPNBdw'
    'aUfJXuIFU2sCyPgOTTpSorxFNOP4XfpO+4ZglvWeirigLnC6R39Hx9OmD4Rw4Bmjr02kT3lhAhVO'
    'EWfXIhy60RTCRFP0fi9hEhuEctXyUs7QDYZ/hGpDShibI+01hYY0ED2Fh2o6Sx3f4q5hH55AKYA9'
    'hVDWO5s6kQxIzJW+FM4L81AuqEIj5hmvimr3iJ7Zq7h/iJ+T4mZYepWCeeh0Pp+9m52f/fRp8m72'
    '7uTiBEVzNNgbfHcwHIxG+4cH3w9GgzeDvS4B7oAYMcT5X0/2Do8QZm8wOmKgwWg4GuwPDqCSgdr/'
    '7qAVSrY2OZ19+fnth7PT2d8m/3CaHQ4PDg22L1/PPk6g1V9GTaB9JE6he3d+YhHoAh4AqGz39BNT'
    'Rg/lI/PkiJ98ME8k/s/myWgoH/1sPZPUvtW49oBz3/PDv/1sPzyUXPrbmf30gJ+eOE/3JezEwbAv'
    'STr9+fzi80fu7P7gCHhxgD+jvf2DwfeK7pOL2ennTxeTTxezi398mXjH93vFCAD+ODk/P/lpMnt3'
    '9tPk/KIF/ECDo3CBYppdwCC1AJs+zC4+nM/OJ19/gSjaovoQ/nuDY2kAJx9Pzj74YA6UUp1la4gY'
    '1us8wxD7GjVDWsF0TqsH13Be6hl9GRLXepZN7/583o2iaa8G89mGAewRKAou+NkuwSY9tU8/2UBM'
    'pMbwwSlDahUCtegwKzdiEZa0nAgditGgVzegsdC99nYYnG201fBHPNS9c9xxl8jGBufYQjtW6BtR'
    'aVqWW1GMkdXdj6JYPAQ/0abyKviaQ0h4eoI7cW7BX/Nyk1a4zde2i0zm7nSVcuLBJLtOMyEwkQCR'
    'nSNvrkF5rZJu1ERRbmljdbxr8Ju1Ntv5Kl2QOzo86KLjwO5emEI8sP8GHhD7InK+37+vm4ujg2jq'
    'wUq8Vp4jfQHH8WWwN0TP43TyFebGl8npOZoQ3h+lYR3eD0cno7d7p+DGcZw0avDwecx9Ei+pmeH9'
    'aCQ7oonZe3cwOXx/pIgZNto8AxnAla4/nB4iaG+vTtD+cDTcG+7LJaZewBuu3Q/xPDgH/xNJOVFJ'
    'HcFw9McShc718H5/v07Uwcnw7fB0+ESi9p5K1BeYePkqv35oUrL/Bik5OKhTcnh4eHT05k2dkq+0'
    '21AG5zI7Zf+pJPxVxKvqJjjLMAOI636Ms/ia1nkRyVfwQ4sqx0WaQxqyw8M6TUenp+/eTSZ1mn6B'
    'cBJm0lexiqmDn/KiurkDzzh4l8bXWV5W6aJ0qTnLKnFdMBnnorhNF4ISBd7HKXwRd/D5iBhzdARE'
    'TDt/TAZHIJPESuWvr67TJMxRQRZre2OfgpVi3VznlLoWF5RwkSOPnPDZPGaUKjQpkmyGazlxdWsv'
    '41wqeFwKu2W/M6zoI+gkBJ7+L6wrEQ9mWQyRJ9BVCz5ppexS0VswTbTfgbBTF4e4r0RWwhCWIUpI'
    'L/i2TYsl7YdcS7TfoJv8dHAtOBMBasm11o7tfevdsG9hF0AGtCE4mC8GixhiQIzEoOF4u6q6bgyp'
    'Vig4DjZFKgNChpiOt6+WQKxdJyD5stHulGJUnP9nb+X2MQWH2NvLLgIc+/FKmojteruHa0kbDLi9'
    '68Q2Go7za9U46g/rXFLlkQ50o8hZXW6tUN/Rq/PBEAzMQPI4PrN4Ml9QDopaY20bSICwW+N4fY5i'
    '4wPcNTQIwGOzcAhZQFyNEofcWYAyG+ADNAY1CWJoCNJkQKfhoRXa2JHwlM0UaanVy/b31QxDupDj'
    'AAitgQzV1BjHlAgxz2Y0Bcb8tN1dgo6O4aenKBvLv+TujesUypyo5RaFXK6CgayfDFGB+wdhubU7'
    'ZuePOOsWy20r8wEDy8HWnRVt3PkbuNEIq7kjpwTVKpe4ochhN0s5eVfdqZ96AH8S+QDXSj/ioA7A'
    'h6f1AKK3HlKq+Z1i2olcCh/KJbrHiY818e4iFM8CQNqTs7aN8lhRzrBPoPyEKIdpiQv5inxB8qIM'
    'TzNcA0tkF1B4Fvm7JFCc6j0RONpNsNZuCSVR4qkiNUGZAmjVowWmGI9xe4b1Jfh0coV0pD/ttXRh'
    'sS2rfI37bnJwthmaWmGPEOKnxRvPUnytMxY66hNWdRVUKZmPJY+Q5FGWi7LXDvw4Vd8UVU9jNK86'
    'wNwtFattJwGr6TQZVQ1cKZXIoya8dBVrqlA+RY1HTqSrFJX+k1Co/SznTK/Xkael/QbVwpNNMply'
    'l67/FdcCWoF+tm8L03C55ko7HhheqsQ1WvTFTBrJYrmtI261EsWFTvlMZ2bSNjXV4CKVUqUIVdjE'
    'rZkjkv+tGlbcusIGA8Rz+NYl2WaIqmPLELsLtkuJbHiqM8k2hFY6MDPRcpb4oe0rEXvlY9tFsqag'
    'A3LcaY6OwkrmQy6wqN4yLek1RBRQzHFFfbGy2RjBN6c5P+814CIfUVzGRMl6DlG8tIPSavx+5hKX'
    'uAaLcMqCyC+ynCrGMKQXL+v1pjbsTtVpV+MubGtS3yALO9BkGZf1vLDtbVI5tavqu+PJi1A+3ski'
    'D/NUyU7uSSCXfbrmM/mnaWkwULm5aZJWD9LyqF3H7h4uwIyG9O+/u2Aqdcm+W9LoosLYHAZV0vPA'
    '+kjXpUS7qe0Ow4b8RCTenljWpoLxWcweAq7DSrdYRlC0OIhKYeiZ95unOpcAyMOmts5rg7HRHqYF'
    'WesR2Mu7Isb8Bky22O89JZ5ukIwGGLF4fDBG3/OA+kjWpazDVWVXiED6cK2GKR7W/apqrhwbCdgL'
    'lIZUSknNMDPIUm57NMA9xZRGPwF3s4vwsOdC+HqGBdQpAneF6taETUDiLL+q2Y6oZ51JsDRxIz5s'
    'F5XbFoXNKFhIahaTfJJmaw0LcdvzQD2vubo1Jh7xaO0RL4AD2AW5u24XOEtGbcZO1yPeClHbrrCX'
    '5ve+D14FBNKyMv9mFNHa0+9eBFSHQsUq3wjH86Cch6J8nvORslF4snVdlzAg8H8yYNO000ZI5MZI'
    '1Gs/w07UqlqmwgQCXhrJTWo6KExar63C7uYRRDosCo8k41kunUH6iGPXBDxupc/r5FnExVVVWOuV'
    '8agWTdd2YXEOVboQ993VshwhUs5xPJJnHtJrUVZmUcSnmOhcy+Xx/p4UrXWiA2Cu7uk7NjZYJ4On'
    'Ky8bVZ1PLjZiFYM74uR0sMYid+8Zw3TsRPR4RXsXmnnb5s8oPpcxGS3KpQS7RchbOFS2LwKU8VPm'
    'FyNh2Ymby8aN1hrzKu41gZ7T1LPDECX4jwciLmTbBPcEIy5lj8o2jKF/9l5nMRiOJ5re9LqVQolG'
    'EblrmKxGPazp+SGf3XDdEJvFK/xmdCQzCR6bcZUjCmWx8bSo87Zp40PK0rLdiNVGFLPnmTk0jdrI'
    'tVtKHrbsKicNyXDTxljiUwJqn25UrJdSnFYd+n/7N/6oma5axkxt+hz9J21bo2ag5aDh5x0CY2Fv'
    'ygvX9cFGz2+bVxOFStnbtXbSs5dWfNyb2suoRlMS9joH6eHvWCs1OPwLpDY/ZVvNRVCs48I8oxW2'
    'szBRpKR4JpPzaP+7g4Y+Ymu3gw+E/1FGKDRsQbFKKy90i3VuyGp1sOe1pkI87XONam5WIsO6nS6W'
    'AfJNJl3KoT4tttimITEBZC+Q5NnGn32nXsCDWfrnt89XTXp2sW+uKW80qU0vfcpGxrxl0pgODOIR'
    'Ui7oNQB9kirLWFZVRcc4YJ6Q62dauZ58jsachJKmBCs9M2Sq36RSiH6qjgJSCvJCdP4MGOlinOoG'
    's2HmEEnd4XnmpAxe8p0t6gYWc+vKIAjOKgjwU5A/ebcJYGm9L4ZP1EKDjYtjEtCJt3xvC5g8CLzu'
    '4geJrbxJwVwkgbxghSHLALPF6HAjVpGI4hUmU+tbPCYr9xoP934LeQ0F+Loz6yNuiPfwGp+E7qjg'
    'OzDU5Rd0F4bnjgvOFcdLLkQ24+NQfEOGPoKAF6jMxCoESdwAGyVBfEh1oy9RSC43Ki0avgH5Sl/I'
    'eyjmFEWfDvVT3uXG7YyQy/aGugrfUcGPR+/lwRs7a1/BjAng2DphKesOa6de3FMH+O/eJlon67kH'
    'Nh2Uofz0ww94hv9fQXjvPS9DSe+VKv1uGDXP25kTCyubCKtxaEqOh53aDU9puOCZOQIJ7a2YEfZZ'
    'IQtUVyeFCmjhWX9ktPmqcdBIAq08ae500kv222gN2gWCfhxvMJ3Ryt/HHi3reEGZD1D85REPqO1c'
    'htDof0j15AHbiI4BhnxiIPjTWC2IybmgZbKRwqM4cvyoZAAwjgkxlVqDb3zIF594jk9iL/d2jLL2'
    'RJC6+nRyIOzNqUVUE03Q+QNBN2JpJslpj3iDfrAHP5IPnqETBPWKRsHeB5ST0e3XN81Il1vfQFg8'
    'J0jrPfvW6NkjvVNtuj1Uhg3H2dNqEeOFB79gxE/nfMCs3eCFV3gJWgEeECjaEskiI9exGSn8TNSs'
    'gacsTIaVAyNhLo+RYs0t+Fq/GUeaPqEuo1nli7gCJ8kcvnmL95nRomKp74hbz0WC5uYUXMb0KsUq'
    'JQkjAAi6A07cVxn1nU7RmHM4bJft0ZBHThIVkyC5CKZ2AvjvcOo68NOOc+a4TGqHjmFgluqgNF5r'
    'hcQttVqWX5uSpfGHC+Rxj4eck/oW2NCS2pEqEZcUXGIgqJzNEY4QWW4+3jbj9jt2++10hxaiZb89'
    'p6l3dm6/tXOqCWE64sevxQ+X/OVyan/kv/yMmKC3xsWtkkTcH9dS+Up9gR5HDZ9LRk49xqUyQOO7'
    'GUcJyCx9XURKV/6EUpIjmPB7jVxTbVM2BfUguZTg2Fsg59j6ClTXREkVjuwjr4ttQYiWVH9pqqlU'
    'Cij/YUwN1lMViCRjES26oFKnTrZzHYW0QXb/dX8ka4+tuf/K0hfmqhSYcLcNFrkWmi+l+NG5QIDu'
    'kxjWb6RouY3C20sHg76Rou02igaKJlfo5kBbNYXSDQaHNS3o5jZ0Sun+Qn2HjLpkItK3Z2p33dFK'
    'rhCiI6E0IYuSXDLG824IIPM4Y7o4C/FPzYVrwHIRQuuRw02ARW6oK+2AGByzH+Tz2jlThmUNr/Hm'
    'd5koaoil0pnz4TsgrKEpYhRM2ZTHSjGXKXbyIq0psmchpX2zjrcp3Sm6zhHvfWFT2AtSl23caZp3'
    'HEoT0nZ9qG/FYA3o6sbRG74wI69q+mXpcYRpGLlhj6R2ai71UvmEtfO8HiRPIxd4jxvdB9F/jFDd'
    'CqL7bLWBXxtaZvmEtjY4tcZytJjQqyDkr40LMSSURVrER5Wde/EQwQ76RzX6GwPFXvBSq8SlpRGX'
    'WiEiqnr+mY2BOwbNHdHXlPTWDoZoCb6LV0swSEqC5T2IaXI/tquznBswgnBNf9qTFgg6TveTajPn'
    'UkENLtEo4GzpWKeuZCBoTLDfODiSEyJnVcyLPVdRsTUWJqI94LGBplDhlKLy8tLyW39sMLFxql73'
    'yXaP7G5IP8mdhvIsfXgGcdM9edg9y9uOfBfJlCWPGjWGSl+vTrnKuWEuu3yXa7fXuMClyzan25Mo'
    'lPdCqzgz61bcsLlzAOYH79GVdsq6chciA7HC+xUEWjdzUZI2WHd5QZuemhoUHzBYeM8gHaGHRqBN'
    'OhUDHwdpJdZlaPGED+vLVe9LqOl4M2mJN+uiiQzXhA40RYRE6DP+nhVcjqEo2jDFO26XkrCI0ly2'
    'seR8bSj5k40mkimhlFCjbABrEoZGZWlWi0y9qK6moQUK0ZmDljGmE+ds0aisPuckxxWYLQNU9vtz'
    'LfgCZLz9DG+rKeVJydOT84k5KckZLOZ26i7tR/TsMk6j6BdJ1td8ACgnc/A4GNVqydSilmoqYe44'
    '2KvVw9uX+7jbVu5sDdf6diNStYkC4ETW5927Pi6d78LIG6HHvMZe75UHxe4KtCNYq8B7tDsrmBtE'
    'ffVYMGs1q3nprYYpWG11VCqYS6DOCfSTiKPj4yLltbT0arNM+9b2cJ/zvvhg6G9W3t6xXMNtcoVS'
    'HR7FoXKtHsNTHxBdy0O8TIvzMldn7rVxWCWUAxaIEhZp1b/CiMRg0Bnnx+Z0VJOAZu/52sBa//XZ'
    'pLb+e2ZlP8dr7ql3974ZOnpcH/Q3q21ZEybWdvVptqem2SMyZuYN49YC1DaN9PStSxFYUAgQmmpv'
    'XbZqPSzi7RwjJ9YuUZNiq0JdSNx6DRGBmmXSlxtzTmtmu87bHtTyNFUmba3w3mMfd/WcZtSepLcR'
    'u5KnNavuY026utytq8tqHLXWAt1xoH1l/yjYdXxDoar66HUqNwjWVVvILdOGsndScvyjiJWShllp'
    'JMPtqkzquGlma/mG7jyu1fY0rLIEH23ZJ4UNFF65pyuuKIuqTgCnVrU0bWr5m9aVH2vUxy9duXWU'
    'dVZIjWiTLdJOt6zopdup30q6xOAh3am/g3pKgdptSer5Uh5j4iBrCryp2OQEuHb45gypHRqGrT4a'
    'Js2xSQaYu/48LtMF148hrECt7rO19YPlXmvrxad61zj/3Oidt3ptsH1Y6qPdQgYdAveTo06Ie0mC'
    '4HqLHN/hPDnHi/18lg4BHbHdKT7uUV8/snhb3TCmZndirtXsB/zorjSrCaZ8d7XWwZC1fSOxKdJb'
    'MAmgpCt33nlPnfonn4vnWzsd1klRhxzwYj6e/X3yruHHrNN7vxXoU45Fiy2wgiVEIF06UzVLCKnp'
    'rt+bo6DpMeTKwpB9VajbYgcdcj/FFCF2y1Ej3E7s1+a6KWfRF70SUihEr34DTHkEse2LPIZWed81'
    'Pnt0co3hVpRrs6ZdP9OgokD1pZLrr1D0qL06ESIuVg+tDeIQP0l3NpskxKqPjspAYkyLO9A/vbc6'
    'zlKtY7sOPbUu+uKu5/TV+GxWUzsWBuq4vRYEEVd3eX+Vzou4eOjfpvmKVvvKNof96RySFZ80y1sc'
    'dO90H9VlQKm2Pu3c30FwKVzUjYBITRxnqaKFYXUVLZWAli22RCUvJPESX9myyOC04DV8iFzoVLt+'
    'la9oHT15Qn8eZWHT1VQZaz4mt7HIJ2kajXdJRFXyTgBV9VHb9ozRoXeo9eVeKlDQ36SLZb9cx6sV'
    'sKeVlX6N+pi880VVp0ogSvPKvE2Rz/U7EOUrA4L5lguvixjzT2RpkgvaeBoAqhPnDliThpLyW/pw'
    'tXezihciIEHH9fuY3qpHi68moqRsS2ij6NJ6cZe2JjSgnO9dvCwgACrj4nqLraX0BsK7+OG/CEy+'
    'QQ5wSUKXQmww8QdTOmP4taJXKsZ4ioFe0Acl5l15QmZm9qjXqdpdA2zc+0Hny9fPbyezz58+/INf'
    'LPa0xSeVO8lYQlxxrt1pdRmuesE3eb0Wf8SdYATkjD7ePQxM+1Pn+ASno4Zpct8L+CTGKp7jbhXv'
    'N5hUDr9XjKgZ0koSxgPNgDB4IZPZam9jMcW4WWTuGJRJRRuBSWDm8WWZ8o4Hp/e7ZbgRhOcwKYGP'
    'dxmsUpkHgxdUjIPf/m2nNbUeHvEc+oCgNtavc8E7KjgJQZ20dl+exLlPcaavp3A2r7CguXNF6Wfn'
    'D2Ul1pN7vP9d5vK+KM3rMFOU6RjktQuMw5GKVGI7p1VwRpnMu4jxLT+884ZZao9scembMRgXbdvI'
    'XTTeIgQk3rOcTcK5pdf84kh+qWOeYT/CFyBL8tWT8B07ETryBkWRlxK5a6cp+RNlM/wOIl4UwW0J'
    'v7vtl1V5iCMOuwlRdCkpD9XYAh87M2isOI1TflymPRKnMf5qplHr0ca78HqKDWN7kEH9jHlT2z7f'
    'S68N2BTgHtz3ePZj3vw2U3tjQGUtF886gGwDUjKxTT/JMeC7TOWEZSVkNiHlGeHui/LF8CAZkB4G'
    '/ili6LU21h39QIh79klpoJr+IeZKJeSyHVGonHKxDqFB6x18/GbOwfzoQF4sj8VAFF8y37HeztT9'
    'n0y+XmQu79k+Omi9DjzqQalLRpd2At9Ofjr7RDrn7P3Z6cnFhJ4CavNiI2iIHk4+vfMByq7Qtely'
    'IOmdtcwN7GyP3yDs3IYvX1uWl6SKuR+6mnw7X5IWdRCsRsn8aONkLisArOOlAOgyxDo9Pswwy5fW'
    'JfnIl5zexgsEWanP+AbbHEYzdCkhNPmlVGR04UD3bt6N8Or32ktgrgbU9RCgcVKoQ+UtmGUH9PGK'
    'QZbg61e71EADfwvVdrNYe5Bs15sy/K0xHxX9x3ZfmtO2q+ftLCFfqiGF+aWBgR5qgfTgkpOd21Rf'
    'fI1SIt1MJ9LNBJ9lkcSSpqjX+3fE8mgOxZNkWa9RaJcZgpzRW6ytJGhbeKo24eGD75QZjVl51gsp'
    'nRMWMrGGPAAnt8byEY4bifCN84Hy5J8DmPART23BF8Z869abZtxO5Oc3FmIvPFnr9VemqA4P4iQh'
    'DdT24ohH5hFzlFn/YriXDEDhkf33i3tdtpV6jBwdE28TfJNkfs0KA6eG8inzu5qV8EweTPTD47Nh'
    'F1Bci6RxySYgURr+N/8EahNyOWiEQd8Ku4rX8yQOiuOgsPSJyuq+8SrBrnTnMcrR3dWqoqZfsG6r'
    '9uDrYYGgR7VH4c4taSXoFQ1QXb+ACF+MPeMMM/nWaa3k42ut3BM9MVpGoVpv6j3vvoaHFNcRerLE'
    'kUdTA9SztbE672BeYz0otll4qbsASKd4ZQG+fntsQX05+zLZeYk6VIEpZ1d5N/nl088fPjjv9cE0'
    'xgFj17pzgFHSBv+S0WzmD22rmmQ2PEX1InD9znSoRyf/kOvspeZ1oVOp3YBeHXUw59jwfCkWjKaR'
    'StfGFw7hGwT92ZS35M9ac0BlG/NjpcYbOcdNFUL38Yfdj2fnH08uTv+K/u6L8lgGpuMX5Wt4oM0m'
    'f/d5wOC6gSD23Pl6S65nz7VINn2Rb4yv0pUYlw/lgMfYzuDb5b6/UOccX5tDlOu0XMcVyHWJUwED'
    'bugNjTs5m4CR50/UsVihRrcQcnxLmbmnsFLy9GoVQIuAjrvLoQkNHCpGhVamtUPAr3Lz6Ior9WL4'
    'wUlxvUXr+wW/FVIY4w0agFksy8Iuvbq+i1EXerbOW4g8sNw/fGsrLSGNu0Arvqj7JscbxcehreUg'
    'eE0YN0FF7UgFLTFVDxsxTjG3VWHHV9G31pKsVLefFHTAHOA4DRS/hzLNlicrPhnAx6aDieq55iLo'
    '40IzGabYQU03B2pNel0vUMc7V+m8FV6npRhwrUd1Q9YLri+VLZva5xx1FdVUWw1zEkKOCAZNuj+v'
    'dH3mRsHOCL3njqMMOtXBNXWsA+Fuvh58pT8hsRNHLvi/eHf9yeEJaL2b7dUVvdhDOxim9Uv5EV8z'
    'qkMagrQiwB73zvhZigann7m5gK+bOyGeFdqh+M141WmsVqvsXbVXZkjgsxlOU7s+jDfdnoVVvnBZ'
    'f4+iHd1AsPY+3Pj6oFQTMxop4DxV249YOXcrNeI1C1Z+UnGb6wijz+t1xBT3OzWNLp8fo5qSOqqH'
    'n6l6H6pD1ytcctWaS2HCw5NRW9fqGmNX1wws8bZOoSy2KezK1QKCr/FXKhPHpXw64/kudMuLslBa'
    'rNcYol3NtDNhdzNYz+IL2gfAPqPLbPBsPcjZbIbWYjaTeddsOjr/HyQMeR8='
)
_GENERATOR_SHA256 = '32b67d0f14326bd1623d99b4f45f297b30b307f99119a5e87f4572453696a2ec'


def _load_generator():
    import types

    raw = zlib.decompress(base64.b64decode(_GENERATOR_ZLIB_B64))
    module = types.ModuleType('generate_corpus')
    module.__dict__['__name__'] = 'generate_corpus'
    exec(compile(raw, 'generate_corpus.py', 'exec'), module.__dict__)
    return raw, module


GENERATOR_BYTES, gen = _load_generator()

SPLITS = ('heldout', 'visible')

REQUIRED_KEYS = ('object', 'canonical_der', 'verdict', 'first_rejecting_element')

# Static proxies for the documented 'no subprocesses, no network calls': the
# modules whose only purpose is spawning processes or speaking to the network.
FORBIDDEN_MODULES = {
    'subprocess', 'multiprocessing', 'socket', 'ssl', 'http', 'urllib',
    'ftplib', 'telnetlib', 'smtplib', 'requests',
}
# os-level helpers that spawn or exec a child process.
FORBIDDEN_OS_CALLS = {
    'system', 'popen', 'fork', 'forkpty', 'posix_spawn', 'posix_spawnp',
    'spawnl', 'spawnle', 'spawnlp', 'spawnlpe', 'spawnv', 'spawnve',
    'spawnvp', 'spawnvpe', 'execv', 'execve', 'execvp', 'execvpe',
    'execl', 'execle', 'execlp', 'execlpe',
}

BASE64_RE = re.compile(r'\A[A-Za-z0-9+/]+={0,2}\Z')

# sha256 of the reference NDJSON the pinned generator emits for each split.
REFERENCE_DIGESTS = {
    'heldout': '235ac6dac8e46ff273b1e376c6e8f945428214d44302168cbfa678bc2d6b6766',
    'visible': '7c385a51875a19d8ca8e5fc9076248e72eb7f12f358c0ce9438ab9203d908ecb',
}


# --------------------------------------------------------------------------
# reference corpus: rebuilt in the verifier from the task's pinned generator
# --------------------------------------------------------------------------

def _build_reference():
    '''Regenerate the visible and held-out splits exactly as the task does.

    Mirrors generate_corpus.main for the two graded splits.  The generator is
    deterministic (pinned SEED; the held-out split uses no randomness), and this
    is the same reference the Gold reproduces (111/111 objects).
    '''
    cert_objs = gen.build_split('o', gen.CERT_CASES, 20)
    lib_objs = gen.build_split('o', gen.LIB_CASES, 20)
    for o in cert_objs:
        o['logged'] = True
    for o in lib_objs:
        o['logged'] = False
    visible = cert_objs + lib_objs
    order = list(range(len(visible)))
    random.Random(gen.SEED ^ 0x5A5A).shuffle(order)
    visible = [visible[i] for i in order]
    for i, o in enumerate(visible):
        o['object'] = 'o%04d.der' % (i + 1)

    held_cases = gen.graded(gen.MIXED_CASES + gen.LIB_CASES + gen.CERT_CASES)
    held = gen.build_split('h', held_cases, len(held_cases))
    for i, o in enumerate(held):
        o['object'] = 'h%04d.der' % (i + 1)
    return {'visible': visible, 'heldout': held}


def _reference_records(objs):
    '''The record the generator itself records for each object.'''
    return [
        {
            'object': o['object'],
            'canonical_der': base64.b64encode(o['canonical']).decode('ascii'),
            'verdict': o['verdict'],
            'first_rejecting_element': o['offset'],
        }
        for o in objs
    ]


def _reference_ndjson(records):
    '''Reproduce the generator's reference.ndjson text byte for byte.'''
    return ''.join(json.dumps(r) + '\n' for r in records)


def _tlv_offsets(data):
    '''Absolute tag offsets of every element the input file carries.'''
    offsets = set()

    def walk(el):
        offsets.add(el.off)
        for kid in el.kids:
            walk(kid)

    try:
        root, _certs, pattr = gen.locate(data)
    except Exception:  # pragma: no cover - defensive
        return offsets
    walk(root)
    for a, _b in pattr:
        if 0 <= a < len(data):
            try:
                walk(gen.parse_el(data, a, len(data)))
            except Exception:  # pragma: no cover - defensive
                pass
    return offsets


# --------------------------------------------------------------------------
# running the artifact
# --------------------------------------------------------------------------

class _Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run(script, corpus, cwd, timeout=RUN_TIMEOUT):
    argv = ['python3', str(script), '--corpus', str(corpus)]
    env = {
        'PATH': '/usr/local/bin:/usr/bin:/bin',
        'HOME': str(cwd),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
    }
    kwargs = dict(
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )
    try:
        # The trusted frame runs as root; drop the child to an unprivileged
        # user so it cannot read the stripped /tests tree.
        return subprocess.run(argv, user='nobody', extra_groups=[], **kwargs)
    except (TypeError, ValueError, PermissionError, KeyError, OSError):
        return subprocess.run(argv, **kwargs)


def _parse_lines(text):
    '''(records, errors) -- never raises, so a broken artifact fails tests.'''
    records, errors = [], []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            errors.append((i, 'blank line'))
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            errors.append((i, 'not JSON: %r' % (exc,)))
            continue
        if not isinstance(obj, dict):
            errors.append((i, 'not a JSON object'))
            continue
        records.append(obj)
    return records, errors


def _records(harness, split):
    return harness['parsed'][split][0]


@pytest.fixture(scope='session')
def harness():
    root = Path(tempfile.mkdtemp(prefix='adjver-'))
    os.chmod(root, 0o755)

    reference = _build_reference()
    h = {
        'root': root,
        'ref': reference,
        'ref_records': {},
        'ref_ndjson': {},
        'parsed': {},
        'runs': {},
        'mixed_expected': {},
    }

    script = root / 'artifact' / 'adjudicate.py'
    run_dir = root / 'run'
    (root / 'artifact').mkdir()
    run_dir.mkdir()
    os.chmod(root / 'artifact', 0o755)
    os.chmod(run_dir, 0o777)

    artifact_ok = ARTIFACT.is_file()
    if artifact_ok:
        shutil.copyfile(ARTIFACT, script)
        os.chmod(script, 0o644)

    for split, objs in reference.items():
        corpus = root / split / 'corpus'
        corpus.mkdir(parents=True)
        os.chmod(root / split, 0o755)
        os.chmod(corpus, 0o755)
        for o in objs:
            path = corpus / o['object']
            path.write_bytes(o['data'])
            os.chmod(path, 0o644)
        recs = _reference_records(objs)
        h['ref_records'][split] = recs
        h['ref_ndjson'][split] = _reference_ndjson(recs)
        if artifact_ok:
            h['runs'][split] = _run(script, corpus, run_dir)
        else:
            h['runs'][split] = _Result(127, '', 'artifact missing: %s' % ARTIFACT)
        h['parsed'][split] = _parse_lines(h['runs'][split].stdout)

    # A corpus of odd-named .der files plus a decoy non-.der file.
    mixed = root / 'mixed'
    mixed.mkdir()
    os.chmod(mixed, 0o755)
    picks = [(reference['heldout'][0], 'alpha.der'),
             (reference['heldout'][1], 'zeta.der')]
    for obj, name in picks:
        path = mixed / name
        path.write_bytes(obj['data'])
        os.chmod(path, 0o644)
        h['mixed_expected'][name] = obj
    decoy = mixed / 'notes.txt'
    decoy.write_text('not an ASN.1 object\n', encoding='utf-8')
    os.chmod(decoy, 0o644)

    empty = root / 'empty'
    empty.mkdir()
    os.chmod(empty, 0o755)

    for split, corpus in (('mixed', mixed), ('empty', empty)):
        if artifact_ok:
            h['runs'][split] = _run(script, corpus, run_dir)
        else:
            h['runs'][split] = _Result(127, '', 'artifact missing: %s' % ARTIFACT)
        h['parsed'][split] = _parse_lines(h['runs'][split].stdout)

    if artifact_ok:
        h['rerun'] = _run(script, root / 'heldout' / 'corpus', run_dir)
    else:
        h['rerun'] = _Result(127, '', 'artifact missing: %s' % ARTIFACT)
    return h


# --------------------------------------------------------------------------
# R1 -- interface, self-containment, ordering, schema
# --------------------------------------------------------------------------

def test_r1_program_is_single_self_contained_stdlib_artifact():
    assert ARTIFACT.is_file(), 'declared artifact %s is missing' % ARTIFACT
    src = ARTIFACT.read_text(encoding='utf-8', errors='replace')
    assert src.strip(), 'artifact is empty'
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        pytest.fail('artifact is not valid Python: %s' % (exc,))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split('.')[0])
    bad_imports = sorted(imported & FORBIDDEN_MODULES)
    assert not bad_imports, 'artifact imports forbidden module(s): %s' % bad_imports

    os_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == 'os':
                os_calls.add(node.attr)
    bad_calls = sorted(os_calls & FORBIDDEN_OS_CALLS)
    assert not bad_calls, 'artifact uses process-spawning os call(s): %s' % bad_calls


def test_r1_exits_zero_and_emits_ndjson_for_every_split(harness):
    for split in SPLITS:
        run = harness['runs'][split]
        assert run.returncode == 0, (
            '%s: exit status %r, stderr=%r' % (split, run.returncode, run.stderr[-500:])
        )
        _recs, errors = harness['parsed'][split]
        assert not errors, '%s: stdout is not one JSON object per line: %s' % (
            split,
            errors[:5],
        )


def test_r1_one_record_per_input_file_in_ascending_order(harness):
    for split in SPLITS:
        expected = [o['object'] for o in harness['ref'][split]]
        records = _records(harness, split)
        got = [r.get('object') for r in records]
        assert len(got) == len(expected), (
            '%s: %d record(s) for %d input file(s)' % (split, len(got), len(expected))
        )
        assert got == sorted(expected), '%s: records are not in ascending filename order' % split
        assert got == expected, '%s: object names do not match the input basenames' % split


def test_r1_record_schema_and_types(harness):
    for split in SPLITS:
        records = _records(harness, split)
        assert records, '%s: no records emitted' % split
        for i, rec in enumerate(records):
            missing = [k for k in REQUIRED_KEYS if k not in rec]
            assert not missing, '%s: record %d is missing key(s) %s' % (split, i, missing)
            assert isinstance(rec['object'], str), '%s: record %d object is not str' % (split, i)
            assert isinstance(rec['canonical_der'], str), (
                '%s: record %d canonical_der is not str' % (split, i)
            )
            assert rec['verdict'] in ('accept', 'reject'), (
                '%s: record %d verdict %r is not accept/reject' % (split, i, rec['verdict'])
            )
            fre = rec['first_rejecting_element']
            assert fre is None or (isinstance(fre, int) and not isinstance(fre, bool)), (
                '%s: record %d first_rejecting_element %r is neither int nor null'
                % (split, i, fre)
            )


def test_r1_non_der_files_ignored_and_empty_corpus_clean(harness):
    mixed = harness['runs']['mixed']
    assert mixed.returncode == 0, 'mixed corpus: exit %r stderr=%r' % (
        mixed.returncode,
        mixed.stderr[-300:],
    )
    records, errors = harness['parsed']['mixed']
    assert not errors, 'mixed corpus: %s' % (errors[:3],)
    names = [r.get('object') for r in records]
    assert names == ['alpha.der', 'zeta.der'], (
        'mixed corpus: expected exactly the two .der inputs, got %r' % (names,)
    )
    for rec in records:
        obj = harness['mixed_expected'][rec['object']]
        assert rec['verdict'] == obj['verdict']
        assert base64.b64decode(rec['canonical_der'], validate=True) == obj['canonical']

    empty = harness['runs']['empty']
    assert empty.returncode == 0, 'empty corpus: exit %r' % (empty.returncode,)
    assert empty.stdout.strip() == '', 'empty corpus: expected no output, got %r' % (
        empty.stdout[:200],
    )


# --------------------------------------------------------------------------
# R2 -- canonical DER re-encoding
# --------------------------------------------------------------------------

def test_r2_canonical_der_re_encodes_the_same_abstract_value(harness):
    mismatches = []
    checked = 0
    total = sum(len(harness['ref'][s]) for s in SPLITS)
    for split in SPLITS:
        for obj, rec in zip(harness['ref'][split], _records(harness, split)):
            tag = '%s/%s' % (split, obj['object'])
            field = rec.get('canonical_der')
            if not isinstance(field, str) or not BASE64_RE.match(field) or len(field) % 4:
                mismatches.append('%s: canonical_der is not padded standard-alphabet base64' % tag)
                continue
            try:
                raw = base64.b64decode(field, validate=True)
            except (binascii.Error, ValueError) as exc:
                mismatches.append('%s: canonical_der is not decodable base64 (%s)' % (tag, exc))
                continue
            if base64.b64encode(raw).decode('ascii') != field:
                mismatches.append('%s: canonical_der base64 is not in canonical form' % tag)
                continue
            if raw != obj['canonical']:
                mismatches.append(
                    '%s: re-encoded DER differs from the reference (%d bytes vs %d)'
                    % (tag, len(raw), len(obj['canonical']))
                )
                continue
            checked += 1
    assert checked == total, '%d of %d objects carried a comparable canonical_der' % (
        checked,
        total,
    )
    assert not mismatches, '%d canonical_der mismatch(es): %s' % (
        len(mismatches),
        mismatches[:8],
    )


# --------------------------------------------------------------------------
# R3 -- the appliance's verdict
# --------------------------------------------------------------------------

def test_r3_verdict_matches_the_appliance_reference(harness):
    mismatches = []
    labels = {}
    for split in SPLITS:
        for obj, rec in zip(harness['ref'][split], _records(harness, split)):
            labels.setdefault(obj['label'], set()).add(obj['verdict'])
            if rec.get('verdict') != obj['verdict']:
                mismatches.append(
                    '%s/%s: verdict %r, appliance %r'
                    % (split, obj['object'], rec.get('verdict'), obj['verdict'])
                )
    assert not mismatches, '%d verdict mismatch(es): %s' % (len(mismatches), mismatches[:8])
    # The reference really is two-dimensional: the same construct is rejected or
    # tolerated according to the parse path that reaches it.
    assert labels.get('cert-issuer-rdn-unsorted') == {'reject'}, (
        'reference drift: unsorted RDN inside a certificate is not rejected'
    )
    assert labels.get('cms-digest-algs-unsorted') == {'accept'}, (
        'reference drift: unsorted SET OF in the CMS envelope is not accepted'
    )
    verdicts = {o['verdict'] for s in SPLITS for o in harness['ref'][s]}
    assert verdicts == {'accept', 'reject'}, 'reference split lost a verdict class'


# --------------------------------------------------------------------------
# R4 -- first rejecting element
# --------------------------------------------------------------------------

def test_r4_first_rejecting_element_matches_reference(harness):
    mismatches = []
    for split in SPLITS:
        for obj, rec in zip(harness['ref'][split], _records(harness, split)):
            tag = '%s/%s' % (split, obj['object'])
            fre = rec.get('first_rejecting_element')
            if obj['verdict'] == 'accept':
                if fre is not None:
                    mismatches.append('%s: accepted object reports offset %r' % (tag, fre))
                continue
            if not isinstance(fre, int) or isinstance(fre, bool):
                mismatches.append('%s: rejected object reports %r, expected %d'
                                  % (tag, fre, obj['offset']))
                continue
            if fre != obj['offset']:
                mismatches.append('%s: offset %d, appliance %d' % (tag, fre, obj['offset']))
                continue
            offsets = _tlv_offsets(obj['data'])
            if fre not in offsets:
                mismatches.append('%s: offset %d is not a tag octet of the input' % (tag, fre))
    assert not mismatches, '%d first_rejecting_element mismatch(es): %s' % (
        len(mismatches),
        mismatches[:8],
    )


# --------------------------------------------------------------------------
# integrity / anti-cheat anchors
# --------------------------------------------------------------------------

def test_no_duplicate_records_and_exact_counts(harness):
    for split in SPLITS:
        expected = [o['object'] for o in harness['ref'][split]]
        records = _records(harness, split)
        names = [r.get('object') for r in records]
        assert len(names) == len(expected), (
            '%s: %d record(s) for %d input file(s)' % (split, len(names), len(expected))
        )
        assert len(set(names)) == len(names), '%s: a record was emitted more than once' % split
        assert sorted(names) == sorted(expected), '%s: record set differs from the input set' % split


def test_output_is_deterministic_across_runs(harness):
    first = harness['runs']['heldout']
    rerun = harness['rerun']
    assert rerun.returncode == 0, 'second run: exit %r' % (rerun.returncode,)
    assert rerun.stdout == first.stdout, 'the artifact is not reproducible across runs'


def test_reference_corpus_matches_pinned_gold_digests(harness):
    assert hashlib.sha256(GENERATOR_BYTES).hexdigest() == _GENERATOR_SHA256
    for split in SPLITS:
        text = harness['ref_ndjson'][split]
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        assert digest == REFERENCE_DIGESTS[split], (
            '%s: reference NDJSON drifted (got %s)' % (split, digest)
        )
