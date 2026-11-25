"""Executed inside a read-only reviewer container before the trusted judge."""
import hashlib
import json
import os
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
for filename, expected in config['artifacts'].items():
    with Path(filename).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise SystemExit('reviewer executable/source digest mismatch')
os.execv(config['command'][0], config['command'])
