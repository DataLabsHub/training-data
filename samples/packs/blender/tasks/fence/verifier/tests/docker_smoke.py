#!/usr/bin/env python3
"""Target-worker acceptance helper. Requires Docker; not run in this delivery.

Builds once; repeats fresh reset -> unchanged FAIL -> oracle PASS three times.
Results are written outside the immutable release package to a caller-chosen path.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--skip-build', action='store_true')
args = parser.parse_args()
out = args.output.absolute()
if out.exists():
    parser.error('output must be a new directory')
if not shutil.which('docker'):
    parser.error('Docker unavailable; this is not a simulated test')
out.mkdir(parents=True)
cli = [sys.executable, str(ROOT / 'runner/envctl.py')]
summary = {'status': 'RUNNING', 'trials': [], 'scope': 'procedural_fixture_only'}


def call(*arguments):
    subprocess.run(cli + list(map(str, arguments)), check=True)


try:
    if not args.skip_build: call('build')
    for index in range(1, 4):
        workspace = out / f'episode-{index}'
        unchanged = out / f'unchanged-{index}'
        solved = out / f'solved-{index}'
        call('reset', '--workspace', workspace)
        shutil.copyfile(workspace / 'initial.blend', workspace / 'submission.blend')
        call('grade', '--task', 'fence-grounding-v0', '--workspace', workspace, '--reports', unchanged)
        negative = json.loads((unchanged / 'result.json').read_text(encoding="utf-8"))
        assert negative['status'] == 'FAIL' and negative['reward'] == 0, negative
        call('oracle', '--workspace', workspace)
        call('grade', '--task', 'fence-grounding-v0', '--workspace', workspace, '--reports', solved)
        positive = json.loads((solved / 'result.json').read_text(encoding="utf-8"))
        assert positive['status'] == 'PASS' and positive['reward'] == 1, positive
        summary['trials'].append({'index': index, 'unchanged': negative, 'oracle': positive})
    summary['status'] = 'PASS'
except Exception as exc:
    summary['status'] = 'ERROR'
    summary['reason'] = str(exc)
finally:
    (out / 'smoke-summary.json').write_text(json.dumps(summary, indent=2), encoding="utf-8")
raise SystemExit(0 if summary['status'] == 'PASS' else 2)
