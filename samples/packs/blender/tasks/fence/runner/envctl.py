#!/usr/bin/env python3
"""Local Docker adapter: build/reset/oracle/grade. No host Blender required."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT
sys.path.insert(0, str(ROOT / 'verifier'))


def run(args):
    print('+ ' + ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=False).returncode


from runtime import IMAGE, sandbox, resolve_image, lock_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'build', 'reset', 'oracle', 'grade', 'render'])
    parser.add_argument('--workspace', type=Path, default=Path('runs/episode-001'))
    parser.add_argument('--reports', type=Path, default=Path('runs/reports-001'))
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--request', type=Path, help='render.views JSON; omit for automatic views')
    parser.add_argument('--backend', choices=['docker', 'local'], default='docker')
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--resolution', type=int, nargs=2)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--task', choices=['general', 'fence-grounding-v0'], default='general')
    parser.add_argument('--farm-reviewer', type=Path)
    parser.add_argument('--farm-review-rounds', type=int, default=2)
    parser.add_argument('--allow-local-reviewer', action='store_true')
    args = parser.parse_args()
    if args.command == 'grade':
        from reward import main as grade_main
        options = ['--submission', str(args.workspace / ('result.blend' if (args.workspace / 'result.blend').exists() else 'submission.blend')), '--reports', str(args.reports),
                   '--task', args.task, '--backend', args.backend, '--blender', args.blender,
                   '--image', args.image, '--timeout', str(args.timeout)]
        if args.resolution:
            options += ['--resolution', *map(str, args.resolution)]
        if args.samples is not None:
            options += ['--samples', str(args.samples)]
        if args.farm_reviewer:
            options += ['--farm-reviewer', str(args.farm_reviewer)]
        options += ['--farm-review-rounds', str(args.farm_review_rounds)]
        if args.allow_local_reviewer:
            options += ['--allow-local-reviewer']
        return grade_main(options)
    if args.command == 'render':
        from render_eval import main as render_main
        options = ['--submission', str(args.workspace / ('result.blend' if (args.workspace / 'result.blend').exists() else 'submission.blend')),
                   '--output', str(args.reports), '--backend', args.backend,
                   '--blender', args.blender, '--image', args.image, '--timeout', str(args.timeout)]
        if args.request:
            options += ['--request', str(args.request)]
        if args.resolution:
            options += ['--resolution', *map(str, args.resolution)]
        if args.samples is not None:
            options += ['--samples', str(args.samples)]
        return render_main(options)
    if args.command == 'check':
        result = {'docker_cli': bool(shutil.which('docker')),
                  'task': 'fence-grounding-v0', 'full_farm': 'NEEDS_REVIEW_WITHOUT_CONFIGURED_REVIEWER',
                  'commercial_ready': False}
        if result['docker_cli']:
            result['docker_daemon'] = subprocess.run(['docker', 'info'], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=15).returncode == 0
        print(json.dumps(result, indent=2))
        return 0 if result.get('docker_daemon') else 2
    if not shutil.which('docker'):
        print('ERROR: Docker CLI unavailable. Install/start Docker on the target worker.', file=sys.stderr)
        return 2
    if args.command == 'build':
        code = run(['docker', 'build', '--platform', 'linux/amd64', '-t', args.image, TASK])
        if code == 0:
            lock_image(args.image)
        return code
    args.image = resolve_image(args.image)
    workspace = args.workspace.absolute()
    reports = args.reports.absolute()
    for path in (workspace, reports):
        if ',' in str(path):
            parser.error('Docker --mount paths must not contain commas')
    if args.command == 'reset':
        # Never delete a colleague's work. Every reset needs a fresh directory.
        if workspace.exists():
            parser.error('reset requires a new workspace path; existing contents are never removed')
        workspace.mkdir(parents=True)
        return run(sandbox(args.image) + ['--mount', f'type=bind,src={workspace},dst=/workspace',
            args.image, 'cp', '/opt/public/initial.blend', '/workspace/initial.blend'])
    if not workspace.is_dir():
        parser.error('workspace does not exist; run reset first')
    if args.command == 'oracle':
        return run(sandbox(args.image) + ['--mount', f'type=bind,src={workspace},dst=/workspace',
            '--mount', f'type=bind,src={TASK / "verifier/oracle"},dst=/solution,readonly',
            args.image, 'bash', '/solution/solve.sh'])
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
