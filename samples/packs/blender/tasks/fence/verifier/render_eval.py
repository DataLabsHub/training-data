#!/usr/bin/env python3
"""Render evidence from a .blend using auto planning or a render.views request.

Docker is the default. Local Blender is an explicit option for trusted artifacts.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from view_plan import validate
from runtime import IMAGE, sandbox, resolve_image
from worker_runtime import WorkerRuntime
from support import Deadline, run_native, BudgetExceeded, file_preflight, freeze_submission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--backend', choices=['docker', 'local'], default='docker')
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--resolution', type=int, nargs=2)
    parser.add_argument('--samples', type=int)
    args = parser.parse_args(argv)
    source = args.submission.resolve()
    output = args.output.resolve()
    if not source.is_file() or args.submission.is_symlink():
        parser.error('submission must be a regular .blend file, not a symlink')
    issue = file_preflight(args.submission)
    if issue:
        parser.error(issue)
    if source.parent == output or source.parent in output.parents or output in source.parents:
        parser.error('output must be outside the submission directory')
    if output.exists():
        parser.error('output must be a new directory')
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    executable = args.blender if args.backend == 'local' else 'docker'
    if not shutil.which(executable):
        parser.error('runtime unavailable: ' + executable)
    request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else {'action': 'render.views', 'args': {'mode': 'auto'}}
    request_args = request.setdefault('args', {})
    if args.resolution:
        request_args['resolution'] = args.resolution
    if args.samples is not None:
        request_args['samples'] = args.samples
    request_args.pop('output_dir', None)
    try:
        validate(request)
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.backend == 'docker' and any(',' in str(p) for p in (source, output, ROOT)):
        parser.error('Docker mount paths cannot contain commas')
    output.mkdir(parents=True)
    try:
        if args.backend == 'docker':
            args.image = resolve_image(args.image)
        source = freeze_submission(source, output / 'input_snapshot')
    except Exception as exc:
        parser.error('render_preparation_failed:' + str(exc))
    normalized = output / 'request.json'
    normalized.write_text(json.dumps(request, indent=2), encoding="utf-8")
    # All produced paths live under a fresh run directory. The worker creates views/.
    worker = ROOT / 'render_views.py'
    workers = WorkerRuntime(args, source, output, ROOT, Deadline(args.timeout))
    command = workers.command(worker, ['--submission', str(source) if args.backend=='local' else '/input/'+source.name,
               '--request', str(normalized) if args.backend=='local' else '/output/request.json',
               '--output', str(output/'views') if args.backend=='local' else '/output/views'], 'render')
    status = {'status': 'ERROR', 'training_mask': 0, 'reward': None}
    try:
        returncode = workers.run(command, 'render')
        manifest_file = output / 'views/manifest.json'
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
        if returncode == 0 and manifest.get('status') == 'COMPLETE':
            status.update(status='COMPLETE', manifest='views/manifest.json', view_count=len(manifest['views']))
        else:
            status.update(reason='render_timeout' if returncode==-1 else 'render_failed', exit_code=returncode, detail=manifest.get('error'))
    except (subprocess.TimeoutExpired, BudgetExceeded) as exc:
        status['reason'] = 'render_timeout' if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
        # Killing the Docker client alone does not terminate its container.
        cid = output / 'container.cid'
        if args.backend == 'docker' and cid.exists():
            try:
                subprocess.run(['docker', 'rm', '-f', cid.read_text(encoding="utf-8").strip()],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                status['cleanup_warning'] = 'container cleanup failed; inspect container.cid'
    except Exception as exc:
        status['reason'] = type(exc).__name__ + ': ' + str(exc)
    (output / 'render_result.json').write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
