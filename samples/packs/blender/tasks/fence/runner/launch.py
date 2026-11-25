"""Run a modeling script with Blender; this launcher does not assign rewards."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path, default=ROOT / 'starter/main.py')
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--initial', type=Path, help='Optional initial .blend; otherwise use the task initial state')
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--export-glb', action='store_true', help='Optional export; not used by the existing verifier')
    parser.add_argument('--render', action='store_true', help='Optional preview from the submitted camera; not a verifier score')
    args = parser.parse_args(argv)
    if not args.script.is_file():
        parser.error('modeling script does not exist')
    if not shutil.which(args.blender):
        parser.error('Blender executable unavailable')
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(p.is_file() or p.is_symlink() for p in out.rglob('*'))):
        parser.error('output must be new or contain only empty directories')
    out.mkdir(parents=True, exist_ok=True)
    command = [args.blender, '--background', '--factory-startup', '--disable-autoexec',
               '--python-exit-code', '3', '--python', str(ROOT / 'runner/execute.py'), '--',
               '--script', str(args.script.resolve()), '--output', str(out)]
    if args.initial:
        command += ['--initial', str(args.initial.resolve())]
    if args.export_glb:
        command.append('--export-glb')
    if args.render:
        command.append('--render')
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
