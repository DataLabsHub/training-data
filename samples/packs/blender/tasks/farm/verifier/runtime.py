"""Evaluator resource policy and immutable Docker image identity."""
import json
import os
from pathlib import Path
import re
import subprocess
from support import POLICY, dump_json, read_json

HERE = Path(__file__).resolve().parent
IMAGE = 'blender-rl-' + HERE.parent.name.replace('_', '-') + ':0.4'
LOCK = HERE / 'runtime-image.json'


def lock_image(image):
    result = subprocess.run(['docker', 'image', 'inspect', image], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=20, check=True, encoding='utf-8')
    record = json.loads(result.stdout)[0]
    identity = record['Id']
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
        raise ValueError('invalid Docker image identity')
    if record.get('Os') != 'linux' or record.get('Architecture') != 'amd64':
        raise ValueError('release runtime requires linux/amd64')
    dump_json(LOCK, {'source_ref': image, 'image_id': identity, 'repo_digests': record.get('RepoDigests', []),
                     'os': record['Os'], 'architecture': record['Architecture'],
                     'blender_version': POLICY['blender_version']})
    return identity


def resolve_image(image):
    if re.fullmatch(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}', image):
        return image
    if not LOCK.is_file():
        raise RuntimeError('runtime_image_not_locked: build with runner/envctl.py build first')
    data = read_json(LOCK)
    if data.get('source_ref') != image:
        raise RuntimeError('runtime_lock_reference_mismatch: build or supply an immutable digest')
    if data.get('blender_version') != POLICY['blender_version']:
        raise RuntimeError('runtime_lock_version_mismatch')
    identity = data.get('image_id', '')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
        raise ValueError('invalid runtime image lock')
    # Never fall back to a mutable tag. Docker fails if this exact image is absent.
    return identity


def sandbox(image):
    uid = os.getuid() if hasattr(os, 'getuid') else 1000
    gid = os.getgid() if hasattr(os, 'getgid') else 1000
    if uid == 0:
        uid, gid = 1000, 1000
    return ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none',
            '--cap-drop', 'ALL', '--user', f'{uid}:{gid}', '--security-opt', 'no-new-privileges',
            '--pids-limit', str(POLICY['pids']), '--memory', str(POLICY['memory_mb'])+'m',
            '--cpus', str(POLICY['cpus']), '--memory-swap', str(POLICY['memory_mb'])+'m', '--read-only',
            '--ipc', 'private', '--shm-size', '64m', '--ulimit', 'core=0:0', '--ulimit', 'nofile=4096:4096',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size='+str(POLICY['tmpfs_mb'])+'m',
            '-e', 'HOME=/tmp', '-e', 'PYTHONUTF8=1', '-e', 'LANG=C.UTF-8',
            '-e', 'PYTHONDONTWRITEBYTECODE=1']
