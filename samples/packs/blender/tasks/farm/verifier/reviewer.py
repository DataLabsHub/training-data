"""Bounded, identity-bound reviewer execution. Docker is the production default."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from support import POLICY, identity_hash, read_json, strict_json, source_identity


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_reviewer(path, submission_dir, allow_local=False, calibration_mode=False, evidence_runtime=None):
    path = Path(path).resolve()
    if Path(submission_dir).resolve() in path.parents:
        raise ValueError('reviewer config belongs to submission directory')
    config = read_json(path)
    for field in ('model', 'model_version', 'prompt_file'):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError('reviewer identity field required: ' + field)
    backend = config.get('backend', 'docker')
    if backend not in ('docker', 'local') or (backend == 'local' and not allow_local):
        raise ValueError('local reviewer requires --allow-local-reviewer (development only)')
    command = config.get('command')
    if not isinstance(command, list) or not command or not all(isinstance(v, str) and v for v in command):
        raise ValueError('reviewer command must be argv list')
    prompt = Path(config['prompt_file']).resolve()
    if Path(submission_dir).resolve() in prompt.parents or not prompt.is_file():
        raise ValueError('prompt must be an evaluator-owned file')
    artifacts = config.get('executable_artifacts')
    if not isinstance(artifacts, dict) or command[0] not in artifacts:
        raise ValueError('executable_artifacts must include the actual executable and reviewer source files')
    for filename, expected in artifacts.items():
        if not isinstance(filename, str) or not (filename.startswith('/') if backend == 'docker' else Path(filename).is_absolute()) or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('invalid executable artifact identity')
        if backend == 'local' and (Path(submission_dir).resolve() in Path(filename).resolve().parents or sha(filename) != expected):
            raise ValueError('reviewer executable digest mismatch')
    for arg in command[1:]:
        if arg.endswith(('.py', '.sh', '.js')) and arg not in artifacts:
            raise ValueError('reviewer entry script must have a digest')
    image = config.get('image')
    if backend == 'docker' and (not isinstance(image, str) or not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', image)):
        raise ValueError('reviewer image must be pinned by registry digest')
    identity = {'backend': backend, 'image': image, 'model': config['model'], 'model_version': config['model_version'],
                'command': command, 'executable_artifacts': artifacts, 'prompt_sha256': sha(prompt),
                'config_sha256': sha(path), 'protocol': 'farm.review.v2',
                'settings_sha256': identity_hash({k: v for k, v in config.items() if k not in ('calibration_file', 'prompt_file')}),
                'evidence_runtime': evidence_runtime, 'verifier_source_sha256': source_identity(),
                'rubric_sha256': sha(Path(__file__).resolve().parent / 'rubric.json'), 'policy_sha256': identity_hash(POLICY)}
    fingerprint = identity_hash(identity)
    calibration = None
    if backend == 'docker' and not calibration_mode:
        calibration_path = Path(config.get('calibration_file', '')).resolve()
        if not calibration_path.is_file() or Path(submission_dir).resolve() in calibration_path.parents:
            raise ValueError('production reviewer requires evaluator-owned calibration report')
        calibration = read_json(calibration_path)
        from calibrate import validate_certificate, fingerprint as calibration_fingerprint
        validate_certificate(calibration, calibration_fingerprint(identity))
        identity['calibration_sha256'] = sha(calibration_path)
        fingerprint = identity_hash(identity)
    return {'config': config, 'path': path, 'prompt': prompt, 'identity': identity, 'fingerprint': fingerprint}


def bounded_process(command, payload, timeout, log, env=None):
    """Drain both pipes concurrently; kill on timeout or either output limit."""
    options = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, **options)
    output = bytearray()
    errors = bytearray()
    overflow = threading.Event()
    failures = []
    def drain(stream, target, limit):
        try:
            while True:
                block = stream.read(65536)
                if not block:
                    break
                if len(target) + len(block) > limit:
                    overflow.set()
                    break
                target.extend(block)
        except OSError as exc:
            failures.append(exc)
        finally:
            stream.close()
    def feed():
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    threads = [threading.Thread(target=drain, args=(process.stdout, output, POLICY['max_review_stdout_bytes']), daemon=True),
               threading.Thread(target=drain, args=(process.stderr, errors, POLICY['max_review_stderr_bytes']), daemon=True),
               threading.Thread(target=feed, daemon=True)]
    for thread in threads:
        thread.start()
    expires = time.monotonic() + timeout
    failure = None
    while process.poll() is None or any(t.is_alive() for t in threads):
        if overflow.is_set():
            failure = 'reviewer_output_budget_exceeded'
            break
        if time.monotonic() >= expires:
            failure = 'reviewer_timeout'
            break
        time.sleep(0.01)
    # A normally exiting reviewer must not leave detached helper descendants.
    if failure or os.name == 'posix':
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=10)
    for thread in threads:
        thread.join(timeout=1)
    Path(log).write_bytes(errors)
    if overflow.is_set():
        failure = 'reviewer_output_budget_exceeded'
    if failure or failures or process.returncode:
        raise RuntimeError(failure or 'reviewer_process_failed')
    return strict_json(bytes(output).decode('utf-8'))


def invoke(reviewer, packet, folder, round_index, timeout):
    config = reviewer['config']
    if sha(reviewer['path']) != reviewer['identity']['config_sha256'] or sha(reviewer['prompt']) != reviewer['identity']['prompt_sha256']:
        raise ValueError('reviewer_config_or_prompt_changed')
    packet = json.loads(json.dumps(packet, ensure_ascii=False))
    packet['reviewer_identity_hash'] = reviewer['fingerprint']
    cid = None
    environment = None
    if reviewer['identity']['backend'] == 'docker':
        from runtime import sandbox
        cid = Path(folder) / f'reviewer_{round_index:02d}.cid'
        for entry in packet['evidence']:
            entry['path'] = '/evidence/' + entry['id']
        runtime_config = Path(folder) / f'reviewer_execution_{round_index:02d}.json'
        runtime_config.write_text(json.dumps({'command': config['command'], 'artifacts': config['executable_artifacts']}), encoding='utf-8')
        command = sandbox(config['image']) + ['-i']
        # A named evaluator-managed network may be required for a remote judge API.
        network = config.get('network', 'none')
        if not isinstance(network, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', network) or network in ('host', 'container'):
            raise ValueError('invalid reviewer network')
        command[command.index('--network') + 1] = network
        for name in config.get('pass_env', []):
            if not isinstance(name, str) or not re.fullmatch('[A-Z][A-Z0-9_]*', name):
                raise ValueError('invalid reviewer environment variable name')
            command += ['-e', name]
        command += ['--cidfile', str(cid), '--mount', f'type=bind,src={folder},dst=/evidence,readonly',
                    '--mount', f'type=bind,src={Path(__file__).resolve().parent},dst=/protocol,readonly',
                    '--mount', f'type=bind,src={reviewer["prompt"]},dst=/reviewer-prompt.txt,readonly',
                    '-e', 'REVIEWER_PROMPT_FILE=/reviewer-prompt.txt', '--entrypoint', 'python3', config['image'],
                    '/protocol/reviewer_entry.py', '/evidence/' + runtime_config.name]
    else:
        for filename, expected in reviewer['identity']['executable_artifacts'].items():
            if sha(filename) != expected:
                raise ValueError('reviewer_executable_changed')
        command = config['command']
        environment = dict(os.environ, PYTHONUTF8='1', REVIEWER_PROMPT_FILE=str(reviewer['prompt']))
    try:
        response = bounded_process(command, json.dumps(packet, ensure_ascii=False, allow_nan=False).encode('utf-8'), timeout,
                                   Path(folder) / f'reviewer_{round_index:02d}.log', environment)
    finally:
        if cid and cid.is_file():
            subprocess.run(['docker', 'rm', '-f', cid.read_text(encoding='utf-8').strip()],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
    if sha(reviewer['path']) != reviewer['identity']['config_sha256'] or sha(reviewer['prompt']) != reviewer['identity']['prompt_sha256']:
        raise ValueError('reviewer_config_or_prompt_changed')
    if reviewer['identity']['backend'] == 'local':
        for filename, expected in reviewer['identity']['executable_artifacts'].items():
            if sha(filename) != expected:
                raise ValueError('reviewer_executable_changed')
    if response.get('reviewer_identity_hash') != reviewer['fingerprint']:
        raise ValueError('reviewer_identity_binding_mismatch')
    return response
