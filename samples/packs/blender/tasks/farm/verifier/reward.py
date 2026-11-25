#!/usr/bin/env python3
"""rl_env terminal evaluator: general eligibility followed by a task evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from general_contract import CHECKS, check, publish, result, validate_checks, validate_sparse_result
from runtime import IMAGE, sandbox, resolve_image
from worker_runtime import WorkerRuntime
from support import run_native, BudgetExceeded, POLICY, file_preflight, freeze_submission, Deadline, identity_hash, source_identity, read_json


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, default=ROOT.parent / 'output/result.blend')
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--task', choices=['general', 'farm-reconstruction-v0'], default='general')
    parser.add_argument('--backend', choices=['docker', 'local'], default='docker')
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--timeout', type=int, default=POLICY['timeout_sec'])
    parser.add_argument('--resolution', type=int, nargs=2, default=None)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--farm-reviewer', type=Path, help='Evaluator-owned reviewer command config, outside submission directory')
    parser.add_argument('--farm-review-rounds', type=int, default=2)
    parser.add_argument('--allow-local-reviewer', action='store_true', help='Development only: allow an unsandboxed reviewer')
    parser.add_argument('--calibration-mode', action='store_true', help='Collect judge calibration results; NEVER emits training reward')
    args = parser.parse_args(argv)
    args.evaluation_id = uuid.uuid4().hex
    args.source_identity = source_identity(ROOT)
    args.resolution = args.resolution or ([960, 720] if args.task == 'farm-reconstruction-v0' else [320, 240])
    args.samples = args.samples if args.samples is not None else (16 if args.task == 'farm-reconstruction-v0' else 4)
    if not 0 <= args.farm_review_rounds <= 4:
        parser.error('farm review rounds must be in [0,4]')
    source, out = args.submission.absolute(), args.reports.resolve()
    args.submission_dir = source.parent.resolve()
    deadline = Deadline(args.timeout)
    args.deadline = deadline
    if out.exists():
        parser.error('reports must be a new directory')
    if args.timeout <= 0 or not 1 <= args.samples <= 4096 or any(not 16 <= n <= 8192 for n in args.resolution):
        parser.error('invalid timeout, samples or resolution')
    if source.parent.resolve() == out or source.parent.resolve() in out.parents or out in source.resolve().parents:
        parser.error('reports must be outside the submission directory')
    if args.backend == 'docker' and any(',' in str(p) for p in (source, out, ROOT)):
        parser.error('Docker mount paths must not contain commas')
    out.mkdir(parents=True)
    checks = [check(k) for k in CHECKS]
    artifact_hash = None
    task_result = None
    def finish():
        data = result(checks, artifact_hash, args.task, task_result)
        data['evaluation_id'] = args.evaluation_id
        data['verifier_source_sha256'] = args.source_identity
        data['runtime'] = {'image': args.image if args.backend == 'docker' else None, 'backend': args.backend, 'policy_hash': identity_hash(POLICY), 'expected_blender': POLICY['blender_version']}
        if args.calibration_mode:
            data['candidate_evaluation'] = {k: data[k] for k in ('status', 'reward', 'training_mask', 'accepted')}
            data.update(status='NEEDS_REVIEW', reward=None, training_mask=0, accepted=None, reason='calibration_only')
        publish(out, data)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0 if data['status'] in ('PASS', 'FAIL') else 2
    try:
        preflight = file_preflight(source)
    except OSError as exc:
        checks[0] = check('project_readable', 'UNKNOWN', 'preflight_io_error:' + type(exc).__name__)
        return finish()
    if preflight:
        checks[0] = check('project_readable', 'FAIL', preflight)
        return finish()
    if not shutil.which(args.blender if args.backend == 'local' else 'docker'):
        checks[0] = check('project_readable', 'UNKNOWN', 'runtime_unavailable')
        return finish()
    try:
        if args.backend == 'docker':
            args.image = resolve_image(args.image)
    except Exception as exc:
        checks[0] = check('project_readable', 'UNKNOWN', 'runtime_identity_failed:' + str(exc))
        return finish()
    try:
        source = freeze_submission(source, out / 'input_snapshot')
        artifact_hash = digest(source)
    except ValueError as exc:
        checks[0] = check('project_readable', 'FAIL', 'invalid_submission_bundle:' + str(exc))
        return finish()
    except Exception as exc:
        checks[0] = check('project_readable', 'UNKNOWN', 'preparation_failed:' + str(exc))
        return finish()
    # Checks are implemented by the evaluator; the agent cannot supply measurements.
    worker = ROOT / 'execution_check.py'
    workers = WorkerRuntime(args, source, out, ROOT, deadline)
    command, run = workers.command, workers.run
    input_arg = str(source) if args.backend == 'local' else '/input/' + source.name
    output_arg = str(out / 'general') if args.backend == 'local' else '/output/general'
    code = run(command(worker, ['--submission', input_arg, '--output', output_arg,
                '--resolution', *map(str, args.resolution), '--samples', str(args.samples)], 'general'), 'general')
    report = out / 'general/general_checks.json'
    if code == 0 and report.is_file():
        try:
            measured = read_json(report)
            if measured['artifact_hash'] != artifact_hash or digest(source) != artifact_hash:
                raise ValueError('artifact changed')
            checks = validate_checks(measured['checks'])
            for item in checks:
                item['evidence'] = ['general/general_checks.json'] + ['general/' + p for p in item['evidence']]
        except (ValueError, KeyError, TypeError, OSError):
            checks = [check(k, 'UNKNOWN', 'invalid_or_stale_evaluator_report') for k in CHECKS]
    else:
        checks = [check(k, 'UNKNOWN', { -1:'evaluator_timeout', -3:'evaluator_output_budget_or_unsafe_output', -4:'evaluator_oom_needs_triage' }.get(code, 'evaluator_failed_needs_triage')) for k in CHECKS]
    general = result(checks, artifact_hash)
    if general['general_status'] != 'PASS' or args.task == 'general':
        return finish()
    if args.task == 'farm-reconstruction-v0':
        from render_eval import evaluate
        try:
            task_result = evaluate(args, source, out, checks, artifact_hash, command, run)
        except Exception as exc:
            task_result = {'status': 'ERROR', 'training_mask': 0, 'reward': None,
                           'reason': 'farm_evaluation_failed:' + type(exc).__name__ + ':' + str(exc)}
        if digest(source) != artifact_hash:
            checks[0] = check('project_readable', 'UNKNOWN', 'artifact_changed_during_farm_evaluation')
        return finish()
    task_dir = ROOT
    options = ['--submission', input_arg, '--report-dir', str(out / 'task') if args.backend == 'local' else '/output/task']
    if args.backend == 'local': options += ['--blender', args.blender]
    cmd = command(task_dir / 'task_check.py', options, 'task', python=True)
    code = run(cmd, 'task')
    task_file = out / 'task/result.json'
    if code in (0, 2) and task_file.is_file():
        try:
            task_result = validate_sparse_result(read_json(task_file))
        except (ValueError, TypeError, OSError):
            task_result = None
    if task_result is None:
        task_result = {'status': 'ERROR', 'reason': 'task_evaluator_failed', 'reward': None, 'training_mask': 0}
    if digest(source) != artifact_hash:
        checks[0] = check('project_readable', 'UNKNOWN', 'artifact_changed_during_task_evaluation')
    return finish()


if __name__ == '__main__':
    raise SystemExit(main())
