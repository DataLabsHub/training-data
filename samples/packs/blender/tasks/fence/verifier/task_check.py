"""Fresh extraction -> trusted metrics -> sparse reward.

Exit 0 means a completed PASS/FAIL evaluation. ERROR/BLOCKED exit 2 and omit
reward.txt. Run in a separate trusted container for meaningful isolation.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from constraint_check import TARGETS, score
from support import file_preflight, run_native

HERE = Path(__file__).resolve().parent


def publish(directory, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'reward.txt').unlink(missing_ok=True)
    (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result['status'] in ('PASS', 'FAIL'):
        (directory / 'reward.txt').write_text(str(result['reward']) + '\n', encoding="utf-8")


def run_blender(executable, script, args, log, timeout=60):
    return run_native([executable, '--background', '--factory-startup',
        '--disable-autoexec', '--python-exit-code', '3', '--python', str(script),
        '--', *map(str, args)], log, timeout, Path(log).parent)



def verify(submission, report_dir, blender='blender'):
    submission, report_dir = Path(submission), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Delete stale success before any subprocess or preflight can fail.
    (report_dir / 'reward.txt').unlink(missing_ok=True)
    result = {'status': 'ERROR', 'training_mask': 0, 'reason': 'incomplete_verification'}
    try:
        preflight = file_preflight(submission)
        if preflight:
            result = {'status': 'FAIL', 'reward': 0.0, 'training_mask': 1, 'reason': preflight}
        elif not shutil.which(blender):
            result['reason'] = 'blender_runtime_unavailable'
        else:
            with tempfile.TemporaryDirectory(prefix='blender-verifier-') as tmp:
                tmp = Path(tmp)
                baseline, baseline_json, candidate_json = (tmp / x for x in
                    ['baseline.blend', 'baseline.json', 'candidate.json'])
                # Establish grader health independently before blaming an artifact.
                rc = run_blender(blender, HERE / 'fixture.py', [baseline], report_dir / 'baseline.log')
                if rc != 0:
                    raise RuntimeError('baseline_generation_failed')
                rc = run_blender(blender, HERE / 'extract_scene.py', [baseline, baseline_json], report_dir / 'extract_baseline.log')
                if rc != 0:
                    raise RuntimeError('baseline_extraction_failed')
                before = json.loads(baseline_json.read_text(encoding="utf-8"))
                health = score(before, before)
                if set(health['failures']) != {name + ':not_grounded' for name in TARGETS}:
                    raise RuntimeError('baseline_contract_drift')
                if submission.is_symlink() or not submission.is_file():
                    result = {'status': 'FAIL', 'reward': 0.0, 'training_mask': 1, 'reason': 'missing_or_symlink_submission'}
                elif submission.stat().st_size > 512 * 1024 * 1024:
                    result = {'status': 'FAIL', 'reward': 0.0, 'training_mask': 1, 'reason': 'artifact_exceeds_512_mib'}
                else:
                    rc = run_blender(blender, HERE / 'extract_scene.py', [submission, candidate_json], report_dir / 'extract_submission.log')
                    if rc != 0:
                        # Native crashes/OOM/runtime faults need triage, not automatic negative training.
                        result = {'status': 'ERROR', 'training_mask': 0, 'reason': 'submission_extraction_failed_needs_triage', 'exit_code': rc}
                    else:
                        result = score(before, json.loads(candidate_json.read_text(encoding="utf-8")))
    except subprocess.TimeoutExpired:
        result = {'status': 'ERROR', 'training_mask': 0, 'reason': 'verifier_timeout_needs_triage'}
    except Exception as exc:
        result = {'status': 'ERROR', 'training_mask': 0, 'reason': type(exc).__name__ + ': ' + str(exc)}
    publish(report_dir, result)
    return 0 if result['status'] in ('PASS', 'FAIL') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='/task/output/result.blend')
    parser.add_argument('--report-dir', default='/logs/verifier')
    parser.add_argument('--blender', default='blender')
    args = parser.parse_args()
    raise SystemExit(verify(args.submission, args.report_dir, args.blender))
