"""General v1 is an eligibility gate, not an appearance reward."""
import json
from pathlib import Path

VERSION = 'general-basic-v1'
CHECKS = ('project_readable', 'geometry_valid', 'resources_complete', 'mesh_diagnostics', 'inspection_render')
REQUIRED = tuple(c for c in CHECKS if c != 'mesh_diagnostics')


def check(check_id, status='UNKNOWN', reason='not_checked', measurements=None, evidence=None):
    return {'check_id': check_id, 'status': status, 'target': 'submission',
            'measurements': measurements or {}, 'reason': reason,
            'evidence': evidence or [], 'checker_version': VERSION}


def result(checks, artifact_hash=None, task='general', task_result=None):
    by_id = {c['check_id']: c for c in checks}
    checks = [dict(by_id.get(k, check(k)), artifact_hash=artifact_hash) for k in CHECKS]
    failed = [c['check_id'] for c in checks if c['check_id'] in REQUIRED and c['status'] == 'FAIL']
    unknown = [c['check_id'] for c in checks if c['check_id'] in REQUIRED and c['status'] == 'UNKNOWN']
    # A proven hard failure is decisive even if downstream checks were skipped.
    gate = 'FAIL' if failed else ('UNKNOWN' if unknown else 'PASS')
    out = {'schema': 'rl_env.evaluation.v1', 'rubric_version': VERSION,
           'artifact_hash': artifact_hash, 'task': task, 'general_status': gate,
           'checks': checks, 'failed_requirements': failed, 'unknown_requirements': unknown,
           'quality_score': None, 'accepted': False if failed else None,
           'reward': None, 'training_mask': 0, 'task_result': task_result,
           'status': gate, 'scope': 'basic_eligibility_not_visual_quality'}
    if gate == 'FAIL':
        out.update(reward=0.0, training_mask=1)
    elif gate == 'UNKNOWN':
        out['status'] = 'ERROR'
    elif task == 'general':
        out['reason'] = 'eligible_for_task_evaluation_no_quality_reward'
    elif task_result is None:
        out.update(status='BLOCKED', reason='task_evaluator_unavailable')
    elif task_result.get('status') in ('PASS', 'FAIL') and task_result.get('training_mask') == 1:
        out.update(status=task_result['status'], reward=task_result['reward'], training_mask=1,
                   accepted=task_result['status'] == 'PASS')
    else:
        out.update(status=task_result.get('status', 'ERROR'), reason='task_evaluation_incomplete')
    if gate == 'PASS' and task_result is not None:
        for key in ('quality_score', 'criterion_scores', 'accepted', 'failed_requirements', 'unknown_requirements'):
            if key in task_result:
                out[key] = task_result[key]
        out['task_rubric_version'] = task_result.get('rubric_version')
    return out


def publish(folder, data):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'reward.txt').unlink(missing_ok=True)
    (folder / 'result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if data['training_mask'] == 1 and data['reward'] is not None:
        (folder / 'reward.txt').write_text(str(data['reward']) + '\n', encoding="utf-8")


def validate_checks(checks):
    """Validate native worker messages before they can affect controller rewards."""
    from pathlib import PurePosixPath
    if not isinstance(checks,list) or len(checks)!=len(CHECKS):
        raise ValueError('incomplete general report')
    if any(not isinstance(c,dict) for c in checks) or {c.get('check_id') for c in checks}!=set(CHECKS):
        raise ValueError('duplicate or unknown general check')
    for c in checks:
        allowed={'PASS','FAIL','UNKNOWN'} | ({'N_A'} if c['check_id'] in ('resources_complete','mesh_diagnostics') else set())
        if c.get('status') not in allowed or not isinstance(c.get('reason'),str) or not isinstance(c.get('measurements'),dict):
            raise ValueError('invalid native check schema')
        refs=c.get('evidence')
        if not isinstance(refs,list) or any(not isinstance(ref,str) or not ref or PurePosixPath(ref).is_absolute() or '..' in PurePosixPath(ref).parts or '\\' in ref for ref in refs):
            raise ValueError('native evidence path escape')
    return checks


def validate_sparse_result(data):
    if not isinstance(data,dict) or data.get('status') not in ('PASS','FAIL','ERROR','BLOCKED','NEEDS_REVIEW'):
        raise ValueError('invalid task result schema')
    if data['status'] in ('PASS','FAIL'):
        if type(data.get('training_mask')) is not int or data['training_mask']!=1 or type(data.get('reward')) not in (int,float):
            raise ValueError('invalid sparse reward')
        if data['reward']!=(1 if data['status']=='PASS' else 0):raise ValueError('sparse reward conflicts with status')
    elif data.get('training_mask')!=0 or data.get('reward') is not None:
        raise ValueError('unresolved task returned a reward')
    return data
