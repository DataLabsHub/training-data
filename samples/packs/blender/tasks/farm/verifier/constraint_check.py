"""Validate evaluator-owned review evidence and aggregate the frozen Farm rubric."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_rubric(rubric):
    items=rubric['atomic_criteria']['items']
    ids=[i['criterion_id'] for i in items]
    if len(ids)!=len(set(ids)) or sum(i['weight'] for i in items)!=100:raise ValueError('invalid rubric weights or IDs')
    for dim in rubric['dimensions']:
        if sum(i['weight'] for i in items if i['dimension']==dim['id'])!=dim['weight']:raise ValueError('dimension budget mismatch')
    return items


def evidence_roles(evidence, refs):
    roles = set()
    for ref in refs:
        roles.update(evidence[ref].get('roles', []))
    if 'gray_reverse' in roles:
        roles.add('gray')
    images = {evidence[r].get('sha256', r) for r in refs
              if evidence[r]['kind'] == 'image' and 'reference' not in evidence[r].get('roles', [])}
    if len(images) >= 2:
        roles.add('multiview')
    return roles, len(images)


def score(rubric, packet, review=None):
    items = validate_rubric(rubric)
    definitions = {i['criterion_id']: i for i in items}
    evidence = {e['id']: e for e in packet['evidence']}
    if len(evidence) != len(packet['evidence']):
        raise ValueError('duplicated evidence ID')
    received = {}
    validity = {'status': 'UNKNOWN', 'reason': 'awaiting_independent_validity_review', 'evidence': []}
    if review is not None:
        for field in ('artifact_hash', 'reference_hash', 'rubric_hash', 'reviewer_identity_hash'):
            if field in packet and review.get(field) != packet[field]:
                raise ValueError('review binding mismatch: ' + field)
        for item in review.get('criterion_scores', []):
            cid = item['criterion_id']
            if cid in received or cid not in definitions:
                raise ValueError('unknown or duplicated criterion')
            item = dict(item)
            if item.get('status') not in ('SCORED', 'UNKNOWN'):
                raise ValueError('N_A and weight overrides are not allowed')
            if any(k in item for k in ('weight', 'required', 'pass_score')):
                raise ValueError('review cannot override rubric')
            if not isinstance(item.get('reason'), str) or not item['reason'].strip():
                raise ValueError('all items require a reason')
            if item['status'] == 'SCORED':
                if type(item.get('score')) is not int or not 0 <= item['score'] <= 4:
                    raise ValueError('invalid score')
                refs = item.get('evidence', [])
                if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in evidence for r in refs):
                    raise ValueError('unknown evidence reference')
                images = [r for r in refs if evidence[r]['kind'] == 'image' and r != 'reference.png']
                if not cid.startswith('F9.') and not images:
                    raise ValueError('visual judgement requires candidate images')
                if cid.startswith('F2.') and 'main.png' not in refs:
                    raise ValueError('composition requires original main view')
                if cid.startswith(('F2.', 'F3.', 'F4.', 'F5.', 'F6.')) and 'reference.png' not in refs:
                    raise ValueError('likeness judgement requires reference')
                roles, views = evidence_roles(evidence, refs)
                policy = definitions[cid]['evidence_policy']
                missing = sorted(set(policy['required_roles']) - roles)
                if missing or views < policy.get('min_submission_views', 0):
                    item.update(status='UNKNOWN', score=None, reason='required_evidence_missing:' + ','.join(missing))
                # Programmatically observed round-trip failure cannot be overruled by a judge.
                if cid == 'F9.roundtrip' and item['status'] == 'SCORED':
                    statuses = [evidence[r].get('measurement_status') for r in refs if 'roundtrip' in evidence[r].get('roles', [])]
                    if 'FAIL' in statuses:
                        item.update(score=0, reason='trusted_roundtrip_failed')
                    elif not statuses or any(v != 'PASS' for v in statuses):
                        item.update(status='UNKNOWN', score=None, reason='roundtrip_status_unknown')
            elif item.get('score') is not None:
                raise ValueError('UNKNOWN score must be null')
            received[cid] = item
        proposed = review.get('submission_validity')
        if proposed is not None:
            if not isinstance(proposed, dict) or proposed.get('status') not in ('VALID', 'INVALID', 'UNKNOWN'):
                raise ValueError('invalid submission_validity schema')
            if not isinstance(proposed.get('reason'), str) or not proposed['reason'].strip():
                raise ValueError('validity requires an observed reason')
            refs = proposed.get('evidence', [])
            if not isinstance(refs, list) or any(not isinstance(r, str) or r not in evidence for r in refs):
                raise ValueError('validity refers to unknown evidence')
            roles, _ = evidence_roles(evidence, refs)
            validity = dict(proposed)
            if validity['status'] in ('VALID', 'INVALID'):
                required = set(rubric['submission_validation']['protocol']['invalid_required_roles'])
                if not required <= roles:
                    validity = {'status': 'UNKNOWN', 'reason': 'validity_evidence_incomplete', 'evidence': refs}
            if validity['status'] == 'INVALID':
                if validity.get('code') not in rubric['submission_validation']['protocol']['invalid_codes']:
                    raise ValueError('unrecognized invalid submission code')
    editability = received.get('F9.editability', {})
    if editability.get('status') == 'SCORED' and editability.get('score') == 0:
        # Score zero's frozen anchor confirms the core native/editable capability is absent.
        validity = {'status': 'INVALID', 'code': 'core_not_editable',
                    'reason': 'confirmed_zero_editability', 'evidence': editability.get('evidence', [])}
    rows = []
    for definition in items:
        cid = definition['criterion_id']
        item = received.get(cid, {'status': 'UNKNOWN', 'score': None, 'reason': 'awaiting_trusted_evidence_review', 'evidence': []})
        rows.append({'criterion_id': cid, 'dimension': definition['dimension'], 'title': definition['title'],
                     'weight': definition['weight'], 'required': definition['required'], 'pass_score': definition['pass_score'],
                     'status': item['status'], 'score': item.get('score'),
                     'satisfied': None if item['status'] == 'UNKNOWN' else item['score'] >= definition['pass_score'],
                     'reason': item['reason'], 'evidence': item.get('evidence', []), 'asset_revision': packet['artifact_hash']})
    unknown = [r['criterion_id'] for r in rows if r['status'] == 'UNKNOWN']
    failed = [r['criterion_id'] for r in rows if r['required'] and r['satisfied'] is False]
    contribution = sum(r['weight'] * r['score'] / 4 for r in rows if r['status'] == 'SCORED')
    q = None if unknown else round(contribution, 6)
    accepted = False if failed else (None if unknown else q >= rubric['acceptance']['quality_threshold'])
    if validity['status'] == 'UNKNOWN':
        accepted = None
    reward = None
    status = 'NEEDS_REVIEW'
    mask = 0
    reason = 'unresolved_items_or_validity_have_no_scalar_reward'
    if validity['status'] == 'INVALID':
        reward, accepted, status, mask = 0.0, False, 'FAIL', 1
        reason = 'confirmed_invalid_submission'
    elif validity['status'] == 'VALID' and not unknown:
        reward = round(rubric['reward']['quality_coefficient'] * q / 100 + rubric['reward']['acceptance_coefficient'] * int(accepted), 6)
        status, mask, reason = ('PASS' if accepted else 'FAIL'), 1, 'evaluation_complete'
    return {'schema': 'rl_env.farm_result.v2', 'status': status, 'rubric_version': rubric['rubric_version'],
            'rubric_hash': packet['rubric_hash'], 'artifact_hash': packet['artifact_hash'], 'reference_hash': packet['reference_hash'],
            'reviewer_identity_hash': packet.get('reviewer_identity_hash'), 'submission_validity': validity,
            'invalid_submission': validity['status'] == 'INVALID', 'criterion_scores': rows, 'quality_score': q,
            'known_contribution': contribution, 'evidence_coverage_weight': sum(r['weight'] for r in rows if r['status'] == 'SCORED'),
            'failed_requirements': failed, 'unknown_requirements': unknown, 'accepted': accepted,
            'reward': reward, 'training_mask': mask, 'reason': reason}


def write_html_report(path, rubric, packet, result):
    """Human-readable criterion checklist with local evidence links."""
    from html import escape
    from urllib.parse import quote
    definitions={i['criterion_id']:i for i in rubric['atomic_criteria']['items']}
    blocks=[]
    for row in result['criterion_scores']:
        rule=definitions[row['criterion_id']]
        facts=''.join('<li>'+escape(t)+'</li>' for t in rule['observable_requirements'])
        links=' '.join('<a href="'+quote(ref,safe='/')+'">'+escape(ref)+'</a>' for ref in row['evidence'])
        label='未知' if row['satisfied'] is None else ('满足' if row['satisfied'] else '不满足')
        blocks.append('<section><h2>'+escape(row['criterion_id']+' '+row['title'])+'</h2>'+
          '<p>权重 '+str(row['weight'])+' · 分数 '+str(row['score'])+' · '+label+'</p><ul>'+facts+'</ul><p>'+escape(row['reason'])+'</p>'+links+'</section>')
    previews=''.join('<figure><img width="420" src="'+quote(e['id'],safe='/')+'"><figcaption>'+escape(e['id'])+'</figcaption></figure>'
                     for e in packet['evidence'] if e['kind']=='image')
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Farm逐项评测</title><style>body{max-width:1100px;margin:32px auto;font:16px/1.6 sans-serif;padding:16px}section{border-top:1px solid #ccc;padding:12px}figure{display:inline-block;vertical-align:top;margin:8px}img{max-width:100%}h2{font-size:19px}a{overflow-wrap:anywhere}</style><h1>Farm 专用评测</h1><p>'+escape(result['status'])+' · Q='+str(result['quality_score'])+' · reward='+str(result['reward'])+'</p><p>产物 '+escape(result['artifact_hash'])+'</p><p>未完成项不生成正式奖励；此报告不将渲染成功视作质量通过。</p>'+previews+''.join(blocks)+'</html>'
    Path(path).write_text(html, encoding="utf-8")
