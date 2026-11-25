#!/usr/bin/env python3
"""Human-labeled validity, 31-criterion accuracy and repeatability calibration.

No synthetic fixture or manually asserted PASS can replace this aggregation.
Manifest v2: cases contain artifact_sha256, category, annotations (two raters),
optional adjudication, and results [{path, sha256}] from calibration-mode trials.
"""
import argparse
import math
from pathlib import Path
import re
from collections import Counter
from support import identity_hash, read_json, dump_json
from constraint_check import file_hash, validate_rubric
HERE = Path(__file__).resolve().parent
POLICY = read_json(HERE / 'calibration-policy.json')
RUBRIC = read_json(HERE / 'rubric.json')
ITEMS = validate_rubric(RUBRIC)
REQUIRED_ATTACKS = set(POLICY['required_attack_categories'])
HEX = re.compile('[0-9a-f]{64}')


def fingerprint(identity):
    return identity_hash({k: v for k, v in identity.items() if k not in ('config_sha256', 'calibration_sha256')})


def _scores(scores):
    if not isinstance(scores, dict) or set(scores) != {c['criterion_id'] for c in ITEMS}:
        raise ValueError('all 31 human criterion scores are required')
    if any(type(v) is not int or not 0 <= v <= 4 for v in scores.values()):
        raise ValueError('human scores must be integers in [0,4]')
    return scores


def gold(case):
    annotations = case.get('annotations', [])
    if len(annotations) < 2 or len({a.get('annotator_id') for a in annotations}) != len(annotations):
        raise ValueError('two distinct human annotators are required')
    for row in annotations:
        if not isinstance(row.get('annotator_id'), str) or not row['annotator_id'].strip(): raise ValueError('missing annotator')
        if row.get('validity') not in ('VALID', 'INVALID'): raise ValueError('unresolved human validity')
        if row['validity'] == 'VALID': _scores(row.get('scores'))
    disagreement = []
    for left in range(len(annotations)):
        for right in range(left+1, len(annotations)):
            a,b = annotations[left],annotations[right]
            if a['validity']==b['validity']=='VALID':
                disagreement += [abs(a['scores'][k]-b['scores'][k]) for k in a['scores']]
    differing = any(a['validity']!=annotations[0]['validity'] or a.get('scores')!=annotations[0].get('scores') for a in annotations)
    chosen = case.get('adjudication') if differing else annotations[0]
    if differing:
        if (not isinstance(chosen,dict) or chosen.get('annotator_id') in {a['annotator_id'] for a in annotations}
                or not chosen.get('annotator_id') or not isinstance(chosen.get('reason'),str) or not chosen['reason'].strip()):
            raise ValueError('disagreeing labels require independent human adjudication')
    if chosen.get('validity') not in ('VALID','INVALID'): raise ValueError('unresolved adjudication')
    if chosen['validity']=='VALID': _scores(chosen.get('scores'))
    return chosen, disagreement


def aggregate(manifest, base):
    if manifest.get('schema') != 'farm.calibration_manifest.v2' or manifest.get('purpose') != 'held_out_acceptance':
        raise ValueError('held-out calibration manifest v2 is required')
    if manifest.get('labels_complete') is not True or not manifest.get('dataset_id'):
        raise ValueError('human labels have not been completed')
    if manifest.get('rubric_sha256') != file_hash(HERE/'rubric.json'):
        raise ValueError('labels refer to a different rubric')
    identities, artifacts, runs, report_hashes = set(),set(),set(),set()
    counts=Counter(); attacks=Counter(); bands=set(); records=[]; failures=[]
    per_criterion={c['criterion_id']:[] for c in ITEMS}
    q_errors=[]; reward_errors=[]; acceptance=[]; human_errors=[]; repeat_q=[]; repeat_reward=[]
    within=[]; repeated=0
    for case in manifest.get('cases',[]):
        artifact=case.get('artifact_sha256','')
        if not HEX.fullmatch(artifact) or artifact in artifacts: raise ValueError('duplicate or invalid artifact identity')
        artifacts.add(artifact)
        label, disagreement=gold(case); human_errors.extend(disagreement)
        valid=label['validity']=='VALID'; counts[label['validity']]+=1
        if not valid: attacks[case['category']]+=1
        scores=label.get('scores')
        quality=sum(c['weight']*scores[c['criterion_id']]/4 for c in ITEMS) if valid else None
        accepted=bool(valid and quality>=RUBRIC['acceptance']['quality_threshold'] and all(scores[c['criterion_id']]>=c['pass_score'] for c in ITEMS if c['required']))
        expected_reward=RUBRIC['reward']['quality_coefficient']*quality/100+RUBRIC['reward']['acceptance_coefficient']*accepted if valid else 0
        if valid:
            counts['gold_accepted' if accepted else 'gold_rejected']+=1
            bands.add('low' if quality<50 else ('medium' if quality<85 else 'high'))
        trials=[]; case_errors=[]; case_q=[]; case_reward=[]; case_accept=[]
        if len(case.get('results',[]))>10: raise ValueError('at most ten trials per case')
        for entry in case.get('results',[]):
            path=(base/entry['path']).resolve()
            actual_hash=file_hash(path)
            if actual_hash!=entry.get('sha256') or actual_hash in report_hashes: raise ValueError('stale or duplicated trial report')
            report_hashes.add(actual_hash)
            report=read_json(path)
            identity=report.get('reviewer_identity')
            if (report.get('schema')!='rl_env.farm_result.v2' or report.get('evaluation_mode')!='calibration'
                    or not identity or identity.get('backend')!='docker'
                    or not identity.get('evidence_runtime') or identity['evidence_runtime'].get('backend')!='docker'):
                raise ValueError('a real Docker calibration-mode report is required')
            if report.get('reviewer_identity_hash')!=identity_hash(identity): raise ValueError('reviewer identity mismatch')
            if identity.get('rubric_sha256')!=manifest['rubric_sha256']: raise ValueError('rubric identity mismatch')
            identities.add(fingerprint(identity))
            run=report.get('evaluation_id')
            if not isinstance(run,str) or not re.fullmatch('[0-9a-f]{32}',run) or run in runs: raise ValueError('trial must have a unique evaluator run ID')
            runs.add(run)
            if report.get('artifact_hash')!=artifact: raise ValueError('trial artifact mismatch')
            actual=report.get('submission_validity',{}).get('status')
            if actual not in ('VALID','INVALID','UNKNOWN'): raise ValueError('invalid observed validity')
            predicted=report.get('candidate_evaluation')
            if not isinstance(predicted,dict): raise ValueError('missing candidate evaluation')
            value=predicted.get('reward'); q=report.get('quality_score')
            complete=type(value) in (int,float) and math.isfinite(value) and 0<=value<=1 and predicted.get('training_mask')==1
            if actual=='INVALID' and (not complete or value!=0 or predicted.get('accepted') is not False or predicted.get('status')!='FAIL'):
                raise ValueError('invalid trial violated hard zero')
            if complete and actual=='VALID':
                prediction=report.get('criterion_scores',[])
                canonical={v['criterion_id']:v for v in prediction}
                if len(prediction)!=len(ITEMS) or set(canonical)!={c['criterion_id'] for c in ITEMS} or any(v.get('status')!='SCORED' or type(v.get('score')) is not int or not 0<=v['score']<=4 for v in canonical.values()):
                    raise ValueError('complete valid trial lacks fixed criterion scores')
                expected_q=sum(c['weight']*canonical[c['criterion_id']]['score']/4 for c in ITEMS)
                expected_a=bool(expected_q>=RUBRIC['acceptance']['quality_threshold'] and all(canonical[c['criterion_id']]['score']>=c['pass_score'] for c in ITEMS if c['required']))
                expected_r=RUBRIC['reward']['quality_coefficient']*expected_q/100+RUBRIC['reward']['acceptance_coefficient']*expected_a
                if type(q) not in (int,float) or not math.isfinite(q) or abs(q-expected_q)>1e-5 or abs(value-expected_r)>1e-5 or predicted.get('accepted') is not expected_a or predicted.get('status')!=('PASS' if expected_a else 'FAIL'):
                    raise ValueError('trial conflicts with immutable reward formula')
            if actual=='UNKNOWN' and complete: raise ValueError('unknown validity produced training reward')
            counts['trials']+=1;counts['unknown']+=not complete
            counts['invalid_trials' if not valid else 'valid_trials']+=1
            counts['false_positive']+=not valid and actual=='VALID'
            counts['false_negative']+=valid and actual=='INVALID'
            if valid and complete:
                rows={v['criterion_id']:v for v in report.get('criterion_scores',[])}
                if actual!='VALID' or set(rows)!=set(scores) or any(v.get('status')!='SCORED' or type(v.get('score')) is not int or not 0<=v['score']<=4 for v in rows.values()):
                    # Misclassified valid cases incur maximum score error, never disappear from quality metrics.
                    errors={k:4 for k in scores}; measured_q=0; measured_accepted=False
                else:
                    measured_q=sum(c['weight']*rows[c['criterion_id']]['score']/4 for c in ITEMS)
                    measured_accepted=bool(measured_q>=RUBRIC['acceptance']['quality_threshold'] and all(rows[c['criterion_id']]['score']>=c['pass_score'] for c in ITEMS if c['required']))
                    recomputed=RUBRIC['reward']['quality_coefficient']*measured_q/100+RUBRIC['reward']['acceptance_coefficient']*measured_accepted
                    if type(q) not in (int,float) or not math.isfinite(q) or abs(q-measured_q)>1e-5 or abs(value-recomputed)>1e-5 or predicted.get('accepted')!=measured_accepted:
                        raise ValueError('trial result is inconsistent with fixed rubric')
                    errors={k:abs(rows[k]['score']-scores[k]) for k in scores}
                case_errors.append(errors);case_q.append(abs(measured_q-quality));case_reward.append(abs(value-expected_reward));case_accept.append(measured_accepted==accepted)
                counts['quality_trials']+=1
            trials.append({'report_sha256':actual_hash,'evaluation_id':run,'validity':actual,'reward':value if complete else None,'quality':q if complete else None})
        if not trials: raise ValueError('each labeled case needs actual evaluation results')
        counts['false_positive_cases']+=not valid and any(v['validity']=='VALID' for v in trials)
        counts['false_negative_cases']+=valid and any(v['validity']=='INVALID' for v in trials)
        if case_errors:
            for key in per_criterion:
                error=max(v[key] for v in case_errors);per_criterion[key].append(error);within.append(error<=1)
            q_errors.append(max(case_q));reward_errors.append(max(case_reward));acceptance.append(all(case_accept))
        # Case-level failure accounting prevents successful repeats from hiding a bad case.
        if any(v['reward'] is None for v in trials): counts['unresolved_cases']+=1
        if valid and all(v['reward'] is not None for v in trials): counts['quality_cases']+=1
        if valid and len(trials)>=POLICY['minimum_repeats']:
            repeated+=1
            if len({v['validity'] for v in trials})!=1 or any(v['reward'] is None for v in trials): failures.append('repeat_validity_or_completion_changed')
            rewards=[v['reward'] for v in trials if v['reward'] is not None]
            qualities=[v['quality'] for v in trials if type(v['quality']) in (int,float) and math.isfinite(v['quality'])]
            if rewards: repeat_reward.append(max(rewards)-min(rewards))
            if qualities: repeat_q.append(max(qualities)-min(qualities))
        records.append({'artifact_sha256':artifact,'label':label['validity'],'category':case['category'],'human_label_sha256':identity_hash({'annotations':case['annotations'],'adjudication':case.get('adjudication')}),'trials':trials})
    if len(identities)!=1: raise ValueError('exactly one reviewer/runtime identity is required')
    def mean(values, empty=1e9): return sum(values)/len(values) if values else empty
    metrics={
        'false_positive_rate':counts['false_positive_cases']/max(1,counts['INVALID']),
        'false_negative_rate':counts['false_negative_cases']/max(1,counts['VALID']),
        'unknown_rate':counts['unresolved_cases']/max(1,len(artifacts)),
        'quality_mae':mean(q_errors),'criterion_mae':mean([x for v in per_criterion.values() for x in v]),
        'each_criterion_mae':{k:mean(v) for k,v in per_criterion.items()},
        'within_one_point':mean(within,0),'reward_mae':mean(reward_errors),'acceptance_agreement':mean(acceptance,0),
        'repeat_quality_range':max(repeat_q,default=0),'repeat_reward_range':max(repeat_reward,default=0),
        'human_score_disagreement':mean(human_errors,0)}
    minimums={'VALID':'minimum_valid_cases','INVALID':'minimum_invalid_cases','quality_cases':'minimum_quality_cases','gold_accepted':'minimum_gold_accepted','gold_rejected':'minimum_gold_rejected'}
    for count,limit in minimums.items():
        if counts[count]<POLICY[limit]: failures.append('insufficient_'+count.lower())
    if repeated<POLICY['minimum_repeat_cases']: failures.append('insufficient_repeat_cases')
    if not set(POLICY['required_quality_bands'])<=bands: failures.append('quality_bands_missing')
    for category in REQUIRED_ATTACKS:
        if attacks[category]<POLICY['minimum_per_attack_category']: failures.append('insufficient_attack_'+category)
    for key in ('false_positive_rate','false_negative_rate','unknown_rate','quality_mae','criterion_mae','reward_mae','repeat_quality_range','repeat_reward_range','human_score_disagreement'):
        if metrics[key]>POLICY['max_'+key]: failures.append(key)
    if max(metrics['each_criterion_mae'].values())>POLICY['max_each_criterion_mae']: failures.append('per_criterion_accuracy')
    for key in ('within_one_point','acceptance_agreement'):
        if metrics[key]<POLICY['minimum_'+key]: failures.append(key)
    return {'schema':'farm.calibration.v2','status':'FAIL' if failures else 'PASS','scope':'validity_quality_and_repeatability',
            'policy_sha256':file_hash(HERE/'calibration-policy.json'),'rubric_sha256':manifest['rubric_sha256'],
            'dataset_id':manifest['dataset_id'],'reviewer_fingerprint':next(iter(identities)),
            'valid_cases':counts['VALID'],'invalid_cases':counts['INVALID'],'quality_cases':counts['quality_cases'],
            'repeated_cases':repeated,'attack_counts':dict(attacks),'counts':dict(counts),'metrics':metrics,
            'failures':sorted(set(failures)),'records':records,'labels_sha256':identity_hash(manifest)}


def validate_certificate(report, expected_fingerprint):
    if (report.get('schema')!='farm.calibration.v2' or report.get('scope')!='validity_quality_and_repeatability'
            or report.get('status')!='PASS' or report.get('failures')!=[]
            or report.get('policy_sha256')!=file_hash(HERE/'calibration-policy.json')
            or report.get('rubric_sha256')!=file_hash(HERE/'rubric.json')
            or report.get('reviewer_fingerprint')!=expected_fingerprint):
        raise ValueError('calibration certificate missing, stale or failed; v2 full-quality calibration required')
    for key in ('valid_cases','invalid_cases','quality_cases'):
        if type(report.get(key)) is not int or report[key]<POLICY['minimum_'+key]: raise ValueError('insufficient calibration samples')
    if type(report.get('repeated_cases')) is not int or report['repeated_cases']<POLICY['minimum_repeat_cases']: raise ValueError('repeat calibration missing')
    if any(report.get('attack_counts',{}).get(c,0)<POLICY['minimum_per_attack_category'] for c in REQUIRED_ATTACKS): raise ValueError('adversarial coverage missing')
    records=report.get('records',[])
    if (not isinstance(records,list) or len(records)!=report['valid_cases']+report['invalid_cases']
            or len({r.get('artifact_sha256') for r in records})!=len(records)
            or sum(r.get('label')=='VALID' for r in records)!=report['valid_cases']
            or any(not HEX.fullmatch(r.get('artifact_sha256','')) or not HEX.fullmatch(r.get('human_label_sha256','')) or not r.get('trials') for r in records)):
        raise ValueError('calibration provenance records missing or inconsistent')
    metrics=report.get('metrics',{})
    for key in ('false_positive_rate','false_negative_rate','unknown_rate','quality_mae','criterion_mae','reward_mae','repeat_quality_range','repeat_reward_range','human_score_disagreement'):
        value=metrics.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or value>POLICY['max_'+key]: raise ValueError('calibration metric failed: '+key)
    for key in ('within_one_point','acceptance_agreement'):
        value=metrics.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not POLICY['minimum_'+key]<=value<=1: raise ValueError('calibration agreement failed')
    errors=metrics.get('each_criterion_mae',{})
    if set(errors)!={c['criterion_id'] for c in ITEMS} or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=POLICY['max_each_criterion_mae'] for v in errors.values()): raise ValueError('criterion calibration missing')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=aggregate(read_json(args.manifest),args.manifest.resolve().parent)
    dump_json(args.output,report)
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__': raise SystemExit(main())
