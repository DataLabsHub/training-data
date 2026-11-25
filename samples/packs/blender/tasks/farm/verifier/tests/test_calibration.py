"""SYNTHETIC accounting fixtures, not actual human/model calibration evidence."""
import copy
import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calibrate import aggregate, REQUIRED_ATTACKS, ITEMS, validate_certificate, fingerprint
from support import dump_json, read_json, identity_hash
from constraint_check import file_hash
HERE=Path(__file__).resolve().parents[1]

class CalibrationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.identity={'backend':'docker','image':'test-digest','config_sha256':'test-config',
    'evidence_runtime':{'backend':'docker','image':'test-runtime'},'rubric_sha256':file_hash(HERE/'rubric.json')}
  self.manifest={'schema':'farm.calibration_manifest.v2','purpose':'held_out_acceptance','dataset_id':'SYNTHETIC-UNIT-TEST',
    'labels_complete':True,'rubric_sha256':self.identity['rubric_sha256'],'cases':[]}
  categories=sorted(REQUIRED_ATTACKS)
  for index in range(40):
   valid=index>=20;label='VALID' if valid else 'INVALID';artifact=f'{index:064x}'
   scores={c['criterion_id']:([1,3,4][index%3] if not c['criterion_id'].startswith('F9.') else 4) for c in ITEMS}
   ann={'validity':label}
   if valid:ann['scores']=scores
   case={'artifact_sha256':artifact,'category':categories[index%4] if not valid else 'ordinary',
     'annotations':[dict(ann,annotator_id='SYNTHETIC-RATER-A'),dict(ann,annotator_id='SYNTHETIC-RATER-B')],'results':[]}
   q=sum(c['weight']*scores[c['criterion_id']]/4 for c in ITEMS) if valid else None
   accepted=bool(valid and q>=85 and all(scores[c['criterion_id']]>=c['pass_score'] for c in ITEMS if c['required']))
   for repeat in range(3 if 20<=index<25 else 1):
    path=self.root/f'{index}-{repeat}.json'
    report={'schema':'rl_env.farm_result.v2','artifact_hash':artifact,'submission_validity':{'status':label},
      'evaluation_mode':'calibration','evaluation_id':f'{index*10+repeat:032x}',
      'candidate_evaluation':{'status':'PASS' if accepted else 'FAIL','accepted':accepted,'training_mask':1,'reward':round(.7*q/100+.3*accepted,6) if valid else 0},
      'reviewer_identity':self.identity,'reviewer_identity_hash':identity_hash(self.identity), 'quality_score':q,
      'criterion_scores':[{'criterion_id':k,'status':'SCORED','score':v} for k,v in scores.items()] if valid else []}
    dump_json(path,report);case['results'].append({'path':path.name,'sha256':file_hash(path)})
   self.manifest['cases'].append(case)
 def edit(self,index,fn,repeat=0):
  entry=self.manifest['cases'][index]['results'][repeat];path=self.root/entry['path'];report=read_json(path)
  fn(report);dump_json(path,report);entry['sha256']=file_hash(path)
 def aggregate(self):return aggregate(self.manifest,self.root)
 def test_full_threshold_accounting(self):
  result=self.aggregate();self.assertEqual(result['status'],'PASS',result['failures'])
  validate_certificate(result,fingerprint(self.identity))
 def test_duplicate_artifacts_rejected(self):
  self.manifest['cases'][1]=self.manifest['cases'][0]
  with self.assertRaises(ValueError):self.aggregate()
 def test_incomplete_labels_rejected(self):
  self.manifest['labels_complete']=False
  with self.assertRaises(ValueError):self.aggregate()
 def test_arbitrary_validity_rejected(self):
  self.edit(0,lambda r:r['submission_validity'].update(status='IGNORED'))
  with self.assertRaises(ValueError):self.aggregate()
 def test_trial_hash_tampering_rejected(self):
  (self.root/'0-0.json').write_text('{}',encoding='utf-8')
  with self.assertRaises(ValueError):self.aggregate()
 def test_replayed_run_id_rejected(self):
  self.edit(1,lambda r:r.update(evaluation_id='0'*32))
  with self.assertRaises(ValueError):self.aggregate()
 def test_insufficient_cases_fail(self):
  self.manifest['cases'].pop();self.assertEqual(self.aggregate()['status'],'FAIL')
 def test_false_acceptance_cannot_hide_in_repeats(self):
  def accept_invalid(r):
   r['submission_validity']['status']='VALID';r['quality_score']=100
   r['candidate_evaluation'].update(status='PASS',accepted=True,reward=1)
   r['criterion_scores']=[{'criterion_id':c['criterion_id'],'status':'SCORED','score':4} for c in ITEMS]
  for index in (0,1):self.edit(index,accept_invalid)
  r=self.aggregate();self.assertEqual((r['status'],r['metrics']['false_positive_rate']),('FAIL',.1))
 def test_mixed_identities_rejected(self):
  def mutate(r):
   r['reviewer_identity']['image']='other';r['reviewer_identity_hash']=identity_hash(r['reviewer_identity'])
  self.edit(0,mutate)
  with self.assertRaises(ValueError):self.aggregate()
 def test_local_evidence_rejected(self):
  self.edit(0,lambda r:r['reviewer_identity']['evidence_runtime'].update(backend='local'))
  with self.assertRaises(ValueError):self.aggregate()
 def test_missing_quality_coverage_fails(self):
  for i in range(20,40):
   scores={c['criterion_id']:4 for c in ITEMS}
   for a in self.manifest['cases'][i]['annotations']:a['scores']=scores
  r=self.aggregate();self.assertEqual(r['status'],'FAIL');self.assertIn('quality_bands_missing',r['failures'])
 def test_human_disagreement_requires_adjudicator(self):
  self.manifest['cases'][20]['annotations'][1]['scores']=dict(self.manifest['cases'][20]['annotations'][1]['scores'],**{ITEMS[0]['criterion_id']:0})
  with self.assertRaises(ValueError):self.aggregate()
 def test_missing_repeatability_fails(self):
  for row in self.manifest['cases']:row['results']=row['results'][:1]
  self.assertIn('insufficient_repeat_cases',self.aggregate()['failures'])
 def test_certificate_cannot_upgrade_v1(self):
  r=self.aggregate();r['schema']='farm.calibration.v1'
  with self.assertRaises(ValueError):validate_certificate(r,fingerprint(self.identity))
 def test_certificate_rejects_nan_or_incomplete_provenance(self):
  r=self.aggregate();r['metrics']['quality_mae']=float('nan')
  with self.assertRaises(ValueError):validate_certificate(r,fingerprint(self.identity))
  r=self.aggregate();r['records']=[]
  with self.assertRaises(ValueError):validate_certificate(r,fingerprint(self.identity))
 def test_invalid_positive_reward_rejected(self):
  self.edit(0,lambda r:r['candidate_evaluation'].update(reward=.9))
  with self.assertRaises(ValueError):self.aggregate()
