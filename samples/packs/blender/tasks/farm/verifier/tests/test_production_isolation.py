"""Controller adversarial tests. Docker interactions here are explicit test doubles."""
import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import support
from worker_runtime import WorkerRuntime

class IsolationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  (self.root/'input').mkdir();(self.root/'report').mkdir()
  self.source=self.root/'input/result.blend';self.source.write_bytes(b'BLENDER-fixture')
  self.verifier=Path(__file__).resolve().parents[1]
  self.args=S(backend='docker',image='sha256:'+'a'*64,blender='blender')
  self.runner=WorkerRuntime(self.args,self.source,self.root/'report',self.verifier,support.Deadline(20))
 def command(self):return self.runner.command(self.verifier/'execution_check.py',['--submission','/input/result.blend','--output','/output/general'],'general')
 def test_no_writable_alias_for_submission_or_previous_evidence(self):
  cmd=self.command();mounts=[cmd[i+1] for i,v in enumerate(cmd) if v=='--mount']
  writable=[v for v in mounts if not v.endswith(',readonly')]
  self.assertEqual(len(writable),1);self.assertIn('dst=/scratch',writable[0])
  self.assertIn('/.worker-staging/general',writable[0]);self.assertIn('/scratch/result',cmd)
  self.assertFalse(any('dst=/output' in m for m in writable))
 def test_worker_cannot_choose_output_escape(self):
  for target in ('/outside','/output/../escape','/output'):
   with self.assertRaises(ValueError):self.runner.command(self.verifier/'execution_check.py',['--output',target],'escape')
 def test_hardlinked_bundle_file_rejected(self):
  os.link(self.source,self.root/'input/alias')
  with self.assertRaises(ValueError):support.freeze_submission(self.source,self.root/'frozen')
 def test_directory_swap_attack_cannot_copy_outside_root(self):
  (self.root/'input/textures').mkdir();(self.root/'outside').mkdir();(self.root/'outside/secret').write_bytes(b'OUTSIDE')
  real=os.open;swapped=False
  def racing(name,flags,*args,**kwargs):
   nonlocal swapped
   if name=='textures' and kwargs.get('dir_fd') is not None and not swapped:
    swapped=True;(self.root/'input/textures').rmdir();(self.root/'input/textures').symlink_to(self.root/'outside',target_is_directory=True)
   return real(name,flags,*args,**kwargs)
  with patch('support.os.open',side_effect=racing):
   with self.assertRaises(ValueError):support.freeze_submission(self.source,self.root/'frozen')
  self.assertFalse((self.root/'frozen/textures/secret').exists())
 def test_depth_limit(self):
  directory=self.root/'input'
  for i in range(34):directory=directory/'d';directory.mkdir()
  with self.assertRaises(support.BudgetExceeded):support.freeze_submission(self.source,self.root/'frozen')
 def test_json_exponent_overflow_rejected(self):
  with self.assertRaises(ValueError):support.strict_json('{"reward":1e309}')
 def fake_docker(self,*args,**kwargs):
  return subprocess.CompletedProcess(args[0],0,stdout=json.dumps({'Status':'exited','ExitCode':0,'OOMKilled':False,'Error':''}))
 def test_output_symlink_cannot_replace_prior_evidence(self):
  cmd=self.command();stage,_=self.runner.stages['general'];(stage/'result').mkdir();(stage/'result/escape').symlink_to(self.source)
  (self.root/'report/general.cid').write_text('b'*64,encoding='utf-8')
  with patch('worker_runtime.run_native',return_value=0),patch('worker_runtime.subprocess.run',side_effect=self.fake_docker):
   code=self.runner.run(cmd,'general')
  self.assertEqual(code,-3);self.assertFalse((self.root/'report/general').exists());self.assertEqual(self.source.read_bytes(),b'BLENDER-fixture')
 def test_docker_cleanup_even_on_timeout(self):
  cmd=self.command();(self.root/'report/general.cid').write_text('c'*64,encoding='utf-8')
  with patch('worker_runtime.run_native',side_effect=subprocess.TimeoutExpired('test',1)),patch('worker_runtime.subprocess.run',side_effect=self.fake_docker) as docker:
   self.assertEqual(self.runner.run(cmd,'general'),-1)
  self.assertTrue(any(call.args[0][:3]==['docker','rm','-f'] for call in docker.call_args_list))
  self.assertTrue(self.runner.events[-1]['container_removed'])
 def test_oom_report_has_no_promoted_result(self):
  cmd=self.command();stage,_=self.runner.stages['general'];(stage/'result').mkdir();(stage/'result/general_checks.json').write_text('{}',encoding='utf-8')
  (self.root/'report/general.cid').write_text('d'*64,encoding='utf-8')
  def docker(command,**kwargs):return subprocess.CompletedProcess(command,0,stdout='{"OOMKilled":true,"Status":"exited","ExitCode":137}')
  with patch('worker_runtime.run_native',return_value=137),patch('worker_runtime.subprocess.run',side_effect=docker):self.assertEqual(self.runner.run(cmd,'general'),-4)
  self.assertFalse((self.root/'report/general').exists())
 def test_native_descendant_is_killed_after_parent_exit(self):
  if os.name!='posix':self.skipTest('POSIX controller behavior')
  marker=self.root/'unexpected'
  child='import time;from pathlib import Path;time.sleep(.3);Path('+repr(str(marker))+').write_text("leaked")'
  script='import subprocess,sys;subprocess.Popen([sys.executable,"-c",'+repr(child)+'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)'
  self.assertEqual(support.run_native([sys.executable,'-c',script],self.root/'log',2,self.root),0)
  time.sleep(.4);self.assertFalse(marker.exists())
 def test_fast_worker_final_write_cannot_bypass_report_budget(self):
  script='import time;from pathlib import Path;time.sleep(.1);Path('+repr(str(self.root/'report/payload'))+').write_bytes(b"x"*4096)'
  with patch.dict(support.POLICY,{'max_report_bytes':1024}):
   with self.assertRaisesRegex(support.BudgetExceeded,'report_output_budget'):
    support.run_native([sys.executable,'-c',script],self.root/'log',2,self.root/'report')
 def test_missing_container_identity_cannot_promote_results(self):
  cmd=self.command();stage,_=self.runner.stages['general'];(stage/'result').mkdir()
  with patch('worker_runtime.run_native',return_value=0):self.assertEqual(self.runner.run(cmd,'general'),-2)
  self.assertFalse((self.root/'report/general').exists())
  self.assertEqual(self.runner.events[-1]['reason'],'container_identity_missing')
 def test_container_exit_mismatch_cannot_promote_results(self):
  cmd=self.command();stage,_=self.runner.stages['general'];(stage/'result').mkdir()
  (self.root/'report/general.cid').write_text('e'*64,encoding='utf-8')
  def docker(command,**kwargs):return subprocess.CompletedProcess(command,0,stdout='{"OOMKilled":false,"Status":"exited","ExitCode":3}')
  with patch('worker_runtime.run_native',return_value=0),patch('worker_runtime.subprocess.run',side_effect=docker):self.assertEqual(self.runner.run(cmd,'general'),-2)
  self.assertFalse((self.root/'report/general').exists())
 def test_non_blend_payload_is_deterministic_failure(self):
  self.source.write_bytes(b'not a blend file')
  self.assertEqual(support.file_preflight(self.source),'not_a_blend_file')
 def test_compression_bomb_bounded_before_native_parser(self):
  import gzip
  self.source.write_bytes(gzip.compress(b'BLENDER-v522'+b'x'*4096))
  with patch.dict(support.POLICY,{'max_submission_bytes':1024}):
   with self.assertRaises(support.BudgetExceeded):support.native_file_preflight(self.source)
 def test_compressed_non_blend_is_invalid(self):
  import gzip
  self.source.write_bytes(gzip.compress(b'not-a-blend-payload'))
  with self.assertRaises(support.InvalidSubmission):support.native_file_preflight(self.source)
 def test_valid_small_compressed_header_accepted_by_budget_screen(self):
  import gzip
  self.source.write_bytes(gzip.compress(b'BLENDER-v522'+b'x'*4096))
  support.native_file_preflight(self.source)
 def test_malformed_native_checks_cannot_bypass_gate(self):
  from general_contract import validate_checks,check,CHECKS
  rows=[check(k,'PASS') for k in CHECKS]
  validate_checks(rows)
  for mutation in ('N_A','missing','duplicate','path_escape','not_a_list'):
   import copy
   data=copy.deepcopy(rows)
   if mutation=='N_A':data[0]['status']='N_A'
   if mutation=='missing':data.pop()
   if mutation=='duplicate':data[0]=data[1]
   if mutation=='path_escape':data[0]['evidence']=['../reward.txt']
   if mutation=='not_a_list':data={}
   with self.assertRaises(ValueError):validate_checks(data)
 def test_sparse_reward_cannot_forge_fractional_or_nan_pass(self):
  from general_contract import validate_sparse_result
  for value in (0,.9,float('nan'),True):
   with self.assertRaises(ValueError):validate_sparse_result({'status':'PASS','training_mask':1,'reward':value})
