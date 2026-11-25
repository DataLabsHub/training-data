"""Native evidence + synthetic review protocol smoke. NOT a quality calibration."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--submission',type=Path,required=True);p.add_argument('--blender',required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
prompt=a.output/'prompt.txt';prompt.write_text('SYNTHETIC PROTOCOL TEST ONLY',encoding='utf-8')
script=ROOT/'verifier/tests/farm_protocol_reviewer.py'
config=a.output/'test-reviewer.json'
config.write_text(json.dumps({'backend':'local','model':'synthetic-test-double','model_version':'v2',
 'prompt_file':str(prompt.resolve()),'command':[sys.executable,str(script)],
 'executable_artifacts':{str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (sys.executable,script)}}),encoding='utf-8')
with (a.output/'run.log').open('wb') as log:
 r=subprocess.run([sys.executable,str(ROOT/'verifier/reward.py'),'--submission',str(a.submission),'--reports',str(a.output/'report'),'--task','farm-reconstruction-v0','--backend','local','--blender',a.blender,'--farm-reviewer',str(config),'--allow-local-reviewer','--resolution','320','240','--samples','2'],stdout=log,stderr=subprocess.STDOUT,timeout=180)
result=json.loads((a.output/'report/result.json').read_text(encoding="utf-8"))
assert r.returncode==2,result
assert result['status']=='NEEDS_REVIEW' and result['reward'] is None,result
assert result['task_result']['candidate_evaluation']['reward']==1,result
assert len(result['criterion_scores'])==31
assert (a.output/'report/farm/tool_00_00/geometry.json').exists()
assert (a.output/'report/farm/tool_00_01/render/barn_detail.png').exists()
assert (a.output/'report/farm/report.html').exists()
summary={'status':'PASS','scope':'SYNTHETIC_REVIEW_PROTOCOL_ONLY_NOT_MODEL_QUALITY','native_evidence':'PASS',
 'tool_rounds':'geometry query + gray closeup PASS','fixed_criteria':31,'reward_aggregation':'PASS',
 'original_farm':'NOT_RUN: artifact unavailable','production_visual_judge':'NOT_CONFIGURED'}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2), encoding="utf-8");print(json.dumps(summary,indent=2))
