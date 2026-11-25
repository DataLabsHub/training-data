"""Real Blender integration regression; creates a new output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--blender',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
with (a.output/'fixture.log').open('wb') as log:
    subprocess.run([a.blender,'--background','--factory-startup','--disable-autoexec','--python-exit-code','3','--python',str(ROOT/'verifier/tests/general_fixtures.py'),'--',str(a.output/'inputs')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
(a.output/'inputs/corrupt.blend').write_bytes(b'not a blend file')
cases=[('baseline','general','PASS',None),('packed','general','PASS',None),
       ('missing_resource','general','FAIL',0),('unused_missing_resource','general','PASS',None),
       ('diagnostic','general','PASS',None),('empty','general','FAIL',0),
       ('corrupt','general','FAIL',0),('baseline','fence-grounding-v0','FAIL',0),
       ('solved','fence-grounding-v0','PASS',1)]
summary=[]
for name,task,status,reward in cases:
    source=a.output/'inputs'/(name+'.blend'); before=hashlib.sha256(source.read_bytes()).hexdigest()
    folder=a.output/(name+'-'+task)
    with (a.output/(folder.name+'.log')).open('wb') as log:
        run=subprocess.run([sys.executable,str(ROOT/'verifier/reward.py'),'--submission',str(source),'--reports',str(folder),'--task',task,'--backend','local','--blender',a.blender,'--resolution','160','120','--samples','2'],stdout=log,stderr=subprocess.STDOUT,timeout=180)
    r=json.loads((folder/'result.json').read_text(encoding="utf-8"))
    assert r['status']==status,(name,task,r)
    assert r['reward']==reward,(name,task,r)
    assert r['training_mask']==(0 if reward is None else 1)
    assert (folder/'reward.txt').exists()==(reward is not None)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    if name=='diagnostic':
        d=next(c for c in r['checks'] if c['check_id']=='mesh_diagnostics')
        assert d['status']=='FAIL',d
    summary.append({'fixture':name,'task':task,'status':status,'reward':reward,'exit_code':run.returncode,'source_unchanged':True})
(a.output/'summary.json').write_text(json.dumps({'status':'PASS','cases':summary},indent=2), encoding="utf-8")
print(json.dumps(summary,indent=2))
