"""SYNTHETIC TEST DOUBLE. Never configure this as a training/production judge."""
import json,sys
p=json.load(sys.stdin)
binding={k:p[k] for k in ('artifact_hash','reference_hash','rubric_hash','reviewer_identity_hash')}
refs=[e['id'] for e in p['evidence']]
if not any(e.startswith('tool_') for e in refs):
 binding['tool_requests']=[{'action':'inspect.geometry','targets':['Barn']},
   {'action':'render.views','args':{'mode':'auto','groups':[{'name':'barn_detail','objects':['Barn']}],'gray_override':True}}]
else:
 binding['submission_validity']={'status':'VALID','reason':'Synthetic fixture only, not visual judgement','evidence':refs}
 binding['criterion_scores']=[{'criterion_id':c['criterion_id'],'status':'SCORED','score':4,
  'reason':'SYNTHETIC PROTOCOL TEST ONLY; does not evaluate the rendered farm quality.','evidence':refs} for c in p['rubric']['atomic_criteria']['items']]
print(json.dumps(binding))
