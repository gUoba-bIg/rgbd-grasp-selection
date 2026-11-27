"""Zero-model/zero-step cross-environment witness. Only redirect the frozen worker's output base."""
import hashlib, json, os, sys
from pathlib import Path
B=Path(__file__).resolve().parents[1];V=B.parent;T=V/'cold';P=T/'p'
role='rank';arm,tid=sys.argv[1:]
R=P/'IP140_R2_R6formal720_static_distance_uncertified_resume01'
F=P/'IP140_R2_R4dev72_cross_dim_explicit_resume01'
def read(p):return json.loads(p.read_bytes())
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
    with p.open('x',encoding='utf8')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
case=next(c for c in read(B/'metadata/cold_preregistration.json')['cases'] if c['trial_id']==tid)
uid=case['unit'];base=V/'gates'/arm;d=base/'units'/uid
assert Path(sys.prefix).resolve()==T/'env/rank' and Path(sys.base_prefix).resolve()==T/'bin' and d.is_dir()
assert Path.cwd().resolve()==R/'workspaces'/'rank_gate04'
sys.path[:0]=list(map(str,[B/'adapters',R/'code',R/'implementation',F/'implementation',P]))
dll=[os.add_dll_directory(str(p))for p in [P/'vendor/ProgramFiles/Git/mingw64/bin',T/'bin',T/'bin/DLLs'] if p.is_dir()]
os.environ['PROGRAMFILES']=str(P/'vendor/ProgramFiles')
from font_scope import install
cached=list((V/'gates/A/units'/uid/'mpl').glob('fontlist-v*.json')) if arm=='B' else []
if arm=='B':assert len(cached)==1
install(T,'rank',d/'mpl',d/'font_source.json','subset' if arm=='B'else'full',cached[0] if cached else None)
import task_worker, common
assert common.workspace_ok() and task_worker.HERE==R
orig_bind=task_worker.bind_engine
guard={'native_step_attempts':0,'bindings':0}
def bind_no_step():
    orig_bind()
    p=sys.modules['pybullet']
    def deny(*a,**k):
        guard['native_step_attempts']+=1;raise RuntimeError('case ranking probe may not run any physics')
    p.stepSimulation=deny;guard['bindings']+=1
task_worker.bind_engine=bind_no_step
task_worker.HERE=base
task_worker.rank_and_E(uid)
assert guard=={'native_step_attempts':0,'bindings':1} and 'torch'not in sys.modules
q=read(d/'orders.json');s=read(d/'selection.json')
order={k:v for k,v in q.items() if k not in ('scoring_sec','pool_build_sec')}
core={k:v for k,v in s.items() if k not in ('eligibility','native_eligibility_api_sha256')}
e=[{k:v for k,v in row.items() if k!='planning_seconds'} for row in s['eligibility']]
expected=read(B/'metadata/smoke_expected.json')[tid]
# The previous live driver compares these same scientific fields and plans.
api=read(d/'eligibility_api/summary.json')
def enc(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def canonical(obj):return hashlib.sha256(enc(obj)).hexdigest()
checks={
'registered_unit_leaf':d.name==core['unit_id']==uid,
'pool_bytes':h(d/'pool.npz')==case['inputs']['pool_sha256'],
'candidate_domain_ledger_bytes':h(d/'candidate_domain_ledger.json')==case['inputs']['exclusion_sha256']==q['candidate_domain_ledger_sha256'],
'complete_orders_scores_risks_and_components':enc(order)==enc(expected['orders']),
'full_presel_eligibility_and_plans':enc(e)==enc(expected['eligibility']),
'unit_selections_and_platform_input_hash':enc(core)==enc(expected['selection_core']),
'independent_method_rankings':all(read(d/('ranking_'+m+'.json'))['complete_order']==order['orders_by_pool_index'][m] for m in ('Raw','A','C0')),
'E_count':sum(x['eligible'] for x in e)==case['old_E_count'],
'all_API_returned_without_steps': api['calls']==api['returns'] and not api['exceptions'] and api['actual_native_step_calls']==0 and api['calls']['setCollisionFilterPair']==14}
put(d/'probe.json',dict(case=tid,unit=uid,arm=arm,checks=checks,passed=all(checks.values()),observed_E=sum(x['eligible'] for x in e),pool_sha256=h(d/'pool.npz'),ledger_sha256=h(d/'candidate_domain_ledger.json'),selection_core_sha256=canonical(core),full_eligible_sha256=canonical(e),all_score_fields_sha256=canonical(order),pool_count=order['pool_count'],methods=core['selected'],policy_src=str(sys.modules['policy_rank'].__file__),policy_sha256=h(Path(sys.modules['policy_rank'].__file__)),worker_sha256=h(Path(task_worker.__file__)),models=0,physical_steps=0,guard=guard))
if not all(checks.values()):raise RuntimeError(str([k for k,passed in checks.items() if not passed]))
