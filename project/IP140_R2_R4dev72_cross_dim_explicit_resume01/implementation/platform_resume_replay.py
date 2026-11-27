"""No engine/worker dispatch. Synthetic fixtures and frozen-file identities only."""
import ast
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock
from platform_imports import ROOT,VERSION
import platform_trial as worker
from scene_world import confirm_same_settle

OLD=ROOT/'IP140_R2_R4dev72_raw_a_c0_paired01'
AUDIT=VERSION/'audit'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,obj):
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(obj,allow_nan=False,sort_keys=True),encoding='utf8')
def expect_error(cases,name,fn):
    try:fn()
    except (ValueError,RuntimeError,KeyError) as e:
        cases.append(dict(name=name,passed=True,exception=str(e)))
    else:raise AssertionError(name+' accepted')
def ok(cases,name,condition,**fields):
    if not condition:raise AssertionError(name)
    cases.append(dict(name=name,passed=True,**fields))

def main():
    if Path.cwd().resolve()!=ROOT:raise ValueError('explicit project root required')
    passed=[]
    independent=['platform_plan','platform_select','platform_static','platform_episode','platform_evaluator',
                 'platform_support','platform_imports','platform_trace','scene_camera','scene_asset','scene_world']
    same={}
    for mod in independent:
        filename=mod+'.py'
        orig=sha(OLD/'implementation'/filename);current=sha(VERSION/'implementation'/filename)
        same[filename]=dict(old=orig,current=current)
        ok(passed,'frozen source '+mod,orig==current)
    old_stats=read(OLD/'audit/platform_zero_interface_receipt.json')
    static=read(OLD/'audit/platform_zero_challenge/static_preflight/path_result.json')
    ok(passed,'archive ordinary postlock path rejection unchanged',
       old_stats['physical_steps']==0 if 'physical_steps' in old_stats else old_stats['native_physics_steps']==0)
    ok(passed,'ordinary missing bilateral entries is an adjudicated path failure',
       worker.candidate_static_failure(static)=='path_failure')
    ok(passed,'actual native integrity fault takes precedence over collision',
       worker.dynamic_category(dict(violations=['nonpad_robot_object_contact','unknown_contact_identity'])) is None)
    ok(passed,'known prohibited collision is an adjudicated failure',
       worker.dynamic_category(dict(violations=['nonpad_robot_object_contact']))=='forbidden_collision')
    ok(passed,'unrecognized native return stays indeterminate',
       worker.dynamic_category(dict(violations=['unregistered_opaque_fault'])) is None)
    ok(passed,'nonempty pool with all IK failures is no_executable_candidate',
       worker.no_selection_outcome([dict(eligible=False,reason='static IK failed to converge')])['failure_category']=='no_executable_candidate')
    sources={}
    for p in sorted((OLD/'scenes').glob('*/settled_state.json')):
        a=read(p);sources[p.parent.name]=sha(p)
        ok(passed,'archived scene-state comparator '+p.parent.name,confirm_same_settle(a,copy.deepcopy(a)) is True)
    assert len(sources)==12
    changed=copy.deepcopy(a);changed['object_pose'][0][0]+=1e-9
    expect_error(passed,'paired initial object pose mismatch',lambda:confirm_same_settle(a,changed))
    with tempfile.TemporaryDirectory(prefix='platform_synthetic_only_',dir=AUDIT) as td:
        base=Path(td)
        intentional='offline_fixture';u='synthetic-empty'
        pool=base/'units'/u/'pool.npz';pool.parent.mkdir(parents=True);pool.write_bytes(b'fixture:empty-pool')
        stable=base/'scenes'/'synthetic'/'settled_state.json';save(stable,dict(synthetic=True))
        order=base/'units'/u/'ranking_Raw.json'
        save(order,dict(unit_id=u,method='Raw',shared_pool_sha256=sha(pool),complete_order=[]))
        sp=base/'units'/u/'selection.json'
        save(sp,dict(unit_id=u,used_ground_truth=False,never_fallback_after_postlock_safety=True,
                     selected={m:None for m in ('Raw','A','C0')},eligibility=[]))
        path=base/'trials'/intentional/'intent.json'
        def rel(p):return p.relative_to(base).as_posix()
        orig=dict(trial_id=intentional,unit_id=u,method='Raw',scene_key='synthetic',
                  pool_path=rel(pool),pool_sha256=sha(pool),ranking_path=rel(order),ranking_sha256=sha(order),
                  stable_state_path=rel(stable),stable_state_sha256=sha(stable),selection_sha256=sha(sp),
                  selected_candidate_id=None,selected_pool_index=None,no_postlock_fallback=True,ranking_uses_ground_truth=False,attempt_limit=1)
        def seal(i):
            save(path,i)
            files={rel(path):sha(path)}
            for j in range(71):
                pp=base/'trials'/str(j)/'intent.json';save(pp,{'synthetic_ordinal':j});files[rel(pp)]=sha(pp)
            save(base/'sealed_intents.json',dict(version=base.name,method_budget=72,files=files))
        seal(orig)
        snapshots=[]
        with mock.patch.object(worker,'VERSION',base),mock.patch.object(worker,'connection',side_effect=AssertionError('unexpected connection')) as forbidden:
            bound,selection,row=worker.frozen_intent(path)
            ok(passed,'complete synthetic intent is bound before adjudication',row is None and bound['trial_id']==intentional)
            # Run the empty-result branch against disposable fixtures only. No real ledger, native API, or official trial.
            assert worker.worker(path)==0
            res=read(path.parent/'method_result.json')
            ok(passed,'empty-domain method exits before any connection or fallback',
               forbidden.call_count==0 and res['failure_category']=='no_eligible_candidate'
               and res['physical_steps']==res['physical_returns']==0 and not res['grasp_success']
               and res['shared_fault'] is False and res['trial_completed'] is False
               and res['engineering_gate_passed'] is False and res['negative_control_passed'] is None
               and res['no_postlock_fallback'] is True and res['selected_pool_index'] is None,
               outcome={k:res[k] for k in ('failure_category','physical_steps','physical_returns','trial_completed','engineering_gate_passed','grasp_success','negative_control_passed')})
            tamper=dict(orig,ranking_sha256='0'*64);seal(tamper)
            expect_error(passed,'ranking bytes changed',lambda:worker.frozen_intent(path))
            seal(dict(orig,stable_state_sha256='0'*64))
            expect_error(passed,'stable state hash changed',lambda:worker.frozen_intent(path))
            seal(dict(orig,pool_sha256='0'*64));expect_error(passed,'pool hash changed',lambda:worker.frozen_intent(path))
            seat=dict(orig,selected_pool_index=0);seal(seat)
            expect_error(passed,'postlock fallback disallowed',lambda:worker.frozen_intent(path))
            seal(orig);save(path,dict(orig,attempt_limit=2))
            expect_error(passed,'post-seal intent mutation',lambda:worker.frozen_intent(path))
    expr=(VERSION/'implementation/platform_trial.py').read_text(encoding='utf-8-sig')
    ok(passed,'unknown or missing step return stops batch with null scientific outcome',
       "returned['unknown_steps']" in expr and "returned['step_returns']!=trace.returns['stepSimulation']" in expr and 'grasp_success=None' in expr)
    ok(passed,'native engine was never imported','pybullet' not in sys.modules)
    out=dict(version=VERSION.name,official_method_intents_created=0,official_method_trials_counted=0,
             native_connections=0,native_physics_calls=0,archived_engineering_experiments_rerun=0,
             result='pass',test_count=len(passed),cases=passed,
             frozen_execution_and_evaluation_files=same,registered_prior_stabilized_snapshots=sources,
             actual_worker_sha256=sha(VERSION/'implementation/platform_trial.py'),previous_worker_sha256=sha(OLD/'implementation/platform_trial.py'),
             synthetic_case_source_sha256=sha(Path(__file__)),counter_source='parent run_trials.py dispatch counts a method before subprocess return, including an empty pool',
             scope='synthetic empty-intent return, identity and decision-dispatch regression; no physical grasp or method outcome')
    p=AUDIT/'platform_resume_interface_receipt.json'
    with p.open('x',encoding='utf8') as fp:json.dump(out,fp,ensure_ascii=False,indent=2)
    print(str(p),len(passed))
if __name__=='__main__':main()

