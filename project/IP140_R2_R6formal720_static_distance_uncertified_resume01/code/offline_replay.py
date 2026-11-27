"""Offline gate. The real contact solver, native logger and robot are never invoked."""
from pathlib import Path
import copy,math,sys,time,traceback,hashlib
from common import *
OLD=ROOT/'IP140_R2_R6formal720_raw_a_c0_paired01'
sys.path[:0]=[str(HERE/'implementation'),str(R4/'implementation'),str(ROOT)]
from platform_imports import bind_frozen_libraries
bind_frozen_libraries()
from result_classification import post_static,classify_summary,DETAIL,validsummary,rowowner,numbers
from contact_semantics import validate_entry_certificate,registered_surface
def run():
    d=HERE/'audit/offline';d.mkdir(exist_ok=False);cases=[]
    def record(name,ok,**info):
        cases.append(dict(name=name,passed=bool(ok),**info))
        require(ok,'offline mismatch '+name)
    def integrity_negative(name,f):
        try:f()
        except (ValueError,TypeError,KeyError) as e:record(name,'shared_native_or_evidence_integrity' in str(e),verdict='shared_fault',error=str(e));return
        record(name,False,verdict='incorrectly_accepted')
    prior=read(HERE/'protocol/continuation.json')
    for tid,wanted in [('R6FORMAL-001',DETAIL),('R6FORMAL-007','static_admitted'),('R6FORMAL-010',DETAIL)]:
        base=HERE/'trials'/tid;proof=post_static(base,intent=read(base/'intent.json'));h=put(d/(tid+'.json'),proof)
        record(tid,proof['detail_class']==wanted,actual=proof['detail_class'],expected=wanted,proof_sha256=h,case_source=str((OLD/'trials'/tid).relative_to(ROOT)))
        if tid=='R6FORMAL-001':
            sr=read(base/'static_preflight/path_result.json');roots=read(base/'static_preflight/left_right_contact_entry_roots.json')
            repeat=[classify_summary(sr,roots,proof['certificate_sides'],proof['native']) for _ in range(2)]
            record('001_deterministic_same_frozen_inputs',repeat==[DETAIL,DETAIL],repeated=repeat,native_steps=0)
            for name,field,value in [('missing_return','native_returns',proof['native']['native_calls']-1),('api_exception','exceptions',1),('nonfinite','finite',False),('identity_mismatch','body_link_ownership',False),('physics_in_static','actual_physics_steps',1)]:
                bad=dict(proof['native']);bad[field]=value;integrity_negative(name,lambda:classify_summary(sr,roots,proof['certificate_sides'],bad))
        if tid=='R6FORMAL-007':
            r=read(base/'episode/result.json');q=read(base/'episode/final_480_steps.json')['observed_rows']
            good=len(q)==480 and [x['step'] for x in q]==list(range(4249,4729)) and all(x['lift_m']>=.10 and x['effective_left_target_contact'] and x['effective_right_target_contact'] and not x['aggregate_table_support'] and not x['slipped'] and not x['engineering_violations'] for x in q)
            record('007_historical_dynamics_adjudication',good and r['grasp_success'] and r['engineering_gate_passed'] and r['step_returns']==4728 and not r['unknown_steps'],steps_original=r['step_returns'],new_steps=0)
            root=read(base/'static_preflight/left_right_contact_entry_roots.json')['sides']['left']
            def validator(a):return validate_entry_certificate(a,robot=root['robot_body_id'],target=root['target_body_id'],pad=root['pad_link_index'])
            for name,distance,should_fail in [('exact_1nm',1e-9,False),('next_float_above_1nm',math.nextafter(1e-9,math.inf),True),('negative_next_float_below_minus_1nm',math.nextafter(-1e-9,-math.inf),True)]:
                v=copy.deepcopy(root);v['entry_audit']['signed_distance_m']=distance
                errors=validator(v)
                record(name,('entry_signed_distance_residual' in errors)==should_fail,distance_m=distance,expected_distance_rejection=should_fail,all_original_validator_reasons=errors)
    archive=PLATFORM/'surface_replay21.json';cases_in=read(archive)
    pos=neg=0
    for orig in cases_in['cases']:
        e=orig['evidence'];name=orig['name']
        if 'expected_allowed_type' not in e or 'point_m' not in e:continue
        from contact_entry import validate_contact_row
        import numpy as np
        side=name.split('_')[0];pad=13 if side=='left' else 15
        pos3=np.array(e['point_m'],float);direction=np.array(e['normal'],float);sep=3e-11
        pa=pos3.copy();pb=pa+sep*direction;nb=-direction
        if name.endswith('_witness_residual'):pb[2]+=2e-9
        row=[0,0,2,pad-1 if name.endswith('_wrong_pad_link') else pad,-1,pa.tolist(),pb.tolist(),nb.tolist(),sep,0.,0.,[0.,1.,0.],0.,[1.,0.,0.]]
        if name.endswith('_reversed_endpoints'):
            row[1:5]=[2,0,-1,pad];row[5:8]=[pb.tolist(),pa.tolist(),direction.tolist()]
        v=registered_surface(side,pos3,direction)
        verdict=validate_contact_row(row,robot=0,target=2,pad_index=pad,W_T_pad=np.eye(4),W_T_G=np.eye(4),side=side,require_face=True)
        want=e['expected_allowed_type'];ok=verdict['passed'];pos+=want is not None;neg+=want is None
        record('frozen_surface/'+name,ok==(want is not None) and (not ok or v['surface_type']==want) and verdict['violations']==e['actual_violations'],expected=want,accepted=ok,actual_surface=v['surface_type'],original_validator_violations=verdict['violations'])
    require(pos and neg,'no historical geometry controls')
    hist=next((R4/'trials'/'R4DEV-001').glob('static_preflight/path_result.json'));sr=read(hist);root=read(hist.parent/'left_right_contact_entry_roots.json')
    record('R4_original_no_bilateral_control',not sr['passed'] and sr['exception']=="ValueError('both_registered_entries_required_before_lift')" and not root['both_passed'],original_rejection=sr['exception'],historical_static_result_sha256=sha(hist))
    actualsum=read(HERE/'trials/R6FORMAL-001/static_api/summary.json')
    bad=copy.deepcopy(actualsum);bad['returns']['getClosestPoints']-=1;integrity_negative('real_summary_one_missing_return',lambda:validsummary(bad,0))
    bad=copy.deepcopy(actualsum);bad['exceptions']=[['getClosestPoints','injected']];integrity_negative('real_summary_raised_exception',lambda:validsummary(bad,0))
    from math import nan,inf
    for label,v in [('nan',nan),('negative_inf',-inf),('positive_inf',inf)]:integrity_negative('native_nested_'+label,lambda:numbers({'q':[0.,v]}))
    raw=read(OLD/'audit/stop_forensics/native_bracket_audit.json')['evidence_queries'][0]['native_row']
    domains={0:set(range(-1,17)),1:{-1},2:{-1}}
    for label,column,value in [('unknown_target',2,99),('wrong_pad',3,88),('wrong_target_link',4,7)]:
        fake=raw.copy();fake[column]=value;integrity_negative(label,lambda:rowowner(fake,domains,[0,2,.20],{},'getClosestPoints'))
    for k in prior['adjudicated_unchanged']:
        a=OLD/'trials'/k/'method_result.json';b=HERE/'trials'/k/'method_result.json'
        record('immutable_return/'+k,sha(a)==sha(b) and read(a)==read(b),original_result_sha256=sha(a),new_dynamics=0)
    put(d/'report.json',dict(passed=all(r['passed'] for r in cases),test_count=len(cases),cases=cases,historical_surface_positive=pos,historical_surface_negative=neg,archive_source=sha(archive),frozen_contact_semantics_sha256=sha(PLATFORM/'contact_semantics.py'),frozen_root_solver_sha256=sha(PLATFORM/'contact_entry.py'),connections=0,new_physical_steps=0))
    print('offline',len(cases),'passed',pos,neg,flush=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        put(HERE/'audit/offline/failed.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()))
        raise
