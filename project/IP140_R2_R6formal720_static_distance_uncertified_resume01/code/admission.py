"""Adopt eight adjudications, seal the overlay and release the 712 still-unstarted intents."""
import ast,shutil
from common import *
OLD=ROOT/'IP140_R2_R6formal720_raw_a_c0_paired01'
def main():
    scope=read(HERE/'protocol/continuation.json');cases=read(HERE/'audit/offline/report.json');require(cases['passed'] and all(z['passed'] for z in cases['cases']) and cases['connections']==cases['new_physical_steps']==0,'offline gate')
    import_paths=read(HERE/'audit/import_check.json');compat=read(HERE/'audit/livepath_compatibility.json')
    require(import_paths['no_engine_module_loaded'] and compat['dynamic007']['actual_physics_steps']==4728 and compat['new_physics_steps']==0 and compat['empty004']['actual_physics_steps']==0,'native receipt replay')
    h=read(HERE/'audit/bytewise_inheritance.json');d=read(HERE/'preregistration/design.json');seal=read(HERE/'sealed_intents.json')
    require(sha(HERE/'sealed_intents.json')==sha(OLD/'sealed_intents.json')==h['original_seal_sha256'] and sha(HERE/'preregistration/design.json')==h['frozen_design_sha256']==seal['design_sha256'],'original seal')
    for k,ref in h['inherited'].items():
        if k in read(HERE/'audit/adapter_hashes.json'):continue
        require(sha(HERE/k)==ref['sha256'],'inherited data changed '+k)
    for k,r in read(HERE/'audit/adapter_hashes.json').items():require(sha(OLD/k)==r['original'] and sha(HERE/k)==r['continuation'],'adapter drift')
    require(all(sha(HERE/k)==want for k,want in seal['files'].items()) and len(seal['files'])==720,'unsealed intent')
    require(not any((HERE/'run/jobs/method'/k).exists() or (OLD/'run/jobs/method'/k).exists() for k in scope['allowed_once_in_original_order']),'an allowed trial already dispatched')
    tid=scope['supplemental_only'][0];dest=HERE/'trials'/tid;orig=read(HERE/'history'/tid/'method_result.json')
    replay=read(HERE/'audit/offline'/(tid+'.json'));require(replay['detail_class']=='static_distance_uncertified' and orig['shared_fault'] and replay['native']['actual_physics_steps']==orig['physical_steps']==orig['physical_returns']==0,'001 supplemental evidence')
    out=dict(orig);out.pop('traceback',None)
    out.update(shared_fault=False,failure_category='path_failure',detail_class='static_distance_uncertified',failure_detail=read(dest/'static_preflight/path_result.json')['first_failure'],engineering_gate_passed=False,grasp_success=False,trial_completed=False,forbidden_collision=False,physical_steps=0,physical_returns=0,new_native_calls=0,adjudication_only=True,
        original_method_result_sha256=sha(OLD/'trials'/tid/'method_result.json'),original_stop_sha256=sha(OLD/'run/METHOD_STOP.json'),offline_evidence_sha256=sha(HERE/'audit/offline'/(tid+'.json')))
    rh=put(dest/'method_result.json',out)
    put(HERE/'run/jobs/method'/tid/'return.json',dict(completed=True,id=tid,intent_sha256=sha(dest/'intent.json'),candidate_id=read(dest/'intent.json')['selected_candidate_id'],method_result_sha256=rh,worker_claim_sha256=sha(OLD/'run/jobs/method'/tid/'launch.json'),elapsed_sec=None,utc=now(),receipt_kind='offline_supplemental_adjudication',new_method_dispatch=False,new_physics_steps=0))
    put(HERE/'protocol/frozen_rule.json',dict(detail_class='static_distance_uncertified',engineering_gate_passed=False,grasp_success=False,physical_steps=0,root_distance_tolerance_m=1e-9,bracket_tolerance_m=1e-10,scope='Finite identity-bound, complete static native calls and failed legal entry certificate. Retain original reasons, brackets and rejected surfaces. Existing geometric, physical, scoring and success predicates stay in their sealed implementations.',priority='Missing returns, nonfinite data, corrupted or mismatched identities, unexpected API exceptions and independent integrity faults stop dispatch.',original_001_untouched=True,replay_sha256=sha(HERE/'audit/offline/report.json')))
    code={p.relative_to(HERE).as_posix():sha(p) for b in ['implementation','code'] for p in (HERE/b).glob('*.py')}
    for p in code:ast.parse((HERE/p).read_text(encoding='utf8'))
    extra=['protocol/continuation.json','protocol/frozen_rule.json','audit/import_check.json','audit/livepath_compatibility.json','audit/bytewise_inheritance.json','audit/adapter_hashes.json','audit/continuation.diff','history/run/METHOD_STOP.json']
    for k in scope['adjudicated_unchanged']+scope['supplemental_only']:
        extra+=['trials/'+k+'/method_result.json','run/jobs/method/'+k+'/return.json']
    extra+= [p.relative_to(HERE).as_posix() for p in (HERE/'audit/offline').glob('*.json')]
    code.update({k:sha(HERE/k) for k in extra})
    resource=read(HERE/'audit/resource_preflight.json')
    free=shutil.disk_usage(HERE).free;require(free>resource['space_reserve_bytes']+resource['estimated_bytes_legacy_case_mix'],'available space')
    put(HERE/'protocol/release.json',dict(version=HERE.name,parent_version=OLD.name,utc=now(),files=code,total=720,new_allowed=712,old_adjudicated=7,offline_supplements=1,prior_budget_reused=True,parent_intents_sha256=sha(OLD/'sealed_intents.json'),parent_stop_sha256=sha(OLD/'run/METHOD_STOP.json'),replay_passed=True,replay_cases=cases['test_count'],history_verified_files=h['prior_full_manifest_files'],disk_available_bytes=free,freeze_source_input_hashes=True,ablation=0))
    print('admitted; adopted 7, supplemental 1, authorized 712',flush=True)
if __name__=='__main__':main()
