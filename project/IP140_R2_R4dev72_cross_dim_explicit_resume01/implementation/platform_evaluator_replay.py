"""Compare the bound R4 evaluator with the original returned historical engineering telemetry."""
import gzip, json
from platform_imports import ROOT,VERSION,bind_frozen_libraries
bind_frozen_libraries()
from scene_world import read,write,registered_scene
from platform_select import digest
from platform_evaluator import ObjectEvaluator
from platform_support import bind_asset
from platform_trial import candidate_static_failure
def main():
    origin=ROOT/'IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01/episodes/14'
    seen=0;diffkeys={}
    with gzip.open(origin/'telemetry.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['event']=='initial_configuration':
                ec=r['evaluator_context']
                asset=registered_scene('scene-021-p2')['asset']
                bind_asset(ROOT/asset['urdf_relative'],ec['object_mass_kg'])
                ev=ObjectEvaluator(r['body'],ec['object_body_id'],ec['table_body_id'],r['links'],
                       object_mass_kg=ec['object_mass_kg'],gravity_world_m_s2=ec['gravity_world_m_s2'])
            elif r['event']=='state':
                gt=r['object_evaluation']
                tcp=[row[3] for row in r['W_T_link_post']['ip140_palm'][:3]]
                val=ev.observe(r['index'],r['phase'],r['raw_manifolds'],r['actual_arm_and_finger_states'][-2:],r['object_state'],tcp,
                               engineering_violations=r['violations'])
                seen+=1
                for k in set(val)|set(gt):
                    if json.dumps(val.get(k),sort_keys=True)!=json.dumps(gt.get(k),sort_keys=True):
                        diffkeys[k]=diffkeys.get(k,0)+1
                        if diffkeys[k]==1: print('first mismatch',r['index'],k,str(val.get(k))[:1000],str(gt.get(k))[:1000])
    expected=read(origin/'result.json')['evaluator_terminal_result']
    normalized=dict(expected)
    normalized['trial_completed']=normalized.pop('episode_completed')
    del normalized['slot_passed']
    normalized['negative_control_passed']=None
    final=ev.finish()
    delta=[k for k in set(final)|set(normalized) if final.get(k)!=normalized.get(k)]
    path=VERSION/'audit/platform_zero_challenge/static_preflight/path_result.json'
    safety=read(path)
    receipt=dict(archived_steps_replayed=seen,archived_source=str(origin.relative_to(ROOT)),archived_telemetry_sha256=digest(origin/'telemetry.jsonl.gz'),
       archived_recorded_result_sha256=digest(origin/'result.json'),evaluator_decision_mismatch_counts=diffkeys,
       terminal_mismatch_fields=sorted(delta),success_semantics_equal=not delta and not diffkeys,
       ordinary_locked_candidate_path_result_sha256=digest(path),ordinary_locked_candidate_failure_category=candidate_static_failure(safety),
       static_states=safety['snapshots_completed'],returned_engineering_result=final['grasp_success'],
       new_method_trials=0,physics_steps=0,full_challenge_passed=(not diffkeys and not delta and seen==4728 and candidate_static_failure(safety)=='path_failure'))
    write(VERSION/'audit/platform_execution_interface_v2.json',receipt)
    if not receipt['full_challenge_passed']:raise RuntimeError('archived evaluator/interface mismatch: '+repr(receipt))
    print(receipt)
if __name__=='__main__':main()


