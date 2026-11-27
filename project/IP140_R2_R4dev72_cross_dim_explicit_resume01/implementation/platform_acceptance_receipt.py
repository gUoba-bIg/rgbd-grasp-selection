import ast, hashlib, json
from pathlib import Path
from platform_imports import ROOT,VERSION,PLATFORM
from platform_trial import candidate_static_failure, dynamic_category
from scene_world import write
def main():
    q=VERSION/'audit/platform_zero_challenge'
    def read(p):return json.loads(p.read_bytes())
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    cp=read(q/'eligibility_api/summary.json');cs=read(q/'static_api/summary.json')
    selection=read(q/'selection.json');static=read(q/'static_preflight/path_result.json')
    eval_receipt=read(VERSION/'audit/platform_execution_interface_v2.json')
    for obj in (cp,cs):
        assert obj['actual_native_step_calls']==0 and not obj['exceptions'] and obj['calls']['connect']==obj['returns']['disconnect']==1 and obj['returns']['setCollisionFilterPair']==14
    assert selection['selected']=={'Raw':0,'A':2,'C0':2} and len(selection['eligibility'])==4
    assert static['snapshots_completed']==3216 and static['first_failure']['snapshot']==3217 and candidate_static_failure(static)=='path_failure'
    assert eval_receipt['full_challenge_passed'] and eval_receipt['physics_steps']==0
    unchanged_frozen_reference=ast.dump(next(n for n in ast.parse((PLATFORM/'aggregate_support.py').read_text(encoding='utf-8-sig')).body if getattr(n,'name',None)=='aggregate_table_contacts'))
    support=ast.dump(next(n for n in ast.parse((VERSION/'implementation/platform_support.py').read_text(encoding='utf-8-sig')).body if getattr(n,'name',None)=='aggregate_table_contacts'))
    assert support==unchanged_frozen_reference
    outputs=['eligibility_api/calls.jsonl.gz','eligibility_api/summary.json','eligibility_setup/eligibility_setup.json','platform_input.json','selection.json','handoff_intent.json','archived_settle_pose.json','static_api/calls.jsonl.gz','static_api/summary.json','static_setup/setup.json','static_preflight/dual_channel_snapshots.jsonl.gz','static_preflight/path_result.json','static_preflight/left_right_contact_entry_roots.json']
    code=['implementation/platform_imports.py','implementation/platform_plan.py','implementation/platform_select.py','implementation/platform_static.py','implementation/platform_episode.py','implementation/platform_support.py','implementation/platform_evaluator.py','implementation/platform_trace.py','implementation/platform_trial.py','implementation/scene_world.py','implementation/scene_asset.py','audit/platform_execution_interface_v2.json']
    files={str((q/p).relative_to(VERSION)):sha(q/p) for p in outputs}
    files.update({p:sha(VERSION/p) for p in code})
    result=dict(interface_closed=True,method_trials=0,native_physics_steps=0,new_dynamical_grasp_results=0,
      actual_p3_pool_rows=4,actual_target_free_selection_by_method=selection['selected'],selected_candidate_for_handoff='archived-021-qual-probe:p3:0',
      locked_before_target_loaded=True,no_fallback=True,static_safety_result='path_failure',static_attempted_states=3217,
      static_returned_states=3216,registered_contact_entries=0,error=static['first_failure'],api_native_connections={'eligibility':1,'locked_safety':1,'dynamic':0},
      full_historical_telemetry_evaluation_diff_count=len(eval_receipt['terminal_mismatch_fields']),original_engineering_ledger_reopened=False,
      completed_historical_snapshot_comparison=eval_receipt['archived_steps_replayed'],new_grasp_success=None,
      caveat='Raw candidate was rejected by the pre-lift bilateral root gate. New method physics has no returned results in this pretrial test. Existing smoke17 success only verifies identical evaluation of already recorded telemetry.',
      hashes=files)
    write(VERSION/'audit/platform_zero_interface_receipt.json',result)
    print(result['interface_closed'])
if __name__=='__main__':main()

