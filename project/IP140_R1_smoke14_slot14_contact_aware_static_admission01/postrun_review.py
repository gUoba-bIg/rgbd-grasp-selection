"""Read-only extraction after stopped smoke14; no engine import or experiment calls."""
from pathlib import Path
import gzip,hashlib,json,datetime
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,obj):
 with (HERE/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
result=read('result.json');api=read('static_api_audit_r3.json');replay=read('positive_negative_replay.json')
assert result['snapshots_completed']==2748 and not result['static_admission_passed']
assert api['passed'] and api['physics_steps']==0 and api['new_charges']==0
records={}
with gzip.open(HERE/'dual_channel_snapshots.jsonl.gz','rt',encoding='utf-8') as f:
 for n,line in enumerate(f,1):
  r=json.loads(line)
  assert r['snapshot']==n
  if n in (2747,2748):records[n]=r
assert n==2748
(HERE/'postrun_evidence').mkdir(exist_ok=False)
for n,r in records.items():put('postrun_evidence/snapshot_'+str(n)+'.json',r)
last=records[2748]
manifolds=last['reference_channel']['raw_manifolds']
assert len(manifolds)==1
contact=manifolds[0]
assert contact[8]>0 and all(contact[k]==0 for k in (9,10,12))
closest=last['reference_channel']['raw_closest']['robot_target']
wanted={374455,374456,374459,374460,374523,374524,374527,374528}
source_api=[]
with gzip.open(HERE/'full_api/calls.jsonl.gz','rt',encoding='utf-8') as f:
 for n,line in enumerate(f,1):
  if n in wanted:source_api.append(dict(jsonl_line_1based=n,raw_record=json.loads(line)))
assert len(source_api)==8
put('postrun_evidence/api_excerpt.json',dict(source='full_api/calls.jsonl.gz',source_sha256=sha(HERE/'full_api/calls.jsonl.gz'),records=source_api,raw_timestamps_preserved=True))
put('first_failure_positive_gap.json',dict(version=HERE.name,status='implementation_semantics_defect_unresolved',source='dual_channel_snapshots.jsonl.gz',source_sha256=sha(HERE/'dual_channel_snapshots.jsonl.gz'),jsonl_line_1based=2748,phase='close',phase_tick=12,q_ref_m=last['q_ref_m'],q_proxy_m=last['q_proxy_m'],raw_manifold=contact,raw_robot_target_closest=closest,original_violations=last['violations'],classification_records=last['reference_channel']['classifications'],raw_normal_tangential_forces_N=[contact[k] for k in (9,10,12)],native_manifold_distance_m=contact[8],diagnosis='Manifold presence was treated as contact even with positive signed distance and exactly zero reported forces; fresh closest remains positive. Source classifier requires an entry certificate before the distance sign crossing exists.',source_rule='contact_semantics.py: hit=manifold or d<=0; admission demanded when hit',execution_first_failure_preserved=True,model_collision_failure_demonstrated=False,engine_bug_demonstrated=False,new_connection_after_stop=False))
hist=replay['historical_reclassification']
put('smoke13_2757_2758_reclassification.json',dict(version=HERE.name,scope='Historical raw-reference semantic mapping only; no new root certificate or proxy observation for snapshots 2757/2758',historical_reclassification=hist,actual_smoke14_roots=read('left_right_contact_entry_roots.json'),actual_smoke14_stop_snapshot=2748,actual_smoke14_reached_2757_or_2758=False,conditional_label='post_contact_motor_reference_overtravel',conditions='Both fresh same-side entry and the independent proxy must pass before current admission. Historical interpolation does not meet this condition.',new_admission_passed=False,original_smoke13_outcome_unchanged=True,dynamic_actual_minus_1mm_gate_unchanged=True,missing_values_not_filled=True))
observations=hist['observations']
historical_tip=[dict(snapshot=x['snapshot'],distance_m=x['raw_contact'][8],tip_depth_m=x['effective_face_audit']['tip_depth_m'],tip_edge_risk=x['effective_face_audit']['tip_edge_risk'],source_sha256=x['source_sha256']) for x in observations]
put('first_contact_tip_edge_evidence.json',dict(version=HERE.name,new_left_entry_root=None,new_right_entry_root=None,new_first_zero_distance_contact=None,new_tip_edge_at_entry=None,reason='Stopped at close tick12 with positive clearance, before entry localization; both new roots remain unobserved.',historical_only=historical_tip,historical_interpolation_m=hist['q_zero_linear_interpolation_m'],historical_interpolation_is_not_new_root=True,historical_tip_flag_not_dynamic_failure=True,current_positive_gap_witness_audits=last['reference_channel']['pad_witness_audits']))
put('missing_evidence.json',dict(version=HERE.name,planned_snapshots=4728,actual_paired_snapshots=2748,actual_successful_snapshots=2747,first_failed_snapshot=2748,unvisited_snapshots={'first':2749,'last':4728,'count':1980},new_left_root=None,new_right_root=None,new_root_queries=0,proxy_contact_stop_behavior_exercised=False,post_contact_cache_isolation_branch_exercised=False,loaded_contact_dynamics=None,physical_steps=0,new_charges=0,full_static_admission_passed=False,reason='First real zero-step result hit a classifier implementation defect; stopped and disconnected, no retry or synthesized completion.'))
put('postrun_coverage_gap.json',dict(version=HERE.name,original_offline_suite=dict(total=99,passed=99,source='positive_negative_replay.json',sha256=sha(HERE/'positive_negative_replay.json')),new_observed_counterexample=dict(source='first_failure_positive_gap.json',phase='close',manifold_distance_m=contact[8],forces_N=[contact[k] for k in (9,10,12)],actual_classifier_rejected=True),original_suite_coverage_complete=False,added_runtime_fix=False,proposed_minimal_change='For close-only registered pad-target proximity with positive distance, finite correctly identified face witness and zero force, preserve manifold as a near-contact record without claiming entry. Keep actual zero-distance positive-to-negative bracket/root mandatory. All nonpad and preclose gates unchanged.',not_implemented=True,new_execution_requires_review=True))
put('postrun_evidence_manifest.json',dict(version=HERE.name,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),inputs={x:sha(HERE/x) for x in ['result.json','static_api_audit_r3.json','positive_negative_replay.json','left_right_contact_entry_roots.json','dual_channel_snapshots.jsonl.gz','full_api/calls.jsonl.gz']},scope='Read-only extraction; does not connect or load an engine, query collisions, move state, change original records or change the ledger',physical_steps=0,new_charges=0))
print(json.dumps(dict(extracted=True,snapshots=2748,missing=1980,physical_steps=0,new_charges=0),ensure_ascii=False))
