"""Read independent precheck and dynamic slot-14 API streams and episode evidence. Never connect to PyBullet."""
import base64
import gzip
import hashlib
import json
from collections import Counter
from math import isfinite
from copy import deepcopy
from pathlib import Path

from damping_write import OBJECT_URDF, validate_damping_write
from structural_filter import STRUCTURAL_PAIRS
from base_table_mount import validate as validate_base_table


HERE = Path(__file__).resolve().parent
PHASES = (('settle',240),('opening',480),('transit',672),('pregrasp',672),
          ('approach',672),('close',480),('lift',792),('hold',720))
PHASE = [phase for phase, count in PHASES for _ in range(count)]


def read_optional(relative):
    path = HERE / relative
    return json.loads(path.read_bytes()) if path.is_file() else None


def read_gzip(relative):
    with gzip.open(HERE / relative, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def audit_connection(base, scope):
    base=Path(base)
    violations = []
    counts, returns, errors = Counter(), Counter(), Counter()
    filters, projected, loads, creations, object_writes = [], [], [], [], []
    contact_by_step, object_by_step, joints_by_step, state_rows = {}, {}, {}, {}
    # Facts are read in API sequence order even when the gate stopped mid-admission.
    seen_links = {}
    bindings = dict(robot=None, object=None, table=None, client=None)
    total = 0
    pending = None
    first_step_seq = None
    disconnect_seq = None
    trace_path = 'full_api/calls.jsonl.gz'
    trace_present = (base / trace_path).is_file()
    def local_optional(relative):
        path=base/relative
        return json.loads(path.read_bytes()) if path.is_file() else None
    gate = local_optional('zero_readback.json') or {}
    session = read_optional('session_result.json') or {}
    offline = read_optional('offline_gate_replay.json') or {}
    receipt = local_optional('damping_write_receipt.json')
    charged = (HERE/'episodes/14/precharge.json').is_file()
    unknown_calls=[]; reset_calls=[]; gravity_calls=[]

    def issue(label):
        if label not in violations:
            violations.append(label)

    if trace_present:
        for row in read_gzip((base/trace_path).relative_to(HERE)):
            seq, kind = row[:2]
            if kind == 'call':
                if pending is not None or seq != total + 1:
                    issue('missing_or_out_of_order_API_call_response')
                    if pending is not None: unknown_calls.append(pending)
                pending = row
                total += 1
                method = row[3]
                counts[method] += 1
                if disconnect_seq is not None:
                    issue('API_call_after_disconnect')
                if method == 'stepSimulation' and first_step_seq is None:
                    first_step_seq = seq
                if method in ('resetJointState','resetBasePositionAndOrientation','resetBaseVelocity'):
                    reset_calls.append(dict(seq=seq,method=method,args=row[4],kwargs=row[5]))
                    if scope=='dynamic' and first_step_seq is not None: issue('dynamic_reset_after_first_step')
                    if method!='resetJointState' or not row[4] or row[4][0]!=bindings['robot']:
                        issue('reset_outside_registered_robot_joint_setup')
                if method=='createConstraint': issue('prohibited_attachment_constraint_API')
                continue
            if pending is None or pending[0] != seq or kind not in ('return','error'):
                issue('orphan_or_invalid_API_response')
                pending = None
                continue
            call = pending
            method, args, kw = call[3:6]
            pending = None
            value = row[3]
            ok = kind == 'return'
            (returns if ok else errors)[method] += 1
            if method in ('connect','disconnect','loadURDF','changeDynamics') or seq == first_step_seq:
                projected.extend((call,row))
            if method == 'connect' and ok:
                bindings['client'] = value
            elif method == 'disconnect' and ok:
                disconnect_seq = seq
            elif method == 'loadURDF' and ok:
                path = args[0] if args else kw.get('fileName')
                loads.append(dict(seq=seq, body=value, path=path))
                if path and Path(path).as_posix().endswith(OBJECT_URDF):
                    bindings['object'] = value
                elif path and Path(path).name == 'ur5e_ip140_r1.urdf':
                    bindings['robot'] = value
            elif method == 'setGravity':
                gravity_calls.append(dict(seq=seq,args=args,kwargs=kw,returned=ok,value=value))
            elif method == 'createMultiBody' and ok:
                creations.append(dict(seq=seq, body=value, args=args, kwargs=kw))
                if len(creations) == 1:
                    bindings['table'] = value
            elif method == 'getJointInfo' and ok and args and args[0] == bindings['robot']:
                name = base64.b64decode(value[12]['bytes_b64']).decode()
                seen_links[name] = value[0]
            elif method.startswith('setCollisionFilter'):
                filters.append(dict(seq=seq, method=method,args=args,kwargs=kw,returned=ok,value=value))
            elif method == 'changeDynamics' and (args[0] if args else kw.get('bodyUniqueId')) == bindings['object']:
                object_writes.append(dict(seq=seq,args=args,kwargs=kw,returned=ok,value=value))
            elif method in ('createConstraint','resetBasePositionAndOrientation','resetBaseVelocity') and bindings['object'] is not None:
                issue('prohibited_attachment_or_teleport_API')
            step = returns['stepSimulation']
            if step and ok:
                if method == 'getContactPoints' and not args and set(kw) <= {'physicsClientId'}:
                    contact_by_step[step] = value
                elif method in ('getBasePositionAndOrientation','getBaseVelocity') and args and args[0] == bindings['object']:
                    object_by_step.setdefault(step,{})[method] = value
                elif method == 'getJointStates' and args and args[0] == bindings['robot'] and len(value)==8:
                    joints_by_step[step] = value
        if pending is not None:
            unknown_calls.append(pending)
            issue('last_API_call_has_no_response')
        api_summary = local_optional('full_api/summary.json')
        if not api_summary or (api_summary['calls'] != dict(counts) or api_summary['returns'] != dict(returns)
                or api_summary['total_calls'] != total):
            issue('compressed_stream_disagrees_with_API_summary')
        for blob, size in (api_summary or {}).get('lossless_raw_blobs_bytes',{}).items():
            with gzip.open(base/'full_api'/(blob+'.bin.gz'),'rb') as stream:
                raw = stream.read()
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != blob:
                issue('camera_or_array_blob_integrity')
        if counts['connect']!=1 or returns['connect']>1: issue('more_than_one_connection_attempt_in_scope')
        if returns['connect'] and returns['disconnect'] != 1:
            issue('live_connection_disconnect_not_confirmed')

    robot, obj, table, client = (bindings[k] for k in ('robot','object','table','client'))
    expected_filters=[[robot,robot,*[seen_links.get(n) for n in pair],0] for pair in STRUCTURAL_PAIRS]
    expected_filters.append([robot,table,seen_links.get('base_link_inertia'),-1,0])
    for i, f in enumerate(filters):
        if (i>=14 or f['method']!='setCollisionFilterPair' or not f['returned']
                or f['value'] is not None or f['kwargs']!={'physicsClientId':client}
                or f['args']!=expected_filters[i]):
            issue('invalid_filter_pair_arguments_sequence_or_return')
    if gate.get('passed') and len(filters)!=14:
        issue('admitted_without_14_exact_filters')
    damping_violations = None
    if receipt is not None and obj is not None and client is not None:
        damping_violations = validate_damping_write(receipt, obj, client, projected)
        for reason in damping_violations:
            issue('write_proof:'+reason)
    elif gate.get('passed'):
        issue('admitted_without_damping_receipt')
    if (gate.get('passed') or gate.get('physics_step_calls') is not None) and gate.get('physics_step_calls') != 0:
        issue('zero_snapshot_step_count')
    if scope=='path_precheck' and counts['stepSimulation']: issue('physical_call_in_static_precheck')
    if scope=='dynamic' and counts['stepSimulation'] and (not charged or not gate.get('passed') or not offline.get('passed')):
        issue('physical_call_without_charge_and_both_admissions')
    if scope=='dynamic' and bool(session.get('charged')) != charged:
        issue('precharge_receipt_and_session_disagree')
    telem = 'episodes/14/telemetry.jsonl.gz'
    states, intents, acknowledgments = Counter(), 0, 0
    support_steps = aggregate_support_steps = left_steps = right_steps = bilateral_steps = 0
    mass_record=local_optional('zero_step/object021_readback.json') or {}
    actual_mass=(mass_record.get('dynamics') or [None])[0]
    gravity_readback=gate.get('physics_readback',{})
    gravity=[gravity_readback.get('gravityAcceleration'+axis) for axis in 'XYZ']
    if gravity==[None,None,None]: gravity=None
    gravity_z=gravity[2] if gravity is not None else None
    weight=None if actual_mass is None or gravity_z is None else actual_mass*abs(gravity_z)
    if gate.get('passed'):
        if len(gravity_calls)!=1 or gravity_calls[0]['args']!=[0,0,-9.81] or not gravity_calls[0]['returned']:
            issue('gravity_audited_setter_mismatch')
    if gate.get('passed') and (actual_mass!=.04 or gravity!=[0.,0.,-9.81]
            or gate.get('gravity_world_m_s2_readback')!=gravity): issue('aggregate_context_not_frozen')
    mount_steps = 0
    mount_failed_steps = []
    actual_baseline = None
    max_lift = None
    final_rows = []
    if scope=='dynamic' and (HERE/telem).is_file():
        for row in read_gzip(telem):
            event = row['event']
            if event == 'step_intent':
                intents += 1
                if row['index'] != intents: issue('intent_order')
            elif event == 'step_return':
                acknowledgments += 1
                if row['index'] != acknowledgments: issue('step_return_order')
            elif event == 'state':
                step = row['index']
                states[row['phase']] += 1
                if not (1<=step<=4728 and row['phase']==PHASE[step-1]) or step != len(state_rows)+1:
                    issue('state_phase_or_order')
                state_rows[step] = True
                mount = row.get('base_table_mount_audit')
                if mount is None:
                    issue('missing_per_step_mount_invariant')
                else:
                    mount_steps += 1
                    independent = validate_base_table(mount['observation'])
                    if independent != mount:
                        issue('mount_invariant_recalculation_disagrees')
                    if not mount.get('passed'):
                        mount_failed_steps.append(step)

                contacts = row['raw_manifolds']
                if contacts != contact_by_step.get(step):
                    issue('telemetry_contacts_vs_API')
                if row['actual_arm_and_finger_states'] != joints_by_step.get(step):
                    issue('telemetry_joint_states_vs_API')
                truth = row['object_state']
                raw_obj = object_by_step.get(step,{})
                if raw_obj.get('getBasePositionAndOrientation') != [truth['position'],truth['orientation']] or (
                        raw_obj.get('getBaseVelocity') != [truth['linear_velocity'],truth['angular_velocity']]):
                    issue('telemetry_object_pose_or_twist_vs_API')
                if step == 240:
                    actual_baseline = truth['position'][2]
                lift = None if actual_baseline is None else truth['position'][2]-actual_baseline
                if lift is not None:
                    max_lift = lift if max_lift is None else max(lift,max_lift)
                pads = set()
                support = False
                present=False; legal_forces=[]; aggregation_errors=[]
                for c in contacts:
                    finite_contact=(len(c)==14 and all(isinstance(v,(int,float)) and isfinite(v) for v in
                        (*c[:5],*c[5],*c[6],*c[7],*c[8:11],*c[11],c[12],*c[13])))
                    roles = {c[1]:c[3],c[2]:c[4]}
                    if set(roles)=={obj,table}:
                        present=True
                        if not finite_contact: aggregation_errors.append('nonfinite_or_incomplete_contact')
                        elif roles[obj]!=-1 or roles[table]!=-1: aggregation_errors.append('wrong_support_link')
                        elif c[8]<-.001: aggregation_errors.append('external_penetration_below_minus_1mm')
                        else: legal_forces.append(max(c[9],0.))
                    if c[9]>.2 and c[8]>=-.001 and roles.get(obj)==-1:
                        if roles.get(table)==-1:
                            support = True
                        if robot in roles:
                            pads.add(roles[robot])
                left, right = [seen_links.get('ip140_'+s+'_pad') in pads for s in ('left','right')]
                aggregate_force=sum(legal_forces)
                if not isfinite(aggregate_force): aggregation_errors.append('nonfinite_aggregate_table_force')
                aggregate=None if aggregation_errors else aggregate_force>.2
                fraction=None if weight is None or not isfinite(aggregate_force) else aggregate_force/weight
                aggregate_expected=dict(table_contact_present=present,
                    table_normal_force_sum_N=aggregate_force if isfinite(aggregate_force) else None,
                    table_load_fraction=fraction,aggregate_table_support=aggregate)
                aggregate_support_steps += int(aggregate is True)
                support_steps += int(support)
                left_steps += int(left)
                right_steps += int(right)
                bilateral_steps += int(left and right)
                decision = row['object_evaluation']
                if row.get('violations') and decision.get('engineering_passed') is not False:
                    issue('engineering_failure_recorded_as_pass')
                if any(decision.get(k)!=v for k,v in aggregate_expected.items()):
                    issue('aggregate_support_vs_independent_raw_sum')
                if aggregation_errors and (decision.get('engineering_passed') is not False or
                        not set(aggregation_errors).issubset(set(decision.get('violations',[])))):
                    issue('invalid_support_contact_hidden_by_aggregation')
                if (decision['lift_m'], decision['effective_left_target_contact'],
                    decision['effective_right_target_contact'], decision['table_support']) != (lift,left,right,support):
                    issue('raw_observations_disagree_with_contact_or_lift_decision')
                if step >= 4249:
                    final_rows.append(dict(step=step,lift_m=lift,effective_left_target_contact=left,
                            effective_right_target_contact=right,effective_table_support=support,
                            slipped=decision['slipped'],engineering_violations=row['violations'],**aggregate_expected))
    if scope=='dynamic' and (intents,acknowledgments)!=(counts['stepSimulation'],returns['stepSimulation']):
        issue('telemetry_intents_or_returns_vs_API')
    if scope=='dynamic' and (set(state_rows) != set(contact_by_step) or len(state_rows)!=returns['stepSimulation']):
        issue('returned_steps_with_missing_comparable_state')
    if gate.get('passed') and obj is None: issue('missing_object_identity_in_live_trace')
    terminal = read_optional('episodes/14/final_480_steps.json') if scope=='dynamic' else None
    if terminal and ([{k:v[k] for k in ('step','lift_m','effective_left_target_contact',
                'effective_right_target_contact','effective_table_support','table_contact_present',
                'table_normal_force_sum_N','table_load_fraction','aggregate_table_support','slipped','engineering_violations')} for v in terminal['observed_rows']]!=final_rows
                or terminal['actual_observed']!=len(final_rows)):
        issue('terminal_interval_vs_telemetry')
    if final_rows and terminal is None: issue('missing_terminal_interval_record')
    res = read_optional('episodes/14/result.json') if scope=='dynamic' else None
    if res:
        if (res['step_intents'],res['step_returns']) != (intents,acknowledgments):
            issue('episode_count_vs_streams')
        if res['slot_passed'] != (res['engineering_gate_passed'] and res['grasp_success']):
            issue('episode_boolean_semantics')
        if res['grasp_success'] and (len(final_rows)!=480 or
                any(v['lift_m']<.10 or not(v['effective_left_target_contact'] and v['effective_right_target_contact'])
                    or v['aggregate_table_support'] is not False or v['slipped'] or v['engineering_violations'] for v in final_rows)):
            issue('terminal_observables_do_not_support_recorded_success')
    initial_path = 'zero_step/initial_world_contacts.json'
    initial = local_optional(initial_path) or {}
    failed_observations = initial.get('failures',[])
    first_rejected = None
    if failed_observations:
        first = failed_observations[0]
        idx = first['index']
        records = (initial['world_manifolds'] if first.get('query')=='getContactPoints' else
                   initial.get('direct_closest',{}).get(first.get('group'),[]))
        first_rejected = dict(file=initial_path,failure=first,
                              raw_witness=records[idx] if 0<=idx<len(records) else None)
    readback_fields = [f for f in gate.get('fields',[]) if f.get('status')!='pass']
    trace_ran = bool(counts['stepSimulation'])
    report = dict(version=HERE.name,scope=scope,passed=not violations,violations=violations,
                  observed=trace_present,offline_passed=offline.get('passed'),session_stop=session.get('stop'),
                  zero_passed=gate.get('passed'),zero_required_mismatches=gate.get('required_mismatches'),
                  zero_snapshot_step_calls=gate.get('physics_step_calls'),
                  zero_first_rejected_geometry=first_rejected,zero_first_failed_field=(
                      readback_fields[0] if readback_fields else None),
                  zero_object_table_manifold_indices=initial.get('object_table_support_manifold_indices'),
                  zero_evidence_files=sorted(str(p.relative_to(HERE).as_posix()) for p in (base/'zero_step').glob('*.json')),
                  bindings=bindings,urdf_loads=loads,static_body_creations=creations,
                  counts=dict(calls=dict(counts),returns=dict(returns),exceptions=dict(errors),total_calls=total),
                  filter_calls=filters,filter_count=len(filters),external_filter_count=sum(
                      f['args'][:2] != [robot,robot] for f in filters),object_writes=object_writes,
                  damping_violations=damping_violations,damping_audit_observed=damping_violations is not None,
                  first_step_api_sequence=first_step_seq,disconnect_api_sequence=disconnect_seq,
                  physics_step_calls=counts['stepSimulation'],physics_step_returns=returns['stepSimulation'],
                  unknown_physics_steps=counts['stepSimulation']-returns['stepSimulation'],
                  precharged=charged if scope=='dynamic' else False,expected_cumulative_starts=8+int(charged),expected_remaining=6-int(charged),
                  telemetry_observed=scope=='dynamic' and (HERE/telem).is_file(),states_per_phase=dict(states),
                  settle_baseline_z_m=actual_baseline,max_observed_lift_m=max_lift,
                  contact_step_counts=dict(left=left_steps,right=right_steps,bilateral=bilateral_steps,
                      table_support=support_steps,aggregate_table_support=aggregate_support_steps) if trace_ran else None,
                  final480_observed=len(final_rows),final480_range=[4249,4728],physics_observed=trace_ran,base_table_mount_steps=mount_steps,base_table_mount_failed_steps=mount_failed_steps,
                  unknown_calls=unknown_calls,reset_calls=reset_calls,
                  support_context=dict(actual_mass_kg=actual_mass,gravity_z_m_s2=gravity_z,
                    gravity_evidence='getPhysicsEngineParameters gravityAccelerationXYZ corroborated against setGravity trace',gravity_setter_calls=gravity_calls,weight_N=weight),
                  final480_independent=final_rows)
    return report


def audit():
    """Separate session identities; fourteen filters are required in EACH admitted session."""
    pre=audit_connection(HERE/'path_precheck','path_precheck')
    dynamic=audit_connection(HERE,'dynamic')
    violations=[scope+':'+v for scope,record in (('precheck',pre),('dynamic',dynamic)) for v in record['violations']]
    def issue(value):
        if value not in violations: violations.append(value)
    session=read_optional('session_result.json') or {}
    path=read_optional('path_precheck/result.json')
    initial=read_optional('ledger_initial.json') or {}
    ledger=read_optional('slot_ledger.json')
    outcome=read_optional('result.json')
    precharge=read_optional('episodes/14/precharge.json')
    charged=precharge is not None
    calls=dynamic['physics_step_calls'];returned=dynamic['physics_step_returns']
    if pre['physics_step_calls']: issue('precheck_must_remain_zero_physics')
    if dynamic['observed'] and not (path and path.get('passed') and pre.get('zero_passed')):
        issue('dynamic_connection_without_passed_static_path')
    if calls and not charged: issue('unbilled_first_step')
    if charged and not calls: issue('precharge_without_recorded_first_call_requires_reconciliation')
    if bool(session.get('charged'))!=charged: issue('session_precharge_disagreement')
    if path:
        if path.get('physics_step_calls')!=0 or path.get('charged') is not False:
            issue('static_path_charged_or_physics')
        count=0;failures=[]
        telemetry=HERE/'path_precheck/path_snapshots.jsonl.gz'
        if telemetry.exists():
            for r in read_gzip(telemetry.relative_to(HERE)):
                count+=1
                if r.get('snapshot')!=count or r.get('physics_step_calls')!=0:
                    issue('static_snapshot_sequence_or_physics')
                if not r.get('passed'): failures.append(count)
        if count!=path.get('snapshots_checked'): issue('static_snapshot_count_vs_result')
        if failures and failures!=[count]: issue('static_path_not_stopped_at_first_rejection')
        if path.get('passed') and (count!=4728 or failures): issue('static_full_path_pass_without_complete_evidence')
    if precharge and (precharge.get('slot')!=14 or precharge.get('parent_slot')!=7
            or precharge.get('starts_before')!=8 or precharge.get('starts_after')!=9
            or precharge.get('remaining_after')!=5 or precharge.get('total')!=14):
        issue('precharge_budget_or_slot_identity')
    if ledger:
        if (ledger.get('starts')!=8+int(charged) or ledger.get('existing_budget')!=14
                or ledger.get('total_slots')!=14 or ledger.get('remaining_total_budget')!=6-int(charged)
                or ledger.get('slots',[])[:13]!=initial.get('slots',[])[:13]
                or ledger['slots'][13].get('charged')!=charged
                or sum(bool(s.get('charged')) for s in ledger['slots'])!=8+int(charged)
                or any(s.get('charged') for s in ledger['slots'][7:12])):
            issue('ledger_history_budget_or_slot14_mismatch')
        if ledger.get('physical_steps')!=initial.get('physical_steps',0)+returned:
            issue('cumulative_returned_step_ledger_mismatch')
    else: issue('missing_sealed_ledger')
    if outcome:
        for key in ('episode_completed','engineering_gate_passed','grasp_success','slot_passed'):
            if not calls and outcome.get(key) is not None: issue('unrun_physics_misrepresented_'+key)
        if (outcome.get('slot')!=14 or outcome.get('charged')!=charged
                or outcome.get('actual_step_calls')!=calls or outcome.get('actual_step_returns')!=returned
                or outcome.get('total_budget')!=14 or outcome.get('starts_after')!=8+int(charged)):
            issue('sealed_result_vs_observed_budget')
    else: issue('missing_sealed_result')
    prior=(initial.get('slots') or [{}]*7)[6]
    historical=read_optional(prior['result']) if prior.get('result') else None
    if not (prior.get('charged') and historical and historical.get('grasp_success') is False
            and historical.get('slot_passed') is False): issue('historical_slot7_failure_not_preserved')
    report=deepcopy(dynamic)
    report.update(schema='IP140-smoke13-two-connection-independent-audit-v1',
        passed=not violations,violations=violations,
        precheck_connection=pre,dynamic_connection=dynamic,
        static_path_result=path,static_path_passed=path.get('passed') if path else None,
        connection_filter_rule='14 per admitted connection: 13 internal + 1 exact fixed base-table; never sum across connections',
        filter_calls_per_connection=dict(precheck=pre['filter_count'],dynamic=dynamic['filter_count']),
        total_physics_step_calls=pre['physics_step_calls']+calls,
        total_physics_step_returns=pre['physics_step_returns']+returned,
        precharged=charged,expected_cumulative_starts=8+int(charged),expected_remaining=6-int(charged),
        total_budget=14,independent_confirmation=False,not_p3=True,
        historical_slot7_result_unchanged=True if historical and historical.get('slot_passed') is False else False)
    with (HERE/'api_and_episode_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return report
