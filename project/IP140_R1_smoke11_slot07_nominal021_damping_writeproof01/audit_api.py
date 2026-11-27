"""Read the closed slot-7 API stream and episode evidence. Never connect to PyBullet."""
import base64
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from damping_write import OBJECT_URDF, validate_damping_write
from structural_filter import STRUCTURAL_PAIRS


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


def audit():
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
    trace_present = (HERE / trace_path).is_file()
    gate = read_optional('zero_readback.json') or {}
    session = read_optional('session_result.json') or {}
    offline = read_optional('offline_gate_replay.json') or {}
    receipt = read_optional('damping_write_receipt.json')
    charged = (HERE/'episodes/07/precharge.json').is_file()

    def issue(label):
        if label not in violations:
            violations.append(label)

    if trace_present:
        for row in read_gzip(trace_path):
            seq, kind = row[:2]
            if kind == 'call':
                if pending is not None or seq != total + 1:
                    issue('missing_or_out_of_order_API_call_response')
                pending = row
                total += 1
                method = row[3]
                counts[method] += 1
                if disconnect_seq is not None:
                    issue('API_call_after_disconnect')
                if method == 'stepSimulation' and first_step_seq is None:
                    first_step_seq = seq
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
            elif method in ('createConstraint','resetBasePositionAndOrientation') and bindings['object'] is not None:
                # Slot 7 only permits initial placement through the registered URDF load.
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
            issue('last_API_call_has_no_response')
        api_summary = read_optional('full_api/summary.json')
        if not api_summary or (api_summary['calls'] != dict(counts) or api_summary['returns'] != dict(returns)
                or api_summary['total_calls'] != total):
            issue('compressed_stream_disagrees_with_API_summary')
        for blob, size in (api_summary or {}).get('lossless_raw_blobs_bytes',{}).items():
            with gzip.open(HERE/'full_api'/(blob+'.bin.gz'),'rb') as stream:
                raw = stream.read()
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != blob:
                issue('camera_or_array_blob_integrity')
        if returns['connect'] and returns['disconnect'] != 1:
            issue('live_connection_disconnect_not_confirmed')

    robot, obj, table, client = (bindings[k] for k in ('robot','object','table','client'))
    for i, f in enumerate(filters):
        if (i >= len(STRUCTURAL_PAIRS) or f['method']!='setCollisionFilterPair' or not f['returned']
                or f['value'] is not None or f['kwargs']!={'physicsClientId':client}
                or f['args']!=[robot,robot,*[seen_links.get(n) for n in STRUCTURAL_PAIRS[i]],0]):
            issue('invalid_filter_pair_arguments_sequence_or_return')
    if gate.get('passed') and len(filters) != 13:
        issue('admitted_without_13_internal_filters')
    damping_violations = None
    if receipt is not None and obj is not None and client is not None:
        damping_violations = validate_damping_write(receipt, obj, client, projected)
        for reason in damping_violations:
            issue('write_proof:'+reason)
    elif gate.get('passed'):
        issue('admitted_without_damping_receipt')
    if (gate.get('passed') or gate.get('physics_step_calls') is not None) and gate.get('physics_step_calls') != 0:
        issue('zero_snapshot_step_count')
    if counts['stepSimulation'] and (not charged or not gate.get('passed') or not offline.get('passed')):
        issue('physical_call_without_charge_and_both_admissions')
    if bool(session.get('charged')) != charged:
        issue('precharge_receipt_and_session_disagree')
    ledger = read_optional('slot_ledger.json')
    if ledger:
        if (ledger.get('starts') != 7+int(charged) or
            ledger['slots'][6]['charged'] != charged or
            any(s['charged'] for s in ledger['slots'][7:12])):
            issue('cumulative_ledger_or_future_slot_mismatch')
    telem = 'episodes/07/telemetry.jsonl.gz'
    states, intents, acknowledgments = Counter(), 0, 0
    support_steps = left_steps = right_steps = bilateral_steps = 0
    actual_baseline = None
    max_lift = None
    final_rows = []
    if (HERE/telem).is_file():
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
                for c in contacts:
                    roles = {c[1]:c[3],c[2]:c[4]}
                    if c[9]>.2 and c[8]>=-.001 and roles.get(obj)==-1:
                        if roles.get(table)==-1:
                            support = True
                        if robot in roles:
                            pads.add(roles[robot])
                left, right = [seen_links.get('ip140_'+s+'_pad') in pads for s in ('left','right')]
                support_steps += int(support)
                left_steps += int(left)
                right_steps += int(right)
                bilateral_steps += int(left and right)
                decision = row['object_evaluation']
                if (decision['lift_m'], decision['effective_left_target_contact'],
                    decision['effective_right_target_contact'], decision['table_support']) != (lift,left,right,support):
                    issue('raw_observations_disagree_with_contact_or_lift_decision')
                if step >= 4249:
                    final_rows.append(dict(step=step,lift_m=lift,effective_left_target_contact=left,
                            effective_right_target_contact=right,effective_table_support=support))
    if (intents,acknowledgments)!=(counts['stepSimulation'],returns['stepSimulation']):
        issue('telemetry_intents_or_returns_vs_API')
    if set(state_rows) != set(contact_by_step) or len(state_rows)!=returns['stepSimulation']:
        issue('returned_steps_with_missing_comparable_state')
    if gate.get('passed') and obj is None: issue('missing_object_identity_in_live_trace')
    terminal = read_optional('episodes/07/final_480_steps.json')
    if terminal and ([{k:v[k] for k in ('step','lift_m','effective_left_target_contact',
                'effective_right_target_contact','effective_table_support')} for v in terminal['observed_rows']]!=final_rows
                or terminal['actual_observed']!=len(final_rows)):
        issue('terminal_interval_vs_telemetry')
    if final_rows and terminal is None: issue('missing_terminal_interval_record')
    res = read_optional('episodes/07/result.json')
    if res:
        if (res['step_intents'],res['step_returns']) != (intents,acknowledgments):
            issue('episode_count_vs_streams')
        if res['slot_passed'] != (res['engineering_gate_passed'] and res['grasp_success']):
            issue('episode_boolean_semantics')
        if res['grasp_success'] and (len(final_rows)!=480 or
                any(v['lift_m']<.10 or not(v['effective_left_target_contact'] and v['effective_right_target_contact'])
                    or v['effective_table_support'] for v in final_rows)):
            issue('terminal_observables_do_not_support_recorded_success')
    initial_path = 'zero_step/initial_world_contacts.json'
    initial = read_optional(initial_path) or {}
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
    report = dict(version=HERE.name,passed=not violations,violations=violations,
                  observed=trace_present,offline_passed=offline.get('passed'),session_stop=session.get('stop'),
                  zero_passed=gate.get('passed'),zero_required_mismatches=gate.get('required_mismatches'),
                  zero_snapshot_step_calls=gate.get('physics_step_calls'),
                  zero_first_rejected_geometry=first_rejected,zero_first_failed_field=(
                      readback_fields[0] if readback_fields else None),
                  zero_object_table_manifold_indices=initial.get('object_table_support_manifold_indices'),
                  zero_evidence_files=sorted(str(p.relative_to(HERE).as_posix()) for p in (HERE/'zero_step').glob('*.json')),
                  bindings=bindings,urdf_loads=loads,static_body_creations=creations,
                  counts=dict(calls=dict(counts),returns=dict(returns),exceptions=dict(errors),total_calls=total),
                  filter_calls=filters,filter_count=len(filters),external_filter_count=sum(
                      f['args'][:2] != [robot,robot] for f in filters),object_writes=object_writes,
                  damping_violations=damping_violations,damping_audit_observed=damping_violations is not None,
                  first_step_api_sequence=first_step_seq,disconnect_api_sequence=disconnect_seq,
                  physics_step_calls=counts['stepSimulation'],physics_step_returns=returns['stepSimulation'],
                  unknown_physics_steps=counts['stepSimulation']-returns['stepSimulation'],
                  precharged=charged,expected_cumulative_starts=7+int(charged),expected_remaining=6-int(charged),
                  telemetry_observed=(HERE/telem).is_file(),states_per_phase=dict(states),
                  settle_baseline_z_m=actual_baseline,max_observed_lift_m=max_lift,
                  contact_step_counts=dict(left=left_steps,right=right_steps,bilateral=bilateral_steps,
                      table_support=support_steps) if trace_ran else None,
                  final480_observed=len(final_rows),final480_range=[4249,4728],physics_observed=trace_ran)
    with (HERE/'api_and_episode_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return report
