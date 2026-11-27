"""One permanently sealed R4 method. No rank fallback and no engineering-slot identifiers."""
import argparse
import gzip
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from platform_imports import VERSION, ROOT, bind_frozen_libraries
bind_frozen_libraries()
from platform_trace import R4Trace
from platform_static import scan
from platform_episode import run_method
from platform_support import bind_asset
from scene_world import setup_world, settle_state, confirm_same_settle, write, read

def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def frozen_intent(path):
    if Path.cwd().resolve()!=ROOT or path.name!='intent.json' or path.parent.parent!=VERSION/'trials':
        raise RuntimeError('wrong method work directory')
    sealed=read(VERSION/'sealed_intents.json');r=read(path);ident=path.parent.name
    if (sealed['version']!=VERSION.name or sealed['method_budget']!=72 or len(sealed['files'])!=72
            or sealed['files'].get(path.relative_to(VERSION).as_posix())!=sha(path)):
        raise RuntimeError('intent not present in complete pre-outcome seal')
    if r['trial_id']!=ident or r['attempt_limit']!=1 or r['no_postlock_fallback'] is not True or r['ranking_uses_ground_truth'] is not False:
        raise RuntimeError('unbound trial or forbidden fallback')
    for tag in ('pool','stable_state','ranking'):
        q=(VERSION/r[tag+'_path']).resolve()
        if not q.is_relative_to(VERSION) or sha(q)!=r[tag+'_sha256']:raise RuntimeError(tag+' identity changed')
    directory=VERSION/'units'/r['unit_id']
    sp=directory/'selection.json'
    if sha(sp)!=r['selection_sha256']:raise RuntimeError('selection changed')
    s=read(sp);idx=r['selected_pool_index']
    if s['used_ground_truth'] is not False or s['selected'][r['method']]!=idx:
        raise RuntimeError('selection not irrevocably locked')
    rank=read(VERSION/r['ranking_path'])
    eligibility=s['eligibility']
    if (rank['unit_id']!=r['unit_id'] or rank['method']!=r['method'] or rank['shared_pool_sha256']!=r['pool_sha256']
            or s['unit_id']!=r['unit_id'] or s['never_fallback_after_postlock_safety'] is not True
            or any(type(i) is not int for i in rank['complete_order'])
            or sorted(rank['complete_order'])!=list(range(len(eligibility)))):
        raise RuntimeError('rank order, pool or unit binding failed')
    for i,record in enumerate(eligibility):
        if type(record['index']) is not int or record['index']!=i or type(record['eligible']) is not bool:
            raise RuntimeError('unbound candidate')
    actual=next((k for k in rank['complete_order'] if s['eligibility'][k]['eligible']),None)
    if actual!=idx:raise RuntimeError('not the first eligible candidate')
    row=None if idx is None else s['eligibility'][idx]
    if (row is None)!=(r['selected_candidate_id'] is None) or row is not None and r['selected_candidate_id']!=row['plan']['candidate_id']:
        raise RuntimeError('intent and motion plan disagree')
    return r,s,row

def no_selection_outcome(eligibility):
    if not eligibility:
        return dict(failure_category='no_eligible_candidate',failure_detail=[],
                    no_candidate_subtype='empty_shared_eligible_pool')
    return dict(failure_category='no_executable_candidate',
                failure_detail=[row['reason'] for row in eligibility],
                no_candidate_subtype='all_eligible_checks_failed')

@contextmanager
def connection(directory,scope):
    import pybullet as backend
    trace=R4Trace(backend,directory,scope)
    client=None
    try:
        client=trace.connect(trace.DIRECT)
        if client<0:raise RuntimeError('DIRECT connection failed')
        yield trace,client
    finally:
        try:
            if client is not None and client>=0:trace.disconnect(physicsClientId=client)
        finally:trace.close()

def checked_zero(ctx):
    p=ctx.p
    if p.actual_step_calls or p.counts['stepSimulation'] or p.exceptions or p.counts['setCollisionFilterPair']!=14 or p.returns['setCollisionFilterPair']!=14:
        raise RuntimeError('incorrect zero-step session')
    return dict(passed=True,dynamic_execution_authorized=False,eligible_slots_after_zero_gate=[],
                object_body_id=ctx.target,table_body_id=ctx.table)

CANDIDATE_GEOMETRY={
    'pad_target_contact_before_close','nonpad_target_collision','simultaneous_nonpad_target_collision_at_entry',
    'finger_independent_robot_target_collision','proxy_pad_target_penetration','entry_not_on_registered_surface',
    'entry_surface_not_registered','pad_witness_position_outside_registered_surfaces','entry_normal_wrong_inner_face_direction',
    'unregistered_self_collision','robot_table_collision','reference_negative_or_manifold_before_legal_entry',
}
COMPANION={
    'missing_per_side_entry_certificate','entry_solver_not_passed','entry_witness_face_audit_missing_or_failed',
    'pad_contact_without_legal_entry','entry_nonpad_target_clearance_unproven','no_clean_positive_to_nonpositive_entry',
}
MOUNT_COMPANION={'mount_pair_identity','other_robot_table_manifold'}
COLLISION={'nonpad_robot_object_contact','robot_table_contact','unregistered_self_manifold','external_penetration_below_minus_1mm'}
MOTION={'hold_tcp_drift','executed_approach_tracking','finger_speed_above_0.04_m_s','finger_motor_force_above_20_N',
        'measured_slider_limit','unloaded_midline_or_symmetry','preopening_not_maintained','opening_stability','pad_parallelism'}
MOTION.update(f'arm_{i}.{flag}' for i in range(6) for flag in ('joint_range','joint_speed_limit','effort_saturated','hold_q','hold_dq'))


def witnessed_nonmount_table(audit,raw,ctx):
    """A foreign robot/table row is rejected twice by the mount-specific reporter.
    Review its native identity and replay every true mount gate, on a copy, before
    assigning the duplicated reporter labels to that candidate's forbidden collision.
    """
    from base_table_mount import validate
    from platform_evaluator import _contact_values
    if not audit or set(audit.get('violations',()))!=MOUNT_COMPANION or not isinstance(raw,list):
        return False
    observation=audit.get('observation',{})
    base=ctx.links['base_link_inertia']
    if (observation.get('robot'),observation.get('table'),observation.get('base_index'),observation.get('stage'))!=(ctx.robot,ctx.table,base,'dynamic'):
        return False
    if observation.get('manifolds')!=raw or set(validate(observation)['violations'])!=MOUNT_COMPANION:
        return False
    links=set(ctx.links.values())|{-1}
    kept=[]
    for row in raw:
        if _contact_values(row) is None:return False
        if {row[1],row[2]}!={ctx.robot,ctx.table}:
            kept.append(row)
            continue
        robot_link,table_link=(row[3],row[4]) if row[1]==ctx.robot else (row[4],row[3])
        if table_link!=-1 or robot_link not in links or robot_link==base:return False
    return len(kept)<len(raw) and validate(dict(observation,manifolds=kept))['passed']


def read_returned_record(path,event,index):
    with gzip.open(path,'rt',encoding='utf8') as fp:
        for line in fp:
            record=json.loads(line)
            if event is None and record.get('snapshot')==index:return record
            if event is not None and record.get('event')==event and record.get('index')==index:return record
    raise ValueError('missing recorded failure witness')


def candidate_static_failure(result,*,directory=None,ctx=None):
    reasons=result.get('violations',[])
    if result.get('exception') is not None:
        return 'path_failure' if not reasons and result['exception']=="ValueError('both_registered_entries_required_before_lift')" else None
    leaves=set()
    channels=set()
    for reason in reasons:
        labels=reason.split(':')
        if len(labels)<2 or labels[0] not in ('reference','proxy'):return None
        if labels[-1] in MOUNT_COMPANION:
            if len(labels)!=2:return None
            channels.add(labels[0])
        elif len(labels)==3 and labels[1] in ('left','right'):
            leaves.add(labels[2])
        elif (len(labels) in (4,5) and labels[1] in ('manifold','robot_target','robot_target_supplemental','robot_self','robot_table')
              and labels[2].isdigit() and (len(labels)==4 or labels[3]=='witness')):
            leaves.add(labels[-1])
        else:return None
    if leaves-(CANDIDATE_GEOMETRY|COMPANION) or not leaves&CANDIDATE_GEOMETRY:return None
    if channels:
        if directory is None or ctx is None or 'robot_table_collision' not in leaves:return None
        rec=read_returned_record(directory/'dual_channel_snapshots.jsonl.gz',None,result['first_failure']['snapshot'])
        if rec.get('violations')!=reasons:return None
        for tag in channels:
            ch=rec[tag+'_channel']
            if not witnessed_nonmount_table(ch.get('base_table_mount_audit'),ch.get('raw_manifolds'),ctx):return None
            if not any(s.startswith('manifold:') and s.endswith(':robot_table_collision') for s in ch['violations']):return None
    return 'path_failure'


def dynamic_category(row,*,directory=None,ctx=None):
    violations=set(row.get('violations',[]))
    rec=None
    if directory is not None and ctx is not None and violations&COLLISION:
        from platform_evaluator import _contact_values
        rec=read_returned_record(directory/'telemetry.jsonl.gz','state',row['step_returns'])
        if set(rec.get('violations',[]))!=violations or not rec.get('raw_manifolds'):return None
        legal={ctx.robot:set(ctx.links.values())|{-1},ctx.table:{-1},ctx.target:{-1}}
        for contact in rec['raw_manifolds']:
            if _contact_values(contact) is None:return None
            if any(body not in legal or link not in legal[body] for body,link in ((contact[1],contact[3]),(contact[2],contact[4]))):return None
    if violations & MOUNT_COMPANION:
        if directory is None or ctx is None or not (MOUNT_COMPANION|{'robot_table_contact'})<=violations:return None
        if not witnessed_nonmount_table(rec.get('base_table_mount_audit'),rec.get('raw_manifolds'),ctx):return None
        violations-=MOUNT_COMPANION
    if violations:
        if violations-(MOTION|COLLISION):return None
        return 'forbidden_collision' if violations&COLLISION else 'motion_hard_gate'
    if row.get('trial_completed') and row.get('engineering_gate_passed'):
        if row['grasp_success']:return 'success'
        causes=set(row.get('grasp_failure_reasons',[]))
        if causes-{'object_slipped','no_bilateral_grasp_observed','missing_final_480_valid_hold_steps','terminal_height_requirement',
                   'terminal_bilateral_contact_requirement','terminal_table_support_observed','terminal_aggregate_table_support_observed'}:return None
        if 'object_slipped' in causes:return 'slip'
        if 'no_bilateral_grasp_observed' in causes:return 'closure_failure'
        if 'terminal_height_requirement' in causes:return 'insufficient_lift'
        if 'terminal_bilateral_contact_requirement' in causes:return 'closure_failure'
        if causes&{'terminal_table_support_observed','terminal_aggregate_table_support_observed','missing_final_480_valid_hold_steps'}:
            return 'hold_failure'
        return None
    if row.get('failure_reason')=="RuntimeError('frozen wall/simulation timeout')":return 'timeout'
    return None


def worker(path):
    intent,selection,row=frozen_intent(path)
    destination=path.parent
    write(destination/'worker_open.json',dict(trial_id=intent['trial_id'],intent_sha256=sha(path),selected_pool_index=intent['selected_pool_index']))
    result=dict(trial_id=intent['trial_id'],method=intent['method'],unit_id=intent['unit_id'],selected_pool_index=intent['selected_pool_index'],
        trial_completed=False,engineering_gate_passed=False,grasp_success=False,negative_control_passed=None,shared_fault=False,
        physical_steps=0,physical_returns=0,failure_category=None,failure_detail=None,forbidden_collision=False,
        planning_sec=sum(x['planning_seconds'] for x in selection['eligibility']),static_sec=None,execution_sec=None,
        planning_basis='shared pool eligibility over all candidates, distinct from one locked candidate static check',
        no_postlock_fallback=True)
    trace=None;dynstart=None
    try:
        if row is None:
            result.update(no_selection_outcome(selection['eligibility']))
        else:
            plan=row['plan'];started=time.perf_counter()
            with connection(destination/'static_api','contact_aware_static') as (p,client):
                ctx=setup_world(p,client,intent['scene_key'],destination/'static_setup',settled_state=VERSION/intent['stable_state_path'])
                sr=scan(p,client,ctx.robot,ctx.links,plan,checked_zero(ctx),intent=intent,output_root=destination/'static_preflight')
                if p.exceptions or p.actual_step_calls or p.counts['stepSimulation']:raise RuntimeError('invalid safety-only native audit')
            result['static_sec']=time.perf_counter()-started
            if not sr['passed']:
                code=candidate_static_failure(sr,directory=destination/'static_preflight',ctx=ctx)
                if code is None:raise RuntimeError('unclassified safety admission error: '+str(sr.get('first_failure')))
                result.update(failure_category=code,failure_detail=sr['first_failure'])
            else:
                roots=read(destination/'static_preflight/left_right_contact_entry_roots.json')
                if sr['snapshots_completed']!=4728 or not roots['both_passed'] or sr['filter_call_count']!=14:
                    raise RuntimeError('incomplete native preflight')
                receipt=destination/'safety_receipt.json'
                write(receipt,dict(passed=True,intent_sha256=sha(path),static_steps=0,path_digest=sha(destination/'static_preflight/path_result.json'),both_registered_contact_entries=True))
                dynstart=time.perf_counter()
                with connection(destination/'dynamic_api','dynamic') as (trace,client):
                    ctx=setup_world(trace,client,intent['scene_key'],destination/'dynamic_setup')
                    checked_zero(ctx)
                    bind_asset(ctx.object_urdf_path,ctx.object_mass_kg)
                    def authorize():
                        trace.authorize_dynamic(client,intent_path=path,intent_sha256=sha(path),static_receipt=receipt)
                        return dict(intent_sha256=sha(path),safety_receipt_sha256=sha(receipt))
                    def verify_settle(c):
                        observed=settle_state(c);write(destination/'returned_settle_state.json',observed)
                        confirm_same_settle(VERSION/intent['stable_state_path'],observed)
                    episode=destination/'episode';episode.mkdir(exist_ok=False)
                    returned=run_method(ctx,plan,episode,intent=intent,before_first_step=authorize,verify_settle=verify_settle)
                    result.update(physical_steps=trace.actual_step_calls,physical_returns=trace.returns['stepSimulation'],
                                  episode_result_sha256=sha(episode/'result.json'),execution_sec=time.perf_counter()-dynstart)
                    code=dynamic_category(returned,directory=episode,ctx=ctx)
                    if trace.exceptions or returned['unknown_steps'] or code is None or returned['step_returns']!=trace.returns['stepSimulation']:
                        raise RuntimeError('unknown native, telemetry or evaluation result: '+str(returned['failure_reason']))
                    result.update(trial_completed=returned['trial_completed'],engineering_gate_passed=returned['engineering_gate_passed'],
                                grasp_success=returned['grasp_success'],forbidden_collision=code=='forbidden_collision',
                                failure_category=None if code=='success' else code,failure_detail=returned['failure_reason'])
    except Exception as e:
        result.update(shared_fault=True,failure_category='shared_integrity_fault',failure_detail=repr(e),traceback=traceback.format_exc(),
                      engineering_gate_passed=None,grasp_success=None,forbidden_collision=None)
        if trace is not None:result.update(physical_steps=trace.actual_step_calls,physical_returns=trace.returns['stepSimulation'])
        if dynstart is not None:result['execution_sec']=time.perf_counter()-dynstart
    write(destination/'method_result.json',result)
    return 2 if result['shared_fault'] else 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('intent',type=Path)
    raise SystemExit(worker(parser.parse_args().intent.resolve()))


