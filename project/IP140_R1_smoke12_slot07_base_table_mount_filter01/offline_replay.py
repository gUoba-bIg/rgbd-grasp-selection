"""Slot 7 synthetic returned-state sequences, evaluated with the live pure gates."""
import hashlib
import json
import math
from pathlib import Path
from object_evaluator import ObjectEvaluator, PHASE_AT_STEP, PHASE_STEPS, validate_filter_calls
from structural_filter import STRUCTURAL_PAIRS
from offline_checks import check_imports, check_schedule, damping_cases
from mount_offline import mount_cases

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREV=ROOT/'IP140_R1_smoke07_mount_baseline_invariant01'
LINKS=PREV/'zero_step/all_links_readback.json'
OUT=HERE/'offline_gate_replay.json'
SOURCES=(LINKS,ROOT/'configs/simulation_v52r2.json',ROOT/'scripts/sim_v52r2_episode.py',
         HERE/'approved_scope.md',HERE/'object_evaluator.py',HERE/'structural_filter.py',HERE/'speed_policy.py',
         Path(__file__).resolve())
N=len(PHASE_AT_STEP)
FINGERS=((.03,0.,(0.,)*6,0.),(.03,0.,(0.,)*6,0.))

def safe_json(v):
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    if isinstance(v,dict):return {k:safe_json(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [safe_json(x) for x in v]
    return v

def hashes():
    paths=set(SOURCES) | set(HERE.glob('*.py')) | {HERE/'trajectory_plan.json', HERE/'preregistration.json',
                                                   HERE/'model/manifest.json', HERE/'model/ur5e_ip140_r1.urdf', HERE/'base_table_registration.json'}
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

class ReplayFailed(Exception):
    def __init__(self, rows):
        self.rows = list(rows)
        super().__init__(rows[-1]['name'])

class CheckedResults(list):
    def append(self, row):
        super().append(row)
        if row.get('passed') is not True:
            raise ReplayFailed(self)

def contact(a,b,la,lb,fn=.3,d=0.):
    return [0,a,b,la,lb,[0.,0.,d],[0.,0.,0.],[0.,0.,1.],d,fn,0.,[1.,0.,0.],0.,[0.,1.,0.]]

def build():
    links={n:r['index'] for n,r in json.loads(LINKS.read_bytes()).items()}
    robot,target,table=0,91,92
    left,right=links['ip140_left_pad'],links['ip140_right_pad']
    results=CheckedResults()
    results.append(check_imports())
    results.append(check_schedule())
    for damping_case in damping_cases():
        results.append(damping_case)
    def object_at(z):
        return dict(body_id=target,position=[.48,.06,z],orientation=[0.,0.,0.,1.],
                    linear_velocity=[0.,0.,0.],angular_velocity=[0.,0.,0.])
    def target_contacts(force=.3,distance=0.):
        return [contact(robot,target,left,-1,force,distance),contact(robot,target,right,-1,force,distance)]
    def case(name,mutate=None,*,success=False,engineering=True,hard=None,extra=False):
        e=ObjectEvaluator(robot,target,table,links);sample=None;last=None
        for i,phase in enumerate(PHASE_AT_STEP,1):
            z=.125 if phase in ('lift','hold') else 0.
            raw=target_contacts() if phase in ('close','lift','hold') else [contact(target,table,-1,-1)]
            state=object_at(z);tcp=[.48,.06,z];fingers=FINGERS
            if mutate:raw,state,tcp,fingers=mutate(i,phase,raw,state,tcp,fingers)
            last=e.observe(i,phase,raw,fingers,state,tcp)
            if i in (240,4249,4728) or not last['passed']:
                sample=dict(step=i,phase=phase,raw_contacts=raw,object_state=state,tcp_world_position=tcp,
                            raw_finger_states=fingers,result=last)
            if not last['passed']:break
        if extra and last['passed']:
            last=e.observe(N+1,'hold',target_contacts(),FINGERS,object_at(.125),[.48,.06,.125])
        final=e.finish()
        ok=(final['slot_passed']==success and final['grasp_success']==success and
            final['engineering_gate_passed']==engineering and
            final['episode_completed']==(final['observed_steps']==N) and
            final['slot_passed']==(final['grasp_success'] and final['engineering_gate_passed']) and
            (not engineering or final['episode_completed']) and
            (hard is None or hard in final['failure_reason']))
        results.append(dict(name=name,passed=ok,expected_grasp_success=success,expected_engineering_passed=engineering,
                            expected_rejection=hard,selected_raw_input_and_result=safe_json(sample),actual=final))
    case('nominal_bilateral_final_480',success=True)
    def boundaries(i,p,r,s,t,f):
        if p in ('lift','hold'):s['position'][2]=.10;t[2]=.10
        if p in ('close','lift','hold'):r=target_contacts(25.,-.001)
        return r,s,t,((.03,.040,(0.,)*6,20.),(.03,-.040,(0.,)*6,-20.))
    case('height_0p10_distance_minus1mm_speed_and_motor_at_limit',boundaries,success=True)
    def reverse(i,p,r,s,t,f):
        for c in r:
            c[1],c[2]=c[2],c[1];c[3],c[4]=c[4],c[3];c[5],c[6]=c[6],c[5];c[7]=[-v for v in c[7]]
        return r,s,t,f
    case('reversed_native_endpoint_order',reverse,success=True)
    def window479(i,p,r,s,t,f):
        if i==N-479:r=r[:1]
        return r,s,t,f
    case('hold479_only',window479,hard='missing_final_480_valid_hold_steps')
    def stale480(i,p,r,s,t,f):
        if i>N-240:r=r[:1]
        return r,s,t,f
    case('earlier480_does_not_replace_terminal480',stale480,hard='missing_final_480_valid_hold_steps')
    def insufficient(i,p,r,s,t,f):
        if p=='hold':s['position'][2]=math.nextafter(.1,-math.inf)
        return r,s,t,f
    case('height_nextafter_below_0p10',insufficient,hard='missing_final_480_valid_hold_steps')
    def single(i,p,r,s,t,f):
        if p in ('close','lift','hold'):r=r[:1]
        return r,s,t,f
    case('single_pad_is_ordinary_failure',single,hard='missing_final_480_valid_hold_steps')
    def no_grasp(i,p,r,s,t,f):
        return [contact(target,table,-1,-1)],object_at(0.),[.48,.06,.125],f
    case('never_grasped_is_completed_failure',no_grasp,hard='missing_final_480_valid_hold_steps')
    def subthreshold(i,p,r,s,t,f):
        if p in ('close','lift','hold'):r=target_contacts(.2)
        return r,s,t,f
    case('exact0p2_is_not_effective',subthreshold,hard='missing_final_480_valid_hold_steps')
    def support(i,p,r,s,t,f):
        if p=='hold':r.append(contact(target,table,-1,-1,1.))
        return r,s,t,f
    case('hold_table_support_fails_grasp',support,hard='missing_final_480_valid_hold_steps')
    def forcezero(i,p,r,s,t,f):
        if p=='hold':r.append(contact(target,table,-1,-1,0.))
        return r,s,t,f
    case('zero_force_table_manifold_recorded_not_load_support',forcezero,success=True)
    def slip(i,p,r,s,t,f):
        # A formed grasp loses bilateral contact for 48 steps and drops below .01 m;
        # recovery later cannot clear the existing slipped flag.
        if 3300<=i<3348:r=[];s=object_at(0.);t=[.48,.06,.20]
        return r,s,t,f
    case('48_step_loss_and_drop_remains_slipped_after_recovery',slip,hard='object_slipped')
    def separation_slip(i,p,r,s,t,f):
        if 3300<=i<3348:r=[];t=[.48,.06,.4]
        return r,s,t,f
    case('48_step_loss_and_separation',separation_slip,hard='object_slipped')
    for link in ('ip140_palm','ip140_left_bone','ip140_right_bone','wrist_3_link','upper_arm_link'):
        def nonpad(i,p,r,s,t,f,n=link):
            if i==3217:r.append(contact(robot,target,links[n],-1,0.))
            return r,s,t,f
        case('target_'+link,nonpad,engineering=False,hard='nonpad_robot_object_contact')
    for link in ('ip140_left_pad','ip140_palm','base_link','base_link_inertia'):
        def hit_table(i,p,r,s,t,f,n=link):
            if i==1:r.append(contact(robot,table,links[n],-1,0.))
            return r,s,t,f
        case('all_robot_table_forbidden_'+link,hit_table,engineering=False,hard='robot_table_contact')
    for which in ('target','support'):
        def penetration(i,p,r,s,t,f,k=which):
            if i==1:
                r=([contact(robot,target,left,-1,.3,math.nextafter(-.001,-math.inf))] if k=='target' else
                   [contact(target,table,-1,-1,.3,math.nextafter(-.001,-math.inf))])
            return r,s,t,f
        case('external_penetration_'+which,penetration,engineering=False,hard='external_penetration_below_minus_1mm')
    def unknown(i,p,r,s,t,f):
        if i==1:r.append(contact(robot,93,left,-1,0.))
        return r,s,t,f
    case('unregistered_body_even_zero_force',unknown,engineering=False,hard='unknown_contact_identity')
    def wrong_link(i,p,r,s,t,f):
        if i==1:r=[contact(robot,target,left,1)]
        return r,s,t,f
    case('wrong_target_link',wrong_link,engineering=False,hard='wrong_object_link')
    def self_contact(i,p,r,s,t,f):
        if i==1:r=[contact(robot,robot,*[links[x] for x in STRUCTURAL_PAIRS[-1]],0.)]
        return r,s,t,f
    case('filtered_pair_reappears',self_contact,engineering=False,hard='filtered_pair_manifold')
    def nan_contact(i,p,r,s,t,f):
        if i==1:r[0][11][0]=float('nan')
        return r,s,t,f
    case('nonfinite_contact_tangent',nan_contact,engineering=False,hard='nonfinite_or_incomplete_contact')
    def nan_object(i,p,r,s,t,f):
        if i==240:s['linear_velocity'][0]=float('nan')
        return r,s,t,f
    case('nonfinite_settle_baseline',nan_object,engineering=False,hard='object_identity_or_nonfinite_state')
    def nan_tcp(i,p,r,s,t,f):
        if i==1:t[0]=float('inf')
        return r,s,t,f
    case('nonfinite_tcp',nan_tcp,engineering=False,hard='nonfinite_tcp')
    def speed(i,p,r,s,t,f):
        if i==1:f=((.03,math.nextafter(.04,math.inf),(0.,)*6,0.),f[1])
        return r,s,t,f
    case('nextafter_observed_speed_fails',speed,engineering=False,hard='finger_speed_above_0.04_m_s')
    def force(i,p,r,s,t,f):
        if i==1:f=(f[0],(.03,0.,(0.,)*6,-math.nextafter(20.,math.inf)))
        return r,s,t,f
    case('nextafter_motor_force_fails',force,engineering=False,hard='finger_motor_force_above_20_N')
    case('extra_step_fails',extra=True,engineering=False,hard='unexpected_phase_or_extra_step')
    # A discontinuous returned-step sequence cannot supply a complete evidence window.
    e=ObjectEvaluator(robot,target,table,links)
    observed=e.observe(2,'settle',[],FINGERS,object_at(0.),[.48,.06,0.])
    results.append(dict(name='unknown_or_skipped_return',passed=not observed['passed'] and
                         'step_index_discontinuity' in observed['violations'],actual=observed))
    calls=[dict(method='setCollisionFilterPair',sequence=i+1,args=[robot,robot,links[a],links[b],0],
                kwargs={'physicsClientId':0},returned=True,return_value=None) for i,(a,b) in enumerate(STRUCTURAL_PAIRS)]
    for name,c,allowed in [('thirteen_internal_filters',calls,True),
                          ('object_filter_forbidden',calls+[dict(calls[-1],sequence=14,args=[robot,target,left,-1,0])],False),
                          ('table_filter_forbidden',calls[:-1]+[dict(calls[-1],args=[target,table,-1,-1,0])],False)]:
        out=validate_filter_calls(robot,target,table,links,c)
        results.append(dict(name=name,passed=out['passed']==allowed,actual=out))
    original_names=[row['name'] for row in results]
    expected=json.loads((ROOT/'IP140_R1_smoke11_slot07_nominal021_damping_writeproof01/offline_gate_replay.json').read_bytes())
    if original_names!=[row['name'] for row in expected['cases']] or len(original_names)!=56:
        raise ValueError('Original 56 case identities or order changed')
    for row in mount_cases(links):
        results.append(row)
    return safe_json(dict(original_case_count=56,extended_case_count=len(results)-56,schema='IP140_R1_smoke12_base_table_mount_offline_v1',passed=all(r['passed'] for r in results),
             cases=results,case_count=len(results),source_sha256=hashes(),
             synthetic_robot_body=robot,synthetic_object_body=target,synthetic_table_body=table,
             note='Synthetic returned-state sequences exercise the runtime evaluator; no physical object or engine exists.',
             time_step_s=1/240,phases=PHASE_STEPS,required_steps=N,final_hold_interval=[4249,4728],
             settle_baseline_rule='Freeze full evaluation-only state at returned step240, no new stability threshold.',
             slip_source='configs/simulation_v52r2.json criteria and scripts/sim_v52r2_episode.py; source hashes included',
             support_rule='Record all target-table manifolds, effective load support uses original strict Fn>0.2 N.',
             base_table_policy='One exact fixed installation pair is audited and physically filtered. Any runtime robot-table manifold still fails. Original 56 cases preserved in order; added mounting cases use the same pure invariant validator as zero and runtime.'))

def validate_offline(report):
    # Receipt binding only. Revalidation must never start a second formal replay.
    if not isinstance(report,dict) or report.get('passed') is not True or not OUT.is_file():
        return False
    if report != json.loads(OUT.read_bytes()) or report.get('source_sha256')!=hashes():
        return False
    cases=report.get('cases',[])
    names={r['name'] for r in cases}
    return (len(names)==len(cases)==report['case_count'] and all(r['passed'] is True for r in cases)
            and {'all_local_entry_points_and_dependencies_import','4728_frozen_openloop_references_and_per_tick_command_counts',
                 'same_client_single_synchronous_audited_write','missing_linearDamping','missing_angularDamping',
                 'nominal_bilateral_final_480','single_pad_is_ordinary_failure','extra_step_fails',
                 'mount_prefilter_registered_evidence','mount_postfilter_invariant',
                 'mount_native_endpoint_order_reversed','mount_dynamic_reappearance',
                 'full14_exact_filter_sequence','full14_reject_fifteenth_filter'}<=names
            and report.get('original_case_count')==56 and report.get('extended_case_count',0)>0)


def persist_report():
    if Path.cwd().resolve()!=ROOT: raise ValueError('explicit project-root working directory required')
    # Opening exclusively reserves the official replay run. An interrupted run is never overwritten.
    with OUT.open('x',encoding='utf8',newline='\n') as output:
        try:
            data=build()
        except ReplayFailed as failed:
            data=dict(schema='IP140_R1_smoke12_base_table_mount_offline_v1', passed=False,
                      first_failure=str(failed), cases=failed.rows, case_count=len(failed.rows), source_sha256=hashes())
        except Exception as exc:
            import traceback
            data=dict(schema='IP140_R1_smoke12_base_table_mount_offline_v1', passed=False, first_failure=repr(exc),
                      cases=[], case_count=0, exception_traceback=traceback.format_exc())
        json.dump(safe_json(data),output,indent=2,ensure_ascii=False,allow_nan=False)
        output.write('\n')
    return data

if __name__=='__main__': raise SystemExit(0 if persist_report()['passed'] else 2)
