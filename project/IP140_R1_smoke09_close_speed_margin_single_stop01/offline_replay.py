"""Offline classified fixtures and archived motion checks for the corrective slot 13. No simulation library or world is used."""
import hashlib
import json
import math
from pathlib import Path
from stop_evaluator import (DT_S, FORBIDDEN_CLASS, PHASE_AT_STEP, PHASE_STEPS, RIGHT_PAD,
                            StopEvaluator, validate_filter_calls)
from structural_filter import STRUCTURAL_PAIRS
from static_proof import additional_proofs, PROOF_SOURCES

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREV=ROOT/'IP140_R1_smoke07_mount_baseline_invariant01'
LINKS=PREV/'zero_step'/'all_links_readback.json'
N=len(PHASE_AT_STEP)
STILL=(0.05,0.,(0.,)*6,0.)
DEFAULT_FINGERS=(STILL,STILL)
OUTPUT=HERE/'offline_gate_replay.json'
SCHEMA='IP140_R1_smoke09_stop_offline_v1'
SOURCES=(LINKS,PREV/'preregistration.json',HERE/'approved_task.txt',HERE/'stop_evaluator.py',
         HERE/'structural_filter.py',Path(__file__).resolve(),*PROOF_SOURCES)

def source_hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}

def _row(robot,fixture,link,force=.25,distance=0.):
    # Synthetic native-schema record. p_A-p_B=d*n, normal is on fixture.
    return [0,robot,fixture,link,-1,[.50,-.025+distance,.28],[.50,-.025,.28],
            [0.,1.,0.],distance,force,0.,[1.,0.,0.],0.,[0.,0.,1.]]

def run_replay():
    lr=json.loads(LINKS.read_bytes())
    links={n:r['index'] for n,r in lr.items()}
    cfg=json.loads((PREV/'preregistration.json').read_bytes())
    def find(obj):
        if isinstance(obj,dict):
            if obj.get('candidate_id')=='IP140-BENCH-06:single_sided_stop':return True
            return any(find(v) for v in obj.values())
        return isinstance(obj,list) and any(find(v) for v in obj)
    if not find(cfg):raise ValueError('missing frozen slot 6 candidate')
    robot,fixture=0,91 # artificial different IDs; all joint indices are archived actual readbacks
    right=links[RIGHT_PAD]
    source=lambda **kw: _row(robot,fixture,right,**kw)
    results=[]
    def case(group,name,tweak=None,*,success=False,code=None,pattern='terminal_24',phase_error=None,extra=False):
        ev=StopEvaluator(robot,fixture,links)
        selected=None
        target=N-23 if phase_error is None else phase_error
        count_left=0;count_right=0;statuses=[];near=[];hard=None
        for i,s in enumerate(PHASE_AT_STEP,1):
            rows=[];fingers=DEFAULT_FINGERS
            if pattern=='terminal_24' and i>=N-23:rows=[source()]
            elif pattern=='terminal_23' and i>=N-22:rows=[source()]
            elif pattern=='interrupted' and i>=N-24 and i!=N-11:rows=[source()]
            elif pattern=='old_only' and N-64<=i<=N-41:rows=[source()]
            if tweak and (i==target or (i>target and group in (1,3))):
                rows,fingers=tweak(rows,i)
            rr=ev.observe(i,s,rows,fingers)
            if rows or fingers!=DEFAULT_FINGERS:
                count_right+=int(rr['valid_right_contact'])
                count_left+=int(any(row[1:5]==[robot,fixture,links['ip140_left_pad'],-1] for row in rows))
                near.extend(rr['near_cap'])
                statuses.extend(c.get('classification') for c in rr['contacts'] if 'classification' in c)
                if selected is None or not rr['passed']:
                    selected=safe_json(dict(step=i,phase=s,raw_manifolds=rows,raw_finger_states=fingers,
                                            evaluator_result=rr))
            if not rr['passed']:
                hard=rr;break
        if extra and hard is None:
            rr=ev.observe(N+1,phase_error,[],DEFAULT_FINGERS)
            selected=dict(step=N+1,phase=phase_error,evaluator_result=rr)
        end=ev.finish()
        classifier_ok=(not success or FORBIDDEN_CLASS in statuses)
        near_ok=(not name.startswith('at_20N') or bool(near))
        ok=(end['slot_passed'] is success and end['expected_negative_detected'] is success and
            end['grasp_success'] is False and (code is None or code in end['violations']) and
            classifier_ok and near_ok)
        results.append(dict(number=group,variant=name,passed=ok,expected_slot_passed=success,
                        expected_rejection=code,pattern=pattern,source='synthetic native-schema contact / archived link indices',
                        selected_input_and_observation=selected,actual=end,valid_right_steps=count_right,
                        left_fixture_step_count=count_left,near_cap_sample_count=len(near),
                        classifications=sorted(set(statuses))))

    case(1,'last_24_right',success=True)
    def reverse(_rows,i):
        r=source(force=.25);r[1],r[2]=r[2],r[1];r[3],r[4]=r[4],r[3]
        r[5],r[6]=r[6],r[5];r[7]=[0.,-1.,0.]
        return [r],DEFAULT_FINGERS
    case(1,'reversed_body_order',reverse,success=True)
    def at_bound(_rows,i):
        return [source(force=24.,distance=-.001)],(STILL,(.025,-.04,(0.,)*6,-20.))
    case(1,'at_20N_motor_and_minus_1mm_and_speed_boundary',at_bound,success=True)
    for p in ('terminal_23','interrupted','old_only'):
        case(2,p,pattern=p,code='missing_final_24_consecutive')
    for force in (.2,.1999999,0.):
        case(3,'subthreshold_'+str(force),lambda r,i,f=force:([source(force=f)],DEFAULT_FINGERS),
             code='missing_final_24_consecutive')
    def other(name):return _row(robot,fixture,links[name])
    case(4,'left_even_zero_force',lambda r,i:([_row(robot,fixture,links['ip140_left_pad'],0.)],DEFAULT_FINGERS),code='wrong_robot_link_on_fixture')
    case(5,'bilateral',lambda r,i:(r+[other('ip140_left_pad')],DEFAULT_FINGERS),code='wrong_robot_link_on_fixture')
    for part in ('ip140_palm','ip140_left_bone','ip140_right_bone','wrist_3_link','upper_arm_link'):
        case(6,part,lambda r,i,n=part:([other(n)],DEFAULT_FINGERS),code='wrong_robot_link_on_fixture')
    case(7,'minus_1mm_exceeded',lambda r,i:([source(distance=math.nextafter(-.001,-math.inf))],DEFAULT_FINGERS),
         code='penetration_below_minus_1mm')
    def changed(k,val):
        def mutate(rows,i):
            r=source();r[k]=val;return [r],DEFAULT_FINGERS
        return mutate
    for k,value,why in ((2,fixture+1,'unregistered_external_manifold'),(1,robot+100,'unregistered_external_manifold'),
                       (4,1,'wrong_fixture_link'),(3,98,'wrong_robot_link_on_fixture')):
        case(8,'changed_native_field_'+str(k),changed(k,value),code=why)
    case(9,'nonfinite_normal_force',changed(9,float('nan')),code='nonfinite_or_incomplete_contact')
    case(9,'nonfinite_witness',changed(5,[float('inf'),0.,0.]),code='nonfinite_or_incomplete_contact')
    case(9,'overflow_vector_impulse',lambda r,i:([r[0][:10]+[1.e308,[1.e308,0.,0.]]+r[0][12:]],DEFAULT_FINGERS),
         code='nonfinite_impulse')
    for label,side,value,reason in (
            ('velocity',0,(0.,math.nextafter(.04,math.inf),(0.,)*6,0.),'finger_speed_above_0.04_m_s'),
            ('motor_force',1,(0.,0.,(0.,)*6,-20.00001),'finger_motor_force_above_20_N'),
            ('nonfinite_velocity',0,(0.,float('inf'),(0.,)*6,0.),'nonfinite_or_incomplete_finger_state')):
        def bad(rows,i,s=side,v=value):
            states=list(DEFAULT_FINGERS);states[s]=v;return rows,states
        case(9,label,bad,code=reason)
    calls=[dict(method='setCollisionFilterPair',sequence=i+1,pair=p,args=[robot,robot,links[p[0]],links[p[1]],0],
               kwargs={'physicsClientId':0},returned=True,return_value=None) for i,p in enumerate(STRUCTURAL_PAIRS)]
    for label,seq,need in [('13_registered_internal',calls,None),
                           ('extra_fixture_filter',calls+[dict(calls[-1],sequence=14,args=[robot,fixture,right,-1,0])],'fixture_filter'),
                           ('fixture_replaces_internal',calls[:-1]+[dict(calls[-1],args=[robot,fixture,right,-1,0])],'fixture_filter')]:
        actual=validate_filter_calls(robot,fixture,links,seq)
        ok=actual['passed'] if need is None else not actual['passed'] and need in actual['violations']
        results.append(dict(number=10,variant=label,passed=ok,expected_rejection=need,actual=actual,
                            synthetic_filter_records=seq))
    total=0
    for phase,count in PHASE_STEPS[:-1]:
        total+=count
        case(11,phase+'_has_fixture',lambda r,i:([source(force=0.)],DEFAULT_FINGERS),
             phase_error=total,code='fixture_contact_outside_close')
    for p in ('lift','hold'):
        # Valid tail first; even a single extra return invalidates the final outcome.
        case(12,'after_recognition_'+p,phase_error=p,extra=True,code='lift_or_hold_forbidden')
    proof=additional_proofs(safe_json(results))
    return dict(schema=SCHEMA,passed=all(r['passed'] for r in results) and proof['passed'],
                static_proofs=proof,case_groups=list(range(1,13)),
                source_sha256=source_hashes(),synthetic_fixture_body=fixture,synthetic_robot_body=robot,
                rubric='Post-step STOP classifier. Synthetic records exercise the live gate; SI units. '
                       'A single registered right-pad point must exceed 0.2 N; subthreshold points are not summed. '
                       'All fixtures here are offline test inputs. Nonfinite test values are serialized as explicit strings.',
                dt_s=DT_S,case_count=len(results),cases=results)

def safe_json(obj):
    if isinstance(obj,float) and not math.isfinite(obj):return str(obj)
    if isinstance(obj,dict):return {k:safe_json(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [safe_json(v) for v in obj]
    return obj

def validate_offline(report):
    if not isinstance(report,dict) or report.get('passed') is not True:return False
    expected=safe_json(run_replay())
    return safe_json(report)==expected and expected['passed']

def persist_report(path=None):
    path=OUTPUT if path is None else Path(path)
    if Path.cwd().resolve()!=ROOT or path.resolve()!=OUTPUT:raise ValueError('unexpected offline output location')
    report=safe_json(run_replay())
    with path.open('x',encoding='utf8',newline='\n') as fp:
        json.dump(report,fp,indent=2,ensure_ascii=False,allow_nan=False)
        fp.write('\n')
    return report

if __name__=='__main__':
    raise SystemExit(0 if persist_report()['passed'] else 2)
