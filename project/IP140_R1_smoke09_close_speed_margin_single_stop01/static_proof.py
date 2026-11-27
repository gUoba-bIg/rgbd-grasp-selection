"""Deterministic, zero-physics control and classification proofs for smoke09."""
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/'IP140_R1_smoke08_single_sided_stop01'
EVENT=OLD/'episodes/06/telemetry.jsonl.gz'
OLD_REPORT=OLD/'offline_gate_replay.json'
IDENTICAL=('model/ur5e_ip140_r1.urdf','model/manifest.json','fixture_static.py',
           'structural_filter.py','mountpair.py','mountpair_registration.json','seam_geometry.py')
PROOF_SOURCES=(Path(__file__).resolve(),EVENT,OLD_REPORT,ROOT/'configs/simulation_v52r2.json',
               *(d/name for d in (HERE,OLD) for name in
                 ('trajectory.py','trajectory_plan.json','single_stop.py','execution_control.py','zero_gate.py','preregistration.json',*IDENTICAL)),
               HERE/'speed_policy.py')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def file_sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def isolated_capture(directory,old):
    # A clean interpreter imports only this version's modules. The recording
    # object exposes two motor setters and has no ability to connect or integrate.
    cmd=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--capture',str(directory),str(int(old))]
    result=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if result.returncode!=0: raise ValueError('isolated command replay: '+result.stderr.decode('utf8',errors='replace')[-2000:])
    return json.loads(result.stdout)

def recorder(directory,old):
    sys.path.insert(0,str(directory))
    import single_stop as sim
    import trajectory as traj
    if not old: import speed_policy as sp
    if Path(sim.__file__).resolve().parent!=directory or Path(traj.__file__).resolve().parent!=directory:
        raise RuntimeError('module identity')
    plan=traj.validate_plan()
    limits=read(ROOT/'configs/simulation_v52r2.json')['physics']['arm_forces']
    class Tape:
        POSITION_CONTROL=2
        def setJointMotorControlArray(self,*a,**kw):self.rows.append(('arm',a,kw))
        def setJointMotorControl2(self,*a,**kw):self.rows.append(('finger',a,kw))
    t=Tape()
    result=[]
    for phase,count in traj.PHASES:
        for tick in range(1,count+1):
            qa,dqa,qf=traj.reference(plan,phase,tick)
            kw={} if old else dict(phase=phase,reference_speed=sp.phase_speed(phase),
                                   motor_max_velocity=sp.phase_speed(phase))
            t.rows=[]
            sim.log_motor_commands(t,0,0,[2,3,4,5,6,7],[12,14],qa,qf,limits,**kw)
            result.append([phase,tick,qa.tolist(),dqa.tolist(),qf,t.rows])
    negatives={}
    if not old:
        qa,dq,f=traj.reference(plan,'close',101)
        over=math.nextafter(.039,math.inf)
        for key,rs,ms in [('reference',over,.039),('maxVelocity',.039,over)]:
            t.rows=[]
            try:sim.log_motor_commands(t,0,0,[2,3,4,5,6,7],[12,14],qa,f,limits,phase='close',reference_speed=rs,motor_max_velocity=ms)
            except ValueError as ex: negatives[key]=dict(reason=str(ex),setter_count=len(t.rows),tested_m_s=over)
            else: negatives[key]=dict(reason='accepted',setter_count=len(t.rows),tested_m_s=over)
        try:sp.finger_target('close',101,1./240.,.1,reference_speed=over)
        except ValueError as ex:negatives['trajectory_override']=str(ex)
    return dict(records=result,negative_controls=negatives)

def compare_commands():
    old=isolated_capture(OLD,True);new=isolated_capture(HERE,False)
    A=old['records'];B=new['records']
    mismatch=[];close=[];opening=[];side_targets=[];nonclose=0
    for idx,(a,b) in enumerate(zip(A,B),1):
        ph,k,qa,dq,f,calls=b
        problems=[]
        if len(calls)!=3 or len(a[5])!=3:raise ValueError('expected one arm and two finger setters')
        if a[:4]!=b[:4] or a[5][0]!=calls[0]:problems.append('arm_reference_or_setter')
        if not (0.<=f<=.070):problems.append('out_of_bounds')
        for side,(ac,bc) in enumerate(zip(a[5][1:],calls[1:])):
            pa=ac[2];pb=bc[2]
            target=pb['targetPosition'];mx=pb['maxVelocity']
            if target!=f or not 0.<=target<=.07:problems.append('finger_setter_target')
            if ph!='close':
                if ac!=bc or f!=a[4] or mx!=.04:problems.append('non_close_changed')
            else:
                if ac[:2]!=bc[:2] or pa.keys()!=pb.keys() or any(v!=pb[n] for n,v in pa.items()
                                if n not in ('targetPosition','maxVelocity')):problems.append('other_close_setter_changed')
                if mx!=.039 or pa['maxVelocity']!=.040 or pb['force']!=20.:problems.append('close_limit')
                if target!=max(0.,.050-.039*k*(1./240.)):problems.append('close_reference_value')
            side_targets.append(target)
        if ph=='close':close.append(f)
        else:
            nonclose+=1
            if ph=='opening':
                opening.append(f)
                if f!=min(.05,.040*k*(1./240.)):problems.append('opening_ramp')
        if problems:mismatch.append([idx,problems])
    first=next((k for k,v in enumerate(close,1) if v==0.),None)
    remaining=len(close)-first if first is not None else -1
    neg=new['negative_controls']
    reject=all(neg[k]['reason']==reason and neg[k]['setter_count']==0 for k,reason in
               [('reference','invalid_reference_speed'),('maxVelocity','invalid_motor_max_velocity')])
    correct=(not mismatch and len(A)==len(B)==3216 and nonclose==2736 and len(opening)==len(close)==480
             and first==308 and remaining>=172 and all(v==0. for v in close[first-1:])
             and reject and neg.get('trajectory_override')=='invalid_reference_speed')
    return dict(passed=correct,isolation='two fresh -I -B interpreters; recording PyBullet-shaped setter, no engine imported',
                old_full_reference_and_setters_sha256=digest(A),new_full_reference_and_setters_sha256=digest(B),
                compared_arm_ticks=len(B),unchanged_nonclose_ticks=nonclose,per_tick_mismatches=mismatch,
                fingers_and_setters_compared=len(side_targets),bounds_observed_m=[min(side_targets),max(side_targets)],
                opening_ticks=len(opening),opening_step1_reference_m=opening[0],opening_final_m=opening[-1],
                close_ticks=len(close),close_first_zero_tick=first,reference_tick307_m=close[306],reference_tick308_m=close[307],
                subsequent_zero_ticks_after_first=remaining,close_setter_negative_tests=neg)

def archived_contact():
    from stop_evaluator import StopEvaluator
    ev=None;last=None;real=None;line_number=0;prior_ok=True
    with gzip.open(EVENT,'rt',encoding='utf8') as f:
        for line_number,line in enumerate(f,1):
            r=json.loads(line)
            if r['event']=='initial_configuration':
                body=r['body'];fixture=r['fixture_evaluator_id'];links=r['links']
                ev=StopEvaluator(body,fixture,links)
            if r['event']!='state':continue
            states=r['actual_arm_and_finger_states'][-2:]
            last=ev.observe(r['index'],r['phase'],r['raw_manifolds'],states)
            if r['index']==2837:
                real=r;break
            prior_ok=prior_ok and last['passed']
    if real is None:raise ValueError('missing historic failure')
    dq=real['actual_arm_and_finger_states'][-2][1]
    hard='finger_speed_above_0.04_m_s'
    from stop_evaluator import PHASE_AT_STEP
    def boundary(speed):
        e=StopEvaluator(body,fixture,links)
        st=(.05,speed,(0.,)*6,0.)
        row=e.observe(1,PHASE_AT_STEP[0],[],(st,st))
        return dict(speed_m_s=speed,passed=row['passed'],violations=row['violations'])
    bounds=[boundary(x) for x in (.040,-.040,math.nextafter(.040,math.inf),-math.nextafter(.040,math.inf))]
    ok=(prior_ok and dq==-.04002018523442052 and last['violations']==[hard] and
        all(x['passed']==(i<2) and (i<2 or hard in x['violations']) for i,x in enumerate(bounds)))
    return dict(passed=ok,archived_step=real['index'],phase=real['phase'],phase_tick=real['phase_tick'],
        telemetry_path=EVENT.relative_to(ROOT).as_posix(),source_line_1based=line_number,telemetry_sha256=file_sha(EVENT),
        raw_dq_m_s=[s[1] for s in states],raw_reported_motor_N=[s[3] for s in states],
        expected_threshold_m_s=.040,prior_2836_passed=prior_ok,real_gate=last,observed_boundaries=bounds)

def engine_parameter_calls(path):
    code=ast.parse(path.read_text(encoding='utf-8-sig'))
    return [ast.unparse(n) for n in ast.walk(code) if isinstance(n,ast.Call) and (
        isinstance(n.func,ast.Attribute) and n.func.attr in ('changeDynamics','setGravity','setPhysicsEngineParameter') or
        n.args and isinstance(n.args[0],ast.Attribute) and n.args[0].attr in ('changeDynamics','setGravity','setPhysicsEngineParameter'))]

def unchanged_parameters():
    a=read(OLD/'preregistration.json');b=read(HERE/'preregistration.json')
    old=next(r for r in a['slots'] if r['slot']==6);new=next(r for r in b['slots'] if r['slot']==13)
    slot_fields={k:dict(old=old[k],new=new[k],same=old[k]==new[k]) for k in ('candidate_world','T_W_G_m','pre_open_m','fixture')}
    files={name:{'old_sha256':file_sha(OLD/name),'new_sha256':file_sha(HERE/name)} for name in IDENTICAL}
    old_plan=read(OLD/'trajectory_plan.json');new_plan=read(HERE/'trajectory_plan.json')
    plans={k:old_plan[k]==new_plan[k] for k in ('candidate_world','W_T_G_goal','W_T_G_home','home_joints',
           'waypoints','phase_steps','pre_open_m','limits_rad_rad_s','pregrasp_m','tool0_z_transit_m')}
    config=ROOT/'configs/simulation_v52r2.json'
    cfg=file_sha(config)
    params=[engine_parameter_calls(d/'zero_gate.py') for d in (OLD,HERE)]
    ok=(all(r['same'] for r in slot_fields.values()) and all(v['old_sha256']==v['new_sha256'] for v in files.values())
        and bool(params[0]) and params[0]==params[1] and all(plans.values()) and old_plan['hashes']['config']==new_plan['hashes']['config']==cfg)
    return dict(passed=ok,slot_fields=slot_fields,identical_files=files,trajectories=plans,
           shared_config_sha256=cfg,actual_config=read(config)['physics'],
           frozen_zero_engine_setter_expressions=params[0],current_zero_engine_setter_expressions=params[1])

def additional_proofs(cases):
    prev=read(OLD_REPORT)
    before=prev['cases']
    legacy=dict(passed=prev['passed'] and len(before)==len(cases)==37 and before==cases,
                old_report_sha256=file_sha(OLD_REPORT),groups=len(set(r['number'] for r in cases)),
                equal_case_transcripts=sum(a==b for a,b in zip(before,cases)),old_cases_sha256=digest(before),
                new_cases_sha256=digest(cases))
    result=dict(legacy_case_equivalence=legacy,observed_speed=archived_contact(),
                all_command_ticks=compare_commands(),unchanged_parameters=unchanged_parameters())
    return dict(passed=all(v['passed'] for v in result.values()),**result)

if __name__=='__main__' and len(sys.argv)==4 and sys.argv[1]=='--capture':
    json.dump(recorder(Path(sys.argv[2]).resolve(),sys.argv[3]=='1'),sys.stdout,allow_nan=False,separators=(',',':'))
