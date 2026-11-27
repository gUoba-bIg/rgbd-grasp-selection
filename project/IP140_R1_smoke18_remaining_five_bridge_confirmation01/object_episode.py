"""Frozen nominal and miss scene execution. Robot commands use the candidate and clock only."""
import math
import json
import time
import traceback
from pathlib import Path
import numpy as np

from episode_log import Journal, save_rgb, safe_values
from base_table_mount import capture as capture_base_table
from structural_filter import audit_filtered, audit_seams, current_frames
from object_evaluator import ObjectEvaluator, PHASE_STEPS
from speed_policy import FINGER_OBSERVED_HARD_LIMIT_M_S, phase_speed, validate_speeds
from batch_context import SLOT, SLOT_DIR, OBJECT_ID, KIND
from trajectory import PLAN, PHASES, DT, fk, read, reference, sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SIDES=('left','right')
PAD=tuple(f'ip140_{s}_pad' for s in SIDES)
BONE=tuple(f'ip140_{s}_bone' for s in SIDES)
PALM='ip140_palm'
MATRIX_ZERO_M=1e-9
LIMIT=20.

def transform(p, xyz, quat):
    T = np.eye(4)
    T[:3, :3] = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    T[:3, 3] = xyz
    return T


def corner_template(p, body, idx, side, cid):
    row, = p.getCollisionShapeData(body, idx, physicsClientId=cid)
    if row[2] != p.GEOM_BOX or row[0:2] != (body,idx):
        raise RuntimeError('pad primitive or identity differs from gate')
    hx, hy, hz = np.array(row[3]) / 2
    y = hy if side == 'left' else -hy
    inside = np.array([[x,y,z,1.] for x in (-hx,hx) for z in (-hz,hz)]).T
    # This frozen backend reports primitive offsets relative to the local inertial frame.
    # The link pose applied each tick expects a link-relative corner template.
    state = p.getLinkState(body, idx, computeForwardKinematics=1, physicsClientId=cid)
    link_T_inertial = transform(p,state[2],state[3])
    inertial_T_shape = transform(p,row[5],row[6])
    return link_T_inertial @ inertial_T_shape @ inside


def surfaces(frames, corners, requested_width):
    inv_g = np.linalg.inv(frames[PALM])
    points = [inv_g @ frames[name] @ local for name,local in zip(PAD,corners)]
    y_left, y_right = [float(pts[1].mean()) for pts in points]
    gaps = points[1][1] - points[0][1]
    # Never snap the stored raw gaps to zero. Epsilon applies only to their
    # matrix-derived zero boundary; closest points/manifolds are checked separately.
    data = dict(corners_G_m=[pts[:3].T.tolist() for pts in points],gaps_m=gaps.tolist(),
                gap_m=y_right-y_left, midline_m=(y_right+y_left)/2,
                abs_symmetry_error_m=abs(y_left+y_right), spread_m=float(np.ptp(gaps)),
                left_error_m=y_left+requested_width/2,right_error_m=y_right-requested_width/2,
                gap_error_m=(y_right-y_left)-requested_width)
    if not all(np.isfinite(pts).all() for pts in points):
        return data, False, False
    nonnegative = float(gaps.min()) >= -MATRIX_ZERO_M
    good = (abs(data['gap_error_m']) <= .001 and abs(data['midline_m']) <= .0005
        and abs(data['left_error_m']) <= .0005 and abs(data['right_error_m']) <= .0005
        and data['abs_symmetry_error_m'] <= .001 and data['spread_m'] <= .0005 and nonnegative)
    return data, good, nonnegative


def capture(p,cid,body,g_idx,directory,name):
    st=p.getLinkState(body,g_idx,computeForwardKinematics=1,physicsClientId=cid)
    view=p.computeViewMatrixFromYawPitchRoll(st[4],.35,46,-25,0,2)
    proj=p.computeProjectionMatrixFOV(47.,1.4,.01,3.)
    rgb=p.getCameraImage(560,400,viewMatrix=view,projectionMatrix=proj,
                         renderer=p.ER_TINY_RENDERER,physicsClientId=cid)[2]
    save_rgb(directory/name,560,400,np.asarray(rgb,dtype=np.uint8).tobytes())


def pose_drift(initial, current):
    # Relative transform initial-TCP <- current-TCP. Rotation magnitude via atan2
    # avoids acos sensitivity at rest; near-pi rotations give the principal angle.
    relative=np.eye(4)
    r_initial=initial[:3,:3]
    r=relative[:3,:3]=r_initial.T @ current[:3,:3]
    t=relative[:3,3]=r_initial.T @ (current[:3,3]-initial[:3,3])
    skew=[r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]]
    angle=math.degrees(math.atan2(float(np.linalg.norm(skew)),float(np.trace(r))-1.))
    return dict(initial_T_current=relative.tolist(),translation_m=float(np.linalg.norm(t)),rotation_deg=angle)


def serial_frames(frames):
    return {name:pose.tolist() for name,pose in frames.items()}





def log_motor_commands(p,cid,body,arm,jpr,qarm,qfinger,arm_limits,*,phase,reference_speed,motor_max_velocity):
    validate_speeds(phase,reference_speed,motor_max_velocity)
    # UR5e and both sliders have the same finite setters as registered in smoke07.
    p.setJointMotorControlArray(body,arm,p.POSITION_CONTROL,targetPositions=qarm.tolist(),
        forces=arm_limits,positionGains=[.08]*6,velocityGains=[1.]*6,physicsClientId=cid)
    for j in jpr:
        p.setJointMotorControl2(body,j,p.POSITION_CONTROL,targetPosition=qfinger,targetVelocity=0.,
            positionGain=.10,velocityGain=1.,force=LIMIT,maxVelocity=motor_max_velocity,physicsClientId=cid)


def arm_checks(states,qref,limits,efforts,hold_q=None):
    result=[]
    for i,(state,ref,lim,eff) in enumerate(zip(states,qref,limits,efforts)):
        q,dq,wrench,tau=state
        if not all(math.isfinite(x) for x in (q,dq,*wrench,tau)):result.append(f'arm_{i}.nonfinite')
        if not lim[0]<=q<=lim[1]:result.append(f'arm_{i}.joint_range')
        if abs(dq)>lim[2]:result.append(f'arm_{i}.joint_speed_limit')
        if abs(tau)>=eff:result.append(f'arm_{i}.effort_saturated')
        if hold_q is not None:
            if abs(q-hold_q[i])>.01:result.append(f'arm_{i}.hold_q')
            if abs(dq)>.01:result.append(f'arm_{i}.hold_dq')
    return result





def get_object_state(p,cid,body):
    xyz,xyzw=p.getBasePositionAndOrientation(body,physicsClientId=cid)
    linear,angular=p.getBaseVelocity(body,physicsClientId=cid)
    return dict(body_id=body,position=xyz,orientation=xyzw,linear_velocity=linear,angular_velocity=angular)


def classify_outcome(kind, completed, engineering_passed, grasp_success):
    if kind not in ('nominal', 'miss'):
        raise ValueError('unregistered slot kind')
    return completed is True and engineering_passed is True and grasp_success is (kind == 'nominal')


def run_slot(p,client_id,body,slot,directory,*,evaluator_context,before_first_step):
    directory=Path(directory).resolve()
    if (slot.get('slot')!=SLOT or slot.get('candidate_id')!=read(PLAN)['candidate_id'] or slot.get('kind')!=KIND
            or slot.get('object_id')!=OBJECT_ID or slot.get('not_p3') is not True
            or directory!=SLOT_DIR/'episodes'/f'{SLOT:02d}' or not directory.is_dir() or not callable(before_first_step)):
        raise ValueError('requires prepared registered bridge slot and durable first-step callback')
    log=Journal(directory)
    summary=dict(slot=SLOT,kind=KIND,expected_grasp_success=KIND=='nominal',candidate_id=slot['candidate_id'],object_id=OBJECT_ID,not_p3=True,
                 episode_completed=False,engineering_gate_passed=False,grasp_success=False,slot_passed=False,
                 failure_stage=None,failure_reason=None,first_step_charge_completed=False,
                 phases_completed=[],step_counts={},violations=[],opening_tail_pass=0,
                 near_cap_step_counts={'left':0,'right':0},screenshot_errors=[])
    palm_idx=None
    terminal_rows=[]
    ev=None
    phase='pre_physics_setup'
    try:
        plan=read(PLAN);gate=read(SLOT_DIR/'zero_readback.json')
        if (not gate['passed'] or gate['trajectory_plan_sha256']!=sha(PLAN)
                or gate['object_body_id']!=evaluator_context['object_body_id']
                or gate['table_body_id']!=evaluator_context['table_body_id']
                or tuple(PHASE_STEPS)!=PHASES):
            raise ValueError('runtime inputs differ from admitted zero-step world')
        infos=[p.getJointInfo(body,j,physicsClientId=client_id) for j in range(p.getNumJoints(body,physicsClientId=client_id))]
        links={row[12].decode():row[0] for row in infos}
        joint_ids={row[1].decode():row[0] for row in infos}
        names=read(ROOT/'data/simulation_v52/robot/robot_assets.json')['arm_joint_names']
        arm=[joint_ids[n] for n in names];jpr=[links[n] for n in BONE]
        palm_idx=links[PALM]
        efforts=read(ROOT/'configs/simulation_v52r2.json')['physics']['arm_forces']
        qhome=np.asarray(plan['home_joints'])
        corners=[corner_template(p,body,links[n],s,client_id) for n,s in zip(PAD,SIDES)]
        previous=current_frames(p,body,links,client_id)
        initial_G=previous[PALM].copy()
        states=p.getJointStates(body,arm+jpr,physicsClientId=client_id)
        object_id=evaluator_context['object_body_id'];table_id=evaluator_context['table_body_id']
        ev=ObjectEvaluator(body,object_id,table_id,links,object_mass_kg=evaluator_context['object_mass_kg'],gravity_world_m_s2=evaluator_context['gravity_world_m_s2'])
        initial_object=get_object_state(p,client_id,object_id)
        log.record('initial_configuration',body=body,evaluator_context=evaluator_context,
            initial_object_state=initial_object,initial_W_T_link=serial_frames(previous),
            initial_joint_states=states,joint_indices=arm+jpr,links=links,
            arm_effort_caps_Nm=efforts,dt_s=DT,trajectory_plan_sha256=sha(PLAN),
            control_provenance='only the candidate-based frozen reference and tick index',
            object_state_use='independent post-return evaluation',
            observation_time='solver rows reference collision state; instantaneous closest and FK use returned state')
        if len(states)!=8 or any(abs(v[0]-ref)>1e-12 or v[1]!=0. for v,ref in zip(states,[*qhome,0.,0.])):
            raise ValueError('unexpected initial robot state')
        capture(p,client_id,body,palm_idx,directory,'initial.png')
        arrived_G=arrived_q=lift_G=lift_q=None
        stall_age=[0,0]
        for phase,count in PHASES:
            summary['step_counts'][phase]=0
            for k in range(1,count+1):
                if time.monotonic()-log.first_wall>=120. or log.steps_started>=7200:
                    raise RuntimeError('frozen wall/simulation timeout')
                qarm,dqref,qfinger=reference(plan,phase,k)
                command_speed=phase_speed(phase)
                command=dict(phase=phase,phase_tick=k,arm_target_rad=qarm.tolist(),
                    arm_reference_slope_rad_s=dqref.tolist(),finger_target_m=qfinger,
                    arm_mode='POSITION_CONTROL',arm_position_gain=.08,arm_velocity_gain=1.,
                    arm_force_cap_Nm=efforts,finger_mode='POSITION_CONTROL',finger_position_gain=.1,
                    finger_velocity_gain=1.,finger_target_velocity=0.,single_force_cap_N=20.,
                    reference_speed_m_s=command_speed,motor_max_velocity_m_s=command_speed,
                    observed_velocity_limit_m_s=FINGER_OBSERVED_HARD_LIMIT_M_S)
                log.record('setter_intent',command=command)
                log_motor_commands(p,client_id,body,arm,jpr,qarm,qfinger,efforts,phase=phase,
                                   reference_speed=command_speed,motor_max_velocity=command_speed)
                if log.steps_started==0:
                    receipt=before_first_step()
                    log.record('first_step_reservation',receipt=receipt)
                log.intent(phase,command)
                p.stepSimulation(physicsClientId=client_id)
                if log.steps_started == 1:
                    summary['first_step_charge_completed'] = True
                    log.record('first_native_call_charged', receipt_path='native_call_observed.json')
                log.returned()
                summary['step_counts'][phase]+=1
                states=p.getJointStates(body,arm+jpr,physicsClientId=client_id)
                frames=current_frames(p,body,links,client_id)
                raw=p.getContactPoints(physicsClientId=client_id)
                fingers=states[-2:]
                widths,within,nonnegative=surfaces(frames,corners,2*qfinger)
                truth=get_object_state(p,client_id,object_id)
                structural=audit_filtered(p,body,links,client_id,raw,current_frames_w=frames,
                                             q_by_side={side:st[0] for side,st in zip(SIDES,fingers)})
                active=audit_seams(p,body,links,client_id,raw,require_pad_witness=min(widths['gaps_m'])<=.002,
                                  manifold_frames=previous,current_frames_w=frames,dt=DT)
                base_table=capture_base_table(p,client_id,body,table_id,links,stage='dynamic',raw_manifolds=raw)
                violations=structural['violations']+active['violations']+base_table['violations']
                if not all(np.isfinite(pose).all() for pose in frames.values()):
                    violations.append('nonfinite_returned_link_pose')
                if phase in ('settle','opening'):
                    hold_q,hold_G=qhome,initial_G
                elif phase=='close':
                    hold_q,hold_G=arrived_q,arrived_G
                elif phase=='hold':
                    hold_q,hold_G=lift_q,lift_G
                else:
                    hold_q=hold_G=None
                violations+=arm_checks(states[:6],qarm,plan['limits_rad_rad_s'],efforts,hold_q)
                reference_G=fk(qarm)
                tracking=pose_drift(reference_G,frames[PALM])
                holding=pose_drift(hold_G,frames[PALM]) if hold_G is not None else None
                if holding and not (holding['translation_m']<=.005 and holding['rotation_deg']<=3.):
                    violations.append('hold_tcp_drift')
                approach_goal=None
                if phase=='approach' and k==count:
                    approach_goal=pose_drift(np.asarray(plan['W_T_G_goal']),frames[PALM])
                    if not (approach_goal['translation_m']<=.005 and approach_goal['rotation_deg']<=3.):
                        violations.append('executed_approach_tracking')
                    arrived_G=frames[PALM].copy();arrived_q=[x[0] for x in states[:6]]
                if phase=='lift' and k==count:
                    lift_G=frames[PALM].copy();lift_q=[x[0] for x in states[:6]]
                if any(not -1e-5<=st[0]<=.07001 for st in fingers):
                    violations.append('measured_slider_limit')
                if not nonnegative:violations.append('nonfinite_or_negative_pad_corners')
                if widths['spread_m']>.0005:violations.append('pad_parallelism')
                if phase in ('settle','opening','transit','pregrasp','approach'):
                    if abs(widths['midline_m'])>.0005 or widths['abs_symmetry_error_m']>.001:
                        violations.append('unloaded_midline_or_symmetry')
                    if phase in ('transit','pregrasp','approach') and not within:
                        violations.append('preopening_not_maintained')
                    if phase=='opening' and k>count-24:
                        if within:summary['opening_tail_pass']+=1
                        else:violations.append('opening_stability')
                decision=ev.observe(log.steps_returned,phase,raw,fingers,truth,frames[PALM][:3,3].tolist(),
                                    engineering_violations=violations)
                violations=decision['violations']
                if 4249<=log.steps_returned<=4728:
                    terminal_rows.append(dict(step=log.steps_returned,phase=phase,hold_tick=k,
                        object_state=truth,lift_m=decision['lift_m'],tcp_world=frames[PALM].tolist(),
                        effective_left_target_contact=decision['effective_left_target_contact'],
                        effective_right_target_contact=decision['effective_right_target_contact'],
                        effective_table_support=decision['table_support'],
                        **{f:decision[f] for f in ('table_contact_present','table_normal_force_sum_N','table_load_fraction','aggregate_table_support')},slipped=decision['slipped'],
                        eligible=decision['eligible_hold_step'],raw_manifolds=raw,
                        contacts=decision['contacts'],engineering_violations=violations))
                for item in decision['near_cap']:
                    summary['near_cap_step_counts'][item['side']]+=1
                for i,st in enumerate(fingers):
                    stalled=phase in ('close','lift','hold') and st[0]>.0005 and st[0]-qfinger>.0005 and abs(st[1])<.001
                    stall_age[i]=stall_age[i]+1 if stalled else 0
                log.record('state',index=log.steps_returned,phase=phase,phase_tick=k,command=command,
                    joint_indices=arm+jpr,actual_arm_and_finger_states=states,
                    arm_q_rad=[s[0] for s in states[:6]],arm_dq_rad_s=[s[1] for s in states[:6]],
                    arm_torque_Nm=[s[3] for s in states[:6]],
                    arm_error_to_command_rad=[s[0]-v for s,v in zip(states,qarm.tolist())],
                    arm_velocity_error_rad_s=[s[1]-v for s,v in zip(states,dqref.tolist())],
                    measured_q_m=[s[0] for s in fingers],actual_speed_m_s=[s[1] for s in fingers],
                    sim_motor_force_N=[s[3] for s in fingers],commanded_q_m=qfinger,geometry=widths,
                    within_unloaded_geometry=within,stall_age=stall_age,tcp_W_T_goal_command=reference_G.tolist(),
                    W_T_link_post=serial_frames(frames),manifold_frame_history_step=log.steps_returned-1,
                    tcp_command_tracking=tracking,tcp_hold_drift=holding,approach_goal_error=approach_goal,
                    object_state=truth,object_relative_G=(np.linalg.inv(frames[PALM]) @ transform(p,truth['position'],truth['orientation'])).tolist(),
                    raw_manifolds=raw,object_evaluation=decision,
                    filtered_pairs_audit=structural,seam_audit=active,base_table_mount_audit=base_table,violations=violations)
                if violations:
                    summary['violations']=violations
                    raise RuntimeError(f'earliest_violation_at_step_{log.steps_returned}:{violations!r}')
                previous=frames
            summary['phases_completed'].append(phase)
        outcome=ev.finish()
        complete=(log.steps_started==log.steps_returned==4728 and p.returns['stepSimulation']==4728
            and summary['opening_tail_pass']==24 and summary['phases_completed']==[name for name,_ in PHASES])
        if not complete:raise RuntimeError('incomplete scheduled physical-return evidence')
        summary.update(outcome)
        summary['episode_completed']=True
        summary['engineering_gate_passed']=outcome['engineering_gate_passed'] and not summary['violations']
        summary['slot_passed']=classify_outcome(KIND,summary['episode_completed'],summary['engineering_gate_passed'],outcome['grasp_success'])
        summary['outcome_class'] = ('nominal_success' if outcome['grasp_success'] else 'nominal_grasp_failure') if KIND=='nominal' else (
            'miss_false_success' if outcome['grasp_success'] else 'expected_miss_detected')
        summary['grasp_failure_reasons'] = list(outcome['failure_reason'])
        if KIND == 'miss':
            summary['failure_reason'] = ['misaligned_negative_grasped'] if outcome['grasp_success'] else []
            summary['failure_stage'] = 'hold' if outcome['grasp_success'] else None
    except Exception as exc:
        summary.update(episode_completed=log.steps_returned==4728,engineering_gate_passed=False,outcome_class='engineering_failure',
                       grasp_success=False,slot_passed=False,failure_stage=phase,
                       failure_reason=repr(exc),traceback=traceback.format_exc())
        log.record('stop',reason=summary['failure_reason'])
    finally:
        if ev is not None:
            summary['evaluator_terminal_result']=ev.finish()
        if palm_idx is not None:
            try:capture(p,client_id,body,palm_idx,directory,'terminal.png')
            except Exception as exc:summary['screenshot_errors'].append(repr(exc))
        interval_path=directory/'final_480_steps.json'
        with interval_path.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(safe_values(dict(required_interval=[4249,4728],actual_observed=len(terminal_rows),
                    steps_returned=log.steps_returned,observed_rows=terminal_rows)),
                stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            stream.write('\n')
        summary['terminal_observations_file']=interval_path.name
        summary['terminal_observations_sha256']=sha(interval_path)
        summary['terminal_observation_count']=len(terminal_rows)
        log.finish(**summary)
    return read(directory/'result.json')

