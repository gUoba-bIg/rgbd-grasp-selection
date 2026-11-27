"""Corrective slot 13. Fixed commands use only the registered candidate, offline plan and clock."""
import math
import time
import traceback
from pathlib import Path
import numpy as np

from episode_log import Journal, save_rgb
from structural_filter import audit_filtered, audit_seams, current_frames
from stop_evaluator import StopEvaluator, PHASE_STEPS
from speed_policy import FINGER_OBSERVED_HARD_LIMIT_M_S, phase_speed, validate_speeds
from trajectory import ID, PLAN, PHASES, DT, fk, read, reference, sha

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


def run_slot(p,client_id,body,slot,directory,*,fixture_body):
    directory=Path(directory).resolve()
    if (slot.get('slot')!=13 or slot.get('candidate_id')!=ID or slot.get('kind')!='single_sided_stop_corrective'
            or slot.get('execution_order')!=7 or slot.get('repeat_of_slot')!=6
            or slot.get('development_corrective_repeat') is not True or slot.get('independent_confirmation') is not False
            or directory!=HERE/'episodes'/'13' or not (directory/'precharge.json').is_file()):
        raise ValueError('requires exclusive slot 13 precharge')
    log=Journal(directory)
    summary=dict(slot=13,candidate_id=ID,execution_order=7,repeat_of_slot=6,
                 kind='single_sided_stop_corrective',development_corrective_repeat=True,independent_confirmation=False,slot_passed=False,expected_negative_detected=False,grasp_success=False,
                 phases_completed=[],step_counts={},violations=[],opening_tail_pass=0,near_cap_step_counts={'left':0,'right':0},
                 screenshot_errors=[])
    palm_idx=None
    ev=None
    phase=None
    try:
        plan=read(PLAN)
        gate=read(HERE/'zero_readback.json')
        if (not gate['passed'] or gate['fixture_body_id']!=fixture_body
                or gate['trajectory_plan_sha256']!=sha(PLAN) or tuple(PHASE_STEPS)!=PHASES):
            raise ValueError('runtime inputs differ from admitted zero-step world')
        infos=[p.getJointInfo(body,j,physicsClientId=client_id) for j in range(p.getNumJoints(body,physicsClientId=client_id))]
        links={row[12].decode():row[0] for row in infos}
        joint_ids={row[1].decode():row[0] for row in infos}
        names=read(ROOT/'data/simulation_v52/robot/robot_assets.json')['arm_joint_names']
        arm=[joint_ids[n] for n in names]
        jpr=[links[n] for n in BONE]
        palm_idx=links[PALM]
        efforts=read(ROOT/'configs/simulation_v52r2.json')['physics']['arm_forces']
        qhome=np.asarray(plan['home_joints'])
        corners=[corner_template(p,body,links[n],s,client_id) for n,s in zip(PAD,SIDES)]
        prev_frames=current_frames(p,body,links,client_id)
        init_G=prev_frames[PALM].copy()
        initial=p.getJointStates(body,arm+jpr,physicsClientId=client_id)
        log.record('initial_configuration',body=body,fixture_evaluator_id=fixture_body,joint_indices=arm+jpr,
            links=links,initial_states=initial,initial_W_T_link=serial_frames(prev_frames),arm_effort_caps_Nm=efforts,
            dt_s=DT,trajectory_plan_sha256=sha(PLAN),candidate_id=ID,
            observation_time='raw solver: just-returned integration; manifold collision frame: prior state; closest and FK: current state',
            reported_wrench='PyBullet joint reactions in returned raw state; physical source is simulation')
        if len(initial)!=8 or any(abs(v[0]-ref)>1e-12 or v[1]!=0. for v,ref in zip(initial,[*qhome,0.,0.])):
            raise ValueError('unexpected initial state after zero gate')
        ev=StopEvaluator(body,fixture_body,links)
        capture(p,client_id,body,palm_idx,directory,'initial.png')
        arrived_G=arrived_q=None
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
                             finger_velocity_gain=1.,finger_target_velocity=0.,single_force_cap_N=20.,single_speed_cap_m_s=command_speed,
                             reference_speed_m_s=command_speed,motor_max_velocity_m_s=command_speed,
                             observed_velocity_limit_m_s=FINGER_OBSERVED_HARD_LIMIT_M_S)
                log.record('setter_intent',command=command)
                log_motor_commands(p,client_id,body,arm,jpr,qarm,qfinger,efforts,
                                   phase=phase,reference_speed=command_speed,motor_max_velocity=command_speed)
                log.intent(phase,command)
                p.stepSimulation(physicsClientId=client_id)
                log.returned()
                summary['step_counts'][phase]+=1
                states=p.getJointStates(body,arm+jpr,physicsClientId=client_id)
                frames=current_frames(p,body,links,client_id)
                raw=p.getContactPoints(physicsClientId=client_id)
                fingers=states[-2:]
                widths,within,nonnegative=surfaces(frames,corners,2*qfinger)
                right=ev.observe(log.steps_returned,phase,raw,fingers)
                structural=audit_filtered(p,body,links,client_id,raw,current_frames_w=frames,
                                             q_by_side={side:st[0] for side,st in zip(SIDES,fingers)})
                active=audit_seams(p,body,links,client_id,raw,
                    require_pad_witness=min(widths['gaps_m'])<=.002,manifold_frames=prev_frames,
                    current_frames_w=frames,dt=DT)
                violations=list(right['violations'])+structural['violations']+active['violations']
                hold_q=qhome if phase in ('settle','opening') else arrived_q if phase=='close' else None
                violations+=arm_checks(states[:6],qarm,plan['limits_rad_rad_s'],efforts,hold_q)
                pose_tracking=pose_drift(fk(qarm),frames[PALM])
                hold_G=init_G if phase in ('settle','opening') else arrived_G if phase=='close' else None
                holding=pose_drift(hold_G,frames[PALM]) if hold_G is not None else None
                if holding and not (holding['translation_m']<=.005 and holding['rotation_deg']<=3.):
                    violations.append('hold_tcp_drift')
                approach_goal=None
                if phase=='approach' and k==count:
                    approach_goal=pose_drift(np.asarray(plan['W_T_G_goal']),frames[PALM])
                    if not (approach_goal['translation_m']<=.005 and approach_goal['rotation_deg']<=3.):
                        violations.append('executed_approach_tracking')
                    arrived_G=frames[PALM].copy()
                    arrived_q=[x[0] for x in states[:6]]
                if any(not -1e-5<=st[0]<=.07001 for st in fingers):violations.append('measured_slider_limit')
                if not nonnegative:violations.append('nonfinite_or_negative_pad_corners')
                if widths['spread_m']>.0005:violations.append('pad_parallelism')
                # Midline/gap tracking belongs to unloaded geometry; one-sided closure
                # necessarily breaks symmetry. Its raw measurements are still logged.
                if phase!='close':
                    if abs(widths['midline_m'])>.0005 or widths['abs_symmetry_error_m']>.001:
                        violations.append('unloaded_midline_or_symmetry')
                    if phase in ('transit','pregrasp','approach') and not within:
                        violations.append('preopening_not_maintained')
                    if phase=='opening' and k>count-24:
                        if within:summary['opening_tail_pass']+=1
                        else:violations.append('opening_stability')
                for item in right['near_cap']:
                    summary['near_cap_step_counts'][item['side']]+=1
                for i,st in enumerate(fingers):
                    stalled=phase=='close' and st[0]>.0005 and st[0]-qfinger>.0005 and abs(st[1])<.001
                    stall_age[i]=stall_age[i]+1 if stalled else 0
                log.record('state',index=log.steps_returned,phase=phase,phase_tick=k,command=command,
                    joint_indices=arm+jpr,actual_arm_and_finger_states=states,
                    arm_q_rad=[s[0] for s in states[:6]],arm_dq_rad_s=[s[1] for s in states[:6]],
                    arm_torque_Nm=[s[3] for s in states[:6]],arm_error_to_command_rad=[s[0]-v for s,v in zip(states,qarm.tolist())],
                    arm_velocity_error_rad_s=[s[1]-v for s,v in zip(states,dqref.tolist())],
                    measured_q_m=[s[0] for s in fingers],actual_speed_m_s=[s[1] for s in fingers],
                    sim_motor_force_N=[s[3] for s in fingers],commanded_q_m=qfinger,geometry=widths,within_unloaded_geometry=within,
                    stall_age=stall_age,tcp_W_T_goal_command=fk(qarm).tolist(),W_T_link_post=serial_frames(frames),
                    manifold_frame_history_step=log.steps_returned-1,tcp_command_tracking=pose_tracking,
                    tcp_hold_drift=holding,approach_goal_error=approach_goal,raw_manifolds=raw,
                    contact_evaluation=right,filtered_pairs_audit=structural,seam_audit=active,violations=violations)
                if violations:
                    summary['violations']=violations
                    raise RuntimeError(f'earliest_violation_at_step_{log.steps_returned}:{violations!r}')
                prev_frames=frames
            summary['phases_completed'].append(phase)
        outcome=ev.finish()
        complete=(log.steps_started==log.steps_returned==3216 and p.returns['stepSimulation']==3216
            and summary['opening_tail_pass']==24 and summary['phases_completed']==[name for name,_ in PHASES])
        if not (complete and outcome['slot_passed']):
            summary['violations']=outcome.get('violations') or ['terminal_right_contact_requirement_or_phase_budget']
            raise RuntimeError('incomplete_terminal_acceptance')
        summary.update(slot_passed=True,expected_negative_detected=True,reason='registered_right_pad_non_target_terminal_contact')
    except Exception as exc:
        summary.update(reason=repr(exc),traceback=traceback.format_exc())
        log.record('stop',reason=summary['reason'])
    finally:
        if ev is not None:
            summary['classifier_terminal_result']=ev.finish()
        if palm_idx is not None:
            try:capture(p,client_id,body,palm_idx,directory,'terminal.png')
            except Exception as exc:summary['screenshot_errors'].append(repr(exc))
        log.finish(**summary)
    return read(directory/'result.json')

