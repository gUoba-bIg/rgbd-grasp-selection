"""Smoke05 empty-stroke slot 3. The caller supplies its gated, filtered connection.
Every command and observed encoder, separation, witness and contact remains unsnapped.
"""
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from episode_log import Journal, save_rgb
from structural_filter import STRUCTURAL_PAIRS, INTERNAL_SEAMS, audit_filtered, audit_seams, current_frames

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RATE = 240
DT = 1 / RATE
SPEED = .040
LIMIT = 20.
MATRIX_ZERO_M = 1.0e-9
SIDES = ('left', 'right')
PAD = tuple(f'ip140_{s}_pad' for s in SIDES)
BONE = tuple(f'ip140_{s}_bone' for s in SIDES)
PALM = 'ip140_palm'
STATIONS = (0, 35, 70, 105, 140, 105, 70, 35, 0)
RAIL_MIN_M = -1.e-5
RAIL_MAX_M = .07001
SLOT_ID = 'IP140-BENCH-EMPTY-03:post-fixed-seam-filter'
ARM_MAX_POSITION_ERROR_RAD = .010
ARM_MAX_SPEED_RAD_S = .010
TCP_MAX_DRIFT_M = .005
TCP_MAX_DRIFT_DEG = 3.


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


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


def scan_contacts(p, cid, body, by_name, by_index, require_pad_witness, frames_pre, frames_post):
    manifold = p.getContactPoints(physicsClientId=cid)
    checks = {}
    violations = []
    for name, audit, options in (
            ('filtered_structural_pairs', audit_filtered, {'current_frames_w':frames_post}),
            ('internal_seams', audit_seams, {'require_pad_witness':require_pad_witness,
             'manifold_frames':frames_pre, 'current_frames_w':frames_post, 'dt':DT})):
        try:
            result = audit(p, body, by_name, cid, manifold, **options)
        except Exception as exc:
            result = {'passed':False, 'violations':['audit_exception'], 'error':repr(exc)}
        checks[name] = result
        if result.get('passed') is not True:
            violations += [name+'.'+str(reason) for reason in result.get('violations', [])] or [name+'.failed']

    registered = {frozenset(pair) for pair in (*STRUCTURAL_PAIRS, *INTERNAL_SEAMS)}
    impulses = []
    for i, pt in enumerate(manifold):
        a = by_index.get(pt[3], str(pt[3])) if pt[1]==body else f'body{pt[1]}_link{pt[3]}'
        b = by_index.get(pt[4], str(pt[4])) if pt[2]==body else f'body{pt[2]}_link{pt[4]}'
        d, fn, ft1, ft2 = pt[8], pt[9], pt[10], pt[12]
        impulses.append(dict(contact=i,normal_Ns=fn*DT, lateral_1_signed_Ns=ft1*DT,
                             lateral_2_signed_Ns=ft2*DT))
        if not all(math.isfinite(v) for v in (*pt[5],*pt[6],*pt[7],d,fn,ft1,ft2,*pt[11],*pt[13])):
            violations.append(f'nonfinite_manifold:{a}|{b}')
        elif pt[1]==body and pt[2]==body:
            if frozenset((a,b)) not in registered:
                violations.append(f'unregistered_self_contact:{a}|{b}')
        elif d < -.001 or fn > .2:
            violations.append(f'forbidden_external_contact:{a}|{b}')
    return manifold, impulses, checks, violations


def run_slot(p,client_id,body,slot,directory):
    """Only after charge_slot. Persist every attempt outcome, including exceptions."""
    directory=Path(directory).resolve()
    number=slot['slot']
    if (number!=3 or slot['candidate_id']!=SLOT_ID or slot['kind']!='empty_stroke'
            or tuple(slot['opening_sequence_mm'])!=STATIONS):
        raise ValueError('only the smoke05 frozen empty slot 3 is permitted')
    if directory!=HERE/'episodes'/f'{number:02d}' or not (directory/'precharge.json').is_file():
        raise ValueError('exclusive slot precharge required')
    if slot['seconds_each']!=2.0 or slot['tail_s']!=.1 or slot['q_initial_m']!=[0,0]:
        raise ValueError("slot timing/initial state is not the frozen empty stroke")
    log=Journal(directory)
    summary=dict(slot=number,slot_passed=False,reason=None,violations=[],
                 attempt_kind='empty_stroke',tested_stations_mm=[],screenshot_errors=[])
    idx_g=None
    finite_measurement_count=0
    maximum_speed=0.
    maximum_force=0.
    try:
        info=[p.getJointInfo(body,j,physicsClientId=client_id) for j in range(p.getNumJoints(body,physicsClientId=client_id))]
        joint_by_name={row[1].decode():row[0] for row in info}
        by_name={row[12].decode():row[0] for row in info}
        by_index={v:k for k,v in by_name.items()}
        by_index[-1]='BASE'
        idx_g=by_name[PALM]
        jpr=[by_name[name] for name in BONE]
        pad_idx=[by_name[name] for name in PAD]
        cfg=read(ROOT/'configs/simulation_v52r2.json')
        historic=read(ROOT/'data/simulation_v52/robot/robot_assets.json')
        arm=[joint_by_name[n] for n in historic['arm_joint_names']]
        home=cfg['home_joints']
        arm_limits=cfg['physics']['arm_forces']
        if len(arm_limits)!=len(arm) or any(not math.isfinite(v) or v<=0. for v in arm_limits):
            raise RuntimeError('invalid frozen arm effort limits')
        raw_initial=p.getJointStates(body,arm+jpr,physicsClientId=client_id)
        frames_pre=current_frames(p,body,by_name,client_id)
        initial_frames={name:frame.copy() for name,frame in frames_pre.items()}
        log.record('initial_configuration',body=body,links=by_index,joints=arm+jpr,
                   candidate_id=SLOT_ID,initial_W_T_link=serial_frames(initial_frames),tcp_link=PALM,
                   states=raw_initial,arm_home=home,arm_joint_names=historic['arm_joint_names'],
                   arm_effort_cap_Nm=arm_limits,steps_frequency_Hz=RATE,step_seconds=DT,
                   contact_impulse_convention='reported signed component force x fixed step duration',
                   stations_mm=STATIONS)
        if any(abs(v[0]-t)>1e-12 or v[1]!=0 for v,t in zip(raw_initial,[*home,0.,0.])):
            raise RuntimeError('initial encoder state inconsistent with frozen bench')
        log.record('arm_hold_command',joints=arm,target=home,forces=cfg['physics']['arm_forces'],
                   control_mode='POSITION_CONTROL',kp=.08,kd=1.)
        p.setJointMotorControlArray(body,arm,p.POSITION_CONTROL,targetPositions=home,
            forces=cfg['physics']['arm_forces'],positionGains=[.08]*6,
            velocityGains=[1.]*6,physicsClientId=client_id)
        corners=[corner_template(p,body,j,s,client_id) for j,s in zip(pad_idx,SIDES)]
        capture(p,client_id,body,idx_g,directory,'initial.png')
        previous_width=0.
        for station_mm in STATIONS:
            width=station_mm/1000.
            delta=(width-previous_width)/2
            for k in range(1,481):
                if time.monotonic()-log.first_wall>=120. or log.steps_started>=30*RATE:
                    if time.monotonic()-log.first_wall>=120.:
                        raise RuntimeError('hard_time_limit_wall_120_s')
                    raise RuntimeError('hard_time_limit_sim_30_s')
                travel=SPEED*k*DT
                q=width/2 if travel>=abs(delta) else previous_width/2+math.copysign(travel,delta)
                if not (0. <= q <= .070):
                    raise RuntimeError('out_of_range_setpoint_before_setter')
                command=dict(mode='POSITION_CONTROL',target_q_m=q,target_velocity=0.,Kp=.1,Kd=1.,
                             max_force_N_per_finger=LIMIT,max_speed_m_s_per_finger=SPEED,
                             station_width_mm=station_mm,station_tick=k)
                log.record("setter_intent",command=command,joints=jpr)
                for j in jpr:
                    p.setJointMotorControl2(body,j,p.POSITION_CONTROL,targetPosition=q,targetVelocity=0.,
                        positionGain=.1,velocityGain=1.,force=LIMIT,maxVelocity=SPEED,physicsClientId=client_id)
                log.record("setter_pair_returned",joints=jpr)
                log.intent('empty_stroke',command)
                p.stepSimulation(physicsClientId=client_id)
                log.returned()
                states=p.getJointStates(body,arm+jpr,physicsClientId=client_id)
                frames_post=current_frames(p,body,by_name,client_id)
                faces,within,nonnegative=surfaces(frames_post,corners,width)
                near_closed=(q==0. and max(abs(v) for v in faces['gaps_m'])<=.001)
                manifold,impulses,checks,violations=scan_contacts(p,client_id,body,by_name,
                    by_index,require_pad_witness=near_closed or log.steps_returned==1,
                    frames_pre=frames_pre,frames_post=frames_post)
                drift=pose_drift(initial_frames[PALM],frames_post[PALM])
                other_tool_drift=(pose_drift(initial_frames['tool0'],frames_post['tool0'])
                                  if 'tool0' in initial_frames else None)
                if not (drift['translation_m']<=TCP_MAX_DRIFT_M and drift['rotation_deg']<=TCP_MAX_DRIFT_DEG):
                    violations.append('tcp_pose_drift')
                arm_error=[st[0]-goal for st,goal in zip(states,home)]
                arm_torque=[st[3] for st in states[:len(arm)]]
                for name,st,err,cap in zip(historic['arm_joint_names'],states[:len(arm)],arm_error,arm_limits):
                    if not all(math.isfinite(v) for v in (st[0],st[1],*st[2],st[3])):
                        violations.append(name+'.nonfinite_arm_feedback')
                    if abs(err)>ARM_MAX_POSITION_ERROR_RAD:
                        violations.append(name+'.arm_home_position_error')
                    if abs(st[1])>ARM_MAX_SPEED_RAD_S:
                        violations.append(name+'.arm_speed')
                    if abs(st[3])>=cap:
                        violations.append(name+'.arm_motor_saturation')
                fingers=states[-2:]
                for side,state in zip(SIDES,fingers):
                    pos,speed,_,force=state
                    if not all(math.isfinite(v) for v in (pos,speed,force)):
                        violations.append(side+'.nonfinite')
                    else:
                        finite_measurement_count+=1
                        maximum_speed=max(maximum_speed,abs(speed))
                        maximum_force=max(maximum_force,abs(force))
                        if abs(speed)>SPEED:violations.append(side+'.actual_speed_above_0.04_m_s')
                        if abs(force)>LIMIT:violations.append(side+'.reported_force_above_20_N')
                        if not RAIL_MIN_M<=pos<=RAIL_MAX_M:
                            violations.append(side+'.outside_measured_rail_band')
                outside_nominal=any(not 0.<=st[0]<=.070 for st in fingers)
                left_y=faces['left_error_m']-width/2
                right_y=faces['right_error_m']+width/2
                band_tracking=(abs(faces['gap_m']-2*q)<=.001 and abs(left_y+q)<=.0005
                               and abs(right_y-q)<=.0005)
                if outside_nominal and not band_tracking:
                    violations.append('rail_band_face_tracking')
                if (abs(faces['midline_m'])>.0005 or faces['abs_symmetry_error_m']>.001
                        or faces['spread_m']>.0005):
                    violations.append('midline_symmetry_or_parallelism')
                if not nonnegative:violations.append('nonfinite_faces_or_negative_geometry')
                if k>456 and not within:violations.append('stable_window_geometry')
                log.record('state',index=log.steps_returned,station_width_mm=station_mm,station_tick=k,
                    actual_arm_and_finger_states=states,joints=arm+jpr,sim_motor_force_N=[s[3] for s in fingers],
                    geometry=faces,within_frozen_tolerance=within,in_settled_window=k>456,
                    measured_q_m=[st[0] for st in fingers],commanded_q_m=q,
                    measured_outside_nominal=outside_nominal,dynamic_face_tracking_in_rail_band=band_tracking,
                    q_arm_rad=[st[0] for st in states[:len(arm)]],dq_arm_rad_s=[st[1] for st in states[:len(arm)]],
                    arm_error_from_home_rad=arm_error,arm_torque_Nm=arm_torque,arm_cap_Nm=arm_limits,
                    W_T_link_post=serial_frames(frames_post),manifold_frame_history_step=log.steps_returned-1,
                    tcp_G_drift=drift,tool0_drift=other_tool_drift,raw_manifolds=manifold,
                    contact_impulses_Ns=impulses,filtered_pairs_audit=checks['filtered_structural_pairs'],
                    seam_audit=checks['internal_seams'],violations=violations)
                if violations:
                    summary['violations']=violations
                    raise RuntimeError(f'earliest_violation_at_step_{log.steps_returned}:'+','.join(violations))
                # No state resets or pose changes between a post-step frame and the
                # next solver collision detection. Step 0 refers to initial_W_T_link.
                frames_pre=frames_post
            summary['tested_stations_mm'].append(station_mm)
            previous_width=width
        summary['slot_passed']=True
        summary['reason']='completed_all_4320_steps'
    except Exception as exc:
        summary['reason']=repr(exc)
        summary['traceback']=traceback.format_exc()
        log.record('stop',reason=summary['reason'])
    finally:
        if idx_g is not None:
            try:capture(p,client_id,body,idx_g,directory,'terminal.png')
            except Exception as exc:summary['screenshot_errors'].append(repr(exc))
        summary.update(max_abs_measured_speed_m_s=maximum_speed if finite_measurement_count else None,
                       max_abs_reported_motor_force_N=maximum_force if finite_measurement_count else None)
        log.finish(**summary)
    return read(directory/'result.json')
