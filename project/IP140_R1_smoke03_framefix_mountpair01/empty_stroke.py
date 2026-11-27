"""Smoke03 slots 1/2, empty stroke only. Caller supplies its existing gated world.
No simulator import, connect, load, restore, constraint, filtering, or retry here.
All positions and dimensions are SI. Physical contacts never use an epsilon.
"""
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

from episode_log import Journal, save_rgb
from mountpair import audit_mount

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
# Pad↔bone shares an assembly plane. Never disable this pair: on an empty
# no-load bench a resolved seam contact impulse or negative separation is anomalous.
PAIRS = [(PAD[0], PAD[1])]
PAIRS += [(PALM, n) for n in BONE + PAD]
PAIRS += list(zip(PAD, BONE))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def transform(p, xyz, quat):
    T = np.eye(4)
    T[:3, :3] = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    T[:3, 3] = xyz
    return T


def link_pose(p, state):
    # Double precision world link frame from world COM and local inertial pose.
    return transform(p, state[0], state[1]) @ np.linalg.inv(transform(p, state[2], state[3]))


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


def surfaces(p, body, g_idx, pad_idx, corners, cid, requested_width):
    g, left, right = p.getLinkStates(body,[g_idx,*pad_idx],
        computeForwardKinematics=1,physicsClientId=cid)
    inv_g = np.linalg.inv(link_pose(p,g))
    points = [inv_g @ link_pose(p,st) @ local for st,local in zip((left,right),corners)]
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


def scan_contacts(p,cid,body,by_name,by_index,arm_adjacent,initial,near_closed):
    manifold = p.getContactPoints(physicsClientId=cid)
    geometry = {}
    violations = []
    for a,b in PAIRS:
        rows=p.getClosestPoints(body,body,.002,linkIndexA=by_name[a],
                                linkIndexB=by_name[b],physicsClientId=cid)
        geometry[a+'|'+b]=rows
        if any(not math.isfinite(pt[8]) or pt[8]<0. for pt in rows):
            violations.append(f'negative_or_invalid_closest_distance:{a}|{b}')
        if not rows and ((a,b)!=PAIRS[0] or initial or near_closed):
            violations.append(f'missing_assembly_witness:{a}|{b}')
    # Keep every raw mount contact. The interface remains physically enabled.
    # Its signed distance is exempt only if the specific local fit is certified.
    try:
        mount = audit_mount(p,body,by_name,cid,manifolds=manifold)
    except Exception as exc:
        mount = dict(passed=False,violations=['mount_audit_exception'],error=repr(exc))
    if mount.get('passed') is not True:
        violations.append('fixed_mount_witness_not_certified')
    internal={frozenset(pair) for pair in PAIRS}
    for pt in manifold:
        a=by_index.get(pt[3],f'link:{pt[3]}')
        b=by_index.get(pt[4],f'link:{pt[4]}')
        d,fn,ft1,ft2=pt[8],pt[9],pt[10],pt[12]
        if not all(math.isfinite(v) for v in (d,fn,ft1,ft2)):
            violations.append(f'nonfinite_manifold:{a}|{b}')
            continue
        same_robot=pt[1]==body and pt[2]==body
        if same_robot and frozenset((a,b)) in internal:
            is_pads=frozenset((a,b))==frozenset(PAD)
            impulse=(fn!=0. or ft1!=0. or ft2!=0.)
            if d<0. or fn>.2 or (impulse and (initial or not is_pads)):
                violations.append(f'endpoint_or_seam:{a}|{b}')
        elif same_robot and {a,b}=={'wrist_3_link',PALM}:
            # Any unexpected local witness already failed the per-step audit above.
            continue
        elif same_robot and frozenset((a,b)) in arm_adjacent:
            # Only historical six pairs entirely within UR5e; no new gripper pair can match.
            continue
        elif d<-.001 or fn>.2:
            violations.append(f'forbidden_contact:{a}|{b}')
    return manifold,geometry,mount,violations


def run_slot(p,client_id,body,slot,directory):
    """Only after charge_slot. Persist every attempt outcome, including exceptions."""
    directory=Path(directory).resolve()
    number=slot['slot']
    if number not in (1,2) or slot['kind']!='empty_stroke' or tuple(slot['opening_sequence_mm'])!=STATIONS:
        raise ValueError('only frozen unloaded slots 1 and 2 are permitted')
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
        arm_adjacent={frozenset(pair) for pair in cfg['allowed_self_collision_pairs']
                      if all(name in ('base_link_inertia','shoulder_link','upper_arm_link','forearm_link',
                                      'wrist_1_link','wrist_2_link','wrist_3_link') for name in pair)}
        raw_initial=p.getJointStates(body,arm+jpr,physicsClientId=client_id)
        log.record('initial_configuration',body=body,links=by_index,joints=arm+jpr,
                   states=raw_initial,arm_home=home,step_seconds=DT,stations_mm=STATIONS)
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
                    raise RuntimeError('hard_time_limit')
                q=previous_width/2+math.copysign(min(abs(delta),SPEED*k*DT),delta)
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
                faces,within,nonnegative=surfaces(p,body,idx_g,pad_idx,corners,client_id,width)
                near_closed=(q==0. and max(abs(v) for v in faces['gaps_m'])<=.001)
                manifold,prox,mount,violations=scan_contacts(p,client_id,body,by_name,
                    by_index,arm_adjacent,initial=log.steps_returned==1,near_closed=near_closed)
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
                        if not 0.<=pos<=.070:violations.append(side+'.prismatic_limit')
                if not nonnegative:violations.append('nonfinite_faces_or_negative_geometry')
                if k>456 and not within:violations.append('stable_window_geometry')
                log.record('state',index=log.steps_returned,station_width_mm=station_mm,station_tick=k,
                    actual_arm_and_finger_states=states,joints=arm+jpr,sim_motor_force_N=[s[3] for s in fingers],
                    geometry=faces,within_frozen_tolerance=within,in_settled_window=k>456,
                    raw_manifolds=manifold,closest_points_by_named_pair=prox,mount_audit=mount,
                    normal_impulses_Ns=[pt[9]*DT for pt in manifold],violations=violations)
                if violations:
                    summary['violations']=violations
                    raise RuntimeError(f'earliest_violation_at_step_{log.steps_returned}:'+','.join(violations))
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
