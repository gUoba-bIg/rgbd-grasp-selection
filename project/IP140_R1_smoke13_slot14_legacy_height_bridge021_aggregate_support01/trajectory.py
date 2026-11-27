"""Offline six-joint kinematics and shared pre-registered slot 14 development bridge commands, SI units."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from speed_policy import finger_target

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MODEL=HERE/'model/ur5e_ip140_r1.urdf'
CONFIG=ROOT/'configs/simulation_v52r2.json'
ASSETS=ROOT/'data/simulation_v52/robot/robot_assets.json'
PLAN=HERE/'trajectory_plan.json'
ID='ip140bridge:slot14:parent-slot07:undo-legacy-z021'
PHASES=(('settle',240),('opening',480),('transit',672),('pregrasp',672),('approach',672),('close',480),('lift',792),('hold',720))
MOVING=('transit','pregrasp','approach')
DT=1/240.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def axis_rot(axis,theta):
    x,y,z=axis
    cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+math.sin(theta)*cross+2*math.sin(theta/2)**2*(cross@cross)


@lru_cache(1)
def chain():
    joints=ET.parse(MODEL).getroot().findall('joint')
    parents={j.find('child').get('link'):j for j in joints}
    ordered=[];link='ip140_palm'
    while link in parents:
        j=parents[link];o=j.find('origin')
        xyz=np.fromstring(o.get('xyz'),sep=' ')
        r,p,y=np.fromstring(o.get('rpy'),sep=' ')
        T=np.eye(4)
        T[:3,:3]=axis_rot((0,0,1),y)@axis_rot((0,1,0),p)@axis_rot((1,0,0),r)
        T[:3,3]=xyz
        rev=j.get('type')=='revolute'
        if not rev and j.get('type')!='fixed':raise ValueError('unregistered joint on TCP chain')
        axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if rev else None
        lim=j.find('limit')
        limits=[float(lim.get(k)) for k in ('lower','upper','velocity')] if rev else None
        ordered.append((j.get('name'),T,axis,limits))
        link=j.find('parent').get('link')
    ordered.reverse()
    if link!='world' or [n for n,t,a,l in ordered if a is not None]!=read(ASSETS)['arm_joint_names']:
        raise ValueError('actual TCP chain identity')
    return ordered


def fk(q,jacobian=False):
    T=np.eye(4)
    axes=[];points=[];i=0
    for _,origin,axis,_ in chain():
        T=T@origin
        if axis is not None:
            if jacobian:
                axes.append(T[:3,:3]@axis)
                points.append(T[:3,3].copy())
            T[:3,:3]=T[:3,:3]@axis_rot(axis,q[i])
            i+=1
    if len(q)!=i:raise ValueError('joint count')
    if not jacobian:return T
    J=np.vstack((np.cross(axes,T[:3,3]-points).T,np.array(axes).T))
    return T,J


def rotation_vector(R):
    skew=np.array((R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]))
    mag=np.linalg.norm(skew)
    c=float(np.trace(R)-1)
    if mag<1.e-13 and c<0:raise ValueError('ambiguous IK rotation branch')
    return skew*.5 if mag<1e-13 else skew*(math.atan2(mag,c)/mag)


def solve(goal,initial):
    q=np.array(initial,dtype=float)
    if q.shape!=(6,) or not np.isfinite(q).all() or np.asarray(goal).shape!=(4,4) or not np.isfinite(goal).all():
        raise ValueError('nonfinite or malformed IK target/seed')
    for iteration in range(251):
        T,J=fk(q,True)
        e=np.r_[goal[:3,3]-T[:3,3],rotation_vector(goal[:3,:3]@T[:3,:3].T)]
        if not np.isfinite(T).all() or not np.isfinite(J).all() or not np.isfinite(e).all():
            raise ValueError('nonfinite IK intermediate')
        if np.linalg.norm(e[:3])<1e-10 and np.linalg.norm(e[3:])<1e-10:
            return q.tolist()
        delta=J.T@np.linalg.solve(J@J.T+(.005**2)*np.eye(6),e)
        if not np.isfinite(delta).all():raise ValueError('nonfinite IK increment')
        q+=delta/max(1.,np.linalg.norm(delta)/.18)  # numerical search step scaling, never state clamping
    raise ValueError('static IK failed to converge')


def build():
    cfg=read(CONFIG)
    slot=read(HERE/'preregistration.json')['slots'][13]
    if (slot['candidate_id']!=ID or slot['slot']!=14 or slot['kind']!='nominal'
            or slot['object_id']!='021' or slot['not_p3'] is not True):
        raise ValueError('slot 14 development-bridge identity')
    raw=np.array(slot['candidate_world']); goal=np.eye(4)
    goal[:3,:3]=raw[4:13].reshape(3,3)
    goal[:3,3]=raw[13:16]+raw[3]*goal[:3,0]
    if not np.isfinite(raw).all():raise ValueError('nonfinite candidate')
    if not np.allclose(goal,slot['T_W_G_m'],atol=1e-12,rtol=0) or raw[1]!=0.06898170991686796 or slot['pre_open_m']!=raw[1]:
        raise ValueError('preopening or candidate / tool transform mismatch')
    mount=np.asarray(read(HERE/'model/manifest.json')['F_T_G'])
    flange_goal=goal@np.linalg.inv(mount)
    flange_transit=flange_goal.copy();flange_transit[2,3]=cfg['motion']['transit_tool_z']
    pre=goal.copy();pre[:3,3]-=cfg['motion']['pregrasp_m']*goal[:3,0]
    initial=cfg['home_joints']
    q=initial
    limits=np.array([l for n,t,a,l in chain() if l is not None])
    rows=[]
    for phase,T in zip(MOVING,(flange_transit@mount,pre,goal)):
        solved=solve(T,q)
        # A revolute joint has equivalent 2*pi windings. Follow the shortest
        # legal branch from the preceding knot, prior to any physical run.
        next_q=[]
        for angle,start,(lower,upper,_) in zip(solved,q,limits):
            equivalents=[angle+k*2*math.pi for k in range(-2,3)
                         if lower<=angle+k*2*math.pi<=upper]
            if not equivalents:raise ValueError('no feasible IK winding')
            next_q.append(min(equivalents,key=lambda val:abs(val-start)))
        a=np.asarray(q);b=np.asarray(next_q)
        # Cubic smoothstep is monotone, with analytic maximum derivative 1.5/T.
        if (np.any(b<limits[:,0]) or np.any(b>limits[:,1]) or
                np.any(1.5*np.abs(b-a)/2.5>limits[:,2])):
            raise ValueError('frozen joint/velocity path limit')
        rows.append(dict(phase=phase,start_q_rad=q,target_q_rad=next_q,target_W_T_G=T.tolist(),
                         FK_W_T_G=fk(next_q).tolist(),moving_ticks=600,settling_ticks=72))
        q=next_q
    lift_goal=goal.copy();lift_goal[2,3]+=cfg['motion']['lift_m']
    lift_q=solve(lift_goal,q)
    lift_start=np.asarray(q);lift_end=np.asarray(lift_q)
    if (np.any(lift_end<limits[:,0]) or np.any(lift_end>limits[:,1])
            or np.any(1.5*np.abs(lift_end-lift_start)/3.>limits[:,2])):
        raise ValueError('frozen lift joint/velocity path limit')
    if cfg['motion']['lift_m']!=.15 or cfg['motion']['lift_seconds']!=3:
        raise ValueError('lift differs from frozen target delta 150 mm / 3 s')
    return dict(passed=True,slot=14,parent_slot=7,development_corrective_bridge=True,independent_confirmation=False,candidate_id=ID,candidate_world=raw.tolist(),pre_open_m=float(raw[1]),phase_steps=list(map(list,PHASES)),
        home_joints=initial,W_T_G_home=fk(initial).tolist(),W_T_G_goal=goal.tolist(),waypoints=rows,
        lift=dict(start_q_rad=q,target_q_rad=lift_q,moving_ticks=720,settling_ticks=72,
                  target_W_T_G=lift_goal.tolist(),distance_world_z_m=.15,
                  interpolation='historical joint smoothstep; target translated 150 mm along world Z'),
        limits_rad_rad_s=limits.tolist(),tool0_z_transit_m=cfg['motion']['transit_tool_z'],pregrasp_m=cfg['motion']['pregrasp_m'],
        interpolation='u^2(3-2u), u=min(tick,600)/600, 72 settling steps; targetVelocity=0, finite frozen arm effort.',
        references=['IP140-R1_参数与证据附表_小修02.pdf, appendix A and S2-S5',
                    'configs/simulation_v52r2.json:motion','scripts/sim_v52r2_episode.py:move'],
        hashes={'model':sha(MODEL),'config':sha(CONFIG),'arm_asset_manifest':sha(ASSETS),
                'kinematics':sha(Path(__file__)),'speed_policy':sha(HERE/'speed_policy.py')})


def validate_plan():
    plan=read(PLAN)
    if plan!=build():raise ValueError('trajectory differs from deterministic registered plan')
    return plan


def reference(plan,phase,tick):
    if phase not in dict(PHASES) or not 1<=tick<=dict(PHASES)[phase]:raise ValueError('phase/tick')
    if phase in MOVING:
        row=plan['waypoints'][MOVING.index(phase)]
        a=np.asarray(row['start_q_rad']);b=np.asarray(row['target_q_rad'])
        u=min(1.,tick/600.);f=u*u*(3.-2.*u)
        q=a+(b-a)*f;dq=(b-a)*6.*u*(1-u)/2.5
    elif phase=='lift':
        a=np.asarray(plan['lift']['start_q_rad']);b=np.asarray(plan['lift']['target_q_rad'])
        u=min(1.,tick/720.);f=u*u*(3.-2.*u)
        q=a+(b-a)*f;dq=(b-a)*6*u*(1-u)/3.
    elif phase=='hold':
        q=np.asarray(plan['lift']['target_q_rad']);dq=np.zeros(6)
    elif phase=='close':
        q=np.array(plan['waypoints'][-1]['target_q_rad']);dq=np.zeros(6)
    else:
        q=np.array(plan['home_joints']);dq=np.zeros(6)
    finger=finger_target(phase,tick,DT,plan['pre_open_m'])
    return q,dq,finger

