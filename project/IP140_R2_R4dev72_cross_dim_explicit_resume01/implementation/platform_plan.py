"""Frozen IP140 motion knots for a genuine P3 row. No Robotiq height offset."""
import math
import numpy as np
from platform_imports import bind_frozen_libraries
bind_frozen_libraries()
from trajectory import HERE,MODEL,ASSETS,CONFIG,read,sha,chain,solve,PHASES,MOVING,fk
def H(r,t):
    x=np.eye(4);x[:3,:3]=r;x[:3,3]=t
    return x

def ip140(row,WC,FG):
    """R4 calibrations have to be supplied from their own paired rendering."""
    row=np.asarray(row,dtype=np.float64)
    if row.shape!=(17,) or not np.isfinite(row).all() or not np.isfinite(WC).all():
        raise ValueError("invalid input")
    R=row[4:13].reshape(3,3); depth=float(row[3]); width=float(row[1]);t=row[13:16]
    if not 0<=width<=.14:raise ValueError("frozen opening out of bounds")
    CG=H(R,t+R[:,0]*depth)
    WG=WC@CG
    return WG,WG@np.linalg.inv(FG),width


def plan_for(raw_camera,T_W_C,candidate_id):
    camera=np.asarray(T_W_C,dtype=float)
    if camera.shape!=(4,4) or not np.isfinite(camera).all() or not np.allclose(camera[3],[0,0,0,1],rtol=0,atol=1e-12):
        raise RuntimeError('invalid calibration')
    raw_camera=np.asarray(raw_camera,dtype=float)
    if raw_camera.shape!=(17,) or not np.isfinite(raw_camera).all():raise RuntimeError('invalid P3 row')
    r=raw_camera[4:13].reshape(3,3)
    if np.max(np.abs(r.T@r-np.eye(3)))>=1e-4 or abs(np.linalg.det(r)-1)>=1e-4:raise RuntimeError('invalid P3 orientation')
    cfg=read(CONFIG)
    goal,flange,width=ip140(raw_camera,camera,np.asarray(read(HERE/'model/manifest.json')['F_T_G']))
    raw=raw_camera.copy()
    raw[4:13]=goal[:3,:3].ravel()
    raw[13:16]=(camera@np.r_[raw_camera[13:16],1])[:3]
    if np.max(np.abs(raw[13:16]+raw[3]*goal[:3,0]-goal[:3,3]))>1e-9:raise RuntimeError('P3 tip chain')
    if goal[2,3]<0:raise ValueError('target_contact_below_known_table')
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
    return dict(passed=True,engineering_bridge_confirmation=False,algorithm_validation=True,candidate_id=candidate_id,candidate_world=raw.tolist(),pre_open_m=float(raw[1]),phase_steps=list(map(list,PHASES)),
        home_joints=initial,W_T_G_home=fk(initial).tolist(),W_T_G_goal=goal.tolist(),waypoints=rows,
        lift=dict(start_q_rad=q,target_q_rad=lift_q,moving_ticks=720,settling_ticks=72,
                  target_W_T_G=lift_goal.tolist(),distance_world_z_m=.15,
                  interpolation='historical joint smoothstep; target translated 150 mm along world Z'),
        limits_rad_rad_s=limits.tolist(),tool0_z_transit_m=cfg['motion']['transit_tool_z'],pregrasp_m=cfg['motion']['pregrasp_m'],
        interpolation='u^2(3-2u), u=min(tick,600)/600, 72 settling steps; targetVelocity=0, finite frozen arm effort.',
        references=['IP140-R1_参数与证据附表_小修02.pdf, appendix A and S2-S5',
                    'configs/simulation_v52r2.json:motion','scripts/sim_v52r2_episode.py:move'],
        hashes={'model':sha(MODEL),'config':sha(CONFIG),'arm_asset_manifest':sha(ASSETS),
                'kinematics':sha(HERE/'trajectory.py'),'speed_policy':sha(HERE/'speed_policy.py')})

