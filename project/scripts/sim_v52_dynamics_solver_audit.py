"""Zero-physics audit of preserved sources, URDF mass matrix and existing traces.

Does not import any experiment/robot module or PyBullet. Refuses output overwrite.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/simulation_v52_dynamics_audit/solver_control.json'
assert Path.cwd().resolve() == ROOT
assert not OUT.exists(), 'Preserve existing audit result'

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def identity(path):
    path = ROOT / path
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def vec(text):
    return np.fromstring(text or '0 0 0', sep=' ')

def skew(a):
    x,y,z=a
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=float)

def rot(axis, angle):
    a=axis/np.linalg.norm(axis); K=skew(a)
    return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)

def origin(node):
    T=np.eye(4)
    if node is not None:
        r,p,y=vec(node.get('rpy'))
        T[:3,:3]=rot(np.array([0.,0,1]),y)@rot(np.array([0.,1,0]),p)@rot(np.array([1.,0,0]),r)
        T[:3,3]=vec(node.get('xyz'))
    return T

cfg=read('configs/simulation_v52r2.json')
manifest=read(cfg['robot_manifest'])
structure=read('data/simulation_v52/robot/structure_check.json')
tree=ET.parse(ROOT/cfg['robot_urdf']).getroot()
joints={j.get('name'):j for j in tree.findall('joint')}
order=[x['joint'] for x in structure['joint_table'] if x['joint'] in joints and joints[x['joint']].get('type') in ('revolute','continuous','prismatic')]
assert len(order)==12
links={l.get('name'):l for l in tree.findall('link')}
index={n:i for i,n in enumerate(order)}
roots=set(links)-{j.find('child').get('link') for j in joints.values()}

def kinematics(q):
    poses={name:np.eye(4) for name in roots}
    ancestors={name:[] for name in roots}
    axes={};positions={};pending=list(joints.values())
    while pending:
        progress=False
        for j in pending[:]:
            parent=j.find('parent').get('link'); child=j.find('child').get('link'); name=j.get('name')
            if parent not in poses: continue
            T=poses[parent]@origin(j.find('origin')); motion=np.eye(4)
            ancestors[child]=list(ancestors[parent])
            if name in index:
                axis=vec(j.find('axis').get('xyz'))
                axes[name]=T[:3,:3]@axis;positions[name]=T[:3,3]
                ancestors[child].append(name)
                if j.get('type')=='prismatic':motion[:3,3]=axis*q[name]
                else:motion[:3,:3]=rot(axis,q[name])
            poses[child]=T@motion
            pending.remove(j);progress=True
        assert progress, 'URDF tree is disconnected'
    return poses,ancestors,axes,positions

def mass_matrix(q):
    poses,ancestors,axes,positions=kinematics(q)
    M=np.zeros((12,12))
    for name,link in links.items():
        inert=link.find('inertial')
        if inert is None:continue
        mass=float(inert.find('mass').get('value'))
        Iel=inert.find('inertia')
        I=np.array([[float(Iel.get('ixx')),float(Iel.get('ixy')),float(Iel.get('ixz'))],
                    [float(Iel.get('ixy')),float(Iel.get('iyy')),float(Iel.get('iyz'))],
                    [float(Iel.get('ixz')),float(Iel.get('iyz')),float(Iel.get('izz'))]])
        T=poses[name]@origin(inert.find('origin'));com=T[:3,3];Iw=T[:3,:3]@I@T[:3,:3].T
        Jv=np.zeros((3,12));Jw=np.zeros((3,12))
        for joint in ancestors[name]:
            k=index[joint];axis=axes[joint]
            if joints[joint].get('type')=='prismatic':Jv[:,k]=axis
            else:Jv[:,k]=np.cross(axis,com-positions[joint]);Jw[:,k]=axis
        M += mass*(Jv.T@Jv)+Jw.T@Iw@Jw
    return M

def nominal(qdriver, arm):
    q={n:0. for n in order}
    q.update(zip(manifest['arm_joint_names'],arm));q[manifest['master_joint']]=qdriver
    for row in manifest['mimic_joints']:q[row['joint']]=qdriver*row['multiplier']+row['offset']
    return q

def matrix_audit(name,q,basis):
    M=mass_matrix(q);eigen=np.linalg.eigvalsh(M);assert eigen.min()>0
    W=np.linalg.inv(M);J=[];rows=[]
    for row in cfg['coupling_repair']['tree']:
        a=index[row['a']];b=index[row['b']];ratio=row['ratio']
        j=np.zeros(12);j[a]=1;j[b]=ratio;J.append(j)
        src=float(W[a,a]+ratio*ratio*W[b,b]);full=float(j@W@j)
        rows.append(dict(**row,d_source=src,d_combined=full,cross_terms=full-src,
                         d_combined_over_d_source=full/src,
                         isolated_row_iteration_error_factor=1-full/src))
    J=np.array(J);A=J@W@J.T;D=np.array([x['d_source'] for x in rows])
    # Linear unbounded sequential sweep, omitting motors, contacts, limits and ERP.
    def sweep(diagonal):
        B=np.eye(5)
        for k in range(5):B[k,:]-=A[k,:]@B/diagonal[k]
        return float(max(abs(np.linalg.eigvals(B))))
    return dict(name=name,posture_basis=basis,positions=q,mass_matrix=M.tolist(),
                minimum_mass_eigenvalue=float(eigen.min()),mass_condition=float(eigen.max()/eigen.min()),
                rows=rows,combined_constraint_gram=A.tolist(),gram_condition=float(np.linalg.cond(A)),
                idealized_sweep_spectral_radius_source_diag=sweep(D),
                idealized_sweep_spectral_radius_full_diag=sweep(np.diag(A)))

# Pure XML FK cross-check to the previously saved static calibration, not new Bullet calls.
fkchecks=[]
for row in structure['fk_checks']:
    poses,*_=kinematics(nominal(row['q'],[0.]*6))
    inv=np.linalg.inv(poses[manifest['gripper_base_link']])
    centers=np.array([(inv@poses[name])[:3,3] for name in manifest['pad_links']])
    old=min(manifest['opening_calibration'],key=lambda v:abs(v['driver_rad']-row['q']))
    # Saved Bullet structure report stores exact TCP; the calibration grid is not exact at every q.
    err=float(np.max(np.abs(centers.mean(axis=0)-np.array(row['tcp_in_base_m']))))
    assert err<2e-6
    fkchecks.append(dict(q=row['q'],max_tcp_error_vs_saved_static_report_m=err))

report=read('results/simulation_v52/admission/report_048.json')
inputs=[identity('results/simulation_v52/admission/report_048.json')]
phase_stats=collections.defaultdict(lambda:dict(samples=0,row_observations=0,saturated_at_99pct=0,max_abs_torque_nm=0.,max_mimic_error_rad=0.,row_max_abs_torque_nm=[0.]*5))
episode_stats=[]; slot4trace=None;slot4result=None;max_torque=0.;total_samples=0;bad_shape=0
for row in report['rows']:
    path=row['directory']+'/trace.json';inputs.append(identity(path));trace=read(path)
    result=row['result']; complete=[]; erows=[];sat=0;mx=0.
    for sample in trace:
        states=sample.get('gear_constraint_states')
        if states is None:continue
        if not (len(states)==5 and all(isinstance(s,list) and len(s)==1 and isinstance(s[0],(int,float)) for s in states)):
            bad_shape+=1;continue
        values=[abs(x[0]) for x in states];max_torque=max(max_torque,max(values));mx=max(mx,max(values));total_samples+=1
        saturated=sum(v>=.99*cfg['physics']['gear_force'] for v in values);sat+=saturated
        ps=phase_stats[sample['phase']];ps['samples']+=1;ps['row_observations']+=5;ps['saturated_at_99pct']+=saturated
        ps['max_abs_torque_nm']=max(ps['max_abs_torque_nm'],max(values));ps['max_mimic_error_rad']=max(ps['max_mimic_error_rad'],sample['mimic_error_rad'])
        ps['row_max_abs_torque_nm']=[max(a,b) for a,b in zip(ps['row_max_abs_torque_nm'],values)]
        q={manifest['master_joint']:sample['driver_rad'],**sample['follower_joint_positions']}
        rr=[q[t['a']]+t['ratio']*q[t['b']] for t in cfg['coupling_repair']['tree']]
        erows.append(dict(step=sample['step'],phase=sample['phase'],row_residuals_rad=rr,
                          max_row_residual_rad=max(abs(x) for x in rr),mimic_error_rad=sample['mimic_error_rad'],
                          max_abs_gear_torque_nm=max(values)))
    if erows:
        episode_stats.append(dict(slot_id=row['slot_id'],kind=result['slot']['kind'],
                                  sampled_steps=len(erows),max_abs_gear_torque_nm=mx,
                                  saturation_row_observations_at_99pct=sat,
                                  max_sampled_row_residual_rad=max(x['max_row_residual_rad'] for x in erows),
                                  max_mimic_error_per_tick_rad=result.get('max_mimic_error_rad'),
                                  peak_sampled_mimic_record=max(erows,key=lambda x:x['mimic_error_rad'])))
    if row['slot_id']==4:slot4trace=trace;slot4result=result;slot4residuals=erows

matrix_cases=[]
approach=next(x['joints'] for x in slot4result['planning'] if x['name']=='approach')
for q in [0.,.35,.5,.695]:
    matrix_cases.append(matrix_audit('nominal_home_q_'+str(q),nominal(q,cfg['home_joints']),
        'Frozen home arm posture and exact nominal mimic relationship; not a recorded dynamic posture'))
for phase in ['close','hold']:
    sample=max((x for x in slot4trace if x['phase']==phase),key=lambda x:x['mimic_error_rad'])
    arm=approach if phase=='close' else next(x['joints'] for x in slot4result['planning'] if x['name']=='lift')
    q=nominal(sample['driver_rad'],arm);q.update(sample['follower_joint_positions'])
    matrix_cases.append(matrix_audit('slot4_sampled_'+phase+'_step_'+str(sample['step']),q,
        'Six gripper angles from saved trace; six arm angles from planned phase endpoint, not measured actual joint state'))

archive='runtime/simulation_v52/cache/http-v2/4/5/7/b/d/457bdc0f13fc3174ca2e66c339d420a683bd3fc6fb4f34259b382951.body'
wheel='runtime/simulation_v52/cache/wheels/72/95/1d/b336e5ee612ae9a019bfff4dc0bedd100ee6f0570db205fdf8/pybullet-3.2.7-cp312-cp312-win_amd64.whl'
installed=list((ROOT/'runtime/simulation_v52/venv/Lib/site-packages').glob('pybullet*.pyd'))
assert len(installed)==1
with zipfile.ZipFile(ROOT/wheel) as z:
    entry=next(n for n in z.namelist() if n.startswith('pybullet') and n.endswith('.pyd'))
    wheel_binary_sha=hashlib.sha256(z.read(entry)).hexdigest()
binding=dict(archive=identity(archive),wheel=identity(wheel),installed_binary=identity(installed[0]),
             wheel_binary_entry=entry,wheel_binary_sha256=wheel_binary_sha,
             installed_matches_wheel=wheel_binary_sha==identity(installed[0])['sha256'])
assert binding['installed_matches_wheel']
source_files=[identity(p) for p in sorted((ROOT/'results/simulation_v52_dynamics_audit/source').rglob('*')) if p.is_file()]
frozen=[identity(p) for p in ['scripts/sim_v52_episode.py','scripts/sim_v52r1_episode.py','scripts/sim_v52r2_episode.py','scripts/sim_v52_robot_assets.py','configs/simulation_v52.json','configs/simulation_v52r1.json','configs/simulation_v52r2.json',cfg['robot_manifest'],cfg['robot_urdf']]]
data=dict(schema='v52_zero_step_solver_control_audit_v1',new_physics_steps=0,new_episodes=0,new_neural_forwards=0,
    imports_pybullet=False,old_experiments_modified=False,not_a_new_admission_result=True,
    source_binding=binding,source_copies=source_files,frozen_inputs=frozen,old_trace_inputs=inputs,
    source_findings=dict(gear_equation='q_A + ratio*q_B = relativePositionTarget; no aux link used',
        jacobian_dof_index='6 + link.m_dofOffset',both_endpoint_velocity_and_impulse_present=True,
        same_body_effective_mass_cross_terms_present=False,
        source_denominator='J_A W J_A^T + J_B W J_B^T',combined_denominator='(J_A+J_B) W (J_A+J_B)^T',
        root_cause_proven=False,global_noncontact_erp_default=.2,gear_erp=.2,effective_gear_position_erp=.04,
        gear_cap_nm=80.,gear_impulse_cap_nm_s=80/240,master_motor_cap_nm=10.,master_motor_impulse_cap_nm_s=10/240,
        motor_kp=.1,motor_kd=1.,motor_max_target_velocity_rad_s=.5,follower_motor_cap_nm=0.,
        constraint_state_units='generalized torque N*m for these revolute gear rows; applied impulse divided by solver timestep'),
    trace_review=dict(samples_with_five_gear_states=total_samples,gear_row_observations=total_samples*5,
        invalid_shapes=bad_shape,max_sampled_abs_gear_torque_nm=max_torque,
        saturation_definition='abs(torque)>=0.99*80 N*m; not a modified admission threshold',
        total_saturated_row_observations=sum(x['saturation_row_observations_at_99pct'] for x in episode_stats),
        phase_summary=dict(phase_stats),episodes=episode_stats,slot4_row_residuals=slot4residuals,
        limitations=['0.1 s snapshots; cannot exclude between-snapshot saturation','No saved joint velocities, actual master torque, per-iteration residual, or actual mass matrix',
                     'gear cap lack of saturation does not prove unlimited-force behavior or exclude motor saturation']),
    xml_mass_estimate=dict(method='Sum m*Jv.T*Jv + Jw.T*(R*I*R.T)*Jw for all links; fixed world, 12 active URDF DOFs; COM and inertia-frame origins included',
        dof_order=order,old_saved_fk_crosscheck=fkchecks,cases=matrix_cases,
        limitations=['No new Bullet import or mass-matrix query','URDF-to-solver loaded inertia equivalence is not dynamically verified',
                     'Fixed base; all 12 joint DOFs free in instantaneous mass matrix; motors/contact/limits absent from idealized sweep',
                     'Saved trace lacks actual arm angles; planned arm angles used for slot4 estimates',
                     'Spectral radius is for isolated linear gear subsystem only, not full solver prediction']),
    proposed_diagnostics=dict(status='design_only_not_executed_requires_review',runs_cap=12,
        design='3 conditions x dt{1/240,1/480} x iterations{150,300}, one per cell',
        conditions=['no object, free closure','no object, each pad 1 N opening-directed calibrated force','reproduce sphere 021 slot4 geometry'],
        mandatory_logs=['every tick q/dq of all 12 DOFs','master requested and actual applied torque','five row torque/cap and q_A+r*q_B and qdot residual',
                        'actual full 12DOF M and d_combined/d_source, or explicitly mark unavailable','all contact force/distance/link IDs','solver iteration residual if instrumented; API alone cannot provide it'],
        cautions=['dt change also changes ERP/dt and master kp/dt; this is whole-discretization sensitivity, not pure solver proof',
                  'When changing dt, issue force/cap commands after dt is set','No force increase, new mechanism, network, or success-threshold relaxation',
                  '12 runs can classify triggers but cannot causally prove missing-cross-term mechanism without targeted independent evidence']))
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),samples=total_samples,peak_gear_torque_nm=max_torque,
    saturated_row_observations=data['trace_review']['total_saturated_row_observations'],
    matrix_ratios={x['name']:[round(t['d_combined_over_d_source'],6) for t in x['rows']] for x in matrix_cases},
    new_physics_steps=0),ensure_ascii=False))
