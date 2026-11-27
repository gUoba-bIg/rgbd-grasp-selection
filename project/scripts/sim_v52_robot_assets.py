"""Build pinned official UR5e/Robotiq assets; inspection never steps physics.

Source repositories are immutable. All conversion products belong to v52.
Transforms T_A_B map B coordinates into A. No grasp episodes are executed here.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/simulation_v52/robot'
UR = ROOT / 'third_party/simulation_v52/ur_description'
RQ_REPO = ROOT / 'third_party/simulation_v52/robotiq'
RQ = RQ_REPO / 'grippers/robotiq_description'
URDF = OUT / 'ur5e_robotiq_2f140.urdf'
MANIFEST = OUT / 'robot_assets.json'
ARM_NAMES = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
             'wrist_1_joint','wrist_2_joint','wrist_3_joint']
MASTER = 'rq_finger_joint'
PADS = ['rq_left_inner_finger_pad', 'rq_right_inner_finger_pad']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def xyz(text): return np.fromstring(text or '0 0 0',sep=' ')

def transform(origin):
    T=np.eye(4)
    if origin is not None:
        T[:3,3]=xyz(origin.get('xyz'))
        T[:3,:3]=Rotation.from_euler('xyz',xyz(origin.get('rpy'))).as_matrix()
    return T

def joint_data(tree):
    return {j.get('name'):j for j in tree.findall('joint')}

def fk(tree, driver):
    """Exact URDF forward kinematics with the official mimic multipliers."""
    joints=joint_data(tree)
    children={j.find('child').get('link') for j in joints.values()}
    roots=[l.get('name') for l in tree.findall('link') if l.get('name') not in children]
    poses={r:np.eye(4) for r in roots}
    pending=list(joints.values())
    while pending:
        changed=False
        for j in pending[:]:
            par=j.find('parent').get('link')
            if par not in poses: continue
            val=driver if j.get('name')==MASTER else 0.
            m=j.find('mimic')
            if m is not None:
                assert m.get('joint')==MASTER
                val=driver*float(m.get('multiplier','1'))+float(m.get('offset','0'))
            motion=np.eye(4)
            if j.get('type') in ('revolute','continuous'):
                axis=xyz(j.find('axis').get('xyz'))
                motion[:3,:3]=Rotation.from_rotvec(axis*val).as_matrix()
            poses[j.find('child').get('link')]=poses[par]@transform(j.find('origin'))@motion
            pending.remove(j);changed=True
        if not changed: raise ValueError('Non-tree robot')
    return poses

def opening_geometry(tree, driver):
    poses=fk(tree,driver)
    inv=np.linalg.inv(poses['rq_robotiq_140_base_link'])
    pads=[inv@poses[x] for x in PADS]
    centers=np.array([x[:3,3] for x in pads])
    direction=(centers[1]-centers[0]);direction/=np.linalg.norm(direction)
    # Official collision box is 30 x 70 x 7.5 mm; local z is pad thickness.
    half=np.array([.03,.07,.0075])/2
    support=[sum(abs(direction@T[:3,:3])*half) for T in pads]
    width=float(np.linalg.norm(centers[1]-centers[0])-sum(support))
    return {'driver_rad':float(driver),'inner_pad_gap_m':width,
            'tcp_in_gripper_base_m':centers.mean(axis=0).tolist(),
            'pad_centers_in_gripper_base_m':centers.tolist(),
            'closing_axis_in_gripper_base':direction.tolist(),
            'pad_normal_in_gripper_base':[T[:3,2].tolist() for T in pads]}

def export():
    import xacro
    import xacro.substitution_args
    import trimesh
    if MANIFEST.exists(): raise FileExistsError('Refuse to overwrite prior robot asset manifest')
    OUT.mkdir(parents=True,exist_ok=True)
    packages={'ur_description':UR,'robotiq_description':RQ}
    # Only the package resolver is provided, in memory; no ROS/global env changes.
    xacro.substitution_args._eval_find=lambda name: str(packages[name]).replace('\\','/')
    wrapper='''<?xml version="1.0"?>
<robot xmlns:xacro="http://wiki.ros.org/xacro" name="v52_ur5e_robotiq2f140">
 <xacro:include filename="$(find ur_description)/urdf/ur_macro.xacro"/>
 <xacro:include filename="$(find robotiq_description)/urdf/robotiq_2f_140_macro.urdf.xacro"/>
 <link name="world"/>
 <xacro:ur_robot name="ur5e" tf_prefix="" parent="world"
 joint_limits_parameters_file="$(find ur_description)/config/ur5e/joint_limits.yaml"
 kinematics_parameters_file="$(find ur_description)/config/ur5e/default_kinematics.yaml"
 physical_parameters_file="$(find ur_description)/config/ur5e/physical_parameters.yaml"
 visual_parameters_file="$(find ur_description)/config/ur5e/visual_parameters.yaml"
 smart_panel="false"><origin xyz="0 0 0" rpy="0 0 0"/></xacro:ur_robot>
 <xacro:robotiq_gripper name="robotiq" prefix="rq_" parent="tool0" include_ros2_control="false">
 <origin xyz="0 0 0" rpy="0 0 ${-pi/2}"/>
 </xacro:robotiq_gripper>
</robot>'''
    wrapper_path=OUT/'ur5e_2f140.urdf.xacro'
    wrapper_path.write_text(wrapper,encoding='utf-8')
    xml=xacro.process_file(str(wrapper_path)).toxml()
    (OUT/'expanded_official_before_adaptation.urdf').write_text(xml,encoding='utf-8')
    tree=ET.fromstring(xml)
    meshes=[]
    mesh_dir=OUT/'meshes';mesh_dir.mkdir(exist_ok=True)
    memo={}
    for mesh in tree.findall('.//mesh'):
        uri=mesh.get('filename')
        pkg,rel=uri.removeprefix('package://').split('/',1)
        source=packages[pkg]/rel
        if uri not in memo:
            dst=mesh_dir/(pkg+'_'+rel.replace('/','_'))
            method='byte_copy'
            if source.suffix.lower()=='.dae':
                dst=dst.with_suffix('.obj')
                scene=trimesh.load(source,force='scene')
                converted=scene.to_geometry()
                # Physical robot rendering does not use texture appearance as perception evidence.
                converted.export(dst)
                method='collada_scene_transforms_applied_to_obj_no_geometry_simplification'
            else: shutil.copy2(source,dst)
            memo[uri]=dst
            meshes.append({'source':str(source.relative_to(ROOT)), 'source_sha256':sha(source),
                           'output':str(dst.relative_to(ROOT)),'output_sha256':sha(dst),'method':method})
        mesh.set('filename',memo[uri].relative_to(OUT).as_posix())
    # Explicit massless dummy frames avoid Bullet's default 1 kg fallback.
    # Stock pads omit mass: supply declared 5 g rectangular-pad inertias, not a measured value.
    inertial_assumptions=[]
    for link in tree.findall('link'):
        if link.find('inertial') is not None: continue
        name=link.get('name');mass=.005 if name in PADS else 0.
        dims=np.array([.03,.07,.0075]);I=mass/12*np.array([dims[1]**2+dims[2]**2,dims[0]**2+dims[2]**2,dims[0]**2+dims[1]**2])
        inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'origin',xyz='0 0 0',rpy='0 0 0')
        ET.SubElement(inertial,'mass',value=str(mass));ET.SubElement(inertial,'inertia',ixx=str(I[0]),iyy=str(I[1]),izz=str(I[2]),ixy='0',ixz='0',iyz='0')
        inertial_assumptions.append({'link':name,'mass_kg':mass,'basis':'5g_pad_assumption' if mass else 'massless_coordinate_frame'})
    ET.indent(tree)
    ET.ElementTree(tree).write(URDF,encoding='utf-8',xml_declaration=True)
    calibration=[opening_geometry(tree,q) for q in np.linspace(0,.695,140)]
    gaps=[x['inner_pad_gap_m'] for x in calibration]
    assert all(a>b for a,b in zip(gaps,gaps[1:])), 'opening must strictly decrease'
    mimic=[]
    for j in tree.findall('joint'):
        m=j.find('mimic')
        if m is not None:
            mimic.append({'joint':j.get('name'),'master':m.get('joint'),
                          'multiplier':float(m.get('multiplier','1')),'offset':float(m.get('offset','0')),
                          'lower':float(j.find('limit').get('lower')),'upper':float(j.find('limit').get('upper'))})
    licenses=[]
    for repo in (UR,RQ_REPO):
        for path in repo.rglob('*'):
            if path.is_file() and path.name.upper().startswith(('LICENSE','COPYING')) and '.git' not in path.parts:
                licenses.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    sources=[]
    for repo,url in [(UR,'https://github.com/UniversalRobots/Universal_Robots_ROS2_Description'),(RQ_REPO,'https://github.com/robotiq/ros')]:
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
        assert not status, 'Official source changed'
        sources.append({'url':url,'commit':commit,'clean_worktree':True})
    data={'schema':'simulation_v52_robot_assets_v1','status':'exported_not_physics_validated',
          'python':str(ROOT/'runtime/simulation_v52/venv/Scripts/python.exe'),
          'urdf':str(URDF),'urdf_sha256':sha(URDF),'source_repositories':sources,
          'licenses':licenses,'mesh_assets':meshes,'inertial_assumptions':inertial_assumptions,
          'arm_joint_names':ARM_NAMES,'arm_ee_link':'tool0','gripper_base_link':'rq_robotiq_140_base_link',
          'T_tool0_gripper_base':np.eye(4).tolist(),'mount_assumption':'coincident flange plane; stock macro +pi/2 cancelled by supplied -pi/2; no adapter thickness',
          'master_joint':MASTER,'master_range_rad':[0.,.695],'mimic_joints':mimic,'pad_links':PADS,
          'opening_calibration':calibration,'tcp_definition':'midpoint of two official pad collision-box centers; varies with q; orientation equals gripper base',
          'width_definition':'distance between inner collision faces, using exact official FK, not width/2',
          'candidate_axes':{'approach_gripper_base':[0,0,1],'closing_gripper_base':[0,1,0],
                            'R_gripperBase_grasp':[[0,0,-1],[0,1,0],[1,0,0]]},
          'limitations':['UR default nominal kinematics, not a particular calibrated real robot',
                         'Robotiq upstream 2f140 description has not been hardware validated',
                         'PyBullet mesh collision uses convex hulls per collision link; no physical grasp has been run',
                         'pads have explicitly assumed mass; motor/contact dynamics require engineering admission'],
          'grasp_episodes_executed':0}
    dump(MANIFEST,data)
    print(json.dumps({'exported':str(URDF),'gap_open_m':gaps[0],'gap_closed_m':gaps[-1],'mimic_count':len(mimic)}))

def driver_for_width(width, manifest=None):
    m=manifest or json.loads(MANIFEST.read_text(encoding='utf-8'))
    cal=m['opening_calibration']
    lo,hi=cal[-1]['inner_pad_gap_m'],cal[0]['inner_pad_gap_m']
    if width<lo-1e-9 or width>hi+1e-9: raise ValueError(f'width {width} outside [{lo},{hi}]; no silent clipping')
    return float(np.interp(width,[x['inner_pad_gap_m'] for x in cal][::-1],[x['driver_rad'] for x in cal][::-1]))

def tcp_offset_for_driver(q, manifest=None):
    m=manifest or json.loads(MANIFEST.read_text(encoding='utf-8'))
    cal=m['opening_calibration']; grid=[r['driver_rad'] for r in cal]
    return np.array([np.interp(q,grid,[r['tcp_in_gripper_base_m'][i] for r in cal]) for i in range(3)])

def bullet_indices(p, body):
    joint_map={};link_map={}
    for i in range(p.getNumJoints(body)):
        info=p.getJointInfo(body,i);joint_map[info[1].decode()]=i;link_map[info[12].decode()]=i
    return joint_map,link_map

def reset_gripper(p, body, q, manifest=None):
    """Reset only for initialization/static structure checks, never during a grasp."""
    m=manifest or json.loads(MANIFEST.read_text(encoding='utf-8'))
    joints,_=bullet_indices(p,body)
    if not m['master_range_rad'][0]<=q<=m['master_range_rad'][1]:raise ValueError(q)
    p.resetJointState(body,joints[MASTER],q)
    for row in m['mimic_joints']:
        p.resetJointState(body,joints[row['joint']],q*row['multiplier']+row['offset'])

def install_mimic_constraints(p, body, max_force=20., manifest=None):
    """Enforce original mimic relationship with Bullet gear constraints; no physics steps."""
    m=manifest or json.loads(MANIFEST.read_text(encoding='utf-8'))
    joints,_=bullet_indices(p,body); master=joints[MASTER]; ids=[]
    for row in m['mimic_joints']:
        child=joints[row['joint']]
        p.setJointMotorControl2(body,child,p.VELOCITY_CONTROL,force=0)
        cid=p.createConstraint(body,master,body,child,p.JOINT_GEAR,[1,0,0],[0,0,0],[0,0,0])
        p.changeConstraint(cid,gearRatio=-1./row['multiplier'],maxForce=max_force,erp=.2)
        ids.append(cid)
    return ids

def command_gripper(p, body, q, force=10., velocity=.5, manifest=None):
    m=manifest or json.loads(MANIFEST.read_text(encoding='utf-8'))
    if not m['master_range_rad'][0]<=q<=m['master_range_rad'][1]:raise ValueError(q)
    joints,_=bullet_indices(p,body)
    p.setJointMotorControl2(body,joints[MASTER],p.POSITION_CONTROL,targetPosition=q,force=force,maxVelocity=velocity)

def inspect():
    import pybullet as p
    target=OUT/'structure_check.json'
    if target.exists():raise FileExistsError('Do not overwrite prior inspection')
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    cid=p.connect(p.DIRECT)
    try:
        if Path.cwd().resolve()!=ROOT:raise RuntimeError('Explicit project root cwd required for Bullet Windows Unicode workaround')
        body=p.loadURDF(URDF.relative_to(ROOT).as_posix(),useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE)
        joints,links=bullet_indices(p,body)
        rows=[]
        for i in range(p.getNumJoints(body)):
            info=p.getJointInfo(body,i)
            rows.append({'index':i,'joint':info[1].decode(),'type':info[2],'link':info[12].decode(),
                         'lower':info[8],'upper':info[9],'mass_kg':p.getDynamicsInfo(body,i)[0],
                         'collision_shape_count':len(p.getCollisionShapeData(body,i))})
        assert all(joints[x]>=0 for x in ARM_NAMES)
        assert all(len(p.getCollisionShapeData(body,links[x]))>0 for x in PADS)
        checks=[]
        for q in [0.,.2,.4,.6,.695]:
            reset_gripper(p,body,q,m)
            base=p.getLinkState(body,links[m['gripper_base_link']],computeForwardKinematics=True)
            invp,invq=p.invertTransform(base[4],base[5])
            centers=[]
            for name in PADS:
                state=p.getLinkState(body,links[name],computeForwardKinematics=True)
                pos,orn=p.multiplyTransforms(invp,invq,state[4],state[5]);centers.append(pos)
            expected=opening_geometry(ET.parse(URDF).getroot(),q)
            err=float(np.max(np.abs(np.array(centers)-expected['pad_centers_in_gripper_base_m'])))
            assert err<2e-6,err
            checks.append({'q':q,'max_pad_center_error_m':err,'gap_m':expected['inner_pad_gap_m'],
                           'tcp_in_base_m':expected['tcp_in_gripper_base_m']})
        reset_gripper(p,body,0,m)
        gear_ids=install_mimic_constraints(p,body,manifest=m)
        command_gripper(p,body,0.,manifest=m)
        data={'status':'static_structure_pass','physics_steps':0,'grasp_episodes_executed':0,
              'arm_joint_indices':[joints[x] for x in ARM_NAMES],
              'arm_ee_link_index':links['tool0'],'gripper_base_link_index':links[m['gripper_base_link']],
              'master_joint_index':joints[MASTER],'pad_link_indices':[links[x] for x in PADS],
              'gear_constraints_created':len(gear_ids),'joint_table':rows,'fk_checks':checks,
              'pybullet_api_version':p.getAPIVersion(),
              'packages':{x:importlib.metadata.version(x) for x in ['pybullet','numpy','scipy','trimesh','xacro']},
              'urdf_sha256':sha(URDF),'note':'No time integration; dynamic mimic/contact admission belongs to root bounded episodes.'}
        dump(target,data)
        print(json.dumps({k:data[k] for k in ['status','arm_joint_indices','arm_ee_link_index','master_joint_index','pad_link_indices','gear_constraints_created','physics_steps']}))
    finally:p.disconnect(cid)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['export','inspect']);args=parser.parse_args()
    {'export':export,'inspect':inspect}[args.action]()
