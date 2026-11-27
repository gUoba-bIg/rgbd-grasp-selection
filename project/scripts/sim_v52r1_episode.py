"""One budgeted engineering episode; no test labels in the action interface."""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet
from pybullet_utils.bullet_client import BulletClient
from scripts.sim_v52_common import ROOT, CONFIG, read, write, append, now, identity
CONFIG=ROOT/'configs/simulation_v52r1.json'
from scripts.sim_v52_math import grasp_to_tool, observed_proposal, unproject, angle_error
from scripts.sim_v52_robot_assets import bullet_indices, reset_gripper, install_mimic_constraints, command_gripper, driver_for_width, tcp_offset_for_driver

class ExecutionRejected(Exception):
    pass

def load_robot(client, cfg, manifest, dynamic):
    flags = client.URDF_USE_INERTIA_FROM_FILE | client.URDF_USE_SELF_COLLISION | client.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    body = client.loadURDF(cfg['robot_urdf'], basePosition=[0,0,0], useFixedBase=True, flags=flags)
    joints, links = bullet_indices(client, body)
    arm = [joints[name] for name in manifest['arm_joint_names']]
    for index, value in zip(arm, cfg['home_joints']):
        client.resetJointState(body,index,value)
    reset_gripper(client,body,0.,manifest)
    pads = [links[name] for name in manifest['pad_links']]
    for pad in pads:
        client.changeDynamics(body,pad,lateralFriction=cfg['physics']['pad_friction'],restitution=0.,collisionMargin=.0005)
    pairs = cfg.get('allowed_self_collision_pairs', [])
    for a,b in pairs:
        client.setCollisionFilterPair(body,body,links[a] if a!='BASE' else -1,links[b] if b!='BASE' else -1,0)
    if dynamic:
        install_mimic_constraints(client,body,max_force=cfg['physics']['gear_force'],manifest=manifest)
        command_gripper(client,body,0.,force=cfg['physics']['gripper_torque'],manifest=manifest)
        client.setJointMotorControlArray(body,arm,client.POSITION_CONTROL,targetPositions=cfg['home_joints'],forces=cfg['physics']['arm_forces'])
    return dict(body=body, arm=arm, joints=joints, links=links, pads=pads, ee=links[manifest['gripper_base_link']])

def make_table(client, cfg):
    shape = client.createCollisionShape(client.GEOM_BOX,halfExtents=[.6,.6,.025])
    visual = client.createVisualShape(client.GEOM_BOX,halfExtents=[.6,.6,.025],rgbaColor=[.65,.65,.65,1])
    return client.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,baseVisualShapeIndex=visual,
                                  basePosition=[.45,0,cfg['table_z']-.025])

def tool_state(client, robot):
    row = client.getLinkState(robot['body'],robot['ee'],computeForwardKinematics=True)
    return np.asarray(row[4]), Rotation.from_quat(row[5]).as_matrix()

def plan_ik(client, robot, goal, cfg, start):
    for j,q in zip(robot['arm'],start):
        client.resetJointState(robot['body'],j,float(q))
    target_q = Rotation.from_matrix(goal[:3,:3]).as_quat().tolist()
    movable = sorted([(i,client.getJointInfo(robot['body'],i)) for i in range(client.getNumJoints(robot['body'])) if client.getJointInfo(robot['body'],i)[3]>=0],key=lambda item:item[1][3])
    # Solver qIndex order includes downstream gripper DOFs. Extract by qIndex.
    for _ in range(3):
        sol = client.calculateInverseKinematics(robot['body'],robot['ee'],goal[:3,3].tolist(),target_q,
                                                maxNumIterations=1000,residualThreshold=1e-6,jointDamping=[.005]*len(movable))
        by_index = {info[3]:float(value) for (_,info),value in zip(movable,sol)}
        qarm = [by_index[client.getJointInfo(robot['body'],j)[3]] for j in robot['arm']]
        for j,q in zip(robot['arm'],qarm):
            client.resetJointState(robot['body'],j,q)
        pos,rot=tool_state(client,robot)
        if np.linalg.norm(pos-goal[:3,3])<=cfg['criteria']['ik_position_m'] and angle_error(rot,goal[:3,:3])<=cfg['criteria']['ik_angle_rad']:
            break
    error = dict(position_m=float(np.linalg.norm(pos-goal[:3,3])), angle_rad=angle_error(rot,goal[:3,:3]))
    if error['position_m']>cfg['criteria']['ik_position_m'] or error['angle_rad']>cfg['criteria']['ik_angle_rad']:
        raise ExecutionRejected('ik_unreachable:'+json.dumps(error))
    for j,q in zip(robot['arm'],qarm):
        info=client.getJointInfo(robot['body'],j)
        if q<info[8]-1e-6 or q>info[9]+1e-6:
            raise ExecutionRejected('joint_limit')
    return qarm,error

def planned_collision(client, robot, table, start, end, cfg):
    for t in np.linspace(0,1,41):
        for j,q in zip(robot['arm'],(1-t)*np.asarray(start)+t*np.asarray(end)):
            client.resetJointState(robot['body'],j,float(q))
        client.performCollisionDetection()
        for contact in client.getContactPoints(bodyA=robot['body'],bodyB=table):
            if contact[3] not in cfg['base_table_allowed_link_indices'] and contact[8]<-cfg['criteria']['penetration_m']:
                return 'planned_table_collision'
        for contact in client.getContactPoints(bodyA=robot['body'],bodyB=robot['body']):
            if contact[8]<-cfg['criteria']['penetration_m']:
                return 'planned_self_collision'
    return None

def run_episode(cfg, slot, directory):
    started=time.monotonic()
    manifest=read(ROOT/cfg['robot_manifest'])
    p=BulletClient(connection_mode=pybullet.DIRECT)
    planner=None
    result=dict(at=now(),slot=slot,success=False,grasp_formed=False,slipped=False,forbidden_collision=False,
                outcome='not_started',expected_control=slot['kind'],physics_steps=0,
                actual_neural_forwards=0,policy='observation_geometry_engineering_only',
                new_object_attachment_constraints=0,object_teleports_after_initialization=0,
                config_identity=identity(CONFIG),source_identity=identity(__file__))
    trace=[]
    collisions=[]
    try:
        dt=cfg['physics']['time_step']
        p.setTimeStep(dt)
        p.setGravity(0,0,-9.81)
        p.setPhysicsEngineParameter(numSolverIterations=cfg['physics']['solver_iterations'],deterministicOverlappingPairs=1)
        table=make_table(p,cfg)
        robot=load_robot(p,cfg,manifest,True)
        asset=cfg['objects'][slot['object_id']]
        obj=p.loadURDF(asset['urdf'],basePosition=[*slot['xy'],cfg['table_z']+asset['height_m']/2+.003],
                       useFixedBase=False,flags=p.URDF_USE_INERTIA_FROM_FILE)
        p.changeDynamics(obj,-1,lateralFriction=cfg['physics']['object_friction'],restitution=0.,linearDamping=.04,angularDamping=.04)
        for _ in range(round(1/dt)):
            p.stepSimulation()
            result['physics_steps']+=1
        # Evaluation-only truth, never passed to observed_proposal or IK.
        initial_eval_position=np.array(p.getBasePositionAndOrientation(obj)[0])
        result['evaluation_initial_position']=initial_eval_position.tolist()
        camera=cfg['camera']
        view=p.computeViewMatrix(camera['eye'],camera['target'],camera['up'])
        proj=p.computeProjectionMatrixFOV(camera['fov'],camera['width']/camera['height'],camera['near'],camera['far'])
        frame=p.getCameraImage(camera['width'],camera['height'],view,proj,renderer=p.ER_TINY_RENDERER,
                                flags=p.ER_NO_SEGMENTATION_MASK)
        depth=np.asarray(frame[3]).reshape(camera['height'],camera['width'])
        rgba=np.asarray(frame[2],dtype=np.uint8).reshape(camera['height'],camera['width'],4)
        from PIL import Image
        Image.fromarray(rgba[:,:,:3]).save(directory/'initial_rgb.png')
        np.savez_compressed(directory/'observed_rgbd.npz',depth_buffer=depth,rgb=rgba[:,:,:3],view=view,projection=proj)
        cloud=unproject(depth,view,proj)
        candidate,observation=observed_proposal(cloud,slot['yaw'],cfg['table_z'],cfg['workspace'])
        if candidate[1]>manifest['opening_calibration'][0]['inner_pad_gap_m']:
            raise ExecutionRejected('gripper_width_infeasible')
        q_open=driver_for_width(float(candidate[1]),manifest)
        q_contact=driver_for_width(max(observation['estimated_object_width_m'],.002),manifest)
        pad_z=float(tcp_offset_for_driver(q_contact,manifest)[2])
        # Engineering proposal only. Respect the entire known robot envelope,
        # including pad half-length while closing, before constructing controls.
        preliminary=grasp_to_tool(np.eye(4),candidate,pad_z)
        minimum_tool_z=cfg['table_z']+cfg['engineering_repair']['downward_collision_envelope_m']+cfg['criteria']['contact_table_clearance_m']
        height_correction=max(0.,minimum_tool_z-float(preliminary[2,3]))
        candidate[15]+=height_correction
        result['engineering_proposal_height_adjustment_m']=height_correction
        result['minimum_tool_height_from_robot_geometry_m']=minimum_tool_z
        if slot['kind']=='miss':
            candidate[13]+=cfg['negative_controls']['miss_x_m']
        elif slot['kind']=='table_collision':
            # GraspNet x axis points down, so contact_z = translation_z-depth.
            candidate[15]=cfg['table_z']+cfg['negative_controls']['contact_z_relative_table_m']+candidate[3]
        goal=grasp_to_tool(np.eye(4),candidate,pad_z)
        result['observation']=observation
        result['engineering_height_basis']='Frozen robot collision envelope plus existing known-table clearance; no object truth and no historical P3 candidate edit'
        result['candidate_world']=candidate.tolist()
        result['goal_tool_world']=goal.tolist()
        result['estimated_contact_driver_rad']=q_contact
        result['requested_opening_m']=float(candidate[1])
        result['open_driver_rad']=q_open
        result['tcp_offset_for_estimated_contact_m']=pad_z
        write(directory/'policy_input.json',dict(candidate_world=candidate.tolist(),observation=observation,goal_tool_world=goal.tolist()),once=True)
        planner=BulletClient(connection_mode=pybullet.DIRECT)
        plan_table=make_table(planner,cfg)
        plan_robot=load_robot(planner,cfg,manifest,False)
        reset_gripper(planner,plan_robot['body'],q_open,manifest)
        pre=goal.copy();pre[2,3]+=cfg['motion']['pregrasp_m']
        transit=goal.copy();transit[2,3]=cfg['motion']['transit_tool_z']
        high=goal.copy();high[2,3]+=cfg['motion']['lift_m']
        current=list(cfg['home_joints'])
        paths=[]
        for name,target in [('transit',transit),('pregrasp',pre),('approach',goal),('lift',high)]:
            target_joints,err=plan_ik(planner,plan_robot,target,cfg,current)
            collision=planned_collision(planner,plan_robot,plan_table,current,target_joints,cfg)
            paths.append(dict(name=name,goal=target.tolist(),joints=target_joints,error=err,collision=collision))
            if collision:
                result['planning']=paths
                raise ExecutionRejected(collision)
            current=target_joints
        result['planning']=paths
        contact_center=candidate[13:16]+candidate[4:13].reshape(3,3)[:,0]*candidate[3]
        if contact_center[2]<cfg['table_z']+cfg['criteria']['contact_table_clearance_m']:
            # Same observed-candidate geometric gate for every arm/control label.
            result['known_table_geometric_guard']=True
            raise ExecutionRejected('target_contact_below_known_table')

        hold_streak=0
        bilateral_loss=0
        phase='ready'
        def tick():
            nonlocal hold_streak,bilateral_loss
            if time.monotonic()-started>cfg['limits']['episode_wall_seconds']:
                raise TimeoutError('episode_wall_limit')
            if result['physics_steps']*dt>cfg['limits']['episode_sim_seconds']:
                raise TimeoutError('episode_sim_limit')
            p.stepSimulation();result['physics_steps']+=1
            pos=np.array(p.getBasePositionAndOrientation(obj)[0])
            delta=float(pos[2]-initial_eval_position[2])
            contacts=p.getContactPoints(bodyA=robot['body'],bodyB=obj)
            pads=set()
            for c in contacts:
                active=c[9]>cfg['criteria']['contact_force_n'] or c[8]<-cfg['criteria']['penetration_m']
                if c[3] in robot['pads']:
                    if c[9]>cfg['criteria']['contact_force_n']:
                        pads.add(c[3])
                elif active:
                    collisions.append(dict(step=result['physics_steps'],kind='nonpad_object',link=c[3],force=c[9],distance=c[8]))
            for c in p.getContactPoints(bodyA=robot['body'],bodyB=table):
                if c[3] not in cfg['base_table_allowed_link_indices'] and (c[9]>cfg['criteria']['contact_force_n'] or c[8]<-cfg['criteria']['penetration_m']):
                    collisions.append(dict(step=result['physics_steps'],kind='robot_table',link=c[3],force=c[9],distance=c[8]))
            for c in p.getContactPoints(bodyA=robot['body'],bodyB=robot['body']):
                if c[9]>cfg['criteria']['contact_force_n'] or c[8]<-cfg['criteria']['penetration_m']:
                    collisions.append(dict(step=result['physics_steps'],kind='robot_self',links=[c[3],c[4]],force=c[9],distance=c[8]))
            if collisions:
                result['forbidden_collision']=True
            bilateral=len(pads)==2
            support=any(c[9]>cfg['criteria']['contact_force_n'] for c in p.getContactPoints(bodyA=obj,bodyB=table))
            if phase in ('lift','hold') and bilateral and not support and delta>=cfg['criteria']['grasp_formed_lift_m']:
                result['grasp_formed']=True
            bilateral_loss=0 if bilateral else bilateral_loss+1
            driver=p.getJointState(robot['body'],robot['joints'][manifest['master_joint']])[0]
            ee_pos,ee_rot=tool_state(p,robot)
            pad_center=ee_pos+ee_rot@tcp_offset_for_driver(driver,manifest)
            separation=float(np.linalg.norm(pos-pad_center))
            if result['grasp_formed'] and bilateral_loss*dt>=cfg['criteria']['slip_loss_seconds'] and (delta<cfg['criteria']['grasp_formed_lift_m'] or separation>cfg['criteria']['slip_separation_m']):
                result['slipped']=True
            valid_hold=phase=='hold' and delta>=cfg['criteria']['lift_m'] and not support and not result['forbidden_collision'] and result['grasp_formed'] and bilateral and not result['slipped']
            hold_streak=hold_streak+1 if valid_hold else 0
            result['final_continuous_hold_seconds']=hold_streak*dt
            result['final_bilateral_contact']=bilateral
            result['max_lift_m']=max(result.get('max_lift_m',0.),delta)
            result['max_continuous_hold_seconds']=max(result.get('max_continuous_hold_seconds',0.),hold_streak*dt)
            mimic_error=max(abs(p.getJointState(robot['body'],robot['joints'][r['joint']])[0]-(driver*r['multiplier']+r['offset'])) for r in manifest['mimic_joints'])
            result['max_mimic_error_rad']=max(result.get('max_mimic_error_rad',0.),mimic_error)
            result['actual_tcp_shift_from_nominal_max_m']=max(result.get('actual_tcp_shift_from_nominal_max_m',0.),abs(float(tcp_offset_for_driver(driver,manifest)[2])-pad_z))
            if result['physics_steps']%24==0:
                trace.append(dict(step=result['physics_steps'],phase=phase,object_eval_position=pos.tolist(),lift_m=delta,
                                  bilateral_contact=bilateral,supported=support,driver_rad=driver,mimic_error_rad=mimic_error))

        def move(target,seconds):
            start=np.array([p.getJointState(robot['body'],j)[0] for j in robot['arm']])
            steps=round(seconds/dt)
            for k in range(steps):
                a=(k+1)/steps;a=a*a*(3-2*a)
                q=(1-a)*start+a*np.array(target)
                p.setJointMotorControlArray(robot['body'],robot['arm'],p.POSITION_CONTROL,targetPositions=q.tolist(),
                    forces=cfg['physics']['arm_forces'],positionGains=[.08]*6,velocityGains=[1.]*6)
                tick()
            for _ in range(round(.3/dt)):
                tick()

        phase='opening'
        command_gripper(p,robot['body'],q_open,force=cfg['physics']['gripper_torque'],manifest=manifest)
        for _ in range(round(.8/dt)):
            tick()
        phase='transit';move(paths[0]['joints'],cfg['motion']['transit_seconds'])
        phase='pregrasp';move(paths[1]['joints'],cfg['motion']['pregrasp_seconds'])
        phase='approach';move(paths[2]['joints'],cfg['motion']['approach_seconds'])
        actual_pos,actual_rot=tool_state(p,robot)
        result['executed_approach_error']=dict(position_m=float(np.linalg.norm(actual_pos-goal[:3,3])),angle_rad=angle_error(actual_rot,goal[:3,:3]))
        if result['executed_approach_error']['position_m']>cfg['criteria']['ik_position_m'] or result['executed_approach_error']['angle_rad']>cfg['criteria']['ik_angle_rad']:
            raise ExecutionRejected('executed_pose_tracking_failed')
        phase='close'
        command_gripper(p,robot['body'],manifest['master_range_rad'][1],force=cfg['physics']['gripper_torque'],velocity=.5,manifest=manifest)
        for _ in range(round(cfg['motion']['close_seconds']/dt)):
            tick()
        phase='lift';move(paths[3]['joints'],cfg['motion']['lift_seconds'])
        phase='hold'
        for _ in range(round(cfg['motion']['hold_seconds']/dt)):
            tick()
        result['dynamic_coupling_passed']=result.get('max_mimic_error_rad',1e10)<=cfg['criteria']['mimic_error_rad']
        result['success']=result.get('final_continuous_hold_seconds',0)>=cfg['criteria']['hold_seconds'] and result.get('final_bilateral_contact',False) and not result['slipped'] and not result['forbidden_collision'] and result['dynamic_coupling_passed']
        result['outcome']='success' if result['success'] else ('forbidden_collision' if result['forbidden_collision'] else ('slip' if result['slipped'] else 'grasp_or_lift_failed'))
        if not result['dynamic_coupling_passed']:
            result['outcome']='dynamic_coupling_failed'
        end=p.getCameraImage(camera['width'],camera['height'],view,proj,renderer=p.ER_TINY_RENDERER,flags=p.ER_NO_SEGMENTATION_MASK)
        Image.fromarray(np.asarray(end[2],dtype=np.uint8).reshape(camera['height'],camera['width'],4)[:,:,:3]).save(directory/'final_rgb.png')
    except ExecutionRejected as exc:
        result['success']=False
        result['outcome']=str(exc)
        result['infeasible']=True
    except BaseException as exc:
        result['success']=False
        result['outcome']='engineering_exception'
        result['exception']=repr(exc)
        result['traceback']=traceback.format_exc()
    finally:
        result['wall_seconds']=time.monotonic()-started
        result['simulation_seconds']=result['physics_steps']*cfg['physics']['time_step']
        result['collision_events_count']=len(collisions)
        write(directory/'collision_events.json',collisions,once=True)
        write(directory/'trace.json',trace,once=True)
        if planner is not None:
            planner.disconnect()
        p.disconnect()
        write(directory/'result.json',result,once=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--slot',type=int,required=True)
    parser.add_argument('--attempt-dir',required=True)
    args=parser.parse_args()
    if not os.environ.get('SIM_V52_CONTROLLER_TOKEN'):
        raise RuntimeError('Use controller; all attempts must be budgeted before execution')
    directory=ROOT/args.attempt_dir
    if (directory/'result.json').exists():
        raise RuntimeError('No overwrite')
    config=read(CONFIG)
    result=run_episode(config,config['slots'][args.slot],directory)
    print(json.dumps(dict(outcome=result['outcome'],success=result['success'],wall_seconds=result['wall_seconds'])))
