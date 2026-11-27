"""Create one preregistered static stop and independently read its base and collision state.
This module supplies evaluation metadata and has no motion-control entry point.
"""
import math
import numpy as np


def _agrees(a, b, tol=1e-9):
    actual=np.asarray(a,dtype=float)
    target=np.asarray(b,dtype=float)
    return actual.shape==target.shape and bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(target))
                                              and np.allclose(actual,target,atol=tol,rtol=0.))


def build_and_read_fixture(p, client, robot, slot):
    f=slot["fixture"]
    goal=np.asarray(slot["T_W_G_m"],dtype=float)
    center_g=np.array([-.030,.030,0.])
    R=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    dims=np.array([.020,.010,.018])
    world=np.array([.50,-.03,.28])
    expected_props={"base_mass_kg":0., "lateral_friction":.6, "restitution":0.,
                    "rolling_friction":0.,"spinning_friction":0.,"margin_m":.0005}
    prereg={"full_dimensions":_agrees(f["full_sides_xyz_in_G_m"],dims),
            "mount_frame":_agrees(f["center_in_goal_G_m"],center_g),
            "world_position":_agrees(f["center_world_m"],world),
            "world_rotation":_agrees(f["R_W_fixture"],R),
            "target_frame_consistent":_agrees(goal[:3,:3]@center_g+goal[:3,3],world) and _agrees(goal[:3,:3],R),
            "frozen_parameters":all(f[k]==v for k,v in expected_props.items()),
            "fixed_non_target_stop":f["fixed_to_world"] is True and f["evaluation_class"]=="non_target_stop"}
    if not all(prereg.values()):
        return dict(passed=False,body_id=None,registration_checks=prereg,violations=[k for k,v in prereg.items() if not v])
    kw={"physicsClientId":client}
    extents=(np.asarray(f["full_sides_xyz_in_G_m"])/2).tolist()
    shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=extents,**kw)
    visual=p.createVisualShape(p.GEOM_BOX,halfExtents=extents,rgbaColor=[.94,.54,.12,1.],**kw)
    if shape<0 or visual<0:
        raise ValueError("Missing fixture collision or visual shape ID.")
    # The approved R is rotation +pi/2 about the world y axis, independently verified below.
    q=p.getQuaternionFromEuler((0.,math.pi/2,0.))
    body=p.createMultiBody(baseMass=f["base_mass_kg"],baseCollisionShapeIndex=shape,baseVisualShapeIndex=visual,
                           basePosition=f["center_world_m"],baseOrientation=q,**kw)
    if body<0 or body==robot:
        raise ValueError("Fixture base must have its own world body ID.")
    p.changeDynamics(body,-1,lateralFriction=f["lateral_friction"],restitution=f["restitution"],
                     rollingFriction=f["rolling_friction"],spinningFriction=f["spinning_friction"],
                     collisionMargin=f["margin_m"],**kw)
    tag_key="evaluation_class"
    tag=p.addUserData(body,tag_key,f[tag_key],**kw)
    tag_id=p.getUserDataId(body,tag_key,**kw)
    tag_value=p.getUserData(tag_id,**kw) if tag_id>=0 else None
    label=tag_value.decode("utf8") if isinstance(tag_value,bytes) else tag_value
    raw_dyn=p.getDynamicsInfo(body,-1,**kw)
    sh=p.getCollisionShapeData(body,-1,**kw)
    pos,quat=p.getBasePositionAndOrientation(body,**kw)
    rotation=np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3,3)
    ids=[p.getBodyUniqueId(i,**kw) for i in range(p.getNumBodies(**kw))]
    n=p.getNumJoints(body,**kw)
    vel=p.getBaseVelocity(body,**kw)
    aabb=p.getAABB(body,-1,**kw)
    checks={"world_contains_exactly_robot_and_fixture":len(ids)==2 and set(ids)=={robot,body},
            "whole_body_fixed_base":n==0 and raw_dyn[0]==0.,
            "label_on_registered_base":tag>=0 and tag_id==tag and label==f["evaluation_class"],
            "world_position":_agrees(pos,world),"world_rotation":_agrees(rotation,R),
            "size":len(sh)==1 and _agrees(sh[0][3],dims),
            "collision_identity":len(sh)==1 and sh[0][0]==body and sh[0][1]==-1 and sh[0][2]==p.GEOM_BOX,
            "collision_local_pose":len(sh)==1 and _agrees(sh[0][5],[0.,0.,0.]) and _agrees(
                np.asarray(p.getMatrixFromQuaternion(sh[0][6])).reshape(3,3),np.eye(3)),
            "zero_initial_velocity":_agrees(vel,[[0.,0.,0.],[0.,0.,0.]])}
    for key,index in {"base_mass_kg":0,"lateral_friction":1,"restitution":5,
                      "rolling_friction":6,"spinning_friction":7,"margin_m":11}.items():
        checks[key]=_agrees(raw_dyn[index],f[key],1e-12)
    return dict(passed=all(checks.values()),body_id=body,base_link=-1,collision_shape_id=shape,
                visual_shape_id=visual,shape_readback=sh,center_world_m=pos,orientation_xyzw=quat,
                R_W_fixture_readback=rotation.tolist(),raw_base_dynamics=raw_dyn,base_velocity=vel,
                scene_body_ids=ids,world_aabb_readback=aabb,fixture_joint_count=n,registration=f,
                checked_against_slot=slot["candidate_id"],registration_checks=prereg,checks=checks,
                evaluation_class_readback=label,user_data_id=tag_id,violations=[k for k,v in checks.items() if not v])


def fixture_initial_contacts(p, client, robot, fixture, manifolds, phase):
    """This receives the unfiltered world stream and obtains a second body-pair-specific readback."""
    external=[row for row in manifolds if row[1]!=robot or row[2]!=robot]
    pair_rows=p.getContactPoints(bodyA=robot,bodyB=fixture,physicsClientId=client)
    return dict(phase=phase,passed=not external and not pair_rows,world_external_manifolds=external,
                robot_fixture_manifolds=pair_rows,all_world_manifolds=manifolds,robot_id=robot,fixture_id=fixture)

