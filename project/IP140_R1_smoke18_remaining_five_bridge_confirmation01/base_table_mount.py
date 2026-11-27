"""Exact fixed-world base/table installation gate. All distances remain raw."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REG = json.loads((HERE / 'base_table_registration.json').read_bytes())
D_REF = REG['reference_signed_distance_m']
TOL = 1e-9
DT = REG['dt_s']
BASE_NAME = 'base_link_inertia'
FOOTPRINT = np.asarray(REG['base_collision_mesh_xy_ccw'])
FOOT_EDGES = np.roll(FOOTPRINT, -1, axis=0) - FOOTPRINT
FOOT_LENGTHS = np.linalg.norm(FOOT_EDGES, axis=1)


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (tuple, list, np.ndarray)):
        return all(finite(x) for x in value)
    return not isinstance(value, (int, float, np.number)) or math.isfinite(value)


def frame(position, quaternion):
    x,y,z,w = quaternion
    result = np.eye(4)
    result[:3,:3] = [[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                    [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                    [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]]
    result[:3,3] = position
    return result


def rotation_error(a,b):
    rotation = np.asarray(a)[:3,:3] @ np.asarray(b)[:3,:3].T
    skew = [rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],rotation[1,0]-rotation[0,1]]
    return float(math.atan2(np.linalg.norm(skew),np.trace(rotation)-1))


def pose_error(a,b):
    a,b = np.asarray(a),np.asarray(b)
    return dict(translation_m=float(np.linalg.norm(a[:3,3]-b[:3,3])),rotation_rad=rotation_error(a,b))


def within_footprint(point):
    delta = np.asarray(point[:2])-FOOTPRINT
    inward = (FOOT_EDGES[:,0]*delta[:,1]-FOOT_EDGES[:,1]*delta[:,0])/FOOT_LENGTHS
    return float(np.min(inward)) >= -TOL, float(np.min(inward))


def check_row(row,robot,table,base_index):
    """Role-normalize one native closest/contact tuple and verify the mount geometry."""
    errors=[]
    output=dict(raw=row)
    if (not isinstance(row,(tuple,list)) or len(row)!=14 or not finite(row)
            or any(not isinstance(row[i],(tuple,list)) or len(row[i])!=3 for i in (5,6,7,11,13))):
        return dict(passed=False,violations=['mount_nonfinite_or_malformed_row'],raw=row)
    if row[1:5] == [robot,table,base_index,-1] or tuple(row[1:5]) == (robot,table,base_index,-1):
        robot_point,table_point,normal = map(np.asarray,(row[5],row[6],row[7]))
    elif tuple(row[1:5]) == (table,robot,-1,base_index):
        robot_point,table_point = map(np.asarray,(row[6],row[5]))
        normal = -np.asarray(row[7])
    else:
        return dict(passed=False,violations=['mount_pair_identity'],raw=row)
    if base_index != 1 or robot==table or type(robot) is not int or type(table) is not int:
        errors.append('mount_registered_identity')
    distance=float(row[8])
    drift=abs(distance-D_REF)
    residual=float(np.linalg.norm(robot_point-table_point-distance*normal))
    normal_length=float(np.linalg.norm(normal))
    direction=float(np.linalg.norm(normal-[0.,0.,1.]))
    inside_robot,robot_xy = within_footprint(robot_point)
    inside_table,table_xy = within_footprint(table_point)
    if drift>TOL:errors.append('mount_distance_baseline')
    if residual>TOL:errors.append('mount_witness_residual')
    if abs(normal_length-1)>TOL:errors.append('mount_normal_unit')
    if direction>TOL:errors.append('mount_normal_direction')
    if (abs(table_point[2])>TOL or not -.15-TOL<=table_point[0]<=1.05+TOL
            or not -.6-TOL<=table_point[1]<=.6+TOL or not inside_table):
        errors.append('mount_table_installation_region')
    if abs(robot_point[2]-D_REF)>TOL or not inside_robot:
        errors.append('mount_base_installation_region')
    forces=np.asarray([row[9],row[10],row[12]],dtype=float)
    impulse=DT*(row[9]*np.asarray(row[7])+row[10]*np.asarray(row[11])+row[12]*np.asarray(row[13]))
    # Each signed impulse component and the vector magnitude are independently retained.
    impulse_norm=float(np.linalg.norm(impulse))
    if np.max(np.abs(forces))>1e-9:errors.append('mount_nonzero_contact_force')
    if impulse_norm>1e-12 or np.max(np.abs(forces*DT))>1e-12:
        errors.append('mount_nonzero_contact_impulse')
    output.update(passed=not errors,violations=errors,robot_point_W_m=robot_point.tolist(),
                  table_point_W_m=table_point.tolist(),normal_table_to_robot_W=normal.tolist(),
                  signed_distance_m=distance,baseline_drift_m=drift,witness_residual_m=residual,
                  normal_length=normal_length,normal_direction_error=direction,
                  base_footprint_inward_distance_m=robot_xy,table_footprint_inward_distance_m=table_xy,
                  raw_Fn_Ft1_Ft2_N=forces.tolist(),dt_s=DT,impulse_W_Ns=impulse.tolist(),
                  scalar_impulses_Ns=(forces*DT).tolist(),impulse_norm_Ns=impulse_norm)
    return output


def validate(observation):
    """Same pure gate for pre-filter, post-filter and returned physical states."""
    o=observation
    errors=[]; rows=[]
    if not isinstance(o,dict) or not finite(o):
        return dict(passed=False,violations=['mount_nonfinite_observation'],observation=o)
    try:
        stage=o['stage'];robot=o['robot'];table=o['table'];idx=o['base_index']
        if stage not in ('pre','post','dynamic'):errors.append('mount_unknown_stage')
        if type(robot) is not int or type(table) is not int or robot<0 or table<0 or robot==table or idx!=1:
            errors.append('mount_registered_identity')
        for name in ('base_W_T','table_W_T'):
            t=np.asarray(o[name],dtype=float)
            if (t.shape!=(4,4) or np.max(np.abs(t[3]-[0,0,0,1]))>TOL
                    or np.max(np.abs(t[:3,:3]@t[:3,:3].T-np.eye(3)))>TOL
                    or abs(np.linalg.det(t[:3,:3])-1)>TOL):
                errors.append('mount_invalid_transform')
        relative=np.linalg.inv(o['table_W_T'])@o['base_W_T']
        delta=pose_error(relative,REG['reference_table_T_base'])
        if delta['translation_m']>TOL or delta['rotation_rad']>TOL:errors.append('mount_fixed_transform')
        base_absolute=pose_error(o['base_W_T'],REG['reference_base_W_T'])
        table_absolute=pose_error(o['table_W_T'],REG['reference_table_W_T'])
        if any(v>TOL for t in (base_absolute,table_absolute) for v in t.values()):
            errors.append('mount_absolute_world_pose')
        reference=REG['baseline_observation']
        for name in ('root_pose','base_dynamics','table_dynamics','base_aabb','table_aabb'):
            actual=np.asarray(o[name],dtype=float) if name not in ('root_pose','base_dynamics','table_dynamics') else None
            # Nested native dynamics tuples have ragged shapes; recurse numerically.
            def same(a,b):
                if isinstance(b,(list,tuple)):
                    return isinstance(a,(list,tuple)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
                return isinstance(a,(int,float)) and abs(a-b)<=TOL
            if not same(o[name],reference[name]):errors.append('mount_frozen_'+name)
        if o['table_dynamics'][0]!=0:errors.append('mount_table_not_static')
        closest=o['closest'];manifolds=o['manifolds']
        if not closest:errors.append('mount_missing_closest')
        for row in closest:
            result=check_row(row,robot,table,idx);rows.append(result);errors.extend(result['violations'])
        mount_manifolds=[]
        for row in manifolds:
            if not isinstance(row,(list,tuple)) or len(row)!=14 or not finite(row):
                errors.append('mount_invalid_world_manifold');continue
            if set(row[1:3])=={robot,table}:
                result=check_row(row,robot,table,idx)
                if not result['passed']:errors.extend(result['violations'])
                if (tuple(row[1:5]) not in ((robot,table,idx,-1),(table,robot,-1,idx))):
                    errors.append('other_robot_table_manifold')
                else:mount_manifolds.append(result)
        if stage=='pre' and not mount_manifolds:errors.append('mount_missing_pre_manifold')
        if stage in ('post','dynamic') and mount_manifolds:errors.append('mount_filtered_manifold_reappeared')
        output=dict(observation=o,relative_table_T_base=relative.tolist(),fixed_transform_error=delta,
                    absolute_base_error=base_absolute,absolute_table_error=table_absolute,
                    closest_checks=rows,mount_manifold_checks=mount_manifolds,
                    margin_diagnostic_m=[r.get('signed_distance_m',D_REF)+o['base_dynamics'][11] for r in rows])
    except (KeyError,TypeError,ValueError,IndexError,np.linalg.LinAlgError) as exc:
        errors.append('mount_incomplete_observation:'+repr(exc));output=dict(observation=o)
    output.update(passed=not errors,violations=list(dict.fromkeys(errors)))
    return output


def capture(p,client,robot,table,links,stage,raw_manifolds=None):
    """Only query current values. No cached step-zero transforms or physics calls."""
    idx=links[BASE_NAME];kw=dict(physicsClientId=client)
    state=p.getLinkState(robot,idx,computeForwardKinematics=1,**kw)
    table_pose=p.getBasePositionAndOrientation(table,**kw)
    o=dict(stage=stage,robot=robot,table=table,base_index=idx,
           base_W_T=frame(state[4],state[5]).tolist(),table_W_T=frame(*table_pose).tolist(),
           root_pose=p.getBasePositionAndOrientation(robot,**kw),
           base_dynamics=p.getDynamicsInfo(robot,idx,**kw),table_dynamics=p.getDynamicsInfo(table,-1,**kw),
           base_aabb=p.getAABB(robot,idx,**kw),table_aabb=p.getAABB(table,-1,**kw),
           closest=p.getClosestPoints(robot,table,.01,linkIndexA=idx,linkIndexB=-1,**kw),
           manifolds=p.getContactPoints(**kw) if raw_manifolds is None else raw_manifolds)
    return validate(o)


def static_assets(p,client,robot,table,links):
    """Bind actual load flags, link topology, shape identity, and frozen asset hashes."""
    idx=links[BASE_NAME];kw=dict(physicsClientId=client);errors=[]
    expected_path=(ROOT/REG['base_collision_mesh_path']).resolve()
    digests={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in REG['source_sha256']}
    if digests!=REG['source_sha256']:errors.append('mount_source_hashes')
    infos=[p.getJointInfo(robot,i,**kw) for i in (0,idx)]
    if any(info[2]!=p.JOINT_FIXED for info in infos) or infos[0][16]!=-1 or infos[1][16]!=0:
        errors.append('mount_base_fixed_chain')
    shape=p.getCollisionShapeData(robot,idx,**kw);table_shapes=p.getCollisionShapeData(table,-1,**kw)
    if len(shape)!=1 or len(table_shapes)!=1:errors.append('mount_collision_count')
    else:
        a,b=shape[0],table_shapes[0]
        raw_name=a[4].decode('utf8') if isinstance(a[4],bytes) else a[4]
        mesh_identity=(ROOT/raw_name).resolve()
        if mesh_identity!=expected_path:errors.append('mount_base_mesh_path')
        if (a[0:3]!=(robot,idx,p.GEOM_MESH) or tuple(a[3])!=(1.,1.,1.)
                or not np.allclose(a[5],REG['base_collision_local_pose'][0],atol=1e-12,rtol=0)
                or not np.allclose(a[6],REG['base_collision_local_pose'][1],atol=1e-12,rtol=0)):
            errors.append('mount_base_shape_metadata')
        if (b[0:3]!=(table,-1,p.GEOM_BOX) or tuple(b[3])!=(1.2,1.2,.05)
                or tuple(b[5])!=(0.,0.,0.) or tuple(b[6])!=(0.,0.,0.,1.)):
            errors.append('mount_table_shape_metadata')
    robot_load=None;table_create=None;pending={}
    for row in p.audit_rows:
        if row[1]=='call' and row[3] in ('loadURDF','createMultiBody'):
            pending[row[0]]=row
        if row[1]=='return' and row[0] in pending:
            call=pending[row[0]]
            if call[3]=='loadURDF' and row[3]==robot:robot_load=call
            if call[3]=='createMultiBody' and row[3]==table:table_create=call
    if robot_load is None or robot_load[5].get('useFixedBase') is not True or robot_load[5].get('physicsClientId')!=client:
        errors.append('mount_robot_fixed_load')
    if table_create is None or table_create[5].get('baseMass')!=0 or table_create[5].get('physicsClientId')!=client:
        errors.append('mount_table_creation')
    return dict(passed=not errors,violations=errors,source_sha256=digests,
                robot_load_call=robot_load,table_create_call=table_create,base_joint_topology=infos,
                base_collision_shapes=shape,table_collision_shapes=table_shapes)


def apply_mount_filter(p,client,robot,table,links,persist):
    """One external filter is only reached after preserved pre-filter evidence passes."""
    p.performCollisionDetection(physicsClientId=client)
    assets=static_assets(p,client,robot,table,links)
    pre=capture(p,client,robot,table,links,'pre')
    persist('base_table_mount_assets.json',assets)
    persist('base_table_mount_pre.json',pre)
    if not assets['passed'] or not pre['passed']:
        return dict(passed=False,pre=pre,post=None,assets=assets,violations=assets['violations']+pre['violations'])
    if p.counts['setCollisionFilterPair']!=13 or p.returns['setCollisionFilterPair']!=13:
        raise ValueError('Expected exactly thirteen normal robot-only filters before mount filter')
    idx=links[BASE_NAME];kw=dict(physicsClientId=client);seq=p.sequence+1
    args=[robot,table,idx,-1,0]
    persist('base_table_filter_intent.json',dict(api_sequence=seq,args=args,kwargs=kw,physical_steps=p.counts['stepSimulation']))
    returned=p.setCollisionFilterPair(*args,**kw)
    persist('base_table_filter_return.json',dict(api_sequence=seq,returned=returned,normal_return=True))
    if returned is not None or p.counts['setCollisionFilterPair']!=14 or p.returns['setCollisionFilterPair']!=14:
        raise ValueError('Exact-fourteenth mount setter failed')
    p.performCollisionDetection(**kw)
    post=capture(p,client,robot,table,links,'post')
    persist('base_table_mount_post.json',post)
    return dict(passed=post['passed'],pre=pre,post=post,assets=assets,violations=post['violations'],
                filter_call=dict(api_sequence=seq,args=args,kwargs=kw,returned=returned))
