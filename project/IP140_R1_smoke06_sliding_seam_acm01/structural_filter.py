"""Smoke06 exact thirteen physical pair exclusions and one active seam."""
from datetime import datetime, timezone
from math import isfinite
import numpy as np
import hashlib
from pathlib import Path
from mountpair import fixed_mount, audit_mount, _witness, registration
from seam_geometry import (ACTIVE, EPS, BONES, PADS, PAD_MOUNTS, PALM, check_row,
                           face_boxes, current_frames, inverse, rigid, SLIDING, check_manifold, row_issues, local_xyz)

CONTACT_NUMERIC_ZERO_M=EPS
CHAIN=("base_link_inertia","shoulder_link","upper_arm_link","forearm_link","wrist_1_link","wrist_2_link","wrist_3_link")
ARM_PAIRS=tuple(zip(CHAIN,CHAIN[1:]))
MOUNT=("wrist_3_link",PALM)
STRUCTURAL_PAIRS=(*ARM_PAIRS,MOUNT,*PAD_MOUNTS,*SLIDING)
INTERNAL_SEAMS=ACTIVE
QUERY_M=.002
DT=1./240.

def utc(): return datetime.now(timezone.utc).isoformat()
def ids_for(links,pair):
    a,b=(links[name] for name in pair)
    if a==b or a<0 or b<0: raise ValueError('invalid named link ids')
    return a,b
def matches(row,body,ids):
    return row[1]==body and row[2]==body and (row[3],row[4]) in (ids,ids[::-1])
def closest(p,body,ids,cid):
    return p.getClosestPoints(body,body,QUERY_M,linkIndexA=ids[0],linkIndexB=ids[1],physicsClientId=cid)

def audit_seams(p,body,link_by_name,client_id,manifolds,require_pad_witness=False,*,
                manifold_frames=None,current_frames_w=None,dt=None):
    issues=[];pairs={}
    frames=current_frames_w if current_frames_w is not None else current_frames(p,body,link_by_name,client_id)
    h=DT if dt is None else dt
    if h!=DT: raise ValueError('actual substep differs from 1/240')
    for pair in ACTIVE:
        ids=ids_for(link_by_name,pair);key='|'.join(pair)
        near=closest(p,body,ids,client_id)
        cm=[row for row in manifolds if matches(row,body,ids)]
        # Inner faces lie at y=0 in each pad link. Measure the actual normal gap in palm frame.
        r=frames[PALM][:3,1]
        face_centers=[frames[name][:3,3]+.030*frames[name][:3,0] for name in PADS]
        gap=float(r@(face_centers[1]-face_centers[0]))
        if not isfinite(gap):issues.append(key+':nonfinite_pad_gap')
        if not near and (cm or require_pad_witness or gap<=QUERY_M):issues.append(key+':missing_closest')
        ca=[check_row(pair,ids,r,frames,body,h) for r in near]
        ma=[check_manifold(pair,ids,r,body,h) for r in cm]
        for record in ma:
            record['classification']='zero_force_touching_manifold' if record['passed'] else 'rejected_active_manifold'
        for src,records in (('current_closest',ca),('solver_manifold',ma)):
            for i,record in enumerate(records):
                issues.extend(f'{key}:{src}:{i}:{v}' for v in record['violations'])
        pairs[key]=dict(link_ids=ids,actual_face_gap_m=gap,raw_points=near,manifolds=cm,closest_audit=ca,manifold_audit=ma,
                        residuals_m=[r.get('witness_residual_m') for r in ca],
                        min_signed_distance_m=min((r[8] for r in near),default=None))
    return dict(passed=not issues,violations=issues,pairs=pairs,numeric_zero_band_m=EPS,query_distance_m=QUERY_M)

def mounting_geometry(p,body,names,ids,cid):
    # getCollisionShapeData in this build is local to the inertial frame.
    shape_data={};issues=[]
    sign=-1 if names==PAD_MOUNTS[0] else 1
    for name,j,lateral,thickness in zip(names,ids,(sign*.006,sign*.001),(.008,.002)):
        rows=p.getCollisionShapeData(body,j,physicsClientId=cid)
        dyn=p.getDynamicsInfo(body,j,physicsClientId=cid)
        if len(rows)!=1 or tuple(rows[0][0:3])!=(body,j,p.GEOM_BOX):
            issues.append(name+':shape_or_identity');continue
        row=rows[0]; expected=np.array([.060,thickness,.020])
        T=rigid(dyn[3],dyn[4])@rigid(row[5],row[6])
        ref=np.eye(4);ref[:3,3]=(.030,lateral,0.)
        errors=(np.max(np.abs(np.asarray(row[3])-expected)),np.max(np.abs(T-ref)))
        if not all(float(x)<=EPS for x in errors):issues.append(name+':box_dimensions_or_pose')
        shape_data[name]=dict(actual_dimensions_m=row[3],actual_link_T_box=T.tolist(),
                          expected_dimensions_m=expected.tolist(),expected_link_T_box=ref.tolist(),
                          max_dimension_error_m=float(errors[0]),max_transform_error=float(errors[1]))
    return shape_data,issues

def audit_pad_mounts(p,body,links,cid,snapshot,world):
    results={}
    for side,pair in zip(('left','right'),PAD_MOUNTS):
        ids=ids_for(links,pair)
        joint=p.getJointInfo(body,ids[1],physicsClientId=cid)
        issues=[]
        if (joint[0],joint[1].decode(),joint[2],joint[12].decode(),joint[16])!=(
                ids[1],f'ip140_{side}_pad_mount',p.JOINT_FIXED,pair[1],ids[0]):
            issues.append('not_exact_fixed_child')
        relative=inverse(world[pair[0]])@world[pair[1]]
        if not np.max(np.abs(relative-np.eye(4)))<=EPS:issues.append('relative_pose_not_identity')
        shapes,shape_issues=mounting_geometry(p,body,pair,ids,cid);issues+=shape_issues
        query=snapshot['pairs']['|'.join(pair)]
        if not query['closest']:issues.append('no_explicit_closest')
        checks={}
        for src,raw in query.items():
            if src not in ('closest','manifolds'):continue
            checked=([check_row(pair,ids,r,world,body,DT) for r in raw] if src=='closest'
                     else [check_manifold(pair,ids,r,body,DT) for r in raw])
            if any(abs(r[8])>EPS for r in raw):issues.append(src+':mount_faces_not_tangent')
            for r in checked:issues.extend(src+':'+v for v in r['violations'])
            checks[src]=checked
        results[side]=dict(passed=not issues,violations=issues,names=pair,link_ids=ids,
                           joint_name=joint[1].decode(),joint_type=joint[2],parent_id=joint[16],
                           actual_bone_T_pad=relative.tolist(),registered_faces_local_m=face_boxes(pair),
                           collision_shapes=shapes,witness_audit=checks,raw_witness_records=query)
    return results

def audit_scene_manifolds(body,link_by_name,manifolds):
    structural={frozenset(ids_for(link_by_name,p)):p for p in STRUCTURAL_PAIRS}
    allowed=frozenset(ids_for(link_by_name,PADS))
    violations=[]
    for i,row in enumerate(manifolds):
        if len(row)<5:
            violations.append('unregistered_empty_scene_manifold:incomplete_identity')
            continue
        ids=frozenset((row[3],row[4]))
        if row[1]==body and row[2]==body:
            if ids in structural:
                violations.append('filtered_manifold:'+'|'.join(structural[ids]))
                continue
            if ids==allowed:continue
        violations.append(f'unregistered_empty_scene_manifold:{i}:{row[1:5]}')
    return dict(passed=not violations,violations=violations)

def slide_coordinates(p,body,links,cid):
    rows=p.getJointStates(body,[links[n] for n in BONES],physicsClientId=cid)
    return {side:row[0] for side,row in zip(('left','right'),rows)}

def joint_transform(pair,world,q):
    relative=inverse(world[pair[0]])@world[pair[1]]
    expected=np.eye(4)
    if pair==MOUNT:
        expected=np.asarray(registration()['T_wrist_G'])
    elif pair in SLIDING:
        side='left' if pair[1] in (BONES[0],PADS[0]) else 'right'
        sign=-1 if side=='left' else 1
        expected[:3,3]=(-.060,sign*q[side],0.)
    elif pair not in PAD_MOUNTS:
        raise ValueError('no registered model for this transform')
    pe=float(np.max(np.abs(relative[:3,3]-expected[:3,3])))
    re=float(np.max(np.abs(relative[:3,:3]-expected[:3,:3])))
    return dict(passed=pe<=EPS and re<=EPS,actual=relative.tolist(),expected=expected.tolist(),
                translation_max_error_m=pe,rotation_element_max_error=re)

def audit_filtered(p,body,link_by_name,client_id,manifolds,current_frames_w=None,*,q_by_side=None):
    pairs={};issues=[]
    world=current_frames_w if current_frames_w is not None else current_frames(p,body,link_by_name,client_id)
    q=q_by_side if q_by_side is not None else slide_coordinates(p,body,link_by_name,client_id)
    for pair in STRUCTURAL_PAIRS:
        key='|'.join(pair);ids=ids_for(link_by_name,pair)
        cm=[r for r in manifolds if matches(r,body,ids)]
        pairs[key]=dict(link_ids=ids,manifolds=cm)
        if cm:issues.append('filtered_manifold:'+key)
        if pair in ARM_PAIRS:continue
        raw=closest(p,body,ids,client_id)
        transform=joint_transform(pair,world,q);checks=[]
        if not transform['passed']:issues.append(key+':assembly_transform')
        if not raw:issues.append(key+':missing_explicit_closest')
        for row in raw:
            if pair==MOUNT:
                faults=row_issues(row,body,ids)
                if faults:
                    audit=dict(passed=False,violations=faults)
                else:
                    audit=_witness(row,inverse(world[PALM]),ids)
                    if row[8]<-EPS:audit['violations'].append('penetration_below_numeric_band')
                    audit['passed']=not audit['violations']
            else:
                audit=check_row(pair,ids,row,world,body,DT)
            checks.append(audit)
            issues.extend(key+':'+v for v in audit['violations'])
        pairs[key].update(current_closest=raw,witness_audit=checks,relative_transform=transform)
    return dict(passed=not issues,violations=issues,pairs=pairs,actual_q_m=q,
                current_mount_closest=pairs['|'.join(MOUNT)]['current_closest'],
                W_T_G_at_current_closest=world[PALM].tolist(),mount_closest_role='dynamic hard gate; section 3 applies to all seven including wrist')

def static_slides(p,body,links,cid,snapshot,world,q):
    results={}
    for pair in SLIDING:
        key='|'.join(pair);ids=ids_for(links,pair)
        side=0 if pair[1] in (BONES[0],PADS[0]) else 1
        slide_id=links[BONES[side]]
        ji=p.getJointInfo(body,slide_id,physicsClientId=cid)
        violations=[]
        parent_ok=(ji[2],ji[16],ji[12].decode())==(p.JOINT_PRISMATIC,links[PALM],BONES[side])
        expected_axis=(0.,-1. if side==0 else 1.,0.)
        if (not parent_ok or np.max(np.abs(np.array(ji[13])-expected_axis))>EPS
                or abs(ji[8])>EPS or abs(ji[9]-.07)>EPS):violations.append('slide_topology_axis_or_limit')
        transform=joint_transform(pair,world,q)
        if not transform['passed']:violations.append('assembly_transform')
        raw=snapshot['pairs'][key]['closest']
        if not raw:violations.append('no_explicit_closest')
        checks=[check_row(pair,ids,r,world,body,DT) for r in raw]
        for rec in checks:violations.extend(rec['violations'])
        results[key]=dict(passed=not violations,violations=violations,relative_transform=transform,
                          q_m=q,slide_id=slide_id,axis=ji[13],limits=ji[8:10],registered_faces=face_boxes(pair),
                          explicit_closest_audit=checks)
    return results

def static_metadata(p,body,links,cid,snapshot):
    names=tuple(dict.fromkeys(n for pair in STRUCTURAL_PAIRS for n in pair))
    states=p.getLinkStates(body,[links[n] for n in names],computeForwardKinematics=1,physicsClientId=cid)
    frames={n:rigid(r[0],r[1])@inverse(rigid(r[2],r[3])) for n,r in zip(names,states)}
    metadata={}
    for n,r in zip(names,states):
        idx=links[n];ji=p.getJointInfo(body,idx,physicsClientId=cid)
        shapes=[]
        for shape in p.getCollisionShapeData(body,idx,physicsClientId=cid):
            kind=shape[2];rel=shape[4].decode() if isinstance(shape[4],bytes) else shape[4]
            geom=dict(id=shape[1],kind=kind,local_link_T_shape=(rigid(r[2],r[3])@rigid(shape[5],shape[6])).tolist(),
                      body=shape[0],actual_shape_dimensions_or_mesh_scale=shape[3],filename=rel)
            if kind==p.GEOM_MESH:
                geom['dimension_convention']='mesh scale (not physical box lengths)'
                asset=Path(rel).resolve()
                if not asset.is_relative_to(Path(__file__).resolve().parent.parent):
                    raise ValueError('mesh path outside frozen project')
                geom['mesh_sha256']=hashlib.sha256(asset.read_bytes()).hexdigest()
            elif kind==p.GEOM_BOX:
                geom['dimension_convention']='box side lengths, metres'
            shapes.append(geom)
        metadata[n]=dict(name=n,link_id=idx,joint_name=ji[1].decode(),joint_type=ji[2],parent_id=ji[16],
                         joint_axis=ji[13],limits=ji[8:10],local_joint_frame=ji[14:16],
                         actual_W_T_link=frames[n].tolist(),engine_world_AABB_m=p.getAABB(body,idx,physicsClientId=cid),
                         collision_shapes=shapes)
    for pair in STRUCTURAL_PAIRS:
        row=snapshot['pairs']['|'.join(pair)]
        row['link_records']=[metadata[n] for n in pair]
        row['relative_transform']=(inverse(frames[pair[0]])@frames[pair[1]]).tolist()
        if pair==MOUNT:
            row['registered_faces']=registration()
        elif pair not in ARM_PAIRS:
            row['registered_faces']=dict(frame='each endpoint link',bounds_m=face_boxes(pair))
        else:
            witnesses=[]
            for contact in (*row['closest'],*row['manifolds']):
                order=(5,6) if contact[3]==links[pair[0]] else (6,5)
                witnesses.append([local_xyz(frames[n],contact[k]).tolist() for n,k in zip(pair,order)])
            row['registered_faces']=dict(kind='articulated mesh assembly. No registered planar box face.',
                                          sampled_surface_witnesses_local_m=witnesses)
def _snapshot(p,body,links,cid,stage):
    p.performCollisionDetection(physicsClientId=cid)
    rows=p.getContactPoints(physicsClientId=cid)
    frames=current_frames(p,body,links,cid)
    out=dict(stage=stage,utc=utc(),physics_steps=0,W_T_G=frames[PALM].tolist(),
             W_T_link={k:v.tolist() for k,v in frames.items()},raw_manifolds=rows,pairs={})
    for pair in STRUCTURAL_PAIRS:
        ids=ids_for(links,pair)
        out['pairs']['|'.join(pair)]=dict(link_ids=ids,closest=closest(p,body,ids,cid),
                                  manifolds=[r for r in rows if matches(r,body,ids)])
    q=slide_coordinates(p,body,links,cid)
    out['sliding_mounts']=static_slides(p,body,links,cid,out,frames,q)
    static_metadata(p,body,links,cid,out)
    out['pad_mounts']=audit_pad_mounts(p,body,links,cid,out,frames)
    out['seam_monitor']=audit_seams(p,body,links,cid,rows,require_pad_witness=True,
                                  manifold_frames=frames,current_frames_w=frames,dt=DT)
    return out

def apply_and_verify(p,body,link_by_name,client_id,persist=None):
    def evidence(name,payload):
        if persist is not None:persist(name,payload)
    ids=[ids_for(link_by_name,pair) for pair in STRUCTURAL_PAIRS]
    unique={frozenset(i) for i in ids}
    if len(unique)!=13 or any(frozenset(ids_for(link_by_name,q)) in unique for q in ACTIVE):
        raise ValueError('inconsistent named pair registry')
    pre=_snapshot(p,body,link_by_name,client_id,'pre_filter')
    issues=[]
    for pair in ARM_PAIRS:
        a,b=ids_for(link_by_name,pair);ji=p.getJointInfo(body,b,physicsClientId=client_id)
        if ji[16]!=a or ji[12].decode()!=pair[1] or ji[2]!=p.JOINT_REVOLUTE:issues.append('arm_adjacency:'+str(pair))
    for pair,ab in zip(STRUCTURAL_PAIRS,ids):
        key='|'.join(pair)
        record=pre['pairs'][key]
        why=[]
        if not record['closest']:why.append('missing_explicit_closest')
        for src in ('closest','manifolds'):
            for i,r in enumerate(record[src]):
                if len(r)!=14 or any(len(r[k])!=3 for k in (5,6,7,11,13)):
                    why.append(f'{src}[{i}]:incomplete_record')
                    continue
                fields=(*r[:5],*r[5],*r[6],*r[7],*r[8:11],*r[11],r[12],*r[13])
                if not all(isinstance(x,(int,float)) and isfinite(x) for x in fields):why.append(f'{src}[{i}]:nonfinite')
                if not matches(r,body,ab):why.append(f'{src}[{i}]:pair_identity')
        record['record_validation']=dict(passed=not why,violations=why)
        issues.extend('pre:'+key+':'+v for v in why)
    pre['mount_topology']=fixed_mount(p,body,link_by_name,client_id)
    pre['mount_witness']=audit_mount(p,body,link_by_name,client_id,manifolds=pre['raw_manifolds'])
    evidence('filter_pre.json',pre)
    for key in ('mount_topology','mount_witness','seam_monitor'):
        if not pre[key]['passed']:issues.append('pre:'+key)
    for side,report in pre['pad_mounts'].items():
        if not report['passed']:issues.append('pre:pad_mount:'+side)
    for key,report in pre['sliding_mounts'].items():
        if not report['passed']:issues.append('pre:slide:'+key)
    ops=[]
    response=dict(passed=False,violations=issues,pre=pre,post=None,pair_ids=ids,setter_calls=ops)
    if issues:return response
    for seq,(pair,(a,b)) in enumerate(zip(STRUCTURAL_PAIRS,ids),1):
        entry=dict(sequence=seq,method='setCollisionFilterPair',pair=pair,args=[body,body,a,b,0],
                   kwargs={'physicsClientId':client_id},started_utc=utc())
        evidence(f'filter_setter_{seq:02d}_intent.json',entry)
        value=p.setCollisionFilterPair(body,body,a,b,0,physicsClientId=client_id)
        entry.update(returned=True,return_value=value,returned_utc=utc())
        ops.append(entry);evidence(f'filter_setter_{seq:02d}_returned.json',entry)
    post=_snapshot(p,body,link_by_name,client_id,'post_filter')
    for key,row in post['pairs'].items():
        if row['manifolds']:issues.append('residual_filtered_manifold:'+key)
    if not post['seam_monitor']['passed']:issues.append('post:seam_monitor')
    for side,report in post['pad_mounts'].items():
        if not report['passed']:issues.append('post:pad_mount:'+side)
    for key,report in post['sliding_mounts'].items():
        if not report['passed']:issues.append('post:slide:'+key)
    post['empty_scene_identity']=audit_scene_manifolds(body,link_by_name,post['raw_manifolds'])
    issues.extend('post:'+v for v in post['empty_scene_identity']['violations'])
    post['preservation_evidence']=dict(setter_count=len(ops),public_filter_getter=False,
        scope='exact thirteen named same-body setters; active opposing-pad pair and whole-world manifold monitoring')
    evidence('filter_post.json',post)
    response.update(passed=not issues,post=post)
    return response
