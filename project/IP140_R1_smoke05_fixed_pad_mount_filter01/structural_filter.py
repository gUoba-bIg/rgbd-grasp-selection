"""Smoke05 exact nine physical pair exclusions and five active-seam watchdogs."""
from datetime import datetime, timezone
from math import isfinite
import numpy as np
from mountpair import fixed_mount, audit_mount
from seam_geometry import (ACTIVE, EPS, BONES, PADS, PAD_MOUNTS, PALM, check_row,
                           face_boxes, current_frames, inverse, rigid)

CONTACT_NUMERIC_ZERO_M=EPS
CHAIN=("base_link_inertia","shoulder_link","upper_arm_link","forearm_link","wrist_1_link","wrist_2_link","wrist_3_link")
ARM_PAIRS=tuple(zip(CHAIN,CHAIN[1:]))
MOUNT=("wrist_3_link",PALM)
STRUCTURAL_PAIRS=(*ARM_PAIRS,MOUNT,*PAD_MOUNTS)
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
        if not near and (pair!=PADS or require_pad_witness):issues.append(key+':missing_closest')
        ca=[check_row(pair,ids,r,frames,body,h) for r in near]
        if cm and manifold_frames is None: raise ValueError('solver collision sample needs its own frame')
        ma=[check_row(pair,ids,r,manifold_frames,body,h) for r in cm]
        for record in ma:
            record['classification']='zero_force_touching_manifold' if record['passed'] else 'rejected_active_manifold'
        for src,records in (('current_closest',ca),('solver_manifold',ma)):
            for i,record in enumerate(records):
                issues.extend(f'{key}:{src}:{i}:{v}' for v in record['violations'])
        pairs[key]=dict(link_ids=ids,raw_points=near,manifolds=cm,closest_audit=ca,manifold_audit=ma,
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
            checked=[check_row(pair,ids,r,world,body,DT) for r in raw]
            if any(abs(r[8])>EPS for r in raw):issues.append(src+':mount_faces_not_tangent')
            for r in checked:issues.extend(src+':'+v for v in r['violations'])
            checks[src]=checked
        results[side]=dict(passed=not issues,violations=issues,names=pair,link_ids=ids,
                           joint_name=joint[1].decode(),joint_type=joint[2],parent_id=joint[16],
                           actual_bone_T_pad=relative.tolist(),registered_faces_local_m=face_boxes(pair),
                           collision_shapes=shapes,witness_audit=checks,raw_witness_records=query)
    return results

def audit_filtered(p,body,link_by_name,client_id,manifolds,current_frames_w=None):
    pairs={};issues=[]
    world=current_frames_w if current_frames_w is not None else current_frames(p,body,link_by_name,client_id)
    for pair in STRUCTURAL_PAIRS:
        key='|'.join(pair);ids=ids_for(link_by_name,pair);cm=[r for r in manifolds if matches(r,body,ids)]
        pairs[key]=dict(link_ids=ids,manifolds=cm)
        if cm:issues.append(key+':filtered_manifold')
        if pair in PAD_MOUNTS:
            raw=closest(p,body,ids,client_id)
            checks=[check_row(pair,ids,r,world,body,DT) for r in raw]
            if not raw or any(abs(r[8])>EPS for r in raw):issues.append(key+':fixed_faces_not_touching')
            for r in checks:issues.extend(key+':'+v for v in r['violations'])
            pairs[key].update(current_closest=raw,witness_audit=checks)
    return dict(passed=not issues,violations=issues,pairs=pairs,
                current_mount_closest=closest(p,body,ids_for(link_by_name,MOUNT),client_id),
                W_T_G_at_current_closest=world[PALM].tolist(),
                mount_closest_role='poststep explicit static-geometry diagnostics; solver manifolds checked separately')

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
    out['pad_mounts']=audit_pad_mounts(p,body,links,cid,out,frames)
    out['seam_monitor']=audit_seams(p,body,links,cid,rows,require_pad_witness=True,
                                  manifold_frames=frames,current_frames_w=frames,dt=DT)
    return out

def apply_and_verify(p,body,link_by_name,client_id,persist=None):
    def evidence(name,payload):
        if persist is not None:persist(name,payload)
    ids=[ids_for(link_by_name,pair) for pair in STRUCTURAL_PAIRS]
    unique={frozenset(i) for i in ids}
    if len(unique)!=9 or any(frozenset(ids_for(link_by_name,q)) in unique for q in ACTIVE):
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
    post['preservation_evidence']=dict(setter_count=len(ops),public_filter_getter=False,
        scope='exact nine pair setters, active five seam and whole-world manifold monitoring, links separate')
    evidence('filter_post.json',post)
    response.update(passed=not issues,post=post)
    return response
