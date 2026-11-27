"""The registered fixed wrist/palm mounting-face and relative-pose checks."""
import hashlib
import json
import math
from seam_geometry import inverse
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAIR = ("wrist_3_link", "ip140_palm")
FLANGE = "flange"


@lru_cache(maxsize=1)
def registration():
    data = json.loads((HERE/"mountpair_registration.json").read_text(encoding="utf-8"))
    policy=data["stage_policy"]
    if (tuple(data["pair"]) != PAIR or policy["pre_filter_zero_step"]["wrist_palm_pair_filter_applied"] is not False
            or policy["post_filter_zero_step_and_dynamic"]["wrist_palm_pair_filter_applied"] is not True):
        raise ValueError("Invalid fixed mounting pair registration")
    if data.get("signed_distance_reference_m") != D_REF_M or data.get("signed_distance_tolerance_m") != D_TOL_M:
        raise ValueError("Unexpected mount baseline in registration")
    return data


def _frame(p, xyz, quat):
    t = np.eye(4)
    t[:3,:3] = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3,3)
    t[:3,3] = xyz
    return t


def _world(p, s):
    return _frame(p,s[0],s[1]) @ np.linalg.inv(_frame(p,s[2],s[3]))


def _frames(p, body, links, client_id):
    states=p.getLinkStates(body,[links[PAIR[0]],links[PAIR[1]]],computeForwardKinematics=1,
                           physicsClientId=client_id)
    wrist,palm=(_world(p,s) for s in states)
    wrist_to_palm=np.linalg.inv(wrist) @ palm
    residual=np.abs(wrist_to_palm-np.asarray(registration()["T_wrist_G"]))
    return np.linalg.inv(palm),wrist_to_palm,float(np.max(residual[:3,:3])),float(np.max(residual[:3,3]))


def fixed_mount(p, body, link_by_name, client_id):
    """Read both named fixed joint records and compare the full rigid transform."""
    reg=registration()
    violations=[]
    links=reg["required_fixed_path"]
    topology=[]
    for parent,child,expected in zip(links,links[1:],reg["required_joints"]):
        row=p.getJointInfo(body,link_by_name[child],physicsClientId=client_id)
        unit=dict(parent_index=row[16],child_index=row[0],child_link=row[12].decode(),
                  joint_name=row[1].decode(),joint_type=row[2])
        topology.append(unit)
        if (row[16]!=link_by_name[parent] or row[0]!=link_by_name[child] or
            row[12].decode()!=child or row[1].decode()!=expected or row[2]!=p.JOINT_FIXED):
            violations.append("not_registered_fixed_path:"+expected)
    chain_passed=not violations
    _,relative,ar,pr=_frames(p,body,link_by_name,client_id)
    transform_passed=bool(ar<=1e-9 and pr<=reg["floating_zero_m"] and np.all(np.isfinite(relative)))
    if not transform_passed:
        violations.append("wrist_palm_transform_changed")
    margins=[]
    for name,key in zip(PAIR,("wrist_margin_m","palm_margin_m")):
        actual=p.getDynamicsInfo(body,link_by_name[name],physicsClientId=client_id)[11]
        margins.append(actual)
        if abs(actual-reg[key])>1.e-12 or not np.isfinite(actual):
            violations.append("unexpected_collision_margin:"+name)
    return dict(passed=not violations,violations=violations,topology=topology,
                body_id=body,registered_pair_ids=[link_by_name[n] for n in PAIR],
                actual_T_wrist_G=relative.tolist(),rotation_max_error=ar,translation_max_error_m=pr,
                collision_margins_m=margins,fixed_chain_passed=chain_passed,relative_transform_passed=transform_passed,
                measurement_stage="pre_filter_zero_step",applicable_filter_policy=reg["stage_policy"]["pre_filter_zero_step"],
                registration_sha256=hashlib.sha256((HERE/"mountpair_registration.json").read_bytes()).hexdigest())


def _witness(row, inv_g, pair_ids):
    reg=registration()
    eps=reg["floating_zero_m"]
    # Normalize identities without dropping the original on-A/on-B normal conventions.
    palm_on_b=row[4]==pair_ids[1]
    wrist_i,palm_i=(5,6) if palm_on_b else (6,5)
    wrist=(inv_g@np.r_[row[wrist_i],1.])[:3]
    palm=(inv_g@np.r_[row[palm_i],1.])[:3]
    normal=inv_g[:3,:3]@np.asarray(row[7])*(1. if palm_on_b else -1.)
    d=float(row[8])
    wrist_lo,wrist_hi=np.asarray(reg["wrist_mesh_AABB_G_m"])
    wrist_lo=wrist_lo-reg["wrist_margin_m"]
    wrist_hi=wrist_hi+reg["wrist_margin_m"]
    extent=np.asarray(reg["palm_yz_half_m"])
    back=reg["palm_back_x_G"]
    wm=reg["palm_margin_m"]
    v=[]
    if not np.isfinite([d,row[9],row[10],row[12],*palm,*wrist,*normal]).all():
        v.append("nonfinite_mount_witness")
    if not back-eps<=palm[0]<=back+wm+eps:
        v.append("palm_witness_outside_rear_mating_surface")
    if not (wrist_lo[0]-eps<=wrist[0]<=wrist_hi[0]+eps):
        v.append("wrist_witness_outside_mesh_plus_existing_margin")
    for label,w in (("palm",palm),("wrist",wrist)):
        if (np.any(np.abs(w[1:])>extent+eps) or np.any(w[1:]<wrist_lo[1:]-eps)
                or np.any(w[1:]>wrist_hi[1:]+eps)):
            v.append(label+"_witness_outside_common_footprint")
    if not (-normal[0]>max(abs(normal[1]),abs(normal[2])) and
            abs(np.linalg.norm(normal)-1.)<=1e-9):
        v.append("contact_normal_not_on_rear_face")
    residual=float(np.linalg.norm(wrist-palm-normal*d))
    if not residual<=eps:
        v.append("inconsistent_separation_vector")
    return dict(passed=not v,violations=v,raw=row,world_ids=[row[1],row[2]],link_ids=[row[3],row[4]],
                witness_in_G_m=dict(wrist=wrist.tolist(),palm=palm.tolist()),
                normal_on_palm_in_G=normal.tolist(),signed_distance_m=d,
                normal_force_N=row[9],lateral_force_1_N=row[10],lateral_force_2_N=row[12],
                separation_vector_residual_m=residual)


def audit_mount(p,body,link_by_name,client_id,manifolds=None):
    """Check the registered face using both independent nearest query and solver rows."""
    inv_g,T,ar,pr=_frames(p,body,link_by_name,client_id)
    pair_ids=tuple(link_by_name[n] for n in PAIR)
    if len(set(pair_ids))!=2:
        raise ValueError("mount names must refer to two distinct links")
    closest=p.getClosestPoints(body,body,registration()["closest_query_m"],linkIndexA=pair_ids[0],
                              linkIndexB=pair_ids[1],physicsClientId=client_id)
    if manifolds is None:
        manifolds=p.getContactPoints(bodyA=body,bodyB=body,physicsClientId=client_id)
    def matches(row):
        return row[1]==body and row[2]==body and (row[3],row[4]) in (pair_ids,pair_ids[::-1])
    v=[]
    if not (pr<=registration()["floating_zero_m"] and ar<=1.e-9):
        v.append("fixed_mount_transform_changed")
    if not closest or not all(matches(row) for row in closest):
        v.append("missing_or_unexpected_exact_pair_witness")
    near=[_witness(row,inv_g,pair_ids) for row in closest if matches(row)]
    contacts=[_witness(row,inv_g,pair_ids) for row in manifolds if matches(row)]
    for group,items in (("closest",near),("manifold",contacts)):
        for i,entry in enumerate(items):
            v.extend(f"{group}:{i}:{why}" for why in entry["violations"])
    return dict(passed=not v,violations=v,closest=near,manifolds=contacts,
                transform_rotation_error=ar,transform_translation_error_m=pr,
                actual_T_wrist_G=T.tolist(),measurement_stage="pre_filter_zero_step",
                applicable_filter_policy=registration()["stage_policy"]["pre_filter_zero_step"])


D_REF_M = -0.0010361016226507268
D_TOL_M = 1e-9
POSE_TOL_M = 1e-9
POSE_TOL_RAD = 1e-9

def _chain_valid(topology,body,ids):
    """Validate the two engine joint records, including their common flange index."""
    if not topology or not isinstance(topology,dict) or tuple(topology.get('registered_pair_ids',()))!=ids:
        return False
    if topology.get('body_id')!=body or len(ids)!=2 or len(set(ids))!=2:
        return False
    joints=topology.get('topology',[])
    if len(joints)!=2 or not all(isinstance(j,dict) for j in joints):
        return False
    first,second=joints
    flange=first.get('child_index')
    if not isinstance(flange,int) or flange<0 or flange in ids:return False
    for row, parent, child, name, joint in (
            (first,ids[0],flange,FLANGE,'wrist_3-flange'),
            (second,flange,ids[1],PAIR[1],'ip140_mount')):
        if (row.get('parent_index'),row.get('child_index'),row.get('child_link'),
                row.get('joint_name'),row.get('joint_type'))!=(parent,child,name,joint,4):
            return False
    return topology.get('fixed_chain_passed') is True

def _proper_frame(T):
    return (T.shape==(4,4) and np.isfinite(T).all() and
            np.max(np.abs(T[3]-[0.,0.,0.,1.]))<=1e-9 and
            np.max(np.abs(T[:3,:3].T@T[:3,:3]-np.eye(3)))<=1e-9 and
            abs(np.linalg.det(T[:3,:3])-1.)<=1e-9)

def _complete_row(row,ids,body):
    if len(row)!=14 or any(not isinstance(row[k],(tuple,list,np.ndarray)) or len(row[k])!=3
                           for k in (5,6,7,11,13)):
        return ['contact_record_shape']
    fields=(*row[:5],*row[5],*row[6],*row[7],*row[8:11],*row[11],row[12],*row[13])
    issues=[]
    if not all(isinstance(v,(int,float,np.integer,np.floating)) and math.isfinite(v) for v in fields):
        issues.append('nonfinite_field')
    if (row[1],row[2])!=(body,body) or (row[3],row[4]) not in (ids,ids[::-1]):
        issues.append('pair_identity')
    return issues

def check_mount_invariant(body, ids, closest_rows, manifold_rows, W_T_wrist, W_T_G, topology, *, stage):
    """One measurement rule for the registered fixed mount; no simulator is used here."""
    if stage not in ('pre_filter_zero_step','post_filter_zero_step','dynamic'):
        raise ValueError('invalid mount gate stage')
    ids=tuple(ids);issues=[]
    chain=_chain_valid(topology,body,ids)
    if not chain:issues.append('fixed_chain_topology')
    if not closest_rows:issues.append('missing_closest')
    if stage!='pre_filter_zero_step' and len(manifold_rows)>0:issues.append('mount_solver_manifold')
    W_T_wrist=np.asarray(W_T_wrist,dtype=float);W_T_G=np.asarray(W_T_G,dtype=float)
    frame_ok=bool(_proper_frame(W_T_wrist) and _proper_frame(W_T_G))
    T=None;inv_g=None;te=None;angle=None
    if frame_ok:
        T=inverse(W_T_wrist)@W_T_G;inv_g=inverse(W_T_G)
        expected=np.asarray(registration()['T_wrist_G'])
        te=float(np.linalg.norm(T[:3,3]-expected[:3,3]))
        R=expected[:3,:3].T@T[:3,:3]
        # atan2 resolves angles near zero without rounding them to zero via acos.
        v=(R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1])
        angle=math.atan2(.5*math.hypot(*v),.5*(np.trace(R)-1.))
        if not te<=POSE_TOL_M:issues.append('fixed_transform_translation')
        if not angle<=POSE_TOL_RAD:issues.append('fixed_transform_rotation')
    else:
        issues.append('invalid_world_frame')
    near=[];pre_manifold=[]
    for source,rows,output in (('closest',closest_rows,near),('manifold',manifold_rows,pre_manifold)):
        for row in rows:
            problems=_complete_row(row,ids,body)
            detail=dict(raw=row)
            if not problems and frame_ok:
                detail=_witness(row,inv_g,ids)
                face_issues=list(detail['violations'])
                if any(v!='inconsistent_separation_vector' for v in face_issues):problems.append('mount_face')
                d=row[8];delta=d-D_REF_M
                resid=math.hypot(*(row[5][k]-row[6][k]-d*row[7][k] for k in range(3)))
                unit=abs(math.hypot(*row[7])-1.)
                if resid>D_TOL_M or 'inconsistent_separation_vector' in face_issues:problems.append('witness_residual')
                if unit>1e-9:problems.append('nonunit_normal')
                if source=='closest' and abs(delta)>D_TOL_M:problems.append('mount_distance_baseline')
                detail.update(delta_d_m=delta,full_witness_residual_world_m=resid,
                              normal_unit_error=unit,face_details=face_issues)
            elif not frame_ok:problems.append('invalid_world_frame')
            detail.update(passed=not problems,violations=problems)
            output.append(detail);issues.extend(problems)
    deviations=[entry['delta_d_m'] for entry in near if 'delta_d_m' in entry]
    return dict(passed=not issues,violations=list(dict.fromkeys(issues)),stage=stage,pair=PAIR,
                body_id=body,registered_pair_ids=ids,fixed_chain_passed=chain,
                fixed_joint_records=topology.get('topology',[]) if isinstance(topology,dict) else [],
                d_ref_m=D_REF_M,d_tolerance_m=D_TOL_M,
                delta_d_m=deviations[0] if len(deviations)==1 else None,
                distance_deviations_m=deviations,max_abs_distance_deviation_m=max(map(abs,deviations),default=None),
                translation_tolerance_m=POSE_TOL_M,rotation_tolerance_rad=POSE_TOL_RAD,
                translation_error_m=te,rotation_error_rad=angle,actual_T_wrist_G=None if T is None else T.tolist(),
                raw_closest=closest_rows,raw_manifolds=manifold_rows,witnesses=near,
                pre_or_unexpected_manifold_witnesses=pre_manifold)
