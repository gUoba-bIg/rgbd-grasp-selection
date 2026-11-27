"""Narrow, instrumented exception for the rigid wrist-to-palm mounting face only.

No collision filtering or physics settings are changed here. Use the raw-contact
watchdog for all remaining (especially moving) pairs.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAIR = ("wrist_3_link", "ip140_palm")
FLANGE = "flange"


@lru_cache(maxsize=1)
def registration():
    data = json.loads((HERE/"mountpair_registration.json").read_text(encoding="utf-8"))
    if tuple(data["pair"]) != PAIR or data["solver_pair_filter_applied"]:
        raise ValueError("Invalid fixed mounting pair registration")
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
    _,relative,ar,pr=_frames(p,body,link_by_name,client_id)
    if not (ar<=1e-9 and pr<=reg["floating_zero_m"]):
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
                collision_margins_m=margins,physical_filter_set=False,
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
                actual_T_wrist_G=T.tolist(),solver_response_enabled=True)

