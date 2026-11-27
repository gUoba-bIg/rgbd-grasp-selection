"""IP140 smoke04. Exactly seven named, same-body structural solver exclusions.
Queries never advance/reset the world. Every other link/body pair keeps its filter.
"""
from datetime import datetime, timezone
from math import dist, hypot, isfinite, sqrt
from mountpair import fixed_mount, audit_mount, _world

CONTACT_NUMERIC_ZERO_M = 1e-9
CHAIN = ("base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
         "wrist_1_link", "wrist_2_link", "wrist_3_link")
MOUNT = ("wrist_3_link", "ip140_palm")
STRUCTURAL_PAIRS = tuple(zip(CHAIN, CHAIN[1:])) + (MOUNT,)
PADS = ("ip140_left_pad", "ip140_right_pad")
INTERNAL_SEAMS = (PADS,) + tuple(("ip140_palm", f"ip140_{side}_{part}")
            for side in ("left", "right") for part in ("pad", "bone")) + tuple(
            (f"ip140_{side}_pad", f"ip140_{side}_bone") for side in ("left", "right"))
QUERY_M = .002


def utc():
    return datetime.now(timezone.utc).isoformat()


def ids_for(links, pair):
    a,b=(links[name] for name in pair)
    if a==b or a<0 or b<0:
        raise ValueError("non-distinct or invalid registered link id")
    return a,b


def matches(row, body, ids):
    return row[1]==body and row[2]==body and (row[3],row[4]) in (ids,ids[::-1])


def closest(p,body,ids,cid):
    return p.getClosestPoints(body,body,QUERY_M,linkIndexA=ids[0],linkIndexB=ids[1],physicsClientId=cid)


def witness_residual(row):
    # PyBullet normal is on B, directed towards A; all operands are world-frame.
    return sqrt(sum(((row[5][k]-row[6][k])-row[7][k]*row[8])**2 for k in range(3)))


def audit_seams(p,body,link_by_name,client_id,manifolds,require_pad_witness=False):
    """Raw distances are retained. The band applies only to these named seams."""
    pairs={}
    issues=[]
    for pair in INTERNAL_SEAMS:
        key="|".join(pair)
        ids=ids_for(link_by_name,pair)
        near=closest(p,body,ids,client_id)
        cm=[r for r in manifolds if matches(r,body,ids)]
        residuals=[]
        if not near and (pair!=PADS or require_pad_witness):
            issues.append(key+":missing_near_witness")
        for i,r in enumerate(near):
            residual=witness_residual(r)
            residuals.append(residual)
            if not matches(r,body,ids):
                issues.append(key+":wrong_witness_identity")
            if not all(isfinite(x) for x in (*r[5],*r[6],*r[7],r[8],r[9],r[10],r[12],residual)):
                issues.append(key+":nonfinite_witness")
            elif residual>CONTACT_NUMERIC_ZERO_M or abs(hypot(*r[7])-1.)>1e-9:
                issues.append(key+":invalid_witness_vector")
            if r[8]<-CONTACT_NUMERIC_ZERO_M:
                issues.append(key+":distance_below_numeric_band")
            if any(v!=0 for v in (r[9],r[10],r[12])):
                issues.append(key+":nonzero_contact_force_in_near_query")
        if cm:
            issues.append(key+":seam_manifold")
            if any(any(v!=0 for v in (r[9],r[10],r[12])) for r in cm):
                issues.append(key+":nonzero_seam_contact_force")
        pairs[key]=dict(link_ids=ids,raw_points=near,manifolds=cm,residuals_m=residuals,
                        min_signed_distance_m=min((r[8] for r in near),default=None))
    return dict(passed=not issues,violations=issues,pairs=pairs,
                numeric_zero_band_m=CONTACT_NUMERIC_ZERO_M,query_distance_m=QUERY_M)


def frame_of_g(p,body,links,cid):
    state=p.getLinkState(body,links[MOUNT[1]],computeForwardKinematics=1,physicsClientId=cid)
    return _world(p,state).tolist()


def audit_filtered(p,body,link_by_name,client_id,manifolds):
    """After filtering, never transform a cached solver manifold with a later FK.
    Current closest witnesses and current G frame are diagnostic only.
    """
    issues=[]
    pairs={}
    for pair in STRUCTURAL_PAIRS:
        ids=ids_for(link_by_name,pair)
        rows=[r for r in manifolds if matches(r,body,ids)]
        pairs["|".join(pair)]=dict(link_ids=ids,manifolds=rows)
        if rows:issues.append("filtered_manifold:"+"|".join(pair))
    mount_ids=ids_for(link_by_name,MOUNT)
    q=closest(p,body,mount_ids,client_id)
    return dict(passed=not issues,violations=issues,pairs=pairs,current_mount_closest=q,
                W_T_G_at_current_closest=frame_of_g(p,body,link_by_name,client_id),
                mount_closest_role="current pose diagnostic only; structural overlap persists geometrically")


def _snapshot(p,body,links,cid,stage):
    p.performCollisionDetection(physicsClientId=cid)
    rows=p.getContactPoints(physicsClientId=cid)
    out=dict(stage=stage,utc=utc(),physics_steps=0,W_T_G=frame_of_g(p,body,links,cid),
             witness_frame="raw contact and closest world positions; W_T_G obtained at this same unchanged zero-step state",
             raw_manifolds=rows,pairs={})
    for pair in STRUCTURAL_PAIRS:
        ids=ids_for(links,pair);key="|".join(pair)
        out["pairs"][key]=dict(link_ids=ids,closest=closest(p,body,ids,cid),
                               manifolds=[r for r in rows if matches(r,body,ids)])
    return out


def apply_and_verify(p,body,link_by_name,client_id,persist=None):
    def evidence(filename,payload):
        if persist is not None:persist(filename,payload)
    ids=[ids_for(link_by_name,pair) for pair in STRUCTURAL_PAIRS]
    if len({frozenset(ab) for ab in ids})!=7 or any(frozenset(ids_for(link_by_name,q)) in
           {frozenset(ab) for ab in ids} for q in INTERNAL_SEAMS):
        raise ValueError("filter list collides with itself or protected seams")
    pre=_snapshot(p,body,link_by_name,client_id,"pre_filter")
    for pair, ab in zip(STRUCTURAL_PAIRS[:-1],ids[:-1]):
        joint=p.getJointInfo(body,ab[1],physicsClientId=client_id)
        if joint[16]!=ab[0] or joint[12].decode()!=pair[1] or joint[2]!=p.JOINT_REVOLUTE:
            raise ValueError("arm adjacency not supported by model: "+"|".join(pair))
    pre["mount_topology"]=fixed_mount(p,body,link_by_name,client_id)
    pre["mount_witness"]=audit_mount(p,body,link_by_name,client_id,manifolds=pre["raw_manifolds"])
    pre["seam_monitor"]=audit_seams(p,body,link_by_name,client_id,pre["raw_manifolds"],require_pad_witness=True)
    evidence("filter_pre.json",pre)
    ops=[];issues=[]
    for key in ("mount_topology","mount_witness","seam_monitor"):
        if not pre[key]["passed"]:issues.append("pre:"+key)
    response=dict(passed=False,violations=issues,pre=pre,post=None,pair_ids=ids,setter_calls=ops)
    if issues:
        return response
    for seq,(pair,(a,b)) in enumerate(zip(STRUCTURAL_PAIRS,ids),1):
        entry=dict(sequence=seq,method="setCollisionFilterPair",pair=pair,
                   args=[body,body,a,b,0],kwargs={"physicsClientId":client_id},started_utc=utc())
        evidence(f"filter_setter_{seq:02d}_intent.json",entry)
        value=p.setCollisionFilterPair(body,body,a,b,0,physicsClientId=client_id)
        entry.update(returned=True,return_value=value,returned_utc=utc())
        ops.append(entry)
        evidence(f"filter_setter_{seq:02d}_returned.json",entry)
    post=_snapshot(p,body,link_by_name,client_id,"post_filter")
    for key,row in post["pairs"].items():
        if row["manifolds"]:
            issues.append("residual_filtered_manifold:"+key)
    post["seam_monitor"]=audit_seams(p,body,link_by_name,client_id,post["raw_manifolds"],require_pad_witness=True)
    if not post["seam_monitor"]["passed"]:issues.append("post:seam_monitor")
    post["preservation_evidence"]=dict(setter_count=len(ops),
        scope="only listed same-body pairs; entire-world manifold monitoring and protected-seam closest queries live",
        public_filter_getter=False,
        limitation="closest queries are diagnostic and do not, in isolation, prove broadphase enablement")
    evidence("filter_post.json",post)
    response.update(passed=not issues,post=post)
    return response

