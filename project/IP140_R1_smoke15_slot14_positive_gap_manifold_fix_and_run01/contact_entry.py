"""Deterministic, zero-integration pad/target entry localization.

No connection is opened here.  Only a caller-owned static precheck connection
may be used.  Contact roots are geometric findings, never dynamics evidence or
commands for a controller.  The target body is never reset or moved.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from seam_geometry import BONES, PADS, PALM, POSE_LINKS, rigid, inverse, local_xyz

HERE=Path(__file__).resolve().parent
MODEL=HERE/"model/ur5e_ip140_r1.urdf"
MANIFEST=HERE/"model/manifest.json"
Q_MIN_M=0.
Q_MAX_M=.070
ROOT_BRACKET_TOL_M=1e-10
ROOT_DISTANCE_TOL_M=1e-9
GEOMETRY_TOL_M=1e-9
NORMAL_UNIT_TOL=1e-9
MAX_BISECTIONS=48
QUERY_DISTANCE_M=.20
TIP_EDGE_RISK_DISTANCE_M=.001


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe(value):
    """Keep nonfinite evidence explicit without emitting invalid JSON numbers."""
    if isinstance(value,np.ndarray):return _safe(value.tolist())
    if isinstance(value,np.generic):return _safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return {"nonfinite_raw":repr(value)}
    if isinstance(value,bytes):return {"bytes_hex":value.hex()}
    if isinstance(value,(list,tuple)):return [_safe(x) for x in value]
    if isinstance(value,dict):return {str(k):_safe(v) for k,v in value.items()}
    return value


def _write(path,value):
    with Path(path).open("x",encoding="utf8",newline="\n") as stream:
        json.dump(_safe(value),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def model_face_spec():
    """Verify rather than infer pad frames from names or hardware descriptions."""
    mf=json.loads(MANIFEST.read_text(encoding="utf8"))
    root=ET.parse(MODEL).getroot()
    spec={}
    for side,pad,bone,sign in zip(("left","right"),PADS,BONES,(-1,1)):
        component=mf["components"][pad]
        link=root.find(f"./link[@name='{pad}']")
        collision=link.find("collision")
        dims=np.fromstring(collision.find("geometry/box").get("size"),sep=" ")
        origin=np.fromstring(collision.find("origin").get("xyz"),sep=" ")
        rpy=np.fromstring(collision.find("origin").get("rpy"),sep=" ")
        joint=root.find(f"./joint[@name='ip140_{side}_pad_mount']")
        slide=root.find(f"./joint[@name='ip140_{side}_slide']")
        lim=slide.find("limit")
        if (not np.array_equal(dims,[.060,.002,.020])
            or not np.array_equal(origin,[.030,sign*.001,0.]) or np.any(rpy)
            or component["box_size_m"]!=dims.tolist()
            or component["center_in_link_m"]!=origin.tolist()
            or joint.get("type")!="fixed" or joint.find("parent").get("link")!=bone
            or joint.find("child").get("link")!=pad
            or joint.find("origin").get("xyz")!="0 0 0"
            or joint.find("origin").get("rpy")!="0 0 0"
            or not np.array_equal(np.fromstring(slide.find("origin").get("xyz"),sep=" "),[-.060,0.,0.])
            or np.any(np.fromstring(slide.find("origin").get("rpy"),sep=" "))
            or not np.array_equal(np.fromstring(slide.find("axis").get("xyz"),sep=" "),[0.,float(sign),0.])
            or float(lim.get("lower"))!=Q_MIN_M or float(lim.get("upper"))!=Q_MAX_M):
            raise ValueError("Frozen pad model identity/face definition differs")
        spec[side]=dict(pad_link=pad,bone_link=bone,
            effective_inner_face_pad_link_m={"x":[0.,.060],"y":[0.,0.],"z":[-.010,.010]},
            effective_inner_face_G_x_m=[-.060,0.],
            inward_normal_in_pad_link=[0.,-sign,0.],
            tip_plane_in_pad_link_x_m=.060,palm_end_in_pad_link_x_m=0.,
            tip_edge_risk_distance_m=TIP_EDGE_RISK_DISTANCE_M,
            tip_edge_risk_scope="descriptive static proximity only; not a dynamic gate or new success threshold")
    return dict(model_sha256=sha(MODEL),manifest_sha256=sha(MANIFEST),sides=spec,
        statement="The effective inner face is local y=0. Local y=+/-0.002 m is the fixed bone/pad back seam.")


def validate_contact_row(row,*,robot,target,pad_index,W_T_pad,W_T_G,side,require_face=False):
    """Normalize native endpoint ordering; validate exact body/link identity."""
    errors=[];result={"passed":False,"violations":errors}
    if (not isinstance(row,(list,tuple)) or len(row)!=14
        or any(not isinstance(row[k],(list,tuple)) or len(row[k])!=3 for k in (5,6,7,11,13))):
        errors.append("contact_row_shape");return result
    flat=(*row[:5],*row[5],*row[6],*row[7],*row[8:11],*row[11],row[12],*row[13])
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in flat):
        errors.append("contact_nonfinite");return result
    if side not in ("left","right"):
        errors.append("side_identity");return result
    forward=(row[1],row[2],row[3],row[4])==(robot,target,pad_index,-1)
    reverse=(row[1],row[2],row[3],row[4])==(target,robot,-1,pad_index)
    if not (forward or reverse):errors.append("pad_target_body_link_identity");return result
    # native n is normalOnB; pad-to-object normal reverses for pad-as-A.
    pad_point=np.asarray(row[5 if forward else 6],dtype=float)
    object_point=np.asarray(row[6 if forward else 5],dtype=float)
    n_native=np.asarray(row[7],dtype=float)
    n_pad_to_object=-n_native if forward else n_native
    norm_error=abs(float(np.linalg.norm(n_native))-1.)
    residual=float(np.linalg.norm(np.asarray(row[5])-np.asarray(row[6])-float(row[8])*n_native))
    if norm_error>NORMAL_UNIT_TOL:errors.append("normal_not_unit")
    if residual>GEOMETRY_TOL_M:errors.append("full_witness_residual")
    W_T_pad=np.asarray(W_T_pad,dtype=float);W_T_G=np.asarray(W_T_G,dtype=float)
    if (W_T_pad.shape!=(4,4) or W_T_G.shape!=(4,4)
        or not np.isfinite(W_T_pad).all() or not np.isfinite(W_T_G).all()):
        errors.append("frame_nonfinite_or_shape");return result
    point_local=local_xyz(W_T_pad,pad_point)
    point_G=local_xyz(W_T_G,pad_point)
    face_outside=max(0.,-point_local[0],point_local[0]-.060,abs(point_local[1]),abs(point_local[2])-.010)
    on_face=bool(face_outside<=GEOMETRY_TOL_M)
    if require_face and not on_face:errors.append("entry_witness_outside_registered_effective_face")
    inward=np.array([0.,1. if side=="left" else -1.,0.])
    n_pad=W_T_pad[:3,:3].T@n_pad_to_object
    inward_alignment=float(n_pad@inward)
    if require_face and inward_alignment<=0:errors.append("entry_normal_wrong_inner_face_direction")
    tip_depth=float(.060-point_local[0])
    result.update(passed=not errors,signed_distance_m=float(row[8]),native_endpoint_order="pad_object" if forward else "object_pad",
        pad_witness_world_m=pad_point.tolist(),object_witness_world_m=object_point.tolist(),
        pad_witness_pad_link_m=point_local.tolist(),pad_witness_G_m=point_G.tolist(),
        native_normal_on_B=n_native.tolist(),normal_pad_to_object_world=n_pad_to_object.tolist(),
        normal_pad_to_object_pad=n_pad.tolist(),normal_pad_to_object_G=(W_T_G[:3,:3].T@n_pad_to_object).tolist(),
        normal_unit_error=norm_error,full_witness_residual_m=residual,face_outside_m=float(face_outside),
        on_effective_inner_face=on_face,inward_normal_alignment=inward_alignment,
        tip_depth_m=tip_depth,tip_edge_risk=bool(tip_depth<=TIP_EDGE_RISK_DISTANCE_M),
        tip_edge_risk_scope="static descriptive flag; no dynamic success/failure meaning",
        raw_normal_and_tangential_force_N=[row[9],row[10],row[12]])
    return result


def find_contact_entry(p,client,robot,target,links,*,side,arm_joint_indices,arm_q,
                      q_clear_m,q_overlap_m,other_finger_q_m,output_dir,source_snapshot=None):
    """Locate one existing positive-to-nonpositive bracket, without dynamics.

    Caller must restore its original command-reference snapshot after return.
    Every sampled q and raw contact is retained.  Nonpad overlap at an initial
    deeply closed bracket endpoint is evidence, not an entry collision: the
    final localized root must have no nonpad manifold or negative clearance.
    No range/threshold is relaxed and no controller target is rewritten.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    result=dict(version=HERE.name,side=side,passed=False,status="not_started",
        robot_body_id=robot,target_body_id=target,
        pad_link_index=links[PADS[0 if side=="left" else 1]] if side in ("left","right") else None,
        phase="close",source_snapshot=source_snapshot,physics_step_calls=0,charged=False,
        geometric_entry_only=True,dynamic_contact_force_verified=False,
        caller_must_restore_command_reference=True,object_pose_modified=False,
        root_bracket_tolerance_m=ROOT_BRACKET_TOL_M,root_distance_tolerance_m=ROOT_DISTANCE_TOL_M,
        max_bisections=MAX_BISECTIONS,query_distance_m=QUERY_DISTANCE_M,
        q_clear_input_m=q_clear_m,q_overlap_input_m=q_overlap_m,
        other_finger_q_m=other_finger_q_m,queries_completed=0,bisections=0,
        model_face_spec=model_face_spec(),source_sha256=sha(__file__),violations=[])
    samples=[];query_path=out/"queries.jsonl.gz"
    stream=None
    try:
        if (side not in ("left","right") or len(arm_joint_indices)!=6 or len(arm_q)!=6
            or not all(math.isfinite(float(x)) for x in (*arm_q,q_clear_m,q_overlap_m,other_finger_q_m))
            or not Q_MIN_M<=q_overlap_m<=q_clear_m<=Q_MAX_M
            or not Q_MIN_M<=other_finger_q_m<=Q_MAX_M):
            raise ValueError("invalid_finite_registered_bounds")
        if isinstance(source_snapshot,dict):
            for key in ("previous_snapshot","current_snapshot","previous_signed_distance_m",
                        "current_signed_distance_m","precontact_clean"):
                if key in source_snapshot:result[key]=source_snapshot[key]
            if source_snapshot.get("phase","close")!="close":
                raise ValueError("entry_context_not_close")
            result["precontact_context_source"]="caller verified previous snapshot and current proxy; not inferred by root solver"
        if p.step_authorized is not False or p.counts.get("stepSimulation",0)!=0:
            raise ValueError("static_only_connection_required")
        if p.counts.get("setCollisionFilterPair",0)!=14:
            raise ValueError("requires_exactly_fourteen_existing_filters")
        kw=dict(physicsClientId=client)
        object_initial=p.getBasePositionAndOrientation(target,**kw)
        result["object_initial_pose"]=object_initial
        selected=0 if side=="left" else 1
        pad=PADS[selected];pad_index=links[pad]
        bone_indices=[links[n] for n in BONES]
        all_indices=list(arm_joint_indices)+bone_indices
        names=(*POSE_LINKS,"tool0") if "tool0" in links else POSE_LINKS
        stream=gzip.open(query_path,"xt",encoding="utf8",newline="\n")
        def query(q,label):
            seq_start=p.sequence+1
            target_q=[other_finger_q_m,other_finger_q_m];target_q[selected]=float(q)
            for index,value in zip(all_indices,[*arm_q,*target_q]):
                p.resetJointState(robot,index,targetValue=float(value),targetVelocity=0.,**kw)
            raw_states=p.getLinkStates(robot,[links[n] for n in names],computeForwardKinematics=1,**kw)
            frames={n:rigid(r[0],r[1])@inverse(rigid(r[2],r[3])) for n,r in zip(names,raw_states)}
            raw_closest=p.getClosestPoints(robot,target,QUERY_DISTANCE_M,**kw)
            # Each root query is transient closest-point geometry only.  A
            # deliberately overlapping reference must not seed the persistent
            # manifold cache subsequently observed by the proxy channel.
            manifolds=None
            pad_rows=[];nonpad=[];invalid=[]
            for i,row in enumerate(raw_closest):
                if not isinstance(row,(list,tuple)) or len(row)!=14:
                    invalid.append("malformed_robot_target_closest");continue
                roles={row[1]:row[3],row[2]:row[4]}
                if set(roles)!={robot,target} or roles[target]!=-1:
                    invalid.append("wrong_robot_target_closest_identity");continue
                if roles[robot]==pad_index:
                    check=validate_contact_row(row,robot=robot,target=target,pad_index=pad_index,
                        W_T_pad=frames[pad],W_T_G=frames[PALM],side=side)
                    pad_rows.append(dict(raw_index=i,raw=row,audit=check))
                    invalid.extend(check["violations"])
                elif roles[robot] not in [links[n] for n in PADS]:
                    if not all(math.isfinite(float(v)) for v in (*row[:5],*row[5],*row[6],*row[7],*row[8:11],*row[11],row[12],*row[13])):
                        invalid.append("nonfinite_nonpad_closest")
                    if row[8]<0:nonpad.append(dict(source="closest",raw_index=i,raw=row))
            if not pad_rows:invalid.append("missing_pad_target_closest")
            chosen=min(pad_rows,key=lambda r:(r["raw"][8],r["raw_index"])) if pad_rows else None
            object_now=p.getBasePositionAndOrientation(target,**kw)
            states=p.getJointStates(robot,all_indices,**kw)
            if object_now!=object_initial:invalid.append("object_pose_changed")
            if p.counts.get("stepSimulation",0)!=0 or p.step_authorized is not False:
                invalid.append("physical_step_or_authorization_forbidden")
            if p.counts.get("setCollisionFilterPair",0)!=14:invalid.append("filter_count_changed")
            record=dict(query_index=len(samples)+1,label=label,side=side,q_m=float(q),
                other_finger_q_m=other_finger_q_m,robot_arm_q_rad=list(arm_q),all_reset_targets=[*arm_q,*target_q],
                joint_states=states,raw_link_states=raw_states,world_link_frames=frames,
                object_pose=object_now,raw_robot_target_closest=raw_closest,raw_world_manifolds=manifolds,
                raw_manifolds_requested=False,persistent_manifold_evidence_observed=False,
                query_mode="fresh_closest_only; no global collision dispatch or cached-contact read",
                pad_closest_audit=pad_rows,selected_raw_index=None if chosen is None else chosen["raw_index"],
                signed_distance_m=None if chosen is None else chosen["raw"][8],
                nonpad_target_collisions=nonpad,violations=list(dict.fromkeys(invalid)),
                physics_step_calls=0,api_sequence_first=seq_start,api_sequence_last=p.sequence)
            stream.write(json.dumps(_safe(record),ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
            samples.append(record);result["queries_completed"]=len(samples)
            if invalid:raise ValueError(";".join(dict.fromkeys(invalid)))
            return record
        clear=query(float(q_clear_m),"input_clear_endpoint")
        overlap=query(float(q_overlap_m),"input_overlap_endpoint")
        if not clear["signed_distance_m"]>0 or not overlap["signed_distance_m"]<=0:
            result["status"]="not_bracketed";result["violations"].append("requires_d_clear_positive_d_overlap_nonpositive")
            return result
        for i in range(MAX_BISECTIONS):
            if clear["q_m"]-overlap["q_m"]<=ROOT_BRACKET_TOL_M:
                break
            middle=(clear["q_m"]+overlap["q_m"])/2.
            trial=query(middle,f"bisection_{i+1:02d}")
            result["bisections"]=i+1
            if trial["signed_distance_m"]>0:clear=trial
            else:overlap=trial
        # Repeat the clear-side root once as a final explicit evidence query;
        # this is a preregistered deterministic evaluation, not a retry.
        root=query(clear["q_m"],"final_clear_side_root")
        result["bracket"]={"q_clear_m":clear["q_m"],"q_overlap_m":overlap["q_m"],
            "d_clear_m":clear["signed_distance_m"],"d_overlap_m":overlap["signed_distance_m"],
            "width_m":clear["q_m"]-overlap["q_m"]}
        chosen=root["raw_robot_target_closest"][root["selected_raw_index"]]
        final=validate_contact_row(chosen,robot=robot,target=target,pad_index=pad_index,
            W_T_pad=root["world_link_frames"][pad],W_T_G=root["world_link_frames"][PALM],side=side,require_face=True)
        errors=list(final["violations"])
        if result["bracket"]["width_m"]>ROOT_BRACKET_TOL_M:errors.append("root_bracket_not_converged")
        if abs(root["signed_distance_m"])>ROOT_DISTANCE_TOL_M:errors.append("root_residual_exceeds_tolerance")
        if root["nonpad_target_collisions"]:errors.append("simultaneous_nonpad_target_collision_at_entry")
        result.update(passed=not errors,status="geometric_entry_localized" if not errors else "entry_not_admitted",
            violations=list(dict.fromkeys(errors)),q_entry_m=root["q_m"],entry_query_index=root["query_index"],
            entry_audit=final,nonpad_target_collisions_at_entry=root["nonpad_target_collisions"],
            entry_nonpad_gate="fresh closest geometric intersection only; actual proxy manifolds independently required",
            persistent_manifold_evidence_observed=False,
            endpoint_or_intermediate_nonpad_collision_queries=[r["query_index"] for r in samples if r["nonpad_target_collisions"]],
            endpoint_nonpad_semantics="All endpoint/intermediate overlaps retained. Only final localized entry establishes simultaneous-entry collision; deep closed reference endpoint is not a loaded trajectory.")
    except BaseException as exc:
        result.update(passed=False,status="contact_entry_exception",exception=repr(exc),traceback=traceback.format_exc())
        result["violations"].append(repr(exc))
    finally:
        if stream is not None:stream.close()
        result["physics_step_calls"]=int(p.counts.get("stepSimulation",0))
        result["filter_call_count"]=int(p.counts.get("setCollisionFilterPair",0))
        if query_path.exists():result["query_log_sha256"]=sha(query_path)
        _write(out/"result.json",result)
    return result
