"""One-shot static command-reference path audit; never integrates dynamics.

Called only by the root launcher inside a dedicated zero-step connection.  A
motor target is not a predicted loaded joint state.  Therefore an overlap in
these snapshots blocks admission, but is not reported as a physical failure.
No scene object is moved and no command is altered using scene truth.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

from trajectory import HERE, ROOT, PLAN, PHASES, DT, fk, read, reference, sha
from seam_geometry import PALM, PADS, BONES, current_frames
from structural_filter import STRUCTURAL_PAIRS, audit_filtered, audit_seams
from base_table_mount import capture as capture_base_table


QUERY_CLEARANCE_M = .005
STATIC_NUMERIC_ZERO_M = 1e-9
EXTERNAL_PENETRATION_LOWER_M = -.001
EXPECTED_PHASE_STEPS = (("settle",240),("opening",480),("transit",672),
    ("pregrasp",672),("approach",672),("close",480),("lift",792),("hold",720))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,bytes): return value.decode("utf8")
    raise TypeError(type(value).__name__)


def _write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
                               default=_json)+"\n",encoding="utf8")


def _numeric_contact(row):
    try:
        values=[*row[:5],*row[5],*row[6],*row[7],row[8],row[9],row[10],
                *row[11],row[12],*row[13]]
        return len(row)>=14 and all(math.isfinite(float(v)) for v in values)
    except (TypeError,ValueError,IndexError):
        return False


def classify_contact(row, *, source, phase, robot, target, table, links):
    """Conservative static checks; preserve unmodified raw contact separately."""
    issues=[]
    if not _numeric_contact(row):
        return dict(classification="malformed_or_nonfinite",violations=["nonfinite_or_malformed_contact"])
    a,b,la,lb=map(int,(row[1],row[2],row[3],row[4]))
    d=float(row[8]);manifold=source=="manifold"
    pair=frozenset((la,lb))
    registered={frozenset((links[x],links[y])) for x,y in STRUCTURAL_PAIRS}
    pads={links[x] for x in PADS}
    if a==robot and b==robot:
        if la==lb:
            # Explicit same-shape closest query is not a physical contact pair.
            classification="same_link_closest_query" if not manifold else "invalid_same_link_manifold"
            if manifold: issues.append("same_link_manifold")
        elif pair in registered:
            classification="registered_internal_assembly"
            if manifold: issues.append("filtered_internal_pair_reappeared")
            # Exact baseline/witness gates are evaluated by audit_filtered.
        elif pair==frozenset(pads):
            classification="active_opposing_pads"
            # Full registered seam tests are evaluated by audit_seams.
        else:
            classification="unregistered_self_pair"
            if manifold or d < -STATIC_NUMERIC_ZERO_M:
                issues.append("unregistered_self_collision")
    elif {a,b}=={robot,table}:
        robot_link=la if a==robot else lb
        table_link=lb if b==table else la
        if robot_link==links["base_link_inertia"] and table_link==-1:
            classification="registered_fixed_world_mount"
            if manifold: issues.append("filtered_base_table_pair_reappeared")
            # Negative frozen baseline is checked by capture_base_table.
        else:
            classification="forbidden_robot_table"
            if manifold or d < -STATIC_NUMERIC_ZERO_M:
                issues.append("robot_table_collision")
    elif {a,b}=={robot,target}:
        robot_link=la if a==robot else lb
        target_link=lb if b==target else la
        if target_link!=-1:
            classification="invalid_target_link";issues.append("wrong_target_link")
        elif robot_link in pads:
            classification="pad_target_reference_geometry"
            if (manifold or d<0) and phase not in ("close","lift","hold"):
                issues.append("pad_target_contact_before_close")
        else:
            classification="forbidden_nonpad_target"
            if manifold or d < -STATIC_NUMERIC_ZERO_M:
                issues.append("nonpad_target_collision")
        if d < EXTERNAL_PENETRATION_LOWER_M:
            issues.append("external_penetration_below_minus_1mm")
    elif {a,b}=={target,table}:
        classification="registered_target_table_support_geometry"
        if la!=-1 or lb!=-1: issues.append("wrong_support_link")
        if d < EXTERNAL_PENETRATION_LOWER_M:
            issues.append("support_penetration_below_minus_1mm")
    else:
        classification="unknown_body_or_environment"
        issues.append("unknown_contact_identity")
    return dict(classification=classification,violations=issues,
                source=source,body_a=a,body_b=b,link_a=la,link_b=lb,
                signed_distance_m=d)


def scan_path(p,client,body,links,manifest,registration,gate,output_dir):
    """Read every planned command; reset robot only in this precheck session.

    The caller owns connect/disconnect.  This function never calls connect,
    loadURDF, stepSimulation, motor setters, object reset, or filter setters.
    First violated gate terminates the audit without consuming slot14.
    """
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    result_path=out/"result.json"
    telemetry_path=out/"path_snapshots.jsonl.gz"
    if result_path.exists() or telemetry_path.exists():
        raise FileExistsError("Static path evidence exists; no repeat/overwrite permitted")
    total=sum(n for _,n in EXPECTED_PHASE_STEPS)
    result=dict(version=HERE.name,slot=14,started_at=_utc(),passed=False,
        status="not_started",charged=False,physics_step_calls=0,
        command_reference_snapshots_planned=total,snapshots_checked=0,
        phases_checked={},first_failure=None,violations=[],
        reference_semantics="robot motor target snapshots; target object stays at initial pose",
        loaded_state_prediction=False,dynamics_validated=False,
        object_pose_used_to_change_control=False,
        conservative_rule="Any static hard-gate failure blocks admission. A loaded target overlap is not a simulated failure.",
        coverage_scope="Every reference snapshot up to first failure; no claim for unvisited snapshots or inter-sample continuum",
        clearance_query_radius_m=QUERY_CLEARANCE_M,
        source_sha256={"path_precheck.py":sha(Path(__file__)),"trajectory_plan.json":sha(PLAN)},
        planned_phase_steps=[list(x) for x in EXPECTED_PHASE_STEPS])
    phase_counts=Counter();classification_counts=Counter();minimum_clearances={}
    step=0;phase="admission";tick=0
    try:
        if p.step_authorized is not False: raise ValueError("precheck_step_authorized_must_be_false")
        if p.counts.get("stepSimulation",0)!=0: raise ValueError("precheck_has_physics_calls")
        if p.counts.get("setCollisionFilterPair",0)!=14 or p.returns.get("setCollisionFilterPair",0)!=14:
            raise ValueError("requires_exactly_14_normal_filters")
        if not gate.get("passed") or gate.get("physics_step_calls")!=0:
            raise ValueError("actual_zero_gate_not_passed")
        if tuple(PHASES)!=EXPECTED_PHASE_STEPS: raise ValueError("phase_schedule_changed")
        plan=read(PLAN)
        if plan.get("slot")!=14 or gate.get("trajectory_plan_sha256")!=sha(PLAN):
            raise ValueError("wrong_slot_or_unadmitted_trajectory")
        target=int(gate["object_body_id"]);table=int(gate["table_body_id"])
        kw=dict(physicsClientId=client)
        infos=[p.getJointInfo(body,j,**kw) for j in range(p.getNumJoints(body,**kw))]
        by_joint={row[1].decode():row[0] for row in infos}
        arm_names=read(ROOT/"data/simulation_v52/robot/robot_assets.json")["arm_joint_names"]
        arm=[by_joint[n] for n in arm_names];fingers=[links[n] for n in BONES]
        joint_indices=arm+fingers
        limits=np.asarray(plan["limits_rad_rad_s"],dtype=float)
        if limits.shape!=(6,3): raise ValueError("invalid_arm_limits")
        target_pose=p.getBasePositionAndOrientation(target,**kw)
        table_pose=p.getBasePositionAndOrientation(table,**kw)
        result.update(robot_body_id=body,target_body_id=target,table_body_id=table,
                      actual_client_id=client,robot_joint_indices=joint_indices,
                      target_initial_pose=target_pose,table_initial_pose=table_pose)
        with gzip.open(telemetry_path,"xt",encoding="utf8") as stream:
            for phase,count in EXPECTED_PHASE_STEPS:
                for tick in range(1,count+1):
                    step+=1
                    q,dq,qfinger=reference(plan,phase,tick)
                    q=np.asarray(q,dtype=float);dq=np.asarray(dq,dtype=float)
                    problems=[]
                    if q.shape!=(6,) or dq.shape!=(6,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)) or not math.isfinite(qfinger):
                        raise ValueError("nonfinite_or_malformed_reference")
                    if np.any(q<limits[:,0]) or np.any(q>limits[:,1]): problems.append("arm_reference_outside_limits")
                    if np.any(np.abs(dq)>limits[:,2]): problems.append("arm_reference_speed_outside_limits")
                    if not 0<=qfinger<=.070: problems.append("finger_reference_outside_command_limits")
                    if problems: raise ValueError(";".join(problems))
                    expected=fk(q)
                    for index,value in zip(joint_indices,[*q,float(qfinger),float(qfinger)]):
                        p.resetJointState(body,index,targetValue=float(value),targetVelocity=0.,**kw)
                    p.performCollisionDetection(**kw)
                    actual_states=p.getJointStates(body,joint_indices,**kw)
                    frames=current_frames(p,body,links,client)
                    translation_error=None;rotation_matrix_error=None
                    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(frames[PALM])):
                        problems.append("nonfinite_fk")
                    else:
                        translation_error=float(np.linalg.norm(frames[PALM][:3,3]-expected[:3,3]))
                        rotation_matrix_error=float(np.max(np.abs(frames[PALM][:3,:3]-expected[:3,:3])))
                        if translation_error>1e-9 or rotation_matrix_error>1e-9:
                            problems.append("engine_fk_pure_fk_mismatch")
                    raw=p.getContactPoints(**kw)
                    queries={"robot_self":p.getClosestPoints(body,body,QUERY_CLEARANCE_M,**kw),
                        "robot_target":p.getClosestPoints(body,target,QUERY_CLEARANCE_M,**kw),
                        "robot_table":p.getClosestPoints(body,table,QUERY_CLEARANCE_M,**kw),
                        "target_table":p.getClosestPoints(target,table,QUERY_CLEARANCE_M,**kw)}
                    filtered=audit_filtered(p,body,links,client,raw,current_frames_w=frames,
                        q_by_side={"left":actual_states[-2][0],"right":actual_states[-1][0]})
                    seams=audit_seams(p,body,links,client,raw,
                        require_pad_witness=2*qfinger<=.002,manifold_frames=frames,current_frames_w=frames,dt=DT)
                    mount=capture_base_table(p,client,body,table,links,"dynamic",raw_manifolds=raw)
                    problems.extend(filtered["violations"]+seams["violations"]+mount["violations"])
                    classifications=[]
                    for source,rows in [("manifold",raw),*queries.items()]:
                        for row_index,row in enumerate(rows):
                            c=classify_contact(row,source=source,phase=phase,robot=body,target=target,table=table,links=links)
                            c.update(row_index=row_index,query=source)
                            classifications.append(c);classification_counts[c["classification"]]+=1
                            problems.extend(f"{source}:{row_index}:{v}" for v in c["violations"])
                            if "signed_distance_m" in c and c["classification"]!="same_link_closest_query":
                                key=f'{c["classification"]}:{c["body_a"]}/{c["link_a"]}:{c["body_b"]}/{c["link_b"]}'
                                old=minimum_clearances.get(key)
                                if old is None or c["signed_distance_m"]<old["signed_distance_m"]:
                                    minimum_clearances[key]=dict(snapshot=step,phase=phase,phase_tick=tick,signed_distance_m=c["signed_distance_m"])
                    target_now=p.getBasePositionAndOrientation(target,**kw)
                    table_now=p.getBasePositionAndOrientation(table,**kw)
                    if target_now!=target_pose or table_now!=table_pose: problems.append("world_object_pose_changed_in_static_scan")
                    if p.counts.get("stepSimulation",0)!=0 or p.step_authorized is not False:
                        problems.append("static_scan_physics_forbidden")
                    if p.counts.get("setCollisionFilterPair",0)!=14 or p.returns.get("setCollisionFilterPair",0)!=14:
                        problems.append("filter_count_changed_during_scan")
                    unique=list(dict.fromkeys(problems))
                    record=dict(snapshot=step,phase=phase,phase_tick=tick,physics_step_calls=0,
                        arm_target_q_rad=q.tolist(),arm_reference_dq_rad_s=dq.tolist(),finger_target_q_m=qfinger,
                        reset_state_is_reference_not_loaded_prediction=True,actual_states_after_reset=actual_states,
                        expected_W_T_G=expected,actual_W_T_G=frames[PALM],
                        fk_translation_error_m=translation_error,fk_rotation_matrix_error=rotation_matrix_error,
                        raw_world_manifolds=raw,raw_closest=queries,contact_classifications=classifications,
                        filtered_internal_audit=filtered,active_pad_audit=seams,base_table_audit=mount,
                        target_pose=target_now,table_pose=table_now,passed=not unique,violations=unique)
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False,default=_json)+"\n")
                    stream.flush()
                    phase_counts[phase]+=1;result["snapshots_checked"]=step
                    if unique:
                        result["first_failure"]=dict(snapshot=step,phase=phase,phase_tick=tick,violations=unique,
                            telemetry_path=str(telemetry_path.relative_to(ROOT)),jsonl_record_1based=step,
                            loaded_target_reference_warning=phase in ("close","lift","hold"))
                        result["violations"]=unique
                        result["status"]="blocked_static_admission_unproven"
                        return result
            result["passed"]=True
            result["status"]="passed_full_command_reference_static_path"
    except BaseException as exc:
        result["status"]="blocked_static_precheck_exception"
        result["exception"]=repr(exc);result["traceback"]=traceback.format_exc()
        result["violations"].append(repr(exc))
        result["first_failure"]=dict(snapshot=step,phase=phase,phase_tick=tick,exception=repr(exc))
    finally:
        result["ended_at"]=_utc()
        result["phases_checked"]=dict(phase_counts)
        result["snapshots_remaining_unchecked"]=total-result["snapshots_checked"]
        result["classification_counts"]=dict(classification_counts)
        result["minimum_clearances_m"]=minimum_clearances
        result["physics_step_calls"]=int(p.counts.get("stepSimulation",0))
        result["filter_call_count"]=int(p.counts.get("setCollisionFilterPair",0))
        result["filter_normal_return_count"]=int(p.returns.get("setCollisionFilterPair",0))
        result["api_counts_at_path_end"]=dict(p.counts)
        if result["physics_step_calls"] or result["filter_call_count"]!=14:
            result["passed"]=False;result["violations"].append("forbidden_physics_or_filter_count")
        if telemetry_path.exists():
            result["telemetry_sha256"]=sha(telemetry_path)
            result["telemetry_bytes"]=telemetry_path.stat().st_size
        _write(result_path,result)
    return result
