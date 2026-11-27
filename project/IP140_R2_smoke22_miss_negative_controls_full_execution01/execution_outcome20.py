"""Batch continuation uses returned physical observations and verified interface status."""
import re

# These are physical observations produced by the existing frozen evaluator.
# Unknown returns, bad identities, filter/geometry corruption and unrecognized errors stop the batch.
PHYSICAL_VIOLATIONS = frozenset({
    "external_penetration_below_minus_1mm","finger_speed_above_0.04_m_s","finger_motor_force_above_20_N",
    "nonpad_robot_object_contact","robot_table_contact","unregistered_self_manifold",
    "hold_tcp_drift","executed_approach_tracking","measured_slider_limit","pad_parallelism",
    "unloaded_midline_or_symmetry","preopening_not_maintained","opening_stability"
})

def verify_candidate_contact_identity(directory, episode):
    """Only known physical pairs, never a missing response or a filtering defect."""
    import gzip
    import json
    from contact_semantics import raw_contact_finite
    from static_identity import read
    zero=read(directory/"zero_readback.json")
    robot,target,table=(zero[k] for k in ("body_id","object_body_id","table_body_id"))
    links={r["index"] for r in read(directory/"zero_step/all_links_readback.json").values()} | {-1}
    mount=read(directory/"zero_step/all_links_readback.json")["base_link_inertia"]["index"]
    samples=directory/"episodes"/f"{episode['slot']:02d}"/"telemetry.jsonl.gz"
    last=None
    with gzip.open(samples,"rb") as stream:
        for line in stream:
            row=json.loads(line)
            if "raw_manifolds" in row:last=row
    if not (last and last["index"]==episode["step_returns"] and
            last["violations"]==episode["violations"] and last["raw_manifolds"]):
        raise ValueError("Returned collision lacks a matching observed physical state")
    for raw in last["raw_manifolds"]:
        if not raw_contact_finite(raw):raise ValueError("Collision lacks a finite native record")
        endpoints={(raw[1],raw[3]),(raw[2],raw[4])}
        if endpoints=={(robot,mount),(table,-1)}:raise ValueError("Filtered installation pair returned")
        for b,l in endpoints:
            if not ((b==robot and l in links) or (b in (target,table) and l==-1)):
                raise ValueError("Unregistered collision endpoint")

def returned_disposition(episode, api_steps):
    n=episode["step_returns"]
    if (type(n) is not int or not 1<=n<=4728 or episode["unknown_steps"] or n!=api_steps or
        episode["step_intents"]!=n or sum(episode["step_counts"].values())!=n):
        raise ValueError("Native physical/evaluator return count mismatch")
    if type(episode["grasp_success"]) is not bool or episode["kind"]!="miss":
        raise ValueError("Missing physical grasp result or unexpected trial kind")
    if episode['episode_completed'] is True and episode['engineering_gate_passed'] is True and n==4728:
        required=episode['grasp_success'] is (episode['kind']=='nominal')
        if episode['slot_passed'] is not required:raise ValueError("inconsistent complete result")
        return "engineering_complete"
    violations=episode.get("violations") or []
    def physical(k):
        return (k in PHYSICAL_VIOLATIONS or bool(re.fullmatch(r"arm_[0-5]\.(joint_range|joint_speed_limit|effort_saturated|hold_q|hold_dq)",k)))
    first=episode.get("evaluator_terminal_result",{}).get("first_failure") or {}
    if (violations and all(physical(k) for k in violations)
            and first.get("step")==n and first.get("violations")==violations
            and episode["failure_reason"]==repr(RuntimeError(f"earliest_violation_at_step_{n}:{violations!r}"))
            and episode["slot_passed"] is False and episode["engineering_gate_passed"] is False):
        return "physical_episode_terminated_on_frozen_gate"
    raise ValueError("Shared evaluator, interface or engineering gate anomaly")
    

def candidate_static_disposition(static, read, path):
    """A geometric rejection can continue only with a clean zero-step trace."""
    import gzip
    import json
    import math
    from static_identity import sha
    root_file=static/"left_right_contact_entry_roots.json"
    roots=read(root_file)["sides"]
    snapshot_file=static/"dual_channel_snapshots.jsonl.gz"
    fail=path.get("first_failure") or {}
    if (path["passed"] is not False or path.get("physics_step_calls")!=0 or path.get("filter_call_count")!=14
            or path.get("target_pose_modified") is not False or path.get("dual_channel_sha256")!=sha(snapshot_file)):
        return None
    proof=None
    errors = fail.get("violations") or []
    collision_reasons={"nonpad_target_collision","finger_independent_robot_target_collision","robot_table_collision",
                       "unregistered_self_collision","support_penetration_below_minus_1mm",
                       "pad_target_contact_before_close","proxy_penetration_below_minus_1mm"}
    if errors and all(issue.rsplit(":",1)[-1] in collision_reasons for issue in errors):
        proof=dict(reason="candidate_path_collision",gate_violations=errors)
    elif fail.get("phase")=="close" and errors:
        # A geometric surface rejection can generate secondary certificate and point failures.
        tails={"entry_surface_not_registered","entry_witness_outside_registered_effective_face",
             "entry_solver_not_passed","entry_witness_face_audit_missing_or_failed","entry_not_on_registered_surface",
             "pad_contact_without_legal_entry","missing_per_side_entry_certificate",
             "reference_negative_or_manifold_before_legal_entry"}
        if any(x.rsplit(":",1)[-1] not in tails for x in fail["violations"]):
            return None
        limited=[]
        for side,cert in roots.items():
            if cert.get("passed") is True:continue
            a=cert.get("entry_audit") or {}
            b=cert.get("bracket") or {}
            if (cert.get("status")!="entry_not_admitted" or cert.get("violations")!=["entry_surface_not_registered"]
                    or cert.get("nonpad_target_collisions_at_entry") or cert.get("current_snapshot")!=fail["snapshot"]
                    or cert.get("previous_snapshot")!=fail["snapshot"]-1 or not cert["previous_signed_distance_m"]>0>=cert["current_signed_distance_m"]
                    or not b["d_clear_m"]>0>=b["d_overlap_m"] or b["width_m"]>cert["root_bracket_tolerance_m"]
                    or not a.get("registered_surface") or a.get("full_witness_residual_m",1)>1e-9
                    or sha(static/"contact_entries"/side/"queries.jsonl.gz")!=cert["query_log_sha256"]):
                return None
            limited.append(dict(side=side,registered_surface=a["registered_surface"],q_entry_m=cert["q_entry_m"]))
        if limited:proof=dict(reason="first_geometric_crossing_outside_registered_R2_surface",rejected_entries=limited)
    if not proof:return None
    from static_admission_check import verify_static_native_api, PHASES
    from contact_semantics import validate_entry_certificate
    zero=read(static/"zero_readback.json")
    api_report=read(static/"full_api/summary.json")
    n=0
    links=read(static/"zero_step"/"all_links_readback.json")
    robot=path["body_robot"];target=path["body_object"]
    verify_static_native_api(static,zero,{k:v["index"] for k,v in links.items()},api_report)
    for side,cert in roots.items():
        folder=static/"contact_entries"/side
        if (cert!=read(folder/"result.json") or cert["query_log_sha256"]!=sha(folder/"queries.jsonl.gz")
                or cert.get("physics_step_calls")!=0 or cert.get("filter_call_count")!=14):
            return None
        if cert["passed"] and validate_entry_certificate(cert,robot=robot,target=target,
                                                        pad=links["ip140_"+side+"_pad"]["index"]):
            return None
    planned=[(phase,tick) for phase,count in PHASES for tick in range(1,count+1)]
    with gzip.open(snapshot_file,"rb") as stream:
        for n,line in enumerate(stream,1):
            r=json.loads(line)
            if r["snapshot"]!=n or r["physical_steps"]!=0 or r["charged"] is not False or (r["phase"],r["phase_tick"])!=planned[n-1]:
                return None
            clean=n<fail["snapshot"]
            if clean and (r["passed"] is not True or r["violations"]):return None
            if not clean and (r["passed"] is not False or r["violations"]!=fail["violations"]):return None
            for name in ("reference_channel","proxy_channel"):
                c=r[name]
                if clean and (c["passed"] is not True or c["violations"]):return None
            if not clean and proof["reason"]=="candidate_path_collision":
                # Bind violations to finite raw evidence with known native endpoints.
                from contact_semantics import raw_contact_finite
                valid_links={v["index"] for v in links.values()} | {-1}
                table=zero["table_body_id"]
                for issue in errors:
                    lane,source,index,reason=issue.split(":")
                    channel=r[lane+"_channel"]
                    rows=channel["raw_manifolds"] if source=="manifold" else channel["raw_closest"][source]
                    raw=rows[int(index)]
                    if not raw_contact_finite(raw):return None
                    for body,link in ((raw[1],raw[3]),(raw[2],raw[4])):
                        if not ((body==robot and link in valid_links) or (body in (table,target) and link==-1)):
                            return None
    if n!=path["snapshots_completed"] or n!=fail["snapshot"]:return None
    return dict(**proof,last_static_snapshot=n,first_failure=fail,available_roots=sorted(roots),
                native_static_api_sha256=sha(static/"full_api/calls.jsonl.gz"),root_summary_sha256=sha(root_file),
                dual_channel_snapshots_sha256=sha(snapshot_file),path_result_sha256=sha(static/"path_result.json"))
