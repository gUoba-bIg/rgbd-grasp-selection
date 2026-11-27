"""Batch continuation uses returned physical observations and verified interface status."""
import re

# These are physical observations produced by the existing frozen evaluator.
# Unknown returns, bad identities, filter/geometry corruption and unrecognized errors stop the batch.
PHYSICAL_VIOLATIONS = frozenset({
    "external_penetration_below_minus_1mm","finger_speed_above_0.04_m_s","finger_motor_force_above_20_N",
    "hold_tcp_drift","executed_approach_tracking","measured_slider_limit","pad_parallelism",
    "unloaded_midline_or_symmetry","preopening_not_maintained","opening_stability"
})

def returned_disposition(episode, api_steps):
    n=episode["step_returns"]
    if (type(n) is not int or not 1<=n<=4728 or episode["unknown_steps"] or n!=api_steps or
        episode["step_intents"]!=n or sum(episode["step_counts"].values())!=n):
        raise ValueError("Native physical/evaluator return count mismatch")
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
    missing=[s for s in ("left","right") if s not in roots]
    no_entry=(fail=={"snapshot":3217,"phase":"lift","phase_tick":1,
              "error":"ValueError('both_registered_entries_required_before_lift')"}
        and path.get("snapshots_completed")==3216 and bool(missing) and not path.get("violations")
        and all(row.get("passed") is True for row in roots.values()))
    proof=None
    if no_entry:
        proof=dict(reason="required_bilateral_entry_not_observed_by_end_of_close",missing_sides=missing)
    elif fail.get("phase")=="close" and fail.get("violations"):
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
    n=0; seen={}; observations={}
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
            clean=no_entry or n<fail["snapshot"]
            if clean and (r["passed"] is not True or r["violations"]):return None
            if not clean and (r["passed"] is not False or r["violations"]!=fail["violations"]):return None
            for name in ("reference_channel","proxy_channel"):
                c=r[name]
                if clean and (c["passed"] is not True or c["violations"]):return None
                if no_entry and r["phase"]=="close":
                    if "robot_target" not in c["raw_closest"]:return None
                    for side in missing:
                        pad=links["ip140_"+side+"_pad"]["index"]
                        query=c["raw_closest"]
                        observations[name+"/"+side]=observations.get(name+"/"+side,0)+1
                        for block in ("robot_target","robot_target_supplemental"):
                            for x in query.get(block,[]):
                                if {(x[1],x[3]),(x[2],x[4])}!={(robot,pad),(target,-1)}:continue
                                if not math.isfinite(x[8]) or x[8]<=0:return None
                                k=name+"/"+side
                                seen[k]=min(seen.get(k,x[8]),x[8])
    if n!=path["snapshots_completed"] or (not no_entry and n!=fail["snapshot"]):return None
    return dict(**proof,last_static_snapshot=n,first_failure=fail,available_roots=sorted(roots),
                queried_missing_side_positive_minima_m=seen,close_states_with_query=observations,
                native_static_api_sha256=sha(static/"full_api/calls.jsonl.gz"),root_summary_sha256=sha(root_file),
                dual_channel_snapshots_sha256=sha(snapshot_file),path_result_sha256=sha(static/"path_result.json"))
