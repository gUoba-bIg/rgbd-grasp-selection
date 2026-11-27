"""Batch continuation uses returned physical observations and verified interface status."""
import re

# These are physical observations produced by the existing frozen evaluator.
# Unknown returns, bad identities, filter/geometry corruption and unrecognized errors stop the batch.
PHYSICAL_VIOLATIONS = frozenset({
    "nonpad_robot_object_contact","robot_table_contact","other_robot_table_manifold","unregistered_self_manifold",
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
    
def outside_crossing(static, read, path):
    roots=read(static/"left_right_contact_entry_roots.json")["sides"]
    tick=(path.get("first_failure") or {}).get("snapshot")
    for side, record in roots.items():
        a=record.get("entry_audit",{})
        bracket=record.get("bracket",{})
        issues=record.get("violations")
        if (record.get("status")=="entry_not_admitted" and record.get("phase")=="close"
            and record.get("current_snapshot")==tick and record.get("previous_snapshot")==tick-1
            and record.get("previous_signed_distance_m",0)>0>=record.get("current_signed_distance_m",1)
            and bracket.get("d_clear_m",0)>0>=bracket.get("d_overlap_m",1)
            and issues==["entry_witness_outside_registered_effective_face"]
            and a.get("on_effective_inner_face") is False and a.get("face_outside_m",0)>1e-9):
            return dict(side=side,state=tick,q_entry_m=record["q_entry_m"],face_outside_m=a["face_outside_m"],
                        evidence=f"contact_entries/{side}/result.json")
    return None
