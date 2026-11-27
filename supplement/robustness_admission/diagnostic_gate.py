"""Diagnostic-only removal of the contact-entry certificate for the 40 sealed cases.
All calculations and original accusations remain in the snapshot stream. No native call is replaced.
"""
from collections import Counter
from common_v1 import need

ROOT_CERTIFICATE={
    "entry_surface_not_registered","root_residual_exceeds_tolerance","root_bracket_not_converged",
    "requires_d_clear_positive_d_overlap_nonpositive","entry_solver_not_passed",
    "entry_witness_face_audit_missing_or_failed","entry_signed_distance_residual","entry_not_on_registered_surface",
    "entry_root_distance_signs","entry_root_bracket_range_width","entry_root_bracket_missing",
    "entry_positive_to_contact_transition_unproven","no_clean_positive_to_nonpositive_entry",
}
PAD_CERTIFICATE={"pad_contact_without_legal_entry","missing_per_side_entry_certificate",
                 "reference_negative_or_manifold_before_legal_entry"}

def certificate_only(reason,phase,channel,links,ctx):
    if phase!="close":return False
    parts=reason.split(":")
    if len(parts)==2 and parts[0] in ("left","right"):
        return parts[1] in ROOT_CERTIFICATE
    if len(parts)!=3 or parts[-1] not in PAD_CERTIFICATE or not parts[1].isdigit():
        return False
    source,index=parts[0],int(parts[1])
    if source=="manifold":stream=channel["raw_manifolds"]
    elif source in ("robot_target","robot_target_supplemental"):stream=channel["raw_closest"][source]
    else:return False
    need(stream is not None and index<len(stream),"missing native point for diagnostic exclusion")
    native=stream[index]
    robot,target=ctx
    sides={links["ip140_left_pad"],links["ip140_right_pad"]}
    return (native[1]==robot and native[2]==target and native[3] in sides and native[4]==-1) or (
            native[2]==robot and native[1]==target and native[4] in sides and native[3]==-1)

def install(static):
    original_classify=static.classify_channel;original_lift=static.pre_lift_gate;counts=Counter()
    def classify(data,phase,**kw):
        original_classify(data,phase,**kw)
        saved=list(data["violations"])
        skipped=[v for v in saved if certificate_only(v,phase,data,kw["links"],(kw["robot"],kw["target"]))]
        if skipped:
            data["unmodified_V1_channel_violations"]=saved;data["diagnostic_certificate_only_exclusions"]=skipped
            counts.update(skipped)
            data["violations"]=[v for v in saved if v not in skipped];data["passed"]=not data["violations"]
        return data["passed"]
    def lift(roots,scoped_miss):
        need(scoped_miss is False and set(roots)<={"left","right"},"diagnostic scope")
        try:return original_lift(roots,False)
        except ValueError as exc:
            need(str(exc) in {"both_registered_entries_required_before_lift","existing_entry_certificate_failed_or_unknown_side"},"unrecognized pre-lift error")
            return dict(pre_lift_bilateral_entry_absent=True,diagnostic_only_certificate_omitted=True,
                        absent_or_uncertified=[s for s in ("left","right") if not roots.get(s,{}).get("passed",False)],
                        original_V1_exception=repr(exc))
    static.classify_channel=classify;static.pre_lift_gate=lift
    return counts

def self_test():
    links={"ip140_left_pad":9,"ip140_right_pad":12}
    data={"raw_manifolds":[[0,0,1,9,-1],[0,0,1,6,-1]],"raw_closest":{"robot_target":[[0,0,1,9,-1]]}}
    tests=[("left:root_residual_exceeds_tolerance","close",True),("manifold:0:missing_per_side_entry_certificate","close",True),
        ("manifold:1:missing_per_side_entry_certificate","close",False),("manifold:0:missing_per_side_entry_certificate","approach",False)]
    for k in ("robot_table_collision","proxy_pad_target_penetration","fresh_full_3d_residual","pad_target_contact_before_close",
              "nonpad_target_collision","unregistered_self_collision","nonfinite","pad_witness_position_outside_registered_surfaces"):
        tests.append(("manifold:0:"+k,"close",False))
    for reason,phase,answer in tests:
        need(certificate_only(reason,phase,data,links,(0,1))==answer,"waiver scope changed "+reason)
    need(not certificate_only("left:normal_not_unit","close",data,links,(0,1)),"invalid normal")
    return dict(passed=True,test_count=len(tests)+1)
