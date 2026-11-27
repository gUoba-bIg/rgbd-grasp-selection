"""Pure replay of static admission contact classification; never opens PyBullet.

All contacts below are synthetic component-test inputs, not measured outcomes.
Default invocation prints deterministic JSON.  --write-once creates the report
only if absent, so later independent recomputation cannot overwrite evidence.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import path_precheck as implementation

HERE=Path(__file__).resolve().parent
REPORT=HERE/"static_path_offline_replay.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value,float) and not math.isfinite(value):
        return {"nonfinite_synthetic_value":repr(value)}
    if isinstance(value,(list,tuple)):return [serializable(x) for x in value]
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    return value


def report():
    links={name:i for i,name in enumerate(dict.fromkeys(
        name for pair in implementation.STRUCTURAL_PAIRS for name in pair))}
    pad_l,pad_r=[links[x] for x in implementation.PADS]
    bone_l,bone_r=[links[x] for x in implementation.BONES]
    base=links["base_link_inertia"]
    robot,target,table=1,2,3
    def row(a,b,la,lb,d=0.):
        return [0,a,b,la,lb,[0.,0.,0.],[0.,0.,0.],[0.,0.,1.],d,
                0.,0.,[1.,0.,0.],0.,[0.,1.,0.]]
    cases=[]
    def add(name,r,expected_rejected,*,source="manifold",phase="close",required_reason=None):
        observed=implementation.classify_contact(deepcopy(r),source=source,phase=phase,
            robot=robot,target=target,table=table,links=links)
        rejected=bool(observed["violations"])
        passed=rejected==expected_rejected and (required_reason is None or required_reason in observed["violations"])
        cases.append(dict(name=name,input_kind="synthetic_not_physical_measurement",
            input=serializable(r),source=source,phase=phase,expected_rejected=expected_rejected,
            required_reason=required_reason,observed=serializable(observed),passed=passed))
    # These first twelve preserve the assertions initially executed by the
    # path-precheck implementer.  Empty violations are a component result only.
    add("01_registered_pad_target_close",row(robot,target,pad_l,-1),False)
    add("02_pad_target_before_close",row(robot,target,pad_l,-1),True,phase="approach",
        required_reason="pad_target_contact_before_close")
    add("03_nonpad_target",row(robot,target,bone_l,-1),True,
        required_reason="nonpad_target_collision")
    add("04_below_minus_1mm",row(robot,target,pad_l,-1,-.0010000000000001),True,
        required_reason="external_penetration_below_minus_1mm")
    add("05_exact_minus_1mm",row(robot,target,pad_l,-1,-.001),False)
    add("06_target_table_registered_support",row(target,table,-1,-1),False)
    add("07_pad_table_forbidden",row(robot,table,pad_l,-1),True,
        required_reason="robot_table_collision")
    add("08_filtered_base_table_reappears",row(robot,table,base,-1),True,
        required_reason="filtered_base_table_pair_reappeared")
    add("09_registered_mount_negative_closest_delegated",row(robot,table,base,-1,-.0010000957),False,
        source="robot_table")
    add("10_wrong_external_body",row(robot,4,pad_l,-1),True,
        required_reason="unknown_contact_identity")
    add("11_wrong_target_link",row(robot,target,pad_l,0),True,
        required_reason="wrong_target_link")
    add("12_nonfinite_distance",row(robot,target,pad_l,-1,float("nan")),True,
        required_reason="nonfinite_or_malformed_contact")
    # Extra identity, contact-role, finite-data, and boundary regressions.
    add("13_reversed_registered_pad_target",row(target,robot,-1,pad_r),False)
    add("14_reversed_forbidden_nonpad",row(target,robot,-1,bone_r),True,
        required_reason="nonpad_target_collision")
    add("15_wrong_support_link",row(target,table,1,-1),True,
        required_reason="wrong_support_link")
    add("16_wrong_robot_body",row(99,target,pad_l,-1),True,
        required_reason="unknown_contact_identity")
    add("17_wrong_robot_link",row(robot,target,999,-1),True,
        required_reason="nonpad_target_collision")
    add("18_wrong_table_identity",row(target,99,-1,-1),True,
        required_reason="unknown_contact_identity")
    a,b=implementation.STRUCTURAL_PAIRS[0]
    add("19_internal_filtered_pair_manifold",row(robot,robot,links[a],links[b]),True,
        required_reason="filtered_internal_pair_reappeared")
    add("20_same_link_manifold_invalid",row(robot,robot,pad_l,pad_l),True,
        required_reason="same_link_manifold")
    add("21_same_link_closest_not_physical_pair",row(robot,robot,pad_l,pad_l,-.01),False,
        source="robot_self")
    add("22_active_pad_seam_delegated",row(robot,robot,pad_l,pad_r),False)
    add("23_unregistered_self_even_zero_force",row(robot,robot,base,pad_l),True,
        required_reason="unregistered_self_collision")
    add("24_unregistered_self_negative_closest",row(robot,robot,base,pad_l,-2e-9),True,
        source="robot_self",required_reason="unregistered_self_collision")
    r=row(robot,target,pad_l,-1);r[9]=float("inf")
    add("25_nonfinite_force",r,True,required_reason="nonfinite_or_malformed_contact")
    r=row(robot,target,pad_l,-1);r[7][2]=float("nan")
    add("26_nonfinite_normal",r,True,required_reason="nonfinite_or_malformed_contact")
    add("27_missing_contact_fields",row(robot,target,pad_l,-1)[:10],True,
        required_reason="nonfinite_or_malformed_contact")
    add("28_closest_pad_negative_before_close",row(robot,target,pad_l,-1,-1e-6),True,
        source="robot_target",phase="approach",required_reason="pad_target_contact_before_close")
    add("29_target_table_too_deep",row(target,table,-1,-1,-.002),True,
        required_reason="support_penetration_below_minus_1mm")
    add("30_loaded_reference_overlap_blocks_not_dynamic_result",row(robot,target,pad_r,-1,-.02),True,
        source="robot_target",phase="close",required_reason="external_penetration_below_minus_1mm")
    add("31_nextafter_minus_1mm_fails",row(robot,target,pad_l,-1,math.nextafter(-.001,-math.inf)),True,
        required_reason="external_penetration_below_minus_1mm")
    add("32_nonpad_positive_clearance_not_contact",row(robot,target,bone_l,-1,.003),False,
        source="robot_target")
    source=Path(implementation.__file__).read_text(encoding="utf8")
    tree=ast.parse(source)
    scanner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="scan_path")
    method_calls=[n.func.attr for n in ast.walk(scanner) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    forbidden={"connect","loadURDF","stepSimulation","setCollisionFilterPair",
        "setJointMotorControlArray","setJointMotorControl2","resetBasePositionAndOrientation","resetBaseVelocity"}
    scanner_source=inspect.getsource(implementation.scan_path)
    static_checks={
        "no_forbidden_direct_api_call":not (set(method_calls)&forbidden),
        "physics_authorization_false_guard_present":"p.step_authorized is not False" in scanner_source,
        "charged_is_false":"charged=False" in scanner_source,
        "loaded_prediction_is_false":"loaded_state_prediction=False" in scanner_source,
        "dynamics_validated_is_false":"dynamics_validated=False" in scanner_source,
        "blocking_status_is_static_unproven":"blocked_static_admission_unproven" in scanner_source,
        "unvisited_snapshots_disclosed":"snapshots_remaining_unchecked" in scanner_source,
        "reference_warning_at_close_lift_hold":'phase in ("close","lift","hold")' in scanner_source,
        "all_4728_references_registered":sum(n for _,n in implementation.EXPECTED_PHASE_STEPS)==4728,
        "no_pybullet_imported":"pybullet" not in sys.modules,
    }
    return dict(version=HERE.name,script_sha256=digest(Path(__file__)),
        classifier_sha256=digest(Path(implementation.__file__)),passed=all(c["passed"] for c in cases) and all(static_checks.values()),
        case_count=len(cases),cases=cases,synthetic_links=links,static_source_checks=static_checks,
        engine_connections=0,physics_steps=0,slot_starts=0,charged=False,
        scope="Component classification and source-guard inspection only; not geometry, loaded feasibility, collision dynamics, or path execution evidence.",
        delegated_gates="Registered negative base-table/wrist distances and active seams require their independent baseline and witness audits. An empty classifier violation list does not bypass these gates.",
        loaded_reference_caveat="Close motor q=0 snapshots can be unreachable under contact. A hard-gate failure blocks static admission and cannot be called a physical failure or proof the method is infeasible.",
        recomputation="Run this script without --write-once to print an independently recomputable deterministic report; --write-once refuses an existing report.")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["--write-once"]):raise SystemExit("Use no argument or --write-once")
    result=report()
    data=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if sys.argv[1:]==["--write-once"]:
        with REPORT.open("x",encoding="utf8") as stream:stream.write(data)
        print(json.dumps(dict(passed=result["passed"],case_count=result["case_count"],report=str(REPORT),
            report_sha256=digest(REPORT),engine_connections=0,physics_steps=0),ensure_ascii=False))
    else:print(data,end="")
    raise SystemExit(0 if result["passed"] else 1)
