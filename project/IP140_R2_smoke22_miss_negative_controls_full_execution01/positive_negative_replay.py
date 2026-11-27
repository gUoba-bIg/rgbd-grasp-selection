"""Smoke17 phase scope regression. Pure replay of archived observations or named synthetic inputs."""
from copy import deepcopy
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

from contact_semantics import audit_pad_witness, classify_contact, validate_entry_certificate
from contact_entry import validate_contact_row
from object_evaluator import ObjectEvaluator, PHASE_AT_STEP

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / "IP140_R1_smoke16_slot14_static_role_split_and_run01"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def save(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

def displaced(row, new_d):
    moved = deepcopy(row)
    moved[8] = new_d
    moved[6] = (np.asarray(moved[5])-new_d*np.asarray(moved[7])).tolist()
    return moved

def build():
    trace = OLD / "static_preflight/dual_channel_snapshots.jsonl.gz"
    zero = OLD / "static_preflight/zero_readback.json"
    identities = OLD / "static_preflight/zero_step/all_links_readback.json"
    roots = OLD / "static_preflight/left_right_contact_entry_roots.json"
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        state = json.loads(next(itertools.islice(stream, 3279, None)))
    assert (state["snapshot"],state["phase"],state["phase_tick"],state["passed"]) == (3280,"lift",64,False)
    frame = state["proxy_channel"]
    links = {name:row["index"] for name,row in read(identities).items()}
    robot, table, target = (read(zero)[name] for name in ("body_id","table_body_id","object_body_id"))
    pad = links["ip140_left_pad"]
    native_pair = {(robot,pad),(target,-1)}
    fresh = next(r for r in frame["raw_closest"]["robot_target"] if {(r[1],r[3]),(r[2],r[4])}==native_pair)
    cache = next(r for r in frame["raw_manifolds"] if {(r[1],r[3]),(r[2],r[4])}==native_pair)
    cert = read(roots)["sides"]["left"]
    W_T_pad = np.asarray(frame["W_T_links"]["ip140_left_pad"])
    dims = read(identities)["ip140_left_pad"]["collision"][0][3]
    radius = read(identities)["ip140_left_pad"]["dynamics"][-1]
    assert max(abs(np.asarray(dims)-[.060,.002,.020]))<1e-15 and radius==.0005
    # Tip section of left pad runs from (L-r,0) to (L,-r). Both arms of arc must fit z span.
    pad_is_A = (fresh[1],fresh[3])==(robot,pad)
    world_point = fresh[5 if pad_is_A else 6]
    p = (np.linalg.inv(W_T_pad) @ np.r_[world_point,1])[:3]
    n = W_T_pad[:3,:3].T @ (np.asarray(fresh[7])*(-1 if pad_is_A else 1))
    center2 = np.array([dims[0]-radius,-radius])
    radial = p[:2]-center2
    radius_error = float(abs(np.linalg.norm(radial)-radius))
    normal_error = float(np.linalg.norm(np.r_[radial/radius,0.]-n))
    full = float(np.linalg.norm(np.asarray(fresh[5])-np.asarray(fresh[6])-fresh[8]*np.asarray(fresh[7])))
    geom = dict(readback_box_dimensions_m=dims,readback_margin_m=radius,tip_round_xy_center_m=center2.tolist(),
        native_fresh=fresh,native_cached=cache,working_W_T_pad=W_T_pad.tolist(),local_pad_witness_m=p.tolist(),
        tip_depth_m=float(dims[0]-p[0]),normal_pad_to_target_local=n.tolist(),
        signed_distance_m=fresh[8],full_3D_witness_residual_m=full,
        inner_plane_abs_offset_m=float(abs(p[1])),signed_tip_arc_radial_components_m=radial.tolist(),
        z_inside_straight_cylinder_section=bool(abs(p[2])<=dims[2]/2-radius),absolute_radius_error_m=radius_error,
        outward_normal_difference_norm=normal_error,phase_tick=state["phase_tick"],snapshot=state["snapshot"],
        units="m, unit normal",archived_failure=state["violations"],source_trace=str(trace.relative_to(ROOT)),
        classification="rounded_tip_edge_contact_risk",scope="Static geometry; no physical grasp outcome measured.")
    tests=[]
    def record(name, ok, **evidence):
        tests.append(dict(name=name,passed=bool(ok),**evidence))
    record("independent_round_arc_on_frozen_radius_with_outward_normal",radius_error<=1e-9 and normal_error<=1e-9
           and min(radial)>0 and geom["z_inside_straight_cylinder_section"] and abs(p[1])>1e-9 and full<=1e-9,result=geom)
    def witness(r, source, phase):
        return audit_pad_witness(r,source=source,phase=phase,robot=robot,target=target,pad_index=pad,
                                 W_T_pad=W_T_pad,W_T_G=frame["W_T_G"],side="left",require_face=True)
    def classify(r, phase, source="robot_target", matches=None):
        return classify_contact(r,source=source,phase=phase,channel="proxy",robot=robot,target=target,table=table,
                                links=links,entry_by_pad={pad:cert},fresh_closest=matches if matches is not None else [r])
    for phase in ("lift","hold","close","approach"):
        for row, source in [(fresh,"robot_target"),(cache,"manifold")]:
            observed=witness(row,source,phase)
            cls=classify(row,phase,source,[fresh])
            late=phase in ("lift","hold")
            expected = observed["passed"] and observed["surface"]["label"]=="rounded_tip_edge_contact_risk"
            if phase == "approach" and source == "robot_target":
                expected = expected and "pad_target_contact_before_close" in cls["violations"]
            elif phase != "approach":
                expected = expected and not cls["violations"]
            record("actual_3280_"+source+"_"+phase,expected,phase=phase,expected_surface_admitted=True,scope="R2 surface test; phase and bracket eligibility checked separately",
                   witness=observed,classification=cls,synthetic_phase=(phase!="lift"))
    strict=validate_contact_row(fresh,robot=robot,target=target,pad_index=pad,W_T_pad=W_T_pad,
                                 W_T_G=frame["W_T_G"],side="left",require_face=True)
    # This known arc is a legitimate R2 surface, but it is a non-root lift read.
    # It must not replace a localized root query or generate an entry certificate.
    proof=deepcopy(cert)
    proof["entry_audit"]=strict
    errors=validate_entry_certificate(proof,robot=robot,target=target,pad=pad)
    record("rounded_surface_is_not_by_itself_an_entry_certificate",strict["passed"] and bool(errors),root_query=strict,errors=errors)
    invalid=deepcopy(cert);invalid["previous_signed_distance_m"]=0.
    errors=validate_entry_certificate(invalid,robot=robot,target=target,pad=pad)
    record("nonpositive_previous_endpoint_cannot_certify_entry",bool(errors),errors=errors)
    for phase in ("lift","hold"):
        for name,first,second,la,lb,expected in (
            ("bone_target",robot,target,links["ip140_left_bone"],-1,True),
            ("palm_target",robot,target,links["ip140_palm"],-1,True),
            ("arm_target",robot,target,links["wrist_1_link"],-1,True),
            ("other_robot_table",robot,table,pad,-1,False),
            ("environment",robot,101,pad,-1,False),
            ("unknown_robot_link",robot,target,999,-1,False),
            ("wrong_target_link",robot,target,pad,2,False),
            ("unregistered_self",robot,robot,pad,links["shoulder_link"],False),
            ("filtered_wrist_palm_reappeared",robot,robot,links["wrist_3_link"],links["ip140_palm"],False),
            ("filtered_base_table_reappeared",robot,table,links["base_link_inertia"],-1,False)):
            row=deepcopy(fresh);row[1:5]=[first,second,la,lb]
            classification=classify(row,phase,"manifold",[row])
            record(phase+"_"+name,bool(not classification["violations"])==expected,synthetic=True,result=classification,expected_accepted=expected)
        row=deepcopy(fresh);row[9]=float("nan")
        cls=classify(row,phase)
        record(phase+"_nonfinite_target_read_rejected",bool(cls["violations"]),synthetic_mutation="Fn NaN",result=cls)
        changed=deepcopy(fresh);changed[6][2]+=2.1e-9
        audit=witness(changed,"robot_target",phase)
        record(phase+"_fresh_raw_witness_residual_reject",not audit["passed"],synthetic=True,result=audit)
        changed=deepcopy(cache);changed[8]+=2e-9
        audit=witness(changed,"manifold",phase)
        record(phase+"_persistent_normal_projection_reject",not audit["passed"],synthetic=True,result=audit)
    changed=displaced(fresh,-.002)
    for phase,expect in (("approach",False),("close",False),("lift",True),("hold",True)):
        obs=classify(changed,phase)
        record("2mm_fixed_target_proxy_"+phase,(not obs["violations"])==expect,result=obs,synthetic=True)
    # The original live evaluator must continue to reject measured forbidden contacts and strict bound crossings.
    ev=lambda: ObjectEvaluator(robot,target,table,links)
    wrong=deepcopy(fresh);wrong[3]=links["ip140_left_bone"];wrong[9]=.5
    for label,row in (("nonpad_contact",wrong),("penetration",displaced(fresh,math.nextafter(-.001,-math.inf)))):
        classification,violations,_,_=ev()._contact(row,0)
        record("dynamic_"+label+"_rejected",bool(violations),synthetic=True,result=classification,violations=violations)
    obs_obj=dict(body_id=target,position=[0.,0.,0.],orientation=[0,0,0,1],linear_velocity=[0]*3,angular_velocity=[0]*3)
    finger=[[0,0,[0]*6,0],[0,0,[0]*6,0]]
    fast=deepcopy(finger);fast[0][1]=math.nextafter(.04,math.inf)
    out=ev().observe(1,"settle",[],fast,obs_obj,[0,0,0])
    record("dynamic_nextafter_speed_hard_failure",not out["passed"],synthetic=True,result=out)
    grips=[]
    for name in ("ip140_left_pad","ip140_right_pad"):
        row=displaced(fresh,0.);row[1:5]=[robot,target,links[name],-1];row[9]=1.;grips.append(row)
    supportrow=displaced(fresh,0.);supportrow[1:5]=[target,table,-1,-1];supportrow[9]=.12
    for scenario in ("valid","aggregate_support","slip"):
        evaluator=ev();obj=deepcopy(obs_obj)
        for step,phase in enumerate(PHASE_AT_STEP,1):
            rows=list(grips) if phase in ("close","lift","hold") else []
            obj["position"][2]=.11 if phase in ("lift","hold") else 0.
            if step>=4249 and scenario=="aggregate_support":
                rows+=[supportrow,supportrow]
            if step>=4249 and scenario=="slip":
                obj["position"][2]=0.; rows=[]
            evaluator.observe(step,phase,rows,finger,obj,[0.,0.,.11])
        out=evaluator.finish()
        match=out["episode_completed"] and out["engineering_gate_passed"] and (out["grasp_success"]==(scenario=="valid"))
        if scenario=="slip":match &= out["slipped"] and out["grasp_formed"]
        if scenario=="aggregate_support":match &= evaluator.terminal_counts["aggregate_table_supported"]==480
        record("frozen_dynamic_terminal_"+scenario,match,synthetic=True,physical_steps=0,
               terminal_counts=evaluator.terminal_counts,expected_grasp_success=(scenario=="valid"),result=out)
    files={"archive/smoke16_trace":trace,"archive/smoke16_links":identities,"archive/smoke16_zero":zero,"archive/smoke16_roots":roots}
    for name in ["positive_negative_replay.py","contact_entry.py","contact_semantics.py","contact_aware_path.py","pad_surface.py","object_evaluator.py","aggregate_support.py"]:
        files[name]=HERE/name
    return dict(version=HERE.name,passed=all(x["passed"] for x in tests),total_cases=len(tests),cases=tests,
                connections=0,physics_steps=0,new_charges=0,charged=False,
                source_sha256={k:sha(path) for k,path in files.items()},geometry=geom,
                notes="True native archived rows are marked by sample and source. Deliberate mutations and phase substitutions are offline tests. No simulator was imported.")

def persist_report():
    report=build()
    save(HERE/"positive_negative_replay.json",report)
    save(HERE/"static_gate_semantics.json",dict(version=HERE.name,
        strict_pre_lift=["entry bracket, identity, root, registered inner plane or inner-tip quarter cylinder with frozen R2 tolerances","proxy/contact reference and original prohibited contacts","672-step approach included"],
        fixed_target_lift_hold_geometry=["all robot-target manifolds and fresh records retained","local surface, normal, depth, distance and nonpad identity diagnostic","target motion unknown"],
        full_path_hard=["body/link identity, numeric integrity and witness residual checks","table, environment and self collision","filters, fixed mounts, FK/TCP, limits, velocity, API, zero steps, no target reset"],
        physical_gates="Unchanged frozen dynamic evaluator; the static role split has no authority over outcome, forces, sliding, aggregate support or terminal height.",
        report_sha256=sha(HERE/"positive_negative_replay.json"),source_sha256=report["source_sha256"]))
    return report

if __name__=="__main__":
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--offline-only"]:
        raise SystemExit("project root plus --offline-only required")
    report=persist_report()
    print(json.dumps({k:report[k] for k in ("passed","total_cases","connections","physics_steps")}))
    if not report["passed"]:print([c["name"] for c in report["cases"] if not c["passed"]])
    raise SystemExit(0 if report["passed"] else 1)
