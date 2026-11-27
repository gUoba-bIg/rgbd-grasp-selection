"""Read-only decoding of two finite pad-clearance hard-stop fields returned by V1 itself.

The V1 episode already stopped. No simulator connection, action or state change is used here.
"""
from common_v1 import *
from math import isfinite
import sys, copy, traceback
from runtime_stages import STUDY,native_setup
from method_trial import validate

CORNER="nonfinite_or_negative_pad_corners"
PAIR="ip140_left_pad|ip140_right_pad"
CLOSEST=PAIR+":current_closest:0:penetration_below_numeric_band"

def finite(x):
    if isinstance(x,float): need(isfinite(x),"nonfinite physical evidence")
    elif isinstance(x,(list,tuple)):
        for y in x:finite(y)
    elif isinstance(x,dict):
        for y in x.values():finite(y)

def gap_fields(row,robot,ids):
    finite(row)
    issues=row["violations"]
    need(issues and len(issues)==len(set(issues)) and set(issues)<={CORNER,CLOSEST},"unrecognized native stop")
    pair=row["seam_audit"]["pairs"][PAIR]
    need(tuple(pair["link_ids"])==ids and not pair["manifolds"],"missing/wrong pad identities or real manifold")
    need(all(not (cp[1]==cp[2]==robot and set(cp[3:5])==set(ids)) for cp in row["raw_manifolds"]),"actual opposing pad manifold")
    eps=row["seam_audit"]["numeric_zero_band_m"]
    need(eps==1e-9,"V1 recorded numeric band")
    geometry=row["geometry"];a,b=geometry["corners_G_m"]
    need(len(a)==len(b)==4 and all(len(p)==3 and all(type(v) in (int,float) and isfinite(v) for v in p) for p in a+b),"missing/invalid pad corners")
    gaps=[q[1]-p[1] for p,q in zip(a,b)]
    need(gaps==geometry["gaps_m"] and (min(gaps)<-eps)==(CORNER in issues),"corner condition not explained by finite numbers")
    points=pair["raw_points"];audit=pair["closest_audit"];need(len(points)==len(audit),"incomplete native closest audit")
    for i,cp in enumerate(points):
        need(len(cp)==14 and cp[1:3]==[robot,robot] and set(cp[3:5])==set(ids) and
             all(type(cp[k]) in (int,float) and isfinite(cp[k]) for k in (8,9,10,12)) and
             all(cp[k]==0 for k in (9,10,12)),"unexpected native closest ownership or force")
        want=["penetration_below_numeric_band"] if cp[8]<-eps else []
        need(audit[i]["violations"]==want,"additional witness failure")
    mins=[i for i,x in enumerate(points) if x[8]<-eps]
    need(mins==([0] if CLOSEST in issues else []),"closest condition not explained by finite geometry")
    need(set(row["seam_audit"]["violations"])==({CLOSEST} if CLOSEST in issues else set()),"other seam event")
    return {"raw_violations":issues,"all_numeric_fields_finite":True,"registered_band_m":eps,"raw_gaps_m":gaps,
            "raw_closest_rows":points,"opposing_solver_manifold_count":0,"raw_corners_G_m":geometry["corners_G_m"],
            "observed_geometry_safety_failure":True,"engine_return_unknown":False}

def last_return(tid):
    folder=STUDY/"trials"/tid;episode=folder/"episode";r=read(episode/"result.json")
    need(r["trial_id"]==tid and r["outcome_class"]=="engineering_failure" and not r["engineering_gate_passed"] and not r["grasp_success"]
         and r["unknown_steps"]==0 and 0<r["step_intents"]==r["step_returns"]<=4728,"incomplete or indeterminate physics")
    need(sha(episode/"telemetry.jsonl.gz")==r["telemetry_sha256"],"telemetry drift")
    n=0;last=None
    def bad(x):raise ValueError("nonfinite telemetry "+x)
    with gzip.open(episode/"telemetry.jsonl.gz","rb") as i:
        for line in i:
            v=json.loads(line,parse_constant=bad)
            if v.get("event")=="state":
                n+=1;need(v["index"]==n,"state sequence")
                if n<r["step_returns"]:need(not v["violations"],"did not stop on earliest violation")
                last=v
    need(n==r["step_returns"] and last["violations"]==r["violations"] and last["phase"]==r["failure_stage"],"short/misowned stop")
    return r,last

def libraries():
    need(sys.prefix==str(C/"env/physics"),"wrong read-only decoder environment")
    os.environ["PROGRAMFILES"]=str(C/"p/vendor/ProgramFiles")
    sys.path[:0]=[str(C/"p"/R6NAME/"code"),str(C/"p"/R6NAME/"implementation"),str(C/"p"/R4/"implementation"),str(C/"p")]
    import platform_imports
    platform_imports.bind_frozen_libraries()
    from result_classification import native_rows
    return native_rows

def tests(row,robot,pads):
    gap_fields(row,robot,pads)
    changes=[]
    def nan(x):x["geometry"]["corners_G_m"][0][0][1]=float("nan")
    def missing(x):x["geometry"].pop("gaps_m")
    def wrong(x):x["seam_audit"]["pairs"][PAIR]["link_ids"][0]=-99
    def unknown(x):x["violations"]=["new_unregistered_failure"]
    def true_manifold(x):x["seam_audit"]["pairs"][PAIR]["manifolds"]=[[1]]
    def changed_raw(x):x["seam_audit"]["pairs"][PAIR]["raw_points"][0][8]=0.
    def changed_corner(x):x["geometry"]["corners_G_m"][0][0][1]=0.
    for name,fn in [("NaN",nan),("missing",missing),("wrong_owner",wrong),("unknown",unknown),("actual_manifold",true_manifold),
                    ("closest_witness_changed",changed_raw),("corner_witness_changed",changed_corner)]:
        trial=copy.deepcopy(row);fn(trial)
        try:gap_fields(trial,robot,pads)
        except (ValueError,RuntimeError,KeyError):changes.append(name)
        else:raise AssertionError("invalid fixture accepted "+name)
    return {"actual_finite_stop_reproduced":True,"rejected_negative_fixtures":changes,"source":"ADM-001 stored telemetry","new_physical_steps":0}

def reconcile(tid,*,dry=False):
    folder=STUDY/"trials"/tid
    ticket,it,sel,record=validate(tid,"diagnostic")
    # All other failed native or scientific records remain blocking.
    raw,last=last_return(tid)
    need(raw["candidate_id"]==it["selected_candidate_id"] and raw["method"]==it["method"] and
         read(folder/"start.json")["sha256"]==ENGINE and read(folder/"safety_receipt.json")["intent_sha256"]==sha(ticket),
         "intent/model owner mismatch")
    w=read(folder/"dynamic_setup/zero_step/initial_world.json")
    robot,target,table=[w["registered_"+k] for k in ("robot","object","table")]
    tb=read(folder/"dynamic_setup/zero_step/table_readback.json")
    need(w["passed"] and tb["body_id"]==table and len({robot,target,table})==3,"world ownership")
    links={k:v["index"] for k,v in read(folder/"dynamic_setup/robot_link_readback.json").items()}
    pads=tuple(links["ip140_"+s+"_pad"] for s in ("left","right"))
    data=gap_fields(last,robot,pads)
    native_rows=libraries()
    from platform_trial import dynamic_category
    from types import SimpleNamespace
    original_category=dynamic_category(raw,directory=folder/"episode",ctx=SimpleNamespace(robot=robot,target=target,table=table,links=links))
    need(original_category is None,"V1 already returned a method detail; cannot substitute a new label")
    native=native_rows(folder/"dynamic_api",(robot,target,table,tb["client_id"]),links,steps=raw["step_returns"])
    need(native["exceptions"]==0 and native["native_returns"]==native["native_calls"] and native["finite"] and native["body_link_ownership"],"unverified raw physics")
    static=read(folder/"static_preflight/path_result.json");roots=read(folder/"static_preflight/left_right_contact_entry_roots.json")
    staticlog=read(folder/"static_integrity.json")
    need(static["passed"] and static["snapshots_completed"]==4728 and static["physical_steps"]==0 and static["filter_call_count"]==14 and
         staticlog["native"]["finite"] and staticlog["native"]["body_link_ownership"] and staticlog["native"]["exceptions"]==0,
         "unverified original safety pass")
    proof=dict(utc=utc(),trial_id=tid,step=last["index"],phase=last["phase"],phase_tick=last["phase_tick"],
        episode_result_sha256=sha(folder/"episode/result.json"),intent_sha256=sha(ticket),telemetry_sha256=raw["telemetry_sha256"],
        native=native,scientific_category_return=original_category,episode_outcome_class=raw["outcome_class"],finite_stop_fields=data,
        thresholds_source={p.name:sha(p) for p in [C/"p"/R4/"implementation/platform_episode.py",C/"p"/PL/"seam_geometry.py",
             C/"p"/PL/"structural_filter.py",C/"p"/R6NAME/"implementation/platform_trial.py"]},new_physics_calls=0)
    if dry:
        proof["tests"]=tests(last,robot,pads)
        return proof
    prefix="study/trials/"+tid+"/"
    proof_hash=write(prefix+"decoded_stop_fields.json",proof)
    write(prefix+"dynamic_integrity.json",native)
    t0=datetime.fromisoformat(read(folder/"start.json")["utc"])
    ended=datetime.fromtimestamp((folder/"dynamic_api/summary.json").stat().st_mtime,timezone.utc)
    result=dict(trial_id=tid,cohort="diagnostic",kind=it["supplement_kind"],scene=it["scene_key"],unit=it["unit_id"],object=it["object_id"],
        condition=it["condition"],method=it["method"],candidate=it["selected_candidate_id"],selected=True,parent_trial=it["parent_trial"],
        seed=it["worker_seed"],intent_sha256=sha(ticket),physical_calls=raw["step_intents"],physical_returns=raw["step_returns"],
        trial_completed=raw["trial_completed"],engineering_gate_passed=False,grasp_success=False,diagnostic_certificate_exception=it["supplement_kind"]=="diagnostic_uncertified",
        full_static_path=True,bilateral_certificate=roots["both_passed"],static_snapshots=static["snapshots_completed"],dynamic_outcome_observed=True,
        forbidden_collision=False,slipped=raw.get("slipped"),max_lift_m=raw.get("max_lift_m"),final_hold_streak=raw.get("final_hold_streak"),
        unavailable_terminal_metrics=[k for k in ("slipped","max_lift_m","final_hold_streak") if k not in raw],
        detail="finite_opposing_pad_clearance_hard_stop",failure=raw["failure_reason"],shared_failure=False,utc=utc(),
        elapsed_s=(ended-t0).total_seconds(),elapsed_basis="start receipt to native disconnect summary file time",
        decoded_stop_fields_sha256=proof_hash,underlying_V1_episode_sha256=proof["episode_result_sha256"])
    write(prefix+"result.json",result)
    return proof

if __name__=="__main__":
    mode,tid=sys.argv[1:]
    if mode in ("offline","verify_fields"):
        proof=reconcile(tid,dry=True);write("admission/"+("diagnostic_receipt_replay" if mode=="offline" else "diagnostic_native_label_verification")+".json",proof)
    elif mode=="existing":reconcile(tid)
    elif mode=="once":
        try:
            from method_trial import run
            run(tid,"diagnostic")
        except Exception as exc:
            # Only an already-returned and completely evidenced V1 hard stop can be interpreted.
            # Preserve the outer reporting exception even when underlying physics is fully known.
            write("study/trials/"+tid+"/first_reporting_exception.json",dict(utc=utc(),exception=repr(exc),traceback=traceback.format_exc()))
            reconcile(tid)
    else:raise RuntimeError("unknown caller")
