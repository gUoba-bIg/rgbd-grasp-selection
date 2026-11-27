"""One hashed supplementary intent, one V1 native connection per phase, no alternative candidate."""
from common_v1 import *
from runtime_stages import STUDY,native_setup
import random
import sys
import time
import traceback

def validate(tid,cohort):
    ticket=STUDY/"trials"/tid/"intent.json";it=read(ticket)
    seal=read(STUDY/"preregistration"/(cohort+"_intents.json"))
    need(ticket.parent.name==it["trial_id"]==tid and sha(ticket)==seal["sha256"][tid] and len(seal["order"])==len(set(seal["order"]))==seal["count"],"invalid supplement reservation")
    need(seal["count"]==60 if cohort=="diagnostic" else 0<=seal["count"]<=72,"budget size")
    source=ROOT/R6NAME/"trials"/it["parent_trial"]/"intent.json";parent=read(source)
    need(sha(source)==it["parent_intent_sha256"] and parent["scene_key"]==it["scene_key"] and parent["worker_seed"]==it["worker_seed"],"parent seed/scene")
    expected_prefix="frozen_r6_units" if cohort=="diagnostic" else "units"
    if cohort=="robustness":
        idx=OUT/"preregistration/observation_fingerprint_index.json"
        need(it["actual_unit_inputs_path"]==idx.relative_to(OUT).as_posix() and sha(idx)==it["actual_unit_inputs_sha256"],"supplement model input index")
        need(it["model_input_fingerprints"]==read(idx)["units"][it["unit_id"]]["model_input_fingerprints"] and
             it["parent_model_input_fingerprints"]==parent["model_input_fingerprints"],"condition-specific tensor ownership")
    need(Path(it["selection_path"]).parts==(expected_prefix,it["unit_id"],"selection.json"),"selected input location")
    for tag in ("stable_state","observation","pool","raw_decoder","ranking","selection","scoring_components","exclusion"):
        rel=Path(it[tag+"_path"]);need(not rel.is_absolute() and ".." not in rel.parts,"invalid input path")
        need(sha(STUDY/rel)==it[tag+"_sha256"],"input hash drift: "+tag)
    sel=read(STUDY/it["selection_path"]);ranking=read(STUDY/it["ranking_path"])
    need(it["attempt_limit"]==1 and it["ranking_uses_ground_truth"] is False and it["no_postlock_fallback"] is True and not it["r5_interface_in_chain"] and not sel["used_ground_truth"],"invalid choice scope")
    i=it["selected_pool_index"];eligible=sel["eligibility"]
    need(ranking["unit_id"]==it["unit_id"]==sel["unit_id"] and ranking["method"]==it["method"] and ranking["shared_pool_sha256"]==it["pool_sha256"],"score source")
    need(sorted(ranking["complete_order"])==list(range(len(eligible))) and all(type(j) is int for j in ranking["complete_order"]),"complete shared order")
    need(all(type(row["index"]) is int and row["index"]==j and type(row["eligible"]) is bool for j,row in enumerate(eligible)),"eligibility record")
    need(next((j for j in ranking["complete_order"] if eligible[j]["eligible"]),None)==i==sel["selected"][it["method"]],"first eligible candidate")
    record=eligible[i] if i is not None else None
    if record:need(record["plan"]["candidate_world"]==it["selected_p3"]["world_17"] and record["plan"]["candidate_id"]==it["selected_candidate_id"],"locked pose")
    if cohort=="diagnostic":
        reg=OUT/"preregistration/diagnostic_sample.json";case=next(c for c in read(reg)["cases"] if c["supplement_trial_id"]==tid)
        need(sha(reg)==it["supplement_selection_manifest"] and "diagnostic_"+case["group"]==it["supplement_kind"] and
             case["trial"]==it["parent_trial"] and case["candidate"]==it["selected_candidate_id"] and parent["selected_p3"]==it["selected_p3"],"diagnostic identity")
    return ticket,it,sel,record

def diagnostic_post(dest,it,ctx,sr):
    # Same native stream, scope, roots, identities and finite-return checks as V1.
    import result_classification as cl
    base=dest/"static_preflight";cl.numbers(sr)
    need(sr["trial_id"]==it["trial_id"] and sr["method"]==it["method"] and sr["candidate_id"]==it["selected_candidate_id"],"scan owner")
    need(sr["physical_steps"]==sr["physics_step_calls"]==0 and sr["filter_call_count"]==14 and not sr["target_pose_modified"] and not sr["proxy_used_by_controller"],"scope changed")
    roots,checked,wanted=cl.roots_and_queries(base,sr,ctx.links)
    native=cl.native_rows(dest/"static_api",(ctx.robot,ctx.target,ctx.table,ctx.client),ctx.links,watch=wanted)
    need(native["finite"] and native["body_link_ownership"] and native["actual_physics_steps"]==0 and native["native_calls"]==native["native_returns"],"incomplete native geometry")
    if sr["passed"]:
        need(sr["snapshots_completed"]==4728 and not sr["violations"] and sr["first_failure"] is None,"incomplete guarded path")
        detail="V1_path_complete_with_diagnostic_certificate_exception" if not roots["both_passed"] else "static_admitted"
    else:detail=cl.classify_summary(sr,roots,checked,native)
    return dict(detail_class=detail,path_passed=sr["passed"],native=native,certificate_sides=checked,
                actual_both_certified=roots["both_passed"],frozen_root_tolerance=cl.DISTANCE,static_result_sha256=sha(base/"path_result.json"))

def run(tid,cohort):
    import numpy as np
    from diagnostic_gate import install
    tic=time.perf_counter();ticket,it,s,row=validate(tid,cohort)
    tw=native_setup("physics",STUDY);random.seed(it["worker_seed"]);np.random.seed(it["worker_seed"]);tw.bind_engine()
    from platform_trial import connection,checked_zero,candidate_static_failure,dynamic_category,no_selection_outcome
    from platform_support import bind_asset
    from platform_episode import run_method
    from scene_world import setup_world,settle_state,confirm_same_settle
    from result_classification import post_static,native_rows,DETAIL
    import platform_static
    dest=ticket.parent; prefix="study/trials/"+tid+"/"
    waive=it["supplement_kind"]=="diagnostic_uncertified"
    counts=install(platform_static) if waive else {}
    res=dict(trial_id=tid,cohort=cohort,kind=it["supplement_kind"],scene=it["scene_key"],unit=it["unit_id"],object=it["object_id"],condition=it["condition"],
        method=it["method"],candidate=it["selected_candidate_id"],selected=(row is not None),parent_trial=it["parent_trial"],
        seed=it["worker_seed"],intent_sha256=sha(ticket),physical_calls=0,physical_returns=0,
        trial_completed=False,engineering_gate_passed=False,grasp_success=False,diagnostic_certificate_exception=waive,
        full_static_path=False,bilateral_certificate=None,dynamic_outcome_observed=False,forbidden_collision=False,slipped=None,
        max_lift_m=None,final_hold_streak=None,shared_failure=False)
    write(prefix+"start.json",dict(utc=utc(),pid=os.getpid(),cwd=str(Path.cwd()),binary=str(Path(sys.modules["pybullet"].__file__).resolve()),sha256=sha(Path(sys.modules["pybullet"].__file__)),intent_sha256=sha(ticket)))
    if row is None:
        with connection(dest/"no_candidate_zero_api","contact_aware_static") as (trace,client):
            ctx=setup_world(trace,client,it["scene_key"],dest/"no_candidate_zero_setup",settled_state=STUDY/it["stable_state_path"]);checked_zero(ctx)
        native_rows(dest/"no_candidate_zero_api",(ctx.robot,ctx.target,ctx.table,ctx.client),ctx.links)
        res.update(detail="empty_old_E" if s["eligibility"] else "empty_pool",failure=no_selection_outcome(s["eligibility"]))
    else:
        with connection(dest/"static_api","contact_aware_static") as (p,client):
            ctx=setup_world(p,client,it["scene_key"],dest/"static_setup",settled_state=STUDY/it["stable_state_path"])
            sr=platform_static.scan(p,client,ctx.robot,ctx.links,row["plan"],checked_zero(ctx),intent=it,output_root=dest/"static_preflight")
            need(not p.exceptions and not p.actual_step_calls and not p.counts["stepSimulation"],"unknown static native return")
        report=diagnostic_post(dest,it,ctx,sr) if waive else post_static(dest,intent=it,links=ctx.links)
        roots=read(dest/"static_preflight/left_right_contact_entry_roots.json")
        res.update(full_static_path=sr["passed"],bilateral_certificate=roots["both_passed"],static_snapshots=sr["snapshots_completed"])
        write(prefix+"static_integrity.json",dict(**report,diagnostic_excluded_labels=dict(counts)))
        if not sr["passed"]:
            category="path_failure" if report["detail_class"]==DETAIL else candidate_static_failure(sr,directory=dest/"static_preflight",ctx=ctx)
            need(category is not None,"unknown V1 static failure "+repr(sr["first_failure"]))
            res.update(detail=report["detail_class"] or category,failure=sr["first_failure"])
        else:
            need(sr["snapshots_completed"]==4728 and (roots["both_passed"] or waive) and sr["filter_call_count"]==14,"incomplete safety gate")
            rec=dest/"safety_receipt.json"
            write(prefix+rec.name,dict(passed=True,intent_sha256=sha(ticket),static_steps=0,path_digest=sha(dest/"static_preflight/path_result.json"),
                both_registered_contact_entries=roots["both_passed"],only_certificate_waived=waive,fully_certified_v1=roots["both_passed"]))
            with connection(dest/"dynamic_api","dynamic") as (trace,client):
                ctx=setup_world(trace,client,it["scene_key"],dest/"dynamic_setup");checked_zero(ctx)
                bind_asset(ctx.object_urdf_path,ctx.object_mass_kg)
                def authorize():
                    trace.authorize_dynamic(client,intent_path=ticket,intent_sha256=sha(ticket),static_receipt=rec)
                    return dict(intent_sha256=sha(ticket),safety_receipt_sha256=sha(rec))
                def verify_settle(c):
                    actual=settle_state(c);write(prefix+"returned_settle_state.json",actual)
                    confirm_same_settle(STUDY/it["stable_state_path"],actual)
                episode=dest/"episode";episode.mkdir(exist_ok=False)
                returned=run_method(ctx,row["plan"],episode,intent=it,before_first_step=authorize,verify_settle=verify_settle)
                code=dynamic_category(returned,directory=episode,ctx=ctx)
                need(not trace.exceptions and not returned["unknown_steps"] and code is not None and returned["step_returns"]==trace.returns["stepSimulation"],"unknown returned dynamics "+str(returned.get("failure_reason")))
                need(trace.actual_step_calls==trace.returns["stepSimulation"] and sha(episode/"telemetry.jsonl.gz")==returned["telemetry_sha256"],"unbound returned telemetry")
                terminal_fields=("slipped","max_lift_m","final_hold_streak")
                missing=[key for key in terminal_fields if key not in returned]
                need(not returned["trial_completed"] or not missing,"completed episode is missing a terminal metric")
                res.update(physical_calls=trace.actual_step_calls,physical_returns=trace.returns["stepSimulation"],
                    dynamic_outcome_observed=True,trial_completed=returned["trial_completed"],engineering_gate_passed=returned["engineering_gate_passed"],
                    grasp_success=returned["grasp_success"],detail=code,failure=returned.get("failure_reason"),forbidden_collision=code=="forbidden_collision",
                    unavailable_terminal_metrics=missing,slipped=returned.get("slipped"),max_lift_m=returned.get("max_lift_m"),final_hold_streak=returned.get("final_hold_streak"))
            native=native_rows(dest/"dynamic_api",(ctx.robot,ctx.target,ctx.table,ctx.client),ctx.links,steps=res["physical_returns"])
            write(prefix+"dynamic_integrity.json",native)
    res.update(utc=utc(),elapsed_s=time.perf_counter()-tic)
    write(prefix+"result.json",res)
if __name__=="__main__":
    try:run(sys.argv[1],sys.argv[2])
    except BaseException as e:
        print(traceback.format_exc(),flush=True)
        disk()
        write("study/trials/"+sys.argv[1]+"/shared_or_unknown_stop.json",dict(utc=utc(),exception=repr(e),traceback=traceback.format_exc(),outcome=None))
        raise
