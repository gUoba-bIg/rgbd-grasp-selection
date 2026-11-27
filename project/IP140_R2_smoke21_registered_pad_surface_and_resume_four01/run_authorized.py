"""Run one registered smoke21 slot, with a new static and a new dynamic DIRECT connection."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import traceback

from batch_context import VERSION, ROOT, SLOT, SLOT_DIR, OBJECT_ID, KIND
from static_identity import read, sha, write_once, verify, now

from candidate_binding20 import assert_bound, binding, RESULT_ID_FIELDS, rejected_ledger_row
from execution_outcome20 import returned_disposition, candidate_static_disposition

BUDGET = 14
BATCH_STARTS = 10
EPISODE = SLOT_DIR / "episodes" / f"{SLOT:02d}"
LEDGER = VERSION / "slot_ledger.json"


class StaticCandidateRejected(Exception):
    pass


def relative(path):
    return path.relative_to(VERSION).as_posix()


def load_registration():
    registration = read(SLOT_DIR / "preregistration.json")
    slot = registration["slots"][SLOT - 1]
    if (slot["slot"], slot["kind"], slot["object_id"]) != (SLOT, KIND, OBJECT_ID):
        raise ValueError("registered slot identity mismatch")
    if read(SLOT_DIR / "trajectory_plan.json")["candidate_id"] != slot["candidate_id"]:
        raise ValueError("trajectory candidate mismatch")
    assert_bound(read(LEDGER)["slots"][SLOT-1],binding(SLOT))
    return registration, slot


def ledger_before():
    initial = read(VERSION / "ledger_initial.json")
    current = read(LEDGER)
    charged_order = initial["charged_in_actual_execution_order"].copy()
    starts = initial["starts"]
    for number in range(1, BUDGET+1):
        row = current["slots"][number-1]
        if 9 <= number < SLOT:
            previous = VERSION/"slots"/f"{number:02d}"
            result = read(previous/"session_result.json")
            if result["batch_can_continue"] is not True or result["starts_before"] != starts:
                raise ValueError("Prior slot failed or chronological accounting differs")
            if row["charged"] is True:
                if not result["charged"] or not row.get("actual_step_calls"):
                    raise ValueError("Prior charge lacks observed native steps")
                charged_order.append(number)
                starts += 1
            elif row != rejected_ledger_row(initial["slots"][number-1], previous/"static_rejection.json"):
                raise ValueError("Prior uncharged slot lacks a verified geometric rejection")
            if result["starts_after"] != starts:
                raise ValueError("Prior closing receipt differs")
        elif row != initial["slots"][number-1]:
            raise ValueError("Historical or future slot was modified")
    if (current["starts"] != starts or current["existing_budget"] != BUDGET
            or current["remaining_total_budget"] != BUDGET-starts
            or current["charged_in_actual_execution_order"] != charged_order):
        raise ValueError("Cumulative charge mismatch")
    if current["slots"][SLOT - 1]["charged"] or (SLOT == 9 and sha(LEDGER) != sha(VERSION / "ledger_initial.json")):
        raise ValueError("slot already started or baseline changed")
    return current


def commit_ledger(value, stage):
    pending = VERSION / f"slot_ledger.{SLOT:02d}.{stage}.pending"
    write_once(pending, value)
    expected = sha(pending)
    os.replace(pending, LEDGER)
    if sha(LEDGER) != expected:
        raise RuntimeError("atomic ledger readback failed")


def book_start(initial, slot):
    ledger = copy.deepcopy(initial)
    ledger.update(schema="IP140_R2_smoke21_budget", version=VERSION.name, at_utc=now(),
                  starts=initial["starts"]+1, remaining_total_budget=BUDGET-initial["starts"]-1,
                  unused_original_budget=BUDGET-initial["starts"]-1, budget_added=0,
                  starts_this_version=initial["starts"]-BATCH_STARTS+1, prior_starts=BATCH_STARTS,
                  prior_step_returns=read(VERSION / "ledger_initial.json")["physical_steps"],
                  preregistration_sha256=sha(SLOT_DIR / "preregistration.json"),
                  prior_final_ledger=dict(path="ledger_initial.json", sha256=sha(VERSION/"ledger_initial.json")),
                  charged_in_actual_execution_order=initial["charged_in_actual_execution_order"]+[SLOT],
                  terminal=f"slot{SLOT}_first_native_invocation")
    row = ledger["slots"][SLOT-1]
    row.update(charged=True, state="started", 
               candidate_id=slot["candidate_id"], kind=KIND, object_id=OBJECT_ID, not_p3=True,
               reservation=relative(EPISODE / "reservation.json"),
               first_native_invocation=relative(EPISODE / "native_call_observed.json"),
               first_native_invocation_sha256=sha(EPISODE / "native_call_observed.json"))
    write_once(EPISODE / "ledger_first_invocation.json", ledger)
    commit_ledger(ledger, "start")


def finish_accounting(initial, trace, status):
    calls = int(trace.actual_step_calls) if trace is not None else 0
    returns = int(trace.returns.get("stepSimulation", 0)) if trace is not None else 0
    status.update(actual_step_calls=calls, actual_step_returns=returns, unknown_step_returns=calls-returns)
    if not (0 <= returns <= calls <= 4728):
        raise RuntimeError("invalid native step counts")
    if not calls:
        if (EPISODE/"native_call_dispatch.json").exists() or (EPISODE/"native_call_observed.json").exists():
            raise RuntimeError("first native dispatch needs evidence reconciliation")
        if read(LEDGER) != initial:
            raise RuntimeError("ledger changed without native step")
        if status.get("static_candidate_rejection") and status.get("batch_can_continue") is True:
            if status["dynamic_connection_attempted"]:
                raise RuntimeError("Geometric nonadmission cannot contain a dynamic connection")
            receipt_path=SLOT_DIR/"static_rejection.json"
            identity=binding(SLOT)
            write_once(receipt_path,dict(**{k:identity[k] for k in RESULT_ID_FIELDS},
                       version=VERSION.name,charged=False,step_calls=0,static_admitted=False,
                       rejection=status["static_candidate_rejection"],
                       static_api_summary_sha256=sha(SLOT_DIR/"static_preflight/full_api/summary.json")))
            ledger=copy.deepcopy(initial)
            ledger["slots"][SLOT-1]=rejected_ledger_row(initial["slots"][SLOT-1],receipt_path)
            ledger.update(at_utc=now(),terminal=f"slot{SLOT}_static_inadmissible_uncharged")
            commit_ledger(ledger,"candidate_rejection")
        return
    ledger = read(LEDGER)
    if ledger != read(EPISODE/"ledger_first_invocation.json") or ledger["starts"]!=initial["starts"]+1:
        raise RuntimeError("charge evidence mismatch")
    outcome_path = EPISODE / "result.json"
    outcome = read(outcome_path) if outcome_path.exists() else None
    ledger.update(at_utc=now(), returned_episodes=initial["returned_episodes"]+int(outcome is not None),
                  unknown_started_episodes=initial.get("unknown_started_episodes",0)+int(outcome is None),
                  physical_steps=initial["physical_steps"]+returns,
                  terminal=f"slot{SLOT}_returned" if outcome else f"slot{SLOT}_unknown")
    baseline=read(VERSION/"ledger_initial.json")
    batch_calls=calls+sum(ledger["slots"][i-1]["actual_step_calls"] for i in range(9,SLOT))
    batch_returns=ledger["physical_steps"]-baseline["physical_steps"]
    ledger.update(steps_this_version=batch_returns,physical_step_calls_this_version=batch_calls,
                  unknown_physical_step_returns_this_version=batch_calls-batch_returns)
    row = ledger["slots"][SLOT-1]
    row.update(actual_step_calls=calls, actual_step_returns=returns, unknown_step_returns=calls-returns,
               state="returned" if outcome is not None else "started_outcome_unknown")
    if outcome is not None:
        assert_bound(outcome,binding(SLOT))
        for key in ("episode_completed","engineering_gate_passed","grasp_success","slot_passed","failure_stage","failure_reason","outcome_class"):
            row[key]=outcome.get(key)
        row.update(result=relative(outcome_path), result_sha256=sha(outcome_path),
                   telemetry=relative(EPISODE/"telemetry.jsonl.gz"),telemetry_sha256=sha(EPISODE/"telemetry.jsonl.gz"),
                   state="returned_passed" if outcome["slot_passed"] else "returned_failed")
    if any(ledger["slots"][i]!=initial["slots"][i] for i in range(BUDGET) if i!=SLOT-1):
        raise RuntimeError("another slot's ledger changed")
    status.update(charged=True,starts_after=ledger["starts"],remaining_after=ledger["remaining_total_budget"])
    write_once(EPISODE/"ledger_final.json",ledger)
    commit_ledger(ledger, "final")


def check_disconnect(directory):
    zero=read(directory/"zero_step/result.json")
    api=read(directory/"full_api/summary.json")
    if zero.get("disconnected") is not True or zero.get("disconnect_exception"):
        raise ValueError("DIRECT disconnect unconfirmed")
    if (api["calls"]!=api["returns"] or api["exceptions"] or api["returns"].get("connect")!=1 or
        api["returns"].get("disconnect")!=1 or api["returns"].get("setCollisionFilterPair")!=14):
        raise ValueError("API exception, unmatched return or incorrect session scope")
    return api


def main():
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--static-admission-then-slot-once"]:
        raise SystemExit("Explicit project root and one-slot authorization flag required")
    if Path(sys.executable).resolve()!=(ROOT/read(ROOT/"configs/simulation_v52e.json")["python"]).resolve():
        raise SystemExit("Frozen interpreter required")
    verify()
    initial=ledger_before()
    registration, slot=load_registration()
    for name in ("session_claim.json","session_result.json","static_preflight","zero_step","full_api","episodes"):
        if (SLOT_DIR/name).exists():
            raise SystemExit("Existing evidence prevents retry: "+name)
    write_once(SLOT_DIR/"session_claim.json",dict(slot=SLOT,pid=os.getpid(),at=now(),max_connections=2,
               input_identity_sha256=sha(VERSION/"static_input_identity.json"),starts_before=initial["starts"]))
    initial_digest=sha(LEDGER)
    write_once(SLOT_DIR/"ledger_before.json",initial)
    status=dict(version=VERSION.name,slot=SLOT,object_id=OBJECT_ID,kind=KIND,candidate_id=slot["candidate_id"],
                starts_before=initial["starts"],starts_after=initial["starts"],remaining_after=BUDGET-initial["starts"],
                total_budget=BUDGET,charged=False,batch_can_continue=False,at=now(),
                registered_surface_revision="IP140_R2",development_corrective_slot9=(SLOT==9),
                static_connection_attempted=False,dynamic_connection_attempted=False)
    trace=None
    try:
        for filename in ("offline_gate_replay.json","positive_negative_replay.json","schema_replay19.json","scope_replay20.json","surface_replay21.json"):
            report=read(VERSION/filename)
            if not (report["passed"] is True and report["cases"] and all(c["passed"] is True for c in report["cases"])):
                raise ValueError("offline admission failed: "+filename)
        from zero_scene import run_zero, gate_valid
        from contact_aware_path import scan
        from static_admission_check import verify_static_complete
        static=SLOT_DIR/"static_preflight"
        def preflight(p,client,body,links,manifest,loaded,gate):
            if (not gate_valid(gate) or p.step_authorized or p.counts["stepSimulation"] or
                p.connection_scope!="contact_aware_static" or gate["dynamic_execution_authorized"] or loaded!=registration):
                raise ValueError("static handoff identity mismatch")
            scan(p,client,body,links,manifest,loaded,gate,output_root=static)
        status.update(phase="static_zero_and_path",static_connection_attempted=True)
        run_zero(on_pass=preflight,output_root=static,session_kind="contact_aware_static")
        static_api=check_disconnect(static)
        if static_api["calls"].get("stepSimulation",0):
            raise RuntimeError("physical step in static connection")
        if not gate_valid(read(static/"zero_readback.json")):
            raise ValueError("static zero failed")
        path=read(static/"path_result.json")
        status["static_snapshots_completed"]=path["snapshots_completed"]
        status["static_first_failure"]=path.get("first_failure")
        if path["passed"] is not True:
            geometric=candidate_static_disposition(static,read,path)
            if geometric:
                status["static_candidate_rejection"]=geometric
                raise StaticCandidateRejected(geometric["reason"])
            raise ValueError("Shared or unclassified static failure")
        admission=verify_static_complete()
        if admission.get("passed") is not True:raise ValueError("static receipt failed")
        write_once(SLOT_DIR/"static_admission_receipt.json",admission)
        write_once(SLOT_DIR/"dynamic_connection_claim.json",dict(at=now(),allowed_slot=SLOT,prior_static_disconnected=True,
                   static_admission_sha256=sha(SLOT_DIR/"static_admission_receipt.json"),one_connection_only=True))
        def dynamic(p,client,body,links,manifest,loaded,gate):
            nonlocal trace
            trace=p
            if (not gate_valid(gate) or loaded!=registration or read(SLOT_DIR/"zero_readback.json")!=gate
                or p.step_authorized or p.counts["stepSimulation"] or p.connection_scope!="dynamic"
                or gate["dynamic_execution_authorized"] is not True or gate["eligible_slots_after_zero_gate"]!=[SLOT]):
                raise RuntimeError("dynamic zero handoff mismatch")
            verify()
            EPISODE.mkdir(parents=True,exist_ok=False)
            write_once(EPISODE/"scene_state.json",dict(slot=SLOT,at=now(),client=client,robot=body,
                       object=gate["object_body_id"],table=gate["table_body_id"],fresh_dynamic_scene=True,
                       zero_sha256=sha(SLOT_DIR/"zero_readback.json")))
            def commit(sequence,returned_normally,error):
                dispatch=EPISODE/"native_call_dispatch.json"
                if read(dispatch)["api_sequence"]!=sequence or sha(LEDGER)!=initial_digest:
                    raise RuntimeError("first native dispatch unconfirmed or unexpected ledger state")
                write_once(EPISODE/"native_call_observed.json",dict(slot=SLOT,client_id=client,method="stepSimulation",
                           api_sequence=sequence,at=now(),starts_before=initial["starts"],starts_after=initial["starts"]+1,
                           remaining_after=BUDGET-initial["starts"]-1,charged=True,first_call_returned_normally=returned_normally,
                           native_error=error,dispatch_sha256=sha(dispatch),reservation_sha256=sha(EPISODE/"reservation.json")))
                book_start(initial,slot)
            def reserve():
                if (p.step_authorized or sha(LEDGER)!=initial_digest or p.counts["stepSimulation"] or
                    p.protected_object!=(gate["object_body_id"],client) or not p.protected_write_started):
                    raise ValueError("reservation prerequisites failed")
                receipt=dict(candidate_vector_sha256=binding(SLOT)["candidate_vector_sha256"],slot=SLOT,object_id=OBJECT_ID,kind=KIND,candidate_id=slot["candidate_id"],
                             not_p3=True,charged=False,starts_before=initial["starts"],starts_after=initial["starts"],
                             remaining_after=BUDGET-initial["starts"],total=BUDGET,client_id=client,object_body_id=gate["object_body_id"],
                             at=now(),prior_ledger_sha256=initial_digest,zero_sha256=sha(SLOT_DIR/"zero_readback.json"),
                             static_admission_sha256=sha(SLOT_DIR/"static_admission_receipt.json"),
                             asset_typed_zero_sha256=sha(SLOT_DIR/"asset_typed_zero_receipt.json"))
                write_once(EPISODE/"reservation.json",receipt)
                p.authorize_physics(EPISODE/"reservation.json",LEDGER,client=client,on_first_observed=commit)
                return receipt
            from object_episode import run_slot
            obj=read(SLOT_DIR/"zero_step"/f"object{OBJECT_ID}_readback.json")
            gravity=gate["gravity_world_m_s2_readback"]
            if len(gravity)!=3 or not all(math.isfinite(x) for x in gravity):
                raise ValueError("actual gravity not available")
            status["phase"]="dynamic_episode"
            try:
                run_slot(p,client,body,slot,EPISODE,evaluator_context=dict(object_body_id=gate["object_body_id"],
                         table_body_id=gate["table_body_id"],object_mass_kg=obj["dynamics"][0],gravity_world_m_s2=gravity),
                         before_first_step=reserve)
            finally:
                p.step_authorized=False
        status.update(phase="dynamic_zero",dynamic_connection_attempted=True)
        run_zero(on_pass=dynamic,output_root=SLOT_DIR,session_kind="dynamic")
        api=check_disconnect(SLOT_DIR)
        if not gate_valid(read(SLOT_DIR/"zero_readback.json")):
            raise ValueError("dynamic zero failed")
        episode=read(EPISODE/"result.json")
        assert_bound(episode,binding(SLOT))
        disposition=returned_disposition(episode,api["returns"].get("stepSimulation",0))
        status.update(stop=disposition,batch_can_continue=True,slot_passed=episode["slot_passed"],
                      grasp_success=episode["grasp_success"],episode_sha256=sha(EPISODE/"result.json"))
    except StaticCandidateRejected as exc:
        status.update(stop="candidate_static_inadmissible",batch_can_continue=True,reason=str(exc),
                      static_admitted=False,grasp_success=None,slot_passed=None,engineering_gate_passed=None)
    except BaseException as exc:
        status.update(stop="shared_engineering_stop",batch_can_continue=False,error=repr(exc),traceback=traceback.format_exc())
    finally:
        try:
            finish_accounting(initial,trace,status)
            verify()
        except BaseException as exc:
            status.update(stop="accounting_review_required",batch_can_continue=False,accounting_error=repr(exc),
                          accounting_traceback=traceback.format_exc())
        status["finished_utc"]=now()
        write_once(SLOT_DIR/"session_result.json",status)
        print(json.dumps(status,ensure_ascii=True))
    return 0 if status["batch_can_continue"] else 2


if __name__=="__main__":
    raise SystemExit(main())
