"""One preflight, then at most one fresh dynamic slot14. Never retries a connection."""
from pathlib import Path
import copy
import json
import math
import os
import sys
import traceback
from static_identity import HERE,ROOT,read,sha,write_once,verify,now
from input_identity import REGISTERED_ID


def atomic_ledger(value,tag,*,exact_bytes=None):
    target=HERE/"slot_ledger.json"
    temp=HERE/("slot_ledger."+tag+".pending")
    if target.resolve()!=HERE.resolve()/"slot_ledger.json" or temp.parent.resolve()!=HERE.resolve():
        raise ValueError("ledger path escaped independent version")
    payload=exact_bytes if exact_bytes is not None else (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")
    with temp.open("xb") as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(temp,target)
    if target.read_bytes()!=payload:raise RuntimeError("atomic ledger readback mismatch")


def validate_initial_ledger():
    initial=read(HERE/"ledger_initial.json")
    if sha(HERE/"slot_ledger.json")!=sha(HERE/"ledger_initial.json"):
        raise RuntimeError("no re-entry after any ledger change")
    if ((initial["starts"],initial["existing_budget"],initial["remaining_total_budget"])!=(8,14,6)
        or initial["slots"][13]["charged"] or any(x["charged"] for x in initial["slots"][7:12])
        or initial["slots"][13]["candidate_id"]!=REGISTERED_ID):
        raise RuntimeError("only unstarted bridge slot14 may be considered")
    return initial


def finalize_ledger(initial,trace,status):
    """Preserve every attempt; a reservation without any native-call intent is not a start."""
    dest=HERE/"episodes"/"14"
    reserve=dest/"reservation.json"
    observed=dest/"native_call_observed.json"
    calls=int(trace.actual_step_calls) if trace is not None else 0
    returns=int(trace.returns.get("stepSimulation",0)) if trace is not None else 0
    status.update(actual_step_calls=calls,actual_step_returns=returns,unknown_step_returns=calls-returns)
    if returns>calls:raise RuntimeError("impossible physical API accounting")
    if calls==0:
        if reserve.exists():
            write_once(dest/"reservation_unused.json",dict(at=now(),reason="Reservation without native invocation",
                reservation_sha256=sha(reserve),physics_step_calls=0,physics_step_returns=0))
        if observed.exists() or sha(HERE/"slot_ledger.json")!=sha(HERE/"ledger_initial.json"):
            raise RuntimeError("unexplained accounting evidence without actual native invocation")
        status.update(charged=False,starts_after=8,remaining_after=6,total_budget=14)
        return
    if not reserve.is_file() or not observed.is_file():raise RuntimeError("actual native call proof missing; preserve and review")
    ledger=copy.deepcopy(initial)
    episode_file=dest/"result.json";episode=read(episode_file) if episode_file.is_file() else None
    ledger.update(schema="IP140_smoke16_final_ledger",version=HERE.name,at_utc=now(),starts=9,existing_budget=14,
        total_slots=14,remaining_total_budget=5,starts_this_version=1,steps_this_version=returns,
        physical_step_calls_this_version=calls,unknown_physical_step_returns_this_version=calls-returns,
        physical_steps=initial.get("physical_steps",0)+returns,
        returned_episodes=initial.get("returned_episodes",8)+(1 if episode is not None else 0),
        unknown_started_episodes=initial.get("unknown_started_episodes",0)+(0 if episode is not None else 1),
        charged_in_actual_execution_order=initial["charged_in_actual_execution_order"]+[14],
        terminal="slot14_returned" if episode is not None else "slot14_started_outcome_unknown",
        additional_corrective_slot_available=False,prior_final_ledger=dict(path="ledger_initial.json",sha256=sha(HERE/"ledger_initial.json")))
    entry=ledger["slots"][13]
    entry.update(charged=True,state=("returned_passed" if episode.get("slot_passed") is True else "returned_failed") if episode else "started_outcome_unknown",
        reservation="episodes/14/reservation.json",reservation_sha256=sha(reserve),
        first_native_invocation="episodes/14/native_call_observed.json",first_native_invocation_sha256=sha(observed),
        result="episodes/14/result.json" if episode else None,result_sha256=sha(episode_file) if episode else None,
        actual_step_calls=calls,actual_step_returns=returns,unknown_step_returns=calls-returns,
        smoke16_record="session_result.json",independent_confirmation=False,not_p3=True)
    if (dest/"telemetry.jsonl.gz").is_file():
        entry.update(telemetry="episodes/14/telemetry.jsonl.gz",telemetry_sha256=sha(dest/"telemetry.jsonl.gz"))
    if episode:
        for name in ("episode_completed","engineering_gate_passed","grasp_success","slot_passed","failure_stage","failure_reason"):
            entry[name]=episode.get(name)
    if ledger["slots"][:13]!=initial["slots"][:13]:raise RuntimeError("historical slots changed")
    write_once(dest/"ledger_final.json",ledger)
    atomic_ledger(ledger,"final")
    status.update(charged=True,starts_after=9,remaining_after=5,total_budget=14,
        episode_result_sha256=sha(episode_file) if episode else None)


def main():
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--static-admission-then-slot14-once"]:
        raise SystemExit("Explicit project root and one-slot authorization flag required")
    expected=ROOT/read(ROOT/"configs/simulation_v52e.json")["python"]
    if Path(sys.executable).resolve()!=expected.resolve():raise SystemExit("Frozen isolated interpreter required")
    forbidden=("session_claim.json","session_result.json","dynamic_connection_claim.json","static_admission_receipt.json",
        "static_preflight","zero_step","zero_readback.json","full_api","episodes","STOP.json","result.json")
    for name in forbidden:
        if (HERE/name).exists():raise SystemExit("Existing evidence prevents retry: "+name)
    initial=validate_initial_ledger()
    status=dict(version=HERE.name,slot=14,parent_slot=7,at=now(),phase="offline",charged=False,
        starts_before=8,starts_after=8,total_budget=14,remaining_after=6,static_connection_attempted=False,dynamic_connection_attempted=False)
    state={}
    write_once(HERE/"session_claim.json",dict(pid=os.getpid(),start=now(),allowed_slot=14,
        max_static_connections=1,max_dynamic_connections=1,max_new_attempts=1,
        input_identity_sha256=sha(HERE/"static_input_identity.json")))
    try:
        verify()
        for name in ("positive_negative_replay.json","offline_gate_replay.json"):
            report=read(HERE/name)
            if report.get("passed") is not True or not report.get("cases") or not all(x.get("passed") is True for x in report["cases"]):
                status.update(stop="offline_replay_failed",failed_report=name);return
        from zero_scene import run_zero,gate_valid
        from contact_aware_path import scan
        from static_admission_check import verify_static_complete
        found={}
        def static_admitted(p,client,body,links,manifest,registration,gate):
            if (not gate_valid(gate) or p.step_authorized or p.counts["stepSimulation"]
                or p.connection_scope!="contact_aware_static" or gate.get("dynamic_execution_authorized") is not False):
                raise RuntimeError("static zero gate mismatch")
            status["phase"]="full_static_preflight"
            found.update(scan(p,client,body,links,manifest,registration,gate,output_root=HERE/"static_preflight"))
        status.update(phase="static_zero",static_connection_attempted=True)
        run_zero(on_pass=static_admitted,output_root=HERE/"static_preflight",session_kind="contact_aware_static")
        static_final=read(HERE/"static_preflight/zero_step/result.json")
        static_api=read(HERE/"static_preflight/full_api/summary.json")
        if (static_final.get("disconnected") is not True or static_final.get("disconnect_exception")
            or static_api["calls"].get("disconnect")!=1 or static_api["returns"].get("disconnect")!=1
            or static_api["calls"].get("stepSimulation",0)):
            raise RuntimeError("preflight disconnect/zero-physics proof missing")
        static_zero=read(HERE/"static_preflight/zero_readback.json")
        if not gate_valid(static_zero) or found.get("passed") is not True:
            status.update(stop="static_admission_blocked",static_zero_passed=static_zero.get("passed"),
                static_path_passed=found.get("passed"),static_snapshots_completed=found.get("snapshots_completed",0));return
        admission=verify_static_complete()
        write_once(HERE/"static_admission_receipt.json",admission)
        verify()
        write_once(HERE/"dynamic_connection_claim.json",dict(at=now(),pid=os.getpid(),one_connection_only=True,
            allowed_slot=14,prior_static_disconnected=True,static_admission_sha256=sha(HERE/"static_admission_receipt.json")))
        status.update(phase="fresh_dynamic_zero",dynamic_connection_attempted=True)
        def dynamic_admitted(p,client,body,links,manifest,registration,gate):
            state.update(trace=p,client=client)
            if (not gate_valid(gate) or read(HERE/"zero_readback.json")!=gate
                or p.counts["stepSimulation"] or p.step_authorized or p.connection_scope!="dynamic"
                or gate.get("dynamic_execution_authorized") is not True or gate.get("eligible_slots_after_zero_gate")!=[14]
                or registration!=read(HERE/"preregistration.json") or registration["slots"][13]["candidate_id"]!=REGISTERED_ID):
                raise RuntimeError("fresh dynamic zero handoff invalid")
            verify()
            dest=HERE/"episodes"/"14";dest.mkdir(parents=True,exist_ok=False)
            write_once(dest/"scene_state.json",dict(at=now(),client=client,robot=body,object=gate["object_body_id"],table=gate["table_body_id"],
                fresh_dynamic_scene=True,zero_sha256=sha(HERE/"zero_readback.json"),static_admission_sha256=sha(HERE/"static_admission_receipt.json")))
            def reserve_before_first_step():
                if (p.counts["stepSimulation"] or p.step_authorized or state.get("reserved")
                    or sha(HERE/"slot_ledger.json")!=sha(HERE/"ledger_initial.json")
                    or p.protected_object!=(gate["object_body_id"],client) or not p.protected_write_started):
                    raise RuntimeError("one-time reservation guard")
                receipt=dict(slot=14,parent_slot=7,candidate_id=REGISTERED_ID,not_p3=True,development_corrective_bridge=True,
                    independent_confirmation=False,charged=False,starts_before=8,starts_after=8,remaining_after=6,total=14,at=now(),
                    client_id=client,robot_body_id=body,object_body_id=gate["object_body_id"],
                    static_admission_sha256=sha(HERE/"static_admission_receipt.json"),zero_sha256=sha(HERE/"zero_readback.json"),
                    offline_sha256=sha(HERE/"offline_gate_replay.json"),positive_gap_offline_sha256=sha(HERE/"positive_negative_replay.json"),
                    inputs_sha256=sha(HERE/"static_input_identity.json"),ledger_initial_sha256=sha(HERE/"ledger_initial.json"),
                    accounting_phase="durable reservation immediately before first native step call")
                write_once(dest/"reservation.json",receipt)
                p.authorize_physics(dest/"reservation.json",HERE/"slot_ledger.json",client=client,on_first_observed=commit_first_call)
                state["reserved"]=True
                return receipt

            def commit_first_call(sequence,returned_normally,error):
                reserve=dest/"reservation.json"
                dispatch=dest/"native_call_dispatch.json"
                if not dispatch.is_file() or read(dispatch)["api_sequence"]!=sequence:
                    raise RuntimeError("native dispatch evidence unavailable; review first-call outcome")
                proof=dict(slot=14,client_id=client,method="stepSimulation",api_sequence=sequence,
                    starts_before=8,starts_after=9,remaining_after=5,total=14,charged=True,at=now(),
                    first_call_returned_normally=returned_normally,native_error=error,
                    physics_result_known=returned_normally,dispatch_sha256=sha(dispatch),reservation_sha256=sha(reserve))
                write_once(dest/"native_call_observed.json",proof)
                ledger=copy.deepcopy(initial)
                ledger.update(schema="IP140_smoke16_first_invocation_ledger",version=HERE.name,at_utc=now(),starts=9,
                    existing_budget=14,total_slots=14,remaining_total_budget=5,starts_this_version=1,
                    charged_in_actual_execution_order=initial["charged_in_actual_execution_order"]+[14],
                    terminal="slot14_first_native_invocation_observed")
                ledger["slots"][13].update(charged=True,state="started",
                    reservation="episodes/14/reservation.json",first_native_invocation="episodes/14/native_call_observed.json",
                    first_native_invocation_sha256=sha(dest/"native_call_observed.json"))
                write_once(dest/"ledger_first_invocation.json",ledger)
                atomic_ledger(ledger,"first_invocation")
            obj=read(HERE/"zero_step/object021_readback.json")
            gravity=gate["gravity_world_m_s2_readback"]
            if not isinstance(gravity,(list,tuple)) or len(gravity)!=3 or not all(math.isfinite(v) for v in gravity):
                raise RuntimeError("actual gravity readback unavailable")
            from object_episode import run_slot
            status["phase"]="slot14"
            try:
                result=run_slot(p,client,body,registration["slots"][13],dest,evaluator_context=dict(
                    object_body_id=gate["object_body_id"],table_body_id=gate["table_body_id"],
                    object_mass_kg=obj["dynamics"][0],gravity_world_m_s2=gravity),before_first_step=reserve_before_first_step)
                state["episode_result"]=result
                status["stop"]="slot14_returned" if p.actual_step_calls else "pre_step_setup_failed"
            finally:p.step_authorized=False
        run_zero(on_pass=dynamic_admitted,output_root=HERE,session_kind="dynamic")
        final=read(HERE/"zero_step/result.json");api=read(HERE/"full_api/summary.json")
        if final.get("disconnected") is not True or api["returns"].get("disconnect")!=1:
            raise RuntimeError("dynamic disconnect outcome unconfirmed")
        status.setdefault("stop","dynamic_zero_blocked")
    except BaseException as exc:
        status.update(stop="exception_stop",error=repr(exc),traceback=traceback.format_exc())
    finally:
        try:
            finalize_ledger(initial,state.get("trace"),status)
            verify()
        except BaseException as exc:
            status.update(accounting_error=repr(exc),accounting_traceback=traceback.format_exc(),manual_review_required=True)
        status["finished_utc"]=now()
        write_once(HERE/"session_result.json",status)
        print(json.dumps(status,ensure_ascii=True))

if __name__=="__main__":main()
