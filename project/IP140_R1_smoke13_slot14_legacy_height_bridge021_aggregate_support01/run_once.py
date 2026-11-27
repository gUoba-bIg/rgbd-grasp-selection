"""Single authorized bridge admission. Precheck is a separate zero-step connection."""
import os,sys,traceback,json
from pathlib import Path
from input_identity import HERE,ROOT,REGISTERED_ID,digest,read,timestamp,verify,write_once

def main():
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--offline-precheck-zero-and-slot-14"]:
        raise SystemExit("Explicit root and slot14 flag required")
    expected=ROOT/read(ROOT/"configs/simulation_v52e.json")["python"]
    if Path(sys.executable).resolve()!=expected.resolve(): raise SystemExit("Frozen interpreter required")
    for name in ("session_claim.json","offline_gate_replay.json","STOP.json","result.json","full_api","episodes","zero_step","zero_readback.json","slot_ledger.json","path_precheck"):
        if (HERE/name).exists(): raise SystemExit("Existing output prevents retry: "+name)
    initial=read(HERE/"ledger_initial.json")
    status=dict(version=HERE.name,slot=14,parent_slot=7,at=timestamp(),phase="identity",charged=False)
    write_once(HERE/"session_claim.json",dict(pid=os.getpid(),start=timestamp(),allowed_slot=14,input_identity_sha256=digest(HERE/"input_identity.json")))
    try:
        verify()
        status["phase"]="offline_replay"
        from offline_replay import persist_report
        replay=persist_report()
        if not replay.get("passed") or not replay.get("cases") or not all(r.get("passed") is True for r in replay["cases"]):
            status["stop"]="offline_replay_failed";return
        verify()
        from static_path_offline_replay import report as static_replay
        static_check=static_replay()
        if static_check!=read(HERE/"static_path_offline_replay.json") or not static_check["passed"]:
            status["stop"]="static_path_offline_replay_failed";return
        if not read(HERE/"candidate_difference_proof.json")["passed"]:
            status["stop"]="candidate_proof_failed";return
        from zero_scene import run_zero,gate_valid
        from path_precheck import scan_path
        status["phase"]="path_precheck_zero"
        path_result={}
        def static_admitted(p,client,body,links,model,registration,gate):
            if not gate_valid(gate) or p.counts["stepSimulation"]!=0 or p.connection_scope!="path_precheck":
                raise RuntimeError("Precheck zero gate mismatch")
            status["phase"]="full_static_path"
            path_result.update(scan_path(p,client,body,links,model,registration,gate,HERE/"path_precheck"))
        run_zero(on_pass=static_admitted,output_root=HERE/"path_precheck",session_kind="path_precheck")
        zero_pre=read(HERE/"path_precheck/zero_readback.json")
        if not gate_valid(zero_pre):
            status["stop"]="precheck_zero_gate_blocked";return
        if path_result.get("passed") is not True:
            status["stop"]="static_path_admission_blocked";return
        pre_api=read(HERE/"path_precheck/full_api/summary.json")
        pre_final=read(HERE/"path_precheck/zero_step/result.json")
        if pre_api["calls"].get("stepSimulation",0): raise RuntimeError("Physical step in precheck")
        if (pre_final.get("disconnected") is not True or pre_final.get("disconnect_exception")
            or pre_api["calls"].get("disconnect")!=1 or pre_api["returns"].get("disconnect")!=1):
            raise RuntimeError("Precheck disconnect not confirmed; fresh dynamic blocked")
        verify()
        status["phase"]="fresh_dynamic_zero"
        def admitted(p,client,body,links,model,registration,gate):
            persisted=read(HERE/"zero_readback.json")
            if (not gate_valid(gate) or persisted!=gate or gate["physics_step_calls"]!=0 or p.counts["stepSimulation"]!=0
                or p.connection_scope!="dynamic" or registration["slots"][13]["candidate_id"]!=REGISTERED_ID
                or registration!=read(HERE/"preregistration.json") or path_result.get("passed") is not True):
                raise RuntimeError("Fresh dynamic admission mismatch")
            verify()
            dest=HERE/"episodes"/"14";dest.mkdir(parents=True,exist_ok=False)
            write_once(dest/"scene_state.json",dict(client=client,robot=body,object=gate["object_body_id"],table=gate["table_body_id"],utc=timestamp(),zero_sha256=digest(HERE/"zero_readback.json"),static_path_sha256=digest(HERE/"path_precheck/result.json")))
            def charge():
                if (p.counts["stepSimulation"] or p.step_authorized or status["charged"] or p.connection_scope!="dynamic"
                    or not p.protected_write_started or p.protected_object!=(gate["object_body_id"],client)):
                    raise RuntimeError("First-step billing guard")
                record=dict(slot=14,parent_slot=7,candidate_id=REGISTERED_ID,not_p3=True,development_corrective_bridge=True,
                    independent_confirmation=False,starts_before=8,starts_after=9,remaining_after=5,total=14,ts=timestamp(),
                    zero_sha256=digest(HERE/"zero_readback.json"),offline_sha256=digest(HERE/"offline_gate_replay.json"),
                    inputs_sha256=digest(HERE/"input_identity.json"),static_path_sha256=digest(HERE/"path_precheck/result.json"))
                write_once(dest/"precharge.json",record)
                status["charged"]=True;p.step_authorized=True
                return record
            obj=read(HERE/"zero_step/object021_readback.json")
            gravity=gate["gravity_world_m_s2_readback"]
            if not isinstance(gravity,(list,tuple)): gravity=[0.,0.,gravity]
            from object_episode import run_slot
            status["phase"]="episode"
            try:
                run_slot(p,client,body,registration["slots"][13],dest,evaluator_context=dict(
                    object_body_id=gate["object_body_id"],table_body_id=gate["table_body_id"],
                    object_mass_kg=obj["dynamics"][0],gravity_world_m_s2=gravity),before_first_step=charge)
                status["stop"]="slot14_returned" if status["charged"] else "setup_failed_before_first_step"
                status["episode_result_sha256"]=digest(dest/"result.json")
            finally:p.step_authorized=False
        run_zero(on_pass=admitted,session_kind="dynamic")
        status.setdefault("stop","dynamic_zero_gate_blocked")
    except BaseException as exc:
        status.update(stop="exception_stop",error=repr(exc),traceback=traceback.format_exc())
    finally:
        status["finished_utc"]=timestamp()
        write_once(HERE/"session_result.json",status)
        print(json.dumps(status,ensure_ascii=True))
if __name__=="__main__": main()
