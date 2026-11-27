"""One static-only connection; never starts a physical slot."""
import os,sys,json,traceback
from pathlib import Path
from static_identity import HERE,ROOT,sha,read,write_once,verify,now
def main():
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--zero-only-contact-aware-review"]:
        raise SystemExit("Explicit root and static-only flag required")
    expected=ROOT/read(ROOT/"configs/simulation_v52e.json")["python"]
    if Path(sys.executable).resolve()!=expected.resolve():raise SystemExit("Frozen interpreter required")
    for rel in ("session_claim.json","session_result.json","zero_step","full_api","zero_readback.json","dual_channel_snapshots.jsonl.gz","path_result.json","contact_entries","STOP.json","result.json"):
        if (HERE/rel).exists():raise SystemExit("Existing output prevents duplicate connection: "+rel)
    status=dict(version=HERE.name,phase="offline",started_at=now(),physics_steps=0,new_charges=0,charged=False,slot14_started=False)
    write_once(HERE/"session_claim.json",dict(pid=os.getpid(),at=now(),identity_sha256=sha(HERE/"static_input_identity.json"),
        static_connections_allowed=1,physical_steps_allowed=0,slots_allowed=[]))
    try:
        verify()
        replay=read(HERE/"positive_negative_replay.json")
        if not replay.get("passed") or not replay.get("cases") or not all(x.get("passed") is True for x in replay["cases"]):
            raise ValueError("offline replay failed; no connection")
        status["phase"]="zero"
        from zero_scene import run_zero,gate_valid
        from contact_aware_path import scan
        found={}
        def admitted(p,client,body,links,manifest,registration,gate):
            if not gate_valid(gate) or p.step_authorized or p.counts["stepSimulation"]:raise ValueError("static gate mismatch")
            status["phase"]="dual_channel_static_path"
            found.update(scan(p,client,body,links,manifest,registration,gate))
        run_zero(on_pass=admitted,session_kind="contact_aware_static")
        final=read(HERE/"zero_step/result.json")
        api=read(HERE/"full_api/summary.json")
        if not final.get("disconnected") or api["calls"].get("disconnect")!=1 or api["returns"].get("disconnect")!=1:
            raise ValueError("disconnect not confirmed")
        if api["calls"].get("stepSimulation",0):raise ValueError("forbidden physics call")
        status.update(stop=found.get("status","zero_gate_blocked"),zero_passed=read(HERE/"zero_readback.json").get("passed"),
            path_passed=found.get("passed"),snapshots_completed=found.get("snapshots_completed",0))
        verify()
    except BaseException as exc:status.update(stop="static_review_exception",exception=repr(exc),traceback=traceback.format_exc())
    finally:
        status["ended_at"]=now();write_once(HERE/"session_result.json",status)
        print(json.dumps(status,ensure_ascii=True))
if __name__=="__main__":main()
