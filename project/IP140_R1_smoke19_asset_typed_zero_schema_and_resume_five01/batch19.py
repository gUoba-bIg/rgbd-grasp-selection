"""One fixed-order batch; a single native stop cannot be retried by this runner."""
from pathlib import Path
import os
import subprocess
import sys
from static_identity import HERE,ROOT,ORDER,read,write_once,now,verify

def main():
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--execute-frozen-five-once"]:
        raise SystemExit("Expected project root and explicit smoke19 batch invocation")
    if Path(sys.executable).resolve()!=(ROOT/read(ROOT/"configs/simulation_v52e.json")["python"]).resolve():
        raise SystemExit("Frozen Python required")
    verify()
    if (HERE/"batch_session_result.json").exists() or any((HERE/"slots"/f"{i:02d}"/"session_claim.json").exists() for i in ORDER):
        raise SystemExit("Prior execution evidence present")
    write_once(HERE/"batch_session_claim.json",dict(version=HERE.name,pid=os.getpid(),started_utc=now(),order=ORDER,budget_before=9))
    statuses=[]
    for i in ORDER:
        slot=HERE/"slots"/f"{i:02d}"
        env={**os.environ,"IP140_SLOT":str(i),"PYTHONUTF8":"1","PYTHONDONTWRITEBYTECODE":"1"}
        with (slot/"worker_stdout.log").open("xb") as output:
            proc=subprocess.run([sys.executable,"-B",str(HERE/"run_authorized.py"),"--static-admission-then-slot-once"],
                                cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,check=False)
        report=slot/"session_result.json"
        status=read(report) if report.is_file() else {"batch_can_continue":False,"stop":"entry_failed_before_session_result","worker_log":str((slot/"worker_stdout.log").relative_to(HERE))}
        statuses.append(dict(slot=i,exit_code=proc.returncode,session=status))
        if proc.returncode!=0 or not status["batch_can_continue"]:
            break
    ledger=read(HERE/"slot_ledger.json")
    write_once(HERE/"batch_session_result.json",dict(version=HERE.name,finished_utc=now(),workers=statuses,
               charged=ledger["starts"]-9,cumulative_starts=ledger["starts"],remaining=ledger["remaining_total_budget"],
               following_slots_not_entered=list(ORDER[len(statuses):]),retries=0))
    return 0 if all(s["session"]["batch_can_continue"] for s in statuses) and len(statuses)==5 else 2

if __name__=="__main__":raise SystemExit(main())
