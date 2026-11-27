"""Fresh method executions; each ledger identity has a single create-only dispatch."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import sys
import time
from common_v1 import *
from study_runner import STUDY, environment
from diagnostic_gate import self_test

def scientific_inputs():
    return {f.name:sha(f) for f in (WORK/"supplement").glob("*.py") if f.name in
        {"common_v1.py","method_trial.py","method_seal.py","diagnostic_gate.py","runtime_stages.py","study_runner.py","run_sealed_methods.py","input_fingerprints.py"}}

def one(tid,cohort,lane,contract):
    from method_trial import validate
    disk();need(scientific_inputs()==contract,"method packaging changed")
    need(not (STUDY/"method_STOP.json").exists(),"shared stop")
    path,it,_,_=validate(tid,cohort)
    work=STUDY/"run/jobs"/tid;work.mkdir(exist_ok=False)
    label="study/run/jobs/"+tid
    start=write(label+"/dispatch.json",dict(utc=utc(),tid=tid,lane=lane,intent=sha(path),order=it["execution_order"],permit_once=True))
    cwd=STUDY/"workspaces"/lane
    env=environment(cwd,"physics",it["worker_seed"],work)
    with (work/"stdout.log").open("xb") as so,(work/"stderr.log").open("xb") as se:
        child=subprocess.Popen([str(C/"env/physics/Scripts/python.exe"),"-B","-s",str(WORK/"supplement/method_trial.py"),tid,cohort],cwd=cwd,env=env,stdout=so,stderr=se)
        code=child.wait()
    if code:
        err=(work/"stderr.log").read_text("utf8",errors="replace")
        raise RuntimeError(f"{tid} shared/unknown (exit {code}): "+err[-2600:])
    f=path.parent/"result.json";r=read(f)
    need(r["physical_calls"]==r["physical_returns"] and not r["shared_failure"] and r["intent_sha256"]==sha(path) and
         r["candidate"]==it["selected_candidate_id"] and type(r["grasp_success"]) is bool,"unbound method return")
    return write(label+"/return.json",dict(utc=utc(),tid=tid,dispatch_sha256=start,result_sha256=sha(f),detail=r["detail"],physical_calls=r["physical_calls"]))

def main():
    disk();need((OUT/"admission/cold_3of3.json").exists(),"unverified cold")
    freeze={k:read(STUDY/"preregistration"/(k+"_intents.json")) for k in ("robustness","diagnostic")}
    need(freeze["diagnostic"]["count"]==60 and freeze["robustness"]["count"]<=72,"budget")
    contract=scientific_inputs()
    write("study/preregistration/physical_release.json",dict(utc=utc(),sources=contract,certificate_scope_self_test=self_test(),
        seal={k:sha(STUDY/"preregistration"/(k+"_intents.json")) for k in freeze},examples_are_new_own_runs=True,
        attempts_per_identity=1,disk=disk()))
    rows=[(tid,c) for c in freeze for tid in freeze[c]["order"]]
    lanes=[f"cpu{i:02}" for i in range(1,5)];todo=iter(rows);running={};first=None;done={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        def submit(lane):
            x=next(todo,None)
            if x is not None:
                tid,coh=x
                running[pool.submit(one,tid,coh,lane,contract)]=(tid,lane)
        for l in lanes:submit(l)
        while running:
            complete,_=wait(running,timeout=45,return_when=FIRST_COMPLETED)
            disk();free=[]
            for future in complete:
                tid,lane=running.pop(future);free.append(lane)
                try:done[tid]=future.result()
                except Exception as exc:
                    if first is None:
                        first=repr(exc);write("study/method_STOP.json",dict(utc=utc(),trial=tid,error=first,
                             remaining_not_dispatched=True,launched_workers_finish_safely=list(x[0] for x in running.values())))
            if complete and (len(done)%12==0 or first):print("method receipts",len(done),"of",len(rows),"shared_stop",first,flush=True)
            if first is None:
                for l in free:submit(l)
    write("study/run/method_ledger.json",dict(utc=utc(),assigned=len(rows),adjudicated=len(done),returned=done,first_shared_fault=first,disk=disk()))
    need(first is None and len(done)==len(rows),"partial method batch")
if __name__=="__main__":main()
