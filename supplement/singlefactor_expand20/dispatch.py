"""Single-use continuation. Process completions drive dispatch; cohort checkpoints occur at 40."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from paths import *
from method_trial_expand import validate
from study_runner import environment
import shutil
import subprocess
import traceback

CODE=("paths.py","draw.py","sample.py","baseline.py","prepare.py","reported_gap.py","build_worker.py","method_trial_expand.py","dispatch.py")
def dependencies():
    sources={name:sha(OUT/"src"/name) for name in CODE}
    for name in ("common_v1.py","method_seal.py","runtime_stages.py","study_runner.py","diagnostic_return_fields.py","rank_stability.py"):
        sources["prior_wrapper/"+name]=sha(PREVIOUS_CODE/name)
    return sources

def receive(tid,lane,claim,prior_hash):
    base=STUDY/"trials"/tid; new=STUDY/"run/jobs"/tid
    it=read(base/"intent.json");r=read(base/"result.json")
    need(prior_hash==sha(base/"intent.json")==r["intent_sha256"] and r["trial_id"]==tid and r["seed"]==it["worker_seed"]
         and r["method"]==it["method"] and r["candidate"]==it["selected_candidate_id"] and r["unit"]==it["unit_id"]
         and r["physical_calls"]==r["physical_returns"] and r["physical_returns"] in range(4729)
         and type(r["grasp_success"]) is bool and r["shared_failure"] is False and r["diagnostic_certificate_exception"] is False,
         "unbound method execution "+tid)
    if r["physical_calls"]:
        native=read(base/"dynamic_integrity.json")
        need(native["finite"] and native["body_link_ownership"] and native["exceptions"]==0 and native["native_calls"]==native["native_returns"]
             and native["actual_physics_steps"]==r["physical_calls"],"incomplete dynamic evidence "+tid)
    return write(f"study/run/jobs/{tid}/return.json",dict(utc=utc(),tid=tid,lane=lane,dispatch_sha256=claim,
         result_sha256=sha(base/"result.json"),detail=r["detail"],physical_calls=r["physical_calls"]))

def one(tid,lane,release):
    need(dependencies()==release["code"],"runner changed")
    path,it,_,_=validate(tid,"robustness")
    expected=release["intents"][tid]
    need(sha(path)==expected and read(OUT/"preregistration/intents240.json")["by_trial"][tid]["origin"]=="new","frozen reservation")
    folder=STUDY/"run/jobs"/tid;folder.mkdir(exist_ok=False)
    claim=write(folder/"dispatch.json",dict(utc=utc(),tid=tid,lane=lane,intent=expected,order=it["execution_order"],permit_once=True))
    cwd=STUDY/"workspaces"/lane
    env=environment(cwd,"physics",it["worker_seed"],folder)
    shellbox=Path(os.environ["SYSTEMROOT"])/"System32/WindowsPowerShell/v1.0"
    env["PATH"]+=os.pathsep+str(shellbox)
    with (folder/"stdout.log").open("xb") as out,(folder/"stderr.log").open("xb") as err:
        command=[str(C/"env/physics/Scripts/python.exe"),"-B","-s","-u","-X","utf8",str(OUT/"src/method_trial_expand.py"),tid,"robustness"]
        child=subprocess.run(command,cwd=cwd,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL)
    if child.returncode:
        text=(folder/"stderr.log").read_text(encoding="utf8",errors="replace")
        raise RuntimeError(f"worker {tid} exited {child.returncode}: {text[-2300:]}")
    return receive(tid,lane,claim,expected)

def checkpoint(k,done,release):
    health=disk()
    need(dependencies()==release["code"] and all(sha(OUT/r)==h for r,h in release["protected_inputs"].items()),"execution seal changed")
    write(f"admission/checkpoints/after_{k:03}.json",dict(utc=utc(),new_completed=k,checksum_verified=True,
          shared_faults=[],disk=health,verified_returns=done))
    print("Checkpoint",k,"new; free GiB",round(health["free_bytes"]/1024**3,2),flush=True)
    return health["above_floor"]

def main():
    contract=read(OUT/"preregistration/intents240.json")
    todo=contract["new"];need(len(todo)==len(set(todo))==168 and len(contract["carried"])==72,"invalid budget")
    # Preflight every new intent and its science input hashes before any native connection.
    validated={t:sha(validate(t,"robustness")[0]) for t in todo}
    health=disk();need(health["above_floor"],"free space below 80 GiB before dispatch")
    release=dict(utc=utc(),code=dependencies(),intents=validated,initial_disk=health,workers=4,
        attempts_per_new_intent=1,protected_inputs={p:sha(OUT/p) for p in (
        "admission/v1_identity.json","preregistration/sample80.json","preregistration/intents240.json",
        "preregistration/observation_fingerprint_index.json","admission/carried72_reconciliation.json","preregistration/unit_input_hashes.json",
        "study/preregistration/design.json","study/preregistration/design_seal.json","study/preregistration/robustness_intents.json")},
        new_P3_calls=0,new_P5_calls=0,new_static_and_dynamic_use_frozen_V1=True)
    write("study/preregistration/physical_release.json",release)
    done={};fault=None;paused=None;lanes=[f"cpu{k:02}" for k in range(1,5)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0,168,40):
            waiting=iter(todo[offset:offset+40]);running={}
            def submit(lane):
                nonlocal paused
                if paused or fault:return
                free=shutil.disk_usage("E:/").free
                if free<FLOOR:
                    paused=dict(utc=utc(),reason="disk_below_80_GiB",observed_bytes=free);return
                tid=next(waiting,None)
                if tid is not None:running[pool.submit(one,tid,lane,release)]=(tid,lane)
            for lane in lanes:submit(lane)
            while running:
                ready,_=wait(running,return_when=FIRST_COMPLETED)
                idle=[]
                for f in ready:
                    tid,lane=running.pop(f);idle.append(lane)
                    try:done[tid]=f.result()
                    except Exception as ex:
                        if fault is None:
                            fault=dict(utc=utc(),tid=tid,reason=repr(ex),traceback=traceback.format_exc(),
                                       safe_drain_in_flight=[t for t,l in running.values()])
                            write("study/run/first_shared_fault.json",fault)
                if fault is None and not paused:
                    for lane in idle:submit(lane)
            if paused or fault is not None:break
            try:
                if not checkpoint(len(done),done,release):
                    paused=dict(utc=utc(),reason="checkpoint_disk_below_80_GiB")
            except Exception as ex:
                fault=dict(utc=utc(),tid=None,reason=repr(ex),traceback=traceback.format_exc(),safe_drain_in_flight=[])
                write("study/run/first_shared_fault.json",fault)
            if paused or fault is not None:break
    status="shared_fault" if fault else "paused" if paused else "complete"
    write("study/run/new168_ledger.json",dict(utc=utc(),status=status,assigned=168,returned=done,count=len(done),first_shared_fault=fault,
        disk_pause=paused,never_dispatched=[t for t in todo if not (STUDY/"run/jobs"/t/"dispatch.json").exists()],resubmissions=0))
    if status!="complete":raise RuntimeError(str(fault or paused))
    need(len(done)==168,"missing adjudication")
    print("All 168 new methods returned",flush=True)
if __name__=="__main__":main()
