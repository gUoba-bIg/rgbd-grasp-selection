"""Continue only never-started V1 diagnostic intents after the read-only return audit.

No intent, launch, native stream or prior result is replaced. Physics remains in the original
sealed method_trial and its evaluator; only interpretation of the returned hard-stop fields is added.
"""
from common_v1 import *
from after_network import call
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from run_sealed_methods import scientific_inputs
from study_runner import STUDY,environment
from method_trial import validate
import sys,time

CODE=WORK/"supplement/diagnostic_return_fields.py"
PY=C/"env/physics/Scripts/python.exe"
CONT=STUDY/"method_reporting_resume01"
PREFIX="study/method_reporting_resume01/"

def returned(tid):
    job=STUDY/"run/jobs"/tid;d=STUDY/"trials"/tid
    it=read(d/"intent.json");r=read(d/"result.json");claim=read(job/"dispatch.json")
    need(r["trial_id"]==tid==it["trial_id"]==claim["tid"] and it["selected_candidate_id"]==r["candidate"] and
         r["intent_sha256"]==claim["intent"]==sha(d/"intent.json") and r["physical_calls"]==r["physical_returns"] and
         type(r["grasp_success"]) is bool and not r["shared_failure"],"new or retained result ownership")
    return dict(utc=utc(),tid=tid,dispatch_sha256=sha(job/"dispatch.json"),result_sha256=sha(d/"result.json"),
                detail=r["detail"],physical_calls=r["physical_calls"])

def one(tid,lane,contract):
    disk();verify_contract(contract)
    need(not (CONT/"STOP.json").exists(),"shared stop")
    src,it,s,row=validate(tid,"diagnostic")
    job=STUDY/"run/jobs"/tid;need(not job.exists() and not (src.parent/"start.json").exists() and not (src.parent/"result.json").exists(),"already dispatched")
    job.mkdir()
    prefix="study/run/jobs/"+tid+"/"
    write(prefix+"dispatch.json",dict(utc=utc(),tid=tid,intent=sha(src),order=it["execution_order"],lane=lane,permit_once=True,resumed_from=PREFIX+"release.json"))
    cwd=STUDY/"workspaces"/lane;e=environment(cwd,"physics",it["worker_seed"],job)
    e["PATH"]+=os.pathsep+str(Path(os.environ["SystemRoot"])/"System32/WindowsPowerShell/v1.0")
    with (job/"stdout.log").open("xb") as so,(job/"stderr.log").open("xb") as se:
        p=subprocess.run([str(PY),"-B","-s",str(CODE),"once",tid],cwd=cwd,env=e,stdout=so,stderr=se)
    if p.returncode:
        raise RuntimeError(tid+" returned an unresolved failure: "+(job/"stderr.log").read_text(encoding="utf8",errors="replace")[-1600:])
    return write(prefix+"return.json",returned(tid))

def verify_contract(c):
    need(scientific_inputs()==c["original_source_hashes"] and c["return_interpreter"]==sha(CODE) and c["continuation"]==sha(Path(__file__)) and
         c["diagnostic_intents"]==sha(STUDY/"preregistration/diagnostic_intents.json") and
         c["robust_intents"]==sha(STUDY/"preregistration/robustness_intents.json"),"source or intent drift")

def main():
    disk()
    need(not CONT.exists() and not (STUDY/"STOP.json").exists(),"another attempt or observation integrity stop")
    oldfile=STUDY/"run/method_ledger.json";old=read(oldfile);paused=read(STUDY/"method_STOP.json")
    plans={k:read(STUDY/"preregistration"/(k+"_intents.json")) for k in ("robustness","diagnostic")}
    all_ids=plans["robustness"]["order"]+plans["diagnostic"]["order"]
    jobs={d.name for d in (STUDY/"run/jobs").iterdir() if d.is_dir()}
    done=dict(old["returned"])
    need(len(all_ids)==len(set(all_ids))==old["assigned"]==132 and len(done)==old["adjudicated"]==74 and
         paused["trial"]=="ADM-001" and jobs-set(done)=={"ADM-001"} and set(plans["robustness"]["order"])<=set(done),"unexpected prior launch state")
    for tid,h in done.items():
        f=STUDY/"run/jobs"/tid/"return.json";r=read(f);obs=returned(tid)
        need(h==sha(f) and all(obs[k]==r[k] for k in ("tid","dispatch_sha256","result_sha256","detail","physical_calls")),"retained receipt drift "+tid)
    proof=read(OUT/"admission/diagnostic_native_label_verification.json")
    need(proof["scientific_category_return"] is None and proof["episode_outcome_class"]=="engineering_failure" and
         proof["step"]==3041 and proof["native"]["actual_physics_steps"]==3041 and proof["native"]["finite"] and proof["native"]["body_link_ownership"] and
         proof["native"]["exceptions"]==0 and len(proof["tests"]["rejected_negative_fixtures"])==7 and proof["new_physics_calls"]==0,"original returned hard stop not verified")
    contract=dict(original_source_hashes=read(STUDY/"preregistration/physical_release.json")["sources"],return_interpreter=sha(CODE),continuation=sha(Path(__file__)),
                  diagnostic_intents=sha(STUDY/"preregistration/diagnostic_intents.json"),robust_intents=sha(STUDY/"preregistration/robustness_intents.json"))
    verify_contract(contract)
    todo=[t for t in all_ids if t not in jobs]
    need(len(todo)==57 and all(t.startswith("ADM-") for t in todo),"not exclusively untouched intents")
    release=write(PREFIX+"release.json",dict(utc=utc(),disk=disk(),code=contract,prior_partial_ledger_sha256=sha(oldfile),
                  prior_stop_sha256=sha(STUDY/"method_STOP.json"),offline_replay_sha256=sha(OUT/"admission/diagnostic_native_label_verification.json"),
                  retained_delivered=list(done),already_run_only_decode="ADM-001",never_started=todo,retries=0))
    call("decode_existing_ADM001",PY,"diagnostic_return_fields.py","existing","ADM-001")
    done["ADM-001"]=write("study/run/jobs/ADM-001/return.json",dict(**returned("ADM-001"),decoded_without_new_execution=True,continuation_release_sha256=release))
    lanes=["cpu01","cpu02","cpu03","cpu04"];queue=iter(todo);inflight={};err=None
    with ThreadPoolExecutor(max_workers=4) as pool:
        def dispatch(lane):
            tid=next(queue,None)
            if tid:inflight[pool.submit(one,tid,lane,contract)]=(lane,tid)
        for lane in lanes:dispatch(lane)
        while inflight:
            ready,_=wait(inflight,timeout=45,return_when=FIRST_COMPLETED)
            disk();available=[]
            for fut in ready:
                lane,tid=inflight.pop(fut);available.append(lane)
                try:done[tid]=fut.result()
                except Exception as exc:
                    if err is None:
                        err=repr(exc);write(PREFIX+"STOP.json",dict(utc=utc(),tid=tid,error=err,remaining_not_dispatched=True,
                                            already_dispatched=[item[1] for item in inflight.values()]))
            if ready and (len(done)%12==0 or err):print("adjudicated",len(done),"/",len(all_ids),err,flush=True)
            if err is None:
                for lane in available:dispatch(lane)
    need(sha(oldfile)==read(CONT/"release.json")["prior_partial_ledger_sha256"],"historical pause ledger changed")
    final_name="study/run/method_ledger_continued.json" if err is None else PREFIX+"partial_ledger.json"
    h=write(final_name,dict(utc=utc(),assigned=len(all_ids),adjudicated=len(done),returned=done,first_shared_fault=err,
            prior_pause_sha256=sha(STUDY/"method_STOP.json"),release_sha256=release,method_fingerprint=contract,disk=disk()))
    need(err is None and set(done)==set(all_ids),"new shared stop, preserve batch")
    ranking=C/"env/rank/Scripts/python.exe"
    call("continued_statistics",ranking,"final_statistics.py")
    write("admission/completed_stages.json",dict(utc=utc(),disk=disk(),three_cold_verified=True,original_redesign=False,
          inference_pairs=720,three_method_rankings=2160,method_ledger_sha256=h,ledger_path=final_name,
          supplementary_native_stop_reconciled_from_existing_evidence="ADM-001"))
    call("continued_report",ranking,"reporting.py")
    call("continued_review_package",sys.executable,"deliver.py")
if __name__=="__main__":main()
