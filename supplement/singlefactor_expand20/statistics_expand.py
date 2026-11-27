"""Adjudicate the sealed expansion once, with paired base-scene cluster intervals."""
from paths import *
from collections import Counter
from cluster_statistics import estimate, SEED, REPS
from ranking_analysis import csv_bytes
from terminal_verification import reconcile

OUTCOMES=("success","empty_pool","empty_old_E","static_distance_uncertified","other_static_rejection","dynamic_hard_gate",
          "forbidden_collision","slip","closure_failure","insufficient_lift","hold_failure","timeout")
PAIRS=(("A","Raw"),("C0","Raw"),("C0","A"))

def classify(r,it):
    detail=r["detail"];steps=r["physical_returns"]
    if not steps:
        if detail in ("empty_pool","empty_old_E"):
            need(it["no_candidate"] and not r["selected"],"empty outcome with a selected candidate")
            return detail
        need(r["selected"] and not r["full_static_path"] and not r["grasp_success"] and r["physical_calls"]==0,"ambiguous static stop")
        need(detail in ("path_failure","static_distance_uncertified"),"unknown static detail "+repr(detail))
        return "static_distance_uncertified" if detail=="static_distance_uncertified" else "other_static_rejection"
    need(r["selected"] and r["full_static_path"] and r["dynamic_outcome_observed"] and not it["no_candidate"],"dynamic ownership")
    if r["grasp_success"]:
        need(detail=="success" and steps==4728 and r["engineering_gate_passed"] and r["trial_completed"],"incomplete successful grasp")
        return "success"
    if detail in ("motion_hard_gate","finite_opposing_pad_clearance_hard_stop"):return "dynamic_hard_gate"
    need(detail in OUTCOMES[6:],"unknown returned failure detail "+repr(detail))
    return detail

def sumrow(rows):
    n=len(rows)
    tally=dict.fromkeys(OUTCOMES,0);tally.update(Counter(r["failure_class"] for r in rows))
    def freq(col):
        return estimate(rows,lambda r:int(r[col]),lambda r:1)
    return dict(n=n,base_scenes=len({r["scene"] for r in rows}),origin_counts=dict(Counter(r["origin"] for r in rows)),
        selected=sum(r["selected"] for r in rows),static_admitted=sum(r["static_admitted"] for r in rows),
        entered_dynamics=sum(r["physics"] for r in rows),steps=sum(r["physical_steps"] for r in rows),
        success=sum(r["success"] for r in rows),grasp_success=freq("success"),static_admission_all=freq("static_admitted"),
        selection_coverage=freq("selected"),conditional_success=estimate(rows,lambda r:int(r["success"]),lambda r:int(r["physics"])),
        outcomes=tally,empty_E_including_empty_pool=tally["empty_old_E"]+tally["empty_pool"],
        exact_details=dict(Counter(r["detail"] for r in rows)))

def paircalc(units,c,first,second):
    data=[]
    for u,methods in units.items():
        a=methods[first];b=methods[second]
        if c!="all" and a["condition"]!=c:continue
        need(a["scene"]==b["scene"] and a["condition"]==b["condition"] and a["pool_sha256"]==b["pool_sha256"]
             and a["state_sha256"]==b["state_sha256"],"pair input differs")
        data.append(dict(unit=u,scene=a["scene"],condition=a["condition"],first_trial=a["trial_id"],second_trial=b["trial_id"],
                         first_success=a["success"],second_success=b["success"],same_lock=a["candidate"]==b["candidate"],
                         first_reason=a["failure_class"],second_reason=b["failure_class"],delta=int(a["success"])-int(b["success"])))
    n=len(data);diff=estimate(data,lambda r:r["delta"],lambda r:1)
    first_only=[r["unit"] for r in data if r["delta"]==1]
    second_only=[r["unit"] for r in data if r["delta"]==-1]
    need(diff["numerator"]==len(first_only)-len(second_only),"discordant cells do not agree")
    return dict(condition=c,first=first,second=second,n=n,scene_clusters=diff["clusters"],difference=diff,
        first_only_success_rate=estimate(data,lambda r:int(r["delta"]==1),lambda r:1),
        first_only_failure_rate=estimate(data,lambda r:int(r["delta"]==-1),lambda r:1),
        first_only_success=len(first_only),second_only_success=len(second_only),first_only_failure=len(second_only),
        second_only_failure=len(first_only),both_success=sum(r["first_success"] and r["second_success"] for r in data),
        both_failure=sum(not r["first_success"] and not r["second_success"] for r in data),
        different_candidates=sum(not r["same_lock"] for r in data),first_only_units=first_only,second_only_units=second_only,details=data)

def main():
    newbook=read(STUDY/"run/new168_ledger.json");seal=read(OUT/"preregistration/intents240.json")
    release=read(STUDY/"preregistration/physical_release.json")
    need(newbook["status"]=="complete" and not newbook["first_shared_fault"] and not newbook["never_dispatched"]
         and set(newbook["returned"])==set(seal["new"]) and len(seal["new"])==168,"new cohort incomplete")
    rows=[];links=[];windows=[];units={}
    reuse=read(OUT/"admission/carried72_reconciliation.json")
    for tid in [*seal["carried"],*seal["new"]]:
        bind=seal["by_trial"][tid];d=STUDY/"trials"/tid;j=STUDY/"run/jobs"/tid
        r=read(d/"result.json");it=read(d/"intent.json");claim=read(j/"dispatch.json");receipt=read(j/"return.json")
        it_hash=sha(d/"intent.json");ret_hash=sha(j/"return.json");res_hash=sha(d/"result.json")
        need(it_hash==bind["intent_sha256"]==r["intent_sha256"]==claim["intent"] and receipt["dispatch_sha256"]==sha(j/"dispatch.json")
             and receipt["result_sha256"]==res_hash and r["trial_id"]==it["trial_id"]==tid==receipt["tid"]==claim["tid"] and
             r["physical_calls"]==r["physical_returns"]==receipt["physical_calls"] and not r["shared_failure"],"receipt/intent mismatch "+tid)
        source=PRIOR/"study/trials"/tid if bind["origin"]=="carried" else d
        if bind["origin"]=="carried":
            need(reuse[tid]["original_result_sha256"]==res_hash and reuse[tid]["original_return_sha256"]==ret_hash,"retained method changed")
        else:
            need(newbook["returned"][tid]==ret_hash and release["intents"][tid]==it_hash,"missing dispatch binding")
        need(r["candidate"]==it["selected_candidate_id"]==bind["candidate"] and r["method"]==bind["method"] and r["unit"]==bind["unit_id"]
             and r["condition"]==bind["condition"] and r["seed"]==bind["worker_seed"] and r["diagnostic_certificate_exception"] is False,"condition/identity drift")
        detailed=classify(r,it)
        verified=reconcile(source,r)
        windows.append(verified)
        row=dict(trial_id=tid,origin=bind["origin"],hash_rank=bind["hash_rank"],unit=r["unit"],scene=r["scene"],object=r["object"],condition=r["condition"],
                 method=r["method"],candidate=r["candidate"],old_E_count=it["old_E_count"],pool_count=it["common_pool_count"],
                 selected=bool(r["selected"]),static_admitted=bool(r["full_static_path"]),physics=bool(r["physical_returns"]),physical_steps=r["physical_returns"],
                 completed=r["trial_completed"],engineering_gate_passed=r["engineering_gate_passed"],success=r["grasp_success"],
                 failure_class=detailed,detail=r["detail"],failure=r["failure"],forbidden_collision=bool(r["forbidden_collision"]),
                 slip=r["slipped"],max_lift_m=r["max_lift_m"],final_hold_streak=r["final_hold_streak"],seed=r["seed"],
                 state_sha256=it["stable_state_sha256"],pool_sha256=it["pool_sha256"],intent_sha256=it_hash,result_sha256=res_hash)
        group=units.setdefault(row["unit"],{});need(row["method"] not in group,"duplicate method-unit")
        group[row["method"]]=row;rows.append(row)
        links.append(dict(trial_id=tid,origin=bind["origin"],source_directory=str(source),summary_result=f"study/trials/{tid}/result.json",
                          intent_sha256=it_hash,result_sha256=res_hash,return_sha256=ret_hash,dispatch_sha256=receipt["dispatch_sha256"],
                          native_step_calls=r["physical_calls"],native_step_returns=r["physical_returns"],
                          terminal_window_sha256=verified.get("window_sha256"),telemetry_sha256=verified.get("source_telemetry_sha256")))
    need(len(units)==80 and all(set(x)==set(METHODS) for x in units.values()) and len(rows)==240,"panel incomplete")
    need(all(r["static_admitted"]==r["physics"] for r in rows),"native execution did not follow a passed static path")
    for c in FACTORS:
        for m in METHODS:
            rs=[r for r in rows if r["method"]==m and r["condition"]==c]
            need(len(rs)==20 and sum(r["origin"]=="carried" for r in rs)==6 and sorted(r["hash_rank"] for r in rs)==list(range(1,21)),"balanced complete panel")
    groups=[];pairs=[]
    for c in ("all",*FACTORS):
        for m in METHODS:
            rs=[r for r in rows if r["method"]==m and (c=="all" or r["condition"]==c)]
            groups.append(dict(condition=c,method=m,**sumrow(rs)))
        for a,b in PAIRS:pairs.append(paircalc(units,c,a,b))
    book=dict(version=OUT.name,closed_utc=utc(),trial_count=len(rows),previous_returns=72,new_returns=168,
        count_unit=len(units),unique_base_scenes=len({r["scene"] for r in rows}),new_model_calls=0,additional_runs_per_retained_identity=0,
        native_returns_total=sum(r["physical_steps"] for r in rows),native_returns_new=sum(r["physical_steps"] for r in rows if r["origin"]=="new"),
        physics_total=sum(r["physics"] for r in rows),physics_new=sum(r["physics"] for r in rows if r["origin"]=="new"),
        shared_faults=[],retries=0,new_dispatch_ledger_sha256=sha(STUDY/"run/new168_ledger.json"),prespecified_intent_seal_sha256=sha(OUT/"preregistration/intents240.json"),
        receipts=links)
    write("ledger240.json",book)
    write("statistics/execution240.json",dict(groups=groups,records=rows))
    write("statistics/paired_effects.json",dict(cluster_unit="base_scene",bootstrap_replicates=REPS,seed=SEED,
         percentile=[2.5,97.5],contrast_unit="success probability difference, first minus second",groups=pairs))
    write("admission/terminal_window_verification.json",dict(methods=len(windows),with_physics=sum(w["physical_outcome_observed"] for w in windows),
         full_windows=sum(w.get("replayed_all_480",False) for w in windows),rows=windows))
    flat=[{k:(json.dumps(v,ensure_ascii=False,separators=(",",":")) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in rows]
    exclusive_bytes(OUT/"statistics/trial_outcomes.csv",csv_bytes(flat))
    print("240 linked method results; each entered physical execution verified against its original terminal window",flush=True)
if __name__=="__main__":main()
