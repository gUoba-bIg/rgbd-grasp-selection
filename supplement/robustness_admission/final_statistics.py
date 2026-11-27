"""Keep original cohorts, selected factor trials, controls and censored diagnostic outcomes separate."""
from collections import Counter,defaultdict
from itertools import combinations
from common_v1 import *
from cluster_statistics import estimate
from history import table
from evidence_ledger import method_ledger
STUDY=OUT/"study"
def div(n,d):return n/d if d else None
def ratio(rows,n,d=lambda r:1):return estimate(rows,n,d)
def counted(rs):
    return dict(n=len(rs),selected=sum(r["selected"] for r in rs),dynamics=sum(r["physical_returns"]>0 for r in rs),
        steps=sum(r["physical_returns"] for r in rs),success=sum(r["grasp_success"] for r in rs),
        outcome=dict(Counter(r["detail"] for r in rs)),all_attempt_success=ratio(rs,lambda r:r["grasp_success"]),
        static_path_completed=ratio(rs,lambda r:r["full_static_path"]),conditional_success=ratio(rs,lambda r:r["grasp_success"],lambda r:r["physical_returns"]>0),
        forbidden_collision=ratio(rs,lambda r:r["forbidden_collision"]))
def matrix(rows,weight=False):
    w=lambda r:r["design_weight"] if weight else 1
    def weightif(pred):return lambda r:w(r)*int(pred(r))
    truth=lambda r:r["dynamic_outcome_observed"]
    tp=weightif(lambda r:truth(r) and r["V1_entry_certified"] and r["grasp_success"])
    tn=weightif(lambda r:truth(r) and not r["V1_entry_certified"] and not r["grasp_success"])
    fp=weightif(lambda r:truth(r) and r["V1_entry_certified"] and not r["grasp_success"])
    fn=weightif(lambda r:truth(r) and not r["V1_entry_certified"] and r["grasp_success"])
    metrics={}
    for label,a,b in (("false_rejection_FN_over_TP_plus_FN",fn,tp),("sensitivity",tp,fn),("specificity",tn,fp),
                       ("positive_predictive_value",tp,fp),("negative_predictive_value",tn,fn)):
        metrics[label]=ratio(rows,a,lambda r:a(r)+b(r))
    return dict(cells={k:sum(f(r) for r in rows) for k,f in (("TP",tp),("FN",fn),("FP",fp),("TN",tn))},
        no_new_physical_outcome={str(cert):sum(w(r) for r in rows if r["V1_entry_certified"]==cert and not truth(r)) for cert in (True,False)},
        total_assigned=sum(w(r) for r in rows),metrics=metrics)
def run():
    book=read(method_ledger());need(book["first_shared_fault"] is None and book["adjudicated"]==book["assigned"],"incomplete supplemental execution")
    cases={r["supplement_trial_id"]:r for r in read(OUT/"preregistration/diagnostic_sample.json")["cases"]}
    rs={"diagnostic":[],"robustness":[]}
    for t in book["returned"]:
        base=STUDY/"trials"/t;r=read(base/"result.json")
        returned=read(STUDY/"run/jobs"/t/"return.json")
        need(sha(base/"result.json")==returned["result_sha256"] and sha(base/"intent.json")==r["intent_sha256"] and r["physical_calls"]==r["physical_returns"],"incomplete returned trial")
        if r["cohort"]=="diagnostic":
            c=cases[t];parent=ROOT/R6NAME/"trials"/c["trial"]
            original=read(parent/"static_preflight/left_right_contact_entry_roots.json")
            r.update(control_group=c["group"],V1_entry_certified=original["both_passed"],V1_static_path_passed=read(parent/"static_preflight/path_result.json")["passed"],
                V1_roots_sha256=sha(parent/"static_preflight/left_right_contact_entry_roots.json"),V1_strict_detail=c["source_detail"],
                design_weight=1/c["inclusion_fraction"])
            labels={v.split(":")[-1] for v in (r.get("failure") or {}).get("violations",[])} if isinstance(r.get("failure"),dict) else set()
            definite={"nonpad_target_collision","robot_table_collision","pad_target_contact_before_close",
                      "unregistered_self_collision","finger_independent_robot_target_collision","simultaneous_nonpad_target_collision_at_entry"}
            if r["dynamic_outcome_observed"]:
                if r["grasp_success"]:r["diagnostic_interpretation"]="successful_monitored_grasp"
                elif r["forbidden_collision"]:r["diagnostic_interpretation"]="observed_prohibited_collision"
                elif r["detail"]=="finite_opposing_pad_clearance_hard_stop":r["diagnostic_interpretation"]="finite_opposing_pad_clearance_hard_stop"
                else:r["diagnostic_interpretation"]="observed_unsuccessful_execution"
            elif labels&definite:r["diagnostic_interpretation"]="proven_independent_path_or_clearance_violation"
            elif "proxy_pad_target_penetration" in labels:r["diagnostic_interpretation"]="static_proxy_penetration_physical_outcome_unverified"
            else:r["diagnostic_interpretation"]="physical_outcome_unverified"
            if not r["V1_entry_certified"] and r["grasp_success"]:
                r["admission_interpretation"]="possible_false_rejection_observed_success"
            elif r["forbidden_collision"] or r["detail"] in ("finite_opposing_pad_clearance_hard_stop","motion_hard_gate") or labels&definite:
                r["admission_interpretation"]="documented_violation_on_specific_path"
            elif r["dynamic_outcome_observed"]:
                r["admission_interpretation"]="observed_grasp_success" if r["grasp_success"] else "observed_failed_attempt_without_general_impossibility_proof"
            else:
                r["admission_interpretation"]="insufficient_physical_evidence"
        rs[r["cohort"]].append(r)
    for k,rows in rs.items():
        table(k+"/trial_outcomes.csv",[{a:(json.dumps(b,ensure_ascii=False,allow_nan=False,separators=(",",":")) if isinstance(b,(list,dict)) else b) for a,b in r.items()} for r in rows])
    robust=rs["robustness"];diags=rs["diagnostic"];need(len(diags)==60,"diagnostic groups")
    groups=[]
    for cond in ("all",*MODES[1:5]):
        for m in METHODS:
            arr=[r for r in robust if r["method"]==m and (cond=="all" or r["condition"]==cond)]
            groups.append(dict(method=m,condition=cond,**counted(arr)))
    units=defaultdict(dict)
    for r in robust:units[r["unit"]][r["method"]]=r
    pairs=[]
    for a,b in combinations(METHODS,2):
        items=[]
        for uid,m in units.items():
            need(set(m)==set(METHODS),"incomplete sampled triplet")
            one,two=m[a],m[b];items.append(dict(scene=one["scene"],delta=int(two["grasp_success"])-int(one["grasp_success"]),different=one["candidate"]!=two["candidate"]))
        pairs.append(dict(method_a=a,method_b=b,n=len(items),b_minus_a=ratio(items,lambda r:r["delta"]),
                b_only=sum(r["delta"]==1 for r in items),a_only=sum(r["delta"]==-1 for r in items),selected_diversion=ratio(items,lambda r:r["different"])))
    write("robustness/sampled_execution.json",dict(groups=groups,paired=pairs,assigned=len(robust),physical_trials=sum(r["physical_returns"]>0 for r in robust),all_attempt_denominator_by_method=len(units)))
    allcounts=[]
    for name in ("uncertified","success_control","collision_control"):
        g=[r for r in diags if r["control_group"]==name]
        allcounts.append(dict(group=name,interpretation_counts=dict(Counter(r["diagnostic_interpretation"] for r in g)),
                              attribution_counts=dict(Counter(r["admission_interpretation"] for r in g)),**counted(g)))
    u=[r for r in diags if r["control_group"]=="uncertified"]
    r0=ratio(u,lambda r:r["grasp_success"]);r1=ratio(u,lambda r:r["grasp_success"] or not r["dynamic_outcome_observed"])
    write("diagnostic/validity.json",dict(groups=allcounts,verified_sample_matrix=matrix(diags),weighted_sampled_frames_matrix=matrix(diags,True),
        observed_rescue_per_assigned_uncertified=r0,observed_rescue_per_executed=ratio(u,lambda r:r["grasp_success"],lambda r:r["dynamic_outcome_observed"]),
        worst_case_all_unexecuted_could_succeed_upper_bound=r1,observed_fraction_211_frame=ratio(u,lambda r:r["design_weight"]*r["grasp_success"],lambda r:r["design_weight"]),
        caveat="Matrices compare presence of both original entry certificates by stop against fresh observed grasp outcomes. Static-only outcomes remain unverified. Positive and negative controls are outcome-enriched. No population specificity/equivalence claim. Bounds address unverified physical outcomes among sampled candidates only."))
if __name__=="__main__":run()
