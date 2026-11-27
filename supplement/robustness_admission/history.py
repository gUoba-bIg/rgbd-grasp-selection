"""Derive V1 observational statistics from original intents, per-trial results and selection receipts."""
import csv
from itertools import combinations
from collections import Counter,defaultdict
from functools import lru_cache
import io
from common_v1 import *
from cluster_statistics import estimate
from rank_stability import load_unit,compare,self_test
import numpy as np

def table(name,rows):
    path=OUT/name;need(path.resolve().is_relative_to(OUT),"output")
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return sha(path)

@lru_cache(maxsize=380)
def unit(folder,uid):
    d=ROOT/folder/"units"/uid;sel=read(d/"selection.json")
    o=read(d/("orders.json" if folder==R6NAME else "scoring_components.json"))
    return sel,o

def extract(folder,ledger):
    rows=read(ROOT/folder/"analysis"/ledger)["rows"];result=[]
    expected={"R6":720,"R7":240}
    stage="R6" if folder==R6NAME else "R7"
    need(len({r["trial_id"] for r in rows})==len(rows)==expected[stage],"unique historical trials")
    for r in rows:
        m=r["method"];uid=r["unit_id"];tid=r["trial_id"];d=ROOT/folder/"trials"/tid
        s,o=unit(folder,uid);i=s["selected"][m]
        it=read(d/"intent.json");res=read(d/"method_result.json");pick=it["selected_p3"]
        need(sha(d/"intent.json")==r["intent_sha256"] and sha(d/"method_result.json")==r["result_sha256"],"frozen outcome "+tid)
        need((it["selected_pool_index"],it["method"],it["unit_id"])==(i,m,uid),"selection bind "+tid)
        # R6FORMAL-001 has an immutable stop in its initial batch and a separate, frozen
        # adjudication in the resumed batch. Ledger truth must match the latter receipt.
        need(res["physical_returns"]==r["physical_returns"] and res["grasp_success"]==r["grasp_success"],"adjudication "+tid)
        rk=ROOT/folder/"units"/uid/("ranking_"+m+".json")
        v=read(rk);sc=v["scores_by_pool_index"];eligible=[j for j in v["complete_order"] if s["eligibility"][j]["eligible"]]
        need((eligible[0] if eligible else None)==i and all(np.isfinite(sc)),"ranking with old E "+tid)
        selected_score=gap=proxy=contact=None
        if i is not None:
            selected_score=sc[i]
            if len(eligible)>1:gap=selected_score-sc[eligible[1]]
            if m!="Raw":
                detail=o["candidate_details"][i]
                component=detail["score_components"][m] if stage=="R6" else detail["arms"][m]
                proxy=component["collision_proxy"];contact=component["contact_support"]
                need(abs(float(component.get("rank_score",component.get("score")))-selected_score)<1e-12,"logged score component")
        registered_admission=d/"safety_receipt.json"
        admitted=bool((read(registered_admission) or {}).get("passed")) if registered_admission.exists() else False
        need(admitted==(r["physical_returns"]>0),"static safety receipt/entry accounting "+tid)
        if i is not None:need(pick["candidate_id"]==r["selected_candidate_id"],"candidate receipt "+tid)
        scene=it["scene_key"];need(uid==scene+"-"+it["condition"],"scene grouping")
        result.append(dict(stage=stage,trial=tid,unit=uid,scene=scene,object=it["object_id"],condition=it["condition"],method=m,
            candidate=r["selected_candidate_id"],pool_index=i,pose_sha256=pick["camera_row_sha256"] if pick else None,
            pool_n=it["common_pool_count"],E_n=len(eligible),selected=int(i is not None),
            observed_score=selected_score,margin_in_E=gap,locked_pool_rank=(v["complete_order"].index(i)+1 if i is not None else None),
            risk_proxy=proxy,contact_support=contact,detail=r["detail_class"],admitted=int(admitted),
            dynamics=int(r["physical_returns"]>0),native_returns=r["physical_returns"],success=int(r["grasp_success"]),
            forbidden=int(r["forbidden_collision"]),result_sha256=r["result_sha256"],intent_sha256=r["intent_sha256"]))
    return result

def rates(rs):
    return {name:estimate(rs,lambda r,n=n:r[n],lambda r,d=d:1 if d is None else r[d])
            for name,n,d in (("all_attempt_success","success",None),("selected","selected",None),("static_admission","admitted",None),
             ("physical_entry","dynamics",None),("conditional_success","success","dynamics"),("forbidden_collision","forbidden",None))}

def curves(rows):
    points=[]
    def ratio(rs,attr,test,operator):
        conditional=[r for r in rs if r[attr] is not None and operator(r[attr],test)]
        threshold=lambda r:r[attr] is not None and operator(r[attr],test)
        coverage=estimate(rs,lambda r:int(threshold(r)),lambda r:1)
        risk=estimate(rs,lambda r:int(threshold(r))*(1-r["success"]),lambda r:int(threshold(r)))
        return dict(threshold=test,n_selected=len(conditional),assigned=len(rs),coverage=coverage,empirical_failure_risk=risk)
    import operator
    for m in sorted({r["method"] for r in rows}):
        rs=[r for r in rows if r["method"]==m]
        for attr,op,descending in (("observed_score",operator.ge,True),("risk_proxy",operator.le,False)):
            keys=sorted({r[attr] for r in rs if r[attr] is not None},reverse=descending)
            if not keys:continue
            points.append(dict(method=m,criterion=attr,threshold=None,n_selected=0,assigned=len(rs),coverage={"value":0.0},empirical_failure_risk={"value":None,"ci95":None}))
            for val in keys:points.append(dict(method=m,criterion=attr,**ratio(rs,attr,val,op)))
    return points

def associations(rows):
    out=[]
    for m in sorted({r["method"] for r in rows}):
        rs=[r for r in rows if r["method"]==m]
        for field in ("observed_score","margin_in_E","locked_pool_rank","risk_proxy"):
            active=[r for r in rs if r[field] is not None]
            if not active:continue
            cuts=sorted(set(np.quantile([r[field] for r in active],[.25,.50,.75]).tolist()))
            groups=defaultdict(list)
            for r in rs:groups["undefined" if r[field] is None else str(np.searchsorted(cuts,r[field],side="left"))].append(r)
            for group,g in groups.items():
                out.append(dict(method=m,field=field,closed_upper_cutpoints=cuts,bin=group,assigned=len(g),
                    static_admission=estimate(g,lambda r:r["admitted"],lambda r:1),
                    all_attempt_success=estimate(g,lambda r:r["success"],lambda r:1)))
    return out

def execute():
    need((OUT/"admission/v1_identity.json").is_file(),"unverified original files")
    write("analysis/stability_self_test.json",self_test())
    rr=extract(R6NAME,"formal_ledger.json");aa=extract(R7NAME,"ablation_ledger.json")
    overlap=set(x["unit"] for x in aa)
    need(len(overlap)==80,"R7 80-unit subset")
    allrows=rr+aa;table("analysis/historical_trial_paths.csv",allrows)
    result={};rates_table=[];pairs=[];match_history=[];matches={}
    for population,rows in (("R6_240_each",rr),("R7_80_each",aa),("common_80_each",[r for r in allrows if r["unit"] in overlap])):
        units=defaultdict(dict)
        for r in rows:units[r["unit"]][r["method"]]=r
        result[population]=dict(assigned=len(rows),scenes=len(set(r["scene"] for r in rows)),units=len(units))
        methods=sorted({r["method"] for r in rows})
        for m in methods:
            subsets=[("all",[r for r in rows if r["method"]==m])]
            for field in ("condition","object"):
                subsets+=[(str(k),[r for r in rows if r["method"]==m and r[field]==k]) for k in sorted({r[field] for r in rows})]
            for label,rs in subsets:
                rates_table.append(dict(population=population,method=m,stratum=label,n=len(rs),paths=dict(Counter(r["detail"] for r in rs)),**rates(rs)))
        for m,n in combinations(methods,2):
            paired=[]
            for uid,u in units.items():
                a,b=u[m],u[n]
                paired.append(dict(scene=a["scene"],different=int(a["candidate"]!=b["candidate"]),both_absent=int(not a["selected"] and not b["selected"]),
                    delta=b["success"]-a["success"],b_only=int(b["success"]>a["success"]),a_only=int(a["success"]>b["success"])))
            pairs.append(dict(population=population,a=m,b=n,n=len(paired),abstention_pair_count=sum(r["both_absent"] for r in paired),
                candidate_diversion=estimate(paired,lambda r:r["different"],lambda r:1),b_minus_a=estimate(paired,lambda r:r["delta"],lambda r:1),
                b_only=sum(r["b_only"] for r in paired),a_only=sum(r["a_only"] for r in paired)))
    for scene in sorted({r["scene"] for r in rr}):
        directory=ROOT/R6NAME/"units";a=load_unit(directory/(scene+"-clean"));b=load_unit(directory/(scene+"-joint"))
        r,w=compare(a,b,scene,"joint");match_history.extend(r);matches[scene]=w
    table("analysis/historical_clean_joint_pairs.csv",match_history)
    write("analysis/historical_pose_matches.json",matches);write("analysis/historical_stability_summary.json",stability_summary(match_history))
    write("analysis/method_rates.json",rates_table);write("analysis/paired_methods.json",pairs)
    write("analysis/risk_coverage.json",curves(allrows));write("analysis/score_margin_and_outcomes.json",associations(allrows))
    write("analysis/scope.json",result)

def stability_summary(rows):
    summary=[]
    for m in METHODS:
        for c in sorted({r["condition"] for r in rows}):
            group=[r for r in rows if r["method"]==m and r["condition"]==c]
            out=dict(method=m,condition=c,scenes=len(group))
            for field in ("top1","locked"):
                out[field+"_counts"]=dict(Counter(r[field] for r in group))
                out[field+"_change"]=estimate(group,lambda r:int(r[field] in ("pose_changed","presence_changed")),lambda r:1)
                out[field+"_change_both_present"]=estimate(group,lambda r:int(r[field]=="pose_changed"),lambda r:int(r[field] in ("pose_changed","pose_matched")))
            for field in ("score_tau_b","stable_order_tau","top1_exact_pose_and_dimensions",*["top%d_jaccard"%k for k in (1,3,5,10,20)],*["top%d_overlap_coefficient"%k for k in (1,3,5,10,20)]):
                out[field]=estimate(group,lambda r:float(r[field]) if r[field] is not None else None,lambda r:1)
            summary.append(out)
    return summary

if __name__=="__main__":execute()
