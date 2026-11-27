"""Cross-check the paper's paired denominators, intervals and every receipt using raw result files."""
from paths import *
from collections import defaultdict
import numpy as np

def check():
    design=read(OUT/"preregistration/intents240.json");book=read(OUT/"ledger240.json");newreturns=read(STUDY/"run/new168_ledger.json")
    reported=read(OUT/"statistics/execution240.json")["groups"];contrasts=read(OUT/"statistics/paired_effects.json")["groups"]
    panel=defaultdict(dict)
    for link in book["receipts"]:
        tid=link["trial_id"];binding=design["by_trial"][tid];source=Path(link["source_directory"])
        need(sha(source/"result.json")==link["result_sha256"] and sha(STUDY/"trials"/tid/"intent.json")==link["intent_sha256"],"independent result or intent hash")
        r=read(source/"result.json")
        need(r["method"]==binding["method"] and r["candidate"]==binding["candidate"] and r["scene"]==binding["base_scene"],"individual trial identity")
        need((tid in newreturns["returned"])==(binding["origin"]=="new"),"carried trial dispatched")
        panel[(r["scene"],r["condition"])][r["method"]]=r
    need(len(panel)==80 and all(set(p)==set(METHODS) for p in panel.values()),"paired identities")
    for g in reported:
        group=[x[g["method"]] for (s,c),x in panel.items() if g["condition"]=="all" or g["condition"]==c]
        need(len(group)==g["n"] and sum(q["grasp_success"] for q in group)==g["success"] and
             sum(q["full_static_path"] for q in group)==g["static_admitted"] and sum(q["physical_returns"]>0 for q in group)==g["entered_dynamics"]
             and sum(g["outcomes"].values())==g["n"],"method totals")
    for r in contrasts:
        per_scene=defaultdict(list);positive=[];negative=[]
        for (scene,c),methods in panel.items():
            if c!=r["condition"] and r["condition"]!="all":continue
            a,b=methods[r["first"]],methods[r["second"]]
            delta=int(a["grasp_success"])-int(b["grasp_success"])
            per_scene[scene].append(delta)
            if delta==1: positive.append(a["unit"])
            if delta==-1:negative.append(a["unit"])
        names=sorted(per_scene);n=len(names)
        rng=np.random.default_rng(1262275604)
        cluster_weights=rng.multinomial(n,[1/n]*n,size=20000)
        sizes=np.array([len(per_scene[s]) for s in names])
        deltas=np.array([sum(per_scene[s]) for s in names])
        diffs=np.sum(cluster_weights*deltas,axis=1)/np.sum(cluster_weights*sizes,axis=1)
        bounds=np.percentile(diffs,[2.5,97.5])
        need(np.max(np.abs(bounds-np.asarray(r["difference"]["ci95"])))<1e-12 and
             abs(deltas.sum()/sizes.sum()-r["difference"]["value"])<1e-12,"paired estimate or cluster interval")
        need(sorted(positive)==sorted(r["first_only_units"]) and sorted(negative)==sorted(r["second_only_units"]),"discordant units")
        for sign,field in ((1,"first_only_success_rate"),(-1,"first_only_failure_rate")):
            counts=np.array([per_scene[s].count(sign) for s in names])
            fraction=np.sum(cluster_weights*counts,axis=1)/np.sum(cluster_weights*sizes,axis=1)
            need(np.allclose(np.percentile(fraction,[2.5,97.5]),r[field]["ci95"],rtol=0,atol=1e-12),"discordant rate interval")
    checkpoints=[read(OUT/f"admission/checkpoints/after_{v:03}.json") for v in (40,80,120,160,168)]
    need(all(c["shared_faults"]==[] and c["disk"]["above_floor"] and not c["disk"]["events"] and c["disk"]["dirty"]==0 for c in checkpoints),"disk or shared-fault checkpoint")
    write("admission/independent_statistical_verification.json",dict(utc=utc(),individually_verified_trials=240,paired_units=80,
       method_strata_verified=len(reported),paired_cluster_intervals_verified=len(contrasts),exclusive_rate_intervals_verified=2*len(contrasts),resamples_per_contrast=20000,
       prefix_trials_not_redispatched=72,checkpoints=[x["new_completed"] for x in checkpoints],scientific_steps=0))
if __name__=="__main__":check()
