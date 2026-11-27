"""Derive hash-bound supplementary intents, keeping all scientific fields under their original owners."""
from common_v1 import *
STUDY=OUT/"study"
def lock(row,src,uid,m,tid,kind,source_root,unit_prefix,unitslot):
    d=source_root/uid
    original=ROOT/R6NAME/"trials"/src/"intent.json"
    bound=read(original)
    s=read(d/"selection.json");i=s["selected"][m]
    o=read(d/"orders.json")
    need(bound["method"]==m and bound["scene_key"]==row["scene"] and not bound["r5_interface_in_chain"],"parent intent")
    if kind.startswith("diagnostic"):
        need(i==bound["selected_pool_index"] and bound["selected_candidate_id"]==unitslot["candidate"] and bound["worker_seed"]==unitslot["worker_seed"],"original diagnostic seed/candidate")
    p3=None
    if i is not None:
        import numpy as np
        with np.load(d/"pool.npz",allow_pickle=False) as pool:c=pool["candidate_camera"][i];ix=int(pool["source_indices"][i])
        cd=o["candidate_details"][i]
        p3=dict(native_source_index=ix,pool_index=i,candidate_id=f"{uid}:p3:{i}",camera_row=c.tolist(),camera_original_dtype=c.dtype.str,
                camera_row_sha256=hashlib.sha256(c.tobytes()).hexdigest(),world_17=cd["candidate_world"],world_T_G=cd["world_T_G"],
                pre_open_m=cd["pre_open_m"],legacy_height_correction_m=0.)
        need(not cd["old_Robotiq_height_adjustment_applied"] and p3["candidate_id"]==s["eligibility"][i]["plan"]["candidate_id"],"original pose or plan")
        if kind.startswith("diagnostic"):need(p3==bound["selected_p3"],"original 17D/pose or source identity")
    prefixes={"pool":"pool.npz","raw_decoder":"raw_decoder.npy","exclusion":"candidate_domain_ledger.json","ranking":"ranking_"+m+".json","selection":"selection.json","scoring_components":"orders.json"}
    it=dict(bound,trial_id=tid,method=m,unit_id=uid,condition=row["condition"],selected_pool_index=i,
        selected_candidate_id=p3["candidate_id"] if p3 else None,selected_p3=p3,no_candidate=i is None,common_pool_count=o["pool_count"],
        old_E_count=sum(a["eligible"] for a in s["eligibility"]),scope="V1 supplementary only",execution_order=unitslot["order"],
        parent_trial=src,parent_intent_sha256=sha(original),supplement_kind=kind,supplement_selection_manifest=unitslot["binding"],
        condition_seed=row.get("condition_seed",bound["condition_seed"]),supplementary_input_identity=True)
    for tag,fn in prefixes.items():
        it[tag+"_path"]=(Path(unit_prefix)/uid/fn).as_posix();it[tag+"_sha256"]=sha(d/fn)
    need(it["old_E_count"]==len([a for a in s["eligibility"] if a["eligible"]]),"eligibility counts")
    for tag in ("stable_state","observation"):
        need(sha(STUDY/it[tag+"_path"])==it[tag+"_sha256"],"stable scene identity")
    if kind=="robustness":
        it["parent_model_input_fingerprints"]=bound["model_input_fingerprints"]
        it["model_input_fingerprints"]=unitslot["fingerprints"]
        it["actual_unit_inputs_sha256"]=unitslot["fingerprint_index"]
        it["actual_unit_inputs_path"]="preregistration/observation_fingerprint_index.json"
        it["parent_source_bundle_sha256"]=bound["source_bundle_sha256"]
    return write("study/trials/"+tid+"/intent.json",it)

def seal(kind):
    import json
    from rank_stability import load_unit,compare
    from history import table,stability_summary
    diag=read(OUT/"preregistration/diagnostic_sample.json");jobs=read(OUT/"preregistration/observation_plan.json")["jobs"]
    plan={j["unit_id"]:j for j in jobs};members={};order=[]
    originaldesign=read(ROOT/R6FIRST/"preregistration/design.json")
    lookup={(r["unit_id"],r["method"]):r["trial_id"] for r in originaldesign["trials"]}
    if kind=="diagnostic":
        binding=sha(OUT/"preregistration/diagnostic_sample.json")
        for k,r in enumerate(diag["cases"],1):
            t=r["supplement_trial_id"];m=r["method"];uid=r["unit"];x=dict(r,order=k,binding=binding)
            members[t]=lock(r,r["trial"],uid,m,t,"diagnostic_"+r["group"],ROOT/R6NAME/"units","frozen_r6_units",x)
            order.append(t)
    else:
        need((STUDY/"run/rank_completed.json").is_file(),"missing new P3/P5/old E returns")
        from input_fingerprints import build
        fingerprint_rows,fingerprint_index=build(jobs)
        matches=[];pairs={};selected={};results=[]
        for j in jobs:
            if j["condition"]=="clean":continue
            base=j["scene"];uid=j["unit_id"];clean=base+"-clean"
            a=load_unit(STUDY/"units"/clean);b=load_unit(STUDY/"units"/uid)
            rows,mp=compare(a,b,base,j["condition"])
            matches.extend(rows);pairs[uid]=mp
            # Distinguish method divergence from across-observation change.
            for i,m in enumerate(METHODS):
                for n in METHODS[i+1:]:
                    results.append(dict(scene=base,condition=j["condition"],method1=m,method2=n,
                        first_selection=b[3][m],second_selection=b[3][n],different=b[3][m]!=b[3][n]))
            if j["condition"]!="joint" and any(r["top1"] in ("presence_changed","pose_changed") for r in rows):
                selected.setdefault(j["condition"],[]).append(uid)
        for j in jobs:
            if j["condition"]=="clean":
                u=load_unit(STUDY/"units"/j["unit_id"])[3]
                for i,m in enumerate(METHODS):
                    for n in METHODS[i+1:]:
                        results.append(dict(scene=j["scene"],condition="clean",method1=m,method2=n,first_selection=u[m],second_selection=u[n],different=u[m]!=u[n]))
        table("robustness/selection_divergence.csv",results);table("robustness/rank_comparisons.csv",matches)
        write("robustness/pose_correspondences.json",pairs)
        write("robustness/rank_summary.json",stability_summary(matches))
        ordered={c:sorted(selected.get(c,[]),key=lambda uid:plan[uid]["screening_order_hash"]) for c in MODES[1:5]}
        sample={c:v[:6] for c,v in ordered.items()}
        binding=write("preregistration/robustness_sample.json",dict(utc=utc(),all_changed_in_hash_order=ordered,sampled=sample,selection_reads_outcomes=False))
        for condition,items in sample.items():
            for uid in items:
                j=plan[uid]
                for m in j["worker_method_order"]:
                    k=len(order)+1;t=f"ROB-{k:03}";src=lookup[j["frozen_reference_unit"],m]
                    record=dict(order=k,binding=binding,fingerprints=fingerprint_rows[uid]["model_input_fingerprints"],fingerprint_index=fingerprint_index)
                    members[t]=lock(j,src,uid,m,t,"robustness",STUDY/"units","units",record);order.append(t)
    write(f"study/preregistration/{kind}_intents.json",dict(utc=utc(),kind=kind,order=order,count=len(members),sha256=members))
if __name__=="__main__":
    import sys
    seal(sys.argv[1])
