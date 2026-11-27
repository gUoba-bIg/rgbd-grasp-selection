"""Offline pre-admission regression. No network inference, engine, or formal unit exports."""
import json
from pathlib import Path
import sys
import numpy as np
import policy_rank as policy

V=Path(__file__).resolve().parents[1]
OLD=V.parent/"IP140_R2_R4dev72_raw_a_c0_paired01"
UNIT="scene-017-p3-clean"

def native(scores,dtype=np.float32):
    a=np.zeros((len(scores),17),dtype=dtype)
    a[:,0]=scores
    a[:,1:4]=[.08,.02,.02]
    a[:,4:13]=np.eye(3).reshape(-1)
    a[:,13]=np.arange(len(a))*.11
    a[:,15]=.8
    a[:,16]=-1
    return a

def domain_rows(a,mask=None):
    if mask is None:mask=np.ones(len(a),dtype=bool)
    return policy.pool_domain_audit(a,mask,policy.decoder_domain(a),"offline")

def main():
    results=[]
    def record(name,ok,observed):
        results.append(dict(test=name,passed=bool(ok),observed=observed))
    def hard(name,a,tag):
        try:domain_rows(a)
        except ValueError as e:record(name,str(e).startswith(tag),str(e))
        else:record(name,False,"unexpected_pass")
    for dtype in (np.float32,np.float64):
        suffix=np.dtype(dtype).name
        small=np.nextafter(dtype(0),dtype(1))
        a=native([dtype(-0.),dtype(0.),small,-small],dtype)
        _,top,z=domain_rows(a)
        record(suffix+"_signed_zero_lowest_positive_negative",top==[2,0,1] and
               z["rows"][0]["score_sign_bit"] and not z["rows"][1]["score_sign_bit"] and
               z["rows"][3]["domain_exclusion_reasons"]==[policy.NEGATIVE_SCORE_REASON],top)
        a=native([1,1,1],dtype)
        a[:,1]=[np.nextafter(dtype(.1),dtype(0)),dtype(.1),np.nextafter(dtype(.1),dtype(1))]
        _,top,z=domain_rows(a)
        record(suffix+"_original_dtype_width_cap",top==[0,1] and z["rows"][2]["domain_exclusion_reasons"]==["native_width_above_decoder_dtype_0p1_m"],a[:,1].tolist())
        for column,name in [(1,"width"),(2,"height"),(3,"depth")]:
            a=native([1,1,1],dtype);a[:,column]=[0,-.01,.01]
            _,top,z=domain_rows(a);reasons=[r["domain_exclusion_reasons"] for r in z["rows"]]
            record(suffix+"_zero_and_negative_"+name,top==[2] and all("nonpositive_native_"+name in rs for rs in reasons[:2]),reasons)
        for col in (0,1,4,13,16):
            for val,desc in ((np.nan,"NaN"),(np.inf,"PosInf"),(-np.inf,"NegInf")):
                a=native([-1],dtype);a[0,col]=val
                hard(f"{suffix}_{desc}_col{col}_even_on_domain_exclusion",a,"nonfinite_native_decoder_row:0")
        for change,name in [(np.diag([1.,1.,-1.]),"reflection"),(np.diag([1.,1.01,1.]),"nonorthogonal")]:
            a=native([-1],dtype);a[0,4:13]=change.reshape(-1)
            hard(suffix+"_"+name+"_despite_negative_score",a,"invalid_native_rotation:0")
    a=native([-.5,-.25])
    _,top,z=domain_rows(a)
    record("all_valid_geometry_negative_native_scores_yield_legitimate_empty_pool",not top and z["domain_eligible_count"]==0 and len(z["rows"])==2,z)
    a=native([.8,.8,.8,.8]);a[1,13:16]=a[0,13:16]
    _,top,z=domain_rows(a,np.array([True,True,False,True]))
    record("observation_filter_and_frozen_nms_before_stable_score_order",len(top)==2 and top==sorted(top) and 2 not in top and top[-1]==3 and z["after_nms_count"]==2,top)
    a=native(np.full(22,.75));_,top,z=domain_rows(a)
    record("ties_at_top20_boundary_retained_by_original_native_index",top==list(range(20)) and z["rows"][-1]["exclusion_reasons"]==["outside_original_score_top20"],top)
    a=native([]);_,top,z=domain_rows(a)
    record("empty_original_decode",top==[] and len(z["rows"])==0,z)
    # Native inputs are read directly from the archived failed export. Reproduce its old observation mask and its pinned NMS.
    folder=OLD/"units"/UNIT
    with np.load(folder/"network.npz",allow_pickle=False) as f:
        a=f["candidates"]; cloud,_=policy.points(f["depth"],f["valid"],f["K"],f["original_size"])
    with np.load(folder/"pool.npz",allow_pickle=False) as f:
        old_pool=f["source_indices"].tolist();old_mask=f["eligibility_mask"];old_rows=f["candidate_camera"]
    before=a.tobytes()
    obs,voxels=policy.filter_candidates(a,cloud,policy.GraspGroup)
    record("frozen_observation_filter_and_independent_geometry",np.array_equal(obs,old_mask) and np.array_equal(obs,policy.independent_mask(a,voxels)),np.flatnonzero(obs).tolist())
    group=policy.GraspGroup(a[obs].copy()).nms(.03,np.pi/6);group.sort_by_score()
    original=policy.exact_indices(a,group[:20].grasp_group_array,np)
    record("historical_pool_reproduces_5",original==old_pool==[11,0,6,3,10] and np.array_equal(old_rows,a[old_pool]),original)
    shared,top,z=domain_rows(a,obs)
    negative=np.flatnonzero(a[:,0]<0).tolist()
    record("historical_15_raw_two_finite_negative_5_to_4",len(a)==15 and negative==[1,10] and top==[11,0,6,3] and all(z["rows"][i]["domain_exclusion_reasons"]==[policy.NEGATIVE_SCORE_REASON] for i in negative),
           dict(raw=len(a),negative_indices=negative,negative_values=a[negative,0].tolist(),before=old_pool,after=top))
    record("native_vector_bytes_unchanged_and_shared_pool_complete",a.tobytes()==before and a[top].tobytes()==old_rows[:4].tobytes() and
           z["methods_share_one_pool"]==["Raw","A","C0"],policy.sha(folder/"network.npz"))
    history=json.loads((OLD/"audit/score_domain_failure.json").read_bytes())
    record("historical_source_manifest_matches_bytes",history["units"][5]["artifacts"]["network.npz"]==policy.sha(folder/"network.npz") and history["units"][5]["artifacts"]["pool.npz"]==policy.sha(folder/"pool.npz"),dict(native=policy.sha(folder/"network.npz"),archived_pool=policy.sha(folder/"pool.npz")))
    # Regression of the original positive/mixed cases: retained order and raw bytes match the historical eligible subset.
    stable_export=[]
    for u in history["units"][:5]:
        unit=OLD/"units"/u["unit"]
        with np.load(unit/"network.npz",allow_pickle=False) as f:candidates=f["candidates"]
        with np.load(unit/"pool.npz",allow_pickle=False) as f:
            old_idx=f["source_indices"];ok=f["eligibility_mask"]
        _,new_idx,report=domain_rows(candidates,ok)
        expected=old_idx[candidates[old_idx,0]>=0].tolist()
        stable_export.append(dict(unit=u["unit"],expected=expected,actual=new_idx))
    record("five_original_completed_rank_export_pools_retain_order",all(s["expected"]==s["actual"] for s in stable_export),stable_export)
    # Full official rank entry on synthetic empty/excluded sets, using only archived RGB-D image arrays.
    fixtures=V/"audit"/"boundary_full_runs";fixtures.mkdir(exist_ok=False)
    with np.load(folder/"network.npz",allow_pickle=False) as f:
        base={k:f[k] for k in f.files if k not in ("candidates","pixels")}
    invalid=native([.5]*7)
    invalid[:2,1]=[0.,-.02];invalid[2:4,2]=[0.,-.02];invalid[4:6,3]=[0.,-.02]
    invalid[6,1]=np.nextafter(np.float32(.1),np.float32(1.))
    for name,arr in (("native_empty",native([])),("all_scores_negative",native([-.3,-.1])),("invalid_dimensions",invalid)):
        dd=fixtures/name;dd.mkdir()
        with (dd/"network.npz").open("xb") as f:
            np.savez_compressed(f,**base,candidates=arr,pixels=np.zeros(len(arr),dtype=np.int64))
        with (dd/"network.json").open("x",encoding="utf8") as f:
            json.dump(dict(synthetic_boundary_fixture=True,network_output_sha256=policy.sha(dd/"network.npz"),
                           inherited_observation_payload=str(folder.relative_to(OLD)),physics_steps=0,network_inference=0),f)
        policy.run(dd)
        ledger=json.loads((dd/"candidate_domain_ledger.json").read_bytes())
        ranks=json.loads((dd/"orders.json").read_bytes())
        with np.load(dd/"pool.npz",allow_pickle=False) as f:
            shape=list(f["candidate_camera"].shape);idx=f["source_indices"].tolist()
        record("full_policy_rank_"+name,shape==[0,17] and idx==[] and ledger["rows"] is not None and
               len(ledger["rows"])==len(arr) and ranks["orders_by_pool_index"]=={"Raw":[],"A":[],"C0":[]} and
               ranks["pool_disposition"]==ledger["pool_disposition"]=="no_eligible_candidate" and
               ranks["shared_pool_sha256"]==policy.sha(dd/"pool.npz") and
               ranks["candidate_domain_ledger_sha256"]==policy.sha(dd/"candidate_domain_ledger.json") and
               all(r["observed_filter_evaluated"]==(name!="invalid_dimensions") for r in ledger["rows"]),
               dict(native_rows=len(arr),eligible_shape=shape,orders=ranks["orders_by_pool_index"],
                    pool_disposition=ranks["pool_disposition"],original_native_hash=policy.sha(dd/"network.npz"),
                    eligible_pool_hash=policy.sha(dd/"pool.npz"),domain_hash=policy.sha(dd/"candidate_domain_ledger.json")))
    all_ok=all(r["passed"] for r in results)
    out=dict(passed=all_ok,total=len(results),passed_count=sum(r["passed"] for r in results),tests=results,
             historical_replayed_raw_domain_ledger=z,source_sha256=policy.sha(Path(__file__)),
             new_shared_pool_source_sha256=policy.sha(Path(policy.__file__)),physics_steps=0,method_trials=0,
             new_formal_units_exported=0,network_forward_calls=0)
    dest=V/"audit/boundary_replay_final.json"
    with dest.open("x",encoding="utf8") as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(out["passed_count"],"/",len(results),"passed;",dest.name,policy.sha(dest))
    if not all_ok:raise SystemExit(1)

if __name__=="__main__":main()

