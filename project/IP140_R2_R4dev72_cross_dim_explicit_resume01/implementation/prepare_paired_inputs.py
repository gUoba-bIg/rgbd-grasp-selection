"""Generate each observation-conditioned P3 pool once, then export all three G1 rankings."""
import os, json, sys, subprocess, hashlib, time, shutil
from cross_unit_audit import review
from pathlib import Path
import numpy as np

V=Path(__file__).resolve().parents[1];ROOT=V.parent
def rd(p):return json.loads(p.read_bytes())
def sh(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def once(p,value):
    with p.open("x",encoding="utf8") as fp:json.dump(value,fp,indent=2,ensure_ascii=False,allow_nan=False)
def invoke(envname,script,args,log):
    exe=ROOT/f"runtime/envs/{envname}/Scripts/python.exe"
    env=dict(os.environ,PYTHONUTF8="1",PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    start=time.perf_counter()
    with log.open("xb") as fp:
        ret=subprocess.run([str(exe),"-B",str(V/"implementation"/script),*map(str,args)],cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT)
    if ret.returncode!=0:raise RuntimeError(f"preseal {script} exited {ret.returncode}; {log}")
    return time.perf_counter()-start

def main():
    if Path.cwd().resolve()!=ROOT:raise ValueError("project working directory required")
    if not rd(V/"audit/zero_interface_admission.json")["passed"]:raise RuntimeError("zero-physics admission missing")
    if rd(V/"development_ledger.json")["method_trials_counted"]!=0:raise RuntimeError("method already started")
    design=rd(V/"preregistration/paired_design.json")
    base=sorted(set(u["base_scene"] for u in design["units"]),key=lambda s:(("021","014","017").index(s.split("-")[1]),int(s.split("-p")[-1])))
    for b in base:
        for filename in ("observed_rgbd.npz","settled_state.json"):
            if not (V/"scenes"/b/filename).is_file():raise RuntimeError("incomplete base scene "+b)
    dst=V/"units";dst.mkdir(exist_ok=False)
    receipt=[]
    for ordinal,u in enumerate(design["units"],1):
        key=u["unit_id"]; d=dst/key;d.mkdir()
        scene=V/"scenes"/u["base_scene"];obs=scene/"observed_rgbd.npz"
        seedkey=base.index(u["base_scene"])+1
        once(d/"network_intent.json",dict(unit_id=key,sensor_input=obs.relative_to(V).as_posix(),input_sha256=sh(obs),
             scene_key=seedkey,frame=0,condition=u["condition"],selected_methods=u["method_order"],physical_outcome_seen=False))
        nt=invoke("vision","audited_forward.py",[obs,d,seedkey,0,u["condition"]],d/"network.log")
        with np.load(d/"network.npz",allow_pickle=False) as nn:
            with (d/"raw_decoder_pool.npy").open("xb") as f:np.save(f,nn["candidates"],allow_pickle=False)
        rt=invoke("grasp_eval","policy_rank.py",[d],d/"scoring.log")
        o=rd(d/"orders.json");n=rd(d/"network.json")
        assert o["candidate_domain_ledger_sha256"]==sh(d/"candidate_domain_ledger.json")
        regression=review(key,ordinal)
        once(d/"cross_field_regression.json",regression)
        assert o["truth_read"] is False and n["networks_returned"]=={"P3":1,"P5_fixed":1}
        assert o["shared_pool_sha256"]==sh(d/"pool.npz") and n["observation_sha256"]==sh(obs)
        with np.load(d/"pool.npz",allow_pickle=False) as p,np.load(d/"network.npz",allow_pickle=False) as nn:
            rows=p["candidate_camera"].tolist()
            assert len(rows)==o["pool_count"] and all(sorted(seq)==list(range(len(rows))) for seq in o["orders_by_pool_index"].values())
            once(d/"platform_input.json",dict(unit_id=key,T_W_C=nn["world_T_camera"].tolist(),candidate_rows=rows,rankings=o["orders_by_pool_index"]))
        for m in ("Raw","A","C0"):
            once(d/("ranking_"+m+".json"),dict(unit_id=key,method=m,shared_pool_sha256=o["shared_pool_sha256"],
                 scores_by_pool_index=o["scores"][m],complete_order=o["orders_by_pool_index"][m],
                 scores_source_sha256=sh(d/"orders.json"),candidate_ids_in_order=[f"{key}:p3:{i}" for i in o["orders_by_pool_index"][m]]))
        material=dict(unit_id=key,base_scene=u["base_scene"],condition=u["condition"],pool_count=len(rows),
             original_rgbd=sh(obs),stable_state=sh(scene/"settled_state.json"),observation_and_inference=sh(d/"network.npz"),
             network=sh(d/"network.json"),pool=o["shared_pool_sha256"],orders=sh(d/"orders.json"),one_pool=True,
             raw_decoded_pool_sha256=sh(d/"raw_decoder_pool.npy"),exclusion_ledger_sha256=sh(d/"candidate_domain_ledger.json"),
             original_network_reused=False,native_inferences_underlying_unit=1,
             candidate_generation_calls=1,segmentation_calls=1,network_process_s=nt,ranking_process_s=rt)
        once(d/"unit_receipt.json",material);receipt.append(material)
        print(key,len(rows),flush=True)
    assert len(receipt)==24
    once(V/"pooled_observations_receipt.json",dict(unit_count=24,ranked_methods=72,raw_pools=24,eligible_pools=24,exclusion_ledgers=24,
        original_network_pairs_reused=0,new_network_pairs=24,old_ranking_used_for_formal_policy=False,units=receipt))
    units=[]
    for row in receipt:
        k=row["unit_id"];d=V/"units"/k
        r=rd(d/"cross_field_regression.json")
        assert r["passed"]
        units.append(dict(unit_id=k,N=r["N"],seven_prior_regression_ordinal=r["ordinal"] if r["ordinal"]<=7 else None,
                     real_network_output_sha256=sh(d/"network.npz"),raw_sha256=sh(d/"raw_decoder_pool.npy"),
                     pool_sha256=sh(d/"pool.npz"),scores_sha256=sh(d/"orders.json"),domain_sha256=sh(d/"candidate_domain_ledger.json"),
                     fields_sha256=sh(d/"cross_field_regression.json"),forward_source_receipt=sh(d/"forward_audit/runtime.json")))
    once(V/"audit/cross_all_units.json",dict(passed=True,unit_count=24,independent_new_P3_processes=24,units=units))

if __name__=="__main__":main()
