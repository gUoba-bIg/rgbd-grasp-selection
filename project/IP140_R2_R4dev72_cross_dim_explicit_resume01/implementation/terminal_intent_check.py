"""Independently verify every intent against its native row, model input tensors, registration and pre-outcome seal."""
import hashlib
import json
from pathlib import Path
import numpy as np

V=Path(__file__).resolve().parents[1]
ROOT=V.parent

def rd(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def tensor_sh(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def require(ok,reason):
    if not ok:raise ValueError(reason)
def same(x,y):return x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()

def verify_seal():
    terminal=V/"audit/terminal_intent_admission.json"
    if terminal.exists():raise FileExistsError(terminal)
    seal=rd(V/"sealed_intents.json");ledger=rd(V/"development_ledger.json")
    require(ledger["method_trials_counted"]==0 and not list((V/"trials").glob("*/method_result.json")) and not list((V/"trials").glob("*/claim.json")),"terminal audit after outcome or claim")
    require(seal["method_budget"]==72 and seal["unit_count"]==24 and seal["intent_count"]==72 and seal["trial_results_at_seal"]==0,"incomplete pre-outcome seal")
    for path,h in (seal["inputs"]|seal["files"]).items():
        p=(V/path).resolve()
        require(p.is_relative_to(V) and sha(p)==h,"sealed file changed "+path)
    dep_path=V/"audit/intent_dependencies.json";dep=rd(dep_path);registered=rd(V/"audit/input_files.json")
    require(dep["origin_registry_sha256"]==sha(V/"audit/input_files.json"),"invalid asset freeze")
    files=dep["project_relative_sha256"];dep_hash=sha(dep_path)
    for k,h in files.items():
        p=(ROOT/k).resolve();require(p.is_relative_to(ROOT) and sha(p)==h,"source input changed "+k)
    require(all(files[k]==v["sha256"] for k,v in registered["files"].items()),"registry not fully bound")
    for p in (V/"implementation").rglob("*.py"):
        require(files[p.relative_to(ROOT).as_posix()]==sha(p),"unbound implementation source")
    reqs=("audit/cross_all_units.json","audit/cross_gpu_boundary/receipt.json","audit/boundary_replay_final.json","audit/reuse_receipt.json")
    require(all(rd(V/q)["passed"] is True and q in seal["inputs"] for q in reqs),"incomplete source admission")
    cross=rd(V/"audit/cross_source/site_audit.json");require(files[cross["source"]]==cross["after_sha256"],"cross source identity")
    design=rd(V/"preregistration/paired_design.json")
    prereg=rd(V/"preregistration/input_contract.json")
    ctrl=design["control_order"];by_id={t["trial_id"]:i+1 for i,t in enumerate(ctrl)}
    bases=sorted({u["base_scene"] for u in design["units"]},key=lambda s:(("021","014","017").index(s.split("-")[1]),int(s.split("-p")[-1])))
    require(len(bases)==12 and seal["balanced_order"]==[r["trial_id"] for r in ctrl]==[r["trial_id"] for r in ledger["trials"]],"sequence")
    outputs=[]
    for u in design["units"]:
        key=u["unit_id"];d=V/"units"/key;scene=V/"scenes"/u["base_scene"]
        original=scene/"observed_rgbd.npz";stable=scene/"settled_state.json"
        selection=rd(d/"selection.json");o=rd(d/"orders.json");net=rd(d/"network.json")
        audit=rd(d/"forward_audit/row_comparison.json");ni=rd(d/"network_intent.json")
        seedkey=bases.index(u["base_scene"])+1;seed=prereg["p3_base_seed"]+1000*seedkey
        require(selection["used_ground_truth"] is False and selection["never_fallback_after_postlock_safety"] is True and selection["selection_input_sha256"]==sha(d/"platform_input.json"),"selection source")
        require(audit["unit"]==ni["unit_id"]==key and audit["passed"] is True and audit["condition"]==ni["condition"]==u["condition"]==net["condition"] and
                audit["scene"]==ni["scene_key"]==seedkey and audit["seed"]==seed and audit["frame"]==ni["frame"]==0,"condition/seed")
        h=sha(original);require(h==net["observation_sha256"]==ni["input_sha256"] and (V/ni["sensor_input"]).resolve()==original.resolve(),"stable sensor")
        with np.load(d/"forward_audit/diagnostic_decoder_inputs.npz",allow_pickle=False) as inp:
            for method,tag,shape in (("P3","point_clouds",(1,20000,3)),("P5","p5_forward_input",(1,5,480,640))):
                tensor=inp[tag];measured=audit["native_forward_inputs"][method]
                require(shape==tensor.shape==tuple(measured["shape"]) and str(tensor.dtype)==measured["dtype"]=="float32" and
                        tensor_sh(tensor)==measured["sha256"],"actual input tensor "+key+" "+method)
        with np.load(d/"network.npz",allow_pickle=False) as nn:
            native=nn["candidates"];wc=np.asarray(nn["world_T_camera"],dtype=float)
        require(same(native,np.load(d/"raw_decoder_pool.npy",allow_pickle=False)),"native decoder bytes")
        with np.load(d/"pool.npz",allow_pickle=False) as pp:ix=pp["source_indices"];rows=pp["candidate_camera"]
        require(same(rows,native[ix]) and len(ix)==len(selection["eligibility"]),"pool to original bytes")
        subset=[r for r in ctrl if r["unit_id"]==key];require(len(subset)==3 and {r["method"] for r in subset}=={"Raw","A","C0"},"paired triplet")
        for plan in subset:
            method=plan["method"];trial=plan["trial_id"];path=V/"trials"/trial/"intent.json";t=rd(path)
            require(seal["files"][path.relative_to(V).as_posix()]==sha(path),"sealed trial")
            require((t["trial_id"],t["unit_id"],t["method"],t["condition"],t["scene_key"],t["board_position_index"])==
                    (trial,key,method,u["condition"],u["base_scene"],u["position_index"]),"intent identity")
            require(t["execution_order_1based"]==by_id[trial] and t["within_unit_order"]==plan["within_unit_index"] and t["unit_order_1based"]==u["unit_order"],"order binding")
            require(t["seed_policy"]==dict(order=design["order_seed"],p3_base=prereg["p3_base_seed"],p3_sampling=seed,scene_key=seedkey,frame=0,condition_recipe=net["corruption"]),"seed binding")
            require(t["actual_input_fingerprints"]==audit["native_forward_inputs"] and t["weights_sha256"]==net["weights"] and
                    all(files[k]==v for k,v in net["weights"].items()) and t["cross_source_sha256"]==cross["after_sha256"],"input/weights identities")
            bindings={"stable_observation":original,"stable_state":stable,"pool":d/"pool.npz","ranking":d/f"ranking_{method}.json",
                      "raw_decoder":d/"raw_decoder_pool.npy","exclusion_ledger":d/"candidate_domain_ledger.json",
                      "actual_input_hashes":d/"forward_audit/row_comparison.json","dependencies_manifest":dep_path}
            for tag,p in bindings.items():
                require((V/t[tag+"_path"]).resolve()==p.resolve() and t[tag+"_sha256"]==sha(p),"intent file "+tag)
            require(t["network_sha256"]==sha(d/"network.npz") and t["sensor_and_observation_sha256"]==sha(d/"unit_receipt.json")
                    and t["asset_identity_sha256"]==sha(V/"audit/input_files.json") and t["selection_sha256"]==sha(d/"selection.json"),"remaining source bindings")
            rank=rd(bindings["ranking"]);order=rank["complete_order"]
            require(sorted(order)==list(range(len(ix))) and order==o["orders_by_pool_index"][method] and rank["method"]==method
                    and rank["shared_pool_sha256"]==t["pool_sha256"]==o["shared_pool_sha256"] and rank["scores_source_sha256"]==sha(d/"orders.json"),"ranking binding")
            idx=next((i for i in order if selection["eligibility"][i]["eligible"]),None)
            require(t["selected_pool_index"]==selection["selected"][method]==idx and t["no_candidate"]==(idx is None) and
                    t["empty_common_pool"]==(len(ix)==0) and t["no_postlock_fallback"] and t["negative_control_passed"] is None,"irrevocable selection")
            q=t["formal_p3_candidate"]
            if idx is None:
                require(q is None and t["selected_candidate_id"] is None and t["expected_stage"]==("no_eligible_candidate" if len(ix)==0 else "no_executable_candidate"),"empty choice")
            else:
                row=native[ix[idx]];x=row.astype(float);r=x[4:13].reshape(3,3)
                world=x.copy();world[13:16]=wc[:3,:3]@x[13:16]+wc[:3,3];world[4:13]=(wc[:3,:3]@r).ravel()
                cg=np.eye(4);cg[:3,:3]=r;cg[:3,3]=x[13:16]+x[3]*r[:,0]
                require(t["selected_candidate_id"]==f"{key}:p3:{idx}" and q["native_index"]==int(ix[idx]) and q["pool_index"]==idx and
                        q["raw_bytes_sha256"]==tensor_sh(row) and q["original_dtype"]==row.dtype.str and q["camera_row_bytes_hex"]==row.tobytes().hex()
                        and same(row,np.asarray(q["camera_17"],dtype=row.dtype)) and
                        np.allclose(q["world_17"],world,atol=1e-12,rtol=0) and np.allclose(q["world_T_G"],wc@cg,rtol=0,atol=1e-12)
                        and q["pre_open_m"]==x[1] and q["legacy_height_adjustment_m"]==0.,"native row/world mapping")
            outputs.append(dict(trial_id=trial,unit_id=key,method=method,pool_index=idx,
                native_index=None if idx is None else int(ix[idx]),intent_sha256=sha(path),pool_sha256=t["pool_sha256"],
                stable_state_sha256=t["stable_state_sha256"],native_forward_inputs=audit["native_forward_inputs"]))
    require(len(outputs)==72 and len({o["unit_id"] for o in outputs})==24 and len({o["trial_id"] for o in outputs})==72,"terminal count")
    report=dict(passed=True,verified_trial_intents=72,verified_triplets=24,trial_results_before_admission=0,
        sealed_intents_sha256=sha(V/"sealed_intents.json"),source_manifest_sha256=dep_hash,verifier_sha256=sha(Path(__file__)),trials=outputs)
    with terminal.open("x",encoding="utf8") as out:json.dump(report,out,indent=2,ensure_ascii=False,allow_nan=False)
    return report

if __name__=="__main__":verify_seal()

