"""One task, one isolated current directory and durable job claim. Frozen kernels own all decisions."""
import sys,os,random,time,json,importlib.util,traceback
from pathlib import Path
import numpy as np
from common import *
sys.path[:0]=[str(HERE/"implementation"),str(R4/"implementation"),str(ROOT)]
def bind_engine():
    spec=importlib.util.spec_from_file_location("pybullet",str(BINARY));p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    sys.modules["pybullet"]=p
    require(Path(p.__file__).resolve()==BINARY and sha(BINARY)=="882cd523a255e66c94d5b91c119f9c688fdbc0956161eff30f2171e21677ab49","native binary identity")
def make_unit(u,scene):
    from policy_network import generate
    import torch
    from src.grasp_prerequisite_data_v27 import SOURCE
    import graspnet,loss_utils,modules,label_generation
    domain=ROOT/"runtime/build/graspnet/utils/loss_utils.py"
    require(Path(loss_utils.__file__).resolve()==domain and sha(domain)=="0a8dea7a09020f869c329781a6b2a713f8ba62d385d07dab43deaab55e007eb7" and
        all(f is loss_utils.batch_viewpoint_params_to_matrix for f in (modules.batch_viewpoint_params_to_matrix,label_generation.batch_viewpoint_params_to_matrix,graspnet.pred_decode.__globals__["batch_viewpoint_params_to_matrix"])),"active P3 decoder")
    require(torch.__version__=="2.6.0+cu126" and torch.cuda.is_available(),"frozen inference environment")
    d=HERE/"units"/u["unit_id"];d.mkdir(exist_ok=False)
    obs=HERE/"scenes"/scene["base_scene"]/"observed_rgbd.npz"
    origin=read(obs.parent/"receipt.json")
    require(origin["observation_rgbd_sha256"]==sha(obs) and origin["actual_steps"]==origin["step_returns"]==240 and origin["method_trials"]==0,"stable observation binding")
    put(d/"forward_intent.json",dict(unit=u["unit_id"],source=sha(obs),point_sampling_seed=scene["sampling_seed"],noise_seed=scene["condition_seed"][u["condition"]],
        torch=torch.__version__,active_decoder_sha256=sha(domain),camera_policy_inputs=["rgb","depth_buffer","view","projection"],weights={"P3":sha(ROOT/"experiments/candidate_v30/epoch_18_final.pt"),"P5":sha(ROOT/"experiments/p5_fusion_v38/fixed/final.pt")}))
    generate(obs,d,scene["scene_index"],0,u["condition"])
    net=read(d/"network.json")
    require(net["networks_returned"]=={"P3":1,"P5_fixed":1} and net["network_output_sha256"]==sha(d/"network.npz") and net["observation_sha256"]==sha(obs),"network returns")
    with np.load(d/"network.npz",allow_pickle=False) as n:
        a=n["candidates"]
        require(a.shape==(len(a),17) and np.isfinite(a).all(),"nonfinite or invalid P3 raw rows")
        with (d/"raw_decoder.npy").open("xb") as out:np.save(out,a,allow_pickle=False)
def rank_and_E(u):
    from policy_rank import run
    d=HERE/"units"/u;run(d)
    o=read(d/"orders.json");require(not o["truth_read"] and o["physics_steps"]==0 and o["shared_pool_sha256"]==sha(d/"pool.npz"),"pool owner or truth")
    with np.load(d/"pool.npz",allow_pickle=False) as pool,np.load(d/"network.npz",allow_pickle=False) as n:
        rows=pool["candidate_camera"]; inds=pool["source_indices"];T=n["world_T_camera"].tolist()
        require(np.array_equal(rows,n["candidates"][inds]),"pool source index")
    for m,rank in o["orders_by_pool_index"].items():
        scores=o["scores"][m]
        require(len(scores)==len(rows) and np.isfinite(scores).all() and sorted(rank)==list(range(len(rows))),"rank identity")
        put(d/("ranking_"+m+".json"),dict(method=m,unit_id=u,scores_by_pool_index=scores,complete_order=rank,shared_pool_sha256=o["shared_pool_sha256"],
            scores_source_sha256=sha(d/"orders.json"),candidate_ids_in_order=[f"{u}:p3:{i}" for i in rank]))
    data=dict(unit_id=u,T_W_C=T,candidate_rows=rows.tolist(),rankings=o["orders_by_pool_index"]);inp=put(d/"platform_input.json",data)
    bind_engine()
    from platform_select import evaluate
    from platform_trace import R4Trace
    from scene_world import load_eligibility_world
    p=R4Trace(sys.modules["pybullet"],d/"eligibility_api","eligibility");client=p.connect(p.DIRECT)
    try:
        ctx=load_eligibility_world(p,client,d/"eligibility_setup");sel=evaluate(data,ctx)
    finally:p.disconnect(physicsClientId=client);p.close()
    require(p.counts==p.returns and not p.exceptions and not p.actual_step_calls and p.counts["setCollisionFilterPair"]==14 and len(sel["eligibility"])==len(rows),"incomplete prelock E")
    sel.update(selection_input_sha256=inp,native_eligibility_api_sha256=sha(d/"eligibility_api/calls.jsonl.zst"))
    put(d/"selection.json",sel)
def main():
    kind,key,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    require(workspace_ok(),"unisolated process cwd")
    job=HERE/"run"/"jobs"/kind/key
    lane=Path.cwd()
    import msvcrt
    with (lane/"lane.lock").open("a+b") as lock:
        if lock.seek(0,2)==0:lock.write(b"1");lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            t=time.monotonic();claim=put(job/"launch.json",dict(pid=os.getpid(),cwd=str(lane),temporary=os.environ["TMP"],seed=seed,kind=kind,id=key,utc=now()))
            random.seed(seed);np.random.seed(seed)
            design=read(HERE/"preregistration/design.json")
            require(sha(HERE/"preregistration/design.json")==read(HERE/"preregistration/design_seal.json")["design_sha256"],"formal design drift")
            if kind=="scene":
                bind_engine()
                from scene_acquire import observation
                observation(key,HERE/"audit/input_preflight.json")
            elif kind=="forward":
                u=next(x for x in design["units"] if x["unit_id"]==key);scene=next(x for x in design["scenes"] if x["base_scene"]==u["scene"])
                make_unit(u,scene)
            elif kind=="rank":rank_and_E(key)
            elif kind=="method":
                bind_engine()
                from platform_trial import worker
                dest=HERE/"trials"/key
                require(worker(dest/"intent.json")==0,"shared frozen trial fault: "+str(read(dest/"method_result.json").get("failure_detail")))
            else:raise ValueError("unknown task")
            put(job/"return.json",dict(completed=True,kind=kind,key=key,worker_claim_sha256=claim,elapsed_sec=time.monotonic()-t,utc=now()))
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        p=HERE/"run"/"jobs"/sys.argv[1]/sys.argv[2]/"exception.json"
        put(p,dict(error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        raise
