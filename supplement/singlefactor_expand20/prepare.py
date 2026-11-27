"""Seal all 240 method identities, validate the original 72, and create private worker paths."""
from paths import *
import _winapi
import method_seal

def junction(dst, src):
    need(not os.path.lexists(dst) and src.is_dir() and dst.is_relative_to(STUDY),"invalid private reference")
    _winapi.CreateJunction(str(src.resolve()), str(dst))
    need(dst.is_junction() and dst.resolve()==src.resolve(), "private reference failed")

def setup_paths():
    junction(STUDY/"units",PRIOR/"study/units")
    junction(STUDY/"scenes",ROOT/R6NAME/"scenes")
    for name in ("design.json","design_seal.json"):
        source=ROOT/R6FIRST/"preregistration"/name
        exclusive_bytes(STUDY/"preregistration"/name,source.read_bytes())
    for k in range(1,5):
        lane=STUDY/"workspaces"/f"cpu{k:02}"
        lane.mkdir()
        junction(lane/"data",C/"p/data")
        junction(lane/PL,C/"p"/PL)
        write(lane/"workspace.json",dict(lane=lane.name,private=True,project=str(C/"p")))
    rel="preregistration/observation_fingerprint_index.json"
    exclusive_bytes(OUT/rel,(PRIOR/rel).read_bytes())

def recheck_inputs(uid,job,fp,idx):
    manifest={}
    files=("candidate_domain_ledger.json","forward_done.json","forward_intent.json","network.json","network.npz",
           "orders.json","platform_input.json","pool.npz","ranking_A.json","ranking_C0.json","ranking_Raw.json",
           "rank_done.json","raw_decoder.npy","selection.json")
    for name in files:
        rel="study/units/"+uid+"/"+name
        manifest[name]=verify_old(rel,idx)
    base=PRIOR/"study/units"/uid
    need(manifest["network.npz"]==fp["network_output_sha256"] and manifest["network.json"]==fp["network_receipt_sha256"],"inference fingerprint "+uid)
    need(job["observation_sha256"]==fp["observation_sha256"]==sha(ROOT/job["observation_path"]) and job["stable_state_sha256"]==sha(ROOT/job["stable_state_path"]),"input scene "+uid)
    network=read(base/"network.json")
    need(network["networks_returned"]=={"P3":1,"P5_fixed":1} and network["condition"]==job["condition"] and network["physics_steps"]==0,"prediction provenance")
    need(network["observation_sha256"]==job["observation_sha256"] and network["corruption"]["seed"]==job["condition_seed"],"corruption binding")
    return dict(files=manifest,scene=job["scene"],stable_state=job["stable_state_sha256"],observation=fp["observation_sha256"],tensors=fp["model_input_fingerprints"])

def validate_old_result(tid,it,idx,book,old):
    base=f"study/trials/{tid}/"
    job=f"study/run/jobs/{tid}/"
    # Rehash every archived native stream, snapshot and receipt for this carried method.
    details={rel:verify_old(rel,idx) for rel in idx if rel.startswith((base,job))}
    need(details and details[base+"intent.json"]==old["sha256"][tid],"old intent identity "+tid)
    res=read(PRIOR/base/"result.json");ret=read(PRIOR/job/"return.json");disp=read(PRIOR/job/"dispatch.json")
    need(book["returned"][tid]==details[job+"return.json"] and ret["dispatch_sha256"]==details[job+"dispatch.json"]
         and ret["result_sha256"]==details[base+"result.json"],"old return chain "+tid)
    need(res["intent_sha256"]==disp["intent"]==details[base+"intent.json"] and
         res["trial_id"]==ret["tid"]==disp["tid"]==tid and res["method"]==it["method"] and res["unit"]==it["unit_id"]
         and res["candidate"]==it["selected_candidate_id"] and res["seed"]==it["worker_seed"] and res["condition"]==it["condition"]
         and res["physical_calls"]==res["physical_returns"]==ret["physical_calls"] and not res["shared_failure"]
         and not res["diagnostic_certificate_exception"] and type(res["grasp_success"]) is bool,"original cohort mismatch "+tid)
    if res["physical_returns"]:
        native=read(PRIOR/base/"episode/result.json")
        need(native["telemetry_sha256"]==details[base+"episode/telemetry.jsonl.gz"] and native["step_returns"]==res["physical_returns"],"old native stream")
    for rel in [base+"intent.json",base+"result.json",job+"return.json",job+"dispatch.json"]:
        need(exclusive_bytes(OUT/rel,(PRIOR/rel).read_bytes())==details[rel],"reuse bytes")
    return dict(original_output=str(PRIOR),declared_inputs_rebuilt=True,original_intent_sha256=details[base+"intent.json"],
                original_result_sha256=details[base+"result.json"],original_return_sha256=details[job+"return.json"],
                checked_original_file_hashes=details,additional_runs=0)

def main():
    need((OUT/"admission/v1_identity.json").exists(),"frozen identity not checked")
    sample=read(OUT/"preregistration/sample80.json")
    binding=sha(OUT/"preregistration/sample80.json")
    need(sample["prior_prefix_verified"],"sample not sealed")
    idx=original_manifest()
    for rel in ["study/preregistration/robustness_intents.json","study/run/method_ledger_continued.json",
                "preregistration/observation_plan.json","preregistration/observation_fingerprint_index.json"]:
        verify_old(rel,idx)
    old=read(PRIOR/"study/preregistration/robustness_intents.json")
    book=read(PRIOR/"study/run/method_ledger_continued.json")
    need(book["first_shared_fault"] is None and old["count"]==72,"previous robust ledger not final")
    old_binding=sha(PRIOR/"preregistration/robustness_sample.json")
    jobs={j["unit_id"]:j for j in read(PRIOR/"preregistration/observation_plan.json")["jobs"]}
    fp=read(PRIOR/"preregistration/observation_fingerprint_index.json")["units"]
    fph=sha(PRIOR/"preregistration/observation_fingerprint_index.json")
    design=read(ROOT/R6FIRST/"preregistration/design.json")
    source={(r["unit_id"],r["method"]):r["trial_id"] for r in design["trials"]}
    setup_paths()
    method_seal.STUDY=STUDY
    computed={}
    def capture(relative,obj):
        computed[obj["trial_id"]]=obj
    method_seal.write=capture
    ownership={}; reuse={}; inputs={}; old_order=[]; new_order=[]
    for origin in ("carried","new"):
        for factor in FACTORS:
            for selected in sample["selections"][factor]:
                if selected["origin"]!=origin:continue
                uid=selected["unit_id"];row=jobs[uid]
                inputs[uid]=recheck_inputs(uid,row,fp[uid],idx)
                for method in selected["method_order"]:
                    if origin=="carried":
                        ordinal=len(old_order)+1; tid=old["order"][ordinal-1]; old_order.append(tid)
                    else:
                        ordinal=73+len(new_order); tid=f"EXP-{len(new_order)+1:03}";new_order.append(tid)
                    parent=source[(row["frozen_reference_unit"],method)]
                    values=dict(order=ordinal,binding=old_binding if origin=="carried" else binding,
                                fingerprints=fp[uid]["model_input_fingerprints"],fingerprint_index=fph)
                    method_seal.lock(row,parent,uid,method,tid,"robustness",PRIOR/"study/units","units",values)
                    it=computed.pop(tid)
                    if origin=="carried":
                        original=read(PRIOR/f"study/trials/{tid}/intent.json")
                        need(original==it,"rebuilt historical method intent differs: "+tid)
                        reuse[tid]=validate_old_result(tid,it,idx,book,old)
                        ticket_sha=reuse[tid]["original_intent_sha256"]
                    else:
                        ticket_sha=write(f"study/trials/{tid}/intent.json",it)
                    ownership[tid]=dict(trial_id=tid,origin=origin,ordinal=ordinal,unit_id=uid,method=method,condition=factor,
                        hash_rank=selected["rank"],base_scene=row["scene"],candidate=it["selected_candidate_id"],worker_seed=it["worker_seed"],
                        ranking_sha256=it["ranking_sha256"],pool_sha256=it["pool_sha256"],parent_intent_sha256=it["parent_intent_sha256"],
                        intent_sha256=ticket_sha)
    need(len(inputs)==80 and len(ownership)==240 and len(old_order)==72 and len(new_order)==168 and old_order==old["order"],"budget or order")
    need(len({(x["unit_id"],x["method"]) for x in ownership.values()})==240,"duplicate method assignment")
    write("admission/carried72_reconciliation.json",reuse)
    write("preregistration/unit_input_hashes.json",inputs)
    manifest=write("study/preregistration/robustness_intents.json",dict(kind="robustness",order=new_order,count=168,sha256={t:ownership[t]["intent_sha256"] for t in new_order}))
    write("preregistration/intents240.json",dict(utc=utc(),version=OUT.name,methods=list(METHODS),N=240,carried=old_order,new=new_order,
         counts=dict(per_method=80,per_condition_per_method=20),by_trial=ownership,new_dispatch_manifest=manifest,
         hash_list_sha256=binding,input_manifest_sha256=sha(OUT/"preregistration/unit_input_hashes.json"),
         previous_results_receipt_sha256=sha(OUT/"admission/carried72_reconciliation.json")))
    print("Sealed 80 units and 240 intents; 72 returned bytes verified; 168 new",flush=True)
if __name__=="__main__":
    main()
