"""Collect per-observation execution and reproducibility evidence from fresh generated files."""
from common_v1 import *
from collections import Counter
import gzip
import csv
def build():
    study=OUT/"study";plans=read(OUT/"preregistration/observation_plan.json")["jobs"]
    need(read(study/"run/rank_completed.json")["count"]==720,"rank completion")
    dest=OUT/"robustness";dest.mkdir(exist_ok=True);fn=dest/"unit_input_output_sha256.csv.gz"
    counts=Counter();arch=Counter()
    with gzip.open(fn,"xt",encoding="utf8",newline="") as out:
        w=csv.writer(out);w.writerow(["unit","base_scene","condition","sampling_seed","corruption_seed","input_rgbd_sha256","settled_state_sha256",
            "P5_real_returns","P3_real_returns","fresh_depth_network_sha256","raw_decoder_sha256","eligibility_ledger_sha256","pool_sha256",
            "Raw_ranking_sha256","A_ranking_sha256","C0_ranking_sha256","old_E_receipt_sha256","P5_checkpoint_sha256","P3_checkpoint_sha256","joint_component_recomposition","old_observation_array_match","old_decisions_match"])
        for p in plans:
            d=study/"units"/p["unit_id"]
            net=read(d/"network.json");rank=read(d/"rank_done.json");forward=read(d/"forward_done.json")
            need(net["networks_returned"]=={"P3":1,"P5_fixed":1} and net["physics_steps"]==rank["physics_steps"]==0,"unexpected inference or rank count")
            need(net["condition"]==p["condition"] and p["observation_sha256"]==net["observation_sha256"] and net["corruption"]["seed"]==p["condition_seed"],"observation/source binding")
            counts[p["condition"]]+=1
            same=forward["old_new"]["all_array_bytes_equal"] if forward["old_new"] else None
            recomposed=net["corruption"].get("joint_recomposition_exact")
            prior=rank["old_scientific_output_equal"]
            if p["condition"] in ("clean","joint"):
                need(same is True and prior is True,"baseline change");arch["fresh_anchor_units_matching"]+=1
            else:
                need(recomposed is True and same is None and prior is None,"single factor field identity")
                arch["field_reconstructions_matching"]+=1
            weights=net["weights"]
            w.writerow([p["unit_id"],p["scene"],p["condition"],p["sampling_seed"],p["condition_seed"],p["observation_sha256"],p["stable_state_sha256"],
                1,1,sha(d/"network.npz"),sha(d/"raw_decoder.npy"),sha(d/"candidate_domain_ledger.json"),rank["pool_sha256"],
                *[sha(d/("ranking_"+m+".json")) for m in METHODS],rank["selection_sha256"],
                weights["experiments/p5_fusion_v38/fixed/final.pt"],weights["experiments/candidate_v30/epoch_18_final.pt"],recomposed,same,prior])
    need(counts=={m:120 for m in MODES},"scene completeness")
    write("robustness/inference_reconciliation.json",dict(utc=utc(),by_mode=dict(counts),P3_returns=sum(counts.values()),P5_returns=sum(counts.values()),
          fresh_method_rankings=3*sum(counts.values()),source_checks=arch,unit_index_sha256=sha(fn),physics_steps_in_observation_and_ranking=0))
if __name__=="__main__":build()
