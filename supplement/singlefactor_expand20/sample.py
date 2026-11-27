"""Seal the 20-unit prefixes before allowing access to historical method receipts."""
from paths import *
import subprocess
import zipfile

def main():
    health=disk()
    write("admission/disk_start.json",health)
    need(health["above_floor"],"E free space below 80 GiB")
    manifest=original_manifest()
    frozen_zip=PRIOR/"delivery/V1_supplement_robustness_admission01_review.zip"
    need(sha(frozen_zip)==OLD_ZIP_SHA,"previous delivery drift")
    with zipfile.ZipFile(frozen_zip) as z:
        old_sums=z.read("SHA256SUMS.txt").decode("utf8").splitlines()
    zip_sums={n:h for h,n in (s.split("  ",1) for s in old_sums)}
    previous_files=["preregistration/robustness_sample.json","admission/outcome_blind_draw_verification.json",
                    "admission/outcome_blind_draw.json","preregistration/observation_plan.json"]
    for rel in previous_files:
        h=verify_old(rel,manifest)
        need(zip_sums[rel]==h,"original registration differs from previous bound review archive")
    draw=read(OUT/"preregistration/replayed_frame.json")
    old=read(PRIOR/previous_files[0])
    sealed_draw=read(PRIOR/previous_files[2])
    plan={j["unit_id"]:j for j in read(PRIOR/previous_files[3])["jobs"]}
    need(draw["opened_sources"]==sealed_draw["opened_sources"] and len(draw["opened_sources"])==1201,"sampling access scope")
    need(draw["all_changed_in_hash_order"]==old["all_changed_in_hash_order"]==sealed_draw["all_changed_in_hash_order"],"eligibility frame drift")
    selections={}
    for factor in FACTORS:
        full=draw["all_changed_in_hash_order"][factor]
        expected=sorted(full,key=lambda u:hashlib.sha256((v1.NAME+"|robustness|"+u).encode()).hexdigest())
        need(expected==full and len(set(full))==len(full)>=20,"duplicate or unsorted frame")
        need(full[:6]==old["sampled"][factor]==draw["sampled"][factor],"original first six do not match")
        selections[factor]=[dict(unit_id=u,base_scene=plan[u]["scene"],condition=factor,rank=k,
             screening_hash=plan[u]["screening_order_hash"],method_order=plan[u]["worker_method_order"],
             origin="carried" if k<=6 else "new") for k,u in enumerate(full[:20],1)]
    data=dict(version=OUT.name,utc=utc(),previous_version=v1.NAME,tag=TAG,commit=COMMIT,prior_zip_sha256=OLD_ZIP_SHA,
         prior_registration_sha256=sha(PRIOR/previous_files[0]),blind_replay_sha256=sha(OUT/"preregistration/replayed_frame.json"),
         sample_rule="First 20 in original salted unit_id SHA256 order among any-method pre-E Top-1 pose or presence changes relative to matched clean.",
         prior_prefix_verified=True,frames=draw["populations"],selections=selections,
         counts=dict(units=80,carried_units=24,new_units=56,carried_trials=72,new_trials=168,total_method_trials=240),
         new_execution_order="depth_only, shift_only, missing_only, block_only; within each the frozen hash ranks 7..20 and original per-unit method order",
         maximum_attempts_per_new_identity=1,shared_fault_stop=True,checkpoints_new=[40,80,120,160,168],minimum_disk_free_GiB=80,
         statistics=dict(cluster="base_scene",bootstrap_repetitions=20000,seed=1262275604,
             contrasts=["A-Raw","C0-Raw","C0-A"],ranking_scope="primary sampled 20 per factor; contextual full frame 120 per factor separately",
             no_outcome_filter=True,unobserved_counterfactuals_remain_missing=True))
    write("preregistration/sample80.json",data)
    print({c:(draw["populations"][c],len(rows)) for c,rows in selections.items()},flush=True)
if __name__=="__main__":
    main()
