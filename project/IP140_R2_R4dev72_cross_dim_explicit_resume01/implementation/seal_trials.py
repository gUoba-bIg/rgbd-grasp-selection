"""Seal observation-only qualification, the balanced schedule and all 72 intents before any attempt."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
from intent_provenance import dependencies, extras
from terminal_intent_check import verify_seal
V=Path(__file__).resolve().parents[1];ROOT=V.parent
ENG=ROOT/"IP140_R2_smoke22_miss_negative_controls_full_execution01"
def sh(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def rd(p):return json.loads(p.read_bytes())
def once(p,value):
    with p.open("x",encoding="utf8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def rel(p):return p.relative_to(V).as_posix()

def main():
    if Path.cwd().resolve()!=ROOT or rd(V/"development_ledger.json")["method_trials_counted"]!=0:raise RuntimeError("scope")
    assert rd(V/"audit/zero_interface_admission.json")["passed"] and rd(V/"pooled_observations_receipt.json")["unit_count"]==24
    assert rd(V/"audit/cross_all_units.json")["passed"]
    assert rd(V/"audit/cross_gpu_boundary/receipt.json")["passed"]
    assert rd(V/"audit/boundary_replay_final.json")["passed"]
    assert rd(V/"audit/reuse_receipt.json")["passed"]
    assert rd(V/"audit/cross_source/site_audit.json")["other_missing_dim_callsites"]==0
    manifest_hash=dependencies()
    design=rd(V/"preregistration/paired_design.json"); f={};inputs=set()
    for u in design["units"]:
        key=u["unit_id"];d=V/"units"/key
        env=dict(os.environ,PYTHONUTF8="1",PYTHONDONTWRITEBYTECODE="1")
        with (d/"qualification.log").open("xb") as log:
            outcome=subprocess.run([sys.executable,"-B",str(V/"implementation/platform_select.py"),str(d)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if outcome.returncode:raise RuntimeError("shared qualification failure at "+key)
        eligible=rd(d/"selection.json");scores=rd(d/"orders.json");shared_pool=d/"pool.npz"
        assert eligible["used_ground_truth"] is False and eligible["selection_input_sha256"]==sh(d/"platform_input.json")
        assert scores["shared_pool_sha256"]==sh(shared_pool) and len(eligible["eligibility"])==scores["pool_count"]
        state=V/"scenes"/u["base_scene"]/"settled_state.json"
        assert state.exists()
        for m in u["method_order"]:
            planned=next(t for t in design["control_order"] if t["method"]==m and t["unit_id"]==key)
            poolidx=eligible["selected"][m]
            expected=next((i for i in scores["orders_by_pool_index"][m] if eligible["eligibility"][i]["eligible"]),None)
            assert expected==poolidx
            dest=V/"trials"/planned["trial_id"];dest.mkdir(parents=True,exist_ok=False)
            r=d/("ranking_"+m+".json");assert rd(r)["shared_pool_sha256"]==sh(shared_pool)
            item=dict(trial_id=planned["trial_id"],unit_id=key,method=m,scene_key=u["base_scene"],condition=u["condition"],
                board_position_index=u["position_index"],selected_pool_index=poolidx,
                selected_candidate_id=None if poolidx is None else eligible["eligibility"][poolidx]["candidate_id"],
                selection_sha256=sh(d/"selection.json"),stable_state_path=rel(state),stable_state_sha256=sh(state),
                pool_path=rel(shared_pool),pool_sha256=sh(shared_pool),ranking_path=rel(r),ranking_sha256=sh(r),
                sensor_and_observation_sha256=sh(d/"unit_receipt.json"),asset_identity_sha256=sh(V/"audit/input_files.json"),
                raw_decoder_path=rel(d/"raw_decoder_pool.npy"),raw_decoder_sha256=sh(d/"raw_decoder_pool.npy"),
                exclusion_ledger_path=rel(d/"candidate_domain_ledger.json"),exclusion_ledger_sha256=sh(d/"candidate_domain_ledger.json"),
                independent_new_budget=False,negative_control_passed=None,belongs_to="exposed_R4_development",attempt_limit=1,
                ranking_uses_ground_truth=False,no_postlock_fallback=True,
                expected_stage=("no_eligible_candidate" if scores["pool_count"]==0 else "no_executable_candidate") if poolidx is None else "postlock_safety_then_frozen_execution",
                outcome_fields="Unobserved until consumed; grasp outcome is never written to a pre-result intent.")
            order_idx=design["control_order"].index(planned)+1
            item.update(extras(u,d,planned,m,poolidx,manifest_hash,order_idx))
            p=dest/"intent.json";once(p,item);f[rel(p)]=sh(p)
        inputs.update(x for x in d.rglob("*") if x.is_file())
    assert len(f)==72
    for folder in ("implementation","preregistration","scenes"):
        inputs.update(x for x in (V/folder).rglob("*") if x.is_file() and "__pycache__" not in x.parts)
    inputs.update(V/name for name in ("audit/input_files.json","audit/zero_interface_admission.json","pooled_observations_receipt.json","audit/intent_dependencies.json","audit/cross_all_units.json","audit/cross_source/site_audit.json","audit/cross_gpu_boundary/receipt.json","audit/reuse_receipt.json","audit/boundary_replay_final.json"))
    assert not list((V/"trials").glob("*/claim.json"))
    once(V/"sealed_intents.json",dict(version=V.name,method_budget=72,unit_count=24,intent_count=72,
        created_utc=datetime.now(timezone.utc).isoformat(),files=f,inputs={rel(p):sh(p) for p in sorted(inputs)},
        trial_results_at_seal=0,balanced_order=[t["trial_id"] for t in design["control_order"]]))
    verify_seal()
if __name__=="__main__":main()
