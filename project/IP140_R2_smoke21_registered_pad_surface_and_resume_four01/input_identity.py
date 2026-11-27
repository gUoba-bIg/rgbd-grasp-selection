"""Immutable inputs for one authorized offline + zero + conditional slot14 session."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / "IP140_R1_smoke12_slot07_base_table_mount_filter01"
IDENTITY = HERE / "input_identity.json"
REGISTERED_ID = "ip140bridge:slot14:parent-slot07:undo-legacy-z021"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_once(path, value):
    import os
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def check_prior():
    plan = read(HERE / "preregistration.json")
    old_plan = read(OLD / "preregistration.json")
    ledger = read(HERE / "ledger_initial.json")
    prior = read(OLD / "slot_ledger.json")
    if (plan["slots"][:13] != old_plan["slots"] or plan["eligible_slots_after_zero_gate"] != [14]
            or plan["slots"][13]["candidate_id"] != REGISTERED_ID
            or ledger["starts"] != prior["starts"] or prior["starts"] != 8
            or prior["remaining_total_budget"] != 5 or ledger["existing_budget"] != 14
            or ledger["remaining_total_budget"] != 6
            or ledger["charged_in_actual_execution_order"] != [1, 2, 3, 4, 5, 6, 13, 7]
            or not ledger["slots"][6]["charged"] or ledger["slots"][13]["charged"]
            or prior["slots"][5]["first_failed_step"] != 2837
            or prior["slots"][12]["independent_confirmation"] is not False
            or any(x["charged"] for x in ledger["slots"][7:12])):
        raise ValueError("slot identity, historical record or approved budget mismatch")
    hist=read(OLD/"episodes/07/result.json")
    if hist["grasp_success"] is not False or hist["slot_passed"] is not False:
        raise ValueError("historical slot7 failure changed")



def freeze():
    check_prior()
    local = set(HERE.glob("*.py"))
    local.update(HERE / name for name in (
        "preregistration.json", "ledger_initial.json", "trajectory_plan.json", "mountpair_registration.json",
        "approved_scope.md", "base_table_registration.json", "candidate_difference_proof.json", "prior_package_verification.json", "history_snapshot_before.json", "static_path_offline_replay.json", "model/ur5e_ip140_r1.urdf", "model/manifest.json"))
    local.update((HERE / "static_precheck").glob("*"))
    local.update(f for f in (HERE / "history").rglob("*") if f.is_file())
    external = {}
    def bind(path, expected):
        if digest(ROOT / path) != expected:
            raise ValueError("source hash mismatch: " + path)
        external[path] = expected
    sources = read(HERE/"static_precheck/asset_source_audit.json")
    for entry in sources["files"]:
        bind(entry["path"], entry["frozen_sha256"])
    damping = read(HERE/"static_precheck/damping_readback_blocker.json")
    bind(damping["binary_path"], damping["binary_sha256"])
    for entry in damping["source_evidence"]:
        bind(entry["path"], entry["sha256"])
    model = read(HERE/"model/manifest.json")
    for key in ("provenance","reused_arm_asset_sha256"):
        for path, expected in model[key].items():
            bind(path, expected)
    for entry in read(HERE/"history/sources.json"):
        bind(entry["source"], entry["sha256"])
        if digest(HERE / entry["package"]) != entry["sha256"]:
            raise ValueError("historical copy mismatch: " + entry["package"])
    for f in (HERE/"history"/OLD.name).rglob("*"):
        if f.is_file(): bind((OLD/f.relative_to(HERE/"history"/OLD.name)).relative_to(ROOT).as_posix(), digest(f))
    for entry in read(HERE/"trajectory_plan.json")["hashes"]:
        if entry in ("config","arm_asset_manifest"):
            path = "configs/simulation_v52r2.json" if entry=="config" else "data/simulation_v52/robot/robot_assets.json"
            bind(path,read(HERE/"trajectory_plan.json")["hashes"][entry])
    for f in (OLD/'zero_step').glob('*.json'):
        bind(f.relative_to(ROOT).as_posix(), digest(f))
    for rel, expected in read(HERE/'base_table_registration.json')['source_sha256'].items():
        bind(rel,expected)
    ext_cfg="configs/simulation_v52e.json"  
    bind(ext_cfg,digest(ROOT/ext_cfg))
    for row in read(HERE/"prior_package_verification.json")["packages"]:
        if not row["passed"]: raise ValueError("prior package verification failed")
        bind(row["zip_path"],row["sha256"])
        bind(row["manifest_path"],row["manifest_sha256"])
    for row in read(HERE/"candidate_difference_proof.json").get("sources",[]):
        if isinstance(row,dict) and "path" in row and "sha256" in row: bind(row["path"],row["sha256"])
    for rel, row in read(HERE/"history_snapshot_before.json")["files"].items():
        bind(rel,row["sha256"])
    receipt = dict(version=HERE.name, created_utc=timestamp(), new_files={
        f.relative_to(HERE).as_posix():digest(f) for f in sorted(local) if f.is_file()},
        external_files=external, old_input_manifest_sha256=digest(OLD/"SHA256SUMS.txt"))
    write_once(IDENTITY, receipt)
    return receipt


def verify():
    check_prior()
    recorded=read(IDENTITY)
    for base, group in ((HERE, "new_files"),(ROOT,"external_files")):
        for rel, expected in recorded[group].items():
            if digest(base/rel) != expected:
                raise ValueError("frozen file changed: " + rel)
    return recorded
