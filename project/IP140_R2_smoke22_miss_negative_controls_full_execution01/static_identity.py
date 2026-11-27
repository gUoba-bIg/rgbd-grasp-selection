"""Immutable inputs, retained history and monotonic batch accounting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT / "IP140_R2_smoke21_registered_pad_surface_and_resume_four01"
OLD = ROOT / "IP140_R1_smoke13_slot14_legacy_height_bridge021_aggregate_support01"  # archival sealing helper provenance
IDENTITY = HERE / "static_input_identity.json"
ORDER = (10,11,12)
INHERITED_REPLAYS = ("positive_negative_replay.json", "schema_replay19.json", "scope_replay20.json", "surface_replay21.json")
NEW_REPLAYS = ("offline_gate_replay.json", "miss_scope_replay22.json", "accounting_control_replay22.json")

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write_once(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())

def invariant():
    old = read(HERE / "ledger_initial.json")
    from candidate_binding20 import initial_from_parent, binding, PARENT_LEDGER, BINDINGS, assert_bound
    ids=read(BINDINGS)["slots"]
    if (sha(PARENT_LEDGER)!=sha(PREVIOUS/"slot_ledger.json") or old!=initial_from_parent(read(PARENT_LEDGER),ids)
        or any(ids[str(i)]!=binding(i) for i in ORDER)):
        raise ValueError("Bridge identities or historical ledger differ")
    current = read(HERE/"slot_ledger.json")
    delta = current["starts"] - 11
    if not 0 <= delta <= 3 or current["existing_budget"] != 14 or current["remaining_total_budget"] != 3-delta:
        raise ValueError("Budget outside frozen limits")
    charged_order=[i for i in ORDER if current["slots"][i-1]["charged"]]
    if len(charged_order)!=delta:raise ValueError("Actual charged count mismatch")
    for i,(prior,item) in enumerate(zip(old["slots"],current["slots"],strict=True),1):
        if i in ORDER:assert_bound(item,ids[str(i)])
        if i not in charged_order and prior != item:
            from candidate_binding20 import rejected_ledger_row
            receipt=HERE/"slots"/f"{i:02d}"/"static_rejection.json"
            if i not in ORDER or not receipt.is_file() or item!=rejected_ledger_row(prior,receipt):
                raise ValueError(f"Historical or unstarted slot changed without static rejection: {i}")
    if current["charged_in_actual_execution_order"] != old["charged_in_actual_execution_order"] + charged_order:
        raise ValueError("Charge order differs from frozen schedule")
    return current

def freeze():
    invariant()
    for name in INHERITED_REPLAYS:
        if sha(HERE/name)!=sha(PREVIOUS/name):
            raise ValueError("Inherited replay content mismatch: "+name)
    reports = [read(HERE/n) for n in INHERITED_REPLAYS+NEW_REPLAYS]
    if any(not r["passed"] or not r["cases"] or not all(c["passed"] for c in r["cases"]) for r in reports):
        raise ValueError("Offline validation has not passed")
    external = dict(read(PREVIOUS/"static_input_identity.json")["external_files"])
    external.update({k:v["sha256"] for k,v in read(HERE/"history_snapshot_before.json")["files"].items()})
    # Bind asset files directly for all five active registrations, including meshes/configs they reference.
    for rel, digest in read(HERE/"external_assets18.json")["files"].items():
        if rel in external and external[rel] != digest: raise ValueError("Conflicting asset hash: "+rel)
        external[rel]=digest
    for rel,h in external.items():
        if sha(ROOT/rel)!=h:raise ValueError("External identity mismatch: "+rel)
    schemas=read(HERE/"inherited_byte_identities.json")["files"]
    for rel, digest in schemas.items():
        if sha(HERE/rel)!=digest or sha(PREVIOUS/rel)!=digest: raise ValueError("Changed inherited registration/control: "+rel)
    paths=sorted(p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p!=HERE/"slot_ledger.json")
    v=dict(version=HERE.name,allowed_slots=list(ORDER),physical_steps_preflight_allowed=0,
           static_and_dynamic_connections_per_slot=[1,1],max_dynamic_steps_per_slot=4728,ledger_initial_sha256=sha(HERE/"ledger_initial.json"),
           created_utc=now(),local_files={p.relative_to(HERE).as_posix():sha(p) for p in paths},external_files=external)
    write_once(IDENTITY,v)
    return v

def verify():
    invariant()
    v=read(IDENTITY)
    if v["allowed_slots"]!=list(ORDER) or v["physical_steps_preflight_allowed"]!=0:
        raise ValueError("Wrong authorization scope")
    for base,key in ((HERE,"local_files"),(ROOT,"external_files")):
        for name,h in v[key].items():
            if sha(base/name)!=h:raise ValueError("Changed frozen input: "+name)
    return v
