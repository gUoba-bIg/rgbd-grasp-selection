"""Immutable inputs, retained history and monotonic batch accounting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT / "IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01"
OLD = ROOT / "IP140_R1_smoke13_slot14_legacy_height_bridge021_aggregate_support01"  # archival sealing helper provenance
IDENTITY = HERE / "static_input_identity.json"
ORDER = (8,9,10,11,12)

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
    if sha(HERE/"ledger_initial.json") != sha(PREVIOUS/"slot_ledger.json"):
        raise ValueError("Initial ledger differs from smoke17")
    current = read(HERE/"slot_ledger.json")
    delta = current["starts"] - 9
    if not 0 <= delta <= 5 or current["existing_budget"] != 14 or current["remaining_total_budget"] != 5-delta:
        raise ValueError("Budget outside frozen limits")
    for i,(prior,item) in enumerate(zip(old["slots"],current["slots"],strict=True),1):
        if i not in ORDER[:delta] and prior != item:
            raise ValueError(f"Historical or unstarted slot changed: {i}")
        if i in ORDER[:delta] and not item["charged"]:
            raise ValueError(f"Missing charge for registered start: {i}")
    if current["charged_in_actual_execution_order"] != old["charged_in_actual_execution_order"] + list(ORDER[:delta]):
        raise ValueError("Charge order differs from frozen schedule")
    return current

def freeze():
    invariant()
    reports = [read(HERE/n) for n in ("positive_negative_replay.json","offline_gate_replay.json")]
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
