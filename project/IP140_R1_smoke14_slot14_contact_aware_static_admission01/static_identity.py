"""Immutable smoke14 inputs; static review has no physical authorization."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/"IP140_R1_smoke13_slot14_legacy_height_bridge021_aggregate_support01"
IDENTITY=HERE/"static_input_identity.json"
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write_once(p,x):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def now():return datetime.now(timezone.utc).isoformat()
def invariant():
    inv=read(HERE/"candidate_invariance.json")
    if inv.get("passed") is not True:raise ValueError("immutable candidate/control proof failed")
    if sha(HERE/"slot_ledger.json")!=sha(OLD/"slot_ledger.json"):raise ValueError("ledger changed")
    ledger=read(HERE/"slot_ledger.json")
    if (ledger["starts"],ledger["existing_budget"],ledger["remaining_total_budget"])!=(8,14,6):raise ValueError("budget mismatch")
    if ledger["slots"][13]["charged"] or any(x["charged"] for x in ledger["slots"][7:12]):raise ValueError("frozen slot started")
    for name in ("trajectory.py","trajectory_plan.json","preregistration.json","object_episode.py","object_evaluator.py","aggregate_support.py","speed_policy.py","model/ur5e_ip140_r1.urdf","model/manifest.json","structural_filter.py","seam_geometry.py","mountpair.py","base_table_mount.py","damping_write.py","object_scene_static.py"):
        if sha(HERE/name)!=sha(OLD/name):raise ValueError("frozen implementation changed: "+name)
def freeze():
    invariant()
    paths=[p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    old=read(OLD/"input_identity.json")
    ext=dict(old["external_files"])
    for rel,row in read(HERE/"history_snapshot_before.json")["files"].items():ext[rel]=row["sha256"]
    for row in read(HERE/"cache_query_source_audit.json")["sources"]:
        ext[row["path"]]=row["sha256"]
    for rel,digest in ext.items():
        if sha(ROOT/rel)!=digest:raise ValueError("historical input changed: "+rel)
    obj=dict(version=HERE.name,created_at=now(),stepSimulation_allowed=False,slot14_execution_allowed=False,
        static_connections_allowed=1,new_charges_allowed=0,ledger_sha256=sha(HERE/"slot_ledger.json"),
        local_files={p.relative_to(HERE).as_posix():sha(p) for p in paths},external_files=ext)
    write_once(IDENTITY,obj);return obj
def verify():
    invariant();v=read(IDENTITY)
    if v["stepSimulation_allowed"] or v["slot14_execution_allowed"]:raise ValueError("no execution authorization")
    for base,key in ((HERE,"local_files"),(ROOT,"external_files")):
        for rel,d in v[key].items():
            if sha(base/rel)!=d:raise ValueError("frozen content changed: "+rel)
    return v
