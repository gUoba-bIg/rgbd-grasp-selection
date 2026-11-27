"""Smoke15 immutable scientific inputs and conditional one-slot authorization."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/"IP140_R1_smoke13_slot14_legacy_height_bridge021_aggregate_support01"
PREVIOUS=ROOT/"IP140_R1_smoke14_slot14_contact_aware_static_admission01"
IDENTITY=HERE/"static_input_identity.json"
FROZEN_CORE=("trajectory.py","trajectory_plan.json","frozen_trajectory_commands.jsonl.gz","preregistration.json","object_episode.py","object_evaluator.py","aggregate_support.py","speed_policy.py","model/ur5e_ip140_r1.urdf","model/manifest.json","structural_filter.py","seam_geometry.py","mountpair.py","base_table_mount.py","damping_write.py","object_scene_static.py","episode_log.py","contact_entry.py")
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write_once(p,x):
 with Path(p).open("x",encoding="utf-8",newline="\n") as f:
  json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
def now():return datetime.now(timezone.utc).isoformat()
def invariant():
 inv=read(HERE/"candidate_invariance.json")
 if inv.get("passed") is not True:raise ValueError("immutable input proof failed")
 initial=read(HERE/"ledger_initial.json")
 if sha(HERE/"ledger_initial.json")!=sha(PREVIOUS/"slot_ledger.json"):raise ValueError("initial ledger mismatch")
 if (initial["starts"],initial["existing_budget"],initial["remaining_total_budget"])!=(8,14,6):raise ValueError("initial budget")
 actual=read(HERE/"slot_ledger.json")
 if any(x["charged"] for x in actual["slots"][7:12]):raise ValueError("frozen slots8-12 charged")
 if actual==initial:return_core=True
 else:
  if (actual["starts"],actual["existing_budget"],actual["remaining_total_budget"])!=(9,14,5) or actual["slots"][13].get("charged") is not True:raise ValueError("illegal ledger transition")
  if actual["slots"][:13]!=initial["slots"][:13]:raise ValueError("historical slots changed")
  charge=read(HERE/"episodes/14/precharge.json")
  if charge.get("slot")!=14 or charge.get("starts_before")!=8 or charge.get("starts_after")!=9:raise ValueError("charge proof")
 for name in FROZEN_CORE:
  if sha(HERE/name)!=sha(PREVIOUS/name):raise ValueError("immutable implementation changed: "+name)
def freeze():
 invariant()
 skip={"slot_ledger.json","static_input_identity.json"}
 paths=[p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.relative_to(HERE).as_posix() not in skip]
 ext=dict(read(PREVIOUS/"static_input_identity.json")["external_files"])
 for rel,row in read(HERE/"history_snapshot_before.json")["files"].items():ext[rel]=row["sha256"]
 for rel,d in ext.items():
  if sha(ROOT/rel)!=d:raise ValueError("historical input changed: "+rel)
 obj=dict(version=HERE.name,created_at=now(),static_connections_allowed=1,
  conditional_dynamic_connections_allowed=1,slot14_execution_allowed="only_after_full_static_admission",
  physical_steps_preflight_allowed=0,max_dynamic_steps=4728,allowed_slot=14,new_charges_allowed=1,
  ledger_initial_sha256=sha(HERE/"ledger_initial.json"),
  local_files={p.relative_to(HERE).as_posix():sha(p) for p in paths},external_files=ext)
 write_once(IDENTITY,obj);return obj
def verify():
 invariant();v=read(IDENTITY)
 if v.get("allowed_slot")!=14 or v.get("new_charges_allowed")!=1 or v.get("physical_steps_preflight_allowed")!=0:raise ValueError("scope mismatch")
 for base,key in ((HERE,"local_files"),(ROOT,"external_files")):
  for rel,d in v[key].items():
   if sha(base/rel)!=d:raise ValueError("frozen content changed: "+rel)
 return v
