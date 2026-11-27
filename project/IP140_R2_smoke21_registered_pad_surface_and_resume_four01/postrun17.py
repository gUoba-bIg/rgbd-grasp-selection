"""Read-only smoke17 evidence audit, closeout and packaging. No engine imports."""
from pathlib import Path
from collections import Counter
import json,gzip,hashlib,zipfile,difflib
from static_identity import HERE,ROOT,PREVIOUS,read,sha,write_once,verify,now
def optional(path):return read(path) if path.is_file() else None
def api_audit(scope):
 directory=HERE/scope if scope else HERE
 path=directory/"full_api/calls.jsonl.gz"
 if not path.exists():return dict(present=False,calls={},returns={},errors={},passed=None)
 calls=Counter();returns=Counter();errors=Counter();pending={};filters=[];bad=[];first=None;reset_after=[]
 with gzip.open(path,"rt",encoding="utf-8") as f:
  for line_no,line in enumerate(f,1):
   row=json.loads(line);seq,event=row[:2]
   if event=="call":
    method=row[3];calls[method]+=1;pending[seq]=method
    if method=="stepSimulation" and first is None:first=line_no
    if method=="setCollisionFilterPair":filters.append(dict(line=line_no,raw=row))
    if first and method in ("resetJointState","resetBasePositionAndOrientation","resetBaseVelocity"):reset_after.append(row)
   elif event in ("return","error"):
    method=pending.pop(seq,None)
    if method is None:bad.append("orphan_return")
    elif event=="return":returns[method]+=1
    else:errors[method]+=1
 if calls["connect"]!=1 or calls["disconnect"]!=1 or returns["disconnect"]!=1:bad.append("connect_disconnect_mismatch")
 if calls["setCollisionFilterPair"]!=14:bad.append("filter_count_not_14")
 if scope and calls["stepSimulation"]:bad.append("physics_in_preflight")
 if reset_after:bad.append("reset_after_first_step")
 return dict(present=True,passed=not bad and not pending,calls=dict(calls),returns=dict(returns),errors=dict(errors),
  unknown_calls=pending,filters=filters,first_step_api_line=first,reset_after_first_step=reset_after,violations=bad,
  source=path.relative_to(HERE).as_posix(),sha256=sha(path))
def finish():
 verify()
 session=read(HERE/"session_result.json")
 pre=optional(HERE/"static_preflight/path_result.json")
 roots=optional(HERE/"static_preflight/left_right_contact_entry_roots.json")
 zero=optional(HERE/"static_preflight/zero_readback.json")
 preapi=api_audit("static_preflight");dynapi=api_audit("")
 ledger=read(HERE/"slot_ledger.json")
 ep=optional(HERE/"episodes/14/result.json")
 api_intents=dynapi["calls"].get("stepSimulation",0)
 native_proof=optional(HERE/"episodes/14/native_call_observed.json")
 physical=0 if native_proof is None and api_intents==0 else session.get("physical_step_calls_this_version",ledger.get("physical_step_calls_this_version") if native_proof is not None else None)
 if not isinstance(physical,int) or physical<0 or physical>api_intents:raise ValueError("native invocation evidence missing or inconsistent; preserve dispatch and review")
 charged=physical>0
 if preapi["calls"].get("stepSimulation",0):raise ValueError("forbidden preflight physics")
 if charged:
  if (ledger["starts"],ledger["remaining_total_budget"])!=(9,5) or not ledger["slots"][13]["charged"]:raise ValueError("actual step not charged")
 else:
  if sha(HERE/"slot_ledger.json")!=sha(HERE/"ledger_initial.json"):raise ValueError("no-step ledger altered")
 if any(x["charged"] for x in ledger["slots"][7:12]):raise ValueError("unauthorized original slot")
 count=0;first_failed=None;classes=Counter();complete=False
 snapshots=HERE/"static_preflight/dual_channel_snapshots.jsonl.gz"
 if snapshots.exists():
  with gzip.open(snapshots,"rt",encoding="utf-8") as f:
   for line in f:
    r=json.loads(line);count+=1
    if r["snapshot"]!=count:raise ValueError("noncontinuous snapshot sequence")
    if not r["passed"] and first_failed is None:first_failed=dict(snapshot=count,phase=r["phase"],phase_tick=r["phase_tick"],violations=r["violations"])
    for channel in ("reference_channel","proxy_channel"):
     for c in r[channel]["classifications"]:classes[c["classification"]]+=1
  complete=bool(count==4728 and first_failed is None and pre and pre["passed"] and roots and roots["both_passed"])
 if charged and not complete:raise ValueError("dynamic without complete static")
 result=dict(version=HERE.name,recorded_utc=now(),status=session.get("stop"),
  static_admission_passed=complete,static_snapshot_count=count,static_snapshot_planned=4728,
  static_first_failure=first_failed,static_result=pre,entry_roots=roots,
  zero_checks_count=len(zero["fields"]) if zero else 0,zero_passed=zero.get("passed") if zero else None,
  charged=charged,new_charges=int(charged),slot14_started=charged,
  api_step_intents=api_intents,first_native_call_evidence=native_proof,physical_step_calls=physical,physical_step_returns=dynapi["returns"].get("stepSimulation",0),
  unknown_physical_steps=physical-dynapi["returns"].get("stepSimulation",0),
  cumulative_starts=ledger["starts"],total_budget=14,remaining=ledger["remaining_total_budget"],slots8_to_12_frozen=True,
  episode_result=ep,episode_completed=None if ep is None else ep.get("episode_completed"),
  engineering_gate_passed=None if ep is None else ep.get("engineering_gate_passed"),
  grasp_success=None if ep is None else ep.get("grasp_success"),slot_passed=None if ep is None else ep.get("slot_passed"),
  independent_confirmation=False,algorithm_effects_tested=False,IP140_overall_passed=None,
  budget_sha256=sha(HERE/"slot_ledger.json"),prohibited_followup_started=False)
 write_once(HERE/"result.json",result)
 write_once(HERE/"complete_api_audit.json",dict(version=HERE.name,static=preapi,dynamic=dynapi,
  static_snapshot_classifications=dict(classes),charged=charged,physical_steps=physical,
  caution="API evidence consistency does not imply geometry or grasp success"))
 before=read(HERE/"history_snapshot_before.json");changed=[]
 for rel,row in before["files"].items():
  p=ROOT/rel
  if (sha(p),p.stat().st_size,p.stat().st_mtime_ns)!=(row["sha256"],row["bytes"],row["mtime_ns"]):changed.append(rel)
 write_once(HERE/"history_integrity_after.json",dict(passed=not changed,checked_files=len(before["files"]),changed=changed,utc=now()))
 if changed:raise ValueError("historical files modified")
 write_once(HERE/"STOP.json",dict(version=HERE.name,reason=result["status"],slot14_started=charged,slot8_to_12_started=False,
  new_connections_authorized=False,retry_allowed=False,result_sha256=sha(HERE/"result.json"),utc=now()))
 return result
def archive():
 verify()
 assert (HERE/"审查报告.md").is_file() and (HERE/"result.json").is_file()
 differences=[]
 for p in sorted(HERE.rglob("*.py")):
  a=PREVIOUS/p.relative_to(HERE)
  if a.is_file() and sha(a)==sha(p):continue
  differences.extend(difflib.unified_diff(a.read_text("utf-8-sig").splitlines(True) if a.is_file() else [],
   p.read_text("utf-8-sig").splitlines(True),fromfile=a.relative_to(ROOT).as_posix() if a.is_file() else "/dev/null",tofile=p.relative_to(ROOT).as_posix()))
 with (HERE/"implementation.diff").open("x",encoding="utf-8",newline="\n") as f:f.writelines(differences)
 files=[p for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".zip",".sha256")]
 with (HERE/"DIRECTORY.tsv").open("x",encoding="utf-8",newline="\n") as f:
  f.write("relative_path\tbytes\n")
  for p in files:f.write(p.relative_to(HERE).as_posix()+"\t"+str(p.stat().st_size)+"\n")
 files.append(HERE/"DIRECTORY.tsv")
 sums={p.relative_to(HERE).as_posix():sha(p) for p in files}
 with (HERE/"SHA256SUMS.txt").open("x",encoding="utf-8",newline="\n") as f:
  f.write("SHA256\tbytes\trelative_path\n")
  for p in files:f.write(sums[p.relative_to(HERE).as_posix()]+"\t"+str(p.stat().st_size)+"\t"+p.relative_to(HERE).as_posix()+"\n")
 files.append(HERE/"SHA256SUMS.txt");sums["SHA256SUMS.txt"]=sha(HERE/"SHA256SUMS.txt")
 zpath=HERE/(HERE.name+"_审查包.zip")
 with zipfile.ZipFile(zpath,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in files:z.write(p,p.relative_to(HERE).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in (".gz",".png") else zipfile.ZIP_DEFLATED)
 with zipfile.ZipFile(zpath) as z:
  if set(z.namelist())!=set(sums):raise ValueError("ZIP membership mismatch")
  for rel,d in sums.items():
   with z.open(rel) as f:
    if hashlib.file_digest(f,"sha256").hexdigest()!=d:raise ValueError("ZIP hash "+rel)
 with Path(str(zpath)+".sha256").open("x",encoding="utf-8") as f:f.write(sha(zpath)+"  "+zpath.name+"\n")
 return dict(path=str(zpath),bytes=zpath.stat().st_size,sha256=sha(zpath),files=len(files))
