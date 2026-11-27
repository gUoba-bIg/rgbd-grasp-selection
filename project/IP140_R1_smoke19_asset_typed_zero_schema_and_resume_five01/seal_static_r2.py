"""Seal static-only outcomes and archive evidence, without an engine import."""
from pathlib import Path
import json,hashlib,zipfile,difflib
from static_identity import HERE,ROOT,OLD,sha,read,write_once,verify,now
def finish():
    session=read(HERE/"session_result.json")
    zero=read(HERE/"zero_readback.json") if (HERE/"zero_readback.json").exists() else {}
    path=read(HERE/"path_result.json") if (HERE/"path_result.json").exists() else {}
    api=read(HERE/"full_api/summary.json") if (HERE/"full_api/summary.json").exists() else {}
    if api.get("calls",{}).get("stepSimulation",0):raise ValueError("Unauthorized physics needs explicit reconciliation")
    if sha(HERE/"slot_ledger.json")!=sha(OLD/"slot_ledger.json"):raise ValueError("ledger was changed")
    successful=bool(path.get("passed") and path.get("snapshots_completed")==4728 and zero.get("passed") and session.get("path_passed"))
    if successful and session.get("stop")!="target-independent path passed; loaded target contact dynamics deferred":
        raise ValueError("static status mismatch")
    result=dict(version=HERE.name,static_admission_passed=successful,
        status="target-independent path passed; loaded target contact dynamics deferred" if successful else session.get("stop"),
        physical_steps=0,new_charges=0,charged=False,slot14_started=False,cumulative_starts=8,total_budget=14,remaining=6,
        slots8_to_12_frozen=True,dynamic_execution_authorized=False,grasp_success=None,slot_passed=None,
        loaded_path_passed=None,IP140_overall_passed=None,
        zero_passed=zero.get("passed"),snapshots_completed=path.get("snapshots_completed",0),
        first_failure=path.get("first_failure"),exceptions=session.get("exception") or path.get("exception") or zero.get("exception"),
        connect_calls=api.get("calls",{}).get("connect",0),disconnect_returns=api.get("returns",{}).get("disconnect",0),
        filter_call_count=api.get("calls",{}).get("setCollisionFilterPair",0),
        ledger_sha256=sha(HERE/"slot_ledger.json"),utc=now())
    write_once(HERE/"result.json",result)
    write_once(HERE/"STOP.json",dict(version=HERE.name,utc=now(),reason=result["status"],physical_steps=0,new_charges=0,
        slot14_started=False,slots8_to_12_started=False,requires_new_review=True,result_sha256=sha(HERE/"result.json")))
    write_once(HERE/"ledger_unchanged_proof.json",dict(original_source=(OLD/"slot_ledger.json").relative_to(ROOT).as_posix(),
        original_sha256=sha(OLD/"slot_ledger.json"),copy_sha256=sha(HERE/"slot_ledger.json"),byte_identical=True,
        cumulative_starts=8,total_budget=14,remaining=6,slot14_uncharged=True))
    before=read(HERE/"history_snapshot_before.json");bad=[];after={}
    for rel,row in before["files"].items():
        p=ROOT/rel;v=dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns);after[rel]=v
        if v!=row:bad.append(rel)
    write_once(HERE/"history_integrity_after.json",dict(passed=not bad,checked_files=len(after),mismatches=bad,files=after,utc=now()))
    verify()
    return result
def archive():
    verify()
    if not read(HERE/"static_api_audit_r3.json").get("passed"):raise ValueError("final API audit not passed")
    if not (HERE/"审查报告.md").is_file():raise ValueError("report required")
    diffs=[]
    for p in sorted(HERE.glob("*.py")):
        before=OLD/p.name
        lines=before.read_text("utf-8-sig").splitlines(True) if before.exists() else []
        diffs.extend(difflib.unified_diff(lines,p.read_text("utf-8-sig").splitlines(True),
            fromfile=before.relative_to(ROOT).as_posix() if before.exists() else "/dev/null",tofile=p.relative_to(ROOT).as_posix()))
    with (HERE/"implementation.diff").open("x",encoding="utf-8",newline="\n") as f:f.writelines(diffs)
    files=[p for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".zip",".sha256")]
    with (HERE/"DIRECTORY.tsv").open("x",encoding="utf-8",newline="\n") as f:
        f.write("relative_path\tbytes\n")
        for p in files:f.write(p.relative_to(HERE).as_posix()+"\t"+str(p.stat().st_size)+"\n")
    files.append(HERE/"DIRECTORY.tsv")
    hashes={p.relative_to(HERE).as_posix():sha(p) for p in files}
    checksum=HERE/"SHA256SUMS.txt"
    with checksum.open("x",encoding="utf-8",newline="\n") as f:
        f.write("SHA256\tbytes\trelative_path\n")
        for p in files:f.write(hashes[p.relative_to(HERE).as_posix()]+"\t"+str(p.stat().st_size)+"\t"+p.relative_to(HERE).as_posix()+"\n")
    files.append(checksum);hashes[checksum.name]=sha(checksum)
    zpath=HERE/(HERE.name+"_审查包.zip")
    with zipfile.ZipFile(zpath,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,p.relative_to(HERE).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in (".gz",".png") else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(zpath) as z:
        if len(z.namelist())!=len(files) or set(z.namelist())!=set(hashes):raise ValueError("ZIP membership")
        for rel,d in hashes.items():
            with z.open(rel) as f:
                if hashlib.file_digest(f,"sha256").hexdigest()!=d:raise ValueError("ZIP SHA mismatch:"+rel)
    with Path(str(zpath)+".sha256").open("x",encoding="utf-8") as f:f.write(sha(zpath)+"  "+zpath.name+"\n")
    return dict(zip=zpath.name,sha256=sha(zpath),bytes=zpath.stat().st_size,files=len(files))
