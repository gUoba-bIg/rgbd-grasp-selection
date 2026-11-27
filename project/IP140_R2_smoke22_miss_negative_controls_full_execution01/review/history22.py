"""Seal original files before execution and independently recheck them after the batch."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone

V = Path(__file__).resolve().parents[1]
ROOT = V.parent
P = ROOT / "IP140_R2_smoke21_registered_pad_surface_and_resume_four01"
P_HASH = "7281a32135c26377a3dc15a0785d4ebcd4edde51dc26ccffa9d7a3a7024a4f7a"

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def load(path):
    return json.loads(path.read_bytes())

def save(path,data):
    with path.open("x",encoding="utf8") as out:
        json.dump(data,out,ensure_ascii=False,indent=2,allow_nan=False)

def snapshot(dirs):
    files={}
    for name in dirs:
        for path in sorted((ROOT/name).rglob("*")):
            if path.is_file():
                st=path.stat()
                files[path.relative_to(ROOT).as_posix()]=dict(sha256=sha(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return files

def archive_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def before():
    parentzip=P/(P.name+"_审查包.zip")
    if sha(parentzip)!=P_HASH:
        raise ValueError("smoke21 published ZIP digest mismatch")
    hist=load(P/"history_snapshot_before.json")
    dirs=hist["directories"]+[P.name]
    files=snapshot(dirs)
    changed=[r for r,x in hist["files"].items() if files.get(r)!=x]
    if changed:
        raise ValueError("historical file changed: "+str(changed))
    parent_manifest=(P/"SHA256SUMS.txt").read_text("utf8").splitlines()[1:]
    parent_entries={}
    for line in parent_manifest:
        h,size,n=line.split("\t",2)
        entry=files[(P/Path(n)).relative_to(ROOT).as_posix()]
        assert entry["sha256"]==h and entry["bytes"]==int(size),n
        parent_entries[n]=h
    parent_entries["SHA256SUMS.txt"]=sha(P/"SHA256SUMS.txt")
    with zipfile.ZipFile(parentzip) as z:
        assert len(z.namelist())==len(parent_entries) and set(z.namelist())==set(parent_entries)
        for name,expected in parent_entries.items():
            with z.open(name) as inzip:
                assert hashlib.file_digest(inzip,"sha256").hexdigest()==expected,name
    clock=datetime.now(timezone.utc).isoformat()
    save(V/"history_snapshot_before.json",dict(version=V.name,utc=clock,directories=dirs,files=files))
    base=load(P/"inherited_byte_identities.json")["files"]
    # Three authorized slot inputs, the frozen physical and evaluation modules and historical regressions.
    keep={n for n in base if not n.startswith("slots/") or n.startswith(("slots/10/","slots/11/","slots/12/"))}
    keep.update(["contact_semantics.py","contact_entry.py","registered_pad_surface21.py","pad_surface.py",
                 "offline_replay.py","offline_checks.py","episode_log.py","dynamic_api.py","aggregate_support_replay.py",
                 "asset_typed_schema.json","acquisition_policy.json","base_table_registration.json","mountpair_registration.json",
                 "surface_replay21.json","scope_replay20.json","positive_negative_replay.json","schema_replay19.json",
                 "static_gate_semantics.json"])
    keep.update(q.relative_to(V).as_posix() for q in (V/"static_precheck").rglob("*") if q.is_file())
    verified={}
    for name in sorted(keep):
        local_hash=sha(V/name)
        assert parent_entries[name]==sha(P/name)==local_hash,name
        verified[name]=local_hash
    save(V/"inherited_byte_identities.json",dict(version=V.name,parent=P.name,parent_zip_sha256=P_HASH,
         verified_parent_archive_members=len(parent_entries),files=verified,historical_files=len(files),
         inherited_offline_receipts=["surface_replay21.json","scope_replay20.json","positive_negative_replay.json","schema_replay19.json"]))
    print(len(files),len(verified),len(parent_entries))

def after():
    frozen=load(V/"history_snapshot_before.json")
    end=snapshot(frozen["directories"])
    original=frozen["files"]
    differences=[k for k in sorted(end.keys()|original.keys()) if original.get(k)!=end.get(k)]
    evidence=dict(checked_files=len(original),checked_directories=len(frozen["directories"]),differences=differences,
        passed=not differences,source_snapshot_sha256=sha(V/"history_snapshot_before.json"),utc=datetime.now(timezone.utc).isoformat())
    save(V/"history_integrity_after.json",evidence)
    print(evidence)
    if differences:raise ValueError("Historical files changed")

if __name__=="__main__":
    if Path.cwd().resolve()!=ROOT:raise RuntimeError("Expected project root")
    {"before":before,"after":after}[sys.argv[1]]()
