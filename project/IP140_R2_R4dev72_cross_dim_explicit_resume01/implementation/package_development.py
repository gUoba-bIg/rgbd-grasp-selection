"""Package completed or explicitly stopped R4 artifacts. Verify every archived byte."""
from pathlib import Path
import hashlib
import json
import os
import zipfile

V=Path(__file__).resolve().parents[1]
ROOT=V.parent

def hash_stream(stream):
    return hashlib.file_digest(stream,"sha256").hexdigest()

def sha(p):
    with p.open("rb") as f:return hash_stream(f)

def inventory():
    files=[]
    for f in V.rglob("*"):
        if f.is_symlink():
            raise ValueError("External or symbolic artifact: "+str(f))
        if f.is_file() and "__pycache__" not in f.parts and f.suffix!=".pyc":
            files.append(f)
    return sorted(files,key=lambda f:f.relative_to(V).as_posix())

def new_text(path,value):
    with path.open("x",encoding="utf8",newline="\n") as f:f.write(value)

def main():
    summary=V/"analysis/paired_statistics.json"
    if not summary.is_file():raise RuntimeError("the final stopped/completed analysis is missing")
    counts=json.loads(summary.read_bytes())
    ledger=json.loads((V/"development_ledger.json").read_bytes())
    if counts["counted"]!=ledger["method_trials_counted"]:raise ValueError("ledger changed after statistics")
    output=ROOT/(V.name+"_审查包.zip");pending=ROOT/(V.name+"_审查包.partial.zip")
    outside_digest=ROOT/(output.name+".sha256")
    manifest=V/"SHA256SUMS.txt"
    if any(p.exists() for p in (manifest,output,pending,outside_digest)):
        raise FileExistsError("an earlier package/manifest exists; preserve it")
    files=inventory()
    hashes={p.relative_to(V).as_posix():sha(p) for p in files}
    new_text(manifest,"".join(h+"  "+name+"\n" for name,h in hashes.items()))
    hashes[manifest.name]=sha(manifest)
    with zipfile.ZipFile(pending,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name in sorted(hashes):
            p=V/name
            method=zipfile.ZIP_STORED if p.suffix.lower() in {".npz",".gz",".zip",".jpg",".png",".mp4",".7z"} else zipfile.ZIP_DEFLATED
            z.write(p,arcname=V.name+"/"+name,compress_type=method,compresslevel=1)
    with zipfile.ZipFile(pending,"r") as z:
        members=z.infolist()
        if len(members)!=len(hashes) or len({m.filename for m in members})!=len(hashes):
            raise ValueError("Archive member count or duplicate name")
        for member in members:
            relative=Path(member.filename).relative_to(V.name).as_posix()
            with z.open(member,"r") as f:
                if hash_stream(f)!=hashes[relative]:raise ValueError("ZIP member integrity failure: "+relative)
    if set(hashes)!={p.relative_to(V).as_posix() for p in inventory()}:
        raise ValueError("Files changed during package creation")
    for name,digest in hashes.items():
        if sha(V/name)!=digest:raise ValueError("Source changed during package creation: "+name)
    digest=sha(pending)
    # Windows rename raises if output already exists. No historical deliverable is replaced.
    os.rename(pending,output)
    new_text(outside_digest,digest+"  "+output.name+"\n")
    print(json.dumps(dict(file=output.name,sha256=digest,bytes=output.stat().st_size,members=len(hashes),verified=True),ensure_ascii=False))
if __name__=="__main__":main()

