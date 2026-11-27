import gzip,hashlib,io,json,shutil,sys,time
from pathlib import Path
import psutil
import codec
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
R4=ROOT/"IP140_R2_R4dev72_cross_dim_explicit_resume01"
def dig(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    records=[];tmp=HERE/"run"/"codec_calibration";tmp.mkdir(exist_ok=False)
    paths=[*(R4/"trials").glob("*/static_preflight/dual_channel_snapshots.jsonl.gz"),*(R4/"trials").glob("*/static_api/calls.jsonl.gz"),
           *(R4/"trials").glob("*/episode/telemetry.jsonl.gz"),*(R4/"trials").glob("*/dynamic_api/calls.jsonl.gz")]
    lookup={}
    for p in paths:lookup.setdefault((p.parent.name,p.name),[]).append(p)
    for (kind,name),files in lookup.items():
        files=sorted(files,key=lambda f:f.stat().st_size);p=files[len(files)//2];records_group=[]
        for after in (0,480):
            path=tmp/(kind+"_"+str(after)+".zst");raw_digest=hashlib.sha256();ordinary=io.BytesIO();n=0
            with gzip.open(p,"rb") as src,codec.open(path,"xb") as out,gzip.GzipFile(fileobj=ordinary,mode="wb",compresslevel=1,mtime=0) as comparison:
                for _ in range(after):src.readline()
                for line in src:
                    out.write(line);comparison.write(line);raw_digest.update(line);n+=len(line)
                    if kind=="static_preflight":
                        out.flush();comparison.flush()
                    if n>=24*2**20:break
            with codec.open(path) as check:
                if hashlib.file_digest(check,"sha256").digest()!=raw_digest.digest():raise ValueError("codec verification")
            records_group.append(dict(skip_rows=after,plain_bytes=n,gzip_bytes=len(ordinary.getbuffer()),zstd_bytes=path.stat().st_size,plain_sha256=raw_digest.hexdigest()))
        records.append(dict(category=kind,file=p.relative_to(ROOT).as_posix(),legacy_total=sum(q.stat().st_size for q in files),samples=records_group))
    usage=shutil.disk_usage(ROOT);ram=psutil.virtual_memory()
    old=sum(p.stat().st_size for p in R4.rglob("*") if p.is_file())
    # The stream classes above account for most bytes. Other archives are carried forward at their old size.
    predicted=10*(old-sum(g["legacy_total"] for g in records)+sum(g["legacy_total"]*max(s["zstd_bytes"]/s["gzip_bytes"] for s in g["samples"]) for g in records))
    policy=dict(free_bytes=usage.free,ram_total=ram.total,ram_available=ram.available,cores=psutil.cpu_count(logical=False),logical_cpus=psutil.cpu_count(),
        cpu_workers=min(4,max(1,psutil.cpu_count(logical=False)-2),max(1,(ram.available-4*2**30)//(2*2**30))),gpu_inference_workers=1,
        estimated_bytes_legacy_case_mix=round(predicted),guard_factor=1.35,space_reserve_bytes=12*2**30,
        estimate_basis="Cold paired-case historical size, measured lossless codec on archival rows only, conditional on observed type mix; actual disk checked per job. Raw/uncompressed content and statuses are preserved.",
        codec_library_sha256=dig(codec.DLL),codec_library=str(codec.DLL),source_sha256=dig(Path(codec.__file__)),samples=records,physics_steps=0)
    policy["admitted"]=predicted*1.35+policy["space_reserve_bytes"] < usage.free
    with (HERE/"audit/resource_preflight.json").open("x",encoding="utf8") as f:json.dump(policy,f,indent=2)
    print(policy["admitted"],policy["cpu_workers"],round(predicted/2**30,2),round(usage.free/2**30,2),flush=True)
if __name__=="__main__":main()
