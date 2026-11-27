"""Archive builder only. Source directories are opened read-only."""
from pathlib import Path
import hashlib, json, zipfile, os, gzip, stat
from datetime import datetime, timezone
V=Path(__file__).resolve().parents[2]
REPO=V/'v1_baseline'; ROOT=V.parent; ARCH=V/'v1_archive'
P0='IP140_R2_R6formal720_raw_a_c0_paired01'
R6='IP140_R2_R6formal720_static_distance_uncertified_resume01'
R7='IP140_R2_R7ablation240_mechanism01'
R8='IP140_R2_R8_journal_manuscript01'
R4='IP140_R2_R4dev72_cross_dim_explicit_resume01'
PL='IP140_R2_smoke22_miss_negative_controls_full_execution01'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')
def check(cond,msg):
    if not cond:raise ValueError(str(msg))
def walk(d):
    for dirpath,ds,fs in os.walk(d):
        ds[:]=sorted(x for x in ds if x not in {'.git','__pycache__','.pytest_cache'} and not (Path(dirpath)/x).is_junction())
        for f in sorted(fs):
            p=Path(dirpath)/f
            if not f.startswith(('~$','.~')) and p.suffix.lower() not in ('.pyc','.pyo') and not p.is_symlink():yield p
def manifest(f):
    return {k:h for h,k in (line.split('  ',1) for line in f.read_text('utf-8-sig').splitlines())}
def pack(name,files):
    final=ARCH/(name+'.zip');scratch=ARCH/(name+'.partial')
    idx=REPO/'locks'/(name+'.jsonl.gz')
    check(not any(x.exists() for x in [scratch,final,idx]),name)
    folded=set();n=total=0
    with zipfile.ZipFile(scratch,'x',allowZip64=True) as z, gzip.open(idx,'xt',encoding='utf8',newline='\n') as out:
        for rel,(p,expected) in sorted(files.items()):
            check(not Path(rel).is_absolute() and '..' not in Path(rel).parts and ':' not in rel and rel.casefold() not in folded,rel);folded.add(rel.casefold())
            p=Path(p);size=p.stat().st_size;info=zipfile.ZipInfo(rel,(2026,1,1,0,0,0));info.file_size=size
            info.compress_type=zipfile.ZIP_STORED if size>8000000 or p.suffix.lower() in ('.npz','.gz','.zip','.whl','.pt','.pyd','.zst','.png','.jpg','.pdf','.dll') else zipfile.ZIP_DEFLATED
            h=hashlib.sha256()
            with p.open('rb') as a,z.open(info,'w',force_zip64=True) as b:
                while chunk:=a.read(2**20):
                    h.update(chunk);b.write(chunk)
            dig=h.hexdigest();check(not expected or dig==expected,(rel,expected,dig))
            out.write(json.dumps([rel,size,dig],ensure_ascii=False,separators=(',',':'))+'\n');n+=1;total+=size
    scratch.rename(final)
    d=dict(name=final.name,sha256=sha(final),size=final.stat().st_size,member_manifest=idx.relative_to(REPO).as_posix(),member_manifest_sha256=sha(idx),members=n,unpacked_bytes=total)
    put(REPO/'locks'/(name+'.receipt.json'),d);print(name,n,total,flush=True);return d
