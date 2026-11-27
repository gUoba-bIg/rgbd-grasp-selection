"""Final04 provenance and operational supplement. No scientific imports."""
from pathlib import Path, PurePosixPath
import csv, gzip, hashlib, io, json, os, stat, zipfile
V=Path(__file__).resolve().parents[1]
ROOT=V.parent
OLD=ROOT/'V1_repro_dependency_closure03'
T=V/'cold'
A=V/'provenance_evidence_appendix'
def digest(b): return hashlib.sha256(b).hexdigest()
def sha(f):
    with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def read(f):return json.loads(f.read_bytes())
def write(f,data):
    with f.open('x',encoding='utf8',newline='\n') as s:json.dump(data,s,ensure_ascii=False,indent=2,allow_nan=False);s.write('\n')
def safe(rel):
    p=PurePosixPath(rel)
    if p.is_absolute() or '..' in p.parts or ':' in rel or '\\' in rel:raise ValueError(rel)
    return rel
sums={k:v for v,k in (l.split('  ',1) for l in (OLD/'delivery/SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines())}
for r in ('audit/missing_dependencies.json','audit/missing_dependencies.tsv','audit/runtime_file_open_graph.jsonl.gz','supply/manifest.json'):
    assert sha(OLD/r)==sums[r],r
rows=read(OLD/'audit/missing_dependencies.json')['rows']
with (OLD/'audit/missing_dependencies.tsv').open(encoding='utf-8-sig',newline='')as f:t=list(csv.DictReader(f,delimiter='\t'))
assert len(rows)==len(t)==158 and set(r['registered_path']for r in rows)==set(r['registered_path']for r in t)
assert all(r['opened_as_runtime_dependency_in_preflight'] is False for r in rows)
manifest=[]
zpath=A/'registered_originals.zip'
with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for r in sorted(rows,key=lambda x:x['registered_path']):
        rel=safe(r['registered_path']);p=(ROOT/rel).resolve(strict=True)
        assert p==Path(r['historical_file']).resolve() and p.is_relative_to(ROOT) and not (T/'p'/rel).exists()
        b=p.read_bytes();h=digest(b);assert len(b)==r['bytes'] and h==r['historical_sha256']
        prefix=None
        if r['expected_sha256'] is None:
            prefix=read(ROOT/safe(r['caller']))['zero_api_call_prefix'];assert 0<prefix['bytes']<=len(b) and digest(b[:prefix['bytes']])==prefix['sha256']
        else:assert h==r['expected_sha256']
        manifest.append(dict(registered_path=rel,bytes=len(b),whole_sha256=h,registered_whole_sha256=r['expected_sha256'],registered_prefix_verified=prefix,caller=r['caller'],pointer=r['called_at'],historical_mtime_ns=p.stat().st_mtime_ns,three_case_precheck_opens=0))
        z.writestr(rel,b)
with zipfile.ZipFile(zpath) as z:
    assert len(z.namelist())==len(set(z.namelist()))==158
    for r in manifest:assert digest(z.read(r['registered_path']))==r['whole_sha256']
assert sum(r['registered_prefix_verified']is not None for r in manifest)==3
write(A/'manifest.json',dict(files=manifest,total_bytes=sum(r['bytes']for r in manifest),archive_sha256=sha(zpath),classification='provenance only; excluded from scientific deployment'))
with (A/'SHA256SUMS.txt').open('x',encoding='utf8') as f:
    for n in ('registered_originals.zip','manifest.json'):f.write(sha(A/n)+'  '+n+'\n')
for f in A.iterdir(): f.chmod(stat.S_IREAD)
# Only the independently verified four runtime originals go into the clean environment.
m=read(OLD/'supply/manifest.json');vol=OLD/'supply'/Path(m['volume']).name
assert len(m['files'])==4 and sha(vol)==m['sha256']=='8b8d8e721bfaf00c0d83f719053bbb951af1eacdfea512ccacd39621ba5e04bf'
with zipfile.ZipFile(vol)as z:
    for r in m['files']:
        rel=safe(r['registered_path']);assert rel not in {i['registered_path'] for i in manifest}
        b=z.read(rel);out=T/'p'/rel;assert out.resolve().is_relative_to(T/'p') and digest(b)==r['sha256']==sha(ROOT/rel) and len(b)==r['bytes']
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb')as f:f.write(b)
        assert sha(out)==r['sha256'];out.chmod(stat.S_IREAD)
write(V/'audit/runtime_supplement.json',dict(source_volume=str(vol),sha256=m['sha256'],files=m['files'],mounted_provenance_files=0))
print('provenance',len(manifest),sum(r['bytes']for r in manifest),'prefixes',3,'runtime supplement',4)
