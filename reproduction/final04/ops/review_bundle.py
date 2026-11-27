"""Create an evidence-only review ZIP, verifying every member at readback. Raw and native data remain local."""
from pathlib import Path
import gzip,hashlib,json,os,subprocess,zipfile
V=Path(__file__).resolve().parents[1];P=V.parent;B=V/'adapter';T=V/'cold';D=V/'delivery';R=T/'p/IP140_R2_R6formal720_static_distance_uncertified_resume01'
def sha(p):
    with p.open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()
def read(p):return json.loads(p.read_bytes())
def put(p,x):
    with p.open('x',encoding='utf8')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def tree(path):
    for dn,ds,fs in os.walk(path):
        root=Path(dn);ds[:]=[d for d in ds if not (root/d).is_junction() and not (root/d).is_symlink() and d!='.git']
        for n in fs:yield root/n
files={};index=[]
def add(p,n=None):
    n=n or p.relative_to(V).as_posix()
    if not p.is_file() or n.startswith('/') or '..' in n.split('/') or ':' in n or (n in files and files[n]!=p):raise ValueError(n)
    files[n]=p
prior='7fcfa642154ab0872df35571a1e049c4c7b2c237'
with (D/'adapter.patch').open('xb')as f:subprocess.run(['git','-C',str(B),'diff','--binary',prior,'HEAD','--'],stdout=f,check=True)
diff=read(V/'audit/final_cold_outcome.json')
for folder in (V/'audit',V/'ops'):
    for p in tree(folder):add(p)
for name in ('preflight.py','driver.py','ranking_fixture.py','cold_stage.py','font_scope.py','data_scope.py','probes.py','amsi_origin.py'):
    add(B/'adapters'/name)
for name in ('cold_preregistration.json','smoke_expected.json','stage_seeds.json','binary_bindings.json'):add(B/'metadata'/name)
for f in (B/'locks').glob('requirements*'):add(f)
for name in ('setup_v1.ps1','verify_v1.ps1','run_smoke_v1.ps1','summarize_v1.ps1','preflight_v1.ps1'):add(B/name)
for name in ('manifest.json','SHA256SUMS.txt'):add(V/'provenance_evidence_appendix'/name)
for f in tree(T/'setup_logs'):
    if f.parent==T/'setup_logs' and f.suffix=='.json':add(f)
add(T/'setup_complete.json')
large_jsons={'structural_filter_readback.json','base_table_mount_registration.json','mountpair_readback.json'}
for root in (V/'gates',T/'cold_receipts',R/'trials',R/'units',R/'run/jobs'):
    if not root.exists():continue
    for p in tree(root):
        name=p.relative_to(V).as_posix();size=p.stat().st_size
        is_cache=any(n in ('mpl','scratch','method_temp','forward_temp','rank_temp','__pycache__')for n in p.relative_to(root).parts)
        if is_cache:continue
        index.append({'path':name,'bytes':size,'sha256':sha(p)})
        if 'cold_receipts/' in name or (p.suffix in ('.json','.jsonl','.log') and size<3_000_000 and p.name not in large_jsons) or (p.name in ('pool.npz','raw_decoder.npy','final_480_steps.jsonl.gz')):
            add(p)
with gzip.open(D/'local_raw_index.jsonl.gz','xt',encoding='utf8')as f:
    for r in sorted(index,key=lambda x:x['path']):f.write(json.dumps(r,ensure_ascii=False)+'\n')
for f in (D/'adapter.patch',D/'local_raw_index.jsonl.gz',D/'V1_final04_report.md'):add(f)
old=P/'V1_repro_dependency_closure03'
for name in ('audit/missing_dependencies.tsv','audit/three_case_gate.json','audit/host_data_dependencies.json','receipts/STOP.json','supply/manifest.json','supply/v1_smoke11_registered_sources.zip','delivery/verification.json'):add(old/name,'history_03/'+name)
for key in ('reports/cold_execution_ledger.json','recovery02_ops/v1_cold_receipt02.json','recovery02_ops/DISPATCH_STOP02.json','v2_control02/config/stage_budget.json','v2_control02/config/development_registry.json'):
    add(P/'IP140_R3_V2_original_title_recovery01'/key,'history_preregistered/'+key)
bundle=D/'V1_repro_final04_review.zip';sums=D/'SHA256SUMS.txt'
with sums.open('x',encoding='utf8',newline='\n') as f:
    for n,p in sorted(files.items()):f.write(sha(p)+'  '+n+'\n')
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for n,p in sorted(files.items()):z.write(p,n)
    z.write(sums,'SHA256SUMS.txt')
with zipfile.ZipFile(bundle)as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(files)+1
    assert z.read('SHA256SUMS.txt')==sums.read_bytes()
    for l in sums.read_text(encoding='utf8').splitlines():
        h,n=l.split('  ',1)
        with z.open(n) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
outer=sha(bundle)
with (D/(bundle.name+'.sha256')).open('x',encoding='utf8',newline='\n')as f:f.write(outer+'  '+bundle.name+'\n')
app=V/'provenance_evidence_appendix/registered_originals.zip'
put(D/'verification.json',dict(archive=bundle.name,sha256=outer,bytes=bundle.stat().st_size,contents_verified=len(files),raw_index_count=len(index),raw_index_bytes=sum(r['bytes']for r in index),provenance_separate_archive=dict(path=app.relative_to(V).as_posix(),sha256=sha(app),bytes=app.stat().st_size),cold_case_acceptances=diff['case_acceptances'],old_tag=read(V/'audit/preservation_closing.json')['tag']))
print(bundle.name,len(files),outer)
