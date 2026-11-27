"""Initial final04 cold invocation. A successful gate is required and the exclusive reservation is never resumed."""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
V=Path(__file__).resolve().parents[1];B=V/'adapter';T=V/'cold'
def read(f):return json.loads(f.read_bytes())
def sha(f):
    with f.open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()
assert Path(sys.executable).resolve()==T/'bin/python.exe'
g=V/'audit/font_AB_and_rank_gate.json';h=V/'audit/preservation_opening.json';app=V/'provenance_evidence_appendix/manifest.json';sup=V/'audit/runtime_supplement.json'
assert read(V/'audit/pre_cold_storage.json')['passed']
assert read(g)['passed'] and read(h)['passed'] and len(read(sup)['files'])==4
assert all(not (T/'p'/i['registered_path']).exists() for i in read(app)['files'])
assert all(shutil.disk_usage(f'{volume}:/').free>=100*2**30 for volume in ('C','E'))
registry=B/'metadata/cold_preregistration.json';cases=read(registry)['cases']
assert [c['expected_physical_returns']for c in cases]==[4728,0,0] and [c['trial_id']for c in cases]==['R6FORMAL-007','R6FORMAL-004','R6FORMAL-001']
policy={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case_slots':[(c['trial_id'],c['unit'],c['expected_detail'],c['worker_seed'],c['expected_physical_returns'],1)for c in cases],
'P3_call_cap':3,'P5_call_cap':3,'total_native_step_dispatch_cap':4728,'algorithm_retries':0,'physics_retries':0,'packaging_retries':0,'max_workers':1,'stop_at_first_incomplete':True,
'freeze_sources':{str(f.relative_to(V)):sha(f)for f in [g,h,app,sup,registry,B/'metadata/smoke_expected.json',*sorted((B/'adapters').glob('*.py'))]}}
with (V/'audit/final_cold_budget.json').open('x',encoding='utf8')as s:json.dump(policy,s,ensure_ascii=False,indent=2);s.flush();os.fsync(s.fileno())
subprocess.run([str(T/'bin/python.exe'),'-I','-B',str(B/'adapters/driver.py'),str(T)],cwd=V.parent,check=True)
