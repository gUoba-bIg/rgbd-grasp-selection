"""Final independent accounting, all-file integrity, and real cold re-entry for P4."""
import os,sys,json,hashlib,subprocess,time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
BASE=ROOT/'results/P4/full_baseline_v22r2';LOG=ROOT/'logs/P4/full_baseline_v22r2/final_readonly'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stat(p):
    s=p.stat()
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=h,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def accounting(p):
    events=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    i=[e.get('call_id',e.get('id')) for e in events if e['kind']=='forward_intent'];r=[e.get('call_id',e.get('id')) for e in events if e['kind']=='forward_returned']
    assert len(i)==len(set(i)) and len(r)==len(set(r)) and set(r)<=set(i)
    return dict(forward_intents=len(i),known_forward_returns=len(r),unknown_forward_outcomes=len(set(i)-set(r)),fit_optimizer_returns=sum(e['kind']=='optimizer_returned' and e.get('phase')=='fit' for e in events),research_optimizer_returns=sum(e['kind']=='optimizer_returned' and e.get('phase')=='research' for e in events))
def main():
    LOG.mkdir(parents=True,exist_ok=True);assert not (LOG/'result.json').exists()
    final=read(BASE/'closeout.json');assert final['usable'] and final['all_tasks_percent']==29 and final['experiment_percent']==36.25
    c=read(ROOT/'configs/p4_full_baseline_v22r2.json');cfg=ROOT/'configs/p4_full_baseline_v22r2.json'
    assert stat(cfg)['sha256']==read(ROOT/'logs/P4/full_baseline_v22r2/freeze.json')['protocol_sha256']
    files={ROOT/p for p in c['protected']}|{p for p in BASE.rglob('*') if p.is_file()}|{cfg}
    versions=['full_baseline_v22','full_baseline_v22r1','full_baseline_v22r2'];counts={v:accounting(ROOT/f'logs/P4/{v}/calls.jsonl') for v in versions};counts['representative_link']=accounting(ROOT/'logs/P4/full_baseline_v22r2/representative_link/calls.jsonl')
    for v,a in counts.items():assert all(a[k]==final['accounting_by_version'][v][k] for k in a)
    total={k:sum(a[k] for a in counts.values()) for k in next(iter(counts.values()))};assert all(final['total_new_P4_accounting'][k]==v for k,v in total.items())
    assert total['research_optimizer_returns']==6400 and total['fit_optimizer_returns']==500 and total['forward_intents']<=35000
    exact=total['forward_intents'] if total['unknown_forward_outcomes']==0 else None;assert exact==final['total_new_P4_accounting']['actual_forward_calls_exact']
    ledgers=[ROOT/f'logs/P4/{v}/calls.jsonl' for v in versions]+[ROOT/'logs/P4/full_baseline_v22r2/representative_link/calls.jsonl',ROOT/'logs/P4/full_baseline_v22r2/evaluation/official_calls.jsonl'];files.update(ledgers)
    official=[json.loads(s) for s in ledgers[-1].read_text(encoding='utf-8').splitlines()];oi={e['id']:e for e in official if e['kind']=='intent'};orr={e['id'] for e in official if e['kind']=='returned'}
    assert orr<=set(oi) and len(oi)<=24 and len({(e['arm'],e['scope']) for e in oi.values()})==22
    le=[json.loads(s) for s in ledgers[-2].read_text(encoding='utf-8').splitlines()];si={e['id'] for e in le if e['kind']=='sampling_intent'};sr={e['id'] for e in le if e['kind']=='sampling_returned'};assert sr<=si and len(si)<=24
    scopes=read(ROOT/'logs/P4/full_baseline_v22r2/scope_readonly.json');migration=read(ROOT/'logs/P4/full_baseline_v22r2/independent_migration_readonly.json');assert scopes['status']==migration['status']=='passed'
    for p in [ROOT/'logs/P4/full_baseline_v22r2/scope_readonly.json',ROOT/'logs/P4/full_baseline_v22r2/independent_migration_readonly.json']:files.add(p)
    for p,h in final['evidence'].items():assert stat(BASE/p)['sha256']==h
    for p,v in c['protected'].items():assert stat(ROOT/p)==v, 'Frozen historical file changed: '+p
    before={p.relative_to(ROOT).as_posix():stat(p) for p in sorted(files)};write(LOG/'before.json',before)
    start=time.monotonic();command=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B','scripts/p4_full_baseline_v22r2.py','run']
    proc=subprocess.run(command,cwd=ROOT,capture_output=True,env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'),timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    (LOG/'cold.stdout.log').write_bytes(proc.stdout);(LOG/'cold.stderr.log').write_bytes(proc.stderr)
    assert proc.returncode==0 and b'already complete' in proc.stdout,'Cold re-entry failed'
    after={p.relative_to(ROOT).as_posix():stat(p) for p in sorted(files)};assert before==after,'Cold re-entry modified frozen input/result/call ledger'
    size=sum(p.stat().st_size for kind in ['experiments','results/P4','logs/P4'] for v in versions for p in (ROOT/(kind+'/'+('p4_'+v if kind=='experiments' else v))).rglob('*') if p.is_file())
    assert size<16*1024**3
    folders=[ROOT/f'logs/P4/{v}' for v in versions]+[ROOT/'logs/P4/full_baseline_v22r2/evaluation',ROOT/'logs/P4/full_baseline_v22r2/representative_link']
    seconds=sum(read(p).get('seconds',0) for d in folders for p in d.glob('invocation_*.json'));assert seconds<14400
    result=dict(status='passed',at=datetime.now().astimezone().isoformat(),verified_files=len(files),all_bytes_sizes_mtimes_unchanged=True,real_cold_reentry=dict(command=command,exit_code=proc.returncode,seconds=time.monotonic()-start,new_forward=0,new_training=0,new_sampling=0,new_official_calls=0),accounting=total,actual_forward_calls_exact=exact,official=dict(intents=len(oi),returned=len(orr),unknown=len(set(oi)-orr)),sampling=dict(intents=len(si),returned=len(sr),unknown=len(si-sr)),recorded_compute_invocation_seconds=seconds,version_artifact_bytes=size,progress=dict(all_tasks=29,experiments=36.25),original_pipeline_state_unchanged=True,next_part_started=False,automation_created=False)
    write(LOG/'result.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
