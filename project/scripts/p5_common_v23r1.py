"""P5 one authorized strength repair; separate immutable records and global limits."""
import os,json,hashlib,uuid,time,shutil
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/P5/quality_gate_v23r1';LOG=ROOT/'logs/P5/quality_gate_v23r1';RUN=ROOT/'experiments/p5_quality_gate_v23r1';CONFIG=ROOT/'configs/p5_quality_gate_v23r1.json'
OLD_BASE=ROOT/'results/P5/quality_gate_v23';OLD_LOG=ROOT/'logs/P5/quality_gate_v23';OLD_RUN=ROOT/'experiments/p5_quality_gate_v23'
def now():return datetime.now().astimezone().isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
    p=Path(p);s=p.stat();return dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def need(x,m):
    if not x:raise RuntimeError(m)
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def once(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():need(read(p)==x,'Immutable record differs '+str(p));return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def event(kind,**x):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**x),ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def events(prior=False):
    folders=[OLD_LOG,LOG] if prior else [LOG]
    return [json.loads(line) for folder in folders if (folder/'calls.jsonl').exists() for line in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
def identity(r):
    if 'call_id' in r:need('id' not in r or r['id']==r['call_id'],'Conflicting recorded identity');value=r['call_id']
    elif r.get('phase')=='engineering_tests' and r['kind'] in ('forward_intent','forward_returned'):value=r.get('id')
    else:raise RuntimeError('Missing recorded call identity; unknown outcomes cannot be inferred')
    need(isinstance(value,str) and bool(value),'Invalid identity');return value
def counts(prior=False):
    es=events(prior);result={}
    for name in ['forward','optimizer','official_AP']:
        a=[identity(r) for r in es if r['kind']==name+'_intent'];b=[identity(r) for r in es if r['kind']==name+'_returned'];need(len(set(a))==len(a) and len(set(b))==len(b) and set(b)<=set(a),'Invalid ledger '+name);result[name]=dict(intents=len(a),known_returns=len(b),unknown=len(set(a)-set(b)))
    result['optimizer_returns_by_phase']={p:sum(r['kind']=='optimizer_returned' and r.get('phase')==p for r in es) for p in sorted({r.get('phase','unspecified') for r in es if r['kind']=='optimizer_returned'})}
    return result
def usage(prior=False):return sum(read(p)['seconds'] for folder in ([OLD_LOG,LOG] if prior else [LOG]) for p in folder.glob('usage_*.json'))
def output_bytes():return sum(p.stat().st_size for d in [BASE,LOG,RUN,OLD_BASE,OLD_LOG,OLD_RUN] for p in d.rglob('*') if p.is_file())
def status(phase,**x):atomic(ROOT/'runtime/pipeline/p5_part_v23r1.json',dict(at=now(),phase=phase,all_tasks_percent=37,experiment_percent=46.25,**x))
def verify():
    c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['protocol_sha256'],'Repair protocol changed')
    for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Frozen evidence changed '+p)
    return c
def save_research_snapshot(p):
    paths=[x for d in [BASE,RUN] for x in d.rglob('*') if x.is_file()];snapshot={x.relative_to(ROOT).as_posix():stat(x) for x in paths};once(p,snapshot);return snapshot
