"""Local audit records and finite budgets for P5 v23."""
import os,json,hashlib,uuid,time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/P5/quality_gate_v23';LOG=ROOT/'logs/P5/quality_gate_v23';RUN=ROOT/'experiments/p5_quality_gate_v23';CONFIG=ROOT/'configs/p5_quality_gate_v23.json'
def now():return datetime.now().astimezone().isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
    p=Path(p);s=p.stat();return dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def need(v,m):
    if not v:raise RuntimeError(m)
def once(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():need(read(p)==x,'Immutable record differs: '+str(p));return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def event(kind,**x):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**x),ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def events():return [json.loads(x) for x in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else []
def status(phase,**x):
    row=dict(at=now(),phase=phase,all_tasks_percent=29,experiment_percent=36.25,**x);p=ROOT/'runtime/pipeline/p5_part_v23.json';temp=p.with_suffix('.'+uuid.uuid4().hex+'.tmp');temp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,p)
def verify():
    c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['protocol_sha256'],'P5 config changed')
    for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Frozen source/input/evidence changed: '+p)
    return c
