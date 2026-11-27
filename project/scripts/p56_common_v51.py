"""Isolated v51 joint-use validation utilities; historical files remain read-only."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,uuid,shutil
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
LOG=ROOT/'logs/joint_validation_v51'
OUT=ROOT/'results/joint_validation_v51'
CONFIG=ROOT/'configs/p56_joint_v51.json'
STATE=ROOT/'runtime/pipeline/p56_joint_v51.json'
PROTOCOL=ROOT/'docs/P5-P6联合候选选择验证协议_v51.txt'
REPORT=ROOT/'docs/P5-P6联合候选选择验证结果_v51.txt'
SCENES=(2,6,7,11,13,15,18,22,33,39)
CONDITIONS=('clean','noise5','missing15','shift3','joint')
GROUPS=('A','B','C','D','R','C0','D0')
ORIGINAL_STATE='b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'

def now():return datetime.now().astimezone().isoformat()
def need(v,m):
    if not bool(v):raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def encode(v):
    if hasattr(v,'tolist'):return v.tolist()
    if isinstance(v,Path):return rel(v)
    raise TypeError(type(v).__name__)
def identity(p):
    p=Path(p);st=p.stat()
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=h,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def atomic(p,v):
    p=Path(p);need(p.resolve().is_relative_to(ROOT),'Output outside project')
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False,default=encode);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def once(p,v):
    p=Path(p);need(not p.exists(),'Existing immutable v51 artifact: '+str(p));atomic(p,v)
def append(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(v,ensure_ascii=False,allow_nan=False,default=encode)+'\n');f.flush();os.fsync(f.fileno())
def events(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()] if Path(p).exists() else []
def ledger(p):
    starts={};ends={}
    for r in events(p):
        need(r['event'] in ('intent','returned'),'Invalid ledger event')
        target=starts if r['event']=='intent' else ends
        need(r['id'] not in target,'Duplicate ledger ID');target[r['id']]=r
    need(set(ends)<=set(starts),'Return without intent')
    for k in ends:need(ends[k]['kind']==starts[k]['kind'],'Ledger kind differs')
    return dict(intents=len(starts),known_returns=len(ends),unknown=sorted(set(starts)-set(ends)),
                returned_by_kind={k:sum(v['kind']==k for v in ends.values()) for k in sorted({v['kind'] for v in starts.values()})})
def status(phase,**kw):
    atomic(STATE,dict(at=now(),phase=phase,current_part='P5_P6_joint_use_v51',
       all_tasks_percent=45,experiment_percent=56.25,P5_original_scientific_passed=False,
       P6_original_scientific_passed=False,expansion_started=False,recurring_automation=False,**kw))
def case_key(scene,condition):return f'scene_{scene:04d}_0064_{condition}'
def case_definitions():
    return [dict(scene=s,frame=64,condition=c,source_dir=f'results/P7/risk_v31/cases/validation/{case_key(s,c)}',
        ranking_file=f'results/joint_validation_v51/cases/{case_key(s,c)}/ranking.json') for s in SCENES for c in CONDITIONS]
@contextmanager
def own_lock():
    import msvcrt
    path=ROOT/'runtime/pipeline/p56_joint_v51.lock'
    with path.open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
def verify(full=False):
    need(Path.cwd().resolve()==ROOT,'Explicit project working directory required')
    c=read(CONFIG);need(identity(CONFIG)==read(LOG/'freeze.json')['config'],'Frozen configuration changed')
    for name,expected in c['protected'].items():
        p=ROOT/name;st=p.stat()
        need(st.st_size==expected['bytes'] and st.st_mtime_ns==expected['mtime_ns'],'Protected metadata changed '+name)
        if full:need(identity(p)==expected,'Protected bytes changed '+name)
    need(identity(ROOT/'runtime/pipeline/state.json')['sha256']==ORIGINAL_STATE,'Historical state changed')
    return c
def child_guard():
    s=read(STATE)
    need(bool(s.get('token')) and os.environ.get('P56_V51_TOKEN')==s['token'],'Missing controller token')
    need(time.time()<s['deadline_epoch'],'Cumulative compute wall cap reached')
    need(not (LOG/'STOP').exists(),'STOP requested')
def handoff(message):
    p=ROOT/'docs/实验自动续接说明.txt';folder=LOG/'handoff_history';folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,folder/(uuid.uuid4().hex+'_'+p.name))
    previous=p.read_text(encoding='utf-8-sig')
    p.write_text('最新：'+now()+' '+message+'\n\n历史：\n'+previous,encoding='utf-8')
def output_bytes():return sum(p.stat().st_size for d in (LOG,OUT) if d.exists() for p in d.rglob('*') if p.is_file())
