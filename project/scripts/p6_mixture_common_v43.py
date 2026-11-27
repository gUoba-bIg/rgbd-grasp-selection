"""Shared finite P6 v43 bookkeeping. No previous experiment is mutated."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="2"
import sys,json,time,hashlib,uuid,shutil
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
LOG=ROOT/"logs/P6/mixture_v43";OUT=ROOT/"results/P6/mixture_v43"
CONF=ROOT/"configs/p6_mixture_v43.json"
STATE=ROOT/"runtime/pipeline/p6_mixture_v43.json"
REPORT=ROOT/"docs/P6共享对应与混合不确定性结果_v43.txt"

def now():return datetime.now().astimezone().isoformat()
def need(value,message):
    if not bool(value):raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def identity(p):
    p=Path(p);s=p.stat()
    return dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def relative(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+"."+uuid.uuid4().hex+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    os.replace(tmp,p)
def once(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
def append(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())
def events():
    p=LOG/"calls.jsonl"
    return [json.loads(v) for v in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
def ledger():
    starts={};returns={}
    for e in events():
        d=starts if e["event"]=="intent" else returns
        need(e["id"] not in d,"Duplicate call id")
        d[e["id"]]=e
    need(set(returns)<=set(starts),"Return without intent")
    for k,e in returns.items():need(e["kind"]==starts[k]["kind"] and e["key"]==starts[k]["key"],"Changed call identity")
    return dict(intents=len(starts),known_returns=len(returns),unknown=sorted(set(starts)-set(returns)),
                counts={k:sum(e["kind"]==k for e in returns.values()) for k in sorted({e["kind"] for e in starts.values()})})
CALL_LIMITS={"feature_generate":2160,"feature_replay":2160,"warp_fit":1,"depth_fit":1,"evaluation":864,"evaluation_replay":864}
_CALL_STARTED=None
_CALL_KEYS=None
def call(kind,key,fn):
    global _CALL_STARTED,_CALL_KEYS
    if _CALL_STARTED is None:
        history=events();need(not ledger()["unknown"],"Unknown prior calls require audit")
        _CALL_STARTED={k:sum(e["event"]=="intent" and e["kind"]==k for e in history) for k in CALL_LIMITS}
        _CALL_KEYS={(e["kind"],e["key"]) for e in history if e["event"]=="intent"}
    need(kind in CALL_LIMITS and _CALL_STARTED[kind]<CALL_LIMITS[kind],"Call budget exceeded "+kind)
    need((kind,key) not in _CALL_KEYS,"Duplicate research call "+kind+"/"+key)
    _CALL_STARTED[kind]+=1;_CALL_KEYS.add((kind,key))
    cid=uuid.uuid4().hex
    append(LOG/"calls.jsonl",dict(at=now(),event="intent",id=cid,kind=kind,key=key))
    value=fn()
    append(LOG/"calls.jsonl",dict(at=now(),event="returned",id=cid,kind=kind,key=key))
    return value
def archive(p,tag):
    p=Path(p)
    if p.exists():
        d=LOG/"history";d.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,d/(tag+"_"+uuid.uuid4().hex+"_"+p.name))
def status(phase,**values):
    x=dict(at=now(),phase=phase,current_part="P6_mixture_v43",all_tasks_percent=45,experiment_percent=56.25,
        P6_full_passed=False,P5_scientific_adoption=False,P10_still_stopped=True,recurring_automation=False)
    x.update(values);atomic(STATE,x)
def supervise(phase,**values):
    p=ROOT/"runtime/pipeline/part_level_supervision.json";archive(p,phase)
    x=read(p);x.update(recorded_at=now(),current_authorized_part="P6_mixture_v43",current_execution_state=phase,
        all_tasks_percent_planning_estimate=45,experiment_percent_planning_estimate=56.25,
        current_runner="scripts/p6_mixture_delivery_v43.py",stop_after="P6_mixture_v43",stop_after_part="P6_mixture_v43",
        automatic_scheduler_recreated=False,automatic_resume_authorized=False,
        user_authorization="2026-11-01用户授权方法设计、内部检查、后台运行P6对应实验，整部分收尾停止")
    x.update(values);atomic(p,x)
def handoff(message,tag):
    p=ROOT/"docs/实验自动续接说明.txt";archive(p,tag)
    p.write_text("最新："+now()+" "+message+"\n\n历史：\n"+p.read_text(encoding="utf-8-sig"),encoding="utf-8")
def verify():
    need(Path.cwd().resolve()==ROOT,"Explicit project working directory required")
    c=read(CONF);need(sha(CONF)==read(LOG/"freeze.json")["config_sha256"],"Frozen config changed")
    for name,value in c["protected"].items():need(identity(ROOT/name)==value,"Frozen dependency changed: "+name)
    need(c["protected"]["runtime/pipeline/state.json"]["sha256"]=="b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76","Historical state changed")
    return c
def peak_rss():
    from scripts.p6_correspondence_v40 import peak_rss as f
    return f()
def storage():return sum(p.stat().st_size for folder in (LOG,OUT) if folder.exists() for p in folder.rglob("*") if p.is_file())
class Guard:
    def __init__(self,stage):
        self.stage=stage;self.start=time.monotonic();self.last=-1e6;self.maxrss=0
        self.path=LOG/("usage_"+stage+"_"+uuid.uuid4().hex+".json")
        self.sample(True)
    def sample(self,force=False):
        elapsed=time.monotonic()-self.start
        if not force and elapsed-self.last<2:return
        self.last=elapsed;self.maxrss=max(self.maxrss,peak_rss());size=storage()
        atomic(self.path,dict(at=now(),stage=self.stage,seconds=elapsed,peak_rss=self.maxrss,output_bytes=size))
        need(elapsed<3600,"Stage exceeded total job budget")
        need(self.maxrss<3*1024**3,"3GiB process peak RSS limit")
        need(size<512*1024**2,"512MiB total output limit")
        need(not (LOG/"STOP").exists(),"User stop marker")
    def __call__(self):self.sample()
    def close(self):self.sample(True)

