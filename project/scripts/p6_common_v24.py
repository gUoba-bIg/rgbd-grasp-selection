"""Finite P6 development stage: audit, protection, actual invocation accounting."""
import os,json,hashlib,uuid,time,shutil
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/P6/image_geometry_v24';LOG=ROOT/'logs/P6/image_geometry_v24';CONFIG=ROOT/'configs/p6_image_geometry_v24.json';OLD=ROOT/'results/P5/quality_gate_v23'
def now():return datetime.now().astimezone().isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(x,m):
 if not x:raise RuntimeError(m)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
 p=Path(p);s=p.stat();return dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def once(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():need(read(p)==x,'Immutable record differs '+str(p));return
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def event(kind,**x):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,at=now(),**x))+'\n');f.flush();os.fsync(f.fileno())
def events():return [json.loads(x) for x in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else []
def counts():
 e=events();result={}
 for k in ['forward','corruption_replay']:
  a=[r['call_id'] for r in e if r['kind']==k+'_intent'];b=[r['call_id'] for r in e if r['kind']==k+'_returned'];need(len(a)==len(set(a)) and len(b)==len(set(b)) and set(b)<=set(a),'Invalid ledger');result[k]=dict(intents=len(a),known_returns=len(b),unknown=len(set(a)-set(b)))
 return result
def usage():return sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'))
def storage():return sum(p.stat().st_size for d in [BASE,LOG] for p in d.rglob('*') if p.is_file())
def verify():
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['protocol_sha256'],'P6 config changed')
 for p,v in c['protected'].items():need(stat(ROOT/p)==v,'P6 frozen source changed '+p)
 return c
def status(phase,**x):atomic(ROOT/'runtime/pipeline/p6_part_v24.json',dict(at=now(),phase=phase,all_tasks_percent=37,experiment_percent=46.25,**x))
def name(r,condition):return f"{condition}_scene_{r['scene']:04d}_{r['frame']}"
def case_order(c):
 monitor={(r['scene'],r['frame']) for r in c['monitor_records']}
 return [(r,d) for early in [True,False] for d in c['core_conditions'] for r in c['development_records'] if ((r['scene'],r['frame']) in monitor)==early]
class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.previous=usage();self.start=time.monotonic();self.path=LOG/('usage_'+uuid.uuid4().hex+'.json');self.save('running')
 def save(self,state):atomic(self.path,dict(seconds=time.monotonic()-self.start,phase=self.phase,state=state,at=now()))
 def guard(self):
  self.save('running');r=self.c['resources'];need(self.previous+time.monotonic()-self.start<r['seconds'],'P6 time budget exhausted');need(storage()<r['new_output_bytes'],'P6 output budget');need(storage()+self.c['old_P6_output_bytes']<r['cumulative_output_bytes'],'P6 cumulative output budget');need(shutil.disk_usage(ROOT).free>=r['disk_reserve_bytes'],'P6 disk reserve')
  import ctypes
  class Counters(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
  m=Counters();m.cb=ctypes.sizeof(m);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p;ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong];need(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(m),m.cb)!=0,'Memory measurement failed');need(m.PeakWorkingSetSize<r['rss_bytes'],'P6 memory limit')
 def intent(self,kind,**x):
  n=sum(r['kind']==kind+'_intent' for r in events());need(n<self.c['resources'][kind+'_limit'],'P6 call budget '+kind);cid=uuid.uuid4().hex;event(kind+'_intent',call_id=cid,phase=self.phase,**x);return cid
 def returned(self,kind,cid,**x):event(kind+'_returned',call_id=cid,phase=self.phase,**x)
