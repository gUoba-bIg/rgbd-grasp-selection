"""One finite P6 scientific repair; old v24 remains immutable."""
import json,sys,time,uuid,os,shutil
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,need,sha,stat,atomic,once,now,name,array_sha
BASE=ROOT/'results/P6/error_mixture_v25';LOG=ROOT/'logs/P6/error_mixture_v25';CONFIG=ROOT/'configs/p6_error_mixture_v25.json'

def verify():
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'v25 config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Protected file changed '+p)
 return c

def event(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,at=now(),**kw))+'\n');f.flush();os.fsync(f.fileno())

def events():return [json.loads(s) for s in (LOG/'calls.jsonl').read_text().splitlines()] if (LOG/'calls.jsonl').exists() else []
def counts():
 rows=events();out={}
 for kind in ['quality_forward','segmentation_forward','corruption_replay','error_model_fit','mixture_prediction']:
  aa=[x['call_id'] for x in rows if x['kind']==kind+'_intent'];bb=[x['call_id'] for x in rows if x['kind']==kind+'_returned'];need(len(set(aa))==len(aa) and len(set(bb))==len(bb) and set(bb)<=set(aa),'Invalid call ledger');out[kind]=dict(intents=len(aa),known_returns=len(bb),unknown=len(set(aa)-set(bb)))
 return out
def invoke(kind,fn,**kw):
 c=read(CONFIG);need(counts()[kind]['intents']<c['limits'][kind],'Call limit '+kind);uid=uuid.uuid4().hex;event(kind+'_intent',call_id=uid,**kw);result=fn();event(kind+'_returned',call_id=uid,**kw);return result
def usage():return sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'))
def storage():return sum(p.stat().st_size for d in [BASE,LOG] for p in d.rglob('*') if p.is_file())
def status(phase,**kw):atomic(ROOT/'runtime/pipeline/p6_part_v25.json',dict(at=now(),phase=phase,all_tasks_percent=41,experiment_percent=51.25,**kw))
class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.start=time.monotonic();self.previous=usage();self.file=LOG/('usage_'+uuid.uuid4().hex+'.json');self.save('running')
 def save(self,state):atomic(self.file,dict(at=now(),phase=self.phase,seconds=time.monotonic()-self.start,state=state))
 def guard(self):
  self.save('running');need(self.previous+time.monotonic()-self.start<self.c['limits']['seconds'],'Time budget exhausted');need(storage()<self.c['limits']['bytes'],'Output budget exhausted');need(storage()+self.c['previous_P6_bytes']<6*1024**3,'Cumulative P6 bytes exceeded');need(shutil.disk_usage(ROOT).free>60*1024**3,'Disk reserve')
  import ctypes
  class C(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['peak','working','a','b','c','d','e','f']]
  v=C();v.cb=ctypes.sizeof(v);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p;ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(C),ctypes.c_ulong]
  need(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(v),v.cb)!=0,'Memory query failed');need(v.peak<6*1024**3,'Memory budget')
def save_sample(p,a,meta):
 import numpy as np
 need(not p.exists(),'Uncommitted sample '+str(p));p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**a);once(p.with_suffix('.json'),dict(meta,cache_sha256=sha(p)))
def load_sample(p):
 import numpy as np
 row=read(p.with_suffix('.json'));need(sha(p)==row['cache_sha256'],'Sample changed')
 with np.load(p,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
 return a,row
