"""P10 fixed held-out evaluation: immutable evidence and finite accounting."""
import os,sys,json,time,uuid,shutil,threading
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,need,now
from scripts.p8_repair_common_v33 import sealed,load,artifact
BASE=ROOT/'results/P10/heldout_v36';LOG=ROOT/'logs/P10/heldout_v36';DATA=ROOT/'data/raw/p10_v36';CONF=ROOT/'configs/p10_heldout_v36.json';ACONF=ROOT/'configs/p10_assets_v36.json'
LOCK=threading.RLock()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');start=time.monotonic()
 while True:
  try:os.replace(tmp,p);return
  except PermissionError as e:
   if getattr(e,'winerror',None) not in (5,32) or time.monotonic()-start>=2:raise
   time.sleep(.05)
def event(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with LOCK,(LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def events():return [json.loads(s) for s in (LOG/'calls.jsonl').read_text().splitlines()] if (LOG/'calls.jsonl').exists() else []
def counts():
 rows=events();out={}
 for k in read(CONF)['call_limits']:
  a=[r['call_id'] for r in rows if r['kind']==k+'_intent'];z=[r['call_id'] for r in rows if r['kind']==k+'_returned'];need(len(a)==len(set(a)) and len(z)==len(set(z)) and set(z)<=set(a),'Invalid call ledger');out[k]=dict(attempts=len(a),known_returns=len(z),unknown=sorted(set(a)-set(z)))
 return out
_counts=None
def invoke(kind,fn,**kw):
 global _counts
 if _counts is None:_counts={k:v['attempts'] for k,v in counts().items()}
 need(_counts[kind]<read(CONF)['call_limits'][kind],'P10 call budget '+kind);cid=uuid.uuid4().hex;event(kind+'_intent',call_id=cid,**kw);_counts[kind]+=1;t=time.monotonic();v=fn();event(kind+'_returned',call_id=cid,seconds=time.monotonic()-t);return v,cid
def verify():
 c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['sha256'],'P10 protocol changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Frozen P10 source changed '+p)
 return c
def parents(c):
 for p,s in read(ROOT/c['parent_binding']).items():need(stat(ROOT/p)==s,'Historical artifact changed '+p)
def status(phase,**kw):atomic(ROOT/'runtime/pipeline/p10_part_v36.json',dict(at=now(),phase=phase,all_tasks_percent=72,experiment_percent=90.,stop_after='P10',P11_started=False,recurring_automation=False,**kw))
def cases(c,scene=None):return [dict(split=split,scene=s,frame=64,condition=d) for split,scenes in c['groups'].items() for s in scenes if scene is None or s==scene for d in c['conditions']]
def key(r):return f"scene_{r['scene']:04d}_0064_{r['condition']}"
def folder(r):return BASE/'cases'/key(r)
def payload(p,**kw):
 import numpy as np
 p=Path(p);need(not p.exists(),'Unreceipted output; preserve '+str(p));p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.'+uuid.uuid4().hex+'.npz');np.savez_compressed(t,**kw);os.replace(t,p);sealed(p.with_suffix('.json'),dict(artifact=stat(p)))
def environment(gpu=False):
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
 os.environ['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
 for p in [ROOT,ROOT/'runtime/build/graspnetAPI_eval_v7']:sys.path.insert(0,str(p))
 p=str(ROOT/'runtime/envs/grasp_eval/Lib/site-packages')
 if p not in sys.path:sys.path.append(p)
class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.start=time.monotonic();self.prior=sum(read(p)['seconds'] for p in LOG.glob('usage_'+phase+'_*.json'));self.path=LOG/('usage_'+phase+'_'+uuid.uuid4().hex+'.json');self.last_save=-1e30;self.last_space=-1e30
 def save(self,state):atomic(self.path,dict(at=now(),seconds=time.monotonic()-self.start,state=state));self.last_save=time.monotonic()
 def guard(self):
  from scripts.p3_train_v27 import resource_guard
  elapsed=time.monotonic()-self.start;need(self.prior+elapsed<self.c['phase_seconds'][self.phase],'Cumulative phase limit '+self.phase);resource_guard(self.c)
  if time.monotonic()-self.last_save>2:self.save('running')
  if time.monotonic()-self.last_space>30:need(sum(p.stat().st_size for base in [BASE,LOG] for p in base.rglob('*') if p.is_file())<self.c['limits']['output_bytes'],'Output limit');self.last_space=time.monotonic()
def asset_inputs():return load(BASE/'assets.json')['files']
def verify_case_inputs(c,r):
 refs=asset_inputs()
 for tail in ['rgb/0064.png','depth/0064.png','meta/0064.mat']:
  p=DATA/f"scenes/scene_{r['scene']:04d}/realsense"/tail;need(stat(p)==refs[p.relative_to(ROOT).as_posix()],'Input changed '+str(p))
