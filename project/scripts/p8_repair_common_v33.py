"""P8 repair ledger, fixed resource limits and frozen-parent reuse."""
import os,sys,json,time,uuid,shutil
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
BASE=ROOT/'results/P8/whole_cloud_repair_v33';LOG=ROOT/'logs/P8/whole_cloud_repair_v33';CONF=ROOT/'configs/p8_whole_cloud_repair_v33.json'
P7=ROOT/'results/P7/risk_v31';P8=ROOT/'results/P8/ablation_v32'
def sealed(p,x):once(p,x);once(p.with_suffix(p.suffix+'.receipt.json'),stat(p))
def load(p):need(stat(p)==read(p.with_suffix(p.suffix+'.receipt.json')),'Immutable artifact changed '+str(p));return read(p)
def artifact(p):need(stat(p)==load(p.with_suffix('.json'))['artifact'],'Immutable array changed '+str(p));return p
def event(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def events():return [json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else []
def counts():
 rows=events();out={}
 for k in read(CONF)['call_limits']:
  a=[r['call_id'] for r in rows if r['kind']==k+'_intent'];b=[r['call_id'] for r in rows if r['kind']==k+'_returned'];need(len(set(a))==len(a) and len(set(b))==len(b) and set(b)<=set(a),'Invalid repair ledger');out[k]=dict(attempts=len(a),known_returns=len(b),unknown=sorted(set(a)-set(b)))
 return out
_counts=None
def invoke(kind,fn,**kw):
 global _counts
 if _counts is None:_counts={k:v['attempts'] for k,v in counts().items()}
 need(_counts[kind]<read(CONF)['call_limits'][kind],'Repair invocation cap '+kind);cid=uuid.uuid4().hex;event(kind+'_intent',call_id=cid,**kw);_counts[kind]+=1;t=time.monotonic();v=fn();event(kind+'_returned',call_id=cid,seconds=time.monotonic()-t);return v,cid
def verify():
 c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['sha256'],'Repair config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Repair dependency changed '+p)
 return c
def verify_parents(c,guard=None):
 for n,(p,s) in enumerate(read(ROOT/c['parent_binding']).items()):
  if guard and n%500==0:guard()
  need(stat(ROOT/p)==s,'Historical P7/P8 material changed '+p)
def status(phase,**kw):atomic(ROOT/'runtime/pipeline/p8_repair_v33.json',dict(at=now(),phase=phase,all_tasks_percent=64,experiment_percent=80.,P9_started=False,recurring_automation=False,stop_after='P8_repair',**kw))
class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.start=time.monotonic();self.previous=sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'));self.path=LOG/('usage_'+phase+'_'+uuid.uuid4().hex+'.json');self.last=-1e30
 def save(self,state):atomic(self.path,dict(at=now(),phase=self.phase,seconds=time.monotonic()-self.start,state=state))
 def guard(self):
  from scripts.p3_train_v27 import resource_guard
  self.save('running');need(self.previous+time.monotonic()-self.start<self.c['resources']['seconds'],'P8 repair cumulative wall budget');resource_guard({'limits':self.c['resources']})
  if time.monotonic()-self.last>10:
   size=sum(p.stat().st_size for d in [BASE,LOG] for p in d.rglob('*') if p.is_file());need(size<self.c['resources']['output_bytes'],'Repair output limit');self.last=time.monotonic()
def dataset(split):
 import numpy as np
 p=P7/'predictions'/f'{split}.npz';meta=load(p.with_suffix('.json'));need(stat(p)==meta['artifact'],'P7 cached predictions changed')
 with np.load(p,allow_pickle=False) as a:y=a['labels'].astype(float);z=a['logits'].astype(float);features=a['features'].copy()
 scene_ids=np.empty(len(y),np.int64);covered=np.zeros(len(y),bool)
 for r in meta['spans']:
  sl=slice(r['start'],r['end']);need(not covered[sl].any(),'Overlapping prediction spans');covered[sl]=True;scene_ids[sl]=r['scene'];d=P7/'cases'/split/f"scene_{r['scene']:04d}_0064_{r['condition']}";labs=load(d/'labels.json');fp=d/'features.npz';fm=load(d/'features.json');need(stat(fp)==fm['artifact'],'P7 feature source changed')
  with np.load(fp) as a:need(a['indices'].tolist()==labs['indices'] and np.array_equal(features[sl],a['features']),'P7 feature/pool mismatch')
  need(np.array_equal(y[sl],[int(x['positive']) for x in labs['records']]),'P7 cached labels differ');need(r['frame']==64,'Unexpected P7 frame')
 need(covered.all() and np.isfinite(z).all() and np.isin(y,[0,1]).all(),'Incomplete prediction dataset')
 return dict(y=y,z=z,scenes=scene_ids,spans=meta['spans'])
def support(y):return dict(n=len(y),positive=int(sum(y)),negative=int(len(y)-sum(y)))
def enough(y):s=support(y);return s['positive']>=30 and s['negative']>=30
