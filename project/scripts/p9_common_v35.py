"""One finite P9 experiment with separate immutable artifacts and budgets."""
import os,sys,json,time,uuid
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p8_repair_common_v33 import sealed,load,artifact,support,enough
BASE=ROOT/'results/P9/risk_seeds_v35';LOG=ROOT/'logs/P9/risk_seeds_v35';EXP=ROOT/'experiments/risk_seeds_v35';CONF=ROOT/'configs/p9_risk_seeds_v35.json'
P7=ROOT/'results/P7/risk_v31';P8=ROOT/'results/P8/ablation_v32';P34=ROOT/'results/P8/temperature_repair_v34'
def events():return [json.loads(s) for s in (LOG/'calls.jsonl').read_text().splitlines()] if (LOG/'calls.jsonl').exists() else []
def event(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def counts():
 rows=events();out={}
 for k in read(CONF)['call_limits']:
  a=[r['call_id'] for r in rows if r['kind']==k+'_intent'];b=[r['call_id'] for r in rows if r['kind']==k+'_returned'];need(len(a)==len(set(a)) and len(b)==len(set(b)) and set(b)<=set(a),'Invalid P9 call ledger');out[k]=dict(attempts=len(a),known_returns=len(b),unknown=sorted(set(a)-set(b)))
 return out
_counts=None
def invoke(kind,fn,**kw):
 global _counts
 if _counts is None:_counts={k:v['attempts'] for k,v in counts().items()}
 need(_counts[kind]<read(CONF)['call_limits'][kind],'P9 call limit '+kind);cid=uuid.uuid4().hex;event(kind+'_intent',call_id=cid,**kw);_counts[kind]+=1;t=time.monotonic();v=fn();event(kind+'_returned',call_id=cid,seconds=time.monotonic()-t);return v,cid
def verify():
 c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['sha256'],'P9 config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'P9 frozen source changed '+p)
 return c
def parents(c,guard=None):
 for n,(p,s) in enumerate(read(ROOT/c['parent_binding']).items()):
  if guard and n%500==0:guard()
  need(stat(ROOT/p)==s,'Historical artifact changed '+p)
def payload(p,**a):
 import numpy as np
 need(not p.exists(),'Unsealed payload exists; preserve before recovery');p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.'+uuid.uuid4().hex+'.npz');np.savez_compressed(temp,**a);os.replace(temp,p);sealed(p.with_suffix('.json'),dict(artifact=stat(p)))
def status(phase,**kw):atomic(ROOT/'runtime/pipeline/p9_part_v35.json',dict(at=now(),phase=phase,all_tasks_percent=64,experiment_percent=80.,stop_after='P9',P10_started=False,recurring_automation=False,**kw))
def pool(split):
 import numpy as np
 with np.load(artifact(P7/'predictions'/f'{split}.npz')) as a:d={k:a[k].copy() for k in a.files}
 d['spans']=load(P7/'predictions'/f'{split}.json')['spans'];d['scenes']=np.concatenate([np.full(r['end']-r['start'],r['scene'],np.int64) for r in d['spans']]);need(len(d['scenes'])==len(d['labels']),'Pool spans differ');return d
def cases(c):return [dict(split='validation',scene=s,frame=64,condition=cond) for s in c['groups']['validation'] for cond in c['conditions']]
def key(r):return f"scene_{r['scene']:04d}_0064_{r['condition']}"
def seedpath(seed):return BASE/'seeds'/str(seed)
class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.start=time.monotonic();self.prior=sum(read(p)['seconds'] for p in LOG.glob('usage_'+phase+'_*.json'));self.path=LOG/('usage_'+phase+'_'+uuid.uuid4().hex+'.json');self.last=-1e30
 def save(self,state):atomic(self.path,dict(at=now(),seconds=time.monotonic()-self.start,state=state))
 def guard(self):
  from scripts.p3_train_v27 import resource_guard
  self.save('running');need(self.prior+time.monotonic()-self.start<self.c['phase_seconds'][self.phase],'P9 cumulative phase time limit '+self.phase);resource_guard(self.c)
  if time.monotonic()-self.last>20:need(sum(p.stat().st_size for d in [BASE,LOG,EXP] for p in d.rglob('*') if p.is_file())<self.c['limits']['output_bytes'],'P9 output limit');self.last=time.monotonic()
