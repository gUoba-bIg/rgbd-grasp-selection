"""P8 finite ablations: immutable artifacts and actual invocation accounting."""
import os,sys,json,time,uuid
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
BASE=ROOT/'results/P8/ablation_v32';LOG=ROOT/'logs/P8/ablation_v32';CONFIG=ROOT/'configs/p8_ablation_v32.json'
OLD=ROOT/'results/P7/risk_v31';_counts=None
def event(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def events():return [json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else []
def counts():
 e=events();out={}
 for k in read(CONFIG)['call_limits']:
  a=[r['call_id'] for r in e if r['kind']==k+'_intent'];b=[r['call_id'] for r in e if r['kind']==k+'_returned'];need(len(set(a))==len(a) and len(set(b))==len(b) and set(b)<=set(a),'Invalid P8 ledger')
  out[k]=dict(intents=len(a),known_returns=len(b),unknown_ids=sorted(set(a)-set(b)))
 return out
def invoke(kind,fn,**kw):
 global _counts
 if _counts is None:_counts={k:v['intents'] for k,v in counts().items()}
 need(_counts[kind]<read(CONFIG)['call_limits'][kind],'P8 invocation limit '+kind);cid=uuid.uuid4().hex;event(kind+'_intent',call_id=cid,**kw);_counts[kind]+=1;t=time.monotonic();value=fn();event(kind+'_returned',call_id=cid,seconds=time.monotonic()-t);return value,cid
def sealed(p,x):once(p,x);once(p.with_suffix(p.suffix+'.receipt.json'),stat(p))
def load(p):need(stat(p)==read(p.with_suffix(p.suffix+'.receipt.json')),'Sealed artifact changed '+str(p));return read(p)
def payload(p,**arrays):
 import numpy as np
 need(not p.exists(),'Unsealed existing payload; preserve and recover explicitly '+str(p));p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays);sealed(p.with_suffix('.json'),dict(artifact=stat(p)))
def artifact(p):need(stat(p)==load(p.with_suffix('.json'))['artifact'],'Payload changed '+str(p));return p
def verify():
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'P8 config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'P8 frozen dependency changed '+p)
 return c
def cases(c):return [dict(split=sp,scene=s,condition=cond,frame=64) for sp in ['calibration','validation'] for s in c['groups'][sp] for cond in c['conditions']]
def key(r):return f"scene_{r['scene']:04d}_0064_{r['condition']}"
def oldfolder(r):return OLD/'cases'/r['split']/key(r)
def folder(r):return BASE/'cases'/r['split']/key(r)
def oldobservations(r):
 p=oldfolder(r)/'observations.npz';need(stat(p)==load(oldfolder(r)/'generation.json')['artifact'],'P7 source changed');return p
def status(phase,**kw):atomic(ROOT/'runtime/pipeline/p8_part_v32.json',dict(at=now(),phase=phase,all_tasks_percent=56,experiment_percent=70.,stop_after='P8',P9_started=False,recurring_automation=False,**kw))
class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.start=time.monotonic();self.prior=sum(read(p)['seconds'] for p in LOG.glob('usage_'+phase+'_*.json'));self.path=LOG/('usage_'+phase+'_'+uuid.uuid4().hex+'.json');self.last=-1e30
 def save(self,state):atomic(self.path,dict(at=now(),seconds=time.monotonic()-self.start,state=state))
 def guard(self):
  from scripts.p3_train_v27 import resource_guard
  self.save('running');need(self.prior+time.monotonic()-self.start<self.c['phase_seconds'][self.phase],'P8 cumulative wall limit '+self.phase);resource_guard(self.c)
  if time.monotonic()-self.last>30:
   total=sum(p.stat().st_size for d in [BASE,LOG,ROOT/'experiments/p8_ablation_v32'] for p in d.rglob('*') if p.is_file());need(total+512*1024**2<self.c['limits']['output_bytes'],'P8 output limit');self.last=time.monotonic()
