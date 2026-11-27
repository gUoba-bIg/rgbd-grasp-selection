"""Finite temperature-only repair, separate ledger and immutable-parent binding."""
import os,sys,json,time,uuid
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p8_repair_common_v33 import sealed,load,artifact,dataset,support,enough
BASE=ROOT/'results/P8/temperature_repair_v34';LOG=ROOT/'logs/P8/temperature_repair_v34';CONF=ROOT/'configs/p8_temperature_repair_v34.json'
P7=ROOT/'results/P7/risk_v31';P8=ROOT/'results/P8/ablation_v32';P33=ROOT/'results/P8/whole_cloud_repair_v33'
def events():return [json.loads(x) for x in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else []
def event(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def counts():
 rows=events();out={}
 for kind in read(CONF)['call_limits']:
  a=[x['call_id'] for x in rows if x['kind']==kind+'_intent'];b=[x['call_id'] for x in rows if x['kind']==kind+'_returned'];need(len(a)==len(set(a)) and len(b)==len(set(b)) and set(b)<=set(a),'Invalid temperature ledger');out[kind]=dict(attempts=len(a),known_returns=len(b),unknown=sorted(set(a)-set(b)))
 return out
_counts=None
def invoke(kind,fn,**kw):
 global _counts
 if _counts is None:_counts={k:v['attempts'] for k,v in counts().items()}
 need(_counts[kind]<read(CONF)['call_limits'][kind],'Temperature invocation cap '+kind);cid=uuid.uuid4().hex;event(kind+'_intent',call_id=cid,**kw);_counts[kind]+=1;start=time.monotonic();value=fn();event(kind+'_returned',call_id=cid,seconds=time.monotonic()-start);return value,cid
def verify():
 c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['sha256'],'Temperature config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Frozen repair source changed '+p)
 return c
def verify_parents(c,guard=None):
 for n,(p,s) in enumerate(read(ROOT/c['parent_binding']).items()):
  if guard and n%500==0:guard()
  need(stat(ROOT/p)==s,'Historical material changed '+p)
def status(phase,**kw):atomic(ROOT/'runtime/pipeline/p8_temperature_v34.json',dict(at=now(),phase=phase,all_tasks_percent=64,experiment_percent=80.,P9_started=False,recurring_automation=False,stop_after='P8_temperature_repair',**kw))
class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.start=time.monotonic();self.prior=sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'));self.path=LOG/('usage_'+phase+'_'+uuid.uuid4().hex+'.json');self.last=-1e30
 def save(self,state):atomic(self.path,dict(at=now(),phase=self.phase,seconds=time.monotonic()-self.start,state=state))
 def guard(self):
  from scripts.p3_train_v27 import resource_guard
  self.save('running');need(self.prior+time.monotonic()-self.start<900,'Temperature cumulative time limit');resource_guard({'limits':self.c['resources']})
  if time.monotonic()-self.last>10:
   need(sum(p.stat().st_size for d in [BASE,LOG] for p in d.rglob('*') if p.is_file())<self.c['resources']['output_bytes'],'Temperature output limit');self.last=time.monotonic()
