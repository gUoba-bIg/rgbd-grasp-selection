"""Finite P7 observable-risk pipeline; scientific gates remain separate."""
import os,sys,time,json,uuid,shutil
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
CONFIG=ROOT/'configs/risk_calibration_v31.json';LOG=ROOT/'logs/P7/risk_v31';BASE=ROOT/'results/P7/risk_v31'
_CALL_COUNTS=None

def event(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())

def invoke(kind,fn,**kw):
 global _CALL_COUNTS
 if _CALL_COUNTS is None:
  _CALL_COUNTS={k:0 for k in read(CONFIG)['call_limits']};events=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else []
  for r in events:
   if r['kind'].endswith('_intent'):_CALL_COUNTS[r['kind'][:-7]]+=1
 c=read(CONFIG);need(_CALL_COUNTS[kind]<c['call_limits'][kind],'P7 actual invocation budget '+kind)
 cid=uuid.uuid4().hex;event(kind+'_intent',call_id=cid,**kw);_CALL_COUNTS[kind]+=1;t=time.monotonic();value=fn();event(kind+'_returned',call_id=cid,seconds=time.monotonic()-t);return value,cid

def sealed(path,value):once(path,value);once(path.with_suffix(path.suffix+'.receipt.json'),stat(path))
def load_sealed(path):need(stat(path)==read(path.with_suffix(path.suffix+'.receipt.json')),'P7 sealed artifact changed');return read(path)
def verify():
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'P7 config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'P7 protected source changed '+p)
 need(read(ROOT/'results/P3/candidate_v30/closeout.json')['candidate_usable_for_P7'],'P3 baseline not usable')
 return c

def verify_case_inputs(c,case):
 key=f"scenes/scene_{case['scene']:04d}/realsense/";receipts=c['image_inputs']
 for tail in ['rgb/0064.png','depth/0064.png','meta/0064.mat']:
  need(stat(ROOT/c['data_root']/(key+tail))==receipts[key+tail],'P7 raw input changed '+key+tail)

def cases(c):
 return [dict(split=split,scene=s,frame=64,condition=d) for split in ['train','calibration','validation'] for s in c['groups'][split] for d in c['conditions']]
def folder(case):return BASE/'cases'/case['split']/f"scene_{case['scene']:04d}_0064_{case['condition']}"
def status(phase,**kw):atomic(ROOT/'runtime/pipeline/risk_v31.json',dict(at=now(),phase=phase,all_tasks_percent=47,experiment_percent=58.75,**kw))

class Budget:
 def __init__(self,c,phase):self.c=c;self.phase=phase;self.start=time.monotonic();self.prior=sum(read(p)['seconds'] for p in LOG.glob('usage_'+phase+'_*.json'));self.path=LOG/('usage_'+phase+'_'+uuid.uuid4().hex+'.json');self.storage_checked=-1e30
 def save(self,state):atomic(self.path,dict(at=now(),seconds=time.monotonic()-self.start,state=state))
 def guard(self):
  from scripts.p3_train_v27 import resource_guard
  self.save('running');need(self.prior+time.monotonic()-self.start<self.c['phase_seconds'][self.phase],'P7 cumulative '+self.phase+' time exhausted');resource_guard(self.c)
  if time.monotonic()-self.storage_checked>=30:
   total=sum(p.stat().st_size for d in [BASE,LOG,ROOT/'experiments/risk_v31'] for p in d.rglob('*') if p.is_file());need(total+512*1024**2<self.c['limits']['output_bytes'],'P7 output budget/reserve reached');self.storage_checked=time.monotonic()
