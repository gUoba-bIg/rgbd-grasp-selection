"""Source-bound P2/P3 prerequisite acquisition and explicit continuation authority."""
import os,sys,time,json,uuid,shutil,threading
from pathlib import Path
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
LOG=ROOT/'logs/P2P3/prerequisites_v26';BASE=ROOT/'results/P2P3/prerequisites_v26';DATA=ROOT/'data/raw/prerequisites_v26';CONFIG=ROOT/'configs/prerequisites_data_v26.json'
WRITE_LOCK=threading.Lock()

def events(kind,**kw):
 with WRITE_LOCK:
  LOG.mkdir(parents=True,exist_ok=True)
  with (LOG/'acquisition.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush()

def verify():
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'data_freeze.json')['sha256'],'Data protocol changed')
 for p,v in c['protected'].items():need(stat(ROOT/p)==v,'Protected source changed '+p)
 return c

def status(phase,**kw):atomic(ROOT/'runtime/pipeline/prerequisites_v26.json',dict(at=now(),phase=phase,all_tasks_percent=41,experiment_percent=51.25,automatic_P7_authorized_after_gates=True,**kw))

def guard(c,started,extra=0):
 need(time.monotonic()-started<c['limits']['seconds'],'Acquisition time limit reached; resumable stop')
 need(shutil.disk_usage(ROOT).free>=60*1024**3+extra,'Disk reserve reached')
