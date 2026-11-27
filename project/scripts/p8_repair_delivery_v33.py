"""Single finite P8 repair chain. Stops after closeout and never starts P9."""
import os,sys,time,uuid,subprocess,traceback,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_repair_common_v33 import *
def log(kind,**kw):
 with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def run():
 c=verify();old=[json.loads(s) for s in (LOG/'controller.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'controller.jsonl').exists() else [];unknown={r['call_id'] for r in old if r['kind']=='child_intent'}-{r['call_id'] for r in old if r['kind']=='child_returned'};need(not unknown,'Unknown previous child return; inspect orphan process before recovery');prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));start=time.monotonic();up=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
 def guard():atomic(up,dict(at=now(),seconds=time.monotonic()-start));need(prior+time.monotonic()-start<1800,'Repair controller cumulative wall limit')
 for name,script,args,output in [('calibration','scripts/p8_repair_run_v33.py',[],BASE/'result.json'),('full_readonly','scripts/p8_repair_review_v33.py',[],LOG/'readonly.json'),('cold_reentry','scripts/p8_repair_review_v33.py',['--cold'],LOG/'cold.json'),('finish','scripts/p8_repair_finish_v33.py',[],BASE/'closeout.json')]:
  if output.exists():load(output);continue
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B',script,*args]
  from scripts.pipeline import project_lock
  with project_lock(ROOT):pass
  log('child_intent',call_id=cid,phase=name,command=cmd)
  with (LOG/f'{name}_{cid}.stdout.log').open('xb') as out,(LOG/f'{name}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);atomic(ROOT/'runtime/pipeline/p8_repair_delivery_v33.json',dict(at=now(),phase=name,controller_pid=os.getpid(),child_pid=proc.pid,all_tasks_percent=64,experiment_percent=80.,P9_started=False))
   try:
    while proc.poll() is None:guard();time.sleep(2)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
   code=proc.returncode
  log('child_returned',call_id=cid,phase=name,exit_code=code);need(code==0,'Repair stage failed; evidence retained: '+name)
 r=load(BASE/'closeout.json');guard();log('completed_stopped',scientific_usable=r['scientific_usable']);atomic(ROOT/'runtime/pipeline/p8_repair_delivery_v33.json',dict(at=now(),phase='P8_repair_completed_stopped',scientific_usable=r['scientific_usable'],all_tasks_percent=64,experiment_percent=80.,P9_started=False))
if __name__=='__main__':
 import msvcrt
 LOG.mkdir(parents=True,exist_ok=True)
 with (ROOT/'runtime/pipeline/p8_repair_v33.lock').open('a+b') as lock:
  lock.seek(0,2)
  if lock.tell()==0:lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  try:run()
  except Exception as e:
   log('failure_preserved',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e));atomic(ROOT/'runtime/pipeline/p8_repair_delivery_v33.json',dict(at=now(),phase='engineering_failure_preserved',error=repr(e),all_tasks_percent=64,experiment_percent=80.,P9_started=False));raise
  finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
