"""One finite P8 controller. No recurring schedule, no P9 command."""
import os,sys,time,uuid,subprocess,traceback,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_common_v32 import *
def controller_event(kind,**kw):
 with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def run():
 c=verify();old=[json.loads(s) for s in (LOG/'controller.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'controller.jsonl').exists() else [];unknown={r['call_id'] for r in old if r['kind']=='child_intent'}-{r['call_id'] for r in old if r['kind']=='child_returned'};need(not unknown,'Unresolved previous child; inspect orphan processes first');start=time.monotonic();prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
 def guard():atomic(usage,dict(seconds=time.monotonic()-start,at=now()));need(prior+time.monotonic()-start<c['wall_seconds'],'Cumulative P8 controller wall budget')
 def command(name,env,script,args):
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script,*args]
  from scripts.pipeline import project_lock
  with project_lock(ROOT):pass
  controller_event('child_intent',call_id=cid,phase=name,command=cmd)
  with (LOG/f'{name}_{cid}.stdout.log').open('xb') as out,(LOG/f'{name}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);atomic(ROOT/'runtime/pipeline/p8_delivery_v32.json',dict(at=now(),phase=name,controller_pid=os.getpid(),child_pid=proc.pid,call_id=cid,all_tasks_percent=56,experiment_percent=70.,stop_after='P8',P9_started=False))
   try:
    while proc.poll() is None:guard();time.sleep(5)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
   code=proc.returncode
  controller_event('child_returned',call_id=cid,phase=name,exit_code=code);need(code==0,'P8 stage failed; preserve evidence: '+name)
 steps=[('ablation_fit','vision','scripts/p8_fit_v32.py',[],BASE/'fitting_complete.json'),('foreground_generate','vision','scripts/p8_generate_v32.py',[],BASE/'generation_complete.json'),('features_select','grasp_eval','scripts/p8_geometry_v32.py',['prepare'],BASE/'selection_complete.json'),('official_scoring','grasp_eval','scripts/p8_geometry_v32.py',['score'],BASE/'scoring_complete.json'),('mask_AP','p4_eval_v22','scripts/p8_mask_v32.py',[],BASE/'mask_summary.json'),('calibration_metrics','vision','scripts/p8_metrics_v32.py',[],BASE/'metrics.json'),('mask_readonly','p4_eval_v22','scripts/p8_mask_v32.py',['--review'],LOG/'mask_readonly.json'),('full_readonly','grasp_eval','scripts/p8_review_v32.py',[],LOG/'readonly.json'),('cold_reentry','vision','scripts/p8_review_v32.py',['--cold'],LOG/'cold.json'),('finish','vision','scripts/p8_finish_v32.py',[],BASE/'closeout.json')]
 for name,env,script,args,done in steps:
  if done.exists():load(done);continue
  command(name,env,script,args)
 result=load(BASE/'closeout.json');guard();atomic(ROOT/'runtime/pipeline/p8_delivery_v32.json',dict(at=now(),phase='P8_completed_stopped',all_tasks_percent=64,experiment_percent=80.,scientific_usable=result['scientific_usable'],P9_started=False));controller_event('completed_stopped',scientific_usable=result['scientific_usable'])
if __name__=='__main__':
 import msvcrt
 LOG.mkdir(parents=True,exist_ok=True)
 with (ROOT/'runtime/pipeline/p8_delivery_v32.lock').open('a+b') as lock:
  lock.seek(0,2)
  if lock.tell()==0:lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  try:run()
  except Exception as exc:
   controller_event('failure_preserved',error=repr(exc),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(exc));atomic(ROOT/'runtime/pipeline/p8_delivery_v32.json',dict(at=now(),phase='engineering_failure_preserved',error=repr(exc),all_tasks_percent=56,experiment_percent=70.,P9_started=False));raise
  finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
