"""One finite continuation: existing P3 training, whole P3 acceptance, then P7 and stop."""
import os,sys,json,time,uuid,subprocess,traceback,ctypes,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
LOG=ROOT/'logs/P3/delivery_v31';STATE=ROOT/'runtime/pipeline/p3_to_p7_v31.json';CONF=ROOT/'configs/p3_to_p7_v31.json'

def event(kind,**kw):
 with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())

def progress():
 for p in ['results/P7/risk_v31/closeout.json','results/P3/candidate_v30/closeout.json']:
  if (ROOT/p).exists():r=read(ROOT/p);return {k:r[k] for k in ['all_tasks_percent','experiment_percent']}
 return dict(all_tasks_percent=43,experiment_percent=53.75)

def state(phase,**kw):atomic(STATE,dict(at=now(),phase=phase,**progress(),P8_allowed=False,recurring_automation=False,**kw))

def record_stop(reason):
 current=progress();hand=ROOT/'docs/实验自动续接说明.txt';stamp=uuid.uuid4().hex
 shutil.copy2(hand,LOG/('handoff_before_stop_'+stamp+'.txt'))
 hand.write_text('最新：'+now()+' 有限P3修复→P7链路已停止：'+reason+'。全部任务'+str(current['all_tasks_percent'])+'%、实验'+str(current['experiment_percent'])+'%。旧失败、调用与检查点保留，P8/P9及周期任务未启动。状态runtime/pipeline/p3_to_p7_v31.json。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 once(LOG/('stop_'+stamp+'.json'),dict(at=now(),reason=reason,**current))

def wait_existing(c,guard):
 k=ctypes.windll.kernel32;k.OpenProcess.restype=ctypes.c_void_p;k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];k.CloseHandle.argtypes=[ctypes.c_void_p];k.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_ulonglong)]*4;k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
 for item in c['existing_training_processes']:
  h=k.OpenProcess(0x100000|0x1000,False,item['pid'])
  if not h:continue
  try:
   values=[ctypes.c_ulonglong() for _ in range(4)];need(k.GetProcessTimes(h,*[ctypes.byref(v) for v in values]),'Cannot verify process creation identity');need(values[0].value==item['creation_filetime'],'Recorded training PID was reused; audit required')
   state('waiting_existing_P3_remaining_training',existing_training_pid=item['pid'],P7_started=False)
   while k.WaitForSingleObject(h,0)==258:guard();time.sleep(5)
   code=ctypes.c_ulong();need(k.GetExitCodeProcess(h,ctypes.byref(code)),'Cannot read existing process exit');event('existing_training_observed_exit',pid=item['pid'],exit_code=code.value);need(code.value==0,'Existing P3 training failed; checkpoints preserved, no duplicate restart')
  finally:k.CloseHandle(h)
 need((ROOT/'logs/P3/candidate_v30/epoch_18.json').exists(),'P3 remaining training incomplete; no duplicate restart')

def run():
 c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['sha256'],'Finite delivery config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Finite delivery source changed '+p)
 prior_events=[json.loads(s) for s in (LOG/'controller.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'controller.jsonl').exists() else []
 unknown={r['call_id'] for r in prior_events if r['kind']=='child_intent'}-{r['call_id'] for r in prior_events if r['kind']=='child_returned'}
 need(not unknown,'Previous child return unknown; inspect processes before manual recovery')
 start=time.monotonic()
 def guard():need(time.monotonic()-start<c['wall_seconds'],'Finite chain wall-time budget exhausted')
 def command(name,env,script,*args):
  guard();state(name);cid=uuid.uuid4().hex;cmd=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script,*args]
  from scripts.pipeline import project_lock
  with project_lock(ROOT):pass
  event('child_intent',call_id=cid,name=name,command=cmd)
  with (LOG/(name+'_'+cid+'.stdout.log')).open('xb') as out,(LOG/(name+'_'+cid+'.stderr.log')).open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);state(name,child_pid=proc.pid,call_id=cid)
   try:
    while proc.poll() is None:guard();time.sleep(5)
   except BaseException:
    # Kill only this owned child tree on a controller failure; keep all intent/return evidence.
    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
   code=proc.returncode
  event('child_returned',call_id=cid,name=name,exit_code=code);need(code==0,'Stage failed; retained logs/checkpoints: '+name)
 wait_existing(c,guard)
 if not (ROOT/'logs/P3/candidate_v30/final_readonly.json').exists():
  command('P3_final_generate','vision','scripts/p3_generate_v30.py','--stage','final')
  command('P3_final_official_score','grasp_eval','scripts/p3_score_v30.py','--stage','final')
  command('P3_final_readonly','grasp_eval','scripts/p3_review_v30.py','--stage','final')
 if not (ROOT/'logs/P3/candidate_v30/final_cold.json').exists():command('P3_final_cold','grasp_eval','scripts/p3_review_v30.py','--stage','final','--cold')
 if not (ROOT/'results/P3/candidate_v30/closeout.json').exists():command('P3_final_closeout','vision','scripts/p3_finish_v30.py')
 result=read(ROOT/'results/P3/candidate_v30/closeout.json');need(result['engineering_passed'],'P3 closeout incomplete')
 if not result['candidate_usable_for_P7']:
  state('P3_completed_scientific_gate_failed_stopped',P7_started=False);record_stop('P3固定范围已验收，但候选科学可用性未达门槛；P7未启动，保留负结果待具体修复决策');return
 command('P7_prepare','vision','scripts/p7_prepare_v31.py')
 for name,env,script,artifact,args in [
  ('P7_generate','vision','scripts/p7_generate_v31.py','results/P7/risk_v31/generation_complete.json',[]),
  ('P7_score','grasp_eval','scripts/p7_score_v31.py','results/P7/risk_v31/scoring_complete.json',[]),
  ('P7_fit','vision','scripts/p7_fit_v31.py','results/P7/risk_v31/fit_result.json',[]),
  ('P7_model_identity','vision','scripts/p7_model_identity_v31.py','results/P7/risk_v31/model_identity.json',[]),
  ('P7_full_review','grasp_eval','scripts/p7_review_v31.py','logs/P7/risk_v31/readonly.json',[]),
  ('P7_cold','grasp_eval','scripts/p7_review_v31.py','logs/P7/risk_v31/cold.json',['--cold']),
  ('P7_finish','vision','scripts/p7_finish_v31.py','results/P7/risk_v31/closeout.json',[])]:
  if not (ROOT/artifact).exists():command(name,env,script,*args)
 result=read(ROOT/'results/P7/risk_v31/closeout.json');state('P7_completed_stopped',result='results/P7/risk_v31/closeout.json',scientific_usable=result['scientific_usable'],P8_started=False);event('delivery_complete_stopped',scientific_usable=result['scientific_usable'])

if __name__=='__main__':
 import msvcrt
 LOG.mkdir(parents=True,exist_ok=True)
 with (ROOT/'runtime/pipeline/p3_to_p7_v31.lock').open('a+b') as lock:
  lock.seek(0,2)
  if lock.tell()==0:lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  try:run()
  except Exception as exc:event('delivery_failure',error=repr(exc),traceback=traceback.format_exc());state('engineering_failure_preserved',error=repr(exc));record_stop('工程入口失败：'+repr(exc));raise
  finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
