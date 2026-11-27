"""One finite, locked delivery chain. No schedule, P8, or unbounded retries."""
import os,sys,json,time,uuid,subprocess,traceback,ctypes,contextlib,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
LOG=ROOT/'logs/P2P3/delivery_v28';STATE=ROOT/'runtime/pipeline/p23_to_p7_v28.json';CONF=ROOT/'configs/p23_to_p7_v28.json'

def event(kind,**kw):
 with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())

def progress():
 p=ROOT/'results/P2P3/candidate_v27/closeout.json'
 if p.exists():return dict(all_tasks_percent=47,experiment_percent=58.75)
 return dict(all_tasks_percent=43,experiment_percent=53.75)

def state(phase,**kw):atomic(STATE,dict(at=now(),phase=phase,**progress(),P8_allowed=False,recurring_automation=False,**kw))

def record_stop(reason):
 stamp=uuid.uuid4().hex;current=progress();hand=ROOT/'docs/实验自动续接说明.txt'
 if hand.exists():
  shutil.copy2(hand,LOG/('handoff_before_stop_'+stamp+'.txt'));hand.write_text('最新：'+now()+' 有限P2/P3→P7链路已停止：'+reason+'。全部'+str(current['all_tasks_percent'])+'%、实验'+str(current['experiment_percent'])+'%。失败日志、未知调用、检查点保留；不得重复启动或进入P8/P9/周期任务。以runtime/pipeline/p23_to_p7_v28.json及controller.jsonl实际证据为准。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 report=ROOT/'docs/P2P3至P7有限链路停止记录.txt';report.write_text('停止记录 '+now()+'\n'+reason+'\n'+json.dumps(current,ensure_ascii=False)+'\n科学门槛失败：原负结果保留，不继续昂贵下游；后续修复须针对本次训练/候选/标签证据并保持原隔离，另建有预算版本。工程失败：先核对应stderr、未知调用及真实孤儿进程，使用原入口或新修订恢复，禁止手工改complete。\n',encoding='utf-8')

def alive(pid):
 k=ctypes.windll.kernel32;k.OpenProcess.restype=ctypes.c_void_p;k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];k.CloseHandle.argtypes=[ctypes.c_void_p];h=k.OpenProcess(0x100000|0x1000,False,pid)
 if not h:return False
 try:return k.WaitForSingleObject(h,0)==258
 finally:k.CloseHandle(h)

def run():
 c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['sha256'],'Delivery config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Delivery source changed '+p)
 started=time.monotonic()
 def timeguard():need(time.monotonic()-started<c['wall_seconds'],'Finite delivery wall limit reached')
 def command(name,env,script,*args):
  timeguard();state(name);cid=uuid.uuid4().hex;cmd=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script,*args];event('child_intent',call_id=cid,name=name,command=cmd)
  from scripts.pipeline import project_lock
  with project_lock(ROOT):pass
  with (LOG/(name+'_'+cid+'.stdout.log')).open('xb') as out,(LOG/(name+'_'+cid+'.stderr.log')).open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);state(name,child_pid=proc.pid,call_id=cid)
   while proc.poll() is None:timeguard();time.sleep(5)
   code=proc.returncode
  event('child_returned',call_id=cid,name=name,exit_code=code);need(code==0,'Finite stage failed; logs/checkpoints retained: '+name)
 def completed_epoch(n):return (ROOT/f'logs/P2P3/candidate_v27/epoch_{n:02d}.json').exists()
 if not completed_epoch(3):
  state('waiting_existing_P3_first3',existing_pid=c['existing_training_pid'])
  while alive(c['existing_training_pid']):timeguard();time.sleep(5)
  need(completed_epoch(3),'Existing P3 first-three training stopped before completion; no duplicate restart')
 if not (ROOT/'logs/P2P3/candidate_v27/early_readonly.json').exists():
  command('P3_early_generate','vision','scripts/p3_generate_v27.py','--stage','early');command('P3_early_official_score','grasp_eval','scripts/p3_score_v27.py','--stage','early');command('P3_early_readonly','vision','scripts/p3_review_v27.py','--stage','early')
 early_review=read(ROOT/'logs/P2P3/candidate_v27/early_readonly.json');need(early_review['engineering_passed'],'Early full review missing')
 for p,s in early_review['files'].items():need(stat(ROOT/p)==s,'Reviewed early result changed '+p)
 need(early_review['scientific_usable']==read(ROOT/'results/P2P3/candidate_v27/early_gate.json')['continue_training'],'Early scientific decision mismatch')
 if not read(ROOT/'results/P2P3/candidate_v27/early_gate.json')['continue_training']:
  if not (ROOT/'logs/P2P3/candidate_v27/early_cold.json').exists():command('P3_early_cold','vision','scripts/p3_review_v27.py','--stage','early','--cold')
  state('P3_early_scientific_gate_failed_stopped',P7_started=False,reason='Candidate effective-support gate failed; preserve negative, no automatic scope expansion');event('scientific_stop',stage='P3_early');record_stop('P3首3轮候选有效性关口未通过，未运行剩余训练和P7；详见results/P2P3/candidate_v27/early_gate.json');return
 if not completed_epoch(18):command('P3_train_remaining','vision','scripts/p3_train_v27r1.py','--through','18')
 if not (ROOT/'logs/P2P3/candidate_v27/final_readonly.json').exists():
  command('P3_final_generate','vision','scripts/p3_generate_v27.py','--stage','final');command('P3_final_official_score','grasp_eval','scripts/p3_score_v27.py','--stage','final');command('P3_final_readonly','vision','scripts/p3_review_v27.py','--stage','final')
 if not (ROOT/'logs/P2P3/candidate_v27/final_cold.json').exists():command('P3_final_cold','vision','scripts/p3_review_v27.py','--stage','final','--cold')
 if not (ROOT/'results/P2P3/candidate_v27/closeout.json').exists():command('P3_final_closeout','vision','scripts/p3_finish_v27.py')
 if not read(ROOT/'results/P2P3/candidate_v27/closeout.json')['candidate_usable_for_P7']:state('P3_scientific_gate_failed_stopped',P7_started=False);record_stop('P3全固定训练和验证范围已完成，但候选可用性关口未通过，P7未启动');return
 command('P7_prepare','vision','scripts/p7_prepare_v28.py')
 if not (ROOT/'results/P7/risk_v28/generation_complete.json').exists():command('P7_generate','vision','scripts/p7_generate_v28.py')
 if not (ROOT/'results/P7/risk_v28/scoring_complete.json').exists():command('P7_score','grasp_eval','scripts/p7_score_v28.py')
 if not (ROOT/'results/P7/risk_v28/fit_result.json').exists():command('P7_fit','vision','scripts/p7_fit_v28.py')
 if not (ROOT/'results/P7/risk_v28/model_identity.json').exists():command('P7_model_identity','vision','scripts/p7_model_identity_v28.py')
 if not (ROOT/'logs/P7/risk_v28/readonly.json').exists():command('P7_full_review','grasp_eval','scripts/p7_review_v28.py')
 if not (ROOT/'logs/P7/risk_v28/cold.json').exists():command('P7_cold','grasp_eval','scripts/p7_review_v28.py','--cold')
 if not (ROOT/'results/P7/risk_v28/closeout.json').exists():command('P7_finish','vision','scripts/p7_finish_v28.py')
 result=read(ROOT/'results/P7/risk_v28/closeout.json');atomic(STATE,dict(at=now(),phase='P7_completed_stopped',result='results/P7/risk_v28/closeout.json',all_tasks_percent=result['all_tasks_percent'],experiment_percent=result['experiment_percent'],scientific_usable=result['scientific_usable'],P8_started=False,recurring_automation=False));event('delivery_complete_stopped',scientific_usable=result['scientific_usable'])

if __name__=='__main__':
 import msvcrt
 LOG.mkdir(parents=True,exist_ok=True);p=ROOT/'runtime/pipeline/p23_to_p7_v28.lock'
 with p.open('a+b') as lock:
  lock.seek(0,2)
  if lock.tell()==0:lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  try:run()
  except Exception as exc:event('delivery_failure',error=repr(exc),traceback=traceback.format_exc());state('engineering_failure_preserved',error=repr(exc));record_stop('工程入口失败：'+repr(exc));raise
  finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
