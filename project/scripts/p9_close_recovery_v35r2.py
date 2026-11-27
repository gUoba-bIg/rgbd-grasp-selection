"""Bounded Windows retry for mutable cold/finish usage writes only."""
import os,sys,time,uuid,json,shutil,subprocess,traceback,runpy
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p9_recovery_v35r1 import environment,verify_recovery
from scripts.p9_common_v35 import *
from scripts.p9_lifecycle_v35 import control
from scripts.pipeline import project_lock
REC=LOG/'close_recovery_v35r2';RC=ROOT/'configs/p9_close_recovery_v35r2.json'
ORIGINAL_ATOMIC=atomic
def replace_bounded(src,dst,replace=os.replace,clock=time.monotonic,sleep=time.sleep,notify=None):
 start=clock();attempt=0
 while True:
  try:replace(src,dst);return
  except PermissionError as e:
   if getattr(e,'winerror',None) not in (5,32):raise
   elapsed=clock()-start
   if notify:notify(attempt,elapsed,e)
   if elapsed>=2.0:raise
   sleep(min(.02*(2**min(attempt,4)),2.0-elapsed));attempt+=1
def usage_atomic(p,x):
 p=Path(p)
 if p.parent!=LOG or p.suffix!='.json' or not p.name.startswith(('usage_cold_','usage_finish_')):return ORIGINAL_ATOMIC(p,x)
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
 def notify(attempt,elapsed,error):
  with (REC/'write_retries.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),pid=os.getpid(),path=p.relative_to(ROOT).as_posix(),attempt=attempt,elapsed=elapsed,error=repr(error)))+'\n');f.flush();os.fsync(f.fileno())
 replace_bounded(tmp,p,notify=notify)
def preflight():
 checks=[]
 class FakeClock:
  def __init__(self):self.t=0.
  def now(self):return self.t
  def sleep(self,dt):self.t+=dt
 def denied(code):
  e=PermissionError('injected Windows replacement error');e.winerror=code;return e
 p=REC/'synthetic_atomic.json';p.write_text('old',encoding='utf-8');tmp=REC/'synthetic_atomic.tmp';expected='{"evidence":"unchanged payload"}\n';tmp.write_text(expected,encoding='utf-8');tries=[];clock=FakeClock()
 def flaky(a,b):
  tries.append(1)
  if len(tries)<=2:raise denied(5 if len(tries)==1 else 32)
  os.replace(a,b)
 replace_bounded(tmp,p,replace=flaky,clock=clock.now,sleep=clock.sleep);need(p.read_text()==expected and len(tries)==3,'Transient sharing conflict recovery failed');checks.append('WinError 5 and 32 recover with identical payload')
 clock=FakeClock();tries=[];tmp.write_text('retained_failure',encoding='utf-8');before=stat(p)
 def permanent(a,b):tries.append(1);raise denied(5)
 try:replace_bounded(tmp,p,replace=permanent,clock=clock.now,sleep=clock.sleep)
 except PermissionError:pass
 else:raise RuntimeError('Permanent denial was swallowed')
 need(clock.t==2. and stat(p)==before and tmp.read_text()=='retained_failure','Bounded failure lost evidence');checks.append('Permanent sharing denial stops at two seconds and retains source and destination')
 for code in [13]:
  clock=FakeClock();tries=[]
  def other(a,b):tries.append(1);raise denied(code)
  try:replace_bounded(tmp,p,replace=other,clock=clock.now,sleep=clock.sleep)
  except PermissionError:pass
  else:raise RuntimeError('Non-sharing exception swallowed')
  need(len(tries)==1 and clock.t==0,'Non-sharing error retried')
 checks.append('Other permission error fails immediately')
 probe=REC/'scope_probe.json';usage_atomic(probe,dict(value=7));need(read(probe)=={'value':7},'Out-of-scope original write failed');checks.append('Non-usage paths retain original atomic function')
 return checks
def verify_close():
 c=verify_recovery();r=read(RC);need(sha(RC)==read(REC/'freeze.json')['sha256'],'Close recovery config changed')
 for p,s in r['protected'].items():need(stat(ROOT/p)==s,'Close recovery protected file changed '+p)
 need((LOG/'calls.jsonl').read_bytes()==(REC/'calls_before.jsonl').read_bytes(),'New scientific call during closure')
 return c
def prepare():
 from datetime import datetime
 need(not RC.exists(),'Close recovery already frozen');REC.mkdir(parents=True,exist_ok=True);c=verify_recovery();ro=load(LOG/'readonly.json');need(ro['passed'] and not (LOG/'cold.json').exists() and not (BASE/'closeout.json').exists(),'Unexpected closure state')
 rows=[json.loads(x) for x in (LOG/'controller.jsonl').read_text().splitlines()];need({x['call_id'] for x in rows if x['kind']=='child_intent'}=={x['call_id'] for x in rows if x['kind']=='child_returned'},'Unknown child return')
 need(all(not v['unknown'] for v in counts().values()),'New unknown research call')
 for name in ['6ad5f25c35b5479a883eefd773040bb0','574af407c91d4cee924961ec2c21a05a']:
  p=LOG/f'cold_reentry_{name}.stderr.log';need(b'PermissionError: [WinError 5]' in p.read_bytes(),'Unexpected cold failure');shutil.copy2(p,REC/p.name)
 for p in [LOG/'calls.jsonl',LOG/'controller.jsonl',ROOT/'runtime/pipeline/p9_delivery_v35.json',ROOT/'runtime/pipeline/p9_part_v35.json',ROOT/'docs/实验自动续接说明.txt',ROOT/'runtime/pipeline/part_level_supervision.json']:
  shutil.copy2(p,REC/('calls_before.jsonl' if p.name=='calls.jsonl' else p.name))
 for p in LOG.glob('usage_cold_*'):shutil.copy2(p,REC/p.name)
 failed='574af407c91d4cee924961ec2c21a05a';intent=next(x for x in rows if x['kind']=='child_intent' and x['call_id']==failed);ret=next(x for x in rows if x['kind']=='child_returned' and x['call_id']==failed)
 once(LOG/('usage_cold_failure_tail_'+uuid.uuid4().hex+'.json'),dict(at=now(),seconds=(datetime.fromisoformat(ret['at'])-datetime.fromisoformat(intent['at'])).total_seconds(),state='conservative_extra_full_failed_child_wall',original_usage_retained=True))
 checks=preflight()
 for p,s in ro['files'].items():need(stat(ROOT/p)==s,'Accepted result changed before closure '+p)
 parents(c)
 paths=[Path(__file__),ROOT/'docs/P9冷重入写入恢复_v35r2.txt',LOG/'readonly.json',LOG/'readonly.json.receipt.json',REC/'calls_before.jsonl',ROOT/'runtime/pipeline/state.json']+[p for p in REC.iterdir() if p.is_file()]
 once(RC,dict(at=now(),protected={p.relative_to(ROOT).as_posix():stat(p) for p in paths},scope='Only mutable usage_cold/usage_finish JSON atomic replacement on Windows sharing errors',retry_seconds=2.,scientific_changes=0,old_budgets_unchanged=True,normal_child_attempts_remaining=1));once(REC/'freeze.json',dict(at=now(),sha256=sha(RC)));sealed(REC/'preflight.json',dict(at=now(),passed=True,checks=checks,accepted_artifacts_verified=len(ro['files']),historical_files_verified=len(read(ROOT/c['parent_binding'])),new_research_calls=0));verify_close();print('P9 closure engineering preflight/freeze passed',flush=True)
def child(phase):
 need(phase in ['cold_reentry','finish'],'Only cold and finish authorized');environment();verify_close()
 import scripts.p9_common_v35 as common
 common.atomic=usage_atomic
 script,args=('scripts/p9_review_v35.py',['--cold']) if phase=='cold_reentry' else ('scripts/p9_lifecycle_v35.py',['finish'])
 sys.argv=[script]+args;runpy.run_path(str(ROOT/script),run_name='__main__')
def deliver():
 c=verify_close();prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));start=time.monotonic();usage=LOG/('controller_usage_close_recovery_'+uuid.uuid4().hex+'.json')
 def guard():atomic(usage,dict(at=now(),seconds=time.monotonic()-start));need(prior+time.monotonic()-start<c['wall_seconds'],'Original cumulative wall cap')
 for phase,done in [('cold_reentry',LOG/'cold.json'),('finish',BASE/'closeout.json')]:
  if done.exists():load(done);continue
  with project_lock(ROOT):pass
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B','scripts/p9_close_recovery_v35r2.py','child',phase];control('child_intent',call_id=cid,phase=phase,command=cmd,recovery='v35r2')
  with (LOG/f'{phase}_{cid}.stdout.log').open('xb') as out,(LOG/f'{phase}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase=phase,controller_pid=os.getpid(),child_pid=proc.pid,all_tasks_percent=64,experiment_percent=80.,P10_started=False,recovery='v35r2'))
   try:
    while proc.poll() is None:guard();time.sleep(2)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
  control('child_returned',call_id=cid,phase=phase,exit_code=proc.returncode,recovery='v35r2');need(proc.returncode==0,'Closure recovery failed; retained '+phase)
 guard();r=load(BASE/'closeout.json');control('completed_stopped',scientific_usable=r['scientific_usable'],recovery='v35r2');atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase='P9_completed_stopped',scientific_usable=r['scientific_usable'],all_tasks_percent=72,experiment_percent=90.,P10_started=False,recovery='v35r2'))
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':
  with project_lock(ROOT):prepare()
 elif mode=='child':child(sys.argv[2])
 elif mode=='deliver':
  import msvcrt
  with (ROOT/'runtime/pipeline/p9_v35.lock').open('r+b') as f:
   msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   try:deliver()
   except Exception as e:
    control('failure_preserved',recovery='v35r2',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e),recovery='v35r2');atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase='engineering_failure_preserved',error=repr(e),all_tasks_percent=64,experiment_percent=80.,P10_started=False,recovery='v35r2'));raise
   finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
 else:raise RuntimeError('Unknown close recovery mode')
