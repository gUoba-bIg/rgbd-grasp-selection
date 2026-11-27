"""Immutable environment routing only; resumes the frozen v35 scientific code."""
import os,sys,time,uuid,shutil,subprocess,traceback,runpy
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p9_common_v35 import *
REC=LOG/'environment_recovery_v35r1';RC=ROOT/'configs/p9_environment_recovery_v35r1.json'
STEPS=[('prediction','scripts/p9_geometry_v35.py',['predict'],BASE/'selection_complete.json'),('scoring','scripts/p9_geometry_v35.py',['score'],BASE/'scoring_complete.json'),('metrics','scripts/p9_metrics_v35.py',[],BASE/'metrics.json'),('full_readonly','scripts/p9_review_v35.py',[],LOG/'readonly.json'),('cold_reentry','scripts/p9_review_v35.py',['--cold'],LOG/'cold.json'),('finish','scripts/p9_lifecycle_v35.py',['finish'],BASE/'closeout.json')]
def environment():
 path=str(ROOT/'runtime/envs/grasp_eval/Lib/site-packages')
 if path not in sys.path:sys.path.append(path)
 import torch,numpy,scipy,open3d
 need(torch.__version__=='2.6.0+cu126' and numpy.__version__=='1.26.4' and scipy.__version__=='1.14.1' and open3d.__version__=='0.20.0','Existing dependency version changed')
 need(Path(torch.__file__).is_relative_to(ROOT/'runtime/envs/vision') and Path(open3d.__file__).is_relative_to(ROOT/'runtime/envs/grasp_eval'),'Dependency priority changed')
 return dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,open3d=open3d.__version__,torch_path=torch.__file__,open3d_path=open3d.__file__)
def verify_recovery():
 c=verify();r=read(RC);need(sha(RC)==read(REC/'freeze.json')['sha256'],'Recovery protocol changed')
 for p,s in r['protected'].items():need(stat(ROOT/p)==s,'Recovery dependency changed '+p)
 need((LOG/'calls.jsonl').read_bytes().startswith((REC/'calls_before.jsonl').read_bytes()),'Original ledger prefix changed');return c
def prepare():
 need(not RC.exists(),'Recovery already frozen');c=verify();REC.mkdir(parents=True,exist_ok=True);env=environment()
 from scripts.p9_geometry_v35 import Geometry
 geo=Geometry(c);need(len(geo.friction)>0,'Official scoring adapter configuration unavailable')
 pending_rows=[json.loads(s) for s in (LOG/'controller.jsonl').read_text().splitlines()];pending={r['call_id'] for r in pending_rows if r['kind']=='child_intent'}-{r['call_id'] for r in pending_rows if r['kind']=='child_returned'};need(not pending,'Unknown previous child, check processes first');ledger=counts();need(ledger['official_score']['attempts']==0 and ledger['risk_forward']['known_returns']==67,'Failure boundary differs');need(all(not x['unknown'] for x in ledger.values()),'Unexpected unknown research invocation');load(BASE/'training_complete.json');load(BASE/'calibration_complete.json')
 for p in [LOG/'calls.jsonl',LOG/'controller.jsonl',ROOT/'runtime/pipeline/p9_part_v35.json',ROOT/'runtime/pipeline/p9_delivery_v35.json']:
  name='calls_before.jsonl' if p.name=='calls.jsonl' else p.name;shutil.copy2(p,REC/name)
 failed=LOG/'prediction_1b2ab1d063c440f99daf8d3e4047d723.stderr.log';need("No module named 'open3d'" in failed.read_text(errors='replace'),'Unexpected original failure');paths=[Path(__file__),ROOT/'docs/P9评价环境路由修复_v35r1.txt',failed,REC/'calls_before.jsonl',REC/'controller.jsonl',ROOT/'runtime/envs/vision/Scripts/python.exe']+[p for p in BASE.rglob('*') if p.is_file()]+[p for p in EXP.rglob('*') if p.is_file()]
 for module in list(sys.modules.values()):
  value=getattr(module,'__file__',None)
  if value:
   p=Path(value)
   if p.is_file() and p.is_relative_to(ROOT/'runtime/envs/grasp_eval'):paths.append(p)
 import open3d
 for p in Path(open3d.__file__).parent.rglob('*'):
  if p.is_file() and p.suffix in ['.pyd','.dll']:paths.append(p)
 protected={p.relative_to(ROOT).as_posix():stat(p) for p in paths};once(RC,dict(at=now(),environment=env,protected=protected,original_config=stat(CONF),scope='resolve existing evaluator dependencies at path tail; frozen scientific source unchanged',training_repeated=False,original_limits_unchanged=True));once(REC/'freeze.json',dict(at=now(),sha256=sha(RC)));sealed(REC/'preflight.json',dict(at=now(),passed=True,actual_geometry_constructor_passed=True,existing_environment_priority_passed=True,old_training_calibration_preserved=True,unknown_new_calls=0,new_training=0,new_forwards=0,new_official_scores=0));verify_recovery();print('P9 environment recovery frozen; bound files',len(protected),flush=True)
def deliver():
 from scripts.p9_lifecycle_v35 import control
 from scripts.pipeline import project_lock
 c=verify_recovery();environment();prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));start=time.monotonic();usage=LOG/('controller_usage_recovery_'+uuid.uuid4().hex+'.json')
 def guard():atomic(usage,dict(at=now(),seconds=time.monotonic()-start));need(prior+time.monotonic()-start<c['wall_seconds'],'Original controller wall cap exceeded')
 for phase,script,args,done in STEPS:
  if done.exists():load(done);continue
  with project_lock(ROOT):pass
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B','scripts/p9_recovery_v35r1.py','child',phase];control('child_intent',call_id=cid,phase=phase,command=cmd,recovery='v35r1')
  with (LOG/f'{phase}_{cid}.stdout.log').open('xb') as out,(LOG/f'{phase}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase=phase,controller_pid=os.getpid(),child_pid=proc.pid,all_tasks_percent=64,experiment_percent=80.,P10_started=False,recovery='v35r1'))
   try:
    while proc.poll() is None:guard();time.sleep(2)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
  control('child_returned',call_id=cid,phase=phase,exit_code=proc.returncode,recovery='v35r1');need(proc.returncode==0,'P9 recovered stage failed; evidence retained: '+phase)
 guard();r=load(BASE/'closeout.json');control('completed_stopped',scientific_usable=r['scientific_usable'],recovery='v35r1');atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase='P9_completed_stopped',scientific_usable=r['scientific_usable'],all_tasks_percent=72,experiment_percent=90.,P10_started=False,recovery='v35r1'))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 mode=sys.argv[1]
 if mode=='prepare':
  with project_lock(ROOT):prepare()
 elif mode=='child':
  environment();verify_recovery();phase=sys.argv[2];matches=[r for r in STEPS if r[0]==phase];need(len(matches)==1,'Unregistered recovery phase');_,script,args,_=matches[0];sys.argv=[script]+args;runpy.run_path(str(ROOT/script),run_name='__main__')
 elif mode=='deliver':
  import msvcrt
  with (ROOT/'runtime/pipeline/p9_v35.lock').open('r+b') as f:
   msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   try:deliver()
   except Exception as e:
    from scripts.p9_lifecycle_v35 import control
    control('failure_preserved',recovery='v35r1',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e),recovery='v35r1');atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase='engineering_failure_preserved',error=repr(e),all_tasks_percent=64,experiment_percent=80.,P10_started=False,recovery='v35r1'));raise
   finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
 else:raise RuntimeError('Unknown recovery mode')
