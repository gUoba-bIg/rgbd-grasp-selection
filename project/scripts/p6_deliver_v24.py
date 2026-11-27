"""P6 finite delivery: separate stages, one project lock per active worker."""
import os,sys,subprocess,time
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import *
def main():
 need(not (LOG/'delivery_started.json').exists(),'P6 controller already registered; inspect actual process and stage logs');verify();once(LOG/'delivery_started.json',dict(at=now(),pid=os.getpid(),finite=True,next_part=False,automatic_schedule=False))
 stages=[('preflight','p4_eval_v22','scripts/p6_compute_v24.py',['--screen']),('preflight_readonly','p4_eval_v22','scripts/p6_accept_v24.py',['--screen']),('quality','vision','scripts/p6_quality_v24.py',[]),('geometry','p4_eval_v22','scripts/p6_compute_v24.py',[]),('readonly','p4_eval_v22','scripts/p6_accept_v24.py',[]),('finish','grasp_eval','scripts/p6_finish_v24.py',[])]
 for tag,env,script,args in stages:
  out=LOG/(tag+'.stdout.log');err=LOG/(tag+'.stderr.log');need(not out.exists() and not err.exists(),'Previous stage attempt needs review');command=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script]+args;cid=uuid.uuid4().hex;event('stage_intent',call_id=cid,phase=tag,command=command);start=time.monotonic()
  with out.open('x',encoding='utf-8') as so,err.open('x',encoding='utf-8') as se:p=subprocess.run(command,cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
  event('stage_returned',call_id=cid,phase=tag,returncode=p.returncode,seconds=time.monotonic()-start);need(p.returncode==0,'P6 stage failed: '+tag)
 once(LOG/'delivery_complete.json',dict(at=now(),stopped=True,next_part_started=False,final_readonly_sha256=sha(LOG/'final_readonly.json')));print('P6 finite stage complete and stopped; P7 not started',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:once(LOG/('failure_'+uuid.uuid4().hex+'.json'),dict(at=now(),error=repr(e),stopped=True));status('execution_failed',error=repr(e));raise
