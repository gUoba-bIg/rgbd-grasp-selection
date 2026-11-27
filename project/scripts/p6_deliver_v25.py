"""Finite one-repair controller, never starts P7."""
import sys,subprocess
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v25 import *

def stage(tag,env,script,args=[]):
 out=LOG/(tag+'.stdout.log');err=LOG/(tag+'.stderr.log');need(not out.exists() and not err.exists(),'Existing attempt needs inspection');cmd=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script]+args;uid=uuid.uuid4().hex;event('stage_intent',call_id=uid,tag=tag,command=cmd)
 with out.open('x',encoding='utf-8') as so,err.open('x',encoding='utf-8') as se:p=subprocess.run(cmd,cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
 event('stage_returned',call_id=uid,tag=tag,returncode=p.returncode);need(p.returncode==0,'Stage failed: '+tag)
def main():
 verify();need(not (LOG/'delivery_started.json').exists(),'Already started; inspect instead');once(LOG/'delivery_started.json',dict(at=now(),pid=os.getpid(),finite=True,P7=False))
 stage('data','vision','scripts/p6_data_v25.py');stage('data_review','p4_eval_v22','scripts/p6_review_v25.py',['data']);stage('fit','p4_eval_v22','scripts/p6_evaluate_v25.py',['fit']);stage('holdout','p4_eval_v22','scripts/p6_evaluate_v25.py',['holdout']);stage('holdout_review','p4_eval_v22','scripts/p6_review_v25.py',['holdout'])
 if read(BASE/'holdout_summary.json')['ready']:
  stage('development','p4_eval_v22','scripts/p6_evaluate_v25.py',['development']);stage('development_review','p4_eval_v22','scripts/p6_review_v25.py',['development'])
 stage('finish','grasp_eval','scripts/p6_finish_v25.py');once(LOG/'delivery_complete.json',dict(at=now(),stopped=True,P7_started=False,final_sha256=sha(LOG/'final_readonly.json')));print('P6 repair complete, stopped',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:once(LOG/('failure_'+uuid.uuid4().hex+'.json'),dict(at=now(),error=repr(e),stopped=True));status('execution_failed',error=repr(e));raise
