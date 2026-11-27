"""Exclusive method execution. For an empty selection, record a fresh paired zero-step world."""
import sys,os,time,random,traceback,msvcrt
import numpy as np
from common import *
from task_worker import bind_engine
def main():
    key=sys.argv[1];seed=int(sys.argv[2]);lane=Path.cwd()
    require(workspace_ok(),'private workspace')
    job=HERE/'run/jobs/method'/key;dst=HERE/'trials'/key;ip=dst/'intent.json'
    with (lane/'lane.lock').open('a+b') as guard:
        if guard.seek(0,2)==0:guard.write(b'1');guard.flush()
        guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        try:
            started=time.monotonic();h=put(job/'launch.json',dict(pid=os.getpid(),cwd=str(lane),temporary=os.environ['TMP'],seed=seed,id=key,utc=now()))
            random.seed(seed);np.random.seed(seed);bind_engine()
            from platform_trial import frozen_intent,worker,connection,checked_zero
            from scene_world import setup_world
            intent,selection,row=frozen_intent(ip);require(intent['worker_seed']==seed,'method RNG identity')
            if row is None:
                with connection(dst/'no_candidate_zero_api','contact_aware_static') as (trace,client):
                    ctx=setup_world(trace,client,intent['scene_key'],dst/'no_candidate_zero_setup',settled_state=HERE/intent['stable_state_path'])
                    checked_zero(ctx)
                require(trace.actual_step_calls==0 and trace.counts==trace.returns and not trace.exceptions,'unknown empty-selection return')
                put(dst/'no_candidate_zero_receipt.json',dict(independent_connection=True,stable_state_sha256=intent['stable_state_sha256'],pool_sha256=intent['pool_sha256'],physical_steps=0,no_method_action=True,native_api_summary_sha256=sha(dst/'no_candidate_zero_api/summary.json')))
            code=worker(ip)
            r=read(dst/'method_result.json')
            require(code==0 and not r['shared_fault'] and type(r['grasp_success']) is bool and r['physical_steps']==r['physical_returns'],'shared frozen-trial failure: '+str(r.get('failure_detail')))
            if row is not None:
                static=read(dst/'static_api/summary.json');require(static['calls']==static['returns'] and static['actual_native_step_calls']==0 and not static['exceptions'],'static call receipt')
            if r['physical_steps']:
                api=read(dst/'dynamic_api/summary.json');ep=read(dst/'episode/result.json')
                require(api['calls']==api['returns'] and not api['exceptions'] and ep['step_returns']==r['physical_returns']==api['actual_native_step_calls'] and ep['unknown_steps']==0 and ep['telemetry_sha256']==sha(dst/'episode/telemetry.jsonl.gz'),'dynamic outcome integrity')
                if r['grasp_success']:
                    require(ep['step_returns']==4728 and ep['engineering_gate_passed'] and ep['final_hold_streak']>=480 and not ep['slipped'],'success receipt')
            put(job/'return.json',dict(completed=True,id=key,intent_sha256=sha(ip),candidate_id=intent['selected_candidate_id'],method_result_sha256=sha(dst/'method_result.json'),worker_claim_sha256=h,elapsed_sec=time.monotonic()-started,utc=now()))
        finally:
            guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_UNLCK,1)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        put(HERE/'run/jobs/method'/sys.argv[1]/'exception.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
