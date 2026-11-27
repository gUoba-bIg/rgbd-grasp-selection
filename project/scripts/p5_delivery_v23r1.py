"""One finite local P5 delivery sequence; never starts another main part."""
import os,sys,ctypes,subprocess,time,uuid
from ctypes import wintypes
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23r1 import *

def main():
    pid=int(sys.argv[1]);receipt=LOG/'delivery_started.json';need(not receipt.exists(),'Delivery controller already registered; inspect process and existing stage outputs')
    frozen=read(LOG/'delivery_freeze.json');need(all(stat(ROOT/p)==v for p,v in frozen['protected'].items()),'Delivery source changed')
    once(receipt,dict(at=now(),pid=os.getpid(),waiting_existing_training_pid=pid,finite_main_part='P5',automatic_schedule=False))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE;kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD;kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x100000|0x1000,False,pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(handle,5000)==258:pass
        finally:kernel.CloseHandle(handle)
    early=(BASE/'finite_stop.json').exists();need(early or (BASE/'exports_complete.json').exists(),'Training stopped before finite endpoint; inspect logs; no downstream experiment')
    stages=[('evaluation_prepare','p4_eval_v22','scripts/p5_evaluate_v23r1.py','prepare'),('screen_readonly','p4_eval_v22','scripts/p5_evaluate_v23r1.py','screen')]
    if not early:stages += [('official_evaluation','p4_eval_v22','scripts/p5_evaluate_v23r1.py','compute'),('independent_readonly','p4_eval_v22','scripts/p5_evaluate_v23r1.py','verify')]
    stages += [('figures','grasp_eval','scripts/p5_figures_v23r1.py',None),('cold_reentry_and_closeout','p4_eval_v22','scripts/p5_closeout_v23r1.py',None)]
    for tag,env,script,mode in stages:
        out=LOG/(tag+'.stdout.log');err=LOG/(tag+'.stderr.log');need(not out.exists() and not err.exists(),'Existing stage attempt must be inspected before retry: '+tag);cmd=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script]+([mode] if mode else []);cid=uuid.uuid4().hex;event('delivery_stage_intent',call_id=cid,phase=tag,command=cmd);start=time.monotonic()
        with out.open('x',encoding='utf-8') as so,err.open('x',encoding='utf-8') as se:p=subprocess.run(cmd,cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        event('delivery_stage_returned',call_id=cid,phase=tag,exit_code=p.returncode,seconds=time.monotonic()-start);need(p.returncode==0,'P5 delivery failed in '+tag+'; preserve logs and do not start next part')
    once(LOG/'delivery_complete.json',dict(at=now(),P5_complete=True,stopped=True,next_part_started=False,final_readonly_sha256=sha(LOG/'final_readonly.json')))
    print('P5 complete, stopped for user review; no P6 or scheduler started',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:once(LOG/('delivery_failure_'+uuid.uuid4().hex+'.json'),dict(at=now(),error=repr(e),stopped=True,no_downstream_part_started=True));raise
