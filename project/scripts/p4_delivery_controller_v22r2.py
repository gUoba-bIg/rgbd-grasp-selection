"""One finite P4 delivery chain. Not a scheduler; no retries or next-part work."""
import os,sys,json,time,argparse,ctypes,subprocess,msvcrt,uuid
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
STATE=ROOT/'runtime/pipeline/p4_delivery_v22r2.json'
LOG=ROOT/'logs/P4/full_baseline_v22r2/controller'
BASE=ROOT/'results/P4/full_baseline_v22r2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def status(phase,**x):
    delivered=x.get('delivered_progress',{});row=dict(at=datetime.now().astimezone().isoformat(),phase=phase,pid=os.getpid(),all_tasks_percent=delivered.get('all_tasks',26),experiment_percent=delivered.get('experiments',32.5),recurring_scheduler=False,**x)
    tmp=STATE.with_suffix('.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,STATE)
    with (LOG/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def call(label,environment,arguments,timeout):
    command=[str(ROOT/f'runtime/envs/{environment}/Scripts/python.exe'),'-B',*arguments]
    for path in [LOG/(label+'.stdout.log'),LOG/(label+'.stderr.log')]:
        if path.exists():raise RuntimeError('Preserve previous controller invocation; log already exists: '+str(path))
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',TEMP=str(ROOT/'runtime/tmp/p4_eval_v22'),TMP=str(ROOT/'runtime/tmp/p4_eval_v22'))
    with (LOG/(label+'.stdout.log')).open('xb') as out,(LOG/(label+'.stderr.log')).open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        status(label,child_pid=process.pid,command=command)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Task-scoped process tree only; retain all records, no restart.
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],cwd=ROOT,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            raise RuntimeError(label+' exceeded frozen invocation budget')
    if code!=0:raise RuntimeError(label+' returned '+str(code)+'; preserve logs and repair internally')
def main():
    p=argparse.ArgumentParser();p.add_argument('--training-pid',required=True,type=int);a=p.parse_args();LOG.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runtime/pipeline/p4_delivery_v22r2.lock').open('a+b');lock.seek(0,2)
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        if (BASE/'closeout.json').exists():raise RuntimeError('P4 already delivered; do not continue')
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x100000,False,a.training_pid)
        if handle:
            status('waiting_for_existing_training',training_pid=a.training_pid);start=time.monotonic()
            try:
                while kernel.WaitForSingleObject(handle,30000)==258:
                    if time.monotonic()-start>14400:raise RuntimeError('Training wait cap reached; no replacement training launched')
            finally:kernel.CloseHandle(handle)
        progress=read(BASE/'progress.json')
        if not (BASE/'training_and_export.json').exists() or progress['status']!='training_export_finished_next_internal_acceptance':raise RuntimeError('Training stopped before completed export: '+str(progress))
        if not read(BASE/'training_and_export.json')['preliminary_baseline_usable']:raise RuntimeError('Training/export complete but frozen adoption gates failed; internal repair decision required')
        call('evaluation_compute','p4_eval_v22',['scripts/p4_complete_evaluation_v22r2.py','compute'],3600)
        call('evaluation_full_readonly','p4_eval_v22',['scripts/p4_complete_evaluation_v22r2.py','verify'],3600)
        for index in range(20):call(f'representative_{index:02d}','vision',['scripts/p4_representative_link_v22r2.py','frame','--index',str(index)],300)
        call('representative_full_readonly','p4_eval_v22',['scripts/p4_representative_link_v22r2.py','verify'],300)
        call('whole_P4_closeout','p4_eval_v22',['scripts/p4_closeout_v22r2.py','close'],300)
        final=read(BASE/'closeout.json');status('P4_delivered_stopped_for_user_review',delivered_progress=dict(all_tasks=final['all_tasks_percent'],experiments=final['experiment_percent']),child_pid=None)
        print('Whole P4 delivered. Controller exited; no next part or recurring automation.',flush=True)
    except BaseException as e:status('internal_failure_preserved',error=repr(e));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
