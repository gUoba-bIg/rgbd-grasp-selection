"""One writer, exclusive job reservation and separate workspace for each native process."""
import time,subprocess,shutil,sys
from collections import deque
from common import *

def seed_for(kind,key):
    return int.from_bytes(hashlib.sha256((HERE.name+'|'+kind+'|'+key).encode()).digest()[:4],'big')
def start(kind,key,seed,lane):
    reserve=read(HERE/'audit/resource_preflight.json')['space_reserve_bytes']
    require(shutil.disk_usage(HERE).free>reserve,'insufficient space to keep raw evidence')
    cwd=HERE/'workspaces'/lane
    temp=cwd/'tmp'/(kind+'_'+key)
    temp.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONHASHSEED=str(seed),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',TMP=str(temp),TEMP=str(temp),TMPDIR=str(temp),CUDA_CACHE_PATH=str(temp/'cuda'),TORCH_EXTENSIONS_DIR=str(temp/'torch'),NUMBA_CACHE_DIR=str(temp/'numba'),MPLCONFIGDIR=str(temp/'mpl'))
    exe=ROOT/('runtime/envs/vision/Scripts/python.exe' if kind=='forward' else 'runtime/envs/grasp_eval/Scripts/python.exe' if kind=='rank' else 'runtime/simulation_v52e/venv/Scripts/python.exe')
    job=HERE/'run/jobs'/kind/key;job.mkdir(parents=True,exist_ok=False)
    cmd=[str(exe),'-B',str(HERE/'code/task_worker.py'),kind,key,str(seed)]
    put(job/'dispatch.json',dict(kind=kind,id=key,seed=seed,command=cmd,cwd=str(cwd),utc=now(),single_attempt=True))
    with (job/'stdout.log').open('xb') as out,(job/'stderr.log').open('xb') as err:
        return subprocess.Popen(cmd,cwd=cwd,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)

def phase(kind,items,lanes,on_return=None):
    queue=deque(items);live={};fail=[];done=0
    while queue or live:
        for lane in lanes:
            if fail or not queue:break
            if lane in live:continue
            key,seed=queue.popleft()
            try:live[lane]=(key,start(kind,key,seed,lane))
            except BaseException as e:fail.append(dict(id=key,detail=repr(e),utc=now()));break
        finished=[]
        for lane,(key,p) in live.items():
            code=p.poll()
            if code is None:continue
            finished.append(lane)
            job=HERE/'run/jobs'/kind/key
            if code or not (job/'return.json').is_file():
                fail.append(dict(id=key,exit_code=code,detail=read(job/'exception.json') if (job/'exception.json').is_file() else 'missing worker return',utc=now()))
            else:
                done+=1
            if on_return is not None:
                try:on_return(key,code,done)
                except BaseException as e:fail.append(dict(id=key,detail=repr(e),utc=now()))
        for lane in finished:del live[lane]
        if finished or fail:
            rep(HERE/'run'/(kind+'_progress.json'),dict(phase=kind,planned=len(items),returned_ok=done,active={l:k for l,(k,_) in live.items()},unlaunched=len(queue),faults=fail,utc=now()))
        if fail and not live:break
        if live:time.sleep(.35)
    if fail:raise RuntimeError(kind+' shared fault '+str(fail))
    put(HERE/'run'/(kind+'_complete.json'),dict(phase=kind,returned=done,utc=now()))
def prepare():
    require((HERE/'audit/input_preflight.json').is_file(),'preflight missing')
    design=read(HERE/'preregistration/design.json')
    cpus=['cpu01','cpu02','cpu03','cpu04']
    for k,items,lanes in [('scene',[x['base_scene'] for x in design['scenes']],cpus),('forward',[x['unit_id'] for x in design['units']],['gpu01']),('rank',[x['unit_id'] for x in design['units']],cpus)]:
        phase(k,[(x,seed_for(k,x)) for x in items],lanes)
if __name__=='__main__':
    try:prepare()
    except BaseException as e:
        import traceback
        put(HERE/'run/PREP_STOP.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),formal_method_results=len(list((HERE/'trials').glob('*/method_result.json')))))
        raise
