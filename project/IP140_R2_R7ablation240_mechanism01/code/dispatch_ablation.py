"""Four private lanes, exclusive reservations, immutable checkpoint chain, no method retries."""
import os, time, sys, stat, shutil, subprocess, traceback
from collections import deque
from common import *
from freeze_sample import ARMS

class Guard:
    def __init__(self):
        seal=read(HERE/'sealed_intents.json');sp=HERE/'preregistration/input_release.json'
        require(seal['version']==HERE.name and seal['method_budget']==len(seal['files'])==240 and seal['input_release_sha256']==sha(sp),'incomplete ablation seal')
        self.paths={ROOT/k:h for k,h in read(sp)['files'].items()}
        self.paths.update({HERE/k:h for k,h in seal['files'].items()})
        self.paths[sp]=sha(sp);self.paths[HERE/'sealed_intents.json']=sha(HERE/'sealed_intents.json')
        self.stats={}
        self.check(True)
    def check(self,full=False):
        for p,h in self.paths.items():
            st=p.stat();sig=st.st_size,st.st_mtime_ns,st.st_ctime_ns
            if full or self.stats.get(p)!=sig:
                require(sha(p)==h,'protected hash changed '+str(p))
            self.stats[p]=sig

def start(t,lane):
    key=t['trial_id'];seed=t['worker_seed'];cwd=HERE/'workspaces'/lane;job=HERE/'run/jobs/method'/key
    tmp=cwd/'tmp'/key;tmp.mkdir(parents=True,exist_ok=False);job.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONHASHSEED=str(seed),
               TMP=str(tmp),TEMP=str(tmp),TMPDIR=str(tmp),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(tmp/'mpl'),NUMBA_CACHE_DIR=str(tmp/'numba'),TORCH_EXTENSIONS_DIR=str(tmp/'torch'))
    exe=ROOT/'runtime/simulation_v52e/venv/Scripts/python.exe'
    args=[str(exe),'-B',str(HERE/'code/method_entry.py'),key,str(seed)]
    put(job/'dispatch.json',dict(id=key,method=t['method'],unit_id=t['unit_id'],seed=seed,attempt=1,cwd=str(cwd),temporary=str(tmp),command=args,utc=now(),method_budget=240))
    with (job/'stdout.log').open('xb') as out,(job/'stderr.log').open('xb') as err:
        return subprocess.Popen(args,cwd=cwd,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)

def verify(key):
    folder=HERE/'trials'/key;job=HERE/'run/jobs/method'/key;receipt=read(job/'return.json');res=read(folder/'method_result.json');intent=read(folder/'intent.json')
    require(receipt['completed'] and receipt['id']==intent['trial_id']==res['trial_id']==key and res['method']==intent['method'] and res['unit_id']==intent['unit_id'],'trial identity')
    require(receipt['method_result_sha256']==sha(folder/'method_result.json') and receipt['intent_sha256']==sha(folder/'intent.json') and receipt['worker_claim_sha256']==sha(job/'launch.json'),'unbound worker return')
    require(not res['shared_fault'] and type(res['grasp_success']) is bool and res['physical_steps']==res['physical_returns'] and receipt['candidate_id']==intent['selected_candidate_id'],'unknown result')
    return dict(trial_id=key,method_result_sha256=receipt['method_result_sha256'],return_sha256=sha(job/'return.json'))

def main():
    guard=Guard();design=read(HERE/'preregistration/design.json')
    require(sha(HERE/'preregistration/design.json')==read(HERE/'sealed_intents.json')['design_sha256'],'design changed')
    require(len(design['trials'])==240 and {t['method'] for t in design['trials']}==set(ARMS),'unregistered methods')
    require(not (HERE/'run/jobs/method').exists(),'existing reservations: duplicate launch refused')
    # Exclusive file is the durable dispatcher claim. It survives even an interrupted scheduler.
    put(HERE/'run/dispatcher_claim.json',dict(pid=os.getpid(),utc=now(),seal=sha(HERE/'sealed_intents.json'),budget=240,workers=4))
    pending=deque(design['trials']);live={};done=[];faults=[];trail=None;last_guard=0.;minimum=shutil.disk_usage(HERE).free
    lanes=[f'cpu{i:02d}' for i in range(1,5)];last_band='normal'
    def status(stage,free):
        return dict(stage=stage,planned=240,completed=len(done),started=240-len(pending),
            unlaunched=len(pending),active={lane:v[0] for lane,v in live.items()},faults=faults,
            minimum_free_bytes=minimum,free_bytes=free,latest_checkpoint=trail,utc=now())
    def publish(stage,free):
        row=status(stage,free)
        rep(HERE/'run/progress.json',row)
        with (HERE/'run/progress.jsonl').open('a',encoding='utf8',newline='\n') as f:
            f.write(json.dumps(row,separators=(',',':'),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    publish('starting',minimum)
    while pending or live:
        free=shutil.disk_usage(HERE).free;minimum=min(minimum,free)
        band='pause' if free<120*2**30 else 'warn' if free<150*2**30 else 'normal'
        if band!=last_band:
            put(HERE/'run/disk_events'/f'{time.time_ns()}.json',dict(band=band,free=free,utc=now()))
            last_band=band
        if time.monotonic()-last_guard>=60 and not faults:
            try:guard.check()
            except BaseException as exc:faults.append(dict(kind='protected_input',error=repr(exc),utc=now()))
            last_guard=time.monotonic()
        changed=False
        for lane,(key,p) in list(live.items()):
            rc=p.poll()
            if rc is None:continue
            del live[lane];changed=True
            try:
                require(rc==0,'worker exit '+str(rc))
                done.append(verify(key))
                if len(done)%60==0:
                    out=HERE/'checkpoints'/f'{len(done):03d}.json'
                    trail=put(out,dict(completed=len(done),returns=list(done),prior=trail,utc=now()))
                    out.chmod(stat.S_IREAD)
                    print('R7 CHECKPOINT',len(done),trail,flush=True)
            except BaseException as exc:
                faults.append(dict(trial=key,kind='shared_or_unknown_return',error=repr(exc),exit_code=rc,utc=now()))
        if not faults and band!='pause':
            for lane in lanes:
                if not pending or lane in live:continue
                t=pending[0]
                try:
                    p=start(t,lane);pending.popleft();live[lane]=(t['trial_id'],p);changed=True
                except BaseException as exc:
                    faults.append(dict(trial=t['trial_id'],kind='dispatch_error',error=repr(exc),utc=now()));break
        if changed or (faults and not live):publish('draining_on_fault' if faults else 'executing',free)
        if faults and not live:break
        if pending or live:time.sleep(1 if band!='pause' or live else 30)
    if not faults:guard.check(True)
    if faults:
        put(HERE/'run/STOP.json',status('stopped_shared_fault',shutil.disk_usage(HERE).free))
        raise RuntimeError('shared fault; see STOP.json')
    require(len(done)==len(set(r['trial_id'] for r in done))==240,'missing final returns')
    publish('execution_closed',shutil.disk_usage(HERE).free)
    put(HERE/'run/method_complete.json',dict(completed=240,unknown=0,remaining=0,terminal_checkpoint_sha256=trail,minimum_free_bytes=minimum,full_hash_check=True,utc=now()))
    print('240/240 completed',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (HERE/'run/STOP.json').exists():
            put(HERE/'run/controller_error.json',dict(utc=now(),error=traceback.format_exc()))
        raise
