"""Consume the terminal 720-intent seal once. Workers never share writable method state."""
import os,time,shutil,subprocess,sys,traceback,stat
from collections import deque
from common import *
OLD=ROOT/'IP140_R2_R6formal720_raw_a_c0_paired01'
def method_start(t,lane):
    key=t['trial_id'];seed=t['worker_seed'];require(shutil.disk_usage(HERE).free>12*2**30,'storage reserve breached')
    cwd=HERE/'workspaces'/lane;job=HERE/'run/jobs/method'/key;temp=cwd/'tmp'/key
    temp.mkdir(parents=True,exist_ok=False);job.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONHASHSEED=str(seed),TMP=str(temp),TEMP=str(temp),TMPDIR=str(temp),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(temp/'mpl'),NUMBA_CACHE_DIR=str(temp/'numba'),TORCH_EXTENSIONS_DIR=str(temp/'torch'))
    exe=ROOT/'runtime/simulation_v52e/venv/Scripts/python.exe'
    cmd=[str(exe),'-B',str(HERE/'code/method_entry.py'),key,str(seed)]
    put(job/'dispatch.json',dict(id=key,seed=seed,cwd=str(cwd),temporary=str(temp),command=cmd,utc=now(),formal_budget=720,attempt=1))
    with (job/'stdout.log').open('xb') as out,(job/'stderr.log').open('xb') as err:
        return subprocess.Popen(cmd,cwd=cwd,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
class Guard:
    def __init__(self):
        self.files={};self.stats={}
        seal=read(HERE/'sealed_intents.json')
        require(seal['method_budget']==720 and len(seal['files'])==720 and seal['unit_manifest_sha256']==sha(HERE/'preregistration/actual_unit_inputs.json') and seal['source_seal_sha256']==sha(HERE/'preregistration/executing_source.json'),'terminal seal missing')
        self.files.update({ROOT/k:h for k,h in read(HERE/'preregistration/frozen_inputs.json').items()})
        local=read(HERE/'preregistration/executing_source.json')['local_sources']
        self.files.update({OLD/k:h for k,h in local.items()})
        signed=read(HERE/'protocol/release.json')
        require(signed['replay_passed'] and signed['total']==720 and signed['new_allowed']==712 and signed['old_adjudicated']==7,'incomplete release')
        self.files.update({HERE/k:h for k,h in signed['files'].items()})
        self.files[HERE/'protocol/release.json']=sha(HERE/'protocol/release.json')
        require(sha(OLD/'sealed_intents.json')==sha(HERE/'sealed_intents.json')==signed['parent_intents_sha256'],'original intent seal changed');self.files.update({HERE/k:h for k,h in seal['files'].items()})
        for v in read(HERE/'preregistration/actual_unit_inputs.json')['units'].values():
            self.files.update({HERE/k:h for k,h in v['source_files'].items()})
            base=HERE/'scenes'/v['scene'];z=v['scene_hashes']
            for name,key in [('observed_rgbd.npz','observation_sha256'),('settled_state.json','state_sha256'),('receipt.json','receipt_sha256')]:self.files[base/name]=z[key]
        for name in ['sealed_intents.json','preregistration/actual_unit_inputs.json','preregistration/design.json','preregistration/design_seal.json','preregistration/executing_source.json','preregistration/frozen_inputs.json']:
            self.files[HERE/name]=sha(HERE/name)
        self.check(True)
    def check(self,full=False):
        for p,h in self.files.items():
            st=p.stat();sig=(st.st_mtime_ns,st.st_size,st.st_ctime_ns)
            if full or self.stats.get(p)!=sig:require(sha(p)==h,'frozen hash drift '+str(p))
            self.stats[p]=sig
def checkpoint(done,queue,live,faults,trail):
    n=len(done);name=f'{n:03d}';p=HERE/'checkpoints'/(name+'.json')
    data=dict(planned=720,completed_results=n,returned_method_evidence=list(done),unlaunched=len(queue),in_flight={l:k for l,(k,_) in live.items()},unknown_or_fault=faults,prior_checkpoint_sha256=trail,utc=now())
    h=put(p,data);p.chmod(stat.S_IREAD);return h
def main():
    guard=Guard();d=read(HERE/'preregistration/design.json');scope=read(HERE/'protocol/continuation.json');prior=set(scope['adjudicated_unchanged']+scope['supplemental_only']);q=deque(t for t in d['trials'] if t['trial_id'] not in prior);live={};done=[];faults=[];trail=None
    require([t['trial_id'] for t in q]==scope['allowed_once_in_original_order'] and len(q)==712,'pending identity/order drift')
    for key in sorted(prior):
        pr=HERE/'run/jobs/method'/key/'return.json';result=HERE/'trials'/key/'method_result.json';require(read(pr)['method_result_sha256']==sha(result),'previous formal result identity')
        done.append(dict(trial_id=key,method_result_sha256=sha(result),return_sha256=sha(pr),origin='offline_supplement' if key in scope['supplemental_only'] else 'sealed_prior_result'))
    start=HERE/'run/method_batch_started.json';put(start,dict(utc=now(),pid=os.getpid(),sealed_intents=sha(HERE/'sealed_intents.json'),guarded_files=len(guard.files)))
    lanes=['cpu01','cpu02','cpu03','cpu04'];last_guard=time.monotonic()
    unit_lane={u['unit_id']:lanes[(u['unit_order']-1)%len(lanes)] for u in d['units']}
    require(len(unit_lane)==240 and unit_lane==read(OLD/'run/method_initial_status.json')['fixed_unit_lane'],'original lane design')
    pending={lane:deque(t for t in q if unit_lane[t['unit_id']]==lane) for lane in lanes}
    require(sum(map(len,pending.values()))==712,'independent lane membership')
    try:
        while q or live:
            if not faults and time.monotonic()-last_guard>=40:
                try:guard.check();last_guard=time.monotonic()
                except BaseException as e:faults.append(dict(type='frozen_input_integrity',error=repr(e),utc=now()))
            for lane in lanes:
                if not q or faults:break
                if lane in live or not pending[lane]:continue
                t=pending[lane].popleft();q.remove(t);key=t['trial_id']
                try:live[lane]=(key,method_start(t,lane))
                except BaseException as e:faults.append(dict(trial=key,error=repr(e),utc=now()));break
            returns=[]
            for lane,(key,p) in live.items():
                rc=p.poll()
                if rc is not None:returns.append((lane,key,rc))
            for lane,key,rc in returns:
                del live[lane]
                job=HERE/'run/jobs/method'/key
                if rc or not (job/'return.json').exists():
                    faults.append(dict(trial=key,exit_code=rc,record=read(job/'exception.json') if (job/'exception.json').exists() else 'unknown method return',utc=now()))
                else:
                    proof=read(job/'return.json');dst=HERE/'trials'/key/'method_result.json'
                    require(proof['id']==key and proof['method_result_sha256']==sha(dst),'unaligned method receipt '+key)
                    done.append(dict(trial_id=key,method_result_sha256=proof['method_result_sha256'],return_sha256=sha(job/'return.json')))
                    if len(done)%60==0:trail=checkpoint(done,q,live,faults,trail)
            if returns or (not (HERE/'run/method_initial_status.json').exists()) or (faults and not live):
                if not (HERE/'run/method_initial_status.json').exists():put(HERE/'run/method_initial_status.json',dict(utc=now(),fixed_unit_lane=unit_lane))
                rep(HERE/'run/progress.json',dict(stage='formal_stopping' if faults else 'formal_methods',planned=720,started=len(list((HERE/'run/jobs/method').glob('*/launch.json'))),completed=len(done),unlaunched=len(q),active={l:i for l,(i,_) in live.items()},faults=faults,latest_checkpoint=trail,utc=now()))
            if faults and not live:break
            if live:time.sleep(.5)
        if not faults:guard.check(True)
        require(not faults and len(done)==720,'shared formal integrity fault: '+str(faults))
        put(HERE/'run/method_complete.json',dict(planned=720,returned=720,terminal_checkpoint_sha256=trail,terminal_input_guard=True,utc=now()))
        rep(HERE/'run/progress.json',dict(stage='formal_execution_closed',planned=720,started=720,completed=720,active={},utc=now()))
    except BaseException as e:
        put(HERE/'run/METHOD_STOP.json',dict(error=repr(e),utc=now(),traceback=traceback.format_exc(),received=done,faults=faults,remaining_not_adjudicated=720-len(done)));raise
if __name__=='__main__':main()
