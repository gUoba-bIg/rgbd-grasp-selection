"""Administrative resume of the original formal workers after the storage audit."""
import json
import msvcrt
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback
from collections import deque
from contextlib import ExitStack
from pathlib import Path

from common import HERE, ROOT, now, put, read, require, sha
from formal_dispatch import Guard, OLD, checkpoint, method_start

AUDIT = ROOT / 'R6_storage_audit01'
CONTROL = HERE / 'protocol/storage_resume01'
JOBS = HERE / 'run/jobs/method'
LANES = ('cpu01', 'cpu02', 'cpu03', 'cpu04')
GIB = 2**30
INVENTORY_SHA = '01803e411784b8115518a10ad366e40d9966f141d1e46931143a64425fd05e3a'

def band(free):
    if free < 120*GIB:
        return 'pause'
    if free < 150*GIB:
        return 'warn'
    return 'normal'

def verified_return(key, intent_hash):
    job = JOBS/key
    receipt_path = job/'return.json'
    receipt = read(receipt_path)
    rfile = HERE/'trials'/key/'method_result.json'
    result = read(rfile)
    require(receipt['completed'] is True and receipt['id']==result['trial_id']==key
            and receipt['intent_sha256']==intent_hash and receipt['method_result_sha256']==sha(rfile)
            and receipt['worker_claim_sha256']==sha(job/'launch.json')
            and not result['shared_fault'] and type(result['grasp_success']) is bool
            and result['physical_returns']==result['physical_steps'], 'method evidence '+key)
    return {'trial_id':key,'method_result_sha256':receipt['method_result_sha256'],'return_sha256':sha(receipt_path)}

def preflight():
    invfile=AUDIT/'storage_inventory.json'
    require(sha(invfile)==INVENTORY_SHA,'reviewed storage inventory digest')
    for line in (AUDIT/'SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        h,p=line.split('  ',1)
        require(sha(AUDIT/p)==h,'storage audit hash '+p)
    inv=read(invfile);ledger=inv['pause_reconciliation'];pending_ids=set(ledger['protected_pending_ids'])
    require(ledger['judged_with_matching_result_and_receipt']==265 and ledger['pending_unlaunched']==455
            and len(pending_ids)==455 and ledger['unknown_or_exception']==0,'storage ledger counts')
    audit_hash={r['path'].split('/',1)[1]:r['sha256'] for r in inv['files'] if r['path'].startswith('continuation/')}
    guard=Guard()
    def anchor(rel):
        require(sha(HERE/rel)==audit_hash[rel],'audited evidence drift '+rel)
        guard.files[HERE/rel]=audit_hash[rel]
    frozen=[rel for rel in audit_hash if rel.startswith(('code/','implementation/','preregistration/','protocol/','checkpoints/'))
            or rel in ('sealed_intents.json','run/progress.json','run/method_initial_status.json')]
    for rel in frozen:anchor(rel)
    design=read(HERE/'preregistration/design.json')
    sealed=read(HERE/'sealed_intents.json')
    require(len(design['trials'])==len(sealed['files'])==720,'design length')
    trials={t['trial_id']:t for t in design['trials']}
    actual={p.name for p in JOBS.iterdir() if p.is_dir()}
    require(actual==set(trials)-pending_ids and len(actual)==265 and not list(JOBS.glob('*/exception.json')),
            'actual started set and audit differ')
    require({p.parent.name for p in (HERE/'trials').glob('*/method_result.json')}==actual,'stray result')
    lanes=read(HERE/'run/method_initial_status.json')['fixed_unit_lane']
    require(lanes==read(OLD/'run/method_initial_status.json')['fixed_unit_lane'],'original lane identity')
    require(len(lanes)==240 and set(lanes.values())==set(LANES),'four fixed lanes')
    with ExitStack() as handles:
        for lane in LANES:
            d=HERE/'workspaces'/lane
            require(read(d/'workspace.json') is not None and (d/'data').resolve()==ROOT/'data'
                    and (d/'IP140_R2_smoke22_miss_negative_controls_full_execution01').resolve()==ROOT/'IP140_R2_smoke22_miss_negative_controls_full_execution01','lane assets')
            f=handles.enter_context((d/'lane.lock').open('rb'));f.seek(0)
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            handles.callback(lambda fh=f:(fh.seek(0),msvcrt.locking(fh.fileno(),msvcrt.LK_UNLCK,1)))
    proofs={};ordered_times={}
    for key,t in trials.items():
        rel='trials/'+key+'/intent.json';anchor(rel)
        require(audit_hash[rel]==sealed['files'][rel]==sha(OLD/rel),'sealed intent drift '+key)
        if key in actual:
            for file in ['dispatch.json','launch.json','return.json']:anchor('run/jobs/method/'+key+'/'+file)
            anchor('trials/'+key+'/method_result.json')
            proofs[key]=verified_return(key,audit_hash[rel])
            j=read(JOBS/key/'return.json');i=read(HERE/rel);dispatch=read(JOBS/key/'dispatch.json')
            require(j['candidate_id']==i['selected_candidate_id'] and dispatch['id']==key
                    and dispatch['attempt']==1 and dispatch['seed']==t['worker_seed'],'original receipt candidate/seed '+key)
            ordered_times[key]=j['utc']
        else:
            require({p.name for p in (HERE/'trials'/key).iterdir()}=={'intent.json'} and not (JOBS/key).exists()
                    and not any((HERE/'workspaces'/lane/'tmp'/key).exists() for lane in LANES),
                    'previously touched pending intent '+key)
    cpfiles=sorted((HERE/'checkpoints').glob('*.json'))
    require([p.name for p in cpfiles]==['060.json','120.json','180.json','240.json'],'checkpoint set')
    prefix=[];trail=None
    for idx,p in enumerate(cpfiles,1):
        cp=read(p);rows=cp['returned_method_evidence']
        require(cp['completed_results']==len(rows)==60*idx and cp['prior_checkpoint_sha256']==trail and rows[:len(prefix)]==prefix,'broken checkpoint chain')
        for item in rows[len(prefix):]:
            proof=proofs[item['trial_id']]
            require(all(item[k]==proof[k] for k in proof),'historical checkpoint ownership')
        prefix=rows;trail=sha(p)
    prefix_ids={a['trial_id'] for a in prefix}
    require(len(prefix_ids)==240 and prefix_ids<=actual,'checkpoint identity')
    done=prefix+[proofs[k] for k in sorted(actual-prefix_ids,key=lambda k:(ordered_times[k],k))]
    pending=[t for t in design['trials'] if t['trial_id'] in pending_ids]
    for lane in LANES:
        ordered=[t['trial_id'] for t in design['trials'] if lanes[t['unit_id']]==lane]
        mask=[k in pending_ids for k in ordered]
        require(mask==sorted(mask),'out-of-order resume in '+lane)
    external=read(HERE/'preregistration/frozen_inputs.json')
    guard.files[Path(__file__).resolve()]=sha(Path(__file__).resolve())
    free=shutil.disk_usage(ROOT.anchor).free
    base=dict(utc=now(),storage_inventory_sha256=INVENTORY_SHA,original_intent_seal=sha(HERE/'sealed_intents.json'),
              source_and_input_guard_files=len(guard.files),frozen_external_source_asset_hashes=len(external),
              audited_source_and_protocol_paths=frozen,prior_results_and_receipts=done,prior_240_checkpoint_sha256=trail,
              approved_pending_once=[t['trial_id'] for t in pending],passed_preflight=True,verified_intents=720,
              previous_adjudicated=265,pending=455,workers=list(LANES),source=guard.files[Path(__file__).resolve()],
              native_engine_unchanged=True,no_trial_replay=True,exclusive_lane_locks_checked=True,volume=ROOT.drive,free_bytes=free,
              warning_strictly_below_bytes=150*GIB,pause_strictly_below_bytes=120*GIB,drain_on_pause=True,
              archive='one transferable lightweight core only; all native evidence remains in place')
    guard.files[CONTROL/'preflight.json']=put(CONTROL/'preflight.json',base)
    return guard,done,trail,pending,lanes,sealed['files']

class DiskControl:
    def __init__(self):
        self.warning=False;self.latched=False;self.events=0;self.minimum=None;self.last=None
    def sample(self):
        free=shutil.disk_usage(ROOT.anchor).free;self.last=free
        self.minimum=free if self.minimum is None else min(self.minimum,free)
        level=band(free)
        if level!='normal' and not self.warning:
            self.warning=True;self.event('warning_below_150_GiB',free)
        elif level=='normal' and self.warning:
            self.warning=False;self.event('recovered_to_at_least_150_GiB',free)
        if level=='pause' and not self.latched:
            self.latched=True;self.event('pause_below_120_GiB_drain_existing',free)
        return free
    def event(self,name,free):
        self.events+=1
        put(CONTROL/'disk_events'/f'{self.events:04d}_{name}.json',dict(event=name,utc=now(),volume=ROOT.drive,free_bytes=free,free_GiB=free/GIB))
        print(name, 'available GiB', round(free/GIB,3), flush=True)

def dispatch(guard, done, trail, pending, owners, sealed):
    queues={lane:deque(t for t in pending if owners[t['unit_id']]==lane) for lane in LANES}
    live={};faults=[];disk=DiskControl();manual=False;launched=[];timer=time.monotonic();last_status=0.0
    def pending_rows():
        return [t for q in queues.values() for t in q]
    def status(stage):
        row=dict(stage=stage,utc=now(),planned=720,completed=len(done),started=265+len(launched),
            newly_started_since_audit=len(launched),unlaunched=len(pending_rows()),active={a:t for a,(t,_) in live.items()},
            free_bytes=disk.last,minimum_seen_bytes=disk.minimum,disk_warn=disk.warning,disk_paused=disk.latched,
            manual_pause=manual,faults=list(faults),latest_checkpoint=trail)
        with (CONTROL/'progress.jsonl').open('a',encoding='utf8',newline='\n') as f:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
    def register(kind,exc,key=None):
        faults.append(dict(kind=kind,trial=key,error=str(exc),utc=now()))
    def remaining():
        return sum(len(q) for q in queues.values())
    disk.sample();status('resumed_after_storage_audit')
    while remaining() or live:
        changed=False
        for lane,(key,p) in list(live.items()):
            rc=p.poll()
            if rc is None:continue
            del live[lane];changed=True
            try:
                if rc or not (JOBS/key/'return.json').is_file():
                    e=JOBS/key/'exception.json'
                    raise RuntimeError({'exit_code':rc,'record':read(e) if e.exists() else 'missing worker return'})
                row=verified_return(key,sealed['trials/'+key+'/intent.json'])
                done.append(row)
                guard.files[HERE/'trials'/key/'method_result.json']=row['method_result_sha256']
                guard.files[JOBS/key/'return.json']=row['return_sha256']
                if len(done)%60==0:
                    trail=checkpoint(done,pending_rows(),live,faults,trail)
            except Exception as exc:register('shared_method_return',exc,key)
        try:
            disk.sample()
            if not faults and time.monotonic()-timer>=40:
                guard.check();timer=time.monotonic()
        except Exception as exc:register('shared_storage_or_integrity',exc)
        if (CONTROL/'PAUSE_REQUEST.json').exists():manual=True
        if not faults and not disk.latched and not manual:
            for lane in LANES:
                if lane in live or not queues[lane]:continue
                t=queues[lane][0];key=t['trial_id']
                try:
                    disk.sample()
                    if disk.latched or (CONTROL/'PAUSE_REQUEST.json').exists():
                        manual=(CONTROL/'PAUSE_REQUEST.json').exists();break
                    require(not (JOBS/key).exists(), 'duplicate assignment '+key)
                    p=method_start(t,lane)
                except Exception as exc:
                    register('shared_dispatch_or_storage',exc,key);break
                queues[lane].popleft();launched.append(key);live[lane]=(key,p);changed=True
        if changed or time.monotonic()-last_status>=30 or ((faults or disk.latched or manual) and not live):
            state='draining_shared_integrity' if faults else 'draining_low_disk' if disk.latched else 'draining_user_pause' if manual else 'formal_methods'
            status(state);last_status=time.monotonic()
        if (faults or disk.latched or manual) and not live:break
        if live:time.sleep(.8)
    if faults or disk.latched or manual:
        stage='stopped_shared_integrity' if faults else 'paused_low_disk' if disk.latched else 'paused_by_user'
        put(CONTROL/'stopped_after_drain.json',dict(stage=stage,utc=now(),completed=len(done),active=0,
            new_dispatches=launched,remaining_ids=[x['trial_id'] for x in pending_rows()],faults=faults,last_free_bytes=disk.last))
        status(stage);return False
    guard.check(True)
    require(len(done)==len({x['trial_id'] for x in done})==720 and len(launched)==len(set(launched))==455,'terminal dispatch cardinality')
    original=read(CONTROL/'preflight.json')
    require(set(launched)==set(original['approved_pending_once']) and set(launched).isdisjoint(x['trial_id'] for x in original['prior_results_and_receipts']),'terminal new/prior attribution')
    put(HERE/'run/method_complete.json',dict(planned=720,returned=720,terminal_checkpoint_sha256=trail,terminal_input_guard=True,utc=now()))
    put(CONTROL/'dispatch_complete.json',dict(utc=now(),original_adjudicated=265,newly_adjudicated=455,
            unique_total=720,actual_dispatch_order_since_resume=launched,active_workers=0,minimum_free_bytes_sampled=disk.minimum,
            disk_events=disk.events,terminal_checkpoint_sha256=trail,baseline_sha256=sha(CONTROL/'preflight.json')))
    status('formal_execution_closed');return True

def finish():
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for stage in ('analyze','delivery'):
        result=subprocess.run([sys.executable,'-B',str(HERE/'code'/(stage+'.py'))],cwd=ROOT,env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        require(result.returncode==0,stage+' returned '+str(result.returncode)+' '+result.stderr.decode('utf8',errors='replace')[-2300:])
    v=read(HERE/'delivery/verification.json')
    require(v['all_included_member_hashes_verified'] and list((HERE/'delivery').glob('*.zip'))==[HERE/v['core_path']]
            and sha(HERE/v['core_path'])==v['core_sha256'],'expected one verified lightweight ZIP')
    print('formal 720 complete;',v['core_path'],v['core_sha256'],flush=True)

def main():
    assert band(150*GIB)=='normal' and band(150*GIB-1)=='warn' and band(120*GIB)=='warn' and band(120*GIB-1)=='pause'
    CONTROL.mkdir(parents=True,exist_ok=True)
    with (CONTROL/'session.lock').open('xb') as f:
        f.write(b'1');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:
            put(CONTROL/'controller_start.json',dict(pid=os.getpid(),utc=now(),scope='original remaining 455; 4 isolated lanes',source_sha256=sha(__file__)))
            ctx=preflight()
            complete=dispatch(*ctx)
        except BaseException as exc:
            put(CONTROL/'controller_error.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()))
            raise
        finally:
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    if complete:
        try:finish()
        except BaseException as exc:
            put(CONTROL/'delivery_error.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()))
            raise
if __name__=='__main__':main()
