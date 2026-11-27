"""Execute a sealed development schedule exactly once and preserve unknown returns."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

V=Path(__file__).resolve().parents[1]
ROOT=V.parent
LEDGER=V/'development_ledger.json'

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,obj,mode='x'):
    with p.open(mode,encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
def commit_ledger(data):
    tmp=V/'ledger_transaction.tmp'
    dump(tmp,data)
    os.replace(tmp,LEDGER)
def event(**fields):
    with (V/'trial_events.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(dict(utc=utc(),**fields),ensure_ascii=False,allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
class Guard:
    def __init__(self,expected):
        self.expected=expected
        self.fingerprints={}
    def check(self,force=False):
        for p,expected in self.expected.items():
            stat=p.stat()
            fingerprint=(stat.st_size,stat.st_mtime_ns)
            if force or self.fingerprints.get(p)!=fingerprint:
                if digest(p)!=expected:raise ValueError('sealed input changed: '+str(p))
                self.fingerprints[p]=fingerprint
def main():
    if Path.cwd().resolve()!=ROOT:raise ValueError('working directory')
    ledger=read(LEDGER)
    if ledger['method_trials_counted'] or ledger['budget']!=72:raise RuntimeError('already counted or wrong budget')
    seal=read(V/'sealed_intents.json')
    assert seal['trial_results_at_seal']==0 and seal['intent_count']==72
    terminal_path=V/"audit/terminal_intent_admission.json"
    terminal=read(terminal_path)
    if not (terminal["passed"] is True and terminal["trial_results_before_admission"]==0 and terminal["verified_trial_intents"]==72
            and terminal["verified_triplets"]==24 and terminal["sealed_intents_sha256"]==digest(V/"sealed_intents.json")):
        raise RuntimeError("terminal_intent_admission_required_before_first_trial")
    order=seal['balanced_order']
    assert len(order)==len(set(order))==72 and order==[row['trial_id'] for row in ledger['trials']]
    paths={V/k:v for k,v in (seal['inputs']|seal['files']).items()}
    for k,m in read(V/'audit/input_files.json')['files'].items():
        paths[ROOT/k]=m['sha256']
    paths[terminal_path]=digest(terminal_path)
    paths[V/'sealed_intents.json']=digest(V/'sealed_intents.json')
    g=Guard(paths)
    audit=dict(utc=utc(),pid=os.getpid(),schedule_sha256=digest(V/'sealed_intents.json'),ledger_sha256_before=digest(LEDGER))
    dump(V/'development_run_claim.json',audit)
    expected_state={}
    try:
        g.check(force=True)
        for row in ledger['trials']:
            path=V/'trials'/row['trial_id']/'intent.json'
            intent=read(path)
            assert all(intent[k]==row[k] for k in ('trial_id','unit_id','method'))
            row['execution_intent_sha256']=seal['files'][path.relative_to(V).as_posix()]
            common=(intent['pool_sha256'],intent['stable_state_sha256'],intent['selection_sha256'])
            if expected_state.setdefault(row['unit_id'],common)!=common:raise ValueError('unpaired method')
        if len(expected_state)!=24:raise ValueError('missing paired units')
        ledger['stage']='sealed_scheduled'
        commit_ledger(ledger)
        for row in ledger['trials']:
            g.check()
            dest=V/'trials'/row['trial_id'];intent=dest/'intent.json'
            dump(dest/'claim.json',dict(trial_id=row['trial_id'],utc=utc(),intent_sha256=digest(intent),attempt=1))
            row['counted']=True
            row['state']='started_awaiting_worker'
            ledger['method_trials_counted']+=1
            ledger['remaining']=72-ledger['method_trials_counted']
            ledger['stage']='running'
            commit_ledger(ledger)
            event(trial_id=row['trial_id'],action='dispatch',counted=True)
            env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
            with (dest/'worker.log').open('xb') as f:
                run=subprocess.run([sys.executable,'-B',str(V/'implementation/platform_trial.py'),str(intent)],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            resultpath=dest/'method_result.json'
            if not resultpath.is_file():raise RuntimeError('unknown worker return '+row['trial_id']+' exit='+str(run.returncode))
            result=read(resultpath)
            row['result_sha256']=digest(resultpath)
            if result['trial_id']!=row['trial_id'] or result.get('negative_control_passed','absent') is not None:
                raise RuntimeError('worker identity/field mismatch '+row['trial_id'])
            if result['shared_fault'] or run.returncode:
                raise RuntimeError('shared worker fault '+row['trial_id']+' '+str(result.get('failure_detail') or result.get('failure_reason') or result.get('reason') or result.get('failure_category') or run.returncode))
            for k in ('trial_completed','engineering_gate_passed','grasp_success'):
                if type(result[k]) is not bool:raise RuntimeError('unadjudicated '+row['trial_id']+' '+k)
                row[k]=result[k]
            if row['grasp_success'] and not (row['engineering_gate_passed'] and row['trial_completed']):raise ValueError('success without frozen gates')
            row.update(state='adjudicated',failure_class=result.get('failure_category'),physical_steps=result['physical_steps'])
            ledger['trial_results_available']+=1
            commit_ledger(ledger)
            event(trial_id=row['trial_id'],action='adjudicated',result_sha256=row['result_sha256'])
            print(row['trial_id'],row['state'],row['grasp_success'],flush=True)
        g.check(force=True)
        ledger['stage']='complete_72'
        commit_ledger(ledger)
    except BaseException as error:
        for row in ledger['trials']:
            if row['state']=='started_awaiting_worker':row['state']='shared_fault_outcome_unknown'
        ledger['stage']='stopped_shared_integrity_fault'
        ledger['shared_integrity_faults']=[str(error)]
        commit_ledger(ledger)
        dump(V/'STOP.json',dict(reason=str(error),utc=utc(),shared_integrity_fault=True,method_trials_counted=ledger['method_trials_counted']))
        raise
if __name__=='__main__':main()
