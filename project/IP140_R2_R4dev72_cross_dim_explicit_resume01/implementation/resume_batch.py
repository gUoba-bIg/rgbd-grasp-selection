"""Progress from validated inherited inputs to the existing 72-trial authorization, once."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,sys,traceback
from run_trials import commit_ledger,dump,utc
V=Path(__file__).resolve().parents[1];ROOT=V.parent
def read(p):return json.loads(p.read_bytes())
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def execute(label,script,args=()):
    folder=V/'run_evidence';target=folder/(label+'.log')
    with target.open('xb') as f:
        status=subprocess.run([sys.executable,'-B',str(V/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                              env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'))
    if status.returncode:raise RuntimeError(f'{label} returned {status.returncode}; evidence {target.relative_to(V).as_posix()}')
def main():
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('working directory')
    if (V/'STOP.json').exists() or read(V/'development_ledger.json')['method_trials_counted']:raise RuntimeError('terminal or used authorization')
    (V/'run_evidence').mkdir()
    dump(V/'run_evidence/claim.json',dict(utc=utc(),pid=os.getpid(),budget=72,more_than_one_launch=False))
    stage='validate_resume'
    try:
        execute(stage,'implementation/admit_cross_source.py')
        stage='regenerate_pools_and_rankings';execute(stage,'implementation/prepare_paired_inputs.py')
        stage='seal_all_method_intents';execute(stage,'implementation/seal_trials.py')
        stage='execute_frozen_methods';execute(stage,'implementation/run_trials.py')
        print('returned 72-trial dispatch',flush=True)
    except BaseException as exc:
        detail=dict(stage=stage,utc=utc(),exception=repr(exc),traceback=traceback.format_exc())
        dump(V/'run_evidence/stage_failure.json',detail)
        if not (V/'STOP.json').exists():
            ledger=read(V/'development_ledger.json')
            ledger['stage']='stopped_shared_integrity_fault'
            ledger['shared_integrity_faults']=[repr(exc)]
            commit_ledger(ledger)
            dump(V/'STOP.json',dict(reason=str(exc),stage=stage,utc=utc(),shared_integrity_fault=True,
                 method_trials_counted=ledger['method_trials_counted'],remaining=ledger['remaining'],
                 source='run_evidence/'+stage+'.log'))
        raise
if __name__=='__main__':main()
