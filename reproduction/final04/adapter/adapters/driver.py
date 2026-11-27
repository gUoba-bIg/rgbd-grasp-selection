"""One-shot independent reproduction coordinator. All persistent outputs are under the fresh installation."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,hashlib,subprocess
B=Path(__file__).resolve().parents[1]
R6='IP140_R2_R6formal720_static_distance_uncertified_resume01'
ORIG='IP140_R2_R6formal720_raw_a_c0_paired01';PL='IP140_R2_smoke22_miss_negative_controls_full_execution01'
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def put(p,d):
    with p.open('x',encoding='utf8') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')
def go(t):
    assert Path(sys.executable).resolve()==t/'bin/python.exe' and read(t/'setup_complete.json')['isolated']
    required=['vision','rank','physics']
    for role in required:
        pre=read(t/'setup_logs'/('preflight_'+role+'.json'))
        if pre['role']!=role or pre['model_forwards'] or pre['physical_steps']:raise RuntimeError('missing fresh import preflight '+role)
    receipt=t/'cold_receipts';receipt.mkdir(exist_ok=False)
    registry=read(B/'metadata/cold_preregistration.json');p=t/'p';version=p/R6;cwd=version/'workspaces/cpu01';cwd.mkdir(parents=True,exist_ok=False)
    system=Path(os.environ['SYSTEMROOT']);shell=system/'System32/WindowsPowerShell/v1.0/powershell.exe'
    for name in ['data',PL]:
        dst=cwd/name;src=p/name
        assert src.is_dir() and not dst.exists() and src.resolve().is_relative_to(t)
        env=os.environ.copy();env.update(LOCAL_COLD_DIR=str(dst),LOCAL_COLD_SRC=str(src))
        subprocess.run([str(shell),'-NoProfile','-NonInteractive','-Command',"$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:LOCAL_COLD_DIR -Target $env:LOCAL_COLD_SRC | Out-Null"],check=True,env=env,cwd=t)
        assert dst.is_junction() and dst.resolve()==src.resolve()
    put(cwd/'workspace.json',{'private_replay':True,'local_only':str(cwd)})
    binary=t/'bin/python.exe'
    log=receipt/'ledger.jsonl'
    def event(**kw):
        with log.open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(time_utc=utc(),**kw),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    known={'forward':'vision','rank':'rank','method':'physics'}
    event(event='admission',registry_sha256=hashlib.sha256((B/'metadata/cold_preregistration.json').read_bytes()).hexdigest(),cases=registry['order'])
    for c in registry['cases']:
        tid=c['trial_id'];unit=c['unit'];o=receipt/tid;o.mkdir();(version/'run/jobs/method'/tid).mkdir(parents=True)
        for role,alias in known.items():
            seed=c['worker_seed'] if role=='method' else int.from_bytes(hashlib.sha256((ORIG+'|'+role+'|'+unit).encode()).digest()[:4],'big')
            origin=read(B/'metadata/stage_seeds.json')[tid][role]
            if seed!=origin:raise ValueError('original job seed '+role)
            scratch=o/(role+'_temp');scratch.mkdir()
            e={k:v for k,v in os.environ.items() if k.upper() in ('WINDIR','SYSTEMROOT','COMSPEC','SYSTEMDRIVE','NUMBER_OF_PROCESSORS','PROCESSOR_ARCHITECTURE','PATHEXT','COMPUTERNAME','USERNAME')}
            for d in ('LOCALAPPDATA','APPDATA','USERPROFILE','HOME'):
                dd=t/'user'/d;dd.mkdir(exist_ok=True,parents=True);e[d]=str(dd)
            e.update(PATH=os.pathsep.join(map(str,[t/'env'/alias/'Scripts',t/'bin',t/'bin/DLLs',system/'System32',system])),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONHASHSEED=str(seed),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch),CUDA_CACHE_PATH=str(scratch/'cuda'),TORCH_EXTENSIONS_DIR=str(scratch/'torch'),MPLCONFIGDIR=str(scratch/'mpl'),NUMBA_CACHE_DIR=str(scratch/'numba'))
            if role!='forward':e['CUDA_VISIBLE_DEVICES']=''
            cmd=[str(t/'env'/alias/'Scripts/python.exe'),'-B','-s',str(B/'adapters/cold_stage.py'),str(t),role,tid]
            put(o/(role+'_invocation.json'),dict(command=cmd,cwd=str(cwd),environment=e,seed=seed))
            event(event='stage_dispatch',trial=tid,role=role)
            with (o/(role+'_stdout.log')).open('xb') as so,(o/(role+'_stderr.log')).open('xb') as se:
                code=subprocess.run(cmd,env=e,cwd=cwd,stdout=so,stderr=se).returncode
            event(event='stage_return',trial=tid,role=role,code=code)
            if code or not (o/(role+'_return.json')).is_file():
                put(receipt/'STOP.json',dict(time_utc=utc(),trial=tid,stage=role,exit_code=code,stderr=(o/(role+'_stderr.log')).read_text('utf8',errors='replace'),next_actions='Stop V2; reserve evidence. No automatic reruns.'))
                raise RuntimeError(f'Cold gate stopped at {tid} / {role}.')
    put(receipt/'complete.json',dict(utc=utc(),passed=True,trials=registry['order']))
if __name__=='__main__':go(Path(sys.argv[1]).resolve())
