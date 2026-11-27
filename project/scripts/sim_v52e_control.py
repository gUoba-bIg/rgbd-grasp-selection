"""Freeze and run one logged zero-step interface admission in the isolated build."""
import sys,os,json,hashlib,uuid,subprocess,time,argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sim_v52_common import read,write,identity,relative,now,append,events,lock
LOG=ROOT/'logs/simulation_v52e';OUT=ROOT/'results/simulation_v52e'
CFG=ROOT/'configs/simulation_v52e.json';STATE=ROOT/'runtime/pipeline/simulation_v52e.json'
WORKER=ROOT/'scripts/sim_v52e_zero_step.py';PYTHON=ROOT/'runtime/simulation_v52e/venv/Scripts/python.exe'

def history():
    data=read(LOG/'historical_snapshot.json')['protected']
    bad=[p for p,x in data.items() if not (ROOT/p).is_file() or identity(ROOT/p)!=x]
    if bad:raise RuntimeError('Historical changes: '+repr(bad[:10]))
    return len(data)

def status(phase,physics_steps=0,**fields):
    write(STATE,dict(at=now(),phase=phase,physics_steps=physics_steps,M1_implemented=False,new_12_started=False,new48_started=False,
        R4_authorized=False,automatic_retry=False,old_scope_all_tasks_percent=45,old_scope_experiment_percent=56.25,
        expanded_scope_percent=None,**fields))

def freeze():
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52e.lock'):
        if CFG.exists():raise RuntimeError('No frozen config overwrite')
        assert read(LOG/'build_result.json')['exit_code']==0
        install=read(LOG/'install_result.json');assert install['exit_code']==0 and len(install['extensions'])==1
        assert read(LOG/'independent_source_review_v2.json')['passed']
        history()
        base=read(ROOT/'configs/simulation_v52r2.json')
        assert base['physics']['gripper_torque']==10 and base['physics']['gear_force']==80 and base['criteria']['mimic_error_rad']==.03
        extension=next(iter(install['extensions']))
        paths={PYTHON,ROOT/extension,WORKER,Path(__file__),ROOT/'configs/simulation_v52r2.json',ROOT/base['robot_manifest'],
            ROOT/base['robot_urdf'],LOG/'authorization.json',LOG/'build_result.json',LOG/'install_result.json',
            LOG/'independent_source_review_v2.json',LOG/'patch/patch_manifest.json',LOG/'patch/parameter_route.patch',LOG/'patch/build_metadata.patch'}
        for p in (ROOT/'data/simulation_v52/robot').rglob('*'):
            if p.is_file():paths.add(p)
        patch=read(LOG/'patch/patch_manifest.json');src=ROOT/'runtime/simulation_v52e/src/pybullet-3.2.7'
        for row in patch['modified_files']:
            p=src/row['path'];assert identity(p)['sha256']==row['after']['sha256'];paths.add(p)
        cfg=dict(schema='v52e_zero_step_v1',at=now(),execution_authorized=True,python=relative(PYTHON),pybullet_extension=extension,
            expected_distribution_version='3.2.7+v52e1',base_config='configs/simulation_v52r2.json',robot_urdf=base['robot_urdf'],robot_manifest=base['robot_manifest'],
            targets=dict(lateralFriction=1.,collisionMargin=.0005,restitution=0.),max_connections=1,max_physics_steps=0,
            limits=dict(wall_seconds=120,rss_bytes=6442450944),unchanged_caps=dict(master_Nm=10,gear_Nm=80,mimic_rad=.03),
            bindings={relative(p):identity(p) for p in sorted(paths)})
        write(CFG,cfg,once=True);write(LOG/'freeze.json',dict(at=now(),config=identity(CFG),acceptance_starts=0,physics_steps=0),once=True)
        status('frozen_zero_step_admission_ready',acceptance_starts=0)
        print(json.dumps(dict(frozen=True,bindings=len(paths),physics_steps=0)))

def verify(cfg):
    assert identity(CFG)==read(LOG/'freeze.json')['config']
    for p,x in cfg['bindings'].items():assert identity(ROOT/p)==x,p

def run():
    import psutil
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52e.lock'):
        cfg=read(CFG);verify(cfg);history()
        ledger=LOG/'acceptance_attempts.jsonl'
        if ledger.exists():raise RuntimeError('Already charged; no retry or duplicate run')
        for p in psutil.process_iter(['pid','name','cmdline']):
            cmd=' '.join(p.info.get('cmdline') or [])
            if p.pid!=os.getpid() and (p.info.get('name') or '').lower().startswith('python') and ('sim_v52e_zero_step.py' in cmd or 'sim_v52d_episode.py' in cmd):
                raise RuntimeError('Another acceptance/physics worker active')
        attempt=uuid.uuid4().hex;token=uuid.uuid4().hex+uuid.uuid4().hex
        directory=OUT/'attempts'/('001_ZERO001_'+attempt[:8]);directory.mkdir(parents=True,exist_ok=False)
        charge=dict(schema='sim_v52e_precharge_v1',at=now(),attempt_id=attempt,unit_id='ZERO001',charged_index=1,
            token_hash=hashlib.sha256(token.encode()).hexdigest(),config_identity=identity(CFG),worker_identity=identity(WORKER))
        write(directory/'precharge.json',charge,once=True)
        command=[str(PYTHON),'-B',relative(WORKER),'--config',relative(CFG),'--attempt-dir',relative(directory)]
        intent=dict(event='intent',at=now(),attempt_id=attempt,unit_id='ZERO001',charged_index=1,directory=relative(directory),
            precharge=identity(directory/'precharge.json'),config=identity(CFG),worker=identity(WORKER),command=command)
        append(ledger,intent);write(directory/'intent.json',intent,once=True)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',SIM_V52E_CONTROLLER_TOKEN=token,
            TEMP=str(ROOT/'runtime/simulation_v52e/temp'),TMP=str(ROOT/'runtime/simulation_v52e/temp'))
        start=time.monotonic();child=None;termination=None;code=None;peak=0
        try:
            with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                status('zero_step_admission_running',acceptance_starts=1,child_pid=child.pid)
                while child.poll() is None:
                    tree=[]
                    try:
                        parent=psutil.Process(child.pid);tree=[parent]+parent.children(recursive=True)
                        peak=max(peak,sum(p.memory_info().rss for p in tree))
                    except psutil.Error:pass
                    if time.monotonic()-start>120 or peak>cfg['limits']['rss_bytes'] or (LOG/'STOP').exists():
                        termination='resource_or_explicit_stop'
                        for p in reversed(tree):
                            try:p.kill()
                            except psutil.Error:pass
                        if child.poll() is None:child.kill()
                        break
                    time.sleep(.1)
                code=child.wait(timeout=10)
        except BaseException as exc:
            termination=repr(exc)
            if child is not None and child.poll() is None:child.kill();child.wait(timeout=10)
        result_path=directory/'result.json';result=read(result_path) if result_path.exists() else None
        known=bool(result and result.get('attempt_id')==attempt and result.get('config_identity')==intent['config'] and result.get('worker_identity')==intent['worker'])
        ret=dict(event='returned',at=now(),attempt_id=attempt,unit_id='ZERO001',charged_index=1,exit_code=code,termination=termination,
            wall_seconds=time.monotonic()-start,peak_sampled_tree_rss_bytes=peak,result_path=relative(result_path),
            result=identity(result_path) if result_path.exists() else None,known_result=known)
        append(ledger,ret);write(directory/'resource_result.json',ret,once=True)
        passed=bool(known and code==0 and termination is None and result['admission_passed'] and result['actual_physics_steps']==0 and not result['unknown_calls'])
        verify(cfg);protected=history()
        write(OUT/'execution_summary.json',dict(at=now(),parameter_admission_passed=passed,charged_starts=1,known_result=known,
            unknown_result=not known,actual_physics_steps=0 if known else None,step_status='known_zero' if known else 'unknown_result_retained',
            result_path=relative(result_path),resources=ret,protected_files_unchanged=protected,
            M1_implemented=False,new12_started=False,new48_started=False,R4_started=False,review_required=True),once=True)
        status('zero_step_complete_stopped_for_review' if passed else 'zero_step_failed_stopped_for_review',
            physics_steps=0 if known else None,step_status='known_zero' if known else 'unknown_result_retained',
            acceptance_starts=1,parameter_admission_passed=passed,known_result=known,result_path=relative(result_path),review_required=True)
        print(json.dumps(dict(passed=passed,known_result=known,physics_steps=0 if known else None,result_path=relative(result_path))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.command]()
