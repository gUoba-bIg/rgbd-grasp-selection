"""One-shot, precharged 12-unit M0 diagnostic controller. No resume/retry command."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sim_v52_common import ROOT,read,write,identity,relative,now,append,events,lock

LOG=ROOT/'logs/simulation_v52f'
OUT=ROOT/'results/simulation_v52f'
CONFIG=ROOT/'configs/simulation_v52fr1.json'
STATE=ROOT/'runtime/pipeline/simulation_v52f.json'
LEDGER=LOG/'attempts.jsonl'
PYTHON=ROOT/'runtime/simulation_v52e/venv/Scripts/python.exe'
WORKER=ROOT/'scripts/sim_v52fr1_episode.py'
PLAN=ROOT/'results/simulation_v52_dynamics_audit/proposed_diagnostic_plan.json'


def once_sync(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def status(phase,**data):
    write(STATE,dict(at=now(),phase=phase,model='M0',authorized_attempts=12,
        user_authorization='User approved exactly twelve repaired-interface M0 diagnostics using isolated v52e engine; stop for review',
        old_v52_attempts=48, M1_implemented=False, R4_authorized=False,
        automatic_retries=False, automatic_expansion=False, recurring_automation=False,
        all_tasks_percent_legacy=45,experiment_percent_legacy=56.25,
        expanded_scope_percent=None,**data))


def historical_verify():
    data=read(LOG/'historical_snapshot.json')['protected']
    bad=[name for name,value in data.items() if not (ROOT/name).is_file() or identity(ROOT/name)!=value]
    if bad:raise RuntimeError('Historical files changed: '+repr(bad[:15]))
    return len(data)


def authorize():
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52f.lock'):
        if (LOG/'authorization.json').exists():raise RuntimeError('Already authorized; no duplicate preparation')
        from scripts.sim_v52e_control import history
        history()
        protected=dict(read(ROOT/'logs/simulation_v52e/historical_snapshot.json')['protected'])
        prior=read(ROOT/'results/simulation_v52e/delivery_closeout_revision1.json')
        for name,value in prior['artifacts'].items():
            if name=='docs/实验自动续接说明.txt':continue
            assert identity(ROOT/name)==value,name
            protected[name]=value
        for directory in ('runtime/simulation_v52e','logs/simulation_v52e','results/simulation_v52e',
                          'logs/simulation_v52d','results/simulation_v52d'):
            for path in (ROOT/directory).rglob('*'):
                if path.is_file():protected[relative(path)]=identity(path)
        for pattern in ('scripts/*v52d*.py','scripts/*v52e*.py','configs/*v52e*.json','docs/v52e*.txt','docs/v52d*.txt'):
            for path in ROOT.glob(pattern):
                if path.is_file():protected[relative(path)]=identity(path)
        for name in ('runtime/pipeline/simulation_v52_dynamics_run.json','runtime/pipeline/simulation_v52e.json'):
            protected[name]=identity(ROOT/name)
        once_sync(LOG/'historical_snapshot.json',dict(at=now(),protected=protected,
            live_handoff_exception='Prefix only with entire prior bytes preserved'))
        once_sync(LOG/'authorization.json',dict(at=now(),authorized=True,model='M0',attempt_cap=12,
            user_instruction='Exactly original twelve N/L/C combinations under repaired isolated v52e engine; read back pads before each physics step; mismatch or runtime failure stops; no retries or expansion.',
            plan=identity(PLAN),new_48_authorized=False,R4_72_authorized=False,formal_720_authorized=False,
            ablation_240_authorized=False,M1_authorized=False,old_study_results_retained=True,physics_started=0))
        depdir=ROOT/'runtime/simulation_v52f/deps'
        depdir.mkdir(parents=True,exist_ok=False)
        oldsite=ROOT/'runtime/simulation_v52/venv/Lib/site-packages'
        for name in ('scipy','scipy.libs','scipy-1.15.3.dist-info'):
            shutil.copytree(oldsite/name,depdir/name,copy_function=shutil.copy2)
        files={relative(p):identity(p) for p in sorted(depdir.rglob('*')) if p.is_file()}
        for path,value in files.items():
            suffix=(ROOT/path).relative_to(depdir)
            assert identity(oldsite/suffix)==value,path
        once_sync(ROOT/'runtime/simulation_v52f/dependency_manifest.json',dict(at=now(),version='scipy1.15.3',
            source='Unchanged original simulation_v52 scipy files copied to a v52f-only import directory',files=files))
        for name in ('temp','cache'):(ROOT/'runtime/simulation_v52f'/name).mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'docs/实验自动续接说明.txt',LOG/'resume_before_v52f.txt')
        status('implementation_and_preflight',attempts_charged=0,actual_physics_steps_returned=0)
        print(json.dumps(dict(authorized=True,protected=len(protected),dependency_files=len(files),attempts_charged=0)))


def source_bindings():
    prior=read(ROOT/'configs/simulation_v52d.json')
    paths={ROOT/name for name in prior['bindings']}
    paths.update((ROOT/'scripts').glob('sim_v52f_*.py'))
    paths.update((ROOT/'scripts').glob('sim_v52fr1_*.py'))
    paths.update([ROOT/'configs/simulation_v52f.json',LOG/'freeze.json',ROOT/'logs/simulation_v52f/reviews/preflight_20261103_160413_1793693053913924000.json'])
    paths.update([LOG/'authorization.json',ROOT/'configs/simulation_v52d.json',PYTHON,
        ROOT/'runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd',
        ROOT/'runtime/simulation_v52f/dependency_manifest.json',ROOT/'docs/v52f_修复后M0十二格复验协议.txt',
        ROOT/'scripts/sim_v52_math.py',ROOT/'runtime/simulation_v52e/venv/Lib/site-packages/pybullet_utils/bullet_client.py',
        ROOT/'results/simulation_v52e/attempts/001_ZERO001_2b433b45/afterR.json',
        ROOT/'logs/simulation_v52e/install_result.json',ROOT/'logs/simulation_v52e/independent_readback_review_v2.json'])
    return {relative(p):identity(p) for p in sorted(paths)}


def draft():
    historical_verify()
    plan=read(PLAN);auth=read(LOG/'authorization.json')
    assert auth['authorized'] and auth['plan']==identity(PLAN)
    base=read(ROOT/'configs/simulation_v52r2.json')
    config=dict(schema='sim_v52f_execution_v1',at=now(),execution_authorized=True,model='M0',
        base_config='configs/simulation_v52r2.json',plan_path=relative(PLAN),
        python=relative(PYTHON),expected_distribution_version='3.2.7+v52e1',
        pybullet_extension='runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd',
        dependency_dir='runtime/simulation_v52f/deps',dependency_manifest='runtime/simulation_v52f/dependency_manifest.json',
        parameter_reference_readback='results/simulation_v52e/attempts/001_ZERO001_2b433b45/afterR.json',
        parameter_targets=dict(lateralFriction=1.0,collisionMargin=.0005,restitution=0.),
        action_path=plan['fixed']['action_input'],combinations=plan['combinations'],
        resources=plan['resources'],criteria=base['criteria'],
        authorization_path=relative(LOG/'authorization.json'),
        proposed_file_flags_are_historical=True,
        current_authority='This explicit new authorization supersedes only the no-run flag for the approved twelve-unit plan; it does not alter historical files.',
        bindings=source_bindings())
    path=LOG/'preparation'/('config_draft_'+uuid.uuid4().hex[:8]+'.json')
    once_sync(path,config)
    print(relative(path))


def freeze(draft_path,review_path):
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52f.lock'):
        if CONFIG.exists():raise RuntimeError('Already frozen; no overwrite')
        cfg=read(draft_path);review=read(review_path)
        assert review['review_passed'] and review['config_identity']==identity(draft_path)
        assert review.get('new_physics_steps',review.get('physics_steps',None))==0
        for name,value in cfg['bindings'].items():assert identity(ROOT/name)==value,name
        historical_verify()
        shutil.copy2(draft_path,CONFIG)
        once_sync(LOG/'freeze_r1.json',dict(at=now(),config=identity(CONFIG),
            review_path=relative(review_path),review=identity(review_path),
            draft_path=relative(draft_path),attempts_started=0))
        status('frozen_ready',attempts_charged=0,config=identity(CONFIG))
        print('Frozen; no diagnostic worker launched by freeze.')


def verify_frozen(cfg):
    f=read(LOG/'freeze_r1.json')
    assert identity(CONFIG)==f['config']
    assert identity(ROOT/f['review_path'])==f['review']
    for name,value in cfg['bindings'].items():assert identity(ROOT/name)==value,name
    plan=read(PLAN)
    assert cfg['combinations']==plan['combinations'] and cfg['resources']==plan['resources']
    assert cfg['model']=='M0' and cfg['execution_authorized'] is True
    assert cfg['criteria']['mimic_error_rad']==.03
    assert cfg['python']==relative(PYTHON) and cfg['expected_distribution_version']=='3.2.7+v52e1'
    assert cfg['parameter_targets']==dict(lateralFriction=1.0,collisionMargin=.0005,restitution=0.)
    for path,value in read(ROOT/cfg['dependency_manifest'])['files'].items():assert identity(ROOT/path)==value,path


def valid_result(intent,return_event):
    path=ROOT/intent['directory']/'result.json'
    if return_event is None or not path.exists() or return_event.get('result')!=identity(path):return None
    try:r=read(path)
    except (ValueError,OSError):return None
    if r.get('unit_id')!=intent['unit_id'] or r.get('attempt_id')!=intent['attempt_id']:return None
    if r.get('config_identity')!=intent['config'] or r.get('source_identity')!=intent['worker']:return None
    if r.get('status') not in ('completed','failed'):return None
    return r


def ledger_rows():
    ee=events(LEDGER);ii=[x for x in ee if x['event']=='intent'];rr=[x for x in ee if x['event']=='returned']
    assert len({x['attempt_id'] for x in ii})==len(ii)
    assert len({x['unit_id'] for x in ii})==len(ii)
    assert len({x['attempt_id'] for x in rr})==len(rr)
    assert len(ii)<=12
    rmap={x['attempt_id']:x for x in rr};assert set(rmap).issubset({x['attempt_id'] for x in ii})
    rows=[]
    for i in ii:
        ret=rmap.get(i['attempt_id']);value=valid_result(i,ret)
        rows.append(dict(intent=i,returned=ret,result=value,unknown=value is None))
    return rows


def stop_child(child):
    import psutil
    descendants=[]
    try:descendants=psutil.Process(child.pid).children(recursive=True)
    except psutil.Error:pass
    for p in reversed(descendants):
        try:p.kill()
        except psutil.Error:pass
    if child.poll() is None:child.kill()
    child.wait(timeout=20)
    survivors=[]
    for p in descendants:
        try:
            if p.is_running():survivors.append(p.pid)
        except psutil.Error:pass
    return survivors


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def run():
    import psutil
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52f.lock'):
        cfg=read(CONFIG);verify_frozen(cfg);historical_verify()
        if events(LEDGER) or (LOG/'session.json').exists():
            raise RuntimeError('This session is one-shot. Existing charged/session record forbids restart or automatic continuation.')
        # No overlapping physical runner, regardless of stale status files.
        for p in psutil.process_iter(['pid','name','cmdline']):
            if p.pid==os.getpid():continue
            cmd=' '.join(p.info.get('cmdline') or [])
            if (p.info.get('name') or '').lower().startswith('python') and any(x in cmd for x in ('sim_v52fr1_episode.py','sim_v52f_episode.py','sim_v52d_episode.py','sim_v52e_zero_step.py','sim_v52r2_episode.py','sim_v52r2_control.py')):
                raise RuntimeError('Another project physics process is active: '+str(p.pid))
        start=time.monotonic();cumulative=0.;returned_count=0
        once_sync(LOG/'session.json',dict(at=now(),controller_pid=os.getpid(),config=identity(CONFIG),attempt_cap=12,
            restart_authorized=False,plan_rows=12))
        phase='completed_stopped_for_review'
        for number,unit in enumerate(cfg['combinations'],1):
            res=cfg['resources']
            if (LOG/'STOP').exists():phase='stop_requested';break
            if time.monotonic()-start>=res['total_session_wall_cap_s'] or cumulative>=res['total_worker_wall_cap_s']:
                phase='budget_exhausted_stopped';break
            if shutil.disk_usage(ROOT).free<res['minimum_free_disk_bytes']:
                phase='insufficient_disk_stopped';break
            if directory_bytes(OUT)+directory_bytes(LOG)>res['output_cap_bytes']:
                phase='output_budget_stopped';break
            attempt_id=uuid.uuid4().hex;token=uuid.uuid4().hex+uuid.uuid4().hex
            directory=OUT/'attempts'/f'{number:03d}_{unit["id"]}_{attempt_id[:8]}'
            directory.mkdir(parents=True,exist_ok=False)
            charge=dict(schema='sim_v52f_precharge_v1',at=now(),attempt_id=attempt_id,charged_index=number,
                unit_id=unit['id'],unit=unit,token_hash=hashlib.sha256(token.encode()).hexdigest(),
                config_identity=identity(CONFIG),worker_identity=identity(WORKER))
            once_sync(directory/'precharge.json',charge)
            command=[str(PYTHON),'-B',relative(WORKER),'--unit',unit['id'],'--attempt-dir',relative(directory),'--config',relative(CONFIG)]
            intent=dict(event='intent',at=now(),attempt_id=attempt_id,charged_index=number,unit_id=unit['id'],unit=unit,
                directory=relative(directory),command=command,config=identity(CONFIG),worker=identity(WORKER),
                precharge=identity(directory/'precharge.json'))
            append(LEDGER,intent)
            once_sync(directory/'intent.json',intent)
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',
                SIM_V52F_CONTROLLER_TOKEN=token,PYTHONPATH=str(ROOT/cfg['dependency_dir']),
                TEMP=str(ROOT/'runtime/simulation_v52f/temp'),TMP=str(ROOT/'runtime/simulation_v52f/temp'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            child=None;termination=None;exit_code=None;peak=0;cpu=0.;begin=time.monotonic();last_size=0.
            try:
                with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                    status('diagnostics_running',controller_pid=os.getpid(),child_pid=child.pid,
                        unit_id=unit['id'],attempts_charged=number,results_returned=returned_count)
                    while child.poll() is None:
                        elapsed=time.monotonic()-begin;rss=0
                        try:
                            root=psutil.Process(child.pid);tree=[root]+root.children(recursive=True)
                            rss=sum(p.memory_info().rss for p in tree)
                            cpu=max(cpu,sum(p.cpu_times().user+p.cpu_times().system for p in tree));peak=max(peak,rss)
                        except psutil.Error:pass
                        reason=None
                        if elapsed>res['per_run_wall_cap_s']:reason='worker_wall_budget'
                        if cumulative+elapsed>res['total_worker_wall_cap_s']:reason='total_worker_wall_budget'
                        if time.monotonic()-start>res['total_session_wall_cap_s']:reason='session_wall_budget'
                        if rss>res['tree_RSS_cap_bytes']:reason='RSS_budget'
                        if (LOG/'STOP').exists():reason='stop_requested'
                        if elapsed-last_size>=1:
                            last_size=elapsed
                            if directory_bytes(OUT)+directory_bytes(LOG)>res['output_cap_bytes']:reason='output_budget'
                            if shutil.disk_usage(ROOT).free<res['minimum_free_disk_bytes']:reason='free_disk_floor'
                        if reason:
                            termination=dict(reason=reason,survivors=stop_child(child));break
                        time.sleep(.25)
                    exit_code=child.wait(timeout=20)
            except BaseException as exc:
                termination=dict(reason='controller_or_launch_exception',exception=repr(exc),traceback=traceback.format_exc(),
                    survivors=stop_child(child) if child is not None else [])
                exit_code=child.returncode if child is not None else None
            elapsed=time.monotonic()-begin;cumulative+=elapsed
            path=directory/'result.json'
            ret=dict(event='returned',at=now(),attempt_id=attempt_id,charged_index=number,unit_id=unit['id'],
                exit_code=exit_code,termination=termination,wall_seconds=elapsed,peak_tree_rss_bytes=peak,
                polled_cpu_seconds_lower_bound=cpu,resource_sampling_period_s=.25,
                result=identity(path) if path.exists() else None,result_path=relative(path),
                resources_path=relative(directory/'resources.json'),result_status='unknown_result')
            value=valid_result(intent,ret)
            if value is not None:ret['result_status']='known_result';returned_count+=1
            append(LEDGER,ret);once_sync(directory/'resources.json',ret)
            print(json.dumps(dict(unit=unit['id'],charged=number,status=value.get('status') if value else 'unknown',
                actual_steps=value.get('actual_physics_steps') if value else None,wall_seconds=elapsed)),flush=True)
            if termination or exit_code!=0 or value is None or value['status']!='completed' or not value.get('telemetry_complete',False) or not value.get('parameter_assertions_passed',False):
                phase='failure_or_unknown_preserved_stopped';break
        report(phase,session_wall=time.monotonic()-start)


def report(phase=None,session_wall=None):
    cfg=read(CONFIG);verify_frozen(cfg)
    rows=ledger_rows()
    complete=(len(rows)==12 and not any(r['unknown'] for r in rows)
        and all(r['result']['status']=='completed' and r['result'].get('telemetry_complete',False) and r['result'].get('parameter_assertions_passed',False)
            and r['result']['actual_physics_steps']==r['intent']['unit']['maximum_step_count']
            and r['returned']['exit_code']==0 and r['returned']['termination'] is None for r in rows))
    known=[r for r in rows if r['result'] is not None]
    totals=dict(at=now(),model='M0',diagnostic_complete=complete,charged_attempts=len(rows),attempt_cap=12,
        results_returned=len(known),completed_results=sum(r['result']['status']=='completed' for r in known),
        unknown_attempts=[r['intent']['attempt_id'] for r in rows if r['unknown']],
        actual_physics_steps_returned=sum(r['result']['actual_physics_steps'] for r in known),
        exact_total_steps_known=not any(r['unknown'] or r['returned'].get('termination')
            or r['result'].get('unknown_step_call') is not None
            or r['result'].get('physics_step_calls_started')!=r['result'].get('physics_step_calls_returned') for r in rows),
        total_worker_wall_seconds=sum(r['returned']['wall_seconds'] for r in rows if r['returned']),
        total_polled_cpu_seconds_lower_bound=sum(r['returned']['polled_cpu_seconds_lower_bound'] for r in rows if r['returned']),
        peak_sampled_tree_rss_bytes=max((r['returned']['peak_tree_rss_bytes'] for r in rows if r['returned']),default=0),
        session_wall_seconds=session_wall,records=rows,
        historical_files_unchanged=historical_verify(),original_48_results_unchanged=True,
        new_training=0,new_neural_forwards=0,M1_implemented=False,new_engineering_admission_started=False,
        R4_started=False,review_required=True,admission_passed=False)
    target=OUT/('execution_report_'+str(len(rows)).zfill(2)+'_'+uuid.uuid4().hex[:8]+'.json')
    once_sync(target,totals)
    status(phase or ('completed_stopped_for_review' if complete else 'partial_stopped_for_review'),
        diagnostic_complete=complete,attempts_charged=len(rows),results_returned=len(known),
        actual_physics_steps_returned=totals['actual_physics_steps_returned'],
        unknown_attempts=totals['unknown_attempts'],report=relative(target))
    print(json.dumps({k:totals[k] for k in ('diagnostic_complete','charged_attempts','results_returned','actual_physics_steps_returned','unknown_attempts')},ensure_ascii=False))
    return totals


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['authorize','draft','freeze','run','report'])
    parser.add_argument('--draft');parser.add_argument('--review');args=parser.parse_args()
    {'authorize':authorize,'draft':draft,'freeze':lambda:freeze(ROOT/args.draft,ROOT/args.review),
     'run':run,'report':report}[args.command]()
