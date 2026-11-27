"""Derive an isolated controller from frozen v52d without changing its source."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'scripts/sim_v52d_control.py').read_text(encoding='utf-8')
source = source.replace('simulation_v52d', 'simulation_v52f').replace('sim_v52d_', 'sim_v52f_').replace('SIM_V52D_', 'SIM_V52F_')
source = source.replace('simulation_v52_dynamics_run.json', 'simulation_v52f.json')
source = source.replace("PYTHON=ROOT/'runtime/simulation_v52/venv/Scripts/python.exe'", "PYTHON=ROOT/'runtime/simulation_v52e/venv/Scripts/python.exe'")
source = source.replace("user_authorization='Explicit approval of twelve N/L/C diagnostics after static audit'", "user_authorization='User approved exactly twelve repaired-interface M0 diagnostics using isolated v52e engine; stop for review'")
start = source.index('def authorize():')
end = source.index('\ndef draft():', start)
source = source[:start] + '''def authorize():
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
    paths.update([LOG/'authorization.json',ROOT/'configs/simulation_v52d.json',PYTHON,
        ROOT/'runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd',
        ROOT/'runtime/simulation_v52f/dependency_manifest.json',ROOT/'docs/v52f_修复后M0十二格复验协议.txt',
        ROOT/'results/simulation_v52e/attempts/001_ZERO001_2b433b45/afterR.json',
        ROOT/'logs/simulation_v52e/install_result.json',ROOT/'logs/simulation_v52e/independent_readback_review_v2.json'])
    return {relative(p):identity(p) for p in sorted(paths)}

''' + source[end:]
source = source.replace("base_config='configs/simulation_v52r2.json',plan_path=relative(PLAN),", "base_config='configs/simulation_v52r2.json',plan_path=relative(PLAN),\n        python=relative(PYTHON),expected_distribution_version='3.2.7+v52e1',\n        pybullet_extension='runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd',\n        dependency_dir='runtime/simulation_v52f/deps',dependency_manifest='runtime/simulation_v52f/dependency_manifest.json',\n        parameter_reference_readback='results/simulation_v52e/attempts/001_ZERO001_2b433b45/afterR.json',\n        parameter_targets=dict(lateralFriction=1.0,collisionMargin=.0005,restitution=0.),")
source = source.replace("('sim_v52f_episode.py','sim_v52r2_episode.py','sim_v52r2_control.py')", "('sim_v52f_episode.py','sim_v52d_episode.py','sim_v52e_zero_step.py','sim_v52r2_episode.py','sim_v52r2_control.py')")
source = source.replace("SIM_V52F_CONTROLLER_TOKEN=token,OMP_NUM_THREADS='1'", "SIM_V52F_CONTROLLER_TOKEN=token,PYTHONPATH=str(ROOT/cfg['dependency_dir']),\n                TEMP=str(ROOT/'runtime/simulation_v52f/temp'),TMP=str(ROOT/'runtime/simulation_v52f/temp'),OMP_NUM_THREADS='1'")
source = source.replace("cfg['criteria']['mimic_error_rad']==.03", "cfg['criteria']['mimic_error_rad']==.03\n    assert cfg['python']==relative(PYTHON) and cfg['expected_distribution_version']=='3.2.7+v52e1'\n    assert cfg['parameter_targets']==dict(lateralFriction=1.0,collisionMargin=.0005,restitution=0.)\n    for path,value in read(ROOT/cfg['dependency_manifest'])['files'].items():assert identity(ROOT/path)==value,path")
source = source.replace("not value.get('telemetry_complete',False):", "not value.get('telemetry_complete',False) or not value.get('parameter_assertions_passed',False):")
source = source.replace("r['result'].get('telemetry_complete',False)\n            and", "r['result'].get('telemetry_complete',False) and r['result'].get('parameter_assertions_passed',False)\n            and")
target=ROOT/'scripts/sim_v52f_control.py'
with target.open('x',encoding='utf-8') as stream:stream.write(source)
print(str(target))
