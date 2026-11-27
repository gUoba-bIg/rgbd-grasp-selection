"""Build/install the isolated interface-only patch. No simulator import or connection."""
import os,sys,subprocess,time,platform,json
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sim_v52_common import read,write,identity,relative,now,lock
NEW=ROOT/'runtime/simulation_v52e';LOG=ROOT/'logs/simulation_v52e'
PYTHON=NEW/'venv/Scripts/python.exe';SRC=NEW/'src/pybullet-3.2.7'

def main():
    assert Path.cwd().resolve()==ROOT
    import psutil
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52e.lock'):
        if (LOG/'build_intent.json').exists():raise RuntimeError('Build already started; inspect retained outcome before any revision')
        patch=read(LOG/'patch/patch_manifest.json')
        for row in patch['modified_files']:
            actual=identity(SRC/row['path']);assert actual['sha256']==row['after']['sha256'] and actual['bytes']==row['after']['bytes']
        command=[str(PYTHON),'-B','-m','pip','wheel','--verbose','--disable-pip-version-check','--no-build-isolation','--no-deps',str(SRC),'--wheel-dir',str(NEW/'wheels')]
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',PYBULLET_BUILD_JOBS='2',
            TEMP=str(NEW/'temp'),TMP=str(NEW/'temp'),PIP_CACHE_DIR=str(NEW/'cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        write(LOG/'build_intent.json',dict(at=now(),command=command,workdir=str(ROOT),python=identity(PYTHON),
            patch=identity(LOG/'patch/patch_manifest.json'),platform=platform.platform(),python_version=sys.version,
            build_jobs=2,wall_limit_seconds=2700,no_simulator_connection=True,
            environment={k:env.get(k) for k in ['TEMP','TMP','PIP_CACHE_DIR','PYBULLET_BUILD_JOBS','INCLUDE','LIB','LIBPATH','VCToolsInstallDir','WindowsSdkDir']}),once=True)
        start=time.monotonic();peak=0;cause=None
        with (LOG/'compile.log').open('xb') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            write(ROOT/'runtime/pipeline/simulation_v52e.json',dict(at=now(),phase='isolated_build_running',build_pid=proc.pid,
                physics_steps=0,acceptance_starts=0,M1_implemented=False,R4_authorized=False,
                old_scope_all_tasks_percent=45,old_scope_experiment_percent=56.25,expanded_scope_percent=None))
            while proc.poll() is None:
                try:
                    tree=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
                    peak=max(peak,sum(p.memory_info().rss for p in tree))
                except psutil.Error:pass
                if time.monotonic()-start>2700 or (LOG/'STOP').exists():
                    cause='build_wall_limit_or_stop'
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.Error:pass
                    break
                time.sleep(1)
            code=proc.wait()
        wheels=list((NEW/'wheels').glob('pybullet-3.2.7+v52e1-*.whl'))
        result=dict(at=now(),exit_code=code,termination=cause,wall_seconds=time.monotonic()-start,peak_sampled_tree_rss_bytes=peak,
            wheels={relative(p):identity(p) for p in wheels},log=identity(LOG/'compile.log'),physics_steps=0)
        write(LOG/'build_result.json',result,once=True)
        if code!=0 or cause or len(wheels)!=1:raise RuntimeError('Build incomplete; retained, no acceptance launch')
        command=[str(PYTHON),'-B','-m','pip','install','--disable-pip-version-check','--no-compile','--no-deps',str(wheels[0])]
        write(LOG/'install_intent.json',dict(at=now(),command=command,wheel=identity(wheels[0]),old_environment_mutation=False),once=True)
        with (LOG/'install.log').open('xb') as log:
            done=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        extensions=list((NEW/'venv/Lib/site-packages').glob('pybullet*.pyd'))
        write(LOG/'install_result.json',dict(at=now(),exit_code=done.returncode,extensions={relative(p):identity(p) for p in extensions},physics_steps=0),once=True)
        assert done.returncode==0 and len(extensions)==1
        write(ROOT/'runtime/pipeline/simulation_v52e.json',dict(at=now(),phase='isolated_build_complete_zero_step_pending',
            physics_steps=0,acceptance_starts=0,M1_implemented=False,R4_authorized=False,
            old_scope_all_tasks_percent=45,old_scope_experiment_percent=56.25,expanded_scope_percent=None))
        print(json.dumps(dict(build_exit=code,install_exit=done.returncode,extension=relative(extensions[0]),physics_steps=0)))

if __name__=='__main__':main()
