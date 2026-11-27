"""Wait for the original preprocessing worker, then seal, execute and close the authorized batch."""
import sys,subprocess,time,traceback
from common import *
def run(name):
    log=HERE/'run'/('stage_'+name)
    exe=ROOT/'runtime/envs/grasp_eval/Scripts/python.exe'
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.with_suffix('.stdout').open('xb') as out,log.with_suffix('.stderr').open('xb') as err:
        ret=subprocess.run([str(exe),'-B',str(HERE/'code'/(name+'.py'))],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    if ret.returncode:
        raise RuntimeError(name+' exit '+str(ret.returncode)+'; '+log.with_suffix('.stderr').read_text(encoding='utf8',errors='replace')[-2200:])
def main():
    put(HERE/'run/controller_start.json',dict(pid=os.getpid(),utc=now(),cwd=str(Path.cwd()),version=HERE.name))
    while not (HERE/'run/rank_complete.json').exists() and not (HERE/'run/PREP_STOP.json').exists():time.sleep(2)
    if not (HERE/'run/PREP_STOP.json').exists():
        try:run('seal')
        except BaseException as e:
            put(HERE/'run/SEAL_STOP.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),formal_methods_started=0))
        else:
            try:run('formal_dispatch')
            except BaseException as e:
                if not (HERE/'run/METHOD_STOP.json').exists():put(HERE/'run/CONTROLLER_STOP.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()))
    run('analyze')
    run('delivery')
if __name__=='__main__':main()
