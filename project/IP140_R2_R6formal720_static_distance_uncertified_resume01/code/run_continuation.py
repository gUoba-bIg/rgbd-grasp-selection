"""One finite execution followed by statistical audit and a transferable core archive."""
import sys,subprocess,os,traceback
from common import *
def run(stage):
    out=HERE/'run'/('stage_'+stage+'.stdout');err=out.with_suffix('.stderr')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with out.open('xb') as of,err.open('xb') as ef:
        code=subprocess.run([sys.executable,'-B',str(HERE/'code'/(stage+'.py'))],cwd=ROOT,env=env,stdout=of,stderr=ef,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    if code:raise RuntimeError(stage+': '+err.read_text(encoding='utf8',errors='replace')[-2000:])
def main():
    put(HERE/'run/controller_start.json',dict(utc=now(),pid=os.getpid(),version=HERE.name,allowed_ablation=0))
    run('admission')
    try:run('formal_dispatch')
    except BaseException as e:
        if not (HERE/'run/METHOD_STOP.json').exists():put(HERE/'run/CONTROLLER_STOP.json',dict(error=repr(e),utc=now()))
    run('analyze');run('delivery')
if __name__=='__main__':main()
