"""Proceed only from the recorded fresh 720-unit ranking release."""
from common_v1 import *
import sys
import time
def call(label,interpreter,script,*args):
    directory=OUT/"execution_logs";directory.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUTF8="1",
             OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1")
    with (directory/(label+".out.log")).open("xb") as out,(directory/(label+".err.log")).open("xb") as err:
        p=subprocess.run([str(interpreter),"-B","-s","-u","-X","utf8",str(WORK/"supplement"/script),*args],env=env,cwd=WORK,stdout=out,stderr=err)
    need(p.returncode==0,"supplement stage "+label+" returned "+str(p.returncode))
    print("returned",label,flush=True)
def main():
    study=OUT/"study"
    while not (study/"run/rank_completed.json").is_file():
        if (study/"STOP.json").is_file():raise RuntimeError("observation / ranking integrity stop; physical dispatch forbidden")
        time.sleep(45)
    checks=[read(study/"run"/(phase+"_completed.json"))["count"] for phase in ("forward","rank")]
    need(checks==[720,720] and not (study/"STOP.json").exists(),"fresh observations and V1 rankings not complete")
    disk()
    rank=C/"env/rank/Scripts/python.exe"
    call("freeze_robustness",rank,"method_seal.py","robustness")
    call("figure_six_modes",rank,"figures.py","prospective")
    call("optional_methods_once",sys.executable,"run_sealed_methods.py")
    call("supplement_statistics",rank,"final_statistics.py")
    write("admission/completed_stages.json",dict(utc=utc(),disk=disk(),three_cold_verified=True,original_redesign=False,
        inference_pairs=720,three_method_rankings=2160,method_ledger_sha256=sha(study/"run/method_ledger.json")))
if __name__=="__main__":main()
