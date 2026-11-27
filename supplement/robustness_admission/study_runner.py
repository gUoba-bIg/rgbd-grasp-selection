"""Orchestrate fresh V1 inference and old-E ranking, with durable unique task ownership."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import random
import sys
import time
from common_v1 import *

STUDY=OUT/"study"
P=C/"p"
def seedof(job,role):
    return int.from_bytes(hashlib.sha256((R6FIRST+"|"+role+"|"+job["frozen_reference_unit"]).encode()).digest()[:4],"big")

def private_junction(dst,src):
    need(not dst.exists() and src.is_dir() and Path(dst).absolute().is_relative_to(STUDY),"unsafe new junction "+str(dst))
    shell=Path(os.environ["SystemRoot"])/"System32/WindowsPowerShell/v1.0/powershell.exe"
    env=dict(os.environ,V1SUP_DST=str(dst),V1SUP_SRC=str(src))
    subprocess.run([str(shell),"-NoProfile","-NonInteractive","-Command","$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:V1SUP_DST -Target $env:V1SUP_SRC | Out-Null"],env=env,check=True)
    need(dst.is_junction() and dst.resolve()==src.resolve(),"declared read-only junction")

def init():
    disk();need((OUT/"admission/cold_3of3.json").is_file(),"V1 cold gate required")
    STUDY.mkdir(exist_ok=False)
    for item in ("preregistration","units","trials","run/jobs","run/groups","workspaces"):
        (STUDY/item).mkdir(parents=True)
    for fn in ("design.json","design_seal.json"):
        original=ROOT/R6FIRST/"preregistration"/fn
        copy=STUDY/"preregistration"/fn
        with original.open("rb") as f,copy.open("xb") as o:shutil.copyfileobj(f,o)
        need(sha(original)==sha(copy),"formal scene registry copy")
    private_junction(STUDY/"scenes",ROOT/R6NAME/"scenes")
    private_junction(STUDY/"frozen_r6_units",ROOT/R6NAME/"units")
    for lane in ["gpu01",*[f"cpu{n:02}" for n in range(1,5)]]:
        dest=STUDY/"workspaces"/lane;dest.mkdir()
        private_junction(dest/"data",P/"data");private_junction(dest/PL,P/PL)
        write(f"study/workspaces/{lane}/workspace.json",dict(lane=lane,project=str(P),private=True))
    sources={p.name:sha(p) for p in (WORK/"supplement").glob("*.py")}
    write("study/preregistration/forward_rank_seal.json",dict(utc=utc(),cold=sha(OUT/"admission/cold_3of3.json"),
        input_plan=sha(OUT/"preregistration/observation_plan.json"),sampling_rules=sha(WORK/"supplement/protocol.json"),
        packaging_sources=sources,engine=ENGINE,roles=["vision","rank"],no_method_results_read=True))

def group_child(role,path):
    from runtime_stages import observe_batch,rank_batch
    cfg=read(path);need(cfg["role"]==role,"child role drift")
    spec=read(OUT/"preregistration/observation_plan.json")
    need(sha(OUT/"preregistration/observation_plan.json")==read(STUDY/"preregistration/forward_rank_seal.json")["input_plan"],"job population changed")
    by_id={v["unit_id"]:v for v in spec["jobs"]}
    tasks=[by_id[n] for n in cfg["ids"]]
    need(all(seedof(j,role)==int(os.environ["PYTHONHASHSEED"]) for j in tasks),"per-worker seed")
    def returned(uid):print("return",role,uid,flush=True)
    if role=="forward":observe_batch(tasks,returned)
    else:rank_batch(tasks,returned)

def environment(lane,alias,seed,work):
    scratch=work/"tmp";scratch.mkdir()
    system=Path(os.environ["SYSTEMROOT"])
    env={k:v for k,v in os.environ.items() if k.upper() in ("WINDIR","SYSTEMROOT","COMSPEC","SYSTEMDRIVE","NUMBER_OF_PROCESSORS","PROCESSOR_ARCHITECTURE","PATHEXT","COMPUTERNAME","USERNAME")}
    for name in ("LOCALAPPDATA","APPDATA","USERPROFILE","HOME"):
        box=lane/("user_"+name);box.mkdir(exist_ok=True);env[name]=str(box)
    env.update(PATH=os.pathsep.join(map(str,[C/"env"/alias/"Scripts",C/"bin",C/"bin/DLLs",system/"System32",system])),
               PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUTF8="1",PYTHONHASHSEED=str(seed),
               OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TEMP=str(scratch),TMP=str(scratch),
               TMPDIR=str(scratch),CUDA_CACHE_PATH=str(lane/"cuda_cache"),TORCH_EXTENSIONS_DIR=str(scratch/"torch"),
               MPLIGNORE_SYSTEM_FONTS="1",MPL_IGNORE_SYSTEM_FONTS="1",MPLBACKEND="Agg",MPLCONFIGDIR=str(lane/"mpl"),NUMBA_CACHE_DIR=str(lane/"numba"))
    if alias!="vision":env["CUDA_VISIBLE_DEVICES"]=""
    return env

def launch(role,group,lane_name,index):
    need(not (STUDY/"STOP.json").exists(),"shared stop already registered")
    label=f"{role}-{index:03}"
    folder=STUDY/"run/groups"/label;folder.mkdir(exist_ok=False)
    lane=STUDY/"workspaces"/lane_name;alias={"forward":"vision","rank":"rank"}[role]
    args=dict(role=role,ids=[j["unit_id"] for j in group],seed=seedof(group[0],role),lane=lane_name)
    write(f"study/run/groups/{label}/claim.json",dict(utc=utc(),**args))
    env=environment(lane,alias,args["seed"],folder)
    cmd=[C/"env"/alias/"Scripts/python.exe","-B","-s",Path(__file__).resolve(),"--child",role,folder/"claim.json"]
    with (folder/"stdout.log").open("xb") as so,(folder/"stderr.log").open("xb") as se:
        child=subprocess.Popen(list(map(str,cmd)),cwd=lane,env=env,stdout=so,stderr=se)
        while child.poll() is None:
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                disk()
    if child.returncode:
        raise RuntimeError(f"{label}: "+(folder/"stderr.log").read_text("utf8",errors="replace")[-2600:])
    suffix="forward_done.json" if role=="forward" else "rank_done.json"
    returns={j["unit_id"]:sha(STUDY/"units"/j["unit_id"]/suffix) for j in group}
    write(f"study/run/groups/{label}/return.json",dict(utc=utc(),returns=returns))
    return returns

def run(role):
    plan=read(OUT/"preregistration/observation_plan.json")["jobs"];groups={}
    for j in plan:groups.setdefault(j["frozen_reference_unit"],[]).append(j)
    if role=="rank":need((STUDY/"run/forward_completed.json").exists(),"all fresh forwards first")
    queues=iter(enumerate(groups.values(),1));lanes=["gpu01"] if role=="forward" else [f"cpu{x:02}" for x in range(1,5)]
    progress=0;last=0;inflight={};first=None
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        def dispatch(lane):
            task=next(queues,None)
            if task is not None:
                i,g=task;inflight[pool.submit(launch,role,g,lane,i)]=lane
        for lane in lanes:dispatch(lane)
        while inflight:
            ready,_=wait(inflight,return_when=FIRST_COMPLETED)
            for f in ready:
                lane=inflight.pop(f)
                try:progress+=len(f.result())
                except Exception as exc:
                    if first is None:
                        first=repr(exc);disk();write("study/STOP.json",dict(utc=utc(),stage=role,reason=first,no_further_dispatch=True))
                if progress//30>last:
                    last=progress//30
                    print(role,progress,"of 720",flush=True)
                    disk()
            if first is None:
                for lane in lanes:
                    if lane not in inflight.values():dispatch(lane)
    need(first is None and progress==720,"incomplete "+role+": "+str(first))
    write(f"study/run/{role}_completed.json",dict(utc=utc(),count=progress,disk=disk()))
if __name__=="__main__":
    if sys.argv[1]=="--child":group_child(sys.argv[2],Path(sys.argv[3]))
    elif sys.argv[1]=="network":
        init();run("forward");run("rank")
