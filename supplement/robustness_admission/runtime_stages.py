"""Independent wrappers. Inputs and scientific functions are loaded from the hash-verified V1 replica."""
from common_v1 import *
import sys
import random
import time

STUDY=OUT/"study"
P=C/"p"
FORM=P/R6NAME
def native_setup(role,project):
    need((OUT/"admission/cold_3of3.json").is_file(),"cold 3/3 required")
    need(Path(sys.prefix).resolve()==C/"env"/role,"wrong interpreter")
    need((Path.cwd()/"data").resolve()==P/"data" and (Path.cwd()/PL).resolve()==P/PL,"wrong asset bindings")
    os.environ["PROGRAMFILES"]=str(P/"vendor/ProgramFiles")
    stage_dirs=[P/"vendor/ProgramFiles/Git/mingw64/bin",C/"bin",C/"bin/DLLs",C/"env"/role/"Lib/site-packages/torch/lib"]
    global _dll
    _dll=[os.add_dll_directory(str(p)) for p in stage_dirs if p.is_dir()]
    sys.path[:0]=[str(FORM/"code"),str(FORM/"implementation"),str(P/R4/"implementation"),str(P)]
    import common,task_worker,platform_imports
    # Only the output/version binding changes. ROOT, the V1 asset resolver and the engine remain sealed.
    common.HERE=project;task_worker.HERE=project;platform_imports.VERSION=project
    need(common.ROOT==P and task_worker.ROOT==P and common.workspace_ok(),"bundle/lane identity")
    return task_worker

def check_archived_forward(uid,actual,origin):
    import numpy as np
    with np.load(actual,allow_pickle=False) as a,np.load(origin,allow_pickle=False) as b:
        need(set(a.files)==set(b.files),"original network field set "+uid)
        for k in a.files:
            need(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),
                 "original network field drift "+uid+"/"+k)
    return {"original_network_sha256":sha(origin),"fresh_network_sha256":sha(actual),"all_array_bytes_equal":True}

def observe_batch(specs,checkpoint):
    import numpy as np
    tw=native_setup("vision",STUDY)
    from src import p5_quality_v23
    from single_factors import project
    frozen_corrupt=p5_quality_v23.corrupt
    from src.grasp_prerequisite_data_v27 import SOURCE
    from policy_network import generate
    import loss_utils
    need(sha(Path(loss_utils.__file__))=="0a8dea7a09020f869c329781a6b2a713f8ba62d385d07dab43deaab55e007eb7","frozen decoder")
    for spec in specs:
        if (OUT/"study/STOP.json").exists():break
        uid=spec["unit_id"];mode=spec["condition"]
        # Match the clean/joint stage seed itself. A factor uses the same seed as its frozen joint parent.
        seed=int.from_bytes(hashlib.sha256((R6FIRST+"|forward|"+spec["frozen_reference_unit"]).encode()).digest()[:4],"big")
        random.seed(seed);np.random.seed(seed)
        s=dict(spec["frozen_scene"])
        s["condition_seed"]={**s["condition_seed"],mode:spec["condition_seed"]}
        p5_quality_v23.corrupt=frozen_corrupt if mode in ("clean","joint") else lambda x,r,e,c:project(x,r,e,c,frozen_corrupt)
        started=time.perf_counter()
        try:tw.make_unit(spec,s)
        finally:p5_quality_v23.corrupt=frozen_corrupt
        unitdir=STUDY/"units"/uid
        fm=read(unitdir/"network.json")
        expected=spec["condition_seed"]
        need(fm["corruption"]["seed"]==expected and fm["networks_returned"]=={"P3":1,"P5_fixed":1} and fm["physics_steps"]==0,"observation seed/return "+uid)
        comparisons=check_archived_forward(uid,unitdir/"network.npz",ROOT/R6NAME/"units"/uid/"network.npz") if mode in ("clean","joint") else None
        write(f"study/units/{uid}/forward_done.json",dict(utc=utc(),elapsed=time.perf_counter()-started,source_reference=spec["frozen_reference_unit"],old_new=comparisons,model_returns=fm["networks_returned"],physical_steps=0))
        checkpoint(uid)

def rank_batch(specs,checkpoint):
    import numpy as np
    tw=native_setup("rank",STUDY)
    for spec in specs:
        if (OUT/"study/STOP.json").exists():break
        uid=spec["unit_id"];dest=STUDY/"units"/uid
        need((dest/"forward_done.json").is_file(),"ranking before forward")
        seed=int.from_bytes(hashlib.sha256((R6FIRST+"|rank|"+spec["frozen_reference_unit"]).encode()).digest()[:4],"big")
        random.seed(seed);np.random.seed(seed);start=time.perf_counter()
        tw.rank_and_E(uid)
        s=read(dest/"selection.json");o=read(dest/"orders.json");same=None
        if spec["condition"] in ("clean","joint"):
            olddir=ROOT/R6NAME/"units"/uid;old=read(olddir/"orders.json");olds=read(olddir/"selection.json")
            for f in ("scores","orders_by_pool_index","candidate_details","shared_pool_sha256","pool_count","decoder_count"):
                need(o[f]==old[f],"old V1 "+uid+" rank "+f)
            def canonical(s):
                return [{k:v for k,v in r.items() if k!="planning_seconds"} for r in s["eligibility"]]
            need(canonical(s)==canonical(olds) and s["selected"]==olds["selected"],"old V1 E/path reproduction "+uid)
            same=True
        write(f"study/units/{uid}/rank_done.json",dict(utc=utc(),elapsed=time.perf_counter()-start,old_scientific_output_equal=same,
            E=sum(r["eligible"] for r in s["eligibility"]),selected=s["selected"],pool_sha256=sha(dest/"pool.npz"),
            selection_sha256=sha(dest/"selection.json"),orders_sha256=sha(dest/"orders.json"),physics_steps=0))
        checkpoint(uid)
