"""Independent reproduction of the four samples from pre-E pose pools and orders only."""
import os,sys
import io,encodings.cp437
from pathlib import Path
from common_v1 import *
from rank_stability import pair_matches
import numpy as np
from collections import Counter

PLAN=OUT/"preregistration/observation_plan.json"
REPLAY=OUT/"admission/outcome_blind_draw.json"

def draw():
    mode_names=("depth_only","shift_only","missing_only","block_only")
    pool_root=OUT/"study/units"
    reads=set();allowed={PLAN.resolve()};writing=False
    def only_inputs(event,args):
        if event!="open":return
        name,mode,flags=args
        if isinstance(name,int):return
        path=Path(os.fsdecode(name)).resolve()
        if writing and path==REPLAY.resolve() and flags&os.O_CREAT:return
        if path not in allowed:raise PermissionError("unregistered sampling input "+str(path))
        need(not flags&(os.O_RDWR|os.O_WRONLY|os.O_APPEND|os.O_TRUNC|os.O_CREAT),"sampling input write")
        reads.add(path)
    # All library loading and native function imports precede this data-access boundary.
    warm=np.zeros((1,17),dtype=np.float32)
    warm[0,0:4]=[1,.05,.02,.02];warm[0,4:13]=np.eye(3).ravel()
    buffer=io.BytesIO();np.savez_compressed(buffer,candidate_camera=warm);buffer.seek(0)
    with np.load(buffer,allow_pickle=False) as n:pair_matches(n["candidate_camera"],warm)
    sys.addaudithook(only_inputs)
    jobs=read(PLAN)["jobs"]
    for j in jobs:
        if j["condition"] in ("clean",*mode_names):
            allowed.update((pool_root/j["unit_id"]/name).resolve() for name in ("pool.npz","orders.json"))
    def get(uid):
        root=pool_root/uid
        with np.load(root/"pool.npz",allow_pickle=False) as f:rows=f["candidate_camera"].copy()
        order=read(root/"orders.json")["orders_by_pool_index"]
        need(all(sorted(order[m])==list(range(len(rows))) for m in METHODS),"incomplete pre-E candidates")
        return rows,{m:(order[m][0] if order[m] else None) for m in METHODS}
    hits={c:[] for c in mode_names};clean={}
    for j in jobs:
        c=j["condition"]
        if c not in hits:continue
        uid,scene=j["unit_id"],j["scene"]
        if scene not in clean:clean[scene]=get(scene+"-clean")
        a,top_a=clean[scene];b,top_b=get(uid)
        pairs,_=pair_matches(a,b)
        changed=any((top_a[m] is None)!=(top_b[m] is None) or
                    (top_a[m] is not None and top_b[m] is not None and pairs.get(top_a[m])!=top_b[m]) for m in METHODS)
        key=hashlib.sha256((NAME+"|robustness|"+uid).encode()).hexdigest()
        need(key==j["screening_order_hash"],"independent hashing vs sealed observation plan")
        if changed:hits[c].append((key,uid))
    ordered={c:[u for _,u in sorted(v)] for c,v in hits.items()}
    need(len(clean)==120 and reads==allowed,"missing or extra sampling inputs")
    data=dict(utc=utc(),populations={c:len(x) for c,x in ordered.items()},all_changed_in_hash_order=ordered,
              sampled={c:x[:6] for c,x in ordered.items()},opened_sources=[str(p.relative_to(OUT)) for p in sorted(reads)],
              access_policy="only frozen observation plan and pool.npz/orders.json for clean and four single factors; every other file read denied",
              reran_models=0,physical_steps=0)
    writing=True
    write("admission/outcome_blind_draw.json",data)

def verify():
    actual=read(REPLAY);sealed=OUT/"preregistration/robustness_sample.json";s=read(sealed)
    need(actual["sampled"]==s["sampled"] and actual["all_changed_in_hash_order"]==s["all_changed_in_hash_order"],"sample identity drift")
    write("admission/outcome_blind_draw_verification.json",dict(utc=utc(),consistent=True,scope="All four entire frames, hash order, and 24 sampled unit identities",
         independent_draw_sha256=sha(REPLAY),original_selection_sha256=sha(sealed),opened_inputs=len(actual["opened_sources"]),
         original_sealer_note="The original common wrapper also opened the diagnostic case registry before branching. It contained historical control-group labels. No such fields fed the sampling expression. This isolated replay denied access to that registry, method receipts and outcomes.",
         fresh_or_repeated_experimental_trials=0))
if __name__=="__main__":
    {"draw":draw,"verify":verify}[sys.argv[1]]()
