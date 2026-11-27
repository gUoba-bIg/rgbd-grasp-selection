"""Pretrial integration proof with archived P3 output. No method or physics is authorized."""
import json
from pathlib import Path
import numpy as np
from platform_select import evaluate,digest,VERSION,ROOT,R4Trace
from scene_world import load_eligibility_world,setup_world,write
from platform_static import scan
import pybullet as backend

def main():
    d=VERSION/'audit'/'algorithm_challenge'
    out=VERSION/'audit'/'platform_zero_challenge'
    out.mkdir(exist_ok=False)
    with np.load(d/'pool.npz') as pool,np.load(d/'network.npz') as nn:
        inp=dict(unit_id='archived-021-qual-probe',T_W_C=nn['world_T_camera'].tolist(),candidate_rows=pool['candidate_camera'].tolist(),rankings=json.loads((d/'orders.json').read_bytes())['orders_by_pool_index'])
    write(out/'platform_input.json',inp)
    p=R4Trace(backend,out/'eligibility_api','eligibility')
    cid=p.connect(p.DIRECT)
    try:
        ctx=load_eligibility_world(p,cid,out/'eligibility_setup')
        selection=evaluate(inp,ctx)
    finally:
        p.disconnect(physicsClientId=cid);p.close()
    selection.update(selection_input_sha256=digest(out/'platform_input.json'),physical_steps=p.counts['stepSimulation'])
    write(out/'selection.json',selection)
    print('Selected irrevocably before target loaded',selection['selected'],flush=True)
if __name__=='__main__':
    main()

