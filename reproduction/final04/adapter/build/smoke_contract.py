from seal_io import *
import sys, subprocess
out={}
def score_payload(x):
    return {k:v for k,v in x.items() if k not in ('scoring_sec','pool_build_sec')}
for c in read(REPO/'metadata/cold_preregistration.json')['cases']:
    d=ROOT/P0/'units'/c['unit'];sel=read(d/'selection.json');orders=read(d/'orders.json')
    out[c['trial_id']]=dict(selection_core={k:v for k,v in sel.items() if k not in ['native_eligibility_api_sha256','eligibility']},
        eligibility=[{k:v for k,v in z.items() if k!='planning_seconds'} for z in sel['eligibility']],
        orders=score_payload(orders),frozen_native_success=c['expected_grasp_success'])
put(REPO/'metadata/smoke_expected.json',out)
# Keep source-checkable build/installation identities for the custom numerical extensions.
pack('native_build_inputs',{p.relative_to(ROOT).as_posix():(p,None) for d in [ROOT/'runtime/downloads/grasp_eval_v7/archives'] for p in d.iterdir() if p.name.endswith(('.tar.gz','.zip'))})
