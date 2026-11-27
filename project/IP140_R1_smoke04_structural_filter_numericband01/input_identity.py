"""Read-only provenance checks for the single admitted session. No backend import."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OLD = PROJECT/'IP140_R1_smoke01'
HISTORICAL_LIST_SHA256 = '86bf8ebd739a0c3b520bb70d246c159ea79dd49368cd182ec27b955807caf794'
PARENT = PROJECT/'IP140_R1_smoke02_pathfix01'
PARENT_LIST_SHA256='5ccc7e88af867b96151d9a49c951412fa405a995872381b668ebf13cf03bf436'
START_INPUTS = ('model/ur5e_ip140_r1.urdf','model/manifest.json','preregistration.json',
                'ledger_initial.json','zero_gate.py','empty_stroke.py','episode_log.py',
                'execution_control.py','input_identity.py','mountpair.py','mountpair_registration.json','full_api.py','structural_filter.py','run_once.py')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def snapshot():
    if Path.cwd().resolve()!=PROJECT:
        raise RuntimeError('explicit project root required')
    parent_index=PARENT/'SHA256SUMS.txt'
    if digest(parent_index)!=PARENT_LIST_SHA256:
        raise RuntimeError('smoke02 historical inventory changed')
    for row in parent_index.read_text('utf-8-sig').splitlines()[1:]:
        h,n,rel=row.split('\t')
        file=PARENT/rel
        if str(file.stat().st_size)!=n or digest(file)!=h:
            raise RuntimeError('smoke02 history mismatch: '+rel)
    if read(PARENT/'slot_ledger.json')['starts']!=0:
        raise RuntimeError('existing 12-slot budget is not unused')
    hist=OLD/'SHA256SUMS.txt'
    if digest(hist)!=HISTORICAL_LIST_SHA256:
        raise RuntimeError('smoke01 file list has changed')
    checked={}
    for row in hist.read_text('utf-8-sig').splitlines()[1:]:
        h,n,rel=row.split('\t')
        file=OLD/rel
        if str(file.stat().st_size)!=n or digest(file)!=h:
            raise RuntimeError('historical binding changed: '+rel)
        checked[rel]=h
    new={k:digest(HERE/k) for k in START_INPUTS}
    for k in ('model/ur5e_ip140_r1.urdf','model/manifest.json','execution_control.py'):
        if new[k]!=checked[k]:raise RuntimeError('file must be byte-identical: '+k)
    registration=read(HERE/'preregistration.json')
    prior=PROJECT/'IP140_R1_smoke03_framefix_mountpair01'
    index=prior/'SHA256SUMS.txt'
    if digest(index)!='801c981c53f4ed2dff9440c58d36d2b513e1525d6a4e868c8570ddcb479e1832':
        raise RuntimeError('smoke03 index has changed')
    for line in index.read_text('utf-8-sig').splitlines()[1:]:
        h,size,rel=line.split('\t')
        if digest(prior/rel)!=h or (prior/rel).stat().st_size!=int(size):
            raise RuntimeError('smoke03 evidence changed: '+rel)
    ledger=read(HERE/'ledger_initial.json')
    if (ledger['starts'],ledger['unused_original_budget'],ledger['slots'][0]['state'])!=(1,11,'returned_failed'):
        raise RuntimeError('inherited budget mismatch')
    if registration['slots']!=read(prior/'preregistration.json')['slots'] or registration['version']!=HERE.name or registration['eligible_slots_after_zero_gate']!=[2]:
        raise RuntimeError('registration mismatch')
    if registration['max_starts']!=12 or [s['slot'] for s in registration['slots']]!=list(range(1,13)):
        raise RuntimeError('invalid budget or slots')
    expected=(4,16,32,6,18,34)
    for slot,num in zip(registration['slots'][6:],expected):
        if not slot['candidate_id'].startswith(f'v52eng:slot{num:02d}:'):
            raise RuntimeError('wrong historical candidate')
    manifest=read(HERE/'model/manifest.json')
    for k,h in {**manifest['provenance'],**manifest['reused_arm_asset_sha256']}.items():
        if digest(PROJECT/k)!=h:raise RuntimeError('upstream asset changed: '+k)
    return dict(schema='ip140_smoke04_inputs_v1',smoke03_inventory_sha256=digest(index), version=HERE.name, smoke02_inventory_sha256=PARENT_LIST_SHA256, new_files=new,
        smoke01_inventory_sha256=HISTORICAL_LIST_SHA256,smoke01_verified_files=len(checked),
        executable_slots=[2],total_budget=12,maximum_direct_connections=1,maximum_robot_loads=1)


def verify():
    snap=read(HERE/'input_identity.json')
    if snap!=snapshot():
        raise RuntimeError('input or historical integrity changed after freeze')
    return snap


if __name__=='__main__':
    value=snapshot()
    with (HERE/'input_identity.json').open('x',encoding='utf-8') as dest:
        json.dump(value,dest,indent=2,ensure_ascii=False)
        dest.write('\n')
