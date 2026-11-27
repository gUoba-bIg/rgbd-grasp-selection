"""Bind the authorized start, lineage, old failed episodes and physical inputs."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
HISTORY={
    'IP140_R1_smoke01':'86bf8ebd739a0c3b520bb70d246c159ea79dd49368cd182ec27b955807caf794',
    'IP140_R1_smoke02_pathfix01':'5ccc7e88af867b96151d9a49c951412fa405a995872381b668ebf13cf03bf436',
    'IP140_R1_smoke03_framefix_mountpair01':'801c981c53f4ed2dff9440c58d36d2b513e1525d6a4e868c8570ddcb479e1832',
    'IP140_R1_smoke04_structural_filter_numericband01':'6b0cca00b8e9b4cb6cc95650b32aa6f6699f44b827dd7cfe271f435496df79a3'}
PREVIOUS=ROOT/'IP140_R1_smoke04_structural_filter_numericband01'
NEW_ID='IP140-BENCH-EMPTY-03:post-fixed-seam-filter'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text('utf-8-sig'))

def snapshot():
    if Path.cwd().resolve()!=ROOT:
        raise RuntimeError('require explicit project working directory')
    for folder,identity in HISTORY.items():
        base=ROOT/folder
        index=base/'SHA256SUMS.txt'
        if digest(index)!=identity:
            raise ValueError('history index changed: '+folder)
        for line in index.read_text('utf-8-sig').splitlines()[1:]:
            expected,size,rel=line.split('\t')
            if (base/rel).stat().st_size!=int(size) or digest(base/rel)!=expected:
                raise ValueError('historical file changed: '+folder+'/'+rel)
    for item in read(HERE/'history/sources.json'):
        if digest(HERE/item['package'])!=item['sha256'] or digest(ROOT/item['source'])!=item['sha256']:
            raise ValueError('history copy mismatch: '+item['package'])
    pri=read(PREVIOUS/'preregistration.json');reg=read(HERE/'preregistration.json')
    reference=pri['slots'].copy()
    reference[2]=dict(reference[1],slot=3,candidate_id=NEW_ID)
    l=read(HERE/'ledger_initial.json')
    if (reg['slots']!=reference or reg['deferred_center']['prior_definition']!=pri['slots'][2]
        or reg['eligible_slots_after_zero_gate']!=[3] or reg['version']!=HERE.name or reg['max_starts']!=12
        or l['starts']!=2 or l['unused_original_budget']!=10 or l['budget_added']!=0
        or any(not s['charged'] or s['state']!='returned_failed' for s in l['slots'][:2])
        or any(s['charged'] for s in l['slots'][2:])):
        raise ValueError('registration or historic budget mismatch')
    if l['preregistration_sha256']!=digest(HERE/'preregistration.json'):
        raise ValueError('ledger not bound')
    for name in ['model/manifest.json','model/ur5e_ip140_r1.urdf','execution_control.py']:
        if digest(HERE/name)!=digest(PREVIOUS/name):
            raise ValueError('frozen model or control changed: '+name)
    manifest=read(HERE/'model/manifest.json')
    for rel,h in {**manifest['provenance'],**manifest['reused_arm_asset_sha256']}.items():
        if digest(ROOT/rel)!=h:
            raise ValueError('physical asset mismatch: '+rel)
    files=sorted({*HERE.glob('*.py'), *[HERE/n for n in ('mountpair_registration.json','ledger_initial.json','preregistration.json','model/manifest.json','model/ur5e_ip140_r1.urdf','history/sources.json')],*[f for f in HERE.glob('*registration*.json')]})
    return dict(schema='ip140_smoke05_inputs_v1',version=HERE.name,executable_slots=[3],attempt_id=NEW_ID,
        total_budget=12,inherited_starts=2,history_inventories=HISTORY,
        maximum_direct_connections=1,maximum_robot_loads=1,new_files={f.relative_to(HERE).as_posix():digest(f) for f in files})

def verify():
    frozen=read(HERE/'input_identity.json')
    if frozen!=snapshot():
        raise RuntimeError('input mismatch after freeze')
    return frozen

if __name__=='__main__':
    bindings=snapshot()
    with (HERE/'input_identity.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(bindings,f,indent=2,ensure_ascii=False)
        f.write('\n')
