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
    'IP140_R1_smoke04_structural_filter_numericband01':'6b0cca00b8e9b4cb6cc95650b32aa6f6699f44b827dd7cfe271f435496df79a3',
    'IP140_R1_smoke05_fixed_pad_mount_filter01':'6c92d61e71c31ef5e73120a6659830d3d04f3f850d8e0d288196e56f16dc3960',
    'IP140_R1_smoke06_sliding_seam_acm01':'de2d751d438a666b93eb04c44e5e8de2dece9cbc54b3e7b6634a87c90ab57d0c'}
PREVIOUS=ROOT/'IP140_R1_smoke06_sliding_seam_acm01'
NEW_ID='IP140-BENCH-EMPTY-05:mount-baseline-invariant'

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
    reference[4]=dict(reference[3],slot=5,candidate_id=NEW_ID,accept='4320 intent/return pairs, zero unknown, all stations and all tail windows complete, no failed hard gate including postfilter fixed wrist baseline invariant.')
    l=read(HERE/'ledger_initial.json')
    if (reg['slots']!=reference or reg['deferred_center']!=pri['deferred_center'] or reg['deferred_offset_positive_y']!=pri['deferred_offset_positive_y'] or reg['deferred_offset_negative_y']['prior_definition']!=pri['slots'][4]
        or reg['eligible_slots_after_zero_gate']!=[5] or reg['version']!=HERE.name or reg['max_starts']!=12
        or l['starts']!=4 or l['unused_original_budget']!=8 or l['physical_steps']!=10 or l['budget_added']!=0
        or any(not s['charged'] or s['state']!='returned_failed' for s in l['slots'][:4])
        or any(s['charged'] for s in l['slots'][4:])):
        raise ValueError('registration or historic budget mismatch')
    if l['preregistration_sha256']!=digest(HERE/'preregistration.json'):
        raise ValueError('ledger not bound')
    for name in ['model/manifest.json','model/ur5e_ip140_r1.urdf','execution_control.py','seam_geometry.py','full_api.py']:
        if digest(HERE/name)!=digest(PREVIOUS/name):
            raise ValueError('frozen model or control changed: '+name)
    manifest=read(HERE/'model/manifest.json')
    for rel,h in {**manifest['provenance'],**manifest['reused_arm_asset_sha256']}.items():
        if digest(ROOT/rel)!=h:
            raise ValueError('physical asset mismatch: '+rel)
    files=sorted({*HERE.glob('*.py'), *[HERE/n for n in ('mountpair_registration.json','ledger_initial.json','preregistration.json','model/manifest.json','model/ur5e_ip140_r1.urdf','history/sources.json','approved_task.txt')],*[f for f in HERE.glob('*registration*.json')]})
    return dict(schema='ip140_smoke07_inputs_v1',version=HERE.name,executable_slots=[5],attempt_id=NEW_ID,
        total_budget=12,inherited_starts=4,history_inventories=HISTORY,
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
