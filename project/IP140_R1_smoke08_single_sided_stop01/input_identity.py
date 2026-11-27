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
HISTORY['IP140_R1_smoke07_mount_baseline_invariant01']='f66b6347f9bffe786c002fa7b32d2abd8c98779a06ed6818d1b8ddf37dae01ec'
PREVIOUS=ROOT/'IP140_R1_smoke07_mount_baseline_invariant01'
NEW_ID='IP140-BENCH-06:single_sided_stop'

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
    l=read(HERE/'ledger_initial.json')
    states=[row['state'] for row in l['slots'][:5]]
    if (reg['slots']!=pri['slots'] or reg['slots'][5]['candidate_id']!=NEW_ID or reg['eligible_slots_after_zero_gate']!=[6]
        or reg['version']!=HERE.name or reg['retry'] or reg['max_starts']!=12
        or reg['approved_filter_order']!=pri['approved_filter_order'] or reg['wrist_invariant']!=pri['wrist_invariant']
        or reg['required_completion']['steps']!=3216 or reg['source_authorization']['sha256']!=digest(HERE/'approved_task.txt')
        or l['starts']!=5 or l['unused_original_budget']!=7 or l['physical_steps']!=4330 or l['budget_added']!=0
        or states!=['returned_failed']*4+['returned_passed'] or not all(x['charged'] for x in l['slots'][:5])
        or any(x['charged'] for x in l['slots'][5:])):
        raise ValueError('frozen scenario or historical ledger mismatch')
    for e in l['slots'][:5]:
        for k in ('result','telemetry'):
            if digest(HERE/e[k])!=e[k+'_sha256']:
                raise ValueError('charged result or telemetry identity changed')
    for reference in reg['references']:
        if digest(ROOT/reference['project_relative'])!=reference['sha256']:
            raise ValueError('frozen plan dependency changed')
    if l['preregistration_sha256']!=digest(HERE/'preregistration.json'):
        raise ValueError('ledger not bound')
    for name in ['model/manifest.json','model/ur5e_ip140_r1.urdf','execution_control.py','seam_geometry.py','full_api.py','mountpair.py','structural_filter.py','mountpair_registration.json']:
        if digest(HERE/name)!=digest(PREVIOUS/name):
            raise ValueError('frozen model or control changed: '+name)
    manifest=read(HERE/'model/manifest.json')
    for rel,h in {**manifest['provenance'],**manifest['reused_arm_asset_sha256']}.items():
        if digest(ROOT/rel)!=h:
            raise ValueError('physical asset mismatch: '+rel)
    files=sorted({*HERE.glob('*.py'), *[HERE/n for n in ('mountpair_registration.json','ledger_initial.json','preregistration.json','model/manifest.json','model/ur5e_ip140_r1.urdf','history/sources.json','approved_task.txt','trajectory_plan.json')],*[f for f in HERE.glob('*registration*.json')]})
    return dict(schema='ip140_smoke08_inputs_v1',version=HERE.name,executable_slots=[6],attempt_id=NEW_ID,
        total_budget=12,inherited_starts=5,history_inventories=HISTORY,
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
