"""Read-only provenance checks for the single admitted session. No backend import."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OLD = PROJECT/'IP140_R1_smoke01'
HISTORICAL_LIST_SHA256 = '86bf8ebd739a0c3b520bb70d246c159ea79dd49368cd182ec27b955807caf794'
START_INPUTS = ('model/ur5e_ip140_r1.urdf','model/manifest.json','preregistration.json',
                'ledger_initial.json','zero_gate.py','empty_stroke.py','episode_log.py',
                'execution_control.py','input_identity.py','freeze_slots.py','run_once.py')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def snapshot():
    if Path.cwd().resolve()!=PROJECT:
        raise RuntimeError('explicit project root required')
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
    if new['preregistration.json']!='73c2ebfbb0f3d4a16ff9051fd36f0fe638905bafedc3a7bf06e3a2d4081a8c50':
        raise RuntimeError('preregistration does not match verified new selection')
    if registration['max_starts']!=12 or [s['slot'] for s in registration['slots']]!=list(range(1,13)):
        raise RuntimeError('invalid budget or slots')
    expected=(4,16,32,6,18,34)
    for slot,num in zip(registration['slots'][6:],expected):
        if not slot['candidate_id'].startswith(f'v52eng:slot{num:02d}:'):
            raise RuntimeError('wrong historical candidate')
    manifest=read(HERE/'model/manifest.json')
    for k,h in {**manifest['provenance'],**manifest['reused_arm_asset_sha256']}.items():
        if digest(PROJECT/k)!=h:raise RuntimeError('upstream asset changed: '+k)
    return dict(schema='ip140_smoke02_inputs_v1', new_files=new,
        smoke01_inventory_sha256=HISTORICAL_LIST_SHA256,smoke01_verified_files=len(checked),
        executable_slots=[1,2],total_budget=12,maximum_direct_connections=1,maximum_robot_loads=1)


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
