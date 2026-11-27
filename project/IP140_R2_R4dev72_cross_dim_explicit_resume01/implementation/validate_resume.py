"""Bind expired parent receipts, unchanged resources and authorized input-domain change."""
import json,hashlib
from pathlib import Path
V=Path(__file__).resolve().parents[1]
ROOT=V.parent
PARENT=ROOT/'IP140_R2_R4dev72_raw_a_c0_paired01'
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('project root required')
    assert read(V/'development_ledger.json')['method_trials_counted']==0
    parent=read(PARENT/'development_ledger.json')
    assert parent['method_trials_counted']==0 and parent['remaining']==parent['budget']==72
    assert read(PARENT/'STOP.json')['method_trials_counted']==0
    r=read(V/'audit/platform_resume_interface_receipt.json')
    assert r['result']=='pass' and r['actual_worker_sha256']==sha(V/'implementation/platform_trial.py')
    boundary=read(V/'audit/boundary_replay_final.json');assert boundary['passed'] and boundary['passed_count']==boundary['total']
    reuse=V/'audit/reuse_receipt.json'
    assert read(reuse)['passed'] and boundary['new_shared_pool_source_sha256']==sha(V/'implementation/policy_rank.py')
    design=read(V/'preregistration/paired_design.json')
    assert len(design['units'])==24 and len(design['control_order'])==72
    sealed=[]
    for name in ('paired_design.json','statistical_protocol.json','input_contract.json'):
        a=V/'preregistration'/name;b=PARENT/'preregistration'/name
        assert sha(a)==sha(b);sealed.append(a)
    for name in ('platform_episode','platform_evaluator','platform_plan','platform_static','platform_support','scene_camera','scene_world','scene_asset','policy_network','platform_select','platform_trace'):
        a=V/'implementation'/(name+'.py');b=PARENT/'implementation'/(name+'.py')
        assert sha(a)==sha(b);sealed.append(a)
    for base in {u['base_scene'] for u in design['units']}:
        files=['observed_rgbd.npz','settled_state.json','receipt.json','api/summary.json']
        for name in files:
            a=V/'scenes'/base/name;b=PARENT/'scenes'/base/name
            assert a.stat().st_size==b.stat().st_size and sha(a)==sha(b)
        rscene=read(V/'scenes'/base/'receipt.json')
        assert rscene['actual_steps']==rscene['step_returns']==240 and rscene['method_trials']==0
    for u in design['units'][:6]:
        key=u['unit_id']
        for n in ('network.npz','network.json','network.log'):
            a=V/'cache'/key/n;b=PARENT/'units'/key/n
            assert a.stat().st_size==b.stat().st_size and sha(a)==sha(b)
    receipt=dict(version=V.name,passed=True,terminal_parent_retained=True,method_trials=0,stepSimulation=0,
        underlying_authorization_total=72,additional_trials=0,frozen_scene_assets_reused=12,reused_network_pairs=6,
        source_authentication={str(p.relative_to(V).as_posix()):sha(p) for p in sealed+[reuse,V/'audit/platform_resume_interface_receipt.json',V/'audit/boundary_replay_final.json',V/'implementation/policy_rank.py']},
        original_interface_path=PARENT.name+'/audit/zero_interface_admission.json',
        original_interface_sha256=sha(PARENT/'audit/zero_interface_admission.json'))
    with (V/'audit/zero_interface_admission.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print('Resume scope and source checks passed')
if __name__=='__main__':main()
