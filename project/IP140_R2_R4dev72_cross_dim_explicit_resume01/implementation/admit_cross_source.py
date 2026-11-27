"""Pre-forward admission. No executable method can be dispatched by this gate."""
import hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parents[1];ROOT=V.parent
PARENT=ROOT/'IP140_R2_R4dev72_nonnegative_score_domain_resume01'
def rd(p):return json.loads(p.read_bytes())
def sh(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('project root')
    gate=rd(V/'audit/reuse_receipt.json')
    assert gate['passed'] and len(gate['parents'])==2 and all(r['counted']==0 and r['remaining']==72 for r in gate['parents'])
    assert rd(V/'development_ledger.json')['method_trials_counted']==0
    inputs=rd(V/'audit/input_files.json')['files']
    source=V/'audit/cross_source/site_audit.json'
    s=rd(source);gpu=rd(V/'audit/cross_gpu_boundary/receipt.json')
    assert s['other_missing_dim_callsites']==0 and s['before_sha256']!=s['after_sha256'] and sh(ROOT/s['source'])==s['after_sha256']==inputs[s['source']]['sha256']
    assert gpu['passed'] and gpu['production_sha256']==s['after_sha256'] and gpu['source_audit_sha256']==sh(source) and gpu['torch']=='2.6.0+cu126' and gpu['physical_steps']==0
    b=rd(V/'audit/boundary_replay_final.json')
    assert b['passed'] and b['total']==b['passed_count']==57
    exact=[]
    for name in ('policy_network','policy_rank','platform_static','platform_episode','scene_world','scene_asset','platform_select','platform_plan','platform_evaluator','platform_support','platform_trace','platform_imports','scene_camera'):
        p=V/'implementation'/(name+'.py')
        assert sh(p)==sh(PARENT/'implementation'/p.name);exact.append(p)
    for p in (V/'preregistration').glob('*.json'):
        assert sh(p)==sh(PARENT/'preregistration'/p.name);exact.append(p)
    files=exact+[source,V/'audit/cross_gpu_boundary/receipt.json',V/'audit/boundary_replay_final.json',V/'implementation/audited_forward.py',V/'audit/reuse_receipt.json',V/'audit/input_files.json']
    result=dict(version=V.name,passed=True,method_trials=0,stepSimulation=0,prior_budget_used=0,authorized_budget_total=72,only_production_algorithm_delta=s['source']+':89, explicit last dimension',approved_source_sha256=s['after_sha256'],previous_source_sha256=s['before_sha256'],pre_forward_artifacts={p.relative_to(V).as_posix():sh(p) for p in files})
    with (V/'audit/zero_interface_admission.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print('Authorized source and stage gates verified')
if __name__=='__main__':main()
