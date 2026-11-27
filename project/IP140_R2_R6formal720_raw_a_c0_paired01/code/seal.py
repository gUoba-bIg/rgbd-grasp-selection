"""Bind every input, selected identity and method intent once before any method result."""
import collections
import numpy as np
from common import *
def fp(a):
    a=np.ascontiguousarray(a);return dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(),shape=list(a.shape),dtype=a.dtype.str)
def rel(p):return p.relative_to(HERE).as_posix()
def pathref(o,key,p):o[key+'_path']=rel(p);o[key+'_sha256']=sha(p)
def main():
    d=read(HERE/'preregistration/design.json');ds=read(HERE/'preregistration/design_seal.json')
    require(ds['design_sha256']==sha(HERE/'preregistration/design.json') and sha(HERE/'preregistration/frozen_inputs.json')==ds['frozen_inputs_sha256'],'design seal')
    inventory=read(HERE/'audit/scene_inventory.json')['scenes'];require(len(inventory)==120 and read(HERE/'audit/tensor_inputs_complete.json')['count']==240,'input inventory')
    for k,n in [('scene',120),('forward',240),('rank',240)]:
        require(read(HERE/'run'/(k+'_complete.json'))['returned']==n,'unreturned preprocessing jobs '+k)
    require(not any((HERE/'trials').iterdir()) and not (HERE/'run/jobs/method').exists(),'prior formal dispatch or mutable intent')
    deps=read(HERE/'preregistration/frozen_inputs.json')
    require(all(sha(ROOT/k)==v for k,v in deps.items()),'source, weight or asset drift')
    local={rel(p):sha(p) for b in ('code','implementation') for p in (HERE/b).glob('*.py')}
    checks=read(HERE/'audit/input_preflight.json')
    require(all(local[k]==v for k,v in checks['source_files'].items()),'source drift during input preparation')
    M=('Raw','A','C0');inputs={};unitlist={u['unit_id']:u for u in d['units']};scenes={s['base_scene']:s for s in d['scenes']}
    source_stamp=put(HERE/'preregistration/executing_source.json',dict(project_inputs=ds['frozen_inputs_sha256'],local_sources=local,native=sha(BINARY),utc=now()))
    for u in d['units']:
        key=u['unit_id'];base=HERE/'units'/key;sc=scenes[u['scene']];iv=inventory[u['scene']]
        network=read(base/'network.json');dom=read(base/'candidate_domain_ledger.json');o=read(base/'orders.json')
        s=read(base/'selection.json');j=read(base/'eligibility_api/summary.json')
        native=read(HERE/'audit/tensor_inputs'/(key+'.json'))
        require(j['calls']==j['returns'] and not j['exceptions'] and j['actual_native_step_calls']==0 and j['calls']['setCollisionFilterPair']==14,'E audit '+key)
        require(network['condition']==u['condition'] and network['observation_sha256']==iv['observation_sha256']==native['observation_sha256'] and native['noise']==network['corruption'] and network['pose_compensation_m']==0 and network['physics_steps']==0,'formal observation lineage')
        require(network['network_output_sha256']==sha(base/'network.npz')==o['network_sha256']==dom['native_network_npz_sha256'],'network digest')
        require(o['shared_pool_sha256']==sha(base/'pool.npz') and o['candidate_domain_ledger_sha256']==sha(base/'candidate_domain_ledger.json') and s['native_eligibility_api_sha256']==sha(base/'eligibility_api/calls.jsonl.zst') and s['selection_input_sha256']==sha(base/'platform_input.json'),'pool and E provenance')
        require(s['used_ground_truth'] is False and o['truth_read'] is False and dom['methods_share_one_pool']==list(M) and s['never_fallback_after_postlock_safety'] is True and s['unit_id']==key,'prohibited split or fallback')
        raw=np.load(base/'raw_decoder.npy',allow_pickle=False)
        with np.load(base/'network.npz',allow_pickle=False) as n,np.load(base/'pool.npz',allow_pickle=False) as pool:
            require(raw.shape==(len(raw),17) and np.array_equal(raw,n['candidates']) and np.isfinite(raw).all(),'native rows')
            for t,field in [('depth','depth'),('valid','valid'),('K','K'),('world_T_camera','world_T_camera'),('native_size','original_size')]:
                require(fp(n[field])==native[t],'independent input mismatch '+key+'/'+t)
            inds=pool['source_indices'];rows=pool['candidate_camera']
            require(np.array_equal(rows,raw[inds]) and len(rows)==len(s['eligibility'])==o['pool_count'] and len(raw)==dom['original_candidate_count'] and np.all(rows[:,0]>=0),'pool binding')
            selected={}
            for m in M:
                rank=read(base/('ranking_'+m+'.json'));idx=s['selected'][m]
                require(rank['shared_pool_sha256']==o['shared_pool_sha256'] and rank['complete_order']==o['orders_by_pool_index'][m] and idx==next((i for i in rank['complete_order'] if s['eligibility'][i]['eligible']),None),'choice closure')
                entry=None
                if idx is not None:
                    rec=s['eligibility'][idx];det=o['candidate_details'][idx];row=rows[idx]
                    require(rec['plan']['candidate_id']==rec['candidate_id'] and np.isfinite(rec['plan']['candidate_world']).all() and np.array_equal(rec['plan']['candidate_world'],det['candidate_world']) and det['source_index']==int(inds[idx]) and det['old_Robotiq_height_adjustment_applied'] is False,'selected pose identity')
                    entry=dict(native_source_index=int(inds[idx]),pool_index=idx,candidate_id=rec['candidate_id'],camera_row=row.tolist(),camera_original_dtype=row.dtype.str,camera_row_sha256=fp(row)['sha256'],world_17=det['candidate_world'],world_T_G=det['world_T_G'],pre_open_m=det['pre_open_m'],legacy_height_correction_m=0.0)
                selected[m]=entry
        files={rel(p):sha(p) for p in [base/f for f in ['network.json','network.npz','raw_decoder.npy','candidate_domain_ledger.json','pool.npz','orders.json','platform_input.json','selection.json','eligibility_api/summary.json','forward_intent.json',*['ranking_'+m+'.json' for m in M]]]}
        files.update({rel(HERE/'audit/tensor_inputs'/(key+'.json')):sha(HERE/'audit/tensor_inputs'/(key+'.json'))})
        inputs[key]=dict(scene=u['scene'],source_files=files,scene_hashes=iv,model_input_fingerprints=native['fingerprints'],selected=selected,pool_size=o['pool_count'],E_count=sum(r['eligible'] for r in s['eligibility']),observation_condition=u['condition'])
    manifest=put(HERE/'preregistration/actual_unit_inputs.json',dict(version=HERE.name,utc=now(),units=inputs,design_sha256=ds['design_sha256'],source_seal_sha256=source_stamp,formal_results_seen=0))
    files={}
    for t in d['trials']:
        u=unitlist[t['unit_id']];v=inputs[u['unit_id']];base=HERE/'units'/u['unit_id'];sd=HERE/'scenes'/u['scene'];sc=scenes[u['scene']];m=t['method'];chosen=v['selected'][m]
        o=dict(trial_id=t['trial_id'],unit_id=u['unit_id'],method=m,scene_key=u['scene'],object_id=u['object_id'],board_position_index=u['position'],repeat=u['repeat'],condition=u['condition'],selected_pool_index=chosen['pool_index'] if chosen else None,selected_candidate_id=chosen['candidate_id'] if chosen else None,selected_p3=chosen,no_candidate=chosen is None,common_pool_count=v['pool_size'],old_E_count=v['E_count'],attempt_limit=1,ranking_uses_ground_truth=False,no_postlock_fallback=True,r5_interface_in_chain=False,negative_control_passed=None,execution_order=t['order'],within_unit_order=t['within_unit'],worker_seed=t['worker_seed'],sampling_seed=sc['sampling_seed'],condition_seed=sc['condition_seed'][u['condition']],model_input_fingerprints=v['model_input_fingerprints'],source_bundle_sha256=source_stamp,actual_unit_inputs_sha256=manifest,asset=sc['asset'],scope='formal, no historical observations or method results')
        for k,p in [('pool',base/'pool.npz'),('raw_decoder',base/'raw_decoder.npy'),('exclusion',base/'candidate_domain_ledger.json'),('ranking',base/('ranking_'+m+'.json')),('selection',base/'selection.json'),('stable_state',sd/'settled_state.json'),('observation',sd/'observed_rgbd.npz'),('scoring_components',base/'orders.json')]:pathref(o,k,p)
        p=HERE/'trials'/t['trial_id']/'intent.json';files[rel(p)]=put(p,o)
    require(len(files)==720 and all(sha(HERE/k)==v for k,v in files.items()),'incomplete terminal intent seal')
    put(HERE/'sealed_intents.json',dict(version=HERE.name,method_budget=720,utc=now(),files=files,design_sha256=ds['design_sha256'],unit_manifest_sha256=manifest,source_seal_sha256=source_stamp,observations=120,paired_units=240,rankings=720,formal_method_results_seen=0))
if __name__=='__main__':main()
