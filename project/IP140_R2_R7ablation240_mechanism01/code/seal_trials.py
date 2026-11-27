"""Sign the actual 240 choices, R6 references and executable paths before opening any trial."""
import ast,sys,collections
import numpy as np
from common import *
from freeze_sample import PARENT,ARMS

def relative(p):return p.relative_to(HERE).as_posix()

def addlocal(values, path):
    values[path.relative_to(ROOT).as_posix()]=sha(path)

def main():
    require(not list((HERE/'trials').iterdir()) and not (HERE/'run/jobs').exists(),'existing R7 trial')
    d=read(HERE/'preregistration/design.json');dseal=read(HERE/'preregistration/design_seal.json')
    require(sha(HERE/'preregistration/design.json')==dseal['design_sha256'],'sample drift')
    previous=read(HERE/'audit/r6_provenance.json');audit=read(HERE/'audit/score_replay.json')
    require(audit['complete'] and audit['units_count']==80 and audit['rankings']==240 and not audit['physics_steps'],'scores incomplete')
    protected=dict(previous['files'])
    addlocal(protected,PARENT/'SHA256SUMS.txt')
    for name,h in ((d['original_proposal'],dseal['plan_sha256']),(d['frozen_definitions'],dseal['definitions_sha256'])):protected[name]=h
    release=read(PARENT/'protocol/release.json')
    diffread=(HERE/'implementation/platform_trial.py').read_bytes()
    taskbytes=(PARENT/'implementation/platform_trial.py').read_bytes()
    a=b"sealed['version']!='IP140_R2_R6formal720_raw_a_c0_paired01' or sealed['method_budget']!=720 or len(sealed['files'])!=720"
    b=b"sealed['version']!=VERSION.name or sealed['method_budget']!=240 or len(sealed['files'])!=240"
    require(taskbytes.count(a)==1 and diffread==taskbytes.replace(a,b),'unexpected platform changes')
    for folder in ('code','implementation'):
        for p in (HERE/folder).glob('*.py'):
            ast.parse(p.read_text(encoding='utf8'))
            addlocal(protected,p)
            if p.name!='platform_trial.py' and (PARENT/relative(p)).exists():
                require(sha(p)==sha(PARENT/relative(p)),'changed shared runtime '+str(p))
    for fil in ('audit/r6_provenance.json','audit/r6_final_state.json','audit/score_replay.json','audit/execution_adapter.diff',
                'preregistration/design.json','preregistration/design_seal.json'):
        addlocal(protected,HERE/fil)
    olddesign=read(PARENT/'preregistration/design.json')
    oldc0={t['unit_id']:t['trial_id'] for t in olddesign['trials'] if t['method']=='C0'}
    parent_index=read(PARENT/'preregistration/actual_unit_inputs.json')['units']
    scenes={s['base_scene']:s for s in d['scenes']};units={u['unit_id']:u for u in d['units']}
    evidence={};components={}
    for uid,u in units.items():
        sd=PARENT/'scenes'/u['scene'];base=HERE/'units'/uid;old=PARENT/'units'/uid; src=parent_index[uid]
        protected.update({(PARENT/k).relative_to(ROOT).as_posix():h for k,h in src['source_files'].items()})
        for n,k in [('settled_state.json','state_sha256'),('observed_rgbd.npz','observation_sha256'),('receipt.json','receipt_sha256')]:
            protected[(sd/n).relative_to(ROOT).as_posix()]=src['scene_hashes'][k]
        sel=read(base/'selection.json');orig=read(old/'selection.json')
        require(sel['eligibility']==orig['eligibility'] and sel['r6_old_E_sha256']==sha(old/'selection.json') and not sel['used_ground_truth'],'E changed')
        require((base/'pool.npz').read_bytes()==(old/'pool.npz').read_bytes() and sha(HERE/'scenes'/u['scene']/'settled_state.json')==src['scene_hashes']['state_sha256'],'stable input')
        pidx=oldc0[uid];pc=PARENT/'trials'/pidx/'intent.json';c0=read(pc)
        require(c0['method']=='C0' and c0['unit_id']==uid and c0['stable_state_sha256']==src['scene_hashes']['state_sha256'] and c0['old_E_count']==sum(k['eligible'] for k in orig['eligibility']),'paired C0 reference')
        evidence[uid]=dict(r6_trial_id=pidx,source_hashes=previous['all_720_opaque_hashes'][pidx],scene=u['scene'],reference_intent_sha256=sha(pc))
        order=read(old/'orders.json');perunit={}
        with np.load(base/'pool.npz',allow_pickle=False) as f:
            rows=f['candidate_camera'];indices=f['source_indices']
            for m in ARMS:
                rank=read(base/('ranking_'+m+'.json'))
                idx=next((i for i in rank['complete_order'] if orig['eligibility'][i]['eligible']),None)
                require(idx==sel['selected'][m] and sorted(rank['complete_order'])==list(range(len(rows))) and rank['shared_pool_sha256']==sha(base/'pool.npz') and np.isfinite(rank['scores_by_pool_index']).all(),'first E or scores')
                chosen=None
                if idx is not None:
                    rec=sel['eligibility'][idx];dt=order['candidate_details'][idx];row=rows[idx]
                    require(rec['plan']['candidate_id']==rec['candidate_id'] and np.array_equal(rec['plan']['candidate_world'],dt['candidate_world']) and int(indices[idx])==dt['source_index'],'motion plan identity')
                    chosen=dict(native_source_index=int(indices[idx]),pool_index=idx,candidate_id=rec['candidate_id'],camera_row=row.tolist(),camera_original_dtype=row.dtype.str,camera_row_sha256=hashlib.sha256(row.tobytes()).hexdigest(),world_17=dt['candidate_world'],world_T_G=dt['world_T_G'],pre_open_m=dt['pre_open_m'],legacy_height_correction_m=0.0)
                perunit[m]=(idx,chosen)
        components[uid]=perunit
        for p in [base/'pool.npz',base/'selection.json',base/'scoring_components.json',HERE/'scenes'/u['scene']/'settled_state.json',*[base/('ranking_'+m+'.json') for m in ARMS]]:
            addlocal(protected,p)
    refh=put(HERE/'preregistration/full_C0_reference_index.json',dict(units=evidence,count=80,outcome_fields_read_in_selection=0))
    addlocal(protected,HERE/'preregistration/full_C0_reference_index.json')
    for path,want in protected.items():require(sha(ROOT/path)==want,'frozen input/source drift '+path)
    release_hash=put(HERE/'preregistration/input_release.json',dict(files=protected,scope='frozen parent inputs/results and code; new outputs use exclusive files',parent=PARENT.name,utc=now()))
    files={}
    for t in d['trials']:
        uid=t['unit_id'];u=units[uid];m=t['method'];sc=scenes[u['scene']];base=HERE/'units'/uid;idx,chosen=components[uid][m]
        ri=evidence[uid];pi=read(PARENT/'trials'/ri['r6_trial_id']/'intent.json')
        trial=dict(trial_id=t['trial_id'],unit_id=uid,method=m,scene_key=u['scene'],object_id=u['object_id'],
            board_position_index=u['position'],repeat=u['repeat'],condition=u['condition'],selected_pool_index=idx,
            selected_candidate_id=chosen['candidate_id'] if chosen else None,selected_p3=chosen,no_candidate=chosen is None,
            old_E_count=pi['old_E_count'],common_pool_count=pi['common_pool_count'],attempt_limit=1,no_postlock_fallback=True,
            ranking_uses_ground_truth=False,r5_interface_in_chain=False,negative_control_passed=None,execution_order=t['order'],
            within_unit_order=t['within_unit'],worker_seed=t['worker_seed'],sampling_seed=sc['sampling_seed'],condition_seed=sc['condition_seed'][u['condition']],
            model_input_fingerprints=pi['model_input_fingerprints'],source_bundle_sha256=release_hash,asset=sc['asset'],
            reference_r6=ri,archived_observation_project_path=(PARENT/pi['observation_path']).relative_to(ROOT).as_posix(),
            observation_sha256=pi['observation_sha256'],raw_decoder_sha256=pi['raw_decoder_sha256'],
            exclusion_sha256=pi['exclusion_sha256'],r6_parent_old_E_sha256=pi['selection_sha256'])
        for k,p in [('pool',base/'pool.npz'),('selection',base/'selection.json'),('scoring_components',base/'scoring_components.json'),
                    ('ranking',base/('ranking_'+m+'.json')),('stable_state',HERE/'scenes'/u['scene']/'settled_state.json')]:
            trial[k+'_path']=relative(p);trial[k+'_sha256']=sha(p)
        f=HERE/'trials'/t['trial_id']/'intent.json';files[relative(f)]=put(f,trial)
    require(len(files)==240 and all(sha(HERE/k)==h for k,h in files.items()),'intent cardinality')
    put(HERE/'sealed_intents.json',dict(version=HERE.name,method_budget=240,files=files,units=80,reference_only=80,reference_index_sha256=refh,
        input_release_sha256=release_hash,design_sha256=dseal['design_sha256'],new_results_seen=0,utc=now()))
    print('sealed all 240 one-shot R7 intents',flush=True)

if __name__=='__main__':
    main()
