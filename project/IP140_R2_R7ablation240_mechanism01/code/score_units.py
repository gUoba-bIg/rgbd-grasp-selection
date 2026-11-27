"""Apply only the three predeclared switches on the immutable R6 candidate pools."""
import os, sys, shutil, traceback
from collections import Counter
from common import *
from freeze_sample import PARENT, ARMS
sys.path.insert(0,str(R4/'implementation'))
from policy_rank import g1_geometry
from src import p56_geometry_v51 as geom
import numpy as np

def copy_input(src,dst):
    if dst.exists():
        require(sha(src)==sha(dst),'refusing conflicting duplicate '+str(dst))
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as a,dst.open('xb') as b:
        shutil.copyfileobj(a,b)
    require(sha(src)==sha(dst),'input copy')

def evaluate(u,binding,prior,model):
    uid=u['unit_id'];old=PARENT/'units'/uid; local=HERE/'units'/uid
    expected=binding['source_files']
    for rel,h in expected.items():
        require(sha(PARENT/rel)==h,'R6 registered unit identity '+rel)
    scene=PARENT/'scenes'/u['scene']
    for name,k in [('observed_rgbd.npz','observation_sha256'),('settled_state.json','state_sha256'),('receipt.json','receipt_sha256')]:
        require(sha(scene/name)==binding['scene_hashes'][k],'frozen scene '+uid)
    copy_input(scene/'settled_state.json',HERE/'scenes'/u['scene']/'settled_state.json')
    copy_input(old/'pool.npz',local/'pool.npz')
    archived=read(old/'orders.json');sel=read(old/'selection.json')
    require(not archived['truth_read'] and not sel['used_ground_truth'] and sel['never_fallback_after_postlock_safety'],'old E provenance')
    require(archived['shared_pool_sha256']==sha(local/'pool.npz') and sha(old/'network.npz')==archived['network_sha256'],'score inputs')
    with np.load(old/'network.npz',allow_pickle=False) as n:
        obs={k:n[k] for k in ['depth','valid','K','original_size','instances','foreground_probability','pixels','candidates']}
    with np.load(local/'pool.npz',allow_pickle=False) as pool:
        rows=pool['candidate_camera'];sources=pool['source_indices'];seeds=pool['seed_pixels']
    require(np.array_equal(rows,obs['candidates'][sources]) and np.array_equal(seeds,obs['pixels'][sources]),'unaligned frozen rows')
    sc=geom.prepare_scene_geometry(obs['depth'],obs['valid'],obs['K'],obs['original_size'])
    scale=np.full(sc['image_shape'],.005)
    zero=np.zeros_like(prior);zi=np.flatnonzero((geom.mixture.CANDIDATES==0).all(axis=1))
    require(len(zi)==1,'unique zero displacement');zero[zi[0]]=1.
    require(len(rows)==len(sel['eligibility'])==archived['pool_count'],'old E dimensionality')
    scores={m:[] for m in ARMS};details=[]
    for i,(row,source,seed) in enumerate(zip(rows,sources,seeds,strict=True)):
        rec=archived['candidate_details'][i]; original=rec['score_components']
        require(int(source)==rec['source_index'] and rec['old_Robotiq_height_adjustment_applied'] is False,'G1 source')
        geometry=g1_geometry(row,sc)
        assoc=geom.association_by_warp(obs['instances'],obs['foreground_probability'],int(seed),geometry['closing_pixels'])
        support=geom.anchor_support_by_warp(sc,geometry,row,scale,model['shape'])
        a=assoc['association'];b=support['side_support']
        require(assoc['target_id']==rec['seed_predicted_local_instance'] and support['anchor_counts']==rec['anchors'],'frozen contact evidence')
        # Component identity checks only. No C0 method trial, reranking or physics is run.
        for key,mode,p in [('A','none',zero),('C0','fixed',prior)]:
            require(geom.aggregate_contact(a,b[mode],p)==original[key]['contact_support'],'frozen contact component drift '+uid)
        parts={}
        for m,assc,side,weights,cp in [
                (ARMS[0],np.ones_like(a),'fixed',prior,'C0'),
                (ARMS[1],a,'fixed',zero,'C0'),
                (ARMS[2],a,'none',prior,'A')]:
            contact=geom.aggregate_contact(assc,b[side],weights)
            collision=original[cp]['collision_proxy']
            value=float(row[0])*contact*(1-collision)
            require(np.isfinite([value,collision,contact]).all() and 0<=contact<=1+np.finfo(float).eps and 0<=collision<=1,'nonfinite/range '+uid)
            scores[m].append(value)
            parts[m]=dict(contact_support=contact,collision_proxy=collision,collision_source=cp,rank_score=value)
        details.append(dict(pool_index=i,native_index=int(source),candidate_id=sel['eligibility'][i]['candidate_id'],
                            seed_instance=assoc['target_id'],seed_pixel=int(seed),raw_score=float(row[0]),
                            association_81=a.tolist(),point_depth_sides=b['none'].tolist(),fixed_depth_sides=b['fixed'].tolist(),arms=parts))
    orders={m:np.argsort(-np.array(ss),kind='stable').tolist() for m,ss in scores.items()}
    choices={m:next((i for i in order if sel['eligibility'][i]['eligible']),None) for m,order in orders.items()}
    comp=put(local/'scoring_components.json',dict(unit_id=uid,shared_pool_sha256=sha(local/'pool.npz'),legacy_orders_sha256=sha(old/'orders.json'),
           no_new_forward=True,fixed_prior=prior.tolist(),zero_offset_index=int(zi[0]),candidate_details=details,truth_read=False,trial_results_read=False))
    put(local/'selection.json',{**sel,'selected':choices,'r6_old_E_sha256':sha(old/'selection.json')})
    for m in ARMS:
        put(local/('ranking_'+m+'.json'),dict(unit_id=uid,method=m,shared_pool_sha256=sha(local/'pool.npz'),
            scores_by_pool_index=scores[m],complete_order=orders[m],candidate_ids_in_order=[sel['eligibility'][i]['candidate_id'] for i in orders[m]],
            scores_source_sha256=comp))
    return dict(unit_id=uid,pool_size=len(rows),old_E_size=sum(e['eligible'] for e in sel['eligibility']),choices=choices,
                scoring_sha256=comp,source_files=expected,scene_hashes=binding['scene_hashes'])

def main():
    require((HERE/'audit/r6_final_state.json').is_file(),'missing parent seal')
    d=read(HERE/'preregistration/design.json')
    require(sha(HERE/'preregistration/design.json')==read(HERE/'preregistration/design_seal.json')['design_sha256'],'sampler changed')
    lib=read(PARENT/'preregistration/frozen_inputs.json')
    for k in ['src/p56_geometry_v51.py','results/P6/local_correspondence_v50/model.json',R4.name+'/implementation/policy_rank.py']:
        require(sha(ROOT/k)==lib[k], 'frozen scoring kernel '+k)
    model=read(ROOT/'results/P6/local_correspondence_v50/model.json')
    prior=geom.mixture.probability_vector(np.asarray(model['empirical_prior']),81)
    registry=read(PARENT/'preregistration/actual_unit_inputs.json')['units']
    audited={}
    for u in d['units']:
        audited[u['unit_id']]=evaluate(u,registry[u['unit_id']],prior,model)
    require(len(audited)==80,'incomplete scores')
    put(HERE/'audit/score_replay.json',dict(units=audited,units_count=80,rankings=240,complete=True,
        reused_observations=True,source_pools_sha256_verified=True,c0_outcomes_accessed=0,physics_steps=0))
    print('80 old E sets, 240 rankings frozen',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        put(HERE/'run/scoring_fault.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()))
        raise
