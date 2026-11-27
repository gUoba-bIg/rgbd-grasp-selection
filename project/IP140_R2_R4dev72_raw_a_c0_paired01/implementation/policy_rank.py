"""Freeze an observation-only common pool and the three G1 rankings. Does not start physics."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='2'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime/build/graspnetAPI_eval_v7')]
from graspnetAPI import GraspGroup
from src.grasp_observed_filter_v29 import filter_candidates,independent_mask
from src.risk_observations_v31 import points
from scripts.verify_grasp_geometry_v10 import exact_indices
from src import p56_geometry_v51 as geom

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def g1_geometry(c,scene):
    _,R,t=geom._candidate(c)
    # The v52 camera-space GraspNet origin is t. Tip G translates +depth on its local x.
    d=float(c[3]);w=float(c[1])
    lo=np.array([[d-.060,-w/2-.010,-.010],[d-.060,w/2,-.010],[d-.070,-.080,-.010],[d-.120,-.080,-.010]])
    hi=np.array([[d,-w/2,.010],[d,w/2+.010,.010],[d-.060,.080,.010],[d-.070,.080,.010]])
    close_lo=np.array([[d-.060,-w/2,-.010],[d-.060,0,-.010]])
    close_hi=np.array([[d,0,.010],[d,w/2,.010]])
    volume=float(np.prod(hi-lo,axis=1).sum())
    if abs(volume-.000216)>1e-15:raise ValueError('G1 box volume')
    boxes=dict(lower=lo,upper=hi,upper_x_closed=[False,False,True,True],closing_lower=close_lo,closing_upper=close_hi,volume_m3=volume,names=['left','right','palm','approach'])
    q=(scene['points']-t)@R
    support=np.all((q>close_lo[0])&(q<close_hi[1]),axis=1)
    p=scene['pixel_indices']
    ids=[p[support&(q[:,1]<0)],p[support&(q[:,1]>=0)]]
    anchors=[geom._fixed_subset(a,geom.FROZEN_METHOD['anchors_per_side']) for a in ids]
    return dict(closing_pixels=p[support],side_anchor_pixels=anchors,side_closing_counts=[len(i) for i in ids],forbidden_boxes=boxes,local_rays=scene['rays']@R,local_origin=-t@R,local_observed=q)

def run(folder):
    before=time.perf_counter();blob=folder/'network.npz';net=json.loads((folder/'network.json').read_bytes())
    if sha(blob)!=net['network_output_sha256']:raise ValueError('Input changed')
    with np.load(blob,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    cand=a['candidates'];cloud,_=points(a['depth'],a['valid'],a['K'],a['original_size'])
    keep,voxels=filter_candidates(cand,cloud,GraspGroup)
    if not np.array_equal(keep,independent_mask(cand,voxels)):raise ValueError('frozen observation-only mask differs')
    if keep.any():
        g=GraspGroup(cand[keep].copy()).nms(.03,np.pi/6);g.sort_by_score()
        ix=exact_indices(cand,g[:20].grasp_group_array,np)
    else:ix=[]
    # P3 pred_decode applied its native 100 mm width cap, so no post-hoc width edit is needed.
    if not np.all((cand[:,1]>=0)&(cand[:,1]<=np.float32(.1))):raise ValueError('P3 width outside frozen decoder range')
    pool_path=folder/'pool.npz'
    with pool_path.open('xb') as f:np.savez_compressed(f,source_indices=np.asarray(ix,np.int64),candidate_camera=cand[ix],seed_pixels=a['pixels'][ix],eligibility_mask=keep)
    pool_seconds=time.perf_counter()-before
    score_start=time.perf_counter()
    scene=geom.prepare_scene_geometry(a['depth'],a['valid'],a['K'],a['original_size'])
    model=json.loads((ROOT/'results/P6/local_correspondence_v50/model.json').read_bytes());prior=np.asarray(model['empirical_prior']);zero=np.zeros(81);zero[np.flatnonzero((geom.mixture.CANDIDATES==0).all(axis=1))[0]]=1
    # The frozen shared helpers also expose non-used learned branches; only fixed/none values go to the ranking.
    scale_map=np.full(scene['image_shape'],.005)
    scores={m:[] for m in ('Raw','A','C0')}; detail=[]
    T=a['world_T_camera']
    for i in ix:
        row=cand[i];geometry=g1_geometry(row,scene)
        assoc=geom.association_by_warp(a['instances'],a['foreground_probability'],int(a['pixels'][i]),geometry['closing_pixels'])
        coll=geom.collision_proxies(scene,geometry,scale_map,model['shape'])
        support=geom.anchor_support_by_warp(scene,geometry,row,scale_map,model['shape'])
        scores['Raw'].append(float(row[0])); components={}
        for m,mode,p in [('A','none',zero),('C0','fixed',prior)]:
            contact=geom.aggregate_contact(assoc['association'],support['side_support'][mode],p)
            collision=coll[mode]['collision_proxy']
            score=float(row[0])*contact*(1-collision)
            scores[m].append(score)
            components[m]=dict(contact_support=contact,collision_proxy=collision,score=score)
        c_w=np.array(row,dtype=np.float64);c_w[13:16]=T[:3,:3]@row[13:16]+T[:3,3];c_w[4:13]=(T[:3,:3]@row[4:13].reshape(3,3)).reshape(-1)
        R=c_w[4:13].reshape(3,3)
        G=np.eye(4);G[:3,:3]=R;G[:3,3]=c_w[13:16]+row[3]*R[:,0]
        detail.append(dict(source_index=int(i),candidate_world=c_w.tolist(),world_T_G=G.tolist(),pre_open_m=float(row[1]),old_Robotiq_height_adjustment_applied=False,score_components=components,seed_predicted_local_instance=assoc['target_id'],anchors=support['anchor_counts']))
    orders={m:np.argsort(-np.array(s),kind='stable').tolist() for m,s in scores.items()}
    report=dict(pool_count=len(ix),decoder_count=len(cand),shared_pool_sha256=sha(pool_path),scores=scores,orders_by_pool_index=orders,candidate_details=detail,eligibility_after_observed_filter=int(keep.sum()),num_voxels=len(voxels),pool_build_sec=pool_seconds,scoring_sec=time.perf_counter()-score_start,network_sha256=sha(blob),source_sha256=sha(__file__),physics_steps=0,G1_volume_m3=.000216,truth_read=False)
    with (folder/'orders.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('network, eligible, shared pool lengths:',len(cand),int(keep.sum()),len(ix))
if __name__=='__main__':run(Path(sys.argv[1]))

