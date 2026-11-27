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

NEGATIVE_SCORE_REASON = "negative_native_score_outside_downstream_multiplicative_domain"


def decoder_domain(candidates):
    """Check structural corruption on every native row before any filtering."""
    if candidates.ndim != 2 or candidates.shape[1] != 17 or candidates.dtype not in (np.float32, np.float64):
        raise ValueError("invalid_native_decoder_matrix")
    native_cap = candidates.dtype.type(.1)
    reasons = []
    for i, row in enumerate(candidates):
        if not np.isfinite(row).all():
            raise ValueError(f"nonfinite_native_decoder_row:{i}")
        rotation = row[4:13].astype(np.float64).reshape(3, 3)
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0) or
                abs(np.linalg.det(rotation) - 1.) > 1e-4):
            raise ValueError(f"invalid_native_rotation:{i}")
        excluded = []
        if row[0] < 0:
            excluded.append(NEGATIVE_SCORE_REASON)
        for name, value in zip(("width", "height", "depth"), row[1:4]):
            if value <= 0:
                excluded.append(f"nonpositive_native_{name}")
        if row[1] > native_cap:
            excluded.append("native_width_above_decoder_dtype_0p1_m")
        reasons.append(excluded)
    return reasons


def observed_rows(candidates, cloud, reasons):
    """Invalid dimensions are not sent to the detector. Negative score does not affect its geometry."""
    normal_geometry = np.array([all(r == NEGATIVE_SCORE_REASON for r in why) for why in reasons],dtype=bool)
    indices = np.flatnonzero(normal_geometry)
    mask, voxels = filter_candidates(candidates[indices], cloud, GraspGroup)
    if not np.array_equal(mask, independent_mask(candidates[indices], voxels)):
        raise ValueError("frozen observation-only mask differs")
    result = np.zeros(len(candidates), dtype=bool)
    result[indices] = mask
    return result, voxels, normal_geometry


def pool_domain_audit(cand, observed_mask, reasons, folder_name, observed_evaluated=None):
    """Exact observed/domain intersection, pinned NMS, stable score/source-index ties."""
    if observed_mask.shape != (len(cand),) or observed_mask.dtype != np.bool_ or len(reasons) != len(cand):
        raise ValueError("invalid_observed_mask_or_native_indices")
    if observed_evaluated is None:
        observed_evaluated=np.ones(len(cand),dtype=bool)
    accepted = np.array([not r for r in reasons], dtype=bool)
    shared = observed_mask & accepted
    viable = np.flatnonzero(shared)
    kept = []
    if len(viable):
        nms = GraspGroup(cand[viable].copy()).nms(.03, np.pi / 6)
        originals = exact_indices(cand[viable], nms.grasp_group_array, np)
        kept = [int(viable[j]) for j in originals]
        # Native score is the primary key; ties retain native decoder order.
        kept.sort(key=lambda i: (-float(cand[i, 0]), i))
    nms_ranks = {k:j for j,k in enumerate(kept)}
    top = kept[:20]
    rows = []
    for i, row in enumerate(cand):
        flags = list(reasons[i])
        if observed_evaluated[i] and not observed_mask[i]:
            flags.append("observed_collision_filter_rejected")
        if shared[i] and i not in nms_ranks:
            flags.append("frozen_nms_suppressed")
        if i in nms_ranks and nms_ranks[i] >= 20:
            flags.append("outside_original_score_top20")
        rows.append(dict(original_index=i, raw_original_17=row.tolist(),
            row_dtype=str(cand.dtype), native_row_sha256=hashlib.sha256(row.tobytes()).hexdigest(),
            raw_score=float(row[0]), score_sign_bit=bool(np.signbit(row[0])),
            domain_eligible=bool(accepted[i]), domain_exclusion_reasons=reasons[i],
            observed_filter_evaluated=bool(observed_evaluated[i]),
            observed_filter_pass=bool(observed_mask[i]) if observed_evaluated[i] else None, eligible_for_nms=bool(shared[i]),
            survived_nms=i in nms_ranks, post_nms_rank_0based=nms_ranks.get(i),
            selected_for_pool=i in top, selected_pool_index=(nms_ranks[i] if i in top else None),
            exclusion_reasons=flags))
    report=dict(unit=folder_name,original_dtype=str(cand.dtype),native_width_cap_m=float(cand.dtype.type(.1)),
        original_candidate_count=len(cand), finite_negative_count=int(np.sum(cand[:,0] < 0)),
        domain_eligible_count=int(accepted.sum()),observed_filter_count=int(observed_mask.sum()),
        domain_and_observed_count=int(shared.sum()), after_nms_count=len(kept), final_pool_count=len(top),
        top20_original_indices=top, unchanged_geometry=True, methods_share_one_pool=["Raw","A","C0"],rows=rows)
    return shared, top, report


def run(folder):
    before=time.perf_counter();blob=folder/'network.npz';net=json.loads((folder/'network.json').read_bytes())
    if sha(blob)!=net['network_output_sha256']:raise ValueError('Input changed')
    with np.load(blob,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    cand=a['candidates']
    reasons=decoder_domain(cand)
    cloud,_=points(a['depth'],a['valid'],a['K'],a['original_size'])
    observed,voxels,evaluated=observed_rows(cand,cloud,reasons)
    keep,ix,domain=pool_domain_audit(cand,observed,reasons,folder.name,evaluated)
    domain["pool_disposition"]="eligible_candidates" if ix else "no_eligible_candidate"
    domain['native_network_npz_sha256']=sha(blob)
    domain_path=folder/'candidate_domain_ledger.json'
    with domain_path.open('x',encoding='utf8') as f:
        json.dump(domain,f,indent=2,allow_nan=False)
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
    report=dict(pool_disposition=domain["pool_disposition"],pool_count=len(ix),decoder_count=len(cand),shared_pool_sha256=sha(pool_path),scores=scores,orders_by_pool_index=orders,candidate_details=detail,eligibility_after_observed_filter=int(observed.sum()),eligibility_after_shared_domain=int(keep.sum()),candidate_domain_ledger_sha256=sha(domain_path),num_voxels=len(voxels),pool_build_sec=pool_seconds,scoring_sec=time.perf_counter()-score_start,network_sha256=sha(blob),source_sha256=sha(__file__),physics_steps=0,G1_volume_m3=.000216,truth_read=False)
    with (folder/'orders.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('network, eligible, shared pool lengths:',len(cand),int(keep.sum()),len(ix))
if __name__=='__main__':run(Path(sys.argv[1]))

