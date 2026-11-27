"""Apply the accepted observable collision filter before every fixed P7 pool."""
import numpy as np
from scripts.p7_common_v31 import ROOT,need,stat,sealed,load_sealed
from src.risk_observations_v31 import points
from src.grasp_observed_filter_v29 import filter_candidates,independent_mask,pinned_detector

def select(candidates,keep,Group):
 from scripts.verify_grasp_geometry_v10 import exact_indices
 if not keep.any():return []
 group=Group(candidates[keep].copy()).nms(.03,np.pi/6);group.sort_by_score()
 return exact_indices(candidates,group[:20].grasp_group_array,np)

def commit_pool(folder,a,case,Group):
 cand=a['candidates'];cloud,_=points(a['observed_depth'],a['observed_valid'],a['K'],a['original_size']);keep,voxels=filter_candidates(cand,cloud,Group)
 need(np.array_equal(keep,independent_mask(cand,voxels)),'Observed eligibility independent algebra mismatch')
 indices=select(cand,keep,Group);path=folder/'pool_observed_filter.npz';need(not path.exists(),'Unsealed pool filter artifact retained')
 np.savez_compressed(path,keep=keep,voxels=voxels)
 sealed(folder/'pool.json',dict(case=case,indices=indices,selection='Observed demo collision filter then raw score top20 after 0.03m/30deg NMS',all_methods_same_pool=True,GT_used=False,filter_artifact=stat(path),filter_parameters=dict(voxel_size=.01,collision_thresh=.01,approach_dist=.05),eligible=int(keep.sum())))
 return indices

def review_pool(folder,a,Group):
 r=load_sealed(folder/'pool.json');path=folder/'pool_observed_filter.npz';need(stat(path)==r['filter_artifact'],'Observed pool filter artifact changed');saved=np.load(path)
 # Independent original-pixel camera reconstruction, followed by the same pinned voxel primitive.
 z=a['observed_depth'].astype(np.float64);v,u=np.indices(z.shape);width,height=a['original_size'];oy=((2*v+1)*height)//960;ox=((2*u+1)*width)//1280;K=a['K'];cloud=np.stack(((ox-K[0,2])*z/K[0,0],(oy-K[1,2])*z/K[1,1],z),-1);valid=a['observed_valid']&(z>0)&(z<2)&np.isfinite(cloud).all(-1)
 voxels=pinned_detector()(cloud[valid].astype(np.float32),voxel_size=.01).scene_points;need(np.array_equal(voxels,saved['voxels']),'Independent filter voxel reconstruction differs')
 keep=independent_mask(a['candidates'],voxels);need(np.array_equal(keep,saved['keep']) and int(keep.sum())==r['eligible'],'Independent candidate eligibility differs')
 need(r['filter_parameters']==dict(voxel_size=.01,collision_thresh=.01,approach_dist=.05) and not r['GT_used'],'Observed pool filter contract differs')
 indices=select(a['candidates'],keep,Group);need(indices==r['indices'],'Fixed outcome-blind pool differs');saved.close();return indices
