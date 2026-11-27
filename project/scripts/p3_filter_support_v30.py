"""Commit and independently review the fixed observable candidate filter."""
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scripts.p6_common_v24 import ROOT,read,stat,need
from scripts.p3_train_v30 import sealed,load_sealed
from src.grasp_observed_filter_v29 import filter_candidates,independent_mask,pinned_detector

def filtered(folder,candidates,data_dir,frame,Group):
 p=folder/'observed_filter.json';a=folder/'observed_filter.npz'
 if p.exists():r=load_sealed(p);need(stat(a)==r['artifact'],'Postprocess payload changed');return np.load(a)['keep']
 from scripts.generate_multiview_grasps_v14r1 import reconstruct_points
 m=loadmat(data_dir/'meta'/f'{frame:04d}.mat',variable_names=['intrinsic_matrix','factor_depth']);depth=np.asarray(Image.open(data_dir/'depth'/f'{frame:04d}.png'));cloud=reconstruct_points(depth,m['intrinsic_matrix'],float(m['factor_depth'].item()))['valid_points'];keep,voxels=filter_candidates(candidates,cloud,Group);need(np.array_equal(keep,independent_mask(candidates,voxels)),'Independent observed filter mismatch');need(not a.exists(),'Unsealed filter artifact retained');np.savez_compressed(a,keep=keep,voxels=voxels);sealed(p,dict(artifact=stat(a),kept=int(keep.sum()),discarded=int((~keep).sum()),GT_used=False,voxel_size=.01,collision_thresh=.01,approach_dist=.05));return keep

def review_filter(folder,candidates,cloud):
 p=folder/'observed_filter.json';r=load_sealed(p);a=folder/'observed_filter.npz';need(stat(a)==r['artifact'],'Filter artifact changed');saved=np.load(a);detector=pinned_detector()(cloud,voxel_size=.01);need(np.array_equal(detector.scene_points,saved['voxels']),'Independent observed voxel reconstruction differs');keep=independent_mask(candidates,detector.scene_points);need(np.array_equal(keep,saved['keep']) and int(keep.sum())==r['kept'],'Independent observed filter decision differs');return keep
