"""Pinned demo collision filter; observations only, no object labels/poses."""
import importlib.util
from pathlib import Path
import numpy as np

def pinned_detector():
 root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p3_observed_detector_v29',root/'runtime/build/graspnet/utils/collision_detector.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.ModelFreeCollisionDetector

def filter_candidates(candidates,points,Group):
 detector=pinned_detector()(points,voxel_size=.01);collision=[]
 for start in range(0,len(candidates),64):collision.extend(detector.detect(Group(candidates[start:start+64]),approach_dist=.05,collision_thresh=.01).tolist())
 return ~np.array(collision,bool),detector.scene_points

def independent_mask(candidates,voxels):
 accepted=[]
 for g in candidates:
  # Preserve upstream float32 parameter arithmetic while forming points in float64.
  height=g[2:3,None];width=g[1:2,None];depth=g[3:4,None];q=(voxels-g[13:16])@g[4:13].reshape(3,3);x,y,z=q[:,0:1].T,q[:,1:2].T,q[:,2:3].T;h=(z>-height/2)&(z<height/2);long=(x>depth-.06)&(x<depth);lo=y>-(width/2+.01);hi=y<(width/2+.01);left=h&long&lo&(y<-width/2);right=h&long&hi&(y>width/2);bottom=h&lo&hi&(x<=depth-.06)&(x>depth-.06-.01);approach=h&lo&hi&(x<=depth-.06-.01)&(x>depth-.06-.01-.05);volume=(height*.06*.01/.01**3)*2+(height*(width+.02)*.01/.01**3)+(height*(width+.02)*.05/.01**3);ratio=(left|right|bottom|approach).sum()/(volume.reshape(-1)[0]+1e-6);accepted.append(bool(ratio<=.01))
 return np.array(accepted,bool)
