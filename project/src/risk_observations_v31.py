"""Observed-input preparation and label-free feature algebra for P7."""
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from pathlib import Path

BASE_FEATURES=['raw_confidence','width_m','insertion_depth_m','seed_z_m','approach_z','quality_sigma_descriptive','instance_mean_sigma_descriptive','predicted_valid','predicted_anomaly','foreground_probability','predicted_instance_area_fraction','predicted_instance_boundary']
LOCAL_FEATURES=['observed_collision','observed_empty','occupied_global','occupied_left','occupied_right','occupied_bottom','occupied_approach']

def prepare(root,scene,frame,condition,epoch):
 import torch
 from src.p5_quality_v23 import corrupt
 d=Path(root)/f'scenes/scene_{scene:04d}/realsense';m=loadmat(d/f'meta/{frame:04d}.mat',variable_names=['intrinsic_matrix','factor_depth']);K=m['intrinsic_matrix'];factor=float(m['factor_depth'].item())
 with Image.open(d/f'rgb/{frame:04d}.png') as im:width,height=im.size;rgb=np.asarray(im.convert('RGB').resize((640,480),Image.Resampling.BILINEAR),np.float32)/255.
 rgb=(rgb-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
 with Image.open(d/f'depth/{frame:04d}.png') as im:depth=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32)/factor
 valid=np.isfinite(depth)&(depth>0);x=np.concatenate((rgb.transpose(2,0,1),np.where(valid,np.minimum(depth,2.)/2,0)[None],valid[None].astype(np.float32)),axis=0).astype(np.float32)
 changed,unused_targets,description=corrupt(torch.from_numpy(x),dict(scene=scene,frame=f'{frame:04d}'),epoch,condition)
 del unused_targets
 return changed.numpy(),K,(width,height),description

def points(observed,valid,K,original_size):
 width,height=original_size;v,u=np.indices(observed.shape);oy=((2*v+1)*height)//960;ox=((2*u+1)*width)//1280;z=observed.astype(np.float64);cloud=np.stack(((ox-K[0,2])*z/K[0,0],(oy-K[1,2])*z/K[1,1],z),-1);mask=valid&(z>0)&(z<2.)&np.isfinite(cloud).all(-1);idx=np.flatnonzero(mask)
 return cloud.reshape(-1,3)[idx].astype(np.float32),idx

def base_features(candidates,pixels,sigma,valid,anomaly,foreground,instances):
 edge=np.zeros(instances.shape,bool);d=(instances[1:]!=instances[:-1]);edge[1:]|=d;edge[:-1]|=d;d=(instances[:,1:]!=instances[:,:-1]);edge[:,1:]|=d;edge[:,:-1]|=d;edge=binary_dilation(edge,iterations=2)
 area=np.bincount(instances.ravel());weighted=np.bincount(instances.ravel(),weights=sigma.ravel());means=weighted/np.maximum(area,1);ids=instances.ravel()[pixels]
 return np.column_stack((candidates[:,0],candidates[:,1],candidates[:,3],candidates[:,15],candidates[:,10],sigma.ravel()[pixels],means[ids],valid.ravel()[pixels],anomaly.ravel()[pixels],foreground.ravel()[pixels],np.where(ids>0,area[ids]/instances.size,0),edge.ravel()[pixels])).astype(np.float32)
