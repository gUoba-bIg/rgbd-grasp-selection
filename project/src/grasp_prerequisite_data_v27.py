"""Explicit-scene GraspNet loader; preserve upstream sampling, map dense labels.

This new wrapper never invokes the upstream 100-scene/test_seen constructor.
Boolean visibility is applied to indices before reading dense arrays, avoiding
whole-object copies. The output must be compared to upstream with identical RNG.
"""
import os,sys,struct,zipfile,collections,functools
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runtime/build/graspnet'
sys.path[:0]=[str(SOURCE/p) for p in ['models','utils','pointnet2','knn','dataset']]
from graspnet_dataset import GraspNetDataset,collate_fn
from data_utils import CameraInfo,create_point_cloud_from_depth_image,get_workspace_mask,remove_invisible_grasp_points

@functools.lru_cache(maxsize=384)
def mapped_npz(path,key):
 path=Path(path)
 with zipfile.ZipFile(path) as z:info=z.getinfo(key+'.npy')
 if info.compress_type!=zipfile.ZIP_STORED:raise ValueError('Dense NPZ must be ZIP_STORED for bounded memory: '+str(path))
 with path.open('rb') as f:
  f.seek(info.header_offset);header=struct.unpack('<4s5H3I2H',f.read(30))
  if header[0]!=b'PK\x03\x04' or header[2]&1:raise ValueError('Invalid inner ZIP header')
  f.seek(header[-2]+header[-1],1);start=f.tell();version=np.lib.format.read_magic(f)
  shape,fortran,dtype=np.lib.format._read_array_header(f,version)
  if dtype.hasobject:raise ValueError('Object dtype refused')
  offset=f.tell();n=int(np.prod(shape,dtype=np.int64))*dtype.itemsize
  if offset+n!=start+info.file_size:raise ValueError('Inner NPY bounds differ')
 return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape,order='F' if fortran else 'C')

class LabelMap(collections.abc.Mapping):
 def __init__(self,root,ids):self.root=Path(root);self.ids=tuple(ids)
 def __iter__(self):return iter(self.ids)
 def __len__(self):return len(self.ids)
 def __getitem__(self,obj_idx):
  if obj_idx not in self.ids:raise KeyError(obj_idx)
  p=self.root/'grasp_label'/f'{obj_idx-1:03d}_labels.npz'
  return (mapped_npz(str(p),'points'),mapped_npz(str(p),'offsets'),mapped_npz(str(p),'scores'),np.load(self.root/'tolerance'/f'{obj_idx-1:03d}_tolerance.npy',mmap_mode='r'))

class CollisionMap(collections.abc.Mapping):
 def __init__(self,root,scenes):self.root=Path(root);self.scenes=tuple(scenes)
 def __iter__(self):return iter(self.scenes)
 def __len__(self):return len(self.scenes)
 def __getitem__(self,scene):
  if scene not in self.scenes:raise KeyError(scene)
  p=self.root/'collision_label'/scene/'collision_labels.npz'
  class Members:
   def __getitem__(self,i):return mapped_npz(str(p),'arr_'+str(i))
  return Members()

class ExplicitSceneDataset(GraspNetDataset):
 def __init__(self,root,scenes,frames,augment=False,image_root=None):
  scenes=list(scenes);frames=list(frames)
  if len(scenes)!=len(set(scenes)) or not set(scenes)<=set(range(100)):raise ValueError('Scene scope invalid')
  if len(frames)!=len(set(frames)) or not set(frames)<=set(range(256)):raise ValueError('Frame scope invalid')
  self.root=str(image_root or root);self.label_root=Path(root);self.num_points=20000;self.remove_outlier=True;self.remove_invisible=True
  self.camera='realsense';self.augment=augment;self.load_label=True;self.split='explicit';self.sceneIds=[f'scene_{s:04d}' for s in scenes]
  ids=sorted(set(int(x)+1 for s in self.sceneIds for x in (self.label_root/'scenes'/s/'object_id_list.txt').read_text().split())-{19})
  self.valid_obj_idxs=ids;self.grasp_labels=LabelMap(root,ids);self.collision_labels=CollisionMap(root,self.sceneIds)
  self.colorpath=[];self.depthpath=[];self.labelpath=[];self.metapath=[];self.scenename=[];self.frameid=[]
  for s in self.sceneIds:
   for frame in frames:
    d=Path(self.root)/'scenes'/s/self.camera
    self.colorpath.append(str(d/'rgb'/f'{frame:04d}.png'));self.depthpath.append(str(d/'depth'/f'{frame:04d}.png'))
    self.labelpath.append(str(d/'label'/f'{frame:04d}.png'));self.metapath.append(str(d/'meta'/f'{frame:04d}.mat'))
    self.scenename.append(s);self.frameid.append(frame)

 def get_data_label(self,index):
  color=np.asarray(Image.open(self.colorpath[index]),dtype=np.float32)/255.;depth=np.asarray(Image.open(self.depthpath[index]));seg=np.asarray(Image.open(self.labelpath[index]));meta=loadmat(self.metapath[index]);scene=self.scenename[index]
  obj_idxs=meta['cls_indexes'].flatten().astype(np.int32);poses=meta['poses'];K=meta['intrinsic_matrix'];factor=meta['factor_depth']
  camera=CameraInfo(1280.,720.,K[0][0],K[1][1],K[0][2],K[1][2],factor)
  cloud=create_point_cloud_from_depth_image(depth,camera,organized=True);mask=depth>0
  if self.remove_outlier:
   d=Path(self.root)/'scenes'/scene/self.camera;trans=np.load(d/'cam0_wrt_table.npy')@np.load(d/'camera_poses.npy')[self.frameid[index]]
   mask &= get_workspace_mask(cloud,seg,trans=trans,organized=True,outlier=.02)
  cloud=cloud[mask];color=color[mask];seg=seg[mask]
  if len(cloud)>=self.num_points:idx=np.random.choice(len(cloud),self.num_points,replace=False)
  else:idx=np.concatenate([np.arange(len(cloud)),np.random.choice(len(cloud),self.num_points-len(cloud),replace=True)])
  cloud=cloud[idx];color=color[idx];seg=seg[idx];objectness=seg.copy();objectness[objectness>1]=1
  out=dict(object_poses_list=[],grasp_points_list=[],grasp_offsets_list=[],grasp_labels_list=[],grasp_tolerance_list=[])
  for i,obj in enumerate(obj_idxs):
   if obj not in self.valid_obj_idxs or (seg==obj).sum()<50:continue
   points,offsets,scores,tolerance=self.grasp_labels[obj];points=np.asarray(points,dtype=np.float32)
   selected=np.arange(len(points))
   if self.remove_invisible:selected=selected[remove_invisible_grasp_points(cloud[seg==obj],points,poses[:,:,i],th=.01)]
   chosen=np.random.choice(len(selected),min(max(int(len(selected)/4),300),len(selected)),replace=False);selected=selected[chosen]
   collision=self.collision_labels[scene][i][selected];ss=np.asarray(scores[selected],dtype=np.float32).copy();tt=np.asarray(tolerance[selected]).copy();ss[collision]=0;tt[collision]=0
   out['object_poses_list'].append(poses[:,:,i]);out['grasp_points_list'].append(points[selected]);out['grasp_offsets_list'].append(np.asarray(offsets[selected],dtype=np.float32));out['grasp_labels_list'].append(ss);out['grasp_tolerance_list'].append(tt)
  if not out['object_poses_list']:raise ValueError('No trainable object in fixed sampled frame')
  if self.augment:cloud,out['object_poses_list']=self.augment_data(cloud,out['object_poses_list'])
  return dict(point_clouds=cloud.astype(np.float32),cloud_colors=color.astype(np.float32),objectness_label=objectness.astype(np.int64),**out)

def to_device(batch,device):
 return {k:[[a.to(device) for a in row] for row in v] if isinstance(v,list) else v.to(device) for k,v in batch.items()}
