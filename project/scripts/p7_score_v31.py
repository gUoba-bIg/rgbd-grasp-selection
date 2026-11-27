"""Outcome-blind fixed pools, observed features, then official geometric labels."""
import os,sys,json,functools,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p7_common_v31 import *

def detector_class():
 spec=importlib.util.spec_from_file_location('p7_pinned_detector',ROOT/'runtime/build/graspnet/utils/collision_detector.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.ModelFreeCollisionDetector

def run():
 import numpy as np,open3d as o3d
 from graspnetAPI import GraspGroup
 from graspnetAPI.utils import eval_utils as official
 from graspnetAPI.utils.config import get_config
 from graspnetAPI.utils.xmlhandler import xmlReader
 from scripts.verify_grasp_geometry_v10 import exact_indices,score_configs
 from src.risk_observations_v31 import points
 c=verify();b=Budget(c,'scoring');root=ROOT/c['data_root'];Detector=detector_class();load_sealed(BASE/'generation_complete.json')
 friction,configs=score_configs((np,o3d,GraspGroup,official,get_config,xmlReader))
 @functools.lru_cache(maxsize=40)
 def model_points(oid):return official.voxel_sample_points(np.asarray(o3d.io.read_point_cloud(str(root/f'models/{oid:03d}/nontextured.ply')).points),.008)
 @functools.lru_cache(maxsize=16)
 def dexmodel(oid):return official.load_dexnet_model(str(root/f'models/{oid:03d}/textured'))
 for n,case in enumerate(cases(c)):
  b.guard();d=folder(case);g=load_sealed(d/'generation.json');need(stat(d/'observations.npz')==g['artifact'],'Observation changed');a=np.load(d/'observations.npz',allow_pickle=False);cand=a['candidates'];poolpath=d/'pool.json'
  if poolpath.exists():pool=load_sealed(poolpath);indices=pool['indices']
  else:
   from scripts.p7_pool_v31 import commit_pool
   indices=commit_pool(d,a,case,GraspGroup)
  selected=cand[indices];fp=d/'features.npz'
  if not (d/'features.json').exists():
   cloud,_=points(a['observed_depth'],a['observed_valid'],a['K'],a['original_size']);detector=Detector(cloud,voxel_size=.005)
   if len(selected):
    co,empty,ious=detector.detect(GraspGroup(selected),approach_dist=.05,collision_thresh=.01,return_empty_grasp=True,empty_thresh=.01,return_ious=True);local=np.column_stack([co,empty]+ious).astype(np.float32)
   else:local=np.empty((0,7),np.float32)
   features=np.concatenate([a['base_features'][indices],local],axis=1);need(features.shape==(len(indices),19) and np.isfinite(features).all(),'Feature shape/finite failure');need(not fp.exists(),'Unsealed feature payload retained');np.savez_compressed(fp,features=features,indices=np.array(indices,dtype=np.int64),observed_voxel_points=detector.scene_points)
   sealed(d/'features.json',dict(artifact=stat(fp),pool=stat(poolpath),GT_accessed_for_features=False))
  need(stat(fp)==load_sealed(d/'features.json')['artifact'],'Observed features changed')
  # Only now access reference objects and poses for labels. They never enter inputs/pool.
  sd=root/f"scenes/scene_{case['scene']:04d}/realsense";parsed=[official.parse_posevector(v) for v in xmlReader(str(sd/'annotations/0064.xml')).getposevectorlist()];ids=[p[0] for p in parsed];poses=[p[1] for p in parsed];models=[model_points(i) for i in ids];transformed=[official.transform_points(p,t) for p,t in zip(models,poses)];world=np.concatenate(transformed);slots=np.concatenate([np.full(len(p),i,np.int32) for i,p in enumerate(transformed)]);assign=slots[official.compute_closest_points(selected[:,13:16],world)] if len(selected) else np.array([],int)
  table=official.transform_points(official.create_table_points(1.,1.,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.008),np.linalg.inv(np.load(sd/'cam0_wrt_table.npy')@np.load(sd/'camera_poses.npy')[64]));world=np.concatenate([world,table]);records=[]
  for index,grasp,slot in zip(indices,selected,assign):
   b.guard();p=d/'labels'/f'{index:04d}.json'
   if p.exists():r=load_sealed(p);need(r['grasp']==grasp.tolist(),'Label grasp identity differs');records.append(r);continue
   def score():
    co,em,dg=official.collision_detection([grasp[None]],[transformed[slot]],[None],[poses[slot]],world,outlier=.05,empty_thresh=10,return_dexgrasps=True);collision=bool(co[0][0]);empty=bool(em[0][0]);score=-1.
    if not collision and dg[0][0] is not None:score=float(official.get_grasp_score(dg[0][0],dexmodel(ids[slot]),friction,configs))
    return dict(collision=collision,empty=empty,score=score)
   value,cid=invoke('official_score',score,case=case,index=int(index));need(value['score'] in [-1.,.2,.4,.6,.8,1.,1.2],'Unsupported label score');r=dict(index=int(index),grasp=grasp.tolist(),object_id=int(ids[slot]),object_slot=int(slot),**value,positive=bool(0<value['score']<=.4 and not value['collision']),call_id=cid);sealed(p,r);records.append(r)
  if case['scene']==c['groups'][case['split']][0] and case['condition']=='clean' and len(selected):
   op=d/'upstream_oracle.json'
   if not op.exists():
    class Dex:
     def __getitem__(self,i):return dexmodel(ids[i])
    def oracle():
     gg,ss,cc=official.eval_grasp(GraspGroup(selected.copy()),models,Dex(),poses,get_config(),table=table,voxel_size=.008,TOP_K=50)
     return dict(groups=[x.tolist() for x in gg],scores=[x.tolist() for x in ss],collisions=[x.tolist() for x in cc])
    value,cid=invoke('upstream_oracle',oracle,case=case);sealed(op,dict(value=value,call_id=cid))
   o=load_sealed(op)['value'];checked=[]
   for gs,ss,cc in zip(o['groups'],o['scores'],o['collisions']):
    for row,s,co in zip(gs,ss,cc):
     match=[r for r in records if r['grasp']==row];need(len(match)==1 and match[0]['score']==s and match[0]['collision']==co,'P7 upstream oracle differs');checked.append(match[0]['index'])
   need(len(checked)<=20,'Oracle scope exceeded')
  result=dict(case=case,indices=indices,positive=sum(r['positive'] for r in records),negative=sum(not r['positive'] for r in records),records=records)
  if (d/'labels.json').exists():need(load_sealed(d/'labels.json')==result,'Repeated labels differ')
  else:sealed(d/'labels.json',result)
  status('scoring',conditions_completed=n+1,total_conditions=500);a.close()
 b.save('complete')
 if not (BASE/'scoring_complete.json').exists():sealed(BASE/'scoring_complete.json',dict(at=now(),conditions=500))

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
