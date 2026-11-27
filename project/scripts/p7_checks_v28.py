"""Before freeze: real-frame input parity, tied metrics, real candidate geometry."""
import os,sys,argparse,json,hashlib,ast
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,once,need,stat,now
LOG=ROOT/'logs/P7/preflight_v28'

def run(mode):
 import numpy as np
 if mode=='vision':
  from src.risk_observations_v28 import prepare
  from src.graspnet_frames import GraspNetFrames
  from src.p5_quality_v23 import corrupt
  from src.risk_metrics_v28 import probability_metrics,ranking
  from scripts.p7_review_v28 import observed_depth,independent_probability,close_values
  root=ROOT/'data/raw/prerequisites_v26';checked=[]
  for scene in [3,4,5]:
   record=dict(scene=scene,frame='0064',files={k:f'scenes/scene_{scene:04d}/realsense/{k}/0064'+('.mat' if k=='meta' else '.png') for k in ['rgb','depth','label','meta']});frames=GraspNetFrames(root,[record],size=(480,640),mode='rgbd',augment=False);x,_=frames.get(0)
   for condition in ['clean','noise5','missing15','shift3','joint']:
    actual,K,size,desc=prepare(root,scene,64,condition,10000);expected,_,meta=corrupt(x,record,10000,condition);need(np.array_equal(actual,expected.numpy()) and desc==meta,'Actual P5/P7 input parity failed');z,v,k,s=observed_depth(dict(data_root='data/raw/prerequisites_v26',quality_screen_epoch=10000),dict(scene=scene,condition=condition));need(np.array_equal(z,actual[3]*2) and np.array_equal(v,actual[4].astype(bool)),'Independent perturbation replay mismatch');checked.append([scene,condition])
  for y,z in [(np.array([1,0,1,0.]),np.array([0.,0.,1.,-1.])),(np.array([1.,0.]),np.array([-40.,40.]))]:close_values(probability_metrics(y,z),independent_probability(y,z))
  y=np.array([1.,0.,1.]);s=np.array([1.,1.,1.]);r=ranking(y,s,[dict(scene=3,condition='clean',start=0,end=0),dict(scene=4,condition='clean',start=0,end=3)]);need(r['p5']==.2 and r['p1']==.5,'Empty-pool denominator/tied order test failed')
  once(LOG/'vision.json',dict(at=now(),passed=True,real_input_cases=checked,tied_metric_checks=2,empty_frame_checks=1,new_neural_forwards=0,new_training_updates=0))
 elif mode=='geometry':
  sys.path.insert(0,str(ROOT/'runtime/build/graspnetAPI_eval_v7'))
  from graspnetAPI import GraspGroup
  from scripts.p7_score_v28 import detector_class
  from scripts.p7_review_v28 import independent_features
  from src.risk_observations_v28 import base_features
  # Fixed geometric probes straddle finger/interior/approach regions and include zero width.
  rng=np.random.RandomState(20261021);cloud=rng.uniform([-.13,-.08,-.02],[.06,.08,.02],size=(12000,3)).astype(np.float32);cand=np.zeros((12,17),np.float32);cand[:,0]=np.linspace(0,1,12);cand[:,1]=np.linspace(0,.1,12);cand[:,2]=.02;cand[:,3]=.04;cand[:,4:13]=np.eye(3).ravel();cand[:,16]=-1;pixels=np.arange(12);instances=np.zeros((480,640),np.int32);instances[:,320:]=1;sigma=np.full((480,640),.01,np.float32);a=dict(candidates=cand,pixels=pixels,instances=instances,sigma=sigma,quality_valid=np.ones_like(sigma),quality_anomaly=np.zeros_like(sigma),foreground_probability=np.ones_like(sigma));detector=detector_class()(cloud,.005);co,empty,ious=detector.detect(GraspGroup(cand),approach_dist=.05,collision_thresh=.01,return_empty_grasp=True,empty_thresh=.01,return_ious=True);actual=np.concatenate([base_features(cand,pixels,sigma,a['quality_valid'],a['quality_anomaly'],a['foreground_probability'],instances),np.column_stack([co,empty]+ious)],1).astype(np.float32);expected=independent_features(a,list(range(12)),detector.scene_points);need(np.allclose(expected,actual,rtol=2e-6,atol=2e-7),'Independent occupied-volume algebra differs');once(LOG/'geometry.json',dict(at=now(),passed=True,geometric_candidates=12,points=12000,maximum_difference=float(np.max(np.abs(actual-expected))),neural_forwards=0,official_quality_scores=0))
 else:
  need(read(LOG/'vision.json')['passed'] and read(LOG/'geometry.json')['passed'] and read(LOG/'real_geometry.json')['passed'],'Missing preflight')
  files=list((ROOT/'scripts').glob('p7_*_v28.py'))+list((ROOT/'src').glob('risk_*_v28.py'))
  for p in files:ast.parse(p.read_text(encoding='utf-8-sig'))
  once(LOG/'checks.json',dict(at=now(),passed=True,vision=stat(LOG/'vision.json'),geometry=stat(LOG/'geometry.json'),real_geometry=stat(LOG/'real_geometry.json'),sources={p.relative_to(ROOT).as_posix():stat(p) for p in files},not_formal_experiment=True));print('P7 preflight complete')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['vision','geometry','finish']);run(p.parse_args().mode)
