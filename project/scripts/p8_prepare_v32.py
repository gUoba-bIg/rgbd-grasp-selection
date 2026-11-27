"""Read-only preflight and immutable P8 protocol freeze."""
import os,sys,ast,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p8_common_v32 import *
def tests():
 import numpy as np
 from src.p8_math_v32 import grasp_ap,probability,ranked_candidates,input_sample
 from scripts.p8_review_v32 import independent_probability
 from scripts.p7_review_v31 import close_values,independent_features
 from src.risk_observations_v31 import base_features
 checks=[]
 for n in [0,1,14,15,16,101]:
  y=np.arange(n)%2;z=np.round(np.linspace(-6,6,n)) if n else np.array([]);close_values(probability(y,z),independent_probability(y,z));checks.append('ECE15 ties/empty '+str(n))
 for scores,collisions in [([],[]),([.2],[False]),([.4,-1,1.2],[False,True,False])]:
  r=grasp_ap(scores,collisions);expected=[[sum(0<s<=mu and not co for s,co in list(zip(scores,collisions))[:k])/k for mu in [.2,.4,.6,.8,1.,1.2]] for k in range(1,51)];need(r['prefix']==expected,'Prefix denominator test');checks.append('AP empty/short prefix')
 a=np.zeros((6,17),np.float32);a[:,13]=np.arange(6);z=np.array([1,1,-2,0,9,0.]);keep=np.array([1,1,1,0,1,1],bool);x,ix=ranked_candidates(a,keep,z);need(ix.tolist()==[4,0,1,5,2],'Deterministic tie order');need(np.array_equal(ranked_candidates(a,keep,z/1.7)[1],ix),'Positive T ordering');checks.append('ordinal tie and temperature invariance')
 parent=read(ROOT/'configs/risk_calibration_v31.json');r=dict(split='train',scene=parent['groups']['train'][0],condition='clean',frame=64);d=OLD/'cases/train'/key(r)
 with np.load(d/'observations.npz') as v:obs={k:v[k] for k in v.files}
 with np.load(d/'features.npz') as v:ix=v['indices'];expected=independent_features(obs,ix,v['observed_voxel_points']);need(np.allclose(expected,v['features'],rtol=2e-6,atol=2e-7),'Existing full feature algebra');checks.append('accepted real training feature reconstruction')
 cloud,pixels,count=input_sample(obs,parent['seed'],r['scene']);need(cloud.shape==(20000,3) and np.all(obs['instances'].ravel()[pixels]>0),'Foreground sampler mapping');need(np.array_equal(pixels,input_sample(obs,parent['seed'],r['scene'])[1]),'Sampler repeatability');checks.append('real observable foreground sample mapping')
 empty={**obs,'instances':np.zeros((480,640),np.int32)};need(len(input_sample(empty,parent['seed'],r['scene'])[0])==0,'No GT/whole fallback on empty foreground');checks.append('empty foreground policy')
 for value in [[],np.empty(0),np.array([True,False])]:need(isinstance(np.asarray(value).tolist(),list),'Oracle list serialization')
 checks.append('upstream empty-list serialization')
 files=list((ROOT/'scripts').glob('p8*_v32.py'))+[ROOT/'src/p8_math_v32.py']
 for p in files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 checks.append('all P8 source syntax');return checks
def run():
 need(not CONFIG.exists(),'Already frozen: use reentry, do not replace');LOG.mkdir(parents=True,exist_ok=True);checks=tests();parent=read(ROOT/'configs/risk_calibration_v31.json');close=load(OLD/'closeout.json');need(close['scientific_usable'],'P7 prerequisite not usable');need(read(ROOT/'logs/P7/risk_v31/cold.json')['engineering_passed'],'P7 cold missing');need(set(parent['groups'])=={'train','calibration','validation'},'Do not include reserved test groups');need([len(parent['groups'][k]) for k in ['train','calibration','validation']]==[70,15,15],'Frozen split changed')
 for p,s in read(ROOT/'logs/P7/risk_v31/cold.json')['files'].items():need(stat(ROOT/p)==s,'P7 parent changed '+p)
 protected=dict(parent['protected'])
 paths=list((ROOT/'scripts').glob('p8*_v32.py'))+[ROOT/p for p in ['src/p8_math_v32.py','docs/P8整体消融协议_v32.txt','docs/分阶段训练方案.html','docs/总进度固定工作包_监督版20261027.txt','configs/risk_calibration_v31.json','logs/P7/risk_v31/cold.json','results/P7/risk_v31/closeout.json','scripts/p7_review_v31.py','scripts/p7_fit_v31.py','scripts/p6_common_v24.py','scripts/p3_train_v27.py','scripts/pipeline.py','scripts/accept_mask_ap_v17.py','scripts/evaluate_mask_ap_v17.py','scripts/p4_complete_evaluation_v22r2.py','scripts/extract_mask_scores_v17.py','scripts/verify_grasp_geometry_v10.py','src/grasp_observed_filter_v29.py','src/risk_observations_v31.py']]
 for p in paths:protected[p.relative_to(ROOT).as_posix()]=stat(p)
 # Snapshot Python helpers reached through inherited imports and pinned COCO code.
 for base in [ROOT/'scripts',ROOT/'src']:
  for p in base.rglob('*.py'):protected[p.relative_to(ROOT).as_posix()]=stat(p)
 for p in (ROOT/'runtime/envs/p4_eval_v22/Lib/site-packages/pycocotools').rglob('*'):
  if p.is_file() and p.suffix in ['.py','.pyd']:protected[p.relative_to(ROOT).as_posix()]=stat(p)
 from graspnetAPI.utils.xmlhandler import xmlReader
 from graspnetAPI.utils.eval_utils import parse_posevector
 references={};objects=set()
 for scene in parent['groups']['calibration']+parent['groups']['validation']:
  sd=ROOT/parent['data_root']/f'scenes/scene_{scene:04d}/realsense'
  for tail in ['annotations/0064.xml','cam0_wrt_table.npy','camera_poses.npy']:
   p=sd/tail;references[p.relative_to(ROOT).as_posix()]=stat(p)
  objects.update(parse_posevector(v)[0] for v in xmlReader(str(sd/'annotations/0064.xml')).getposevectorlist())
 for oid in sorted(objects):
  for tail in ['nontextured.ply','textured.obj','textured.sdf']:
   p=ROOT/parent['data_root']/f'models/{oid:03d}'/tail;references[p.relative_to(ROOT).as_posix()]=stat(p)
 mask_inputs={}
 for scene in parent['groups']['validation']:
  p=ROOT/parent['data_root']/f'scenes/scene_{scene:04d}/realsense/label/0064.png';mask_inputs[p.relative_to(ROOT).as_posix()]=stat(p)
 c=dict(version=32,at=now(),seed=parent['seed'],data_root=parent['data_root'],groups=parent['groups'],conditions=parent['conditions'],decoder=parent['decoder'],candidate_checkpoint=parent['candidate_checkpoint'],groups_definition={'D0':'whole_raw','D1':'foreground_raw','D2':'foreground_simple_sigma','D3':'foreground_risk_no_quality','D4':'foreground_risk_full','D5':'foreground_risk_full_calibrated','D6':'whole_risk_full','D7':'foreground_risk_no_instance'},scope='bounded exposed-development single-seed 15-scene x five-condition x one-view ablation; not independent official90 test or robot success',phase_seconds=dict(generation=1800,fitting=1200,features=10800,scoring=21600,mask=1800,mask_review=1800,metrics=900,review=10800,cold=1800,finish=600),wall_seconds=50400,limits=parent['limits'],call_limits=dict(grasp_forward=156,risk_update=1800,risk_forward=64,risk_audit_forward=4,official_score=31000,upstream_oracle=6,mask_AP=78,temperature_objective=100,temperature_audit=1),mask_inputs=mask_inputs,protected=protected)
 c['reference_inputs']=references;c['call_limits'].update(risk_numpy_forward=530,risk_numpy_audit=540)
 sealed(LOG/'preflight.json',dict(at=now(),passed=True,checks=checks,P7_parent_all_files_unchanged=True,new_training_updates=0,new_grasp_forwards=0,new_official_scores=0,pre_freeze_revision='Explicit NumPy forward accounting and evaluator-reference input binding; prior draft retained in pre_freeze_draft_1'));once(CONFIG,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONFIG)));verify();status('frozen_not_started',preflight_checks=len(checks));print('P8 frozen; all preflight checks passed',len(checks),flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
