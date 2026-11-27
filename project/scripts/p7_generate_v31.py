"""Freeze all 500 observed-condition candidate pools before outcome labels."""
import os,sys,uuid,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p7_common_v31 import *

def run():
 import numpy as np,torch
 from src.grasp_prerequisite_data_v27 import SOURCE
 from graspnet import GraspNet,pred_decode
 from src.p5_quality_v23 import QualityNet
 from src.segmentation import PilotInstanceNet
 from scripts.extract_mask_scores_v17 import decode_with_scores
 from src.risk_observations_v31 import prepare,points,base_features
 c=verify();b=Budget(c,'generation');torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(c['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
 grasp=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04],is_training=False).cuda().eval();grasp.load_state_dict(torch.load(ROOT/c['candidate_checkpoint'],map_location='cpu',weights_only=False)['model_state_dict'])
 quality=QualityNet().cuda().eval();quality.load_state_dict(torch.load(ROOT/c['quality_checkpoint'],map_location='cpu',weights_only=False)['model'])
 seg=PilotInstanceNet('rgbd').cuda().eval();state=torch.load(ROOT/c['segmentation_checkpoint'],map_location='cpu',weights_only=False)['model'];seg.load_state_dict({k[9:]:v for k,v in state.items() if k.startswith('backbone.')});del state
 allcases=cases(c)
 for n,case in enumerate(allcases):
  d=folder(case);done=d/'generation.json'
  if done.exists():r=load_sealed(done);need(stat(d/'observations.npz')==r['artifact'],'P7 observations changed');continue
  b.guard();verify_case_inputs(c,case);d.mkdir(parents=True,exist_ok=True);(x,K,size,corruption),corrupt_call=invoke('corruption',lambda:prepare(ROOT/c['data_root'],case['scene'],case['frame'],case['condition'],c['quality_screen_epoch']),case=case);cloud,source_pixels=points(x[3]*2,x[4].astype(bool),K,size);need(len(cloud)>=20000,'Insufficient observed candidate input')
  rng=np.random.RandomState(c['seed']+case['scene']*1000+case['frame']);sampled=rng.choice(len(cloud),20000,replace=False);input_tensor=torch.from_numpy(cloud[sampled])[None].cuda();image=torch.from_numpy(x)[None].cuda()
  def gf():
   with torch.inference_mode():e=grasp({'point_clouds':input_tensor});a=pred_decode(e)[0].cpu().numpy().copy();sel=torch.nonzero(torch.argmax(e['objectness_score'][0].float(),0)==1).reshape(-1);indices=e['sa1_inds'][0].long()[e['sa2_inds'][0].long()][sel].cpu().numpy()
   torch.cuda.synchronize();return a,source_pixels[sampled[indices]]
  (cand,pixels),gcall=invoke('grasp_forward',gf,case=case)
  def qf():
   with torch.inference_mode():q=quality(image)
   torch.cuda.synchronize();return [q['log_sigma'][0,0].exp().cpu().numpy(),q['valid_logits'][0,0].sigmoid().cpu().numpy(),q['anomaly_logits'][0,0].sigmoid().cpu().numpy()]
  (sigma,qvalid,anomaly),qcall=invoke('quality_forward',qf,case=case)
  def sf():
   with torch.inference_mode():h=seg(image)
   torch.cuda.synchronize();return dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=h['centers'][0,0].sigmoid().cpu().numpy(),offsets=h['offsets'][0].cpu().numpy())
  heads,scall=invoke('segmentation_forward',sf,case=case);instances=decode_with_scores(heads,**c['decoder'])[0];feat=base_features(cand,pixels,sigma,qvalid,anomaly,heads['foreground_probability'],instances)
  identity=[]
  if case['condition']=='clean' and case['split'] in ['train','calibration'] and case['scene']==c['groups'][case['split']][0]:
   (again,again_pixels),cid=invoke('grasp_forward',gf,case=case,identity=True);need(np.array_equal(cand,again) and np.array_equal(pixels,again_pixels),'Candidate identity replay differs');identity.append(cid)
   again,cid=invoke('quality_forward',qf,case=case,identity=True);need(all(np.array_equal(v,w) for v,w in zip([sigma,qvalid,anomaly],again)),'Quality identity replay differs');identity.append(cid)
   again,cid=invoke('segmentation_forward',sf,case=case,identity=True);need(all(np.array_equal(heads[k],again[k]) for k in heads),'Segmentation identity replay differs');identity.append(cid)
  need(feat.shape==(len(cand),12) and np.isfinite(feat).all(),'Invalid observed base features');path=d/'observations.npz';need(not path.exists(),'Unreceipted observation payload; preserve failure')
  np.savez_compressed(path,candidates=cand,pixels=pixels,base_features=feat,observed_depth=x[3]*2,observed_valid=x[4].astype(bool),K=K,original_size=np.array(size),sigma=sigma,quality_valid=qvalid,quality_anomaly=anomaly,instances=instances,**heads)
  source=ROOT/c['data_root']/f"scenes/scene_{case['scene']:04d}/realsense";inputs={p.relative_to(ROOT).as_posix():stat(p) for p in [source/'rgb/0064.png',source/'depth/0064.png',source/'meta/0064.mat']}
  sealed(done,dict(at=now(),case=case,artifact=stat(path),inputs=inputs,corruption=corruption,corruption_call_id=corrupt_call,forward_call_ids=[gcall,qcall,scall],identity_call_ids=identity,reference_geometry_used_for_features=False,P6_probability_used=False));status('generation',conditions_completed=n+1,total_conditions=500)
  del image,input_tensor
 b.save('complete');sealed(BASE/'generation_complete.json',dict(at=now(),conditions=len(allcases)))

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
