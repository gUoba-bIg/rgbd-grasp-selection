"""Frozen upstream networks applied once to each held-out observation."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p10_common_v36 import *
def run():
 environment(gpu=True)
 import numpy as np,torch
 from src.grasp_prerequisite_data_v27 import SOURCE
 from graspnet import GraspNet,pred_decode
 from src.p5_quality_v23 import QualityNet
 from src.segmentation import PilotInstanceNet
 from scripts.extract_mask_scores_v17 import decode_with_scores
 from src.risk_observations_v31 import prepare,points,base_features
 c=verify();b=Budget(c,'generation');load(BASE/'assets.json');torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(c['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
 grasp=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04],is_training=False).cuda().eval();grasp.load_state_dict(torch.load(ROOT/c['candidate_checkpoint'],map_location='cpu',weights_only=False)['model_state_dict'])
 quality=QualityNet().cuda().eval();quality.load_state_dict(torch.load(ROOT/c['quality_checkpoint'],map_location='cpu',weights_only=False)['model'])
 seg=PilotInstanceNet('rgbd').cuda().eval();state=torch.load(ROOT/c['segmentation_checkpoint'],map_location='cpu',weights_only=False)['model'];seg.load_state_dict({k[9:]:v for k,v in state.items() if k.startswith('backbone.')});del state
 allcases=cases(c)
 for n,r in enumerate(allcases,1):
  d=folder(r);done=d/'generation.json'
  if done.exists():g=load(done);need(stat(d/'observations.npz')==g['artifact'],'Generated observations changed');continue
  b.guard();verify_case_inputs(c,r);(x,K,size,corruption),cc=invoke('corruption',lambda:prepare(DATA,r['scene'],64,r['condition'],c['quality_screen_epoch']),case=r);cloud,pix=points(x[3]*2,x[4].astype(bool),K,size);rng=np.random.RandomState(c['seed']+r['scene']*1000+64)
  sampled=rng.choice(len(cloud),20000,replace=len(cloud)<20000) if len(cloud) else np.empty(0,np.int64);image=torch.from_numpy(x)[None].cuda();inp=torch.from_numpy(cloud[sampled])[None].cuda();ids=[];identity=[]
  def gf():
   with torch.inference_mode():e=grasp({'point_clouds':inp});a=pred_decode(e)[0].cpu().numpy().copy();sel=torch.nonzero(torch.argmax(e['objectness_score'][0].float(),0)==1).reshape(-1);ix=e['sa1_inds'][0].long()[e['sa2_inds'][0].long()][sel].cpu().numpy()
   torch.cuda.synchronize();return a,pix[sampled[ix]]
  if len(cloud):(cand,pixels),cid=invoke('grasp_forward',gf,case=r);ids.append(cid)
  else:cand=np.empty((0,17),np.float32);pixels=np.empty(0,np.int64)
  def qf():
   with torch.inference_mode():q=quality(image)
   torch.cuda.synchronize();return [q['log_sigma'][0,0].exp().cpu().numpy(),q['valid_logits'][0,0].sigmoid().cpu().numpy(),q['anomaly_logits'][0,0].sigmoid().cpu().numpy()]
  (sigma,qvalid,anomaly),cid=invoke('quality_forward',qf,case=r);ids.append(cid)
  def sf():
   with torch.inference_mode():h=seg(image)
   torch.cuda.synchronize();return dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=h['centers'][0,0].sigmoid().cpu().numpy(),offsets=h['offsets'][0].cpu().numpy())
  heads,cid=invoke('segmentation_forward',sf,case=r);ids.append(cid);instances=decode_with_scores(heads,**c['decoder'])[0];features=base_features(cand,pixels,sigma,qvalid,anomaly,heads['foreground_probability'],instances)
  if r['condition']=='clean' and r['scene'] in [100,160]:
   if len(cloud):
    (a,p),cid=invoke('grasp_forward',gf,case=r,identity=True);need(np.array_equal(a,cand) and np.array_equal(p,pixels),'Candidate replay differs');identity.append(cid)
   a,cid=invoke('quality_forward',qf,case=r,identity=True);need(all(np.array_equal(v,w) for v,w in zip(a,[sigma,qvalid,anomaly])),'Quality replay differs');identity.append(cid)
   a,cid=invoke('segmentation_forward',sf,case=r,identity=True);need(all(np.array_equal(a[k],heads[k]) for k in a),'Segmentation replay differs');identity.append(cid)
  need(features.shape==(len(cand),12) and np.isfinite(features).all(),'Invalid base features')
  payload(d/'observations.npz',candidates=cand,pixels=pixels,base_features=features,observed_depth=x[3]*2,observed_valid=x[4].astype(bool),K=K,original_size=np.array(size),sigma=sigma,quality_valid=qvalid,quality_anomaly=anomaly,instances=instances,**heads)
  sealed(done,dict(at=now(),case=r,artifact=stat(d/'observations.npz'),corruption=corruption,corruption_call_id=cc,forward_call_ids=ids,identity_call_ids=identity,point_count=len(cloud),sample_with_replacement=bool(0<len(cloud)<20000),GT_used=False));status('generation',conditions_completed=n,total_conditions=len(allcases));del image,inp
 b.save('complete');sealed(BASE/'generation_complete.json',dict(at=now(),conditions=len(allcases),formal_test_frames=90,conditions_per_frame=12))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
