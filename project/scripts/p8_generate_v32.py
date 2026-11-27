"""Generate only foreground-input cases; reuse sealed P7 sensor/segmentation data."""
import os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_common_v32 import *
def run():
 import numpy as np,torch
 from src.grasp_prerequisite_data_v27 import SOURCE
 from graspnet import GraspNet,pred_decode
 from src.p8_math_v32 import input_sample
 c=verify();b=Budget(c,'generation');torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(c['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
 net=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04],is_training=False).cuda().eval();net.load_state_dict(torch.load(ROOT/c['candidate_checkpoint'],map_location='cpu',weights_only=False)['model_state_dict'])
 for n,r in enumerate(cases(c)):
  d=folder(r);done=d/'generation.json'
  if done.exists():g=load(done);artifact(d/'foreground.npz');continue
  b.guard();source=oldobservations(r);a=np.load(source);cloud,pixels,nfg=input_sample(a,c['seed'],r['scene']);ids=[];identity=[]
  if len(cloud):
   inp=torch.from_numpy(cloud)[None].cuda()
   def forward():
    with torch.inference_mode():e=net({'point_clouds':inp});cand=pred_decode(e)[0].cpu().numpy().copy();sel=torch.nonzero(torch.argmax(e['objectness_score'][0].float(),0)==1).reshape(-1);ix=e['sa1_inds'][0].long()[e['sa2_inds'][0].long()][sel].cpu().numpy()
    torch.cuda.synchronize();return cand,pixels[ix]
   (cand,px),cid=invoke('grasp_forward',forward,case=r);ids.append(cid)
   if r['scene']==c['groups'][r['split']][0] and r['condition']=='clean':
    (ca,pp),cid=invoke('grasp_forward',forward,case=r,identity=True);need(np.array_equal(cand,ca) and np.array_equal(px,pp),'P8 deterministic replay differs');identity.append(cid)
   del inp
  else:cand=np.empty((0,17),np.float32);px=np.empty(0,np.int64)
  need(cand.shape==(len(px),17) and np.isfinite(cand).all(),'Invalid P8 candidates');payload(d/'foreground.npz',candidates=cand,pixels=px,sampled_pixels=pixels)
  sealed(done,dict(case=r,source=stat(source),foreground_points=nfg,forward_call_ids=ids,identity_call_ids=identity,empty_policy='zero candidates if fewer than 1024 observed foreground points',GT_used=False));a.close();status('foreground_generation',completed=n+1,total=150)
 b.save('complete');sealed(BASE/'generation_complete.json',dict(cases=150,at=now()))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
