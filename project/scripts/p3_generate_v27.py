"""Generate fixed clean candidates from the explicit-scene model; no GT masks."""
import os,sys,time,json,uuid,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p3_train_v27 import event,CONFIG,LOG,OUT,sealed,load_sealed,resource_guard

def run(stage):
 import numpy as np,torch
 from scipy.io import loadmat
 from PIL import Image
 from src.grasp_prerequisite_data_v27 import SOURCE
 from graspnet import GraspNet,pred_decode
 from scripts.generate_multiview_grasps_v14r1 import reconstruct_points,sample_full_cloud,validate_candidates
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'Protocol changed');epoch=3 if stage=='early' else 18
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Frozen source changed '+p)
 cp=OUT/f'epoch_{epoch:02d}_final.pt';cpstat=stat(cp);need(cpstat==read(LOG/f'epoch_{epoch:02d}.json')['checkpoint'],'Model checkpoint changed')
 torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04],is_training=False).cuda().eval();saved=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict'],strict=True);del saved
 for item in c[stage+'_evaluation']:
  scene,frame=item['scene'],item['frame'];folder=ROOT/f'results/P2P3/candidate_v27/{stage}/scene_{scene:04d}_{frame:04d}';receipt=folder/'generation.json'
  if receipt.exists():
   r=load_sealed(receipt);need(stat(folder/'candidates.npy')==r['candidates'],'Generated cache changed');continue
  prior=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
  need(sum(r['kind']=='generation_intent' for r in prior)<40,'Candidate generation actual call limit reached')
  resource_guard(c,True);folder.mkdir(parents=True,exist_ok=True);d=ROOT/c['data_root']/'scenes'/f'scene_{scene:04d}'/'realsense'
  need(not any((folder/n).exists() for n in ['candidates.npy','candidate_pixels.npy','sampled_points.npy']),'Unsealed generation output preserved; recovery audit required')
  dp=d/'depth'/f'{frame:04d}.png';mp=d/'meta'/f'{frame:04d}.mat';meta=loadmat(mp,variable_names=['intrinsic_matrix','factor_depth']);depth=np.asarray(Image.open(dp));points=reconstruct_points(depth,meta['intrinsic_matrix'],float(meta['factor_depth'].item()));arrays,sampling=sample_full_cloud(points['valid_points'],points['valid_pixel_indices'],seed=c['seed']+scene*1000+frame)
  need(sampling['input_status']=='ready','Insufficient observed input');tensor=torch.from_numpy(arrays['sampled_points']).unsqueeze(0).cuda();cid=uuid.uuid4().hex;event('generation_intent',call_id=cid,stage=stage,scene=scene,frame=frame);start=time.monotonic()
  with torch.inference_mode():
   ep=model({'point_clouds':tensor});candidates=pred_decode(ep)[0].cpu().numpy().copy();selected=torch.nonzero(torch.argmax(ep['objectness_score'][0].float(),0)==1).reshape(-1)
   composed=ep['sa1_inds'][0].long()[ep['sa2_inds'][0].long()][selected].cpu().numpy();pixels=arrays['sampled_pixel_indices'][composed]
  torch.cuda.synchronize();seconds=time.monotonic()-start;event('generation_returned',call_id=cid,candidates=len(candidates),seconds=seconds);validation=validate_candidates(candidates)
  np.save(folder/'candidates.npy',candidates,allow_pickle=False);np.save(folder/'candidate_pixels.npy',pixels,allow_pickle=False);np.save(folder/'sampled_points.npy',arrays['sampled_points'],allow_pickle=False)
  sealed(receipt,dict(at=now(),call_id=cid,scene=scene,frame=frame,candidates=stat(folder/'candidates.npy'),pixels=stat(folder/'candidate_pixels.npy'),sampled_points=stat(folder/'sampled_points.npy'),checkpoint=cpstat,inputs={p.relative_to(ROOT).as_posix():stat(p) for p in [dp,mp]},GT_used_for_generation=False,seed=c['seed']+scene*1000+frame,sampling=sampling,validation=validation,seconds=seconds))
  del ep,tensor
 print(json.dumps(dict(stage=stage,generated_frames=len(c[stage+'_evaluation']))),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['early','final'],required=True);args=parser.parse_args()
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run(args.stage)
