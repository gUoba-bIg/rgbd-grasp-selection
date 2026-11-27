"""Only missing quality outputs; frozen CPU checks happen separately."""
import os,sys,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import *
from scripts.pipeline import project_lock
from scripts import evaluate_segmentation_baseline_v16 as evaluation
from scripts import train_segmentation_baseline_v16 as trainer
from src.p5_quality_v23 import QualityNet,corrupt
from src.p6_image_geometry_v24 import quality
import numpy as np
import torch

def run():
 c=verify()
 if (BASE/'quality_complete.json').exists():print('P6 quality SKIP: complete');return
 b=Budget(c,'quality');outcome='failed'
 try:
  need(trainer.runtime_identity()==c['runtime'],'Inference runtime changed');trainer.reset_random_streams(c['seed']);q=QualityNet().cuda();q.load_state_dict(torch.load(ROOT/c['quality_checkpoint'],map_location='cpu',weights_only=False)['model']);q.eval()
  def infer(r,d):
   b.guard();need(torch.cuda.max_memory_reserved()<c['resources']['gpu_bytes'],'GPU limit');raw=evaluation.prepare_input(ROOT,c,r,'rgbd');cid=b.intent('corruption_replay',case=name(r,d));x,_,meta=corrupt(torch.from_numpy(raw),r,c['quality_screen_epoch'],d);b.returned('corruption_replay',cid,case=name(r,d));cid=b.intent('forward',case=name(r,d))
   with torch.inference_mode():h=q(x[None].cuda())
   torch.cuda.synchronize();b.returned('forward',cid,case=name(r,d));a=dict(sigma=h['log_sigma'][0,0].exp().cpu().numpy(),valid_probability=h['valid_logits'][0,0].sigmoid().cpu().numpy(),anomaly_probability=h['anomaly_logits'][0,0].sigmoid().cpu().numpy());return a,array_sha(x[3].numpy()*2.),meta
  if not (LOG/'quality_identity.json').exists():
   checked=[]
   for r in c['monitor_records'][:2]:
    for d in ['clean','joint']:
     a,z,meta=infer(r,d)
     with np.load(quality(r,d),allow_pickle=False) as old:need(all(np.array_equal(a[k],old[k]) for k in a),'Reused quality input/runtime identity differs')
     checked.append(name(r,d))
   once(LOG/'quality_identity.json',dict(passed=True,bitwise_all_outputs_equal=checked,new_model_calls=4))
  items=[]
  for r,d in case_order(c):
   p=quality(r,d);rp=p.with_suffix('.json');reused=p.is_relative_to(OLD)
   if not rp.exists():
    a,z,meta=infer(r,d);p.parent.mkdir(parents=True,exist_ok=True);need(not p.exists(),'Uncommitted quality cache needs review');np.savez_compressed(p,**a);once(rp,dict(scene=r['scene'],frame=r['frame'],condition=d,cache_sha256=sha(p),observed_depth_sha256=z,corruption=meta,model_sha256=sha(ROOT/c['quality_checkpoint'])))
   need(sha(p)==read(rp)['cache_sha256'],'Quality cache hash mismatch');items.append(dict(case=name(r,d),cache=p.relative_to(ROOT).as_posix(),sha256=sha(p),reused=reused))
   if len(items)%100==0:status('quality_export',completed=len(items),total=800);print('quality '+str(len(items))+'/800',flush=True)
  once(BASE/'quality_complete.json',dict(items=items,total=800,reused_cases=sum(x['reused'] for x in items)));outcome='passed'
 finally:b.save(outcome)
if __name__=='__main__':
 with project_lock(ROOT):run()
