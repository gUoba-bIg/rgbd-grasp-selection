"""Frozen neural inference for train-only repair data; no neural optimization."""
import os,sys
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v25 import *
from scripts.pipeline import project_lock
from scripts import train_segmentation_baseline_v16 as trainer
from scripts import evaluate_segmentation_baseline_v16 as evaluation
from scripts.extract_mask_scores_v17 import decode_with_scores
from src.p5_quality_v23 import QualityNet
from src.segmentation import PilotInstanceNet
from src import p6_image_geometry_v24 as geo
from src.p6_error_mixture_v25 import observable_cells
import numpy as np
import torch

def run():
 c=verify()
 if (BASE/'data_complete.json').exists():print('SKIP complete repair data',flush=True);return
 b=Budget(c,'train_data');outcome='failed'
 try:
  need(trainer.runtime_identity()==c['runtime'],'Inference environment changed');trainer.reset_random_streams(c['seed'])
  quality=QualityNet().cuda();quality.load_state_dict(torch.load(ROOT/c['quality_checkpoint'],map_location='cpu',weights_only=False)['model']);quality.eval()
  seg=PilotInstanceNet('rgbd').cuda();state=torch.load(ROOT/c['fixed_segmentation_checkpoint'],map_location='cpu',weights_only=False)['model'];seg.load_state_dict({k[len('backbone.'):]:v for k,v in state.items() if k.startswith('backbone.')});seg.eval()
  def obs(r,d):
   s=invoke('corruption_replay',lambda:geo.source(c,r,d),case=name(r,d));x=evaluation.prepare_input(ROOT,c,r,'rgbd');x[3]=s['observed']/2;x[4]=s['observed_valid'];return s,torch.from_numpy(x)[None].cuda()
  def qforward(x):
   def f():
    with torch.inference_mode():q=quality(x)
    torch.cuda.synchronize();return {k:q[v][0,0].sigmoid().cpu().numpy() for k,v in [('valid_probability','valid_logits'),('anomaly_probability','anomaly_logits')]}|dict(sigma=q['log_sigma'][0,0].exp().cpu().numpy())
   return invoke('quality_forward',f)
  def sforward(x):
   def f():
    with torch.inference_mode():h=seg(x)
    torch.cuda.synchronize();return dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=h['centers'][0,0].sigmoid().cpu().numpy(),offsets=h['offsets'][0].cpu().numpy())
   h=invoke('segmentation_forward',f);return decode_with_scores(h,**c['decoder'])[0],h
  if not (LOG/'identity.json').exists():
   r=c['development_records'][0]
   for d in ['clean','joint']:
    s,x=obs(r,d);q=qforward(x)
    with np.load(geo.quality(r,d)) as old:need(all(np.array_equal(q[k],old[k]) for k in q),'Old quality identity mismatch')
    if d=='clean':
     m,h=sforward(x)
     with np.load(geo.fixed(r,d)) as old:need(np.array_equal(m,old['instances']) and all(np.array_equal(h[k],old[k]) for k in h),'Fixed model identity mismatch')
   once(LOG/'identity.json',dict(passed=True,quality=2,segmentation=1,bitwise_equal=True))
  total=0
  for split in ['fit','holdout']:
   for r in c[split+'_records']:
    for d in c['core_conditions']:
     stem=name(r,d);p=BASE/'samples'/split/(stem+'.npz')
     if p.with_suffix('.json').exists():load_sample(p);total+=1;continue
     b.guard();need(torch.cuda.max_memory_reserved()<8*1024**3,'GPU memory exceeded');s,x=obs(r,d);q=qforward(x);m,h=sforward(x) if split=='holdout' else (np.zeros((480,640),np.int32),{})
     a=geo.build(c,s,m,q);cell,noise=observable_cells(s['observed'],s['observed_valid'],q['sigma']);idx=a['flat_index']
     sample=dict(flat_index=idx,cell=cell.ravel()[idx],main=a['main'],boundary=a['boundary'],instance=a['instance'],rays=a['rays'],observed=a['depth_observed'],reference=a['depth_reference'],sigma_old=a['sigma'])
     op=BASE/'observations'/split/(stem+'.npz');save_sample(op,dict(sigma=q['sigma'],instances=m,**h),dict(scene=r['scene'],frame=r['frame'],condition=d,observed_sha256=array_sha(s['observed'])))
     save_sample(p,sample,dict(scene=r['scene'],frame=r['frame'],condition=d,split=split,noise_proxy=noise,observations=op.relative_to(ROOT).as_posix(),observations_sha256=sha(op),observed_sha256=array_sha(s['observed'])))
     total+=1
     if total%40==0:status('data',completed=total,total=400);print(f'data {total}/400',flush=True)
  once(BASE/'data_complete.json',dict(cases=total,fit_cases=320,holdout_cases=80,training_network_updates=0));outcome='passed'
 finally:b.save(outcome)
if __name__=='__main__':
 with project_lock(ROOT):run()
