"""Finite P5 quality stage and shared audited model execution, no scheduler."""
import os,sys,time,uuid,shutil,threading,argparse
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23r1 import *
from scripts.pipeline import project_lock
from scripts import train_segmentation_baseline_v16 as trainer
from scripts import evaluate_segmentation_baseline_v16 as evaluation
from scripts.extract_mask_scores_v17 import decode_with_scores
from src.checkpoints import CheckpointStore,atomic_json,fingerprint
from src.graspnet_frames import GraspNetFrames
from src.p5_quality_v23 import QualityNet,GatedModel,Frames,corrupt,quality_loss,CORE,GRADED,SEED,strength as original_strength
from src.p5_metrics_v23 import diagnostics,summary
import torch
import numpy as np

class Session:
    def __init__(self,c):
        self.c=c;self.uid=uuid.uuid4().hex;self.start=time.monotonic();self.ctx={};self.seq=0;self.done=threading.Event()
        e=events();self.forwards=sum(x['kind']=='forward_intent' for x in e);self.optim=sum(x['kind']=='optimizer_returned' for x in e);self.prior_counts=read(OLD_BASE/'closeout.json')['accounting']
        self.previous=usage();self.cumulative_previous=usage(prior=True);self.usage=LOG/('usage_'+self.uid+'.json')
        self.hooks=[];need(trainer.runtime_identity()==c['runtime'],'Runtime mismatch')
        threading.Thread(target=self.watchdog,daemon=True).start();self.save_usage('running')
    def save_usage(self,state,error=None):atomic_json(self.usage,dict(at=now(),seconds=time.monotonic()-self.start,state=state,error=error))
    def watchdog(self):
        if not self.done.wait(max(0,self.c['resources']['wall_seconds']-self.previous)):
            self.save_usage('resource_timeout');event('resource_timeout',session=self.uid);os._exit(124)
    def guard(self):
        r=self.c['resources'];self.save_usage('running')
        need(self.previous+time.monotonic()-self.start<r['wall_seconds'],'New-version time budget exhausted');need(self.cumulative_previous+time.monotonic()-self.start<r['cumulative_wall_seconds'],'Cumulative time budget exhausted')
        need(evaluation.peak_rss()<r['rss_bytes'] and torch.cuda.max_memory_reserved()<r['gpu_bytes'],'Memory budget exhausted')
        need(shutil.disk_usage(ROOT).free>=r['disk_reserve_bytes'],'Disk reserve exhausted')
        need(output_bytes()<r['output_bytes'],'Cumulative output budget exhausted')
    def hook(self,model,name):
        def pre(m,args):
            need(self.forwards<self.c['resources']['max_forward_intents'] and self.forwards+self.prior_counts['forward']['intents']<self.c['resources']['cumulative_forward_intents'],'Forward budget exhausted');self.forwards+=1;self.seq+=1;m._call=self.uid+'_'+str(self.seq);event('forward_intent',call_id=m._call,network=name,**self.ctx)
        def post(m,args,out):torch.cuda.synchronize();event('forward_returned',call_id=m._call,network=name,**self.ctx)
        self.hooks.extend([model.register_forward_pre_hook(pre),model.register_forward_hook(post)])
    def optimizer_intent(self):
        need(self.optim<self.c['resources']['max_optimizer_returns'] and self.optim+self.prior_counts['optimizer']['known_returns']<self.c['resources']['cumulative_optimizer_returns'],'Optimizer budget exhausted');key=uuid.uuid4().hex;event('optimizer_intent',call_id=key,**self.ctx);return key
    def optimizer_returned(self,key,loss,**x):self.optim+=1;event('optimizer_returned',call_id=key,loss=loss,**self.ctx,**x)
    def finish(self,state,error=None):
        self.done.set()
        for h in self.hooks:h.remove()
        self.save_usage(state,error)


def qcheckpoint():return OLD_RUN/'quality_final.pt'
def strength(step):return .5*original_strength(step)
