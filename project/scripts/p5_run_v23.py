"""Finite P5 quality stage and shared audited model execution, no scheduler."""
import os,sys,time,uuid,shutil,threading,argparse
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23 import *
from scripts.pipeline import project_lock
from scripts import train_segmentation_baseline_v16 as trainer
from scripts import evaluate_segmentation_baseline_v16 as evaluation
from scripts.extract_mask_scores_v17 import decode_with_scores
from src.checkpoints import CheckpointStore,atomic_json,fingerprint
from src.graspnet_frames import GraspNetFrames
from src.p5_quality_v23 import QualityNet,GatedModel,Frames,corrupt,quality_loss,CORE,GRADED,SEED,strength
from src.p5_metrics_v23 import diagnostics,summary
import torch
import numpy as np

def prepare():
    need(not CONFIG.exists(),'P5 is already frozen')
    old=read(ROOT/'configs/p4_full_baseline_v22r2.json')
    need(read(LOG/'component_tests.json')['passed'],'Component tests failed')
    need(read(ROOT/'logs/P4/full_baseline_v22r2/final_readonly/result.json')['status']=='passed','P4 final gate missing')
    c={k:old[k] for k in ['input_size','data_root','train_records','development_records','monitor_records','split_path','data_fingerprint','decoder','runtime']}
    c.update(version=23,created_at=now(),seed=SEED,purpose='P5 segmentation quality and gating development, not P8 grasp efficacy',parent='experiments/p4_full_baseline_v22r2/selected/epoch_33_rgbd.pt',quality_updates=240,quality_lr=.001,quality_weight_decay=.0001,quality_amp=False,quality_selection='final update240 only',quality_screen_epoch=10000,core_conditions=list(CORE),graded_conditions=list(GRADED),quality_gates=dict(brier_max=.05,spearman_min=.2,nll_improvement_min=.05,anomaly_ap_min=.02,anomaly_vs_prevalence=1.2),segmentation=dict(max_epochs=10,updates_per_epoch=80,learning_rate=.00001,weight_decay=.0001,amp_initial_scale=128,amp_max_overflow_retries=4,gradient_clip=1.,first_gate_epoch=5,clean_f1_min=.5,stability_max=.05,selection='highest paired clean/joint scene-macro F1, earliest tie',gate_warmup_updates=80,gate_ramp_updates=160,gate_strength_max=.5),adoption=dict(degraded_mask_ap_gain_min=.01,clean_mask_ap_loss_max=.01,improved_scenes_min=6,joint_spatial_ablation_gain='strictly positive against unit and spatial mean'),resources=dict(max_optimizer_returns=2200,max_forward_intents=24000,wall_seconds=10800,output_bytes=16*1024**3,gpu_bytes=14*1024**3,rss_bytes=6*1024**3,disk_reserve_bytes=60*1024**3,max_adjustments=2,max_official_ap_calls=160),protected={})
    paths=['scripts/p5_run_v23.py','scripts/p5_common_v23.py','scripts/test_p5_v23.py','src/p5_quality_v23.py','src/p5_metrics_v23.py','docs/P5整体开发协议_v23.txt','logs/P5/quality_gate_v23/component_tests.json','configs/p4_full_baseline_v22r2.json','results/P4/full_baseline_v22r2/closeout.json','logs/P4/full_baseline_v22r2/final_readonly/result.json',c['parent'],c['split_path']]
    paths += ['scripts/train_segmentation_baseline_v16.py','scripts/evaluate_segmentation_baseline_v16.py','scripts/extract_mask_scores_v17.py','scripts/pipeline.py','src/checkpoints.py','src/graspnet_frames.py','src/segmentation.py','src/segmentation_v2.py']
    for r in c['monitor_records'][:2]:paths.append(f"results/P4/full_baseline_v22r2/exports/rgbd/scene_{r['scene']:04d}_{r['frame']}.npz")
    for r in c['train_records']+c['development_records']:
        for k,p in r['files'].items():
            p=str(Path(c['data_root'])/p).replace('\\','/');s=stat(ROOT/p);need(s['sha256']==r['sha256'][k],'Raw data mismatch '+p);c['protected'][p]=s
    for p in paths:c['protected'][p]=stat(ROOT/p)
    need(len(c['train_records'])==320 and len(c['development_records'])==160 and len(c['monitor_records'])==20,'Unexpected data scope')
    once(CONFIG,c);once(LOG/'freeze.json',dict(at=now(),protocol_sha256=sha(CONFIG),protected_files=len(c['protected']),research_started=False))
    status('frozen_quality_not_started',config=str(CONFIG.relative_to(ROOT)),automatic_scheduler=False)
    print('P5 protocol frozen',flush=True)

class Session:
    def __init__(self,c):
        self.c=c;self.uid=uuid.uuid4().hex;self.start=time.monotonic();self.ctx={};self.seq=0;self.done=threading.Event()
        e=events();self.forwards=sum(x['kind']=='forward_intent' for x in e);self.optim=sum(x['kind']=='optimizer_returned' for x in e)
        self.previous=sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'));self.usage=LOG/('usage_'+self.uid+'.json')
        self.hooks=[];need(trainer.runtime_identity()==c['runtime'],'Runtime mismatch')
        threading.Thread(target=self.watchdog,daemon=True).start();self.save_usage('running')
    def save_usage(self,state,error=None):atomic_json(self.usage,dict(at=now(),seconds=time.monotonic()-self.start,state=state,error=error))
    def watchdog(self):
        if not self.done.wait(max(0,self.c['resources']['wall_seconds']-self.previous)):
            self.save_usage('resource_timeout');event('resource_timeout',session=self.uid);os._exit(124)
    def guard(self):
        r=self.c['resources'];self.save_usage('running')
        need(self.previous+time.monotonic()-self.start<r['wall_seconds'],'Time budget exhausted')
        need(evaluation.peak_rss()<r['rss_bytes'] and torch.cuda.max_memory_reserved()<r['gpu_bytes'],'Memory budget exhausted')
        need(shutil.disk_usage(ROOT).free>=r['disk_reserve_bytes'],'Disk reserve exhausted')
        need(sum(p.stat().st_size for d in [RUN,BASE,LOG] for p in d.rglob('*') if p.is_file())<r['output_bytes'],'Output budget exhausted')
    def hook(self,model,name):
        def pre(m,args):
            need(self.forwards<self.c['resources']['max_forward_intents'],'Forward budget exhausted');self.forwards+=1;self.seq+=1;m._call=self.uid+'_'+str(self.seq);event('forward_intent',call_id=m._call,network=name,**self.ctx)
        def post(m,args,out):torch.cuda.synchronize();event('forward_returned',call_id=m._call,network=name,**self.ctx)
        self.hooks.extend([model.register_forward_pre_hook(pre),model.register_forward_hook(post)])
    def optimizer_intent(self):
        need(self.optim<self.c['resources']['max_optimizer_returns'],'Optimizer budget exhausted');key=uuid.uuid4().hex;event('optimizer_intent',call_id=key,**self.ctx);return key
    def optimizer_returned(self,key,loss,**x):self.optim+=1;event('optimizer_returned',call_id=key,loss=loss,**self.ctx,**x)
    def finish(self,state,error=None):
        self.done.set()
        for h in self.hooks:h.remove()
        self.save_usage(state,error)

def qnetwork():trainer.reset_random_streams(SEED);return QualityNet().cuda()
def qcheckpoint():return RUN/'quality_final.pt'
def identity(s):
    if (BASE/'parent_identity.json').exists():need(read(BASE/'parent_identity.json')['passed'],'Parent identity failed');return
    c=s.c;m=GatedModel(qnetwork(),'fixed').cuda();m.backbone.load_state_dict(torch.load(ROOT/c['parent'],map_location='cpu',weights_only=False)['model']);m.eval();s.hook(m.quality,'quality');s.hook(m.backbone,'segmentation');rows=[]
    for r in c['monitor_records'][:2]:
        s.ctx=dict(phase='parent_identity',scene=r['scene'],frame=r['frame']);x=evaluation.prepare_input(ROOT,c,r,'rgbd')
        with torch.inference_mode():h=m(torch.from_numpy(x)[None].cuda())
        old=np.load(ROOT/f"results/P4/full_baseline_v22r2/exports/rgbd/scene_{r['scene']:04d}_{r['frame']}.npz")
        heads=dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=h['centers'][0,0].sigmoid().cpu().numpy(),offsets=h['offsets'][0].cpu().numpy());equal=all(np.array_equal(old[k],v) for k,v in heads.items());need(equal,'Unit gate differs from accepted P4 parent');rows.append(dict(scene=r['scene'],frame=r['frame'],bitwise_equal=True));old.close()
    once(BASE/'parent_identity.json',dict(passed=True,rows=rows,parent_sha256=sha(ROOT/c['parent']),no_new_training=True));del m;torch.cuda.empty_cache()
def quality_train(s):
    c=s.c
    if qcheckpoint().exists():need(sha(qcheckpoint())==read(RUN/'quality_final.json')['sha256'],'Quality weights changed');return
    q=qnetwork();s.hook(q,'quality');opt=torch.optim.AdamW(q.parameters(),lr=c['quality_lr'],weight_decay=c['quality_weight_decay']);store=CheckpointStore(RUN/'quality_checkpoints',fingerprint(dict(config=sha(CONFIG),phase='quality')),c['data_fingerprint'],keep_last=2)
    ds=Frames(ROOT/c['data_root'],c['train_records'],size=c['input_size'],seed=SEED,augment=True)
    cursor=dict(epoch=0,next_batch=0,global_step=0)
    if (RUN/'quality_checkpoints/latest.json').exists():cursor=store.load_latest(q,opt)['cursor']
    epoch,pos,step=cursor['epoch'],cursor['next_batch'],cursor['global_step']
    while step<c['quality_updates']:
        order=ds.epoch_order(epoch);selected=order[pos:pos+4];s.ctx=dict(phase='quality_training',epoch=epoch+1,update=step+1);s.guard();opt.zero_grad(set_to_none=True);total=0.;components={}
        for i in selected:
            x,t=ds.get(i,epoch);t={k:v[None].cuda() for k,v in t.items() if k.startswith('q_')};loss,parts=quality_loss(q(x[None].cuda()),t);need(bool(torch.isfinite(loss)),'Nonfinite quality loss');(loss/len(selected)).backward();total+=float(loss.detach())/len(selected)
            for k,v in parts.items():components[k]=components.get(k,0.)+float(v)/len(selected)
        norm=torch.nn.utils.clip_grad_norm_(q.parameters(),1.);need(bool(torch.isfinite(norm)),'Nonfinite quality gradients');key=s.optimizer_intent();opt.step();torch.cuda.synchronize();need(all(bool(torch.isfinite(p).all()) for p in q.parameters()),'Nonfinite quality parameters');s.optimizer_returned(key,total,components=components,gradient_norm=float(norm))
        step+=1;pos+=len(selected)
        if pos==len(ds):epoch+=1;pos=0
        if step%20==0 or step==c['quality_updates']:
            store.save(q,opt,epoch,pos,step,True);status('quality_training',updates=step,budget=c['quality_updates'],forward_intents=s.forwards);print(json.dumps(dict(phase='quality_training',step=step,loss=total,components=components)),flush=True)
    torch.save(dict(model=q.state_dict(),updates=step,protocol_sha256=sha(CONFIG)),qcheckpoint());once(RUN/'quality_final.json',dict(sha256=sha(qcheckpoint()),updates=step));del q,opt;torch.cuda.empty_cache()

def quality_screen(s):
    c=s.c;out=BASE/'quality_screen';out.mkdir(parents=True,exist_ok=True);q=qnetwork();q.load_state_dict(torch.load(qcheckpoint(),map_location='cpu',weights_only=False)['model']);q.eval();s.hook(q,'quality');rows=[]
    for condition in CORE:
        ds=Frames(ROOT/c['data_root'],c['monitor_records'],size=c['input_size'],seed=SEED,augment=False,condition=condition)
        for i,r in enumerate(c['monitor_records']):
            stem=f"{condition}_{r['scene']:04d}_{r['frame']}";p=out/(stem+'.json');cache=out/(stem+'.npz')
            if p.exists():row=read(p);need(sha(cache)==row['cache_sha256'],'Screen cache changed');rows.append(row);continue
            s.ctx=dict(phase='quality_screen',condition=condition,scene=r['scene'],frame=r['frame']);s.guard();x,t=ds.get(i,c['quality_screen_epoch'])
            with torch.inference_mode():heads=q(x[None].cuda())
            vp=heads['valid_logits'][0,0].sigmoid().cpu().numpy();sg=heads['log_sigma'][0,0].exp().cpu().numpy();ap=heads['anomaly_logits'][0,0].sigmoid().cpu().numpy()
            targets={k:v.numpy() for k,v in t.items() if k.startswith('q_')};d=diagnostics(vp,sg,ap,targets)
            np.savez_compressed(cache,valid_probability=vp,sigma=sg,anomaly_probability=ap,**targets)
            row=dict(scene=r['scene'],frame=r['frame'],condition=condition,cache_sha256=sha(cache),diagnostics=d);once(p,row);rows.append(row)
    report=summary(rows);once(BASE/'quality_screen.json',report);status('quality_screen_complete',quality_usable=report['quality_usable'],gates=report['gates']);print(json.dumps(report['scene_macro']),flush=True)
    del q;torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','quality','verify']);args=ap.parse_args()
    with project_lock(ROOT):
        if args.mode=='prepare':prepare();return
        c=verify()
        if args.mode=='verify':print('Frozen P5 sources, inputs and P4 evidence unchanged',flush=True);return
        s=Session(c)
        try:identity(s);quality_train(s);quality_screen(s);verify();s.finish('complete')
        except BaseException as e:s.finish('failed',repr(e));status('quality_stage_failed',error=repr(e));raise
if __name__=='__main__':main()
