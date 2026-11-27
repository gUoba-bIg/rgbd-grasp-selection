"""P4 full-resolution development baselines; historical experiments are read-only."""
import os,sys,json,time,uuid,shutil,threading,argparse
from pathlib import Path
from datetime import datetime
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_lock
from scripts import train_segmentation_baseline_v16 as trainer
from scripts import evaluate_segmentation_baseline_v16 as evaluation
from scripts.extract_mask_scores_v17 import decode_with_scores
from src.checkpoints import CheckpointStore,atomic_json,file_sha256,fingerprint
from src.graspnet_frames import GraspNetFrames as OriginalFrames, instance_targets
from src.segmentation import PilotInstanceNet,decode_instances
import numpy as np
import torch
class GraspNetFrames(OriginalFrames):
    def get(self,index,epoch=0):
        inputs,targets=super().get(index,epoch)
        targets['centers']=instance_targets(targets['labels'].numpy(),sigma=6.)['centers']
        return inputs,targets
CONFIG=ROOT/'configs/p4_full_baseline_v22r2.json'
OUT=ROOT/'results/P4/full_baseline_v22r2'
RUN=ROOT/'experiments/p4_full_baseline_v22r2'
LOG=ROOT/'logs/P4/full_baseline_v22r2'
def need(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def once(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():need(read(p)==x,'Immutable record differs: '+str(p));return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def now():return datetime.now().astimezone().isoformat()
def state(p):
    s=p.stat();return dict(sha256=file_sha256(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def prepare():
    need(not CONFIG.exists() and not RUN.exists(),'Revision2 already exists')
    prior=ROOT/'results/P4/full_baseline_v22r1';result=read(prior/'training_and_export.json');sel=read(prior/'selection.json')
    need(result['scientific_gates']==dict(fit=True,rgbd_macro_f1=True,rgbd_foreground_iou=True,monitor_stability=False,noncollapsed_all_scenes=True),'Repair trigger differs')
    need(sel['last_epoch']==30 and sel['no_improvement_epochs']==8,'Unexpected prior stopping point')
    old=read(ROOT/'configs/p4_full_baseline_v22r1.json');c=dict(old);c['protected']=dict(old['protected']);c['created_at']=now();c['revision']='v22r2'
    for folder in ['logs/P4/full_baseline_v22r1','results/P4/full_baseline_v22r1','experiments/p4_full_baseline_v22r1']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():c['protected'][p.relative_to(ROOT).as_posix()]=state(p)
    for p in ['scripts/p4_full_baseline_v22r2.py','scripts/p4_full_baseline_v22r1.py','configs/p4_full_baseline_v22r1.json','configs/p4_complete_evaluation_v22r1.json','configs/p4_representative_link_v22r1.json','configs/p4_closeout_v22r1.json']:
        c['protected'][p]=state(ROOT/p)
    events=[json.loads(x) for v in ['full_baseline_v22','full_baseline_v22r1'] for x in (ROOT/f'logs/P4/{v}/calls.jsonl').read_text(encoding='utf-8').splitlines()]
    intents={x['call_id'] for x in events if x['kind']=='forward_intent'};returns={x['call_id'] for x in events if x['kind']=='forward_returned'}
    optim=sum(x['kind']=='optimizer_returned' for x in events);research=sum(x['kind']=='optimizer_returned' and x['phase']=='research' for x in events)
    need(optim==6100 and research==5600 and intents==returns,'Cumulative budget/accounting differs')
    c['prior_logs']=['logs/P4/full_baseline_v22','logs/P4/full_baseline_v22r1']
    c['repair']=dict(number=2,observed='r1 complete160 RGBD F1=.5622963073892355 and FGIoU=.9213898889864959 passed, last3 monitor span=.0548325466586592 failed original .05. All evidence preserved.',hypothesis='Constant learning rate leaves final updates oscillatory. A single fixed five-epoch 10x lower-learning-rate refinement may stabilize. Prospective repair hypothesis, not claimed result.',changed='New version continues both actual epoch30 optimizer/scaler/RNG checkpoints at lr1e-5; all other training/data/targets/decoder/scientific gates unchanged. Exactly five paired refinement epochs31..35; select earliest best paired mean within these five, never choose based on full160 results. No further parameter trial.',prior_forward_intents=len(intents),prior_known_returns=len(returns),prior_unknown=0,prior_optimizer_returns=optim,prior_research_updates=research,cumulative_budget='Prior5600 research+800 refinement=6400, prior500 fit with no refit gives6900 total accepted updates <=7000. Shared35000 forward intents,4h and output16GiB unchanged.')
    c['training']=dict(old['training'],learning_rate=1e-5,start_epoch=30,max_epochs=35,max_new_research_updates=800,max_engineering_fit_updates=0,selection='Same epoch for both arms: maximum paired scene-macro F1 among fixed refinement epochs31..35, earliest tie; finish all5 regardless of result. No repeated tail trials.')
    c['purpose']='P4 final bounded internal repair2: fixed low-learning-rate stabilization with remaining800 research updates'
    c['continuation']={}
    for arm in ['rgb','rgbd']:
        directory=ROOT/f'experiments/p4_full_baseline_v22r1/research/{arm}/checkpoints';manifest=read(directory/read(directory/'latest.json')['manifest']);path=directory/manifest['file']
        need(manifest['global_step']==2400 and state(path)['sha256']==manifest['sha256'],'Parent checkpoint binding failed')
        c['continuation'][arm]=dict(path=path.relative_to(ROOT).as_posix(),sha256=manifest['sha256'],config_fingerprint=manifest['config_fingerprint'],data_fingerprint=manifest['data_fingerprint'])
    once(CONFIG,c);once(LOG/'freeze.json',dict(at=now(),protocol_sha256=file_sha256(CONFIG),runner_sha256=file_sha256(Path(__file__)),protected_count=len(c['protected']),prior_counts=c['repair']))
    print(json.dumps(dict(status='repair2_frozen',prior_updates=optim,remaining_research_updates=800,prior_forward_intents=len(intents)),ensure_ascii=False),flush=True)

class Session:
    def __init__(self,c):
        self.c=c;self.start=time.monotonic();self.uid=uuid.uuid4().hex;self.seq=0;self.context={};self.done=threading.Event()
        OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
        self.events=LOG/'calls.jsonl';self.forward_total=0;self.optim_total=0
        for ledger in [*(ROOT/p/'calls.jsonl' for p in c['prior_logs']),self.events]:
            if ledger.exists():
                for line in ledger.read_text(encoding='utf-8').splitlines():
                    row=json.loads(line);self.forward_total+=row['kind']=='forward_intent';self.optim_total+=row['kind']=='optimizer_returned'
        self.stream=self.events.open('a',encoding='utf-8',buffering=1)
        self.runtime={'previous_seconds':sum(read(p).get('seconds',0) for directory in [LOG,*(ROOT/p for p in c['prior_logs'])] for p in directory.glob('invocation_*.json'))}
        self.base={p:state(ROOT/p) for p in c['protected']}
        need(all(v['sha256']==c['protected'][p]['sha256'] and v['bytes']==c['protected'][p]['bytes'] for p,v in self.base.items()),'Frozen source/data mismatch')
        need(trainer.runtime_identity()==c['runtime'],'Runtime identity differs')
        threading.Thread(target=self.watchdog,daemon=True).start()
    def event(self,kind,**x):
        self.stream.write(json.dumps(dict(kind=kind,at=now(),**self.context,**x),ensure_ascii=False,allow_nan=False)+'\n');self.stream.flush();os.fsync(self.stream.fileno())
    def watchdog(self):
        if not self.done.wait(max(0,self.c['resources']['wall_seconds']-self.runtime['previous_seconds'])):
            try:self.event('timeout',no_automatic_budget_extension=True)
            finally:os._exit(124)
    def guard(self):
        c=self.c['resources'];need(self.runtime['previous_seconds']+time.monotonic()-self.start<c['wall_seconds'],'P4 wall budget reached')
        need(evaluation.peak_rss()<c['rss_bytes'],'RSS budget reached')
        need(torch.cuda.max_memory_reserved()<c['gpu_bytes'],'GPU budget reached')
        need(shutil.disk_usage(ROOT).free>=c['disk_reserve_bytes'],'Disk reserve reached')
        need(self.optim_total<c['max_recorded_optimizer_updates'],'Optimizer budget reached')
        need(sum(p.stat().st_size for d in [RUN,OUT,LOG,*(ROOT/p for p in ['experiments/p4_full_baseline_v22','results/P4/full_baseline_v22','logs/P4/full_baseline_v22','experiments/p4_full_baseline_v22r1','results/P4/full_baseline_v22r1','logs/P4/full_baseline_v22r1'])] for p in d.rglob('*') if p.is_file())<c['output_bytes'],'Output budget reached')
    def pre(self,model,args):
        need(self.forward_total<self.c['resources']['max_model_forward_calls'],'Forward budget reached')
        self.seq+=1;self.forward_total+=1;model._p4_call=self.uid+'_'+str(self.seq);self.event('forward_intent',call_id=model._p4_call)
    def post(self,model,args,output):torch.cuda.synchronize();self.event('forward_returned',call_id=model._p4_call)
    def model(self,arm):
        trainer.reset_random_streams(self.c['seed']);model=PilotInstanceNet(mode=arm).cuda()
        parent=self.c['parent_checkpoints'][arm];payload=torch.load(ROOT/parent['path'],map_location='cpu',weights_only=False)
        need(payload['config_fingerprint']==parent['config_fingerprint'] and payload['data_fingerprint']==parent['data_fingerprint'],'Parent checkpoint binding differs')
        model.load_state_dict(payload['model'],strict=True);del payload
        model.register_forward_pre_hook(self.pre);model.register_forward_hook(self.post);return model
    def evaluate(self,model,arm,records,export=None):
        model.eval();rows=[];context=dict(self.context)
        for r in records:
            stem=f"scene_{r['scene']:04d}_{r['frame']}";self.context=dict(context,frame=stem)
            path=export/(stem+'.json') if export else None
            if path is not None and path.exists():
                row=read(path);need(state(export/(stem+'.npz'))['sha256']==row['cache_sha256'],'Export cache changed');rows.append(row);continue
            inputs=evaluation.prepare_input(ROOT,self.c,r,arm)
            with torch.inference_mode():heads=model(torch.from_numpy(inputs)[None].cuda())
            need(all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in heads.values()),'Nonfinite/non-FP32 evaluation')
            numpy_heads=dict(foreground_probability=heads['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=heads['centers'][0,0].sigmoid().cpu().numpy(),offsets=heads['offsets'][0].cpu().numpy())
            pred,fg,scores,centers=decode_with_scores(numpy_heads,**self.c['decoder'])
            if export:
                need(np.array_equal(pred,decode_instances(heads,**self.c['decoder'])),'Independent decoders disagree')
                export.mkdir(parents=True,exist_ok=True);np.savez_compressed(export/(stem+'.npz'),instances=pred,foreground=fg,**numpy_heads)
            # Labels are decoded only after output calculation (and after export when requested).
            gt=evaluation.load_labels(ROOT,self.c,r);metrics=evaluation.frame_metrics(pred,fg,gt)
            row=dict(scene=r['scene'],frame=r['frame'],arm=arm,metrics=metrics,scores=scores)
            if export:row['cache_sha256']=file_sha256(export/(stem+'.npz'));once(path,row)
            rows.append(row);self.guard();del heads,inputs,numpy_heads
        scenes=[]
        for scene in sorted({r['scene'] for r in rows}):
            subset=[r for r in rows if r['scene']==scene];tp=sum(r['metrics']['instances']['tp'] for r in subset);fp=sum(r['metrics']['instances']['fp'] for r in subset);fn=sum(r['metrics']['instances']['fn'] for r in subset)
            inter=sum(r['metrics']['foreground_intersection'] for r in subset);union=sum(r['metrics']['foreground_union'] for r in subset)
            scenes.append(dict(scene=scene,frames=len(subset),f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,foreground_iou=inter/union if union else 1.,predicted_instances=sum(r['metrics']['predicted_instances'] for r in subset)))
        self.context=context
        return dict(frames=len(rows),scene_macro_f1=float(np.mean([s['f1'] for s in scenes])),scene_macro_foreground_iou=float(np.mean([s['foreground_iou'] for s in scenes])),scenes=scenes,rows=rows)
    def updates(self,arm,phase,end_epoch=None):
        c=self.c;t=c['training'];directory=RUN/phase/arm;model=self.model(arm)
        opt=torch.optim.AdamW(model.parameters(),lr=c['fitting']['learning_rate'] if phase=='fit' else t['learning_rate'],weight_decay=t['weight_decay']);scaler=torch.amp.GradScaler('cuda',init_scale=t['amp_initial_scale'])
        store=CheckpointStore(directory/'checkpoints',fingerprint(dict(protocol=file_sha256(CONFIG),arm=arm,phase=phase)),c['data_fingerprint'],keep_last=2)
        records=c['fit_records'] if phase=='fit' else c['train_records'];dataset=GraspNetFrames(ROOT/c['data_root'],records,size=c['input_size'],mode=arm,seed=c['seed'],augment=phase!='fit')
        cursor=dict(epoch=0,next_batch=0,global_step=0)
        if (directory/'checkpoints/latest.json').exists():cursor=store.load_latest(model,opt,scaler=scaler)['cursor']
        elif phase=='research':
            parent=c['continuation'][arm];origin=ROOT/parent['path'];need(file_sha256(origin)==parent['sha256'],'Continuation parent changed')
            original_store=CheckpointStore(origin.parent,parent['config_fingerprint'],parent['data_fingerprint'],keep_last=2)
            loaded=original_store.load_latest(model,opt,scaler=scaler);cursor=loaded['cursor'];need(Path(loaded['checkpoint_path'])==origin and cursor==dict(epoch=30,next_batch=0,global_step=2400),'Continuation cursor mismatch')
            before=torch.load(origin,map_location='cpu',weights_only=False)
            need(all(torch.equal(v.detach().cpu(),before['model'][k]) for k,v in model.state_dict().items()),'Parent weights did not load exactly')
            for group in opt.param_groups:group['lr']=t['learning_rate']
            saved=store.save(model,opt,30,0,2400,True,scaler=scaler,extra=dict(phase=phase,arm=arm,continuation_source=parent))
            saved_path=directory/'checkpoints'/saved['file'];after=torch.load(saved_path,map_location='cpu',weights_only=False)
            def exact(x,y):
                if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
                if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
                if isinstance(x,dict):return x.keys()==y.keys() and all(exact(x[k],y[k]) for k in x)
                if isinstance(x,(tuple,list)):return type(x)==type(y) and len(x)==len(y) and all(exact(a,b) for a,b in zip(x,y))
                return x==y
            changed=before['optimizer'];
            for group in changed['param_groups']:group['lr']=1e-5
            need(exact(before['model'],after['model']) and exact(changed,after['optimizer']) and exact(before['scaler'],after['scaler']) and exact(before['rng'],after['rng']),'Continuation migration changed weights/moments/scaler/RNG')
            migration=RUN/'migration';migration.mkdir(parents=True,exist_ok=True);shutil.copy2(saved_path,migration/(arm+'.pt'))
            once(migration/(arm+'.json'),dict(parent=parent,continued_cursor=cursor,only_optimizer_change=dict(learning_rate=1e-5),exact_weights_moments_scaler_rng=True,new_checkpoint_sha256=file_sha256(migration/(arm+'.pt'))))
            del before,after,changed

        epoch=cursor['epoch'];next_batch=cursor['next_batch'];step=cursor['global_step']
        cfg=dict(protocol=dict(gradient_clip_norm=1.),device='cuda',effective_amp=True,amp_max_overflow_retries=t['amp_max_overflow_retries'])
        max_step=c['fitting']['max_updates_per_arm'] if phase=='fit' else end_epoch*80
        history_path=directory/'updates.jsonl';history_path.parent.mkdir(parents=True,exist_ok=True)
        with history_path.open('a',encoding='utf-8') as history:
            while step<max_step:
                model.train();order=list(range(4)) if phase=='fit' else dataset.epoch_order(epoch)
                while next_batch<len(order) and step<max_step:
                    selected=order[next_batch:next_batch+4];self.context=dict(phase=phase,arm=arm,epoch=epoch+1,update=step+1);self.guard()
                    loss,parts,norm,retries=trainer.train_update(model,opt,scaler,dataset,selected,epoch,cfg)
                    self.optim_total+=1;self.event('optimizer_returned',loss=loss,overflow_retries=retries)
                    step+=1;next_batch+=len(selected)
                    if next_batch==len(order):epoch+=1;next_batch=0;ended=True
                    else:ended=False
                    history.write(json.dumps(dict(step=step,epoch=epoch,next_batch=next_batch,loss=loss,components=parts,gradient_norm=norm,overflow_retries=retries))+'\n');history.flush()
                    if step%20==0 or (ended and phase!='fit') or step==max_step:
                        store.save(model,opt,epoch,next_batch,step,True,scaler=scaler,extra=dict(phase=phase,arm=arm))
                        atomic_json(OUT/'progress.json',dict(at=now(),status='running',phase=phase,arm=arm,epoch=epoch,next_batch=next_batch,updates=step,forward_intents_total=self.forward_total,optimizer_returns_total=self.optim_total,all_tasks_percent=26,experiment_percent=32.5))
                    if phase=='fit' and step in c['fitting']['check_steps']:
                        self.context=dict(phase='fit_evaluation',arm=arm,update=step)
                        result=self.evaluate(model,arm,records);passed=all(x['metrics']['instances']['f1_iou50']['value']>=.9 and x['metrics']['foreground_pixel_iou']['value']>=.9 for x in result['rows'])
                        once(directory/f'check_{step:04d}.json',dict(passed=passed,result=result));print(json.dumps(dict(stage='fit',arm=arm,step=step,passed=passed,f1=result['scene_macro_f1'])),flush=True)
                        if passed:
                            once(directory/'complete.json',dict(status='passed',steps=step));del model,opt,scaler;torch.cuda.empty_cache();return
                    if ended:break
        if phase=='fit':raise RuntimeError('Small fixed-data fit gate failed for '+arm)
        self.context=dict(phase='monitor',arm=arm,epoch=end_epoch)
        report=self.evaluate(model,arm,c['monitor_records']);once(OUT/f'epochs/{end_epoch:02d}_{arm}.json',report)
        del model,opt,scaler;torch.cuda.empty_cache();return report
    def finish(self,status,error=None):
        self.done.set();self.stream.close()
        once(LOG/('invocation_'+self.uid+'.json'),dict(at=now(),seconds=time.monotonic()-self.start,status=status,error=error,peak_rss_bytes=evaluation.peak_rss(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved()))

def run_all():
    c=read(CONFIG);freeze=read(LOG/'freeze.json');need(file_sha256(CONFIG)==freeze['protocol_sha256'] and file_sha256(Path(__file__))==freeze['runner_sha256'],'Runner/protocol changed')
    if (OUT/'training_and_export.json').exists():
        print('Training/export already complete; proceed to pending independent P4 acceptance without recomputing.');return
    s=Session(c)
    try:
        for arm in ['rgb','rgbd']:
            need(read(ROOT/f'experiments/p4_full_baseline_v22r1/fit/{arm}/complete.json')['status']=='passed','Inherited fit failed')
        selected=None;best=-1.;stale=0;pair_history=[]
        for epoch in range(31,36):
            scores={}
            for arm in ['rgb','rgbd']:
                path=OUT/f'epochs/{epoch:02d}_{arm}.json';scores[arm]=read(path) if path.exists() else s.updates(arm,'research',epoch)
            score=float(np.mean([scores[a]['scene_macro_f1'] for a in ['rgb','rgbd']]))
            if score>best:
                selected=epoch;best=score;stale=0
                for arm in ['rgb','rgbd']:
                    target=RUN/'selected'/f'epoch_{epoch:02d}_{arm}.pt'
                    if not target.exists():
                        folder=RUN/'research'/arm/'checkpoints';manifest=read(folder/read(folder/'latest.json')['manifest']);payload=torch.load(folder/manifest['file'],map_location='cpu',weights_only=False)
                        need(payload['cursor']['epoch']==epoch,'Selected checkpoint is not requested epoch');target.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(model=payload['model'],epoch=epoch,arm=arm,protocol_sha256=file_sha256(CONFIG)),target);del payload
            else:stale+=1
            pair_history.append(score)
            atomic_json(OUT/'selection.json',dict(selected_epoch=selected,best_paired_monitor_f1=best,last_epoch=epoch,no_improvement_epochs=stale,paired_monitor_history=pair_history))
            print(json.dumps(dict(stage='paired_epoch',epoch=epoch,rgb_f1=scores['rgb']['scene_macro_f1'],rgbd_f1=scores['rgbd']['scene_macro_f1'],best_epoch=selected,patience=stale)),flush=True)
            if epoch==5:need(scores['rgbd']['scene_macro_f1']>=c['scientific_gates']['first_five_epoch_rgbd_monitor_f1_min'],'Five-epoch validity gate failed; no automatic long run')
            # Fixed five-epoch refinement; never inspect a passing tail to stop early.
        summary={}
        for arm in ['rgb','rgbd']:
            s.context=dict(phase='final_export',arm=arm,selected_epoch=selected);model=s.model(arm)
            checkpoint=RUN/'selected'/f'epoch_{selected:02d}_{arm}.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
            summary[arm]=s.evaluate(model,arm,c['development_records'],OUT/'exports'/arm)
            once(OUT/(arm+'_full_development.json'),summary[arm]);del model;torch.cuda.empty_cache()
        gates=dict(fit=True,first_five_parent_gate=read(ROOT/'results/P4/full_baseline_v22r1/epochs/05_rgbd.json')['scene_macro_f1']>=.2,rgbd_macro_f1=summary['rgbd']['scene_macro_f1']>=.5,rgbd_foreground_iou=summary['rgbd']['scene_macro_foreground_iou']>=.7,monitor_stability=len(pair_history)>=3 and max(pair_history[-3:])-min(pair_history[-3:])<=.05,noncollapsed_all_scenes=all(x['predicted_instances']>0 for a in summary.values() for x in a['scenes']))
        need(all(state(ROOT/p)==v for p,v in s.base.items()),'Frozen originals changed during training/evaluation')
        once(OUT/'training_and_export.json',dict(status='completed_training_and_export',selected_epoch=selected,completed_epochs=epoch,new_research_updates_per_arm=(epoch-30)*80,parent_research_updates_per_arm=4000,continued_from_epoch=30,refinement_epochs=5,scientific_gates=gates,preliminary_baseline_usable=all(gates.values()),full_P4_accepted=False,required_remaining=['mask_AP_boundary_error_analysis','representative_grasp_linkage','independent_full_readonly_acceptance','stage_summary'],all_originals_unchanged=True,at=now()))
        atomic_json(OUT/'progress.json',dict(at=now(),status='training_export_finished_next_internal_acceptance',selected_epoch=selected,completed_epochs=epoch,preliminary_baseline_usable=all(gates.values()),all_tasks_percent=26,experiment_percent=32.5))
        s.finish('completed_training_and_export')
    except BaseException as e:
        atomic_json(OUT/'progress.json',dict(at=now(),status='internal_failure_needs_repair',error_type=type(e).__name__,error=str(e),all_tasks_percent=26,experiment_percent=32.5))
        s.finish('failed_preserved',repr(e));raise
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    with project_lock(ROOT):
        if args.mode=='prepare':prepare()
        else:run_all()
if __name__=='__main__':main()
