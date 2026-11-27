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
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet,decode_instances
import numpy as np
import torch
CONFIG=ROOT/'configs/p4_full_baseline_v22.json'
OUT=ROOT/'results/P4/full_baseline_v22'
RUN=ROOT/'experiments/p4_full_baseline_v22'
LOG=ROOT/'logs/P4/full_baseline_v22'
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
    need(not CONFIG.exists() and not RUN.exists(),'Previous P4 v22 protocol/run exists')
    LOG.mkdir(parents=True,exist_ok=True)
    parent=read(ROOT/'configs/segmentation_baseline_v16.json');train=read(ROOT/'results/P4/segmentation_baseline_v16_training.json')
    need(file_sha256(ROOT/'configs/segmentation_baseline_v16.json')==train['protocol_sha256'],'Parent protocol changed')
    manifest=read(ROOT/parent['data_manifest']);split=read(ROOT/parent['split'])
    train_records=manifest['records']['train'];dev_records=manifest['records']['validation']
    need(len(train_records)==320 and len(dev_records)==160,'Declared coverage changed')
    ts={r['scene'] for r in train_records};ds={r['scene'] for r in dev_records}
    need(not ts&ds and ts<=set(split['groups']['train']) and ds<=set(split['groups']['validation']),'Scene leakage')
    fit=[r for r in train_records if r['scene'] in [3,4] and r['frame'] in ['0000','0128']]
    monitor=[r for r in dev_records if r['frame'] in ['0000','0128']]
    need(len(fit)==4 and len(monitor)==20,'Fixed fit/monitor selection changed')
    protected={}
    def protect(path,h=None):
        p=ROOT/path;s=state(p)
        if h:need(s['sha256']==h,'Original identity differs: '+path)
        protected[path]=s
    for p,h in dict(parent['source_hashes'],**parent['input_hashes']).items():protect(p,h)
    for p in ['configs/segmentation_baseline_v16.json','results/P4/segmentation_baseline_v16_training.json','results/P4/baseline_grasp_link_v21/closeout.json','scripts/extract_mask_scores_v17.py','scripts/p4_full_baseline_v22.py','runtime/pipeline/state.json','configs/stages.json','docs/分阶段训练方案.html']:
        protect(p)
    parents={}
    for arm in ['rgb','rgbd']:
        a=train['arms'][arm]['result'];p=Path(a['checkpoint_path']).as_posix();protect(p,a['checkpoint_sha256'])
        parents[arm]=dict(path=p,sha256=a['checkpoint_sha256'],prior_updates=1600,config_fingerprint=a['config_fingerprint'],data_fingerprint=a['data_fingerprint'])
    for rec in train_records+dev_records:
        for kind,p in rec['files'].items():protect((Path(parent['data_root'])/p).as_posix(),rec['sha256'][kind])
    protocol=dict(schema=22,status='frozen',created_at=now(),purpose='Complete P4 development-stage ordinary RGB/RGBD baselines at 640x480, with fixed-budget matched model-only warm starts and explicit failure gates',
        architecture='Unmodified verified PilotInstanceNet from R16; ordinary early fusion replaces failed R18 dual-stream as the development baseline. This is not a repair or efficacy claim for R18.',
        parent_checkpoints=parents,seed=20261022,input_size=[480,640],data_root=parent['data_root'],train_records=train_records,development_records=dev_records,monitor_records=monitor,fit_records=fit,
        split_path=parent['split'],data_fingerprint=fingerprint(dict(train=train_records,development=dev_records)),decoder=parent['decoder'],runtime=parent['runtime'],protected=protected,
        fitting=dict(max_updates_per_arm=200,learning_rate=.001,check_steps=[20,50,100,200],all_four_f1_min=.9,all_four_foreground_iou_min=.9,augment=False),
        training=dict(max_epochs=40,first_gate_epoch=5,early_stopping_patience=8,selection='Same epoch for both arms: maximize mean of the two scene-macro monitor F1 values; strict improvement, earliest tied epoch retained.',learning_rate=.0001,weight_decay=.0001,accumulation=4,batch_size=1,augment=True,gradient_clip_norm=1.,save_every_updates=20,amp_initial_scale=128.,amp_max_overflow_retries=16,max_new_research_updates=6400,max_engineering_fit_updates=400),
        scientific_gates=dict(fit_required=True,first_five_epoch_rgbd_monitor_f1_min=.2,adoption_rgbd_full_development_macro_f1_min=.5,adoption_rgbd_full_development_foreground_iou_min=.7,monitor_last_three_pair_mean_f1_span_max=.05,all_development_scenes_have_instances=True),
        resources=dict(wall_seconds=14400,max_model_forward_calls=35000,max_recorded_optimizer_updates=7000,gpu_bytes=14*1024**3,rss_bytes=6*1024**3,output_bytes=16*1024**3,disk_reserve_bytes=60*1024**3),
        final_evaluation='Export both selected checkpoints on all 160 already-exposed development frames each: FP32 heads, exact unchanged decoder, IoU matching, center confidence, class-agnostic COCO mask AP, boundary error, scene-level diagnostics. Independent cached-output verification. No held-out/test or calibration access.',
        integration='Use selected RGBD at representative existing development scenes to verify original-pixel and candidate linkage under an independently frozen bounded integration contract. Official pretrained grasp weights remain engineering-only with unknown training-scene exposure.',
        repair_policy='Preserve failed versions and evidence. Internal diagnosis/repair is authorized for P4, but frozen contracts must never be altered in place. No unlimited same-hypothesis trials; respect original two-adjustment/resource review rule.',
        completion='Only the complete P4 delivery and truthful usable/conditional/failure decision returns to user for next-part confirmation. Internal fit, epochs, scoring, and checks do not require renewed permission.',
        limitations=['Single-seed development baseline, not P9 three-seed formal result','20 train scenes and 10 repeatedly exposed development scenes; no independent generalization claim','Both arms warm-start from R16 matched 1600-update checkpoints; cumulative training is reported','RGBD retains 6272 extra input-convolution parameters, so exact parameter equality is not claimed','Historical R18 zero-instance/AP result remains unchanged; P5/P8 not passed'])
    # Freeze numerical behavior before training. The unchanged decoder/label target is inherited, not tuned.
    once(CONFIG,protocol)
    once(LOG/'freeze.json',dict(at=now(),protocol_sha256=file_sha256(CONFIG),runner_sha256=file_sha256(Path(__file__)),protected_count=len(protected)))
    plan='P4整体执行协议（v22）\n'+now()+'\n目标：在原640x480尺寸形成RGB和普通RGB-D稳定开发基线、完整评价与误差图，并完成代表性链路验收。原主计划允许结构失效后替换；采用已核验R16普通融合实现，R18负结果保持。\n两臂各从R16最终1600步权重仅加载模型，优化器和随机流重置。原20训练场景320帧与10已暴露开发场景160帧不变，保留验证、校准及测试不打开。\n先固定两训练场景四帧过拟合，最多每臂200更新（单列工程预算），原阈值不变；检查两臂每帧F1及前景IoU均>=0.9。研究训练从原父权重重新开始，不使用拟合探针权重。\n640x480，batch1累积4，AdamW学习率1e-4、权重衰减1e-4、梯度裁剪1、AMP。每臂最多40轮/3200更新，总6400新增研究更新；先5轮检查塌陷，再以两臂场景宏F1均值选择同一最佳轮次，连续8轮无改善提前停。每轮固定20帧监测（每开发场景0000/0128），最终两臂各160完整开发帧评价，监测子集成绩不替代完整成绩。\n预声明采用条件：小样本拟合通过；第5轮RGBD监测F1>=0.2；RGBD完整开发宏F1>=0.5及前景IoU>=0.7；最后三轮配对监测均值跨度<=0.05，各开发场景有实例。这些是开发可用性判断，RGBD无需胜过RGB，完整方法AP/ECE门槛仍由P8判断。\n总资源上限4小时、35000次记录模型前向、7000次记录优化更新（含技术恢复冗余）、GPU14GiB/RSS6GiB、产物16GiB、磁盘余量60GiB。实际前向用durable意图/返回日志，未知调用保留未知；训练工程量与研究量分别报告。每20更新保存可恢复检查点，元数据绑定协议及数据。\n拟合、训练、导出、AP、边界分析、链路验证及独立复验都是P4内部步骤，内部失败保留证据后自行处理；不再逐小轮询问用户。完整P4交付后停止等下一部分确认。自动任务没有恢复。\n全部任务26%，实验32.5%为开始前固定工作包估算；未验收不提前记完成。\n'
    with (ROOT/'docs/P4整体执行协议_v22.txt').open('x',encoding='utf-8') as f:f.write(plan)
    handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,LOG/'handoff_before.txt')
    handoff.write_text('最新：'+now()+' 用户已要求开始P4整体，v22协议已冻结。按小样本拟合→640x480配对基线训练（5轮检查，最多40轮，8轮无改善提前停）→完整开发评价/AP/误差图→代表性链路→独立整体验收推进。内部小轮不等待用户确认。原冻结版本不改，失败另存并按有限预算修复。全部任务26%，实验32.5%，未新增完成分值。自动任务不恢复。详见docs/P4整体执行协议_v22.txt，进程/进度以results/P4/full_baseline_v22/progress.json及日志为准。\n\n历史：\n'+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(dict(status='frozen',source_files=len(protected),max_research_updates=6400,fit_updates_cap=400)))

class Session:
    def __init__(self,c):
        self.c=c;self.start=time.monotonic();self.uid=uuid.uuid4().hex;self.seq=0;self.context={};self.done=threading.Event()
        OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
        self.events=LOG/'calls.jsonl';self.forward_total=0;self.optim_total=0
        if self.events.exists():
            for line in self.events.read_text(encoding='utf-8').splitlines():
                row=json.loads(line);self.forward_total+=row['kind']=='forward_intent';self.optim_total+=row['kind']=='optimizer_returned'
        self.stream=self.events.open('a',encoding='utf-8',buffering=1)
        self.runtime={'previous_seconds':sum(read(p).get('seconds',0) for p in LOG.glob('invocation_*.json'))}
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
        need(sum(p.stat().st_size for d in [RUN,OUT,LOG] for p in d.rglob('*') if p.is_file())<c['output_bytes'],'Output budget reached')
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
            if not (RUN/'fit'/arm/'complete.json').exists():s.updates(arm,'fit')
        selected=None;best=-1.;stale=0;pair_history=[]
        for epoch in range(1,41):
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
            if stale>=8:break
        summary={}
        for arm in ['rgb','rgbd']:
            s.context=dict(phase='final_export',arm=arm,selected_epoch=selected);model=s.model(arm)
            checkpoint=RUN/'selected'/f'epoch_{selected:02d}_{arm}.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
            summary[arm]=s.evaluate(model,arm,c['development_records'],OUT/'exports'/arm)
            once(OUT/(arm+'_full_development.json'),summary[arm]);del model;torch.cuda.empty_cache()
        gates=dict(fit=True,rgbd_macro_f1=summary['rgbd']['scene_macro_f1']>=.5,rgbd_foreground_iou=summary['rgbd']['scene_macro_foreground_iou']>=.7,monitor_stability=len(pair_history)>=3 and max(pair_history[-3:])-min(pair_history[-3:])<=.05,noncollapsed_all_scenes=all(x['predicted_instances']>0 for a in summary.values() for x in a['scenes']))
        need(all(state(ROOT/p)==v for p,v in s.base.items()),'Frozen originals changed during training/evaluation')
        once(OUT/'training_and_export.json',dict(status='completed_training_and_export',selected_epoch=selected,completed_epochs=epoch,new_research_updates_per_arm=epoch*80,parent_research_updates_per_arm=1600,scientific_gates=gates,preliminary_baseline_usable=all(gates.values()),full_P4_accepted=False,required_remaining=['mask_AP_boundary_error_analysis','representative_grasp_linkage','independent_full_readonly_acceptance','stage_summary'],all_originals_unchanged=True,at=now()))
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
