"""One P5 gate-strength repair; frozen old controls are read-only."""
from p5_runtime_v23r1 import *
from scripts.p5_cache_v23r1 import save_cache,load_cache

def make(s,arm):
    c=s.c;trainer.reset_random_streams(SEED);q=QualityNet();q.load_state_dict(torch.load(qcheckpoint(),map_location='cpu',weights_only=False)['model']);m=GatedModel(q,arm).cuda();m.backbone.load_state_dict(torch.load(ROOT/c['parent'],map_location='cpu',weights_only=False)['model']);s.hook(m.quality,'quality');s.hook(m.backbone,'segmentation');return m

def aggregate(rows):
    scenes=[]
    for sc in sorted({r['scene'] for r in rows}):
        sub=[r['metrics'] for r in rows if r['scene']==sc];tp=sum(x['instances']['tp'] for x in sub);fp=sum(x['instances']['fp'] for x in sub);fn=sum(x['instances']['fn'] for x in sub);i=sum(x['foreground_intersection'] for x in sub);u=sum(x['foreground_union'] for x in sub)
        scenes.append(dict(scene=sc,frames=len(sub),f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,foreground_iou=i/u if u else 1.,predicted_instances=sum(x['predicted_instances'] for x in sub)))
    return dict(frames=len(rows),scene_macro_f1=float(np.mean([x['f1'] for x in scenes])),scene_macro_foreground_iou=float(np.mean([x['foreground_iou'] for x in scenes])),scenes=scenes,rows=rows)

def evaluate(s,m,arm,condition,records,folder=None):
    c=s.c;m.eval();rows=[];context=dict(s.ctx)
    for r in records:
        stem=f"scene_{r['scene']:04d}_{r['frame']}";p=folder/(stem+'.json') if folder else None;cache=folder/(stem+'.npz') if folder else None
        if p and p.exists():row=read(p);need(sha(cache)==row['cache_sha256'],'Export changed');rows.append(row);continue
        s.ctx=dict(context,condition=condition,scene=r['scene'],frame=r['frame']);s.guard();raw=evaluation.prepare_input(ROOT,c,r,'rgbd');x,_,meta=corrupt(torch.from_numpy(raw),r,c['quality_screen_epoch'],condition)
        with torch.inference_mode():h=m(x[None].cuda())
        heads=dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=h['centers'][0,0].sigmoid().cpu().numpy(),offsets=h['offsets'][0].cpu().numpy());need(all(v.dtype==np.float32 and np.isfinite(v).all() for v in heads.values()),'Invalid FP32 segmentation export');pred,fg,scores,centers=decode_with_scores(heads,**c['decoder'])
        if folder:
            folder.mkdir(parents=True,exist_ok=True);save_cache(cache,instances=pred,foreground=fg,gate=h['effective_gate'][0,0].cpu().numpy(),**heads)
        gt=evaluation.load_labels(ROOT,c,r);row=dict(scene=r['scene'],frame=r['frame'],arm=arm,condition=condition,ablation=m.ablation,metrics=evaluation.frame_metrics(pred,fg,gt),scores=scores,corruption=meta)
        if folder:row['cache_sha256']=sha(cache);once(p,row)
        rows.append(row);del h,x,raw,heads
    s.ctx=context;return aggregate(rows)

def train_epoch(s,arm,end_epoch):
    c=s.c;t=c['segmentation'];directory=RUN/'segmentation'/arm;monitor=BASE/'epochs'/f'{end_epoch:02d}_{arm}.json'
    if monitor.exists():return read(monitor)
    m=make(s,arm);opt=torch.optim.AdamW(m.backbone.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay']);scaler=torch.amp.GradScaler('cuda',init_scale=t['amp_initial_scale']);store=CheckpointStore(directory/'checkpoints',fingerprint(dict(protocol=sha(CONFIG),arm=arm)),c['data_fingerprint'],keep_last=2);ds=Frames(ROOT/c['data_root'],c['train_records'],size=c['input_size'],seed=SEED,augment=True);cur=dict(epoch=0,next_batch=0,global_step=0)
    if (directory/'checkpoints/latest.json').exists():cur=store.load_latest(m,opt,scaler=scaler)['cursor']
    ep,pos,step=cur['epoch'],cur['next_batch'],cur['global_step'];need(step<=end_epoch*80,'Checkpoint beyond missing monitor; do not overwrite history')
    cfg=dict(protocol=dict(gradient_clip_norm=1.),device='cuda',effective_amp=True,amp_max_overflow_retries=t['amp_max_overflow_retries'])
    while step<end_epoch*80:
        order=ds.epoch_order(ep);sel=order[pos:pos+4];s.ctx=dict(phase='segmentation_training',arm=arm,epoch=ep+1,update=step+1);s.guard();m.train();m.quality.eval();m.strength=strength(step);key=s.optimizer_intent();loss,parts,norm,retries=trainer.train_update(m,opt,scaler,ds,sel,ep,cfg);s.optimizer_returned(key,loss,components=parts,gradient_norm=norm,overflow_retries=retries);step+=1;pos+=len(sel)
        if pos==len(ds):ep+=1;pos=0
        if step%20==0:
            store.save(m,opt,ep,pos,step,True,scaler=scaler);status('segmentation_training',arm=arm,epoch=ep,next_batch=pos,updates=step,max_updates_per_arm=800,forward_intents=s.forwards)
    m.strength=strength(step);s.ctx=dict(phase='monitor',arm=arm,epoch=end_epoch);reports={condition:evaluate(s,m,arm,condition,c['monitor_records']) for condition in ['clean','joint']}
    # The quality parameters must stay exactly at the independently screened head.
    qstate=torch.load(qcheckpoint(),map_location='cpu',weights_only=False)['model'];need(all(torch.equal(v.detach().cpu(),qstate[k]) for k,v in m.quality.state_dict().items()),'Frozen quality head changed')
    saved=RUN/'epochs'/f'{end_epoch:02d}_{arm}.pt';saved.parent.mkdir(parents=True,exist_ok=True)
    if saved.exists():need(read(monitor.with_suffix('.checkpoint.json'))['sha256']==sha(saved),'Epoch checkpoint changed')
    else:torch.save(dict(model=m.state_dict(),epoch=end_epoch,arm=arm,strength=m.strength,protocol_sha256=sha(CONFIG)),saved);once(monitor.with_suffix('.checkpoint.json'),dict(path=saved.relative_to(ROOT).as_posix(),sha256=sha(saved)))
    once(monitor,reports);print(json.dumps(dict(epoch=end_epoch,arm=arm,clean_f1=reports['clean']['scene_macro_f1'],joint_f1=reports['joint']['scene_macro_f1'])),flush=True);del m,opt,scaler;torch.cuda.empty_cache();return reports


def prepare():
    need(not CONFIG.exists(),'Repair is already frozen')
    c=read(ROOT/'configs/p5_quality_gate_v23.json')
    need(read(OLD_LOG/'root_final_review.json')['passed'],'Old final root review not passed')
    c['version']='23r1';c['created_at']=now();c['segmentation']['gate_strength_max']=.25
    c['repair']=dict(authorization='User: 那就重跑把; one proposed strength0.5 to0.25 repair',only_scientific_change='maximum gate suppression0.5 to0.25',quality_retrained=False,fixed_arm_retrained=False,quality_checkpoint=qcheckpoint().relative_to(ROOT).as_posix(),fixed_epoch_source=(OLD_RUN/'epochs').relative_to(ROOT).as_posix(),finite_trial=1,next_trial_or_P6=False,early_screen=dict(epoch=5,records='unchanged20 monitor records',conditions=c['core_conditions'],new_cached_frame_entries=200,clean_f1_min=.5,noncollapsed_clean_scenes=True,continue_only_if_four_degradation_mean_F1_gain_strictly_positive=True,not_final_AP_acceptance=True),reuse_fixed_exports_only_if_same_selected_epoch=True)
    c['resources'].update(max_optimizer_returns=800,max_forward_intents=14000,max_official_ap_calls=160,wall_seconds=3600,cumulative_optimizer_returns=3000,cumulative_forward_intents=35000,cumulative_official_ap_calls=320,cumulative_wall_seconds=10800,max_adjustments=1)
    c['prior_accounting']=counts(prior=True);c['prior_compute_seconds']=usage(prior=True)
    inherited=read(OLD_LOG/'final_readonly.json')['files']
    for p,v in inherited.items():need(stat(ROOT/p)==v,'Prior accepted result changed: '+p)
    c['protected'].update(inherited)
    paths=['scripts/p5_common_v23r1.py','scripts/p5_cache_v23r1.py','scripts/p5_runtime_v23r1.py','scripts/p5_repair_v23r1.py','docs/P5单次门控修复协议_v23r1.txt','configs/p5_quality_gate_v23.json','logs/P5/quality_gate_v23/root_final_review.json','logs/P5/quality_gate_v23/final_readonly.json','logs/P5/quality_gate_v23/calls.jsonl','logs/P5/quality_gate_v23r1/codec_lzma_test.json','logs/P5/quality_gate_v23r1/preflight.json','runtime/pipeline/state.json']
    c['protected'].update({p:stat(ROOT/p) for p in paths})
    need(strength(0)==0 and strength(80)==0 and strength(160)==.125 and strength(240)==.25 and strength(800)==.25,'Strength schedule failed')
    need(c['decoder']==read(ROOT/'configs/p5_quality_gate_v23.json')['decoder'],'Decoder changed')
    once(CONFIG,c);once(LOG/'freeze.json',dict(protocol_sha256=sha(CONFIG),new_model_calls=0,prior_accepted_files_verified=len(inherited),strength_schedule_test_passed=True))
    once(LOG/'segmentation_freeze.json',dict(protocol_sha256=sha(CONFIG),protected={p:c['protected'][p] for p in paths}))
    status('repair_frozen',new_training_updates=0);print('P5 v23r1 frozen',flush=True)


def load_epoch(s,arm,epoch):
    m=make(s,arm);p=(OLD_RUN if arm=='fixed' else RUN)/'epochs'/f'{epoch:02d}_{arm}.pt';payload=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(payload['model']);m.strength=0. if arm=='fixed' else payload['strength'];m.ablation='learned';return m


def early_screen(s):
    path=BASE/'first_five_gate.json'
    if path.exists():return read(path)
    c=s.c;sets=[];values={}
    for arm in ['fixed','quality']:
        m=load_epoch(s,arm,5)
        for condition in CORE:
            name=f'screen_{arm}_learned_{condition}';folder=BASE/'screen'/name;report_path=BASE/'screen'/(name+'.json');s.ctx=dict(phase='early_screen',arm=arm,epoch=5,set=name)
            if report_path.exists():report=read(report_path)
            else:report=evaluate(s,m,arm,condition,c['monitor_records'],folder);once(report_path,report)
            if condition in ['clean','joint']:
                old=read((OLD_BASE if arm=='fixed' else BASE)/'epochs'/f'05_{arm}.json')[condition]
                need(report['scene_macro_f1']==old['scene_macro_f1'] and report['scenes']==old['scenes'],'Early screen differs from epoch5 monitor')
            values[(arm,condition)]=report;sets.append(dict(name=name,kind='screen',arm=arm,ablation='learned',condition=condition,frames=20,official_AP=False,report_path=report_path.relative_to(ROOT).as_posix(),directory=folder.relative_to(ROOT).as_posix(),report_sha256=sha(report_path)))
        del m;torch.cuda.empty_cache()
    gains={d:values['quality',d]['scene_macro_f1']-values['fixed',d]['scene_macro_f1'] for d in CORE if d!='clean'};mean=float(np.mean(list(gains.values())))
    gates={arm:values[arm,'clean']['scene_macro_f1']>=.5 and all(x['predicted_instances']>0 for x in values[arm,'clean']['scenes']) for arm in ['fixed','quality']};gates['mean_degraded_F1_positive']=mean>0
    result=dict(epoch=5,sets=sets,gates=gates,passed=all(gates.values()),mean_degraded_F1_gain=mean,by_condition_F1_gain=gains,not_full_development_or_AP_acceptance=True)
    once(path,result)
    if not result['passed']:once(BASE/'finite_stop.json',dict(reason='Prespecified epoch5 finite continuation screen failed',screen_sha256=sha(path),full160_AP_not_run=True,no_second_repair=True))
    print(json.dumps(dict(early_screen=result['gates'],mean_gain=mean)),flush=True);return result


def training(s):
    if (BASE/'finite_stop.json').exists():return None
    if (BASE/'selection.json').exists():return read(BASE/'selection.json')
    values=[]
    for epoch in range(1,11):
        r={'fixed':read(OLD_BASE/'epochs'/f'{epoch:02d}_fixed.json'),'quality':train_epoch(s,'quality',epoch)}
        score=float(np.mean([x[k]['scene_macro_f1'] for x in r.values() for k in ['clean','joint']]));values.append(dict(epoch=epoch,paired_mean=score))
        if epoch==1:
            a=torch.load(OLD_RUN/'epochs/01_fixed.pt',map_location='cpu',weights_only=False)['model'];b=torch.load(RUN/'epochs/01_quality.pt',map_location='cpu',weights_only=False)['model'];need(a.keys()==b.keys() and all(torch.equal(v,b[k]) for k,v in a.items()),'First80 unit-gate tensors differ');once(BASE/'paired_unit_training.json',dict(epoch=1,updates_per_arm=80,fixed_updates_reused=True,bitwise_all_model_tensors_equal=True));del a,b
        if epoch==5 and not early_screen(s)['passed']:return None
    chosen=max(values,key=lambda x:x['paired_mean']);last=[x['paired_mean'] for x in values[-3:]];selected=dict(epoch=chosen['epoch'],selection_value=chosen['paired_mean'],epoch_values=values,last_three_span=max(last)-min(last),stability_passed=max(last)-min(last)<=.05,fixed_weights_reused_same_epoch=True)
    once(BASE/'selection.json',selected);return selected


def export(s,selected):
    c=s.c;epoch=selected['epoch'];sets=[];old_selection=read(OLD_BASE/'selection.json')
    for arm in ['fixed','quality']:
        reuse=arm=='fixed' and epoch==old_selection['epoch'];m=None if reuse else load_epoch(s,arm,epoch)
        tasks=[('core',condition,'learned',c['development_records']) for condition in CORE]+[('graded',condition,'learned',c['monitor_records']) for condition in GRADED]
        if arm=='quality':tasks += [('ablation',condition,ablation,c['development_records']) for ablation in ['unit','mean'] for condition in ['clean','joint']]
        for kind,condition,ablation,records in tasks:
            name=f'{kind}_{arm}_{ablation}_{condition}';source=OLD_BASE if reuse else BASE;folder=source/'exports'/name;report_path=source/'exports'/(name+'.json')
            if report_path.exists():report=read(report_path)
            else:
                need(not reuse,'Missing reusable fixed output');m.ablation=ablation;s.ctx=dict(phase='final_export',set=name,arm=arm);report=evaluate(s,m,arm,condition,records,folder);once(report_path,report)
            sets.append(dict(name=name,kind=kind,arm=arm,ablation=ablation,condition=condition,frames=len(records),official_AP=kind!='graded',report_sha256=sha(report_path),report_path=report_path.relative_to(ROOT).as_posix(),directory=folder.relative_to(ROOT).as_posix(),reused_fixed_output=reuse));status('final_export',completed_sets=len(sets),total_sets=28,selected_epoch=epoch,forward_intents=s.forwards);print(json.dumps(dict(set=name,reused=reuse,frames=report['frames'],f1=report['scene_macro_f1'])),flush=True)
        del m;torch.cuda.empty_cache()
    once(BASE/'exports_complete.json',dict(selected_epoch=epoch,sets=sets,total_frame_entries=sum(x['frames'] for x in sets),official_AP_sets=14));need(sum(x['frames'] for x in sets)==2520,'Export scope mismatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','verify']);a=p.parse_args()
    with project_lock(ROOT):
        if a.mode=='freeze':prepare();return
        c=verify()
        if a.mode=='verify':print('Repair frozen inputs unchanged');return
        if (BASE/'exports_complete.json').exists() or (BASE/'finite_stop.json').exists():print('Existing completed finite endpoint; zero new computation');return
        s=Session(c)
        try:
            selected=training(s)
            if selected is not None:export(s,selected)
            verify();s.finish('complete');status('screen_stopped_waiting_readonly' if selected is None else 'exports_complete_waiting_evaluation')
        except BaseException as e:s.finish('failed',repr(e));status('repair_failed',error=repr(e));raise
if __name__=='__main__':main()
