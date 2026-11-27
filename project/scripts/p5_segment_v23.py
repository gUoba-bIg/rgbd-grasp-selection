"""P5 paired segmentation, fixed finite epochs, resumable cached exports."""
from p5_run_v23 import *

def freeze():
    c=verify();need(read(BASE/'quality_screen.json')['quality_usable'],'Quality screen failed');need(read(LOG/'quality_review.json')['passed'],'Independent quality review failed')
    need(read(LOG/'execution_tests.json')['passed'],'Execution recovery tests failed')
    paths=['scripts/p5_segment_v23.py','scripts/p5_quality_review_v23.py','experiments/p5_quality_gate_v23/quality_final.pt','results/P5/quality_gate_v23/quality_screen.json','results/P5/quality_gate_v23/parent_identity.json','logs/P5/quality_gate_v23/quality_review.json']
    paths+=['scripts/test_p5_execution_v23.py','logs/P5/quality_gate_v23/execution_tests.json']
    once(LOG/'segmentation_freeze.json',dict(protocol_sha256=sha(CONFIG),protected={p:stat(ROOT/p) for p in paths}))

def binding():
    c=verify();f=read(LOG/'segmentation_freeze.json');need(f['protocol_sha256']==sha(CONFIG),'Segmentation protocol mismatch')
    for p,s in f['protected'].items():need(stat(ROOT/p)==s,'Segmentation dependency changed '+p)
    return c

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
            folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,instances=pred,foreground=fg,gate=h['effective_gate'][0,0].cpu().numpy(),**heads)
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

def training(s):
    if (BASE/'selection.json').exists():return read(BASE/'selection.json')
    values=[]
    for epoch in range(1,11):
        r={arm:train_epoch(s,arm,epoch) for arm in ['fixed','quality']};score=float(np.mean([x[k]['scene_macro_f1'] for x in r.values() for k in ['clean','joint']]));values.append(dict(epoch=epoch,paired_mean=score))
        if epoch==1:
            a=torch.load(RUN/'epochs/01_fixed.pt',map_location='cpu',weights_only=False)['model'];b=torch.load(RUN/'epochs/01_quality.pt',map_location='cpu',weights_only=False)['model'];need(all(torch.equal(v,b[k]) for k,v in a.items()),'First80 unit-gate paired weights not identical');once(BASE/'paired_unit_training.json',dict(epoch=1,updates_per_arm=80,bitwise_all_model_tensors_equal=True));del a,b
        if epoch==5:
            gates={arm:x['clean']['scene_macro_f1']>=.5 and all(z['predicted_instances']>0 for z in x['clean']['scenes']) for arm,x in r.items()};once(BASE/'first_five_gate.json',dict(gates=gates,passed=all(gates.values())));need(all(gates.values()),'First-five segmentation gate failed; preserve negative result, do not full-export')
    chosen=max(values,key=lambda x:x['paired_mean']);last=[x['paired_mean'] for x in values[-3:]];selected=dict(epoch=chosen['epoch'],selection_value=chosen['paired_mean'],epoch_values=values,last_three_span=max(last)-min(last),stability_passed=max(last)-min(last)<=.05)
    once(BASE/'selection.json',selected);return selected

def export(s,selected):
    c=s.c;epoch=selected['epoch'];sets=[]
    for arm in ['fixed','quality']:
        m=make(s,arm);payload=torch.load(RUN/'epochs'/f'{epoch:02d}_{arm}.pt',map_location='cpu',weights_only=False);m.load_state_dict(payload['model']);m.strength=payload['strength']
        tasks=[('core',condition,'learned',c['development_records']) for condition in CORE]+[('graded',condition,'learned',c['monitor_records']) for condition in GRADED]
        if arm=='quality':tasks += [('ablation',condition,ablation,c['development_records']) for ablation in ['unit','mean'] for condition in ['clean','joint']]
        for kind,condition,ablation,records in tasks:
            name=f'{kind}_{arm}_{ablation}_{condition}';m.ablation=ablation;s.ctx=dict(phase='final_export',set=name,arm=arm);report_path=BASE/'exports'/(name+'.json')
            if report_path.exists():report=read(report_path)
            else:report=evaluate(s,m,arm,condition,records,BASE/'exports'/name);once(report_path,report)
            sets.append(dict(name=name,kind=kind,arm=arm,ablation=ablation,condition=condition,frames=len(records),official_AP=kind!='graded',report_sha256=sha(report_path)));status('final_export',completed_sets=len(sets),total_sets=28,selected_epoch=epoch,forward_intents=s.forwards);print(json.dumps(dict(set=name,frames=report['frames'],f1=report['scene_macro_f1'])),flush=True)
        del m;torch.cuda.empty_cache()
    once(BASE/'exports_complete.json',dict(selected_epoch=epoch,sets=sets,total_frame_entries=sum(x['frames'] for x in sets),official_AP_sets=14));need(sum(x['frames'] for x in sets)==2520,'Export scope mismatch')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','verify']);a=p.parse_args()
    with project_lock(ROOT):
        if a.mode=='freeze':freeze();return
        c=binding()
        if a.mode=='verify':print('P5 segmentation frozen inputs unchanged');return
        s=Session(c)
        try:selected=training(s);export(s,selected);binding();s.finish('complete');status('segmentation_exports_complete_waiting_evaluation')
        except BaseException as e:s.finish('failed',repr(e));status('segmentation_stage_failed',error=repr(e));raise
if __name__=='__main__':main()
