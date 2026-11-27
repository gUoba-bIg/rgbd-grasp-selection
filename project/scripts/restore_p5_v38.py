"""P5 v38: one matched independent-depth experiment, full-part finite delivery.

Reuses the frozen v37 cache decoder/auditor and accounting primitives through an
explicit process-local namespace adapter. No v37 file or result is modified.
The root owns authorization.json and all research launch decisions.
"""
import sys
import os
import json
import time
import uuid
import shutil
import argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import restore_quality_v37 as legacy
LOG=ROOT/'logs/P5/fusion_v38'
BASE=ROOT/'results/P5/fusion_v38'
RUN=ROOT/'experiments/p5_fusion_v38'
PROPOSAL=LOG/'proposal.json'
AUTH=LOG/'authorization.json'
RGB_PARENT=ROOT/'experiments/p4_full_baseline_v22r2/selected/epoch_33_rgb.pt'
Q_PARENT=ROOT/'experiments/p5_quality_gate_v23/quality_final.pt'
P4=ROOT/'results/P4/full_baseline_v22r2'
SEED=20261023
MONITORS=(80,640,720,800)
CONDITIONS=('clean','noise5','missing15','shift3','joint')
read,sha,stat,need,atomic,once=legacy.read,legacy.sha,legacy.stat,legacy.need,legacy.atomic,legacy.once
_OLD=read(ROOT/'configs/p5_quality_gate_v23.json')
_SCOPE='screen'


def config():
    # The inherited loader/decoder sees only the predeclared scene lists.
    return dict(_OLD,monitor_records=_OLD['development_records'] if _SCOPE=='full' else _OLD['monitor_records'])


def set_scope(scope):
    global _SCOPE
    need(scope in ('screen','full'),'Unknown evaluation scope')
    _SCOPE=scope
    legacy.LOG,legacy.OUT,legacy.RUN=LOG,BASE/scope,RUN
    legacy.PROPOSAL,legacy.AUTH=PROPOSAL,AUTH
    legacy.config=config


set_scope('screen')
accounting,append=legacy.accounting,legacy.append


def proposal():
    return dict(version='v38',part='P5 whole-part bounded repair',parent=RGB_PARENT.relative_to(ROOT).as_posix(),
        quality_parent=Q_PARENT.relative_to(ROOT).as_posix(),seed=SEED,
        inherited_config='configs/p5_quality_gate_v23.json',inherited_config_sha256=sha(ROOT/'configs/p5_quality_gate_v23.json'),
        input_size=[480,640],train_records=_OLD['train_records'],screen_records=_OLD['monitor_records'],full_records=_OLD['development_records'],
        architecture=dict(rgb='P4 epoch33 RGB backbone, FPN and prediction heads all frozen',
            depth='2to16_k5_s2_GN4_SiLU,16to32_k3_s2_GN8_SiLU,32to48_k3_GN8_SiLU,48to48_k1_zero_output',
            fusion='RGB_FPN + q * independent_depth_features',fixed_q=1,
            quality='frozen P5 valid*exp(-sigma/.005)*(1-.5*anomaly); observable proxy, not calibrated sensor probability',
            original_depth_and_validity_unchanged=True,normalization_after_gate=False),
        training=dict(arms=['fixed','quality'],updates_per_arm=800,total_optimizer_intents=1600,
            batch_accumulation=4,trainable='depth branch only',learning_rate=.001,weight_decay=.0001,
            AMP_initial_scale=128,overflow_retries=0,gradient_clip=1.,selection='final800, no epoch selection',
            monitor_updates=list(MONITORS),monitor_conditions=['clean','joint'],checkpoint_interval=20,
            first_two_accepted_updates_record_actual_gradients=True,recovery_replays_consume_same_budget=True),
        evaluation=dict(conditions=list(CONDITIONS),decoder=_OLD['decoder'],corruption_epoch=_OLD['quality_screen_epoch'],
            screen_frames=280,full_frames=2240,full_reuses_screen_predictions=280,monitor_frames=320,
            ablations=['unit','mean'],ablation_conditions=['clean','joint'],AP_macro='scene equally weighted; each full scene aggregates its 16 frames',
            screen_AP_calls=150,full_new_AP_calls=140,full_RGB_parent_AP_calls=0,
            RGB_parent='Reuse exact P4 heads/masks for clean; RGB unchanged across depth-only corruption',
            RGBD_descriptive='Reuse P4 full160 clean only; historical budget/architecture differ, not matched gate control'),
        screen_rule=dict(numerical_validity=True,clean_macro_F1_min=.5,noncollapsed_every_scene=True,
            trained_branch_participation=True,clean_AP_loss_vs_RGB_max=.01,
            not_required=['positive_AP_gain','six_improved_scenes','positive_quality_ablation','last_three_stability']),
        final_rule=dict(mean_degraded_AP_gain_min=.01,clean_quality_vs_fixed_AP_loss_max=.01,
            clean_both_vs_RGB_AP_loss_max=.01,clean_macro_F1_min=.5,improved_scenes_min=6,
            joint_spatial_quality_better_than_unit_and_mean=True,last_three_paired_F1_span_max=.05,
            last_three_updates=[640,720,800]),
        resources=dict(preflight_forward_intents=5,max_forward_intents=20000,forward_cap_includes_preflight=True,expected_forward_intents=17920,
            max_optimizer_intents=1600,max_official_AP_intents=290,wall_seconds=10800,
            GPU_reserved_bytes=14*1024**3,RSS_bytes=6*1024**3,output_bytes=20*1024**3,disk_reserve_bytes=60*1024**3),
        scope='Previously exposed 20-train/10-development scenes; no P6/P10/test/robot success or autonomous next method',
        stop='End after one screen failure or complete full evaluation/review/cold; no second architecture or extra training',
        parent_reference_files=parent_reference_files(),
        inherited_sources=['scripts/restore_quality_v37.py','src/checkpoints.py','src/segmentation.py','src/p5_quality_v23.py',
            'src/graspnet_frames.py','scripts/train_segmentation_baseline_v16.py','scripts/evaluate_segmentation_baseline_v16.py',
            'scripts/extract_mask_scores_v17.py','scripts/accept_mask_ap_v17.py','scripts/evaluate_mask_ap_v17.py',
            'scripts/p4_complete_evaluation_v22r2.py','scripts/mask_ap_common_v17.py','scripts/pipeline.py'])


def parent_reference_files():
    paths=[RGB_PARENT,Q_PARENT,ROOT/'configs/p5_quality_gate_v23.json',ROOT/'configs/p4_full_baseline_v22r2.json',
        ROOT/'configs/p4_complete_evaluation_v22r2.json',P4/'closeout.json',P4/'training_and_export.json',
        ROOT/'logs/P4/full_baseline_v22r2/final_readonly/result.json',
        ROOT/'logs/P5/quality_gate_v23/segmentation_freeze.json',
        ROOT/'results/P5/quality_gate_v23/quality_screen.json']
    paths+=list((P4/'exports/rgb').glob('*.json'))+list((P4/'exports/rgb').glob('*.npz'))
    paths+=list((P4/'evaluation/official').glob('rgb_scene_*.json'))
    return [p.relative_to(ROOT).as_posix() for p in sorted(set(paths))]


def binding():
    auth=read(AUTH)
    need(auth.get('frozen') is True and auth['proposal_sha256']==sha(PROPOSAL),'Missing root freeze authorization')
    need(auth.get('protected'),'Missing protected manifest')
    for path,value in auth['protected'].items():need(stat(ROOT/path)==value,'Frozen source/input changed: '+path)
    return auth


class Session(legacy.Session):
    def __init__(self,phase,preflight=False):
        super().__init__(phase,preflight)

    def guard(self):
        self.guard_save_only();self.save()
        need(self.previous+time.monotonic()-self.start < (300 if self.preflight else 10800),'Cumulative v38 time cap')
        need(self.max_rss<6*1024**3 and self.max_gpu<14*1024**3,'v38 memory cap')
        need(shutil.disk_usage(ROOT).free>=60*1024**3,'v38 disk reserve')
        need(not (LOG/'STOP').exists(),'Explicit P5 component stop requested')
        used=sum(p.stat().st_size for d in (LOG,BASE,RUN) if d.exists() for p in d.rglob('*') if p.is_file())
        need(used<20*1024**3,'v38 artifact cap')

    def intent(self,family,**values):
        self.guard()
        cap={'forward':5 if self.preflight else 20000-accounting(True)['forward']['intents'],'optimizer':0 if self.preflight else 1600,'official_AP':0 if self.preflight else 290}[family]
        need(self.counts[family]<cap,family+' budget exhausted')
        self.counts[family]+=1;cid=uuid.uuid4().hex
        append(family+'_intent',self.preflight,call_id=cid,**self.ctx,**values)
        return cid


def make_model(arm,session=None):
    torch,_=legacy.torch_setup()
    from src.modal_fusion_v38 import ModalFusionV38
    model=ModalFusionV38(arm)
    model.backbone.load_state_dict(torch.load(RGB_PARENT,map_location='cpu',weights_only=False)['model'])
    model.quality.load_state_dict(torch.load(Q_PARENT,map_location='cpu',weights_only=False)['model'])
    model.cuda()
    if session:
        session.hook(model,'fusion_wrapper')
        session.hook(model.quality,'quality_internal')
    return model


def check():
    if (BASE/'preflight.json').exists():
        need(read(BASE/'preflight.json')['model_sha256']==sha(ROOT/'src/modal_fusion_v38.py'),'Model changed after check')
        return
    need(not legacy.events(True),'Preparation already attempted; preserve evidence before any retry')
    s=Session('preflight',True)
    try:
        torch,_=legacy.torch_setup()
        from src.modal_fusion_v38 import ModalFusionV38
        from scripts.evaluate_segmentation_baseline_v16 import prepare_input
        before={p:stat(ROOT/p) for p in [RGB_PARENT.relative_to(ROOT).as_posix(),Q_PARENT.relative_to(ROOT).as_posix()]}
        parent=make_model('fixed').backbone.eval();s.hook(parent,'RGB_parent')
        record=_OLD['monitor_records'][0]
        for key,path in record['files'].items():need(sha(ROOT/_OLD['data_root']/path)==record['sha256'][key],'Precheck frame changed')
        x=torch.from_numpy(prepare_input(ROOT,_OLD,record,'rgbd'))[None].cuda();xcopy=x.clone()
        with torch.inference_mode():expected=parent(x[:,:3])
        states={};identities={};trainables={}
        for arm in ('fixed','quality'):
            model=make_model(arm,s).eval();seen=[]
            handles=[model.backbone.stem.register_forward_pre_hook(lambda m,a:seen.append(('rgb',torch.equal(a[0],x[:,:3])))),
                model.depth.register_forward_pre_hook(lambda m,a:seen.append(('depth',torch.equal(a[0],x[:,3:5]))))]
            with torch.inference_mode():actual=model(x)
            for h in handles:h.remove()
            identities[arm]=dict(RGB_parent_heads_bitwise_equal=all(torch.equal(expected[k],actual[k]) for k in expected),
                input_paths_exact=seen==[('rgb',True),('depth',True)],raw_input_unchanged=torch.equal(x,xcopy),
                frozen_RGB_and_quality=all(not p.requires_grad for module in [model.backbone,model.quality] for p in module.parameters()))
            states[arm]={k:v.cpu().clone() for k,v in model.state_dict().items()}
            trainables[arm]=[k for k,v in model.named_parameters() if v.requires_grad]
            del model,actual
        need(states['fixed'].keys()==states['quality'].keys() and all(torch.equal(v,states['quality'][k]) for k,v in states['fixed'].items()),'Paired states differ')
        need(trainables['fixed']==trainables['quality'] and all(k.startswith('depth.') for k in trainables['fixed']),'Trainable parameter fairness differs')
        need(all(all(v.values()) for v in identities.values()),'Identity or separation precheck failed')
        # Pure CPU synthetic algebra/gradient; no research data, model forward or optimizer step.
        rgb=torch.tensor([1.,2.]);depth=torch.tensor([3.,4.],requires_grad=True)
        need(torch.equal(ModalFusionV38.fuse(rgb,depth,torch.zeros(2)),rgb),'Zero gate must restore RGB')
        fused=ModalFusionV38.fuse(rgb,depth,torch.tensor([.25,.75]));fused.sum().backward()
        need(torch.equal(depth.grad,torch.tensor([.25,.75])),'Gate must scale depth derivative without altering RGB')
        # Zero output-layer initialization blocks earlier depth gradients only before first update.
        hidden=torch.tensor([2.],requires_grad=True);last=torch.tensor([0.],requires_grad=True)
        (hidden*last).sum().backward();need(hidden.grad.item()==0 and last.grad.item()==2,'Zero-projection gradient algebra differs')
        for path,value in before.items():need(stat(ROOT/path)==value,'Parent changed during preparation')
        once(BASE/'preflight.json',dict(passed=True,identities=identities,paired_state_equal=True,trainable_names=trainables['fixed'],
            model_sha256=sha(ROOT/'src/modal_fusion_v38.py'),parents=before,accounting=accounting(True),
            CPU_synthetic_gate_and_gradient_algebra=True,CPU_research_optimizer_steps=0,
            note='5 entries: RGB parent1 + wrappers2 + internal quality2; individual CNN layers are not extra model entries'))
        s.finish()
    except BaseException as error:s.finish('failed',repr(error));raise


def predict(s,model,record,condition):
    import numpy as np
    import torch
    from src.p5_quality_v23 import corrupt
    from scripts.evaluate_segmentation_baseline_v16 import prepare_input,load_labels,frame_metrics
    from scripts.extract_mask_scores_v17 import decode_with_scores
    x,_,metadata=corrupt(torch.from_numpy(prepare_input(ROOT,_OLD,record,'rgbd')),record,_OLD['quality_screen_epoch'],condition)
    s.ctx=dict(s.ctx,scene=record['scene'],frame=record['frame'],condition=condition)
    model.eval();model.record_diagnostics=True
    with torch.inference_mode():h=model(x[None].cuda())
    heads=dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=h['centers'][0,0].sigmoid().cpu().numpy(),offsets=h['offsets'][0].cpu().numpy())
    need(all(v.dtype==np.float32 and np.isfinite(v).all() for v in heads.values()),'Invalid FP32 prediction')
    pred,fg,scores,_=decode_with_scores(heads,**_OLD['decoder'])
    arrays=dict(**heads,instances=pred,foreground=fg,gate=h['effective_gate'][0,0].cpu().numpy())
    diag={k:v.detach().cpu().tolist() for k,v in h['fusion_diagnostics'].items()}
    return arrays,dict(scene=record['scene'],frame=record['frame'],scores=scores,metrics=frame_metrics(pred,fg,load_labels(ROOT,_OLD,record)),corruption=metadata,fusion_diagnostics=diag)


def save_case(path,arrays,row):
    import numpy as np
    path=Path(path);cache=path.with_suffix('.npz')
    need(not path.exists() and not cache.exists(),'Uncommitted/existing case; preserve before recovery')
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.open('xb') as handle:np.savez_compressed(handle,**arrays)
    committed=dict(row);committed['cache_sha256']=sha(cache)
    once(path,committed)
    return read(path)


def monitor(s,model,arm,step):
    path=BASE/'monitor'/f'{arm}_{step:04d}.json'
    if path.exists():return
    result={}
    for condition in ('clean','joint'):
        rows=[]
        for rec in _OLD['monitor_records']:
            receipt=BASE/'monitor_cache'/f'{arm}_{step:04d}_{condition}'/f"scene_{rec['scene']:04d}_{rec['frame']}.json"
            if receipt.exists():
                row=read(receipt);need(sha(receipt.with_suffix('.npz'))==row['cache_sha256'],'Monitor cache changed')
            else:
                s.ctx=dict(phase='monitor',arm=arm,update=step)
                arrays,row=predict(s,model,rec,condition);row=save_case(receipt,arrays,row)
            rows.append(row)
        result[condition]=legacy.aggregate_f1(rows)
    once(path,result)


def immutable_parent_states(model):
    import torch
    for module,path in [(model.backbone,RGB_PARENT),(model.quality,Q_PARENT)]:
        expected=torch.load(path,map_location='cpu',weights_only=False)['model']
        need(all(torch.equal(v.detach().cpu(),expected[k]) for k,v in module.state_dict().items()),'Frozen RGB/quality parameters changed')


def train_arm(s,arm):
    import torch
    from src.checkpoints import CheckpointStore,fingerprint
    from src.p5_quality_v23 import Frames
    from scripts import train_segmentation_baseline_v16 as trainer
    final=RUN/arm/'final.json'
    if final.exists():need(sha(RUN/arm/'final.pt')==read(final)['sha256'],'Final weights changed');return
    model=make_model(arm,s)
    initial={k:v.detach().cpu().clone() for k,v in model.depth.state_dict().items()}
    optimizer=torch.optim.AdamW(model.depth.parameters(),lr=.001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    store=CheckpointStore(RUN/arm/'checkpoints',fingerprint(dict(proposal=sha(PROPOSAL),arm=arm)),_OLD['data_fingerprint'],keep_last=2)
    cursor=dict(epoch=0,next_batch=0,global_step=0)
    if (RUN/arm/'checkpoints/latest.json').exists():cursor=store.load_latest(model,optimizer,scaler=scaler)['cursor']
    epoch,pos,step=cursor['epoch'],cursor['next_batch'],cursor['global_step']
    dataset=Frames(ROOT/_OLD['data_root'],_OLD['train_records'],size=_OLD['input_size'],seed=SEED,augment=True)
    cfg=dict(protocol=dict(gradient_clip_norm=1.),device='cuda',effective_amp=True,amp_max_overflow_retries=1)
    if step in MONITORS:monitor(s,model,arm,step)
    while step<800:
        s.ctx=dict(phase='training',arm=arm,update=step+1,epoch=epoch+1)
        selected=dataset.epoch_order(epoch)[pos:pos+4];model.train();model.record_diagnostics=False
        gradients={};handles=[]
        if step<2:
            for name,param in [('first',model.depth[0].weight),('projection',model.depth[-1].weight)]:
                def collect(gradient,name=name):gradients[name]=gradients.get(name,0.)+float(gradient.detach().float().abs().sum().cpu())
                handles.append(param.register_hook(collect))
        cid=s.intent('optimizer')
        loss,parts,norm,retries=trainer.train_update(model,optimizer,scaler,dataset,selected,epoch,cfg)
        for handle in handles:handle.remove()
        s.returned('optimizer',cid,loss=loss,components=parts,gradient_norm=norm,overflow_retries=retries)
        step+=1;pos+=len(selected)
        if step<=2:
            evidence=dict(update=step,scaled_accumulated_absolute_gradients=gradients,projection_norm=float(model.depth[-1].weight.detach().float().norm().cpu()),
                note='First-step earlier depth gradient zero follows zero projection; second accepted update checks real propagation')
            once(OUT_PATH(arm,step),evidence)
            if step==2:need(gradients.get('first',0)>0 and gradients.get('projection',0)>0,'New depth path has no gradient after first accepted update')
        if pos==len(dataset):epoch+=1;pos=0
        if step%20==0:
            store.save(model,optimizer,epoch,pos,step,True,scaler=scaler)
            legacy.status('training',arm=arm,logical_updates=step,optimizer=accounting()['optimizer'])
            if step%80==0:print(json.dumps(dict(arm=arm,updates=step)),flush=True)
        if step in MONITORS:monitor(s,model,arm,step)
    immutable_parent_states(model)
    changed={k:float((v.detach().cpu()-initial[k]).float().norm()) for k,v in model.depth.state_dict().items()}
    need(changed['0.weight']>0 and changed['9.weight']>0,'Depth parameters did not update')
    payload=dict(model=model.state_dict(),arm=arm,logical_updates=800,proposal_sha256=sha(PROPOSAL))
    path=RUN/arm/'final.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        need(saved['proposal_sha256']==sha(PROPOSAL) and saved['logical_updates']==800 and saved['arm']==arm,'Uncommitted final identity differs')
        need(all(torch.equal(v.detach().cpu(),saved['model'][k]) for k,v in model.state_dict().items()),'Uncommitted final tensor differs')
    else:
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');torch.save(payload,temporary);os.replace(temporary,path)
    once(final,dict(sha256=sha(path),logical_updates=800,proposal_sha256=sha(PROPOSAL),depth_change_norms=changed,RGB_quality_unchanged=True))
    del model,optimizer,scaler;torch.cuda.empty_cache()


def OUT_PATH(arm,step):return BASE/'participation'/f'{arm}_update_{step}.json'


def tasks():return legacy.task_sets()+[dict(name='baseline_rgb_clean',arm='fixed',ablation='unit',condition='clean',parent_RGB=True)]


def parent_cache(scope,record,item):
    import numpy as np
    original=P4/'exports/rgb'/f"scene_{record['scene']:04d}_{record['frame']}.json"
    row=read(original);cache=original.with_suffix('.npz');need(sha(cache)==row['cache_sha256'],'Parent RGB cache changed')
    with np.load(cache,allow_pickle=False) as values:
        arrays={k:values[k].copy() for k in values.files}
    arrays['gate']=np.ones((480,640),np.float32)
    receipt=BASE/scope/'exports'/item['name']/original.name
    return save_case(receipt,arrays,dict(**row,source_parent_receipt=original.relative_to(ROOT).as_posix(),source_parent_sha256=sha(original),source_parent_cache_sha256=sha(cache),set=item))



def export_binding(scope):
    records=_OLD['monitor_records'] if scope=='screen' else _OLD['development_records']
    return dict(proposal_sha256=sha(PROPOSAL),scope=scope,records=records,decoder=_OLD['decoder'],
        corruption_epoch=_OLD['quality_screen_epoch'],conditions=list(CONDITIONS),tasks=tasks(),
        finals={path.relative_to(ROOT).as_posix():stat(path) for arm in ('fixed','quality') for path in (RUN/arm/'final.pt',RUN/arm/'final.json')},
        RGB_parent=stat(RGB_PARENT),quality_parent=stat(Q_PARENT),
        inherited_config_sha256=sha(ROOT/'configs/p5_quality_gate_v23.json'))


def verify_export_seal(scope):
    seal=read(BASE/scope/'exports_complete.json')
    need(seal['source_binding']==export_binding(scope),'Export checkpoint/protocol/decoder/record identity differs')
    expected=14*(20 if scope=='screen' else 160)
    need(seal['method_frames']==expected and seal['parent_cached_frames']==expected//14,'Export declared scope differs')
    current={p.relative_to(ROOT).as_posix():stat(p) for p in (BASE/scope/'exports').rglob('*') if p.is_file()}
    need(current==seal['inventory'],'Sealed export inventory or bytes changed')
    return seal


def export(s,scope):
    import torch
    set_scope(scope)
    if scope=='full':
        need(read(BASE/'screen/decision.json')['eligible_full_evaluation'],'Screen validity failed; full run forbidden')
        verify_export_seal('screen')
    records=config()['monitor_records'];scope_out=BASE/scope
    for arm in ('fixed','quality'):
        final=read(RUN/arm/'final.json');need(final['logical_updates']==800 and sha(RUN/arm/'final.pt')==final['sha256'],'Final800 missing/changed')
        model=make_model(arm,s);model.load_state_dict(torch.load(RUN/arm/'final.pt',map_location='cpu',weights_only=False)['model'])
        for item in [x for x in legacy.task_sets() if x['arm']==arm]:
            model.ablation=item['ablation']
            for rec in records:
                name=f"scene_{rec['scene']:04d}_{rec['frame']}.json";receipt=scope_out/'exports'/item['name']/name
                if receipt.exists():need(sha(receipt.with_suffix('.npz'))==read(receipt)['cache_sha256'],'Export changed');continue
                inherited=BASE/'screen/exports'/item['name']/name
                if scope=='full' and inherited.exists():
                    row=read(inherited);need(sha(inherited.with_suffix('.npz'))==row['cache_sha256'],'Screen cache changed')
                    receipt.parent.mkdir(parents=True,exist_ok=True)
                    destination=receipt.with_suffix('.npz')
                    if destination.exists():
                        need(sha(destination)==row['cache_sha256'],'Interrupted copy differs; preserve it for review')
                    else:shutil.copyfile(inherited.with_suffix('.npz'),destination)
                    once(receipt,row)
                else:
                    s.ctx=dict(phase='export-'+scope,arm=arm,set=item['name']);arrays,row=predict(s,model,rec,item['condition']);save_case(receipt,arrays,dict(**row,set=item))
        immutable_parent_states(model)
        del model;torch.cuda.empty_cache()
    item=tasks()[-1]
    for rec in records:
        path=scope_out/'exports'/item['name']/f"scene_{rec['scene']:04d}_{rec['frame']}.json"
        if path.exists():need(sha(path.with_suffix('.npz'))==read(path)['cache_sha256'],'Parent-derived cache changed')
        else:parent_cache(scope,rec,item)
    inventory={p.relative_to(ROOT).as_posix():stat(p) for p in (scope_out/'exports').rglob('*') if p.is_file()}
    once(scope_out/'exports_complete.json',dict(scope=scope,method_frames=14*len(records),parent_cached_frames=len(records),inventory=inventory,source_binding=export_binding(scope)))


def stable_values():
    import numpy as np
    values=[float(np.mean([read(BASE/'monitor'/f'{arm}_{step:04d}.json')[condition]['scene_macro_f1'] for arm in ('fixed','quality') for condition in ('clean','joint')])) for step in (640,720,800)]
    return values,max(values)-min(values)


def decide(metrics,scope):
    import numpy as np
    values,span=stable_values();rgb=metrics['baseline_rgb_clean']['macro_AP']
    participation={arm:read(RUN/arm/'final.json') for arm in ('fixed','quality')}
    valid=dict(clean_F1=all(metrics[f'core_{a}_clean']['macro_F1']>=.5 for a in ('fixed','quality')),
        noncollapsed=all(v['predicted_instances']>0 for a in ('fixed','quality') for v in metrics[f'core_{a}_clean']['scenes'].values()),
        branch_participation=all(p['depth_change_norms']['0.weight']>0 and p['depth_change_norms']['9.weight']>0 and p['RGB_quality_unchanged'] for p in participation.values()),
        clean_both_vs_RGB=all(rgb-metrics[f'core_{a}_clean']['macro_AP']<=.01 for a in ('fixed','quality')))
    gain=float(np.mean([metrics[f'core_quality_{d}']['macro_AP']-metrics[f'core_fixed_{d}']['macro_AP'] for d in CONDITIONS[1:]]))
    scenes={sc:float(np.mean([metrics[f'core_quality_{d}']['scenes'][sc]['AP']-metrics[f'core_fixed_{d}']['scenes'][sc]['AP'] for d in CONDITIONS[1:]])) for sc in metrics['core_fixed_clean']['scenes']}
    ablation={a:metrics['core_quality_joint']['macro_AP']-metrics[f'ablation_{a}_joint']['macro_AP'] for a in ('unit','mean')}
    scientific=dict(**valid,stability=span<=.05,degraded_AP_gain=gain>=.01,
        clean_quality_vs_fixed=metrics['core_fixed_clean']['macro_AP']-metrics['core_quality_clean']['macro_AP']<=.01,
        improved_scenes=sum(x>0 for x in scenes.values())>=6,spatial_quality=all(x>0 for x in ablation.values()))
    p4=read(P4/'closeout.json')
    return dict(scope=scope,validity_gates=valid,eligible_full_evaluation=all(valid.values()),scientific_gates=scientific,
        scientific_usable=scope=='full' and all(scientific.values()),scientific_adoption_supported=scope=='full' and all(scientific.values()),full_original_P5_scientific_passed=scope=='full' and all(scientific.values()),
        mean_degraded_mask_AP_gain=gain,clean_quality_vs_fixed_loss=metrics['core_fixed_clean']['macro_AP']-metrics['core_quality_clean']['macro_AP'],
        clean_both_vs_RGB_loss={a:rgb-metrics[f'core_{a}_clean']['macro_AP'] for a in ('fixed','quality')},
        RGB_parent_macro_AP=rgb,RGB_parent_depth_condition_invariant=True,by_scene_gain=scenes,joint_ablation_gain=ablation,
        last_three_paired_F1_values=values,last_three_span=span,
        historical_RGBD_clean=dict(AP=p4['arms']['rgbd']['scene_macro']['AP'],scope='P4 full160 clean; descriptive only, different training/architecture from v38 matched arms'),
        interpretation='Scientific adoption is assessed only on full160; screen scientific numbers are descriptive and do not control expansion',
        no_P6_P10_or_next_method_authorized=True)


def evaluate(s,scope,independent=False):
    import numpy as np
    from scripts import p4_complete_evaluation_v22r2 as inherited
    from scripts import evaluate_mask_ap_v17 as official
    need('torch' not in sys.modules,'CPU evaluate/review must run in a separate process')
    set_scope(scope);scope_out=BASE/scope;seal=verify_export_seal(scope)
    if scope=='full':verify_export_seal('screen')
    metrics={}
    for item in tasks():
        per_scene={}
        for scene in sorted({r['scene'] for r in config()['monitor_records']}):
            s.ctx=dict(phase=('review-' if independent else 'AP-')+scope,set=item['name'],scene=scene);s.guard()
            inputs,matches,diag=legacy.scene_inputs(item,scene,independent)
            path=scope_out/'AP'/item['name']/f'scene_{scene:04d}.json'
            if independent:need(path.exists(),'Independent review cannot create missing AP or source artifacts')
            if path.exists():ap=read(path)
            elif scope=='full' and item.get('parent_RGB'):
                original=P4/'evaluation/official'/f'rgb_scene_{scene:04d}.json';ap=read(original)
                once(path,ap);once(path.with_suffix('.source.json'),dict(source=original.relative_to(ROOT).as_posix(),source_sha256=sha(original),no_new_AP=True))
            else:
                need(not independent,'Independent review found missing AP')
                cid=s.intent('official_AP');ap=official.official_evaluation(inputs);once(path,ap)
                s.returned('official_AP',cid,path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
            if scope=='full' and item.get('parent_RGB'):
                original=P4/'evaluation/official'/f'rgb_scene_{scene:04d}.json'
                expected_source=dict(source=original.relative_to(ROOT).as_posix(),source_sha256=sha(original),no_new_AP=True)
                source_receipt=path.with_suffix('.source.json')
                if independent:need(source_receipt.exists(),'Missing parent AP source receipt in read-only review')
                elif not source_receipt.exists():
                    need(ap==read(original),'Interrupted parent AP copy differs from original')
                    once(source_receipt,expected_source)
                need(read(source_receipt)==expected_source,'Parent AP source binding differs')
            inherited.audit_ap(ap,matches)
            per_scene[str(scene)]=dict(**inherited.primary(ap),**diag['scenes'][0])
        result=dict(scenes=per_scene,macro_AP=float(np.mean([x['AP'] for x in per_scene.values()])),macro_F1=float(np.mean([x['f1'] for x in per_scene.values()])))
        metrics[item['name']]=result;path=scope_out/'metrics'/(item['name']+'.json')
        if independent:need(read(path)==result,'Independent aggregate differs')
        else:once(path,result)
    result=decide(metrics,scope)
    if independent:need(read(scope_out/'decision.json')==result,'Independent decision differs')
    else:once(scope_out/'decision.json',result)
    return result


def review_monitors(s):
    import numpy as np
    from PIL import Image
    from scripts import accept_mask_ap_v17 as reference
    from scripts import p4_complete_evaluation_v22r2 as inherited
    count=0
    for arm in ('fixed','quality'):
        for step in MONITORS:
            expected=read(BASE/'monitor'/f'{arm}_{step:04d}.json')
            for condition in ('clean','joint'):
                rows=[]
                for rec in _OLD['monitor_records']:
                    s.guard();receipt=BASE/'monitor_cache'/f'{arm}_{step:04d}_{condition}'/f"scene_{rec['scene']:04d}_{rec['frame']}.json"
                    row=read(receipt);need(sha(receipt.with_suffix('.npz'))==row['cache_sha256'],'Monitor bytes changed')
                    with np.load(receipt.with_suffix('.npz'),allow_pickle=False) as values:
                        rebuilt=reference.independent_decode(**{k:values[k] for k in ('foreground_probability','center_probability','offsets')},decoder=_OLD['decoder'])
                        need(np.array_equal(rebuilt['instances'],values['instances']) and np.array_equal(rebuilt['foreground'],values['foreground']) and rebuilt['retained_instances']==row['scores'],'Monitor decoding differs')
                    with Image.open(ROOT/_OLD['data_root']/rec['files']['label']) as image:gt=inherited.indexed_resize(np.asarray(image),480,640)
                    inherited.frame_check(rebuilt['instances'],rebuilt['foreground'],gt,row['metrics']);rows.append(row);count+=1
                need(legacy.aggregate_f1(rows)==expected[condition],'Monitor arithmetic differs')
    need(count==320,'Monitor scope incomplete');return count


def snapshot():
    # The root controller owns this one closeout file; it seals it separately.
    # All component artifacts and every other unexpected path remain checked.
    return {p.relative_to(ROOT).as_posix():stat(p) for directory in (BASE,RUN) for p in directory.rglob('*') if p.is_file() and p != BASE/'closeout.json'}


def review(s):
    before=snapshot();calls=accounting();monitor_count=review_monitors(s)
    screen=evaluate(s,'screen',True);scopes=['screen'];outcome=screen
    if screen['eligible_full_evaluation']:
        outcome=evaluate(s,'full',True);scopes.append('full')
    else:need(not (BASE/'full').exists(),'Screen failure unexpectedly accessed full evaluation')
    need(snapshot()==before,'Readonly review changed output inventory or bytes')
    need(accounting()==calls,'Readonly review unexpectedly invoked research')
    once(LOG/'review.json',dict(passed=True,engineering_passed=True,scientific_usable=outcome['scientific_usable'],
        full_original_P5_scientific_passed=outcome['full_original_P5_scientific_passed'],completed_scopes=scopes,
        verified_monitor_frames=monitor_count,new_forward=0,new_optimizer=0,new_official_AP=0,accounting=calls,reviewed_outputs=before))


def cold():
    result=read(LOG/'review.json');need(result['passed'],'Independent review missing');before=accounting()
    need(before['optimizer']['known_returns']==1600 and not before['optimizer']['unknown_ids'],'Require1600 confirmed optimization returns')
    for scope in result['completed_scopes']:verify_export_seal(scope)
    need(snapshot()==result['reviewed_outputs'],'Cold artifact inventory or bytes changed')
    need(accounting()==before,'Cold unexpectedly invoked research')
    once(LOG/'cold.json',dict(passed=True,engineering_passed=True,scientific_usable=result['scientific_usable'],
        full_original_P5_scientific_passed=result['full_original_P5_scientific_passed'],completed_scopes=result['completed_scopes'],
        new_forward=0,new_optimizer=0,new_official_AP=0,accounting=before,
        accounting_fully_observed=all(not v['unknown_ids'] for v in before.values()),unknown_calls_remain_unknown=True,
        end_of_authorized_P5_part=True,next_research_started=False))
    legacy.status('completed_stopped',engineering_passed=True,scientific_usable=result['scientific_usable'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','check','train','export-screen','evaluate-screen','export-full','evaluate-full','review','cold'))
    args=parser.parse_args()
    from scripts.pipeline import project_lock
    with project_lock(ROOT):
        if args.mode=='prepare':once(PROPOSAL,proposal());return
        if args.mode=='check':check();return
        binding();need(read(BASE/'preflight.json')['passed'],'Preparation not accepted')
        if args.mode=='cold':cold();return
        s=Session(args.mode)
        try:
            if args.mode=='train':
                set_scope('screen')
                for arm in ('fixed','quality'):train_arm(s,arm)
                once(BASE/'training_complete.json',dict(updates_per_arm=800,arms={a:read(RUN/a/'final.json') for a in ('fixed','quality')}))
            elif args.mode.startswith('export-'):export(s,args.mode.split('-')[1])
            elif args.mode.startswith('evaluate-'):
                scope=args.mode.split('-')[1]
                if scope=='full':need(read(BASE/'screen/decision.json')['eligible_full_evaluation'],'Full evaluation not eligible')
                evaluate(s,scope)
            else:review(s)
            binding();s.finish()
        except BaseException as error:s.finish('failed',repr(error));legacy.status('failed',mode=args.mode,error=repr(error),accounting=accounting());raise


if __name__=='__main__':main()
