"""Bounded v37 quality-residual experiment. Never mutates earlier frozen versions.

prepare/check do not authorize research. Root must write authorization.json with
frozen=true, proposal_sha256 and protected {relative_path: sha256/bytes/mtime_ns}.
Commands train/export/evaluate/review/cold then enforce that receipt.
"""
import os
import sys
import json
import uuid
import time
import hashlib
import shutil
import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
LOG = ROOT / 'logs/Restore/v37/quality'
OUT = ROOT / 'results/Restore/v37/quality'
RUN = ROOT / 'experiments/restore_quality_v37'
PROPOSAL = LOG / 'proposal.json'
AUTH = LOG / 'authorization.json'
OLD_CONFIG = ROOT / 'configs/p5_quality_gate_v23.json'
Q_PARENT = ROOT / 'experiments/p5_quality_gate_v23/quality_final.pt'
SEED = 20261023
CONDITIONS = ('clean', 'noise5', 'missing15', 'shift3', 'joint')


def now():
    return datetime.now().astimezone().isoformat()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stat(path):
    path = Path(path)
    s = path.stat()
    return dict(sha256=sha(path), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    deadline=time.monotonic()+2.0
    while True:
        try:
            os.replace(temporary,path)
            break
        except OSError as error:
            if getattr(error,'winerror',None) not in (5,32) or time.monotonic()>=deadline:
                raise
            time.sleep(.05)


def once(path, value):
    if Path(path).exists():
        need(read(path) == value, 'Immutable result differs: ' + str(path))
    else:
        atomic(path, value)


def events(preflight=False):
    path = LOG / ('preflight_calls.jsonl' if preflight else 'calls.jsonl')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def append(kind, preflight=False, **values):
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / ('preflight_calls.jsonl' if preflight else 'calls.jsonl')).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(at=now(), kind=kind, **values), allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def accounting(preflight=False):
    rows = events(preflight)
    result = {}
    for family in ('forward', 'optimizer', 'official_AP'):
        intent_ids = [r['call_id'] for r in rows if r['kind'] == family + '_intent']
        return_ids = [r['call_id'] for r in rows if r['kind'] == family + '_returned']
        need(len(intent_ids)==len(set(intent_ids)) and len(return_ids)==len(set(return_ids)), 'Duplicate call identity in '+family)
        intended,returned=set(intent_ids),set(return_ids)
        need(returned <= intended, 'Return without intent')
        result[family] = dict(intents=len(intended), known_returns=len(returned), unknown_ids=sorted(intended-returned))
    return result


def config():
    old = read(OLD_CONFIG)
    need(len(old['train_records']) == 320 and len(old['monitor_records']) == 20, 'Parent cohort differs')
    need(len({x['scene'] for x in old['monitor_records']}) == 10, 'Expected ten exposed development scenes')
    return old


def proposal_body():
    old = config()
    return dict(version='v37', purpose='Restore original quality-aware segmentation hypothesis; bounded exposed-development experiment',
        parent=old['parent'], quality_parent=Q_PARENT.relative_to(ROOT).as_posix(), old_config=OLD_CONFIG.relative_to(ROOT).as_posix(),
        old_config_sha256=sha(OLD_CONFIG), input_size=old['input_size'], seed=SEED,
        train_records=old['train_records'], evaluation_records=old['monitor_records'],
        conditions=list(CONDITIONS), decoder=old['decoder'], corruption_epoch=old['quality_screen_epoch'],
        architecture=dict(adapter='Conv48to16_k3_GroupNorm4_SiLU_Conv16to48_k1', last_layer_zero_initialized=True,
            residual='F + 0.25 * reliability * tanh(adapter(F))', feature_location='FPN combined finest map before refine',
            quality='Frozen P5 valid * exp(-sigma/.005) * (1-.5*anomaly)', fixed_control='same architecture, reliability=1',
            input_depth_validity_unchanged=True),
        training=dict(arms=['fixed','quality'], updates_per_arm=400, total_optimizer_intents=800, batch_accumulation=4,
            learning_rate=1e-5, weight_decay=1e-4, gradient_clip=1., amp_initial_scale=128,
            amp_overflow_retries=0, monitor_updates=[80,240,320,400], checkpoint_every_updates=20,
            selection='final400 only; no epoch selection', recovery_replays_consume_same_800_intent_budget=True),
        evaluation=dict(core_frames=200, ablation_frames=80, total_frames=280, monitor_cached_frames=320,
            ablation_conditions=['clean','joint'], ablations=['unit','mean'],
            AP_unit='scene two-frame aggregate; scene equal-weight macro', official_AP_calls=140, pooled_AP=False,
            readonly='independent cached decoding, matching, precision/recall/scores arrays; zero new forward/AP'),
        gates=dict(mean_degraded_mask_AP_gain_min=.01, clean_mask_AP_loss_max=.01,
            improved_scenes_min=6, clean_F1_validity_min=.5, joint_quality_ablation_strictly_better=True,
            last_three_paired_macro_F1_span_max=.05,last_three_updates=[240,320,400]),
        scientific_scope='Only 20 exposed development frames; passing is a development repair result, not full160-frame adoption, independent generalization or grasp success',
        resources=dict(max_forward_intents=8000, expected_forward_intents=7600, preflight_forward_intents=5,
            max_optimizer_intents=800, max_official_AP_intents=140, wall_seconds=7200,
            gpu_bytes=14*1024**3, rss_bytes=6*1024**3, output_bytes=16*1024**3, disk_reserve_bytes=60*1024**3),
        budget_evidence=dict(P5_r1_400_training_monitor_export_seconds=694.156,
            P5_1600_update_call_seconds=1269.279, P5_2520_evaluation_seconds=564.829,
            P5_2520_review_seconds=508.422, projected_two_r1_plus_scaled_evaluation_seconds=1507.5621,
            approximate_planning_seconds_range=[1500,3000], hard_cap_seconds=7200,
            note='Planning from recorded old runtime, not a completion-time guarantee; new checkpoint and residual overhead may differ'),
        inherited_sources=['src/segmentation.py','src/p5_quality_v23.py','src/graspnet_frames.py','src/checkpoints.py',
            'scripts/train_segmentation_baseline_v16.py','scripts/evaluate_segmentation_baseline_v16.py',
            'scripts/extract_mask_scores_v17.py','scripts/accept_mask_ap_v17.py','scripts/evaluate_mask_ap_v17.py',
            'scripts/p4_complete_evaluation_v22r2.py','scripts/mask_ap_common_v17.py','scripts/pipeline.py'],
        root_authorization_required=True)


def binding():
    proposal = read(PROPOSAL)
    auth = read(AUTH)
    need(auth.get('frozen') is True and auth['proposal_sha256'] == sha(PROPOSAL), 'Root freeze authorization missing or changed')
    need(auth.get('protected'), 'Protected-source/input freeze is required')
    for path, expected in auth['protected'].items():
        need(stat(ROOT / path) == expected, 'Frozen dependency changed: ' + path)
    need(sha(OLD_CONFIG) == proposal['old_config_sha256'], 'Parent protocol changed')
    return proposal


def status(phase, **values):
    atomic(LOG / 'state.json', dict(at=now(), phase=phase, **values))


class Session:
    def __init__(self, phase, preflight=False):
        self.phase, self.preflight = phase, preflight
        self.start = time.monotonic()
        self.uid = uuid.uuid4().hex
        self.usage = LOG / (('preflight_usage_' if preflight else 'usage_') + self.uid + '.json')
        self.previous = sum(read(p)['seconds'] for p in LOG.glob('preflight_usage_*.json' if preflight else 'usage_*.json'))
        if not preflight:
            self.previous += sum(read(p)['seconds'] for p in LOG.glob('preflight_usage_*.json'))
        self.counts = {k:v['intents'] for k,v in accounting(preflight).items()}
        self.ctx = dict(phase=phase)
        self.hooks = []
        self.max_rss = 0
        self.max_gpu = 0
        self.finish_state = 'running'
        self.save()

    def save(self, error=None):
        atomic(self.usage, dict(at=now(), seconds=time.monotonic()-self.start, phase=self.phase,
            state=self.finish_state, error=error, peak_RSS_bytes=self.max_rss, peak_GPU_reserved_bytes=self.max_gpu))

    def guard(self):
        from scripts.evaluate_segmentation_baseline_v16 import peak_rss
        self.max_rss = max(self.max_rss, peak_rss())
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            self.max_gpu = max(self.max_gpu, torch.cuda.max_memory_reserved())
        self.save()
        need(self.previous + time.monotonic()-self.start < (300 if self.preflight else 7200), 'Cumulative v37 time cap reached')
        need(self.max_rss < 6*1024**3 and self.max_gpu < 14*1024**3, 'Memory cap reached')
        need(shutil.disk_usage(ROOT).free >= 60*1024**3, 'Disk reserve reached')
        need(not (LOG/'STOP').exists(), 'Explicit quality-component stop marker exists')
        used = sum(p.stat().st_size for d in (LOG,OUT,RUN) if d.exists() for p in d.rglob('*') if p.is_file())
        need(used < 16*1024**3, 'v37 output cap reached')

    def intent(self, family, **values):
        self.guard()
        limit = {'forward':5 if self.preflight else 8000-accounting(True)['forward']['intents'],'optimizer':0 if self.preflight else 800,'official_AP':0 if self.preflight else 140}[family]
        need(self.counts[family] < limit, family + ' intention budget exhausted')
        self.counts[family] += 1
        cid = uuid.uuid4().hex
        append(family+'_intent', self.preflight, call_id=cid, **self.ctx, **values)
        return cid

    def returned(self, family, cid, **values):
        append(family+'_returned', self.preflight, call_id=cid, **self.ctx, **values)

    def hook(self, model, network):
        pending = []
        def before(module, inputs):
            pending.append(self.intent('forward', network=network))
        def after(module, inputs, output):
            import torch
            torch.cuda.synchronize()
            self.returned('forward', pending.pop(), network=network)
        self.hooks += [model.register_forward_pre_hook(before),model.register_forward_hook(after)]

    def finish(self, state='complete', error=None):
        self.finish_state = state
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.guard_save_only()
        self.save(error)

    def guard_save_only(self):
        from scripts.evaluate_segmentation_baseline_v16 import peak_rss
        self.max_rss = max(self.max_rss, peak_rss())
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            self.max_gpu = max(self.max_gpu, torch.cuda.max_memory_reserved())


def torch_setup():
    import torch
    from scripts import train_segmentation_baseline_v16 as trainer
    need(trainer.runtime_identity() == config()['runtime'], 'Inherited GPU runtime differs')
    trainer.reset_random_streams(SEED, 'cuda')
    return torch, trainer


def make_model(arm, session=None):
    torch, trainer = torch_setup()
    from src.quality_residual_v37 import QualityResidualV37
    model = QualityResidualV37(arm)
    model.backbone.load_state_dict(torch.load(ROOT/config()['parent'], map_location='cpu', weights_only=False)['model'])
    model.quality.load_state_dict(torch.load(Q_PARENT, map_location='cpu', weights_only=False)['model'])
    model.cuda()
    if session:
        session.hook(model, 'segmentation_wrapper')
        session.hook(model.quality, 'quality_internal')
    return model


def preflight():
    if (OUT/'preflight.json').exists():
        result = read(OUT/'preflight.json')
        need(result['source_sha256'] == sha(ROOT/'src/quality_residual_v37.py'), 'Model changed after preflight')
        print(json.dumps(result)); return
    need(not events(True), 'Preflight was already attempted; retain unknown calls and review before retry')
    session = Session('preflight', True)
    try:
        torch, _ = torch_setup()
        from scripts.evaluate_segmentation_baseline_v16 import prepare_input
        old = config()
        need(stat(ROOT/old['parent'])==old['protected'][old['parent']], 'P4 parent differs from original frozen identity')
        qrel=Q_PARENT.relative_to(ROOT).as_posix()
        qfreeze=read(ROOT/'logs/P5/quality_gate_v23/segmentation_freeze.json')
        need(stat(Q_PARENT)==qfreeze['protected'][qrel], 'P5 quality parent differs from original frozen identity')
        for kind,relative in old['monitor_records'][0]['files'].items():
            need(sha(ROOT/old['data_root']/relative)==old['monitor_records'][0]['sha256'][kind], 'Preparation sample changed')
        parent = make_model('fixed').backbone.eval()
        session.hook(parent, 'parent_backbone')
        raw = prepare_input(ROOT, old, old['monitor_records'][0], 'rgbd')
        x = torch.from_numpy(raw)[None].cuda()
        before = x.clone()
        with torch.inference_mode():
            base = parent(x)
        identities, parameters, snapshots = {}, {}, {}
        for arm in ('fixed','quality'):
            model = make_model(arm, session).eval()
            snapshots[arm] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            parameters[arm] = sum(p.numel() for p in model.parameters())
            stem_same = []
            hook = model.backbone.stem.register_forward_pre_hook(lambda _m,args: stem_same.append(torch.equal(args[0],x)))
            with torch.inference_mode():
                value = model(x)
            hook.remove()
            identities[arm] = dict(heads_bitwise_equal=all(torch.equal(base[k],value[k]) for k in base),
                zero_residual=float(value['residual_max_abs'].cpu()) == 0., raw_stem_input_same=all(stem_same),
                frozen_quality=all(not p.requires_grad for p in model.quality.parameters()))
            del model, value
        equal = all(torch.equal(v,snapshots['quality'][k]) for k,v in snapshots['fixed'].items())
        need(equal and parameters['fixed'] == parameters['quality'], 'Paired initialization or capacity differs')
        need(all(all(v.values()) for v in identities.values()) and torch.equal(x,before), 'Identity or input preservation failed')
        result = dict(passed=True, identities=identities, paired_state_equal=equal, parameter_counts=parameters,
            source_sha256=sha(ROOT/'src/quality_residual_v37.py'), parent=stat(ROOT/old['parent']), quality_parent=stat(Q_PARENT),
            accounting=accounting(True), model_entry_policy='1 original backbone + 2 wrapper + 2 internal quality entries = 5; internal stage layers are not separately claimed')
        once(OUT/'preflight.json',result)
        session.finish()
        print(json.dumps(result))
    except BaseException as error:
        session.finish('failed',repr(error)); raise


def train_arm(session, arm):
    import torch
    from src.checkpoints import CheckpointStore, fingerprint
    from src.p5_quality_v23 import Frames
    from scripts import train_segmentation_baseline_v16 as trainer
    old = config()
    final = RUN/arm/'final.json'
    if final.exists():
        need(sha(RUN/arm/'final.pt') == read(final)['sha256'], 'Final checkpoint changed'); return
    model = make_model(arm, session)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    store = CheckpointStore(RUN/arm/'checkpoints', fingerprint(dict(proposal=sha(PROPOSAL),arm=arm)), old['data_fingerprint'], keep_last=2)
    cursor = dict(epoch=0,next_batch=0,global_step=0)
    if (RUN/arm/'checkpoints/latest.json').exists():
        cursor = store.load_latest(model,optimizer,scaler=scaler)['cursor']
    dataset = Frames(ROOT/old['data_root'],old['train_records'],size=old['input_size'],seed=SEED,augment=True)
    epoch,pos,step = cursor['epoch'],cursor['next_batch'],cursor['global_step']
    if step in (80,240,320,400):
        monitor(session,model,arm,step)
    train_cfg = dict(protocol=dict(gradient_clip_norm=1.),device='cuda',effective_amp=True,amp_max_overflow_retries=1)
    while step < 400:
        session.ctx = dict(phase='training',arm=arm,update=step+1,epoch=epoch+1)
        order = dataset.epoch_order(epoch)
        selected = order[pos:pos+4]
        model.train()
        cid = session.intent('optimizer')
        loss,parts,norm,retries = trainer.train_update(model,optimizer,scaler,dataset,selected,epoch,train_cfg)
        session.returned('optimizer',cid,loss=loss,components=parts,gradient_norm=norm,overflow_retries=retries)
        step += 1; pos += len(selected)
        if pos == len(dataset):
            epoch += 1; pos = 0
        if step % 20 == 0:
            store.save(model,optimizer,epoch,pos,step,True,scaler=scaler)
            status('training',arm=arm,logical_updates=step,physical_optimizer_accounting=accounting()['optimizer'])
            print(json.dumps(dict(arm=arm,step=step,loss=loss)),flush=True)
        if step in (80,240,320,400):
            monitor(session,model,arm,step)
    qstate = torch.load(Q_PARENT,map_location='cpu',weights_only=False)['model']
    need(all(torch.equal(v.detach().cpu(),qstate[k]) for k,v in model.quality.state_dict().items()), 'Frozen quality head changed')
    path = RUN/arm/'final.pt'
    if path.exists():
        saved = torch.load(path,map_location='cpu',weights_only=False)
        need(saved['logical_updates']==400 and saved['arm']==arm and saved['proposal_sha256']==sha(PROPOSAL), 'Uncommitted final checkpoint has different identity')
        need(all(torch.equal(v.detach().cpu(),saved['model'][k]) for k,v in model.state_dict().items()), 'Uncommitted final checkpoint differs from recovered state')
        once(final,dict(sha256=sha(path),logical_updates=step,proposal_sha256=sha(PROPOSAL)))
    else:
        temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        torch.save(dict(model=model.state_dict(),logical_updates=step,arm=arm,proposal_sha256=sha(PROPOSAL)),temporary)
        os.replace(temporary,path)
        once(final,dict(sha256=sha(path),logical_updates=step,proposal_sha256=sha(PROPOSAL)))
    del model,optimizer,scaler
    torch.cuda.empty_cache()


def predict(session,model,record,condition):
    import numpy as np
    import torch
    from scripts.evaluate_segmentation_baseline_v16 import prepare_input,load_labels,frame_metrics
    from scripts.extract_mask_scores_v17 import decode_with_scores
    from src.p5_quality_v23 import corrupt
    old = config()
    raw = prepare_input(ROOT,old,record,'rgbd')
    inputs,_,metadata = corrupt(torch.from_numpy(raw),record,old['quality_screen_epoch'],condition)
    session.ctx = dict(session.ctx,scene=record['scene'],frame=record['frame'],condition=condition)
    model.eval()
    with torch.inference_mode():
        h = model(inputs[None].cuda())
    heads = dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),
        center_probability=h['centers'][0,0].sigmoid().cpu().numpy(), offsets=h['offsets'][0].cpu().numpy())
    need(all(v.dtype == np.float32 and np.isfinite(v).all() for v in heads.values()), 'Invalid FP32 cached heads')
    pred,fg,scores,_ = decode_with_scores(heads,**old['decoder'])
    metrics = frame_metrics(pred,fg,load_labels(ROOT,old,record))
    arrays = dict(**heads,instances=pred,foreground=fg,gate=h['effective_gate'][0,0].cpu().numpy())
    return arrays,dict(scene=record['scene'],frame=record['frame'],scores=scores,metrics=metrics,corruption=metadata)


def aggregate_f1(rows):
    import numpy as np
    scenes=[]
    for sc in sorted({r['scene'] for r in rows}):
        counts = {k:sum(r['metrics']['instances'][k] for r in rows if r['scene']==sc) for k in ('tp','fp','fn')}
        den=2*counts['tp']+counts['fp']+counts['fn']
        scenes.append(dict(scene=sc,f1=2*counts['tp']/den if den else 0.,predicted_instances=counts['tp']+counts['fp']))
    return dict(scene_macro_f1=float(np.mean([s['f1'] for s in scenes])),scenes=scenes)


def monitor(session,model,arm,step):
    path=OUT/'monitor'/f'{arm}_{step:04d}.json'
    if path.exists(): return
    import numpy as np
    result={}
    for condition in ('clean','joint'):
        rows=[]
        for rec in config()['monitor_records']:
            receipt=OUT/'monitor_cache'/f'{arm}_{step:04d}_{condition}'/f"scene_{rec['scene']:04d}_{rec['frame']}.json"
            cache=receipt.with_suffix('.npz')
            if receipt.exists():
                row=read(receipt);need(sha(cache)==row['cache_sha256'],'Monitor cache changed')
            else:
                need(not cache.exists(),'Uncommitted monitor cache exists; preserve for review')
                session.ctx=dict(phase='monitor',arm=arm,logical_updates=step)
                arrays,row=predict(session,model,rec,condition)
                cache.parent.mkdir(parents=True,exist_ok=True)
                with cache.open('xb') as handle:np.savez_compressed(handle,**arrays)
                row=dict(**row,cache_sha256=sha(cache));once(receipt,row)
            rows.append(row)
        result[condition]=aggregate_f1(rows)
    once(path,result)
    need(all(s['predicted_instances']>0 for s in result['clean']['scenes']), 'Collapsed clean scene; no expensive expansion')


def task_sets():
    return ([dict(name=f'core_{arm}_{condition}',arm=arm,ablation='learned',condition=condition) for arm in ('fixed','quality') for condition in CONDITIONS]
        + [dict(name=f'ablation_{ablation}_{condition}',arm='quality',ablation=ablation,condition=condition) for ablation in ('unit','mean') for condition in ('clean','joint')])


def export(session):
    import numpy as np
    import torch
    for arm in ('fixed','quality'):
        final = read(RUN/arm/'final.json')
        need(final['logical_updates']==400 and sha(RUN/arm/'final.pt')==final['sha256'],'Final400 missing or changed')
        model = make_model(arm,session)
        model.load_state_dict(torch.load(RUN/arm/'final.pt',map_location='cpu',weights_only=False)['model'])
        for item in [x for x in task_sets() if x['arm']==arm]:
            model.ablation=item['ablation']
            for record in config()['monitor_records']:
                path=OUT/'exports'/item['name']/f"scene_{record['scene']:04d}_{record['frame']}.json"
                cache=path.with_suffix('.npz')
                if path.exists():
                    need(read(path)['cache_sha256']==sha(cache),'Export cache changed');continue
                need(not cache.exists(),'Uncommitted cache exists; preserve and review it before retry')
                session.ctx=dict(phase='export',set=item['name'],arm=arm)
                arrays,row=predict(session,model,record,item['condition'])
                cache.parent.mkdir(parents=True,exist_ok=True)
                with cache.open('xb') as handle:
                    np.savez_compressed(handle,**arrays)
                once(path,dict(**row,set=item,cache_sha256=sha(cache)))
            print(json.dumps(dict(exported=item['name'])),flush=True)
        del model
        torch.cuda.empty_cache()
    inventory={p.relative_to(ROOT).as_posix():stat(p) for directory in ('exports','monitor','monitor_cache') for p in (OUT/directory).rglob('*') if p.is_file()}
    once(OUT/'exports_complete.json',dict(frame_entries=280,sets=task_sets(),inventory=inventory))


def scene_inputs(item,scene,independent):
    import numpy as np
    from PIL import Image
    from scripts import p4_complete_evaluation_v22r2 as inherited
    from scripts import evaluate_mask_ap_v17 as official
    from scripts import accept_mask_ap_v17 as reference
    from scripts.extract_mask_scores_v17 import decode_with_scores
    old=config();images=[];annotations=[];detections=[];matching=[];rows=[]
    for rec in [r for r in old['monitor_records'] if r['scene']==scene]:
        path=OUT/'exports'/item['name']/f"scene_{scene:04d}_{rec['frame']}.json"
        row=read(path);need(sha(path.with_suffix('.npz'))==row['cache_sha256'],'Cache digest differs')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as cache:
            heads={k:cache[k] for k in ('foreground_probability','center_probability','offsets')}
            if independent:
                result=reference.independent_decode(**heads,decoder=old['decoder'])
                pred,fg,scores=result['instances'],result['foreground'],result['retained_instances']
            else:
                pred,fg,scores,_=decode_with_scores(heads,**old['decoder'])
            need(np.array_equal(pred,cache['instances']) and np.array_equal(fg,cache['foreground']) and scores==row['scores'],'Independent decoding differs')
            gate=cache['gate'];need(np.isfinite(gate).all() and gate.min()>=0 and gate.max()<=1,'Invalid reliability')
            if item['arm']=='fixed' or item['ablation']=='unit':need(np.all(gate==1),'Unit gate differs')
            if item['ablation']=='mean':need(np.ptp(gate)==0,'Mean ablation not spatially constant')
        label=ROOT/old['data_root']/rec['files']['label']
        need(sha(label)==rec['sha256']['label'],'Original labels changed')
        with Image.open(label) as im: raw=np.asarray(im)
        high=inherited.indexed_resize(pred,720,1280) if independent else inherited.high_resolution(pred)
        gt_low=inherited.indexed_resize(raw,480,640)
        inherited.frame_check(pred,fg,gt_low,row['metrics'])
        rows.append(row)
        pids=[int(x) for x in np.unique(high) if x>0];gids=[int(x) for x in np.unique(raw) if x>0]
        image_id=scene*1000+int(rec['frame']);images.append(dict(id=image_id,height=720,width=1280));gtids=[];dtids=[]
        for gid in gids:
            mask=raw==gid;annid=len(annotations)+1;gtids.append(annid);yy,xx=np.nonzero(mask)
            annotations.append(dict(id=annid,image_id=image_id,category_id=1,area=int(mask.sum()),iscrowd=0,
                segmentation=official.encode_mask(mask),bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]))
        for score in scores:
            need(score['instance_id'] in pids,'Instance lost original-resolution support')
            detections.append(dict(image_id=image_id,category_id=1,score=score['score'],segmentation=official.encode_mask(high==score['instance_id'])))
            dtids.append(len(detections))
        overlap=reference.dense_overlap(high,raw,pids,gids)['iou']
        matching.append(reference.independent_image_matching(image_id,gtids,dtids,[x['score'] for x in scores],overlap))
    return dict(gt=dict(info={},images=images,categories=[dict(id=1,name='object')],annotations=annotations),detections=detections),matching,aggregate_f1(rows)



def review_monitors(session):
    import numpy as np
    from PIL import Image
    from scripts import p4_complete_evaluation_v22r2 as inherited
    from scripts import accept_mask_ap_v17 as reference
    old=config();count=0
    for arm in ('fixed','quality'):
        for step in (80,240,320,400):
            expected=read(OUT/'monitor'/f'{arm}_{step:04d}.json')
            for condition in ('clean','joint'):
                rows=[]
                for rec in old['monitor_records']:
                    session.guard()
                    receipt=OUT/'monitor_cache'/f'{arm}_{step:04d}_{condition}'/f"scene_{rec['scene']:04d}_{rec['frame']}.json"
                    row=read(receipt);cache=receipt.with_suffix('.npz')
                    need(sha(cache)==row['cache_sha256'],'Monitor bytes changed')
                    with np.load(cache,allow_pickle=False) as values:
                        rebuilt=reference.independent_decode(**{k:values[k] for k in ('foreground_probability','center_probability','offsets')},decoder=old['decoder'])
                        need(np.array_equal(rebuilt['instances'],values['instances']) and np.array_equal(rebuilt['foreground'],values['foreground']) and rebuilt['retained_instances']==row['scores'],'Independent monitor decoder differs')
                    with Image.open(ROOT/old['data_root']/rec['files']['label']) as image:
                        gt=inherited.indexed_resize(np.asarray(image),480,640)
                    inherited.frame_check(rebuilt['instances'],rebuilt['foreground'],gt,row['metrics'])
                    rows.append(row);count+=1
                need(aggregate_f1(rows)==expected[condition],'Independent monitor aggregate differs')
    need(count==320,'Monitor scope incomplete')
    return count


def decision(metrics):
    import numpy as np
    gain=float(np.mean([metrics[f'core_quality_{d}']['macro_AP']-metrics[f'core_fixed_{d}']['macro_AP'] for d in CONDITIONS[1:]]))
    clean_loss=metrics['core_fixed_clean']['macro_AP']-metrics['core_quality_clean']['macro_AP']
    scene_gains={str(sc):float(np.mean([metrics[f'core_quality_{d}']['scenes'][str(sc)]['AP']-metrics[f'core_fixed_{d}']['scenes'][str(sc)]['AP'] for d in CONDITIONS[1:]])) for sc in sorted({r['scene'] for r in config()['monitor_records']})}
    ablations={a:metrics['core_quality_joint']['macro_AP']-metrics[f'ablation_{a}_joint']['macro_AP'] for a in ('unit','mean')}
    monitor_values=[float(np.mean([read(OUT/'monitor'/f'{arm}_{step:04d}.json')[condition]['scene_macro_f1'] for arm in ('fixed','quality') for condition in ('clean','joint')])) for step in (240,320,400)]
    stability_span=max(monitor_values)-min(monitor_values)
    gates=dict(stability=stability_span<=.05,degraded_mask_AP_gain=gain>=.01,clean_mask_AP_preserved=clean_loss<=.01,
        improved_scenes=sum(x>0 for x in scene_gains.values())>=6,
        clean_validity=all(metrics[f'core_{a}_clean']['macro_F1']>=.5 and all(s['predicted_instances']>0 for s in metrics[f'core_{a}_clean']['scenes'].values()) for a in ('fixed','quality')),
        spatial_quality_ablation=all(x>0 for x in ablations.values()))
    return dict(gates=gates,development_repair_supported=all(gates.values()),mean_degraded_mask_AP_gain=gain,
        clean_mask_AP_loss=clean_loss,by_scene_gain=scene_gains,joint_ablation_gain=ablations,
        last_three_paired_macro_F1_values=monitor_values,last_three_span=stability_span,
        full160_adoption_demonstrated=False,independent_generalization_demonstrated=False,
        scope='Fixed 20 exposed development frames x five conditions; engineering completion is separate from scientific support')


def evaluate(session,independent=False):
    import numpy as np
    from scripts import p4_complete_evaluation_v22r2 as inherited
    from scripts import evaluate_mask_ap_v17 as official
    need('torch' not in sys.modules,'Run CPU AP and review in a separate process from GPU commands')
    seal=read(OUT/'exports_complete.json')
    need(seal['frame_entries']==280,'Export set incomplete')
    for path,value in seal['inventory'].items():need(stat(ROOT/path)==value,'Sealed export changed')
    before={p.relative_to(ROOT).as_posix():stat(p) for p in OUT.rglob('*') if p.is_file()} if independent else None
    verified_monitors=review_monitors(session) if independent else 0
    metrics={}
    for item in task_sets():
        scene_values={}
        for scene in sorted({r['scene'] for r in config()['monitor_records']}):
            session.ctx=dict(phase='review' if independent else 'AP',set=item['name'],scene=scene)
            session.guard()
            inputs,matching,diag=scene_inputs(item,scene,independent)
            path=OUT/'AP'/item['name']/f'scene_{scene:04d}.json'
            if path.exists():
                ap=read(path)
            else:
                need(not independent,'Review cannot create missing AP')
                cid=session.intent('official_AP')
                ap=official.official_evaluation(inputs)
                once(path,ap)
                session.returned('official_AP',cid,path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
            inherited.audit_ap(ap,matching)
            scene_values[str(scene)]=dict(**inherited.primary(ap),**diag['scenes'][0])
        value=dict(scenes=scene_values,macro_AP=float(np.mean([x['AP'] for x in scene_values.values()])),
            macro_F1=float(np.mean([x['f1'] for x in scene_values.values()])))
        metrics[item['name']]=value
        path=OUT/'metrics'/(item['name']+'.json')
        if independent:need(read(path)==value,'Cached aggregate differs')
        else:once(path,value)
        print(json.dumps(dict(set=item['name'],review=independent,AP=value['macro_AP'])),flush=True)
    result=decision(metrics)
    if independent:
        need(read(OUT/'decision.json')==result,'Decision re-computation differs')
        for path,value in before.items():need(stat(ROOT/path)==value,'Read-only review modified research output')
        once(LOG/'review.json',dict(passed=True,engineering_passed=True,scientific_usable=result['development_repair_supported'],
            usability_scope=result['scope'],full_original_P5_passed=False,verified_monitor_entries=verified_monitors,frame_entries=280,official_AP_blocks=140,
            new_forwards=0,new_official_AP=0,decision_sha256=sha(OUT/'decision.json'),reviewed_outputs=before))
    else:
        once(OUT/'decision.json',result)
    status('reviewed' if independent else 'evaluated',decision=result,accounting=accounting())


def cold():
    review=read(LOG/'review.json')
    need(review['passed'],'Independent review not complete')
    before=accounting()
    need(before['optimizer']['known_returns']==800 and not before['optimizer']['unknown_ids'], 'Exactly 800 confirmed research updates required')
    for path,value in review['reviewed_outputs'].items():need(stat(ROOT/path)==value,'Completed artifact changed')
    for arm in ('fixed','quality'):
        value=read(RUN/arm/'final.json')
        need(value['logical_updates']==400 and sha(RUN/arm/'final.pt')==value['sha256'],'Final training state differs')
    need(accounting()==before,'Cold entry unexpectedly changed research counts')
    once(LOG/'cold.json',dict(passed=True,engineering_passed=True,scientific_usable=review['scientific_usable'],
        usability_scope=review['usability_scope'],full_original_P5_passed=False,
        accounting_fully_observed=all(not v['unknown_ids'] for v in before.values()),
        unknown_calls_remain_unknown=True,engineering_scope='Verified completed artifacts, not an assertion that all historical call returns were observed',new_training=0,new_forward=0,new_AP=0,accounting=before,
        decision_sha256=sha(OUT/'decision.json')))
    status('complete',accounting=before,decision=read(OUT/'decision.json'))
    print(json.dumps(dict(cold_reentry='passed',new_research_calls=0)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','check','train','export','evaluate','review','cold'))
    parser.add_argument('--arm',choices=('fixed','quality'))
    args=parser.parse_args()
    from scripts.pipeline import project_lock
    with project_lock(ROOT):
        if args.mode=='prepare':
            once(PROPOSAL,proposal_body());print(str(PROPOSAL));return
        if args.mode=='check':preflight();return
        binding()
        need(read(OUT/'preflight.json')['passed'],'Preparation identity check missing')
        if args.mode=='cold':cold();return
        session=Session(args.mode)
        try:
            if args.mode=='train':
                for arm in ([args.arm] if args.arm else ['fixed','quality']):train_arm(session,arm)
                if all((RUN/a/'final.json').exists() for a in ('fixed','quality')):
                    once(OUT/'training_complete.json',dict(arms={a:read(RUN/a/'final.json') for a in ('fixed','quality')},updates_per_arm=400))
            elif args.mode=='export':export(session)
            else:evaluate(session,args.mode=='review')
            binding();session.finish()
        except BaseException as error:
            session.finish('failed',repr(error));status('failed',mode=args.mode,error=repr(error),accounting=accounting());raise


if __name__=='__main__':main()
