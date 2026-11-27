"""One bounded spatial RGB-D correspondence model; original P6 gates unchanged."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[_key]='2'
import sys,json,time,uuid,shutil,argparse,subprocess,traceback
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p6_mixture_common_v43 import now,need,read,identity,sha,atomic,once,append
from scripts.research_cache_recovery_v44r1 import process_tree_rss
from scripts.pipeline import project_lock
LOG=ROOT/'logs/P6/local_correspondence_v50'
OUT=ROOT/'results/P6/local_correspondence_v50'
CONFIG=ROOT/'configs/p6_local_correspondence_v50.json'
STATE=ROOT/'runtime/pipeline/p6_local_correspondence_v50.json'
PROTOCOL=ROOT/'docs/P6空间表面概率模型协议_v50.txt'
REPORT=ROOT/'docs/P6空间表面概率模型结果_v50.txt'
OLD=ROOT/'results/P6/mixture_v43'
PARENT=ROOT/'results/P6/probability_calibration_v46'
DATA=ROOT/'logs/P6/correspondence_v40/data_audit.json'
ORIGINAL_STATE='b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'
SOURCES=['scripts/p6_local_correspondence_v50.py','src/p6_local_correspondence_v50.py',
 'src/p6_local_features_v50.py','scripts/p6_local_data_v50.py','src/p6_local_training_review_v50.py',
 'tests/test_p6_local_correspondence_v50.py','tests/test_p6_local_features_v50.py',
 'tests/test_p6_local_training_review_v50.py','tests/test_p6_local_correspondence_independent_v50.py',
 'tests/test_p6_local_runner_data_v50.py',
 'scripts/check_p6_local_correspondence_v50.py']
GATES=('warp_nll_gain','warp_brier_noninferior','warp95_coverage','warp_set_sharper',
 'warp_multilevel_reliability','worst_scene_warp','worst_scene_depth','source_depth95',
 'source_depth_sharpness','source_depth_nll_gain','target_joint_nll_noninferior',
 'target_missing_brier_noninferior','target_conditional_coverage','source_joint95',
 'target_union_sharpness','target_union95','observation_preserved')
LIMITS={'train_once':1,'fit_posterior':1,'fit_posterior_review':1,'warp_calibration':1,'warp_calibration_replay':1,
 'evaluation_posterior':1,'evaluation_posterior_replay':1}
for _split,_frames in (('fit',48),('calibration',16),('evaluation',16)):
 for _suffix in ('','_review'):
  LIMITS[_split+'_data'+_suffix+'_frame']=_frames
  LIMITS[_split+'_data'+_suffix+'_case']=_frames*27
for _split in ('calibration','evaluation'):
 for _suffix in ('','_review'):
  LIMITS[_split+_suffix+'_frame']=16
  LIMITS[_split+_suffix+'_case']=432
_seen=None

def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()

def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []

def ledger(path=None):
    starts = {}; ends = {}
    for e in events(path or LOG/'calls.jsonl'):
        target = starts if e['event'] == 'intent' else ends
        need(e['event'] in ('intent', 'returned') and e['id'] not in target, 'Invalid duplicate ledger event')
        target[e['id']] = e
    need(set(ends) <= set(starts), 'Unpaired return')
    for cid, e in ends.items():
        need(all(e[k] == starts[cid][k] for k in ('kind', 'key')), 'Return identity differs')
    return dict(intents=len(starts), known_returns=len(ends), unknown=sorted(set(starts)-set(ends)),
                counts={k: sum(e['kind'] == k for e in ends.values()) for k in sorted({e['kind'] for e in starts.values()})})

def call(kind, key, function):
    global _seen
    if _seen is None:
        need(not ledger()['unknown'], 'Prior unknown calls require evidence audit')
        _seen = {(e['kind'], e['key']) for e in events(LOG/'calls.jsonl') if e['event'] == 'intent'}
    need(kind in LIMITS and sum(k == kind for k, _ in _seen) < LIMITS[kind], 'Call cap '+kind)
    need((kind, key) not in _seen, 'Duplicate scientific call')
    _seen.add((kind, key)); cid = uuid.uuid4().hex
    append(LOG/'calls.jsonl', dict(at=now(), event='intent', id=cid, kind=kind, key=key))
    answer = function()
    append(LOG/'calls.jsonl', dict(at=now(), event='returned', id=cid, kind=kind, key=key))
    return answer

def backup(p, tag):
    if p.exists():
        folder = LOG/'history'; folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, folder/(tag+'_'+uuid.uuid4().hex+'_'+p.name))

def status(phase, **fields):
    atomic(STATE, dict(at=now(), phase=phase, all_tasks_percent=45, experiment_percent=56.25,
                      current_part='P6_local_correspondence_v50', P6_scientific_passed=False,
                      P7_started=False, P10_started=False, recurring_automation=False, **fields))

def supervise(phase, **fields):
    path = ROOT/'runtime/pipeline/part_level_supervision.json'; backup(path, phase)
    value = read(path)
    value.update(recorded_at=now(), current_authorized_part='P6_local_correspondence_v50',
        current_execution_state=phase, current_runner=rel(Path(__file__)),
        current_execution_evidence=rel(STATE), next_repair_started=True,
        user_authorization='用户授权实例条件局部RGB-D对应概率网络：固定864训练案例512步一次拟合、完整实际分支验收后停止',
        automatic_resume_authorized=False, automatic_scheduler_recreated=False,
        stop_after='P6_local_correspondence_v50', stop_after_part='P6_local_correspondence_v50',
        remaining_research_updates_cap=1, waiting_for_user_next_part_confirmation=False,
        all_tasks_percent_planning_estimate=45, experiment_percent_planning_estimate=56.25,
        summary_report=None, final_readonly=None, final_cold=None, final_delivery_index=None,
        unchanged_progress_reason='No new main-delivery milestone until complete P6 acceptance')
    value.update(fields); atomic(path, value)

def handoff(message, tag):
    path = ROOT/'docs/实验自动续接说明.txt'; backup(path, tag)
    path.write_text('最新：'+now()+' '+message+'\n\n历史：\n'+path.read_text(encoding='utf-8-sig'), encoding='utf-8')

@contextmanager
def own_lock():
    import msvcrt
    path = ROOT/'runtime/pipeline/p6_local_correspondence_v50.lock'
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0: stream.write(b'0'); stream.flush()
        stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try: yield
        finally: stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

def protected_check(c):
    need(Path.cwd().resolve() == ROOT, 'Explicit project working directory required')
    for name, expected in c['protected'].items():
        need(identity(ROOT/name) == expected, 'Protected dependency changed '+name)
    need(sha(ROOT/'runtime/pipeline/state.json') == ORIGINAL_STATE, 'Historical pipeline state changed')

def verify():
    c = read(CONFIG); need(identity(CONFIG) == read(LOG/'freeze.json')['config'], 'Frozen config changed')
    protected_check(c); return c

def output_bytes():
    return sum(p.stat().st_size for folder in (LOG, OUT) if folder.exists() for p in folder.rglob('*') if p.is_file())

def child_guard():
    state = read(STATE)
    need(os.environ.get('P6_V50_CHILD_TOKEN') == state.get('run_token') and bool(state.get('run_token')), 'Worker requires active controller')
    need(time.time() < state['deadline_epoch'], 'Cumulative compute wall budget exhausted')
    need(not (LOG/'STOP').exists(), 'User STOP marker')
    need(output_bytes() <= 2*1024**3, 'Combined output cap exceeded')

def cache():
    from scripts.p6_mixture_v43 import generated
    return generated()

def training_arrays():
    import numpy as np
    arrays, rows = cache()
    keep = np.array([r['split'] == 'fit' for r in rows])
    fit_rows = [r for r in rows if r['split'] == 'fit']
    need(len(fit_rows) == 1296 and len({r['key'] for r in fit_rows}) == 1296, 'Wrong fit scope')
    return arrays['X'][keep], arrays['y'][keep], fit_rows

def selection_data():
    """Reuse exactly fixed fold0 heldout scores; the full joint model has seen F0."""
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_read
    from src.p6_probability_calibration_v46 import validate_oof_coverage,FOLDS
    x,y,rows=training_arrays();oof=arrays_read(PARENT/'oof.npz');saved=read(PARENT/'fold_0.json');head=saved['head']
    scenes=np.array([r['scene'] for r in rows]);assigned=np.asarray(oof['assigned_folds'])
    need(rows==read(PARENT/'oof_rows.json') and np.array_equal(y,oof['targets']) and np.array_equal(scenes,oof['scenes']),'OOF row/label/scene alignment differs')
    validate_oof_coverage(scenes,assigned,oof['visits'],expected_cases_per_scene=108)
    train=np.flatnonzero(assigned!=0);held=np.flatnonzero(assigned==0)
    need(len(train)==864 and len(held)==432 and not np.intersect1d(train,held).size and set(scenes[held].tolist())==set(FOLDS[0]),'Fixed training/auxiliary partition differs')
    need(saved['training_indices']==train.tolist() and saved['holdout_indices']==held.tolist()
         and saved['training_scenes']==sorted(set(scenes[train].tolist())) and saved['holdout_scenes']==list(FOLDS[0]),'Frozen fixed-head provenance differs')
    center=x[train].mean(axis=(0,1));scale=x[train].std(axis=(0,1));scale=np.where(scale>1e-8,scale,1.)
    prior_counts=np.bincount(y[train],minlength=81).astype(float)+1/81;prior=prior_counts/prior_counts.sum()
    need(head['fitted_cases']==864 and np.array_equal(center,np.asarray(head['center'])) and np.array_equal(scale,np.asarray(head['scale']))
         and np.array_equal(prior,np.asarray(head['empirical_prior'])),'Fixed-head normalization/prior must come only from its864 training rows')
    need(np.array_equal(oof['priors'][held],np.broadcast_to(prior,(432,81))),'Cached auxiliary priors differ from fixed-head training prior')
    provenance=dict(fixed_fold=0,training_indices=train.tolist(),auxiliary_indices=held.tolist(),
        training_scenes=saved['training_scenes'],auxiliary_scenes=saved['holdout_scenes'],
        inputs={rel(p):identity(p) for p in (PARENT/'fold_0.json',PARENT/'oof.npz',PARENT/'oof_rows.json',OLD/'generated.npz')},
        inherited_complete_oof_review=identity(ROOT/'logs/P6/probability_calibration_v46/oof_review.json'),
        auxiliary_unseen_by_head=True,auxiliary_unseen_by_depth_H_or_whole_research=False,new_head_predictions=0)
    parent=read(ROOT/'results/P6/joint_repair_v45/model.json')
    return oof['scores'][held],oof['priors'][held],y[held],scenes[held],[rows[i] for i in held],head,parent,provenance

def reference_candidate(probabilities,priors,targets,scenes):
    """Independent scientific formulas; keep actual auxiliary scene identifiers."""
    import numpy as np
    def statistics(matrix):
        records=[]
        for p,target in zip(matrix,targets):
            order=np.argsort(-p,kind='stable');cumulative=np.cumsum(p[order]);levels={}
            for level in (.5,.8,.95):
                count=min(81,int(np.searchsorted(cumulative,level,side='left'))+1);chosen=order[:count]
                levels[str(level)]=dict(covered=bool(int(target) in chosen),mass=float(p[chosen].sum()),size=count)
            onehot=np.zeros(81);onehot[int(target)]=1.
            records.append(dict(nll=float(-np.log(p[int(target)])),brier=float(np.sum((p-onehot)**2)),levels=levels))
        return dict(nll=float(np.mean([r['nll'] for r in records])),brier=float(np.mean([r['brier'] for r in records])),
            levels={str(level):{field:float(np.mean([r['levels'][str(level)][field] for r in records])) for field in ('covered','mass','size')} for level in (.5,.8,.95)},
            worst=min(float(np.mean([r['levels']['0.95']['covered'] for r,s in zip(records,scenes) if s==scene])) for scene in sorted(set(scenes.tolist()))))
    method,prior=statistics(probabilities),statistics(priors)
    levels={str(level):dict(method=method['levels'][str(level)],prior=prior['levels'][str(level)]) for level in (.5,.8,.95)}
    gaps={key:abs(row['method']['covered']-row['method']['mass']) for key,row in levels.items()}
    gates=dict(warp_nll_gain=prior['nll']-method['nll']>=.05,warp_brier_noninferior=method['brier']<=prior['brier'],
        warp95_coverage=levels['0.95']['method']['covered']>=.90,
        warp_set_sharper=levels['0.95']['method']['size']<=.9*levels['0.95']['prior']['size'],
        warp_multilevel_reliability=all(v<=.05 for v in gaps.values()),worst_scene_warp=method['worst']>=.80)
    return dict(summary=dict(split='fit_auxiliary_calibration',warp_nll=dict(method=method['nll'],prior=prior['nll']),
        warp_brier=dict(method=method['brier'],prior=prior['brier']),warp_levels=levels,worst_scene_warp95=method['worst'],case_count=len(targets)),
        gates={k:bool(v) for k,v in gates.items()},other_five_passed=bool(all(v for k,v in gates.items() if k!='warp_multilevel_reliability')),
        reliability_gap_by_level=gaps,max_reliability_gap=max(gaps.values()))

def warp_metric(weights, target):
    from src import p6_joint_repair_v45 as core
    value = core.warp_statistics(weights, int(target))
    value['levels'] = {str(v['level']): dict(covered=v['covered'], size=v['size'], mass=v['probability'])
                       for v in value.pop('credible_sets')}
    return value


def freeze():
    started=time.monotonic()
    with project_lock(ROOT),own_lock():
        need(not CONFIG.exists(),'Already frozen; use verify/deliver')
        checks=read(LOG/'internal_checks.json');review=read(LOG/'independent_review.json')
        for receipt in (checks,review):
            need(receipt['passed'],'Internal checks/independent review not passed')
            need(set(SOURCES)<=set(receipt['source_sha256']),'Source omitted from review')
            for name,digest in receipt['source_sha256'].items():need(sha(ROOT/name)==digest,'Reviewed source changed '+name)
        from scripts import p6_fixed_mixture_v49 as previous
        from scripts.p6_local_data_v50 import asset_bindings
        from importlib.metadata import version as package_version
        oldconfig=previous.verify();done=read(previous.OUT/'closeout.json')
        need(done['engineering_passed'] and previous.verify_executed()==done['proof'],'Previous completed evidence differs')
        need(done['cold']==identity(previous.LOG/'cold.json') and done['report']==identity(previous.REPORT)
             and done['child_ledger']==previous.ledger(previous.LOG/'children.jsonl'),'Previous cold/report/children differ')
        assets=asset_bindings();need(assets['production_unique_rgb_cap']==80 and assets['full_replay_rgb_cap']==80,'RGB reuse coverage changed')
        protected=dict(oldconfig['protected']);protected.update(assets['dependencies'])
        dependencies=SOURCES+[rel(PROTOCOL),'docs/P6空间表面概率模型独立审查_v50.txt',rel(previous.CONFIG),rel(previous.STATE),rel(previous.REPORT),
            rel(LOG/'internal_checks.json'),rel(LOG/'independent_review.json')]
        dependencies += [rel(p) for folder in (previous.OUT,previous.LOG) for p in folder.rglob('*') if p.is_file()]
        protected.update({name:identity(ROOT/name) for name in dependencies})
        preparation={rel(p):dict(identity=identity(p),record=read(p)) for p in (LOG/'preparation').glob('*.json')}
        measured=sum(float(v['record']['seconds']) for v in preparation.values())
        c=dict(at=now(),version='v50',protected=protected,call_limits=LIMITS,assets=assets,
          environment=dict(python=sys.version,executable=sys.executable,packages={name:package_version(name) for name in ('torch','torchvision','numpy','scipy','Pillow')}),
          controller_wall_seconds=7200,preparation_seconds=measured+30.,preparation=preparation,freeze_reserve_seconds=30,
          cpu_threads=2,process_tree_rss_limit=3*1024**3,output_limit=2*1024**3,free_disk_min=60*1024**3,
          original_gates_unchanged=True,train_cases=864,auxiliary_cases=432,calibration_cases=432,evaluation_cases=432,
          training=dict(scenes=[4,5,9,10,14,16,19,20],auxiliary_scenes=[3,8,12,17],frames_per_scene=4,
              updates=512,batch_size=32,seed=20261102,optimizer='Adam',lr=.001,weight_decay=.001,parameter_count=1193,
              one_fit=True,final_checkpoint_only=True,early_stop=False,validation_selection=False),
          admission=dict(original_six_warp_gates=True,nll_gain_over_frozen_same_fold_head=.01,brier_noninferior_to_same_fold_head=True),
          observations='81 integer candidates; 4x4 fixed spatial cells; 28 observable channels; no GT/source/truewarp/mode/seed in features',
          actual_neural_caps=dict(rgb_forward=160,train_forward=512,train_backward=512,train_update=512,predict_forward=138),
          feature_call_cap=2160,feature_replay_cap=2160,depth_refits=0,shape_refits=0,AP_calls=0,grasp_calls=0,
          budget_scope='One new authorized model experiment including all local preparation and freeze reserve; historical P6 cost preserved separately without false additive total',
          historical_usage=oldconfig.get('historical_usage',{}),previous_usage={rel(p):read(p) for p in previous.LOG.glob('controller_usage_*.json')},
          stages='fit data and full replay -> one512update fit -> evidence verification and all1296 final predictions/replay -> F0 admission -> caldata/replay -> cal6/replay -> cal17/replay -> evaldata/replay -> eval17/replay -> cold -> stop',
          scope='Previously exposed synthetic global integer-warp GraspNet RealSense development scope. Upstream segmenter saw all20 scenes; only this localhead train/F0 scene split is disjoint.',
          training_review_scope='Full training evidence and final-model prediction replay, not a second512-update optimizer refit or per-update numerical gradient replay',
          failed_gate_action='Stop after complete review of actually executed branch; no new parameter selection/fold/model or P7/P10',
          progress_before=[45,56.25],progress_after_full_pass=[47,58.75])
        protected_check(c)
        need(c['preparation_seconds']<7200 and shutil.disk_usage(ROOT).free>=c['free_disk_min'],'Insufficient resources')
        once(CONFIG,c);once(LOG/'freeze.json',dict(at=now(),config=identity(CONFIG)))
        need(time.monotonic()-started<=30,'Freeze reserve exceeded')
        status('frozen_ready');supervise('v50_frozen_ready')
        handoff('P6 v50实例条件局部RGB-D对应概率模型已冻结：864案例512固定步一次训练；F0一次准入，再按原cal6/cal17/eval17门槛执行并完整复验。7200秒累计墙钟帽，2 CPU线程、3GiB实际子进程树RSS、2GiB输出；失败停，不自动后续版本/P7/P10。全部任务总进度45%、实验总进度56.25%。入口scripts/p6_local_correspondence_v50.py deliver；禁止重复。','freeze')
        print(json.dumps(dict(frozen=True,remaining_seconds=7200-c['preparation_seconds'])))


def neural_callback(scope):
    def save(row):
        kind='rgb_forward' if row['stage']=='rgb_forward' else ('train_'+row['stage'] if scope=='train' else 'predict_forward')
        cid=scope+':'+row['stage']+':'+str(row['id'])
        append(LOG/'neural_calls.jsonl',dict(at=now(),event=row['event'],id=cid,kind=kind,key=scope,payload=row))
        if scope=='train':append(LOG/'train_events.jsonl',dict(at=now(),**row))
    return save


def split_rows(split):
    _,meta=cache();rows=[r for r in meta if r['split']==split]
    need(len(rows)==(1296 if split=='fit' else 432) and len({r['key'] for r in rows})==len(rows),'Changed original data scope')
    return rows


def frame_name(record):return f"{record['scene']:04d}_{record['frame']}"


def check_data_certificate(split):
    proof=read(LOG/(split+'_data_review.json'));manifest=read(OUT/(split+'_data.json'))
    need(proof['passed'] and proof['manifest']==identity(OUT/(split+'_data.json')),'Data replay certificate differs')
    need(proof['case_count']==len(manifest['rows']),'Data case certificate differs')
    for name,binding in manifest['artifacts'].items():need(identity(OUT/name)==binding,'Data artifact changed '+name)
    return manifest


def data_generate(split,replay=False):
    import hashlib,numpy as np
    from scripts.p6_local_data_v50 import load_rgb_model,infer_rgb,load_observed_frame,frame_features
    from scripts.p6_mixture_v43 import observation,independent_source,key_for,MODES,WARPS,arrays_save,arrays_read
    from src.p6_local_features_v50 import FEATURE_NAMES,CANDIDATES
    data=read(DATA);metadata=split_rows(split);expected={r['key']:r for r in metadata}
    stage=split+'_data'+('_review' if replay else '')
    if split=='calibration':need(read(LOG/'auxiliary_review.json')['passed'] and read(OUT/'auxiliary.json')['supported'],'F0 not admitted')
    if split=='evaluation':need(read(LOG/'calibration_review.json')['passed'] and read(OUT/'calibration.json')['passed'],'Original cal17 not passed')
    original=read(OUT/(split+'_data.json')) if replay else None
    model=load_rgb_model();artifacts={};rows=[]
    for record in data['records'][split]:
        child_guard();name=frame_name(record)
        def load_and_predict():
            frame=load_observed_frame(data,record)
            maps=infer_rgb(model,frame['rgb'],key=split+'/'+name,on_event=neural_callback(stage+'/'+name))
            return frame,maps
        frame,maps=call(stage+'_frame',name,load_and_predict)
        rgb_path=OUT/'rgb'/split/(name+'.npz');feature_path=OUT/'features'/split/(name+'.npz')
        if replay:
            from scripts.p6_correspondence_v40 import load_frame
            reference=load_frame(data,record,labels=False)
            need(set(frame)==set(reference) and all(np.array_equal(frame[k],reference[k]) for k in frame),'Observable loader differs from frozen preprocessing')
            saved=arrays_read(rgb_path)
            need(set(saved)==set(maps) and all(np.array_equal(saved[k],maps[k]) for k in maps),'Complete RGB output replay differs')
        else:
            rgb_path.parent.mkdir(parents=True,exist_ok=True);arrays_save(rgb_path,**maps)
        values=[];targets=[]
        for delta in WARPS:
            for mode in MODES:
                key=key_for(split,record,delta,mode);row=expected[key]
                def calculate():
                    observed=observation(frame,record,delta,mode)
                    if replay:
                        independent=independent_source(frame,record,delta,mode)
                        need(all(np.array_equal(a,b) for a,b in zip(observed,independent)),'Independent corruption/source differs')
                    z,valid,source=observed
                    need(hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()==row['source_sha'],'Frozen source identity differs')
                    need(int(valid.sum())==row['observed_count'],'Original observation denominator differs')
                    return frame_features(frame,z,valid,maps)
                value=call(stage+'_case',key,calculate)
                need(value.shape==(81,4,4,28) and value.dtype==np.float32 and np.isfinite(value).all(),'Invalid feature block')
                target=int(np.flatnonzero(np.all(CANDIDATES==np.asarray(delta),axis=1))[0])
                values.append(value);targets.append(target);rows.append(row);child_guard()
        features=np.stack(values);targets=np.asarray(targets,dtype=np.int64)
        if replay:
            saved=arrays_read(feature_path)
            need(set(saved)=={'X','y'} and np.array_equal(saved['X'],features) and np.array_equal(saved['y'],targets),'Complete feature/target replay differs')
        else:
            feature_path.parent.mkdir(parents=True,exist_ok=True);arrays_save(feature_path,X=features,y=targets)
        for path in (rgb_path,feature_path):artifacts[path.relative_to(OUT).as_posix()]=identity(path)
    need(rows==metadata,'Frozen data order changed')
    manifest=dict(split=split,rows=rows,feature_names=list(FEATURE_NAMES),shape=[len(rows),81,4,4,28],
      artifacts=artifacts,source_audit=identity(DATA),features_use_GT=False,known_warp_as_training_target_only=True)
    if replay:
        need(manifest==original,'Complete data inventory replay differs');child_guard()
        once(LOG/(stage+'.json'),dict(at=now(),passed=True,manifest=identity(OUT/(split+'_data.json')),
          case_count=len(rows),frame_count=len(data['records'][split]),complete_rgb_and_features_replayed=True,
          independent_source_reconstructed=True,original_loader_crosschecked=True,extra_optimizer_training=False))
    else:once(OUT/(split+'_data.json'),manifest)


def feature_arrays(split):
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_read
    manifest=check_data_certificate(split);data=read(DATA);x=[];y=[]
    for record in data['records'][split]:
        block=arrays_read(OUT/'features'/split/(frame_name(record)+'.npz'));x.append(block['X']);y.append(block['y'])
    x=np.concatenate(x);y=np.concatenate(y)
    need(list(x.shape)==manifest['shape'] and y.shape==(len(x),),'Feature scope changed')
    return x,y,manifest['rows']


def fixed_partition_arrays():
    import numpy as np
    x,y,rows=feature_arrays('fit');saved=read(PARENT/'fold_0.json')
    train=np.asarray(saved['training_indices']);held=np.asarray(saved['holdout_indices']);scenes=np.asarray([r['scene'] for r in rows])
    need(len(train)==864 and len(held)==432 and np.array_equal(np.sort(np.concatenate([train,held])),np.arange(1296)),'Partition coverage changed')
    need(set(scenes[train])=={4,5,9,10,14,16,19,20} and set(scenes[held])=={3,8,12,17},'Scene partition changed')
    oldx,oldy,oldrows=training_arrays();need(rows==oldrows and np.array_equal(y,oldy),'Training label/row alignment changed')
    return x,y,rows,scenes,train,held


def train():
    from src import p6_local_correspondence_v50 as core
    x,y,rows,scenes,train_ids,held=fixed_partition_arrays()
    once(OUT/'training_partition.json',dict(training_indices=train_ids.tolist(),auxiliary_indices=held.tolist(),
      training_scenes=sorted(set(scenes[train_ids].tolist())),auxiliary_scenes=sorted(set(scenes[held].tolist())),
      fixed_fold=identity(PARENT/'fold_0.json'),training_original_frames=32,auxiliary_original_frames=16,
      augmented_training_cases=864,independent_training_scene_units=8,whole_pipeline_independence=False))
    try:
        answer=call('train_once','fixed864_512updates',lambda:core.fit_local_model(x[train_ids],y[train_ids],core.FEATURE_NAMES,
          guard=child_guard,on_event=neural_callback('train'),on_update=lambda row:append(LOG/'train_updates.jsonl',dict(at=now(),**row))))
    except BaseException as error:
        once(OUT/'training_failure.json',dict(at=now(),error=repr(error),partial=getattr(error,'partial_report',None),traceback=traceback.format_exc()));raise
    once(OUT/'head.json',answer['model']);once(OUT/'training_report.json',answer['training_report']);child_guard()


def predict(split,scope,kind):
    from src import p6_local_correspondence_v50 as core
    x,y,rows=feature_arrays(split);head=read(OUT/'head.json')
    p=call(kind,scope,lambda:core.predict_warp(head,x,core.FEATURE_NAMES,guard=child_guard,on_event=neural_callback(scope)))
    return p,y,rows


def auxiliary_result(p,y,rows,reference=False):
    import numpy as np
    from scipy.special import logsumexp
    from src.p6_fixed_mixture_v49 import candidate_metrics
    scores,priors,oldy,oldscenes,oldrows,oldhead,parent,provenance=selection_data()
    ids=np.asarray(read(PARENT/'fold_0.json')['holdout_indices']);head=read(OUT/'head.json')
    need([rows[i] for i in ids]==oldrows and np.array_equal(y[ids],oldy),'Frozen heldout alignment changed')
    need(np.array_equal(np.asarray(head['empirical_prior']),priors[0]),'New/old heads must share exact training prior')
    oldlogits=scores+np.log(priors);oldp=np.exp(oldlogits-logsumexp(oldlogits,axis=1,keepdims=True))
    metric=reference_candidate if reference else candidate_metrics
    new=metric(p[ids],priors,oldy,oldscenes);old=metric(oldp,priors,oldy,oldscenes)
    gates=dict(new['gates'])
    gates['nll_gain_over_frozen_same_fold_head']=old['summary']['warp_nll']['method']-new['summary']['warp_nll']['method']>=.01
    gates['brier_noninferior_to_same_fold_head']=new['summary']['warp_brier']['method']<=old['summary']['warp_brier']['method']
    return dict(summary=new['summary'],gates={k:bool(v) for k,v in gates.items()},supported=bool(all(gates.values())),
      old_head_summary=old['summary'],max_reliability_gap=new['max_reliability_gap'],reliability_gap_by_level=new['reliability_gap_by_level'],
      nll_gain_over_old=old['summary']['warp_nll']['method']-new['summary']['warp_nll']['method'],
      case_count=432,head=identity(OUT/'head.json'),frozen_baseline=identity(PARENT/'fold_0.json'),
      scope='One fixed final-checkpoint auxiliary admission; no candidate/model/calibration selection')


def train_review():
    from src.p6_local_training_review_v50 import validate_training_report
    from scripts.p6_mixture_v43 import arrays_save
    x,y,rows,scenes,train_ids,held=fixed_partition_arrays();head=read(OUT/'head.json');report=read(OUT/'training_report.json')
    proof=validate_training_report(head,report,x[train_ids],y[train_ids],events=events(LOG/'train_events.jsonl'),updates=events(LOG/'train_updates.jsonl'))
    p,actual_y,actual_rows=predict('fit','fit_final','fit_posterior')
    import numpy as np
    need(actual_rows==rows and np.array_equal(actual_y,y),'Fit posterior scope changed')
    arrays_save(OUT/'fit_posterior.npz',probability=p);auxiliary=auxiliary_result(p,y,rows);once(OUT/'auxiliary.json',auxiliary)
    once(LOG/'train_review.json',dict(at=now(),passed=True,proof=proof,
      outputs={name:identity(OUT/name) for name in ('head.json','training_report.json','training_partition.json','fit_posterior.npz','auxiliary.json')},
      logs={name:identity(LOG/name) for name in ('train_events.jsonl','train_updates.jsonl')},
      scope='All512 step/order/finite/counter/model evidence verified and all1296 final predictions produced; no optimizer refit'))


def auxiliary_review():
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_read,close_numeric
    review=read(LOG/'train_review.json');need(review['passed'],'Training review failed')
    for name,binding in review['outputs'].items():need(identity(OUT/name)==binding,'Training output changed')
    for name,binding in review['logs'].items():need(identity(LOG/name)==binding,'Training trace changed')
    p,y,rows=predict('fit','fit_final_replay','fit_posterior_review')
    need(np.array_equal(p,arrays_read(OUT/'fit_posterior.npz')['probability']),'All1296 final prediction replay differs')
    fresh=auxiliary_result(p,y,rows,reference=True);close_numeric(fresh,read(OUT/'auxiliary.json'))
    if fresh['supported']:
        head=read(OUT/'head.json');parent=read(ROOT/'results/P6/joint_repair_v45/model.json')
        once(OUT/'model.json',dict(warp=head,shape=parent['shape'],depth=parent['depth'],empirical_prior=head['empirical_prior'],
          parent_geometry_identity=identity(ROOT/'results/P6/joint_repair_v45/model.json'),head_identity=identity(OUT/'head.json'),
          new_head_fits=1,supervised_calibration_selections=0,depth_scale_refits=0,shape_refits=0,GT_in_predictor=False))
    once(LOG/'auxiliary_review.json',dict(at=now(),passed=True,supported=fresh['supported'],
      outputs={name:identity(OUT/name) for name in ('fit_posterior.npz','auxiliary.json')},
      model=None if not fresh['supported'] else identity(OUT/'model.json'),independent_metric_formulas=True,
      complete_final_prediction_replay_cases=1296,training_gradient_numeric_replay=False))


def warp_result(replay=False):
    import numpy as np
    from scripts.p6_joint_evaluate_v45 import aggregate_warp
    from scripts.p6_mixture_v43 import arrays_save,arrays_read
    need(read(LOG/'auxiliary_review.json')['supported'],'Auxiliary not supported')
    scope='calibration_final'+('_replay' if replay else '')
    from src import p6_local_correspondence_v50 as core
    x,y,meta=feature_arrays('calibration');head=read(OUT/'head.json')
    p=core.predict_warp(head,x,core.FEATURE_NAMES,guard=child_guard,on_event=neural_callback(scope));prior=np.asarray(head['empirical_prior'])
    rows=[dict(**r,warp_method=warp_metric(v,target),warp_prior=warp_metric(prior,target)) for r,v,target in zip(meta,p,y)]
    summary,gates=aggregate_warp(rows,split='calibration')
    if replay:need(np.array_equal(p,arrays_read(OUT/'calibration_posterior.npz')['probability']),'Cal432 posterior replay differs')
    else:arrays_save(OUT/'calibration_posterior.npz',probability=p)
    return dict(rows=rows,summary=summary,gates=gates,passed=all(gates.values()),model_sha256=sha(OUT/'model.json'))


def warp_calibration():once(OUT/'warp_calibration.json',call('warp_calibration','calibration432',lambda:warp_result(False)))


def warp_review():
    from scripts.p6_mixture_v43 import close_numeric
    fresh=call('warp_calibration_replay','calibration432',lambda:warp_result(True));close_numeric(fresh,read(OUT/'warp_calibration.json'))
    once(LOG/'warp_review.json',dict(at=now(),passed=True,outputs={name:identity(OUT/name) for name in ('warp_calibration.json','calibration_posterior.npz','model.json')}))


def evaluation_posterior(replay=False):
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_save,arrays_read
    scope='evaluation_final'+('_replay' if replay else '')
    p,y,rows=predict('evaluation',scope,'evaluation_posterior'+('_replay' if replay else ''))
    if replay:need(np.array_equal(p,arrays_read(OUT/'evaluation_posterior.npz')['probability']),'Evaluation432 posterior replay differs')
    else:arrays_save(OUT/'evaluation_posterior.npz',probability=p)
    return p


def full_evaluate(split,replay=False):
    import numpy as np
    from scripts.p6_joint_evaluate_v45 import evaluate_case,aggregate
    from scripts.p6_mixture_v43 import observation,independent_source,key_for,MODES,WARPS,close_numeric,arrays_read
    from scripts.p6_correspondence_v40 import load_frame
    need(read(OUT/'warp_calibration.json')['passed'] and read(LOG/'warp_review.json')['passed'],'Warp prerequisites failed')
    if split=='evaluation':need(read(OUT/'calibration.json')['passed'] and read(LOG/'calibration_review.json')['passed'],'Full calibration prerequisites failed')
    model=read(OUT/'model.json');prior=np.asarray(model['empirical_prior']);metadata=check_data_certificate(split)['rows']
    p=arrays_read(OUT/'calibration_posterior.npz')['probability'] if split=='calibration' else evaluation_posterior(replay)
    probabilities={r['key']:v for r,v in zip(metadata,p)};data=read(DATA);result=[];stage=split+('_review' if replay else '')
    for record in data['records'][split]:
        child_guard();frame=call(stage+'_frame',frame_name(record),lambda:load_frame(data,record,labels=True))
        for delta in WARPS:
            for mode in MODES:
                key=key_for(split,record,delta,mode)
                def calculate():
                    if replay:
                        a=observation(frame,record,delta,mode);b=independent_source(frame,record,delta,mode)
                        need(all(np.array_equal(v,w) for v,w in zip(a,b)),'Independent observation/source replay differs')
                    return evaluate_case(frame,record,delta,mode,probabilities[key],model['depth'],prior,model['shape'])
                metrics=call(stage+'_case',key,calculate)
                result.append(dict(key=key,split=split,scene=record['scene'],frame=record['frame'],warp=list(delta),mode=mode,**metrics));child_guard()
    summary,gates=aggregate(result,split=split)
    if split=='calibration':need(all(gates[k]==v for k,v in read(OUT/'warp_calibration.json')['gates'].items()),'Original warp gates changed in full cal')
    answer=dict(rows=result,summary=summary,gates=gates,passed=all(gates.values()),model_sha256=sha(OUT/'model.json'),scope=split+'432 known synthetic development cases')
    if replay:
        close_numeric(answer,read(OUT/(split+'.json')))
        once(LOG/(stage+'.json'),dict(at=now(),passed=True,outputs={name:identity(OUT/name) for name in (split+'.json',split+'_posterior.npz','model.json')},
          scope='Complete432 case original17metric replay and independent observation/source-ID reconstruction'))
    else:once(OUT/(split+'.json'),answer)


def run_plan(run):
    for stage in ('fit_data','fit_data_review','train','train_review','auxiliary_review'):run(stage)
    if read(OUT/'auxiliary.json')['supported']:
        for stage in ('calibration_data','calibration_data_review','warp_calibration','warp_review'):run(stage)
        if read(OUT/'warp_calibration.json')['passed']:
            for stage in ('calibration','calibration_review'):run(stage)
            if read(OUT/'calibration.json')['passed']:
                for stage in ('evaluation_data','evaluation_data_review','evaluation','evaluation_review'):run(stage)
    run('cold')


def executed_stages():
    stages=['fit_data','fit_data_review','train','train_review','auxiliary_review']
    if read(OUT/'auxiliary.json')['supported']:
        stages+=['calibration_data','calibration_data_review','warp_calibration','warp_review']
        if read(OUT/'warp_calibration.json')['passed']:
            stages+=['calibration','calibration_review']
            if read(OUT/'calibration.json')['passed']:stages+=['evaluation_data','evaluation_data_review','evaluation','evaluation_review']
    return stages


def verify_executed():
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_read,close_numeric
    from scripts.p6_joint_evaluate_v45 import aggregate_warp,aggregate
    stages=executed_stages();expected={};names=set();neural_expected={'rgb_forward':0,'train_forward':512,'train_backward':512,'train_update':512,'predict_forward':82}
    certificates=['train_review','auxiliary_review'];rgb_frames=0
    for split in ('fit','calibration','evaluation'):
        if split+'_data' not in stages:continue
        manifest=check_data_certificate(split);frames=48 if split=='fit' else 16;count=frames*27
        need(manifest['rows']==split_rows(split),'Frozen row order changed')
        names.add(split+'_data.json');names.update(manifest['artifacts'])
        for suffix in ('','_review'):expected.update({split+'_data'+suffix+'_frame':frames,split+'_data'+suffix+'_case':count})
        neural_expected['rgb_forward']+=2*frames;rgb_frames+=frames
    names.update(['head.json','training_report.json','training_partition.json','fit_posterior.npz','auxiliary.json'])
    expected.update(train_once=1,fit_posterior=1,fit_posterior_review=1)
    aux=read(OUT/'auxiliary.json');auxproof=read(LOG/'auxiliary_review.json')
    need(auxproof['passed'] and auxproof['supported']==aux['supported'] and auxproof['independent_metric_formulas'],'Auxiliary review differs')
    x,y,rows,scenes,train_ids,held=fixed_partition_arrays()
    fresh=auxiliary_result(arrays_read(OUT/'fit_posterior.npz')['probability'],y,rows,reference=True);close_numeric(fresh,aux)
    from src.p6_local_training_review_v50 import validate_training_report
    validate_training_report(read(OUT/'head.json'),read(OUT/'training_report.json'),x[train_ids],y[train_ids],
      events=events(LOG/'train_events.jsonl'),updates=events(LOG/'train_updates.jsonl'))
    gates_by_split={s:{name:None for name in GATES} for s in ('calibration','evaluation')}
    if aux['supported']:
        names.update(['model.json','calibration_posterior.npz','warp_calibration.json']);expected.update(warp_calibration=1,warp_calibration_replay=1)
        neural_expected['predict_forward']+=28;certificates.append('warp_review')
        model=read(OUT/'model.json');head=read(OUT/'head.json');parent=read(ROOT/'results/P6/joint_repair_v45/model.json')
        need(auxproof['model']==identity(OUT/'model.json') and model['warp']==head and model['empirical_prior']==head['empirical_prior']
          and all(model[k]==parent[k] for k in ('depth','shape')),'Model/prior/frozen geometry changed')
        for split in ['warp_calibration']+[s for s in ('calibration','evaluation') if s in stages]:
            result=read(OUT/(split+'.json'))
            summary,gates=aggregate_warp(result['rows'],split='calibration') if split=='warp_calibration' else aggregate(result['rows'],split=split)
            close_numeric(summary,result['summary']);need(gates==result['gates'] and result['passed']==all(gates.values()),'Original gates differ')
            if split!='warp_calibration':
                names.add(split+'.json');certificates.append(split+'_review')
                for suffix in ('','_review'):expected.update({split+suffix+'_frame':16,split+suffix+'_case':432})
            gates_by_split['calibration' if split=='warp_calibration' else split].update(gates)
        if 'evaluation' in stages:
            names.add('evaluation_posterior.npz');expected.update(evaluation_posterior=1,evaluation_posterior_replay=1);neural_expected['predict_forward']+=28
    for stage in certificates:
        proof=read(LOG/(stage+'.json'));need(proof['passed'],'Executed review not passed '+stage)
        for name,binding in proof['outputs'].items():need(identity(OUT/name)==binding,'Reviewed output changed '+name)
        for name,binding in proof.get('logs',{}).items():need(identity(LOG/name)==binding,'Reviewed trace changed '+name)
    calls=ledger();nn=ledger(LOG/'neural_calls.jsonl')
    need(not calls['unknown'] and calls['counts']==expected,'Scientific ledger count differs from executed scope')
    need(not nn['unknown'] and nn['counts']==neural_expected,'Actual neural invocation ledger differs from executed scope')
    children=events(LOG/'children.jsonl');returns=[r for r in children if r['event']=='returned']
    child_account=ledger(LOG/'children.jsonl');intents=[r for r in children if r['event']=='intent']
    need([r['key'] for r in intents] in (stages,stages+['cold']),'Unexpected or repeated child intent')
    if child_account['unknown']:
        pending=[r for r in intents if r['id'] in child_account['unknown']]
        need(len(pending)==1 and pending[0]['key']=='cold' and read(STATE)['phase']=='cold','Unknown child outside active cold review')
    need(all(r['exit_code']==0 for r in returns),'Failed child cannot be accepted')
    childkeys=[r['key'] for r in returns]
    need(childkeys==stages or childkeys==stages+['cold'],'Executed children do not match branch')
    actual={p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    need(actual-{'closeout.json'}==names,'Unreviewed/missing output artifact')
    return dict(stages=stages,calls=calls,neural_calls=nn,outputs={name:identity(OUT/name) for name in sorted(names)},
      auxiliary_supported=aux['supported'],gates_by_split=gates_by_split,new_head_fits=1,returned_optimizer_updates=512,
      training_numeric_gradient_replay=False,whole_pipeline_holdout=False,RGB_original_frames=rgb_frames,
      all17_executed='calibration' in stages,evaluation_executed='evaluation' in stages,
      development_scientific_passed='evaluation' in stages and read(OUT/'evaluation.json')['passed'])


def cold():
    proof=verify_executed();child_guard()
    once(LOG/'cold.json',dict(at=now(),passed=True,fresh_process_pid=os.getpid(),proof=proof,
      scope='Fresh process read-only complete executed inventory, all summary formulas, training evidence, original gates and ledgers; no new neural or optimizer calls'))


def finish(c,guard):
    proof=verify_executed();coldproof=read(LOG/'cold.json')
    need(coldproof['passed'] and coldproof['proof']==proof,'Cold/executed evidence differs');protected_check(c);guard()
    passed=proof['development_scientific_passed'];progress=[47,58.75] if passed else [45,56.25]
    last='evaluation' if proof['evaluation_executed'] else 'calibration' if proof['all17_executed'] else 'warp_calibration' if proof['auxiliary_supported'] else 'auxiliary'
    result=read(OUT/(last+'.json'));failures=[k for k,v in result['gates'].items() if not v]
    lines=['P6实例条件局部RGB-D对应概率模型 v50',now(),'',
      f'全部任务总进度{progress[0]}%；实验总进度{progress[1]}%。本固定候选已停止。',
      '原限定开发P6科学采用='+str(passed)+'；全部已执行内容完整工程复验及真实冷重入=True。',
      '最后阶段='+last+'；失败项='+str(failures),
      '一次864增广案例、8训练场景32原帧、固定512 Adam更新；final checkpoint only。F0四场景432案例只作一次准入，无新校准/验证选模型/第二次训练。',
      '1193参数：28通道→16→8的共享token MLP，3x3局部卷积、固定16区域平均→一个候选分数；81共享候选概率。',
      '新信息为保留空间区域、预测实例边界及观测表面条件；旧16特征已有3/7池化，不称首次多尺度。相对旧线性head也增加非线性容量，未通过本实验分离每项特征因果收益。',
      '分割器来自冻结P4v22r2纯RGB epoch33。它已训练见过全部20场景；只有新localhead的训练与F0场景隔离。depth/H及研究选型亦暴露这些开发场景，不称整体独立测试。',
      'F0须原六warp门槛全过，并相对冻结同fold旧head NLL改善>=0.01、Brier不劣；才依次原cal6→cal17→eval17。未执行指标保留null。',
      '整帧共享warp随机变量、固定训练π、旧depth/H、A/B几何传播、原XYZ/分母/划分/阈值不变。无GT/source/真实warp/mode进入预测特征；标签仅用于训练目标或评价。',
      '训练审查覆盖完整512步顺序、有限轨迹、事件、Adam末态及最终模型；已独立合成梯度/Adam检查，最终1296模型预测全重播。未重训512步，也不声称逐梯度数值全复验。',
      '当前不能作为完整P6科学结果采用；保存该新方法阴性结果，不自动换参数重跑。' if not passed else '可采用为原限定合成开发结果；真实传感器和下游抓取收益仍未由本轮证明。',
      '辅助结果='+json.dumps(read(OUT/'auxiliary.json'),ensure_ascii=False),
      '原科学门槛(null未运行)='+json.dumps(proof['gates_by_split'],ensure_ascii=False),
      '实际科学调用='+json.dumps(proof['calls'],ensure_ascii=False),
      '实际神经/反向/更新调用(intent/return/unknown分离)='+json.dumps(proof['neural_calls'],ensure_ascii=False),
      '预算='+json.dumps({k:c.get(k) for k in ('controller_wall_seconds','preparation_seconds','output_limit','budget_scope')},ensure_ascii=False),
      '仅当前协议usage可核实；历史各版本消耗原样保存，不以名义案例/帧次编造实际前向或全研究精确总耗时。',
      '不自动新版本/P7/P10/周期任务；等待用户查看整部分结果。']
    need(not REPORT.exists(),'Do not overwrite prior report');REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8');guard()
    once(OUT/'closeout.json',dict(at=now(),proof=proof,report=identity(REPORT),cold=identity(LOG/'cold.json'),stopped=True,
      engineering_passed=True,P6_development_scope_passed=passed,all_tasks_percent=progress[0],experiment_percent=progress[1],
      child_ledger=ledger(LOG/'children.jsonl')))
    status('completed_stopped',report=rel(REPORT),current_stage=last,scientific_passed=passed)
    if passed:
        st=read(STATE);st.update(all_tasks_percent=47,experiment_percent=58.75,P6_scientific_passed=True);atomic(STATE,st)
    supervise('v50_completed_stopped',current_runner=None,remaining_research_updates_cap=0,waiting_for_user_next_part_confirmation=True,
      summary_report=rel(REPORT),final_cold=rel(LOG/'cold.json'),final_readonly=rel(LOG/('auxiliary_review.json' if last=='auxiliary' else 'warp_review.json' if last=='warp_calibration' else last+'_review.json')),
      final_delivery_index=rel(OUT/'closeout.json'),all_tasks_percent_planning_estimate=progress[0],experiment_percent_planning_estimate=progress[1])
    handoff(f'P6 v50固定空间表面模型完成已执行范围完整复验和冷重入并停止。最后阶段{last}；F0准入={proof["auxiliary_supported"]}；原P6科学采用={passed}。全部任务总进度{progress[0]}%、实验总进度{progress[1]}%。报告{rel(REPORT)}；不自动新参数/版本/P7/P10/周期任务。','complete')


def deliver():
    started = time.monotonic()
    c = verify()
    with project_lock(ROOT), own_lock():
        need(not ledger()['unknown'] and not ledger(LOG/'children.jsonl')['unknown'] and not ledger(LOG/'neural_calls.jsonl')['unknown'], 'Unknown calls/children: audit before any recovery')
        if (OUT/'closeout.json').exists():
            done = read(OUT/'closeout.json'); need(identity(REPORT) == done['report'] and identity(LOG/'cold.json') == done['cold'], 'Completed receipt changed')
            need(verify_executed() == done['proof'], 'Completed inventory differs')
            need(ledger(LOG/'children.jsonl') == done['child_ledger'], 'Completed child ledger differs')
            print(json.dumps(dict(already_complete=True, no_new_research=True))); return
        need(not (LOG/'children.jsonl').exists(), 'Partial execution preserved: no automatic retry')
        prior = c['preparation_seconds']; run_token = uuid.uuid4().hex
        usage = LOG/('controller_usage_'+uuid.uuid4().hex+'.json'); peak = 0
        deadline = time.time()+c['controller_wall_seconds']-prior-(time.monotonic()-started)
        def guard():
            elapsed = time.monotonic()-started
            atomic(usage, dict(at=now(), seconds=elapsed, preparation_seconds=prior, cumulative_seconds=prior+elapsed, peak_child_tree_bytes=peak))
            need(prior+elapsed < c['controller_wall_seconds'], 'Cumulative7200s budget exhausted')
            need(not (LOG/'STOP').exists(), 'User STOP marker')
            need(output_bytes() <= c['output_limit'], 'Combined output cap')
            need(shutil.disk_usage(ROOT).free >= c['free_disk_min'], 'Free disk floor')
        def run(stage):
            nonlocal peak
            guard(); cid = uuid.uuid4().hex; key = stage
            env = os.environ.copy(); env['P6_V50_CHILD_TOKEN'] = run_token
            command = [str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-X', 'utf8', '-B', rel(Path(__file__)), 'worker', '--stage', stage]
            append(LOG/'children.jsonl', dict(at=now(), event='intent', id=cid, kind='worker', key=key, command=command))
            status(stage, controller_pid=os.getpid(), run_token=run_token, deadline_epoch=deadline)
            with (LOG/(stage+'.stdout.log')).open('xb') as out, (LOG/(stage+'.stderr.log')).open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
                append(LOG/'controller.jsonl', dict(at=now(), event='started', stage=stage, pid=child.pid))
                try:
                    while child.poll() is None:
                        memory, members = process_tree_rss(child.pid); peak = max(peak, memory)
                        need(memory <= c['process_tree_rss_limit'], '3GiB actual child-tree peak RSS cap')
                        guard(); time.sleep(1)
                except BaseException:
                    if child.poll() is None:
                        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: pass
                    append(LOG/'controller.jsonl', dict(at=now(), event='interrupted', stage=stage, pid=child.pid, exit_code=child.poll(), orphan_unconfirmed=child.poll() is None))
                    if child.poll() is not None:
                        append(LOG/'children.jsonl', dict(at=now(), event='returned', id=cid, kind='worker', key=key, exit_code=child.returncode, interrupted=True))
                    raise
            append(LOG/'children.jsonl', dict(at=now(), event='returned', id=cid, kind='worker', key=key, exit_code=child.returncode))
            need(child.returncode == 0, 'Worker failed '+stage); guard()
        try:
            supervise('v50_running')
            run_plan(run); guard(); finish(c, guard); guard()
        except BaseException as e:
            append(LOG/'controller.jsonl', dict(at=now(), event='failure_preserved', error=repr(e), traceback=traceback.format_exc()))
            status('failed_stopped', error=repr(e), unknown=ledger()['unknown'], unknown_children=ledger(LOG/'children.jsonl')['unknown'],unknown_neural=ledger(LOG/'neural_calls.jsonl')['unknown'])
            supervise('v50_failed_stopped', current_runner=None, waiting_for_user_next_part_confirmation=True)
            handoff('P6 v50出现工程或资源异常，失败日志/模型/未知返回保留，先核实际进程与锁再恢复，不自动重复。全部任务总进度45%、实验总进度56.25%；科学通过未成立，不进P7/P10。', 'failure')
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','verify','deliver','worker']);parser.add_argument('--stage');args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='verify':print(json.dumps(dict(passed=True,protected_files=len(verify()['protected']))))
    elif args.mode=='deliver':deliver()
    else:
        verify();child_guard();OUT.mkdir(parents=True,exist_ok=True)
        if args.stage in ('fit_data','calibration_data','evaluation_data'):data_generate(args.stage.removesuffix('_data'))
        elif args.stage in ('fit_data_review','calibration_data_review','evaluation_data_review'):data_generate(args.stage.removesuffix('_data_review'),True)
        elif args.stage=='train':train()
        elif args.stage=='train_review':train_review()
        elif args.stage=='auxiliary_review':auxiliary_review()
        elif args.stage=='warp_calibration':warp_calibration()
        elif args.stage=='warp_review':warp_review()
        elif args.stage in ('calibration','evaluation'):full_evaluate(args.stage)
        elif args.stage in ('calibration_review','evaluation_review'):full_evaluate(args.stage.removesuffix('_review'),True)
        elif args.stage=='cold':cold()
        else:raise ValueError('Unknown worker stage')
        child_guard()
if __name__=='__main__':main()
