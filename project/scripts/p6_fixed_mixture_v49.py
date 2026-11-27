"""One fixed eight-scene predictor and21-point mixture calibration; frozen science gates.

Frozen v43 scientific thresholds and observations are reused, never edited.
No scheduler; no neural inference, AP, depth-scale refit or automatic retry.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import sys, json, time, uuid, shutil, argparse, subprocess, traceback
from pathlib import Path
from contextlib import contextmanager
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts.p6_mixture_common_v43 import now, need, read, identity, sha, atomic, once, append
from scripts.research_cache_recovery_v44r1 import process_tree_rss
from scripts.pipeline import project_lock
LOG = ROOT/'logs/P6/fixed_mixture_v49'
OUT = ROOT/'results/P6/fixed_mixture_v49'
CONFIG = ROOT/'configs/p6_fixed_mixture_v49.json'
STATE = ROOT/'runtime/pipeline/p6_fixed_mixture_v49.json'
PROTOCOL = ROOT/'docs/P6固定模型独立混合校准协议_v49.txt'
REPORT = ROOT/'docs/P6固定模型独立混合校准结果_v49.txt'
OLD = ROOT/'results/P6/mixture_v43'
PARENT = ROOT/'results/P6/probability_calibration_v46'
PREVIOUS = ROOT/'results/P6/geometric_recovery_v48r1'
DATA = ROOT/'logs/P6/correspondence_v40/data_audit.json'
ORIGINAL_STATE = 'b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'
SOURCES = ['scripts/p6_fixed_mixture_v49.py', 'src/p6_fixed_mixture_v49.py',
           'tests/test_p6_fixed_mixture_v49.py', 'tests/test_p6_fixed_mixture_independent_v49.py',
           'scripts/check_p6_fixed_mixture_v49.py']
GATES = ('warp_nll_gain', 'warp_brier_noninferior', 'warp95_coverage', 'warp_set_sharper',
         'warp_multilevel_reliability', 'worst_scene_warp', 'worst_scene_depth',
         'source_depth95', 'source_depth_sharpness', 'source_depth_nll_gain',
         'target_joint_nll_noninferior', 'target_missing_brier_noninferior',
         'target_conditional_coverage', 'source_joint95', 'target_union_sharpness',
         'target_union95', 'observation_preserved')
LIMITS = dict(selection_grid=1, selection_grid_replay=1,
              warp_calibration=1, warp_calibration_replay=1,
              calibration_posterior=1, calibration_review_posterior=1,
              evaluation_posterior=1, evaluation_review_posterior=1,
              calibration_frame=16, calibration_review_frame=16,
              evaluation_frame=16, evaluation_review_frame=16,
              calibration_case=432, calibration_review_case=432,
              evaluation_case=432, evaluation_review_case=432)

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

_seen = None
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
                      current_part='P6_fixed_mixture_v49', P6_scientific_passed=False,
                      P7_started=False, P10_started=False, recurring_automation=False, **fields))
def supervise(phase, **fields):
    path = ROOT/'runtime/pipeline/part_level_supervision.json'; backup(path, phase)
    value = read(path)
    value.update(recorded_at=now(), current_authorized_part='P6_fixed_mixture_v49',
        current_execution_state=phase, current_runner=rel(Path(__file__)),
        current_execution_evidence=rel(STATE), next_repair_started=True,
        user_authorization='用户授权固定fold0预测器及一次21点混合监督校准，保持head与先验一致，完整分支复验后停止',
        automatic_resume_authorized=False, automatic_scheduler_recreated=False,
        stop_after='P6_fixed_mixture_v49', stop_after_part='P6_fixed_mixture_v49',
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
    path = ROOT/'runtime/pipeline/p6_fixed_mixture_v49.lock'
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
    need(os.environ.get('P6_V49_CHILD_TOKEN') == state.get('run_token') and bool(state.get('run_token')), 'Worker requires active controller')
    need(time.time() < state['deadline_epoch'], 'Cumulative compute wall budget exhausted')
    need(not (LOG/'STOP').exists(), 'User STOP marker')
    need(output_bytes() <= 512*1024**2, 'Combined output cap exceeded')

def freeze():
    started = time.monotonic()
    with project_lock(ROOT), own_lock():
        need(not CONFIG.exists(), 'Already frozen; use verify/deliver')
        checks = read(LOG/'internal_checks.json'); review = read(LOG/'independent_review.json')
        for receipt in (checks, review):
            need(receipt['passed'], 'Internal checks or independent code review failed')
            for name, digest in receipt['source_sha256'].items():
                need(sha(ROOT/name) == digest, 'Reviewed source changed '+name)
        need(set(SOURCES) <= set(checks['source_sha256']) and set(SOURCES) <= set(review['source_sha256']), 'Review omitted source')
        from scripts import p6_geometric_recovery_v48r1 as previous
        oldconfig=previous.verify();done=read(PREVIOUS/'closeout.json')
        need(previous.verify_executed()==done['proof'] and done['engineering_passed'],'Prior completed recovery receipt differs')
        need(done['cold']==identity(previous.LOG/'cold.json') and done['report']==identity(previous.REPORT),'Prior cold/report identity differs')
        need(done['child_ledger']==previous.ledger(previous.LOG/'children.jsonl'),'Prior child ledger differs')
        protected=dict(oldconfig['protected'])
        dependencies=SOURCES+[rel(PROTOCOL),'docs/P6固定模型独立混合校准独立审查_v49.txt',
            'configs/p6_geometric_recovery_v48r1.json',rel(previous.REPORT),rel(previous.STATE),
            rel(LOG/'internal_checks.json'),rel(LOG/'independent_review.json')]
        dependencies += [rel(p) for folder in (PARENT,PREVIOUS,previous.LOG) for p in folder.rglob('*') if p.is_file()]
        historical_usage={}
        for name in ('joint_repair_v45','probability_calibration_v46','quality_calibration_v47','geometric_logits_v48','geometric_recovery_v48r1'):
            folder=ROOT/'logs/P6'/name
            records={rel(p):dict(identity=identity(p),record=read(p)) for p in folder.glob('controller_usage_*.json')}
            dependencies+=list(records)
            historical_usage[name]=dict(records=records,missing_or_unknown_usage=not bool(records),
                additive_total_claimed=False,note='Records may already include prior attempts; do not add overlapping or unknown counts')
        protected.update({name: identity(ROOT/name) for name in dependencies})
        measured = sum(float(read(p)['seconds']) for p in (LOG/'preparation').glob('*.json'))
        c = dict(at=now(), version='v49', protected=protected, call_limits=LIMITS,
            controller_wall_seconds=3600, preparation_seconds=measured+30., cpu_threads=2,
            process_tree_rss_limit=3*1024**3, output_limit=512*1024**2, free_disk_min=60*1024**3,
            original_gates_unchanged=True, train_cases=1296, calibration_cases=432, exposed_evaluation_cases=432,
            folds=[[3,8,12,17], [4,9,14,19], [5,10,16,20]],
            budget_scope='New independently authorized protocol; history preserved, not a reset of total research consumption',
            historical_usage=historical_usage,
            selected_head=dict(fold_index=0,training_scenes=[4,5,9,10,14,16,19,20],auxiliary_scenes=[3,8,12,17],
                training_rows=864,auxiliary_rows=432,source=rel(PARENT/'fold_0.json'),new_head_fits=0),
            selection=dict(lambda_grid=[i/20 for i in range(21)],supervised_calibration_selections=1,
                candidate_metric_evaluations=21,replay_candidate_metric_evaluations=21,
                gate_filter='All original warp gates except multilevel reliability must pass',
                objective='Minimize maximum absolute coverage versus selected mass gap over50/80/95',
                tie_rule='Smaller lambda',acceptance_max_gap=.05,gradient_optimizer_calls=0),
            reused=dict(shape_fits=0,depth_scale_refits=0,head_refits=0,parent_geometry=rel(ROOT/'results/P6/joint_repair_v45/model.json'),
                mixture_prior_and_all_prior_baselines='fixed fold0 training empirical prior',
                auxiliary_is_head_holdout_not_whole_pipeline_holdout=True),
            stages='fixed auxiliary grid21+complete replay21 -> if supported same-head cal6+replay -> if pass cal17+replay -> if pass eval17+replay -> cold -> report -> stop',
            failed_gate_action='Stop this candidate after full replay of executed scope; do not tune or run the deferred expensive scope',
            scope='Known synthetic global integer warps on previously exposed GraspNet RealSense development scenes; not independent final test',
            neural_forwards=0, official_AP=0, CAD_renders=0, grasp_calls=0,
            progress_before=[45, 56.25], progress_after_full_pass=[47, 58.75])
        protected_check(c)
        need(c['preparation_seconds'] < 3600 and shutil.disk_usage(ROOT).free >= c['free_disk_min'], 'Insufficient reserved resources')
        once(CONFIG, c); once(LOG/'freeze.json', dict(at=now(), config=identity(CONFIG)))
        need(time.monotonic()-started <= 30, 'Freeze reserve exceeded')
        status('frozen_ready'); supervise('v49_frozen_ready')
        handoff('P6 v49固定预测器混合校准已冻结：只用旧fold0八场景head，F0四场景432辅助案例选择一次21点lambda。21指标评价和完整21复验不等于21个模型训练；head重训=0，监督校准选择=1。通过后同一head和其训练先验用于原cal6→cal17→eval17；失败收尾停止。新协议独立3600秒预算，历史消耗保留不归零。全部任务总进度45%、实验总进度56.25%。入口scripts/p6_fixed_mixture_v49.py deliver；禁止重复。','freeze')
        print(json.dumps(dict(frozen=True, remaining_seconds=3600-c['preparation_seconds'])))

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


def candidate_callback(filename):
    def save(row):append(LOG/filename,dict(at=now(),event='candidate_returned',**row))
    return save


def selection():
    from src import p6_fixed_mixture_v49 as core
    scores,priors,y,scenes,rows,head,parent,provenance=selection_data()
    once(OUT/'selection_provenance.json',provenance);once(OUT/'auxiliary_rows.json',rows)
    try:
        report=call('selection_grid','fixed_F0_grid21',lambda:core.calibrate_fixed_mixture(
            scores,priors,y,scenes,guard=child_guard,on_candidate=candidate_callback('selection_candidates.jsonl')))
    except BaseException as error:
        once(OUT/'selection_failure.json',dict(at=now(),error=repr(error),partial_report=getattr(error,'partial_report',None),traceback=traceback.format_exc()))
        raise
    once(OUT/'selection.json',report);child_guard()


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


def reference_decision(candidates):
    need(len(candidates)==21 and [r['index'] for r in candidates]==list(range(21))
         and [r['lambda_value'] for r in candidates]==[i/20 for i in range(21)],'Incomplete or changed fixed grid')
    feasible=[r for r in candidates if all(v for k,v in r['gates'].items() if k!='warp_multilevel_reliability')]
    selected=min(feasible,key=lambda r:(r['max_reliability_gap'],r['lambda_value'])) if feasible else None
    return dict(selected_index=None if selected is None else selected['index'],selected_lambda=None if selected is None else selected['lambda_value'],
        supported=selected is not None and selected['max_reliability_gap']<=.05,selected=selected)


def reference_selection(scores,priors,y,scenes):
    import numpy as np
    from scipy.special import logsumexp
    logits=np.asarray(scores)+np.log(priors);head=np.exp(logits-logsumexp(logits,axis=1,keepdims=True));candidates=[]
    for index in range(21):
        child_guard();weight=index/20
        probability=head.copy() if index==0 else priors.copy() if index==20 else (1-weight)*head+weight*priors
        row=dict(index=index,lambda_value=weight,**reference_candidate(probability,priors,y,scenes))
        candidates.append(row);candidate_callback('selection_reference_candidates.jsonl')(row)
    return dict(candidates=candidates,**reference_decision(candidates))


def assembled_model(head,parent,calibration):
    from src import p6_fixed_mixture_v49 as core
    head_identity=dict(path=rel(PARENT/'fold_0.json'),identity=identity(PARENT/'fold_0.json'))
    return dict(warp=core.wrap_model(head,calibration,head_identity=head_identity),calibration=calibration,
        shape=parent['shape'],depth=parent['depth'],empirical_prior=head['empirical_prior'],
        fixed_head_identity=head_identity,parent_geometry_identity=identity(ROOT/'results/P6/joint_repair_v45/model.json'),
        supervised_calibration_selections=1,new_head_fits=0,depth_scale_refits=0,shape_fits=0,
        same_head_prior_in_all_method_and_prior_baselines=True,auxiliary_head_holdout_only=True,GT_in_predictor=False)


def selection_review():
    import numpy as np
    from scripts.p6_mixture_v43 import close_numeric
    scores,priors,y,scenes,rows,head,parent,provenance=selection_data();report=read(OUT/'selection.json')
    need(provenance==read(OUT/'selection_provenance.json') and rows==read(OUT/'auxiliary_rows.json'),'Auxiliary provenance changed')
    need(report['completed'] and report['candidate_evaluations']==21 and report['fixed_fold_index']==0
         and report['mixture_grid']==[i/20 for i in range(21)] and np.array_equal(report['empirical_prior'],head['empirical_prior']),'Selection contract differs')
    reference=call('selection_grid_replay','independent_F0_grid21',lambda:reference_selection(scores,priors,y,scenes))
    close_numeric(report['candidates'],reference['candidates'])
    for field in ('selected_index','selected_lambda','selected','supported'):
        close_numeric(report[field],reference[field])
    need(report['passed']==reference['supported'],'Independent decision differs')
    for filename,candidates in [('selection_candidates.jsonl',report['candidates']),('selection_reference_candidates.jsonl',reference['candidates'])]:
        need([{k:v for k,v in r.items() if k not in ('at','event')} for r in events(LOG/filename)]==candidates,'Durable candidate returns differ')
    if report['supported']:once(OUT/'model.json',dict(at=now(),**assembled_model(head,parent,report)))
    names=['selection.json','selection_provenance.json','auxiliary_rows.json']+(['model.json'] if report['supported'] else [])
    child_guard();once(LOG/'selection_review.json',dict(at=now(),passed=True,supported=report['supported'],
        outputs={name:identity(OUT/name) for name in names},
        candidate_logs={name:identity(LOG/name) for name in ('selection_candidates.jsonl','selection_reference_candidates.jsonl')},
        independent_formula_certificate=True,production_candidate_metric_evaluations=21,reference_candidate_metric_evaluations=21,
        complete_actual_auxiliary_scope=True,supervised_calibration_selections=1,new_head_fits=0,
        scope='All21 probabilities, scientific metric formulas, fixed feasible set, maximum-gap minimum and exact tie rule independently replayed; selected wrapper assembled once after acceptance'))


def warp_metric(weights, target):
    from src import p6_fixed_mixture_v49 as core
    value = core.warp_statistics(weights, int(target))
    value['levels'] = {str(v['level']): dict(covered=v['covered'], size=v['size'], mass=v['probability'])
                       for v in value.pop('credible_sets')}
    return value

def warp_result(replay=False):
    import numpy as np
    from src import p6_fixed_mixture_v49 as core
    from scripts.p6_joint_evaluate_v45 import aggregate_warp
    proof=read(LOG/'selection_review.json')
    need(proof['passed'] and proof['supported'] and proof['outputs']['model.json']==identity(OUT/'model.json'),'Accepted fixed model prerequisite differs')
    arrays, meta = cache(); model = read(OUT/'model.json')
    selected = np.array([r['split'] == 'calibration' for r in meta])
    p = core.predict_warp(model['warp'], arrays['X'][selected]); prior = np.array(model['empirical_prior'])
    rows = [dict(**r, warp_method=warp_metric(prob, target), warp_prior=warp_metric(prior, target))
        for r, prob, target in zip([r for r in meta if r['split'] == 'calibration'], p, arrays['y'][selected])]
    summary, gates = aggregate_warp(rows, split='calibration')
    answer = dict(rows=rows, summary=summary, gates=gates, passed=all(gates.values()),
                  scope='432 calibration cached-feature cases; six original warp gates only', model_sha256=sha(OUT/'model.json'))
    child_guard()
    if not replay:
        from scripts.p6_mixture_v43 import arrays_save
        arrays_save(OUT/'calibration_warp.npz', probability=p)
    else:
        from scripts.p6_mixture_v43 import arrays_read
        need(np.array_equal(p, arrays_read(OUT/'calibration_warp.npz')['probability']), 'Posterior replay differs')
    return answer
def warp_calibration():
    result = call('warp_calibration', 'calibration432', lambda: warp_result(False))
    once(OUT/'warp_calibration.json', result)
def warp_review():
    from scripts.p6_mixture_v43 import close_numeric
    need(read(LOG/'selection_review.json')['passed'], 'Independent selection certificate prerequisite failed')
    fresh = call('warp_calibration_replay', 'calibration432', lambda: warp_result(True))
    close_numeric(fresh, read(OUT/'warp_calibration.json'))
    once(LOG/'warp_review.json', dict(at=now(), passed=True, result=identity(OUT/'warp_calibration.json'),
        posterior=identity(OUT/'calibration_warp.npz'), model=identity(OUT/'model.json'),
        scope='Complete deterministic 432 posterior/metric replay; fixed-head auxiliary calibration and independent formula certificate verified'))

def full_evaluate(split, replay=False):
    import numpy as np
    from src import p6_fixed_mixture_v49 as core
    from scripts.p6_joint_evaluate_v45 import evaluate_case, aggregate
    from scripts.p6_mixture_v43 import observation, independent_source, key_for, MODES, WARPS, close_numeric
    from scripts.p6_correspondence_v40 import load_frame
    need(read(OUT/'warp_calibration.json')['passed'] and read(LOG/'warp_review.json')['passed'], 'Warp prerequisite failed')
    if split == 'evaluation':
        need(read(OUT/'calibration.json')['passed'] and read(LOG/'calibration_review.json')['passed'], 'Full calibration prerequisite failed')
    arrays, metadata = cache(); model = read(OUT/'model.json'); prior = np.array(model['empirical_prior'])
    stage = split+('_review' if replay else '')
    select = np.array([r['split'] == split for r in metadata])
    p = call(stage+'_posterior', split+'432', lambda: core.predict_warp(model['warp'], arrays['X'][select]))
    if split == 'calibration':
        from scripts.p6_mixture_v43 import arrays_read
        need(np.array_equal(p, arrays_read(OUT/'calibration_warp.npz')['probability']), 'Full-cal posterior differs from early gate')
    filtered = [r for r in metadata if r['split'] == split]; probabilities = {r['key']: v for r, v in zip(filtered, p)}
    data = read(DATA); result = []; stage = split+('_review' if replay else '')
    for record in data['records'][split]:
        child_guard()
        frame = call(stage+'_frame', f"{record['scene']}/{record['frame']}", lambda: load_frame(data, record, labels=True))
        for delta in WARPS:
            for mode in MODES:
                key = key_for(split, record, delta, mode)
                def calculate():
                    if replay:
                        a = observation(frame, record, delta, mode); b = independent_source(frame, record, delta, mode)
                        need(all(np.array_equal(x, y) for x, y in zip(a, b)), 'Independent corruption/source-ID replay differs')
                    return evaluate_case(frame, record, delta, mode, probabilities[key], model['depth'], prior, model['shape'])
                metrics = call(stage+'_case', key, calculate)
                result.append(dict(key=key, split=split, scene=record['scene'], frame=record['frame'], warp=list(delta), mode=mode, **metrics))
                child_guard()
    summary, gates = aggregate(result, split=split)
    if split == 'calibration':
        early = read(OUT/'warp_calibration.json')
        need(all(gates[k] == v for k, v in early['gates'].items()), 'Full-cal warp gates differ from early gate')
    answer = dict(rows=result, summary=summary, gates=gates, passed=all(gates.values()),
        model_sha256=sha(OUT/'model.json'), scope=split+' 432 known-synthetic development cases')
    if replay:
        close_numeric(answer, read(OUT/(split+'.json')))
        once(LOG/(stage+'.json'), dict(at=now(), passed=True, result=identity(OUT/(split+'.json')),
             scope='Full deterministic numeric replay and independent observation/source-ID reconstruction'))
    else: once(OUT/(split+'.json'), answer)

def executed_stages():
    stages=['selection','selection_review']
    if not read(OUT/'selection.json')['supported']:return stages
    stages+=['warp_calibration','warp_review']
    if read(OUT/'warp_calibration.json')['passed']:
        stages+=['calibration','calibration_review']
        if read(OUT/'calibration.json')['passed']:stages+=['evaluation','evaluation_review']
    return stages


def verify_executed():
    from scripts.p6_joint_evaluate_v45 import aggregate_warp,aggregate
    from scripts.p6_mixture_v43 import close_numeric
    stages=executed_stages();report=read(OUT/'selection.json');review=read(LOG/'selection_review.json')
    expected=dict(selection_grid=1,selection_grid_replay=1)
    need(review['passed'] and review['supported']==report['supported'] and review['independent_formula_certificate'],'Auxiliary complete certificate differs')
    decision=reference_decision(report['candidates'])
    need(report['completed'] is True and report['candidate_evaluations']==21 and report['passed']==decision['supported'],'Auxiliary decision/completion differs')
    for field in ('selected_index','selected_lambda','selected','supported'):close_numeric(report[field],decision[field])
    names={'selection.json','selection_provenance.json','auxiliary_rows.json'}|({'model.json'} if report['supported'] else set())
    need(set(review['outputs'])==names,'Auxiliary reviewed output inventory incomplete')
    for name,binding in review['outputs'].items():need(identity(OUT/name)==binding,'Reviewed auxiliary artifact changed '+name)
    need(set(review['candidate_logs'])=={'selection_candidates.jsonl','selection_reference_candidates.jsonl'},'Candidate trace inventory incomplete')
    for name,binding in review['candidate_logs'].items():
        need(identity(LOG/name)==binding,'Reviewed candidate log changed')
        returned=events(LOG/name)
        need(len(returned)==21 and all(r['event']=='candidate_returned' for r in returned),'Actual candidate return count differs')
        close_numeric([{k:v for k,v in r.items() if k not in ('at','event')} for r in returned],report['candidates'])
    provenance=read(OUT/'selection_provenance.json')
    for name,binding in provenance['inputs'].items():need(identity(ROOT/name)==binding,'Frozen auxiliary input changed '+name)
    need(provenance['inherited_complete_oof_review']==identity(ROOT/'logs/P6/probability_calibration_v46/oof_review.json'),'Inherited OOF review changed')
    gates_by_split={split:{name:None for name in GATES} for split in ('calibration','evaluation')}
    if report['supported']:
        expected.update(warp_calibration=1,warp_calibration_replay=1);names|={'calibration_warp.npz','warp_calibration.json'}
        model=read(OUT/'model.json');head=read(PARENT/'fold_0.json')['head'];parent=read(ROOT/'results/P6/joint_repair_v45/model.json')
        close_numeric({k:v for k,v in model.items() if k!='at'},assembled_model(head,parent,report))
        need(model['empirical_prior']==head['empirical_prior'] and all(model[k]==parent[k] for k in ('shape','depth')),'Fixed eight-scene prior or parent geometry changed')
        warp_proof=read(LOG/'warp_review.json');need(warp_proof['passed'],'Complete cal6 replay not passed')
        for field,name in [('result','warp_calibration.json'),('posterior','calibration_warp.npz'),('model','model.json')]:
            need(warp_proof[field]==identity(OUT/name),'Reviewed cal6 artifact changed')
        for split in ['warp_calibration']+[v for v in ('calibration','evaluation') if v in stages]:
            result=read(OUT/(split+'.json'))
            summary,gates=aggregate_warp(result['rows'],split='calibration') if split=='warp_calibration' else aggregate(result['rows'],split=split)
            close_numeric(summary,result['summary']);need(gates==result['gates'] and result['passed']==all(gates.values()),'Original scientific metric/gates differ')
            if split!='warp_calibration':
                names.add(split+'.json');full_review=read(LOG/(split+'_review.json'))
                need(full_review['passed'] and full_review['result']==identity(OUT/(split+'.json')),'Full original-scope replay differs')
                for stage in (split,split+'_review'):expected.update({stage+'_posterior':1,stage+'_frame':16,stage+'_case':432})
        gates_by_split['calibration'].update(read(OUT/'warp_calibration.json')['gates'])
        for split in ('calibration','evaluation'):
            if split in stages:
                actual=read(OUT/(split+'.json'))['gates'];need(set(actual)==set(GATES),'Original17 gate names differ')
                gates_by_split[split].update(actual)
    calls=ledger();need(not calls['unknown'] and calls['counts']==expected,'Executed ledger differs')
    actual={p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    need(actual-{'closeout.json'}==names,'Unexpected/unreviewed output artifact')
    return dict(stages=stages,calls=calls,outputs={name:identity(OUT/name) for name in sorted(names)},
        auxiliary_supported=report['supported'],selected_lambda=report['selected_lambda'],gates_by_split=gates_by_split,
        supervised_calibration_selections=1,production_auxiliary_candidate_metrics=21,reference_auxiliary_candidate_metrics=21,
        new_head_fits=0,model_assembled=report['supported'],all17_executed='calibration' in stages,evaluation_executed='evaluation' in stages,
        development_scientific_passed='evaluation' in stages and read(OUT/'evaluation.json')['passed'])


def cold():
    proof = verify_executed(); child_guard()
    once(LOG/'cold.json', dict(at=now(), passed=True, fresh_process_pid=os.getpid(),
        scope='Read-only full executed inventory, summaries, original gates and call ledger', proof=proof))

def finish(c,guard):
    proof=verify_executed();cold_proof=read(LOG/'cold.json')
    need(cold_proof['passed'] and cold_proof['proof']==proof,'Cold/executed inventory differs')
    protected_check(c);guard();passed=proof['development_scientific_passed'];progress=[47,58.75] if passed else [45,56.25]
    last=('evaluation' if proof['evaluation_executed'] else 'calibration' if proof['all17_executed'] else 'warp_calibration' if proof['model_assembled'] else 'selection')
    result=read(OUT/(last+'.json'));failures=[k for k,v in result.get('gates',{}).items() if not v]
    selected=read(OUT/'selection.json')
    if last=='selection':failures=['no_feasible_candidate'] if selected['selected'] is None else ['selected_multilevel_reliability']
    lines=['P6固定模型的单次场景隔离混合校准 v49',now(),'',
        f'全部任务总进度{progress[0]}%；实验总进度{progress[1]}%。本固定候选已结束并停止。',
        '科学采用（原限定合成开发范围）='+str(passed)+'；实际执行完整范围工程复验及真实冷重入=True。',
        '最后阶段='+last+'；失败项='+str(failures),
        '辅助21点选择支持='+str(proof['auxiliary_supported'])+'；所选lambda='+str(proof['selected_lambda']),
        '原cal17已执行='+str(proof['all17_executed'])+'；原开发eval17已执行='+str(proof['evaluation_executed']),
        '固定旧v46 fold0八场景评分head：[4,5,9,10,14,16,19,20]864训练案例；辅助F0=[3,8,12,17]432案例。只复用原完整验收OOF分数，不重新head训练或OOF预测。',
        '辅助与本次warphead训练场景隔离；连续深度/H已见过全部12fit包括F0，研究也曾暴露这些场景，不称联合模型整体未见或最终独立测试。',
        '全链保持同一8场景head、同一fold0训练先验π和同一lambda。模型empirical_prior与所有先验基线同π；原12场景depth/H及XYZ传播不变。',
        'p=(1-lambda)*p_head+lambda*π。固定21个j/20候选，先筛原其他五warp约束，再最小化50/80/95最大质量-覆盖偏差；完全相等选较小lambda。所选偏差必须<=0.05；失败不细化网格、不换fold、不重训。',
        '这是一轮有监督校准选择，生产21候选级指标评价+独立公式完整重播21；不是没有拟合/选择，也不是21个训练模型。0新head训练、0新神经训练/前向、0官方AP/CAD/抓取。',
        '辅助选择通过只准入原cal6；其全过才cal17，再全过才eval17，全部已执行内容真实复验。未运行科学关口保持null。',
        '当前不能作为完整P6科学结果采用。' if not passed else '仅通过原限定开发协议，真实传感器校准及抓取收益仍需后续验证。',
        c['scope'],
        '辅助选择结果='+json.dumps({k:v for k,v in selected.items() if k!='candidates'},ensure_ascii=False),
        '全部原科学门槛（null=未运行）='+json.dumps(proof['gates_by_split'],ensure_ascii=False),
        '实际调用账本='+json.dumps(proof['calls'],ensure_ascii=False),
        '本次新独立预算='+json.dumps({k:c.get(k) for k in ('controller_wall_seconds','preparation_seconds','budget_scope')},ensure_ascii=False),
        '可取得历史usage原记录（含准备/冻结预留，可能重叠，不加总或声称全研究精确耗时）='+json.dumps(c.get('historical_usage',{}),ensure_ascii=False),
        '原P5及历史失败/未知结论不改，全部冻结输入/旧版本保留。只更新新v49状态，不自动进入v50/P7/P10或周期任务；等待用户检查。']
    guard();need(not REPORT.exists(),'Existing report must not be overwritten');REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8');guard()
    once(OUT/'closeout.json',dict(at=now(),proof=proof,report=identity(REPORT),cold=identity(LOG/'cold.json'),stopped=True,
        engineering_passed=True,P6_development_scope_passed=passed,all_tasks_percent=progress[0],experiment_percent=progress[1],
        neural_forwards=0,official_AP=0,child_ledger=ledger(LOG/'children.jsonl')))
    status('completed_stopped',report=rel(REPORT),current_stage=last,scientific_passed=passed)
    if passed:
        state=read(STATE);state.update(all_tasks_percent=47,experiment_percent=58.75,P6_scientific_passed=True);atomic(STATE,state)
    supervise('v49_completed_stopped',current_runner=None,remaining_research_updates_cap=0,waiting_for_user_next_part_confirmation=True,
        summary_report=rel(REPORT),final_cold=rel(LOG/'cold.json'),
        final_readonly=rel(LOG/'selection_review.json') if last=='selection' else rel(LOG/'warp_review.json') if last=='warp_calibration' else rel(LOG/(last+'_review.json')),
        final_delivery_index=rel(OUT/'closeout.json'),all_tasks_percent_planning_estimate=progress[0],experiment_percent_planning_estimate=progress[1])
    handoff(f'P6 v49固定8场景head的21点混合监督校准完成已执行范围、独立公式复验及冷重入并停止。最后阶段{last}；辅助支持={proof["auxiliary_supported"]}；原P6科学采用={passed}。全部任务总进度{progress[0]}%、实验总进度{progress[1]}%。报告{rel(REPORT)}。不自动v50/P7/P10或周期任务。','complete')


def deliver():
    started = time.monotonic()
    c = verify()
    with project_lock(ROOT), own_lock():
        need(not ledger()['unknown'] and not ledger(LOG/'children.jsonl')['unknown'], 'Unknown calls/children: audit before any recovery')
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
            need(prior+elapsed < c['controller_wall_seconds'], 'Cumulative3600s budget exhausted')
            need(not (LOG/'STOP').exists(), 'User STOP marker')
            need(output_bytes() <= c['output_limit'], 'Combined output cap')
            need(shutil.disk_usage(ROOT).free >= c['free_disk_min'], 'Free disk floor')
        def run(stage):
            nonlocal peak
            guard(); cid = uuid.uuid4().hex; key = stage
            env = os.environ.copy(); env['P6_V49_CHILD_TOKEN'] = run_token
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
            supervise('v49_running')
            for stage in ('selection', 'selection_review'): run(stage)
            if read(OUT/'selection.json')['supported']:
                for stage in ('warp_calibration', 'warp_review'): run(stage)
                if read(OUT/'warp_calibration.json')['passed']:
                    for stage in ('calibration', 'calibration_review'): run(stage)
                    if read(OUT/'calibration.json')['passed']:
                        for stage in ('evaluation', 'evaluation_review'): run(stage)
            run('cold'); guard(); finish(c, guard); guard()
        except BaseException as e:
            append(LOG/'controller.jsonl', dict(at=now(), event='failure_preserved', error=repr(e), traceback=traceback.format_exc()))
            status('failed_stopped', error=repr(e), unknown=ledger()['unknown'], unknown_children=ledger(LOG/'children.jsonl')['unknown'])
            supervise('v49_failed_stopped', current_runner=None, waiting_for_user_next_part_confirmation=True)
            handoff('P6 v49出现工程或资源异常，失败日志/模型/未知返回保留，先核实际进程与锁再恢复，不自动重复。全部任务总进度45%、实验总进度56.25%；科学通过未成立，不进P7/P10。', 'failure')
            raise

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'verify', 'deliver', 'worker'])
    parser.add_argument('--stage'); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'verify': print(json.dumps(dict(passed=True, protected_files=len(verify()['protected']))))
    elif args.mode == 'deliver': deliver()
    else:
        verify(); child_guard(); OUT.mkdir(parents=True, exist_ok=True)
        if args.stage == 'selection': selection()
        elif args.stage == 'selection_review': selection_review()
        elif args.stage == 'warp_calibration': warp_calibration()
        elif args.stage == 'warp_review': warp_review()
        elif args.stage in ('calibration', 'evaluation'): full_evaluate(args.stage)
        elif args.stage in ('calibration_review', 'evaluation_review'): full_evaluate(args.stage.removesuffix('_review'), True)
        elif args.stage == 'cold': cold()
        else: raise ValueError('Unknown worker stage')
        child_guard()
if __name__ == '__main__': main()
