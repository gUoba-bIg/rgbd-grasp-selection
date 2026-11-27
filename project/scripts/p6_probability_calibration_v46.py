"""One grouped-OOF probability calibration, with fixed gates and full replay.

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
LOG = ROOT/'logs/P6/probability_calibration_v46'
OUT = ROOT/'results/P6/probability_calibration_v46'
CONFIG = ROOT/'configs/p6_probability_calibration_v46.json'
STATE = ROOT/'runtime/pipeline/p6_probability_calibration_v46.json'
PROTOCOL = ROOT/'docs/P6场景分组概率校准协议_v46.txt'
REPORT = ROOT/'docs/P6场景分组概率校准结果_v46.txt'
OLD = ROOT/'results/P6/mixture_v43'
PARENT = ROOT/'results/P6/joint_repair_v45'
DATA = ROOT/'logs/P6/correspondence_v40/data_audit.json'
ORIGINAL_STATE = 'b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'
SOURCES = ['scripts/p6_probability_calibration_v46.py', 'src/p6_probability_calibration_v46.py',
           'tests/test_p6_probability_calibration_v46.py', 'tests/test_p6_calibration_independent_v46.py',
           'scripts/check_p6_probability_calibration_v46.py']
GATES = ('warp_nll_gain', 'warp_brier_noninferior', 'warp95_coverage', 'warp_set_sharper',
         'warp_multilevel_reliability', 'worst_scene_warp', 'worst_scene_depth',
         'source_depth95', 'source_depth_sharpness', 'source_depth_nll_gain',
         'target_joint_nll_noninferior', 'target_missing_brier_noninferior',
         'target_conditional_coverage', 'source_joint95', 'target_union_sharpness',
         'target_union95', 'observation_preserved')
LIMITS = dict(fold_fit=3, fold_prediction=3, fold_prediction_replay=3,
              temperature_fit=1, temperature_certificate=1,
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
                      current_part='P6_probability_calibration_v46', P6_scientific_passed=False,
                      P7_started=False, P10_started=False, recurring_automation=False, **fields))
def supervise(phase, **fields):
    path = ROOT/'runtime/pipeline/part_level_supervision.json'; backup(path, phase)
    value = read(path)
    value.update(recorded_at=now(), current_authorized_part='P6_probability_calibration_v46',
        current_execution_state=phase, current_runner=rel(Path(__file__)),
        current_execution_evidence=rel(STATE), next_repair_started=True,
        user_authorization='2026-11-02用户同意三组场景训练外预测加单参数概率校准，内部检查后后台运行，完整本部分后停止',
        automatic_resume_authorized=False, automatic_scheduler_recreated=False,
        stop_after='P6_probability_calibration_v46', stop_after_part='P6_probability_calibration_v46',
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
    path = ROOT/'runtime/pipeline/p6_probability_calibration_v46.lock'
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
    need(os.environ.get('P6_V46_CHILD_TOKEN') == state.get('run_token') and bool(state.get('run_token')), 'Worker requires active controller')
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
        from scripts import p6_joint_repair_v45 as parent
        oldconfig = parent.verify()
        parent_done = read(PARENT/'closeout.json')
        need(parent.verify_executed() == parent_done['proof'] and parent_done['engineering_passed'], 'Parent completion invalid')
        need(parent_done['proof']['stages'] == ['fit', 'warp_calibration', 'warp_review'], 'Unexpected parent scope')
        protected = dict(oldconfig['protected'])
        dependencies = SOURCES + [rel(PROTOCOL), 'docs/P6场景分组概率校准独立设计审查_v46.txt',
            'configs/p6_joint_repair_v45.json', 'scripts/research_cache_recovery_v44r1.py',
            'src/p6_cache_audit_v44.py', rel(DATA), 'logs/P6/mixture_v43/freeze.json',
            'logs/P6/mixture_v43/review.json', 'logs/P6/mixture_v43/cold.json',
            'results/research_v44r1/closeout.json', 'results/P6/cache_audit_v44/result.json',
            'logs/P6/cache_audit_v44/review.json', 'logs/P6/cache_audit_v44/cold.json',
            rel(LOG/'internal_checks.json'), rel(LOG/'independent_review.json')]
        dependencies += [rel(p) for p in PARENT.rglob('*') if p.is_file()]
        dependencies += ['logs/P6/joint_repair_v45/'+name for name in
                         ('freeze.json', 'warp_review.json', 'cold.json', 'calls.jsonl', 'children.jsonl')]
        protected.update({name: identity(ROOT/name) for name in dependencies})
        measured = sum(float(read(p)['seconds']) for p in (LOG/'preparation').glob('*.json'))
        c = dict(at=now(), version='v46', protected=protected, call_limits=LIMITS,
            controller_wall_seconds=3600, preparation_seconds=measured+30., cpu_threads=2,
            process_tree_rss_limit=3*1024**3, output_limit=512*1024**2, free_disk_min=60*1024**3,
            original_gates_unchanged=True, train_cases=1296, calibration_cases=432, exposed_evaluation_cases=432,
            folds=[[3,8,12,17], [4,9,14,19], [5,10,16,20]],
            optimizer=dict(fits=3, cases_per_fold_fit=864, cases_per_fold_holdout=432,
                maxiter=60, maxfun=100, hard_objectives_per_fit=120, ridge=.001, shared_coefficients=16),
            temperature=dict(fits=1, minimum=0., maximum=1., loss='equal_case_OOF_NLL',
                max_objective_evaluations=64, bracket_tolerance=1e-10, calibration_partition_fits=0),
            reused=dict(full_fit_head_refits=0, shape_fits=0, depth_scale_refits=0, parent_model=rel(PARENT/'model.json')),
            stages='fit -> cached warp cal6+replay -> if pass full cal17+replay -> if pass exposed eval17+replay -> cold -> report -> stop',
            failed_gate_action='Stop this candidate after full replay of executed scope; do not tune or run the deferred expensive scope',
            scope='Known synthetic global integer warps on previously exposed GraspNet RealSense development scenes; not independent final test',
            neural_forwards=0, official_AP=0, CAD_renders=0, grasp_calls=0,
            progress_before=[45, 56.25], progress_after_full_pass=[47, 58.75])
        protected_check(c)
        need(c['preparation_seconds'] < 3600 and shutil.disk_usage(ROOT).free >= c['free_disk_min'], 'Insufficient reserved resources')
        once(CONFIG, c); once(LOG/'freeze.json', dict(at=now(), config=identity(CONFIG)))
        need(time.monotonic()-started <= 30, 'Freeze reserve exceeded')
        status('frozen_ready'); supervise('v46_frozen_ready')
        handoff('P6 v46场景分组概率校准已冻结：固定3个fit场景折各训练轻量head，仅由1296个真OOF预测拟合一次先验锚定tau；复用v45最终head/H/深度尺度。cal6全过才cal17，再全过才开发eval17，失败即完整复验/cold/汇总后停止，不追加tau或v47/P7/P10。全部任务总进度45%、实验总进度56.25%。入口scripts/p6_probability_calibration_v46.py deliver；先核进程及锁，禁止重复。', 'freeze')
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
def fit():
    import numpy as np
    from src import p6_joint_repair_v45 as base
    from src import p6_probability_calibration_v46 as core
    from scripts.p6_mixture_v43 import arrays_save
    x, y, rows = training_arrays(); scenes = np.array([r['scene'] for r in rows])
    scores = np.empty((len(y),81)); priors = np.empty_like(scores)
    assigned = np.full(len(y), -1, dtype=np.int64); visits = np.zeros(len(y), dtype=np.int64)
    for fold in range(3):
        train, holdout = core.fit_fold_indices(scenes, fold)
        need(len(train) == 864 and len(holdout) == 432 and not np.intersect1d(train, holdout).size, 'Bad fixed OOF partition')
        head = call('fold_fit', str(fold), lambda: base.fit_warp(x[train], y[train], guard=child_guard))
        once(OUT/f'fold_{fold}.json', dict(fold=fold, head=head, training_indices=train.tolist(),
            holdout_indices=holdout.tolist(), training_scenes=sorted(set(scenes[train].tolist())),
            holdout_scenes=sorted(set(scenes[holdout].tolist())), input_sha256=sha(OLD/'generated.npz')))
        scores[holdout] = call('fold_prediction', str(fold), lambda: core.score_warp(head, x[holdout]))
        priors[holdout] = np.array(head['empirical_prior']); assigned[holdout] = fold; visits[holdout] += 1
        child_guard()
    core.validate_oof_coverage(scenes, assigned, visits, expected_cases_per_scene=108)
    arrays_save(OUT/'oof.npz', scores=scores, priors=priors, targets=y, scenes=scenes,
        assigned_folds=assigned, visits=visits)
    once(OUT/'oof_rows.json', rows)
    calibration = call('temperature_fit', 'one_OOF_NLL_parameter', lambda: core.calibrate_temperature(scores, priors, y, guard=child_guard))
    parent = read(PARENT/'model.json')
    warp = core.wrap_model(parent['warp'], calibration)
    child_guard()
    once(OUT/'model.json', dict(at=now(), warp=warp, shape=parent['shape'], depth=parent['depth'],
        empirical_prior=parent['empirical_prior'], calibration=calibration,
        input_sha256=sha(OLD/'generated.npz'), parent_model_sha256=sha(PARENT/'model.json'),
        oof_sha256=sha(OUT/'oof.npz'), fit_scenes=read(DATA)['scenes']['fit'],
        calibration_partition_fits=0, depth_scale_refits=0, shape_fits=0, full_head_refits=0, GT_in_predictor=False))

def temperature_certificate(scores, priors, targets, report):
    """Independent convex-NLL optimality certificate; no second parameter fit."""
    import numpy as np
    from scipy.special import logsumexp
    s = np.asarray(scores, float); pi = np.asarray(priors, float); y = np.asarray(targets)
    checked = {}
    def point(tau):
        if tau in checked: return checked[tau]
        logits = np.log(pi)+tau*s; logp = logits-logsumexp(logits, axis=1, keepdims=True)
        probability = np.exp(logp)
        checked[tau] = float(-logp[np.arange(len(y)),y].mean()), float((np.sum(probability*s,axis=1)-s[np.arange(len(y)),y]).mean())
        return checked[tau]
    trace = report['objective_trace']; tau = report['tau']; value, gradient = point(tau)
    need(0 <= tau <= 1 and report['converged'], 'Invalid/unconverged temperature solution')
    need(len(trace) == report['objective_evaluations'] <= 64, 'Temperature objective budget differs')
    for entry in trace:
        v, g = point(entry['tau'])
        need(np.isclose(v,entry['nll'],rtol=1e-11,atol=1e-12) and np.isclose(g,entry['derivative'],rtol=1e-10,atol=1e-12), 'Saved objective trace differs')
    need(np.isclose(value,report['solution_nll'],rtol=1e-11,atol=1e-12) and
         np.isclose(gradient,report['solution_derivative'],rtol=1e-10,atol=1e-12), 'Solution score differs')
    d0, d1 = point(0.)[1], point(1.)[1]
    need(np.isclose(d0,report['derivative0'],atol=1e-12) and np.isclose(d1,report['derivative1'],atol=1e-12), 'Endpoint certificate differs')
    if tau == 0: need(d0 >= -1e-12, 'Lower endpoint not convex optimum')
    elif tau == 1: need(d1 <= 1e-12, 'Upper endpoint not convex optimum')
    else:
        lo, hi = report['bracket']; dlo, dhi = point(lo)[1], point(hi)[1]
        need(0 <= lo <= tau <= hi <= 1 and hi-lo <= 1e-10 and dlo <= 1e-12 and dhi >= -1e-12, 'Interior bracket not a convex optimum certificate')
    need(value <= min(point(0.)[0],point(1.)[0])+1e-10, 'Fitted NLL worse than endpoint')
    need(set(checked) == {entry['tau'] for entry in trace} and len(checked) <= 64, 'Certificate evaluated outside saved trace')
    return dict(passed=True, tau=float(tau), nll=value, derivative=gradient,
                objective_trace_points_verified=len(trace), certificate_objective_evaluations=len(checked), optimization_refits=0)

def review_oof():
    import numpy as np
    from src import p6_joint_repair_v45 as base
    from src import p6_probability_calibration_v46 as core
    from scripts.p6_mixture_v43 import arrays_read, close_numeric
    x, y, rows = training_arrays(); oof = arrays_read(OUT/'oof.npz')
    need(rows == read(OUT/'oof_rows.json'), 'OOF row identity differs')
    scenes = np.array([r['scene'] for r in rows]); visits = np.zeros(len(y),dtype=np.int64)
    need(np.array_equal(y,oof['targets']) and np.array_equal(scenes,oof['scenes']), 'OOF labels/scenes differ')
    for fold in range(3):
        train, holdout = core.fit_fold_indices(scenes, fold); saved = read(OUT/f'fold_{fold}.json'); head=saved['head']
        need(saved['training_indices'] == train.tolist() and saved['holdout_indices'] == holdout.tolist(), 'OOF training/holdout indices differ')
        need(saved['training_scenes'] == sorted(set(scenes[train].tolist())) and saved['holdout_scenes'] == sorted(set(scenes[holdout].tolist())), 'OOF scene provenance differs')
        center, scale = x[train].mean(axis=(0,1)), x[train].std(axis=(0,1)); scale=np.where(scale>1e-8,scale,1.)
        need(np.array_equal(center,np.array(head['center'])) and np.array_equal(scale,np.array(head['scale'])), 'Fold normalization leaks outside training')
        need(np.array_equal(base.empirical_prior(y[train]),np.array(head['empirical_prior'])), 'Fold prior differs from training counts')
        scores = call('fold_prediction_replay', str(fold), lambda: core.score_warp(head,x[holdout]))
        need(np.array_equal(scores,oof['scores'][holdout]) and np.array_equal(np.broadcast_to(head['empirical_prior'],(len(holdout),81)),oof['priors'][holdout]), 'OOF score/prior replay differs')
        visits[holdout]+=1; child_guard()
    core.validate_oof_coverage(scenes,oof['assigned_folds'],oof['visits'],expected_cases_per_scene=108)
    need(np.array_equal(visits,oof['visits']), 'OOF each-case exposure differs')
    model=read(OUT/'model.json'); parent=read(PARENT/'model.json')
    need(all(model[k] == parent[k] for k in ('depth','shape','empirical_prior')), 'Frozen parent depth/H/prior changed')
    close_numeric(model['warp'],core.wrap_model(parent['warp'],model['calibration']))
    certificate = call('temperature_certificate','independent_convex_NLL_check',
        lambda: temperature_certificate(oof['scores'],oof['priors'],y,model['calibration']))
    once(LOG/'oof_review.json',dict(at=now(),passed=True,certificate=certificate,
        outputs={name:identity(OUT/name) for name in ('fold_0.json','fold_1.json','fold_2.json','oof.npz','oof_rows.json','model.json')},
        scope='Every OOF score replayed; fold identities/normalization/prior and convex temperature optimality checked, no model refit'))

def warp_metric(weights, target):
    from src import p6_probability_calibration_v46 as core
    value = core.warp_statistics(weights, int(target))
    value['levels'] = {str(v['level']): dict(covered=v['covered'], size=v['size'], mass=v['probability'])
                       for v in value.pop('credible_sets')}
    return value

def warp_result(replay=False):
    import numpy as np
    from src import p6_probability_calibration_v46 as core
    from scripts.p6_joint_evaluate_v45 import aggregate_warp
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
    review_oof()
    fresh = call('warp_calibration_replay', 'calibration432', lambda: warp_result(True))
    close_numeric(fresh, read(OUT/'warp_calibration.json'))
    once(LOG/'warp_review.json', dict(at=now(), passed=True, result=identity(OUT/'warp_calibration.json'),
        posterior=identity(OUT/'calibration_warp.npz'), model=identity(OUT/'model.json'),
        scope='Complete deterministic 432 posterior/metric replay and full OOF verification; no optimizer refit'))

def full_evaluate(split, replay=False):
    import numpy as np
    from src import p6_probability_calibration_v46 as core
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
    stages = ['fit', 'warp_calibration', 'warp_review']
    warp = read(OUT/'warp_calibration.json')
    if warp['passed']:
        stages += ['calibration', 'calibration_review']
        if read(OUT/'calibration.json')['passed']: stages += ['evaluation', 'evaluation_review']
    return stages
def verify_executed():
    from scripts.p6_joint_evaluate_v45 import aggregate_warp, aggregate
    from scripts.p6_mixture_v43 import close_numeric
    stages = executed_stages(); expected = dict(fold_fit=3, fold_prediction=3, fold_prediction_replay=3,
        temperature_fit=1, temperature_certificate=1, warp_calibration=1, warp_calibration_replay=1)
    model = read(OUT/'model.json')
    parent=read(PARENT/'model.json')
    need(all(model[k] == parent[k] for k in ('depth','shape','empirical_prior')), 'Frozen parent changed')
    oof_proof=read(LOG/'oof_review.json'); need(oof_proof['passed'],'OOF review not passed')
    for name,expected_identity in oof_proof['outputs'].items():
        need(identity(OUT/name) == expected_identity,'OOF reviewed artifact changed '+name)
    proof = read(LOG/'warp_review.json'); need(proof['passed'], 'Warp replay not passed')
    for field, name in [('result', 'warp_calibration.json'), ('posterior', 'calibration_warp.npz'), ('model', 'model.json')]:
        need(proof[field] == identity(OUT/name), 'Reviewed warp artifact changed')
    for split in ['warp_calibration'] + [s for s in ('calibration', 'evaluation') if s in stages]:
        result = read(OUT/(split+'.json'))
        summary, gates = aggregate_warp(result['rows'], split='calibration') if split == 'warp_calibration' else aggregate(result['rows'], split=split)
        close_numeric(summary, result['summary']); need(gates == result['gates'] and result['passed'] == all(gates.values()), 'Stored gates differ')
        if split != 'warp_calibration':
            review = read(LOG/(split+'_review.json'))
            need(review['passed'] and review['result'] == identity(OUT/(split+'.json')), 'Full replay identity differs')
            for stage in (split, split+'_review'): expected.update({stage+'_posterior': 1, stage+'_frame': 16, stage+'_case': 432})
    calls = ledger(); need(not calls['unknown'] and calls['counts'] == expected, 'Executed scientific ledger differs')
    names = {'model.json','fold_0.json','fold_1.json','fold_2.json','oof.npz','oof_rows.json',
             'calibration_warp.npz','warp_calibration.json'} | {s+'.json' for s in ('calibration', 'evaluation') if s in stages}
    actual = {p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    need(actual - {'closeout.json'} == names, 'Unexpected/unreviewed output artifact')
    gates_by_split = {s: {name: None for name in GATES} for s in ('calibration', 'evaluation')}
    gates_by_split['calibration'].update(read(OUT/'warp_calibration.json')['gates'])
    for split in ('calibration', 'evaluation'):
        if split in stages:
            actual_gates = read(OUT/(split+'.json'))['gates']
            need(set(actual_gates) == set(GATES), 'Original17 gate names differ')
            gates_by_split[split].update(actual_gates)
    return dict(stages=stages, calls=calls, outputs={name: identity(OUT/name) for name in sorted(names)},
        gates_by_split=gates_by_split,
        development_scientific_passed='evaluation' in stages and read(OUT/'evaluation.json')['passed'],
        all17_executed='calibration' in stages, evaluation_executed='evaluation' in stages)
def cold():
    proof = verify_executed(); child_guard()
    once(LOG/'cold.json', dict(at=now(), passed=True, fresh_process_pid=os.getpid(),
        scope='Read-only full executed inventory, summaries, original gates and call ledger', proof=proof))

def finish(c, guard):
    proof = verify_executed(); cold_proof = read(LOG/'cold.json')
    need(cold_proof['passed'] and cold_proof['proof'] == proof, 'Cold/executed inventory differs')
    protected_check(c); guard()
    passed = proof['development_scientific_passed']; progress = [47, 58.75] if passed else [45, 56.25]
    last = 'evaluation' if proof['evaluation_executed'] else 'calibration' if proof['all17_executed'] else 'warp_calibration'
    result = read(OUT/(last+'.json')); failures = [k for k, v in result['gates'].items() if not v]
    lines = ['P6场景分组概率校准 v46', now(), '',
        f'全部任务总进度{progress[0]}%；实验总进度{progress[1]}%。当前候选已结束并停止。',
        '科学采用（既定已知合成开发范围）='+str(passed)+'；已执行范围工程复验及真实冷重入=True。',
        '最后执行阶段='+last+'；失败项='+str(failures),
        '原17项完整cal评价已执行='+str(proof['all17_executed'])+'；原17项开发eval已执行='+str(proof['evaluation_executed']),
        '固定3个训练场景折，每折8个场景训练轻量head、4个场景生成OOF；每折独立标准化和训练先验。仅合并OOF的NLL拟合一个tau，未用cal/eval调tau。',
        '先验锚定p_tau正比于pi*exp(tau*score)，tau在[0,1]；tau=1合法，不强行软化。原81候选正支持、v45最终head/H/深度尺度/原XYZ保持冻结，固定5mm t3基线保留。',
        '预先固定顺序：cal位移6门槛失败即停止原始深度/B复算；全过才cal17；再全过才eval17；任何失败不调参、不追加新方案。',
        '没有跑到的门槛保持未验证，不能称全部17项通过。失败候选目前不可作为完整P6科学结果采用。' if not passed else '仅通过本开发协议；不等于真实传感器校准、独立最终测试或抓取收益。',
        c['scope'],
        '0新神经前向/官方AP/抓取/CAD调用；3个OOF临时head拟合+1次单参数校准，最终head/H/depth重训=0；复验重建预测和凸最优证书，不另拟合候选。',
        'tau及OOF损失='+json.dumps({k:v for k,v in read(OUT/'model.json')['calibration'].items() if k in ('tau','solution_nll','uncalibrated_nll','derivative0','derivative1','stopping_reason','objective_evaluations')},ensure_ascii=False),
        '阶段结果='+json.dumps(result['gates'], ensure_ascii=False),
        '全部门槛状态（null表示未执行）='+json.dumps(proof['gates_by_split'], ensure_ascii=False),
        '调用账本='+json.dumps(proof['calls'], ensure_ascii=False),
        '各fold优化状态='+json.dumps([{k:v for k,v in read(OUT/f'fold_{i}.json')['head'].items() if k.startswith('optimizer') or k=='objective_evaluations'} for i in range(3)],ensure_ascii=False),
        '校准损失改善不等于多级门槛通过；8场景临时head与12场景最终head的证据强度可能不同。若仍失败，应按实际可靠性/锐度或A/B失败类型作后续判断，不继续在当前cal上反复挑tau，不将局部结果称完整可用。',
        'P5科学结论不变。所有旧文件/失败/未知保留；原state.json不改。没有进入P7/P10，没有周期任务，等待用户检查后决定下一部分。']
    guard(); need(not REPORT.exists(), 'Existing report must not be overwritten'); REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    guard()
    once(OUT/'closeout.json', dict(at=now(), proof=proof, report=identity(REPORT), cold=identity(LOG/'cold.json'),
        stopped=True, engineering_passed=True, P6_development_scope_passed=passed,
        all_tasks_percent=progress[0], experiment_percent=progress[1], neural_forwards=0, official_AP=0,
        child_ledger=ledger(LOG/'children.jsonl')))
    status('completed_stopped', report=rel(REPORT), current_stage=last, scientific_passed=passed)
    if passed:
        state = read(STATE); state.update(all_tasks_percent=47, experiment_percent=58.75, P6_scientific_passed=True); atomic(STATE, state)
    supervise('v46_completed_stopped', current_runner=None, remaining_research_updates_cap=0,
        waiting_for_user_next_part_confirmation=True, summary_report=rel(REPORT), final_cold=rel(LOG/'cold.json'),
        final_readonly=rel(LOG/'warp_review.json') if last == 'warp_calibration' else rel(LOG/(last+'_review.json')),
        final_delivery_index=rel(OUT/'closeout.json'), all_tasks_percent_planning_estimate=progress[0], experiment_percent_planning_estimate=progress[1])
    handoff(f'P6 v46整体修复已完成本次有界范围及完整复验/冷重入并停止。最后阶段{last}；科学采用={passed}，未运行关口不冒充通过。全部任务总进度{progress[0]}%、实验总进度{progress[1]}%。报告{rel(REPORT)}。等待用户检查，不自动v47/P7/P10或周期任务。', 'complete')

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
            env = os.environ.copy(); env['P6_V46_CHILD_TOKEN'] = run_token
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
            supervise('v46_running')
            for stage in ('fit', 'warp_calibration', 'warp_review'): run(stage)
            if read(OUT/'warp_calibration.json')['passed']:
                for stage in ('calibration', 'calibration_review'): run(stage)
                if read(OUT/'calibration.json')['passed']:
                    for stage in ('evaluation', 'evaluation_review'): run(stage)
            run('cold'); guard(); finish(c, guard); guard()
        except BaseException as e:
            append(LOG/'controller.jsonl', dict(at=now(), event='failure_preserved', error=repr(e), traceback=traceback.format_exc()))
            status('failed_stopped', error=repr(e), unknown=ledger()['unknown'], unknown_children=ledger(LOG/'children.jsonl')['unknown'])
            supervise('v46_failed_stopped', current_runner=None, waiting_for_user_next_part_confirmation=True)
            handoff('P6 v46出现工程或资源异常，失败日志/模型/未知返回保留，先核实际进程与锁再恢复，不自动重复。全部任务总进度45%、实验总进度56.25%；科学通过未成立，不进P7/P10。', 'failure')
            raise

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'verify', 'deliver', 'worker'])
    parser.add_argument('--stage'); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'verify': print(json.dumps(dict(passed=True, protected_files=len(verify()['protected']))))
    elif args.mode == 'deliver': deliver()
    else:
        verify(); child_guard(); OUT.mkdir(parents=True, exist_ok=True)
        if args.stage == 'fit': fit()
        elif args.stage == 'warp_calibration': warp_calibration()
        elif args.stage == 'warp_review': warp_review()
        elif args.stage in ('calibration', 'evaluation'): full_evaluate(args.stage)
        elif args.stage in ('calibration_review', 'evaluation_review'): full_evaluate(args.stage.removesuffix('_review'), True)
        elif args.stage == 'cold': cold()
        else: raise ValueError('Unknown worker stage')
        child_guard()
if __name__ == '__main__': main()
