"""One evidence-gated four-corner quality calibration; frozen scientific gates.

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
LOG = ROOT/'logs/P6/quality_calibration_v47'
OUT = ROOT/'results/P6/quality_calibration_v47'
CONFIG = ROOT/'configs/p6_quality_calibration_v47.json'
STATE = ROOT/'runtime/pipeline/p6_quality_calibration_v47.json'
PROTOCOL = ROOT/'docs/P6质量条件校准协议_v47.txt'
REPORT = ROOT/'docs/P6质量条件校准结果_v47.txt'
OLD = ROOT/'results/P6/mixture_v43'
PARENT = ROOT/'results/P6/probability_calibration_v46'
DATA = ROOT/'logs/P6/correspondence_v40/data_audit.json'
ORIGINAL_STATE = 'b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'
SOURCES = ['scripts/p6_quality_calibration_v47.py', 'src/p6_quality_calibration_v47.py',
           'tests/test_p6_quality_calibration_v47.py', 'tests/test_p6_quality_independent_v47.py',
           'scripts/check_p6_quality_calibration_v47.py']
GATES = ('warp_nll_gain', 'warp_brier_noninferior', 'warp95_coverage', 'warp_set_sharper',
         'warp_multilevel_reliability', 'worst_scene_warp', 'worst_scene_depth',
         'source_depth95', 'source_depth_sharpness', 'source_depth_nll_gain',
         'target_joint_nll_noninferior', 'target_missing_brier_noninferior',
         'target_conditional_coverage', 'source_joint95', 'target_union_sharpness',
         'target_union95', 'observation_preserved')
LIMITS = dict(evidence_compute=1, evidence_replay=1,
              conditional_fit=1, conditional_certificate=1,
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
                      current_part='P6_quality_calibration_v47', P6_scientific_passed=False,
                      P7_started=False, P10_started=False, recurring_automation=False, **fields))
def supervise(phase, **fields):
    path = ROOT/'runtime/pipeline/part_level_supervision.json'; backup(path, phase)
    value = read(path)
    value.update(recorded_at=now(), current_authorized_part='P6_quality_calibration_v47',
        current_execution_state=phase, current_runner=rel(Path(__file__)),
        current_execution_evidence=rel(STATE), next_repair_started=True,
        user_authorization='用户授权复用训练外预测先审查质量相关偏差，有固定证据才一次条件校准与完整分支验收，完成或失败停止',
        automatic_resume_authorized=False, automatic_scheduler_recreated=False,
        stop_after='P6_quality_calibration_v47', stop_after_part='P6_quality_calibration_v47',
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
    path = ROOT/'runtime/pipeline/p6_quality_calibration_v47.lock'
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
    need(os.environ.get('P6_V47_CHILD_TOKEN') == state.get('run_token') and bool(state.get('run_token')), 'Worker requires active controller')
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
        from scripts import p6_probability_calibration_v46 as parent
        oldconfig = parent.verify()
        parent_done = read(PARENT/'closeout.json')
        need(parent.verify_executed() == parent_done['proof'] and parent_done['engineering_passed'], 'Parent completion invalid')
        need(parent_done['cold'] == identity(ROOT/'logs/P6/probability_calibration_v46/cold.json') and
             parent_done['report'] == identity(ROOT/'docs/P6场景分组概率校准结果_v46.txt'), 'Parent cold/report differs')
        need(parent_done['child_ledger'] == parent.ledger(parent.LOG/'children.jsonl'), 'Parent child ledger differs')
        need(parent_done['proof']['stages'] == ['fit', 'warp_calibration', 'warp_review'], 'Unexpected parent scope')
        protected = dict(oldconfig['protected'])
        dependencies = SOURCES + [rel(PROTOCOL), 'docs/P6质量条件校准独立设计审查_v47.txt',
            'configs/p6_probability_calibration_v46.json', 'docs/P6场景分组概率校准结果_v46.txt', 'scripts/research_cache_recovery_v44r1.py',
            'src/p6_cache_audit_v44.py', rel(DATA), 'logs/P6/mixture_v43/freeze.json',
            'logs/P6/mixture_v43/review.json', 'logs/P6/mixture_v43/cold.json',
            'results/research_v44r1/closeout.json', 'results/P6/cache_audit_v44/result.json',
            'logs/P6/cache_audit_v44/review.json', 'logs/P6/cache_audit_v44/cold.json',
            rel(LOG/'internal_checks.json'), rel(LOG/'independent_review.json')]
        dependencies += [rel(p) for p in PARENT.rglob('*') if p.is_file()]
        dependencies += ['logs/P6/probability_calibration_v46/'+name for name in
                         ('freeze.json', 'oof_review.json','warp_review.json','cold.json','calls.jsonl','children.jsonl')]
        protected.update({name: identity(ROOT/name) for name in dependencies})
        measured = sum(float(read(p)['seconds']) for p in (LOG/'preparation').glob('*.json'))
        c = dict(at=now(), version='v47', protected=protected, call_limits=LIMITS,
            controller_wall_seconds=3600, preparation_seconds=measured+30., cpu_threads=2,
            process_tree_rss_limit=3*1024**3, output_limit=512*1024**2, free_disk_min=60*1024**3,
            original_gates_unchanged=True, train_cases=1296, calibration_cases=432, exposed_evaluation_cases=432,
            folds=[[3,8,12,17], [4,9,14,19], [5,10,16,20]],
            evidence=dict(cached_rows=1296, features=['observed_valid_fraction','uncalibrated_posterior_entropy'],
                bins='known pooled median; equal is low', fold_bin_min_rows=108, fold_bin_min_scenes=3,
                derivative_high_minus_low='same strict sign across3folds and absolute mean>=.02', any_feature_support=True),
            optimizer=dict(fits=1, maxiter=60,maxfun=100,hard_objectives=120,ridge=.001,parameters=4,
                bounds=[[0.,1.]]*4,kkt_tolerance=1e-5,nonconvergence_is_engineering_failure=True),
            temperature=dict(loss='equal_case_OOF_NLL plus .5*.001*sum((theta-tau46)^2)',
                calibration_partition_fits=0,unknown_observables_fallback='frozen tau46'),
            reused=dict(full_fit_head_refits=0, shape_fits=0, depth_scale_refits=0, parent_model=rel(PARENT/'model.json')),
            stages='cached evidence+replay -> if supported fit+certificate -> cal6+replay -> if pass cal17+replay -> if pass eval17+replay -> cold -> report -> stop',
            failed_gate_action='Stop this candidate after full replay of executed scope; do not tune or run the deferred expensive scope',
            scope='Known synthetic global integer warps on previously exposed GraspNet RealSense development scenes; not independent final test',
            neural_forwards=0, official_AP=0, CAD_renders=0, grasp_calls=0,
            progress_before=[45, 56.25], progress_after_full_pass=[47, 58.75])
        protected_check(c)
        need(c['preparation_seconds'] < 3600 and shutil.disk_usage(ROOT).free >= c['free_disk_min'], 'Insufficient reserved resources')
        once(CONFIG, c); once(LOG/'freeze.json', dict(at=now(), config=identity(CONFIG)))
        need(time.monotonic()-started <= 30, 'Freeze reserve exceeded')
        status('frozen_ready'); supervise('v47_frozen_ready')
        handoff('P6 v47质量条件校准已冻结：只复用v46的1296个OOF缓存，先一次固定质量证据及完整复验；不支持则零新拟合停止，支持才一次四角条件温度拟合/证书，再原cal6→cal17→eval17。复用原head/H/深度，完成或失败后复验/cold/汇总停止，不自动v48/P7/P10。全部任务总进度45%、实验总进度56.25%。入口scripts/p6_quality_calibration_v47.py deliver；禁止重复。', 'freeze')
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
def oof_inputs():
    """Align and reuse already reviewed OOF predictions; zero new head inference."""
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_read
    from src.p6_probability_calibration_v46 import validate_oof_coverage
    x, y, rows = training_arrays(); oof = arrays_read(PARENT/'oof.npz')
    scenes = np.array([r['scene'] for r in rows])
    need(rows == read(PARENT/'oof_rows.json'), 'Parent OOF row identity differs')
    need(np.array_equal(y, oof['targets']) and np.array_equal(scenes, oof['scenes']), 'Parent OOF label/scene alignment differs')
    validate_oof_coverage(scenes, oof['assigned_folds'], oof['visits'], expected_cases_per_scene=108)
    parent = read(PARENT/'model.json'); tau46 = parent['calibration']['tau']
    need(parent['warp']['calibration'] == parent['calibration'], 'Parent wrapper/global tau differs')
    return x, y, rows, oof, parent, tau46


def evidence_result():
    from src import p6_quality_calibration_v47 as core
    x, y, rows, oof, parent, tau46 = oof_inputs()
    result = core.evidence_gate(x, oof['scores'], oof['priors'], y, oof['scenes'], oof['assigned_folds'], tau46)
    result['inputs'] = {rel(p): identity(p) for p in (PARENT/'oof.npz', PARENT/'oof_rows.json', PARENT/'model.json', OLD/'generated.npz')}
    result['parent_oof_review'] = identity(ROOT/'logs/P6/probability_calibration_v46/oof_review.json')
    result['new_head_fits'] = result['new_head_predictions'] = 0
    return result


def evidence():
    result = call('evidence_compute', 'fixed_cached1296', evidence_result)
    child_guard(); once(OUT/'evidence.json', result)


def evidence_review():
    from scripts.p6_mixture_v43 import close_numeric
    original = read(OUT/'evidence.json')
    fresh = call('evidence_replay', 'fixed_cached1296', evidence_result)
    close_numeric(original, fresh); child_guard()
    once(LOG/'evidence_review.json', dict(at=now(), passed=True, result=identity(OUT/'evidence.json'),
        supported=original['supported'], scope='Full cached1296 fixed evidence replay, alignment and protected-parent provenance; no model fit or head inference'))


def fit():
    from src import p6_quality_calibration_v47 as core
    need(read(OUT/'evidence.json')['supported'] and read(LOG/'evidence_review.json')['passed'], 'Evidence prerequisite failed')
    x, y, rows, oof, parent, tau46 = oof_inputs()
    try:
        calibration = call('conditional_fit', 'one_four_corner_OOF_candidate', lambda:
            core.fit_conditional_temperature(x, oof['scores'], oof['priors'], y, tau46, guard=child_guard,
                on_evaluation=lambda row: append(LOG/'conditional_objective.jsonl', dict(at=now(), **row))))
    except BaseException as exc:
        once(OUT/'failure_fit.json', dict(at=now(), error=repr(exc),
            partial_report=getattr(exc, 'partial_report', None), traceback=traceback.format_exc()))
        raise
    once(OUT/'conditional_fit.json', calibration)
    need(calibration['optimizer_converged'] and calibration['kkt_passed'], 'Conditional optimizer failed; returned report preserved')
    warp = core.wrap_model(parent['warp']['base_model'], calibration)
    child_guard()
    once(OUT/'model.json', dict(at=now(), warp=warp, shape=parent['shape'], depth=parent['depth'],
        empirical_prior=parent['empirical_prior'], calibration=calibration,
        input_sha256=sha(OLD/'generated.npz'), parent_model_sha256=sha(PARENT/'model.json'),
        oof_sha256=sha(PARENT/'oof.npz'), fit_scenes=read(DATA)['scenes']['fit'],
        calibration_partition_fits=0, depth_scale_refits=0, shape_fits=0, full_head_refits=0,
        new_oof_head_fits=0, new_oof_head_predictions=0, GT_in_predictor=False))


def conditional_certificate(features, scores, priors, targets, report):
    """Independent finite convex objective/gradient/KKT replay, no optimizer."""
    import numpy as np
    from scipy.special import logsumexp
    x, s, pi, y = np.asarray(features,float), np.asarray(scores,float), np.asarray(priors,float), np.asarray(targets)
    need(x.shape == (len(y),81,16) and s.shape == pi.shape == (len(y),81), 'Certificate shape mismatch')
    product = x[:,:,0]; conditioned = x[:,:,14]; candidates = np.argmax(product,axis=1)
    edge = product[np.arange(len(y)),candidates]; known = edge > 0; v = np.zeros(len(y))
    v[known] = conditioned[np.arange(len(y))[known],candidates[known]]/edge[known]
    need(np.all((v >= -1e-12)&(v <= 1+1e-12)) and np.allclose(conditioned,product*v[:,None],rtol=1e-10,atol=1e-12), 'Certificate observable mismatch')
    v = np.clip(v,0,1)
    uncalibrated = np.log(pi)+s; logp1 = uncalibrated-logsumexp(uncalibrated,axis=1,keepdims=True)
    h = np.clip(-np.sum(np.exp(logp1)*logp1,axis=1)/np.log(81),0,1)
    z = np.column_stack(((1-v)*(1-h),(1-v)*h,v*(1-h),v*h))
    tau46 = report['tau46']; checked = {}
    need(report['ridge'] == .001 and report['bounds'] == [[0.,1.]]*4 and report['kkt_tolerance'] == 1e-5, 'Changed fixed certificate contract')
    def point(theta):
        key = tuple(theta)
        if key in checked: return checked[key]
        theta = np.asarray(theta,float)
        need(theta.shape == (4,) and np.isfinite(theta).all() and np.all((theta >= 0)&(theta <= 1)), 'Invalid corner')
        tau = np.where(known,z@theta,tau46); logits = np.log(pi)+tau[:,None]*s
        logp = logits-logsumexp(logits,axis=1,keepdims=True); p = np.exp(logp)
        nll = float(-logp[np.arange(len(y)),y].mean()); penalty = float(.5*.001*np.sum((theta-tau46)**2))
        point_gradient = np.sum(p*s,axis=1)-s[np.arange(len(y)),y]
        gradient = z[known].T@point_gradient[known]/len(y)+.001*(theta-tau46)
        checked[key] = dict(value=nll+penalty, gradient=gradient.tolist(), nll=nll, penalty=penalty)
        return checked[key]
    trace = report['objective_trace']; theta = np.asarray(report['theta'],float)
    need(0 < len(trace) == report['objective_evaluations'] <= 120, 'Conditional objective budget differs')
    need(trace[0]['theta'] == [tau46]*4 and any(entry['theta'] == theta.tolist() for entry in trace), 'Initial/final objective provenance differs')
    for entry in trace:
        actual = point(entry['theta'])
        for field in ('value','gradient','nll','penalty'):
            need(np.allclose(actual[field],entry[field],rtol=1e-10,atol=1e-12), 'Saved conditional objective trace differs '+field)
    final = point(theta)
    for field, target in [('value','final_objective'),('gradient','final_gradient'),('nll','final_nll'),('penalty','final_penalty')]:
        need(np.allclose(final[field],report[target],rtol=1e-10,atol=1e-12), 'Final conditional solution differs '+field)
    need(np.isclose(report['baseline_nll'],point([tau46]*4)['nll'],rtol=1e-10,atol=1e-12), 'Baseline NLL differs')
    residual = theta-np.clip(theta-np.asarray(final['gradient']),0,1); norm = float(np.max(np.abs(residual)))
    need(report['optimizer_converged'] is True and report['optimizer_iterations'] <= 60, 'Optimizer did not converge within fixed iterations')
    need(report['kkt_passed'] is True and norm <= 1e-5 and np.isclose(norm,report['kkt_projected_gradient_inf'],rtol=1e-10,atol=1e-12), 'Bounded KKT certificate failed')
    need(final['value'] <= point([tau46]*4)['value']+1e-10, 'Conditional objective worse than feasible frozen model')
    need(report['known_rows'] == int(known.sum()) and report['unknown_rows'] == int((~known).sum()) and report['total_rows'] == len(y), 'Unknown/full denominator accounting differs')
    need(len(checked) <= len(trace), 'Certificate searched outside saved objective trace')
    return dict(passed=True, theta=theta.tolist(), final_objective=final['value'], final_nll=final['nll'],
        final_gradient=final['gradient'], kkt_projected_gradient_inf=norm, kkt_tolerance=1e-5,
        objective_trace_points_verified=len(trace), certificate_objective_evaluations=len(checked), optimization_refits=0,
        known_rows=int(known.sum()), unknown_rows=int((~known).sum()), full_loss_denominator=len(y))


def fit_review():
    from src import p6_quality_calibration_v47 as core
    from scripts.p6_mixture_v43 import close_numeric
    x, y, rows, oof, parent, tau46 = oof_inputs()
    model = read(OUT/'model.json'); calibration = read(OUT/'conditional_fit.json')
    need(model['calibration'] == calibration and calibration['tau46'] == tau46, 'Fit report/model or frozen tau differs')
    need(all(model[k] == parent[k] for k in ('depth','shape','empirical_prior')), 'Frozen parent depth/H/prior changed')
    close_numeric(model['warp'],core.wrap_model(parent['warp']['base_model'],calibration))
    durable_trace = events(LOG/'conditional_objective.jsonl')
    need([{k:v for k,v in row.items() if k != 'at'} for row in durable_trace] == calibration['objective_trace']
         and len(durable_trace) == calibration['objective_evaluations'] <= 120, 'Durable actual objective trace differs')
    certificate = call('conditional_certificate','independent_convex_objective_gradient_KKT', lambda:
        conditional_certificate(x,oof['scores'],oof['priors'],y,calibration))
    child_guard()
    once(LOG/'fit_review.json',dict(at=now(),passed=True,certificate=certificate,
        outputs={name:identity(OUT/name) for name in ('conditional_fit.json','model.json')},
        reused_parent=identity(PARENT/'model.json'), reused_oof=identity(PARENT/'oof.npz'),
        objective_trace=identity(LOG/'conditional_objective.jsonl'),
        scope='Independent all-saved-point objective/gradient and final KKT certificate; frozen head/H/depth/prior identity, zero optimization refits'))


def observable_counts(features):
    """Count fallback usage without another head prediction or fitting pass."""
    import numpy as np
    x = np.asarray(features,float); need(x.ndim == 3 and x.shape[1:] == (81,16), 'Invalid split features')
    product, conditioned = x[:,:,0], x[:,:,14]
    need(np.isfinite(x).all() and np.all(product >= 0) and np.all(conditioned >= 0), 'Invalid split edge features')
    k = np.argmax(product,axis=1); denom = product[np.arange(len(x)),k]; known = denom > 0
    v = np.zeros(len(x)); v[known] = conditioned[np.arange(len(x))[known],k[known]]/denom[known]
    need(np.all((v >= -1e-12)&(v <= 1+1e-12)) and np.allclose(conditioned,product*v[:,None],rtol=1e-10,atol=1e-12), 'Split validity recovery differs')
    return dict(total_rows=len(x),known_rows=int(known.sum()),unknown_rows=int((~known).sum()),unknown_temperature='fixed_tau46')


def warp_metric(weights, target):
    from src import p6_quality_calibration_v47 as core
    value = core.warp_statistics(weights, int(target))
    value['levels'] = {str(v['level']): dict(covered=v['covered'], size=v['size'], mass=v['probability'])
                       for v in value.pop('credible_sets')}
    return value

def warp_result(replay=False):
    import numpy as np
    from src import p6_quality_calibration_v47 as core
    from scripts.p6_joint_evaluate_v45 import aggregate_warp
    arrays, meta = cache(); model = read(OUT/'model.json')
    selected = np.array([r['split'] == 'calibration' for r in meta])
    p = core.predict_warp(model['warp'], arrays['X'][selected]); prior = np.array(model['empirical_prior'])
    rows = [dict(**r, warp_method=warp_metric(prob, target), warp_prior=warp_metric(prior, target))
        for r, prob, target in zip([r for r in meta if r['split'] == 'calibration'], p, arrays['y'][selected])]
    summary, gates = aggregate_warp(rows, split='calibration')
    answer = dict(rows=rows, summary=summary, gates=gates, passed=all(gates.values()),
                  scope='432 calibration cached-feature cases; six original warp gates only', model_sha256=sha(OUT/'model.json'),
                  quality_usage=observable_counts(arrays['X'][selected]))
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
    need(read(LOG/'fit_review.json')['passed'], 'Independent fit certificate prerequisite failed')
    fresh = call('warp_calibration_replay', 'calibration432', lambda: warp_result(True))
    close_numeric(fresh, read(OUT/'warp_calibration.json'))
    once(LOG/'warp_review.json', dict(at=now(), passed=True, result=identity(OUT/'warp_calibration.json'),
        posterior=identity(OUT/'calibration_warp.npz'), model=identity(OUT/'model.json'),
        scope='Complete deterministic 432 posterior/metric replay; prior cached evidence and independent fit certificate verified'))

def full_evaluate(split, replay=False):
    import numpy as np
    from src import p6_quality_calibration_v47 as core
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
        model_sha256=sha(OUT/'model.json'), scope=split+' 432 known-synthetic development cases',
        quality_usage=observable_counts(arrays['X'][select]))
    if replay:
        close_numeric(answer, read(OUT/(split+'.json')))
        once(LOG/(stage+'.json'), dict(at=now(), passed=True, result=identity(OUT/(split+'.json')),
             scope='Full deterministic numeric replay and independent observation/source-ID reconstruction'))
    else: once(OUT/(split+'.json'), answer)

def executed_stages():
    stages = ['evidence', 'evidence_review']
    if not read(OUT/'evidence.json')['supported']: return stages
    stages += ['fit', 'fit_review', 'warp_calibration', 'warp_review']
    if read(OUT/'warp_calibration.json')['passed']:
        stages += ['calibration', 'calibration_review']
        if read(OUT/'calibration.json')['passed']: stages += ['evaluation', 'evaluation_review']
    return stages


def verify_executed():
    from scripts.p6_joint_evaluate_v45 import aggregate_warp, aggregate
    from scripts.p6_mixture_v43 import close_numeric
    stages = executed_stages(); expected = dict(evidence_compute=1, evidence_replay=1)
    evidence_result = read(OUT/'evidence.json'); evidence_proof = read(LOG/'evidence_review.json')
    need(evidence_proof['passed'] and evidence_proof['result'] == identity(OUT/'evidence.json') and
         evidence_proof['supported'] == evidence_result['supported'], 'Reviewed evidence identity/decision differs')
    need(evidence_result['supported'] == evidence_result['passed'] == any(v['passed'] for v in evidence_result['features'].values()), 'Evidence support decision differs')
    for name, binding in evidence_result['inputs'].items():
        need(identity(ROOT/name) == binding, 'Evidence input changed '+name)
    need(evidence_result['parent_oof_review'] == identity(ROOT/'logs/P6/probability_calibration_v46/oof_review.json'), 'Inherited OOF review changed')
    names = {'evidence.json'}
    gates_by_split = {s: {name: None for name in GATES} for s in ('calibration', 'evaluation')}
    if evidence_result['supported']:
        expected.update(conditional_fit=1, conditional_certificate=1, warp_calibration=1, warp_calibration_replay=1)
        names |= {'conditional_fit.json','model.json','calibration_warp.npz','warp_calibration.json'}
        model = read(OUT/'model.json'); parent = read(PARENT/'model.json')
        need(all(model[k] == parent[k] for k in ('depth','shape','empirical_prior')), 'Frozen parent changed')
        need(model['warp']['base_model'] == parent['warp']['base_model'] and model['calibration'] == read(OUT/'conditional_fit.json'), 'Frozen head/report changed')
        fit_proof = read(LOG/'fit_review.json')
        need(fit_proof['passed'] and fit_proof['certificate']['passed'], 'Independent fit certificate not passed')
        for name, binding in fit_proof['outputs'].items():
            need(identity(OUT/name) == binding, 'Reviewed conditional artifact changed '+name)
        need(fit_proof['reused_parent'] == identity(PARENT/'model.json') and fit_proof['reused_oof'] == identity(PARENT/'oof.npz'), 'Fit parent identity changed')
        need(fit_proof['objective_trace'] == identity(LOG/'conditional_objective.jsonl'), 'Reviewed durable trace identity changed')
        trace = events(LOG/'conditional_objective.jsonl')
        need([{k:v for k,v in point.items() if k != 'at'} for point in trace] == model['calibration']['objective_trace'], 'Durable objective trace differs')
        proof = read(LOG/'warp_review.json'); need(proof['passed'], 'Warp replay not passed')
        for field, name in [('result', 'warp_calibration.json'), ('posterior', 'calibration_warp.npz'), ('model', 'model.json')]:
            need(proof[field] == identity(OUT/name), 'Reviewed warp artifact changed')
        for split in ['warp_calibration'] + [s for s in ('calibration', 'evaluation') if s in stages]:
            result = read(OUT/(split+'.json'))
            summary, gates = aggregate_warp(result['rows'], split='calibration') if split == 'warp_calibration' else aggregate(result['rows'], split=split)
            close_numeric(summary, result['summary'])
            need(gates == result['gates'] and result['passed'] == all(gates.values()), 'Stored gates differ')
            if split != 'warp_calibration':
                names.add(split+'.json'); review = read(LOG/(split+'_review.json'))
                need(review['passed'] and review['result'] == identity(OUT/(split+'.json')), 'Full replay identity differs')
                for stage in (split, split+'_review'): expected.update({stage+'_posterior': 1, stage+'_frame': 16, stage+'_case': 432})
        gates_by_split['calibration'].update(read(OUT/'warp_calibration.json')['gates'])
        for split in ('calibration', 'evaluation'):
            if split in stages:
                actual_gates = read(OUT/(split+'.json'))['gates']
                need(set(actual_gates) == set(GATES), 'Original17 gate names differ')
                gates_by_split[split].update(actual_gates)
    calls = ledger(); need(not calls['unknown'] and calls['counts'] == expected, 'Executed scientific ledger differs')
    actual = {p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    need(actual - {'closeout.json'} == names, 'Unexpected/unreviewed output artifact')
    return dict(stages=stages, calls=calls, outputs={name: identity(OUT/name) for name in sorted(names)},
        gates_by_split=gates_by_split, evidence_supported=evidence_result['supported'],
        conditional_fit_executed='fit' in stages,
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
    last = ('evaluation' if proof['evaluation_executed'] else 'calibration' if proof['all17_executed']
            else 'warp_calibration' if proof['conditional_fit_executed'] else 'evidence')
    result = read(OUT/(last+'.json'))
    failures = [k for k, v in result.get('gates',{}).items() if not v]
    lines = ['P6质量条件校准 v47', now(), '',
        f'全部任务总进度{progress[0]}%；实验总进度{progress[1]}%。当前候选已结束并停止。',
        '科学采用（既定已知合成开发范围）='+str(passed)+'；已执行范围工程复验及真实冷重入=True。',
        '最后执行阶段='+last+'；失败项='+str(failures),
        '缓存证据支持唯一候选='+str(proof['evidence_supported'])+'；新条件拟合已执行='+str(proof['conditional_fit_executed']),
        '原17项完整cal评价已执行='+str(proof['all17_executed'])+'；原17项开发eval已执行='+str(proof['evaluation_executed']),
        '复用v46全部1296条训练外head分数、折先验、行和折身份；0新head训练或OOF head预测。tau46已用全部OOF标签训练，当前证据不称整条方法的独立OOF验证。',
        'v为观测有效深度比例，h为未校准tau1后验熵；固定known行合并中位数，等号low，三折同号和固定0.02均差门槛，仅作一次计算准入，不是因果或统计显著性。',
        '证据不支持时不拟合、不跑cal/eval，全部17科学门槛保持null；支持时两质量量始终保留，最多一次四角温度NLL拟合及独立目标/梯度/KKT证书。',
        '固定Z双线性插值，theta∈[0,1]^4，unknown恒定tau46；正则0.001/2乘角点偏差平方，全1296行等权分母。最终head/H/深度尺度/原XYZ及固定5mm t3基线保持冻结。',
        '预先固定顺序：证据及完整replay→唯一fit及证书→原cal6及完整replay→全过才cal17→再全过才eval17；失败不调参、不追加新方案。',
        '没有跑到的门槛保持未验证，不能称全部17项通过。当前不能作为完整P6科学结果采用。' if not passed else '仅通过本开发协议；不等于真实传感器校准、独立最终测试或抓取收益。',
        c['scope'],
        '0新神经前向/官方AP/抓取/CAD调用；0新OOF head拟合/推理，最终head/H/depth重训=0；独立证书不另拟合候选。',
        '证据结果='+json.dumps(read(OUT/'evidence.json'),ensure_ascii=False),
        '全部门槛状态（null表示未执行）='+json.dumps(proof['gates_by_split'], ensure_ascii=False),
        '调用账本='+json.dumps(proof['calls'], ensure_ascii=False)]
    if proof['conditional_fit_executed']:
        report = read(OUT/'conditional_fit.json')
        lines += ['条件拟合='+json.dumps({k:v for k,v in report.items() if k != 'objective_trace'},ensure_ascii=False),
            '阶段结果='+json.dumps(result.get('gates',{}),ensure_ascii=False),
            '折head到full-fit head的分布迁移仍由原cal检验；局部损失改善不等于全部可靠性/锐度/几何门槛通过。']
    else:
        lines += ['本次固定质量证据未支持条件校准：已拒绝启动新拟合与后续评价，避免无证据重复计算；不将资源准入失败写成方法成功。']
    lines += ['P5科学结论不变。所有旧文件/失败/未知保留；原state.json不改。没有进入P7/P10，没有周期任务，等待用户检查后决定下一部分。']
    guard(); need(not REPORT.exists(), 'Existing report must not be overwritten'); REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    guard()
    once(OUT/'closeout.json', dict(at=now(), proof=proof, report=identity(REPORT), cold=identity(LOG/'cold.json'),
        stopped=True, engineering_passed=True, P6_development_scope_passed=passed,
        all_tasks_percent=progress[0], experiment_percent=progress[1], neural_forwards=0, official_AP=0,
        child_ledger=ledger(LOG/'children.jsonl')))
    status('completed_stopped', report=rel(REPORT), current_stage=last, scientific_passed=passed)
    if passed:
        state = read(STATE); state.update(all_tasks_percent=47, experiment_percent=58.75, P6_scientific_passed=True); atomic(STATE, state)
    supervise('v47_completed_stopped', current_runner=None, remaining_research_updates_cap=0,
        waiting_for_user_next_part_confirmation=True, summary_report=rel(REPORT), final_cold=rel(LOG/'cold.json'),
        final_readonly=rel(LOG/'evidence_review.json') if last == 'evidence' else rel(LOG/'warp_review.json') if last == 'warp_calibration' else rel(LOG/(last+'_review.json')),
        final_delivery_index=rel(OUT/'closeout.json'), all_tasks_percent_planning_estimate=progress[0], experiment_percent_planning_estimate=progress[1])
    handoff(f'P6 v47完成本次固定分支及完整复验/冷重入并停止。最后阶段{last}；证据支持={proof["evidence_supported"]}，科学采用={passed}，未运行关口保持null。全部任务总进度{progress[0]}%、实验总进度{progress[1]}%。报告{rel(REPORT)}。等待用户检查，不自动v48/P7/P10或周期任务。', 'complete')


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
            env = os.environ.copy(); env['P6_V47_CHILD_TOKEN'] = run_token
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
            supervise('v47_running')
            for stage in ('evidence', 'evidence_review'): run(stage)
            if read(OUT/'evidence.json')['supported']:
                for stage in ('fit', 'fit_review', 'warp_calibration', 'warp_review'): run(stage)
                if read(OUT/'warp_calibration.json')['passed']:
                    for stage in ('calibration', 'calibration_review'): run(stage)
                    if read(OUT/'calibration.json')['passed']:
                        for stage in ('evaluation', 'evaluation_review'): run(stage)
            run('cold'); guard(); finish(c, guard); guard()
        except BaseException as e:
            append(LOG/'controller.jsonl', dict(at=now(), event='failure_preserved', error=repr(e), traceback=traceback.format_exc()))
            status('failed_stopped', error=repr(e), unknown=ledger()['unknown'], unknown_children=ledger(LOG/'children.jsonl')['unknown'])
            supervise('v47_failed_stopped', current_runner=None, waiting_for_user_next_part_confirmation=True)
            handoff('P6 v47出现工程或资源异常，失败日志/模型/未知返回保留，先核实际进程与锁再恢复，不自动重复。全部任务总进度45%、实验总进度56.25%；科学通过未成立，不进P7/P10。', 'failure')
            raise

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'verify', 'deliver', 'worker'])
    parser.add_argument('--stage'); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'verify': print(json.dumps(dict(passed=True, protected_files=len(verify()['protected']))))
    elif args.mode == 'deliver': deliver()
    else:
        verify(); child_guard(); OUT.mkdir(parents=True, exist_ok=True)
        if args.stage == 'evidence': evidence()
        elif args.stage == 'evidence_review': evidence_review()
        elif args.stage == 'fit': fit()
        elif args.stage == 'fit_review': fit_review()
        elif args.stage == 'warp_calibration': warp_calibration()
        elif args.stage == 'warp_review': warp_review()
        elif args.stage in ('calibration', 'evaluation'): full_evaluate(args.stage)
        elif args.stage in ('calibration_review', 'evaluation_review'): full_evaluate(args.stage.removesuffix('_review'), True)
        elif args.stage == 'cold': cold()
        else: raise ValueError('Unknown worker stage')
        child_guard()
if __name__ == '__main__': main()
