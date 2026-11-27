"""One strictly nested, five-feature geometric logit repair; frozen science gates.

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
LOG = ROOT/'logs/P6/geometric_logits_v48'
OUT = ROOT/'results/P6/geometric_logits_v48'
CONFIG = ROOT/'configs/p6_geometric_logits_v48.json'
STATE = ROOT/'runtime/pipeline/p6_geometric_logits_v48.json'
PROTOCOL = ROOT/'docs/P6几何评分结构修复协议_v48.txt'
REPORT = ROOT/'docs/P6几何评分结构修复结果_v48.txt'
OLD = ROOT/'results/P6/mixture_v43'
PARENT = ROOT/'results/P6/probability_calibration_v46'
PREVIOUS = ROOT/'results/P6/quality_calibration_v47'
DATA = ROOT/'logs/P6/correspondence_v40/data_audit.json'
ORIGINAL_STATE = 'b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'
SOURCES = ['scripts/p6_geometric_logits_v48.py', 'src/p6_geometric_logits_v48.py',
           'tests/test_p6_geometric_logits_v48.py', 'tests/test_p6_geometric_independent_v48.py',
           'scripts/check_p6_geometric_logits_v48.py']
GATES = ('warp_nll_gain', 'warp_brier_noninferior', 'warp95_coverage', 'warp_set_sharper',
         'warp_multilevel_reliability', 'worst_scene_warp', 'worst_scene_depth',
         'source_depth95', 'source_depth_sharpness', 'source_depth_nll_gain',
         'target_joint_nll_noninferior', 'target_missing_brier_noninferior',
         'target_conditional_coverage', 'source_joint95', 'target_union_sharpness',
         'target_union95', 'observation_preserved')
LIMITS = dict(nested_head_fit=3,nested_head_certificate=3,nested_cross_prediction=6,nested_cross_replay=6,
              outer_fit=3,outer_certificate=3,outer_prediction=3,outer_replay=3,nested_gate=1,nested_gate_replay=1,
              final_fit=1,final_certificate=1,warp_calibration=1,warp_calibration_replay=1,
              calibration_posterior=1,calibration_review_posterior=1,evaluation_posterior=1,evaluation_review_posterior=1,
              calibration_frame=16,calibration_review_frame=16,evaluation_frame=16,evaluation_review_frame=16,
              calibration_case=432,calibration_review_case=432,evaluation_case=432,evaluation_review_case=432)


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
                      current_part='P6_geometric_logits_v48', P6_scientific_passed=False,
                      P7_started=False, P10_started=False, recurring_automation=False, **fields))
def supervise(phase, **fields):
    path = ROOT/'runtime/pipeline/part_level_supervision.json'; backup(path, phase)
    value = read(path)
    value.update(recorded_at=now(), current_authorized_part='P6_geometric_logits_v48',
        current_execution_state=phase, current_runner=rel(Path(__file__)),
        current_execution_evidence=rel(STATE), next_repair_started=True,
        user_authorization='用户授权固定五维几何评分结构修复，严格nested训练准入及完整分支验收，完成或失败停止',
        automatic_resume_authorized=False, automatic_scheduler_recreated=False,
        stop_after='P6_geometric_logits_v48', stop_after_part='P6_geometric_logits_v48',
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
    path = ROOT/'runtime/pipeline/p6_geometric_logits_v48.lock'
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
    need(os.environ.get('P6_V48_CHILD_TOKEN') == state.get('run_token') and bool(state.get('run_token')), 'Worker requires active controller')
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
        from scripts import p6_quality_calibration_v47 as previous
        oldconfig = previous.verify(); previous_done = read(PREVIOUS/'closeout.json')
        need(previous.verify_executed() == previous_done['proof'] and previous_done['engineering_passed'], 'v47 completion invalid')
        need(previous_done['cold'] == identity(previous.LOG/'cold.json') and previous_done['report'] == identity(previous.REPORT), 'v47 cold/report differs')
        need(previous_done['child_ledger'] == previous.ledger(previous.LOG/'children.jsonl'), 'v47 child ledger differs')
        protected = dict(oldconfig['protected'])
        dependencies = SOURCES + [rel(PROTOCOL), 'docs/P6几何评分结构修复独立设计审查_v48.txt',
            'configs/p6_quality_calibration_v47.json',rel(previous.REPORT),rel(DATA),
            rel(LOG/'internal_checks.json'),rel(LOG/'independent_review.json')]
        dependencies += [rel(p) for folder in (PARENT,PREVIOUS) for p in folder.rglob('*') if p.is_file()]
        dependencies += [rel(p) for p in previous.LOG.rglob('*') if p.is_file()]
        dependencies += ['results/P6/joint_repair_v45/model.json']
        protected.update({name: identity(ROOT/name) for name in dependencies})
        measured = sum(float(read(p)['seconds']) for p in (LOG/'preparation').glob('*.json'))
        c = dict(at=now(), version='v48', protected=protected, call_limits=LIMITS,
            controller_wall_seconds=3600, preparation_seconds=measured+30., cpu_threads=2,
            process_tree_rss_limit=3*1024**3, output_limit=512*1024**2, free_disk_min=60*1024**3,
            original_gates_unchanged=True, train_cases=1296, calibration_cases=432, exposed_evaluation_cases=432,
            folds=[[3,8,12,17], [4,9,14,19], [5,10,16,20]],
            nested=dict(head_fits=3,head_fit_scenes=4,head_fit_rows=432,cross_score_blocks=6,
                outer_correctors=3,meta_train_rows_per_outer=864,heldout_rows_per_outer=432,
                outer_input='frozen v46 raw tau1 OOF; no v46 tau or v47 theta offset',
                gate='pooled NLL gain>=.01 and each3fold gain>=0 and pooled Brier noninferior',
                features=['rx','ry','rx_squared','rx_times_ry','ry_squared'],residual_scale=8,
                final_corrector_fits_if_nested_pass=1,final_train_rows=1296),
            optimizer=dict(corrector_maxiter=60,corrector_maxfun=100,hard_objectives_per_fit=120,
                ridge=.001,parameters=5,bounds=[[-2.,2.]]*5,initial=[0.]*5,kkt_tolerance=1e-5,
                corrector_ftol=1e-12,corrector_gtol=1e-8,maxls=15,
                head_maxiter=60,head_maxfun=100,head_ftol=1e-8,head_gtol=1e-5,
                nonconvergence_is_engineering_failure=True),
            reused=dict(full_fit_head_refits=0,shape_fits=0,depth_scale_refits=0,parent_model=rel(PARENT/'model.json')),
            stages='nested fit+complete replay/certificates -> if supported final fit+certificate -> cal6+replay -> if pass cal17+replay -> if pass eval17+replay -> cold -> report -> stop',
            failed_gate_action='Stop this candidate after full replay of executed scope; do not tune or run the deferred expensive scope',
            scope='Known synthetic global integer warps on previously exposed GraspNet RealSense development scenes; not independent final test',
            neural_forwards=0, official_AP=0, CAD_renders=0, grasp_calls=0,
            progress_before=[45, 56.25], progress_after_full_pass=[47, 58.75])
        protected_check(c)
        need(c['preparation_seconds'] < 3600 and shutil.disk_usage(ROOT).free >= c['free_disk_min'], 'Insufficient reserved resources')
        once(CONFIG, c); once(LOG/'freeze.json', dict(at=now(), config=identity(CONFIG)))
        need(time.monotonic()-started <= 30, 'Freeze reserve exceeded')
        status('frozen_ready'); supervise('v48_frozen_ready')
        handoff('P6 v48几何评分结构修复已冻结：最多3个四场景head、6块跨组预测、3个严格nested校正器；nested准入全过才最终1次校正器拟合及原cal6→cal17→eval17。全部已执行内容完整复验/独立证书/cold后停止。不使用v46温度或v47条件层，不自动v49/P7/P10。全部任务总进度45%、实验总进度56.25%。入口scripts/p6_geometric_logits_v48.py deliver；禁止重复。','freeze')
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
    """Reuse frozen raw OOF scores, never either later calibration layer."""
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_read
    from src.p6_probability_calibration_v46 import validate_oof_coverage
    x,y,rows = training_arrays(); oof = arrays_read(PARENT/'oof.npz')
    scenes = np.array([r['scene'] for r in rows])
    need(rows == read(PARENT/'oof_rows.json') and np.array_equal(y,oof['targets']) and np.array_equal(scenes,oof['scenes']), 'Frozen OOF row alignment differs')
    validate_oof_coverage(scenes,oof['assigned_folds'],oof['visits'],expected_cases_per_scene=108)
    for outer in range(3):
        held=np.flatnonzero(oof['assigned_folds'] == outer);train=np.flatnonzero(oof['assigned_folds'] != outer)
        saved=read(PARENT/f'fold_{outer}.json')
        need(saved['training_indices'] == train.tolist() and saved['holdout_indices'] == held.tolist()
             and saved['training_scenes'] == sorted(set(scenes[train].tolist()))
             and saved['holdout_scenes'] == sorted(set(scenes[held].tolist())), 'Inherited outer head actual train/heldout provenance differs')
    parent=read(PARENT/'model.json'); original=read(ROOT/'results/P6/joint_repair_v45/model.json')
    need(parent['warp']['base_model'] == original['warp'], 'Raw v45 full head identity differs')
    need(all(parent[k] == original[k] for k in ('shape','depth','empirical_prior')), 'Frozen full parent geometry differs')
    return x,y,rows,oof,parent


def fold_indices(oof,fold):
    import numpy as np
    from src.p6_probability_calibration_v46 import FOLDS
    assigned=np.asarray(oof['assigned_folds']); scenes=np.asarray(oof['scenes'])
    indices=np.flatnonzero(assigned == fold)
    need(len(indices) == 432 and set(scenes[indices].tolist()) == set(FOLDS[fold]), 'Nested fixed fold scope differs')
    return indices


def persist_fit(kind,key,filename,fit_function):
    """All actual optimizer evaluations are durable before a fit can return."""
    trace_path=LOG/(key+'_objective.jsonl')
    def execute():
        return fit_function(lambda row:append(trace_path,dict(at=now(),**row)))
    try: report=call(kind,key,execute)
    except BaseException as exc:
        once(OUT/(key+'_failure.json'),dict(at=now(),error=repr(exc),
            partial_report=getattr(exc,'partial_report',None),traceback=traceback.format_exc()))
        raise
    once(OUT/filename,report)
    need(report['optimizer_converged'], 'Optimizer nonconvergence; full report preserved '+key)
    if kind != 'nested_head_fit': need(report['kkt_passed'], 'Corrector KKT failed; report preserved '+key)
    child_guard(); return report


def trace_check(key,expected):
    rows=events(LOG/(key+'_objective.jsonl'))
    need([{k:v for k,v in r.items() if k != 'at'} for r in rows] == expected and 0 < len(rows) <= 120,
         'Durable actual objective trace differs '+key)
    return identity(LOG/(key+'_objective.jsonl'))


def meta_training(outer,oof):
    """Use only Fj/Fk labels and predictions made by a head trained on the other group."""
    import numpy as np
    from scripts.p6_mixture_v43 import arrays_read
    other=[f for f in range(3) if f != outer]; blocks=[]; provenance=[]
    for target in other:
        source=next(f for f in other if f != target)
        held=fold_indices(oof,target); train=fold_indices(oof,source)
        saved=arrays_read(OUT/f'cross_{source}_to_{target}.npz')
        need(np.array_equal(saved['indices'],held) and not np.intersect1d(train,held).size, 'Cross-score indices differ')
        need(source != outer and target != outer, 'Outer group leaked into meta-training')
        blocks.append((held,saved['scores'],saved['priors']))
        provenance.append(dict(source_head_fold=source,target_fold=target,training_indices=train.tolist(),target_indices=held.tolist()))
    indices=np.concatenate([b[0] for b in blocks]); scores=np.concatenate([b[1] for b in blocks]); priors=np.concatenate([b[2] for b in blocks])
    held=fold_indices(oof,outer)
    need(len(indices) == 864 and len(set(indices.tolist())) == 864 and not np.intersect1d(indices,held).size, 'Outer meta-train overlap')
    return indices,scores,priors,provenance


def nested_fit():
    import numpy as np
    from src import p6_geometric_logits_v48 as core
    from scripts.p6_mixture_v43 import arrays_save
    x,y,rows,oof,parent=oof_inputs(); once(OUT/'nested_rows.json',rows)
    for source in range(3):
        train=fold_indices(oof,source)
        head=persist_fit('nested_head_fit',f'head_{source}',f'head_{source}.json',lambda sink:
            core.fit_base_head(x[train],y[train],guard=child_guard,on_evaluation=sink))
        once(OUT/f'head_{source}_provenance.json',dict(source_fold=source,training_indices=train.tolist(),
            training_scenes=sorted(set(oof['scenes'][train].tolist())),input_sha256=sha(OLD/'generated.npz')))
        for target in range(3):
            if target == source: continue
            held=fold_indices(oof,target)
            score=call('nested_cross_prediction',f'{source}_to_{target}',lambda:core.score_warp(head,x[held]))
            priors=np.broadcast_to(np.asarray(head['empirical_prior']),score.shape).copy()
            arrays_save(OUT/f'cross_{source}_to_{target}.npz',scores=score,priors=priors,indices=held)
            child_guard()
    probability=np.empty((len(y),81)); visits=np.zeros(len(y),dtype=np.int64)
    for outer in range(3):
        train,scores,priors,provenance=meta_training(outer,oof); held=fold_indices(oof,outer)
        correction=persist_fit('outer_fit',f'outer_{outer}',f'outer_{outer}.json',lambda sink:
            core.fit_geometry_residual(scores,priors,y[train],guard=child_guard,on_evaluation=sink))
        once(OUT/f'outer_{outer}_provenance.json',dict(outer_fold=outer,meta_training_indices=train.tolist(),
            heldout_indices=held.tolist(),meta_training_scenes=sorted(set(oof['scenes'][train].tolist())),
            heldout_scenes=sorted(set(oof['scenes'][held].tolist())),cross_training=provenance,
            outer_raw_OOF_identity=identity(PARENT/'oof.npz'),outer_v46_model=identity(PARENT/f'fold_{outer}.json')))
        probability[held]=call('outer_prediction',str(outer),lambda:
            core.predict_residual(oof['scores'][held],oof['priors'][held],correction))
        visits[held]+=1; child_guard()
    need(np.array_equal(visits,np.ones(len(y),dtype=np.int64)), 'Outer predictions lack one exact visit')
    arrays_save(OUT/'nested.npz',probability=probability,visits=visits,assigned_folds=oof['assigned_folds'])
    result=call('nested_gate','heldout1296',lambda:core.nested_gate(oof['scores'],oof['priors'],y,probability,oof['scenes'],oof['assigned_folds']))
    once(OUT/'nested_gate.json',result)


def head_certificate(features,targets,head):
    """Independent original-v45 objective/gradient replay; no stricter optimizer gate."""
    import numpy as np
    from scipy.special import logsumexp
    x=np.asarray(features,float); y=np.asarray(targets); n=len(y)
    center=x.mean(axis=(0,1)); scale=x.std(axis=(0,1)); scale=np.where(scale>1e-8,scale,1.)
    counts=np.bincount(y,minlength=81).astype(float)+1/81; prior=counts/counts.sum()
    need(np.array_equal(center,np.asarray(head['center'])) and np.array_equal(scale,np.asarray(head['scale'])) and
         np.array_equal(prior,np.asarray(head['empirical_prior'])), 'Head fit-only normalizer/prior differs')
    x=np.clip((x-center)/scale,-8.,8.); records=head['objective_evaluation_records']; trace=head['objective_trace']
    need(0 < len(records) == len(trace) == head['objective_evaluations'] <= 120 and head['optimizer_converged'] is True and head['optimizer_iterations'] <= 60, 'Head optimization accounting/convergence differs')
    need(records[0]['theta'] == [0.]*16 and any(r['theta'] == head['coefficients'] for r in records), 'Head initial/final point provenance differs')
    for record,loss in zip(records,trace):
        theta=np.asarray(record['theta']); logits=np.einsum('bkf,f->bk',x,theta)+np.log(prior)[None]
        logp=logits-logsumexp(logits,axis=1,keepdims=True); error=np.exp(logp); error[np.arange(n),y]-=1.
        value=float(-logp[np.arange(n),y].mean()+.5*.001*np.dot(theta,theta))
        gradient=np.einsum('bk,bkf->f',error/n,x)+.001*theta
        need(np.isclose(value,record['value'],rtol=1e-10,atol=1e-12) and np.isclose(value,loss,rtol=1e-10,atol=1e-12)
             and np.allclose(gradient,record['gradient'],rtol=1e-10,atol=1e-12), 'Head objective/gradient trace differs')
    return dict(passed=True,objective_trace_points_verified=len(records),optimization_refits=0,training_rows=n,
        original_optimizer_convergence_required=True,extra_corrector_KKT_not_imposed=True)


def geometric_certificate(scores,priors,targets,report):
    """Independent five-feature convex objective, gradient and bounded KKT replay."""
    import numpy as np
    from scipy.special import logsumexp
    from src.p6_joint_repair_v45 import CANDIDATES
    s,pi,y=np.asarray(scores,float),np.asarray(priors,float),np.asarray(targets); n=len(y)
    logits0=np.log(pi)+s; logp0=logits0-logsumexp(logits0,axis=1,keepdims=True); p0=np.exp(logp0)
    candidates=np.asarray(CANDIDATES,float); mean=p0@candidates; r=(candidates[None]-mean[:,None])/8.
    rx,ry=r[:,:,0],r[:,:,1]; phi=np.stack((rx,ry,rx*rx,rx*ry,ry*ry),axis=-1)
    need(report['ridge'] == .001 and report['bounds'] == [[-2.,2.]]*5 and report['kkt_tolerance'] == 1e-5, 'Corrector fixed contract changed')
    checked={}
    def point(beta):
        key=tuple(beta)
        if key in checked:return checked[key]
        beta=np.asarray(beta,float); need(beta.shape == (5,) and np.isfinite(beta).all() and np.all((beta>=-2)&(beta<=2)), 'Invalid geometric coefficient')
        logits=logits0+phi@beta; logp=logits-logsumexp(logits,axis=1,keepdims=True)
        delta=np.exp(logp); delta[np.arange(n),y]-=1.
        nll=float(-logp[np.arange(n),y].mean()); penalty=float(.5*.001*np.dot(beta,beta))
        gradient=np.einsum('bk,bkf->f',delta/n,phi)+.001*beta
        checked[key]=dict(value=nll+penalty,gradient=gradient.tolist(),nll=nll,penalty=penalty)
        return checked[key]
    trace=report['objective_trace']; beta=np.asarray(report['beta'],float)
    need(0 < len(trace) == report['objective_evaluations'] <= 120 and trace[0]['beta'] == [0.]*5 and
         any(v['beta'] == beta.tolist() for v in trace), 'Corrector objective provenance/budget differs')
    for row in trace:
        actual=point(row['beta'])
        for field in ('value','gradient','nll','penalty'):
            need(np.allclose(actual[field],row[field],rtol=1e-10,atol=1e-12), 'Corrector trace differs '+field)
    final=point(beta)
    for field,target in [('value','final_objective'),('gradient','final_gradient'),('nll','final_nll'),('penalty','final_penalty')]:
        need(np.allclose(final[field],report[target],rtol=1e-10,atol=1e-12), 'Corrector final point differs '+field)
    need(np.isclose(point([0.]*5)['nll'],report['baseline_nll'],rtol=1e-10,atol=1e-12), 'Raw tau1 baseline NLL differs')
    residual=beta-np.clip(beta-np.asarray(final['gradient']),-2.,2.); norm=float(np.max(np.abs(residual)))
    need(report['optimizer_converged'] is True and report['optimizer_iterations'] <= 60 and report['kkt_passed'] is True
         and norm <= 1e-5 and np.isclose(norm,report['kkt_projected_gradient_inf'],rtol=1e-10,atol=1e-12), 'Corrector convergence/KKT failed')
    need(final['value'] <= point([0.]*5)['value']+1e-10 and len(checked)<=len(trace), 'Corrector objective worse or certificate searched new points')
    return dict(passed=True,beta=beta.tolist(),final_objective=final['value'],final_nll=final['nll'],
        final_gradient=final['gradient'],kkt_projected_gradient_inf=norm,kkt_tolerance=1e-5,
        objective_trace_points_verified=len(trace),certificate_objective_evaluations=len(checked),optimization_refits=0)


def nested_names():
    return {'nested_rows.json','nested.npz','nested_gate.json'} | {
        f'{kind}_{fold}{suffix}.json' for kind in ('head','outer') for fold in range(3) for suffix in ('','_provenance')
    } | {f'cross_{source}_to_{target}.npz' for source in range(3) for target in range(3) if source != target}


def nested_review():
    import numpy as np
    from src import p6_geometric_logits_v48 as core
    from scripts.p6_mixture_v43 import arrays_read,close_numeric
    x,y,rows,oof,parent=oof_inputs(); need(rows == read(OUT/'nested_rows.json'), 'Nested row identity differs')
    certificates={}; traces={}
    for source in range(3):
        train=fold_indices(oof,source); head=read(OUT/f'head_{source}.json'); prov=read(OUT/f'head_{source}_provenance.json')
        need(prov == dict(source_fold=source,training_indices=train.tolist(),training_scenes=sorted(set(oof['scenes'][train].tolist())),input_sha256=sha(OLD/'generated.npz')), 'Small head provenance differs')
        traces[f'head_{source}']=trace_check(f'head_{source}',head['objective_evaluation_records'])
        certificates[f'head_{source}']=call('nested_head_certificate',str(source),lambda:head_certificate(x[train],y[train],head))
        for target in range(3):
            if target == source:continue
            held=fold_indices(oof,target); saved=arrays_read(OUT/f'cross_{source}_to_{target}.npz')
            score=call('nested_cross_replay',f'{source}_to_{target}',lambda:core.score_warp(head,x[held]))
            need(np.array_equal(saved['indices'],held) and np.array_equal(saved['scores'],score) and
                 np.array_equal(saved['priors'],np.broadcast_to(head['empirical_prior'],score.shape)), 'Every cross-head prediction must replay exactly')
            child_guard()
    fresh=np.empty((len(y),81)); visits=np.zeros(len(y),dtype=np.int64)
    for outer in range(3):
        train,scores,priors,provenance=meta_training(outer,oof); held=fold_indices(oof,outer)
        correction=read(OUT/f'outer_{outer}.json'); prov=read(OUT/f'outer_{outer}_provenance.json')
        expected=dict(outer_fold=outer,meta_training_indices=train.tolist(),heldout_indices=held.tolist(),
            meta_training_scenes=sorted(set(oof['scenes'][train].tolist())),heldout_scenes=sorted(set(oof['scenes'][held].tolist())),
            cross_training=provenance,outer_raw_OOF_identity=identity(PARENT/'oof.npz'),outer_v46_model=identity(PARENT/f'fold_{outer}.json'))
        need(prov == expected,'Nested corrector train/heldout chain differs')
        traces[f'outer_{outer}']=trace_check(f'outer_{outer}',correction['objective_trace'])
        certificates[f'outer_{outer}']=call('outer_certificate',str(outer),lambda:geometric_certificate(scores,priors,y[train],correction))
        fresh[held]=call('outer_replay',str(outer),lambda:core.predict_residual(oof['scores'][held],oof['priors'][held],correction))
        visits[held]+=1; child_guard()
    saved=arrays_read(OUT/'nested.npz')
    need(np.array_equal(fresh,saved['probability']) and np.array_equal(visits,saved['visits']) and
         np.array_equal(visits,np.ones(len(y),dtype=np.int64)) and np.array_equal(oof['assigned_folds'],saved['assigned_folds']), 'Outer probability/visit replay differs')
    gate=call('nested_gate_replay','heldout1296',lambda:core.nested_gate(oof['scores'],oof['priors'],y,fresh,oof['scenes'],oof['assigned_folds']))
    close_numeric(gate,read(OUT/'nested_gate.json')); child_guard()
    once(LOG/'nested_review.json',dict(at=now(),passed=True,supported=gate['supported'],certificates=certificates,
        outputs={name:identity(OUT/name) for name in sorted(nested_names())},traces=traces,
        reused_parent=identity(PARENT/'model.json'),reused_oof=identity(PARENT/'oof.npz'),input_features=identity(OLD/'generated.npz'),
        scope='Every new head objective/gradient and fit-only normalizer/prior; all6 cross prediction blocks; all3 outer corrector certificates and heldout predictions; entire gate replay, no refit'))


def fit():
    from src import p6_geometric_logits_v48 as core
    need(read(OUT/'nested_gate.json')['supported'] and read(LOG/'nested_review.json')['passed'],'Nested prerequisite failed')
    prerequisite=read(LOG/'nested_review.json')
    for name,binding in prerequisite['outputs'].items():need(identity(OUT/name) == binding,'Nested reviewed prerequisite changed '+name)
    x,y,rows,oof,parent=oof_inputs()
    correction=persist_fit('final_fit','final','geometric_fit.json',lambda sink:
        core.fit_geometry_residual(oof['scores'],oof['priors'],y,guard=child_guard,on_evaluation=sink))
    warp=core.wrap_model(parent['warp']['base_model'],correction)
    once(OUT/'model.json',dict(at=now(),warp=warp,shape=parent['shape'],depth=parent['depth'],empirical_prior=parent['empirical_prior'],
        calibration=correction,input_sha256=sha(OLD/'generated.npz'),parent_model_sha256=sha(PARENT/'model.json'),
        oof_sha256=sha(PARENT/'oof.npz'),fit_scenes=read(DATA)['scenes']['fit'],calibration_partition_fits=0,
        depth_scale_refits=0,shape_fits=0,full_head_refits=0,nested_head_fits=3,raw_tau1=True,v46_tau_used=False,v47_theta_used=False,GT_in_predictor=False))


def fit_review():
    from src import p6_geometric_logits_v48 as core
    from scripts.p6_mixture_v43 import close_numeric
    x,y,rows,oof,parent=oof_inputs();model=read(OUT/'model.json');correction=read(OUT/'geometric_fit.json')
    need(model['calibration'] == correction and all(model[k] == parent[k] for k in ('depth','shape','empirical_prior')),'Frozen parent/report changed')
    close_numeric(model['warp'],core.wrap_model(parent['warp']['base_model'],correction))
    trace=trace_check('final',correction['objective_trace'])
    certificate=call('final_certificate','independent_convex_geometry_KKT',lambda:geometric_certificate(oof['scores'],oof['priors'],y,correction))
    child_guard()
    once(LOG/'fit_review.json',dict(at=now(),passed=True,certificate=certificate,
        outputs={name:identity(OUT/name) for name in ('geometric_fit.json','model.json')},objective_trace=trace,
        reused_parent=identity(PARENT/'model.json'),reused_oof=identity(PARENT/'oof.npz'),
        scope='Independent all-saved-point objective/gradient and bounded KKT certificate; raw final head/H/depth/prior unchanged'))


def warp_metric(weights, target):
    from src import p6_geometric_logits_v48 as core
    value = core.warp_statistics(weights, int(target))
    value['levels'] = {str(v['level']): dict(covered=v['covered'], size=v['size'], mass=v['probability'])
                       for v in value.pop('credible_sets')}
    return value

def warp_result(replay=False):
    import numpy as np
    from src import p6_geometric_logits_v48 as core
    from scripts.p6_joint_evaluate_v45 import aggregate_warp
    fit_proof=read(LOG/'fit_review.json')
    need(fit_proof['passed'] and fit_proof['outputs']['model.json'] == identity(OUT/'model.json'), 'Final model certificate prerequisite changed')
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
    need(read(LOG/'fit_review.json')['passed'], 'Independent fit certificate prerequisite failed')
    fresh = call('warp_calibration_replay', 'calibration432', lambda: warp_result(True))
    close_numeric(fresh, read(OUT/'warp_calibration.json'))
    once(LOG/'warp_review.json', dict(at=now(), passed=True, result=identity(OUT/'warp_calibration.json'),
        posterior=identity(OUT/'calibration_warp.npz'), model=identity(OUT/'model.json'),
        scope='Complete deterministic 432 posterior/metric replay; prior strictly nested gate and independent final fit certificate verified'))

def full_evaluate(split, replay=False):
    import numpy as np
    from src import p6_geometric_logits_v48 as core
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
    stages=['nested_fit','nested_review']
    if not read(OUT/'nested_gate.json')['supported']: return stages
    stages+=['fit','fit_review','warp_calibration','warp_review']
    if read(OUT/'warp_calibration.json')['passed']:
        stages+=['calibration','calibration_review']
        if read(OUT/'calibration.json')['passed']:stages+=['evaluation','evaluation_review']
    return stages


def verify_executed():
    from scripts.p6_joint_evaluate_v45 import aggregate_warp,aggregate
    from scripts.p6_mixture_v43 import close_numeric
    stages=executed_stages();nested=read(OUT/'nested_gate.json');proof=read(LOG/'nested_review.json')
    expected=dict(nested_head_fit=3,nested_head_certificate=3,nested_cross_prediction=6,nested_cross_replay=6,
        outer_fit=3,outer_certificate=3,outer_prediction=3,outer_replay=3,nested_gate=1,nested_gate_replay=1)
    names=nested_names();need(proof['passed'] and proof['supported'] == nested['supported'],'Nested replay decision differs')
    need(set(proof['outputs']) == names and set(proof['certificates']) == {f'{kind}_{i}' for kind in ('head','outer') for i in range(3)},'Nested review inventory incomplete')
    for name,binding in proof['outputs'].items():need(identity(OUT/name) == binding,'Reviewed nested artifact changed '+name)
    for key,binding in proof['traces'].items():
        report=read(OUT/(key+'.json'));expected_trace=report['objective_evaluation_records'] if key.startswith('head_') else report['objective_trace']
        need(trace_check(key,expected_trace) == binding,'Reviewed optimizer trace changed '+key)
    need(set(proof['traces']) == {f'{kind}_{i}' for kind in ('head','outer') for i in range(3)}, 'Nested reviewed trace inventory incomplete')
    need(all(v['passed'] for v in proof['certificates'].values()),'Nested certificate failed')
    need(proof['reused_parent'] == identity(PARENT/'model.json') and proof['reused_oof'] == identity(PARENT/'oof.npz') and
         proof['input_features'] == identity(OLD/'generated.npz'),'Nested original input identity differs')
    gates_by_split={split:{name:None for name in GATES} for split in ('calibration','evaluation')}
    if nested['supported']:
        expected.update(final_fit=1,final_certificate=1,warp_calibration=1,warp_calibration_replay=1)
        names|={'geometric_fit.json','model.json','calibration_warp.npz','warp_calibration.json'}
        model=read(OUT/'model.json');parent=read(PARENT/'model.json')
        need(all(model[k] == parent[k] for k in ('depth','shape','empirical_prior')) and
             model['warp']['base_model'] == parent['warp']['base_model'] and model['calibration'] == read(OUT/'geometric_fit.json'),'Frozen final parent/report changed')
        fit_proof=read(LOG/'fit_review.json');need(fit_proof['passed'] and fit_proof['certificate']['passed'],'Final certificate not passed')
        for name,binding in fit_proof['outputs'].items():need(identity(OUT/name) == binding,'Reviewed final fit artifact changed '+name)
        need(trace_check('final',model['calibration']['objective_trace']) == fit_proof['objective_trace'],'Final trace changed')
        need(fit_proof['reused_parent'] == identity(PARENT/'model.json') and fit_proof['reused_oof'] == identity(PARENT/'oof.npz'),'Final fit input changed')
        warp_proof=read(LOG/'warp_review.json');need(warp_proof['passed'],'Warp replay not passed')
        for field,name in [('result','warp_calibration.json'),('posterior','calibration_warp.npz'),('model','model.json')]:
            need(warp_proof[field] == identity(OUT/name),'Reviewed warp artifact changed')
        for split in ['warp_calibration']+[s for s in ('calibration','evaluation') if s in stages]:
            result=read(OUT/(split+'.json'))
            summary,gates=aggregate_warp(result['rows'],split='calibration') if split == 'warp_calibration' else aggregate(result['rows'],split=split)
            close_numeric(summary,result['summary']);need(gates == result['gates'] and result['passed'] == all(gates.values()),'Stored scientific gates differ')
            if split != 'warp_calibration':
                names.add(split+'.json');review=read(LOG/(split+'_review.json'))
                need(review['passed'] and review['result'] == identity(OUT/(split+'.json')),'Full replay identity differs')
                for stage in (split,split+'_review'):expected.update({stage+'_posterior':1,stage+'_frame':16,stage+'_case':432})
        gates_by_split['calibration'].update(read(OUT/'warp_calibration.json')['gates'])
        for split in ('calibration','evaluation'):
            if split in stages:
                actual=read(OUT/(split+'.json'))['gates'];need(set(actual) == set(GATES),'Original17 gate names differ')
                gates_by_split[split].update(actual)
    calls=ledger();need(not calls['unknown'] and calls['counts'] == expected,'Executed scientific ledger differs')
    actual={p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    need(actual-{'closeout.json'} == names,'Unexpected/unreviewed output artifact')
    return dict(stages=stages,calls=calls,outputs={name:identity(OUT/name) for name in sorted(names)},
        gates_by_split=gates_by_split,nested_supported=nested['supported'],final_fit_executed='fit' in stages,
        development_scientific_passed='evaluation' in stages and read(OUT/'evaluation.json')['passed'],
        all17_executed='calibration' in stages,evaluation_executed='evaluation' in stages)


def cold():
    proof = verify_executed(); child_guard()
    once(LOG/'cold.json', dict(at=now(), passed=True, fresh_process_pid=os.getpid(),
        scope='Read-only full executed inventory, summaries, original gates and call ledger', proof=proof))

def finish(c,guard):
    proof=verify_executed();cold_proof=read(LOG/'cold.json')
    need(cold_proof['passed'] and cold_proof['proof'] == proof,'Cold/executed inventory differs')
    protected_check(c);guard();passed=proof['development_scientific_passed'];progress=[47,58.75] if passed else [45,56.25]
    last=('evaluation' if proof['evaluation_executed'] else 'calibration' if proof['all17_executed']
          else 'warp_calibration' if proof['final_fit_executed'] else 'nested_gate')
    result=read(OUT/(last+'.json'));failures=[k for k,v in result.get('gates',{}).items() if not v]
    lines=['P6几何评分结构修复 v48',now(),'',
        f'全部任务总进度{progress[0]}%；实验总进度{progress[1]}%。本固定候选已结束并停止。',
        '科学采用（既定已知合成开发范围）='+str(passed)+'；已执行完整范围工程复验及真实冷重入=True。',
        '最后执行阶段='+last+'；失败项='+str(failures),
        '严格nested训练准入='+str(proof['nested_supported'])+'；最终1296行校正器拟合已执行='+str(proof['final_fit_executed']),
        '原17项完整cal评价已执行='+str(proof['all17_executed'])+'；原17项开发eval已执行='+str(proof['evaluation_executed']),
        '固定3场景组：每组4场景训练一个原v45 head，6块跨组分数。每个outer校正器只用另两组的交叉head预测及标签训练；其outer预测使用不含该组训练标签的既有v46 raw OOF head。',
        '不使用全OOF拟合的v46 tau或v47 theta作offset，避免把外层标签引入nested链。outer12场景共1296行各一次；每个head/校正器全部实际目标及梯度轨迹均独立核验，不重复拟合。',
        'p0为raw tau1后验，候选位移减p0期望并除8，五维[rx,ry,rx²,rxry,ry²]；beta固定[-2,2]^5，初值0，正则0.001/2乘平方范数。不是再扫温度，允许调整候选相对排序。',
        'nested准入固定：合并NLL改善>=0.01、三折各自NLL改善>=0、合并Brier非劣。准入失败即不训练最终校正器、不运行cal/eval，全部原17门槛保持null。',
        'nested过才一次最终校正器拟合；随后原cal6及完整replay→全过才cal17→全过才eval17；任何失败不调参、不追加版本。',
        '只有已执行范围通过工程复验，未运行科学门槛保持未验证；当前不能作为完整P6科学结果采用。' if not passed else '通过本限定开发协议；不等于真实传感器校准、独立最终测试或真实抓取收益。',
        c['scope'],
        '0新神经前向/官方AP/抓取/CAD调用；最多3个轻量head及3外层+1最终校正器拟合，旧v45最终head/H/深度重训=0。最终head/H/depth/原XYZ、全部81正支持、固定5mm t3基线和原17科学门槛均保留。',
        '严格nested结果='+json.dumps(read(OUT/'nested_gate.json'),ensure_ascii=False),
        '全部门槛状态（null表示未执行）='+json.dumps(proof['gates_by_split'],ensure_ascii=False),
        '调用账本='+json.dumps(proof['calls'],ensure_ascii=False)]
    if proof['final_fit_executed']:
        fit_report=read(OUT/'geometric_fit.json')
        lines+=['最终校正器='+json.dumps({k:v for k,v in fit_report.items() if k != 'objective_trace'},ensure_ascii=False),
                '阶段结果='+json.dumps(result.get('gates',{}),ensure_ascii=False)]
    else:lines+=['严格训练外准入未通过，最终校正器和cal/eval均未启动；负结果完整保留，不能称方法采用成功。']
    lines+=['4场景内部head到8场景外层head再到12场景最终head存在分布迁移，nested与原cal分别检验，不能预设可迁移。',
        '原P5结论不变；历史state.json及所有冻结材料不改。没有进入P7/P10或周期任务，完成本次后等待用户检查。']
    guard();need(not REPORT.exists(),'Existing report must not be overwritten');REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8');guard()
    once(OUT/'closeout.json',dict(at=now(),proof=proof,report=identity(REPORT),cold=identity(LOG/'cold.json'),stopped=True,
        engineering_passed=True,P6_development_scope_passed=passed,all_tasks_percent=progress[0],experiment_percent=progress[1],
        neural_forwards=0,official_AP=0,child_ledger=ledger(LOG/'children.jsonl')))
    status('completed_stopped',report=rel(REPORT),current_stage=last,scientific_passed=passed)
    if passed:
        state=read(STATE);state.update(all_tasks_percent=47,experiment_percent=58.75,P6_scientific_passed=True);atomic(STATE,state)
    supervise('v48_completed_stopped',current_runner=None,remaining_research_updates_cap=0,waiting_for_user_next_part_confirmation=True,
        summary_report=rel(REPORT),final_cold=rel(LOG/'cold.json'),
        final_readonly=rel(LOG/'nested_review.json') if last == 'nested_gate' else rel(LOG/'warp_review.json') if last == 'warp_calibration' else rel(LOG/(last+'_review.json')),
        final_delivery_index=rel(OUT/'closeout.json'),all_tasks_percent_planning_estimate=progress[0],experiment_percent_planning_estimate=progress[1])
    handoff(f'P6 v48已完成固定nested结构修复分支及完整复验/独立证书/冷重入并停止。最后阶段{last}；nested准入={proof["nested_supported"]}，科学采用={passed}，未运行关口保持null。全部任务总进度{progress[0]}%、实验总进度{progress[1]}%。报告{rel(REPORT)}。等待用户检查，不自动v49/P7/P10或周期任务。','complete')


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
            env = os.environ.copy(); env['P6_V48_CHILD_TOKEN'] = run_token
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
            supervise('v48_running')
            for stage in ('nested_fit', 'nested_review'): run(stage)
            if read(OUT/'nested_gate.json')['supported']:
                for stage in ('fit', 'fit_review', 'warp_calibration', 'warp_review'): run(stage)
                if read(OUT/'warp_calibration.json')['passed']:
                    for stage in ('calibration', 'calibration_review'): run(stage)
                    if read(OUT/'calibration.json')['passed']:
                        for stage in ('evaluation', 'evaluation_review'): run(stage)
            run('cold'); guard(); finish(c, guard); guard()
        except BaseException as e:
            append(LOG/'controller.jsonl', dict(at=now(), event='failure_preserved', error=repr(e), traceback=traceback.format_exc()))
            status('failed_stopped', error=repr(e), unknown=ledger()['unknown'], unknown_children=ledger(LOG/'children.jsonl')['unknown'])
            supervise('v48_failed_stopped', current_runner=None, waiting_for_user_next_part_confirmation=True)
            handoff('P6 v48出现工程或资源异常，失败日志/模型/未知返回保留，先核实际进程与锁再恢复，不自动重复。全部任务总进度45%、实验总进度56.25%；科学通过未成立，不进P7/P10。', 'failure')
            raise

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'verify', 'deliver', 'worker'])
    parser.add_argument('--stage'); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'verify': print(json.dumps(dict(passed=True, protected_files=len(verify()['protected']))))
    elif args.mode == 'deliver': deliver()
    else:
        verify(); child_guard(); OUT.mkdir(parents=True, exist_ok=True)
        if args.stage == 'nested_fit': nested_fit()
        elif args.stage == 'nested_review': nested_review()
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
