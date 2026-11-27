"""One frozen development-only P6 repair. No neural inference, AP, or P10 reads.

CLI: proposal -> root writes freeze.json -> deliver -> cold (new process).
All outputs are confined to the new Restore/v37/uncertainty directories.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import inspect
import json
import time
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scipy.stats import t as student_t
from src import uncertainty_restore_v37 as core
from src import p6_image_geometry_v24 as old_geometry
from src.geometry_propagation_v19 import pinhole_jacobian
from scripts.pipeline import project_lock

BASE = ROOT / 'results/Restore/v37/uncertainty'
LOG = ROOT / 'logs/Restore/v37/uncertainty'
OLD = ROOT / 'results/P6/error_mixture_v25'
CONFIG = ROOT / 'configs/p6_error_mixture_v25.json'
CONDITIONS = ('clean', 'noise5', 'missing15', 'shift3', 'joint')
NOISY = ('noise5', 'shift3', 'joint')
LIMITS = dict(seconds=1200, rss_bytes=2 * 1024**3, bytes=256 * 1024**2,
              corruption_replay=400, observable_features=400, error_model_fit=1,
              scale_calibration=1, distribution_prediction=80, independent_recompute=80)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def runtime_identity():
    import scipy
    import PIL
    return dict(python=sys.version.split()[0], executable=str(Path(sys.executable).resolve()),
                numpy=np.__version__, scipy=scipy.__version__, Pillow=PIL.__version__,
                cpu_threads={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def peak_rss_bytes():
    if os.name != 'nt':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in ('peak', 'working', 'a', 'b', 'c', 'd', 'e', 'f')]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    core.require(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(value), value.cb), 'Cannot verify process memory')
    return int(value.peak)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def artifact(path, arrays, metadata):
    core.require(not path.exists() and not path.with_suffix('.json').exists(), 'Uncommitted output exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    once(path.with_suffix('.json'), dict(metadata, sha256=sha(path)))


def cached(path):
    metadata = read(path.with_suffix('.json'))
    core.require(sha(path) == metadata.get('sha256', metadata.get('cache_sha256')), 'Cache identity differs: ' + str(path))
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k].copy() for k in source.files}, metadata


def stem(record, condition):
    return f"{condition}_scene_{record['scene']:04d}_{record['frame']}"


def selected_records():
    c = read(CONFIG)
    scenes = sorted(c['fit_scenes'])
    fit_scenes, calibration_scenes = scenes[:12], scenes[12:]
    core.require(len(scenes) == 16 and len(c['holdout_scenes']) == 4, 'Frozen v25 data scope differs')
    records = dict(fit=[r for r in c['fit_records'] if r['scene'] in fit_scenes],
                   calibration=[r for r in c['fit_records'] if r['scene'] in calibration_scenes],
                   development=c['holdout_records'])
    core.require([len(records[k]) for k in records] == [48, 16, 16], 'Expected 12/4/4 scenes and four frames')
    return c, records


def build_proposal():
    c, records = selected_records()
    inputs = {}
    paths = [CONFIG, ROOT / 'src/uncertainty_restore_v37.py', Path(__file__),
             ROOT / 'src/p6_image_geometry_v24.py', ROOT / 'src/geometry_propagation_v19.py',
             ROOT / 'scripts/p6_common_v24.py', ROOT / 'scripts/pipeline.py']
    for mode, rows in records.items():
        old_split = 'holdout' if mode == 'development' else 'fit'
        for r in rows:
            for kind in ('rgb', 'depth', 'meta'):
                p = ROOT / c['data_root'] / r['files'][kind]
                inputs[relative(p)] = dict(sha256=r['sha256'][kind], bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for d in CONDITIONS:
                for folder in ('samples', 'observations'):
                    p = OLD / folder / old_split / (stem(r, d) + '.npz')
                    meta_path = p.with_suffix('.json')
                    meta = read(meta_path)
                    inputs[relative(p)] = dict(sha256=meta['cache_sha256'], bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                    paths.append(meta_path)
    for p in paths:
        inputs[relative(p)] = dict(sha256=sha(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    return dict(version='v37', component='uncertainty', created=now(), runtime=runtime_identity(),
        scientific_scope='exposed development repair of conditional depth residual; not independent confirmation, full uv covariance, grasp improvement, or completed P6',
        model=dict(family='Student-t', df=3, location_and_log_scale='linear observable features with one fixed L-BFGS fit', max_iterations=60, max_function_evaluations=150, ridge=1e-4, features=list(core.FEATURES),
                   scale_bounds_mm=[.1, 200.], standardized_features_clipped_to=[-8, 8], calibration='one scene/condition/frame-equal empirical multiplier from noisy calibration conditions; no distribution-free or conditional coverage guarantee', geometry_mean_correction=False),
        records=records, scenes={k: sorted({r['scene'] for r in v}) for k, v in records.items()},
        cases=dict(fit=240, calibration=80, development=80), max_points_per_fit_or_calibration_case=1024,
        fit_point_cap=245760, calibration_point_cap=81920, development='reuse exact original 4096 main plus up to1024 boundary union pixels',
        neural_inference=0, neural_optimization=0, grasp_forward=0, official_ap=0, monte_carlo=0,
        instance_boundary_input=False, instance_boundary_reason='old fit observations contain zero placeholders, not predicted instances; only frozen holdout masks define evaluation strata',
        gates=dict(main95_each_noisy=[.90, .98], boundary95_each_noisy_min=.85, worst_scene_main95_min=.80, average_degraded_nll_gain_min=.05,
                   main_mean_width95_each_noisy_max_m=.04, boundary_mean_width95_each_noisy_max_m=.08,
                   sharpness_scope='new engineering upper limits, approximately2x/4x fixed5mm normal width; not physical collision guarantee', observed_xyz_identical=True),
        limits=LIMITS, protected=inputs, old_reference='original depth is synthetic perturbation reference, not independent real sensor ground truth',
        exposure='all20 scenes appeared in upstream training; prior holdout scores exposed; all conclusions are development only',
        stop_rule='one fit and one scale calibration; no automatic model changes or expansion after failure')


def verify_freeze():
    proposal = read(LOG / 'proposal.json')
    freeze = read(LOG / 'freeze.json')
    core.require(freeze.get('authorized') is True, 'Root freeze authorization missing')
    core.require(freeze.get('proposal_sha256') == sha(LOG / 'proposal.json'), 'Proposal changed after root freeze')
    core.require(proposal['limits'] == LIMITS, 'Resource limits changed')
    core.require(proposal['runtime'] == runtime_identity(), 'Frozen numerical runtime changed')
    for name, record in proposal['protected'].items():
        p = ROOT / name
        st = p.stat()
        core.require(st.st_size == record['bytes'] and st.st_mtime_ns == record['mtime_ns'] and sha(p) == record['sha256'], 'Protected input changed: ' + name)
    return proposal


def ledger():
    path = LOG / 'calls.jsonl'
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    result = {}
    for kind in LIMITS:
        if kind in ('seconds', 'bytes', 'rss_bytes'):
            continue
        intents = [r['id'] for r in rows if r['kind'] == kind and r['event'] == 'intent']
        returned = [r['id'] for r in rows if r['kind'] == kind and r['event'] == 'returned']
        core.require(len(intents) == len(set(intents)) and len(returned) == len(set(returned)) and set(returned) <= set(intents), 'Invalid call ledger')
        result[kind] = dict(intents=len(intents), known_returns=len(returned), unknown=len(set(intents) - set(returned)))
    return result


def event(row):
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / 'calls.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(at=now(), **row), ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.previous = sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'))
        self.path = LOG / ('usage_' + uuid.uuid4().hex + '.json')
        self.guard()

    def save(self, outcome):
        # Only this invocation's mutable accounting file; no research evidence overwritten.
        value = dict(at=now(), seconds=time.monotonic() - self.started, outcome=outcome)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value), encoding='utf-8')
        os.replace(temporary, self.path)

    def guard(self):
        self.save('running')
        core.require(self.previous + time.monotonic() - self.started <= LIMITS['seconds'], 'Cumulative CPU-stage wall budget exceeded')
        size = sum(p.stat().st_size for folder in (BASE, LOG) for p in folder.rglob('*') if p.is_file())
        core.require(size <= LIMITS['bytes'], 'Artifact size budget exceeded')
        core.require(peak_rss_bytes() <= LIMITS['rss_bytes'], 'Peak RSS exceeded')

    def invoke(self, kind, fn, **metadata):
        self.guard()
        current = ledger()[kind]
        core.require(current['intents'] < LIMITS[kind], 'Call budget exhausted: ' + kind)
        core.require(current['unknown'] == 0, 'Unreturned call requires explicit recovery: ' + kind)
        identity = uuid.uuid4().hex
        event(dict(id=identity, kind=kind, event='intent', **metadata))
        answer = fn()
        event(dict(id=identity, kind=kind, event='returned', **metadata))
        self.guard()
        return answer


def prepare(proposal, budget):
    endpoint = BASE / 'features_complete.json'
    if endpoint.exists():
        return
    c = read(CONFIG)
    count = 0
    for mode, records in proposal['records'].items():
        split = 'holdout' if mode == 'development' else 'fit'
        for r in records:
            with Image.open(ROOT / c['data_root'] / r['files']['rgb']) as im:
                rgb = np.asarray(im.convert('RGB').resize((640, 480), Image.Resampling.BILINEAR), dtype=np.float64) / 255.
            for condition in CONDITIONS:
                name = stem(r, condition)
                out = BASE / 'features' / mode / (name + '.npz')
                if out.with_suffix('.json').exists():
                    cached(out)
                    count += 1
                    continue
                original, meta = cached(OLD / 'samples' / split / (name + '.npz'))
                maps, map_meta = cached(OLD / 'observations' / split / (name + '.npz'))
                s = budget.invoke('corruption_replay', lambda: old_geometry.source(c, r, condition), case=name, split=mode)
                idx = original['flat_index']
                core.require(np.array_equal(s['observed'].ravel()[idx], original['observed']) and np.array_equal(s['reference'].ravel()[idx], original['reference']), 'Frozen corruption replay differs')
                core.require(np.array_equal(maps['sigma'].ravel()[idx], original['sigma_old']), 'Frozen quality identity differs')
                if mode != 'development':
                    core.require(np.all(original['main']) and not maps['instances'].any(), 'Old fit cache semantics changed')
                    keep = np.arange(min(1024, len(idx)), dtype=np.int64) * len(idx) // min(1024, len(idx))
                    original = {k: v[keep] for k, v in original.items()}
                    idx = original['flat_index']
                x = budget.invoke('observable_features', lambda: core.observable_features(s['observed'], s['observed_valid'], maps['sigma'], rgb, idx), case=name, split=mode)
                arrays = {k: original[k] for k in ('flat_index', 'main', 'boundary', 'rays', 'observed', 'reference', 'sigma_old')}
                artifact(out, dict(arrays, features=x), dict(scene=r['scene'], frame=r['frame'], condition=condition, split=mode,
                    old_sample_sha256=meta['cache_sha256'], old_observation_sha256=map_meta['cache_sha256'], feature_names=list(core.FEATURES), reference_used_as_feature=False))
                count += 1
                if count % 40 == 0:
                    print(f'features {count}/400', flush=True)
    core.require(count == 400, 'Incomplete feature set')
    once(endpoint, dict(cases=count, points_cap_per_fit_or_calibration_case=1024, neural_forward=0, reference_used_as_feature=False))


def stack(mode, noisy_only=False):
    features, residuals, weights, binding = [], [], [], {}
    for p in sorted((BASE / 'features' / mode).glob('*.npz')):
        a, meta = cached(p)
        if noisy_only and meta['condition'] not in NOISY:
            continue
        features.append(a['features'])
        residuals.append(a['observed'] - a['reference'])
        weights.append(np.full(len(a['features']), 1. / len(a['features'])))
        binding[relative(p)] = meta['sha256']
    core.require(len(features) == {'fit': 240, 'calibration': 48 if noisy_only else 80}[mode], 'Incomplete fitting partition')
    return np.concatenate(features), np.concatenate(residuals), np.concatenate(weights), binding


def model_fit(budget):
    endpoint = BASE / 'model_fit.json'
    if endpoint.exists():
        return
    core.require((BASE / 'features_complete.json').exists(), 'Features missing')
    x, y, w, binding = stack('fit')
    model = budget.invoke('error_model_fit', lambda: core.fit(x, y, w, budget.guard), fit_cases=240, maximum_points=245760)
    once(endpoint, dict(model, source_sha256=binding))


def model_calibration(budget):
    endpoint = BASE / 'model_calibrated.json'
    if endpoint.exists():
        return
    fitted = read(BASE / 'model_fit.json')
    x, y, w, binding = stack('calibration', noisy_only=True)
    model = budget.invoke('scale_calibration', lambda: core.calibrate(fitted, x, y, w), calibration_noisy_cases=48, fitted_parameter='one_multiplier')
    once(endpoint, dict(model, calibration_sources=binding, fit_model_sha256=sha(BASE / 'model_fit.json')))


def aggregate(rows):
    conditions = {}
    for condition in CONDITIONS:
        scenes = []
        for scene in sorted({r['scene'] for r in rows}):
            rr = [r for r in rows if r['scene'] == scene and r['condition'] == condition]
            core.require(len(rr) == 4, 'Incomplete scene/condition block')
            groups = {}
            for group in ('main', 'boundary'):
                mm = [r['metrics'][group] for r in rr]
                n = sum(m['n'] for m in mm)
                groups[group] = dict(n=n,
                    coverage=[sum(m['covered'][j] for m in mm) / n for j in range(3)],
                    nll=sum(m['nll_sum'] for m in mm) / n,
                    fixed5mm_nll=sum(m['fixed5mm_nll_sum'] for m in mm) / n,
                    mean_width95_m=sum(m['width95_sum'] for m in mm) / n,
                    raw_rmse3d_m=float(np.sqrt(sum(m['squared_error3d_sum'] for m in mm) / n)))
            scenes.append(dict(scene=scene, groups=groups))
        macro = {g: {k: np.mean([s['groups'][g][k] for s in scenes], axis=0).tolist()
                     for k in ('coverage', 'nll', 'fixed5mm_nll', 'mean_width95_m', 'raw_rmse3d_m')}
                 for g in ('main', 'boundary')}
        conditions[condition] = dict(scenes=scenes, macro=macro)
    worst = min(s['groups']['main']['coverage'][2] for d in NOISY for s in conditions[d]['scenes'])
    gain = float(np.mean([conditions[d]['macro']['main']['fixed5mm_nll'] - conditions[d]['macro']['main']['nll'] for d in CONDITIONS if d != 'clean']))
    gates = dict(noisy_95_coverage=all(.90 <= conditions[d]['macro']['main']['coverage'][2] <= .98 for d in NOISY),
                 boundary_95_coverage=all(conditions[d]['macro']['boundary']['coverage'][2] >= .85 for d in NOISY),
                 worst_scene_95_coverage=worst >= .80, nll_improvement=gain >= .05,
                 main_interval_sharpness=all(conditions[d]['macro']['main']['mean_width95_m'] <= .04 for d in NOISY),
                 boundary_interval_sharpness=all(conditions[d]['macro']['boundary']['mean_width95_m'] <= .08 for d in NOISY))
    return dict(conditions=conditions, gates=gates, all_scientific_gates_passed=all(gates.values()),
                worst_scene_coverage95=worst, mean_degraded_nll_gain=gain, cases=len(rows),
                aggregation='per-scene pooled exact original points, then scene equal; matches v25',
                scientific_scope='exposed development only, conditional depth rank1')


def evaluate(budget):
    endpoint = BASE / 'development_summary.json'
    if endpoint.exists():
        return
    model = read(BASE / 'model_calibrated.json')
    rows = []
    for p in sorted((BASE / 'features' / 'development').glob('*.npz')):
        out = BASE / 'predictions' / p.name
        if out.with_suffix('.json').exists():
            _, row = cached(out)
            rows.append(row)
            continue
        a, meta = cached(p)
        prediction = budget.invoke('distribution_prediction', lambda: core.predict(model, a['features']), case=p.stem)
        geometry = core.propagate_depth_only(prediction, a['observed'], a['rays'])
        core.require(np.array_equal(geometry['observed_xyz'], a['observed'][:, None] * a['rays']), 'Geometry changed')
        metrics = core.case_metrics(a, prediction)
        artifact(out, dict(prediction, **geometry), dict(scene=meta['scene'], frame=meta['frame'], condition=meta['condition'],
            metrics=metrics, features_sha256=meta['sha256'], model_sha256=sha(BASE / 'model_calibrated.json'), observed_geometry_unchanged=True))
        rows.append(read(out.with_suffix('.json')))
    core.require(len(rows) == 80, 'Development evaluation incomplete')
    once(endpoint, aggregate(rows))


def independent_case(a, p, model):
    # Reconstruct parameters and all reported statistics independently of the
    # core's prediction/metrics/propagation implementations.
    raw = (a['features'] - np.array(model['center'])) / np.array(model['normalization_scale'])
    x = np.concatenate((np.ones((len(raw), 1)), np.maximum(-8., np.minimum(8., raw))), axis=1)
    mu = np.einsum('ij,j->i', x, np.array(model['location_coefficients'])) / 1000.
    scale = np.exp(np.maximum(np.log(.1), np.minimum(np.log(200.), np.einsum('ij,j->i', x, np.array(model['log_scale_coefficients']))))) / 1000. * model['calibration_multiplier']
    for key, expected in [('location', mu), ('scale', scale), ('variance', 3 * scale * scale)]:
        np.testing.assert_allclose(p[key], expected, rtol=1e-10, atol=1e-12)
    err = a['observed'] - a['reference']
    pit = student_t.cdf(err, 3, loc=mu, scale=scale)
    nll = -student_t.logpdf(err, 3, loc=mu, scale=scale)
    width = student_t.ppf(.975, 3, loc=mu, scale=scale) - student_t.ppf(.025, 3, loc=mu, scale=scale)
    np.testing.assert_array_equal(p['observed_xyz'], a['observed'][:, None] * a['rays'])
    covariance = a['rays'][:, :, None] * a['rays'][:, None, :] * (3 * scale * scale)[:, None, None]
    np.testing.assert_allclose(p['covariance'], covariance, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(p['predicted_error_mean_xyz'], mu[:, None] * a['rays'], rtol=1e-10, atol=1e-12)
    expected_second = a['rays'][:, :, None] * a['rays'][:, None, :] * (3 * scale * scale + mu * mu)[:, None, None]
    np.testing.assert_allclose(p['error_second_moment_about_observation'], expected_second, rtol=1e-10, atol=1e-14)
    # The frozen R19 Jacobian's third column is the pinhole ray. Numeric
    # comparison verifies algebra only, not Gaussian probability assumptions.
    for i in np.linspace(0, len(mu) - 1, 5, dtype=int):
        ray = a['rays'][i]
        j = pinhole_jacobian(np.array([ray[0], ray[1], a['observed'][i]]), np.eye(3))
        expected = j @ np.diag([0., 0., 3 * scale[i] * scale[i]]) @ j.T
        np.testing.assert_allclose(p['covariance'][i], expected, rtol=1e-10, atol=1e-14)
    out = {}
    for group in ('main', 'boundary'):
        m = a[group]
        out[group] = dict(n=int(m.sum()),
            covered=[int(np.count_nonzero((pit[m] >= (1-q)/2) & (pit[m] <= (1+q)/2))) for q in core.LEVELS],
            nll_sum=float(nll[m].sum()),
            fixed5mm_nll_sum=float((.5 * (err[m] / .005) ** 2 + np.log(.005 * np.sqrt(2 * np.pi))).sum()),
            width95_sum=float(width[m].sum()),
            squared_error3d_sum=float(np.sum((err[m, None] * a['rays'][m]) ** 2)), observed_geometry_changed=False)
    return out


def compare_tree(expected, actual):
    if isinstance(expected, dict):
        core.require(set(expected) == set(actual), 'Result keys differ')
        for k in expected:
            compare_tree(expected[k], actual[k])
    elif isinstance(expected, (list, tuple)):
        core.require(len(expected) == len(actual), 'Result lengths differ')
        for x, y in zip(expected, actual):
            compare_tree(x, y)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        np.testing.assert_allclose(expected, actual, rtol=1e-9, atol=1e-10)
    else:
        core.require(expected == actual, 'Result mismatch')


def review(budget):
    endpoint = BASE / 'review.json'
    if endpoint.exists():
        return
    model = read(BASE / 'model_calibrated.json')
    rows = []
    for path in sorted((BASE / 'predictions').glob('*.npz')):
        p, meta = cached(path)
        a, source = cached(BASE / 'features' / 'development' / path.name)
        core.require(meta['features_sha256'] == source['sha256'] and meta['model_sha256'] == sha(BASE / 'model_calibrated.json'), 'Prediction bindings differ')
        metrics = budget.invoke('independent_recompute', lambda: independent_case(a, p, model), case=path.stem)
        compare_tree(metrics, meta['metrics'])
        rows.append(dict(scene=meta['scene'], frame=meta['frame'], condition=meta['condition'], metrics=metrics))
    compare_tree(aggregate(rows), read(BASE / 'development_summary.json'))
    core.require(len(rows) == 80 and all(v['unknown'] == 0 for v in ledger().values()), 'Incomplete independent review')
    once(endpoint, dict(passed=True, cases=80, original_geometry_identical=True,
        scipy_student_t_distribution_recomputed=True, covariance_and_bias_second_moment_verified=True,
        frozen_R19_depth_column_reused=True, gaussian_ellipsoid_coverage_claimed=False,
        full_pixel_covariance_claimed=False, independent_confirmation=False))


def deliver():
    proposal = verify_freeze()
    if (BASE / 'result.json').exists():
        print(json.dumps(read(BASE / 'result.json'), ensure_ascii=False))
        return
    budget = Budget()
    outcome = 'failed'
    try:
        prepare(proposal, budget)
        model_fit(budget)
        model_calibration(budget)
        evaluate(budget)
        review(budget)
        verify_freeze()
        summary = read(BASE / 'development_summary.json')
        model = read(BASE / 'model_calibrated.json')
        bindings = {relative(p): sha(p) for folder in (BASE / 'features', BASE / 'predictions') for p in folder.rglob('*') if p.is_file()}
        for name in ('features_complete.json', 'model_fit.json', 'model_calibrated.json', 'development_summary.json', 'review.json'):
            bindings[relative(BASE / name)] = sha(BASE / name)
        once(BASE / 'result.json', dict(at=now(), engineering_passed=True,
            scientific_usable=summary['all_scientific_gates_passed'],
            ready_for_bounded_expanded_validation=summary['all_scientific_gates_passed'],
            whole_P6_passed=False, full_method_passed=False, independent_confirmation=False,
            gates=summary['gates'], optimizer_converged=model['optimizer_converged'],
            optimizer_message=model['optimizer_message'], calibration_multiplier=model['calibration_multiplier'],
            condition_metrics={d: summary['conditions'][d]['macro'] for d in CONDITIONS},
            mean_degraded_nll_gain=summary['mean_degraded_nll_gain'], worst_scene_coverage95=summary['worst_scene_coverage95'],
            observed_geometry_unchanged=True, scope=proposal['scientific_scope'], calls=ledger(),
            artifact_sha256=bindings, required_remaining=['independent confirmation', 'full uv/depth modeling if retained by original plan', 'instance-aware geometry integration', 'grasp benefit/collision validation']))
        outcome = 'passed'
        print(json.dumps({k: v for k, v in read(BASE / 'result.json').items() if k != 'artifact_sha256'}, ensure_ascii=False), flush=True)
    except Exception as exc:
        event(dict(kind='stage_failure', event='failure', error=repr(exc)))
        raise
    finally:
        budget.save(outcome)


def cold():
    verify_freeze()
    before = ledger()
    result = read(BASE / 'result.json')
    for name, expected in result['artifact_sha256'].items():
        core.require(sha(ROOT / name) == expected, 'Cold artifact identity differs: ' + name)
    summary, model, reviewed = [read(BASE / name) for name in ('development_summary.json', 'model_calibrated.json', 'review.json')]
    compare_tree(result['gates'], summary['gates'])
    compare_tree(result['condition_metrics'], {d: summary['conditions'][d]['macro'] for d in CONDITIONS})
    compare_tree(result['mean_degraded_nll_gain'], summary['mean_degraded_nll_gain'])
    compare_tree(result['worst_scene_coverage95'], summary['worst_scene_coverage95'])
    compare_tree(result['calibration_multiplier'], model['calibration_multiplier'])
    core.require(result['engineering_passed'] == reviewed['passed'] and result['scientific_usable'] == summary['all_scientific_gates_passed']
                 and result['ready_for_bounded_expanded_validation'] == summary['all_scientific_gates_passed'], 'Cold result semantics differ')
    core.require(not result['whole_P6_passed'] and not result['full_method_passed'] and not result['independent_confirmation'], 'Cold scientific scope overstated')
    core.require(before == ledger() and result['calls'] == before, 'Cold reentry changed research-call ledger')
    evidence = dict(at=now(), passed=True, new_research_calls=0, runtime=runtime_identity(),
                    result_sha256=sha(BASE / 'result.json'), engineering_passed=result['engineering_passed'], scientific_usable=result['scientific_usable'])
    if (LOG / 'cold.json').exists():
        previous = read(LOG / 'cold.json')
        core.require(previous['passed'] and previous['new_research_calls'] == 0 and previous['result_sha256'] == evidence['result_sha256'], 'Previous cold marker differs')
    else:
        once(LOG / 'cold.json', evidence)
    print(json.dumps(dict(cold_passed=True, new_research_calls=0, scientific_usable=result['scientific_usable'])), flush=True)


def self_test():
    # Synthetic arithmetic only: no research data, model.fit, or saved model.
    rng = np.random.default_rng(37)
    x = np.column_stack((np.ones(100), rng.normal(size=(100, len(core.FEATURES)))))
    y = rng.normal(scale=5, size=100)
    w = np.full(100, .01)
    theta = rng.normal(scale=.05, size=x.shape[1] * 2)
    theta[x.shape[1]] = np.log(5.)
    _, gradient = core.objective(theta, x, y, w)
    numerical = []
    for j in range(len(theta)):
        step = np.zeros(len(theta)); step[j] = 1e-5
        numerical.append((core.objective(theta + step, x, y, w)[0] - core.objective(theta - step, x, y, w)[0]) / 2e-5)
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)
    model = dict(features=list(core.FEATURES), degrees_of_freedom=3., center=[0.] * len(core.FEATURES), normalization_scale=[1.] * len(core.FEATURES),
                 location_coefficients=[2.] + [0.] * len(core.FEATURES), log_scale_coefficients=[np.log(5.)] + [0.] * len(core.FEATURES), calibration_multiplier=1.5)
    pred = core.predict(model, np.zeros((7, len(core.FEATURES))))
    err = np.linspace(-.02, .02, 7)
    dist = core.distribution_statistics(pred, err)
    np.testing.assert_allclose(dist['pit'], student_t.cdf(err, 3, loc=.002, scale=.0075))
    np.testing.assert_allclose(dist['logpdf'], student_t.logpdf(err, 3, loc=.002, scale=.0075))
    np.testing.assert_allclose(pred['variance'], 3 * .0075 ** 2)
    rays = rng.normal(size=(7, 3)); rays[:, 2] = 1.
    geom = core.propagate_depth_only(pred, np.ones(7), rays)
    np.testing.assert_array_equal(geom['observed_xyz'], rays)
    for cov, ray in zip(geom['covariance'], rays):
        core.require(np.linalg.matrix_rank(cov, tol=1e-12) == 1, 'Rank1 covariance failed')
        j = pinhole_jacobian(np.array([ray[0], ray[1], 1.]), np.eye(3))
        np.testing.assert_allclose(cov, j @ np.diag([0., 0., 3*.0075**2]) @ j.T)
    sample = dict(features=np.zeros((7, len(core.FEATURES))), observed=np.ones(7), reference=np.ones(7) - err,
                  rays=rays, main=np.ones(7, dtype=bool), boundary=np.array([True, False, True, False, True, False, True]))
    compare_tree(independent_case(sample, dict(pred, **geom), model), core.case_metrics(sample, pred))
    plain_model = dict(model, calibration_multiplier=1.)
    calibration_x = np.zeros((100, len(core.FEATURES)))
    calibration_scores = np.linspace(.1, 6., 100)
    calibrated = core.calibrate(plain_model, calibration_x, .002 + .005 * calibration_scores, np.ones(100))
    expected_multiplier = core.weighted_quantile(calibration_scores, np.ones(100), .95) / student_t.ppf(.975, 3)
    np.testing.assert_allclose(calibrated['calibration_multiplier'], expected_multiplier)
    calibrated_prediction = core.predict(calibrated, calibration_x)
    np.testing.assert_allclose(calibrated_prediction['variance'], 3 * (.005 * expected_multiplier) ** 2)
    z = np.ones((32, 32)); z[:, 16:] = 1.1
    rgb = np.repeat((z - 1.)[:, :, None] * 5., 3, axis=2)
    idx = np.arange(z.size)
    feat = core.observable_features(z, np.ones_like(z, bool), np.full_like(z, .005), rgb, idx)
    core.require(feat.shape == (1024, len(core.FEATURES)), 'Feature layout failed')
    forbidden = {'reference', 'condition', 'scene', 'label', 'instances'}
    core.require(not forbidden.intersection(inspect.signature(core.observable_features).parameters), 'Prohibited feature inputs')
    np.testing.assert_allclose(core.weighted_quantile([1., 2., 9.], [1., 1., 1.], .5), 2.)
    print(json.dumps(dict(passed=True, checks=['analytic_gradient', 'student_t_cdf_pdf_variance', 'same_calibrated_scale', 'zero_geometry_change', 'rank1_R19_covariance', 'independent_review_numerics', 'empirical_calibration', 'feature_signature', 'synthetic_features', 'weighted_quantile'], research_model_calls=0, runtime=runtime_identity(), peak_rss_bytes=peak_rss_bytes(), torch_imported='torch' in sys.modules)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['proposal', 'self-test', 'deliver', 'cold'])
    action = parser.parse_args().action
    if action == 'self-test':
        self_test()
    elif action == 'proposal':
        once(LOG / 'proposal.json', build_proposal())
        print(str(LOG / 'proposal.json'))
    else:
        with project_lock(ROOT):
            deliver() if action == 'deliver' else cold()
