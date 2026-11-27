"""Independent saved-array audit of limited synthetic pinhole propagation.

No random generator, model, real observation loader or production worker is
called. Numerical reference functions below do not import the production core.
Storage receipts and environment bindings are shared infrastructure; NumPy and
SciPy quantile routines are common dependencies, not independent implementations.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import beta, chi2, t
from scripts import geometry_storage_v19 as store

CONFIG = 'configs/geometry_propagation_v19.json'
SOURCE = 'scripts/accept_geometry_propagation_v19.py'
MODES = ('full', 'diagonal')
FORMS = ('linear', 'exact')
LEVELS = (.5, .8, .95)
BATCHES, BATCH_SIZE, SAMPLE_COUNT = 16, 8192, 131072
FIGURES = ('second_order_terms.png', 'moment_consistency.png', 'coverage_diagnostics.png')
require = store.require


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    return value


def finite(value, shape=None):
    require(not np.iscomplexobj(value), 'Complex numerical evidence')
    result = np.asarray(value, dtype=np.float64)
    require((shape is None or result.shape == shape) and np.isfinite(result).all(), 'Invalid numerical evidence')
    return result


def close(actual, expected, path='value', atol=2e-14, rtol=2e-10):
    """Compare a reference subset; integer identities and decisions are exact."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), 'Missing reference fields: '+path)
        for key, value in expected.items(): close(actual[key], value, path+'.'+key, atol, rtol)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        require(isinstance(actual, (list, tuple, np.ndarray)) and len(actual) == len(expected), 'Shape differs: '+path)
        for index, value in enumerate(expected): close(actual[index], value, path+'[%d]'%index, atol, rtol)
    elif expected is None or isinstance(expected, (str, bool, int, np.integer, np.bool_)):
        require(type(actual) is type(plain(expected)) and actual == plain(expected), 'Identity or decision differs: '+path)
    else:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(float(expected))
                and abs(actual-float(expected)) <= atol+rtol*abs(float(expected)), 'Numerical recount differs: '+path)


def independent_seed(case_id, batch_index):
    require(re.fullmatch(r'case_\d{3}', case_id) and 0 <= int(case_id[5:]) < 72
            and type(batch_index) is int and 0 <= batch_index < 16, 'Invalid saved-array identity')
    text = 'R19|20261027|%s|batch_%02d'%(case_id, batch_index)
    return int(hashlib.sha256(text.encode('ascii')).hexdigest()[:8], 16)


def declared_cases():
    cameras = [('K0', [480, 640], [600., 600., 319.5, 239.5]),
               ('K1', [720, 1280], [900., 500., 630., 350.])]
    correlations = [[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        [[1., .3, .4], [.3, 1., -.2], [.4, -.2, 1.]],
        [[1., -.25, -.35], [-.25, 1., .3], [-.35, .3, 1.]]]
    rows = []
    for camera, shape, (fx, fy, cx, cy) in cameras:
        for position, pixel in [('principal', [cx, cy]), ('top_left', [0., 0.]),
                                ('bottom_right', [float(shape[1]-1), float(shape[0]-1)])]:
            for depth in (.5, 1.5):
                for level, scales in enumerate(([.25, .25, .002], [2., 1., .02])):
                    for index, correlation in enumerate(correlations):
                        # Preserve the declared multiplication order for byte-level protocol identity.
                        sigma = np.diag(scales) @ np.asarray(correlation) @ np.diag(scales)
                        rows.append({'case_id': 'case_%03d'%len(rows), 'camera': camera, 'image_size_hw': shape.copy(),
                            'intrinsic': [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], 'position': position,
                            'nominal': pixel+[depth], 'noise_level': 'S%d'%level, 'standard_deviation': list(scales),
                            'correlation_id': 'R%d'%index, 'correlation': correlation,
                            'input_covariance': sigma.tolist(), 'sample_count': SAMPLE_COUNT,
                            'batch_count': BATCHES, 'batch_size': BATCH_SIZE})
    return rows


def project(points, intrinsic):
    points = finite(points); k = finite(intrinsic, (3, 3))
    require(points.ndim in (1, 2) and points.shape[-1] == 3, 'Wrong coordinate shape')
    u, v, z = np.moveaxis(points, -1, 0)
    return np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=-1)


def reference(case, mode):
    require(mode in MODES, 'Unknown covariance mode')
    k = finite(case['intrinsic'], (3, 3)); u, v, z = finite(case['nominal'], (3,))
    s = finite(case['input_covariance'], (3, 3)).copy()
    if mode == 'diagonal': s = np.diag(np.diag(s))
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    require(fx > 0 and fy > 0 and z > 0 and k[0, 1] == k[1, 0] == 0
            and np.array_equal(k[2], [0., 0., 1.]) and np.array_equal(s, s.T), 'Invalid declared pinhole covariance')
    a, b, c, d = z/fx, (u-cx)/fx, z/fy, (v-cy)/fy
    j = np.array([[a, 0., b], [0., c, d], [0., 0., 1.]])
    first = np.array([[a*a*s[0, 0]+2*a*b*s[0, 2]+b*b*s[2, 2],
                      a*c*s[0, 1]+a*d*s[0, 2]+b*c*s[1, 2]+b*d*s[2, 2], a*s[0, 2]+b*s[2, 2]],
                     [0., c*c*s[1, 1]+2*c*d*s[1, 2]+d*d*s[2, 2], c*s[1, 2]+d*s[2, 2]],
                     [0., 0., s[2, 2]]])
    first = first+np.triu(first, 1).T
    h = np.zeros((3, 3)); bias = np.array([s[0, 2]/fx, s[1, 2]/fy, 0.])
    h[0, 0] = (s[0, 0]*s[2, 2]+s[0, 2]*s[0, 2])/(fx*fx)
    h[1, 1] = (s[1, 1]*s[2, 2]+s[1, 2]*s[1, 2])/(fy*fy)
    h[0, 1] = h[1, 0] = (s[0, 1]*s[2, 2]+s[0, 2]*s[1, 2])/(fx*fy)
    location = np.array([b*z, d*z, z])
    return {'input_covariance': s, 'jacobian': j, 'nominal_output': location,
            'first_order_mean': location, 'first_order_covariance': first, 'mean_bias': bias,
            'second_order_covariance_correction': h, 'exact_mean': location+bias,
            'exact_covariance': first+h, 'second_moment_about_nominal': first+h+bias[:, None]*bias[None, :]}


def lower_factor(sigma):
    """Three-variable scalar Cholesky, separate from production LAPACK factorization."""
    s = finite(sigma, (3, 3)); factor = np.zeros((3, 3))
    for row in range(3):
        for column in range(row+1):
            value = s[row, column]-math.fsum(factor[row, k]*factor[column, k] for k in range(column))
            require(column != row or value > 0, 'Main covariance is not positive definite')
            factor[row, column] = math.sqrt(value) if column == row else value/factor[column, column]
    return factor


def batch_moments(points):
    values = finite(points); require(values.ndim == 2 and values.shape[1] == 3 and len(values) >= 2, 'Invalid batch')
    mean = np.sum(values, axis=0, dtype=np.float64)/len(values)
    centered = values-mean
    m2 = np.array([[np.sum(centered[:, i]*centered[:, j], dtype=np.float64) for j in range(3)] for i in range(3)])
    return {'count': len(values), 'mean': mean, 'covariance': m2/(len(values)-1), 'centered_sum_products': m2, 'ddof': 1}


def combined_moments(batches):
    require(len(batches) == 16 and len({b['count'] for b in batches}) == 1, 'Require sixteen equal batches')
    n = batches[0]['count']; means = np.asarray([b['mean'] for b in batches]); cov = np.asarray([b['covariance'] for b in batches])
    mean = np.sum(means, axis=0)/16; primary = np.sum(cov, axis=0)/16
    mean_se = np.sqrt(np.sum((means-mean)**2, axis=0)/(15*16))
    covariance_se = np.sqrt(np.sum((cov-primary)**2, axis=0)/(15*16))
    # Merge sufficient statistics with explicit between-batch outer products.
    total = np.zeros((3, 3))
    for batch in batches:
        shift = np.asarray(batch['mean'])-mean
        total += np.asarray(batch['centered_sum_products'])+n*np.outer(shift, shift)
    return {'batch_count': 16, 'batch_size': n, 'sample_count': n*16, 'contract_sample_count': n == 8192,
        'batch_means': means, 'batch_covariances': cov, 'primary_mean': mean, 'primary_covariance': primary,
        'mean_standard_error': mean_se, 'covariance_standard_error': covariance_se,
        'pooled_mean': mean, 'pooled_covariance': total/(n*16-1),
        'primary_covariance_definition': 'equal_mean_of_sixteen_within_batch_unbiased_covariances'}


def moment_decision(summary, mean, covariance):
    critical = float(t.isf(.01/(2*2592), 15)); result = {'critical_value': critical, 'family_alpha': .01,
        'family_comparisons': 2592, 'degrees_of_freedom': 15, 'family_scope': 'moments_only_not_combined_with_coverage',
        'approximate_batch_normality_required': True, 'exact_finite_sample_family_guarantee': False}
    for name, expected, floor in (('mean', np.asarray(mean), 1e-10), ('covariance', np.asarray(covariance), 1e-14)):
        observed = summary['primary_'+name]; se = summary[name+'_standard_error']; delta = observed-expected
        allowance = critical*se+floor; selected = np.ones(3, bool) if name == 'mean' else np.triu(np.ones((3, 3), bool))
        passed = np.abs(delta) <= allowance
        result[name] = {'observed': observed, 'reference': expected, 'difference': delta, 'standard_error': se,
            'absolute_floor': floor, 'allowance': allowance, 'checked_components': selected,
            'component_passed': passed, 'passed': bool(np.all(passed[selected]))}
    result['passed'] = result['mean']['passed'] and result['covariance']['passed']
    eligible = summary['batch_count'] == 16 and summary['batch_size'] == 8192 and summary['sample_count'] == 131072
    result['eligible_for_contract_gate'] = eligible
    result['contract_gate_passed'] = result['passed'] if eligible else None
    return result


def count_coverage(points, mean, covariance):
    values = finite(points); residual = values-np.asarray(mean); matrix = finite(covariance, (3, 3))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    threshold = max(1e-12*float(eigenvalues[-1]), 1e-18); active = eigenvalues > threshold
    rank = int(active.sum()); require(eigenvalues[0] >= -1e-12*max(eigenvalues[-1], 1e-12), 'Invalid output covariance')
    if rank == 3:
        # Main-case independent reference uses a linear solve, not the production spectral distance.
        solved = np.linalg.solve(matrix, residual.T).T
        distance = np.sum(residual*solved, axis=1); null = np.zeros(len(values))
    else:
        coordinates = residual @ eigenvectors
        distance = np.sum(coordinates[:, active]**2/eigenvalues[active], axis=1)
        null = np.sqrt(np.sum(coordinates[:, ~active]**2, axis=1))
    support = null <= 1e-10
    return {'counts': [int(np.count_nonzero(support & (distance <= chi2.ppf(p, rank)))) for p in LEVELS] if rank else [],
            'sample_count': len(values), 'rank': rank, 'off_support': int(np.count_nonzero(~support)),
            'maximum_null': float(null.max())}


def coverage_decision(rows, role):
    require(role in ('linear', 'exact', 'mismatch') and rows and len({r['rank'] for r in rows}) == 1, 'Coverage identity differs')
    n = sum(r['sample_count'] for r in rows); rank = rows[0]['rank']; off = sum(r['off_support'] for r in rows)
    maximum = max(r['maximum_null'] for r in rows)
    result = {'role': role, 'sample_count': n, 'numerical_rank': rank, 'off_support_count': off,
        'deterministic_support_passed': off == 0, 'maximum_nullspace_residual_metres': maximum,
        'point_mass': rank == 0, 'point_mass_all_in_support': off == 0 if rank == 0 else None,
        'eligible_for_gaussian_coverage_gate': role == 'linear' and rank > 0,
        'contract_sample_count': n == 131072, 'checks': []}
    tail = .01/(2*432)
    for index, p in enumerate(LEVELS if rank else ()):
        k = sum(r['counts'][index] for r in rows)
        low = 0. if k == 0 else float(beta.ppf(tail, k, n-k+1))
        high = 1. if k == n else float(beta.isf(tail, k+1, n-k))
        inside = low <= p <= high
        result['checks'].append({'nominal_probability': p, 'chi_square_threshold': float(chi2.ppf(p, rank)),
            'interval': {'lower': low, 'upper': high, 'successes': k, 'count': n, 'observed_fraction': k/n,
                         'family_alpha': .01, 'family_comparisons': 432, 'individual_confidence': 1-2*tail},
            'observed_minus_nominal': k/n-p, 'nominal_in_interval': inside,
            'gate_passed': inside if role == 'linear' else None})
    result['passed'] = (off == 0 and all(r['gate_passed'] for r in result['checks'])) if role == 'linear' and rank else None
    result['eligible_for_contract_gate'] = role == 'linear' and rank == 3 and n == 131072
    result['contract_gate_passed'] = result['passed'] if result['eligible_for_contract_gate'] else None
    return result


def recount_batch(case, normals):
    z = finite(normals); require(z.ndim == 2 and z.shape[1] == 3 and len(z) >= 2, 'Invalid saved normal array')
    result = {'modes': {}}; values = {}
    for mode in MODES:
        ref = reference(case, mode); factor = lower_factor(ref['input_covariance'])
        delta = np.column_stack([sum(z[:, j]*factor[i, j] for j in range(i+1)) for i in range(3)])
        linear = ref['nominal_output']+np.column_stack([sum(delta[:, j]*ref['jacobian'][i, j] for j in range(3)) for i in range(3)])
        inputs = np.asarray(case['nominal'])+delta; exact = project(inputs, case['intrinsic'])
        row = {'nonpositive_depth_count': int(np.count_nonzero(inputs[:, 2] <= 0)),
               'depth_above_two_metres_count': int(np.count_nonzero(inputs[:, 2] > 2))}
        for form, points, prefix in [('linear', linear, 'first_order'), ('exact', exact, 'exact')]:
            counts = count_coverage(points, ref[prefix+'_mean'], ref[prefix+'_covariance'])
            row[form] = {'statistics': batch_moments(points), 'coverage': coverage_decision([counts], form), '_counts': counts}
        result['modes'][mode] = row; values[mode] = (ref, linear)
    ref = values['diagonal'][0]
    counts = count_coverage(values['full'][1], ref['first_order_mean'], ref['first_order_covariance'])
    result['mismatch'] = coverage_decision([counts], 'mismatch'); result['_mismatch_counts'] = counts
    return result


def public_batch(row):
    return {'modes': {mode: {key: ({k: v for k, v in value.items() if not k.startswith('_')} if key in FORMS else value)
            for key, value in row['modes'][mode].items()} for mode in MODES}, 'mismatch': row['mismatch']}


def recount_summary(case, batches):
    require(len(batches) == 16, 'Incomplete saved case')
    result = {'schema': 19, 'case_id': case['case_id'], 'modes': {}, 'passed': True,
              'production_samples_required': 131072, 'nonlinear_and_mismatch_are_diagnostics_only': True}
    for mode in MODES:
        ref = reference(case, mode); rows = [b['modes'][mode] for b in batches]
        production_reference = {k: v for k, v in ref.items() if k != 'input_covariance'}
        production_reference['input'] = {'covariance': ref['input_covariance']}
        item = {'reference': production_reference,
            'nonpositive_depth_count': sum(r['nonpositive_depth_count'] for r in rows),
            'depth_above_two_metres_count': sum(r['depth_above_two_metres_count'] for r in rows)}
        for form, prefix in [('linear', 'first_order'), ('exact', 'exact')]:
            summary = combined_moments([r[form]['statistics'] for r in rows])
            moments = moment_decision(summary, ref[prefix+'_mean'], ref[prefix+'_covariance'])
            coverage = coverage_decision([r[form]['_counts'] for r in rows], form)
            item[form] = {'summary': summary, 'moments': moments, 'coverage': coverage}
            result['passed'] = result['passed'] and moments['contract_gate_passed'] is True
            if form == 'linear': result['passed'] = result['passed'] and coverage['contract_gate_passed'] is True
        result['passed'] = result['passed'] and item['nonpositive_depth_count'] == 0
        result['modes'][mode] = item
    result['mismatch_full_linear_with_diagonal_covariance'] = coverage_decision([b['_mismatch_counts'] for b in batches], 'mismatch')
    return result


def audit_attempts(directory, case, binding, batch_size=8192, project_root=None, logs_root='logs/P6/geometry_propagation_v19'):
    """Count observable draw evidence; unfinished attempts retain unknown totals."""
    project_root = Path(project_root) if project_root is not None else directory.parent
    parent = directory/'attempts'
    attempts = sorted(parent.iterdir()) if parent.exists() else []
    require(attempts and [p.name for p in attempts] == ['attempt_%06d'%i for i in range(1, len(attempts)+1)]
            and all(p.is_dir() and not p.is_symlink() for p in attempts), 'Attempt sequence is incomplete')
    returned = []; unknown = 0; unobserved = 0; resources = []
    for ordinal, attempt in enumerate(attempts, 1):
        allowed = {'intent.json', 'worker_started.json', 'process_outcome.json', 'worker_result.json'}
        allowed |= {name+'.sha256.json' for name in list(allowed) if name.endswith('.json')}
        for path in attempt.iterdir():
            require(path.name in allowed or (path.name == 'draws' and path.is_dir()), 'Unexpected attempt evidence')
        intent = None
        if (attempt/'intent.json').exists():
            intent = store.load(attempt/'intent.json', binding)
            close(intent, {'schema': 19, 'case_id': case['case_id'], 'attempt': ordinal})
            require(set(intent) == {'schema', 'case_id', 'attempt', 'command', 'stdout', 'stderr'} and isinstance(intent['command'], list)
                    and all(isinstance(part, str) for part in intent['command'])
                    and intent['command'][-4:] == ['--worker-case', case['case_id'], '--attempt', str(ordinal)], 'Worker command identity differs')
            for stream in ('stdout', 'stderr'):
                require(intent[stream] == '%s/%s_attempt_%06d.%s.log'%(logs_root, case['case_id'], ordinal, stream), 'External log path differs')
        if (attempt/'worker_started.json').exists():
            require(store.load(attempt/'worker_started.json', binding) == {'schema': 19, 'case_id': case['case_id'], 'attempt': ordinal}, 'Worker start differs')
        if (attempt/'process_outcome.json').exists():
            outcome = store.load(attempt/'process_outcome.json', binding)
            close(outcome, {'schema': 19, 'case_id': case['case_id'], 'attempt': ordinal})
            require(outcome['status'] in ('exit_observed', 'timeout_killed', 'process_error')
                    and (type(outcome['returncode']) is int if outcome['status'] == 'exit_observed' else outcome['returncode'] is None), 'Unknown worker outcome')
            require(intent is not None, 'Observed process lacks parent intent')
            if outcome['status'] == 'process_error': require(outcome['logs'] is None, 'Unknown writer logs must remain unsealed')
            else:
                require(type(outcome['logs']) is dict and set(outcome['logs']) == {'stdout', 'stderr'}, 'Missing sealed external logs')
                for stream in ('stdout', 'stderr'):
                    require(outcome['logs'][stream] == {'path': intent[stream], 'state': store.state(store.safe(project_root, intent[stream]))},
                            'Observed external log changed SHA, bytes or mtime')
            if outcome['status'] != 'exit_observed': unobserved += 1
        else: unobserved += 1
        attempt_returned = []; attempt_unknown = 0
        draws = sorted((attempt/'draws').iterdir()) if (attempt/'draws').exists() else []
        require(not draws or (attempt/'worker_started.json').exists(), 'Draws lack a worker-start record')
        for draw in draws:
            require(draw.is_dir() and re.fullmatch(r'batch_(0[0-9]|1[0-5])', draw.name), 'Unknown draw evidence')
            require(all(p.name in ('intent.json', 'intent.json.sha256.json', 'outcome.json', 'outcome.json.sha256.json')
                        for p in draw.iterdir()), 'Unexpected draw evidence')
            index = int(draw.name[6:]); expected = {'schema': 19, 'case_id': case['case_id'], 'batch_index': index,
                'seed': independent_seed(case['case_id'], index), 'sample_count': batch_size,
                'generator': 'numpy.RandomState.standard_normal'}
            if (draw/'intent.json').exists(): require(store.load(draw/'intent.json', binding) == expected, 'Draw intent differs')
            if (draw/'outcome.json').exists():
                require((draw/'intent.json').exists(), 'Returned draw lacks intent')
                row = store.load(draw/'outcome.json', binding); close(row, dict(expected, status='returned'))
                require(set(row) == set(expected)|{'status', 'z_sha256', 'z_bytes'} and re.fullmatch('[0-9a-f]{64}', row['z_sha256'])
                        and type(row['z_bytes']) is int and row['z_bytes'] > batch_size*3*8, 'Invalid returned-array digest or size')
                returned.append(row); attempt_returned.append(row)
            else: unknown += 1; attempt_unknown += 1
        if (attempt/'worker_result.json').exists():
            require(attempt_unknown == 0 and (attempt/'worker_started.json').exists(), 'Completed worker has uncertain draw evidence')
            row = store.load(attempt/'worker_result.json', binding)
            close(row, {'schema': 19, 'status': 'case_worker_completed', 'case_id': case['case_id'],
                'new_random_draw_calls': len(attempt_returned), 'new_standard_normal_triples': len(attempt_returned)*batch_size,
                'new_model_forward_calls': 0, 'new_training_updates': 0})
            resources.append(row['resources'])
    uncertain = unknown+unobserved > 0
    accounting = {'worker_attempts': len(attempts), 'returned_random_draw_calls': len(returned),
        'unknown_random_draw_calls': unknown, 'unobserved_worker_exits': unobserved,
        'exact_random_draw_calls': None if uncertain else len(returned),
        'known_standard_normal_triples_lower_bound': len(returned)*batch_size,
        'exact_standard_normal_triples': None if uncertain else len(returned)*batch_size}
    return accounting, returned, resources


def audit_case(directory, case, binding, *, batch_size=8192, guard=None, project_root=None,
               logs_root='logs/P6/geometry_propagation_v19'):
    """Read one complete case; small explicit batch sizes exist only for fixtures."""
    require(directory.name == case['case_id'], 'Case directory identity differs')
    before = store.snapshot(directory)
    completion = store.load(directory/'completion.json', binding)
    require(completion == {'schema': 19, 'case_id': case['case_id'], 'binding_fingerprint': binding['fingerprint'],
        'outputs': {name: value['sha256'] for name, value in before.items()
                    if name not in ('completion.json', 'completion.json.sha256.json')}}, 'Case seal differs')
    require(set(p.name for p in directory.iterdir()) <= {'attempts', 'batches', 'summary.json', 'summary.json.sha256.json',
        'report.json', 'report.json.sha256.json', 'completion.json', 'completion.json.sha256.json'}, 'Unexpected completed case file')
    require(sorted(p.name for p in (directory/'batches').iterdir()) == ['batch_%02d'%i for i in range(16)], 'Missing batch directories')
    accounting, returned, resources = audit_attempts(directory, case, binding, batch_size, project_root, logs_root)
    require(resources, 'Completed case lacks a completed-worker resource record')
    batches = []
    for index in range(16):
        if guard: guard.check()
        folder = directory/('batches/batch_%02d'%index); path = folder/'z.npy'
        require(set(p.name for p in folder.iterdir()) == {'z.npy', 'z.npy.sha256.json', 'statistics.json', 'statistics.json.sha256.json'}, 'Unexpected batch file')
        store.verify_bytes(path, binding)
        require(path.stat().st_size <= batch_size*3*8+65536, 'Oversized normal-array container')
        with path.open('rb') as stream:
            normals = np.load(stream, allow_pickle=False)
            require(stream.read(1) == b'', 'Trailing bytes in normal-array container')
        require(normals.dtype == np.float64 and normals.shape == (batch_size, 3)
                and normals.flags.c_contiguous and np.isfinite(normals).all(), 'Invalid saved normal array')
        seed = independent_seed(case['case_id'], index); checksum = store.digest(path)
        matching = [r for r in returned if r['batch_index'] == index]
        require(matching and all(r['seed'] == seed and r['z_sha256'] == checksum and r['z_bytes'] == path.stat().st_size
                for r in matching), 'Saved Z has no consistent returned-draw provenance')
        stored = store.load(folder/'statistics.json', binding)
        close(stored, {'schema': 19, 'case_id': case['case_id'], 'batch_index': index, 'seed': seed,
            'z_sha256': checksum, 'sample_count': batch_size})
        recounted = recount_batch(case, normals)
        close(stored['values'], plain(public_batch(recounted)), case['case_id']+'/batch_%02d'%index)
        batches.append(recounted)
    summary = recount_summary(case, batches)
    stored_summary = store.load(directory/'summary.json', binding)
    close(stored_summary, plain(summary), case['case_id']+'/summary')
    for mode in MODES:
        close(stored_summary['modes'][mode]['reference'], plain(summary['modes'][mode]['reference']),
              case['case_id']+'/'+mode+'/analytic-reference', atol=1e-18, rtol=1e-10)
    report = {'schema': 19, 'case_id': case['case_id'], 'passed': summary['passed'],
        'summary_sha256': store.digest(directory/'summary.json'), 'draw_accounting': accounting}
    require(store.load(directory/'report.json', binding) == report, 'Case report or accounting differs')
    require(store.snapshot(directory) == before, 'Audit changed production evidence')
    return summary, report, resources


def independent_deterministic(cases, saved):
    """Recheck nominal derivatives, exact fourth-degree moments and six supports."""
    import itertools
    nodes = [-math.sqrt(3.), 0., math.sqrt(3.)]; weights = [1/6, 2/3, 1/6]
    indices = list(itertools.product(range(3), repeat=3))
    z = np.asarray([[nodes[i] for i in row] for row in indices])
    w = np.asarray([math.prod(weights[i] for i in row) for row in indices])
    require(saved['schema'] == 19 and saved['passed'] is True and saved['new_random_draw_calls'] == 0
            and len(saved['case_checks']) == len(cases) and len(saved['rank_checks']) == 6, 'Incomplete deterministic evidence')
    largest = 0.
    for case, original in zip(cases, saved['case_checks']):
        require(original['case_id'] == case['case_id'] and original['passed'] is True, 'Deterministic case identity differs')
        nominal = np.asarray(case['nominal']); steps = [.01, .01, .0001]; ref = reference(case, 'full')
        fd = np.column_stack([(project(nominal+np.eye(3)[i]*h, case['intrinsic'])
                              -project(nominal-np.eye(3)[i]*h, case['intrinsic']))/(2*h) for i, h in enumerate(steps)])
        delta = fd-ref['jacobian']; ratio = np.abs(delta)/(1e-9+1e-7*np.abs(ref['jacobian']))
        require(np.all(ratio <= 1), 'Independent Jacobian check failed'); largest = max(largest, float(ratio.max()))
        close(original['jacobian'], plain({'analytic': ref['jacobian'], 'central_difference': fd, 'difference': delta, 'passed': True}))
        for mode in MODES:
            ref = reference(case, mode); perturbation = z @ lower_factor(ref['input_covariance']).T
            values = {'linear': ref['nominal_output']+perturbation @ ref['jacobian'].T,
                      'exact': project(nominal+perturbation, case['intrinsic'])}
            for form, prefix in [('linear', 'first_order'), ('exact', 'exact')]:
                mean = w @ values[form]; residual = values[form]-mean
                cov = np.einsum('n,ni,nj->ij', w, residual, residual)
                diff_mean = mean-ref[prefix+'_mean']; diff_cov = cov-ref[prefix+'_covariance']
                require(np.all(np.abs(diff_mean) <= 1e-12+1e-12*np.abs(ref[prefix+'_mean']))
                    and np.all(np.abs(diff_cov) <= 1e-14+1e-10*np.abs(ref[prefix+'_covariance'])), 'Independent Gaussian quadrature failed')
                close(original['moments'][mode][form], plain({'mean_difference': diff_mean, 'covariance_difference': diff_cov, 'passed': True}))
    base = dict(cases[0], nominal=[319.5, 239.5, 1.])
    for row, vector, expected in zip(saved['rank_checks'], ([0,0,0], [1,0,0], [0,1,0], [0,0,.01], [1,0,.01], [0,1,.01]), (0,1,1,1,2,2)):
        vector = np.asarray(vector, float); case = dict(base, input_covariance=np.outer(vector, vector), standard_deviation=np.abs(vector))
        ref = reference(case, 'full'); errors = np.asarray([-3., -1., 0., 1., 3.])[:, None]*vector
        for form, points, prefix in [('linear', ref['nominal_output']+errors @ ref['jacobian'].T, 'first_order'),
                                     ('exact', project(np.asarray(case['nominal'])+errors, case['intrinsic']), 'exact')]:
            check = count_coverage(points, ref[prefix+'_mean'], ref[prefix+'_covariance'])
            require(check['off_support'] == 0 and check['rank'] == (expected if form == 'exact' else int(np.any(vector))), 'Rank/support reference failed')
        close(row, {'vector': vector.tolist(), 'expected_exact_rank': expected, 'actual_exact_rank': expected, 'passed': True})
    return {'passed': True, 'case_count': len(cases), 'rank_boundary_count': 6, 'maximum_jacobian_tolerance_ratio': largest,
            'new_random_draw_calls': 0, 'weighted_quadrature_points_per_case_and_mode': 27}


def render_figures(case_rows):
    """Return deterministic PNG bytes; every declared case remains visible."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    output = {}; x = np.arange(len(case_rows))
    colors = {'full': '#176B9B', 'diagonal': '#C65D1B'}
    def finish(figure, name):
        figure.tight_layout(); FigureCanvasAgg(figure)
        stream = io.BytesIO(); figure.savefig(stream, format='png', dpi=150,
            metadata={'Software': 'Synthetic geometry verification'})
        output[name] = stream.getvalue()
    figure = Figure(figsize=(12, 8)); axes = figure.subplots(2, 2)
    for mode in MODES:
        refs = [row['modes'][mode]['reference'] for row in case_rows]
        for axis, component in zip(axes[0], (0, 1)):
            axis.plot(x, [r['mean_bias'][component]*1e6 for r in refs], label=mode, color=colors[mode], linewidth=1)
            axis.axhline(0, color='#777777', linewidth=.5); axis.set_ylabel('Bias (micrometres)')
        axes[1, 0].plot(x, [np.linalg.norm(r['second_order_covariance_correction'])/np.linalg.norm(r['first_order_covariance']) for r in refs],
                       label=mode, color=colors[mode], linewidth=1)
        axes[1, 1].plot(x, [r['second_order_covariance_correction'][0][1]*1e12 for r in refs], label=mode, color=colors[mode], linewidth=1)
    for axis, title in zip(axes.flat, ('Exact mean shift in X', 'Exact mean shift in Y', 'Relative second-order correction', 'Signed XY correction')):
        axis.set_title(title); axis.set_xlabel('Case index (all 72)'); axis.grid(alpha=.2); axis.legend(fontsize=8)
    axes[1, 0].set_ylabel('Frobenius norm H / norm C1'); axes[1, 1].set_ylabel('Hxy (square micrometres)')
    figure.suptitle('Synthetic fixed-camera Gaussian reference; no sensor calibration claim', fontsize=11)
    finish(figure, FIGURES[0])
    figure = Figure(figsize=(12, 8)); axes = figure.subplots(2, 2)
    for i, mode in enumerate(MODES):
        for j, form in enumerate(FORMS):
            axis = axes[i, j]
            for item, color in [('mean', '#176B9B'), ('covariance', '#C65D1B')]:
                ratios = []
                for row in case_rows:
                    check = row['modes'][mode][form]['moments'][item]
                    select = np.asarray(check['checked_components'], bool)
                    ratios.append(float((np.abs(check['difference'])/np.asarray(check['allowance']))[select].max()))
                axis.plot(x, ratios, '.-', markersize=3, linewidth=.6, label=item, color=color)
            axis.axhline(1., color='#AE2222', linestyle='--', label='predeclared limit')
            axis.set_title(mode+' / '+form); axis.set_ylabel('Maximum |difference| / allowance')
            axis.set_xlabel('Case index (all 72)'); axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure.suptitle('Moment consistency against the corresponding analytic mean and covariance', fontsize=11)
    finish(figure, FIGURES[1])
    figure = Figure(figsize=(15, 10)); axes = figure.subplots(3, 3)
    for i, probability in enumerate(LEVELS):
        for j, mode in enumerate((*MODES, 'mismatch')):
            axis = axes[i, j]; axis.axhline(probability, color='#333333', linestyle='--', linewidth=.8, label='nominal')
            if mode in MODES:
                for form, color in [('linear', '#176B9B'), ('exact', '#C65D1B')]:
                    rows = [r['modes'][mode][form]['coverage']['checks'][i]['interval'] for r in case_rows]
                    observed = np.asarray([r['observed_fraction'] for r in rows])
                    if form == 'linear':
                        axis.errorbar(x, observed, yerr=[observed-[r['lower'] for r in rows], [r['upper'] for r in rows]-observed],
                                      color=color, fmt='.', markersize=2, elinewidth=.4, label='linear with CP interval')
                    else: axis.plot(x, observed, '.', color=color, markersize=3, label='exact descriptive')
            else:
                observed = [r['mismatch_full_linear_with_diagonal_covariance']['checks'][i]['interval']['observed_fraction'] for r in case_rows]
                axis.plot(x, observed, '.', color='#6C428D', markersize=3, label='full linear / diagonal C1')
            axis.set_title('%s, nominal %.0f%%'%(mode, probability*100)); axis.set_xlabel('Case index (all 72)')
            axis.set_ylabel('Observed coverage'); axis.grid(alpha=.2); axis.legend(fontsize=7)
    figure.suptitle('Only linear matched coverage is an acceptance gate; exact and mismatch are descriptive', fontsize=11)
    finish(figure, FIGURES[2]); return output


def evidence_state(root, context):
    sources = {name: store.state(Path(context['protocol']['base_python']['path']) if name == 'external_base_python'
               else store.safe(root, name)) for name in context['source_state']}
    external = {name: store.state(store.safe(root, name)) for name in context['report_state']}
    return {'production_tree': store.snapshot(context['directory']), 'sources': sources, 'production_report': external,
            'sealed_external_logs': {name: store.state(store.safe(root, name)) for name in context['log_state']},
            'recovery_proof': store.state(store.safe(root, context['protocol']['outputs']['resume_report']))}


def audit_recovery(root, context):
    """Independently bind the actual two-case and completed process boundaries."""
    protocol = context['protocol']; path = store.safe(root, protocol['outputs']['resume_report'])
    require(not list(path.parent.glob(path.name+'.*.tmp')), 'Completed recovery proof has temporary files')
    proof = store.read(path)
    expected_binding = {'protocol_sha256': store.digest(context['config']),
        'wrapper_sha256': store.digest(root/'scripts/run_geometry_propagation_resume_v19.py'),
        'child_sha256': store.digest(root/'scripts/run_geometry_propagation_v19.py'),
        'interpreter_sha256': store.digest(store.safe(root, protocol['interpreter'])),
        'pipeline_sha256': store.digest(root/'scripts/pipeline.py')}
    require(proof['schema'] == 19 and proof['status'] == 'passed' and proof['binding'] == expected_binding
            and proof['scope'] == 'Actual process boundaries with two cases first; no physical power-cut claim', 'Recovery proof identity differs')
    entry = proof['entry']; relative = context['directory'].relative_to(root).as_posix()+'/'
    tree = {relative+name: row for name, row in context['sealedtree'].items()}
    expected_prefix = {name: value for name, value in tree.items() if name in {
        relative+'binding.json', relative+'binding.json.sha256.json', relative+'deterministic.json', relative+'deterministic.json.sha256.json'}
        or name.startswith(relative+'cases/case_000/') or name.startswith(relative+'cases/case_001/')}
    require(expected_prefix and entry['prefix_before'] == entry['prefix_after'] == expected_prefix,
            'Recovery prefix is incomplete, changed, or includes later-case files')
    require(entry['completed_snapshot'] == dict(tree, **context['report_state']) and entry['byte_and_mtime_identical'] is True
            and entry['read_only_check_passed'] is True, 'Recovery completed evidence differs')
    expected_report_sha = store.digest(store.safe(root, protocol['outputs']['report']))
    require(entry['report_sha256'] == expected_report_sha
            and entry['completion_sha256'] == store.digest(context['directory']/'completion.json'), 'Recovery completion digest differs')
    operations = (('normalization',) if 'normalization_record' in entry else ())+('probe', 'completion', 'read_only')
    for operation in operations:
        row = entry[operation+'_record']
        require(row['schema'] == 19 and row['expected_cases'] == 72, 'Recovery work identity differs')
        for key in ('new_cases', 'committed_cases', 'new_random_draw_calls', 'new_model_forward_calls', 'new_training_updates'):
            require(type(row[key]) is int and row[key] >= 0, 'Invalid recovery counter')
        require(row['new_cases'] <= row['committed_cases'] <= 72 and row['new_random_draw_calls'] <= row['new_cases']*16
                and row['new_model_forward_calls'] == row['new_training_updates'] == 0, 'Recovery work counters disagree')
        if operation == 'normalization':
            require(row['status'] == 'partial' and row['committed_cases'] <= 2
                    and row['new_cases'] == row['new_random_draw_calls'] == 0, 'Normalization advanced beyond the interrupted prefix')
        elif operation == 'probe': require(row['status'] == 'partial' and row['committed_cases'] == 2, 'Missing exact two-case boundary')
        else:
            require(row['committed_cases'] == 72 and row['status'] in ('completed', 'verified_completed_read_only')
                    and row['report_sha256'] == expected_report_sha, 'Recovery final cohort differs')
        if operation == 'read_only' or row['status'] == 'verified_completed_read_only':
            require(row['status'] == 'verified_completed_read_only' and row['new_cases'] == row['new_random_draw_calls'] == 0,
                    'Recovery read-only process performed new work')
        resource = row['resources']
        for key, maximum in [('elapsed_seconds', protocol['resources']['driver_soft_seconds']),
                             ('peak_rss_bytes', protocol['resources']['rss_soft_bytes']),
                             ('current_output_bytes', protocol['resources']['output_limit_bytes'])]:
            require(type(resource[key]) in (int, float) and math.isfinite(resource[key]) and 0 <= resource[key] <= maximum,
                    'Recovery resource evidence exceeds fixed limits')
        require(resource['peak_gpu_allocated_bytes'] == resource['peak_gpu_reserved_bytes'] == 0, 'Recovery GPU use differs')
    return {'passed': True, 'proof_sha256': store.digest(path), 'prefix_file_count': len(expected_prefix),
            'physical_power_loss_tested': False, 'new_random_draw_calls': 0}


def acceptance_paths(root, protocol):
    report = store.safe(root, protocol['outputs']['acceptance_report'])
    figures = store.safe(root, protocol['outputs']['visuals_dir'])
    require(report != store.safe(root, protocol['outputs']['report']) and not report.is_relative_to(store.safe(root, protocol['outputs']['root']))
            and not figures.is_relative_to(store.safe(root, protocol['outputs']['root'])), 'Acceptance overlaps production evidence')
    require(all(name in protocol['budget_paths'] for name in (protocol['outputs']['acceptance_report'],
        protocol['outputs']['acceptance_report']+'.sha256.json', protocol['outputs']['visuals_dir'], protocol['outputs']['logs'])), 'Acceptance budget omissions')
    return report, figures


def run(root=ROOT, config=CONFIG, check_only=False):
    import importlib.metadata
    root = Path(root).resolve(); context = store.completed_context(root, config)
    protocol, binding = context['protocol'], context['binding']
    require(protocol['cases'] == declared_cases(), 'Independent case enumeration differs')
    seeds = [{'case_id': case['case_id'], 'batch_index': i, 'seed': independent_seed(case['case_id'], i)}
             for case in protocol['cases'] for i in range(16)]
    require(protocol['seed_inventory'] == seeds and len({r['seed'] for r in seeds}) == 1152, 'Independent seed inventory differs')
    required = {SOURCE, 'tests/test_geometry_propagation_acceptance_v19.py', 'scripts/run_geometry_propagation_v19.py',
        'scripts/geometry_storage_v19.py', 'src/geometry_propagation_v19.py', 'scripts/multiview_common_v14r1.py',
        'scripts/run_geometry_propagation_resume_v19.py', 'scripts/pipeline.py', 'tests/test_geometry_propagation_wrapper_v19.py',
        'tests/test_geometry_propagation_storage_v19.py', 'tests/test_geometry_propagation_v19.py',
        'docs/第十九轮三维传播合成验证草案.txt'}
    require(required <= set(protocol['source_hashes']), 'Acceptance sources not frozen')
    require({protocol['interpreter'], 'runtime/envs/grasp_eval/pyvenv.cfg'} <= set(protocol['input_hashes']), 'Runtime input freeze incomplete')
    require(protocol['resources'] == {'cpu_threads': 4, 'worker_timeout_seconds': 120, 'driver_soft_seconds': 1800,
        'rss_soft_bytes': 2*1024**3, 'output_limit_bytes': 1024**3, 'disk_reserve_bytes': 60*1024**3}, 'Resource contract changed')
    require(protocol['acceptance_runtime'] == {'matplotlib': importlib.metadata.version('matplotlib')}, 'Plotting runtime changed')
    report_path, visuals = acceptance_paths(root, protocol)
    readonly = check_only or report_path.exists()
    require(not check_only or report_path.exists(), 'Check-only requires completed acceptance')
    guard = store.Guard(root, protocol); guard.check()
    before = evidence_state(root, context)
    recovery = audit_recovery(root, context)
    acceptance_before = ({str(p): store.state(p) for p in (report_path, store.receipt(report_path))}, store.snapshot(visuals)) if readonly else None
    # Only acceptance-owned recovery is permitted. Production evidence is never repaired.
    if not readonly:
        store.recover(report_path, binding)
        store.validate_layout(visuals, [visuals/name for name in FIGURES], binding)
    else:
        store.recover(report_path, binding, readonly=True)
        store.validate_layout(visuals, [visuals/name for name in FIGURES], binding, readonly=True)
    require(set(p.name for p in context['directory'].iterdir()) == {'binding.json', 'binding.json.sha256.json',
        'deterministic.json', 'deterministic.json.sha256.json', 'completion.json', 'completion.json.sha256.json', 'cases'}, 'Unexpected root evidence')
    require(sorted(p.name for p in (context['directory']/'cases').iterdir()) == [c['case_id'] for c in protocol['cases']], 'Case directory cohort differs')
    deterministic = independent_deterministic(protocol['cases'], store.load(context['directory']/'deterministic.json', binding))
    rows = []; accounting = []; resources = []
    for case, production_report in zip(protocol['cases'], context['report']['cases']):
        guard.check(); summary, report, measured = audit_case(context['directory']/'cases'/case['case_id'], case, binding,
            guard=guard, project_root=root, logs_root=protocol['outputs']['logs'])
        require(report == production_report and summary['passed'] is True, 'Independent case did not pass frozen criteria')
        rows.append(plain(summary)); accounting.append(report['draw_accounting']); resources.extend(measured)
    for value in resources:
        require(0 <= value['elapsed_seconds'] <= protocol['resources']['worker_timeout_seconds']
            and 0 <= value['peak_rss_bytes'] <= protocol['resources']['rss_soft_bytes']
            and 0 <= value['current_output_bytes'] <= protocol['resources']['output_limit_bytes']
            and value['peak_gpu_allocated_bytes'] == value['peak_gpu_reserved_bytes'] == 0, 'Worker resource evidence failed')
    require(evidence_state(root, context) == before, 'Independent recount changed source or production evidence')
    figures = {name: store.verify_bytes(visuals/name, binding) for name in FIGURES} if readonly else {}
    if not readonly:
        # Preserve the plotting cache under budgeted logs, outside immutable figures.
        guard.check(8*1024**2)
        visuals.mkdir(parents=True, exist_ok=True)
        cache = store.safe(root, protocol['outputs']['logs'])/'acceptance_plot_cache'
        cache.mkdir(parents=True, exist_ok=True)
        previous = os.environ.get('MPLCONFIGDIR'); os.environ['MPLCONFIGDIR'] = str(cache)
        try: images = render_figures(rows)
        finally:
            if previous is None: os.environ.pop('MPLCONFIGDIR', None)
            else: os.environ['MPLCONFIGDIR'] = previous
        for name, payload in images.items():
            guard.check(len(payload)+4096); store.commit_bytes(visuals/name, payload, binding)
            figures[name] = store.verify_bytes(visuals/name, binding)
    unknown = sum(r['unknown_random_draw_calls'] for r in accounting)
    unobserved = sum(r['unobserved_worker_exits'] for r in accounting)
    returned = sum(r['returned_random_draw_calls'] for r in accounting)
    report = {'schema': 19, 'status': 'accepted_limited_synthetic_geometry', 'binding': binding,
        'production_report_sha256': store.digest(store.safe(root, protocol['outputs']['report'])),
        'production_completion_sha256': store.digest(context['directory']/'completion.json'),
        'case_count': 72, 'batch_count': 1152, 'designed_standard_normal_triples': 9437184,
        'designed_error_inputs': 18874368, 'new_random_draw_calls': 0, 'new_production_samples': 0,
        'new_model_forward_calls': 0, 'new_training_updates': 0, 'gpu_used': False,
        'independent_deterministic': deterministic, 'recovery': recovery, 'cases': rows, 'case_draw_accounting': accounting,
        'draw_accounting': {'returned_random_draw_calls': returned, 'unknown_random_draw_calls': unknown,
            'unobserved_worker_exits': unobserved, 'exact_random_draw_calls': None if unknown+unobserved else returned,
            'known_standard_normal_triples_lower_bound': returned*8192,
            'exact_standard_normal_triples': None if unknown+unobserved else returned*8192},
        'source_and_production_evidence_unchanged': True,
        'completed_worker_resource_record_count': len(resources),
        'worker_attempts_without_completed_resource_record': sum(r['worker_attempts'] for r in accounting)-len(resources),
        'resource_evidence_scope': 'Available completed-worker peaks and current audit limits; unmeasured interrupted peaks are not invented.',
        'source_state_fingerprint': store.fingerprint(before['sources']),
        'production_state_fingerprint': store.fingerprint(before['production_tree']),
        'sealed_external_log_state_fingerprint': store.fingerprint(before['sealed_external_logs']),
        'unsealed_external_log_paths': context['unsealed_log_paths'],
        'figures': figures, 'independent_scope': ['scalar Cholesky and expanded analytic covariance',
            'saved-array means, within-batch covariance, pooled covariance and standard errors',
            'linear-solve coverage counts and independent t/CP decisions', 'attempt provenance and immutable evidence'],
        'shared_dependencies': ['NumPy float64 arithmetic', 'SciPy distribution quantiles', 'schema19 storage and runtime validation'],
        'limitations': ['Two separate nominal alpha=0.01 families; no combined 99 percent guarantee.',
            'Batch-t moment checks are approximate engineering tolerances.',
            'Nonlinear and correlation-mismatch coverage are descriptive.',
            'No sensor calibration, quality-to-variance mapping, real-data or grasp-performance claim.'],
        'full_p6_completion': False, 'physical_power_loss_tested': False}
    if readonly: require(store.load(report_path, binding) == report, 'Saved independent acceptance differs')
    else:
        payload = store.json_bytes(report); guard.check(len(payload)+4096); store.commit_bytes(report_path, payload, binding)
    require(evidence_state(root, context) == before, 'Acceptance publishing changed production evidence')
    if readonly:
        require(acceptance_before == ({str(p): store.state(p) for p in (report_path, store.receipt(report_path))}, store.snapshot(visuals)),
                'Read-only acceptance changed its report or figures')
    return {'schema': 19, 'status': 'verified_read_only' if readonly else report['status'], 'case_count': 72,
            'new_random_draw_calls': 0, 'new_production_samples': 0, 'new_model_forward_calls': 0, 'new_training_updates': 0,
            'report_sha256': store.digest(report_path), 'resources': guard.check()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(config=args.config, check_only=args.check_only), ensure_ascii=False, allow_nan=False), flush=True)
