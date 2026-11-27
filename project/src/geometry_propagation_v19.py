"""Pure float64 synthetic pinhole geometry and diagnostic statistics.

This module performs no sampling, file access, model inference or scheduling.
Input covariance is declared in pixel/metre units; no quality-to-variance map
is defined. Passing a supplied normal array does not generate new samples.
"""
import hashlib
import math
import re

import numpy as np
from scipy.stats import beta, chi2, t


CASE_COUNT = 72
BATCH_COUNT = 16
BATCH_SIZE = 8192
SAMPLES_PER_CASE = BATCH_COUNT * BATCH_SIZE
MOMENT_COMPARISONS = 2592
COVERAGE_COMPARISONS = 432
FAMILY_ALPHA = 0.01
MASTER_SEED = 20261027
COVERAGE_LEVELS = (0.5, 0.8, 0.95)
FD_STEPS = (0.01, 0.01, 0.0001)
NULLSPACE_TOLERANCE_M = 1e-10
UPPER_TRIANGLE = np.triu_indices(3)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape=None, name='array'):
    _require(not np.iscomplexobj(value), 'Complex values are not '+name)
    result = np.asarray(value, dtype=np.float64)
    _require((shape is None or result.shape == shape) and np.isfinite(result).all(),
             'Expected finite float64 '+name)
    return result


def _points(value, name='points'):
    result = _array(value, name=name)
    _require(result.ndim in (1, 2) and result.shape[-1] == 3,
             name+' must have shape [3] or [N,3]')
    return result


def validate_intrinsic(intrinsic):
    """Accept fixed zero-skew pinhole K only; do not discard extra terms."""
    matrix = _array(intrinsic, (3, 3), 'intrinsic')
    _require(matrix[0, 0] > 0 and matrix[1, 1] > 0
             and matrix[0, 1] == matrix[1, 0] == 0
             and np.array_equal(matrix[2], [0., 0., 1.]),
             'Require positive focal lengths and canonical zero-skew K')
    return matrix.copy()


def validate_nominal(nominal):
    value = _array(nominal, (3,), 'nominal (u,v,z)')
    _require(value[2] > 0, 'Nominal depth must be positive')
    return value.copy()


def backproject(samples, intrinsic):
    """Evaluate the polynomial for finite samples, including zero/negative z.

    Samples are never clipped, filtered or resampled. Nominal-point checks and
    physical positive-depth policies are deliberately separate from this map.
    """
    values = _points(samples, 'input coordinates')
    matrix = validate_intrinsic(intrinsic)
    u, v, z = (values[..., index] for index in range(3))
    result = np.stack(((u-matrix[0, 2])*z/matrix[0, 0],
                       (v-matrix[1, 2])*z/matrix[1, 1], z), axis=-1)
    _require(np.isfinite(result).all(), 'Backprojection overflow')
    return result


def pinhole_jacobian(nominal, intrinsic):
    u, v, z = validate_nominal(nominal)
    matrix = validate_intrinsic(intrinsic)
    result = np.array([[z/matrix[0, 0], 0., (u-matrix[0, 2])/matrix[0, 0]],
                     [0., z/matrix[1, 1], (v-matrix[1, 2])/matrix[1, 1]],
                     [0., 0., 1.]], dtype=np.float64)
    _require(np.isfinite(result).all(), 'Jacobian overflow')
    return result


def central_difference_jacobian(nominal, intrinsic, steps=FD_STEPS):
    point = validate_nominal(nominal)
    step = _array(steps, (3,), 'finite difference steps')
    _require(np.all(step > 0) and point[2] > step[2], 'Invalid positive central-difference interval')
    columns = []
    for index in range(3):
        displacement = np.zeros(3, dtype=np.float64)
        displacement[index] = step[index]
        columns.append((backproject(point+displacement, intrinsic)
                        -backproject(point-displacement, intrinsic))/(2*step[index]))
    return np.column_stack(columns)


def jacobian_diagnostics(nominal, intrinsic):
    analytic = pinhole_jacobian(nominal, intrinsic)
    finite_difference = central_difference_jacobian(nominal, intrinsic)
    difference = finite_difference-analytic
    allowance = 1e-9+1e-7*np.abs(analytic)
    return {'analytic': analytic, 'central_difference': finite_difference,
            'difference': difference, 'absolute_difference': np.abs(difference),
            'allowance': allowance, 'maximum_absolute_difference': float(np.abs(difference).max()),
            'maximum_tolerance_ratio': float((np.abs(difference)/allowance).max()),
            'passed': bool(np.all(np.abs(difference) <= allowance)),
            'column_units': ['metres_per_pixel', 'metres_per_pixel', 'metres_per_metre'],
            'steps': list(FD_STEPS)}


def input_covariance_diagnostics(covariance, standard_deviation):
    """Inspect declared mixed-unit covariance without eigenvalue correction.

    Standard deviations are supplied physical scales, not estimates obtained
    from quality scores. Zero-scale rows/columns must be exactly zero.
    """
    matrix = _array(covariance, (3, 3), 'input covariance')
    scale = _array(standard_deviation, (3,), 'declared standard deviations')
    _require(np.all(scale >= 0) and np.all(np.diag(matrix) >= 0), 'Negative input variance or scale')
    active = np.flatnonzero(scale > 0)
    inactive = np.flatnonzero(scale == 0)
    _require(not len(inactive) or (np.all(matrix[inactive, :] == 0) and np.all(matrix[:, inactive] == 0)),
             'Zero-scale dimension has nonzero covariance')
    if len(active):
        correlation = matrix[np.ix_(active, active)]/np.outer(scale[active], scale[active])
        _require(np.isfinite(correlation).all(), 'Input normalization overflow')
        symmetry_error = float(np.max(np.abs(correlation-correlation.T)))
        _require(symmetry_error <= 1e-12 and np.all(np.abs(np.diag(correlation)-1.) <= 1e-12),
                 'Input covariance symmetry or declared standard deviations disagree')
        # Symmetric part is used solely for spectral diagnostics of roundoff;
        # the original matrix is retained and is never projected onto PSD.
        eigenvalues = np.linalg.eigvalsh((correlation+correlation.T)/2)
        threshold = max(1e-12*float(eigenvalues[-1]), 1e-14)
        _require(float(eigenvalues[0]) >= -threshold, 'Input covariance is not positive semidefinite')
        rank = int(np.count_nonzero(eigenvalues > threshold))
    else:
        correlation = np.empty((0, 0), np.float64)
        eigenvalues = np.empty(0, np.float64)
        symmetry_error, threshold, rank = 0., 1e-14, 0
    return {'covariance': matrix.copy(), 'standard_deviation': scale.copy(), 'active_indices': active,
            'normalized_correlation': correlation, 'normalized_eigenvalues': eigenvalues,
            'symmetry_error': symmetry_error, 'rank_threshold': threshold, 'numerical_rank': rank,
            'eigenvalues_modified': False, 'units': ['pixels', 'pixels', 'metres']}


def output_covariance_diagnostics(covariance):
    """Inspect an output covariance in square metres, independently of input rank."""
    matrix = _array(covariance, (3, 3), 'output covariance')
    symmetry_error = float(np.max(np.abs(matrix-matrix.T)))
    _require(symmetry_error <= 1e-14, 'Output covariance is not symmetric')
    eigenvalues, eigenvectors = np.linalg.eigh((matrix+matrix.T)/2)
    maximum = max(0., float(eigenvalues[-1]))
    psd_tolerance = 1e-12*max(maximum, 1e-12)
    threshold = max(1e-12*maximum, 1e-18)
    _require(float(eigenvalues[0]) >= -psd_tolerance, 'Output covariance is not positive semidefinite')
    return {'covariance': matrix.copy(), 'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors,
            'symmetry_error': symmetry_error, 'psd_tolerance': psd_tolerance, 'rank_threshold': threshold,
            'numerical_rank': int(np.count_nonzero(eigenvalues > threshold)), 'eigenvalues_modified': False,
            'units': 'square_metres'}


def propagate_gaussian(nominal, intrinsic, covariance, standard_deviation, mode='full'):
    """Return first-order and exact Gaussian moments for a fixed pinhole model."""
    _require(mode in ('full', 'diagonal'), 'Unknown covariance mode')
    point = validate_nominal(nominal)
    matrix = validate_intrinsic(intrinsic)
    original = input_covariance_diagnostics(covariance, standard_deviation)
    sigma = original['covariance'] if mode == 'full' else np.diag(np.diag(original['covariance']))
    selected = input_covariance_diagnostics(sigma, standard_deviation)
    jacobian = pinhole_jacobian(point, matrix)
    location = backproject(point, matrix)
    first = jacobian @ sigma @ jacobian.T
    fx, fy = matrix[0, 0], matrix[1, 1]
    bias = np.array([sigma[0, 2]/fx, sigma[1, 2]/fy, 0.], dtype=np.float64)
    correction = np.zeros((3, 3), dtype=np.float64)
    correction[0, 0] = (sigma[0, 0]*sigma[2, 2]+sigma[0, 2]**2)/fx**2
    correction[1, 1] = (sigma[1, 1]*sigma[2, 2]+sigma[1, 2]**2)/fy**2
    correction[0, 1] = correction[1, 0] = (sigma[0, 1]*sigma[2, 2]+sigma[0, 2]*sigma[1, 2])/(fx*fy)
    exact = first+correction
    return {'mode': mode, 'nominal': point, 'intrinsic': matrix, 'input': selected, 'jacobian': jacobian,
            'nominal_output': location, 'first_order_mean': location.copy(), 'first_order_covariance': first,
            'mean_bias': bias, 'second_order_covariance_correction': correction,
            'exact_mean': location+bias, 'exact_covariance': exact,
            'second_moment_about_nominal': exact+np.outer(bias, bias),
            'first_order_spectrum': output_covariance_diagnostics(first),
            'exact_spectrum': output_covariance_diagnostics(exact),
            'scope': 'declared_joint_Gaussian_input_fixed_K_no_clipping',
            'quality_to_variance_mapping_defined': False}


def support_distance(points, mean, covariance):
    """Return distance plus null-space residual; off-support distance is infinity."""
    values = _points(points)
    location = _array(mean, (3,), 'mean')
    spectrum = output_covariance_diagnostics(covariance)
    residual = values-location
    coordinates = residual @ spectrum['eigenvectors']
    active = spectrum['eigenvalues'] > spectrum['rank_threshold']
    projected = np.sum(coordinates[..., active]**2/spectrum['eigenvalues'][active], axis=-1)
    null_residual = np.linalg.norm(coordinates[..., ~active], axis=-1)
    _require(np.isfinite(projected).all() and np.isfinite(null_residual).all(), 'Support distance overflow')
    in_support = null_residual <= NULLSPACE_TOLERANCE_M
    return {'squared_distance': np.where(in_support, projected, np.inf),
            'projected_squared_distance': projected, 'nullspace_residual_metres': null_residual,
            'in_support': in_support, 'spectrum': spectrum, 'point_mass': spectrum['numerical_rank'] == 0,
            'nullspace_tolerance_metres': NULLSPACE_TOLERANCE_M}


def enumerate_cases():
    """Return the 72 JSON-compatible declarations, without drawing samples."""
    cameras = [('K0', [480, 640], [[600., 0., 319.5], [0., 600., 239.5], [0., 0., 1.]]),
               ('K1', [720, 1280], [[900., 0., 630.], [0., 500., 350.], [0., 0., 1.]])]
    correlations = [np.eye(3), np.array([[1., .30, .40], [.30, 1., -.20], [.40, -.20, 1.]]),
                    np.array([[1., -.25, -.35], [-.25, 1., .30], [-.35, .30, 1.]])]
    scales = [(.25, .25, .002), (2., 1., .02)]
    result = []
    for camera, shape, intrinsic in cameras:
        positions = [('principal', [intrinsic[0][2], intrinsic[1][2]]), ('top_left', [0., 0.]),
                     ('bottom_right', [float(shape[1]-1), float(shape[0]-1)])]
        for position, pixel in positions:
            for depth in (.5, 1.5):
                for level, scale in enumerate(scales):
                    diagonal = np.diag(scale)
                    for correlation_index, correlation in enumerate(correlations):
                        covariance = diagonal @ correlation @ diagonal
                        input_covariance_diagnostics(covariance, scale)
                        result.append({'case_id': 'case_%03d'%len(result), 'camera': camera, 'image_size_hw': shape.copy(),
                            'intrinsic': [row.copy() for row in intrinsic], 'position': position, 'nominal': pixel+[depth],
                            'noise_level': 'S%d'%level, 'standard_deviation': list(scale), 'correlation_id': 'R%d'%correlation_index,
                            'correlation': correlation.tolist(), 'input_covariance': covariance.tolist(),
                            'sample_count': SAMPLES_PER_CASE, 'batch_count': BATCH_COUNT, 'batch_size': BATCH_SIZE})
    _require(len(result) == CASE_COUNT, 'Case enumeration count changed')
    return result


def batch_seed(case_id, batch_index):
    _require(isinstance(case_id, str) and re.fullmatch(r'case_[0-9]{3}', case_id)
             and 0 <= int(case_id[5:]) < CASE_COUNT, 'Unknown synthetic case identity')
    _require(type(batch_index) is int and 0 <= batch_index < BATCH_COUNT, 'Unknown batch index')
    identity = 'R19|%d|%s|batch_%02d'%(MASTER_SEED, case_id, batch_index)
    return int(hashlib.sha256(identity.encode('ascii')).hexdigest()[:8], 16)


def seed_inventory():
    rows = [{'case_id': 'case_%03d'%case, 'batch_index': batch, 'seed': batch_seed('case_%03d'%case, batch)}
            for case in range(CASE_COUNT) for batch in range(BATCH_COUNT)]
    _require(len({row['seed'] for row in rows}) == len(rows), 'Declared batch seed collision')
    return rows


def batch_statistics(values):
    samples = _points(values, 'sample outputs')
    _require(samples.ndim == 2 and samples.shape[0] >= 2, 'At least two samples are needed for unbiased covariance')
    mean = samples.mean(axis=0, dtype=np.float64)
    centered = samples-mean
    m2 = centered.T @ centered
    covariance = m2/(len(samples)-1)
    _require(np.isfinite(mean).all() and np.isfinite(covariance).all(), 'Sample moment overflow')
    return {'count': len(samples), 'mean': mean, 'covariance': covariance, 'centered_sum_products': m2, 'ddof': 1}


def aggregate_batches(batches):
    """Keep equal-batch primary covariance separate from pooled covariance.

    Exactly 16 independent equal-size batches are required. Small fixture
    batches are supported; the future worker must separately require 8192.
    """
    _require(len(batches) == BATCH_COUNT, 'Require exactly sixteen batch estimates')
    counts, means, covariances = [], [], []
    for batch in batches:
        count = batch['count']
        _require(type(count) is int and count >= 2 and batch['ddof'] == 1, 'Invalid unbiased batch identity')
        mean = _array(batch['mean'], (3,), 'batch mean')
        covariance = _array(batch['covariance'], (3, 3), 'batch covariance')
        m2 = _array(batch['centered_sum_products'], (3, 3), 'batch centered sum products')
        _require(np.allclose(m2, covariance*(count-1), rtol=1e-12, atol=1e-14), 'Batch covariance and centered sums disagree')
        output_covariance_diagnostics(covariance)
        counts.append(count); means.append(mean); covariances.append(covariance)
    _require(len(set(counts)) == 1, 'Equal-size independent batches are required')
    means, covariances = np.stack(means), np.stack(covariances)
    mean = means.mean(axis=0, dtype=np.float64)
    primary_covariance = covariances.mean(axis=0, dtype=np.float64)
    centered_means = means-mean
    pooled_m2 = np.sum(covariances*(counts[0]-1), axis=0)+counts[0]*(centered_means.T @ centered_means)
    return {'batch_count': BATCH_COUNT, 'batch_size': counts[0], 'sample_count': sum(counts),
            'contract_sample_count': counts[0] == BATCH_SIZE, 'batch_means': means, 'batch_covariances': covariances,
            'primary_mean': mean, 'primary_covariance': primary_covariance,
            'mean_standard_error': means.std(axis=0, ddof=1)/math.sqrt(BATCH_COUNT),
            'covariance_standard_error': covariances.std(axis=0, ddof=1)/math.sqrt(BATCH_COUNT),
            'pooled_mean': mean.copy(), 'pooled_covariance': pooled_m2/(sum(counts)-1),
            'primary_covariance_definition': 'equal_mean_of_sixteen_within_batch_unbiased_covariances'}


def moment_tolerance(summary, reference_mean, reference_covariance):
    """Approximate batch-t engineering diagnostic, not an exact family guarantee."""
    _require(summary['batch_count'] == BATCH_COUNT, 'Moment tolerance requires sixteen batches')
    reference_mean = _array(reference_mean, (3,), 'reference mean')
    reference_covariance = _array(reference_covariance, (3, 3), 'reference covariance')
    output_covariance_diagnostics(reference_covariance)
    critical = float(t.isf(FAMILY_ALPHA/(2*MOMENT_COMPARISONS), BATCH_COUNT-1))
    result = {'batch_count': BATCH_COUNT, 'degrees_of_freedom': BATCH_COUNT-1, 'family_alpha': FAMILY_ALPHA,
              'family_comparisons': MOMENT_COMPARISONS, 'critical_value': critical,
              'family_scope': 'moments_only_not_combined_with_coverage',
              'approximate_batch_normality_required': True, 'exact_finite_sample_family_guarantee': False,
              'contract_sample_count': bool(summary['contract_sample_count'])}
    for name, reference, floor in (('mean', reference_mean, 1e-10), ('covariance', reference_covariance, 1e-14)):
        observed = _array(summary['primary_'+name], reference.shape, 'primary '+name)
        se = _array(summary[name+'_standard_error'], reference.shape, name+' standard error')
        _require(np.all(se >= 0), 'Negative standard error')
        difference = observed-reference
        allowance = critical*se+floor
        selected = np.ones(3, bool) if name == 'mean' else np.triu(np.ones((3, 3), bool))
        component_pass = np.abs(difference) <= allowance
        result[name] = {'observed': observed.copy(), 'reference': reference.copy(), 'difference': difference,
                        'standard_error': se.copy(), 'absolute_floor': floor, 'allowance': allowance,
                        'checked_components': selected, 'component_passed': component_pass,
                        'passed': bool(component_pass[selected].all())}
    result['passed'] = result['mean']['passed'] and result['covariance']['passed']
    eligible = (type(summary['batch_count']) is int and summary['batch_count'] == BATCH_COUNT
                and type(summary['batch_size']) is int and summary['batch_size'] == BATCH_SIZE
                and type(summary['sample_count']) is int and summary['sample_count'] == SAMPLES_PER_CASE
                and summary['contract_sample_count'] is True)
    result['eligible_for_contract_gate'] = eligible
    result['contract_gate_passed'] = result['passed'] if eligible else None
    return result


def clopper_pearson(successes, count):
    _require(type(count) is int and count > 0 and type(successes) is int and 0 <= successes <= count,
             'Require integer binomial counts with positive denominator')
    tail = FAMILY_ALPHA/(2*COVERAGE_COMPARISONS)
    low = 0. if successes == 0 else float(beta.ppf(tail, successes, count-successes+1))
    high = 1. if successes == count else float(beta.isf(tail, successes+1, count-successes))
    return {'lower': low, 'upper': high, 'successes': successes, 'count': count,
            'observed_fraction': successes/count, 'family_alpha': FAMILY_ALPHA,
            'family_comparisons': COVERAGE_COMPARISONS, 'individual_confidence': 1-2*tail}


def coverage_from_counts(counts, sample_count, numerical_rank, off_support_count,
                         maximum_nullspace_residual_metres, *, role):
    """Combine fixed-threshold integer counts without retaining output clouds."""
    _require(role in ('linear', 'exact', 'mismatch'), 'Coverage role must be explicit')
    _require(type(sample_count) is int and sample_count > 0 and type(numerical_rank) is int
             and 0 <= numerical_rank <= 3 and type(off_support_count) is int
             and 0 <= off_support_count <= sample_count, 'Invalid coverage support counts')
    residual = float(maximum_nullspace_residual_metres)
    _require(math.isfinite(residual) and residual >= 0
             and (residual <= NULLSPACE_TOLERANCE_M) == (off_support_count == 0),
             'Coverage nullspace residual and counts disagree')
    rank = numerical_rank
    _require(isinstance(counts, (tuple, list)) and len(counts) == (3 if rank else 0)
             and all(type(value) is int and 0 <= value <= sample_count-off_support_count for value in counts)
             and list(counts) == sorted(counts), 'Invalid ordered coverage counts')
    support_passed = off_support_count == 0
    result = {'role': role, 'sample_count': sample_count, 'numerical_rank': rank,
              'off_support_count': off_support_count, 'deterministic_support_passed': support_passed,
              'maximum_nullspace_residual_metres': residual, 'point_mass': rank == 0,
              'point_mass_all_in_support': support_passed if rank == 0 else None,
              'eligible_for_gaussian_coverage_gate': role == 'linear' and rank > 0,
              'contract_sample_count': sample_count == SAMPLES_PER_CASE, 'checks': []}
    if rank:
        for probability, count in zip(COVERAGE_LEVELS, counts):
            threshold = float(chi2.ppf(probability, rank))
            interval = clopper_pearson(count, sample_count)
            within = interval['lower'] <= probability <= interval['upper']
            result['checks'].append({'nominal_probability': probability, 'chi_square_threshold': threshold,
                'interval': interval, 'observed_minus_nominal': count/sample_count-probability,
                'nominal_in_interval': within, 'gate_passed': within if role == 'linear' else None})
    result['passed'] = (result['deterministic_support_passed'] and all(row['gate_passed'] for row in result['checks'])) \
        if result['eligible_for_gaussian_coverage_gate'] else None
    result['eligible_for_contract_gate'] = role == 'linear' and rank == 3 and result['contract_sample_count']
    result['contract_gate_passed'] = result['passed'] if result['eligible_for_contract_gate'] else None
    return result


def coverage_diagnostics(points, mean, covariance, *, role):
    """Only linear Gaussian output has a gate; incomplete-N results are diagnostic."""
    samples = _points(points)
    _require(samples.ndim == 2 and len(samples) > 0, 'Coverage requires nonempty [N,3] samples')
    support = support_distance(samples, mean, covariance)
    rank = support['spectrum']['numerical_rank']
    counts = [int(np.count_nonzero(support['in_support'] & (support['squared_distance'] <= chi2.ppf(p, rank))))
              for p in COVERAGE_LEVELS] if rank else []
    return coverage_from_counts(counts, len(samples), rank, int(np.count_nonzero(~support['in_support'])),
                                float(support['nullspace_residual_metres'].max()), role=role)


def sample_diagnostics(nominal, intrinsic, covariance, standard_deviation, standard_normals, mode='full'):
    """Transform a supplied Z, without RNG calls, clipping, jitter or filtering.

    Full-mode main sampling is positive definite only; rank-deficient algebra
    tests should supply explicit perturbations to backproject/support_distance.
    """
    normals = _points(standard_normals, 'supplied standard normal values')
    _require(normals.ndim == 2 and len(normals) >= 2, 'At least two supplied rows required')
    reference = propagate_gaussian(nominal, intrinsic, covariance, standard_deviation, mode)
    _require(reference['input']['numerical_rank'] == 3, 'Main sample transform requires positive definite input covariance')
    sigma = reference['input']['covariance']
    if mode == 'full':
        try:
            factor = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as error:
            raise ValueError('Cholesky failed; no covariance repair permitted') from error
    else:
        factor = np.diag(_array(standard_deviation, (3,), 'declared standard deviations'))
    perturbations = normals @ factor.T
    coordinates = reference['nominal']+perturbations
    linear = reference['nominal_output']+perturbations @ reference['jacobian'].T
    exact = backproject(coordinates, reference['intrinsic'])
    return {'reference': reference, 'factor': factor, 'perturbations': perturbations, 'input_coordinates': coordinates,
            'linear_points': linear, 'exact_points': exact, 'linear_statistics': batch_statistics(linear),
            'exact_statistics': batch_statistics(exact), 'nonpositive_depth_count': int(np.count_nonzero(coordinates[:, 2] <= 0)),
            'depth_above_two_metres_count': int(np.count_nonzero(coordinates[:, 2] > 2)),
            'samples_removed': 0, 'random_draw_calls': 0, 'new_model_forward_calls': 0,
            'scope': 'supplied_synthetic_values_only_no_quality_to_variance_mapping'}
