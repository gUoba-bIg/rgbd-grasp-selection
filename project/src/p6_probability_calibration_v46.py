"""Finite prior-anchored scalar calibration from scene-held-out predictions.

This module has no I/O, experiment dispatch, metric-gate search or model fitting.
The runner produces three genuinely held-out predictions with v45 fold models.
Only the pooled OOF categorical NLL selects the one evidence-strength parameter.
"""
import copy
import numpy as np
from scipy.special import logsumexp
from src import p6_joint_repair_v45 as base

FOLDS = ((3, 8, 12, 17), (4, 9, 14, 19), (5, 10, 16, 20))
FIT_SCENES = tuple(sorted(scene for fold in FOLDS for scene in fold))
MAX_OBJECTIVE_EVALUATIONS = 64
BRACKET_TOLERANCE = 1e-10
CANDIDATES = base.CANDIDATES
WARP_FEATURES = base.WARP_FEATURES
warp_statistics = base.warp_statistics
require = base.require


def _base_parameters(model):
    required = {'family', 'features', 'candidate_count', 'class_bias_parameters',
                'ridge', 'center', 'scale', 'coefficients', 'empirical_prior'}
    require(isinstance(model, dict) and required <= set(model), 'Incomplete frozen base model')
    require(model['family'] == 'fixed_fit_prior_offset_shared_conditional_evidence'
            and model['features'] == list(WARP_FEATURES) and model['candidate_count'] == 81
            and model['class_bias_parameters'] == 0 and model['ridge'] == .001, 'Wrong frozen base family')
    center, scale, coefficients = (np.asarray(model[k], np.float64) for k in ('center', 'scale', 'coefficients'))
    require(center.shape == scale.shape == coefficients.shape == (16,)
            and all(np.isfinite(v).all() for v in (center, scale, coefficients))
            and np.all(scale > 0), 'Invalid frozen feature/head parameters')
    prior = base.old.probability_vector(model['empirical_prior'], 81)
    require(np.all(prior > 0), 'All81 candidates must retain positive prior support')
    return center, scale, coefficients, prior


def score_warp(base_model, features):
    """Return observable evidence x_standardized@theta, without log-prior."""
    center, scale, coefficients, _ = _base_parameters(base_model)
    a = np.asarray(features, np.float64)
    require(a.ndim in (2, 3) and a.shape[-2:] == (81, 16)
            and a.size > 0 and np.isfinite(a).all(), 'Expected finite [81,16] or [N,81,16] features')
    x = base.old.normalize_features(a, center, scale)
    scores = x@coefficients
    require(np.isfinite(scores).all(), 'Nonfinite evidence scores')
    return scores


def _calibration_arrays(scores, priors, targets, sample_weights):
    s, p, y = np.asarray(scores, np.float64), np.asarray(priors, np.float64), np.asarray(targets)
    require(s.ndim == 2 and s.shape[1] == 81 and len(s) > 0 and p.shape == s.shape
            and np.isfinite(s).all() and np.isfinite(p).all() and np.all(p > 0)
            and np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-12), 'Invalid OOF evidence/prior arrays')
    require(y.shape == (len(s),) and np.issubdtype(y.dtype, np.integer)
            and np.all((y >= 0) & (y < 81)), 'Invalid OOF target indices')
    weights = np.ones(len(s)) if sample_weights is None else np.asarray(sample_weights, np.float64)
    require(weights.shape == (len(s),) and np.isfinite(weights).all() and np.all(weights > 0)
            and np.isfinite(weights.sum()) and weights.sum() > 0, 'Invalid positive calibration weights')
    weights = weights/weights.sum()
    # Row constants cannot affect a categorical distribution. Centering also
    # makes a genuinely flat evidence vector exactly flat for the tie rule.
    centered = s-s.max(axis=1, keepdims=True)
    require(np.isfinite(centered).all(), 'Evidence dynamic range overflows')
    return centered, np.log(p), y.astype(np.int64, copy=False), weights


def _objective(tau, prepared):
    require(type(tau) in (int, float, np.float64, np.float32) and np.isfinite(tau)
            and 0 <= tau <= 1, 'Evidence strength must lie in [0,1]')
    scores, log_prior, targets, weights = prepared
    logits = log_prior+float(tau)*scores
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    probability = np.exp(logp)
    require(np.isfinite(logp).all() and np.isfinite(probability).all() and np.all(probability > 0),
            'Numerical underflow/nonfinite probabilities violate81 positive support')
    target_score = scores[np.arange(len(scores)), targets]
    nll = float(-weights@logp[np.arange(len(scores)), targets])
    derivative = float(weights@(np.sum(probability*scores, axis=1)-target_score))
    require(np.isfinite(nll) and np.isfinite(derivative), 'Nonfinite calibration objective/derivative')
    return nll, derivative


def temperature_objective(tau, scores, priors, targets, sample_weights=None):
    """Pooled proper categorical NLL and its analytic first derivative."""
    return _objective(tau, _calibration_arrays(scores, priors, targets, sample_weights))


def calibrate_temperature(scores, priors, targets, sample_weights=None, guard=lambda: None):
    """One convex minimization, at most64 evaluations; no calibration-set gates.

    Priority is fixed before observing data: D(1)<=0 selects tau=1 (including
    a completely flat objective), then D(0)>=0 selects tau=0, otherwise solve
    the monotone derivative by bisection. No automatic extra fit is performed.
    """
    prepared = _calibration_arrays(scores, priors, targets, sample_weights)
    trace = []
    def evaluate(tau):
        require(len(trace) < MAX_OBJECTIVE_EVALUATIONS, 'Fixed64 objective-evaluation cap exhausted')
        guard()
        value, derivative = _objective(float(tau), prepared)
        row = dict(tau=float(tau), nll=value, derivative=derivative)
        trace.append(row)
        return row
    endpoint0, endpoint1 = evaluate(0.), evaluate(1.)
    d0, d1 = endpoint0['derivative'], endpoint1['derivative']
    require(d0 <= d1+1e-12*max(1., abs(d0), abs(d1)), 'Nonmonotone convex endpoint derivatives')
    if d1 <= 0:
        solution = endpoint1; left = right = endpoint1
        reason = 'flat_identity_tau1' if d0 == d1 == 0 else 'upper_endpoint_tau1'
    elif d0 >= 0:
        solution = endpoint0; left = right = endpoint0; reason = 'lower_endpoint_tau0'
    else:
        left, right = endpoint0, endpoint1
        while right['tau']-left['tau'] > BRACKET_TOLERANCE:
            current = evaluate((left['tau']+right['tau'])/2)
            if current['derivative'] == 0:
                left = right = current
                break
            if current['derivative'] < 0: left = current
            else: right = current
        if left['tau'] == right['tau']:
            solution = left; reason = 'exact_stationary_derivative'
        else:
            solution = evaluate((left['tau']+right['tau'])/2)
            reason = 'bracket_tolerance'
    require(solution['nll'] <= endpoint1['nll']+1e-10, 'Convex solution is worse than feasible identity')
    return dict(family='pooled_oof_prior_anchored_temperature_v46', tau=solution['tau'],
        candidate_count=81, fitted_rows=len(prepared[0]), tau_bounds=[0., 1.],
        objective='pooled_weighted_categorical_NLL', objective_evaluations=len(trace),
        objective_evaluation_cap=MAX_OBJECTIVE_EVALUATIONS, objective_trace=trace,
        derivative0=d0, derivative1=d1, bracket=[left['tau'], right['tau']],
        bracket_derivatives=[left['derivative'], right['derivative']], bracket_tolerance=BRACKET_TOLERANCE,
        solution_nll=solution['nll'], solution_derivative=solution['derivative'],
        uncalibrated_nll=endpoint1['nll'], prior_only_nll=endpoint0['nll'],
        converged=True, stopping_reason=reason, flat_tie_rule='prefer_tau1_preserve_base',
        fold_partition=[list(fold) for fold in FOLDS], calibration_set_used=False,
        evaluation_set_used=False, metric_gate_search=False, hyperparameter_search=False,
        posterior_head_refits_by_this_function=0)


def _check_calibration(report):
    require(isinstance(report, dict) and report.get('family') == 'pooled_oof_prior_anchored_temperature_v46'
            and report.get('candidate_count') == 81 and report.get('converged') is True
            and report.get('tau_bounds') == [0., 1.] and report.get('metric_gate_search') is False,
            'Wrong or incomplete temperature report')
    tau = report.get('tau')
    require(type(tau) in (float, int) and np.isfinite(tau) and 0 <= tau <= 1, 'Invalid fitted temperature')
    require(report.get('objective') == 'pooled_weighted_categorical_NLL'
            and report.get('fold_partition') == [list(fold) for fold in FOLDS]
            and report.get('calibration_set_used') is False and report.get('evaluation_set_used') is False,
            'Changed calibration-data/selection contract')
    return float(tau)


def wrap_model(base_model, calibration_report):
    _base_parameters(base_model); _check_calibration(calibration_report)
    return dict(family='frozen_v45_with_prior_anchored_temperature_v46', candidate_count=81,
        base_model=copy.deepcopy(base_model), calibration=copy.deepcopy(calibration_report),
        original_base_head_refitted=False, all81_positive_support=True)


def predict_warp(wrapper, features):
    require(isinstance(wrapper, dict) and wrapper.get('family') == 'frozen_v45_with_prior_anchored_temperature_v46'
            and wrapper.get('candidate_count') == 81 and 'base_model' in wrapper and 'calibration' in wrapper,
            'Wrong temperature wrapper')
    model = wrapper['base_model']; tau = _check_calibration(wrapper['calibration'])
    scores = score_warp(model, features); prior = _base_parameters(model)[3]
    if tau == 1.:
        # Deliberately preserve floating-point operation order at identity.
        return base.predict_warp(model, features)
    if tau == 0.:
        return np.broadcast_to(prior, scores.shape).copy()
    logits = np.log(prior)+tau*scores
    probability = np.exp(logits-logsumexp(logits, axis=-1, keepdims=True))
    require(np.isfinite(probability).all() and np.all(probability > 0), 'Numeric probabilities lost positive support')
    return probability


def _scene_array(scene_ids):
    scenes = np.asarray(scene_ids)
    require(scenes.ndim == 1 and len(scenes) > 0 and np.issubdtype(scenes.dtype, np.integer)
            and tuple(sorted(set(scenes.tolist()))) == FIT_SCENES,
            'OOF partition requires exactly the twelve frozen fit scenes')
    return scenes


def fit_fold_indices(scene_ids, fold_index):
    """Disjoint train/held-out row indices, with whole scenes kept together."""
    scenes = _scene_array(scene_ids)
    require(type(fold_index) in (int, np.int32, np.int64) and 0 <= fold_index < 3, 'Invalid fixed fold index')
    held = np.isin(scenes, FOLDS[int(fold_index)])
    train_indices, held_indices = np.flatnonzero(~held), np.flatnonzero(held)
    require(len(set(scenes[train_indices].tolist())) == 8 and len(set(scenes[held_indices].tolist())) == 4,
            'Fixed8/4 scene fold differs')
    return train_indices, held_indices


def validate_oof_coverage(scene_ids, assigned_fold_ids, visit_counts, expected_cases_per_scene=None):
    """Check recorded row assignments and explicit visit counts, not model fit provenance.

    The runner must separately bind each temporary model's training indices,
    fold-specific normalization and prior. This function cannot prove those.
    """
    scenes = _scene_array(scene_ids); assigned = np.asarray(assigned_fold_ids); visits = np.asarray(visit_counts)
    require(assigned.shape == visits.shape == scenes.shape and np.issubdtype(assigned.dtype, np.integer)
            and np.issubdtype(visits.dtype, np.integer) and np.all(visits == 1),
            'Every OOF row must have exactly one explicit held-out prediction')
    expected = np.array([next(i for i, fold in enumerate(FOLDS) if int(scene) in fold) for scene in scenes])
    require(np.array_equal(assigned, expected), 'OOF row assigned to a training-seen or wrong fold')
    counts = {str(scene): int(np.sum(scenes == scene)) for scene in FIT_SCENES}
    if expected_cases_per_scene is not None:
        require(type(expected_cases_per_scene) is int and expected_cases_per_scene > 0
                and all(n == expected_cases_per_scene for n in counts.values()), 'Per-scene OOF case counts differ')
    return dict(passed=True, rows=len(scenes), fold_partition=[list(fold) for fold in FOLDS],
                heldout_rows_per_fold=[int(np.sum(expected == i)) for i in range(3)],
                cases_per_scene=counts, explicit_visit_counts_checked=True,
                model_training_provenance_checked=False)
