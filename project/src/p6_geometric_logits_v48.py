"""Finite geometric relative-logit correction on the frozen81 candidate grid.

The five residual features use only candidate positions and the uncorrected
observable posterior. Their quadratic terms are not claimed to be a Gaussian
covariance. No I/O, true warp, noise-mode or source geometry enters prediction.
"""
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from src import p6_joint_repair_v45 as base
from src import p6_probability_calibration_v46 as previous

RIDGE = .001
BOUND = 2.
KKT_TOLERANCE = 1e-5
MAX_EVALUATIONS = 120
FEATURE_NAMES = ('rx', 'ry', 'rx_squared', 'rx_times_ry', 'ry_squared')
CANDIDATES = base.CANDIDATES
FOLDS = previous.FOLDS
require = base.require
score_warp = previous.score_warp
warp_statistics = base.warp_statistics


def _arrays(scores, priors, targets=None):
    s, p = np.asarray(scores, float), np.asarray(priors, float)
    require(s.ndim == 2 and s.shape[1] == 81 and len(s) > 0 and p.shape == s.shape
            and np.isfinite(s).all() and np.isfinite(p).all() and np.all(p > 0)
            and np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-12), 'Invalid score/prior matrices')
    if targets is None: return s, p
    y = np.asarray(targets)
    require(y.shape == (len(s),) and np.issubdtype(y.dtype, np.integer)
            and np.all((y >= 0) & (y < 81)), 'Invalid supervised candidate indices')
    return s, p, y.astype(np.int64, copy=False)


def _log_probability(logits):
    lp = logits-logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(lp)
    require(np.isfinite(lp).all() and np.all(p > 0), 'Underflow/nonfinite probabilities violate positive81 support')
    return p, lp


def baseline_probabilities(scores, priors):
    s, p = _arrays(scores, priors)
    return _log_probability(np.log(p)+s)[0]


def geometric_features(scores, priors):
    """[N,81,5] fixed-scale features; the posterior center is not ground truth."""
    probability = baseline_probabilities(scores, priors)
    center = probability@CANDIDATES.astype(float)
    residual = (CANDIDATES[None].astype(float)-center[:, None])/8.
    rx, ry = residual[..., 0], residual[..., 1]
    phi = np.stack((rx, ry, rx*rx, rx*ry, ry*ry), axis=-1)
    require(np.isfinite(phi).all() and np.max(np.abs(phi)) <= 1.+1e-12, 'Geometric residual outside fixed support')
    return phi


residual_features = geometric_features


def corrected_probabilities(beta, scores, priors):
    s, p = _arrays(scores, priors); beta = np.asarray(beta, float)
    require(beta.shape == (5,) and np.isfinite(beta).all() and np.all(np.abs(beta) <= BOUND), 'Correction outside fixed five-dimensional box')
    baseline = np.log(p)+s
    if np.array_equal(beta, np.zeros(5)):
        return _log_probability(baseline)[0]
    phi = geometric_features(s, p)
    return _log_probability(baseline+phi@beta)[0]


def _objective_details(beta, scores, priors, targets, phi):
    beta = np.asarray(beta, float)
    require(beta.shape == (5,) and np.isfinite(beta).all() and np.all(np.abs(beta) <= BOUND), 'Invalid geometric correction parameters')
    s, p, y = _arrays(scores, priors, targets)
    require(phi.shape == (len(s), 81, 5) and np.isfinite(phi).all(), 'Invalid fixed geometric feature matrix')
    probability, logp = _log_probability(np.log(p)+s+phi@beta)
    nll = float(-np.mean(logp[np.arange(len(s)), y]))
    penalty = float(.5*RIDGE*np.dot(beta, beta))
    gradient = (np.einsum('nk,nkf->nf', probability, phi)-phi[np.arange(len(s)), y]).mean(axis=0)+RIDGE*beta
    require(np.isfinite(nll) and np.isfinite(penalty) and np.isfinite(gradient).all(), 'Nonfinite correction objective')
    return dict(beta=beta.tolist(), value=nll+penalty, gradient=gradient.tolist(), nll=nll, penalty=penalty)


def correction_objective(beta, scores, priors, targets):
    row = _objective_details(beta, scores, priors, targets, geometric_features(scores, priors))
    return row['value'], np.asarray(row['gradient'])


def fit_geometry_residual(scores, priors, targets, guard=lambda: None, on_evaluation=lambda record: None):
    s, p, y = _arrays(scores, priors, targets); phi = geometric_features(s, p); trace = []
    def objective(beta):
        require(len(trace) < MAX_EVALUATIONS, 'Fixed120 correction objective-evaluation cap exhausted')
        guard()
        row = _objective_details(beta, s, p, y, phi)
        trace.append(row); on_evaluation(copy.deepcopy(row))
        return row['value'], np.asarray(row['gradient'])
    try:
        result = minimize(objective, np.zeros(5), method='L-BFGS-B', jac=True, bounds=[(-BOUND, BOUND)]*5,
            options=dict(maxiter=60, maxfun=100, maxls=15, ftol=1e-12, gtol=1e-8))
    except BaseException as error:
        error.partial_report = dict(family='geometric_relative_logit_correction_v48_partial', completed=False,
            objective_trace=copy.deepcopy(trace), returned_objective_evaluations=len(trace), automatic_retry=False)
        raise
    matching = [row for row in trace if np.array_equal(np.asarray(row['beta']), result.x)]
    require(bool(matching), 'Final correction absent from accounted objective trace')
    final = matching[-1]; beta = np.asarray(final['beta']); gradient = np.asarray(final['gradient'])
    projected = beta-np.clip(beta-gradient, -BOUND, BOUND); norm = float(np.max(np.abs(projected)))
    require(final['value'] <= trace[0]['value']+1e-10, 'Returned correction worse than feasible zero correction')
    return dict(family='geometric_relative_logit_correction_v48', beta=final['beta'], ridge=RIDGE,
        bounds=[[-BOUND, BOUND]]*5, feature_names=list(FEATURE_NAMES), residual_divisor=8.,
        posterior_center='uncorrected_tau1_posterior_mean_candidate', feature_standardization='none_fixed_grid_scale',
        objective='equal_case_NLL_plus_fixed_zero_centered_ridge', objective_trace=trace,
        objective_evaluations=len(trace), objective_evaluation_cap=MAX_EVALUATIONS,
        final_objective=final['value'], final_gradient=final['gradient'], final_nll=final['nll'], final_penalty=final['penalty'],
        baseline_nll=trace[0]['nll'], optimizer_converged=bool(result.success), optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit), optimizer_limits=dict(maxiter=60, maxfun=100, maxls=15, ftol=1e-12, gtol=1e-8),
        kkt_projected_gradient=projected.tolist(), kkt_projected_gradient_inf=norm,
        kkt_tolerance=KKT_TOLERANCE, kkt_passed=norm <= KKT_TOLERANCE,
        fitted_rows=len(s), calibration_or_evaluation_used=False, metric_gate_search=False,
        hyperparameter_search=False, automatic_retry=False, class_specific_free_biases=0,
        covariance_matrix_claimed=False, true_geometry_input=False)


def fit_base_head(features, targets, sample_weights=None, guard=lambda: None, on_evaluation=lambda record: None):
    """Exact v45 head algorithm with an additional immutable evaluation sink.

    Reuses the frozen objective and bounded optimizer, without modifying any
    module globals. Head options remain ftol1e-8/gtol1e-5; the correction's
    stricter options and KKT criterion are not imposed on these base heads.
    """
    a, y = np.asarray(features, np.float64), np.asarray(targets)
    require(a.ndim == 3 and a.shape[1:] == (81, len(base.WARP_FEATURES)) and len(a) > 1
            and np.isfinite(a).all(), 'Invalid base-head fit features')
    require(y.shape == (len(a),) and np.issubdtype(y.dtype, np.integer)
            and np.all((y >= 0) & (y < 81)), 'Invalid base-head fit targets')
    weights = base.old.normalized_weights(len(a), sample_weights)
    prior = base.empirical_prior(y, n_candidates=81, alpha=1/81)
    center, scale = a.mean(axis=(0, 1)), a.std(axis=(0, 1))
    scale = np.where(scale > 1e-8, scale, 1.)
    x = base.old.normalize_features(a, center, scale); records = []
    def loss(theta):
        value, gradient = base.warp_objective(theta, x, y, weights, prior)
        row = dict(theta=np.asarray(theta).tolist(), value=float(value), gradient=np.asarray(gradient).tolist())
        records.append(row); on_evaluation(copy.deepcopy(row))
        return value, gradient
    try:
        theta, info = base.old.bounded_fit(loss, np.zeros(len(base.WARP_FEATURES)), guard)
    except BaseException as error:
        error.partial_report = dict(family='v45_base_head_v48_partial', completed=False,
            objective_evaluation_records=copy.deepcopy(records), returned_objective_evaluations=len(records), automatic_retry=False)
        raise
    require(len(records) == info['objective_evaluations'] and [row['value'] for row in records] == info['objective_trace'],
            'Frozen optimizer and persisted head evaluation records disagree')
    matching = [row for row in records if np.array_equal(np.asarray(row['theta']), theta)]
    require(bool(matching), 'Final base head missing from objective records')
    return dict(family='fixed_fit_prior_offset_shared_conditional_evidence', features=list(base.WARP_FEATURES),
        center=center.tolist(), scale=scale.tolist(), coefficients=theta.tolist(),
        empirical_prior=prior.tolist(), candidate_count=81, class_bias_parameters=0, fitted_cases=len(a),
        ridge=.001, prior_dirichlet_alpha=1/81, candidate_order_requires_external_protocol_binding=True,
        calibration_fits=0, objective_evaluation_records=records, final_gradient=matching[-1]['gradient'],
        optimizer_limits=dict(maxiter=60, maxfun=100, maxls=15, ftol=1e-8, gtol=1e-5, hard_objectives=120),
        additional_correction_kkt_gate_applied=False, **info)


def _check_report(report):
    require(isinstance(report, dict) and report.get('family') == 'geometric_relative_logit_correction_v48'
            and report.get('ridge') == RIDGE and report.get('bounds') == [[-BOUND, BOUND]]*5
            and report.get('feature_names') == list(FEATURE_NAMES) and report.get('residual_divisor') == 8.
            and report.get('optimizer_converged') is True and report.get('kkt_passed') is True,
            'Geometric correction fit/certificate not accepted')
    beta = np.asarray(report.get('beta'), float)
    require(beta.shape == (5,) and np.isfinite(beta).all() and np.all(np.abs(beta) <= BOUND), 'Invalid geometric correction box')
    return beta


def predict_residual(scores, priors, report):
    return corrected_probabilities(_check_report(report), scores, priors)


def wrap_model(base_model, report):
    previous._base_parameters(base_model); _check_report(report)
    return dict(family='frozen_v45_with_geometric_logits_v48', candidate_count=81,
        base_model=copy.deepcopy(base_model), calibration=copy.deepcopy(report),
        raw_tau1_base_offset=True, original_base_head_refitted=False)


def predict_warp(wrapper, features):
    require(isinstance(wrapper, dict) and wrapper.get('family') == 'frozen_v45_with_geometric_logits_v48'
            and wrapper.get('candidate_count') == 81 and wrapper.get('raw_tau1_base_offset') is True,
            'Wrong geometric model wrapper')
    model = wrapper['base_model']; beta = _check_report(wrapper['calibration'])
    x = np.asarray(features, float); single = x.ndim == 2
    if single: x = x[None]
    scores = previous.score_warp(model, x)
    if np.array_equal(beta, np.zeros(5)):
        result = base.predict_warp(model, x)
    else:
        prior = previous._base_parameters(model)[3]
        result = corrected_probabilities(beta, scores, np.broadcast_to(prior, scores.shape))
    return result[0] if single else result


def _summary(probability, targets):
    p, y = probability, targets
    chosen = p[np.arange(len(p)), y]
    return dict(n=int(len(p)), nll=float(-np.log(chosen).mean()),
                brier=float((np.sum(p*p, axis=1)-2*chosen+1).mean()))


def nested_gate(scores, priors, targets, corrected_probability, scenes, assigned_folds):
    s, p, y = _arrays(scores, priors, targets); corrected = np.asarray(corrected_probability, float)
    require(corrected.shape == s.shape and np.isfinite(corrected).all() and np.all(corrected > 0)
            and np.allclose(corrected.sum(axis=1), 1., atol=1e-12, rtol=0), 'Invalid held-out corrected probability')
    scenes, assigned = np.asarray(scenes), np.asarray(assigned_folds)
    require(len(scenes) == len(s), 'Nested-case metadata mismatch')
    previous.validate_oof_coverage(scenes, assigned, np.ones(len(s), dtype=np.int64), expected_cases_per_scene=108)
    uncorrected = baseline_probabilities(s, p)
    before, after = _summary(uncorrected, y), _summary(corrected, y)
    folds = []
    for fold in range(3):
        keep = assigned == fold
        b, a = _summary(uncorrected[keep], y[keep]), _summary(corrected[keep], y[keep])
        folds.append(dict(fold=fold, scenes=sorted(set(scenes[keep].tolist())), baseline=b, corrected=a,
                          nll_gain=b['nll']-a['nll']))
    gain = before['nll']-after['nll']
    gates = dict(pooled_nll_gain=gain >= .01, every_fold_nll_noninferior=all(v['nll_gain'] >= 0 for v in folds),
                 pooled_brier_noninferior=after['brier'] <= before['brier'])
    supported = bool(all(gates.values()))
    return dict(supported=supported, passed=supported, total_rows=len(s), baseline=before, corrected=after,
        pooled_nll_gain=gain, fold_results=folds, gates=gates,
        thresholds=dict(pooled_nll_gain_min=.01, every_fold_nll_gain_min=0., pooled_brier_delta_max=0.),
        split_role='fit_internal_nested_candidate_admission', scientific_adoption=False,
        fold_partition=[list(v) for v in FOLDS], calibration_or_evaluation_used=False,
        head_and_meta_training_provenance_must_be_verified_by_runner=True,
        four_scene_to_eight_scene_head_transfer_is_tested_not_assumed=True)
