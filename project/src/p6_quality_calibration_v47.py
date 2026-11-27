"""Fixed observable-quality evidence screen and one convex conditional repair.

No I/O or experiment dispatch. Evidence is descriptive OOF screening, not a
causal finding or independent calibration validation. Unknown validity retains
the frozen global temperature and its contribution to the full NLL denominator.
"""
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from src import p6_probability_calibration_v46 as previous

RIDGE = .001
MAX_EVALUATIONS = 120
KKT_TOLERANCE = 1e-5
MIN_BIN_CASES = 108
MIN_BIN_SCENES = 3
MIN_CONTRAST = .02
require = previous.require
warp_statistics = previous.warp_statistics


def _tau(value):
    require(type(value) in (int, float, np.float64) and np.isfinite(value) and 0 <= value <= 1,
            'Frozen tau46 must be finite in [0,1]')
    return float(value)


def _scores_priors(scores, priors):
    s, p = np.asarray(scores, float), np.asarray(priors, float)
    require(s.ndim == 2 and s.shape[1] == 81 and len(s) > 0 and p.shape == s.shape
            and np.isfinite(s).all() and np.isfinite(p).all() and np.all(p > 0)
            and np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-12), 'Invalid evidence/prior matrices')
    return s, p


def _posterior(scores, priors, tau):
    logits = np.log(priors)+np.asarray(tau)[..., None]*scores
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(logp)
    require(np.isfinite(logp).all() and np.all(p > 0), 'Nonfinite or underflowed81-candidate posterior')
    return p, logp


def quality_observables(features, scores, priors):
    """Recover fixed-interior valid fraction and uncalibrated posterior entropy.

    v=0 on unknown rows is only a storage placeholder. Such rows never use Z
    to choose temperature. No padding count, source-ID, label or mode is used.
    """
    s, p = _scores_priors(scores, priors)
    x = np.asarray(features, float)
    require(x.shape == (len(s), 81, 16) and np.isfinite(x).all(), 'Invalid cached observable feature tensor')
    product, conditioned = x[:, :, 0], x[:, :, 14]
    require(np.all(product >= 0) and np.all(conditioned >= 0), 'Negative physical edge/valid product')
    largest = np.argmax(product, axis=1)
    denominator = product[np.arange(len(s)), largest]
    known = denominator > 0
    v = np.zeros(len(s))
    v[known] = conditioned[np.arange(len(s))[known], largest[known]]/denominator[known]
    require(np.isfinite(v).all() and np.all((v >= -1e-12) & (v <= 1+1e-12)), 'Recovered validity outside[0,1]')
    v = np.clip(v, 0., 1.)
    require(np.allclose(conditioned, product*v[:, None], rtol=1e-10, atol=1e-12),
            'Candidate-wise validity ratios do not agree')
    require(np.all(conditioned[~known] == 0), 'Zero-product case cannot contain a validity product')
    uncalibrated, logp = _posterior(s, p, np.ones(len(s)))
    h = -np.sum(uncalibrated*logp, axis=1)/np.log(81.)
    require(np.isfinite(h).all() and np.all((h >= -1e-12) & (h <= 1+1e-12)), 'Invalid posterior entropy')
    h = np.clip(h, 0., 1.)
    z = np.column_stack(((1-v)*(1-h), (1-v)*h, v*(1-h), v*h))
    return dict(v=v, h=h, valid_fraction=v, ambiguity=h, known=known, Z=z)


def _targets(targets, n):
    y = np.asarray(targets)
    require(y.shape == (n,) and np.issubdtype(y.dtype, np.integer)
            and np.all((y >= 0) & (y < 81)), 'Invalid categorical targets')
    return y.astype(np.int64, copy=False)


def _level_summary(probabilities, targets, indices):
    indices = np.asarray(indices, np.int64)
    if not len(indices): return None
    values = [warp_statistics(probabilities[i], int(targets[i])) for i in indices]
    return {str(level): dict(coverage=float(np.mean([v['credible_sets'][j]['covered'] for v in values])),
                            selected_mass=float(np.mean([v['credible_sets'][j]['probability'] for v in values])),
                            size=float(np.mean([v['credible_sets'][j]['size'] for v in values])))
            for j, level in enumerate((.5, .8, .95))}


def evidence_gate(features, scores, priors, targets, scenes, assigned_folds, tau46):
    s, p = _scores_priors(scores, priors); y = _targets(targets, len(s)); tau46 = _tau(tau46)
    scenes, assigned = np.asarray(scenes), np.asarray(assigned_folds)
    previous.validate_oof_coverage(scenes, assigned, np.ones(len(s), dtype=np.int64), expected_cases_per_scene=108)
    require(len(scenes) == len(s), 'OOF quality metadata row count differs')
    q = quality_observables(features, s, p); known = q['known']
    probability, _ = _posterior(s, p, np.full(len(s), tau46))
    gradient = np.sum(probability*s, axis=1)-s[np.arange(len(s)), y]
    require(np.isfinite(gradient).all(), 'Nonfinite per-case NLL derivative')
    checks = {}
    for name in ('valid_fraction', 'ambiguity'):
        values = q[name]
        median = float(np.median(values[known])) if known.any() else None
        low = known & (values <= median) if median is not None else np.zeros(len(s), bool)
        high = known & (values > median) if median is not None else np.zeros(len(s), bool)
        folds = []
        for fold in range(3):
            groups = {}; masks = {}
            for label, selected in (('low', low), ('high', high)):
                selected = selected & (assigned == fold); masks[label] = selected
                group_scenes = sorted(set(scenes[selected].tolist()))
                groups[label] = dict(n=int(selected.sum()), scenes=group_scenes,
                    mean_nll_gradient=float(gradient[selected].mean()) if selected.any() else None)
            adequate = all(g['n'] >= MIN_BIN_CASES and len(g['scenes']) >= MIN_BIN_SCENES for g in groups.values())
            delta = (groups['high']['mean_nll_gradient']-groups['low']['mean_nll_gradient']) if all(m.any() for m in masks.values()) else None
            folds.append(dict(fold=fold, support_passed=adequate, delta=delta, **groups))
        delta = [f['delta'] for f in folds]
        defined = all(v is not None for v in delta)
        consistent = defined and (all(v > 0 for v in delta) or all(v < 0 for v in delta))
        pooled = float(np.mean(delta)) if defined else None
        passed = all(f['support_passed'] for f in folds) and consistent and pooled is not None and abs(pooled) >= MIN_CONTRAST
        scene_diagnostics = []
        for scene in previous.FIT_SCENES:
            groups = {}
            for label, selected in (('low', low), ('high', high)):
                selected = selected & (scenes == scene)
                groups[label] = dict(n=int(selected.sum()),
                    mean_nll_gradient=float(gradient[selected].mean()) if selected.any() else None)
            contrast = groups['high']['mean_nll_gradient']-groups['low']['mean_nll_gradient'] if all(g['n'] for g in groups.values()) else None
            selected_scene = scenes == scene
            scene_diagnostics.append(dict(scene=int(scene), total_rows=int(selected_scene.sum()),
                known_rows=int(np.sum(selected_scene & known)), unknown_rows=int(np.sum(selected_scene & ~known)),
                delta=contrast, **groups))
        checks[name] = dict(passed=bool(passed), median=median, median_ties='low',
            fold_contrasts=folds, same_sign_across_three_folds=bool(consistent),
            equal_fold_mean_contrast=pooled,
            scene_diagnostics_not_used_for_gate=scene_diagnostics,
            descriptive_levels_only=dict(low=_level_summary(probability, y, np.flatnonzero(low)),
                                         high=_level_summary(probability, y, np.flatnonzero(high))))
    supported = any(value['passed'] for value in checks.values())
    return dict(supported=supported, passed=supported, total_rows=len(s), known_rows=int(known.sum()),
        unknown_rows=int((~known).sum()), tau46=tau46, features=checks,
        counts_by_fold=[dict(fold=fold, total_rows=int(np.sum(assigned == fold)),
            known_rows=int(np.sum((assigned == fold) & known)), unknown_rows=int(np.sum((assigned == fold) & ~known))) for fold in range(3)],
        fold_partition=[list(v) for v in previous.FOLDS],
        thresholds=dict(min_bin_cases_per_fold=MIN_BIN_CASES, min_scenes_per_bin=MIN_BIN_SCENES,
                        absolute_equal_fold_mean_contrast=MIN_CONTRAST, any_prespecified_feature_suffices=True),
        all_row_level_diagnostics=_level_summary(probability, y, np.arange(len(s))),
        unknown_rows_retained_in_all_row_diagnostics=True, unknown_temperature='fixed_tau46',
        predictor_features_fixed_regardless_of_which_screen_passes=True,
        calibration_or_evaluation_rows_used=False, significance_test=False, causal_claim=False,
        screen_scope='Descriptive fit-OOF evidence for one predeclared compute allocation, not scientific adoption',
        oof_model_training_provenance_must_be_verified_by_runner=True)


def _objective_details(theta, scores, priors, targets, z, known, tau46):
    theta = np.asarray(theta, float)
    require(theta.shape == (4,) and np.isfinite(theta).all() and np.all((theta >= 0) & (theta <= 1)), 'Four corners must lie in[0,1]')
    s, p = _scores_priors(scores, priors); y = _targets(targets, len(s)); tau46 = _tau(tau46)
    z, known = np.asarray(z, float), np.asarray(known)
    require(z.shape == (len(s), 4) and np.isfinite(z).all() and np.all(z >= 0)
            and np.allclose(z.sum(axis=1), 1., atol=1e-12, rtol=0)
            and known.shape == (len(s),) and known.dtype == np.bool_, 'Invalid quality interpolation/known flags')
    tau = np.where(known, z@theta, tau46)
    probability, logp = _posterior(s, p, tau)
    nll = float(-np.mean(logp[np.arange(len(s)), y]))
    penalty = float(.5*RIDGE*np.sum((theta-tau46)**2))
    point_gradient = np.sum(probability*s, axis=1)-s[np.arange(len(s)), y]
    gradient = z.T@(point_gradient*known)/len(s)+RIDGE*(theta-tau46)
    require(np.isfinite(nll) and np.isfinite(penalty) and np.isfinite(gradient).all(), 'Nonfinite conditional objective')
    return dict(theta=theta.tolist(), value=nll+penalty, gradient=gradient.tolist(), nll=nll, penalty=penalty)


def conditional_objective(theta, scores, priors, targets, Z, known, tau46):
    result = _objective_details(theta, scores, priors, targets, Z, known, tau46)
    return result['value'], np.asarray(result['gradient'])


def fit_conditional_temperature(features, scores, priors, targets, tau46, guard=lambda: None,
                                on_evaluation=lambda record: None):
    tau46 = _tau(tau46); q = quality_observables(features, scores, priors)
    trace = []
    def objective(theta):
        require(len(trace) < MAX_EVALUATIONS, 'Fixed120 conditional objective-evaluation cap exhausted')
        guard()
        row = _objective_details(theta, scores, priors, targets, q['Z'], q['known'], tau46)
        trace.append(row)
        on_evaluation(copy.deepcopy(row))
        return row['value'], np.asarray(row['gradient'])
    try:
        result = minimize(objective, np.full(4, tau46), method='L-BFGS-B', jac=True, bounds=[(0., 1.)]*4,
            options=dict(maxiter=60, maxfun=100, maxls=15, ftol=1e-12, gtol=1e-8))
    except BaseException as error:
        error.partial_report = dict(family='conditional_quality_temperature_v47_partial',
            completed=False, tau46=tau46, ridge=RIDGE, objective_trace=copy.deepcopy(trace),
            returned_objective_evaluations=len(trace), automatic_retry=False,
            total_rows=len(q['known']), known_rows=int(q['known'].sum()), unknown_rows=int((~q['known']).sum()))
        raise
    matching = [r for r in trace if np.array_equal(np.asarray(r['theta']), result.x)]
    require(bool(matching), 'Optimizer final point absent from accounted objective trace')
    final = matching[-1]; gradient = np.asarray(final['gradient']); theta = np.asarray(final['theta'])
    projected = theta-np.clip(theta-gradient, 0., 1.)
    norm = float(np.max(np.abs(projected)))
    require(final['value'] <= trace[0]['value']+1e-10, 'Returned objective worse than feasible frozen global model')
    return dict(family='conditional_quality_temperature_v47', theta=final['theta'], tau46=tau46,
        ridge=RIDGE, bounds=[[0., 1.]]*4, feature_names=['interior_valid_fraction', 'uncalibrated_posterior_entropy_over_log81'],
        corner_order=['v0h0', 'v0h1', 'v1h0', 'v1h1'], objective='equal_case_OOF_NLL_plus_fixed_corner_ridge',
        objective_trace=trace, objective_evaluations=len(trace), objective_evaluation_cap=MAX_EVALUATIONS,
        final_objective=final['value'], final_gradient=final['gradient'], final_nll=final['nll'], final_penalty=final['penalty'],
        baseline_nll=trace[0]['nll'], optimizer_converged=bool(result.success), optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit), optimizer_limits=dict(maxiter=60, maxfun=100, maxls=15, ftol=1e-12, gtol=1e-8),
        kkt_projected_gradient=projected.tolist(), kkt_projected_gradient_inf=norm,
        kkt_tolerance=KKT_TOLERANCE, kkt_passed=norm <= KKT_TOLERANCE,
        total_rows=len(q['known']), known_rows=int(q['known'].sum()), unknown_rows=int((~q['known']).sum()),
        loss_denominator='all_rows_including_unknown', unknown_temperature='fixed_tau46',
        entropy_from_uncalibrated_tau1=True, feature_selection=False, metric_gate_search=False,
        hyperparameter_search=False, automatic_retry=False, evidence_precondition_requires_runner=True)


def _check_report(report):
    require(isinstance(report, dict) and report.get('family') == 'conditional_quality_temperature_v47'
            and report.get('ridge') == RIDGE and report.get('bounds') == [[0., 1.]]*4
            and report.get('optimizer_converged') is True and report.get('kkt_passed') is True,
            'Conditional optimizer/certificate not accepted')
    theta = np.asarray(report.get('theta'), float); tau46 = _tau(report.get('tau46'))
    require(theta.shape == (4,) and np.isfinite(theta).all() and np.all((theta >= 0) & (theta <= 1))
            and report.get('unknown_temperature') == 'fixed_tau46'
            and report.get('entropy_from_uncalibrated_tau1') is True, 'Changed conditional prediction contract')
    return theta, tau46


def wrap_model(base_model, fit_report):
    previous._base_parameters(base_model); _check_report(fit_report)
    return dict(family='frozen_v45_with_quality_conditional_temperature_v47', candidate_count=81,
        base_model=copy.deepcopy(base_model), calibration=copy.deepcopy(fit_report),
        original_base_head_refitted=False, all81_positive_support=True)


def predict_warp(wrapper, features):
    require(isinstance(wrapper, dict) and wrapper.get('family') == 'frozen_v45_with_quality_conditional_temperature_v47'
            and wrapper.get('candidate_count') == 81, 'Wrong conditional model wrapper')
    model = wrapper['base_model']; theta, tau46 = _check_report(wrapper['calibration'])
    x = np.asarray(features, float); single = x.ndim == 2
    if single: x = x[None]
    scores = previous.score_warp(model, x)
    prior = previous._base_parameters(model)[3]; priors = np.broadcast_to(prior, scores.shape)
    q = quality_observables(x, scores, priors)
    tau = np.where(q['known'], q['Z']@theta, tau46)
    probability, _ = _posterior(scores, priors, tau)
    # Preserve the exact endpoint meanings, including unknown-row fallback.
    probability[tau == 0] = prior
    if np.any(tau == 1): probability[tau == 1] = previous.base.predict_warp(model, x[tau == 1])
    return probability[0] if single else probability
