"""One bounded Newton continuation of the unchanged v45 convex head objective.

This is a new optimizer attempt, not a retroactive success of the old L-BFGS
attempt. The fixed prior, normalization, labels and regularizer are unchanged.
No I/O or project data are read here. Every returned objective and Hessian can
be persisted immediately, including failed/nonconverged attempts.
"""
import copy
import json
import numpy as np
from scipy.special import logsumexp
from src import p6_joint_repair_v45 as base
from src import p6_geometric_logits_v48 as original_geometry

# Unchanged geometry/candidate functions remain the frozen v48 implementation.
fit_geometry_residual = original_geometry.fit_geometry_residual
score_warp = original_geometry.score_warp
predict_residual = original_geometry.predict_residual
wrap_model = original_geometry.wrap_model
warp_statistics = original_geometry.warp_statistics
predict_warp = original_geometry.predict_warp
nested_gate = original_geometry.nested_gate
geometric_features = original_geometry.geometric_features
residual_features = original_geometry.residual_features
correction_objective = original_geometry.correction_objective
FOLDS = original_geometry.FOLDS
CANDIDATES = original_geometry.CANDIDATES

RIDGE = .001
GRADIENT_TOLERANCE = 1e-5
ARMIJO_C1 = 1e-4
SHRINK = .5
MAX_TRIALS = 15
MAX_STEPS = 60
CUMULATIVE_OBJECTIVE_CAP = 120
SOLVER = 'damped_newton_v48r1'
require = base.require


def head_hessian(parameters, normalized_features, weights, prior):
    """Exact weighted softmax feature covariance plus fixed ridge times I."""
    theta = np.asarray(parameters, float)
    x = np.asarray(normalized_features, float)
    w, pi = np.asarray(weights, float), np.asarray(prior, float)
    require(theta.shape == (16,) and np.isfinite(theta).all()
            and x.ndim == 3 and x.shape[1:] == (81, 16) and len(x) > 1
            and np.isfinite(x).all(), 'Invalid Hessian parameter/features')
    require(w.shape == (len(x),) and np.isfinite(w).all() and np.all(w >= 0)
            and np.isclose(w.sum(), 1., atol=1e-12, rtol=0.)
            and pi.shape == (81,) and np.isfinite(pi).all() and np.all(pi > 0)
            and np.isclose(pi.sum(), 1., atol=1e-12, rtol=0.), 'Invalid Hessian weights/prior')
    logits = np.einsum('bkf,f->bk', x, theta)+np.log(pi)[None]
    p = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
    mean = np.einsum('bk,bkf->bf', p, x)
    centered = x-mean[:, None]
    h = np.einsum('bk,bkf,bkg->fg', w[:, None]*p, centered, centered, optimize=True)
    h = .5*(h+h.T)+RIDGE*np.eye(16)
    require(np.isfinite(h).all(), 'Nonfinite exact Hessian')
    return h


def fit_base_head_newton(features, targets, sample_weights=None, *, seed_model=None,
                         seed_identity=None, objective_cap=120, guard=lambda: None,
                         on_evaluation=lambda record: None, on_hessian=lambda record: None,
                         on_step=lambda record: None):
    """Cold zero start, or one declared continuation from a failed v48 head.

    Warm starts require an unchanged fit normalizer/prior and a failed seed.
    The seed's objective count is charged against the same 120-call head cap.
    The new Newton iteration count is separate from the old L-BFGS count.
    """
    a, y = np.asarray(features, np.float64), np.asarray(targets)
    require(a.ndim == 3 and a.shape[1:] == (81, 16) and len(a) > 1
            and np.isfinite(a).all(), 'Invalid Newton fit features')
    require(y.shape == (len(a),) and np.issubdtype(y.dtype, np.integer)
            and np.all((y >= 0) & (y < 81)), 'Invalid Newton fit targets')
    require(type(objective_cap) is int and 1 <= objective_cap <= CUMULATIVE_OBJECTIVE_CAP,
            'Invalid new objective-evaluation budget')
    weights = base.old.normalized_weights(len(a), sample_weights)
    prior = base.empirical_prior(y, n_candidates=81, alpha=1/81)
    center, scale = a.mean(axis=(0, 1)), a.std(axis=(0, 1))
    scale = np.where(scale > 1e-8, scale, 1.)
    x = base.old.normalize_features(a, center, scale)
    initial = np.zeros(16); prior_calls = 0; old_iterations = 0
    identity = dict(kind='zero_initialization')
    if seed_model is not None:
        require(isinstance(seed_model, dict) and seed_model.get('family') == 'fixed_fit_prior_offset_shared_conditional_evidence'
                and seed_model.get('features') == list(base.WARP_FEATURES)
                and seed_model.get('candidate_count') == 81 and seed_model.get('class_bias_parameters') == 0
                and seed_model.get('ridge') == RIDGE and seed_model.get('fitted_cases') == len(a)
                and seed_model.get('prior_dirichlet_alpha') == 1/81
                and seed_model.get('optimizer_converged') is False,
                'Continuation requires the matching failed v45-family seed')
        require(seed_model.get('optimizer_solver') != SOLVER,
                'A second Newton continuation is not authorized')
        require(sample_weights is None, 'Warm continuation is bound to the original equal-case fit')
        require(np.array_equal(np.asarray(seed_model.get('center'), float), center)
                and np.array_equal(np.asarray(seed_model.get('scale'), float), scale)
                and np.array_equal(np.asarray(seed_model.get('empirical_prior'), float), prior),
                'Warm seed normalization/prior does not match this fit partition')
        initial = np.asarray(seed_model.get('coefficients'), float).copy()
        require(initial.shape == (16,) and np.isfinite(initial).all(), 'Invalid failed-head coefficients')
        prior_calls = seed_model.get('objective_evaluations')
        require(type(prior_calls) is int and 0 < prior_calls < CUMULATIVE_OBJECTIVE_CAP
                and objective_cap <= CUMULATIVE_OBJECTIVE_CAP-prior_calls,
                'Continuation exceeds the original cumulative objective cap')
        old_iterations = seed_model.get('optimizer_iterations')
        require(type(old_iterations) is int and old_iterations >= 0, 'Invalid seed iteration accounting')
        require(isinstance(seed_identity, dict) and bool(seed_identity), 'Warm seed identity is required')
        json.dumps(seed_identity, allow_nan=False)
        identity = copy.deepcopy(seed_identity)
    else:
        require(seed_identity is None, 'A cold fit cannot claim a failed-head seed identity')
    records, hessians, steps = [], [], []
    current = None; accepted_steps = 0
    step_cap = min(MAX_STEPS, objective_cap)

    def report(reason, completed):
        theta = initial if current is None else np.asarray(current['theta'])
        gradient = None if current is None else current['gradient']
        norm = None if gradient is None else float(np.max(np.abs(gradient)))
        success = bool(completed and norm is not None and norm <= GRADIENT_TOLERANCE)
        return dict(family='fixed_fit_prior_offset_shared_conditional_evidence', features=list(base.WARP_FEATURES),
            center=center.tolist(), scale=scale.tolist(), coefficients=theta.tolist(), empirical_prior=prior.tolist(),
            candidate_count=81, class_bias_parameters=0, fitted_cases=len(a), ridge=RIDGE,
            prior_dirichlet_alpha=1/81, candidate_order_requires_external_protocol_binding=True, calibration_fits=0,
            optimizer_solver=SOLVER, optimizer_converged=success, optimizer_message=reason,
            optimizer_iterations=accepted_steps, newton_steps_attempted=len(hessians),
            seed_optimizer_iterations=old_iterations, original_iteration_limit_retroactively_passed=False,
            initial_theta=initial.tolist(), seed_identity=copy.deepcopy(identity),
            training_attempt='failed_head_continuation' if seed_model is not None else 'new_cold_newton_head',
            prior_objective_evaluations=prior_calls, objective_evaluations=len(records),
            cumulative_objective_evaluations=prior_calls+len(records), cumulative_objective_cap=CUMULATIVE_OBJECTIVE_CAP,
            objective_evaluation_cap=objective_cap, objective_trace=[r['value'] for r in records],
            objective_evaluation_records=copy.deepcopy(records), hessian_evaluations=len(hessians),
            hessian_evaluation_records=copy.deepcopy(hessians), newton_trace=copy.deepcopy(steps),
            final_objective=None if current is None else current['value'],
            final_objective_index=None if current is None else current['index'], final_gradient=gradient,
            final_gradient_inf=norm, gradient_tolerance=GRADIENT_TOLERANCE,
            objective_not_increased=None if current is None else bool(current['value'] <= records[0]['value']),
            optimizer_limits=dict(max_newton_steps=step_cap, hard_objectives=objective_cap, max_trials_per_step=MAX_TRIALS,
                armijo_c1=ARMIJO_C1, shrink=SHRINK, gradient_inf=GRADIENT_TOLERANCE),
            completed=completed, automatic_retry=False, ftol_success_used=False, hyperparameter_search=False,
            additional_correction_kkt_gate_applied=False,
            solver_changed_not_training_objective=True, historical_failure_preserved=seed_model is not None)

    def objective(theta):
        require(len(records) < objective_cap, 'New objective cap reached')
        guard()
        value, gradient = base.warp_objective(theta, x, y, weights, prior)
        require(np.isfinite(value) and np.isfinite(gradient).all(), 'Nonfinite Newton objective')
        row = dict(index=len(records), theta=np.asarray(theta).tolist(), value=float(value), gradient=gradient.tolist())
        records.append(row); on_evaluation(copy.deepcopy(row))
        return row

    try:
        current = objective(initial)
        reason = 'newton_step_cap_exhausted'
        for step in range(1, step_cap+1):
            gradient = np.asarray(current['gradient'])
            if np.max(np.abs(gradient)) <= GRADIENT_TOLERANCE:
                reason = 'gradient_tolerance_reached'; break
            if len(records) >= objective_cap:
                reason = 'objective_cap_exhausted'; break
            guard()
            h = head_hessian(current['theta'], x, weights, prior)
            hrow = dict(index=len(hessians), theta=list(current['theta']), objective_index=current['index'],
                        hessian=h.tolist(), gradient_inf=float(np.max(np.abs(gradient))))
            hessians.append(hrow); on_hessian(copy.deepcopy(hrow))
            direction = np.linalg.solve(h, -gradient)
            descent = float(gradient@direction)
            require(np.isfinite(direction).all() and np.isfinite(descent) and descent < 0,
                    'Exact Hessian failed to produce a finite descent direction')
            accepted = False
            for trial in range(1, MAX_TRIALS+1):
                if len(records) >= objective_cap:
                    reason = 'objective_cap_exhausted'; break
                alpha = SHRINK**(trial-1)
                candidate = objective(np.asarray(current['theta'])+alpha*direction)
                rhs = current['value']+ARMIJO_C1*alpha*descent
                accepted = bool(candidate['value'] <= rhs)
                row = dict(step=step, trial=trial, origin_objective_index=current['index'],
                    hessian_index=hrow['index'], candidate_objective_index=candidate['index'], alpha=alpha,
                    direction=direction.tolist(), directional_derivative=descent, armijo_rhs=float(rhs),
                    candidate_value=candidate['value'], accepted=accepted)
                steps.append(row); on_step(copy.deepcopy(row))
                if accepted:
                    current = candidate; accepted_steps += 1; break
            if not accepted:
                if reason != 'objective_cap_exhausted': reason = 'armijo_trial_cap_exhausted'
                break
        if np.max(np.abs(current['gradient'])) <= GRADIENT_TOLERANCE:
            reason = 'gradient_tolerance_reached'
        elif len(records) >= objective_cap:
            reason = 'objective_cap_exhausted'
        return report(reason, True)
    except BaseException as error:
        error.partial_report = report('interrupted_or_failed', False)
        raise
