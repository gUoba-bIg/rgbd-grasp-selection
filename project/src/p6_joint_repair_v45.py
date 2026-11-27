"""One finite joint repair: prior-offset correspondence and residual CDF shape.

All prediction inputs are observed features. Warp/noise-mode/source labels are
accepted only by supervised fit functions. Frozen v43 geometry stays untouched.
None denotes the original t3 kernel for explicitly retained fixed baselines.
"""
import numpy as np
from scipy.special import logsumexp, gammaln, stdtr, stdtrit
from src import uncertainty_mixture_v43 as old

CANDIDATES = old.CANDIDATES
WARP_FEATURES = old.WARP_FEATURES
DEPTH_FEATURES = old.DEPTH_FEATURES
LEVELS = old.LEVELS
PROBABILITY_KNOTS = (0., .10, .25, .50, .80, .95, .99)
MINIMUM_KNOT_SPACING = 1e-6
observable_case_features = old.observable_case_features
depth_features = old.depth_features
predict_depth = old.predict_depth
empirical_prior = old.empirical_prior
warp_statistics = old.warp_statistics
backproject_uvz = old.backproject_uvz
jacobian_uvz = old.jacobian_uvz
require = old.require


def warp_objective(parameters, x, targets, weights, prior, ridge=.001):
    """Proper categorical NLL with a fixed fit-prior offset, no class bias."""
    logits = np.einsum('bkf,f->bk', x, parameters)+np.log(prior)[None]
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    delta = np.exp(logp); delta[np.arange(len(x)), targets] -= 1.
    delta *= weights[:, None]
    value = -np.dot(weights, logp[np.arange(len(x)), targets])+.5*ridge*np.dot(parameters, parameters)
    gradient = np.einsum('bk,bkf->f', delta, x)+ridge*parameters
    return float(value), gradient


def fit_warp(features, target_indices, sample_weights=None, guard=lambda: None):
    a, targets = np.asarray(features, np.float64), np.asarray(target_indices)
    require(a.ndim == 3 and a.shape[1:] == (81, len(WARP_FEATURES)) and len(a) > 1 and np.isfinite(a).all(), 'Invalid fit feature tensor')
    require(targets.shape == (len(a),) and np.issubdtype(targets.dtype, np.integer)
            and np.all((targets >= 0) & (targets < 81)), 'Invalid fit warp targets')
    weights = old.normalized_weights(len(a), sample_weights)
    # Match the original count-based smoothed prior; no evaluation frequencies.
    prior = empirical_prior(targets, n_candidates=81, alpha=1/81)
    center, scale = a.mean(axis=(0, 1)), a.std(axis=(0, 1))
    scale = np.where(scale > 1e-8, scale, 1.)
    x = old.normalize_features(a, center, scale)
    theta, info = old.bounded_fit(lambda v: warp_objective(v, x, targets, weights, prior), np.zeros(len(WARP_FEATURES)), guard)
    return dict(family='fixed_fit_prior_offset_shared_conditional_evidence', features=list(WARP_FEATURES),
                center=center.tolist(), scale=scale.tolist(), coefficients=theta.tolist(),
                empirical_prior=prior.tolist(), candidate_count=81, class_bias_parameters=0, fitted_cases=len(a),
                ridge=.001, prior_dirichlet_alpha=1/81, candidate_order_requires_external_protocol_binding=True,
                calibration_fits=0, **info)


def predict_warp(model, features):
    require(model['family'] == 'fixed_fit_prior_offset_shared_conditional_evidence' and model['features'] == list(WARP_FEATURES)
            and model['candidate_count'] == 81 and model['class_bias_parameters'] == 0 and model['ridge'] == .001, 'Wrong warp model')
    x = old.normalize_features(features, model['center'], model['scale'])
    prior = old.probability_vector(model['empirical_prior'], 81)
    theta = np.asarray(model['coefficients'], float)
    require(x.ndim in (2, 3) and x.shape[-2] == 81 and theta.shape == (16,) and np.isfinite(theta).all() and np.all(prior > 0), 'Invalid shared evidence model')
    logits = x@theta+np.log(prior)
    result = np.exp(logits-logsumexp(logits, axis=-1, keepdims=True))
    require(np.isfinite(result).all() and np.all(result > 0), 'Numerical underflow violates positive candidate support; do not truncate')
    return result


def _shape_parameters(kernel):
    require(kernel['family'] == 'symmetric_piecewise_linear_absolute_cdf_exponential_tail'
            and kernel['probability_knots'] == list(PROBABILITY_KNOTS)
            and kernel['minimum_knot_spacing'] == MINIMUM_KNOT_SPACING, 'Unexpected shape family')
    radius = np.asarray(kernel['radius_knots'], np.float64)
    q = np.asarray(PROBABILITY_KNOTS)
    require(radius.shape == q.shape and radius[0] == 0 and np.isfinite(radius).all()
            and np.all(np.diff(radius) > 0), 'Shape knots must increase strictly from zero')
    slopes = np.diff(q)/np.diff(radius)
    rate = slopes[-1]/(1-q[-1])
    require(np.isfinite(rate) and rate > 0 and np.isfinite(slopes).all() and np.all(slopes > 0), 'Invalid tail or density')
    require(np.isclose(kernel['tail_rate'], rate, rtol=1e-13, atol=0.), 'Saved tail rule changed')
    return radius, q, slopes, rate


def fit_residual_shape(residual_m, scale_m, sample_weights, supervision_modes):
    """One weighted quantile construction, with no validation-selected knots.

    Nominal exact zeros are not merged into a continuous noisy-residual law.
    Supervision-mode metadata is NEVER accepted by prediction functions.
    """
    y, scale, modes = np.asarray(residual_m, float), np.asarray(scale_m, float), np.asarray(supervision_modes)
    require(y.ndim == 1 and len(y) > 7 and y.shape == scale.shape == modes.shape and np.isfinite(y).all()
            and np.isfinite(scale).all() and np.all(scale > 0), 'Invalid shape fit observations')
    require(set(modes.tolist()) == {'noise5', 'noise5_missing15'}, 'Shape fitting uses exactly the two supervised noisy modes')
    weights = old.normalized_weights(len(y), sample_weights)
    absolute = np.abs(y/scale); order = np.argsort(absolute, kind='stable')
    absolute, cumulative = absolute[order], np.cumsum(weights[order])
    radius = [0.]; raw_quantiles = [0.]; adjusted = []
    for probability in PROBABILITY_KNOTS[1:]:
        index = min(int(np.searchsorted(cumulative, probability, side='left')), len(absolute)-1)
        raw = float(absolute[index]); value = max(raw, radius[-1]+MINIMUM_KNOT_SPACING)
        raw_quantiles.append(raw); radius.append(value); adjusted.append(value != raw)
    slopes = np.diff(PROBABILITY_KNOTS)/np.diff(radius)
    kernel = dict(family='symmetric_piecewise_linear_absolute_cdf_exponential_tail',
                  probability_knots=list(PROBABILITY_KNOTS), radius_knots=radius, raw_fit_radius_quantiles=raw_quantiles,
                  minimum_knot_spacing=MINIMUM_KNOT_SPACING, spacing_adjustment_count=sum(adjusted),
                  tail_rate=float(slopes[-1]/(1-PROBABILITY_KNOTS[-1])), fitted_points=len(y),
                  fit_supervision_modes=['noise5', 'noise5_missing15'], nominal_used_for_shape_fitting=False,
                  symmetry_assumed_not_proven_conditionally=True, coefficient_optimization_calls=0,
                  calibration_fits=0, hyperparameter_search=False)
    kernel['standardized_variance'] = shape_variance(kernel)
    return kernel


def shape_logpdf(kernel, standardized_residual):
    radius, q, slopes, rate = _shape_parameters(kernel)
    u = np.asarray(standardized_residual, float)
    require(np.isfinite(u).all(), 'Nonfinite standardized residual')
    absolute = np.abs(u)
    index = np.clip(np.searchsorted(radius, absolute, side='right')-1, 0, len(slopes)-1)
    result = np.where(absolute < radius[-1], np.log(slopes[index]/2),
                      np.log((1-q[-1])*rate/2)-rate*(absolute-radius[-1]))
    return result


def shape_pdf(kernel, standardized_residual):
    return np.exp(shape_logpdf(kernel, standardized_residual))


def shape_cdf(kernel, standardized_residual):
    radius, q, slopes, rate = _shape_parameters(kernel)
    u = np.asarray(standardized_residual, float)
    require(not np.isnan(u).any(), 'NaN standardized residual')
    absolute = np.abs(u)
    index = np.clip(np.searchsorted(radius, absolute, side='right')-1, 0, len(slopes)-1)
    # Clamp exponent argument to its tail domain; no overflow from unused branch.
    magnitude_cdf = np.where(absolute < radius[-1], q[index]+slopes[index]*(absolute-radius[index]),
                             1-(1-q[-1])*np.exp(-rate*np.maximum(absolute-radius[-1], 0.)))
    return np.where(u < 0, (1-magnitude_cdf)/2, (1+magnitude_cdf)/2)


def shape_ppf(kernel, probability):
    radius, q, slopes, rate = _shape_parameters(kernel)
    p = np.asarray(probability, float)
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), 'Invalid quantile probabilities')
    magnitude_p = np.abs(2*p-1)
    index = np.clip(np.searchsorted(q, magnitude_p, side='right')-1, 0, len(slopes)-1)
    with np.errstate(divide='ignore'):
        tail = radius[-1]-np.log((1-magnitude_p)/(1-q[-1]))/rate
    magnitude = np.where(magnitude_p < q[-1], radius[index]+(magnitude_p-q[index])/slopes[index], tail)
    return np.where(p < .5, -magnitude, magnitude)


def shape_variance(kernel):
    radius, q, slopes, rate = _shape_parameters(kernel)
    left, right = radius[:-1], radius[1:]
    within = np.sum(np.diff(q)*(left*left+left*right+right*right)/3)
    tail = (1-q[-1])*(radius[-1]**2+2*radius[-1]/rate+2/rate**2)
    value = float(within+tail)
    require(np.isfinite(value) and value > 0, 'Invalid finite shape variance')
    return value


def sample_shape(kernel, size, seed):
    return shape_ppf(kernel, np.random.default_rng(seed).random(size))


def residual_logpdf(residual, scale, kernel=None):
    y, s = np.asarray(residual, float), np.asarray(scale, float)
    require(np.isfinite(y).all() and np.isfinite(s).all() and np.all(s > 0), 'Invalid residual/scale')
    if kernel is None:
        return gammaln(2)-gammaln(1.5)-.5*np.log(3*np.pi)-np.log(s)-2*np.log1p((y/s)**2/3)
    return shape_logpdf(kernel, y/s)-np.log(s)


def depth_logpdf(kernel, residual, scale):
    return residual_logpdf(residual, scale, kernel)


def residual_cdf(residual, scale, kernel=None):
    y, s = np.asarray(residual, float), np.asarray(scale, float)
    require(not np.isnan(y).any() and np.isfinite(s).all() and np.all(s > 0), 'Invalid residual/scale')
    return stdtr(3., y/s) if kernel is None else shape_cdf(kernel, y/s)


def residual_ppf(probability, scale, kernel=None):
    p, s = np.asarray(probability, float), np.asarray(scale, float)
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)) and np.isfinite(s).all() and np.all(s > 0), 'Invalid quantile/scale')
    return (stdtrit(3., p) if kernel is None else shape_ppf(kernel, p))*s


def residual_variance(scale, kernel=None):
    s = np.asarray(scale, float)
    require(np.isfinite(s).all() and np.all(s > 0), 'Invalid variance scale')
    return s*s*(3. if kernel is None else shape_variance(kernel))


def point_mixture_A(observed_uv, observed_depth, scale, probabilities, candidates=CANDIDATES, shape_model=None):
    mixture = old.point_mixture_A(observed_uv, observed_depth, scale, probabilities, candidates)
    if shape_model is not None: _shape_parameters(shape_model)
    mixture['shape_model'] = shape_model
    return mixture


def target_mixture_B(depth, valid, scale_map, target_uv, probabilities, candidates=CANDIDATES, shape_model=None):
    mixture = old.target_mixture_B(depth, valid, scale_map, target_uv, probabilities, candidates)
    if shape_model is not None: _shape_parameters(shape_model)
    mixture['shape_model'] = shape_model
    return mixture


def validate_mixture(mixture):
    require('shape_model' in mixture, 'Mixture must explicitly declare new shape or retained t3 baseline')
    if mixture['shape_model'] is not None: _shape_parameters(mixture['shape_model'])
    return old.validate_mixture(mixture)


def propagate_components(mixture, intrinsic_grid, original_size=None, grid_size=None):
    means, scale, good, p = validate_mixture(mixture)
    xyz = backproject_uvz(means, intrinsic_grid, original_size, grid_size)
    unit = means.copy(); unit[..., 2] = 1.
    rays = backproject_uvz(unit, intrinsic_grid, original_size, grid_size)
    within = np.einsum('nki,nkj,nk->nkij', rays, rays, residual_variance(scale, mixture['shape_model']))
    weights = good*p; mass = weights.sum(axis=1)
    normalized = np.divide(weights, mass[:, None], out=np.zeros_like(weights), where=mass[:, None] > 0)
    mean = np.einsum('nk,nki->ni', normalized, xyz); delta = xyz-mean[:, None]
    covariance = np.einsum('nk,nkij->nij', normalized, within+delta[..., :, None]*delta[..., None, :])
    return dict(component_mean_xyz=xyz, component_covariance_xyz=within, conditional_mean_xyz=mean,
                conditional_covariance_xyz=covariance, represented_probability=mass, conditional_moments_defined=mass > 0,
                moments_are_probability_region=False,
                observed_xyz=backproject_uvz(mixture['observed_uvz'], intrinsic_grid, original_size, grid_size),
                original_observed_xyz_copy=True, original_geometry_overwritten=False,
                continuous_uncertainty='depth_only_within_component', shared_discrete_uncertainty='frame_warp',
                standardized_depth_variance=3. if mixture['shape_model'] is None else shape_variance(mixture['shape_model']))


def sample_shared(mixture, n_draws, seed, intrinsic_grid=None, original_size=None, grid_size=None):
    means, scale, good, p = validate_mixture(mixture)
    require(type(n_draws) is int and n_draws > 0, 'Invalid shared draw count')
    rng = np.random.default_rng(seed); chosen = rng.choice(len(p), size=n_draws, p=p)
    values = np.transpose(means[:, chosen, :], (1, 0, 2)).copy()
    noise = rng.standard_t(3., size=values.shape[:2]) if mixture['shape_model'] is None else shape_ppf(mixture['shape_model'], rng.random(values.shape[:2]))
    values[:, :, 2] += noise*scale[:, chosen].T
    negative = int(np.sum(values[:, :, 2] <= 0)); valid = good[:, chosen].T
    if intrinsic_grid is not None: values = backproject_uvz(values, intrinsic_grid, original_size, grid_size)
    return dict(candidate_indices=chosen, samples=values, valid=valid, frame_warp_shared_across_all_points=True,
                missing_draws=int((~valid).sum()), nonpositive_depth_draws=negative, depth_tails_clipped=False)


def mixture_statistics(mixture, reference_depth, levels=LEVELS):
    means, scale, good, p = validate_mixture(mixture); kernel = mixture['shape_model']
    reference = np.asarray(reference_depth, float)
    require(reference.shape == (len(means),) and np.isfinite(reference).all(), 'Invalid mixture reference')
    weights = good*p; mass = weights.sum(axis=1)
    residual = reference[:, None]-means[:, :, 2]
    logweights = np.full_like(weights, -np.inf); np.log(weights, out=logweights, where=weights > 0)
    logpdf = logsumexp(logweights+residual_logpdf(residual, scale, kernel), axis=1)
    pit = np.divide(np.sum(weights*residual_cdf(residual, scale, kernel), axis=1), mass,
                    out=np.full(len(mass), np.nan), where=mass > 0)
    width = np.divide(np.sum(weights*(2*residual_ppf(.975, scale, kernel)), axis=1), mass,
                      out=np.full(len(mass), np.nan), where=mass > 0)
    active = weights > 0
    low = np.min(np.where(active, means[..., 2], np.inf), axis=1)
    high = np.max(np.where(active, means[..., 2], -np.inf), axis=1)
    span = np.where(mass > 0, high-low, np.nan)
    coverage = []
    for level in levels:
        require(0 < level < 1, 'Invalid coverage level')
        coverage.append((mass > 0) & (pit >= (1-level)/2) & (pit <= (1+level)/2))
    return dict(logpdf_subdensity=logpdf, conditional_pit=pit, conditional_covered=np.asarray(coverage).T,
                represented_probability=mass, missing_probability=1-mass, conditional_eligible=mass > 0,
                within_component_mean_width95=width, component_center_span=span,
                center_span_is_sharpness=False, conditional_coverage_is_full_coverage=False)


def hierarchical_statistics(mixture, reference_depth, target_index=None, levels=LEVELS):
    means, scale, good, p = validate_mixture(mixture); kernel = mixture['shape_model']
    reference = np.asarray(reference_depth, float)
    require(reference.shape == (len(means),) and np.isfinite(reference).all(), 'Invalid credible-region reference')
    if target_index is not None:
        require(isinstance(target_index, (int, np.integer)) and 0 <= target_index < len(p), 'Invalid source warp reference')
    order = np.argsort(-p, kind='stable'); cumulative = np.cumsum(p[order]); rows = []
    for level in levels:
        require(0 < level < 1, 'Invalid joint level')
        q = np.sqrt(level); count = min(len(p), int(np.searchsorted(cumulative, q, side='left'))+1)
        selected = order[:count]; active = good[:, selected]
        radius = residual_ppf((1+q)/2, scale[:, selected], kernel)
        lower, upper = means[:, selected, 2]-radius, means[:, selected, 2]+radius
        coverage = np.any(active & (reference[:, None] >= lower) & (reference[:, None] <= upper), axis=1)
        sort = np.argsort(np.where(active, lower, np.inf), axis=1, kind='stable')
        lo, hi = (np.take_along_axis(a, sort, axis=1) for a in (lower, upper))
        valid = np.take_along_axis(active, sort, axis=1)
        last = np.full(len(means), -np.inf); width = np.zeros(len(means)); pieces = np.zeros(len(means), np.int64)
        for j in range(count):
            exists = valid[:, j]; pieces += exists & (lo[:, j] > last)
            width += np.where(exists, np.maximum(0., hi[:, j]-np.maximum(lo[:, j], last)), 0.)
            last = np.where(exists, np.maximum(last, hi[:, j]), last)
        missing = (~active)@p[selected]; observed = active@p[selected]
        span = np.max(np.where(active, upper, -np.inf), axis=1)-np.min(np.where(active, lower, np.inf), axis=1)
        span = np.where(active.any(axis=1), span, np.nan)
        row = dict(level=float(level), component_central_mass=float(q), selected_warp_indices=selected.copy(),
                   selected_warp_mass=float(p[selected].sum()), selected_warp_count=count,
                   latent_joint_continuous_mass=q*observed, selected_missing_probability=missing,
                   full_missing_probability=np.asarray(mixture['missing_probability']).copy(),
                   selected_latent_mass_is_exact_B_region_probability=False,
                   numerical_union_covered=coverage, numerical_union_width=width, numerical_union_piece_count=pieces,
                   convex_hull_width_diagnostic=span, full_reference_count=len(reference),
                   any_observed_selected_component=active.any(axis=1), shortest_hpd_claim=False)
        if target_index is not None:
            selected_true = bool(target_index in selected); true_valid = good[:, target_index]
            true_radius = residual_ppf((1+q)/2, scale[:, target_index], kernel)
            row.update(true_warp_selected=selected_true, true_component_observed=true_valid.copy(),
                       joint_source_covered=selected_true & true_valid & (np.abs(reference-means[:, target_index, 2]) <= true_radius))
        rows.append(row)
    return rows
