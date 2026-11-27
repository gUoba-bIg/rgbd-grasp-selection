"""Observation-only conditional mixtures for two distinct synthetic targets.

A maps an observed sample to its possible source pixel and source depth.
B maps a fixed RGB pixel to possible depth samples under a shared frame warp.
Missing B mass remains missing. Neither interface replaces observed geometry.
This numerical module has no filesystem, model-forward or reference-data access.
Only fit/statistics functions accept supervision. Same instance is not surface.
"""
import numpy as np
from scipy.ndimage import uniform_filter, median_filter, maximum_filter, minimum_filter
from scipy.optimize import minimize
from scipy.special import logsumexp, gammaln, stdtr, stdtrit

CANDIDATES = np.array(sorted([(x, y) for y in range(-4, 5) for x in range(-4, 5)],
                            key=lambda q: (q[0]**2+q[1]**2, q[1], q[0])), np.int64)
DF = 3.
LEVELS = (.5, .8, .95)
WARP_FEATURES = ('edge_product', 'signed_direction_product', 'absolute_direction_product',
                 'edge_strength_difference', 'strong_edge_overlap', 'rgb_edge_at_depth_edge',
                 'depth_edge_at_rgb_edge', 'pooled3_edge_product', 'pooled7_edge_product',
                 'inside_fraction', 'edge_product_times_dx_over4', 'edge_product_times_dy_over4', 'edge_product_times_radius2_over32',
                 'edge_product_times_noise_proxy', 'edge_product_times_valid_fraction', 'abs_direction_times_valid_fraction')
DEPTH_FEATURES = ('depth_m', 'valid_fraction3', 'valid_fraction7', 'median_deviation_mm',
                  'local_range_mm', 'global_interior_noise_proxy_mm')
SCALE_BOUNDS_M = (1e-4, .05)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def candidates_array(candidates):
    raw = np.asarray(candidates)
    require(raw.ndim == 2 and raw.shape[1] == 2 and len(raw) > 1
            and np.issubdtype(raw.dtype, np.integer), 'Candidates must be integer (dx,dy) pairs')
    require(len(np.unique(raw, axis=0)) == len(raw), 'Duplicate warp candidates')
    return raw.astype(np.int64, copy=False)


def observations(depth, valid):
    z, ok = np.asarray(depth, np.float64), np.asarray(valid)
    require(z.ndim == 2 and min(z.shape) >= 9 and ok.dtype == np.bool_ and ok.shape == z.shape,
            'Expected depth and boolean validity images at least 9x9')
    require(np.isfinite(z).all() and np.all(z[ok] > 0), 'Invalid observed depth')
    return z, ok


def fixed_feature_indices(shape, limit=2048):
    """Fixed 8px margin covers +/-4 queries plus radius3 pooling/gradients."""
    require(len(shape) == 2 and min(shape) > 16 and type(limit) is int and limit > 0,
            'Candidate features need a nonempty fixed eight-pixel interior')
    h, w = shape
    yy, xx = np.mgrid[8:h-8, 8:w-8]
    ids = (yy*w+xx).ravel()
    count = min(len(ids), limit)
    return ids[np.arange(count)*len(ids)//count]


def observable_case_features(rgb, depth, valid, candidates=CANDIDATES):
    """16 independently measured observables, not a temperature on old scores.

    Sample sites use a fixed 8px interior grid, not GT edges or warp-dependent
    masks. The margin covers +/-4 candidate queries plus radius3 pooling.
    Global conditioning statistics use that same fixed interior, never padding.
    The same denominator is retained for all candidates.
    Positive delta means source RGB pixel -> observed depth pixel + delta.
    """
    z, ok = observations(depth, valid)
    color = np.asarray(rgb, np.float64); shifts = candidates_array(candidates)
    require(color.shape == (*z.shape, 3) and np.isfinite(color).all()
            and np.all((color >= 0) & (color <= 1)), 'RGB must be finite [0,1]')
    h, w = z.shape
    require(np.all(np.abs(shifts) <= 4), 'Candidate query exceeds fixed safe border')
    ids = fixed_feature_indices(z.shape)
    yy, xx = np.divmod(ids, w)
    gray = color @ np.array([.299, .587, .114])
    ry, rx = np.gradient(gray); zy, zx = np.gradient(z)
    dense = uniform_filter(ok.astype(float), 3, mode='constant') > .999
    zy, zx = np.where(dense, zy, 0.), np.where(dense, zx, 0.)
    rm, zm = np.hypot(rx, ry), np.hypot(zx, zy)
    re, de = rm/(rm+.02), zm/(zm+.02)
    rx = np.divide(rx, rm, out=np.zeros_like(rx), where=rm > 0)
    ry = np.divide(ry, rm, out=np.zeros_like(ry), where=rm > 0)
    zx = np.divide(zx, zm, out=np.zeros_like(zx), where=zm > 0)
    zy = np.divide(zy, zm, out=np.zeros_like(zy), where=zm > 0)
    r3, d3 = uniform_filter(re, 3), uniform_filter(de, 3)
    r7, d7 = uniform_filter(re, 7), uniform_filter(de, 7)
    domain = np.zeros(z.shape, bool); domain[8:-8, 8:-8] = True
    interior = domain & dense & (zm < .02)
    med = median_filter(z, 3, mode='nearest')
    noise = min(20., float(np.median(np.abs(z[interior]-med[interior])))/.005) if interior.any() else 0.
    valid_fraction = float(ok[domain].mean())
    output = []
    for dx, dy in shifts:
        sx, sy = xx-dx, yy-dy
        inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        u, v, a, b = xx[inside], yy[inside], sx[inside], sy[inside]
        r, d = re[b, a], de[v, u]
        direction = rx[b, a]*zx[v, u]+ry[b, a]*zy[v, u]
        n = len(ids)
        product, absolute = np.sum(r*d)/n, np.sum(r*d*np.abs(direction))/n
        output.append([product, np.sum(r*d*direction)/n, absolute,
                       np.sum(np.abs(r-d))/n, np.sum((r > .5) & (d > .5))/n,
                       np.sum(r*(d > .5))/n, np.sum(d*(r > .5))/n,
                       np.sum(r3[b, a]*d3[v, u])/n, np.sum(r7[b, a]*d7[v, u])/n,
                       float(inside.mean()), product*dx/4., product*dy/4., product*(dx*dx+dy*dy)/32.,
                       product*noise, product*valid_fraction, absolute*valid_fraction])
    result = np.asarray(output, np.float64)
    require(result.shape == (len(shifts), len(WARP_FEATURES)) and np.isfinite(result).all(), 'Invalid candidate features')
    return result


def depth_features(depth, valid, flat_indices):
    """Only observed depth/validity; no supplied true noise sigma or source-id."""
    z, ok = observations(depth, valid)
    raw = np.asarray(flat_indices)
    require(raw.ndim == 1 and np.issubdtype(raw.dtype, np.integer)
            and np.all((raw >= 0) & (raw < z.size)), 'Invalid depth feature indices')
    idx = raw.astype(np.int64, copy=False)
    require(ok.ravel()[idx].all(), 'Depth features require observed samples')
    f3, f7 = (uniform_filter(ok.astype(float), size, mode='constant') for size in (3, 7))
    med = median_filter(z, 3, mode='nearest')
    spread = maximum_filter(np.where(ok, z, -np.inf), 3)-minimum_filter(np.where(ok, z, np.inf), 3)
    interior = ok & (f3 > .999) & (spread < .04)
    noise = float(np.median(np.abs(z[interior]-med[interior]))) if interior.any() else 0.
    maps = [z, f3, f7, np.clip(np.abs(z-med)*1000., 0., 100.),
            np.clip(spread*1000., 0., 200.), np.full_like(z, noise*1000.)]
    result = np.column_stack([m.ravel()[idx] for m in maps])
    require(np.isfinite(result).all(), 'Nonfinite depth features')
    return result


def normalized_weights(n, sample_weights):
    w = np.ones(n) if sample_weights is None else np.asarray(sample_weights, np.float64)
    require(w.shape == (n,) and n > 0 and np.isfinite(w).all() and np.all(w > 0), 'Invalid fit weights')
    return w/w.sum()


def normalize_features(features, center, scale):
    a = np.asarray(features, np.float64)
    c, s = np.asarray(center, np.float64), np.asarray(scale, np.float64)
    require(a.shape[-1:] == c.shape == s.shape and np.isfinite(a).all()
            and np.isfinite(c).all() and np.isfinite(s).all() and np.all(s > 0), 'Invalid feature normalization')
    return np.clip((a-c)/s, -8., 8.)


def warp_objective(parameters, x, targets, weights, ridge=1e-3):
    """Shared conditional multinomial regression plus fitted class intercepts."""
    b, k, f = x.shape
    beta, intercept = parameters[:f], parameters[f:]
    require(len(intercept) == k, 'Wrong categorical parameter count')
    logits = np.einsum('bkf,f->bk', x, beta)+intercept
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    delta = np.exp(logp); delta[np.arange(b), targets] -= 1.
    delta *= weights[:, None]
    value = -np.dot(weights, logp[np.arange(b), targets])+.5*ridge*np.dot(parameters, parameters)
    gradient = np.r_[np.einsum('bk,bkf->f', delta, x), delta.sum(axis=0)]+ridge*parameters
    return float(value), gradient


def bounded_fit(loss, initial, guard):
    trace = []
    def objective(theta):
        require(len(trace) < 120, 'Fixed objective-evaluation budget exhausted')
        guard()
        value, gradient = loss(theta)
        require(np.isfinite(value) and np.isfinite(gradient).all(), 'Nonfinite fit objective')
        trace.append(float(value))
        return value, gradient
    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      options={'maxiter': 60, 'maxfun': 100, 'maxls': 15, 'ftol': 1e-8, 'gtol': 1e-5})
    return result.x, dict(optimizer_converged=bool(result.success), optimizer_message=str(result.message),
                         optimizer_iterations=int(result.nit), objective_evaluations=len(trace),
                         objective_trace=trace, hyperparameter_search=False)


def fit_warp(features, target_indices, sample_weights=None, guard=lambda: None):
    a, target = np.asarray(features, np.float64), np.asarray(target_indices)
    require(a.ndim == 3 and a.shape[2] == len(WARP_FEATURES) and len(a) > 1 and np.isfinite(a).all(), 'Invalid warp fit features')
    require(target.shape == (len(a),) and np.issubdtype(target.dtype, np.integer)
            and np.all((target >= 0) & (target < a.shape[1])), 'Invalid supervised warp labels')
    w = normalized_weights(len(a), sample_weights)
    c, s = a.mean(axis=(0, 1)), a.std(axis=(0, 1)); s = np.where(s > 1e-8, s, 1.)
    x = normalize_features(a, c, s)
    theta, info = bounded_fit(lambda v: warp_objective(v, x, target, w), np.zeros(a.shape[2]+a.shape[1]), guard)
    return dict(family='supervised_shared_conditional_multinomial', features=list(WARP_FEATURES),
                center=c.tolist(), scale=s.tolist(), coefficients=theta[:a.shape[2]].tolist(),
                class_intercepts=theta[a.shape[2]:].tolist(), candidate_count=a.shape[1], fitted_cases=len(a), ridge=.001,
                candidate_order_requires_external_protocol_binding=True, **info)


def predict_warp(model, features):
    require(model['family'] == 'supervised_shared_conditional_multinomial' and model['features'] == list(WARP_FEATURES), 'Wrong warp model')
    x = normalize_features(features, model['center'], model['scale'])
    require(x.ndim in (2, 3) and x.shape[-2] == model['candidate_count'], 'Warp candidate count differs')
    logits = x @ np.asarray(model['coefficients'])+np.asarray(model['class_intercepts'])
    return np.exp(logits-logsumexp(logits, axis=-1, keepdims=True))


def empirical_prior(target_indices, n_candidates=81, alpha=1/81):
    y = np.asarray(target_indices)
    require(y.ndim == 1 and len(y) > 0 and np.issubdtype(y.dtype, np.integer)
            and type(n_candidates) is int and n_candidates > 1 and np.all((y >= 0) & (y < n_candidates))
            and np.isfinite(alpha) and alpha > 0, 'Invalid empirical prior labels')
    counts = np.bincount(y, minlength=n_candidates).astype(np.float64)+alpha
    return counts/counts.sum()


def depth_objective(theta, x, residual_m, weights, ridge=1e-3):
    linear = x @ theta
    low, high = np.log(SCALE_BOUNDS_M)
    logscale = np.clip(linear, low, high); scale = np.exp(logscale)
    r2 = (residual_m/scale)**2
    constant = gammaln(DF/2)-gammaln((DF+1)/2)+.5*np.log(DF*np.pi)
    loss = constant+logscale+.5*(DF+1)*np.log1p(r2/DF)
    delta = (1-(DF+1)*r2/(DF+r2))*((linear > low) & (linear < high))
    gradient = x.T @ (weights*delta); gradient[1:] += ridge*theta[1:]
    return float(weights@loss+.5*ridge*np.dot(theta[1:], theta[1:])), gradient


def fit_depth(features, same_source_residual_m, sample_weights=None, guard=lambda: None):
    a, y = np.asarray(features, np.float64), np.asarray(same_source_residual_m, np.float64)
    require(a.ndim == 2 and a.shape[1] == len(DEPTH_FEATURES) and y.shape == (len(a),)
            and len(a) > len(DEPTH_FEATURES) and np.isfinite(a).all() and np.isfinite(y).all(), 'Invalid source-depth fit')
    c, s = a.mean(axis=0), a.std(axis=0); s = np.where(s > 1e-8, s, 1.)
    x = np.column_stack((np.ones(len(a)), normalize_features(a, c, s)))
    initial = np.zeros(x.shape[1]); initial[0] = np.log(.005)
    weights = normalized_weights(len(y), sample_weights)
    theta, info = bounded_fit(lambda v: depth_objective(v, x, y, weights), initial, guard)
    return dict(family='same_source_zero_location_student_t', features=list(DEPTH_FEATURES), df=DF,
                center=c.tolist(), scale=s.tolist(), log_scale_coefficients=theta.tolist(),
                scale_bounds_m=list(SCALE_BOUNDS_M), residual_definition='observed_z minus clean_z at tracked source_id',
                location_m=0., fitted_points=len(a), calibration_multiplier=1., **info)


def predict_depth(model, features):
    require(model['family'] == 'same_source_zero_location_student_t' and model['features'] == list(DEPTH_FEATURES)
            and model['df'] == DF and model['scale_bounds_m'] == list(SCALE_BOUNDS_M)
            and model['calibration_multiplier'] == 1., 'Wrong or post-hoc modified depth model')
    x = normalize_features(features, model['center'], model['scale'])
    require(x.ndim == 2, 'Depth feature matrix must be two dimensional')
    linear = np.column_stack((np.ones(len(x)), x)) @ np.asarray(model['log_scale_coefficients'])
    return np.exp(np.clip(linear, *np.log(SCALE_BOUNDS_M)))


def probability_vector(probabilities, n_candidates):
    p = np.asarray(probabilities, np.float64)
    require(p.shape == (n_candidates,) and np.isfinite(p).all() and np.all(p >= 0)
            and np.isclose(p.sum(), 1., rtol=0., atol=1e-12), 'Invalid categorical probabilities')
    return p


def point_mixture_A(observed_uv, observed_depth, scale, probabilities, candidates=CANDIDATES):
    """Source coordinates of observed physical samples; no fixed-pixel target claim."""
    shifts = candidates_array(candidates); p = probability_vector(probabilities, len(shifts))
    uv, z, s = np.asarray(observed_uv, np.float64), np.asarray(observed_depth, np.float64), np.asarray(scale, np.float64)
    require(z.ndim == 1 and uv.shape == (len(z), 2) and s.shape == z.shape and np.isfinite(uv).all()
            and np.isfinite(z).all() and np.isfinite(s).all() and np.all(z > 0) and np.all(s > 0), 'Invalid A samples')
    means = np.empty((len(z), len(shifts), 3)); means[:, :, :2] = uv[:, None, :]-shifts
    means[:, :, 2] = z[:, None]
    return dict(interface='A_observed_sample_source', means_uvz=means, scale=np.broadcast_to(s[:, None], means.shape[:2]).copy(),
                component_valid=np.ones(means.shape[:2], bool), probabilities=p.copy(), missing_probability=np.zeros(len(z)),
                observed_uvz=np.column_stack((uv, z)), candidates=shifts.copy(), original_geometry_overwritten=False)


def target_mixture_B(depth, valid, scale_map, target_uv, probabilities, candidates=CANDIDATES):
    """Fixed RGB pixel depth/surface mixture; missing mass is never reassigned."""
    z, ok = observations(depth, valid)
    shifts = candidates_array(candidates); p = probability_vector(probabilities, len(shifts))
    uv, smap = np.asarray(target_uv), np.asarray(scale_map, np.float64)
    require(uv.ndim == 2 and uv.shape[1] == 2 and np.issubdtype(uv.dtype, np.integer)
            and np.all((uv >= 0) & (uv < [z.shape[1], z.shape[0]])), 'Invalid target coordinates')
    require(smap.shape == z.shape and np.isfinite(smap).all() and np.all(smap[ok] > 0), 'Invalid observed scale map')
    query = uv[:, None, :]+shifts[None]
    inside = np.all((query >= 0) & (query < [z.shape[1], z.shape[0]]), axis=2)
    sx, sy = np.clip(query[:, :, 0], 0, z.shape[1]-1), np.clip(query[:, :, 1], 0, z.shape[0]-1)
    good = inside & ok[sy, sx]
    means = np.zeros((*good.shape, 3)); means[:, :, :2] = uv[:, None]
    means[:, :, 2] = np.where(good, z[sy, sx], 0.)
    scale = np.where(good, smap[sy, sx], SCALE_BOUNDS_M[0])
    return dict(interface='B_fixed_rgb_target_depth', means_uvz=means, scale=scale, component_valid=good,
                probabilities=p.copy(), missing_probability=(~good)@p, candidates=shifts.copy(),
                observed_uvz=np.column_stack((uv, z[uv[:, 1], uv[:, 0]])),
                original_geometry_overwritten=False)


def validate_mixture(mixture):
    means, scale, good = (np.asarray(mixture[k]) for k in ('means_uvz', 'scale', 'component_valid'))
    require(means.ndim == 3 and means.shape[2] == 3 and scale.shape == good.shape == means.shape[:2]
            and good.dtype == np.bool_ and np.isfinite(means).all() and np.isfinite(scale).all() and np.all(scale > 0), 'Invalid mixture')
    p = probability_vector(mixture['probabilities'], means.shape[1])
    require(np.allclose(mixture['missing_probability'], (~good)@p, atol=1e-12, rtol=0), 'Missing probability was changed')
    return means, scale, good, p


def backproject_uvz(uvz, intrinsic_grid, original_size=None, grid_size=None):
    """Use original K and exact nearest-source pixels when both sizes are given.

    Sizes are (width,height). Nonintegral resize factors cannot use a scaled K:
    source_x = floor((2*u+1)*original_width/(2*grid_width)), similarly for y.
    Without sizes K describes the input grid (useful for analytic toy checks).
    """
    values, k = np.asarray(uvz, np.float64), np.asarray(intrinsic_grid, np.float64)
    require(values.shape[-1] == 3 and k.shape == (3, 3) and np.isfinite(k).all()
            and k[0, 0] > 0 and k[1, 1] > 0 and np.array_equal(k[2], [0, 0, 1])
            and k[0, 1] == k[1, 0] == 0, 'Expected finite pinhole intrinsics without skew')
    require(np.isfinite(values).all(), 'Nonfinite uvz')
    require((original_size is None) == (grid_size is None), 'Supply both original and grid sizes')
    uv = values[..., :2]
    if original_size is not None:
        original, grid = np.asarray(original_size), np.asarray(grid_size)
        require(original.shape == grid.shape == (2,) and np.issubdtype(original.dtype, np.integer)
                and np.issubdtype(grid.dtype, np.integer) and np.all(original > 0) and np.all(grid > 0)
                and np.array_equal(uv, np.rint(uv)), 'Nearest-source projection requires integer grid pixels')
        uv = ((2*uv.astype(np.int64)+1)*original)//(2*grid)
    z = values[..., 2]
    # Match frozen xyz_at: divide each pixel difference FIRST, then multiply z.
    return np.stack((((uv[..., 0]-k[0, 2])/k[0, 0])*z,
                     ((uv[..., 1]-k[1, 2])/k[1, 1])*z, z), axis=-1)


def jacobian_uvz(uvz, intrinsic_grid):
    values, k = np.asarray(uvz, np.float64), np.asarray(intrinsic_grid, np.float64)
    backproject_uvz(values, k)  # Validate the camera contract.
    out = np.zeros((*values.shape[:-1], 3, 3))
    out[..., 0, 0] = values[..., 2]/k[0, 0]; out[..., 1, 1] = values[..., 2]/k[1, 1]
    out[..., 0, 2] = (values[..., 0]-k[0, 2])/k[0, 0]
    out[..., 1, 2] = (values[..., 1]-k[1, 2])/k[1, 1]; out[..., 2, 2] = 1.
    return out


def propagate_components(mixture, intrinsic_grid, original_size=None, grid_size=None):
    means, scale, good, p = validate_mixture(mixture)
    xyz = backproject_uvz(means, intrinsic_grid, original_size, grid_size)
    unit = means.copy(); unit[..., 2] = 1.
    rays = backproject_uvz(unit, intrinsic_grid, original_size, grid_size)
    within = np.einsum('nki,nkj,nk->nkij', rays, rays, scale**2*DF/(DF-2))
    weights = good*p; mass = weights.sum(axis=1)
    conditional = np.divide(weights, mass[:, None], out=np.zeros_like(weights), where=mass[:, None] > 0)
    mean = np.einsum('nk,nki->ni', conditional, xyz)
    delta = xyz-mean[:, None]
    covariance = np.einsum('nk,nkij->nij', conditional, within+delta[..., :, None]*delta[..., None, :])
    # These descriptive moments MUST NOT replace the retained mixture support.
    return dict(component_mean_xyz=xyz, component_covariance_xyz=within,
                conditional_mean_xyz=mean, conditional_covariance_xyz=covariance, represented_probability=mass,
                conditional_moments_defined=mass > 0, moments_are_probability_region=False,
                observed_xyz=backproject_uvz(mixture['observed_uvz'], intrinsic_grid, original_size, grid_size),
                original_observed_xyz_copy=True, original_geometry_overwritten=False,
                continuous_uncertainty='depth_only_within_component', shared_discrete_uncertainty='frame_warp',
                projection='exact_nearest_source_pixel' if original_size is not None else 'input_grid_pinhole')


def sample_shared(mixture, n_draws, seed, intrinsic_grid=None, original_size=None, grid_size=None):
    means, scale, good, p = validate_mixture(mixture)
    require(type(n_draws) is int and n_draws > 0, 'Invalid draw count')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(p), size=n_draws, p=p)
    values = np.transpose(means[:, chosen, :], (1, 0, 2)).copy()
    values[:, :, 2] += rng.standard_t(DF, size=values.shape[:2])*scale[:, chosen].T
    # No clipping of Student-t tails: unphysical negative-depth draws are exposed.
    valid = good[:, chosen].T
    if intrinsic_grid is not None:
        values = backproject_uvz(values, intrinsic_grid, original_size, grid_size)
    return dict(candidate_indices=chosen, samples=values, valid=valid,
                frame_warp_shared_across_all_points=True, missing_draws=int((~valid).sum()))


def warp_statistics(probabilities, target_index, levels=LEVELS):
    p = probability_vector(probabilities, len(probabilities))
    require(type(target_index) in (int, np.int64, np.int32) and 0 <= target_index < len(p), 'Invalid warp reference')
    order = np.argsort(-p, kind='stable'); cumulative = np.cumsum(p[order])
    sets = []
    for level in levels:
        require(0 < level < 1, 'Invalid coverage level')
        count = min(len(p), int(np.searchsorted(cumulative, level, side='left'))+1)
        selected = order[:count]
        sets.append(dict(level=float(level), covered=bool(target_index in selected), size=count,
                         probability=float(p[selected].sum())))
    onehot = np.zeros(len(p)); onehot[target_index] = 1.
    return dict(nll=float(-np.log(p[target_index])) if p[target_index] > 0 else float('inf'),
                brier=float(np.sum((p-onehot)**2)), credible_sets=sets,
                entropy=float(-np.sum(p[p > 0]*np.log(p[p > 0]))))


def mixture_statistics(mixture, reference_depth, levels=LEVELS):
    """B subdensity proper NLL keeps missing mass; CDF coverage is CONDITIONAL.

    A depth-only scores are marginal diagnostics and not joint uvz coverage.
    Joint A NLL additionally needs the true categorical source/warp probability.
    """
    means, scale, good, p = validate_mixture(mixture)
    reference = np.asarray(reference_depth, np.float64)
    require(reference.shape == (len(means),) and np.isfinite(reference).all(), 'Invalid reference depths')
    weights = good*p; mass = weights.sum(axis=1)
    t = (reference[:, None]-means[:, :, 2])/scale
    logdensity = gammaln((DF+1)/2)-gammaln(DF/2)-.5*np.log(DF*np.pi)-np.log(scale)-.5*(DF+1)*np.log1p(t*t/DF)
    logweights = np.full_like(weights, -np.inf)
    np.log(weights, out=logweights, where=weights > 0)
    logpdf = logsumexp(logweights+logdensity, axis=1)
    pit = np.divide(np.sum(weights*stdtr(DF, t), axis=1), mass, out=np.full(len(mass), np.nan), where=mass > 0)
    width = np.divide(np.sum(weights*(2*stdtrit(DF, .975)*scale), axis=1), mass,
                      out=np.full(len(mass), np.nan), where=mass > 0)
    active = weights > 0
    low = np.min(np.where(active, means[:, :, 2], np.inf), axis=1)
    high = np.max(np.where(active, means[:, :, 2], -np.inf), axis=1)
    span = np.where(mass > 0, high-low, np.nan)
    covered = []
    for level in levels:
        require(0 < level < 1, 'Invalid depth coverage level')
        covered.append((mass > 0) & (pit >= (1-level)/2) & (pit <= (1+level)/2))
    return dict(logpdf_subdensity=logpdf, conditional_pit=pit, conditional_covered=np.asarray(covered).T,
                represented_probability=mass, missing_probability=1-mass, conditional_eligible=mass > 0,
                within_component_mean_width95=width, component_center_span=span,
                center_span_is_sharpness=False, conditional_coverage_is_full_coverage=False)


def hierarchical_statistics(mixture, reference_depth, target_index=None, levels=LEVELS):
    """Predeclared joint credible construction, not shortest/HPD intervals.

    Select warp mass >= sqrt(level), then each selected t interval has mass
    sqrt(level). A tests the true warp AND its source depth. B unions the
    selected observed intervals; missing components remain an explicit symbol.
    Selected missing mass is a latent-component lower bound: the shared missing
    symbol has the FULL missing probability. Overlap can also admit unselected
    continuous components, so selected latent mass is not exact B region mass.
    Numerical coverage arrays retain every requested reference target. Missing
    references cannot silently disappear from the caller's full denominator.
    """
    means, scale, good, p = validate_mixture(mixture)
    reference = np.asarray(reference_depth, np.float64)
    require(reference.shape == (len(means),) and np.isfinite(reference).all(), 'Invalid region references')
    if target_index is not None:
        require(type(target_index) in (int, np.int64, np.int32) and 0 <= target_index < len(p), 'Invalid region warp reference')
    order = np.argsort(-p, kind='stable'); cumulative = np.cumsum(p[order])
    rows = []
    for level in levels:
        require(0 < level < 1, 'Invalid joint coverage level')
        q = np.sqrt(level)
        count = min(len(p), int(np.searchsorted(cumulative, q, side='left'))+1)
        selected = order[:count]; active = good[:, selected]
        radius = stdtrit(DF, (1+q)/2)*scale[:, selected]
        lower, upper = means[:, selected, 2]-radius, means[:, selected, 2]+radius
        coverage = np.any(active & (reference[:, None] >= lower) & (reference[:, None] <= upper), axis=1)
        # Vectorized across targets; interval union does not include empty gaps.
        sort = np.argsort(np.where(active, lower, np.inf), axis=1, kind='stable')
        lo, hi = (np.take_along_axis(a, sort, axis=1) for a in (lower, upper))
        valid = np.take_along_axis(active, sort, axis=1)
        last = np.full(len(means), -np.inf); width = np.zeros(len(means)); pieces = np.zeros(len(means), np.int64)
        for j in range(count):
            exists = valid[:, j]
            pieces += exists & (lo[:, j] > last)
            increment = np.maximum(0., hi[:, j]-np.maximum(lo[:, j], last))
            width += np.where(exists, increment, 0.)
            last = np.where(exists, np.maximum(last, hi[:, j]), last)
        selected_mass = float(p[selected].sum())
        missing_mass = (~active) @ p[selected]
        observed_mass = active @ p[selected]
        span = np.max(np.where(active, upper, -np.inf), axis=1)-np.min(np.where(active, lower, np.inf), axis=1)
        span = np.where(active.any(axis=1), span, np.nan)
        row = dict(level=float(level), component_central_mass=float(q), selected_warp_indices=selected.copy(),
                   selected_warp_mass=selected_mass, selected_warp_count=count,
                   latent_joint_continuous_mass=q*observed_mass, selected_missing_probability=missing_mass,
                   full_missing_probability=np.asarray(mixture['missing_probability']).copy(),
                   selected_latent_mass_is_exact_B_region_probability=False,
                   numerical_union_covered=coverage, numerical_union_width=width, numerical_union_piece_count=pieces,
                   convex_hull_width_diagnostic=span, full_reference_count=len(reference),
                   any_observed_selected_component=active.any(axis=1), shortest_hpd_claim=False)
        if target_index is not None:
            true_valid = good[:, target_index]
            true_radius = stdtrit(DF, (1+q)/2)*scale[:, target_index]
            in_set = bool(target_index in selected)
            row.update(true_warp_selected=in_set, true_component_observed=true_valid.copy(),
                       joint_source_covered=in_set & true_valid & (np.abs(reference-means[:, target_index, 2]) <= true_radius))
        rows.append(row)
    return rows
