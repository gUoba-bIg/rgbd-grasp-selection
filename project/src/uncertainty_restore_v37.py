"""Bounded observable-conditional depth-error model; never changes XYZ.

RGB/depth edge offsets below are observable proxies, NOT estimates of pixel
coordinate covariance. The modeled target is observed_depth-reference_depth
at the SAME frozen pixel. This module has no file or neural-network access.
"""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, median_filter, uniform_filter, shift
from scipy.optimize import minimize
from scipy.special import gammaln, stdtr, stdtrit

DF = 3.0
FEATURES = (
    'log_sigma_over_5mm', 'depth_m', 'valid_fraction3', 'noise_proxy_mm',
    'median_residual_mm', 'range7_mm', 'depth_dx_mm', 'depth_dy_mm',
    'rgb_dx', 'rgb_dy', 'depth_edge_mm', 'rgb_edge',
    'rgb_depth_offset_x_proxy_px', 'rgb_depth_offset_y_proxy_px',
    'alignment_best', 'alignment_gap', 'edge_direction_agreement',
    'depth_dx_times_offset_x', 'depth_dy_times_offset_y', 'missing_fraction7',
)
LEVELS = (.5, .8, .95)


def require(test, message):
    if not bool(test):
        raise ValueError(message)


def observable_features(depth, valid, sigma, rgb, flat_index):
    """Reference/labels/condition/scene are deliberately absent from signature."""
    z = np.asarray(depth, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    sg = np.asarray(sigma, dtype=np.float64)
    color = np.asarray(rgb, dtype=np.float64)
    idx = np.asarray(flat_index, dtype=np.int64)
    require(z.ndim == 2 and min(z.shape) >= 9, 'Expected at least 9x9 depth image')
    require(ok.shape == z.shape == sg.shape and color.shape == (*z.shape, 3), 'Image shape mismatch')
    require(np.isfinite(z).all() and np.isfinite(sg).all() and np.isfinite(color).all(), 'Nonfinite observations')
    require(np.all(sg > 0) and np.all((color >= 0) & (color <= 1)), 'Invalid sigma/RGB units')
    require(idx.ndim == 1 and np.all((idx >= 0) & (idx < z.size)), 'Invalid selected pixels')
    require(ok.ravel()[idx].all(), 'Features requested for invalid depth')
    med = median_filter(z, size=3, mode='nearest')
    hi = maximum_filter(np.where(ok, z, -np.inf), size=7, mode='nearest')
    lo = minimum_filter(np.where(ok, z, np.inf), size=7, mode='nearest')
    spread = np.where(ok, hi - lo, 0.)
    fraction = uniform_filter(ok.astype(float), size=3, mode='nearest')
    noise_region = ok & (fraction > .999) & (spread < .02)
    noise = float(np.median(np.abs(z[noise_region] - med[noise_region]))) if noise_region.any() else 0.
    gray = color @ np.array([.299, .587, .114])
    dy, dx = np.gradient(z)
    ry, rx = np.gradient(gray)
    # Missing-neighbor discontinuities remain separately observable; do not
    # misrepresent them as signed geometric surface gradients.
    dx = np.where(fraction > .999, dx, 0.)
    dy = np.where(fraction > .999, dy, 0.)
    de = np.hypot(dx, dy)
    re = np.hypot(rx, ry)
    dn = np.clip(de / (uniform_filter(de, 9, mode='nearest') + .001), 0., 5.)
    rn = np.clip(re / (uniform_filter(re, 9, mode='nearest') + .01), 0., 5.)
    offsets = [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (-3, 3), (3, -3), (3, 3)]
    scores = np.stack([uniform_filter(dn * shift(rn, (oy, ox), order=0, mode='constant', cval=0., prefilter=False), 7, mode='nearest') for ox, oy in offsets])
    # A soft correlation centroid is a feature, never a geometric correction.
    weights = np.exp(np.clip(scores - scores.max(axis=0), -40., 0.))
    weights /= weights.sum(axis=0)
    ox = np.einsum('khw,k->hw', weights, np.array([o[0] for o in offsets]))
    oy = np.einsum('khw,k->hw', weights, np.array([o[1] for o in offsets]))
    ordered = np.sort(scores, axis=0)
    agreement = np.divide(dx * rx + dy * ry, de * re, out=np.zeros_like(z), where=(de * re > 1e-12))
    maps = [np.log(sg / .005), z, fraction, np.full_like(z, noise * 1000),
            np.clip((z - med) * 1000, -500, 500), np.clip(spread * 1000, 0, 1000),
            np.clip(dx * 1000, -500, 500), np.clip(dy * 1000, -500, 500),
            rx, ry, de * 1000, re, ox, oy, ordered[-1], ordered[-1] - ordered[-2],
            agreement, np.clip(dx * ox * 1000, -1000, 1000), np.clip(dy * oy * 1000, -1000, 1000),
            1 - uniform_filter(ok.astype(float), 7, mode='nearest')]
    result = np.column_stack([a.ravel()[idx] for a in maps])
    require(result.shape == (len(idx), len(FEATURES)) and np.isfinite(result).all(), 'Invalid observable feature matrix')
    return result


def design(features, center, scale):
    x = np.asarray(features, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == len(FEATURES) and np.isfinite(x).all(), 'Invalid features')
    return np.column_stack((np.ones(len(x)), np.clip((x - center) / scale, -8., 8.)))


def objective(parameters, x, residual_mm, weights, ridge=1e-4):
    """Analytic Student-t NLL and gradient, including declared hard scale bounds."""
    p = x.shape[1]
    beta, gamma = parameters[:p], parameters[p:]
    mu = x @ beta
    linear = x @ gamma
    low, high = np.log(.1), np.log(200.)
    log_s = np.clip(linear, low, high)
    s = np.exp(log_s)
    error = residual_mm - mu
    denominator = DF * s * s + error * error
    constant = gammaln(DF / 2.) - gammaln((DF + 1.) / 2.) + .5 * np.log(DF * np.pi)
    value = np.dot(weights, constant + log_s + .5 * (DF + 1.) * np.log1p((error / s) ** 2 / DF))
    penalty = .5 * ridge * (np.dot(beta[1:], beta[1:]) / 25. + np.dot(gamma[1:], gamma[1:]))
    dmu = -(DF + 1.) * error / denominator
    dlog = 1. - (DF + 1.) * error * error / denominator
    dlog *= (linear > low) & (linear < high)
    gradient = np.r_[x.T @ (weights * dmu), x.T @ (weights * dlog)]
    gradient[1:p] += ridge * beta[1:] / 25.
    gradient[p + 1:] += ridge * gamma[1:]
    require(np.isfinite(value + penalty) and np.isfinite(gradient).all(), 'Nonfinite fit objective')
    return float(value + penalty), gradient


def fit(features, residual_m, sample_weights=None, guard=lambda: None):
    """Exactly one fixed-capacity optimizer; no model or hyperparameter search."""
    a = np.asarray(features, dtype=np.float64)
    y = np.asarray(residual_m, dtype=np.float64) * 1000.
    require(a.ndim == 2 and len(a) == len(y) and len(y) > len(FEATURES) and np.isfinite(y).all(), 'Invalid fit observations')
    center = a.mean(axis=0)
    scale = a.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    x = design(a, center, scale)
    w = np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    require(w.shape == y.shape and np.isfinite(w).all() and np.all(w > 0), 'Invalid sample weights')
    w = w / w.sum()
    initial = np.zeros(x.shape[1] * 2)
    initial[x.shape[1]] = np.log(5.)
    trace = []

    def loss(v):
        require(len(trace) < 150, 'Hard limit of150 objective evaluations reached')
        guard()
        value, gradient = objective(v, x, y, w)
        trace.append(value)
        return value, gradient

    result = minimize(loss, initial, method='L-BFGS-B', jac=True,
                      options=dict(maxiter=60, maxfun=150, maxls=15, ftol=1e-7, gtol=1e-5))
    return dict(family='observable_conditional_student_t_depth_residual',
                degrees_of_freedom=DF, features=list(FEATURES), center=center.tolist(),
                normalization_scale=scale.tolist(), location_coefficients=result.x[:x.shape[1]].tolist(),
                log_scale_coefficients=result.x[x.shape[1]:].tolist(), scale_bounds_mm=[.1, 200.],
                calibration_multiplier=1., optimizer_converged=bool(result.success),
                optimizer_message=str(result.message), optimizer_iterations=int(result.nit),
                objective_evaluations=int(result.nfev), objective_trace=trace,
                fitted_points=len(y), geometry_mean_correction=False,
                covariance_scope='rank1 conditional depth; pixels and intrinsics fixed',
                independent_confirmation=False, hyperparameter_search=False)


def predict(model, features):
    require(model['features'] == list(FEATURES) and model['degrees_of_freedom'] == DF, 'Model family differs')
    x = design(features, np.array(model['center']), np.array(model['normalization_scale']))
    mu = (x @ np.array(model['location_coefficients'])) * .001
    scale = np.exp(np.clip(x @ np.array(model['log_scale_coefficients']), np.log(.1), np.log(200.))) * .001
    scale *= float(model.get('calibration_multiplier', 1.))
    require(np.isfinite(mu).all() and np.isfinite(scale).all() and np.all(scale > 0), 'Invalid Student-t parameters')
    return dict(location=mu, scale=scale, variance=scale ** 2 * DF / (DF - 2.))


def weighted_quantile(values, weights, probability):
    v, w = np.asarray(values), np.asarray(weights)
    require(v.ndim == w.ndim == 1 and v.shape == w.shape and len(v) > 0, 'Invalid quantile arrays')
    require(np.isfinite(v).all() and np.isfinite(w).all() and np.all(w > 0) and 0 < probability < 1, 'Invalid quantile values')
    order = np.argsort(v, kind='stable')
    index = np.searchsorted(np.cumsum(w[order]), probability * w.sum(), side='left')
    return float(v[order[min(index, len(v) - 1)]])


def calibrate(model, features, residual_m, sample_weights):
    """One predeclared empirical scale fit; no conditional coverage guarantee."""
    require(float(model.get('calibration_multiplier', 1.)) == 1., 'Calibration requires the uncalibrated fit')
    p = predict(model, features)
    score = np.abs((np.asarray(residual_m) - p['location']) / p['scale'])
    multiplier = weighted_quantile(score, sample_weights, .95) / float(stdtrit(DF, .975))
    require(np.isfinite(multiplier) and multiplier > 0, 'Degenerate empirical calibration')
    updated = dict(model, calibration_multiplier=multiplier,
                   calibration_definition='scene/condition/frame equal weighted empirical .95 absolute standardized residual; noisy calibration conditions only',
                   coverage_guarantee=False)
    return updated


def distribution_statistics(prediction, residual_m):
    residual = np.asarray(residual_m, dtype=np.float64)
    mu, scale = prediction['location'], prediction['scale']
    z = (residual - mu) / scale
    logpdf = (gammaln((DF + 1) / 2) - gammaln(DF / 2) - .5 * np.log(DF * np.pi)
              - np.log(scale) - .5 * (DF + 1) * np.log1p(z * z / DF))
    return dict(pit=stdtr(DF, z), logpdf=logpdf,
                interval_width95=2 * stdtrit(DF, .975) * scale,
                interval_lower95=mu - stdtrit(DF, .975) * scale,
                interval_upper95=mu + stdtrit(DF, .975) * scale)


def propagate_depth_only(prediction, observed_depth, rays):
    """Covariance about distribution mean, while nominal XYZ stays observed.

    The non-Gaussian covariance is NOT itself a 95% probability region.
    Second moment about nominal retains bias^2 for downstream conservative use.
    """
    r = np.asarray(rays, dtype=np.float64)
    z = np.asarray(observed_depth, dtype=np.float64)
    mu, variance = prediction['location'], prediction['variance']
    require(r.shape == (len(z), 3) and np.isfinite(r).all(), 'Invalid camera rays')
    return dict(observed_xyz=z[:, None] * r,
                predicted_error_mean_xyz=mu[:, None] * r,
                covariance=np.einsum('ni,nj,n->nij', r, r, variance),
                error_second_moment_about_observation=np.einsum('ni,nj,n->nij', r, r, variance + mu ** 2))


def case_metrics(sample, prediction):
    residual = sample['observed'] - sample['reference']
    d = distribution_statistics(prediction, residual)
    groups = {}
    for label, mask in [('main', sample['main']), ('boundary', sample['boundary'])]:
        n = int(np.sum(mask))
        require(n > 0, 'Empty evaluation stratum: ' + label)
        pit = d['pit'][mask]
        error = residual[mask]
        groups[label] = dict(n=n, covered=[int(np.sum((pit >= (1 - q) / 2) & (pit <= (1 + q) / 2))) for q in LEVELS],
            nll_sum=float(-d['logpdf'][mask].sum()),
            fixed5mm_nll_sum=float((.5 * (error / .005) ** 2 + np.log(.005 * np.sqrt(2 * np.pi))).sum()),
            width95_sum=float(d['interval_width95'][mask].sum()),
            squared_error3d_sum=float(np.sum((error[:, None] * sample['rays'][mask]) ** 2)),
            observed_geometry_changed=False)
    return groups
