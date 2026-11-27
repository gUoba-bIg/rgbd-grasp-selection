"""Observable global integer RGB/depth alignment; no neural model or I/O.

The predictor never sees a reference image, warp, labels, scene, source-id or
camera ground truth. Source-id/GT utilities below are strictly evaluation.
"""
import numpy as np
from scipy.ndimage import uniform_filter, binary_dilation

WARPS = [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (3, -3), (-3, 3), (-3, -3)]
CANDIDATES = np.array(sorted([(x, y) for y in range(-4, 5) for x in range(-4, 5)], key=lambda q: (q[0]**2 + q[1]**2, q[1], q[0])), dtype=np.int64)


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def spaced(mask, limit):
    indices = np.flatnonzero(mask)
    n = min(len(indices), limit)
    return indices[np.arange(n) * len(indices) // n] if n else np.empty(0, np.int64)


def warp_integer(depth, valid, delta):
    """Forward source pixel (u,v) -> observation (u+dx,v+dy), no interpolation."""
    z, ok = np.asarray(depth), np.asarray(valid, dtype=bool)
    need(z.shape == ok.shape and z.ndim == 2, 'Invalid depth/validity shapes')
    dx, dy = map(int, delta)
    h, w = z.shape
    out = np.zeros_like(z)
    observed_valid = np.zeros_like(ok)
    source_id = np.full(z.shape, -1, np.int64)
    xa, xb = max(0, -dx), min(w, w - dx)
    ya, yb = max(0, -dy), min(h, h - dy)
    if xb > xa and yb > ya:
        destination = np.s_[ya+dy:yb+dy, xa+dx:xb+dx]
        origin = np.s_[ya:yb, xa:xb]
        out[destination] = z[origin]
        observed_valid[destination] = ok[origin]
        source_id[destination] = np.arange(z.size).reshape(z.shape)[origin]
    out[~observed_valid] = 0
    source_id[~observed_valid] = -1
    return out, observed_valid, source_id


def edge_arrays(rgb, observed_depth, observed_valid):
    color = np.asarray(rgb, np.float64)
    z = np.asarray(observed_depth, np.float64)
    ok = np.asarray(observed_valid, bool)
    need(color.shape == (*z.shape, 3) and ok.shape == z.shape and min(z.shape) >= 9, 'Invalid observed images')
    need(np.isfinite(color).all() and np.isfinite(z).all() and np.all((color >= 0) & (color <= 1)), 'Nonfinite or invalid RGB units')
    gray = color @ np.array([.299, .587, .114])
    ry, rx = np.gradient(gray)
    zy, zx = np.gradient(z)
    rm, zm = np.hypot(rx, ry), np.hypot(zx, zy)
    dense = uniform_filter(ok.astype(float), 3, mode='constant', cval=0) > .999
    # Fixed observable support excludes artificial warp padding, not oracle support.
    dense[:8] = dense[-8:] = False
    dense[:, :8] = dense[:, -8:] = False
    selected = spaced(dense & (zm > .01), 4096)
    rx = np.divide(rx, rm, out=np.zeros_like(rx), where=rm > 0)
    ry = np.divide(ry, rm, out=np.zeros_like(ry), where=rm > 0)
    zx = np.divide(zx, zm, out=np.zeros_like(zx), where=zm > 0)
    zy = np.divide(zy, zm, out=np.zeros_like(zy), where=zm > 0)
    return selected, rx, ry, rm / (rm + .02), zx, zy, zm / (zm + .02)


def estimate(rgb, observed_depth, observed_valid):
    """Fixed edge-direction score; all candidates share one selected denominator."""
    indices, rx, ry, rw, zx, zy, zw = edge_arrays(rgb, observed_depth, observed_valid)
    h, w = observed_depth.shape
    v, u = np.divmod(indices, w)
    scores = np.zeros(len(CANDIDATES), np.float64)
    for j, (dx, dy) in enumerate(CANDIDATES):
        xx, yy = u - dx, v - dy
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        # Out-of-image hypotheses contribute zero, with the SAME denominator.
        if inside.any():
            i = indices[inside]
            orientation = np.abs(zx.ravel()[i] * rx[yy[inside], xx[inside]] + zy.ravel()[i] * ry[yy[inside], xx[inside]])
            scores[j] = np.sum(zw.ravel()[i] * rw[yy[inside], xx[inside]] * orientation) / max(1, len(indices))
    best = int(np.argmax(scores))
    ordered = np.sort(scores)
    margin = float(ordered[-1] - ordered[-2])
    usable = len(indices) >= 64 and margin > 1e-12
    raw = CANDIDATES[best].copy() if usable else np.zeros(2, np.int64)
    return dict(scores=scores, raw_uv=raw, edge_indices=indices, usable=bool(usable),
                edge_count=len(indices), peak_margin=margin, best_score=float(scores[best]),
                rejection='none' if usable else ('fewer_than64_edges' if len(indices) < 64 else 'nonunique_peak'))


def ridge_fit(raw_uv, targets_uv):
    raw, target = np.asarray(raw_uv, np.float64), np.asarray(targets_uv, np.float64)
    need(raw.shape == target.shape and raw.ndim == 2 and raw.shape[1] == 2 and len(raw) > 3, 'Invalid fit data')
    design = np.column_stack((raw, np.ones(len(raw))))
    penalty = np.diag([1., 1., 0.])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return dict(family='six_parameter_affine_ridge_from_observable_edge_shift', coefficients=coefficients.tolist(),
                ridge=1., intercept_penalty=0., input_features=['raw_dx_px', 'raw_dy_px', 'constant_one'],
                fitted_cases=len(raw), hyperparameter_search=False,
                output_rule='clip continuous uv to[-4,4], numpy.rint nearest integer with half-to-even; rejected observation returns0')


def predict(model, raw_uv, usable):
    raw = np.asarray(raw_uv, np.float64)
    valid = np.asarray(usable, bool)
    continuous = np.column_stack((raw, np.ones(len(raw)))) @ np.asarray(model['coefficients'])
    applied = np.rint(np.clip(continuous, -4, 4)).astype(np.int64)
    applied[~valid] = 0
    return continuous, applied


def select_reference(depth, valid, eligible, labels):
    """Evaluation-only pixels chosen independently of any estimated alignment."""
    h, w = depth.shape
    domain = np.asarray(eligible, bool).copy()
    domain[:8] = domain[-8:] = False
    domain[:, :8] = domain[:, -8:] = False
    boundary = np.zeros((h, w), bool)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        difference = labels[tuple(a)] != labels[tuple(b)]
        boundary[tuple(a)] |= difference
        boundary[tuple(b)] |= difference
    boundary = binary_dilation(boundary, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], bool))
    main = spaced(domain, 4096)
    extra = spaced(domain & boundary, 1024)
    indices = np.union1d(main, extra)
    need(len(main) > 0 and len(extra) > 0, 'Empty fixed main or GT-boundary evaluation stratum')
    return dict(indices=indices, main=np.isin(indices, main), boundary=boundary.ravel()[indices],
                depth=depth.ravel()[indices], label=labels.ravel()[indices], fixed_domain_count=int(domain.sum()),
                excluded_valid_from_fixed8px_border=int(np.sum(eligible) - domain.sum()))


def xyz_at(uv, depth, intrinsic, original_size, resized_size):
    uv = np.asarray(uv, np.int64)
    width, height = original_size
    w, h = resized_size
    ox = ((2 * uv[:, 0] + 1) * width) // (2 * w)
    oy = ((2 * uv[:, 1] + 1) * height) // (2 * h)
    rays = np.column_stack(((ox - intrinsic[0,2]) / intrinsic[0,0], (oy - intrinsic[1,2]) / intrinsic[1,1], np.ones(len(uv))))
    return depth[:, None] * rays


def evaluate_case(frame, reference, delta, applied_uv, usable, source_id):
    """GT and warp are used only here: tracking, supervision, and evaluation."""
    h, w = frame['depth'].shape
    idx = reference['indices']
    vv, uu = np.divmod(idx, w)
    source_uv = np.column_stack((uu, vv))
    observed_uv = source_uv + np.asarray(delta)
    corrected_uv = observed_uv - applied_uv
    need(np.array_equal(source_id[observed_uv[:,1], observed_uv[:,0]], idx), 'Source-id/warp direction mismatch')
    need(np.all((corrected_uv >= 0) & (corrected_uv < [w, h])), 'Corrected point left fixed evaluation domain')
    z = reference['depth']
    xyz = xyz_at(source_uv, z, frame['K'], frame['original_size'], (w,h))
    obs_xyz = xyz_at(observed_uv, z, frame['K'], frame['original_size'], (w,h))
    corrected_xyz = xyz_at(corrected_uv, z, frame['K'], frame['original_size'], (w,h))
    oracle_xyz = xyz_at(observed_uv - np.asarray(delta), z, frame['K'], frame['original_size'], (w,h))
    need(np.array_equal(oracle_xyz, xyz), 'Known-warp oracle did not exactly restore source geometry')
    original_labels = reference['label']
    raw_labels = frame['labels'][observed_uv[:,1], observed_uv[:,0]]
    corrected_labels = frame['labels'][corrected_uv[:,1], corrected_uv[:,0]]
    raw_valid = frame['valid'][observed_uv[:,1], observed_uv[:,0]]
    raw_reference_depth = frame['depth'][observed_uv[:,1], observed_uv[:,0]]
    cross_instance = raw_labels != original_labels
    depth_jump = raw_valid & (np.abs(raw_reference_depth - z) > .02)
    groups = dict(main=reference['main'], boundary=reference['boundary'],
                  unregistered_same_instance=raw_valid & ~cross_instance,
                  unregistered_cross_instance=cross_instance,
                  unregistered_depth_jump_proxy=depth_jump,
                  unregistered_reference_missing=~raw_valid)
    metrics = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        metrics[name] = dict(n=n, baseline_instance_identity_matches=int(np.sum((raw_labels == original_labels) & mask)),
            estimated_instance_identity_matches=int(np.sum((corrected_labels == original_labels) & mask)) if usable else 0,
            baseline_squared_error3d_sum=float(np.sum((obs_xyz[mask] - xyz[mask]) ** 2)),
            estimated_squared_error3d_sum=float(np.sum((corrected_xyz[mask] - xyz[mask]) ** 2)))
    return dict(error_uv_px=(np.asarray(applied_uv) - delta).tolist(),
        euclidean_error_px=float(np.linalg.norm(np.asarray(applied_uv) - delta)),
        no_correction_error_px=float(np.linalg.norm(delta)), usable=bool(usable),
        same_source_correspondence_exact=bool(usable and np.array_equal(applied_uv, delta)),
        metrics=metrics, oracle_source_identity_exact=True, oracle_error_px=0.,
        original_xyz_overwritten=False, uncertainty_scope='global integer image translation only; no depth-noise/fulluvz/collision calibration')


def summarize(rows):
    scenes = sorted({r['scene'] for r in rows})
    need(len(rows) == len(scenes) * 4 * 9, 'Expected four frames and nine warps for every evaluation scene')
    nonzero = [r for r in rows if r['warp_uv'] != [0,0]]
    clean = [r for r in rows if r['warp_uv'] == [0,0]]
    scene_errors = {str(s): float(np.mean([r['euclidean_error_px'] for r in nonzero if r['scene'] == s])) for s in scenes}
    directions = {str(tuple(d)): float(np.mean([r['euclidean_error_px'] for r in nonzero if r['warp_uv'] == list(d)])) for d in WARPS[1:]}
    mean = float(np.mean(list(scene_errors.values())))
    baseline = float(np.mean([r['no_correction_error_px'] for r in nonzero]))
    fixed_rates, estimated_rates = [], []
    boundary_by_scene = {}
    for s in scenes:
        rr = [r for r in nonzero if r['scene'] == s]
        fixed = np.mean([r['metrics']['boundary']['baseline_instance_identity_matches'] / r['metrics']['boundary']['n'] for r in rr])
        estimated = np.mean([r['metrics']['boundary']['estimated_instance_identity_matches'] / r['metrics']['boundary']['n'] for r in rr])
        fixed_rates.append(fixed); estimated_rates.append(estimated)
        boundary_by_scene[str(s)] = dict(fixed=float(fixed), estimated=float(estimated), delta=float(estimated-fixed))
    summary = dict(nonzero_mean_error_px=mean, no_correction_mean_error_px=baseline,
        improvement_fraction=float(1 - mean / baseline), clean_mean_error_px=float(np.mean([r['euclidean_error_px'] for r in clean])),
        clean_max_error_px=float(max(r['euclidean_error_px'] for r in clean)), worst_scene_error_px=max(scene_errors.values()),
        direction_errors_px=directions, scene_errors_px=scene_errors,
        boundary_retention_fixed_vs_estimated=dict(fixed=float(np.mean(fixed_rates)), estimated=float(np.mean(estimated_rates)), by_scene=boundary_by_scene),
        rejection_count=sum(not r['usable'] for r in rows), total_cases=len(rows),
        exact_source_correspondence_case_fraction=float(np.mean([r['same_source_correspondence_exact'] for r in rows])),
        oracle_all_exact=all(r['oracle_source_identity_exact'] and r['oracle_error_px'] == 0 for r in rows))
    gates = dict(mean_error=mean <= 1.5, improvement=summary['improvement_fraction'] >= .5,
        each_direction=all(v <= 2. for v in directions.values()), worst_scene=summary['worst_scene_error_px'] <= 2.,
        clean_mean=summary['clean_mean_error_px'] <= .5, clean_max=summary['clean_max_error_px'] <= 1.,
        boundary_identity_noninferior=float(np.mean(estimated_rates)) >= float(np.mean(fixed_rates)),
        no_rejected_case=summary['rejection_count'] == 0, oracle_positive_control=summary['oracle_all_exact'])
    return summary, gates
