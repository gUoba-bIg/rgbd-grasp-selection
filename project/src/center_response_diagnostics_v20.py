"""Unfrozen, deterministic production mathematics for cached center diagnostics.

No file access, model, Torch, device access, sampling, training, or reference
implementation imports. Small synthetic shapes are supported; a future runner
must enforce the actual source identities, dimensions and execution budget.
"""
import math

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

SCHEMA = 20
DRAFT_SHA256 = '859b8a1ae1317255da3567aee834ede9862820499afe5d5dbef2cbd63e708357'
QUANTILES = (0., .5, .9, .99, 1.)
HEAD_KEYS = {'foreground_probability', 'center_probability', 'offsets'}
SUM_KEYS = ('p', 'target', 'target_squared', 'squared_error', 'abs_logit_proxy', 'squared_logit_proxy')
TRACE_COUNTS = ('threshold_pixels', 'candidate_count', 'visited_count', 'distance_rejected_count',
                'selected_count', 'unvisited_count', 'zero_assignment_count', 'retained_count')
ZERO_KEYS = ('new_model_forward_calls', 'new_training_updates', 'new_backward_calls',
             'new_official_evaluations', 'new_random_draws')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _shape(shape):
    value = tuple(shape)
    require(len(value) == 2 and all(type(n) is int and n > 0 for n in value), 'Expected positive integer H,W')
    return value


def _labels(labels):
    a = np.asarray(labels)
    require(a.ndim == 2 and a.size > 0 and np.issubdtype(a.dtype, np.integer), 'Labels must be a nonempty integer grid')
    require(int(a.min()) >= 0 and int(a.max()) <= 88, 'Labels must be in 0..88')
    return a.astype(np.int64, copy=False)


def _vector(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and np.isfinite(a).all(), 'Expected finite vector')
    return a


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def zero_work_counts():
    return {key: 0 for key in ZERO_KEYS}


def validate_zero_work(counts):
    require(set(counts) == set(ZERO_KEYS), 'Wrong prohibited-work counters')
    require(all(type(v) is int and v == 0 for v in counts.values()), 'Prohibited work is not strict integer zero')


def instance_targets(labels):
    """Match the legacy NumPy float32 operations, without importing Torch."""
    labels = _labels(labels)
    height, width = labels.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    centers = np.zeros(labels.shape, np.float32)
    offsets = np.zeros((2, height, width), np.float32)
    instances = []
    for object_id in np.unique(labels):
        if object_id == 0:
            continue
        mask = labels == object_id
        cy, cx = int(np.round(yy[mask].mean())), int(np.round(xx[mask].mean()))
        gaussian = np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*3.0**2))
        centers = np.maximum(centers, gaussian)
        offsets[0, mask] = (cx-xx[mask])/width
        offsets[1, mask] = (cy-yy[mask])/height
        instances.append(dict(id=int(object_id), row=cy, col=cx, pixels=int(mask.sum()),
                              centroid_inside=bool(mask[cy, cx])))
    return dict(labels=labels.copy(), foreground=(labels > 0).astype(np.float32)[None],
                centers=centers[None], offsets=offsets, instances=instances)


def resize_labels_training(labels, shape):
    """Pillow NEAREST preprocessing, deliberately distinct from evaluation mapping."""
    labels = _labels(labels)
    height, width = _shape(shape)
    return np.asarray(Image.fromarray(labels.astype(np.uint16)).resize(
        (width, height), Image.Resampling.NEAREST), dtype=np.int64)


def integer_pixel_center_indices(source, destination):
    require(type(source) is int and source > 0 and type(destination) is int and destination > 0,
            'Invalid axis sizes')
    require(2*source*destination < np.iinfo(np.int64).max, 'Axis product exceeds int64')
    return ((2*np.arange(destination, dtype=np.int64)+1)*source)//(2*destination)


def map_grid_integer(array, shape):
    a = np.asarray(array)
    require(a.ndim == 2 and a.size > 0 and a.dtype.kind in 'biu', 'Expected integer or boolean grid')
    height, width = _shape(shape)
    rows = integer_pixel_center_indices(int(a.shape[0]), height)
    cols = integer_pixel_center_indices(int(a.shape[1]), width)
    return a[rows[:, None], cols[None, :]]


def check_heads(heads):
    require(set(heads) == HEAD_KEYS, 'Expected exactly three segmentation heads')
    require(all(isinstance(a, np.ndarray) for a in heads.values()), 'Heads must be arrays')
    foreground, centers, offsets = (heads[k] for k in ('foreground_probability', 'center_probability', 'offsets'))
    require(foreground.ndim == 2 and foreground.size > 0, 'Invalid probability grid')
    require(centers.shape == foreground.shape and offsets.shape == (2,)+foreground.shape, 'Head shapes differ')
    require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in heads.values()), 'Heads must be finite float32')
    require(np.all((foreground >= 0)&(foreground <= 1)) and np.all((centers >= 0)&(centers <= 1)), 'Invalid probability')
    return foreground.shape


def decode_trace(heads):
    """Frozen .5/.35, 9-window, distance 5, cap 64 and area 20 semantics."""
    height, width = check_heads(heads)
    foreground, centers, offsets = (heads[k] for k in ('foreground_probability', 'center_probability', 'offsets'))
    instance_map = np.zeros((height, width), np.int32)
    foreground_mask = foreground >= .5
    maxima = maximum_filter(centers, size=9, mode='constant', cval=-np.inf)
    coordinates = np.argwhere((centers == maxima)&(centers >= .35))
    order = np.argsort(-centers[coordinates[:, 0], coordinates[:, 1]], kind='stable')
    selected, visited, rejected, capped = [], 0, 0, False
    for index in order:
        visited += 1
        candidate = coordinates[index]
        if all(np.linalg.norm(candidate-other) >= 5 for other in selected):
            selected.append(candidate)
            if len(selected) == 64:
                capped = True
                break
        else:
            rejected += 1
    chosen = [dict(selected_center_index=i, row=int(point[0]), col=int(point[1]),
                   score=float(centers[tuple(point)])) for i, point in enumerate(selected)]
    yy, xx = np.nonzero(foreground_mask)
    areas = [0]*len(chosen)
    retained = []
    if selected and len(yy):
        selected_array = np.asarray(selected)
        # Preserve the original mixed int64 / float32 offset arithmetic.
        destinations = np.stack([yy+offsets[1, yy, xx]*height, xx+offsets[0, yy, xx]*width], axis=1)
        nearest = np.empty(len(yy), np.int32)
        for start in range(0, len(yy), 4096):
            distance = ((destinations[start:start+4096, None]-selected_array[None])**2).sum(axis=2)
            nearest[start:start+4096] = distance.argmin(axis=1)
        for i in range(len(chosen)):
            mask = nearest == i
            areas[i] = int(mask.sum())
            if areas[i] >= 20:
                label = len(retained)+1
                instance_map[yy[mask], xx[mask]] = label
                retained.append(dict(instance_id=label, **chosen[i]))
    trace = dict(threshold_pixels=int((centers >= .35).sum()),
                 maximum_meets_center_threshold=bool(np.any(centers >= .35)),
                 center_threshold_float32=float(np.float32(.35)), candidate_count=len(order),
                 visited_count=visited, distance_rejected_count=rejected, selected_count=len(chosen),
                 unvisited_count=len(order)-visited, cap_triggered=capped,
                 candidate_order=[[int(coordinates[i, 0]), int(coordinates[i, 1]),
                                   float(centers[tuple(coordinates[i])])] for i in order],
                 assigned_pixels=areas, zero_assignment_count=sum(n == 0 for n in areas),
                 retained_count=len(retained))
    require(visited == rejected+len(chosen), 'Invalid traversal accounting')
    return dict(instance_map=instance_map, foreground_mask=foreground_mask,
                selected_centers=chosen, retained_instances=retained, trace=trace)


def linear_quantiles(values):
    x = np.sort(_vector(values))
    result = {}
    for q in QUANTILES:
        h = (len(x)-1)*q
        lo, hi = math.floor(h), math.ceil(h)
        result[str(q)] = None if not len(x) else float(x[lo]+(x[hi]-x[lo])*(h-lo))
    return result


def value_summary(values, *, quantiles=False, retain_values=False):
    x = _vector(values)
    total = float(x.sum(dtype=np.float64))
    result = dict(count=len(x), sum=total, mean=_ratio(total, len(x)))
    if quantiles:
        result['quantiles'] = linear_quantiles(x)
    if retain_values:
        result['values'] = x.tolist()
    return result


def distance_regions(shape, centers):
    height, width = _shape(shape)
    coordinates = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    require(np.isfinite(coordinates).all(), 'Nonfinite centers')
    yy, xx = np.mgrid[:height, :width]
    d2 = np.full((height, width), np.inf, np.float64)
    for row, col in coordinates:
        d2 = np.minimum(d2, (yy-row)**2+(xx-col)**2)
    return {'R0': d2 <= 9, 'R1': (d2 > 9)&(d2 <= 81), 'R2': d2 > 81}


def _measure_from_sums(count, sums, full_pixels, extrema):
    require(type(count) is int and count >= 0 and type(full_pixels) is int and full_pixels > 0,
            'Invalid metric denominator')
    return dict(count=count, full_pixels=full_pixels, sums=sums,
                p_mean=_ratio(sums['p'], count), target_mean=_ratio(sums['target'], count),
                target_squared_mean=_ratio(sums['target_squared'], count),
                mse=_ratio(sums['squared_error'], count),
                weighted_mse=_ratio(200*sums['squared_error'], count),
                full_mse_contribution=sums['squared_error']/full_pixels,
                abs_logit_proxy_mean=_ratio(sums['abs_logit_proxy'], count),
                squared_logit_proxy_mean=_ratio(sums['squared_logit_proxy'], count),
                p_min=extrema[0], p_max=extrema[1])


def _measure(p, y, g, mask):
    pv, yv, gv = p[mask], y[mask], g[mask]
    sums = dict(p=float(pv.sum(dtype=np.float64)), target=float(yv.sum(dtype=np.float64)),
                target_squared=float((yv*yv).sum(dtype=np.float64)),
                squared_error=float(((pv-yv)**2).sum(dtype=np.float64)),
                abs_logit_proxy=float(np.abs(gv).sum(dtype=np.float64)),
                squared_logit_proxy=float((gv*gv).sum(dtype=np.float64)))
    extrema = (float(pv.min()), float(pv.max())) if pv.size else (None, None)
    return _measure_from_sums(int(pv.size), sums, int(p.size), extrema)


def center_statistics(probability, targets):
    probability = np.asarray(probability)
    y0 = targets['centers'][0]
    require(probability.dtype == np.float32 and probability.shape == y0.shape
            and np.isfinite(probability).all() and np.all((probability >= 0)&(probability <= 1)), 'Invalid center map')
    p, y = probability.astype(np.float64), y0.astype(np.float64)
    g = 400/p.size*(p-y)*p*(1-p)
    unique = sorted({(row['row'], row['col']) for row in targets['instances']})
    masks = distance_regions(p.shape, unique)
    return dict(global_metrics=_measure(p, y, g, np.ones(p.shape, bool)),
                distance_regions={k: _measure(p, y, g, mask) for k, mask in masks.items()},
                label_regions={k: _measure(p, y, g, mask) for k, mask in
                               {'background': targets['labels'] == 0, 'foreground': targets['labels'] > 0}.items()},
                unique_peak=value_summary([p[row, col] for row, col in unique], quantiles=True, retain_values=True),
                instance_peak=value_summary([p[row['row'], row['col']] for row in targets['instances']],
                                            quantiles=True, retain_values=True),
                unique_peak_fraction=len(unique)/p.size,
                constant_references=dict(zero_mse=float((y*y).mean(dtype=np.float64)),
                    label_mean_constant_mse=float(((y-y.mean(dtype=np.float64))**2).mean(dtype=np.float64))))


def maximum_cardinality_matching(gt_points, predicted_points, radius):
    """Ordered augmenting paths; maximum cardinality, not minimum total distance.

    gt_points are (ID,row,col), predictions are (selected_index,row,col).
    """
    require(type(radius) in (int, float) and math.isfinite(radius) and radius >= 0, 'Invalid radius')
    gt, pred = sorted(gt_points), sorted(predicted_points)
    for points in (gt, pred):
        require(all(len(p) == 3 and type(p[0]) is int and all(math.isfinite(v) for v in p[1:]) for p in points),
                'Invalid point identity')
        require(len({p[0] for p in points}) == len(points), 'Duplicate vertex identity')
    adjacency = []
    for _, row, col in gt:
        edges = [((row-r)**2+(col-c)**2, index, j) for j, (index, r, c) in enumerate(pred)]
        adjacency.append([j for d2, _, j in sorted(edges) if d2 <= radius*radius])
    owner = [-1]*len(pred)

    def augment(i, visited):
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if owner[j] == -1 or augment(owner[j], visited):
                owner[j] = i
                return True
        return False

    for i in range(len(gt)):
        augment(i, set())
    pairs = []
    for j, i in enumerate(owner):
        if i >= 0:
            d2 = (gt[i][1]-pred[j][1])**2+(gt[i][2]-pred[j][2])**2
            pairs.append(dict(gt_id=gt[i][0], selected_center_index=pred[j][0],
                              squared_distance=float(d2), distance=math.sqrt(d2)))
    pairs.sort(key=lambda p: (p['gt_id'], p['selected_center_index']))
    matched = len(pairs)
    return dict(radius=float(radius), gt_count=len(gt), predicted_count=len(pred), matched_count=matched,
                unmatched_gt_count=len(gt)-matched, unmatched_prediction_count=len(pred)-matched,
                recall=_ratio(matched, len(gt)), precision=_ratio(matched, len(pred)), pairs=pairs,
                matched_distances=value_summary([p['distance'] for p in pairs], retain_values=True),
                objective='maximum_cardinality_ordered_augmenting_paths_not_minimum_distance')


def center_locations(probability, instances, selected_centers):
    p = np.asarray(probability)
    require(p.ndim == 2 and p.size and p.dtype == np.float32 and np.isfinite(p).all(), 'Invalid center map')
    height, width = p.shape
    rows = []
    for item in sorted(instances, key=lambda x: x['id']):
        cy, cx = item['row'], item['col']
        require(0 <= cy < height and 0 <= cx < width, 'Center outside grid')
        row = dict(id=item['id'], row=cy, col=cx, probability=float(p[cy, cx]))
        for radius in (5, 10):
            y0, y1 = max(0, cy-radius), min(height, cy+radius+1)
            x0, x1 = max(0, cx-radius), min(width, cx+radius+1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            mask = (yy-cy)**2+(xx-cx)**2 <= radius*radius
            values = p[y0:y1, x0:x1]
            best = float(values[mask].max())
            dy, dx = np.argwhere(mask&(values == best))[0]
            y, x = int(y0+dy), int(x0+dx)
            row['r'+str(radius)] = dict(probability=best, row=y, col=x,
                                       distance=math.hypot(y-cy, x-cx))
        rows.append(row)
    point = np.argwhere(p == p.max())[0]
    top = dict(row=int(point[0]), col=int(point[1]), probability=float(p[tuple(point)]),
               nearest_gt_distance=min((math.hypot(point[0]-r['row'], point[1]-r['col']) for r in rows), default=None))
    unique = sorted({(r['row'], r['col']) for r in rows})
    instance_gt = [(r['id'], r['row'], r['col']) for r in rows]
    unique_gt = [(i, row, col) for i, (row, col) in enumerate(unique)]
    pred = [(r['selected_center_index'], r['row'], r['col']) for r in selected_centers]
    matches = {kind+'_r'+str(radius): maximum_cardinality_matching(points, pred, radius)
               for kind, points in (('instance', instance_gt), ('unique', unique_gt)) for radius in (5, 10)}
    return dict(instances=rows, global_maximum=top, matches=matches)


def representation_summary(labels, targets, original_labels=None):
    low_ids = [int(n) for n in np.unique(labels) if n]
    original_ids = None if original_labels is None else [int(n) for n in np.unique(_labels(original_labels)) if n]
    instances = targets['instances']
    groups = {}
    for item in instances:
        groups.setdefault((item['row'], item['col']), []).append(item['id'])
    near = []
    for i, left in enumerate(instances):
        for right in instances[i+1:]:
            d2 = (left['row']-right['row'])**2+(left['col']-right['col'])**2
            if d2 < 25:
                near.append(dict(ids=[left['id'], right['id']], squared_distance=d2))
    return dict(low_ids=low_ids, original_ids=original_ids,
                vanished_ids=None if original_ids is None else sorted(set(original_ids)-set(low_ids)),
                instances=instances, small_instance_ids=[r['id'] for r in instances if r['pixels'] < 20],
                outside_centroid_ids=[r['id'] for r in instances if not r['centroid_inside']],
                coincident_centers=[dict(row=k[0], col=k[1], ids=v) for k, v in sorted(groups.items()) if len(v) > 1],
                center_pairs_below_five=near)


def diagnose_frame(heads, low_labels, identity, original_labels=None):
    shape = check_heads(heads)
    labels = _labels(low_labels)
    require(labels.shape == shape, 'Label/head grids differ')
    require(set(identity) == {'scene', 'frame', 'condition', 'arm'} and type(identity['scene']) is int
            and identity['scene'] >= 0 and all(isinstance(identity[k], str) and identity[k] for k in ('frame', 'condition', 'arm')),
            'Invalid frame identity')
    if original_labels is not None:
        require(np.array_equal(resize_labels_training(original_labels, shape), labels), 'Training label resize differs')
    targets = instance_targets(labels)
    decoded = decode_trace(heads)
    return dict(schema=SCHEMA, status='unfrozen_component_result', identity=dict(identity), shape=list(shape),
                pixel_count=int(labels.size), representation=representation_summary(labels, targets, original_labels),
                decoder=dict(trace=decoded['trace'], selected_centers=decoded['selected_centers'],
                             retained_instances=decoded['retained_instances']),
                statistics=center_statistics(heads['center_probability'], targets),
                locations=center_locations(heads['center_probability'], targets['instances'], decoded['selected_centers']),
                prohibited_work=zero_work_counts())


def _combine_measures(measures):
    count = sum(m['count'] for m in measures)
    pixels = sum(m['full_pixels'] for m in measures)
    sums = {k: float(np.asarray([m['sums'][k] for m in measures], np.float64).sum(dtype=np.float64)) for k in SUM_KEYS}
    minimum = min((m['p_min'] for m in measures if m['count']), default=None)
    maximum = max((m['p_max'] for m in measures if m['count']), default=None)
    value = _measure_from_sums(count, sums, pixels, (minimum, maximum))
    value['abs_logit_proxy_per_frame'] = sums['abs_logit_proxy']/len(measures)
    value['squared_logit_proxy_per_frame'] = sums['squared_logit_proxy']/len(measures)
    return value


def aggregate_scene(frames):
    frames = list(frames)
    require(bool(frames), 'Scene has no frames')
    keys = [(r['identity']['scene'], r['identity']['condition'], r['identity']['arm']) for r in frames]
    require(len(set(keys)) == 1, 'Mixed scene/condition/arm')
    require(len({r['identity']['frame'] for r in frames}) == len(frames), 'Duplicate frame')
    require(len({tuple(r['shape']) for r in frames}) == 1, 'Mixed frame shapes')
    frames.sort(key=lambda r: r['identity']['frame'])
    for frame in frames:
        require(frame['schema'] == SCHEMA, 'Wrong frame schema')
        validate_zero_work(frame['prohibited_work'])
    stats = [r['statistics'] for r in frames]
    merged = dict(global_metrics=_combine_measures([s['global_metrics'] for s in stats]))
    for group in ('distance_regions', 'label_regions'):
        merged[group] = {key: _combine_measures([s[group][key] for s in stats]) for key in stats[0][group]}
    for group in ('unique_peak', 'instance_peak'):
        merged[group] = value_summary([v for s in stats for v in s[group]['values']], quantiles=True)
    pixels = sum(f['pixel_count'] for f in frames)
    merged['unique_peak_fraction'] = merged['unique_peak']['count']/pixels
    merged['constant_references'] = {k: sum(s['constant_references'][k]*f['pixel_count'] for s, f in zip(stats, frames))/pixels
                                     for k in stats[0]['constant_references']}
    counts = {key: sum(f['decoder']['trace'][key] for f in frames) for key in TRACE_COUNTS}
    cap_count = sum(bool(f['decoder']['trace']['cap_triggered']) for f in frames)
    locations = {}
    rows = [row for f in frames for row in f['locations']['instances']]
    locations['center_probability'] = value_summary([r['probability'] for r in rows])
    for radius in (5, 10):
        for field in ('probability', 'distance'):
            locations['r%d_%s' % (radius, field)] = value_summary([r['r'+str(radius)][field] for r in rows])
    locations['global_maximum_nearest_gt_distance'] = value_summary([
        f['locations']['global_maximum']['nearest_gt_distance'] for f in frames
        if f['locations']['global_maximum']['nearest_gt_distance'] is not None])
    matches = {}
    for key in frames[0]['locations']['matches']:
        values = [f['locations']['matches'][key] for f in frames]
        sums = {k: sum(v[k] for v in values) for k in ('gt_count', 'predicted_count', 'matched_count',
                                                     'unmatched_gt_count', 'unmatched_prediction_count')}
        matches[key] = dict(**sums, recall=_ratio(sums['matched_count'], sums['gt_count']),
            precision=_ratio(sums['matched_count'], sums['predicted_count']),
            matched_distances=value_summary([d for v in values for d in v['matched_distances']['values']]))
    scene, condition, arm = keys[0]
    return dict(schema=SCHEMA, identity=dict(scene=scene, condition=condition, arm=arm),
                frame_count=len(frames), frame_ids=[f['identity']['frame'] for f in frames], shape=frames[0]['shape'],
                pixel_count=pixels, statistics=merged,
                decoder=dict(counts=counts, per_frame={k: v/len(frames) for k, v in counts.items()},
                             cap_frame_count=cap_count, cap_frame_fraction=cap_count/len(frames)),
                locations=locations, matches=matches, prohibited_work=zero_work_counts())


def _macro(values, scene_ids):
    valid = [(scene, value) for scene, value in zip(scene_ids, values) if value is not None]
    require(all(math.isfinite(value) for _, value in valid), 'Nonfinite scene metric')
    return dict(mean=float(np.asarray([v for _, v in valid], np.float64).mean()) if valid else None,
                valid_scene_count=len(valid), total_scene_count=len(scene_ids),
                valid_scenes=[s for s, _ in valid], null_scenes=[s for s, v in zip(scene_ids, values) if v is None])


def scalar_scene_metrics(scene):
    """Explicit scalar projection; counts and extrema are not implicitly averaged."""
    stats = scene['statistics']
    result = {}
    groups = [('global', stats['global_metrics'])]
    groups += [(group+'.'+key, value) for group in ('distance_regions', 'label_regions') for key, value in stats[group].items()]
    for prefix, value in groups:
        for key in ('p_mean', 'target_mean', 'target_squared_mean', 'mse', 'weighted_mse', 'full_mse_contribution',
                    'abs_logit_proxy_mean', 'squared_logit_proxy_mean', 'abs_logit_proxy_per_frame', 'squared_logit_proxy_per_frame'):
            result[prefix+'.'+key] = value[key]
    result['unique_peak_fraction'] = stats['unique_peak_fraction']
    for name, value in stats['constant_references'].items():
        result['constant_references.'+name] = value
    for group in ('unique_peak', 'instance_peak'):
        result[group+'.mean'] = stats[group]['mean']
        for q, value in stats[group]['quantiles'].items():
            result[group+'.mean_of_scene_quantile.'+q] = value
    result.update({'decoder.per_frame.'+k: v for k, v in scene['decoder']['per_frame'].items()})
    result['decoder.cap_frame_fraction'] = scene['decoder']['cap_frame_fraction']
    result.update({'locations.'+k+'.mean': v['mean'] for k, v in scene['locations'].items()})
    for key, value in scene['matches'].items():
        for metric in ('recall', 'precision'):
            result['matches.'+key+'.'+metric] = value[metric]
        result['matches.'+key+'.matched_distance_mean'] = value['matched_distances']['mean']
    return result


def aggregate_cohort(scenes):
    scenes = sorted(scenes, key=lambda s: s['identity']['scene'])
    require(bool(scenes), 'Cohort has no scenes')
    ids = [s['identity']['scene'] for s in scenes]
    require(len(set(ids)) == len(ids), 'Duplicate scene')
    require(len({(s['identity']['condition'], s['identity']['arm']) for s in scenes}) == 1, 'Mixed cohort')
    require(len({tuple(s['shape']) for s in scenes}) == 1, 'Mixed grid shapes')
    for s in scenes:
        require(s['schema'] == SCHEMA, 'Wrong scene schema')
        validate_zero_work(s['prohibited_work'])
    values = [scalar_scene_metrics(s) for s in scenes]
    require(all(set(v) == set(values[0]) for v in values), 'Scene metric keys differ')
    macro = {k: _macro([v[k] for v in values], ids) for k in values[0]}
    return dict(schema=SCHEMA, condition=scenes[0]['identity']['condition'], arm=scenes[0]['identity']['arm'],
                scene_ids=ids, scene_count=len(ids), frame_count=sum(s['frame_count'] for s in scenes),
                macro=macro, aggregation='equal_scene_means; quantile fields are means of within-scene pooled quantiles',
                prohibited_work=zero_work_counts())


def paired_scene_difference(left_scenes, right_scenes):
    """Return right minus left for aligned scenes/frames; null pairs stay null."""
    left_scenes, right_scenes = list(left_scenes), list(right_scenes)
    for side in (left_scenes, right_scenes):
        require(bool(side), 'Empty paired cohort')
        require(len({(s['identity']['condition'], s['identity']['arm']) for s in side}) == 1, 'Mixed paired cohort')
        require(len({tuple(s['shape']) for s in side}) == 1, 'Mixed paired grid shapes')
        for scene in side:
            require(scene['schema'] == SCHEMA, 'Wrong paired schema')
            validate_zero_work(scene['prohibited_work'])
    left = {s['identity']['scene']: s for s in left_scenes}
    right = {s['identity']['scene']: s for s in right_scenes}
    require(len(left) == len(left_scenes) and len(right) == len(right_scenes), 'Duplicate paired scene')
    require(bool(left) and set(left) == set(right), 'Unaligned paired scenes')
    ids = sorted(left)
    differences = []
    for scene in ids:
        a, b = left[scene], right[scene]
        require(a['frame_ids'] == b['frame_ids'] and a['shape'] == b['shape'], 'Unaligned paired frames or grids')
        am, bm = scalar_scene_metrics(a), scalar_scene_metrics(b)
        require(set(am) == set(bm), 'Paired metric keys differ')
        differences.append({key: None if am[key] is None or bm[key] is None else bm[key]-am[key] for key in am})
    macro = {key: _macro([v[key] for v in differences], ids) for key in differences[0]}
    for value in macro.values():
        value['valid_pair_count'] = value['valid_scene_count']
        value['total_pair_count'] = value['total_scene_count']
    return dict(schema=SCHEMA, direction='right_minus_left', scene_ids=ids,
                left_identity={key: left_scenes[0]['identity'][key] for key in ('condition', 'arm')},
                right_identity={key: right_scenes[0]['identity'][key] for key in ('condition', 'arm')},
                by_scene={str(scene): value for scene, value in zip(ids, differences)},
                paired_macro=macro,
                prohibited_work=zero_work_counts())
