"""Unfrozen independent R20 mathematics; no I/O, models, RNG or production imports.

Only synthetic preparation is authorized by this module's present status. Formal
source identities, receipts, 1280 support, locks and cumulative budgets belong to
the future runner. Reported matching pairs are the sole production intermediate
accepted: their legality, distances and maximum cardinality are independently
checked by a unit-capacity residual-network BFS, not the production DFS matcher.
The fixed objective string describes the reported pair policy; it is not a claim
that max flow reproduces its particular nonunique solution or minimum distance.
"""
import math
from collections import deque

import numpy as np
from PIL import Image

DRAFT_SHA256 = '859b8a1ae1317255da3567aee834ede9862820499afe5d5dbef2cbd63e708357'
SCHEMA = 20
ZERO_KEYS = ('new_model_forward_calls', 'new_training_updates', 'new_backward_calls',
             'new_official_evaluations', 'new_random_draws')
SUM_KEYS = ('p', 'target', 'target_squared', 'squared_error', 'abs_logit_proxy', 'squared_logit_proxy')
TRACE_COUNTS = ('threshold_pixels', 'candidate_count', 'visited_count', 'distance_rejected_count',
                'selected_count', 'unvisited_count', 'zero_assignment_count', 'retained_count')
MATCH_KEYS = ('instance_r5', 'instance_r10', 'unique_r5', 'unique_r10')
REPORTED_POLICY = 'maximum_cardinality_ordered_augmenting_paths_not_minimum_distance'
FLOW_POLICY = 'maximum_cardinality_independent_residual_bfs_not_minimum_distance'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def size(value):
    value = tuple(value)
    need(len(value) == 2 and all(type(x) is int and x > 0 for x in value), 'Invalid grid size')
    return value


def label_array(value):
    a = np.asarray(value)
    need(a.ndim == 2 and a.size > 0 and a.dtype.kind in 'iu', 'Invalid label array')
    need(bool(np.all((a >= 0) & (a <= 88))), 'Invalid label IDs')
    return a.astype(np.int64, copy=True)


def zero_work():
    return dict.fromkeys(ZERO_KEYS, 0)


def check_zero_work(value):
    need(type(value) is dict and set(value) == set(ZERO_KEYS), 'Wrong work counter fields')
    need(all(type(v) is int and v == 0 for v in value.values()), 'Work counters must be strict int zero')


def finite_values(values):
    a = np.asarray(list(values), dtype=np.float64)
    need(a.ndim == 1 and bool(np.isfinite(a).all()), 'Invalid finite vector')
    return a


def quotient(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def linear_quantiles(values):
    ordered = sorted(float(v) for v in finite_values(values))
    result = {}
    for q in (0., .5, .9, .99, 1.):
        if not ordered:
            value = None
        else:
            h = (len(ordered)-1)*q
            i, j = math.floor(h), math.ceil(h)
            value = ordered[i] + (ordered[j]-ordered[i])*(h-i)
        result[str(q)] = value
    return result


def summarize(values, quantiles=False, keep=False):
    a = finite_values(values)
    total = math.fsum(a.tolist())
    result = dict(count=int(a.size), sum=total, mean=quotient(total, int(a.size)))
    if quantiles:
        result['quantiles'] = linear_quantiles(a)
    if keep:
        result['values'] = a.tolist()
    return result


def independent_targets(labels):
    labels = label_array(labels)
    height, width = labels.shape
    heat = np.zeros(labels.shape, dtype=np.float32)
    offsets = np.zeros((2, height, width), dtype=np.float32)
    instances = []
    row_axis, col_axis = np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32)
    for identity in np.unique(labels):
        if identity == 0:
            continue
        flat = np.flatnonzero(labels.ravel() == identity)
        rr, cc = flat // width, flat % width
        row = int(np.rint(np.add.reduce(rr.astype(np.float32), dtype=np.float32)/np.float32(len(rr))))
        col = int(np.rint(np.add.reduce(cc.astype(np.float32), dtype=np.float32)/np.float32(len(cc))))
        radial = np.add(np.square(row_axis-np.float32(row))[:, None],
                        np.square(col_axis-np.float32(col))[None, :], dtype=np.float32)
        gaussian = np.exp(np.divide(-radial, np.float32(18), dtype=np.float32), dtype=np.float32)
        np.maximum(heat, gaussian, out=heat)
        offsets[0, rr, cc] = (np.float32(col)-cc.astype(np.float32))/np.float32(width)
        offsets[1, rr, cc] = (np.float32(row)-rr.astype(np.float32))/np.float32(height)
        instances.append(dict(id=int(identity), row=row, col=col, pixels=int(len(flat)),
                              centroid_inside=bool(labels[row, col] == identity)))
    return dict(labels=labels, foreground=(labels != 0).astype(np.float32)[None],
                centers=heat[None], offsets=offsets, instances=instances)


def training_resize(labels, output_hw):
    h, w = size(output_hw)
    image = Image.fromarray(label_array(labels).astype(np.uint16))
    return np.array(image.resize((w, h), resample=Image.Resampling.NEAREST), dtype=np.int64)


def evaluation_resize(array, output_hw):
    a = np.asarray(array)
    h, w = size(output_hw)
    need(a.ndim == 2 and a.size > 0 and a.dtype.kind in 'biu', 'Invalid evaluation grid')
    # Python integer axis arithmetic is independent of NumPy's vector formula.
    rows = [((2*i+1)*a.shape[0])//(2*h) for i in range(h)]
    cols = [((2*j+1)*a.shape[1])//(2*w) for j in range(w)]
    return a[np.ix_(rows, cols)].copy()


def check_heads(heads):
    need(type(heads) is dict and set(heads) == {'foreground_probability', 'center_probability', 'offsets'}, 'Wrong heads')
    need(all(isinstance(a, np.ndarray) and a.dtype == np.float32 and np.isfinite(a).all()
             for a in heads.values()), 'Heads must be finite float32 arrays')
    fg, p, offsets = (heads[k] for k in ('foreground_probability', 'center_probability', 'offsets'))
    need(fg.ndim == 2 and fg.size > 0 and p.shape == fg.shape and offsets.shape == (2,)+fg.shape, 'Wrong head grids')
    need(bool(np.all((fg >= 0)&(fg <= 1))) and bool(np.all((p >= 0)&(p <= 1))), 'Probability outside [0,1]')
    return tuple(int(n) for n in fg.shape)


def independent_decode(heads):
    h, w = check_heads(heads)
    foreground = heads['foreground_probability'] >= np.float32(.5)
    p, offset = heads['center_probability'], heads['offsets']
    # Separable nine-neighbour maxima; no SciPy filter or production decoder.
    padded = np.pad(p, ((4, 4), (0, 0)), constant_values=-np.inf)
    vertical = np.full((h, w), -np.inf, dtype=np.float32)
    for shift in range(9):
        np.maximum(vertical, padded[shift:shift+h], out=vertical)
    padded = np.pad(vertical, ((0, 0), (4, 4)), constant_values=-np.inf)
    local = np.full((h, w), -np.inf, dtype=np.float32)
    for shift in range(9):
        np.maximum(local, padded[:, shift:shift+w], out=local)
    rows, cols = np.nonzero((p == local)&(p >= np.float32(.35)))
    # Explicit score / row-major tie ordering is equivalent to the fixed stable order.
    order = np.lexsort((cols, rows, -p[rows, cols]))
    candidates = [(int(rows[i]), int(cols[i]), float(p[rows[i], cols[i]])) for i in order]
    selected, visited, rejected, capped = [], 0, 0, False
    for row, col, score in candidates:
        visited += 1
        if any((row-c['row'])**2+(col-c['col'])**2 < 25 for c in selected):
            rejected += 1
            continue
        selected.append(dict(selected_center_index=len(selected), row=row, col=col, score=score))
        if len(selected) == 64:
            capped = True
            break
    labels = np.zeros((h, w), dtype=np.int32)
    rr, cc = np.nonzero(foreground)
    areas, retained = [0 for _ in selected], []
    if selected and len(rr):
        target_y = rr.astype(np.float64)+(offset[1, rr, cc]*np.float32(h)).astype(np.float64)
        target_x = cc.astype(np.float64)+(offset[0, rr, cc]*np.float32(w)).astype(np.float64)
        need(np.isfinite(target_y).all() and np.isfinite(target_x).all(), 'Nonfinite offset destination')
        assignment = np.empty(len(rr), dtype=np.int32)
        sy = np.array([c['row'] for c in selected], dtype=np.float64)
        sx = np.array([c['col'] for c in selected], dtype=np.float64)
        for start in range(0, len(rr), 2048):
            stop = min(start+2048, len(rr))
            distance = np.square(target_y[start:stop, None]-sy)+np.square(target_x[start:stop, None]-sx)
            assignment[start:stop] = np.argmin(distance, axis=1)
        areas = np.bincount(assignment, minlength=len(selected)).tolist()
        for index, amount in enumerate(areas):
            if amount >= 20:
                members = assignment == index
                label = len(retained)+1
                labels[rr[members], cc[members]] = label
                retained.append(dict(instance_id=label, **selected[index]))
    return dict(instance_map=labels, foreground_mask=foreground, selected_centers=selected,
                retained_instances=retained, trace=dict(
                    threshold_pixels=int(np.count_nonzero(p >= np.float32(.35))),
                    maximum_meets_center_threshold=bool(np.max(p) >= np.float32(.35)),
                    center_threshold_float32=float(np.float32(.35)), candidate_count=len(candidates),
                    visited_count=visited, distance_rejected_count=rejected, selected_count=len(selected),
                    unvisited_count=len(candidates)-visited, cap_triggered=capped,
                    candidate_order=[list(c) for c in candidates], assigned_pixels=areas,
                    zero_assignment_count=areas.count(0), retained_count=len(retained)))


def region_codes(shape, centers):
    h, w = size(shape)
    unique = sorted(set(tuple(point) for point in centers))
    need(all(len(p) == 2 and all(type(n) is int for n in p) and 0 <= p[0] < h and 0 <= p[1] < w
             for p in unique), 'Invalid region centers')
    # Only two bounded unions are needed; no 88 x H x W distance tensor or pixel loop.
    near, middle = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for row, col in unique:
        y0, y1, x0, x1 = max(0, row-9), min(h, row+10), max(0, col-9), min(w, col+10)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = (yy-row)**2+(xx-col)**2
        near[y0:y1, x0:x1] |= d2 <= 9
        middle[y0:y1, x0:x1] |= d2 <= 81
    return np.where(near, 0, np.where(middle, 1, 2)).astype(np.uint8)


def measure_result(n, sums, total_pixels, low, high):
    need(type(n) is int and 0 <= n <= total_pixels and type(total_pixels) is int and total_pixels > 0,
         'Invalid metric support')
    return dict(count=n, full_pixels=total_pixels, sums=sums, p_mean=quotient(sums['p'], n),
                target_mean=quotient(sums['target'], n), target_squared_mean=quotient(sums['target_squared'], n),
                mse=quotient(sums['squared_error'], n), weighted_mse=quotient(200*sums['squared_error'], n),
                full_mse_contribution=sums['squared_error']/total_pixels,
                abs_logit_proxy_mean=quotient(sums['abs_logit_proxy'], n),
                squared_logit_proxy_mean=quotient(sums['squared_logit_proxy'], n), p_min=low, p_max=high)


def independent_statistics(probability, targets):
    p = np.asarray(probability, dtype=np.float64)
    y = targets['centers'][0].astype(np.float64)
    need(p.shape == y.shape and np.isfinite(p).all() and np.all((p >= 0)&(p <= 1)), 'Invalid center map')
    error = p-y
    derivative = (error*p*(1-p))*(400./p.size)
    quantities = (p, y, np.square(y), np.square(error), np.abs(derivative), np.square(derivative))

    def calculate(mask):
        n = int(np.count_nonzero(mask))
        sums = {key: float(np.add.reduce(value[mask], dtype=np.float64)) for key, value in zip(SUM_KEYS, quantities)}
        return measure_result(n, sums, int(p.size), float(np.min(p[mask])) if n else None,
                              float(np.max(p[mask])) if n else None)

    unique = sorted(set((r['row'], r['col']) for r in targets['instances']))
    codes = region_codes(tuple(int(n) for n in p.shape), unique)
    mean_y = float(np.add.reduce(y.ravel(), dtype=np.float64))/y.size
    return dict(global_metrics=calculate(np.ones(p.shape, dtype=bool)),
                distance_regions={'R'+str(i): calculate(codes == i) for i in range(3)},
                label_regions={'background': calculate(targets['labels'] == 0), 'foreground': calculate(targets['labels'] != 0)},
                unique_peak=summarize((p[r, c] for r, c in unique), True, True),
                instance_peak=summarize((p[r['row'], r['col']] for r in targets['instances']), True, True),
                unique_peak_fraction=len(unique)/p.size,
                constant_references=dict(zero_mse=float(np.add.reduce(np.square(y).ravel(), dtype=np.float64))/y.size,
                    label_mean_constant_mse=float(np.add.reduce(np.square(y-mean_y).ravel(), dtype=np.float64))/y.size))


def _points(points):
    points = sorted(tuple(p) for p in points)
    need(all(len(p) == 3 and type(p[0]) is int and all(type(v) in (int, float) and math.isfinite(v)
             for v in p[1:]) for p in points), 'Invalid matching coordinates')
    need(len({p[0] for p in points}) == len(points), 'Duplicate matching identity')
    return points


def maximum_flow_pairs(gt_points, predicted_points, radius):
    """Edmonds-Karp BFS on a unit-capacity network; no DFS matching routine."""
    gt, pred = _points(gt_points), _points(predicted_points)
    need(type(radius) in (int, float) and math.isfinite(radius) and radius >= 0, 'Invalid radius')
    n, m = len(gt), len(pred)
    source, sink = n+m, n+m+1
    capacity = np.zeros((n+m+2, n+m+2), dtype=np.int8)
    capacity[source, :n] = 1
    capacity[n:n+m, sink] = 1
    edges = []
    for i, (_, y, x) in enumerate(gt):
        for j, (_, yy, xx) in enumerate(pred):
            if (y-yy)**2+(x-xx)**2 <= radius*radius:
                capacity[i, n+j] = 1
                edges.append((i, j))
    residual = capacity.copy()
    while True:
        previous = [-1]*(n+m+2)
        previous[source] = source
        pending = deque([source])
        while pending and previous[sink] == -1:
            vertex = pending.popleft()
            for neighbor in np.flatnonzero(residual[vertex] > 0):
                neighbor = int(neighbor)
                if previous[neighbor] == -1:
                    previous[neighbor] = vertex
                    pending.append(neighbor)
        if previous[sink] == -1:
            break
        vertex = sink
        while vertex != source:
            parent = previous[vertex]
            residual[parent, vertex] -= 1
            residual[vertex, parent] += 1
            vertex = parent
    return [(gt[i][0], pred[j][0]) for i, j in edges if residual[i, n+j] == 0]


def matching_statistics(gt_points, predicted_points, radius, reported_pairs=None):
    gt, pred = _points(gt_points), _points(predicted_points)
    optimum = maximum_flow_pairs(gt, pred, radius)
    by_gt, by_pred = {p[0]: p[1:] for p in gt}, {p[0]: p[1:] for p in pred}
    if reported_pairs is None:
        identities = optimum
    else:
        need(type(reported_pairs) is list, 'Reported matching must be a list')
        identities = []
        for pair in reported_pairs:
            need(type(pair) is dict and set(pair) == {'gt_id', 'selected_center_index', 'squared_distance', 'distance'},
                 'Wrong reported pair fields')
            a, b = pair['gt_id'], pair['selected_center_index']
            need(type(a) is int and type(b) is int and a in by_gt and b in by_pred, 'Unknown matching identity')
            d2 = (by_gt[a][0]-by_pred[b][0])**2+(by_gt[a][1]-by_pred[b][1])**2
            need(d2 <= radius*radius, 'Matching edge outside radius')
            need(type(pair['squared_distance']) is float and pair['squared_distance'] == float(d2)
                 and type(pair['distance']) is float and pair['distance'] == math.sqrt(d2), 'Reported distance differs')
            identities.append((a, b))
        need(len(identities) == len(optimum), 'Reported matching is not maximum cardinality')
        need(len({a for a, _ in identities}) == len(identities) == len({b for _, b in identities}), 'Repeated matched vertex')
        need(identities == sorted(identities), 'Reported pair order differs')
    pairs = []
    for a, b in sorted(identities):
        d2 = (by_gt[a][0]-by_pred[b][0])**2+(by_gt[a][1]-by_pred[b][1])**2
        pairs.append(dict(gt_id=a, selected_center_index=b, squared_distance=float(d2), distance=math.sqrt(d2)))
    count = len(pairs)
    return dict(radius=float(radius), gt_count=len(gt), predicted_count=len(pred), matched_count=count,
                unmatched_gt_count=len(gt)-count, unmatched_prediction_count=len(pred)-count,
                recall=quotient(count, len(gt)), precision=quotient(count, len(pred)), pairs=pairs,
                matched_distances=summarize((r['distance'] for r in pairs), keep=True),
                objective=FLOW_POLICY if reported_pairs is None else REPORTED_POLICY)


def independent_locations(probability, instances, selected, reported_matches=None):
    p = np.asarray(probability)
    need(p.ndim == 2 and p.size and p.dtype == np.float32 and np.isfinite(p).all(), 'Invalid location map')
    h, w = p.shape
    records = []
    for item in sorted(instances, key=lambda r: r['id']):
        row, col = item['row'], item['col']
        need(type(row) is int and type(col) is int and 0 <= row < h and 0 <= col < w, 'Invalid center')
        result = dict(id=item['id'], row=row, col=col, probability=float(p[row, col]))
        for radius in (5, 10):
            y0, y1, x0, x1 = max(0, row-radius), min(h, row+radius+1), max(0, col-radius), min(w, col+radius+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            allowed = (yy-row)**2+(xx-col)**2 <= radius*radius
            masked = np.where(allowed, p[y0:y1, x0:x1], -np.inf)
            local_y, local_x = np.unravel_index(int(np.argmax(masked)), masked.shape)
            y, x = int(y0+local_y), int(x0+local_x)
            result['r'+str(radius)] = dict(probability=float(p[y, x]), row=y, col=x, distance=math.hypot(y-row, x-col))
        records.append(result)
    top_y, top_x = (int(n) for n in np.unravel_index(int(np.argmax(p)), p.shape))
    top = dict(row=top_y, col=top_x, probability=float(p[top_y, top_x]),
               nearest_gt_distance=min((math.hypot(top_y-r['row'], top_x-r['col']) for r in records), default=None))
    unique = sorted(set((r['row'], r['col']) for r in records))
    gt = {'instance': [(r['id'], r['row'], r['col']) for r in records],
          'unique': [(i, point[0], point[1]) for i, point in enumerate(unique)]}
    pred = [(r['selected_center_index'], r['row'], r['col']) for r in selected]
    if reported_matches is not None:
        need(type(reported_matches) is dict and set(reported_matches) == set(MATCH_KEYS), 'Wrong reported match groups')
    matches = {}
    for kind in ('instance', 'unique'):
        for radius in (5, 10):
            key = kind+'_r'+str(radius)
            reported_pairs = None if reported_matches is None else reported_matches[key]['pairs']
            matches[key] = matching_statistics(gt[kind], pred, radius, reported_pairs)
    return dict(instances=records, global_maximum=top, matches=matches)


def independent_representation(labels, targets, original_labels):
    instances = targets['instances']
    low_ids = [int(v) for v in np.unique(labels) if v != 0]
    original_ids = None if original_labels is None else [int(v) for v in np.unique(label_array(original_labels)) if v != 0]
    grouped = {}
    for row in instances:
        grouped.setdefault((row['row'], row['col']), []).append(row['id'])
    pairs = []
    for index, a in enumerate(instances):
        for b in instances[index+1:]:
            d2 = (a['row']-b['row'])**2+(a['col']-b['col'])**2
            if d2 < 25:
                pairs.append(dict(ids=[a['id'], b['id']], squared_distance=d2))
    return dict(low_ids=low_ids, original_ids=original_ids,
                vanished_ids=None if original_ids is None else sorted(set(original_ids)-set(low_ids)),
                instances=instances, small_instance_ids=[r['id'] for r in instances if r['pixels'] < 20],
                outside_centroid_ids=[r['id'] for r in instances if not r['centroid_inside']],
                coincident_centers=[dict(row=p[0], col=p[1], ids=ids) for p, ids in sorted(grouped.items()) if len(ids) > 1],
                center_pairs_below_five=pairs)


def recompute_frame(heads, low_labels, identity, original_labels=None, reported_matches=None):
    shape = check_heads(heads)
    labels = label_array(low_labels)
    need(labels.shape == shape, 'Label and head grids differ')
    need(type(identity) is dict and set(identity) == {'scene', 'frame', 'condition', 'arm'}
         and type(identity['scene']) is int and identity['scene'] >= 0
         and all(type(identity[k]) is str and bool(identity[k]) for k in ('frame', 'condition', 'arm')), 'Invalid frame identity')
    if original_labels is not None:
        need(np.array_equal(training_resize(original_labels, shape), labels), 'Training resize mismatch')
    target = independent_targets(labels)
    decoded = independent_decode(heads)
    return dict(schema=20, status='unfrozen_component_result', identity=dict(identity), shape=list(shape),
                pixel_count=int(labels.size), representation=independent_representation(labels, target, original_labels),
                decoder={k: decoded[k] for k in ('trace', 'selected_centers', 'retained_instances')},
                statistics=independent_statistics(heads['center_probability'], target),
                locations=independent_locations(heads['center_probability'], target['instances'],
                                               decoded['selected_centers'], reported_matches), prohibited_work=zero_work())


def compare_tree(expected, actual, *, abs_tol, rel_tol, path='root'):
    """Exact shape, keys, types, nulls and integers; explicit float tolerances.

    The caller must freeze field-specific tolerances before formal data access.
    This generic comparator does not select or relax that policy.
    """
    need(type(abs_tol) in (int, float) and math.isfinite(abs_tol) and abs_tol >= 0
         and type(rel_tol) in (int, float) and math.isfinite(rel_tol) and rel_tol >= 0, 'Invalid tolerance')
    if type(expected) is dict:
        need(type(actual) is dict and set(actual) == set(expected), path+': keys differ')
        for key in expected:
            compare_tree(expected[key], actual[key], abs_tol=abs_tol, rel_tol=rel_tol, path=path+'.'+key)
    elif type(expected) is list:
        need(type(actual) is list and len(actual) == len(expected), path+': list differs')
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare_tree(left, right, abs_tol=abs_tol, rel_tol=rel_tol, path=path+'['+str(i)+']')
    elif type(expected) is float:
        need(type(actual) is float and math.isfinite(expected) and math.isfinite(actual)
             and abs(expected-actual) <= abs_tol+rel_tol*abs(expected), path+': float differs')
    else:
        need(type(actual) is type(expected) and actual == expected, path+': exact value differs')


def audit_frame(heads, low_labels, identity, reported_frame, *, original_labels=None, abs_tol, rel_tol):
    """Recompute everything from inputs, only using reported pairs as described above."""
    expected = recompute_frame(heads, low_labels, identity, original_labels,
                               reported_frame['locations']['matches'])
    # All original decoding metadata, including float32 scores and threshold,
    # remains exact; tolerances apply only to newly derived floating statistics.
    compare_tree(expected['decoder'], reported_frame['decoder'], abs_tol=0., rel_tol=0., path='root.decoder')
    compare_tree(expected, reported_frame, abs_tol=abs_tol, rel_tol=rel_tol)
    return expected


def pooled_measure(rows):
    rows = list(rows)
    n, total = sum(r['count'] for r in rows), sum(r['full_pixels'] for r in rows)
    sums = {k: math.fsum(r['sums'][k] for r in rows) for k in SUM_KEYS}
    result = measure_result(n, sums, total, min((r['p_min'] for r in rows if r['count']), default=None),
                            max((r['p_max'] for r in rows if r['count']), default=None))
    result.update(abs_logit_proxy_per_frame=sums['abs_logit_proxy']/len(rows),
                  squared_logit_proxy_per_frame=sums['squared_logit_proxy']/len(rows))
    return result


def _check_record(record, level):
    need(type(record) is dict and type(record.get('schema')) is int and record['schema'] == 20, 'Wrong '+level+' schema')
    check_zero_work(record['prohibited_work'])
    size(record['shape'])
    need(type(record['identity']['scene']) is int and record['identity']['scene'] >= 0, 'Wrong scene ID')


def recompute_scene(frames):
    """Pool independently recomputed/audited frames, never unaudited stored JSON."""
    frames = sorted(list(frames), key=lambda r: r['identity']['frame'])
    need(bool(frames), 'Empty scene')
    for f in frames:
        _check_record(f, 'frame')
    identities = [f['identity'] for f in frames]
    need(len({(r['scene'], r['condition'], r['arm']) for r in identities}) == 1, 'Mixed scene')
    need(len({r['frame'] for r in identities}) == len(frames), 'Duplicate frame')
    need(len({tuple(f['shape']) for f in frames}) == 1, 'Mixed shapes')
    stats = [f['statistics'] for f in frames]
    merged = dict(global_metrics=pooled_measure(s['global_metrics'] for s in stats))
    for group, keys in (('distance_regions', ('R0', 'R1', 'R2')), ('label_regions', ('background', 'foreground'))):
        merged[group] = {key: pooled_measure(s[group][key] for s in stats) for key in keys}
    for kind in ('unique_peak', 'instance_peak'):
        merged[kind] = summarize((v for s in stats for v in s[kind]['values']), quantiles=True)
    pixels = sum(f['pixel_count'] for f in frames)
    merged['unique_peak_fraction'] = merged['unique_peak']['count']/pixels
    merged['constant_references'] = {key: math.fsum(s['constant_references'][key]*f['pixel_count']
        for s, f in zip(stats, frames))/pixels for key in ('zero_mse', 'label_mean_constant_mse')}
    counts = {key: sum(f['decoder']['trace'][key] for f in frames) for key in TRACE_COUNTS}
    need(all(type(f['decoder']['trace']['cap_triggered']) is bool for f in frames), 'Cap flag is not bool')
    cap = sum(1 for f in frames if f['decoder']['trace']['cap_triggered'])
    rows = [r for f in frames for r in f['locations']['instances']]
    locations = dict(center_probability=summarize(r['probability'] for r in rows))
    for radius in (5, 10):
        for field in ('probability', 'distance'):
            locations['r%d_%s' % (radius, field)] = summarize(r['r'+str(radius)][field] for r in rows)
    locations['global_maximum_nearest_gt_distance'] = summarize(f['locations']['global_maximum']['nearest_gt_distance']
        for f in frames if f['locations']['global_maximum']['nearest_gt_distance'] is not None)
    matches = {}
    for name in MATCH_KEYS:
        values = [f['locations']['matches'][name] for f in frames]
        sums = {key: sum(r[key] for r in values) for key in ('gt_count', 'predicted_count', 'matched_count',
                                                           'unmatched_gt_count', 'unmatched_prediction_count')}
        matches[name] = dict(**sums, recall=quotient(sums['matched_count'], sums['gt_count']),
                             precision=quotient(sums['matched_count'], sums['predicted_count']),
                             matched_distances=summarize(d for r in values for d in r['matched_distances']['values']))
    return dict(schema=20, identity={key: identities[0][key] for key in ('scene', 'condition', 'arm')},
                frame_count=len(frames), frame_ids=[r['frame'] for r in identities], shape=list(frames[0]['shape']),
                pixel_count=pixels, statistics=merged, decoder=dict(counts=counts,
                    per_frame={key: value/len(frames) for key, value in counts.items()},
                    cap_frame_count=cap, cap_frame_fraction=cap/len(frames)), locations=locations,
                matches=matches, prohibited_work=zero_work())


def project_metrics(scene):
    result = {}
    stats = scene['statistics']
    measures = {'global': stats['global_metrics']}
    for group in ('distance_regions', 'label_regions'):
        measures.update({group+'.'+key: value for key, value in stats[group].items()})
    fields = ('p_mean', 'target_mean', 'target_squared_mean', 'mse', 'weighted_mse', 'full_mse_contribution',
              'abs_logit_proxy_mean', 'squared_logit_proxy_mean', 'abs_logit_proxy_per_frame', 'squared_logit_proxy_per_frame')
    for name, measure in measures.items():
        result.update({name+'.'+key: measure[key] for key in fields})
    result['unique_peak_fraction'] = stats['unique_peak_fraction']
    result.update({'constant_references.'+key: value for key, value in stats['constant_references'].items()})
    for group in ('unique_peak', 'instance_peak'):
        result[group+'.mean'] = stats[group]['mean']
        result.update({group+'.mean_of_scene_quantile.'+key: value for key, value in stats[group]['quantiles'].items()})
    result.update({'decoder.per_frame.'+key: value for key, value in scene['decoder']['per_frame'].items()})
    result['decoder.cap_frame_fraction'] = scene['decoder']['cap_frame_fraction']
    result.update({'locations.'+key+'.mean': value['mean'] for key, value in scene['locations'].items()})
    for group, values in scene['matches'].items():
        for key in ('recall', 'precision'):
            result['matches.'+group+'.'+key] = values[key]
        result['matches.'+group+'.matched_distance_mean'] = values['matched_distances']['mean']
    return result


def scene_mean(values, ids):
    need(len(values) == len(ids), 'Misaligned macro values')
    valid = [i for i, value in enumerate(values) if value is not None]
    need(all(type(values[i]) in (int, float) and math.isfinite(values[i]) for i in valid), 'Nonfinite macro input')
    return dict(mean=quotient(math.fsum(values[i] for i in valid), len(valid)), valid_scene_count=len(valid),
                total_scene_count=len(ids), valid_scenes=[ids[i] for i in valid],
                null_scenes=[ids[i] for i in range(len(ids)) if values[i] is None])


def _cohort(scenes):
    scenes = sorted(list(scenes), key=lambda r: r['identity']['scene'])
    need(bool(scenes), 'Empty cohort')
    for s in scenes:
        _check_record(s, 'scene')
    need(len({s['identity']['scene'] for s in scenes}) == len(scenes), 'Duplicate scene')
    need(len({(s['identity']['condition'], s['identity']['arm']) for s in scenes}) == 1, 'Mixed cohort identities')
    need(len({tuple(s['shape']) for s in scenes}) == 1, 'Mixed cohort grids')
    return scenes


def recompute_cohort(scenes):
    scenes = _cohort(scenes)
    ids = [s['identity']['scene'] for s in scenes]
    metrics = [project_metrics(s) for s in scenes]
    need(all(set(m) == set(metrics[0]) for m in metrics), 'Metric schema differs')
    return dict(schema=20, condition=scenes[0]['identity']['condition'], arm=scenes[0]['identity']['arm'],
                scene_ids=ids, scene_count=len(ids), frame_count=sum(s['frame_count'] for s in scenes),
                macro={key: scene_mean([m[key] for m in metrics], ids) for key in metrics[0]},
                aggregation='equal_scene_means; quantile fields are means of within-scene pooled quantiles',
                prohibited_work=zero_work())


def recompute_paired(left_scenes, right_scenes):
    left, right = _cohort(left_scenes), _cohort(right_scenes)
    ids = [s['identity']['scene'] for s in left]
    need(ids == [s['identity']['scene'] for s in right], 'Unaligned scene pairs')
    differences = []
    for a, b in zip(left, right):
        need(a['frame_ids'] == b['frame_ids'] and a['shape'] == b['shape'], 'Unaligned frame pairs')
        first, second = project_metrics(a), project_metrics(b)
        need(set(first) == set(second), 'Paired metric keys differ')
        differences.append({k: None if first[k] is None or second[k] is None else second[k]-first[k] for k in first})
    macro = {key: scene_mean([d[key] for d in differences], ids) for key in differences[0]}
    for item in macro.values():
        item.update(valid_pair_count=item['valid_scene_count'], total_pair_count=item['total_scene_count'])
    return dict(schema=20, direction='right_minus_left', scene_ids=ids,
                left_identity={k: left[0]['identity'][k] for k in ('condition', 'arm')},
                right_identity={k: right[0]['identity'][k] for k in ('condition', 'arm')},
                by_scene={str(i): d for i, d in zip(ids, differences)}, paired_macro=macro, prohibited_work=zero_work())
