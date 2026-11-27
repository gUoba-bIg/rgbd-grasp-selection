"""Strict preparation JSON contracts; internal consistency is not independent acceptance.

Only the standard library is used. No paths are opened, no production/reference
mathematics is imported, and no formal experiment identity set is inferred.
"""
import json
import math

ALGEBRA_ABS_TOL = 1e-12
ALGEBRA_REL_TOL = 1e-12
THRESHOLD_FLOAT32 = 0.3499999940395355
ZERO_KEYS = ('new_model_forward_calls', 'new_training_updates', 'new_backward_calls', 'new_official_evaluations', 'new_random_draws')
SUM_KEYS = ('p', 'target', 'target_squared', 'squared_error', 'abs_logit_proxy', 'squared_logit_proxy')
TRACE_COUNTS = ('threshold_pixels', 'candidate_count', 'visited_count', 'distance_rejected_count', 'selected_count', 'unvisited_count', 'zero_assignment_count', 'retained_count')
QKEYS = ('0.0', '0.5', '0.9', '0.99', '1.0')
MATCH_KEYS = ('instance_r5', 'instance_r10', 'unique_r5', 'unique_r10')
LOCATION_KEYS = ('center_probability', 'r5_probability', 'r5_distance', 'r10_probability', 'r10_distance', 'global_maximum_nearest_gt_distance')
MEANS = ('p_mean', 'target_mean', 'target_squared_mean', 'mse', 'weighted_mse', 'full_mse_contribution', 'abs_logit_proxy_mean', 'squared_logit_proxy_mean')
SCENE_MEANS = MEANS + ('abs_logit_proxy_per_frame', 'squared_logit_proxy_per_frame')
AGGREGATION = 'equal_scene_means; quantile fields are means of within-scene pooled quantiles'
OBJECTIVE = 'maximum_cardinality_ordered_augmenting_paths_not_minimum_distance'


def need(condition, path, reason):
    if not condition:
        raise ValueError(path + ': ' + reason)


def fields(value, keys, path):
    need(type(value) is dict and set(value) == set(keys), path, 'missing or additional fields')


def integer(value, path, minimum=0, maximum=None):
    need(type(value) is int and value >= minimum and (maximum is None or value <= maximum), path, 'expected bounded strict integer')
    return value


def number(value, path, minimum=None, maximum=None):
    need(type(value) in (int, float), path, 'expected finite number, not boolean')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    need(finite and (minimum is None or value >= minimum) and (maximum is None or value <= maximum), path, 'nonfinite or out-of-range number')
    return value


def boolean(value, path):
    need(type(value) is bool, path, 'expected strict boolean')


def string(value, path):
    need(type(value) is str and bool(value), path, 'expected nonempty string')


def array(value, path, length=None):
    need(type(value) is list and (length is None or len(value) == length), path, 'wrong array type or length')


def close(actual, expected, path):
    if expected is None:
        need(actual is None, path, 'zero denominator must be null')
    else:
        number(actual, path)
        need(math.isclose(actual, expected, abs_tol=ALGEBRA_ABS_TOL, rel_tol=ALGEBRA_REL_TOL), path, 'algebraic inconsistency')


def ratio(actual, numerator, denominator, path):
    close(actual, numerator / denominator if denominator else None, path)


def json_tree(value, path='$', depth=0):
    need(depth <= 64, path, 'excessive JSON depth')
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float):
        number(value, path)
    elif type(value) is list:
        for index, child in enumerate(value): json_tree(child, path+'['+str(index)+']', depth+1)
    elif type(value) is dict:
        need(all(type(key) is str for key in value), path, 'JSON object keys must be strings')
        for key, child in value.items(): json_tree(child, path+'.'+key, depth+1)
    else:
        need(False, path, 'not a native JSON value')


def identity(value, frame=False, scene=True, path='identity'):
    keys = ['condition', 'arm'] + (['scene'] if scene else []) + (['frame'] if frame else [])
    fields(value, keys, path)
    for key in keys:
        if key == 'scene': integer(value[key], path+'.'+key)
        else: string(value[key], path+'.'+key)


def ids(value, path, minimum=0, maximum=None, strings=False):
    array(value, path)
    for item in value:
        if strings: string(item, path)
        else: integer(item, path, minimum, maximum)
    need(value == sorted(set(value)), path, 'identities must be ordered and unique')


def shape(value, path='shape'):
    array(value, path, 2)
    for item in value: integer(item, path, 1)
    return value[0] * value[1]


def point(value, dimensions, path):
    integer(value['row'], path+'.row', 0, dimensions[0]-1)
    integer(value['col'], path+'.col', 0, dimensions[1]-1)


def zero_work(value):
    fields(value, ZERO_KEYS, 'prohibited_work')
    for key in ZERO_KEYS: integer(value[key], 'prohibited_work.'+key, 0, 0)


def start(record, keys):
    json_tree(record)
    fields(record, keys, '$')
    integer(record['schema'], 'schema', 20, 20)
    zero_work(record['prohibited_work'])


def expected(actual, value, path):
    if value is not None:
        json_tree(value, 'expected.'+path)
        def same_types(a, b):
            if type(a) is not type(b): return False
            if type(a) is dict: return set(a) == set(b) and all(same_types(a[key], b[key]) for key in a)
            if type(a) is list: return len(a) == len(b) and all(same_types(x, y) for x, y in zip(a, b))
            return True
        need(same_types(actual, value), path, 'caller-supplied identity types differ')
        need(actual == value, path, 'differs from caller-supplied identity or extent')


def summary(value, path, quantiles=False, values=False, upper=None):
    keys = ['count', 'sum', 'mean'] + (['quantiles'] if quantiles else []) + (['values'] if values else [])
    fields(value, keys, path)
    count = integer(value['count'], path+'.count')
    total = number(value['sum'], path+'.sum', 0)
    if count == 0: need(total == 0, path+'.sum', 'empty sum must be exactly zero')
    if upper is not None: need(total <= count*upper+ALGEBRA_ABS_TOL, path+'.sum', 'sum exceeds bounded sample count')
    ratio(value['mean'], total, count, path+'.mean')
    if value['mean'] is not None: number(value['mean'], path+'.mean', 0, upper)
    if values:
        array(value['values'], path+'.values', count)
        for item in value['values']: number(item, path+'.values', 0, upper)
        close(total, math.fsum(value['values']), path+'.sum')
    if quantiles:
        fields(value['quantiles'], QKEYS, path+'.quantiles')
        ordered = []
        for key in QKEYS:
            q = value['quantiles'][key]
            if count:
                number(q, path+'.quantiles.'+key, 0, upper); ordered.append(q)
            else: need(q is None, path+'.quantiles.'+key, 'empty quantile must be null')
        need(ordered == sorted(ordered), path+'.quantiles', 'quantiles must be monotone')
        if count:
            need(ordered[0]-ALGEBRA_ABS_TOL <= value['mean'] <= ordered[-1]+ALGEBRA_ABS_TOL, path, 'mean outside extreme quantiles')
        if values:
            samples = sorted(value['values'])
            for key in QKEYS:
                h = (count-1)*float(key)
                lower, higher = math.floor(h), math.ceil(h)
                wanted = None if not count else samples[lower]+(samples[higher]-samples[lower])*(h-lower)
                close(value['quantiles'][key], wanted, path+'.quantiles.'+key)


def measure(value, pixels, path, frames=None):
    fields(value, ('count', 'full_pixels', 'sums', 'p_min', 'p_max')+(SCENE_MEANS if frames is not None else MEANS), path)
    count = integer(value['count'], path+'.count', 0, pixels)
    integer(value['full_pixels'], path+'.full_pixels', pixels, pixels)
    fields(value['sums'], SUM_KEYS, path+'.sums')
    for key, total in value['sums'].items():
        number(total, path+'.sums.'+key, 0)
        if not count: need(total == 0, path+'.sums.'+key, 'empty sum must be exactly zero')
        if key in ('p', 'target', 'target_squared', 'squared_error'):
            need(total <= count+ALGEBRA_ABS_TOL, path+'.sums.'+key, 'sum exceeds probability bound')
    sums = value['sums']
    need(sums['target_squared'] <= sums['target']+ALGEBRA_ABS_TOL, path, 'Y squared exceeds Y sum')
    for field, total in (('p_mean', sums['p']), ('target_mean', sums['target']), ('target_squared_mean', sums['target_squared']),
                         ('mse', sums['squared_error']), ('weighted_mse', 200*sums['squared_error']),
                         ('abs_logit_proxy_mean', sums['abs_logit_proxy']), ('squared_logit_proxy_mean', sums['squared_logit_proxy'])):
        ratio(value[field], total, count, path+'.'+field)
    ratio(value['full_mse_contribution'], sums['squared_error'], pixels, path+'.full_mse_contribution')
    if count:
        number(value['p_min'], path+'.p_min', 0, 1); number(value['p_max'], path+'.p_max', 0, 1)
        need(value['p_min'] <= value['p_max'], path, 'extrema reversed')
        need(value['p_min']-ALGEBRA_ABS_TOL <= value['p_mean'] <= value['p_max']+ALGEBRA_ABS_TOL, path, 'mean outside extrema')
    else:
        need(value['p_min'] is None and value['p_max'] is None, path, 'empty extrema must be null')
    if frames is not None:
        ratio(value['abs_logit_proxy_per_frame'], sums['abs_logit_proxy'], frames, path+'.abs_logit_proxy_per_frame')
        ratio(value['squared_logit_proxy_per_frame'], sums['squared_logit_proxy'], frames, path+'.squared_logit_proxy_per_frame')


def statistics(value, pixels, frames=None):
    fields(value, ('global_metrics', 'distance_regions', 'label_regions', 'unique_peak', 'instance_peak', 'unique_peak_fraction', 'constant_references'), 'statistics')
    measure(value['global_metrics'], pixels, 'statistics.global_metrics', frames)
    integer(value['global_metrics']['count'], 'global count', pixels, pixels)
    for group, keys in (('distance_regions', ('R0', 'R1', 'R2')), ('label_regions', ('background', 'foreground'))):
        fields(value[group], keys, 'statistics.'+group)
        for key in keys: measure(value[group][key], pixels, 'statistics.'+group+'.'+key, frames)
        need(sum(value[group][key]['count'] for key in keys) == pixels, group, 'partition count differs')
        for key in SUM_KEYS:
            close(math.fsum(value[group][part]['sums'][key] for part in keys), value['global_metrics']['sums'][key], group+'.'+key)
        nonempty = [value[group][key] for key in keys if value[group][key]['count']]
        need(min(part['p_min'] for part in nonempty) == value['global_metrics']['p_min'] and max(part['p_max'] for part in nonempty) == value['global_metrics']['p_max'], group, 'partition extrema differ from global extrema')
    for key in ('unique_peak', 'instance_peak'): summary(value[key], 'statistics.'+key, True, frames is None, 1)
    unique, instances = value['unique_peak']['count'], value['instance_peak']['count']
    need(unique <= instances and unique <= value['distance_regions']['R0']['count'], 'statistics', 'invalid peak counts')
    need((unique == 0) == (instances == 0), 'statistics', 'unique and instance emptiness differ')
    need(value['global_metrics']['sums']['target']+ALGEBRA_ABS_TOL >= unique, 'statistics', 'peak count exceeds target mass')
    for key in ('unique_peak', 'instance_peak'):
        if value[key]['count']:
            need(value['global_metrics']['p_min'] <= value[key]['quantiles']['0.0'] <= value[key]['quantiles']['1.0'] <= value['global_metrics']['p_max'], key, 'peak sample outside global extrema')
    ratio(value['unique_peak_fraction'], unique, pixels, 'statistics.unique_peak_fraction')
    fields(value['constant_references'], ('zero_mse', 'label_mean_constant_mse'), 'constant_references')
    constants = value['constant_references']; global_measure = value['global_metrics']
    close(constants['zero_mse'], global_measure['target_squared_mean'], 'constant_references.zero_mse')
    number(constants['label_mean_constant_mse'], 'constant_references.label_mean_constant_mse', 0, 1)
    need(constants['label_mean_constant_mse'] <= constants['zero_mse']+ALGEBRA_ABS_TOL, 'constant_references', 'variance exceeds second moment')
    if frames is None:
        close(constants['label_mean_constant_mse'], global_measure['target_squared_mean']-global_measure['target_mean']**2, 'constant_references.label_mean_constant_mse')
    if not instances:
        need(value['distance_regions']['R0']['count'] == value['distance_regions']['R1']['count'] == 0, 'statistics', 'empty centers have near pixels')


def representation(value, dimensions, stats):
    fields(value, ('low_ids', 'original_ids', 'vanished_ids', 'instances', 'small_instance_ids', 'outside_centroid_ids', 'coincident_centers', 'center_pairs_below_five'), 'representation')
    ids(value['low_ids'], 'low_ids', 1, 88)
    for key in ('small_instance_ids', 'outside_centroid_ids'): ids(value[key], key, 1, 88)
    if value['original_ids'] is None: need(value['vanished_ids'] is None, 'vanished_ids', 'unknown original labels require null')
    else:
        ids(value['original_ids'], 'original_ids', 1, 88); ids(value['vanished_ids'], 'vanished_ids', 1, 88)
        need(set(value['low_ids']) <= set(value['original_ids']), 'low_ids', 'unknown new instance')
        need(value['vanished_ids'] == sorted(set(value['original_ids'])-set(value['low_ids'])), 'vanished_ids', 'wrong difference')
    array(value['instances'], 'instances', len(value['low_ids']))
    by_id = {}; coordinates = {}
    for item in value['instances']:
        fields(item, ('id', 'row', 'col', 'pixels', 'centroid_inside'), 'representation.instance')
        integer(item['id'], 'instance.id', 1, 88); point(item, dimensions, 'instance')
        integer(item['pixels'], 'instance.pixels', 1, dimensions[0]*dimensions[1]); boolean(item['centroid_inside'], 'centroid_inside')
        by_id[item['id']] = item; coordinates.setdefault((item['row'], item['col']), []).append(item['id'])
    need(list(by_id) == value['low_ids'] and len(by_id) == len(value['instances']), 'instances', 'instance identity mismatch')
    need(sum(item['pixels'] for item in by_id.values()) == stats['label_regions']['foreground']['count'], 'instances', 'area does not equal foreground count')
    need(value['small_instance_ids'] == [i for i, item in by_id.items() if item['pixels'] < 20], 'small_instance_ids', 'wrong small-instance list')
    need(value['outside_centroid_ids'] == [i for i, item in by_id.items() if not item['centroid_inside']], 'outside_centroid_ids', 'wrong centroid list')
    groups = [dict(row=point[0], col=point[1], ids=members) for point, members in sorted(coordinates.items()) if len(members) > 1]
    array(value['coincident_centers'], 'coincident_centers')
    for item in value['coincident_centers']:
        fields(item, ('row', 'col', 'ids'), 'coincident_center'); point(item, dimensions, 'coincident_center'); ids(item['ids'], 'coincident.ids', 1, 88)
    need(value['coincident_centers'] == groups, 'coincident_centers', 'wrong coincidence groups')
    pairs = []
    for index, left in enumerate(value['instances']):
        for right in value['instances'][index+1:]:
            d2 = (left['row']-right['row'])**2+(left['col']-right['col'])**2
            if d2 < 25: pairs.append(dict(ids=[left['id'], right['id']], squared_distance=d2))
    array(value['center_pairs_below_five'], 'center_pairs_below_five')
    for item in value['center_pairs_below_five']:
        fields(item, ('ids', 'squared_distance'), 'close_pair'); ids(item['ids'], 'close_pair.ids', 1, 88); array(item['ids'], 'close_pair.ids', 2); integer(item['squared_distance'], 'close_pair.d2', 0, 24)
    need(value['center_pairs_below_five'] == pairs, 'center_pairs_below_five', 'wrong close-pair list')
    need(stats['unique_peak']['count'] == len(coordinates) and stats['instance_peak']['count'] == len(by_id), 'representation', 'peak denominator mismatch')
    return by_id, coordinates


def selected(item, dimensions, path, retained=False):
    fields(item, ('selected_center_index', 'row', 'col', 'score')+(('instance_id',) if retained else ()), path)
    integer(item['selected_center_index'], path+'.index', 0, 63); point(item, dimensions, path)
    number(item['score'], path+'.score', THRESHOLD_FLOAT32, 1)
    if retained: integer(item['instance_id'], path+'.instance_id', 1, 64)


def decoder(value, dimensions, stats):
    fields(value, ('trace', 'selected_centers', 'retained_instances'), 'decoder')
    trace = value['trace']; pixels = dimensions[0]*dimensions[1]
    fields(trace, TRACE_COUNTS+('cap_triggered', 'candidate_order', 'assigned_pixels', 'maximum_meets_center_threshold', 'center_threshold_float32'), 'decoder.trace')
    for key in TRACE_COUNTS: integer(trace[key], 'trace.'+key, 0, pixels)
    boolean(trace['cap_triggered'], 'cap_triggered'); boolean(trace['maximum_meets_center_threshold'], 'maximum_meets_center_threshold')
    number(trace['center_threshold_float32'], 'center_threshold_float32'); need(trace['center_threshold_float32'] == THRESHOLD_FLOAT32, 'threshold', 'different frozen float32 threshold')
    need(trace['maximum_meets_center_threshold'] == (trace['threshold_pixels'] > 0), 'threshold', 'inconsistent crossing flag')
    global_stats = stats['global_metrics']
    need(trace['maximum_meets_center_threshold'] == (global_stats['p_max'] >= THRESHOLD_FLOAT32), 'threshold', 'maximum and threshold count differ')
    need(trace['candidate_count'] <= trace['threshold_pixels'], 'trace', 'too many candidates')
    need(bool(trace['candidate_count']) == bool(trace['threshold_pixels']), 'trace', 'threshold crossing has no maximum candidate')
    need(trace['visited_count'] == trace['distance_rejected_count']+trace['selected_count'], 'trace', 'visited decomposition differs')
    need(trace['candidate_count'] == trace['visited_count']+trace['unvisited_count'], 'trace', 'candidate decomposition differs')
    integer(trace['selected_count'], 'selected_count', 0, 64)
    need(trace['cap_triggered'] == (trace['selected_count'] == 64), 'cap_triggered', 'break flag differs')
    if not trace['cap_triggered']: need(trace['unvisited_count'] == 0, 'unvisited_count', 'uncapped traversal incomplete')
    array(trace['candidate_order'], 'candidate_order', trace['candidate_count'])
    candidates = []; seen = set()
    for item in trace['candidate_order']:
        array(item, 'candidate', 3); integer(item[0], 'candidate.row', 0, dimensions[0]-1); integer(item[1], 'candidate.col', 0, dimensions[1]-1); number(item[2], 'candidate.score', THRESHOLD_FLOAT32, 1)
        need(tuple(item[:2]) not in seen, 'candidate', 'duplicate position'); seen.add(tuple(item[:2])); candidates.append(item)
    need(candidates == sorted(candidates, key=lambda p: (-p[2], p[0], p[1])), 'candidate_order', 'unstable order')
    if candidates: need(candidates[0][2] == global_stats['p_max'], 'candidate_order', 'first candidate is not the global maximum')
    array(value['selected_centers'], 'selected_centers', trace['selected_count'])
    for index, item in enumerate(value['selected_centers']):
        selected(item, dimensions, 'selected'); integer(item['selected_center_index'], 'selected.index', index, index)
    # Replay only the persisted candidate list: this does not verify real local maxima.
    picked = []
    for candidate in candidates[:trace['visited_count']]:
        if all((candidate[0]-other[0])**2+(candidate[1]-other[1])**2 >= 25 for other in picked): picked.append(candidate)
        need(len(picked) <= 64, 'trace', 'cap was traversed past')
    need([[item['row'], item['col'], item['score']] for item in value['selected_centers']] == picked, 'selected_centers', 'selection contradicts persisted traversal')
    if trace['cap_triggered']: need(bool(candidates) and picked[-1] == candidates[trace['visited_count']-1], 'cap', 'did not stop at 64th selection')
    array(trace['assigned_pixels'], 'assigned_pixels', trace['selected_count'])
    for count in trace['assigned_pixels']: integer(count, 'assigned_pixels', 0, pixels)
    need(sum(trace['assigned_pixels']) <= pixels, 'assigned_pixels', 'too many assigned pixels')
    need(trace['zero_assignment_count'] == trace['assigned_pixels'].count(0), 'zero_assignment_count', 'wrong zero count')
    retained = [dict(instance_id=index+1, **value['selected_centers'][selected_index]) for index, selected_index in enumerate(i for i, n in enumerate(trace['assigned_pixels']) if n >= 20)]
    array(value['retained_instances'], 'retained_instances', trace['retained_count'])
    for item in value['retained_instances']: selected(item, dimensions, 'retained', True)
    need(value['retained_instances'] == retained, 'retained_instances', 'area filtering or renumbering differs')


def matching(value, path, gt, predictions, radius, frame=True):
    counts = ('gt_count', 'predicted_count', 'matched_count', 'unmatched_gt_count', 'unmatched_prediction_count')
    fields(value, counts+('recall', 'precision', 'matched_distances')+(('radius', 'pairs', 'objective') if frame else ()), path)
    for key in counts: integer(value[key], path+'.'+key)
    g, d, m = value['gt_count'], value['predicted_count'], value['matched_count']
    need(g == (len(gt) if frame else gt) and d == (len(predictions) if frame else predictions) and m <= min(g, d), path, 'matching denominator or capacity differs')
    need(value['unmatched_gt_count'] == g-m and value['unmatched_prediction_count'] == d-m, path, 'unmatched count differs')
    ratio(value['recall'], m, g, path+'.recall'); ratio(value['precision'], m, d, path+'.precision')
    summary(value['matched_distances'], path+'.matched_distances', values=frame, upper=radius)
    integer(value['matched_distances']['count'], path+'.distance_count', m, m)
    if frame:
        number(value['radius'], path+'.radius'); need(value['radius'] == radius and value['objective'] == OBJECTIVE, path, 'wrong radius or objective')
        array(value['pairs'], path+'.pairs', m); left = []; right = []; distances = []
        for pair in value['pairs']:
            fields(pair, ('gt_id', 'selected_center_index', 'squared_distance', 'distance'), path+'.pair')
            integer(pair['gt_id'], path+'.gt_id'); integer(pair['selected_center_index'], path+'.selected_center_index')
            need(pair['gt_id'] in gt and pair['selected_center_index'] in predictions, path, 'unknown matched endpoint')
            a, b = gt[pair['gt_id']], predictions[pair['selected_center_index']]
            d2 = (a[0]-b[0])**2+(a[1]-b[1])**2
            need(d2 <= radius**2, path, 'edge outside radius'); close(pair['squared_distance'], d2, path+'.d2'); close(pair['distance'], math.sqrt(d2), path+'.distance')
            left.append(pair['gt_id']); right.append(pair['selected_center_index']); distances.append(pair['distance'])
        need(left == sorted(set(left)) and len(right) == len(set(right)), path, 'duplicate or unordered matching endpoints')
        need(value['matched_distances']['values'] == distances, path, 'distance list differs')


def locations(value, dimensions, by_id, coordinates, decoded, stats):
    fields(value, ('instances', 'global_maximum', 'matches'), 'locations')
    array(value['instances'], 'locations.instances', len(by_id)); probabilities = {}
    for position, item in enumerate(value['instances']):
        fields(item, ('id', 'row', 'col', 'probability', 'r5', 'r10'), 'locations.instance')
        integer(item['id'], 'location.id', 1, 88); need(item['id'] == list(by_id)[position], 'location.id', 'wrong GT order')
        point(item, dimensions, 'location'); source = by_id[item['id']]
        need((item['row'], item['col']) == (source['row'], source['col']), 'location', 'GT coordinates differ')
        number(item['probability'], 'location.probability', 0, 1)
        coord = (item['row'], item['col'])
        if coord in probabilities: need(probabilities[coord] == item['probability'], 'location', 'shared coordinate probabilities differ')
        probabilities[coord] = item['probability']
        for radius in (5, 10):
            local = item['r'+str(radius)]; fields(local, ('probability', 'row', 'col', 'distance'), 'local'); point(local, dimensions, 'local')
            number(local['probability'], 'local.probability', item['probability'], stats['global_metrics']['p_max'])
            distance = math.hypot(local['row']-item['row'], local['col']-item['col'])
            need(distance <= radius, 'local', 'point outside circular neighborhood'); close(local['distance'], distance, 'local.distance')
        need(item['r10']['probability'] >= item['r5']['probability'], 'local', 'larger neighborhood maximum decreased')
    need(stats['instance_peak']['values'] == [item['probability'] for item in value['instances']], 'instance_peak', 'probability list differs')
    need(stats['unique_peak']['values'] == [probabilities[coord] for coord in sorted(coordinates)], 'unique_peak', 'probability list differs')
    top = value['global_maximum']; fields(top, ('row', 'col', 'probability', 'nearest_gt_distance'), 'global_maximum'); point(top, dimensions, 'global_maximum')
    close(top['probability'], stats['global_metrics']['p_max'], 'global_maximum.probability')
    nearest = min((math.hypot(top['row']-r, top['col']-c) for r, c in coordinates), default=None)
    close(top['nearest_gt_distance'], nearest, 'global_maximum.nearest_gt_distance')
    fields(value['matches'], MATCH_KEYS, 'matches')
    pred = {p['selected_center_index']: (p['row'], p['col']) for p in decoded['selected_centers']}
    gt = {i: (p['row'], p['col']) for i, p in by_id.items()}; unique = dict(enumerate(sorted(coordinates)))
    for key in MATCH_KEYS: matching(value['matches'][key], key, gt if key.startswith('instance') else unique, pred, int(key.split('_r')[1]))


def validate_frame(record, *, expected_identity=None, expected_shape=None):
    start(record, ('schema', 'status', 'identity', 'shape', 'pixel_count', 'representation', 'decoder', 'statistics', 'locations', 'prohibited_work'))
    need(record['status'] == 'unfrozen_component_result', 'status', 'only current preparation status is supported')
    identity(record['identity'], frame=True); expected(record['identity'], expected_identity, 'identity')
    pixels = shape(record['shape']); expected(record['shape'], expected_shape, 'shape'); integer(record['pixel_count'], 'pixel_count', pixels, pixels)
    statistics(record['statistics'], pixels)
    by_id, coordinates = representation(record['representation'], record['shape'], record['statistics'])
    decoder(record['decoder'], record['shape'], record['statistics'])
    locations(record['locations'], record['shape'], by_id, coordinates, record['decoder'], record['statistics'])
    return record


def validate_scene(record, *, expected_identity=None, expected_shape=None, expected_frame_ids=None):
    start(record, ('schema', 'identity', 'frame_count', 'frame_ids', 'shape', 'pixel_count', 'statistics', 'decoder', 'locations', 'matches', 'prohibited_work'))
    identity(record['identity']); expected(record['identity'], expected_identity, 'identity')
    frame_count = integer(record['frame_count'], 'frame_count', 1); ids(record['frame_ids'], 'frame_ids', strings=True); array(record['frame_ids'], 'frame_ids', frame_count)
    expected(record['frame_ids'], expected_frame_ids, 'frame_ids')
    pixels = shape(record['shape'])*frame_count; expected(record['shape'], expected_shape, 'shape'); integer(record['pixel_count'], 'pixel_count', pixels, pixels)
    stats = record['statistics']; statistics(stats, pixels, frame_count)
    decoded = record['decoder']; fields(decoded, ('counts', 'per_frame', 'cap_frame_count', 'cap_frame_fraction'), 'decoder')
    fields(decoded['counts'], TRACE_COUNTS, 'decoder.counts'); fields(decoded['per_frame'], TRACE_COUNTS, 'decoder.per_frame')
    counts = decoded['counts']
    for key in TRACE_COUNTS: integer(counts[key], 'counts.'+key, 0, pixels); ratio(decoded['per_frame'][key], counts[key], frame_count, 'per_frame.'+key)
    need(counts['candidate_count'] <= counts['threshold_pixels'] and counts['selected_count'] <= 64*frame_count, 'counts', 'candidate or cap count exceeded')
    need(counts['visited_count'] == counts['distance_rejected_count']+counts['selected_count'] and counts['candidate_count'] == counts['visited_count']+counts['unvisited_count'], 'counts', 'traversal sums differ')
    need(counts['zero_assignment_count']+counts['retained_count'] <= counts['selected_count'], 'counts', 'retained and zero-assignment overlap')
    cap = integer(decoded['cap_frame_count'], 'cap_frame_count', 0, frame_count)
    need(64*cap <= counts['selected_count'] <= 64*cap+63*(frame_count-cap), 'counts', 'cap frame count contradicts selected total')
    if cap == 0: need(counts['unvisited_count'] == 0, 'counts', 'uncapped scenes have unvisited candidates')
    ratio(decoded['cap_frame_fraction'], cap, frame_count, 'cap_frame_fraction')
    fields(record['locations'], LOCATION_KEYS, 'locations'); gt = stats['instance_peak']['count']
    need(gt <= 88*frame_count, 'statistics.instance_peak', 'GT count exceeds label-ID capacity')
    for key in LOCATION_KEYS:
        upper = 1 if key.endswith('probability') else (5 if key == 'r5_distance' else 10 if key == 'r10_distance' else math.hypot(record['shape'][0]-1, record['shape'][1]-1))
        summary(record['locations'][key], 'locations.'+key, upper=upper)
        count = record['locations'][key]['count']
        need(count <= frame_count if key == 'global_maximum_nearest_gt_distance' else count == gt, 'locations.'+key, 'wrong sample denominator')
        if key == 'global_maximum_nearest_gt_distance':
            need((count == 0) == (gt == 0) and count <= gt, 'locations.'+key, 'effective GT frame count differs')
    for key in ('count', 'sum', 'mean'): close(record['locations']['center_probability'][key], stats['instance_peak'][key], 'locations.center_probability.'+key)
    fields(record['matches'], MATCH_KEYS, 'matches')
    for key in MATCH_KEYS:
        matching(record['matches'][key], key, gt if key.startswith('instance') else stats['unique_peak']['count'], counts['selected_count'], int(key.split('_r')[1]), False)
    for unit in ('instance', 'unique'):
        need(record['matches'][unit+'_r5']['matched_count'] <= record['matches'][unit+'_r10']['matched_count'], 'matches', 'larger radius reduced maximum cardinality')
    return record


def metric_names():
    groups = ['global']+['distance_regions.'+r for r in ('R0', 'R1', 'R2')]+['label_regions.'+r for r in ('background', 'foreground')]
    names = [group+'.'+key for group in groups for key in SCENE_MEANS]
    names += ['unique_peak_fraction', 'constant_references.zero_mse', 'constant_references.label_mean_constant_mse']
    for group in ('unique_peak', 'instance_peak'): names += [group+'.mean']+[group+'.mean_of_scene_quantile.'+q for q in QKEYS]
    names += ['decoder.per_frame.'+key for key in TRACE_COUNTS]+['decoder.cap_frame_fraction']
    names += ['locations.'+key+'.mean' for key in LOCATION_KEYS]
    names += ['matches.'+key+'.'+metric for key in MATCH_KEYS for metric in ('recall', 'precision', 'matched_distance_mean')]
    return names


def macro(value, scene_ids, path, paired=False):
    keys = ('mean', 'valid_scene_count', 'total_scene_count', 'valid_scenes', 'null_scenes')+(('valid_pair_count', 'total_pair_count') if paired else ())
    fields(value, keys, path)
    count = integer(value['valid_scene_count'], path+'.valid_scene_count', 0, len(scene_ids))
    integer(value['total_scene_count'], path+'.total_scene_count', len(scene_ids), len(scene_ids))
    ids(value['valid_scenes'], path+'.valid_scenes'); ids(value['null_scenes'], path+'.null_scenes')
    need(len(value['valid_scenes']) == count and set(value['valid_scenes']).isdisjoint(value['null_scenes']) and sorted(value['valid_scenes']+value['null_scenes']) == scene_ids, path, 'invalid valid/null partition')
    if count: number(value['mean'], path+'.mean', None if paired else 0)
    else: need(value['mean'] is None, path, 'empty macro must be null')
    if paired: integer(value['valid_pair_count'], path+'.valid_pair_count', count, count); integer(value['total_pair_count'], path+'.total_pair_count', len(scene_ids), len(scene_ids))


def unconditional_metric(key):
    return key.startswith(('global.', 'decoder.', 'constant_references.')) or key == 'unique_peak_fraction' or key.endswith(('full_mse_contribution', 'abs_logit_proxy_per_frame', 'squared_logit_proxy_per_frame'))


def metric_upper(key):
    if key.endswith(('weighted_mse',)): return 200
    if key.endswith(('p_mean', 'target_mean', 'target_squared_mean', 'mse', 'full_mse_contribution', 'recall', 'precision')) or key.startswith(('unique_peak', 'instance_peak', 'constant_references.')) or key == 'decoder.cap_frame_fraction' or 'probability.mean' in key: return 1
    if key.startswith('decoder.per_frame.') and key.rsplit('.', 1)[1] in ('selected_count', 'zero_assignment_count', 'retained_count'): return 64
    if key == 'locations.r5_distance.mean' or (key.startswith('matches.') and '_r5.matched_distance_mean' in key): return 5
    if key == 'locations.r10_distance.mean' or (key.startswith('matches.') and '_r10.matched_distance_mean' in key): return 10
    return None


def matched_validity_subsets(metrics, path):
    for key in MATCH_KEYS:
        distance = set(metrics['matches.'+key+'.matched_distance_mean']['valid_scenes'])
        recall = set(metrics['matches.'+key+'.recall']['valid_scenes'])
        precision = set(metrics['matches.'+key+'.precision']['valid_scenes'])
        need(distance <= recall & precision, path+'.matches.'+key, 'matched distance requires both endpoint denominators')


def paired_shared_denominators(values, path):
    conditional = ('p_mean', 'target_mean', 'target_squared_mean', 'mse', 'weighted_mse', 'abs_logit_proxy_mean', 'squared_logit_proxy_mean')
    groups = ['global']+['distance_regions.'+r for r in ('R0', 'R1', 'R2')]+['label_regions.'+r for r in ('background', 'foreground')]
    for group in groups:
        valid = values[group+'.mse'] is not None
        need(all((values[group+'.'+key] is not None) == valid for key in conditional), path+'.'+group, 'conditional paired denominators differ')
    gt_keys = []
    for group in ('instance_peak', 'unique_peak'):
        gt_keys += [group+'.mean']+[group+'.mean_of_scene_quantile.'+q for q in QKEYS]
    gt_keys += ['locations.'+key+'.mean' for key in LOCATION_KEYS]
    gt_keys += ['matches.'+key+'.recall' for key in MATCH_KEYS]
    gt_valid = values['instance_peak.mean'] is not None
    need(all((values[key] is not None) == gt_valid for key in gt_keys), path, 'paired GT-presence denominators differ')
    prediction_valid = values['matches.instance_r5.precision'] is not None
    need(all((values['matches.'+key+'.precision'] is not None) == prediction_valid for key in MATCH_KEYS), path, 'paired prediction-presence denominators differ')
    for key in MATCH_KEYS:
        if values['matches.'+key+'.matched_distance_mean'] is not None:
            need(gt_valid and prediction_valid, path+'.matches.'+key, 'paired distance requires both endpoint denominators')


def validate_cohort(record, *, expected_identity=None, expected_scene_ids=None):
    start(record, ('schema', 'condition', 'arm', 'scene_ids', 'scene_count', 'frame_count', 'macro', 'aggregation', 'prohibited_work'))
    cohort_identity = dict(condition=record['condition'], arm=record['arm']); identity(cohort_identity, scene=False); expected(cohort_identity, expected_identity, 'identity')
    ids(record['scene_ids'], 'scene_ids'); count = integer(record['scene_count'], 'scene_count', 1); array(record['scene_ids'], 'scene_ids', count)
    expected(record['scene_ids'], expected_scene_ids, 'scene_ids'); integer(record['frame_count'], 'frame_count', count)
    need(record['aggregation'] == AGGREGATION, 'aggregation', 'different aggregation definition')
    fields(record['macro'], metric_names(), 'macro')
    for key, value in record['macro'].items():
        macro(value, record['scene_ids'], 'macro.'+key)
        if unconditional_metric(key): need(value['valid_scenes'] == record['scene_ids'], 'macro.'+key, 'unconditional metric cannot be null')
        if value['mean'] is not None: number(value['mean'], 'macro.'+key+'.mean', 0, metric_upper(key))
    for group in ['global']+['distance_regions.'+r for r in ('R0', 'R1', 'R2')]+['label_regions.'+r for r in ('background', 'foreground')]:
        raw, weighted = record['macro'][group+'.mse'], record['macro'][group+'.weighted_mse']
        for field in ('p_mean', 'target_mean', 'target_squared_mean', 'abs_logit_proxy_mean', 'squared_logit_proxy_mean'):
            need(record['macro'][group+'.'+field]['valid_scenes'] == raw['valid_scenes'], 'macro.'+group, 'conditional denominator identities differ')
        need(raw['valid_scenes'] == weighted['valid_scenes'], 'macro', 'weighted metric validity differs')
        close(weighted['mean'], None if raw['mean'] is None else 200*raw['mean'], 'macro.'+group+'.weighted_mse')
    for group in ('distance_regions', 'label_regions'):
        regions = ('R0', 'R1', 'R2') if group == 'distance_regions' else ('background', 'foreground')
        close(math.fsum(record['macro'][group+'.'+region+'.full_mse_contribution']['mean'] for region in regions), record['macro']['global.mse']['mean'], 'macro.'+group+'.contributions')
    gt_valid = record['macro']['instance_peak.mean']['valid_scenes']
    for group in ('unique_peak', 'instance_peak'):
        keys = [group+'.mean']+[group+'.mean_of_scene_quantile.'+q for q in QKEYS]
        for key in keys: need(record['macro'][key]['valid_scenes'] == gt_valid, 'macro.'+key, 'peak denominator identities differ')
        values = [record['macro'][group+'.mean_of_scene_quantile.'+q]['mean'] for q in QKEYS]
        if gt_valid: need(values == sorted(values), 'macro.'+group, 'scene quantile means are not monotone')
    for key in LOCATION_KEYS: need(record['macro']['locations.'+key+'.mean']['valid_scenes'] == gt_valid, 'macro.locations', 'GT-presence validity differs')
    prediction_valid = record['macro']['matches.instance_r5.precision']['valid_scenes']
    for key in MATCH_KEYS:
        need(record['macro']['matches.'+key+'.recall']['valid_scenes'] == gt_valid, 'macro.matches', 'recall GT denominator differs')
        need(record['macro']['matches.'+key+'.precision']['valid_scenes'] == prediction_valid, 'macro.matches', 'precision prediction denominator differs')
    matched_validity_subsets(record['macro'], 'macro')
    close(record['macro']['constant_references.zero_mse']['mean'], record['macro']['global.target_squared_mean']['mean'], 'macro.zero_reference')
    return record


def validate_paired(record, *, expected_identity=None, expected_scene_ids=None):
    start(record, ('schema', 'direction', 'scene_ids', 'left_identity', 'right_identity', 'by_scene', 'paired_macro', 'prohibited_work'))
    need(record['direction'] == 'right_minus_left', 'direction', 'different paired sign')
    for key in ('left_identity', 'right_identity'): identity(record[key], scene=False, path=key)
    expected({key: record[key] for key in ('left_identity', 'right_identity')}, expected_identity, 'identity')
    ids(record['scene_ids'], 'scene_ids'); need(bool(record['scene_ids']), 'scene_ids', 'empty paired cohort'); expected(record['scene_ids'], expected_scene_ids, 'scene_ids')
    fields(record['by_scene'], [str(scene) for scene in record['scene_ids']], 'by_scene'); fields(record['paired_macro'], metric_names(), 'paired_macro')
    for scene, values in record['by_scene'].items():
        fields(values, metric_names(), 'by_scene.'+scene)
        for key, value in values.items():
            if value is not None:
                upper = metric_upper(key)
                number(value, 'by_scene.'+scene+'.'+key, -upper if upper is not None else None, upper)
            if unconditional_metric(key): need(value is not None, 'by_scene.'+scene+'.'+key, 'unconditional paired difference cannot be null')
        paired_shared_denominators(values, 'by_scene.'+scene)
        for group in ['global']+['distance_regions.'+r for r in ('R0', 'R1', 'R2')]+['label_regions.'+r for r in ('background', 'foreground')]:
            raw = values[group+'.mse']; close(values[group+'.weighted_mse'], None if raw is None else 200*raw, 'by_scene.'+scene+'.'+group+'.weighted_mse')
    for key, value in record['paired_macro'].items():
        macro(value, record['scene_ids'], 'paired_macro.'+key, True)
        if value['mean'] is not None:
            upper = metric_upper(key)
            number(value['mean'], 'paired_macro.'+key+'.mean', -upper if upper is not None else None, upper)
        present = [(scene, record['by_scene'][str(scene)][key]) for scene in record['scene_ids'] if record['by_scene'][str(scene)][key] is not None]
        need(value['valid_scenes'] == [scene for scene, _ in present], 'paired_macro.'+key, 'effective paired identities differ')
        ratio(value['mean'], math.fsum(v for _, v in present), len(present), 'paired_macro.'+key+'.mean')
    matched_validity_subsets(record['paired_macro'], 'paired_macro')
    return record


def loads_strict(text, kind, **expectations):
    """Parse native JSON text/bytes; reject repeated keys and nonstandard numbers."""
    need(type(text) in (str, bytes, bytearray), '$', 'expected JSON text')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, '$.'+key, 'duplicate JSON object key'); result[key] = value
        return result
    def constant(value): raise ValueError('Nonstandard JSON number: '+value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    validators = {'frame': validate_frame, 'scene': validate_scene, 'cohort': validate_cohort, 'paired': validate_paired}
    need(kind in validators, 'kind', 'unknown record kind')
    return validators[kind](value, **expectations)
