"""Independent arithmetic audit of fixed-support observed collision diagnostics."""
import argparse
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import local_risk_common_v15 as storage
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAMES = ('0016', '0080', '0160', '0240')
REGIONS = ('global', 'left_finger', 'right_finger', 'bottom', 'approach')
ARRAYS = ('observed_points_float64.npy', 'valid_pixel_indices.npy', 'voxel_points.npy')
PARENT_BINARY = ('valid_points.npy', 'valid_pixel_indices.npy', 'sampled_valid_indices.npy', 'sampled_points.npy',
                 'sampled_pixel_indices.npy', 'forward_bundle.npz', 'candidates.npy', 'seed_provenance.npz')
FIGURE = 'scene_0003_local_observed_collision.png'
TOLERANCE = 1e-12
require = storage.require


def reconstruct_observation(depth, intrinsic, factor):
    require(depth.ndim == 2 and np.issubdtype(depth.dtype, np.number), 'Invalid source depth image')
    camera = np.asarray(intrinsic, np.float64)
    require(camera.shape == (3, 3) and np.isfinite(camera).all() and camera[0, 0] > 0 and camera[1, 1] > 0
            and math.isfinite(factor) and factor > 0, 'Invalid source calibration')
    z = depth.astype(np.float64)/factor
    pixel = np.arange(depth.size, dtype=np.int64)
    good = np.isfinite(z.reshape(-1)) & (z.reshape(-1) > 0) & (z.reshape(-1) <= 2.)
    selected = pixel[good]
    row, col = selected//depth.shape[1], selected % depth.shape[1]
    d = z.reshape(-1)[good]
    points = np.column_stack(((col-camera[0, 2])*d/camera[0, 0], (row-camera[1, 2])*d/camera[1, 1], d))
    return points, selected


def independent_voxel_centers(observed, cached, voxel=.01):
    require(observed.dtype == cached.dtype == np.float64 and observed.shape == (len(observed), 3)
            and cached.shape == (len(cached), 3) and np.isfinite(observed).all() and np.isfinite(cached).all(),
            'Invalid observation or voxel cache')
    if not len(observed):
        require(len(cached) == 0, 'Empty observation acquired voxels')
        return {'voxel_count': 0, 'maximum_absolute_center_difference_m': 0., 'absolute_tolerance_m': TOLERANCE,
                'internal_native_accumulation_order_reproduced': False}
    origin = observed.min(axis=0)-voxel*.5
    grid = np.floor((observed-origin)/voxel).astype(np.int64)
    keys, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(keys), 3), np.float64)
    np.add.at(sums, inverse, observed)
    centers = sums/counts[:, None]
    require(len(cached) == len(keys), 'Voxel count differs from independent grid grouping')
    cache_keys = np.floor((cached-origin)/voxel).astype(np.int64)
    order = np.lexsort((cache_keys[:, 2], cache_keys[:, 1], cache_keys[:, 0]))
    require(np.array_equal(cache_keys[order], keys), 'Voxel cache grid membership differs')
    error = float(np.max(np.abs(cached[order]-centers)))
    require(np.allclose(cached[order], centers, atol=TOLERANCE, rtol=0), 'Voxel centers differ from independently reconstructed means')
    return {'voxel_count': len(keys), 'maximum_absolute_center_difference_m': error, 'absolute_tolerance_m': TOLERANCE,
            'internal_native_accumulation_order_reproduced': False}


def independent_regions(candidate, voxel_points):
    """Separate one-candidate region arithmetic; no detector, group, or block call."""
    require(candidate.shape == (17,) and candidate.dtype == np.float32 and np.isfinite(candidate).all()
            and candidate[1] >= 0 and candidate[2] > 0 and candidate[3] > 0, 'Invalid unchanged candidate')
    require(voxel_points.dtype == np.float64 and voxel_points.shape == (len(voxel_points), 3)
            and np.isfinite(voxel_points).all(), 'Invalid cached voxel points')
    # One-element float32 parameter arrays retain the frozen decoder/API arithmetic.
    width, height, depth = candidate[1:2], candidate[2:3], candidate[3:4]
    coordinates = (voxel_points-candidate[13:16]) @ candidate[4:13].reshape(3, 3)
    x, y, z = coordinates.T
    in_height = (z > -height/2) & (z < height/2)
    along_finger = (x > depth-.06) & (x < depth)
    outer_left, inner_left = y > -(width/2+.01), y < -width/2
    outer_right, inner_right = y < width/2+.01, y > width/2
    at_bottom = (x <= depth-.06) & (x > depth-.06-.01)
    at_approach = (x <= depth-.06-.01) & (x > depth-.06-.01-.05)
    masks = [in_height & along_finger & outer_left & inner_left,
             in_height & along_finger & outer_right & inner_right,
             in_height & outer_left & outer_right & at_bottom,
             in_height & outer_left & outer_right & at_approach]
    union = masks[0] | masks[1] | masks[2] | masks[3]
    counts = dict(zip(REGIONS, [int(np.count_nonzero(mask)) for mask in [union]+masks]))
    require(counts['global'] == sum(counts[name] for name in REGIONS[1:]), 'Disjoint gripper region accounting differs')
    side = height*.06*.01/(.01**3)
    bottom = height*(width+2*.01)*.01/(.01**3)
    approach = height*(width+2*.01)*.05/(.01**3)
    volume = side*2+bottom+approach
    volumes = dict(zip(REGIONS, (volume, side, side, bottom, approach)))
    scores = {name: counts[name]/float((value+1e-6)[0]) for name, value in volumes.items()}
    inner = in_height & along_finger & (~inner_left) & (~inner_right)
    inner_count = int(np.count_nonzero(inner))
    inner_volume = float((height*.06*width/(.01**3))[0])
    applicable = bool(width[0] > 0)
    inner_score = inner_count/inner_volume if applicable else None
    return {'collision': scores['global'] > .01, 'empty_diagnostic': inner_score < .01 if applicable else None,
        'empty_diagnostic_applicable': applicable, 'empty_diagnostic_reason': None if applicable else 'zero_width',
        'occupied_volume_scores': scores, 'region_point_counts': counts,
        'region_voxelized_volumes': {name: float(value[0]) for name, value in volumes.items()},
        'inner_point_count': inner_count, 'inner_voxelized_volume': inner_volume, 'inner_occupied_volume_score': inner_score}


def compare_feature_row(row, frame, index, candidate, expected):
    fields = {'frame', 'original_index', 'raw_confidence', 'candidate_row_sha256', 'collision',
              'empty_diagnostic', 'empty_diagnostic_applicable', 'empty_diagnostic_reason', 'occupied_volume_scores'}
    require(set(row) == fields and row['frame'] == frame and type(row['original_index']) is int and row['original_index'] == index,
            'Feature row identity or schema differs')
    require(row['raw_confidence'] == float(candidate[0])
            and row['candidate_row_sha256'] == hashlib.sha256(candidate.tobytes()).hexdigest(), 'Original candidate row or raw confidence changed')
    for key in ('collision', 'empty_diagnostic', 'empty_diagnostic_applicable', 'empty_diagnostic_reason'):
        require(row[key] == expected[key] and type(row[key]) is type(expected[key]), 'Independent feature diagnostic differs: '+key)
    scores = row['occupied_volume_scores']
    require(set(scores) == set(REGIONS) and all(type(value) in (float, int) and math.isfinite(value) and value >= 0
            and abs(value-expected['occupied_volume_scores'][name]) <= TOLERANCE for name, value in scores.items()),
            'Independent region occupancy differs')


def fraction_value(value=None, reason=None):
    if value is None:
        require(isinstance(reason, str), 'Undefined ratio needs a reason')
        return {'numerator': None, 'denominator': None, 'value': None, 'reason': reason}
    return {'numerator': value.numerator, 'denominator': value.denominator, 'value': float(value), 'reason': None}


def independent_metrics(trace):
    require(all(type(row['collision']) is bool and type(row['predicted_collision']) is bool
                and math.isfinite(row['global_iou']) and row['global_iou'] >= 0
                and row['predicted_collision'] == (row['global_iou'] > .01) for row in trace), 'Invalid metric event/threshold')
    tp = sum(row['collision'] and row['predicted_collision'] for row in trace)
    fn = sum(row['collision'] and not row['predicted_collision'] for row in trace)
    fp = sum(not row['collision'] and row['predicted_collision'] for row in trace)
    tn = sum(not row['collision'] and not row['predicted_collision'] for row in trace)
    positive = [row['global_iou'] for row in trace if row['collision']]
    negative = [row['global_iou'] for row in trace if not row['collision']]
    empty_reason = 'no_reference_support' if not trace else None
    tpr = fraction_value(Fraction(tp, tp+fn)) if positive else fraction_value(reason=empty_reason or 'no_positive_reference')
    fpr = fraction_value(Fraction(fp, fp+tn)) if negative else fraction_value(reason=empty_reason or 'no_negative_reference')
    wins = ties = losses = 0
    for yes in positive:
        for no in negative:
            if yes > no:
                wins += 1
            elif yes == no:
                ties += 1
            else:
                losses += 1
    pairs = len(positive)*len(negative)
    auroc = fraction_value(Fraction(2*wins+ties, 2*pairs)) if pairs else fraction_value(
        reason=empty_reason or ('no_positive_reference' if not positive else 'no_negative_reference'))
    auroc.update(pair_count=pairs, wins=wins, ties=ties, losses=losses)
    return {'support_count': len(trace), 'positive_reference_count': len(positive), 'negative_reference_count': len(negative),
            'confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, 'metrics': {'tpr': tpr, 'fpr': fpr, 'auroc': auroc}}


def equal_frame_metrics(frames):
    result = {}
    for metric in ('tpr', 'fpr', 'auroc'):
        eligible = [frame for frame in FRAMES if frames[frame]['metrics'][metric]['value'] is not None]
        excluded = {frame: frames[frame]['metrics'][metric]['reason'] for frame in FRAMES if frame not in eligible}
        values = [Fraction(frames[frame]['metrics'][metric]['numerator'], frames[frame]['metrics'][metric]['denominator']) for frame in eligible]
        mean = fraction_value(sum(values, Fraction(0))/len(values)) if values else fraction_value(reason='no_eligible_frames')
        result[metric] = {'eligible_frames': eligible, 'excluded_frames': excluded, 'eligible_frame_count': len(eligible), 'mean': mean}
    return result


def verify_inventory(directory, names):
    names = set(names)
    actual = set()
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'Link in sealed inventory')
        name = path.relative_to(directory).as_posix()
        if path.is_dir():
            require(any(item.startswith(name+'/') for item in names), 'Unexpected sealed subdirectory')
        else:
            require(path.is_file(), 'Invalid sealed filesystem object')
            actual.add(name)
    require(actual == names, 'Sealed inventory differs')


def payload_inventory(payloads):
    return {'binding.json'} | {value for name in payloads for value in (name, name+'.sha256.json')}


def verify_manifest(directory, binding, names, schema, frame):
    owner = storage if schema == 15 else storage.storage
    verify_inventory(directory, names)
    completion = owner.load_commit(directory/'completion.json', binding)
    require(completion['schema'] == schema and completion['status'] == 'completed' and completion['frame'] == frame
            and completion['binding_fingerprint'] == binding['fingerprint'], 'Wrong completed frame identity')
    expected = set(names)-{'completion.json', 'completion.json.sha256.json'}
    require(set(completion['outputs']) == expected and all(storage.digest(directory/name) == checksum
            for name, checksum in completion['outputs'].items()), 'Completion manifest differs')
    require(completion['report_sha256'] == storage.digest(directory/'report.json'), 'Completed report checksum differs')
    return completion


def parent_binding(binding, protocol, frame, group):
    plain = {key: value for key, value in binding.items() if key != 'fingerprint'}
    require(binding['fingerprint'] == storage.fingerprint(plain) and binding['schema'] == 14
            and binding['frame'] == frame and binding['input_group'] == group
            and binding['protocol_sha256'] == protocol['shared_inputs']['configs/grasp_multiview_v14r1.json'], 'Wrong parent source binding')


def inspect_parent_generation(root, protocol, frame):
    directory = storage.safe_path(root, protocol['frame_inputs'][frame]['generation_dir'])
    storage.verify_frozen_tree(root, directory, protocol['feature_inputs'][frame])
    before = storage.snapshot(directory)
    binding = storage.read(directory/'binding.json')
    parent_binding(binding, protocol, frame, 'generation_inputs')
    payloads = PARENT_BINARY+('input_commit.json', 'forward_commit.json', 'report.json', 'completion.json')
    completion = verify_manifest(directory, binding, payload_inventory(payloads), 14, frame)
    for name in payloads:
        storage.storage.verify_bytes(directory/name, binding)
    inputs = storage.storage.load_commit(directory/'input_commit.json', binding)
    report = storage.storage.load_commit(directory/'report.json', binding)
    candidates = np.load(directory/'candidates.npy', allow_pickle=False)
    require(candidates.dtype == np.float32 and candidates.ndim == 2 and candidates.shape[1] == 17
            and len(candidates) <= 1024 and np.isfinite(candidates).all() and np.all(candidates[:, 16] == -1)
            and np.all(candidates[:, 1] >= 0) and np.all(candidates[:, 2:4] > 0), 'Invalid unmodified original candidates')
    require(completion['candidate_count'] == report['candidate_count'] == len(candidates)
            and completion['candidates_sha256'] == report['candidates_sha256'] == storage.digest(directory/'candidates.npy')
            and inputs['schema'] == 14 and inputs['frame'] == report['frame'] == frame
            and inputs['binding_fingerprint'] == report['binding_fingerprint'] == binding['fingerprint'], 'Parent generation identity differs')
    require(completion['input_status'] in ('ready', 'insufficient_input')
            and completion['outcome'] == ('insufficient_input' if completion['input_status'] == 'insufficient_input' else
                'generated' if len(candidates) else 'empty_candidates'), 'Parent generation coverage differs')
    require(completion['input_status'] != 'insufficient_input' or len(candidates) == 0, 'Insufficient input emitted candidates')
    return {'directory': directory, 'snapshot': before, 'completion': completion, 'information': inputs['information'],
        'candidates': candidates, 'points': np.load(directory/'valid_points.npy', allow_pickle=False),
        'pixels': np.load(directory/'valid_pixel_indices.npy', allow_pickle=False)}


def inspect_features(root, config, protocol, frame, guard):
    binding = storage.make_binding(root, config, protocol, phase='features', frame=frame)
    parent = inspect_parent_generation(root, protocol, frame)
    candidates = parent['candidates']
    count, blocks = len(candidates), (len(candidates)+7)//8
    directory = storage.frame_directory(root, protocol, frame)
    require(storage.read(directory/'binding.json') == binding, 'Feature source binding differs')
    names = payload_inventory(ARRAYS+('preparation.json', 'report.json', 'completion.json')+
                              tuple('blocks/block_%04d.json' % index for index in range(blocks)))
    completion = verify_manifest(directory, binding, names, 15, frame)
    require(completion['candidate_count'] == count and completion['expected_blocks'] == blocks, 'Dynamic feature block coverage differs')
    preparation = storage.load_commit(directory/'preparation.json', binding)
    report = storage.load_commit(directory/'report.json', binding)
    for name in ARRAYS:
        storage.verify_bytes(directory/name, binding)
    observed, pixels, voxel = [np.load(directory/name, allow_pickle=False) for name in ARRAYS]
    paths = protocol['frame_inputs'][frame]
    with Image.open(storage.safe_path(root, paths['depth'])) as source:
        depth = np.asarray(source)
    require(depth.shape == (720, 1280) and depth.dtype == np.uint16, 'Source depth shape/dtype differs')
    from scipy.io import loadmat
    calibration = loadmat(storage.safe_path(root, paths['meta']), variable_names=['intrinsic_matrix', 'factor_depth'])
    require(np.asarray(calibration['factor_depth']).size == 1, 'Depth factor must be scalar')
    factor = float(np.asarray(calibration['factor_depth']).item())
    intrinsic = np.asarray(calibration['intrinsic_matrix'], np.float64)
    reconstructed, source_pixels = reconstruct_observation(depth, intrinsic, factor)
    require(observed.dtype == np.float64 and np.array_equal(observed, reconstructed)
            and pixels.dtype == np.int64 and np.array_equal(pixels, source_pixels)
            and parent['points'].dtype == np.float32 and np.array_equal(observed.astype(np.float32), parent['points'])
            and parent['pixels'].dtype == np.int64 and np.array_equal(pixels, parent['pixels']), 'Raw depth/float64/parent float32 provenance differs')
    require(parent['information']['shape_hw'] == list(depth.shape) and parent['information']['intrinsic_matrix'] == intrinsic.tolist()
            and parent['information']['factor_depth'] == factor, 'Parent source calibration differs')
    voxel_audit = independent_voxel_centers(observed, voxel)
    source_completion_sha = storage.digest(parent['directory']/'completion.json')
    candidate_sha = storage.digest(parent['directory']/'candidates.npy')
    expected_preparation = {'schema': 15, 'status': 'prepared', 'frame': frame, 'binding_fingerprint': binding['fingerprint'],
        'candidate_count': count, 'candidates_sha256': candidate_sha, 'source_generation_completion_sha256': source_completion_sha,
        'input_status': parent['completion']['input_status'], 'source_generation_outcome': parent['completion']['outcome'],
        'depth_shape_hw': list(depth.shape), 'intrinsic_matrix': intrinsic.tolist(), 'factor_depth': factor,
        'metadata_fields_read': ['intrinsic_matrix', 'factor_depth'], 'maximum_depth_m': 2.,
        'observed_points': len(observed), 'voxel_points': len(voxel), 'voxel_size_m': .01, 'finger_width_m': .01, 'finger_length_m': .06,
        'point_precision': 'float64', 'candidate_precision': 'float32', 'files': {name: storage.digest(directory/name) for name in ARRAYS}}
    require(preparation == expected_preparation, 'Feature preparation disagrees with independent source reconstruction')
    rows, arithmetic = [], []
    for block_index in range(blocks):
        block = storage.load_commit(directory/('blocks/block_%04d.json' % block_index), binding)
        start, stop = block_index*8, min(count, block_index*8+8)
        require(block['schema'] == 15 and block['frame'] == frame and block['binding_fingerprint'] == binding['fingerprint']
                and block['block_index'] == block_index and block['start'] == start and block['stop'] == stop
                and len(block['rows']) == stop-start, 'Feature block partition changed')
        for index, row in enumerate(block['rows'], start):
            expected = independent_regions(candidates[index], voxel)
            compare_feature_row(row, frame, index, candidates[index], expected)
            arithmetic.append(dict(expected, original_index=index))
        rows.extend(block['rows'])
        guard.check()
    counts = {'collision': sum(row['collision'] for row in rows), 'empty_diagnostic': sum(row['empty_diagnostic'] is True for row in rows),
              'empty_not_applicable': sum(not row['empty_diagnostic_applicable'] for row in rows)}
    expected_report = {'schema': 15, 'status': 'completed', 'frame': frame, 'binding_fingerprint': binding['fingerprint'],
        'candidate_count': count, 'expected_blocks': blocks, 'rows': rows, 'candidates_sha256': candidate_sha,
        'source_generation_completion_sha256': source_completion_sha, 'preparation_sha256': storage.digest(directory/'preparation.json'),
        'counts': counts, 'uses_ground_truth_labels': False, 'uses_object_geometry': False, 'gpu_used': False,
        'new_training_steps': 0, 'new_model_forward_calls': 0, 'new_official_geometry_computations': 0,
        'nms_applied': False, 'all_candidates_preserved': True, 'empty_is_rejection': False,
        'score_semantics': 'Observed point count / voxelized geometric volume; not set IoU or calibrated probability',
        'unobserved_surfaces_ruled_safe': False}
    require(report == expected_report, 'Sealed feature report does not reconstruct')
    return {'directory': directory, 'snapshot': storage.snapshot(directory), 'rows': rows, 'candidates': candidates,
        'candidate_count': count, 'expected_blocks': blocks, 'input_status': parent['completion']['input_status'],
        'files': {(directory/name).relative_to(root).as_posix(): storage.digest(directory/name) for name in names},
        'seal': {'binding_fingerprint': binding['fingerprint'], 'completion_sha256': storage.digest(directory/'completion.json'),
                 'report_sha256': completion['report_sha256'], 'candidates_sha256': candidate_sha},
        'audit': {'candidate_count': count, 'observed_point_count': len(observed), 'voxel_audit': voxel_audit,
                  'counts': counts, 'independent_region_arithmetic': arithmetic},
        'parent_directory': parent['directory'], 'parent_snapshot': parent['snapshot']}


def inspect_geometry(root, protocol, frame, feature):
    directory = storage.safe_path(root, protocol['parent_geometry_dirs'][frame])
    storage.verify_frozen_tree(root, directory, protocol['evaluation_inputs'])
    binding = storage.read(directory/'binding.json')
    parent_binding(binding, protocol, frame, 'geometry_inputs')
    preparation = storage.storage.load_commit(directory/'preparation.json', binding)
    count = preparation['selected_count']
    require(type(count) is int and 0 <= count <= feature['candidate_count'], 'Invalid dynamic official support')
    names = payload_inventory(('preparation.json', 'report.json', 'oracle.json', 'completion.json')+
                              tuple('grasps/grasp_%04d.json' % i for i in range(count)))
    completion = verify_manifest(directory, binding, names, 14, frame)
    require(completion['expected_grasps'] == count, 'Official support count differs')
    records = [storage.storage.load_commit(directory/('grasps/grasp_%04d.json' % i), binding) for i in range(count)]
    report, oracle = [storage.storage.load_commit(directory/name, binding) for name in ('report.json', 'oracle.json')]
    hashes = [storage.digest(directory/('grasps/grasp_%04d.json' % i)) for i in range(count)]
    trace = independent_support_join(frame, feature['candidates'], feature['rows'], preparation, records, hashes, report, oracle)
    source = {name[:-5]+'_sha256': storage.digest(directory/name)
              for name in ('binding.json', 'completion.json', 'preparation.json', 'oracle.json', 'report.json')}
    metric = independent_metrics(trace)
    summary = dict(metric, frame=frame, candidate_count=feature['candidate_count'], selected_count=count,
        unscored_candidate_count=feature['candidate_count']-count, input_status=feature['input_status'], label_trace=trace, sources=source)
    return {'directory': directory, 'snapshot': storage.snapshot(directory), 'summary': summary,
            'all_generation_completion_sha256': completion['all_generation_completion_sha256']}


def independent_support_join(frame, candidates, features, preparation, records, hashes, report, oracle):
    count, support = len(candidates), len(records)
    require(preparation['input_dtype'] == 'float32' and preparation['input_count'] == count
            and preparation['selected_count'] == support == len(preparation['selected']) == len(hashes)
            and report['candidate_count'] == report['original_candidate_count'] == count and report['selected_count'] == support
            and len(features) == count, 'Fixed support membership differs')
    clipped = candidates.copy()
    clipped[:, 1] = np.clip(clipped[:, 1], 0, .1)
    seen, trace = set(), []
    for index, (selection, record, checksum) in enumerate(zip(preparation['selected'], records, hashes)):
        original, slot = selection['original_index'], selection['object_slot']
        require(type(original) is int and 0 <= original < count and original not in seen
                and selection['selection_index'] == index and type(slot) is int and 0 <= slot < len(preparation['object_ids'])
                and selection['object_id'] == preparation['object_ids'][slot], 'Ambiguous original candidate support mapping')
        require(all(record.get(key) == value for key, value in selection.items())
                and record['grasp_dtype'] in ('float32', 'float64') and record['grasp'] == clipped[original].tolist(),
                'Official scored row no longer equals original float32 width-clipped candidate')
        require(type(record['collision']) is bool and type(record['empty']) is bool
                and (not record['empty'] or record['collision']) and type(record['score']) in (int, float)
                and record['score'] in (-1., .2, .4, .6, .8, 1., 1.2)
                and (not record['collision'] or record['score'] == -1.) and re.fullmatch('[0-9a-f]{64}', checksum),
                'Invalid official reference label or provenance')
        row = features[original]
        require(row['original_index'] == original and row['frame'] == frame
                and row['candidate_row_sha256'] == hashlib.sha256(candidates[original].tobytes()).hexdigest(), 'Support-to-feature identity differs')
        seen.add(original)
        trace.append({'frame': frame, 'selection_index': index, 'original_index': original, 'object_slot': slot,
            'object_id': record['object_id'], 'object_local_index': record['object_local_index'], 'grasp_record_sha256': checksum,
            'candidate_row_sha256': row['candidate_row_sha256'], 'global_iou': row['occupied_volume_scores']['global'],
            'predicted_collision': row['collision'], 'collision': record['collision'], 'empty': record['empty'], 'friction_score': record['score']})
    expected_oracle = {'groups': [], 'scores': [], 'collision': []}
    for slot in range(len(preparation['object_ids'])):
        group = [record for record in records if record['object_slot'] == slot]
        require([record['object_local_index'] for record in group] == list(range(len(group))), 'Object-local support order differs')
        expected_oracle['groups'].append([record['grasp'] for record in group])
        expected_oracle['scores'].append([record['score'] for record in group])
        expected_oracle['collision'].append([record['collision'] for record in group])
    require(oracle['result'] == expected_oracle and oracle['executed'] is bool(count)
            and report['official_oracle_executed'] is bool(count) and report['official_oracle_exact_match'] is (True if count else None)
            and (not count or support > 0), 'Cached official reference does not match the immutable support')
    order = np.argsort(-np.asarray([record['grasp'][0] for record in records], np.float32))
    require(report['ordered_records'] == [records[int(i)] for i in order], 'Official report ranking differs')
    require(report['collision_including_empty_count'] == sum(row['collision'] for row in records)
            and report['empty_count'] == sum(row['empty'] for row in records)
            and report['positive_friction_score_count'] == sum(row['score'] > 0 for row in records), 'Official collision/friction counts differ')
    return trace


def resource_evidence(record, protocol):
    for key, limit in (('peak_rss_bytes', 'rss_soft_bytes'), ('elapsed_seconds', 'invocation_soft_seconds'),
                       ('current_output_bytes', 'output_limit_bytes')):
        value = record['resources'][key]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= protocol['resources']['features'][limit],
                'Feature recovery exceeded resource bounds')


def validate_work_record(record, operation, frame, candidates):
    count, blocks = len(candidates), (len(candidates)+7)//8
    fields = ('new_blocks', 'reused_blocks', 'committed_blocks', 'expected_blocks', 'new_feature_rows',
              'new_voxelizations', 'new_detector_calls')
    require(record['frame'] == frame and record['candidate_count'] == count
            and all(type(record[key]) is int and record[key] >= 0 for key in fields)
            and record['expected_blocks'] == blocks
            and record['new_blocks']+record['reused_blocks'] == record['committed_blocks'] <= blocks,
            'Feature work counters do not match dynamic block membership')
    for key in ('new_model_forward_calls', 'new_training_steps', 'new_official_geometry_computations'):
        require(type(record[key]) is int and record[key] == 0, 'Forbidden computation in feature recovery')
    recovered = record.get('recovered_completed_probe', False)
    if recovered:
        require(recovered is True and operation == 'probe' and blocks <= 2
                and record['status'] == 'verified_completed_read_only', 'Invalid recovered completed small probe')
    elif operation == 'probe':
        require(record['status'] == ('partial' if blocks > 2 else 'completed'), 'Wrong two-block stopping boundary')
    elif operation == 'complete':
        require(record['status'] in ('completed', 'verified_completed_read_only'), 'Incomplete feature recovery')
    else:
        require(operation == 'read_only' and record['status'] == 'verified_completed_read_only', 'Missing read-only feature check')
    committed = min(2, blocks) if operation == 'probe' else blocks
    require(record['committed_blocks'] == committed, 'Recovery did not stop at declared boundary')
    expected_rows = min(count, 8*committed)-min(count, 8*record['reused_blocks'])
    expected_calls = 0
    for block in range(record['reused_blocks'], committed):
        widths = candidates[block*8:min(count, block*8+8), 1]
        expected_calls += int(np.any(widths > 0))+int(np.any(widths == 0))
    require(record['new_feature_rows'] == expected_rows and record['new_detector_calls'] == expected_calls
            and record['new_voxelizations'] in (0, 1), 'New feature rows/detector calls do not match restored prefix')
    if operation != 'probe' or recovered:
        require(record['new_voxelizations'] == 0, 'Completed preparation was voxelized again')
    if operation == 'read_only' or record['status'] == 'verified_completed_read_only':
        require(all(record[key] == 0 for key in ('new_blocks', 'new_feature_rows', 'new_voxelizations', 'new_detector_calls')),
                'Read-only feature recovery performed computation')


def verify_recovery(root, config, protocol, features):
    proof_path = storage.safe_path(root, protocol['outputs']['features_resume_report'])
    before = {'sha256': storage.digest(proof_path), 'bytes': proof_path.stat().st_size, 'mtime_ns': proof_path.stat().st_mtime_ns}
    proof = storage.read(proof_path)
    expected_binding = {'protocol_sha256': storage.digest(config), 'wrapper_sha256': storage.digest(root/'scripts/run_local_risk_v15.py'),
        'child_sha256': storage.digest(root/'scripts/extract_local_risk_v15.py'), 'common_sha256': storage.digest(root/'scripts/local_risk_common_v15.py'),
        'interpreter_sha256': storage.digest(storage.safe_path(root, protocol['interpreter'])), 'pipeline_sha256': storage.digest(root/'scripts/pipeline.py')}
    require(proof['schema'] == 15 and proof['phase'] == 'features' and proof['status'] == 'passed'
            and proof['binding'] == expected_binding and set(proof['frames']) == set(FRAMES), 'Incomplete or differently bound feature recovery proof')
    details = {}
    for frame in FRAMES:
        entry, feature = proof['frames'][frame], features[frame]
        current = storage.snapshot(feature['directory'])
        require(entry['status'] == 'passed' and entry['byte_and_mtime_identical'] is True
                and entry['completed_snapshot'] == current and bool(entry['prefix_before'])
                and all(current.get(name) == value for name, value in entry['prefix_before'].items())
                and entry['completion_sha256'] == feature['seal']['completion_sha256'], 'Actual prefix or completed byte/size/mtime changed')
        if feature['expected_blocks'] > 2:
            prefix_names = payload_inventory(ARRAYS+('preparation.json', 'blocks/block_0000.json', 'blocks/block_0001.json'))
            require(set(entry['prefix_before']) == prefix_names, 'Recovery prefix was not exactly two feature blocks')
        else:
            require(entry['prefix_before'] == current, 'Completed small-input probe changed')
        for key, operation in (('probe_record', 'probe'), ('completion_record', 'complete'), ('read_only_record', 'read_only')):
            record = entry[key]
            resource_evidence(record, protocol)
            validate_work_record(record, operation, frame, feature['candidates'])
        require(entry['completion_record']['reused_blocks'] >= min(2, feature['expected_blocks']), 'Completed prefix blocks were not reused')
        details[frame] = {key: {field: entry[key][field] for field in ('new_blocks', 'reused_blocks', 'new_feature_rows', 'new_voxelizations', 'new_detector_calls')}
                         for key in ('probe_record', 'completion_record', 'read_only_record')}
    return {'sha256': before['sha256'], 'actual_prefix_and_completed_bytes_sizes_mtimes_verified': True,
            'frames': details, 'scope': 'Process boundaries, not physical power removal'}, proof_path, before


def expected_evaluation_report(protocol, binding, features, geometries, parent_sha):
    frames = {frame: geometries[frame]['summary'] for frame in FRAMES}
    return {'schema': 15, 'status': 'completed_conditional_collision_diagnostic', 'binding': binding,
        'binding_fingerprint': binding['fingerprint'], 'protocol_sha256': binding['protocol_sha256'],
        'feature_seals': {frame: features[frame]['seal'] for frame in FRAMES}, 'parent_acceptance_report_sha256': parent_sha,
        'frames': frames, 'frame_equal_metrics': equal_frame_metrics(frames),
        'score_semantics': 'Observed point count / voxelized geometric volume; not set IoU or calibrated probability',
        'metric_policy': {'reference_event': 'official_collision_including_empty', 'observed_collision_threshold': .01,
            'threshold_comparison': 'strictly_greater', 'risk_score': 'global_occupied_volume_score',
            'higher_score_means': 'collision', 'auroc_ties': 'half_credit_exact_pairs', 'metrics': ['tpr', 'fpr', 'auroc'],
            'aggregation': 'equal_frame_mean_over_eligible_frames_only'},
        'all_features_sealed_before_label_access': True, 'unscored_labels_are_unknown': True,
        'new_feature_computations': 0, 'new_geometry_computations': 0, 'new_official_geometry_computations': 0,
        'new_model_forward_calls': 0, 'new_training_steps': 0, 'new_nms_or_support_selection': False,
        'gpu_used': False, 'efficacy_gate': False, 'method_improvement_claimed': False, 'dataset_ap_computed': False,
        'calibrated_probability_claimed': False, 'independent_confirmation': False, 'limitations': protocol['limitations']}


def scientific_png(frames, means):
    canvas = Image.new('RGB', (1640, 1060), 'white')
    draw = ImageDraw.Draw(canvas)
    heading, text, small = (ImageFont.load_default(size=size) for size in (30, 22, 18))
    draw.text((40, 24), 'Observed local occupancy vs official collision | fixed support', fill='#172634', font=heading)
    draw.text((40, 72), 'Training scene 0003, four related views. Global occupancy > 0.01; empty retained in reference collision.', fill='#455462', font=small)
    draw.text((45, 125), 'A  Reference-support accounting', fill='#172634', font=text)
    columns = [('View', 50), ('All', 155), ('Scored', 255), ('Unknown', 370), ('TP', 520), ('FP', 615), ('TN', 710), ('FN', 805)]
    for label, x in columns:
        draw.text((x, 172), label, fill='#455462', font=text)
    for row_index, frame in enumerate(FRAMES):
        item = frames[frame]
        y = 220+row_index*57
        values = [frame, item['candidate_count'], item['selected_count'], item['unscored_candidate_count']]+[item['confusion'][key] for key in ('tp','fp','tn','fn')]
        for (_, x), value in zip(columns, values):
            draw.text((x, y), str(value), fill='#172634', font=text)
        draw.line((45, y+39, 890, y+39), fill='#e3e8ec')
    draw.text((980, 125), 'B  Equal-view summaries', fill='#172634', font=text)
    names = {'tpr': 'TPR (sensitivity)', 'fpr': 'FPR (false alarm)', 'auroc': 'AUROC (tie = half)'}
    colors = {'tpr': '#2868ad', 'fpr': '#be5d37', 'auroc': '#32855d'}
    for index, metric in enumerate(('tpr','fpr','auroc')):
        item = means[metric]
        y = 190+index*78
        value = item['mean']['value']
        draw.text((980, y), names[metric], fill=colors[metric], font=text)
        label = 'NA' if value is None else '%.6f' % value
        draw.text((980, y+30), label+' | eligible views '+str(item['eligible_frame_count'])+'/4', fill='#172634', font=text)
    draw.text((45, 480), 'C  Predeclared per-view metrics (undefined values are NA, not zero)', fill='#172634', font=text)
    for panel, metric in enumerate(('tpr','fpr','auroc')):
        left, top, width, height = 80+panel*530, 570, 430, 280
        bottom = top+height
        draw.text((left, top-38), names[metric], fill=colors[metric], font=text)
        for tick in range(6):
            y = bottom-tick*height/5
            draw.line((left,y,left+width,y),fill='#e3e8ec')
            draw.text((left-38,y-9),'%.1f' % (tick/5),fill='#455462',font=small)
        for index, frame in enumerate(FRAMES):
            value=frames[frame]['metrics'][metric]['value']
            x=left+25+index*104
            if value is None:
                draw.text((x,bottom-30),'NA',fill='#455462',font=text)
            else:
                y=bottom-value*height
                draw.rectangle((x,y,x+55,bottom),fill=colors[metric])
                draw.text((x-4,max(top-18,y-24)),'%.3f' % value,fill='#172634',font=small)
            draw.text((x-1,bottom+13),frame,fill='#455462',font=small)
    draw.text((40, 928), 'Unknown labels are excluded. Official geometry selected the scored support; this is a conditional diagnostic.', fill='#172634', font=small)
    draw.text((40, 963), 'Occupancy is point count / voxelized volume, not set IoU, calibrated probability, dataset AP, or robot safety.', fill='#455462', font=small)
    draw.text((40, 998), 'No training, inference, new support selection, or force-closure solve. No efficacy threshold or generalization claim.', fill='#455462', font=small)
    output=io.BytesIO()
    canvas.save(output,format='PNG',optimize=False)
    return output.getvalue()


def recover_publication_temporaries(root, protocol, read_only):
    report = storage.safe_path(root, protocol['outputs']['acceptance_report'])
    image = storage.safe_path(root, protocol['outputs']['visuals_dir'])/FIGURE
    for target in (report, image):
        for path in target.parent.glob(target.name+'.*.tmp'):
            require(path.is_file() and not path.is_symlink() and re.fullmatch(re.escape(target.name)+r'\.[0-9a-f]{32}\.tmp', path.name)
                    and not read_only and not report.exists(), 'Unexpected or sealed acceptance temporary')
            path.unlink()  # An exact, uncommitted output temporary only.


def locked_acceptance(root, config, protocol, guard, check_only):
    report_path = storage.safe_path(root, protocol['outputs']['acceptance_report'])
    read_only = check_only or report_path.exists()
    recover_publication_temporaries(root, protocol, read_only)
    features = {}
    for frame in FRAMES:
        features[frame] = inspect_features(root, config, protocol, frame, guard)
        guard.check()
    recovery, proof_path, proof_state = verify_recovery(root, config, protocol, features)
    # This is the first label-bearing binding/read, after all four feature audits.
    feature_files = {name: sha for item in features.values() for name, sha in item['files'].items()}
    evaluation_binding = storage.make_binding(root, config, protocol, phase='evaluation', extra_inputs=feature_files)
    parent_path = storage.safe_path(root, protocol['parent_acceptance_report'])
    require(protocol['parent_acceptance_report'] in protocol['evaluation_inputs'], 'Unbound parent acceptance')
    parent = storage.read(parent_path)
    require(parent['schema'] == 14 and parent['status'] == 'accepted_descriptive_multiview_baseline_engineering'
            and list(parent['frames']) == list(FRAMES), 'Parent engineering acceptance incomplete')
    geometries = {}
    for frame in FRAMES:
        geometries[frame] = inspect_geometry(root, protocol, frame, features[frame])
        source = geometries[frame]['summary']['sources']
        require(parent['frames'][frame]['geometry_completion_sha256'] == source['completion_sha256']
                and parent['frames'][frame]['candidates_sha256'] == features[frame]['seal']['candidates_sha256'], 'Parent accepted different inputs')
        require(geometries[frame]['all_generation_completion_sha256'] == {key: storage.digest(value['parent_directory']/'completion.json')
                for key,value in features.items()}, 'Parent geometry not bound to every generation seal')
        guard.check()
    expected = expected_evaluation_report(protocol, evaluation_binding, features, geometries, storage.digest(parent_path))
    evaluation_path = storage.safe_path(root, protocol['outputs']['evaluation_report'])
    require(storage.load_commit(evaluation_path, evaluation_binding) == expected, 'Independent fixed-support metric report differs')
    evaluation_state = {path: {'sha256': storage.digest(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                        for path in (evaluation_path, storage.receipt_path(evaluation_path))}
    extra = dict(feature_files)
    extra.update({path.relative_to(root).as_posix(): state['sha256'] for path,state in evaluation_state.items()})
    extra[proof_path.relative_to(root).as_posix()] = proof_state['sha256']
    binding = storage.make_binding(root, config, protocol, phase='acceptance', extra_inputs=extra)
    frames, means = expected['frames'], expected['frame_equal_metrics']
    image = scientific_png(frames, means)
    image_path = storage.safe_path(root, protocol['outputs']['visuals_dir'])/FIGURE
    report = {'schema': 15, 'status': 'accepted_conditional_observed_collision_engineering', 'binding': binding,
        'protocol_sha256': storage.digest(config), 'acceptor_sha256': storage.digest(Path(__file__)),
        'evaluation_report_sha256': storage.digest(evaluation_path), 'features': {frame: features[frame]['audit'] for frame in FRAMES},
        'frames': frames, 'frame_equal_metrics': means, 'feature_recovery': recovery,
        'checks': {'all_raw_depth_float64_and_parent_float32_clouds_reconstructed': True,
            'all_voxel_center_sets_independently_reconstructed': True, 'all_candidate_region_counts_and_occupancies_recomputed': True,
            'all_candidates_retained_including_negative_raw_confidence_and_zero_width': True,
            'all_features_sealed_and_audited_before_label_access': True, 'all_parent_seals_and_exact_candidate_support_links_verified': True,
            'cached_official_oracle_consistency_checked_without_resolving_geometry': True,
            'confusion_tpr_fpr_tie_half_auroc_and_eligible_frame_means_independently_recomputed': True,
            'undefined_diagnostics_retained_and_unscored_labels_not_imputed': True,
            'actual_recovery_prefix_and_complete_sha_size_mtime_verified': True},
        'independence_boundary': {'region_and_voxel_math_uses_no_producer_or_official_detector_function': True,
            'voxel_internal_native_accumulation_order_reproduced': False, 'voxel_center_absolute_tolerance_m': TOLERANCE,
            'feature_scalar_absolute_tolerance': TOLERANCE, 'nms_or_object_assignment_reexecuted': False,
            'force_closure_recomputed': False, 'filesystem_access_tracing_performed': False},
        'independent_feature_rows_recomputed': sum(item['candidate_count'] for item in features.values()),
        'new_training_steps': 0, 'new_model_forward_calls': 0, 'new_official_geometry_computations': 0,
        'new_nms_or_support_selection': False, 'gpu_used': False, 'efficacy_gate': False, 'method_improvement_claimed': False,
        'dataset_ap_computed': False, 'calibrated_probability_claimed': False, 'independent_confirmation': False,
        'scope': 'Feature arithmetic verification is not training or force-closure evaluation; four correlated views of one training scene on the fixed geometry-selected support',
        'visual': {'file': protocol['outputs']['visuals_dir']+'/'+FIGURE, 'sha256': hashlib.sha256(image).hexdigest()},
        'limitations': protocol['limitations']}
    for frame in FRAMES:
        feature, geometry = features[frame], geometries[frame]
        require(storage.snapshot(feature['directory']) == feature['snapshot']
                and storage.snapshot(feature['parent_directory']) == feature['parent_snapshot']
                and storage.snapshot(geometry['directory']) == geometry['snapshot'], 'Read-only acceptance changed sealed stage bytes/mtime')
    for path, state in {**evaluation_state, proof_path: proof_state}.items():
        require(state == {'sha256': storage.digest(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}, 'Sealed proof/evaluation changed')
    require(storage.make_binding(root, config, protocol, phase='acceptance', extra_inputs=extra) == binding, 'Bound source or old results changed during acceptance')
    if image_path.parent.exists():
        require(all(path == image_path and path.is_file() and not path.is_symlink() for path in image_path.parent.rglob('*')),
                'Unexpected acceptance visual inventory')
    payload = storage.json_bytes(report)
    guard.check((0 if image_path.exists() else len(image))+(0 if report_path.exists() else len(payload)))
    storage.publish_once(image_path, image, read_only)
    guard.check(0 if report_path.exists() else len(payload))
    storage.publish_once(report_path, payload, read_only)
    return {'status': 'verified_completed_read_only' if read_only else 'passed', 'report_sha256': storage.digest(report_path),
        'figure_sha256': storage.digest(image_path), 'independent_feature_rows_recomputed': report['independent_feature_rows_recomputed'],
        'new_training_steps': 0, 'new_model_forward_calls': 0, 'new_official_geometry_computations': 0, 'resources': guard.check()}


def run(root=ROOT, config=None, check_only=False):
    root = Path(root).resolve()
    config = Path(config or root/'configs/grasp_local_risk_v15.json').resolve()
    require(config.is_relative_to(root), 'Configuration outside project')
    protocol = storage.read(config)
    storage.validate_protocol(protocol)
    require(bool(protocol['source_hashes']), 'Independent acceptance source is not frozen')
    guard = storage.Guard(root, protocol, phase='acceptance')
    guard.check()
    with storage.lock(root/'runtime/locks/grasp_local_risk_v15/acceptance'):
        return locked_acceptance(root, config, protocol, guard, check_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/grasp_local_risk_v15.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.config, args.check_only), ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
