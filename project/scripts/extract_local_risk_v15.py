"""Resumable, label-free local observed occupancy for immutable grasp candidates."""
import argparse
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from scipy.io import loadmat

if __package__:
    from . import local_risk_common_v15 as common
else:
    import local_risk_common_v15 as common

ROOT, CONFIG = common.ROOT, common.CONFIG
require = common.require
SCORE_NAMES = common.SCORE_NAMES
ARRAYS = ('observed_points_float64.npy', 'valid_pixel_indices.npy', 'voxel_points.npy')
PARENT_ARRAYS = ('valid_points.npy', 'valid_pixel_indices.npy', 'sampled_valid_indices.npy',
                 'sampled_points.npy', 'sampled_pixel_indices.npy', 'forward_bundle.npz', 'candidates.npy', 'seed_provenance.npz')
DETECTOR_PATH = 'runtime/build/graspnet/utils/collision_detector.py'
DETECTOR_SHA = 'c8776ae298e2fa1972e94989074c7227467f37803c02217e950eefb0bae74bc8'


def dependencies(root=ROOT):
    require(common.digest(root / DETECTOR_PATH) == DETECTOR_SHA, 'Frozen detector changed')
    from graspnetAPI import GraspGroup
    spec = importlib.util.spec_from_file_location('frozen_local_risk_detector_v15', root / DETECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require('torch' not in sys.modules, 'Model runtime must not be imported')
    return GraspGroup, module.ModelFreeCollisionDetector


def reconstruct_float64(depth, intrinsic, factor, maximum_depth=2.):
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    require(depth.ndim == 2 and intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
            and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0 and math.isfinite(factor) and factor > 0, 'Invalid depth calibration')
    z = depth.astype(np.float64) / factor
    v, u = np.indices(z.shape)
    valid = np.isfinite(z) & (z > 0) & (z <= maximum_depth)
    points = np.stack(((u[valid] - intrinsic[0, 2]) * z[valid] / intrinsic[0, 0],
                       (v[valid] - intrinsic[1, 2]) * z[valid] / intrinsic[1, 1], z[valid]), axis=1)
    return points, np.flatnonzero(valid.reshape(-1)).astype(np.int64)


def load_raw_observation(root, protocol, frame, parent):
    paths = protocol['frame_inputs'][frame]
    with Image.open(common.safe_path(root, paths['depth'])) as image:
        depth = np.asarray(image)
    require(depth.dtype == np.uint16 and depth.shape == (720, 1280), 'Frozen depth shape/dtype differs')
    calibration = loadmat(common.safe_path(root, paths['meta']), variable_names=['intrinsic_matrix', 'factor_depth'])
    factor_array = np.asarray(calibration['factor_depth'])
    require(factor_array.size == 1, 'Depth factor must be scalar')
    factor = float(factor_array.item())
    intrinsic = np.asarray(calibration['intrinsic_matrix'], dtype=np.float64)
    points, pixels = reconstruct_float64(depth, intrinsic, factor, protocol['maximum_depth_m'])
    require(np.array_equal(pixels, parent['pixels']) and np.array_equal(points.astype(np.float32), parent['points']),
            'Float64 reconstruction does not reproduce parent pixels/float32 cloud')
    info = parent['information']
    require(info['shape_hw'] == list(depth.shape) and info['intrinsic_matrix'] == intrinsic.tolist()
            and info['factor_depth'] == factor, 'Parent calibration differs')
    return points, pixels, {'depth_shape_hw': list(depth.shape), 'intrinsic_matrix': intrinsic.tolist(), 'factor_depth': factor}


def validate_candidates(candidates):
    require(candidates.dtype == np.float32 and candidates.ndim == 2 and candidates.shape[1] == 17
            and len(candidates) <= 1024 and np.isfinite(candidates).all(), 'Invalid original candidate array')
    require(np.all(candidates[:, 1] >= 0) and np.all(candidates[:, 2:4] > 0)
            and np.all(candidates[:, 16] == -1), 'Invalid original dimensions/object field')


def read_parent(root, protocol, frame):
    directory = common.safe_path(root, protocol['frame_inputs'][frame]['generation_dir'])
    files = common.verify_frozen_tree(root, directory, protocol['feature_inputs'][frame])
    expected = {'binding.json'}
    for name in PARENT_ARRAYS + ('input_commit.json', 'forward_commit.json', 'report.json', 'completion.json'):
        expected.update((name, name + '.sha256.json'))
    require({path.relative_to(directory).as_posix() for path in directory.iterdir() if path.is_file()} == expected
            and all(path.is_file() for path in directory.iterdir()), 'Parent generation inventory differs')
    binding = common.read(directory / 'binding.json')
    identity = {key: value for key, value in binding.items() if key != 'fingerprint'}
    require(binding['fingerprint'] == common.storage.fingerprint(identity) and binding['frame'] == frame
            and binding['protocol_sha256'] == protocol['shared_inputs']['configs/grasp_multiview_v14r1.json'],
            'Parent generation binding differs')
    for name in PARENT_ARRAYS + ('input_commit.json', 'forward_commit.json', 'report.json', 'completion.json'):
        common.storage.verify_bytes(directory / name, binding)
    completion = common.storage.load_commit(directory / 'completion.json', binding)
    manifest = {name: common.digest(directory / name) for name in expected - {'completion.json', 'completion.json.sha256.json'}}
    require(completion['schema'] == 14 and completion['status'] == 'completed' and completion['frame'] == frame
            and completion['binding_fingerprint'] == binding['fingerprint'] and completion['outputs'] == manifest,
            'Parent generation completion differs')
    candidates = np.load(directory / 'candidates.npy', allow_pickle=False)
    validate_candidates(candidates)
    inputs = common.storage.load_commit(directory / 'input_commit.json', binding)
    report = common.storage.load_commit(directory / 'report.json', binding)
    require(inputs['schema'] == 14 and inputs['status'] == 'inputs_committed'
            and inputs['binding_fingerprint'] == binding['fingerprint']
            and inputs['files'] == {name: common.digest(directory / name) for name in PARENT_ARRAYS[:5]},
            'Parent input commitment differs')
    require(report['schema'] == 14 and report['status'] == 'completed' and report['frame'] == frame
            and report['binding_fingerprint'] == binding['fingerprint']
            and completion['report_sha256'] == common.digest(directory / 'report.json')
            and all(report[key] == completion[key] for key in ('candidate_count', 'candidates_sha256', 'outcome', 'input_status'))
            and report['candidates_sha256'] == common.digest(directory / 'candidates.npy'), 'Parent generation report identity differs')
    points = np.load(directory / 'valid_points.npy', allow_pickle=False)
    pixels = np.load(directory / 'valid_pixel_indices.npy', allow_pickle=False)
    require(points.dtype == np.float32 and points.shape == (len(pixels), 3) and np.isfinite(points).all()
            and pixels.dtype == np.int64 and pixels.ndim == 1 and np.all(np.diff(pixels) > 0)
            and np.all((pixels >= 0) & (pixels < 720 * 1280)) and np.all((points[:, 2] > 0) & (points[:, 2] <= 2.)),
            'Parent cloud/pixels differ')
    require(completion['candidate_count'] == len(candidates) and inputs['frame'] == frame, 'Parent frame/count differs')
    return {'candidates': candidates, 'points': points, 'pixels': pixels, 'information': inputs['information'],
            'input_status': completion['input_status'], 'outcome': completion['outcome'],
            'candidates_sha256': common.digest(directory / 'candidates.npy'),
            'source_generation_completion_sha256': common.digest(directory / 'completion.json')}


def save_json(path, value, binding, guard):
    guard.check(2 * len(common.json_bytes(value)) + 1024)
    result = common.commit(path, value, binding)
    guard.check()
    return result


def save_array(path, value, binding, guard):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    payload = stream.getvalue()
    guard.check(2 * len(payload) + 1024)
    result = common.commit_bytes(path, payload, binding)
    guard.check()
    return result


def load_array(path, binding):
    common.verify_bytes(path, binding)
    return np.load(path, allow_pickle=False)


def detector_from_cache(Detector, points, protocol):
    # The constructor performs voxelization; a committed cache must not repeat it.
    detector = Detector.__new__(Detector)
    detector.finger_width, detector.finger_length = .01, .06
    detector.voxel_size, detector.scene_points = protocol['voxel_size_m'], points
    return detector


def preparation_identity(frame, binding, parent, calibration, points, voxel, files, protocol):
    return {'schema': 15, 'status': 'prepared', 'frame': frame, 'binding_fingerprint': binding['fingerprint'],
            'candidate_count': len(parent['candidates']), 'candidates_sha256': parent['candidates_sha256'],
            'source_generation_completion_sha256': parent['source_generation_completion_sha256'],
            'input_status': parent['input_status'], 'source_generation_outcome': parent['outcome'],
            **calibration, 'metadata_fields_read': ['intrinsic_matrix', 'factor_depth'],
            'maximum_depth_m': protocol['maximum_depth_m'], 'observed_points': len(points), 'voxel_points': len(voxel),
            'voxel_size_m': protocol['voxel_size_m'], 'finger_width_m': .01, 'finger_length_m': .06,
            'point_precision': 'float64', 'candidate_precision': 'float32', 'files': files}


def prepare(root, protocol, frame, directory, binding, parent, guard):
    path = directory / 'preparation.json'
    new_voxelizations = 0
    if path.exists():
        preparation = common.load_commit(path, binding)
        points, pixels, voxel = (load_array(directory / name, binding) for name in ARRAYS)
        calibration = {key: preparation[key] for key in ('depth_shape_hw', 'intrinsic_matrix', 'factor_depth')}
    else:
        points, pixels, calibration = load_raw_observation(root, protocol, frame, parent)
        save_array(directory / ARRAYS[0], points, binding, guard)
        save_array(directory / ARRAYS[1], pixels, binding, guard)
        if (directory / ARRAYS[2]).exists():
            voxel = load_array(directory / ARRAYS[2], binding)
        else:
            _, Detector = dependencies(root)
            guard.check()
            detector = Detector(points, voxel_size=protocol['voxel_size_m'])
            require(detector.finger_width == .01 and detector.finger_length == .06, 'Detector geometry constants changed')
            voxel = np.asarray(detector.scene_points).copy()
            new_voxelizations = 1
            guard.check()
            save_array(directory / ARRAYS[2], voxel, binding, guard)
        preparation = None
    require(points.dtype == voxel.dtype == np.float64 and points.shape == (len(pixels), 3)
            and voxel.ndim == 2 and voxel.shape[1] == 3 and np.isfinite(points).all() and np.isfinite(voxel).all()
            and pixels.dtype == np.int64 and np.array_equal(pixels, parent['pixels'])
            and np.array_equal(points.astype(np.float32), parent['points']) and len(voxel) <= len(points), 'Prepared observation cache differs')
    info = parent['information']
    require(calibration == {'depth_shape_hw': info['shape_hw'], 'intrinsic_matrix': info['intrinsic_matrix'],
                            'factor_depth': info['factor_depth']}, 'Prepared calibration differs')
    files = {name: common.digest(directory / name) for name in ARRAYS}
    expected = preparation_identity(frame, binding, parent, calibration, points, voxel, files, protocol)
    if preparation is None:
        save_json(path, expected, binding, guard)
    else:
        require(preparation == expected, 'Preparation identity differs')
    return voxel, expected, new_voxelizations


def block_rows(detector, candidates, start, stop, protocol, Group, frame):
    selected = candidates[start:stop]
    values = {}
    calls = 0
    for applicable in (True, False):
        indices = np.flatnonzero((selected[:, 1] > 0) if applicable else (selected[:, 1] == 0))
        if not len(indices):
            continue
        returned = detector.detect(Group(selected[indices].copy()), approach_dist=protocol['approach_distance_m'],
            collision_thresh=protocol['collision_threshold'], return_empty_grasp=applicable,
            empty_thresh=protocol['empty_threshold'], return_ious=True)
        calls += 1
        collision, scores = returned[0], returned[-1]
        require(len(scores) == len(SCORE_NAMES), 'Detector score field count differs')
        for position, local_index in enumerate(indices):
            original_index = start + int(local_index)
            values[original_index] = {'frame': frame, 'original_index': original_index,
                'raw_confidence': float(candidates[original_index, 0]),
                'candidate_row_sha256': hashlib.sha256(candidates[original_index].tobytes(order='C')).hexdigest(),
                'collision': bool(collision[position]),
                'empty_diagnostic': bool(returned[1][position]) if applicable else None,
                'empty_diagnostic_applicable': applicable, 'empty_diagnostic_reason': None if applicable else 'zero_width',
                'occupied_volume_scores': {name: float(score[position]) for name, score in zip(SCORE_NAMES, scores)}}
    return [values[index] for index in range(start, stop)], calls


def validate_rows(rows, candidates, start, stop, protocol, frame):
    require(isinstance(rows, list) and len(rows) == stop - start, 'Wrong block row count')
    keys = {'frame', 'original_index', 'raw_confidence', 'candidate_row_sha256', 'collision', 'empty_diagnostic',
            'empty_diagnostic_applicable', 'empty_diagnostic_reason', 'occupied_volume_scores'}
    for index, row in enumerate(rows, start):
        require(set(row) == keys and row['frame'] == frame and type(row['original_index']) is int
                and row['original_index'] == index and row['raw_confidence'] == float(candidates[index, 0])
                and row['candidate_row_sha256'] == hashlib.sha256(candidates[index].tobytes(order='C')).hexdigest(), 'Original candidate identity changed')
        scores = row['occupied_volume_scores']
        require(set(scores) == set(SCORE_NAMES) and all(type(value) in (int, float) and math.isfinite(value)
                and value >= 0 for value in scores.values()), 'Invalid occupied-volume scores')
        require(type(row['collision']) is bool and row['collision'] == (scores['global'] > protocol['collision_threshold']), 'Collision threshold differs')
        applicable = bool(candidates[index, 1] > 0)
        require(row['empty_diagnostic_applicable'] is applicable and
                ((type(row['empty_diagnostic']) is bool and row['empty_diagnostic_reason'] is None) if applicable else
                 (row['empty_diagnostic'] is None and row['empty_diagnostic_reason'] == 'zero_width')), 'Empty diagnostic applicability differs')


def allowed_names(expected_blocks):
    names = {'binding.json'}
    for name in (*ARRAYS, 'preparation.json', 'report.json', 'completion.json',
                 *('blocks/block_%04d.json' % index for index in range(expected_blocks))):
        names.update((name, name + '.sha256.json'))
    return names


def check_inventory(directory, expected_blocks, completed=False):
    allowed = allowed_names(expected_blocks)
    if not directory.exists():
        return
    for path in directory.rglob('*'):
        require(not path.is_symlink() and (path.is_file() or path == directory / 'blocks'), 'Unexpected output directory/link')
    common.recover_temporaries(directory, allowed, check_only=completed)
    if completed:
        require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()} == allowed,
                'Completed feature inventory differs')


def make_report(frame, binding, parent, preparation_sha, rows, expected_blocks):
    return {'schema': 15, 'status': 'completed', 'frame': frame, 'binding_fingerprint': binding['fingerprint'],
        'candidate_count': len(parent['candidates']), 'expected_blocks': expected_blocks, 'rows': rows,
        'candidates_sha256': parent['candidates_sha256'],
        'source_generation_completion_sha256': parent['source_generation_completion_sha256'], 'preparation_sha256': preparation_sha,
        'counts': {'collision': sum(row['collision'] for row in rows),
                   'empty_diagnostic': sum(row['empty_diagnostic'] is True for row in rows),
                   'empty_not_applicable': sum(not row['empty_diagnostic_applicable'] for row in rows)},
        'uses_ground_truth_labels': False, 'uses_object_geometry': False, 'gpu_used': False,
        'new_training_steps': 0, 'new_model_forward_calls': 0, 'new_official_geometry_computations': 0,
        'nms_applied': False, 'all_candidates_preserved': True, 'empty_is_rejection': False,
        'score_semantics': 'Observed point count / voxelized geometric volume; not set IoU or calibrated probability',
        'unobserved_surfaces_ruled_safe': False}


def run(root=ROOT, config=CONFIG, frame=None, max_new_blocks=None, check_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    require(max_new_blocks is None or type(max_new_blocks) is int and max_new_blocks >= 0, 'Invalid block budget')
    require(not check_only or max_new_blocks is None, 'Conflicting check-only budget')
    protocol = common.read(config)
    common.validate_protocol(protocol)
    require(bool(protocol['source_hashes']), 'Feature implementation must be frozen')
    require(frame in protocol['frames'], 'Unknown feature frame')
    guard = common.Guard(root, protocol, phase='features')
    guard.check()
    binding = common.make_binding(root, config, protocol, phase='features', frame=frame)
    parent = read_parent(root, protocol, frame)
    candidates = parent['candidates']
    expected_blocks = math.ceil(len(candidates) / protocol['block_size'])
    directory = common.frame_directory(root, protocol, frame)
    completed = (directory / 'completion.json').exists()
    require(not check_only or completed, 'Check-only requires completed features')
    with common.lock(root / 'runtime/locks/grasp_local_risk_v15/features' / frame):
        check_inventory(directory, expected_blocks, completed)
        missing_block = False
        for index in range(expected_blocks):
            present = (directory / ('blocks/block_%04d.json' % index)).exists()
            require(not (present and missing_block), 'Committed block payloads are not a contiguous prefix')
            missing_block = missing_block or not present
        if completed:
            require(common.read(directory / 'binding.json') == binding, 'Completed feature binding differs')
        else:
            require((directory / 'binding.json').exists() or not directory.exists() or not any(directory.iterdir()), 'Unbound feature artifacts')
            guard.check(2 * len(common.json_bytes(binding)) + 1024)
            common.once(directory / 'binding.json', binding)
        voxel, preparation, voxelizations = prepare(root, protocol, frame, directory, binding, parent, guard)
        rows, new, reused, calls, new_rows = [], 0, 0, 0, 0
        detector = Group = None
        for index in range(expected_blocks):
            guard.check()
            start, stop = index * protocol['block_size'], min(len(candidates), (index + 1) * protocol['block_size'])
            path = directory / ('blocks/block_%04d.json' % index)
            if path.exists():
                block = common.load_commit(path, binding)
                reused += 1
            else:
                require(not completed, 'Completed feature block missing')
                if max_new_blocks is not None and new >= max_new_blocks:
                    break
                if detector is None:
                    Group, Detector = dependencies(root)
                    detector = detector_from_cache(Detector, voxel, protocol)
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=protocol['cpu_threads']):
                    values, block_calls = block_rows(detector, candidates, start, stop, protocol, Group, frame)
                calls += block_calls
                validate_rows(values, candidates, start, stop, protocol, frame)
                block = {'schema': 15, 'frame': frame, 'binding_fingerprint': binding['fingerprint'],
                         'block_index': index, 'start': start, 'stop': stop, 'rows': values}
                save_json(path, block, binding, guard)
                new += 1
                new_rows += stop - start
            require(block['schema'] == 15 and block['frame'] == frame and block['binding_fingerprint'] == binding['fingerprint']
                    and block['block_index'] == index and block['start'] == start and block['stop'] == stop, 'Block identity differs')
            validate_rows(block['rows'], candidates, start, stop, protocol, frame)
            rows.extend(block['rows'])
        status = 'partial'
        if len(rows) == len(candidates):
            report = make_report(frame, binding, parent, common.digest(directory / 'preparation.json'), rows, expected_blocks)
            if completed:
                require(common.load_commit(directory / 'report.json', binding) == report, 'Completed feature report differs')
            else:
                require(common.make_binding(root, config, protocol, phase='features', frame=frame) == binding,
                        'Feature input/source changed during invocation')
                save_json(directory / 'report.json', report, binding, guard)
            outputs = {name: common.digest(directory / name) for name in sorted(allowed_names(expected_blocks)
                       - {'completion.json', 'completion.json.sha256.json'})}
            completion = {key: report[key] for key in ('schema', 'status', 'frame', 'binding_fingerprint', 'candidate_count', 'expected_blocks')}
            completion.update(outputs=outputs, report_sha256=common.digest(directory / 'report.json'))
            if completed:
                require(common.load_commit(directory / 'completion.json', binding) == completion, 'Completed feature manifest differs')
            else:
                save_json(directory / 'completion.json', completion, binding, guard)
            status = 'verified_completed_read_only' if completed else 'completed'
        return {'status': status, 'frame': frame, 'candidate_count': len(candidates), 'new_blocks': new,
                'reused_blocks': reused, 'committed_blocks': new + reused, 'expected_blocks': expected_blocks,
                'new_feature_rows': new_rows,
                'new_voxelizations': voxelizations, 'new_detector_calls': calls,
                'new_model_forward_calls': 0, 'new_training_steps': 0, 'new_official_geometry_computations': 0,
                'resources': guard.check()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--frame', required=True, choices=common.FRAMES)
    parser.add_argument('--max-new-blocks', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(__import__('json').dumps(run(args.root, args.config, args.frame, args.max_new_blocks, args.check_only), sort_keys=True))
