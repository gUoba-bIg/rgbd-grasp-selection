"""Frozen full-cloud grasp generation for four existing training-scene views."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from scipy.io import loadmat

if __package__:
    from . import multiview_common_v14 as common
else:
    import multiview_common_v14 as common

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ['0016', '0080', '0160', '0240']
CHECKPOINT = 'checkpoints/pretrained/checkpoint-rs.tar'
CHECKPOINT_SHA = '60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868'
ARRAYS = ('valid_points', 'valid_pixel_indices', 'sampled_valid_indices', 'sampled_points', 'sampled_pixel_indices')
ARTIFACTS = tuple(name + '.npy' for name in ARRAYS) + ('forward_bundle.npz', 'candidates.npy', 'seed_provenance.npz')
JSON_OUTPUTS = ('input_commit.json', 'forward_commit.json', 'report.json', 'completion.json')
NUMERICAL_ROOTS = ('runtime/build/graspnet', 'runtime/envs/vision/Lib/site-packages/pointnet2',
                   'runtime/envs/vision/Lib/site-packages/knn_pytorch')
NUMERICAL_SUFFIXES = ('.py', '.pyd', '.dll', '.cu', '.cpp', '.h')


def require(value, message):
    if not value:
        raise ValueError(message)


def reconstruct_points(depth, intrinsic, factor):
    """Match pinned upstream operation order before the final float32 conversion."""
    depth, intrinsic = np.asarray(depth), np.asarray(intrinsic, dtype=np.float64)
    require(depth.ndim == 2 and np.issubdtype(depth.dtype, np.number), 'Depth must be a numeric 2D array')
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all() and intrinsic[0, 0] > 0
            and intrinsic[1, 1] > 0 and math.isfinite(factor) and factor > 0, 'Invalid depth calibration')
    xmap, ymap = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    z = depth / factor
    x = (xmap - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (ymap - intrinsic[1, 2]) * z / intrinsic[1, 1]
    cloud = np.stack([x, y, z], axis=-1)
    valid = np.isfinite(cloud).all(-1) & (cloud[..., 2] > 0) & (cloud[..., 2] <= 2.)
    return {'valid_points': cloud[valid].astype(np.float32),
            'valid_pixel_indices': np.flatnonzero(valid.reshape(-1)).astype(np.int64)}


def sample_full_cloud(points, pixels, seed=20261021, size=20000):
    require(points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            and np.all((points[:, 2] > 0) & (points[:, 2] <= 2.)), 'Invalid full-cloud XYZ')
    require(pixels.dtype == np.int64 and pixels.shape == (len(points),) and np.all(pixels >= 0)
            and np.all(np.diff(pixels) > 0), 'Invalid original pixel provenance')
    require(type(seed) is int and type(size) is int and size > 0, 'Invalid sampling settings')
    enough = len(points) >= size
    native = np.random.RandomState(seed).choice(len(points), size, replace=False) if enough else np.empty(0, np.int64)
    indices = native.astype(np.int64)
    arrays = {'valid_points': points, 'valid_pixel_indices': pixels, 'sampled_valid_indices': indices,
              'sampled_points': points[indices], 'sampled_pixel_indices': pixels[indices]}
    return arrays, {'input_status': 'ready' if enough else 'insufficient_input', 'valid_point_count': len(points),
        'sampled_point_count': len(indices), 'sampling_with_replacement': False, 'seed': seed, 'requested_points': size,
        'native_sampling_index_dtype': str(native.dtype), 'native_sampling_indices_sha256': hashlib.sha256(native.tobytes()).hexdigest()}


def validate_candidates(candidates):
    require(candidates.dtype == np.float32 and candidates.ndim == 2 and candidates.shape[1] == 17
            and 0 <= len(candidates) <= 1024 and np.isfinite(candidates).all(), 'Invalid candidate array')
    if not len(candidates):
        return {'candidate_count': 0, 'geometry_checks_applicable': False}
    rotations = candidates[:, 4:13].reshape(-1, 3, 3)
    orthogonal = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.)))
    require(orthogonal < 1e-4 and determinant_error < 1e-4, 'Invalid candidate rotation')
    require(np.all(candidates[:, 0] >= 0) and np.all((candidates[:, 1] >= 0) & (candidates[:, 1] <= .100001))
            and np.allclose(candidates[:, 2], .02)
            and np.isclose(candidates[:, 3, None], [.01, .02, .03, .04], atol=1e-6).any(1).all()
            and np.all((candidates[:, 15] > 0) & (candidates[:, 15] <= 2.)) and np.all(candidates[:, 16] == -1),
            'Candidate dimensions/score/seed/object-ID differ')
    return {'candidate_count': len(candidates), 'geometry_checks_applicable': True,
            'maximum_rotation_orthogonality_error': orthogonal, 'maximum_determinant_error': determinant_error}


def empty_bundle():
    return {'candidates': np.empty((0, 17), np.float32), 'sa1_indices': np.empty(0, np.int64),
            'sa2_indices': np.empty(0, np.int64), 'selected_seed_indices': np.empty(0, np.int64),
            'fp2_xyz': np.empty((0, 3), np.float32), 'legacy_fp2_indices': np.empty(0, np.int64)}


def validate_seed_chain(bundle, arrays):
    require(set(bundle) == set(empty_bundle()), 'Unexpected forward-bundle fields')
    candidates = bundle['candidates']
    validate_candidates(candidates)
    selected, sa1, sa2 = (bundle[key] for key in ('selected_seed_indices', 'sa1_indices', 'sa2_indices'))
    fp2, legacy = bundle['fp2_xyz'], bundle['legacy_fp2_indices']
    if not len(sa1):
        require(all(value.shape == empty_bundle()[name].shape and value.dtype == empty_bundle()[name].dtype
                    for name, value in bundle.items()), 'Invalid insufficient-input bundle')
        return
    require(sa1.dtype == sa2.dtype == selected.dtype == legacy.dtype == np.int64 and sa1.shape == (2048,)
            and sa2.shape == legacy.shape == (1024,) and fp2.dtype == np.float32 and fp2.shape == (1024, 3),
            'Invalid actual seed arrays')
    require(np.all((sa1 >= 0) & (sa1 < len(arrays['sampled_points']))) and np.all((sa2 >= 0) & (sa2 < 2048)),
            'Actual SA index out of range')
    require(np.all((legacy >= 0) & (legacy < len(arrays['sampled_points']))), 'Diagnostic legacy seed index out of range')
    composed = sa1[sa2]
    require(np.array_equal(arrays['sampled_points'][composed], fp2), 'Actual SA seed composition differs from FP2 XYZ')
    require(selected.shape == (len(candidates),) and np.all((selected >= 0) & (selected < 1024))
            and np.all(np.diff(selected) > 0), 'Invalid decoded seed indices')
    require(np.array_equal(candidates[:, 13:16], fp2[selected]), 'Candidate XYZ differs from selected actual seed')


def seed_provenance(bundle, arrays):
    validate_seed_chain(bundle, arrays)
    sources = bundle['sa1_indices'][bundle['sa2_indices']][bundle['selected_seed_indices']] if len(bundle['sa1_indices']) else np.empty(0, np.int64)
    return {**{key: value for key, value in bundle.items() if key != 'candidates'},
            'candidate_sampled_indices': sources, 'candidate_valid_indices': arrays['sampled_valid_indices'][sources],
            'candidate_pixel_indices': arrays['sampled_pixel_indices'][sources]}


def serialize(value):
    stream = io.BytesIO()
    if isinstance(value, dict):
        np.savez(stream, **value)
    else:
        np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def forward_once(root, arrays, protocol, digest):
    """One FP32 model call, using the original custom-kernel numerical settings."""
    import torch
    require(torch.cuda.is_available(), 'Frozen GPU is unavailable')
    actual = {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'gpu': torch.cuda.get_device_name(0)}
    require(all(protocol['model_runtime'][key] == value for key, value in actual.items()), 'Frozen GPU runtime differs')
    checkpoint_path = root / CHECKPOINT
    require(digest(checkpoint_path) == CHECKPOINT_SHA, 'Checkpoint changed before model load')
    source = root / 'runtime/build/graspnet'
    sys.path[:0] = [str(source / folder) for folder in ('models', 'utils', 'pointnet2', 'knn')]
    from graspnet import GraspNet, pred_decode
    seed = protocol['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    limits = protocol['resources']['generation']
    torch.cuda.set_per_process_memory_fraction(min(1., limits['gpu_reserved_soft_bytes'] / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                  cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    require(all(torch.isfinite(value).all().item() for value in net.state_dict().values()), 'Nonfinite model state')
    del checkpoint
    net = net.cuda().eval()
    tensor = torch.from_numpy(arrays['sampled_points']).unsqueeze(0).cuda()
    with torch.inference_mode():
        endpoints = net({'point_clouds': tensor})
        candidates = pred_decode(endpoints)[0].detach().cpu().numpy().copy()
        selected = torch.nonzero(torch.argmax(endpoints['objectness_score'][0].float(), 0) == 1).reshape(-1)
        bundle = {'candidates': candidates,
            'sa1_indices': endpoints['sa1_inds'][0].detach().cpu().numpy().astype(np.int64),
            'sa2_indices': endpoints['sa2_inds'][0].detach().cpu().numpy().astype(np.int64),
            'fp2_xyz': endpoints['fp2_xyz'][0].detach().cpu().numpy().copy(),
            'legacy_fp2_indices': endpoints['fp2_inds'][0].detach().cpu().numpy().astype(np.int64),
            'selected_seed_indices': selected.cpu().numpy().astype(np.int64)}
    torch.cuda.synchronize()
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    require(allocated <= limits['gpu_allocated_soft_bytes'] and reserved <= limits['gpu_reserved_soft_bytes'], 'GPU resource limit exceeded')
    validate_seed_chain(bundle, arrays)
    return bundle, {'model_forward_calls': 1, 'strict_load': True, 'weights_only': True,
        'checkpoint_sha256': CHECKPOINT_SHA, 'precision': 'float32', 'autocast': False,
        'peak_gpu_allocated_bytes': allocated, 'peak_gpu_reserved_bytes': reserved,
        'legacy_fp2_index_matches_composed': bool(np.array_equal(bundle['legacy_fp2_indices'], bundle['sa1_indices'][bundle['sa2_indices']]))}


def frame_input_paths(protocol, frame):
    files = protocol['generation_inputs'][frame]
    paths = {}
    for kind, extension in (('rgb', 'png'), ('depth', 'png'), ('meta', 'mat')):
        matches = [name for name in files if name.endswith('/scene_0003/realsense/' + kind + '/' + frame + '.' + extension)]
        require(len(matches) == 1, 'Frozen frame input is missing or ambiguous: ' + kind)
        paths[kind] = matches[0]
    require(set(paths.values()) == set(files), 'Unexpected generation-frame source, possible label leakage')
    if 'frame_inputs' in protocol:
        require(paths == {kind: protocol['frame_inputs'][frame][kind] for kind in ('rgb', 'depth', 'meta')},
                'Frame input declarations differ')
    return paths


def frozen_numerical_inputs(protocol):
    shared = protocol['generation_shared_inputs']
    required = {CHECKPOINT, 'data/manifests/scene_splits.json'}
    require(required.issubset(shared), 'Missing mandatory shared generation source')
    numerical = {name: sha for name, sha in shared.items() if name not in required}
    require(bool(numerical) and all(any(name.startswith(folder + '/') for folder in NUMERICAL_ROOTS)
        and Path(name).suffix in NUMERICAL_SUFFIXES and '\\' not in name and '..' not in name.split('/')
        for name in numerical), 'Unexpected shared generation source')
    return numerical


def validate_protocol(protocol, frame):
    require(protocol['schema'] == 14 and protocol['frames'] == FRAMES and frame in FRAMES, 'Unsupported frame or protocol')
    require(protocol['scene_id'] == 3 and protocol['camera'] == 'realsense', 'Frozen scene/camera differs')
    require(protocol['seed'] == 20261021 and protocol['num_points'] == 20000 and protocol['maximum_depth_m'] == 2.,
            'Frozen sampling policy differs')
    require(protocol['checkpoint'] == {'path': CHECKPOINT, 'sha256': CHECKPOINT_SHA, 'weights_only': True, 'strict': True},
            'Frozen checkpoint policy differs')
    require(bool(protocol['source_hashes']), 'Source implementation must be frozen before generation')
    require(protocol['generation_shared_inputs'].get(CHECKPOINT) == CHECKPOINT_SHA, 'Checkpoint input binding missing')
    frozen_numerical_inputs(protocol)
    require(protocol['outputs']['frames_root'] == 'results/P3/grasp_multiview_v14/frames', 'Frozen generation destination differs')
    for selected in FRAMES:
        frame_input_paths(protocol, selected)
    for name in protocol['generation_shared_inputs']:
        require('/label/' not in name and '/annotations/' not in name and '/geometry/' not in name,
                'Generation shared inputs contain forbidden labels or geometry')
    limits = protocol['resources']['generation']
    for key, expected in {'cpu_threads': 2, 'rss_soft_bytes': 8 * 1024 ** 3,
        'gpu_allocated_soft_bytes': 4 * 1024 ** 3, 'gpu_reserved_soft_bytes': 4 * 1024 ** 3,
        'invocation_soft_seconds': 300, 'output_limit_bytes': 1024 ** 3, 'disk_reserve_bytes': 60 * 1024 ** 3}.items():
        require(limits[key] == expected, 'Frozen generation resource differs: ' + key)


def numerical_identity(root):
    numerical = {}
    for folder in NUMERICAL_ROOTS:
        for path in sorted(common.safe_path(root, folder).rglob('*')):
            if path.is_file() and path.suffix in NUMERICAL_SUFFIXES:
                numerical[path.relative_to(root).as_posix()] = common.digest(path)
    require(bool(numerical), 'Missing numerical implementation inventory')
    resolved = {}
    for module in ('pointnet2._ext', 'knn_pytorch.knn_pytorch'):
        spec = importlib.util.find_spec(module)
        require(spec is not None and spec.origin is not None, 'Missing CUDA extension: ' + module)
        path = Path(spec.origin).resolve()
        require(path.is_relative_to(root), 'CUDA extension escaped project')
        name = path.relative_to(root).as_posix()
        require(name in numerical and common.digest(path) == numerical[name], 'Resolved CUDA extension is outside bound inventory')
        resolved[module] = {'path': name, 'sha256': numerical[name]}
    return numerical, resolved


def make_generation_binding(root, config, protocol, frame):
    numerical, extensions = numerical_identity(root)
    require(numerical == frozen_numerical_inputs(protocol), 'Actual numerical inventory differs from the frozen shared inputs')
    model_runtime = {key: importlib.metadata.version(package) for key, package in (('torch', 'torch'), ('torchvision', 'torchvision'))}
    require(all(protocol['model_runtime'][key] == value for key, value in model_runtime.items()), 'Frozen model package runtime differs')
    value = common.make_binding(root, config, protocol, input_group='generation_inputs', frame=frame,
                                runtime_name='vision', extra_inputs=numerical)
    value.pop('fingerprint')
    value.update(resolved_extensions=extensions, model_runtime=protocol['model_runtime'],
                 numerical_source_and_extension_hashes=numerical, precision='float32', autocast=False,
                 uses_ground_truth_labels=False, uses_object_geometry=False, training_updates=0)
    value['fingerprint'] = common.fingerprint(value)
    return value


def load_frame_inputs(root, protocol, frame):
    paths = {kind: common.safe_path(root, name) for kind, name in frame_input_paths(protocol, frame).items()}
    with Image.open(paths['rgb']) as image:
        rgb = np.asarray(image.convert('RGB'))
    with Image.open(paths['depth']) as image:
        depth = np.asarray(image)
    require(rgb.shape == (720, 1280, 3) and rgb.dtype == np.uint8 and depth.shape == (720, 1280)
            and depth.dtype == np.uint16, 'Frozen RGB-D image shape/dtype differs')
    calibration = loadmat(paths['meta'], variable_names=['intrinsic_matrix', 'factor_depth'])
    intrinsic = np.asarray(calibration['intrinsic_matrix'], dtype=np.float64)
    factor_values = np.asarray(calibration['factor_depth'])
    require(factor_values.size == 1, 'Depth factor is not scalar')
    factor = float(factor_values.item())
    cloud = reconstruct_points(depth, intrinsic, factor)
    arrays, information = sample_full_cloud(cloud['valid_points'], cloud['valid_pixel_indices'], protocol['seed'], protocol['num_points'])
    information.update(shape_hw=list(depth.shape), intrinsic_matrix=intrinsic.tolist(), factor_depth=factor,
                       metadata_fields_read=['intrinsic_matrix', 'factor_depth'], model_input='XYZ only; RGB verified, not a model feature')
    return arrays, information


def allowed_output_names():
    names = {'binding.json'}
    for name in (*ARTIFACTS, *JSON_OUTPUTS):
        names.update((name, name + '.sha256.json'))
    return names


def validate_inventory(directory, recover=False):
    if not directory.exists():
        return
    allowed = allowed_output_names()
    for path in directory.iterdir():
        require(path.is_file() and not path.is_symlink(), 'Unexpected generation directory or link')
        if path.name in allowed:
            continue
        match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', path.name)
        require(match is not None and match[1] in allowed and recover, 'Unexpected or interrupted generation artifact: ' + path.name)
        path.unlink()


def save_json(path, value, binding, guard):
    guard.check(2 * len(common.json_bytes(value)) + 1024)
    common.commit(path, value, binding)
    guard.check()


def save_binary(path, value, binding, guard):
    payload = serialize(value)
    guard.check(2 * len(payload) + 1024)
    receipt = common.commit_bytes(path, payload, binding)
    guard.check()
    return receipt


def load_array(path, binding, archive=False):
    common.verify_bytes(path, binding)
    if archive:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}
    return np.load(path, allow_pickle=False)


def validate_input_cache(arrays, information, protocol):
    points, pixels = arrays['valid_points'], arrays['valid_pixel_indices']
    require(points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            and np.all((points[:, 2] > 0) & (points[:, 2] <= 2.)), 'Invalid cached full cloud')
    require(pixels.dtype == np.int64 and pixels.shape == (len(points),) and np.all((pixels >= 0) & (pixels < 720 * 1280))
            and np.all(np.diff(pixels) > 0), 'Invalid cached pixel provenance')
    indices = arrays['sampled_valid_indices']
    enough = len(points) >= protocol['num_points']
    require(indices.dtype == np.int64 and indices.shape == ((protocol['num_points'],) if enough else (0,))
            and np.all((indices >= 0) & (indices < len(points))) and len(np.unique(indices)) == len(indices), 'Invalid cached sample indices')
    require(arrays['sampled_points'].dtype == np.float32 and arrays['sampled_points'].shape == (len(indices), 3)
            and arrays['sampled_pixel_indices'].dtype == np.int64 and arrays['sampled_pixel_indices'].shape == indices.shape,
            'Cached sampled array shape/dtype differs')
    require(np.array_equal(arrays['sampled_points'], points[indices]) and np.array_equal(arrays['sampled_pixel_indices'], pixels[indices]),
            'Cached XYZ/pixel/sample chain differs')
    require(information['input_status'] == ('ready' if enough else 'insufficient_input')
            and information['valid_point_count'] == len(points) and information['sampled_point_count'] == len(indices)
            and information['seed'] == protocol['seed'] and information['requested_points'] == protocol['num_points']
            and information['sampling_with_replacement'] is False, 'Cached sampling information differs')
    require(information['native_sampling_index_dtype'] in ('int32', 'int64')
            and information['native_sampling_indices_sha256'] == hashlib.sha256(indices.astype(information['native_sampling_index_dtype']).tobytes()).hexdigest(),
            'Native sampling byte identity differs')


def read_committed_inputs(directory, binding, protocol, frame):
    commit = common.load_commit(directory / 'input_commit.json', binding)
    require(commit['schema'] == 14 and commit['frame'] == frame and commit['binding_fingerprint'] == binding['fingerprint']
            and commit['status'] == 'inputs_committed' and set(commit['files']) == {name + '.npy' for name in ARRAYS},
            'Invalid input commit identity')
    arrays = {name: load_array(directory / (name + '.npy'), binding) for name in ARRAYS}
    require(commit['files'] == {name + '.npy': common.digest(directory / (name + '.npy')) for name in ARRAYS}, 'Input commit inventory differs')
    validate_input_cache(arrays, commit['information'], protocol)
    return arrays, commit['information']


def final_manifest(directory):
    names = ['binding.json']
    for name in (*ARTIFACTS, *JSON_OUTPUTS[:-1]):
        names.extend((name, name + '.sha256.json'))
    return {name: common.digest(directory / name) for name in names}


def build_report(directory, frame, binding, information, bundle, forward_record):
    candidates = bundle['candidates']
    calls = forward_record['forward_calls_committed']
    require(calls == (0 if information['input_status'] == 'insufficient_input' else 1), 'Logical committed forward budget differs')
    if calls == 0:
        require(not len(bundle['sa1_indices']), 'Insufficient input cannot have a model forward')
    else:
        require(len(bundle['sa1_indices']) == 2048, 'Ready input must contain a real committed model seed chain')
    outcome = 'insufficient_input' if calls == 0 else 'empty_candidates' if not len(candidates) else 'generated'
    return {'schema': 14, 'status': 'completed', 'frame': frame, 'binding_fingerprint': binding['fingerprint'],
        'outcome': outcome, 'input_status': information['input_status'], 'input_information': information,
        'candidate_count': len(candidates), 'candidates_sha256': common.digest(directory / 'candidates.npy'),
        'forward_calls_committed': calls, 'forward_evidence': forward_record['evidence'],
        'geometry_checks': validate_candidates(candidates), 'uses_ground_truth_labels': False, 'uses_object_geometry': False,
        'training_updates': 0, 'is_dataset_ap_result': False,
        'limitations': ['Fixed training-scene views; not independent scenes or held-out confirmation',
            'Full-cloud baseline only; no segmentation benefit is inferred',
            'Candidate confidence is not calibrated success probability',
            'Custom CUDA bitwise reproducibility is not guaranteed; committed bundles are never forwarded again',
            'A missing durable bundle may need recomputation after interruption; mismatch with a saved intent is rejected',
            'Empty or insufficient inputs remain valid negative outcomes, without alternate seed or resampling']}


def validate_forward_record(record, binding, frame, directory):
    require(record['schema'] == 14 and record['frame'] == frame and record['binding_fingerprint'] == binding['fingerprint']
            and record['forward_calls_committed'] in (0, 1)
            and record['evidence']['model_forward_calls'] == record['forward_calls_committed']
            and record['forward_bundle_sha256'] == common.digest(directory / 'forward_bundle.npz'), 'Invalid forward commit identity')


def verify_completed(directory, binding, protocol, frame, guard):
    validate_inventory(directory)
    require(common.read(directory / 'binding.json') == binding, 'Completed generation binding differs')
    completion = common.load_commit(directory / 'completion.json', binding)
    require(completion['schema'] == 14 and completion['frame'] == frame and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == binding['fingerprint']
            and completion['outputs'] == final_manifest(directory), 'Completed generation manifest differs')
    arrays, information = read_committed_inputs(directory, binding, protocol, frame)
    bundle = load_array(directory / 'forward_bundle.npz', binding, archive=True)
    validate_seed_chain(bundle, arrays)
    forward_record = common.load_commit(directory / 'forward_commit.json', binding)
    validate_forward_record(forward_record, binding, frame, directory)
    candidates = load_array(directory / 'candidates.npy', binding)
    require(np.array_equal(candidates, bundle['candidates']), 'Exported candidate array changed')
    provenance = load_array(directory / 'seed_provenance.npz', binding, archive=True)
    expected = seed_provenance(bundle, arrays)
    require(set(provenance) == set(expected) and all(np.array_equal(provenance[key], expected[key]) for key in expected),
            'Exported seed provenance differs')
    report = common.load_commit(directory / 'report.json', binding)
    require(report == build_report(directory, frame, binding, information, bundle, forward_record)
            and completion['report_sha256'] == common.digest(directory / 'report.json')
            and all(completion[key] == report[key] for key in ('candidate_count', 'candidates_sha256', 'outcome', 'input_status', 'forward_calls_committed')),
            'Completed generation report differs')
    return {'status': 'verified_completed_read_only', 'frame': frame, 'candidate_count': len(candidates),
            'outcome': report['outcome'], 'input_status': information['input_status'], 'new_model_forward_calls': 0,
            'forward_calls_committed': report['forward_calls_committed'], 'resources': guard.check()}


@contextlib.contextmanager
def watch_resources(guard):
    stop = threading.Event()
    def watch():
        while not stop.wait(.1):
            try:
                guard.check()
            except FileNotFoundError:
                continue  # An atomic rename may race a directory-size sample.
            except Exception as error:
                print(json.dumps({'status': 'resource_limit_exceeded', 'reason': str(error)}), flush=True)
                os._exit(70)
    worker = threading.Thread(target=watch, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(1.)


def run(root, config, frame, stop_after_inputs=False, check_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    require(not (stop_after_inputs and check_only), 'Modes are mutually exclusive')
    protocol = common.read(config)
    validate_protocol(protocol, frame)
    guard = common.Guard(root, protocol, phase='generation')
    guard.check()
    with watch_resources(guard):
        binding = make_generation_binding(root, config, protocol, frame)
        directory = common.frame_directory(root, protocol, frame, 'generation')
        if (directory / 'completion.json').exists():
            return verify_completed(directory, binding, protocol, frame, guard)
        require(not check_only, 'Check-only requires completed generation')
        with common.lock(root / 'runtime/locks/grasp_multiview_v14' / frame):
            validate_inventory(directory, recover=True)
            if not (directory / 'binding.json').exists():
                require(not directory.exists() or not any(directory.iterdir()), 'Unbound generation output')
            guard.check(2 * len(common.json_bytes(binding)) + 1024)
            common.once(directory / 'binding.json', binding)
            if (directory / 'input_commit.json').exists():
                arrays, information = read_committed_inputs(directory, binding, protocol, frame)
            else:
                arrays, information = load_frame_inputs(root, protocol, frame)
                validate_input_cache(arrays, information, protocol)
                receipts = {name + '.npy': save_binary(directory / (name + '.npy'), arrays[name], binding, guard) for name in ARRAYS}
                input_commit = {'schema': 14, 'frame': frame, 'status': 'inputs_committed', 'binding_fingerprint': binding['fingerprint'],
                    'information': information, 'files': {name: receipt['sha256'] for name, receipt in receipts.items()}}
                save_json(directory / 'input_commit.json', input_commit, binding, guard)
            if stop_after_inputs:
                return {'status': 'inputs_committed', 'frame': frame, 'input_status': information['input_status'],
                        'new_model_forward_calls': 0, 'input_commit_sha256': common.digest(directory / 'input_commit.json'),
                        'resources': guard.check()}
            calls = 0
            bundle_path, forward_path = directory / 'forward_bundle.npz', directory / 'forward_commit.json'
            if bundle_path.exists():
                bundle = load_array(bundle_path, binding, archive=True)
                forward_record = common.load_commit(forward_path, binding)
                validate_forward_record(forward_record, binding, frame, directory)
            else:
                if information['input_status'] == 'insufficient_input':
                    bundle, evidence = empty_bundle(), {'model_forward_calls': 0, 'reason': 'insufficient_input'}
                else:
                    # Retain the shared lock used by the frozen earlier generator.
                    with common.lock(root / 'runtime/grasp_foreground_v12/generation_gpu_lock'):
                        guard.check()
                        bundle, evidence = forward_once(root, arrays, protocol, common.digest)
                    calls = 1
                validate_seed_chain(bundle, arrays)
                bundle_sha = hashlib.sha256(serialize(bundle)).hexdigest()
                if forward_path.exists():
                    forward_record = common.load_commit(forward_path, binding)
                    require(forward_record['forward_bundle_sha256'] == bundle_sha, 'Recomputed CUDA output differs from durable forward intent')
                else:
                    forward_record = {'schema': 14, 'frame': frame, 'binding_fingerprint': binding['fingerprint'],
                        'forward_calls_committed': evidence['model_forward_calls'], 'forward_bundle_sha256': bundle_sha, 'evidence': evidence}
                    save_json(forward_path, forward_record, binding, guard)
                save_binary(bundle_path, bundle, binding, guard)
                validate_forward_record(forward_record, binding, frame, directory)
            validate_seed_chain(bundle, arrays)
            save_binary(directory / 'candidates.npy', bundle['candidates'], binding, guard)
            save_binary(directory / 'seed_provenance.npz', seed_provenance(bundle, arrays), binding, guard)
            require(make_generation_binding(root, config, protocol, frame) == binding, 'Generation inputs/source changed during invocation')
            report = build_report(directory, frame, binding, information, bundle, forward_record)
            save_json(directory / 'report.json', report, binding, guard)
            completion = {key: report[key] for key in ('schema', 'frame', 'status', 'binding_fingerprint', 'outcome', 'input_status',
                'candidate_count', 'candidates_sha256', 'forward_calls_committed')}
            completion.update(outputs=final_manifest(directory), report_sha256=common.digest(directory / 'report.json'))
            save_json(directory / 'completion.json', completion, binding, guard)
            return {'status': 'completed', 'frame': frame, 'candidate_count': report['candidate_count'], 'outcome': report['outcome'],
                'input_status': information['input_status'], 'new_model_forward_calls': calls,
                'forward_calls_committed': report['forward_calls_committed'], 'resources': guard.check()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=common.CONFIG)
    parser.add_argument('--frame', required=True, choices=FRAMES)
    parser.add_argument('--stop-after-inputs', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.config, args.frame, args.stop_after_inputs, args.check_only), sort_keys=True))
