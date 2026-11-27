"""Trace frozen grasp seed translations to predicted frame-local instances.

No inference, labels, model geometry, candidate filtering or score changes occur.
Only exact float32 sampled-point equality authorizes an instance association.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True

import numpy as np

DEFAULTS = {'block_size': 16, 'sample_seed': 20261021, 'num_points': 20000,
            'maximum_depth_m': 2.0, 'candidate_count': 236,
            'matching_policy': 'exact_float32_xyz', 'disk_reserve_bytes': 60 * 1024 ** 3,
            'output_limit_bytes': 256 * 1024 ** 2, 'rss_soft_bytes': 4 * 1024 ** 3,
            'invocation_soft_seconds': 300}
ARRAY_NAMES = ('valid_points', 'valid_pixel_indices', 'valid_instance_ids',
               'sampled_indices', 'sampled_points', 'sampled_pixel_indices')
STATUSES = ('assigned', 'background', 'ambiguous', 'unmatched')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def once_json(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Immutable JSON changed: ' + str(path))
    else:
        atomic_json(path, value)


def once_array(path, value):
    path = Path(path)
    if path.exists():
        old = np.load(path, allow_pickle=False)
        require(old.dtype == value.dtype and old.shape == value.shape
                and np.array_equal(old, value), 'Uncommitted array differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / '.process.lock').open('a+b')
    try:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def round_output_files(paths):
    """Count shared outputs once, including pending sibling writes for file targets."""
    files = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.update(item.resolve() for item in path.rglob('*') if item.is_file())
        elif path.is_file():
            files.add(path.resolve())
        if not path.is_dir() and path.parent.is_dir():
            files.update(item.resolve() for item in path.parent.glob(path.name + '.*')
                         if item.is_file() and (item.name.endswith('.tmp') or '.tmp-' in item.name))
    return files


def planned_json_bytes(path, value):
    return 0 if Path(path).exists() else len((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


class Monitor:
    """Soft sampled resource guards; native calls cannot be safely interrupted."""
    def __init__(self, root, out, settings, output_paths=None):
        self.root, self.out, self.settings = root, out, settings
        self.output_paths = list(output_paths or []) + [out, out.with_suffix('.json')]
        self.started, self.peak, self.error = time.monotonic(), 0, None
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.watch, daemon=True)
        self.worker.start()

    def watch(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, resident_memory_bytes())
                if self.peak > self.settings['rss_soft_bytes']:
                    self.error = 'RSS soft limit exceeded'
                if time.monotonic() - self.started > self.settings['invocation_soft_seconds']:
                    self.error = 'Time soft limit exceeded'
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(.25)

    def check(self, additional_bytes=0):
        self.peak = max(self.peak, resident_memory_bytes())
        require(self.peak <= self.settings['rss_soft_bytes'], 'RSS soft limit exceeded')
        require(self.error is None, self.error or '')
        require(time.monotonic() - self.started <= self.settings['invocation_soft_seconds'], 'Time limit exceeded')
        require(shutil.disk_usage(self.root).free >= self.settings['disk_reserve_bytes'], 'Disk reserve unavailable')
        require(type(additional_bytes) is int and additional_bytes >= 0, 'Invalid projected output size')
        size = sum(path.stat().st_size for path in round_output_files(self.output_paths))
        require(size + additional_bytes <= self.settings['output_limit_bytes'], 'Combined round output budget exceeded')

    def close(self):
        self.stop.set()
        self.worker.join(1)


def resident_memory_bytes():
    """Native Windows working-set size, without altering the frozen environment."""
    import ctypes
    from ctypes import wintypes
    require(os.name == 'nt', 'Round 11 resource measurement requires the frozen Windows host')

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                    ('PrivateUsage', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    require(bool(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)),
            'Native process memory query failed: {}'.format(ctypes.get_last_error()))
    return int(counters.WorkingSetSize)


def validate_settings(protocol):
    settings = dict(DEFAULTS)
    settings.update(protocol.get('association', {}))
    for field in ('block_size', 'sample_seed', 'num_points', 'candidate_count'):
        require(type(settings[field]) is int and settings[field] > 0, 'Invalid integer: ' + field)
    require(settings['matching_policy'] == 'exact_float32_xyz', 'Only exact sampled-point matching is permitted')
    require(settings['maximum_depth_m'] == 2.0 and settings['sample_seed'] == 20261021
            and settings['num_points'] == 20000 and settings['candidate_count'] == 236,
            'Round 11 must reproduce frozen smoke settings')
    for field in ('disk_reserve_bytes', 'output_limit_bytes', 'rss_soft_bytes', 'invocation_soft_seconds'):
        require(isinstance(settings[field], (float, int)) and settings[field] > 0, 'Invalid resource limit')
    return settings


def safe_file(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Path escapes project')
    return path


def verify_perception(root, relative):
    perception = safe_file(root, relative)
    completion = read(perception / 'completion.json')
    require(completion['schema'] == 11 and completion['status'] == 'completed'
            and completion['ground_truth_labels_read'] is False, 'Prediction is not sealed label-free output')
    require(completion['outputs']['instances_original.npy'] == digest(perception / 'instances_original.npy'),
            'Predicted mask differs from perception completion')
    perception_binding = read(perception / 'binding.json')
    # Perception uses the already-frozen checkpoints canonical serialization.
    perception_id = hashlib.sha256(json.dumps(perception_binding, sort_keys=True, ensure_ascii=False,
                                               separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(completion['binding_fingerprint'] == perception_id, 'Prediction completion binding differs')
    for name, expected in completion['outputs'].items():
        require(digest(safe_file(perception, name)) == expected, 'Prediction committed output changed: ' + name)
    require(digest(safe_file(root, completion['report_file'])) == completion['report_sha256'], 'Prediction report changed')
    return completion


def make_binding(root, config, protocol, settings):
    require(Path(sys.prefix).resolve() == (root / 'runtime/envs/vision').resolve(), 'Use frozen vision Python')
    hashes = dict(protocol.get('input_hashes', {}))
    hashes.update(protocol.get('source_hashes', {}))
    for name, expected in hashes.items():
        require(digest(safe_file(root, name)) == expected, 'Frozen input/source changed: ' + name)
    for item in protocol['inputs'].values():
        require(digest(safe_file(root, item['path'])) == item['sha256'], 'Configured input hash mismatch')
    require(digest(safe_file(root, protocol['candidate']['path'])) == protocol['candidate']['sha256'], 'Configured candidates changed')
    require(digest(safe_file(root, protocol['candidate']['smoke_path'])) == protocol['candidate']['smoke_sha256'], 'Configured smoke changed')
    required = ['results/P3/inference_smoke.json',
                'results/P3/scene_0003_realsense_0000_candidates.npy',
                'data/raw/graspnet/scenes/scene_0003/realsense/rgb/0000.png',
                'data/raw/graspnet/scenes/scene_0003/realsense/depth/0000.png',
                'data/raw/graspnet/scenes/scene_0003/realsense/meta/0000.mat',
                'scripts/run_grasp_inference_smoke.py',
                'runtime/build/graspnet/models/graspnet.py',
                'runtime/build/graspnet/models/backbone.py',
                'runtime/build/graspnet/pointnet2/pointnet2_modules.py',
                'runtime/build/graspnet/pointnet2/pointnet2_utils.py',
                'runtime/build/graspnet/utils/data_utils.py',
                'results/P3/instance_interface_v11/instances_original.npy',
                'results/P3/instance_interface_v11/binding.json',
                'results/P3/instance_interface_v11/completion.json']
    actual_inputs = {name: digest(root / name) for name in required}
    verify_perception(root, protocol['outputs']['perception_dir'])
    site = root / 'runtime/envs/vision/Lib/site-packages'
    runtime_files = {}
    for name in ('numpy', 'numpy.libs', 'scipy', 'scipy.libs', 'PIL'):
        for path in sorted((site / name).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.pyd', '.dll'):
                runtime_files[path.relative_to(root).as_posix()] = digest(path)
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'Pillow')}
    return {'schema': 11, 'config_sha256': digest(config), 'script_sha256': digest(__file__),
            'settings': settings, 'frozen_hashes': hashes, 'input_hashes': actual_inputs,
            'runtime': {'python': sys.version, 'prefix': str(Path(sys.prefix).resolve()),
                        'executable_sha256': digest(sys.executable), 'packages': versions,
                        'numerical_files': runtime_files, 'cpu_threads': 2, 'gpu_used': False}}


def reconstruct_points(depth, intrinsic, factor, instances, maximum_depth=2.0):
    """Same operation order/dtypes as pinned upstream data_utils, then float32."""
    depth, intrinsic, instances = np.asarray(depth), np.asarray(intrinsic, dtype=np.float64), np.asarray(instances)
    require(depth.ndim == 2 and instances.shape == depth.shape, 'Mask/depth shape mismatch')
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
            and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0 and math.isfinite(factor) and factor > 0,
            'Invalid camera calibration')
    require(np.issubdtype(instances.dtype, np.integer) and (instances >= 0).all()
            and (instances <= np.iinfo(np.int32).max).all(), 'Invalid predicted instance IDs')
    xmap, ymap = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    z = depth / factor
    x = (xmap - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (ymap - intrinsic[1, 2]) * z / intrinsic[1, 1]
    cloud = np.stack([x, y, z], axis=-1)
    valid = np.isfinite(cloud).all(-1) & (cloud[..., 2] > 0) & (cloud[..., 2] <= maximum_depth)
    return {'valid_points': cloud[valid].astype(np.float32),
            'valid_pixel_indices': np.flatnonzero(valid.reshape(-1)),
            'valid_instance_ids': instances[valid].astype(np.int32)}


def sample_indices(count, seed, size, expected_sha256=None):
    require(count >= size, 'Insufficient valid points; no replacement allowed')
    indices = np.random.RandomState(seed).choice(count, size, replace=False)
    actual = hashlib.sha256(indices.tobytes()).hexdigest()
    if expected_sha256 is not None:
        require(actual == expected_sha256, 'Original sampled-index hash mismatch')
    return indices


def reprojection_error(points, pixels, shape, intrinsic):
    """Diagnostic consistency of retained float32 XYZ with its original pixel."""
    require(len(points) > 0 and points.shape == (len(pixels), 3) and np.isfinite(points).all()
            and (points[:, 2] > 0).all(), 'Invalid projected point inputs')
    points = points.astype(np.float64)
    u = points[:, 0] * intrinsic[0, 0] / points[:, 2] + intrinsic[0, 2]
    v = points[:, 1] * intrinsic[1, 1] / points[:, 2] + intrinsic[1, 2]
    du = np.abs(u - pixels % shape[1])
    dv = np.abs(v - pixels // shape[1])
    return {'maximum_absolute_u_pixels': float(du.max()), 'maximum_absolute_v_pixels': float(dv.max()),
            'maximum_euclidean_pixels': float(np.sqrt(du ** 2 + dv ** 2).max())}


def load_originals(root, settings):
    from PIL import Image
    from scipy.io import loadmat
    base = root / 'data/raw/graspnet/scenes/scene_0003/realsense'
    meta = loadmat(base / 'meta/0000.mat', variable_names=['intrinsic_matrix', 'factor_depth'])
    intrinsic = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
    factor = float(np.asarray(meta['factor_depth']).item())
    with Image.open(base / 'depth/0000.png') as image:
        depth = np.asarray(image)
    with Image.open(base / 'rgb/0000.png') as image:
        rgb = np.asarray(image.convert('RGB'))
    instances = np.load(root / 'results/P3/instance_interface_v11/instances_original.npy', allow_pickle=False)
    smoke = read(root / 'results/P3/inference_smoke.json')
    require(smoke['status'] == 'passed', 'Original inference is not accepted')
    for name in ('rgb', 'depth', 'meta'):
        suffix = 'mat' if name == 'meta' else 'png'
        require(digest(base / name / ('0000.' + suffix)) == smoke['protocol']['input_sha256'][name], 'Original input mismatch')
    candidates_path = root / 'results/P3/scene_0003_realsense_0000_candidates.npy'
    require(digest(candidates_path) == smoke['candidates_sha256'], 'Frozen candidate bytes changed')
    require(tuple(smoke['input_shape_hw']) == depth.shape == rgb.shape[:2], 'Original frame shape mismatch')
    require(intrinsic.tolist() == smoke['intrinsic_matrix'] and factor == smoke['factor_depth'], 'Original camera mismatch')
    arrays = reconstruct_points(depth, intrinsic, factor, instances, settings['maximum_depth_m'])
    require(len(arrays['valid_points']) == smoke['valid_points_before_sampling'], 'Original valid-point count mismatch')
    require(len(arrays['valid_points']) == settings.get('valid_point_count', len(arrays['valid_points'])), 'Protocol valid-point count mismatch')
    require(smoke['sampled_indices_sha256'] == settings.get('sampled_indices_sha256', smoke['sampled_indices_sha256']),
            'Protocol sampled-index identity mismatch')
    indices = sample_indices(len(arrays['valid_points']), settings['sample_seed'], settings['num_points'],
                             smoke['sampled_indices_sha256'])
    arrays.update(sampled_indices=indices, sampled_points=arrays['valid_points'][indices],
                  sampled_pixel_indices=arrays['valid_pixel_indices'][indices])
    reprojection = reprojection_error(arrays['valid_points'], arrays['valid_pixel_indices'], depth.shape, intrinsic)
    require(reprojection['maximum_euclidean_pixels'] <= settings.get('reprojection_tolerance_pixels', .001),
            'Pixel reprojection exceeds frozen tolerance')
    labels, counts = np.unique(arrays['valid_instance_ids'], return_counts=True)
    info = {'shape_hw': list(depth.shape), 'intrinsic_matrix': intrinsic.tolist(), 'factor_depth': factor,
            'valid_point_count': len(arrays['valid_points']), 'sampled_point_count': len(indices),
            'sampled_indices_sha256': smoke['sampled_indices_sha256'],
            'per_instance_valid_points': {str(int(k)): int(v) for k, v in zip(labels, counts)},
            'reprojection': reprojection,
            'invalid_depth_pixels': int(depth.size - len(arrays['valid_points']))}
    return arrays, info


def load_candidates(root, settings):
    candidates = np.load(root / 'results/P3/scene_0003_realsense_0000_candidates.npy', allow_pickle=False)
    require(candidates.dtype == np.float32 and candidates.shape == (settings['candidate_count'], 17)
            and np.isfinite(candidates).all(), 'Original candidates must be finite float32 [236,17]')
    require((candidates[:, 16] == -1).all(), 'Original unspecified object IDs changed')
    return candidates


def index_points(sampled_points):
    require(sampled_points.dtype == np.float32 and sampled_points.ndim == 2 and sampled_points.shape[1] == 3
            and np.isfinite(sampled_points).all(), 'Invalid sampled-point array')
    mapping = {}
    for position, point in enumerate(sampled_points):
        mapping.setdefault(tuple(float(x) for x in point), []).append(position)
    return mapping


def associate_rows(candidates, arrays, image_shape, start, stop, mapping=None):
    mapping = index_points(arrays['sampled_points']) if mapping is None else mapping
    require(0 <= start < stop <= len(candidates), 'Invalid block bounds')
    rows = []
    width = image_shape[1]
    for original_index in range(start, stop):
        candidate = candidates[original_index]
        require(candidate.dtype == np.float32 and candidate.shape == (17,) and np.isfinite(candidate).all(), 'Invalid candidate')
        positions = mapping.get(tuple(float(x) for x in candidate[13:16]), [])
        matches = []
        for position in positions:
            valid_index = int(arrays['sampled_indices'][position])
            pixel = int(arrays['valid_pixel_indices'][valid_index])
            matches.append({'sample_position': position, 'valid_point_index': valid_index,
                            'pixel_index': pixel, 'row': pixel // width, 'col': pixel % width,
                            'instance_id': int(arrays['valid_instance_ids'][valid_index])})
        labels = sorted({item['instance_id'] for item in matches})
        instance_id = labels[0] if len(labels) == 1 else None
        status = ('unmatched' if not matches else 'ambiguous' if len(labels) > 1
                  else 'background' if instance_id == 0 else 'assigned')
        kind = ('none' if not matches else 'unique' if len(matches) == 1
                else 'duplicate_conflict' if len(labels) > 1 else 'duplicate_consensus')
        nearest = None
        if not matches:
            distances = np.linalg.norm(arrays['sampled_points'].astype(np.float64) - candidate[13:16].astype(np.float64), axis=1)
            nearest = float(distances.min()) if len(distances) else None
        rows.append({'original_index': original_index, 'candidate_row': candidate.tolist(),
                     'status': status, 'match_kind': kind, 'instance_id': instance_id,
                     'matches': matches, 'nearest_distance_m': nearest})
    return rows


def verify_rows(rows, candidates, arrays, shape, start, stop):
    """Validate cached provenance without reconstructing points or finding matches."""
    require(isinstance(rows, list) and len(rows) == stop - start, 'Wrong cached block size')
    for original, row in zip(range(start, stop), rows):
        require(row['original_index'] == original and row['candidate_row'] == candidates[original].tolist(), 'Candidate row changed')
        require(row['status'] in STATUSES and isinstance(row['matches'], list), 'Invalid association status')
        matches = row['matches']
        positions = [m['sample_position'] for m in matches]
        require(positions == sorted(set(positions)), 'Duplicated/unordered sampled positions')
        for match in matches:
            position, valid = match['sample_position'], match['valid_point_index']
            require(type(position) is int and 0 <= position < len(arrays['sampled_indices']), 'Invalid sampled position')
            require(valid == int(arrays['sampled_indices'][position]), 'Sample provenance changed')
            pixel = int(arrays['valid_pixel_indices'][valid])
            require(match['pixel_index'] == pixel and match['row'] == pixel // shape[1]
                    and match['col'] == pixel % shape[1], 'Pixel provenance changed')
            require(match['instance_id'] == int(arrays['valid_instance_ids'][valid]), 'Instance provenance changed')
            require(np.array_equal(arrays['sampled_points'][position], candidates[original, 13:16]), 'Not an exact XYZ match')
        labels = sorted({m['instance_id'] for m in matches})
        expected_id = labels[0] if len(labels) == 1 else None
        expected_status = ('unmatched' if not matches else 'ambiguous' if len(labels) > 1
                           else 'background' if expected_id == 0 else 'assigned')
        expected_kind = ('none' if not matches else 'unique' if len(matches) == 1
                         else 'duplicate_conflict' if len(labels) > 1 else 'duplicate_consensus')
        require(row['status'] == expected_status and row['instance_id'] == expected_id
                and row['match_kind'] == expected_kind, 'Cached status inconsistent')
        require((bool(matches) and row['nearest_distance_m'] is None) or
                (not matches and isinstance(row['nearest_distance_m'], (float, int))
                 and math.isfinite(row['nearest_distance_m']) and row['nearest_distance_m'] >= 0), 'Invalid nearest diagnostic')


def commit_receipt(path, binding_id):
    once_json(path.with_suffix('.sha256.json'), {'schema': 11, 'binding_fingerprint': binding_id,
                                               'file': path.name, 'sha256': digest(path)})


def verify_receipt(path, binding_id):
    receipt = read(path.with_suffix('.sha256.json'))
    require(receipt == {'schema': 11, 'binding_fingerprint': binding_id, 'file': path.name,
                        'sha256': digest(path)}, 'Receipt mismatch: ' + str(path))


def point_cache(root, out, settings, binding_id, check_only=False):
    receipt_path = out / 'points_commit.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['binding_fingerprint'] == binding_id and set(receipt['files']) == set(n + '.npy' for n in ARRAY_NAMES), 'Invalid point receipt')
        for name, expected in receipt['files'].items():
            require(digest(out / name) == expected, 'Point array hash mismatch: ' + name)
        arrays = {name: np.load(out / (name + '.npy'), allow_pickle=False) for name in ARRAY_NAMES}
        return arrays, receipt['information'], True
    require(not check_only, 'No committed point cache')
    arrays, info = load_originals(root, settings)
    for name, value in arrays.items():
        once_array(out / (name + '.npy'), value)
    once_json(receipt_path, {'schema': 11, 'binding_fingerprint': binding_id, 'information': info,
                            'files': {name + '.npy': digest(out / (name + '.npy')) for name in ARRAY_NAMES}})
    return arrays, info, False


def summary_rows(rows, info):
    counts = {status: sum(row['status'] == status for row in rows) for status in STATUSES}
    per_instance = {}
    for row in rows:
        if row['status'] == 'assigned':
            key = str(row['instance_id'])
            per_instance[key] = per_instance.get(key, 0) + 1
    return {'candidate_count': len(rows), 'status_counts': counts,
            'exact_matched_candidates': len(rows) - counts['unmatched'],
            'duplicate_xyz_candidates': sum(len(row['matches']) > 1 for row in rows),
            'per_instance_candidate_count': per_instance,
            'valid_point_count': info['valid_point_count'],
            'per_instance_valid_points': info['per_instance_valid_points'],
            'sampled_indices_sha256': info['sampled_indices_sha256']}


def complete_report(binding_id, rows, info):
    summary = summary_rows(rows, info)
    checks = {'all_original_candidates_preserved': [r['original_index'] for r in rows] == list(range(len(rows))),
              'all_candidates_exact_sample_matches': summary['status_counts']['unmatched'] == 0,
              'point_counts_conserved': sum(info['per_instance_valid_points'].values()) == info['valid_point_count'],
              'all_candidate_statuses_accounted': sum(summary['status_counts'].values()) == len(rows)}
    return {'schema': 11, 'status': 'passed' if all(checks.values()) else 'failed',
            'acceptance': 'single_frame_prediction_interface_only', 'binding_fingerprint': binding_id,
            'checks': checks, 'summary': summary, 'point_information': info, 'rows': rows,
            'uses_ground_truth_labels': False, 'uses_object_geometry': False, 'gpu_used': False,
            'is_dataset_ap_result': False, 'candidate_scores_modified': False,
            'translation_semantics': 'frozen fp2_xyz surface seed, not the closing/contact center',
            'limitations': ['Predicted instance IDs are frame-local, never dataset object IDs.',
                            'Background and ambiguous associations are retained; no nearest-point fallback.',
                            'Association is not force closure, collision validity, success or segmentation improvement.',
                            'No grasp prediction was rerun; original seed order is not recoverable from XYZ duplicates.']}


def allowed_members(total_blocks):
    allowed = {'binding.json', '.process.lock', 'points_commit.json', 'completion.json'}
    allowed.update(name + '.npy' for name in ARRAY_NAMES)
    allowed.update('blocks/block_{:04d}{}'.format(i, suffix)
                   for i in range(total_blocks) for suffix in ('.json', '.sha256.json'))
    return allowed


def known_temporary(relative, total_blocks):
    match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', relative)
    return bool(match and match.group(1) in allowed_members(total_blocks) - {'.process.lock'})


def validate_inventory(out, total_blocks, complete=False):
    allowed = allowed_members(total_blocks)
    for path in out.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(out).as_posix()
        if known_temporary(relative, total_blocks):
            continue  # Interrupted uncommitted writes are preserved and never accepted as results.
        require(relative in allowed, 'Unexpected cache member: ' + relative)
    if complete:
        require((out / 'completion.json').is_file(), 'Completion missing')


def execute(root, config, max_new_blocks=None, check_only=False):
    protocol = read(config)
    settings = validate_settings(protocol)
    out = safe_file(root, protocol['outputs']['association_dir'])
    report_path = safe_file(root, protocol['outputs']['association_report'])
    require(out == root / 'results/P3/grasp_instance_association_v11'
            and report_path == out.with_suffix('.json')
            and protocol['outputs']['perception_dir'] == 'results/P3/instance_interface_v11',
            'Round 11 artifact paths changed')
    budget_paths = [safe_file(root, value) for value in protocol['outputs'].values()]
    budget_paths += [root / 'results/P3' / name for name in (
        'instance_interface_v11_resume.json', 'grasp_instance_association_v11_resume.json')]
    monitor = Monitor(root, out, settings, budget_paths)
    try:
        monitor.check()
        binding = make_binding(root, config, protocol, settings)
        binding_id = fingerprint(binding)
        candidates = load_candidates(root, settings)
        total_blocks = math.ceil(len(candidates) / settings['block_size'])

        def process():
            if (out / 'binding.json').exists():
                require(read(out / 'binding.json') == binding, 'Association binding changed')
            else:
                require(not check_only, 'Association binding absent')
                existing = [p for p in out.rglob('*') if p.is_file() and p.name != '.process.lock'
                            and not (known_temporary(p.relative_to(out).as_posix(), total_blocks)
                                     and p.relative_to(out).as_posix().startswith('binding.json.'))]
                require(not existing, 'Unbound existing association cache')
                once_json(out / 'binding.json', binding)
            validate_inventory(out, total_blocks)
            completed = (out / 'completion.json').exists()
            require(not check_only or completed, 'Read-only check requires committed completion')
            arrays, info, points_reused = point_cache(root, out, settings, binding_id, check_only or completed)
            require(arrays['valid_points'].shape == (info['valid_point_count'], 3), 'Point shape mismatch')
            require(arrays['sampled_points'].shape == (settings['num_points'], 3), 'Sample shape mismatch')
            rows, reused, added, mapping = [], 0, 0, None
            for block_index in range(total_blocks):
                monitor.check()
                start = block_index * settings['block_size']
                stop = min(start + settings['block_size'], len(candidates))
                path = out / 'blocks' / ('block_{:04d}.json'.format(block_index))
                if path.with_suffix('.sha256.json').exists():
                    require(path.exists(), 'Receipt without block')
                    verify_receipt(path, binding_id)
                    block = read(path)
                    require(block['binding_fingerprint'] == binding_id and block['start'] == start
                            and block['stop'] == stop and block['schema'] == 11, 'Wrong block identity')
                    verify_rows(block['rows'], candidates, arrays, info['shape_hw'], start, stop)
                    reused += 1
                else:
                    require(not (check_only or completed), 'Completion contains uncommitted block')
                    if max_new_blocks is not None and added >= max_new_blocks:
                        break
                    if mapping is None:
                        mapping = index_points(arrays['sampled_points'])
                    new_rows = associate_rows(candidates, arrays, info['shape_hw'], start, stop, mapping)
                    block = {'schema': 11, 'binding_fingerprint': binding_id, 'start': start, 'stop': stop, 'rows': new_rows}
                    once_json(path, block)
                    commit_receipt(path, binding_id)
                    added += 1
                rows.extend(block['rows'])
            if len(rows) != len(candidates):
                return {'status': 'partial', 'new_blocks': added, 'reused_blocks': reused,
                        'completed_candidates': len(rows), 'points_reused': points_reused}
            report = complete_report(binding_id, rows, info)
            if completed:
                require(read(report_path) == report, 'Final report changed')
                completion = read(out / 'completion.json')
                require(completion['binding_fingerprint'] == binding_id, 'Completion binding mismatch')
                expected_files = ['binding.json', 'points_commit.json'] + [n + '.npy' for n in ARRAY_NAMES]
                expected_files += ['blocks/block_{:04d}{}'.format(i, suffix)
                                   for i in range(total_blocks) for suffix in ('.json', '.sha256.json')]
                require(set(completion['files']) == set(expected_files), 'Completion inventory mismatch')
                for name, expected in completion['files'].items():
                    require(digest(out / name) == expected, 'Completion artifact changed: ' + name)
                require(digest(report_path) == completion['report_sha256'], 'Completion report hash mismatch')
            else:
                monitor.check(planned_json_bytes(report_path, report))
                once_json(report_path, report)
                files = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*'))
                         if p.is_file() and p.name != '.process.lock' and not p.name.endswith('.tmp')}
                final_commit = {'schema': 11, 'binding_fingerprint': binding_id,
                          'report_sha256': digest(report_path), 'files': files, 'total_blocks': total_blocks,
                          'status': report['status']}
                monitor.check(planned_json_bytes(out / 'completion.json', final_commit))
                once_json(out / 'completion.json', final_commit)
            monitor.check()
            require('torch' not in sys.modules, 'Association must not import torch')
            return {'status': report['status'], 'new_blocks': added, 'reused_blocks': reused,
                    'points_reused': points_reused, 'check_only': check_only, 'summary': report['summary'],
                    'peak_rss_bytes': monitor.peak, 'elapsed_seconds': time.monotonic() - monitor.started}

        if check_only:
            return process()
        with lock(out):
            return process()
    finally:
        monitor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--max-new-blocks', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    require(args.max_new_blocks is None or args.max_new_blocks > 0, 'max-new-blocks must be positive')
    require(not (args.check_only and args.max_new_blocks is not None), 'Read-only cannot request new blocks')
    config = args.config.resolve() if args.config else root / 'configs/instance_grasp_interface_v11.json'
    result = execute(root, config, args.max_new_blocks, args.check_only)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    require(result['status'] != 'failed', 'Engineering acceptance did not pass; negative outputs retained')


if __name__ == '__main__':
    main()
