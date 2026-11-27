"""Frozen two-arm candidate generation with durable input and forward commits."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import threading
import time
import uuid

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['full_scene', 'predicted_foreground']
CHECKPOINT = 'checkpoints/pretrained/checkpoint-rs.tar'
CHECKPOINT_SHA = '60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868'
OLD_CANDIDATES = 'results/P3/scene_0003_realsense_0000_candidates.npy'
OLD_SMOKE = 'results/P3/inference_smoke.json'
ASSOCIATION = 'results/P3/grasp_instance_association_v11'
PERCEPTION = 'results/P3/instance_interface_v11'
OUTPUT_ROOT = 'results/P3/grasp_foreground_v12'
ARRAYS = ('eligible_indices', 'sampled_eligible_indices', 'sampled_valid_indices', 'sampled_points', 'sampled_pixel_indices')
ARTIFACTS = tuple(name + '.npy' for name in ARRAYS) + ('forward_bundle.npz', 'candidates.npy', 'seed_provenance.npz')
JSON_FILES = ('binding.json', 'input_commit.json', 'report.json', 'completion.json')
ACTIVE_RESOURCE_GUARD = None


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def path_in_root(name):
    path = Path(name)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Path escapes project root')
    return path


def atomic_bytes(path, payload):
    if ACTIVE_RESOURCE_GUARD is not None:
        ACTIVE_RESOURCE_GUARD.check(additional_bytes=len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def once_json(path, value, readonly=False):
    if path.exists():
        require(read(path) == value, 'Immutable report changed: ' + path.name)
    else:
        require(not readonly, 'Missing read-only report: ' + path.name)
        atomic_bytes(path, json_bytes(value))


def serialize(value, archive=False):
    buffer = io.BytesIO()
    if archive:
        np.savez(buffer, **value)
    else:
        np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def artifact(directory, name, binding):
    path = directory / name
    receipt_path = directory / (name + '.pending.json')
    if not receipt_path.exists():
        require(not path.exists(), 'Unbound artifact: ' + name)
        return None
    receipt = read(receipt_path)
    require(receipt['schema'] == 12 and receipt['file'] == name and receipt['binding_fingerprint'] == binding,
            'Artifact binding changed: ' + name)
    if not path.exists():
        return None
    require(path.stat().st_size == receipt['size_bytes'] and digest(path) == receipt['sha256'], 'Artifact corrupt: ' + name)
    return receipt


def commit_artifact(directory, name, payload, binding, evidence=None, readonly=False):
    expected = {'schema': 12, 'file': name, 'binding_fingerprint': binding, 'sha256': hashlib.sha256(payload).hexdigest(),
                'size_bytes': len(payload), 'evidence': evidence or {}}
    prior = artifact(directory, name, binding)
    if prior is not None:
        require(prior == expected, 'Immutable artifact differs: ' + name)
        return prior
    require(not readonly, 'Missing committed artifact: ' + name)
    intent = directory / (name + '.pending.json')
    if intent.exists():
        prior = read(intent)
        require(all(prior[key] == expected[key] for key in ('schema', 'file', 'binding_fingerprint', 'sha256', 'size_bytes')),
                'Interrupted artifact changed: ' + name)
        expected = prior
    else:
        atomic_bytes(intent, json_bytes(expected))
    atomic_bytes(directory / name, payload)
    return artifact(directory, name, binding)


def validate_inventory(directory):
    allowed = set(ARTIFACTS) | set(JSON_FILES) | {name + '.pending.json' for name in ARTIFACTS} | {'.process.lock'}
    for path in directory.iterdir():
        require(path.is_file(), 'Unexpected generation subdirectory')
        require(path.name in allowed or any(re.fullmatch(re.escape(name) + r'\.tmp-[0-9a-f]{32}', path.name) for name in allowed),
                'Unexpected generation artifact: ' + path.name)


@contextlib.contextmanager
def exclusive_lock(directory, readonly=False):
    if not readonly:
        directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.process.lock'
    require(not readonly or path.exists(), 'Missing completed generation lock')
    with path.open('r+b' if readonly else 'a+b') as stream:
        if not readonly:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_protocol(protocol, arm):
    require(protocol['schema'] == 12 and protocol['arms'] == ARMS and arm in ARMS, 'Unsupported arm or protocol')
    require(protocol['frame'] == {'scene': 3, 'camera': 'realsense', 'frame': '0000'}, 'Only frozen training frame is allowed')
    require(protocol['seed'] == 20261021 and protocol['num_points'] == 20000 and protocol['maximum_depth_m'] == 2.,
            'Frozen sampling policy differs')
    require(protocol['checkpoint']['path'] == CHECKPOINT and protocol['checkpoint']['sha256'] == CHECKPOINT_SHA,
            'Frozen model checkpoint differs')
    require(protocol['checkpoint']['weights_only'] is True and protocol['checkpoint']['strict'] is True,
            'Safe strict model load policy differs')
    require(protocol['association_input_dir'] == ASSOCIATION and protocol['outputs']['root'] == OUTPUT_ROOT,
            'Frozen input/output paths differ')
    limits = protocol['generation_resources']
    for key, expected in {'cpu_threads': 2, 'rss_soft_bytes': 8 * 1024 ** 3,
            'gpu_allocated_soft_bytes': 4 * 1024 ** 3, 'gpu_reserved_soft_bytes': 4 * 1024 ** 3,
            'invocation_soft_seconds': 300, 'output_limit_bytes': 512 * 1024 ** 2,
            'disk_reserve_bytes': 60 * 1024 ** 3}.items():
        require(limits[key] == expected, 'Generation resource changed: ' + key)


def verify_seals(protocol):
    association = path_in_root(ASSOCIATION)
    completed = read(association / 'completion.json')
    require(completed['schema'] == 11 and completed['status'] == 'passed', 'R11 association not accepted')
    for name, expected in completed['files'].items():
        path = (association / name).resolve()
        require(path.is_relative_to(association) and digest(path) == expected, 'R11 association seal differs')
    report_path = association.with_suffix('.json')
    require(digest(report_path) == completed['report_sha256'], 'R11 association report differs')
    report = read(report_path)
    require(report['uses_ground_truth_labels'] is False and report['uses_object_geometry'] is False, 'R11 source is not label-free')
    perception = path_in_root(PERCEPTION)
    sealed = read(perception / 'completion.json')
    require(sealed['status'] == 'completed' and sealed['ground_truth_labels_read'] is False, 'R11 prediction not sealed')
    for name, expected in sealed['outputs'].items():
        path = (perception / name).resolve()
        require(path.is_relative_to(perception) and digest(path) == expected, 'R11 prediction seal differs')
    require(digest(path_in_root(sealed['report_file'])) == sealed['report_sha256'], 'R11 prediction report differs')
    return {ASSOCIATION + '/completion.json': digest(association / 'completion.json'),
            PERCEPTION + '/completion.json': digest(perception / 'completion.json')}


def make_binding(protocol, config_path, arm):
    require(Path(sys.prefix).resolve() == path_in_root('runtime/envs/vision'), 'Use frozen vision environment')
    hashes = dict(protocol['input_hashes'], **protocol['source_hashes'])
    required = {ASSOCIATION + '/' + name for name in ('valid_points.npy', 'valid_pixel_indices.npy',
        'valid_instance_ids.npy', 'completion.json', 'binding.json')}
    required.update((PERCEPTION + '/completion.json', PERCEPTION + '/binding.json',
                     OLD_CANDIDATES, OLD_SMOKE, CHECKPOINT))
    require(required.issubset(hashes), 'Missing frozen label-free input hashes')
    for name, expected in hashes.items():
        require('/label/' not in name and '/annotations/' not in name and 'coverage_expanded_train_scene_0003_0000' not in name,
                'Ground-truth diagnostic path forbidden in generation binding')
        require(digest(path_in_root(name)) == expected, 'Frozen source/input changed: ' + name)
    require(digest(path_in_root(CHECKPOINT)) == CHECKPOINT_SHA, 'Checkpoint checksum differs')
    seals = verify_seals(protocol)
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__}
    for package, key in (('torch', 'torch'), ('torchvision', 'torchvision'), ('scipy', 'scipy'), ('Pillow', 'pillow')):
        versions[key] = importlib.metadata.version(package)
    require(all(protocol['runtime'][key] == value for key, value in versions.items()), 'Frozen generation runtime differs')
    numerical = {}
    for folder in ('runtime/build/graspnet', 'runtime/envs/vision/Lib/site-packages/pointnet2',
                   'runtime/envs/vision/Lib/site-packages/knn_pytorch'):
        for path in sorted(path_in_root(folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.pyd', '.dll', '.cu', '.cpp', '.h'):
                numerical[path.relative_to(ROOT).as_posix()] = digest(path)
    require(bool(numerical), 'Missing model source/binary identity')
    resolved_extensions = {}
    for module in ('pointnet2._ext', 'knn_pytorch.knn_pytorch'):
        spec = importlib.util.find_spec(module)
        require(spec is not None and spec.origin is not None, 'Missing CUDA extension: ' + module)
        path = path_in_root(spec.origin)
        name = path.relative_to(ROOT).as_posix()
        require(name in numerical and digest(path) == numerical[name], 'Resolved extension is outside bound binary inventory')
        resolved_extensions[module] = {'path': name, 'sha256': numerical[name]}
    return {'schema': 12, 'arm': arm, 'protocol_sha256': digest(config_path), 'script_sha256': digest(__file__),
        'inputs': hashes, 'r11_seals': seals, 'numerical_source_and_extension_hashes': numerical,
        'resolved_extensions': resolved_extensions,
        'runtime': versions, 'interpreter_sha256': digest(sys.executable),
        'uses_ground_truth_labels': False, 'uses_object_geometry': False, 'new_training_updates': 0,
        'model': 'Frozen GraspNet XYZ-only inference; FP32; no ranking modification',
        'determinism': 'Same original seeds/cuDNN/TF32 settings; custom CUDA bitwise determinism is not guaranteed'}


def sampling_inputs(points, pixels, instances, arm, seed=20261021, size=20000):
    require(arm in ARMS, 'Unknown arm')
    require(points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 3
            and np.isfinite(points).all() and np.all((points[:, 2] > 0) & (points[:, 2] <= 2.)), 'Invalid frozen points')
    require(pixels.dtype == np.int64 and pixels.shape == (len(points),) and np.all((pixels >= 0) & (pixels < 720 * 1280))
            and np.all(np.diff(pixels) > 0), 'Invalid original pixel provenance')
    require(instances.dtype == np.int32 and instances.shape == (len(points),) and np.all(instances >= 0), 'Invalid predicted IDs')
    eligible = np.arange(len(points), dtype=np.int64) if arm == 'full_scene' else np.flatnonzero(instances > 0).astype(np.int64)
    enough = len(eligible) >= size
    sampled = np.random.RandomState(seed).choice(len(eligible), size, replace=False).astype(np.int64) if enough else np.empty(0, np.int64)
    valid = eligible[sampled]
    arrays = {'eligible_indices': eligible, 'sampled_eligible_indices': sampled, 'sampled_valid_indices': valid,
              'sampled_points': points[valid], 'sampled_pixel_indices': pixels[valid]}
    return arrays, {'input_status': 'ready' if enough else 'insufficient_input',
        'valid_point_count': len(points), 'eligible_point_count': len(eligible), 'sampled_point_count': len(sampled),
        'sampling_with_replacement': False, 'seed': seed, 'requested_points': size}


def load_input_arrays(protocol, arm):
    source = path_in_root(ASSOCIATION)
    points, pixels, instances = [np.load(source / name, allow_pickle=False) for name in
                                ('valid_points.npy', 'valid_pixel_indices.npy', 'valid_instance_ids.npy')]
    arrays, info = sampling_inputs(points, pixels, instances, arm, protocol['seed'], protocol['num_points'])
    if arm == 'full_scene':
        smoke = read(path_in_root(OLD_SMOKE))
        # NumPy choice uses native C-long indices: Windows NumPy1.26 returns
        # int32, while the new portable cache deliberately stores int64 indices.
        native_indices = np.random.RandomState(protocol['seed']).choice(len(points), protocol['num_points'], replace=False)
        original_indices = np.load(source / 'sampled_indices.npy', allow_pickle=False)
        require(original_indices.dtype == native_indices.dtype and np.array_equal(original_indices, native_indices)
                and np.array_equal(arrays['sampled_valid_indices'], native_indices), 'Full arm differs from legacy native sampling')
        actual = hashlib.sha256(native_indices.tobytes()).hexdigest()
        require(info['valid_point_count'] == smoke['valid_points_before_sampling']
                and actual == smoke['sampled_indices_sha256'], 'Full arm does not reproduce frozen original sampling')
        info['legacy_full_sampled_indices_sha256'] = actual
        info['legacy_sampling_index_dtype'] = str(native_indices.dtype)
    return arrays, info


def validate_candidates(candidates):
    require(candidates.dtype == np.float32 and candidates.ndim == 2 and candidates.shape[1] == 17
            and 0 <= len(candidates) <= 1024 and np.isfinite(candidates).all(), 'Invalid candidate array')
    if not len(candidates):
        return {'candidate_count': 0, 'geometry_checks_applicable': False}
    rotations = candidates[:, 4:13].reshape(-1, 3, 3)
    orthogonal = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.)))
    require(orthogonal < 1e-4 and determinant_error < 1e-4, 'Invalid candidate rotation')
    require(np.all((candidates[:, 1] >= 0) & (candidates[:, 1] <= .100001)) and np.allclose(candidates[:, 2], .02)
            and np.isclose(candidates[:, 3, None], [.01, .02, .03, .04], atol=1e-6).any(1).all()
            and np.all((candidates[:, 15] > 0) & (candidates[:, 15] <= 2.)) and np.all(candidates[:, 16] == -1),
            'Candidate width/height/depth/seed/object-ID geometry differs')
    return {'candidate_count': len(candidates), 'geometry_checks_applicable': True,
        'maximum_rotation_orthogonality_error': orthogonal, 'maximum_determinant_error': determinant_error}


def validate_seed_chain(bundle, arrays):
    candidates = bundle['candidates']
    validate_candidates(candidates)
    selected = bundle['selected_seed_indices']
    sa1, sa2, fp2 = (bundle[key] for key in ('sa1_indices', 'sa2_indices', 'fp2_xyz'))
    if not len(sa1):
        require(not len(candidates) and not len(selected) and not len(sa2) and fp2.shape == (0, 3), 'Invalid nonforward bundle')
        return
    require(sa1.dtype == sa2.dtype == selected.dtype == np.int64 and sa1.shape == (2048,) and sa2.shape == (1024,)
            and fp2.dtype == np.float32 and fp2.shape == (1024, 3), 'Invalid seed arrays')
    require(np.all((sa1 >= 0) & (sa1 < len(arrays['sampled_points']))) and np.all((sa2 >= 0) & (sa2 < 2048)), 'Seed index out of range')
    composed = sa1[sa2]
    require(np.array_equal(arrays['sampled_points'][composed], fp2), 'Actual SA seed composition does not reproduce fp2 XYZ')
    require(selected.shape == (len(candidates),) and np.all((selected >= 0) & (selected < 1024))
            and np.all(np.diff(selected) > 0), 'Invalid decoded seed indices')
    require(np.array_equal(candidates[:, 13:16], fp2[selected]), 'Candidate seed translation differs from saved source')


def empty_bundle():
    return {'candidates': np.empty((0, 17), np.float32), 'sa1_indices': np.empty(0, np.int64),
            'sa2_indices': np.empty(0, np.int64), 'selected_seed_indices': np.empty(0, np.int64),
            'fp2_xyz': np.empty((0, 3), np.float32), 'legacy_fp2_indices': np.empty(0, np.int64)}


def forward_once(arrays, protocol):
    import torch
    require(torch.cuda.is_available(), 'Frozen GPU is unavailable')
    actual = {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'gpu': torch.cuda.get_device_name(0)}
    require(all(protocol['runtime'][key] == value for key, value in actual.items()), 'Frozen GPU runtime differs')
    require(digest(path_in_root(CHECKPOINT)) == CHECKPOINT_SHA, 'Checkpoint changed before model load')
    source = path_in_root('runtime/build/graspnet')
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
    # Match the original smoke run. This is not a custom-kernel determinism guarantee.
    torch.use_deterministic_algorithms(False)
    limits = protocol['generation_resources']
    torch.cuda.set_per_process_memory_fraction(min(1., limits['gpu_reserved_soft_bytes'] / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                  cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False)
    checkpoint = torch.load(path_in_root(CHECKPOINT), map_location='cpu', weights_only=True)
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


def rss_bytes():
    if os.name != 'nt':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb), 'RSS unavailable')
    return int(value.WorkingSetSize)


class ResourceGuard:
    def __init__(self, protocol):
        self.limits = protocol['generation_resources']
        self.outputs = [path_in_root(name) for name in protocol['outputs'].values()]
        self.outputs += [path_in_root('results/P3/grasp_foreground_' + phase + '_v12_resume.json')
                         for phase in ('generation', 'geometry')]
        self.started, self.peak = time.monotonic(), 0
        self.stop = threading.Event()

    def check(self, additional_bytes=0):
        self.peak = max(self.peak, rss_bytes())
        require(self.peak <= self.limits['rss_soft_bytes'], 'RSS resource limit exceeded')
        require(time.monotonic() - self.started <= self.limits['invocation_soft_seconds'], 'Time limit exceeded')
        require(shutil.disk_usage(ROOT).free - additional_bytes >= self.limits['disk_reserve_bytes'], 'Disk reserve unavailable')
        total = 0
        files = set()
        for output in self.outputs:
            if output.is_file():
                files.add(output)
            elif output.is_dir():
                files.update(path for path in output.rglob('*') if path.is_file())
        for path in files:
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass  # Atomic rename race; next sample measures the final path.
        require(total + additional_bytes <= self.limits['output_limit_bytes'], 'All-arm output limit exceeded')

    def watch(self):
        while not self.stop.wait(self.limits.get('sample_interval_seconds', .1)):
            try:
                self.check()
            except Exception as error:
                print(json.dumps({'status': 'resource_limit_exceeded', 'reason': str(error)}), flush=True)
                os._exit(70)

    def __enter__(self):
        global ACTIVE_RESOURCE_GUARD
        self.check()
        self.previous_guard = ACTIVE_RESOURCE_GUARD
        ACTIVE_RESOURCE_GUARD = self
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        global ACTIVE_RESOURCE_GUARD
        self.stop.set()
        self.thread.join(1)
        ACTIVE_RESOURCE_GUARD = self.previous_guard


def run(config_path, arm, stop_after_inputs=False, check_only=False):
    protocol = read(config_path)
    validate_protocol(protocol, arm)
    directory = path_in_root(protocol['outputs']['root']) / arm / 'generation'
    complete_path = directory / 'completion.json'
    readonly = check_only or complete_path.exists()
    require(not check_only or complete_path.exists(), 'Check-only needs completed generation')
    with ResourceGuard(protocol) as guard:
        binding_record = make_binding(protocol, config_path, arm)
        binding = fingerprint(binding_record)
        with exclusive_lock(directory, readonly):
            validate_inventory(directory)
            if not (directory / 'binding.json').exists():
                require(all(path.name == '.process.lock' or re.fullmatch(r'binding\.json\.tmp-[0-9a-f]{32}', path.name)
                            for path in directory.iterdir()), 'Unbound generation cache')
            once_json(directory / 'binding.json', binding_record, readonly)
            arrays, information = load_input_arrays(protocol, arm)
            receipts = {}
            for name in ARRAYS:
                receipts[name + '.npy'] = commit_artifact(directory, name + '.npy', serialize(arrays[name]), binding, readonly=readonly)
            input_commit = {'schema': 12, 'arm': arm, 'binding_fingerprint': binding, 'status': 'inputs_committed',
                            'information': information, 'files': {name: receipt['sha256'] for name, receipt in receipts.items()}}
            once_json(directory / 'input_commit.json', input_commit, readonly)
            if stop_after_inputs and not readonly:
                result = {'status': 'inputs_committed', 'arm': arm, 'input_status': information['input_status'],
                          'new_model_forward_calls': 0, 'input_commit_sha256': digest(directory / 'input_commit.json')}
                print(json.dumps(result), flush=True)
                return result
            saved = artifact(directory, 'forward_bundle.npz', binding)
            new_calls = 0
            if saved is None:
                require(not readonly, 'Missing completed forward bundle')
                if information['input_status'] == 'insufficient_input':
                    bundle, evidence = empty_bundle(), {'model_forward_calls': 0, 'reason': 'insufficient_input'}
                else:
                    with exclusive_lock(path_in_root('runtime/grasp_foreground_v12/generation_gpu_lock')):
                        bundle, evidence = forward_once(arrays, protocol)
                    new_calls = 1
                saved = commit_artifact(directory, 'forward_bundle.npz', serialize(bundle, True), binding, evidence)
            else:
                with np.load(directory / 'forward_bundle.npz', allow_pickle=False) as archive:
                    bundle = {name: archive[name].copy() for name in archive.files}
            validate_seed_chain(bundle, arrays)
            calls = saved['evidence']['model_forward_calls']
            require(calls == (0 if information['input_status'] == 'insufficient_input' else 1), 'Forward budget/evidence mismatch')
            candidates = bundle['candidates']
            checks = validate_candidates(candidates)
            commit_artifact(directory, 'candidates.npy', serialize(candidates), binding, readonly=readonly)
            provenance = {key: value for key, value in bundle.items() if key != 'candidates'}
            if len(bundle['sa1_indices']):
                source_indices = bundle['sa1_indices'][bundle['sa2_indices']][bundle['selected_seed_indices']]
            else:
                source_indices = np.empty(0, np.int64)
            provenance.update(candidate_sampled_indices=source_indices,
                candidate_valid_indices=arrays['sampled_valid_indices'][source_indices],
                candidate_pixel_indices=arrays['sampled_pixel_indices'][source_indices])
            commit_artifact(directory, 'seed_provenance.npz', serialize(provenance, True), binding, readonly=readonly)
            legacy = {'comparison_applicable': arm == 'full_scene', 'selection_gate': False, 'repeat_on_mismatch': False}
            if arm == 'full_scene':
                old = np.load(path_in_root(OLD_CANDIDATES), allow_pickle=False)
                same = candidates.shape == old.shape and candidates.dtype == old.dtype
                legacy.update(old_candidate_count=len(old), same_shape_dtype=same,
                    bitwise_array_equal=bool(same and candidates.tobytes() == old.tobytes()),
                    maximum_absolute_difference=float(np.max(np.abs(candidates - old))) if same and candidates.size else None)
            outcome = 'insufficient_input' if information['input_status'] == 'insufficient_input' else 'empty_candidates' if not len(candidates) else 'generated'
            report = {'schema': 12, 'status': 'completed', 'arm': arm, 'outcome': outcome,
                'input_status': information['input_status'], 'input_information': information, 'binding_fingerprint': binding,
                'candidate_count': len(candidates), 'candidates_sha256': digest(directory / 'candidates.npy'),
                'forward_calls': calls, 'forward_calls_committed': calls, 'forward_evidence': saved['evidence'], 'geometry_checks': checks,
                'legacy_baseline_diagnostic': legacy, 'uses_ground_truth_labels': False, 'uses_object_geometry': False,
                'new_training_updates': 0, 'is_dataset_ap_result': False,
                'limitations': ['One reused training frame and one sampling seed only',
                    'Foreground union changes point density and context together', 'Candidate score is not calibrated success probability',
                    'Custom CUDA bitwise reproducibility is not guaranteed; completed arrays are reused without forward',
                    'Empty/insufficient candidates are retained as negative results, not retried']}
            once_json(directory / 'report.json', report, readonly)
            names = list(ARTIFACTS) + [name + '.pending.json' for name in ARTIFACTS] + ['binding.json', 'input_commit.json', 'report.json']
            completion = {'schema': 12, 'status': 'completed', 'arm': arm, 'outcome': outcome,
                'input_status': information['input_status'], 'forward_calls': calls, 'forward_calls_committed': calls, 'candidate_count': len(candidates),
                'binding_fingerprint': binding, 'candidates_sha256': digest(directory / 'candidates.npy'),
                'report_sha256': digest(directory / 'report.json'), 'outputs': {name: digest(directory / name) for name in names}}
            once_json(complete_path, completion, readonly)
            validate_inventory(directory)
            guard.check()
            result = {'status': 'verified_completed_read_only' if readonly else 'completed', 'arm': arm,
                'outcome': outcome, 'input_status': information['input_status'], 'candidate_count': len(candidates),
                'new_model_forward_calls': new_calls, 'forward_calls_committed': calls, 'peak_rss_bytes': guard.peak,
                'elapsed_seconds': time.monotonic() - guard.started}
            print(json.dumps(result), flush=True)
            return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/grasp_foreground_comparison_v12.json')
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--stop-after-inputs', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    require(not (args.stop_after_inputs and args.check_only), 'Modes are mutually exclusive')
    run(path_in_root(args.config), args.arm, args.stop_after_inputs, args.check_only)
