"""One frozen RGB-D forward followed by independently resumable instance decoding."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import numpy as np
import PIL
from PIL import Image
import scipy
from scipy.io import loadmat
import torch
import torchvision
from src.checkpoints import atomic_json, atomic_write, file_sha256, fingerprint, _exclusive_lock
from src.segmentation import PilotInstanceNet, decode_instances

PROTOCOL = 'configs/instance_grasp_interface_v11.json'
OUTPUT_DIRECTORY = 'results/P3/instance_interface_v11'
CHECKPOINT_SHA = '357a9cd7cc8e0e5657afd692899badd39fc0058647e6500f70c081e447ff0496'
CONFIG_FP = '7b3b66664954f4c28707c8e166b0abd6788ac593eb7bb60e8f31b26ed67000d1'
DATA_FP = '18e36769bb9018b8854a1e19c96488bf19947a76bd441dc4e311241db0659fad'
CURSOR = {'epoch': 10, 'next_batch': 0, 'global_step': 1600}
FRAME = {'scene': 3, 'camera': 'realsense', 'frame': '0000'}
DECODER = {'foreground_threshold': .5, 'center_threshold': .35, 'min_pixels': 20, 'max_centers': 64}
PREPROCESSING = {'size': [240, 320], 'original_size': [720, 1280], 'mode': 'rgbd',
    'depth_clip_m': 2.0, 'rgb_mean': [.485, .456, .406], 'rgb_std': [.229, .224, .225],
    'rgb_resize': 'bilinear', 'depth_resize': 'nearest', 'mask_resize': 'nearest', 'augment': False}
RESOURCES = {'cpu_threads': 2, 'rss_soft_bytes': 8589934592, 'gpu_allocated_soft_bytes': 2147483648,
    'gpu_reserved_soft_bytes': 2147483648, 'invocation_soft_seconds': 300,
    'output_limit_bytes': 268435456, 'disk_reserve_bytes': 64424509440, 'sample_interval_seconds': .1}
ARTIFACTS = ('input_tensor.npy', 'heads.npz', 'instances_low.npy', 'instances_original.npy')
JSON_FILES = ('binding.json', 'heads_commit.json', 'completion.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def project_path(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Expected forward-slash relative project path')
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe project path')
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Path escapes project root')
    return path


def validate_protocol(protocol):
    require(protocol['schema'] == 11 and protocol['frame'] == FRAME, 'Only the frozen training frame is allowed')
    require(protocol['preprocessing'] == PREPROCESSING and protocol['decoder'] == DECODER,
            'Frozen preprocessing or decoder changed')
    require(protocol['precision'] == 'float32' and protocol['resources'] == RESOURCES, 'Precision/resource budget changed')
    require(protocol['decoder_execution_device'] == 'cpu_from_committed_float32_heads', 'Decoder execution device changed')
    require(protocol['outputs']['perception_dir'] == OUTPUT_DIRECTORY
            and protocol['outputs']['perception_report'] == 'results/P3/instance_interface_v11.json', 'Output directory changed')
    require(protocol['metadata_fields_read'] == ['intrinsic_matrix', 'factor_depth']
            and protocol['deterministic_algorithms'] is True and protocol['allow_tf32'] is False,
            'Metadata or deterministic policy changed')
    require(set(protocol['inputs']) == {'rgb', 'depth', 'meta'}, 'Input allowlist changed')
    for kind, suffix in (('rgb', 'rgb/0000.png'), ('depth', 'depth/0000.png'), ('meta', 'meta/0000.mat')):
        item = protocol['inputs'][kind]
        expected = 'data/raw/graspnet/scenes/scene_0003/realsense/' + suffix
        require(item['path'] == expected and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Frozen input path/hash changed')
    checkpoint = protocol['checkpoint']
    require(checkpoint['sha256'] == CHECKPOINT_SHA and checkpoint['config_fingerprint'] == CONFIG_FP
            and checkpoint['data_fingerprint'] == DATA_FP and checkpoint['cursor'] == CURSOR, 'Checkpoint identity changed')
    require(checkpoint['trusted_local_payload'] is True and checkpoint['path'] == 'experiments/segmentation_coverage_v8/coverage_expanded/checkpoints/checkpoint-1792634787992100100-b20cbc58b2d4.pt',
            'Only the fixed final checkpoint is allowed')


def runtime_identity():
    require(torch.cuda.is_available(), 'Frozen inference GPU is unavailable')
    return {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'pillow': PIL.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}


def make_binding(protocol, protocol_path):
    validate_protocol(protocol)
    actual = runtime_identity()
    require(actual == protocol['runtime'], 'Runtime differs from frozen V8 runtime')
    verified_files = {}
    forbidden = {protocol['audit']['label_path'], protocol['audit']['reference_frame_path']}
    for key in ('input_hashes', 'source_hashes'):
        for name, checksum in protocol[key].items():
            require(name not in forbidden and '/label/' not in name and '/annotations/' not in name,
                    'Diagnostic-only files cannot enter the deployment input allowlist')
            require(file_sha256(project_path(name)) == checksum, 'Frozen common evidence changed: ' + name)
            verified_files[name] = checksum
    input_hashes = {}
    for key, item in protocol['inputs'].items():
        path = project_path(item['path'])
        require(file_sha256(path) == item['sha256'], 'Input digest changed: ' + key)
        input_hashes[key] = dict(item, size_bytes=path.stat().st_size)
    checkpoint = project_path(protocol['checkpoint']['path'])
    require(file_sha256(checkpoint) == CHECKPOINT_SHA, 'Checkpoint digest changed; refusing deserialization')
    metadata_path = project_path(protocol['checkpoint']['manifest_path'])
    require(metadata_path == checkpoint.with_suffix('.json') and file_sha256(metadata_path) == protocol['checkpoint']['manifest_sha256'],
            'Checkpoint manifest file/hash differs')
    metadata = read(metadata_path)
    require(metadata['sha256'] == CHECKPOINT_SHA and metadata['size_bytes'] == checkpoint.stat().st_size
            and metadata['file'] == checkpoint.name and metadata['global_step'] == 1600
            and metadata['config_fingerprint'] == CONFIG_FP and metadata['data_fingerprint'] == DATA_FP,
            'Checkpoint manifest differs')
    run_config_path = checkpoint.parent.parent / 'config.json'
    require(fingerprint(read(run_config_path)) == CONFIG_FP, 'Training config fingerprint differs')
    names = ('scripts/infer_instance_interface_v11.py', 'src/segmentation.py', 'src/checkpoints.py', 'src/graspnet_frames.py')
    binaries = {'python': Path(sys.executable), 'torch_C': Path(torch._C.__file__),
                'numpy_multiarray': Path(np.core._multiarray_umath.__file__)}
    return {'schema': 11, 'frame': FRAME, 'protocol_sha256': file_sha256(protocol_path),
        'sources': {name: file_sha256(ROOT / name) for name in names}, 'inputs': input_hashes, 'verified_common_files': verified_files,
        'checkpoint': dict(protocol['checkpoint'], manifest_sha256=file_sha256(metadata_path),
                           run_config_sha256=file_sha256(run_config_path)),
        'runtime_identity': actual, 'runtime_binaries': {name: {'path': str(path), 'sha256': file_sha256(path)} for name, path in binaries.items()},
        'preprocessing': PREPROCESSING, 'decoder': DECODER, 'precision': 'float32',
        'metadata_variables_read': ['intrinsic_matrix', 'factor_depth'],
        'ground_truth_labels_read': False, 'object_poses_or_ids_read': False,
        'training_performed': False, 'official_test_or_heldout_read': False}


def prepare_inputs(rgb_path, depth_path, meta_path, preprocessing=PREPROCESSING):
    """Replicate only the nonaugmented image branch of V8, never its GT loader."""
    size = tuple(preprocessing['size'])
    target = (size[1], size[0])
    with Image.open(rgb_path) as source:
        original = source.size
        require((original[1], original[0]) == tuple(preprocessing['original_size']), 'Original RGB shape differs')
        rgb = np.asarray(source.convert('RGB').resize(target, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    with Image.open(depth_path) as source:
        require(source.size == original, 'RGB and depth shapes differ')
        depth = np.asarray(source.resize(target, Image.Resampling.NEAREST), dtype=np.float32)
    meta = loadmat(str(meta_path), variable_names=['intrinsic_matrix', 'factor_depth'])
    require('intrinsic_matrix' in meta and 'factor_depth' in meta, 'Missing allowed calibration fields')
    factor_values = np.asarray(meta['factor_depth'])
    require(factor_values.size == 1, 'Depth factor must be scalar')
    factor = float(factor_values.reshape(-1)[0])
    intrinsic = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
    require(np.isfinite(factor) and factor > 0 and intrinsic.shape == (3, 3)
            and np.isfinite(intrinsic).all() and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0,
            'Invalid calibration values')
    depth = depth / factor
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, np.minimum(depth, preprocessing['depth_clip_m']) / preprocessing['depth_clip_m'], 0).astype(np.float32)
    rgb = (rgb - np.asarray(preprocessing['rgb_mean'], dtype=np.float32)) / np.asarray(preprocessing['rgb_std'], dtype=np.float32)
    tensor = np.ascontiguousarray(np.concatenate([rgb.transpose(2, 0, 1), depth[None], valid.astype(np.float32)[None]], axis=0)[None], dtype=np.float32)
    require(np.isfinite(tensor).all(), 'Nonfinite input tensor')
    return tensor, {'original_size': [original[1], original[0]], 'intrinsic_matrix': intrinsic.tolist(), 'factor_depth': factor}


def validate_heads(heads):
    require(set(heads) == {'foreground', 'centers', 'offsets'}, 'Unexpected head names')
    for name, channels in (('foreground', 1), ('centers', 1), ('offsets', 2)):
        array = heads[name]
        require(array.dtype == np.float32 and array.shape == (1, channels, 240, 320)
                and np.isfinite(array).all(), 'Invalid head array: ' + name)


def decode_heads(heads, decoder=DECODER):
    validate_heads(heads)
    low = decode_instances({key: torch.from_numpy(value.copy()) for key, value in heads.items()}, **decoder)
    require(low.dtype == np.int32 and low.shape == (240, 320) and low.min() >= 0 and low.max() <= 64, 'Invalid decoded instances')
    original = np.asarray(Image.fromarray(low).resize((1280, 720), Image.Resampling.NEAREST), dtype=np.int32)
    require(set(np.unique(original)) == set(np.unique(low)), 'Nearest lift lost instance IDs')
    return low, original


def serialized_array(value, archive=False):
    stream = io.BytesIO()
    if archive:
        np.savez(stream, **value)
    else:
        np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def payload_hash(payload):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def validate_members(directory):
    names = set(ARTIFACTS) | set(JSON_FILES) | {name + '.pending.json' for name in ARTIFACTS} | {'.checkpoint.lock'}
    for path in directory.iterdir():
        require(path.is_file(), 'Unexpected output subdirectory: ' + path.name)
        if path.name in names:
            continue
        require(any(re.fullmatch(re.escape(name) + r'\.tmp-[0-9a-f]{32}', path.name) for name in names),
                'Unknown artifact in isolated output directory: ' + path.name)


def read_artifact(directory, name, binding):
    path, receipt_path = directory / name, directory / (name + '.pending.json')
    if not receipt_path.exists():
        require(not path.exists(), 'Unbound output exists: ' + name)
        return None
    receipt = read(receipt_path)
    require(receipt['schema'] == 11 and receipt['file'] == name and receipt['binding_fingerprint'] == binding,
            'Artifact binding differs: ' + name)
    if not path.exists():
        return None
    require(path.stat().st_size == receipt['size_bytes'] and file_sha256(path) == receipt['sha256'],
            'Committed artifact is corrupt: ' + name)
    return receipt


def publish_artifact(directory, name, payload, binding, evidence=None, read_only=False):
    """Write digest intent before bytes, allowing crash-after-rename recovery without inference."""
    receipt = {'schema': 11, 'file': name, 'binding_fingerprint': binding,
               'sha256': payload_hash(payload), 'size_bytes': len(payload), 'evidence': evidence or {}}
    prior = read_artifact(directory, name, binding)
    if prior is not None:
        require(prior == receipt, 'Immutable artifact differs: ' + name)
        return prior
    require(not read_only, 'Required committed artifact missing: ' + name)
    intent_path = directory / (name + '.pending.json')
    if intent_path.exists():
        intent = read(intent_path)
        require(all(intent[key] == receipt[key] for key in ('schema', 'file', 'binding_fingerprint', 'sha256', 'size_bytes')),
                'Interrupted output intent changed: ' + name)
        receipt = intent
    else:
        atomic_json(intent_path, receipt)
    atomic_write(directory / name, lambda stream: stream.write(payload))
    return read_artifact(directory, name, binding)


def publish_json_once(path, value, read_only=False):
    if path.exists():
        require(read(path) == value, 'Immutable report differs: ' + path.name)
    else:
        require(not read_only, 'Required report missing: ' + path.name)
        atomic_json(path, value)


class ResourceGuard:
    """Sample CPU/RSS/time; hard abort is safe because only verified artifacts resume."""
    def __init__(self, directory, resources, output_paths=None):
        self.directory, self.resources = directory, resources
        self.output_paths = output_paths or [directory]
        self.start = time.monotonic()
        self.peak_rss = 0
        self.stop = threading.Event()
        self.problem = None

    def check(self):
        rss = process_rss_bytes()
        self.peak_rss = max(self.peak_rss, rss)
        require(rss <= self.resources['rss_soft_bytes'], 'RSS resource cap exceeded')
        require(time.monotonic() - self.start <= self.resources['invocation_soft_seconds'], 'Invocation time cap exceeded')
        require(shutil.disk_usage(ROOT).free >= self.resources['disk_reserve_bytes'], 'Disk reserve breached')
        files = set()
        for output in self.output_paths:
            if output.is_file():
                files.add(output)
            elif output.is_dir():
                files.update(path for path in output.rglob('*') if path.is_file())
        total = 0
        for path in files:
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                # Atomic publication can rename a temporary between scan and stat.
                # The next sample sees its final name; committed output validation
                # separately treats a missing immutable file as corruption.
                continue
        require(total <= self.resources['output_limit_bytes'], 'Combined round output size cap exceeded')

    def monitor(self):
        while not self.stop.wait(self.resources['sample_interval_seconds']):
            try:
                self.check()
            except Exception as error:
                print(json.dumps({'status': 'resource_limit_exceeded', 'reason': str(error)}), flush=True)
                os._exit(70)

    def __enter__(self):
        self.check()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1)


def process_rss_bytes():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        require(bool(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)),
                'Cannot read process memory counters')
        return int(counters.WorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)


def load_frozen_model(protocol):
    path = project_path(protocol['checkpoint']['path'])
    require(file_sha256(path) == CHECKPOINT_SHA, 'Checkpoint changed immediately before deserialization')
    # This exact digest is a trusted, locally produced complete training payload.
    # Its NumPy RNG state requires weights_only=False; never accept a remote payload here.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(payload['schema_version'] == 1 and payload['config_fingerprint'] == CONFIG_FP
            and payload['data_fingerprint'] == DATA_FP and payload['cursor'] == CURSOR,
            'Deserialized final checkpoint identity differs')
    require(fingerprint(payload['extra']['config']) == CONFIG_FP and payload['extra']['data_fingerprint'] == DATA_FP,
            'Embedded final checkpoint config/data differs')
    model = PilotInstanceNet(mode='rgbd')
    model.load_state_dict(payload['model'], strict=True)
    require(all(torch.isfinite(value).all().item() for value in model.state_dict().values()), 'Nonfinite model state')
    return model.eval()


def forward_once(inputs, protocol):
    gpu_limit = protocol['resources']['gpu_reserved_soft_bytes']
    torch.cuda.set_per_process_memory_fraction(min(1., gpu_limit / torch.cuda.get_device_properties(0).total_memory), device=0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_frozen_model(protocol).to('cuda').eval()
    with torch.inference_mode():
        outputs = model(torch.from_numpy(inputs).to('cuda'))
        heads = {key: value.detach().cpu().numpy().copy() for key, value in outputs.items()}
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    del outputs, model
    torch.cuda.empty_cache()
    validate_heads(heads)
    require(reserved <= gpu_limit and allocated <= protocol['resources']['gpu_allocated_soft_bytes'], 'GPU reserve/allocation cap exceeded')
    return heads, {'model_forward_calls': 1, 'weights_only': False, 'strict_load': True, 'checkpoint_sha256': CHECKPOINT_SHA,
        'checkpoint_cursor': CURSOR, 'inference_precision': 'float32', 'autocast': False,
        'peak_gpu_allocated_bytes': allocated, 'peak_gpu_reserved_bytes': reserved}


def prediction_summary(low, original):
    ids = np.unique(low[low > 0]).tolist()
    return {'predicted_instances': len(ids), 'instance_ids': ids,
        'low_foreground_pixels': int(np.count_nonzero(low)), 'original_foreground_pixels': int(np.count_nonzero(original)),
        'low_pixels_by_instance': {str(key): int(np.count_nonzero(low == key)) for key in ids},
        'original_pixels_by_instance': {str(key): int(np.count_nonzero(original == key)) for key in ids}}


def run(protocol_path=None, stop_after_heads=False, check_only=False):
    protocol_path = project_path(PROTOCOL) if protocol_path is None else Path(protocol_path)
    protocol = read(protocol_path)
    validate_protocol(protocol)
    directory = project_path(protocol['outputs']['perception_dir'])
    report_path = project_path(protocol['outputs']['perception_report'])
    if check_only:
        require((directory / 'completion.json').exists(), 'Check-only requires completed outputs')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(protocol['inference_seed'])
    np.random.seed(protocol['inference_seed'])
    output_paths = [project_path(name) for name in protocol['outputs'].values()] + [project_path(name) for name in (
        'results/P3/instance_interface_v11_resume.json', 'results/P3/grasp_instance_association_v11_resume.json')]
    with ResourceGuard(directory, protocol['resources'], output_paths) as guard:
        binding_record = make_binding(protocol, protocol_path)
        binding = fingerprint(binding_record)
        completed = (directory / 'completion.json').exists()
        readonly = check_only or completed
        if not readonly:
            directory.mkdir(parents=True, exist_ok=True)
        require(directory.is_dir(), 'Missing output directory')
        # Existing lock opened in r+b for a genuinely read-only terminal verification.
        lock = readonly_lock(directory) if readonly else _exclusive_lock(directory)
        with lock:
            validate_members(directory)
            if not (directory / 'binding.json').exists():
                allowed = {'.checkpoint.lock'}
                require(all(path.name in allowed or re.fullmatch(r'binding\.json\.tmp-[0-9a-f]{32}', path.name) for path in directory.iterdir()),
                        'Outputs exist without binding')
            publish_json_once(directory / 'binding.json', binding_record, readonly)
            inputs, calibration = prepare_inputs(**{key + '_path': project_path(value['path']) for key, value in protocol['inputs'].items()})
            input_receipt = publish_artifact(directory, 'input_tensor.npy', serialized_array(inputs), binding, read_only=readonly)
            head_receipt = read_artifact(directory, 'heads.npz', binding)
            new_forwards = 0
            if head_receipt is None:
                require(not readonly, 'Completed head cache is missing')
                heads, evidence = forward_once(inputs, protocol)
                new_forwards = 1
                head_receipt = publish_artifact(directory, 'heads.npz', serialized_array(heads, archive=True), binding, evidence=evidence)
            else:
                with np.load(directory / 'heads.npz', allow_pickle=False) as archive:
                    heads = {key: archive[key].copy() for key in archive.files}
                validate_heads(heads)
            evidence = head_receipt['evidence']
            require(evidence['model_forward_calls'] == 1 and evidence['strict_load'] is True
                    and evidence['checkpoint_sha256'] == CHECKPOINT_SHA and evidence['checkpoint_cursor'] == CURSOR
                    and evidence['inference_precision'] == 'float32' and evidence['autocast'] is False,
                    'Frozen forward evidence differs')
            head_commit = {'schema': 11, 'status': 'heads_committed', 'binding_fingerprint': binding,
                'frame': FRAME, 'calibration': calibration, 'input_tensor': input_receipt, 'heads': head_receipt,
                'ground_truth_labels_read': False, 'model_forward_calls_committed': 1,
                'recovery_note': 'Committed heads are reused. A power cut before durable head bytes may require redoing that one forward.'}
            publish_json_once(directory / 'heads_commit.json', head_commit, readonly)
            if stop_after_heads and not completed:
                guard.check()
                result = {'status': 'heads_committed', 'new_model_forward_calls': new_forwards,
                    'binding_fingerprint': binding, 'heads_commit_sha256': file_sha256(directory / 'heads_commit.json')}
                print(json.dumps(result), flush=True)
                return result
            low, original = decode_heads(heads, protocol['decoder'])
            receipts = [input_receipt, head_receipt]
            for name, value in (('instances_low.npy', low), ('instances_original.npy', original)):
                receipts.append(publish_artifact(directory, name, serialized_array(value), binding, read_only=readonly))
            outputs = {name: file_sha256(directory / name) for name in ('binding.json', 'heads_commit.json') + ARTIFACTS
                       + tuple(name + '.pending.json' for name in ARTIFACTS)}
            report = {'schema': 11, 'status': 'completed', 'binding_fingerprint': binding, 'frame': FRAME,
                'outputs': outputs, 'prediction_counts': prediction_summary(low, original),
                'input_tensor_shape': list(inputs.shape), 'head_shapes': {name: list(value.shape) for name, value in heads.items()},
                'decoder': DECODER, 'calibration': calibration, 'model_forward_calls_committed': 1,
                'ground_truth_labels_read': False, 'training_performed': False,
                'model_ap_or_grasp_success_claim': False, 'zero_prediction_is_valid_negative_result': True,
                'decision': 'prediction_interface_completed_not_effectiveness_acceptance',
                'limitations': ['One already-used training frame only', 'No segmentation accuracy or grasp AP measured',
                               'Resampling cannot restore lost spatial resolution', 'Local instance IDs are not object category IDs']}
            guard.check()
            publish_json_once(report_path, report, readonly)
            completion = dict(report, report_file=protocol['outputs']['perception_report'], report_sha256=file_sha256(report_path))
            guard.check()
            publish_json_once(directory / 'completion.json', completion, readonly)
            guard.check()
            validate_members(directory)
            print(json.dumps({'status': 'verified_completed_read_only' if readonly else 'completed',
                'new_model_forward_calls': new_forwards, 'prediction_counts': report['prediction_counts'],
                'peak_rss_bytes': guard.peak_rss, 'elapsed_seconds': time.monotonic() - guard.start}), flush=True)
            return report


@contextlib.contextmanager
def readonly_lock(directory):
    path = directory / '.checkpoint.lock'
    require(path.is_file(), 'Missing lock on completed cache')
    with path.open('r+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=PROTOCOL)
    parser.add_argument('--stop-after-heads', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    options = parser.parse_args()
    require(not (options.stop_after_heads and options.check_only), 'Modes are mutually exclusive')
    run(project_path(options.config), options.stop_after_heads, options.check_only)
