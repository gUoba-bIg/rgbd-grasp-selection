"""Durable fixed-checkpoint development scores; no reference-label decoding."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import numpy as np
from scripts import probe_dualstream_resources_v18 as durable
from scripts import evaluate_segmentation_baseline_v16 as parent_data

CONFIG = 'configs/dualstream_evaluation_v18.json'
ARMS = ('fixed', 'quality')
CONDITIONS = ('clean', 'noise', 'missing', 'joint')
SIZE = (480, 640)
QUALITY_SIZES = ((120, 160), (60, 80), (30, 40), (15, 20))
SEGMENTATION_KEYS = {'foreground_probability', 'center_probability', 'offsets'}
HEAD_KEYS = SEGMENTATION_KEYS | {'quality_'+str(i) for i in range(4)}
DECODER = {'foreground_threshold': .5, 'center_threshold': .35, 'min_pixels': 20, 'max_centers': 64}
REQUIRED_SOURCES = (
    'scripts/extract_dualstream_scores_v18.py', 'scripts/train_segmentation_dualstream_v18.py',
    'src/segmentation_dualstream_v18.py', 'src/segmentation_quality_v18.py', 'src/graspnet_frames.py',
    'scripts/extract_mask_scores_v17.py', 'scripts/evaluate_segmentation_baseline_v16.py',
    'scripts/probe_dualstream_resources_v18.py', 'scripts/multiview_common_v14r1.py')
require, safe = durable.require, durable.safe
storage = durable.storage
receipt, verify_bytes, load_commit = durable.receipt, durable.verify_bytes, durable.load_commit


def array_payload(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def read_arrays(file):
    with np.load(file, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verify_maps(root, protocol):
    for group in ('source_hashes', 'input_hashes'):
        require(protocol.get(group), 'Evaluation source and input freeze required')
        for name, checksum in protocol[group].items():
            require(isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum)
                and storage.digest(safe(root, name)) == checksum, 'Frozen source/input differs: '+name)


def verify_design(protocol):
    require(protocol['schema'] == 18 and protocol['status'] == 'frozen'
        and protocol['phase'] == 'development_dualstream_evaluation'
        and protocol['arms'] == list(ARMS) and protocol['conditions'] == list(CONDITIONS)
        and protocol['input_size'] == list(SIZE) and protocol['precision'] == 'float32_without_autocast'
        and protocol['gate_strength'] == 1 and protocol['expected_frames_per_arm_condition'] == 160
        and protocol['camera'] == 'realsense' and protocol['decoder'] == DECODER
        and protocol['peak_window'] == 9 and protocol['center_suppression_distance'] == 5,
        'Fixed dual-stream score policy differs')
    require(protocol['resources'] == {'cpu_threads': 4, 'invocation_soft_seconds': 1800,
        'rss_soft_bytes': 4*1024**3, 'gpu_allocated_soft_bytes': int(13.5*1024**3),
        'gpu_reserved_soft_bytes': int(13.5*1024**3), 'output_limit_bytes': 12*1024**3,
        'disk_reserve_bytes': 60*1024**3}, 'Evaluation resource budget differs')
    require(protocol['outputs']['scores_root'] == 'results/P4/dualstream_evaluation_v18/scores',
            'Score output root differs')


def validate_protocol(root, config, protocol, arm, condition):
    verify_design(protocol)
    require(arm in ARMS and condition in CONDITIONS, 'Unknown score arm or condition')
    require(protocol['interpreter'] == 'runtime/envs/vision/Scripts/python.exe'
        and safe(root, protocol['interpreter']) == Path(sys.executable).resolve(), 'Wrong vision interpreter')
    require(all(name in protocol['source_hashes'] for name in REQUIRED_SOURCES), 'Missing score dependency freeze')
    required_inputs = (protocol['parent_training_protocol'], protocol['parent_training_report'],
                       protocol['interpreter'], 'runtime/envs/vision/pyvenv.cfg')
    require(all(name in protocol['input_hashes'] for name in required_inputs), 'Missing score input freeze')
    verify_maps(root, protocol)
    base = protocol['base_python']
    require(Path(base['path']).is_absolute() and Path(base['path']).name.lower() == 'python.exe'
        and Path(sys._base_executable).resolve() == Path(base['path']).resolve()
        and isinstance(base['sha256'], str) and re.fullmatch('[0-9a-f]{64}', base['sha256'])
        and storage.digest(base['path']) == base['sha256'], 'Vision base interpreter differs')
    runtime = protocol['runtime']
    require(platform.python_version() == runtime['python'], 'Python version differs')
    for package in ('torch', 'torchvision', 'numpy', 'scipy', 'Pillow'):
        require(importlib.metadata.version(package) == runtime[package], 'Package version differs: '+package)
    output = safe(root, protocol['outputs']['scores_root'])
    roots = [safe(root, name) for name in protocol['budget_paths']]
    require(any(output == path or output.is_relative_to(path) for path in roots), 'Scores omitted from total output budget')
    require(storage.read(config) == protocol, 'Evaluation protocol changed while reading')


class Guard:
    """Cumulative output union, per-invocation soft time and allocator limits."""
    def __init__(self, root, protocol):
        self.root, self.protocol = Path(root), protocol
        self.started = time.perf_counter()
        self.gpu = False

    def check(self, extra_bytes=0):
        require(type(extra_bytes) is int and extra_bytes >= 0, 'Invalid publication reservation')
        files = set()
        for name in self.protocol['budget_paths']:
            path = safe(self.root, name)
            candidates = list(path.rglob('*')) if path.is_dir() else [path]
            if not path.is_dir():
                candidates += list(path.parent.glob(path.name+'.*.tmp'))
            for file in candidates:
                require(not file.is_symlink(), 'Link in evaluation budget')
                if file.is_file(): files.add(file.resolve())
        size = sum(file.stat().st_size for file in files)
        elapsed, rss = time.perf_counter()-self.started, storage.peak_memory_bytes()
        allocated = reserved = 0
        if self.gpu:
            import torch
            allocated, reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
        limits = self.protocol['resources']
        require(size+extra_bytes <= limits['output_limit_bytes'], 'Evaluation output budget exhausted')
        require(shutil.disk_usage(self.root).free-extra_bytes >= limits['disk_reserve_bytes'], 'Disk reserve exhausted')
        require(elapsed <= limits['invocation_soft_seconds'] and rss <= limits['rss_soft_bytes'], 'Evaluation time or RSS limit exceeded')
        require(allocated <= limits['gpu_allocated_soft_bytes'] and reserved <= limits['gpu_reserved_soft_bytes'], 'GPU allocator limit exceeded')
        return {'elapsed_seconds': elapsed, 'peak_rss_bytes': rss, 'peak_gpu_allocated_bytes': allocated,
                'peak_gpu_reserved_bytes': reserved, 'current_output_bytes': size,
                'scope': 'Boundary-sampled soft limits; an in-flight operation is not preempted'}


def commit(file, value, binding, guard, readonly=False):
    payload = storage.json_bytes(value)
    commit_bytes(file, payload, binding, guard, readonly)


def commit_bytes(file, payload, binding, guard, readonly=False):
    guard.check(0 if readonly or Path(file).exists() else len(payload)+4096)
    durable.commit_bytes(file, payload, binding, readonly)


def training_source(root, parent, parent_file, report, arm, manifest):
    directory = safe(root, parent['arms'][arm]['run_directory'])
    filenames = ('config.json', 'data_manifest.json', 'initialization.json', 'history.json',
                 'progress.json', 'result.json', 'resource_usage.json')
    values = {name: storage.read(directory/name) for name in filenames}
    config, data, result = (values[name] for name in ('config.json', 'data_manifest.json', 'result.json'))
    require(config['schema'] == 18 and config['arm'] == config['mode'] == arm
        and config['protocol'] == parent and config['protocol_sha256'] == storage.digest(parent_file), 'Parent arm config differs')
    require(data['schema'] == 18 and data['source_manifest'] == manifest
        and data['source_manifest_sha256'] == storage.digest(safe(root, parent['data_manifest']))
        and data['records'] == manifest['records'], 'Parent data identity differs')
    # Trainer fingerprints use compact JSON, unlike this script's receipt binding.
    config_hash, data_hash = parent_data.fingerprint(config), parent_data.fingerprint(data)
    require(result['schema'] == 18 and result['status'] == 'trained' and result['arm'] == arm
        and result['optimizer_steps'] == result['planned_steps'] == 800
        and result['cursor'] == {'epoch': 10, 'next_batch': 0, 'global_step': 800}
        and result['validation_evaluated'] is False and result['evaluation'] is None
        and result['automatic_extension'] is False and result['config_fingerprint'] == config_hash
        and result['data_fingerprint'] == data_hash, 'Parent arm is not final-800 training')
    for field, name in (('history_sha256', 'history.json'), ('initialization_sha256', 'initialization.json'),
                        ('resource_usage_sha256', 'resource_usage.json')):
        require(result[field] == storage.digest(directory/name), 'Parent training evidence differs: '+name)
    require(result['resources'] == values['resource_usage.json'], 'Parent resource record differs')
    pointer_file = directory/'checkpoints/latest.json'
    pointer = storage.read(pointer_file)
    require(Path(pointer['manifest']).name == pointer['manifest'], 'Unsafe parent checkpoint manifest')
    metadata_file = safe(directory/'checkpoints', pointer['manifest'])
    require(storage.digest(metadata_file) == pointer['manifest_sha256'], 'Parent checkpoint pointer differs')
    metadata = storage.read(metadata_file)
    require(Path(metadata['file']).name == metadata['file'], 'Unsafe parent checkpoint payload')
    checkpoint = safe(directory/'checkpoints', metadata['file'])
    require(metadata['global_step'] == 800 and metadata['config_fingerprint'] == config_hash
        and metadata['data_fingerprint'] == data_hash and checkpoint.stat().st_size == metadata['size_bytes']
        and storage.digest(checkpoint) == metadata['sha256'] == result['checkpoint_sha256']
        and checkpoint == safe(root, result['checkpoint_path']), 'Parent checkpoint source differs')
    declared = report['arms'][arm]
    require(declared == {'run_directory': directory.relative_to(root).as_posix(), 'result': result,
        'result_sha256': storage.digest(directory/'result.json'), 'config_sha256': storage.digest(directory/'config.json'),
        'history_sha256': storage.digest(directory/'history.json'),
        'initialization_sha256': storage.digest(directory/'initialization.json'), 'checkpoint_sha256': metadata['sha256']},
        'Parent report does not seal this final arm')
    files = [directory/name for name in filenames]+[pointer_file, metadata_file, checkpoint]
    return {'directory': directory, 'files': {file.relative_to(root).as_posix(): storage.digest(file) for file in files},
        'identity': {'checkpoint_sha256': metadata['sha256'], 'result_sha256': declared['result_sha256'],
            'config_fingerprint': config_hash, 'data_fingerprint': data_hash, 'optimizer_steps': 800}}


def read_parent(root, protocol):
    parent_file = safe(root, protocol['parent_training_protocol'])
    report_file = safe(root, protocol['parent_training_report'])
    for file, field in ((parent_file, 'parent_training_protocol_sha256'), (report_file, 'parent_training_report_sha256')):
        require(isinstance(protocol[field], str) and re.fullmatch('[0-9a-f]{64}', protocol[field])
            and storage.digest(file) == protocol[field], 'Parent training binding differs')
    parent, report = storage.read(parent_file), storage.read(report_file)
    require(parent['schema'] == 18 and parent['planned_steps'] == 800 and parent['input_size'] == list(SIZE)
        and set(parent['arms']) == set(ARMS) and parent['seed'] == 20261022, 'Wrong parent training protocol')
    require(report['schema'] == 18 and report['status'] == 'completed'
        and report['research_optimizer_steps_completed'] == 1600
        and report['protocol_sha256'] == protocol['parent_training_protocol_sha256']
        and report['validation_evaluated'] is False and report['automatic_extension'] is False
        and set(report['arms']) == set(ARMS), 'Paired training report is incomplete')
    verify_maps(root, parent)
    for field in ('runtime', 'validation_scenes', 'frames', 'camera', 'interpreter'):
        require(protocol[field] == parent[field], 'Parent/evaluation policy differs: '+field)
    manifest, partitions = parent_data.verified_manifest(root, parent)
    records = partitions['development']
    require(len(records) == 160 and len(parent['validation_scenes']) == 10, 'Require all 160 development frames')
    sources = {arm: training_source(root, parent, parent_file, report, arm, manifest) for arm in ARMS}
    return {'protocol': parent, 'protocol_file': parent_file, 'records': records, 'sources': sources}


def make_binding(root, config, protocol, context, arm, condition):
    body = {'schema': 18, 'phase': 'scores', 'arm': arm, 'condition': condition,
        'protocol_sha256': storage.digest(config), 'source_hashes': protocol['source_hashes'],
        'input_hashes': protocol['input_hashes'], 'parent_training_protocol_sha256': protocol['parent_training_protocol_sha256'],
        'parent_training_report_sha256': protocol['parent_training_report_sha256'],
        'parent_training_files': {name: checksum for source in context['sources'].values() for name, checksum in source['files'].items()},
        'final_checkpoints': {name: source['identity'] for name, source in context['sources'].items()},
        'runtime': protocol['runtime'], 'interpreter_sha256': storage.digest(safe(root, protocol['interpreter'])),
        'base_python': protocol['base_python'], 'input_size': list(SIZE), 'gate_strength': 1.0,
        'precision': 'float32_without_autocast', 'decoder': DECODER, 'peak_window': 9, 'center_suppression_distance': 5,
        'expected_frames': len(context['records']), 'score_definition': 'Retained selected center sigmoid float32; no score calibration'}
    return dict(body, fingerprint=storage.fingerprint(body))


def frame_identity(record):
    require(type(record['scene']) is int and re.fullmatch('[0-9]{4}', record['frame']), 'Invalid development frame identity')
    return {'partition': 'development', 'scene': record['scene'], 'frame': record['frame'], 'camera': 'realsense'}


def frame_directory(directory, record):
    identity = frame_identity(record)
    return Path(directory)/'frames'/('scene_%04d_%s' % (identity['scene'], identity['frame']))


def corruption_identity(index, record, condition):
    scene, frame = int(record['scene']), int(record['frame'])
    value = 'R18|20261022|0|{}|{}|{}'.format(scene, frame, condition)
    return {'condition': condition, 'seed': int(hashlib.sha256(value.encode('ascii')).hexdigest()[:8], 16),
            'index': index, 'epoch': 0, 'scene': scene, 'frame': frame}


def prepare_input(root, context, record, index, condition):
    """Read only RGB/depth/calibration; labels never enter preprocessing."""
    from PIL import Image
    from scipy.io import loadmat
    import torch
    from src.segmentation_quality_v18 import corrupt_depth
    directory = safe(root, context['protocol']['data_root'])
    with Image.open(safe(directory, record['files']['rgb'])) as image:
        original_size = image.size
        rgb = np.asarray(image.convert('RGB').resize((640, 480), Image.Resampling.BILINEAR), np.float32)/255.0
    with Image.open(safe(directory, record['files']['depth'])) as image:
        require(image.size == original_size, 'Source RGB/depth geometry differs')
        depth = np.asarray(image.resize((640, 480), Image.Resampling.NEAREST), np.float32)
    metadata = loadmat(safe(directory, record['files']['meta']), variable_names=['factor_depth'])
    factor = float(np.asarray(metadata['factor_depth']).reshape(-1)[0])
    require(math.isfinite(factor) and factor > 0, 'Invalid depth factor')
    depth = depth/factor
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.where(valid, np.minimum(depth, 2.0)/2.0, 0).astype(np.float32)
    rgb = (rgb-np.asarray([.485, .456, .406], np.float32))/np.asarray([.229, .224, .225], np.float32)
    clean = np.ascontiguousarray(np.concatenate((rgb.transpose(2, 0, 1), normalized[None], valid.astype(np.float32)[None])))
    inputs, qtarget, corruption = corrupt_depth(torch.from_numpy(clean), index, 0, record=record, condition=condition)
    values = np.ascontiguousarray(inputs.numpy(), dtype=np.float32)
    require(values.shape == (5,)+SIZE and np.isfinite(values).all(), 'Evaluation input differs')
    info = {'schema': 18, 'identity': frame_identity(record), 'input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
        'input_shape': list(values.shape), 'input_dtype': 'float32', 'original_size_hw': list(reversed(original_size)),
        'source_sha256': record['sha256'], 'corruption': corruption,
        'quality_target_sha256': hashlib.sha256(qtarget.numpy().tobytes()).hexdigest(),
        'quality_target_is_model_input': False, 'reference_labels_decoded': False}
    return values, info


def validate_input_info(info, record, index, condition):
    require(info['schema'] == 18 and info['identity'] == frame_identity(record)
        and info['input_shape'] == [5, 480, 640] and info['input_dtype'] == 'float32'
        and info['source_sha256'] == record['sha256'] and info['quality_target_is_model_input'] is False
        and info['reference_labels_decoded'] is False, 'Saved feature identity differs')
    for key in ('input_sha256', 'quality_target_sha256'):
        require(isinstance(info[key], str) and re.fullmatch('[0-9a-f]{64}', info[key]), 'Missing feature/target digest')
    require(all(info['corruption'][key] == value for key, value in corruption_identity(index, record, condition).items()),
            'Corruption seed or condition differs')


def load_model(root, context, arm, protocol, guard):
    import torch
    from scripts import train_segmentation_dualstream_v18 as trainer
    require(torch.cuda.is_available(), 'CUDA required for a new model forward')
    require(trainer.runtime_identity() == protocol['runtime'], 'Actual inference runtime differs')
    trainer.reset_random_streams(context['protocol']['seed'], 'cuda')
    model, config, manifest, result, evidence = trainer.load_final_model(context['sources'][arm]['directory'], context['protocol'])
    require(config['arm'] == arm and result['optimizer_steps'] == 800
        and evidence['checkpoint_sha256'] == context['sources'][arm]['identity']['checkpoint_sha256'], 'Loaded final model differs')
    limit = protocol['resources']['gpu_reserved_soft_bytes']
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit/torch.cuda.get_device_properties(0).total_memory), 0)
    torch.set_num_threads(4)
    guard.gpu = True
    model = model.to('cuda').eval()
    guard.check()
    return model


def prepare_device_input(inputs):
    import torch
    return torch.from_numpy(inputs)[None].to('cuda')


def forward_once(model, inputs):
    import torch
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        outputs = model(inputs, gate_strength=1.0)
        heads = {'foreground_probability': outputs['foreground'][0, 0].sigmoid().cpu().numpy(),
                 'center_probability': outputs['centers'][0, 0].sigmoid().cpu().numpy(),
                 'offsets': outputs['offsets'][0].cpu().numpy()}
        require(len(outputs['quality_maps']) == 4, 'Model quality scale count differs')
        heads.update({'quality_'+str(i): value[0, 0].cpu().numpy() for i, value in enumerate(outputs['quality_maps'])})
    return heads


def check_heads(heads):
    require(set(heads) == HEAD_KEYS, 'Unexpected raw-head keys')
    for name in SEGMENTATION_KEYS:
        value = heads[name]
        require(value.shape == ((2,)+SIZE if name == 'offsets' else SIZE)
            and value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid raw segmentation head')
        if name != 'offsets': require(np.all((value >= 0) & (value <= 1)), 'Probability outside [0,1]')
    for index, shape in enumerate(QUALITY_SIZES):
        value = heads['quality_'+str(index)]
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
            and np.all((value >= 0) & (value <= 1)), 'Invalid raw quality map')


def validate_pending_receipt(file, binding):
    value = storage.read(file)
    require(set(value) == {'schema', 'binding_fingerprint', 'sha256', 'bytes'} and value['schema'] == 18
        and value['binding_fingerprint'] == binding['fingerprint'] and type(value['bytes']) is int and value['bytes'] > 0
        and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Malformed interrupted receipt')


def attempts(frame, binding):
    folder = Path(frame)/'attempts'
    result = []
    for index, directory in enumerate(sorted(folder.iterdir()) if folder.exists() else []):
        require(directory.name == '%06d' % index and directory.is_dir() and not directory.is_symlink(), 'Noncontiguous score attempt ledger')
        allowed = {'intent.json', 'outcome.json'}
        for file in directory.iterdir():
            name = re.sub(r'\.[0-9a-f]{32}\.tmp$', '', file.name)
            require(file.is_file() and not file.is_symlink()
                and (name in allowed or name.removesuffix('.sha256.json') in allowed), 'Unexpected attempt file')
        intent_file, outcome_file = directory/'intent.json', directory/'outcome.json'
        intent = load_commit(intent_file, binding) if intent_file.exists() else None
        if intent is None:
            require(not outcome_file.exists() and not receipt(outcome_file).exists(), 'Outcome lacks committed intent')
            if receipt(intent_file).exists(): validate_pending_receipt(receipt(intent_file), binding)
        else:
            require(intent['schema'] == 18 and intent['phase'] == 'model_forward'
                and intent['arm'] == binding['arm'] and intent['condition'] == binding['condition']
                and intent['binding_fingerprint'] == binding['fingerprint'] and intent['attempt_index'] == index
                and frame_directory(Path(frame).parent.parent, intent['identity']) == Path(frame), 'Attempt belongs to another frame')
        outcome = load_commit(outcome_file, binding) if outcome_file.exists() else None
        if outcome is not None:
            require(all(outcome.get(key) == value for key, value in intent.items())
                and outcome['status'] in ('model_returned', 'model_exception'), 'Forward outcome differs')
        elif receipt(outcome_file).exists(): validate_pending_receipt(receipt(outcome_file), binding)
        result.append({'attempt_index': index, 'intent': intent, 'outcome': outcome})
    return result


def attempt_summary(frame, binding):
    rows = attempts(frame, binding)
    returned = sum(row['outcome'] is not None and row['outcome']['status'] == 'model_returned' for row in rows)
    errors = sum(row['outcome'] is not None and row['outcome']['status'] == 'model_exception' for row in rows)
    unknown = len(rows)-returned-errors
    return {'attempts': len(rows), 'model_calls_returned': returned, 'model_exceptions_observed': errors,
        'unknown_outcome_attempts': unknown, 'known_model_call_lower_bound': returned,
        'actual_model_calls_exact': returned if unknown+errors == 0 else None}


def new_heads(root, protocol, context, source, index, frame, binding, guard, model, work):
    inputs, info = prepare_input(root, context, source, index, binding['condition'])
    validate_input_info(info, source, index, binding['condition'])
    commit(frame/'input.json', info, binding, guard)
    device_input = prepare_device_input(inputs)
    attempt_index = len(attempts(frame, binding))
    attempt = frame/'attempts'/('%06d' % attempt_index)
    intent = {'schema': 18, 'phase': 'model_forward', 'arm': binding['arm'], 'condition': binding['condition'],
        'identity': frame_identity(source), 'attempt_index': attempt_index, 'binding_fingerprint': binding['fingerprint'],
        'input_sha256': info['input_sha256'], 'input_record_sha256': storage.digest(frame/'input.json'),
        'checkpoint_sha256': context['sources'][binding['arm']]['identity']['checkpoint_sha256']}
    commit(attempt/'intent.json', intent, binding, guard)
    guard.check(8*1024**2)
    work['new_model_forward_calls'] += 1
    try:
        heads = forward_once(model, device_input)
    except Exception as error:
        commit(attempt/'outcome.json', dict(intent, status='model_exception', error_type=type(error).__name__,
            error=str(error), exact_model_call_count_known=False), binding, guard)
        raise
    payload = array_payload(**heads)
    commit(attempt/'outcome.json', dict(intent, status='model_returned', heads_sha256=hashlib.sha256(payload).hexdigest(),
        heads_bytes=len(payload)), binding, guard)
    # Commit raw evidence before numeric checks or decoding; an existing payload is never overwritten.
    commit_bytes(frame/'heads.npz', payload, binding, guard)


def expected_frame(source, index, frame, binding):
    info = load_commit(frame/'input.json', binding)
    validate_input_info(info, source, index, binding['condition'])
    verify_bytes(frame/'heads.npz', binding)
    heads = read_arrays(frame/'heads.npz'); check_heads(heads)
    raw_sha = storage.digest(frame/'heads.npz')
    require(any(row['outcome'] is not None and row['outcome']['status'] == 'model_returned'
        and row['outcome']['heads_sha256'] == raw_sha and row['outcome']['heads_bytes'] == (frame/'heads.npz').stat().st_size
        and row['intent']['input_sha256'] == info['input_sha256']
        and row['intent']['input_record_sha256'] == storage.digest(frame/'input.json')
        and row['intent']['checkpoint_sha256'] == binding['final_checkpoints'][binding['arm']]['checkpoint_sha256']
        for row in attempts(frame, binding)), 'Raw heads lack matching returned-forward evidence')
    from scripts.extract_mask_scores_v17 import decode_with_scores
    instances, foreground, scores, centers = decode_with_scores({name: heads[name] for name in SEGMENTATION_KEYS}, **DECODER)
    payload = array_payload(instances=instances, foreground=foreground)
    record = {'schema': 18, 'arm': binding['arm'], 'condition': binding['condition'], 'identity': frame_identity(source),
        'canonical_index': index, 'binding_fingerprint': binding['fingerprint'], 'input_sha256': info['input_sha256'],
        'input_record_sha256': storage.digest(frame/'input.json'), 'heads_sha256': raw_sha,
        'prediction_sha256': hashlib.sha256(payload).hexdigest(),
        'checkpoint_sha256': binding['final_checkpoints'][binding['arm']]['checkpoint_sha256'],
        'corruption': info['corruption'], 'instances': scores, 'selected_centers': centers,
        'predicted_instance_count': len(scores), 'quality_means': [float(heads['quality_'+str(i)].mean(dtype=np.float64)) for i in range(4)],
        'quality_target_is_model_input': False, 'reference_labels_decoded': False,
        'scores_are_calibrated_probabilities': False, 'new_training_updates': 0,
        'forward_attempt_accounting': attempt_summary(frame, binding)}
    return payload, record


def recover_pending_heads(frame, binding, guard):
    """A complete atomic temporary with its prior receipt can be adopted on CPU."""
    file = frame/'heads.npz'
    if file.exists() or not receipt(file).exists(): return False
    validate_pending_receipt(receipt(file), binding)
    wanted = storage.read(receipt(file))
    for temporary in sorted(frame.glob('heads.npz.*.tmp')):
        require(re.fullmatch(r'heads\.npz\.[0-9a-f]{32}\.tmp', temporary.name), 'Unknown heads temporary')
        if temporary.stat().st_size == wanted['bytes'] and storage.digest(temporary) == wanted['sha256']:
            commit_bytes(file, temporary.read_bytes(), binding, guard)
            return True
    return False


def validate_layout(directory, records, binding):
    allowed_files = {directory/name for name in ('binding.json', 'report.json', 'completion.json')}
    allowed_dirs = {directory/'frames'}
    gap = False
    for source in records:
        frame = frame_directory(directory, source)
        require(not gap or not frame.exists(), 'Frame artifacts are not a contiguous prefix')
        gap = gap or not (frame/'record.json').exists()
        allowed_dirs.update((frame, frame/'attempts'))
        allowed_files.update(frame/name for name in ('input.json', 'heads.npz', 'prediction.npz', 'record.json'))
        for row in attempts(frame, binding):
            attempt = frame/'attempts'/('%06d' % row['attempt_index'])
            allowed_dirs.add(attempt)
            allowed_files.update((attempt/'intent.json', attempt/'outcome.json'))
    allowed_files |= {receipt(file) for file in list(allowed_files)}
    for file in directory.rglob('*') if directory.exists() else []:
        require(not file.is_symlink(), 'Link in score output tree')
        if file.is_dir():
            require(file in allowed_dirs, 'Unexpected score directory')
        else:
            base = file.with_name(re.sub(r'\.[0-9a-f]{32}\.tmp$', '', file.name))
            require(base in allowed_files, 'Unexpected score artifact')


def make_report(binding, rows):
    accounting = {key: sum(row['forward_attempt_accounting'][key] for row in rows) for key in
        ('attempts', 'model_calls_returned', 'model_exceptions_observed', 'unknown_outcome_attempts', 'known_model_call_lower_bound')}
    accounting['actual_model_calls_exact'] = (accounting['model_calls_returned'] if
        accounting['model_exceptions_observed']+accounting['unknown_outcome_attempts'] == 0 else None)
    return {'schema': 18, 'status': 'completed_development_dualstream_scores', 'arm': binding['arm'], 'condition': binding['condition'],
        'binding_fingerprint': binding['fingerprint'], 'expected_frames': binding['expected_frames'], 'frame_count': len(rows),
        'rows': rows, 'forward_attempt_accounting': accounting, 'nominal_model_forward_calls': binding['expected_frames'],
        'ground_truth_metric_evaluation_performed': False, 'new_training_updates': 0, 'scores_are_calibrated_probabilities': False}


def completion_record(directory, binding):
    files = durable.snapshot(directory)
    return {'schema': 18, 'status': 'completed', 'arm': binding['arm'], 'condition': binding['condition'],
        'binding_fingerprint': binding['fingerprint'], 'expected_frames': binding['expected_frames'],
        'outputs': {name: value['sha256'] for name, value in files.items()
            if not re.fullmatch(r'completion\.json(?:\.sha256\.json)?(?:\.[0-9a-f]{32}\.tmp)?', name)},
        'report_sha256': storage.digest(directory/'report.json')}


def verify_completed(directory, binding, records):
    """NumPy/PIL/stdlib seal audit; no neural import, peak decoder or labels.

    The score extractor separately reconstructs the complete decoder on its
    vision-runtime read-only path. This audit verifies immutable evidence and
    exact selected-center score mapping for the independent metric runtime.
    """
    directory = Path(directory)
    before = durable.snapshot(directory)
    require((directory/'completion.json').exists(), 'Score cohort has not been sealed')
    validate_layout(directory, records, binding)
    require(load_commit(directory/'binding.json', binding) == binding, 'Score binding differs')
    require(load_commit(directory/'completion.json', binding) == completion_record(directory, binding), 'Score completion seal differs')
    rows = []
    for index, source in enumerate(records):
        frame = frame_directory(directory, source)
        info = load_commit(frame/'input.json', binding)
        validate_input_info(info, source, index, binding['condition'])
        verify_bytes(frame/'heads.npz', binding); heads = read_arrays(frame/'heads.npz'); check_heads(heads)
        verify_bytes(frame/'prediction.npz', binding); prediction = read_arrays(frame/'prediction.npz')
        require(set(prediction) == {'instances', 'foreground'}, 'Prediction cache keys differ')
        instances, foreground = prediction['instances'], prediction['foreground']
        require(instances.shape == foreground.shape == SIZE and instances.dtype == np.int32
            and foreground.dtype == np.bool_ and instances.min() >= 0
            and np.array_equal(foreground, heads['foreground_probability'] >= .5)
            and not np.any((instances > 0) & ~foreground), 'Prediction cache geometry/support differs')
        row = load_commit(frame/'record.json', binding)
        require(row['schema'] == 18 and row['arm'] == binding['arm'] and row['condition'] == binding['condition']
            and row['canonical_index'] == index and row['identity'] == frame_identity(source)
            and row['binding_fingerprint'] == binding['fingerprint']
            and row['input_sha256'] == info['input_sha256'] and row['corruption'] == info['corruption']
            and row['input_record_sha256'] == storage.digest(frame/'input.json')
            and row['heads_sha256'] == storage.digest(frame/'heads.npz')
            and row['prediction_sha256'] == storage.digest(frame/'prediction.npz')
            and row['checkpoint_sha256'] == binding['final_checkpoints'][binding['arm']]['checkpoint_sha256']
            and row['quality_target_is_model_input'] is False and row['reference_labels_decoded'] is False
            and row['scores_are_calibrated_probabilities'] is False and row['new_training_updates'] == 0,
            'Score frame provenance differs')
        require(row['forward_attempt_accounting'] == attempt_summary(frame, binding), 'Forward ledger differs')
        require(any(item['outcome'] is not None and item['outcome']['status'] == 'model_returned'
            and item['outcome']['heads_sha256'] == row['heads_sha256']
            and item['intent']['input_record_sha256'] == row['input_record_sha256']
            and item['intent']['input_sha256'] == row['input_sha256']
            and item['intent']['checkpoint_sha256'] == row['checkpoint_sha256']
            for item in attempts(frame, binding)), 'Missing returned-head provenance')
        centers = row['selected_centers']
        require(len(centers) <= 64, 'Too many selected centers')
        for selected, center in enumerate(centers):
            require(set(center) == {'selected_center_index', 'row', 'col', 'score'}
                and center['selected_center_index'] == selected and type(center['row']) is int and type(center['col']) is int
                and 0 <= center['row'] < 480 and 0 <= center['col'] < 640
                and center['score'] == float(heads['center_probability'][center['row'], center['col']]),
                'Selected-center score is not the exact retained float32 value')
        mapping = row['instances']
        require(row['predicted_instance_count'] == len(mapping)
            and np.array_equal(np.unique(instances[instances > 0]), np.arange(1, len(mapping)+1)), 'Instance IDs differ')
        for instance_id, item in enumerate(mapping, 1):
            selected = item['selected_center_index']
            require(type(selected) is int and 0 <= selected < len(centers)
                and item == dict(instance_id=instance_id, **centers[selected])
                and int(np.count_nonzero(instances == instance_id)) >= 20, 'Retained center-to-instance mapping differs')
        require(row['quality_means'] == [float(heads['quality_'+str(i)].mean(dtype=np.float64)) for i in range(4)], 'Quality means differ')
        rows.append(row)
    report = load_commit(directory/'report.json', binding)
    require(report == make_report(binding, rows), 'Score report differs')
    require(before == durable.snapshot(directory), 'Score seal audit changed SHA, size or mtime')
    return report


def verify_scores_seals(root=ROOT, config=CONFIG, protocol=None, context=None):
    """Verify all eight sealed 160-frame cohorts before any reference decoding."""
    root = Path(root).resolve(); config = safe(root, config)
    protocol = storage.read(config) if protocol is None else protocol
    require(storage.read(config) == protocol, 'Metric reader received another evaluation protocol')
    verify_design(protocol); verify_maps(root, protocol)
    require(all(name in protocol['source_hashes'] for name in REQUIRED_SOURCES), 'Missing score dependency freeze')
    base = safe(root, protocol['outputs']['scores_root'])
    require(all((base/condition/arm/'completion.json').exists() for condition in CONDITIONS for arm in ARMS),
            'All 1280 predictions must be sealed before metric input can be opened')
    context = read_parent(root, protocol) if context is None else context
    require(len(context['records']) == 160, 'Score seal cohort does not contain 160 frames')
    bindings, reports, seals = {}, {}, {}
    for condition in CONDITIONS:
        bindings[condition], reports[condition], seals[condition] = {}, {}, {}
        for arm in ARMS:
            directory = base/condition/arm
            binding = make_binding(root, config, protocol, context, arm, condition)
            reports[condition][arm] = verify_completed(directory, binding, context['records'])
            bindings[condition][arm] = binding
            seals[condition][arm] = {key+'_sha256': storage.digest(directory/(key+'.json'))
                                    for key in ('binding', 'report', 'completion')}
    return {'context': context, 'bindings': bindings, 'reports': reports, 'seals': seals}


def execute(root, config, protocol, context, binding, max_new_frames=None, check_only=False, guard=None):
    directory = safe(root, protocol['outputs']['scores_root'])/binding['condition']/binding['arm']
    guard = guard or Guard(root, protocol)
    guard.check()
    complete = (directory/'completion.json').exists()
    readonly = complete or check_only
    require(not check_only or complete, 'Read-only verification requires a completed score cohort')
    validate_layout(directory, context['records'], binding)
    before = durable.snapshot(directory) if readonly else None
    commit(directory/'binding.json', binding, binding, guard, readonly)
    work = {'schema': 18, 'arm': binding['arm'], 'condition': binding['condition'], 'expected_frames': len(context['records']),
        'new_frames': 0, 'reused_frames': 0, 'committed_frames': 0, 'new_model_forward_calls': 0, 'new_training_updates': 0}
    rows, model = [], None
    for index, source in enumerate(context['records']):
        frame = frame_directory(directory, source)
        existed = (frame/'record.json').exists()
        if not existed:
            require(not readonly, 'Completed score cohort has missing records')
            if max_new_frames is not None and work['new_frames'] >= max_new_frames: break
            recover_pending_heads(frame, binding, guard)
        if not (frame/'heads.npz').exists():
            require(not existed and not readonly, 'Committed score frame is missing raw heads')
            with storage.lock(root/'runtime/grasp_foreground_v12/generation_gpu_lock'):
                if model is None: model = load_model(root, context, binding['arm'], protocol, guard)
                new_heads(root, protocol, context, source, index, frame, binding, guard, model, work)
        payload, record = expected_frame(source, index, frame, binding)
        commit_bytes(frame/'prediction.npz', payload, binding, guard, readonly or existed)
        commit(frame/'record.json', record, binding, guard, readonly or existed)
        rows.append(record)
        work['reused_frames' if existed else 'new_frames'] += 1
        guard.check()
    work['committed_frames'] = len(rows)
    if len(rows) == len(context['records']):
        commit(directory/'report.json', make_report(binding, rows), binding, guard, readonly)
        commit(directory/'completion.json', completion_record(directory, binding), binding, guard, readonly)
        if readonly: require(before == durable.snapshot(directory), 'Read-only verification changed SHA, bytes or mtime')
        return dict(work, status='verified_completed_read_only' if readonly else 'completed',
            report_sha256=storage.digest(directory/'report.json'), resources=guard.check())
    return dict(work, status='partial', resources=guard.check())


def run(root=ROOT, config=CONFIG, arm=None, condition=None, max_new_frames=None, check_only=False):
    root = Path(root).resolve(); config = safe(root, config)
    require(max_new_frames is None or type(max_new_frames) is int and 0 <= max_new_frames <= 160, 'Invalid new-frame limit')
    require(not check_only or max_new_frames is None, 'Read-only mode cannot request new work')
    protocol = storage.read(config)
    guard = Guard(root, protocol)
    guard.check()
    validate_protocol(root, config, protocol, arm, condition)
    context = read_parent(root, protocol)
    binding = make_binding(root, config, protocol, context, arm, condition)
    directory = safe(root, protocol['outputs']['scores_root'])/condition/arm
    if check_only or (directory/'completion.json').exists():
        return execute(root, config, protocol, context, binding, max_new_frames, check_only, guard)
    with storage.lock(root/'runtime/locks/dualstream_scores_v18'/condition/arm):
        return execute(root, config, protocol, context, binding, max_new_frames, check_only, guard)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--max-new-frames', type=int)
    parser.add_argument('--check-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    try:
        args = arguments()
        print(json.dumps(run(config=args.config, arm=args.arm, condition=args.condition,
            max_new_frames=args.max_new_frames, check_only=args.check_only), sort_keys=True, allow_nan=False), flush=True)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print('Dual-stream score extraction stopped: '+str(error), file=sys.stderr, flush=True)
        sys.exit(2)
