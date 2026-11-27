"""Paired dual-stream research trainer; explicit frozen quality policy and durable updates."""
import argparse
import ctypes
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import PIL
import scipy
import torch
import torchvision
from scripts import train_segmentation_coverage_v8 as coverage
from scripts import train_segmentation_expanded_v3 as expanded
from scripts.train_segmentation_pilot import prepare_run_files, restore_or_initialize_checkpoint
from src.checkpoints import (CheckpointStore, _exclusive_lock, _capture_rng, _restore_rng,
                             atomic_json, file_sha256, fingerprint)
from src.segmentation_dualstream_v18 import make_paired_models
from src.segmentation_quality_v18 import (gate_strength_for_update, quality_loss, QualityGraspNetFrames, corruption_condition, corruption_metadata)
from scripts import train_segmentation_baseline_v16 as baseline
from scripts import probe_dualstream_resources_v18 as resource_probe
from src.segmentation_v2 import segmentation_loss

ARMS = ('fixed', 'quality')
TRAIN_SCENES = sorted(coverage.OLD_TRAIN + coverage.NEW_TRAIN)
VALIDATION_SCENES = [2, 6, 7, 11, 13, 15, 18, 22, 33, 39]
CONFIG = 'configs/segmentation_dualstream_v18.json'

def project_path(name):
    path = Path(name)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    require(path != ROOT and path.is_relative_to(ROOT), 'Path escaped project')
    return path
model_tensor_fingerprint = coverage.model_tensor_fingerprint
def rng_fingerprint():
    """CPU initialization identity; full device RNG is held by CheckpointStore."""
    numpy_state = np.random.get_state()
    return fingerprint({'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'torch_cpu': model_tensor_fingerprint({'state': torch.get_rng_state()})})


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/segmentation_dualstream_v18.json')
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--crash-after-step', type=int)
    parser.add_argument('--checkpoint-interval', type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    return parser.parse_args(argv)


def source_hashes(arm=None):
    names = set(coverage.source_hashes('coverage_expanded')) | {
        'scripts/train_segmentation_dualstream_v18.py', 'src/segmentation.py', 'src/segmentation_v2.py',
        'src/graspnet_frames.py', 'src/checkpoints.py', 'src/segmentation_dualstream_v18.py', 'src/segmentation_quality_v18.py',
        'scripts/train_segmentation_baseline_v16.py', 'scripts/probe_dualstream_resources_v18.py',
        'scripts/run_dualstream_resources_v18.py'}
    return {name: file_sha256(project_path(name)) for name in sorted(names)}


def verify_frozen_sources(protocol):
    require(protocol.get('status') == 'frozen' and bool(protocol['source_hashes'])
            and bool(protocol['input_hashes']), 'Unfrozen dual-stream sources/inputs')
    actual = source_hashes()
    require(all(protocol['source_hashes'].get(name) == sha for name, sha in actual.items()), 'Training helper source binding differs')
    for group in ('source_hashes', 'input_hashes'):
        for name, sha in protocol[group].items():
            require(file_sha256(project_path(name)) == sha, 'Frozen source/input changed: ' + name)


def runtime_identity():
    return {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'Pillow': PIL.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def ensure_run_target(directory, arm):
    directory = Path(directory).resolve()
    research = ROOT / 'experiments/segmentation_dualstream_v18' / arm
    verification = ROOT / 'runtime/verification/dualstream_training_v18' / arm
    engineering = directory.is_relative_to(verification) and directory != verification
    require(directory == research or engineering, 'Run directory is not an allowed research or v18 verification target')
    require(arm in ARMS, 'Run directory must identify its arm')
    for ancestor in directory.parents:
        if ancestor == ROOT:
            break
        require(not (ancestor / 'config.json').exists(), 'Run cannot be nested inside another experiment')
    if (directory / 'config.json').exists():
        old = read_json(directory / 'config.json')
        require(old.get('schema') == 18 and old.get('arm') == arm, 'Existing directory belongs to another experiment')


def verify_parent_resources(protocol):
    expected = {
        'resource_probe_protocol': 'configs/dualstream_resources_v18.json',
        'resource_probe_report': 'results/P4/dualstream_resources_v18.json',
        'resource_probe_resume_report': 'results/P4/dualstream_resources_v18_resume.json'}
    require(all(protocol.get(key) == value for key, value in expected.items()), 'Missing frozen resource parent')
    paths = {key: project_path(value) for key, value in expected.items()}
    for key, path in paths.items():
        require(protocol['input_hashes'].get(expected[key]) == protocol.get(key+'_sha256') == file_sha256(path),
                'Resource parent SHA differs: '+key)
    parent = read_json(paths['resource_probe_protocol'])
    binding = resource_probe.validate_protocol(ROOT, paths['resource_probe_protocol'], parent)
    report = resource_probe.load_commit(paths['resource_probe_report'], binding)
    resume = read_json(paths['resource_probe_resume_report'])
    require(report['status'] == 'passed_synthetic_gpu_resource_probe'
            and report['binding_fingerprint'] == binding['fingerprint']
            and report['optimizer_updates'] == report['real_frames'] == 0,
            'Resource probe has not passed')
    require(resume['status'] == 'passed' and resume['binding']['protocol_sha256'] == file_sha256(paths['resource_probe_protocol'])
            and resume['binding']['probe_binding_fingerprint'] == binding['fingerprint']
            and resume['report_sha256'] == file_sha256(paths['resource_probe_report'])
            and resume['byte_and_mtime_identical'] is True, 'Resource restart has not passed')
    from scripts.run_dualstream_resources_v18 import completed_snapshot
    require(completed_snapshot(ROOT, parent) == resume['completed_snapshot'], 'Resource parent outputs changed')
    require(protocol['runtime'] == parent['runtime'] and protocol['base_python'] == parent['base_python']
            and protocol['interpreter'] == parent['interpreter'], 'Training runtime differs from resource probe')
    require(Path(sys.executable).resolve() == project_path(protocol['interpreter'])
            and Path(sys._base_executable).resolve() == Path(protocol['base_python']['path']).resolve(),
            'Wrong training interpreter or base interpreter')


def validate_protocol(protocol, args):
    require(protocol.get('status') == 'frozen', 'Unfrozen dual-stream research protocol')
    require(args.device in ('cpu', 'cuda'), 'Unknown training device')
    expected = {'schema': 18, 'seed': 20261022, 'input_size': [480, 640], 'fpn_channels': 128,
        'batch_size': 1, 'accumulation': 4, 'planned_steps': 800, 'equivalent_epochs': 10,
        'total_research_update_budget': 1600, 'segment_boundaries': [100, 400, 800], 'augment': True,
        'learning_rate': .001, 'weight_decay': .0001, 'warmup_steps': 40, 'min_learning_rate': .00001,
        'amp': True, 'amp_initial_scale': 128., 'amp_max_overflow_retries': 16,
        'gradient_clip_norm': 1., 'gradient_clip_policy': 'separate_segmentation_and_quality_heads_after_unscale',
        'save_every_steps': 10, 'keep_last': 3, 'num_workers': 0, 'free_disk_reserve_gib': 60,
        'loss': 'src.segmentation_v2.segmentation_loss', 'center_mse_weight': 200,
        'quality_loss': 'src.segmentation_quality_v18.quality_loss', 'lambda_quality': 1.,
        'gate_schedule': {'zero_through_step': 80, 'linear_until_step': 240, 'maximum': 1.},
        'corruption': {'conditions': ['clean', 'noise', 'missing', 'joint'],
            'assignment': 'canonical_index_plus_epoch_mod_4', 'noise_sigma_m': .02,
            'depth_clip_m': 2., 'missing_rectangle': [120, 160], 'joint_order': ['noise', 'missing'],
            'quality_target': 'M1*exp(-abs(z1-z0)/0.02)', 'quality_scales': 4, 'quality_reduction': 'mean_of_scale_mean_mse'},
        'camera': 'realsense', 'frames': list(range(0, 256, 16)), 'train_scenes': TRAIN_SCENES,
        'validation_scenes': VALIDATION_SCENES, 'data_protocol': 'configs/segmentation_coverage_v8.json',
        'remaining_unopened_validation_scenes': [52, 59, 75, 79, 80]}
    require(all(protocol.get(key) == value for key, value in expected.items()), 'Frozen dual-stream research policy differs')
    limits = protocol['resources']
    expected_resources = {'cpu_threads': 4, 'training_seconds_per_arm': 3600,
        'rss_soft_bytes': 4*1024**3, 'gpu_allocated_soft_bytes': 14495514624,
        'gpu_reserved_soft_bytes': 14495514624, 'output_limit_bytes': 24*1024**3,
        'disk_reserve_bytes': 60*1024**3}
    require(all(limits.get(key) == value for key, value in expected_resources.items()), 'Frozen training resource policy differs')
    require({'experiments/segmentation_dualstream_v18', 'runtime/verification/dualstream_training_v18'}
            <= set(protocol['budget_paths']), 'Training budget omits research or engineering outputs')
    require(set(protocol['arms']) == set(ARMS) and args.arm in ARMS, 'Unexpected dual-stream arm')
    for arm in ARMS:
        require(protocol['arms'][arm] == {'mode': arm, 'input_channels': 5,
            'run_directory': 'experiments/segmentation_dualstream_v18/'+arm}, 'Frozen arm definition differs')
    for name in ('max_steps', 'crash_after_step'):
        value = getattr(args, name)
        require(value is None or type(value) is int and 1 <= value <= 800, 'Invalid absolute step budget')
    require(args.checkpoint_interval is None or type(args.checkpoint_interval) is int
            and 1 <= args.checkpoint_interval <= 10, 'Invalid checkpoint interval')
    ensure_run_target(project_path(args.run_dir), args.arm)
    if project_path(args.run_dir) == ROOT / protocol['arms'][args.arm]['run_directory']:
        require(args.device == 'cuda' and args.checkpoint_interval is None and args.crash_after_step is None,
                'Research runs require frozen CUDA/checkpoint policy; interruption probes are separate')
    verify_frozen_sources(protocol)
    verify_parent_resources(protocol)
    require(runtime_identity() == protocol['runtime'], 'Frozen training runtime changed')


def load_manifest(protocol, arm=None):
    require(arm is None or arm in ARMS, 'Unknown data arm')
    manifest, data_root = baseline.load_manifest(protocol)
    manifest = dict(manifest, schema=18)
    require(len(manifest['records']['train']) == 320, 'Dual-stream training requires exactly 320 frames')
    return manifest, data_root


def reset_random_streams(seed, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def paired_initialization(seed=20261022):
    fixed, quality = make_paired_models(seed)
    first, second = fixed.state_dict(), quality.state_dict()
    equal = first.keys() == second.keys() and all(torch.equal(value, second[name]) for name, value in first.items())
    require(equal, 'Paired dual-stream initialization differs')
    digest = model_tensor_fingerprint(first)
    evidence = {'seed': seed, 'pretrained_weights_used': False, 'parent_checkpoint_used': False,
        'initialization_policy': 'Scratch independent RGB/depth encoders; both arms copied from one CPU model',
        'shared_model_tensor_sha256': digest, 'fixed_initial_model_tensor_sha256': digest,
        'quality_initial_model_tensor_sha256': model_tensor_fingerprint(second), 'common_tensors_equal': equal,
        'initial_output_bitwise_equality_claimed': False}
    reset_random_streams(seed)
    return fixed, quality, evidence


def build_config(protocol, config_path, manifest, paired_evidence, args):
    return {'schema': 18, 'arm': args.arm, 'mode': args.arm, 'protocol': protocol,
        'protocol_sha256': file_sha256(config_path), 'protocol_file': str(config_path.relative_to(ROOT)),
        'data_fingerprint': fingerprint(manifest), 'source_sha256': source_hashes(), 'runtime': runtime_identity(),
        'device': args.device, 'effective_amp': protocol['amp'] and args.device == 'cuda',
        'save_every_optimizer_steps': args.checkpoint_interval or protocol['save_every_steps'],
        'engineering_checkpoint_override': args.checkpoint_interval, 'amp_initial_scale': protocol['amp_initial_scale'],
        'amp_max_overflow_retries': protocol['amp_max_overflow_retries'],
        'num_workers': 0, 'deterministic_algorithms': True, 'paired_initialization': paired_evidence,
        'scheduler_index_rule': 'LambdaLR index k sets next update k+1; clamps at final floor'}


def initialization_evidence(model, optimizer, scheduler, scaler, config):
    require(not optimizer.state_dict()['state'] and scheduler.last_epoch == 0, 'Optimization state is not fresh')
    actual = model_tensor_fingerprint(model.state_dict())
    require(actual == config['paired_initialization'][config['arm'] + '_initial_model_tensor_sha256'], 'Scratch initial model differs')
    return {'schema': 18, 'arm': config['arm'], 'paired_initialization': config['paired_initialization'],
        'initial_model_tensor_sha256': actual, 'optimizer_history_empty': True, 'optimizer_state_entries': 0,
        'scheduler_last_epoch': 0, 'scheduler_state': scheduler.state_dict(), 'initial_used_lr': optimizer.param_groups[0]['lr'],
        'scaler_state': scaler.state_dict(), 'cursor': {'epoch': 0, 'next_batch': 0, 'global_step': 0}, 'history_length': 0,
        'rng_seed': config['protocol']['seed'], 'initial_rng_fingerprint': rng_fingerprint(),
        'initial_rng_fingerprint_scope': 'Python, NumPy and torch CPU; complete device RNG is in every checkpoint',
        'cuda_initialization_seed': config['protocol']['seed'] if config['device'] == 'cuda' else None,
        'optimizer_group_names': [group['name'] for group in optimizer.param_groups],
        'run_config_fingerprint': fingerprint(config)}


def verify_history(history, manifest, protocol, arm=None, final=False):
    require([row['step'] for row in history] == list(range(1, len(history)+1)), 'Noncontiguous training history')
    records, exposures, orders = manifest['records']['train'], {}, {}
    per_epoch = len(records) // protocol['accumulation']
    for step, row in enumerate(history, 1):
        epoch, offset = (step-1)//per_epoch, (step-1)%per_epoch * protocol['accumulation']
        if epoch not in orders:
            orders[epoch] = np.random.RandomState((protocol['seed']+epoch) % 2**32).permutation(len(records)).tolist()
        selected = orders[epoch][offset:offset+protocol['accumulation']]
        expected = [{key: records[index][key] for key in ('scene', 'frame')} for index in selected]
        require(row['epoch'] == epoch and row['samples'] == expected, 'Training sample order/augmentation epoch changed')
        require(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm_before_clip'])
                and all(math.isfinite(value) for value in row['components'].values()), 'Nonfinite training history')
        require(math.isclose(row['learning_rate_used'], expanded.learning_rate_for_update(step, protocol), rel_tol=1e-12, abs_tol=1e-12),
                'Learning rate history changed')
        require(row['gate_strength'] == gate_strength_for_update(step), 'Gate schedule history changed')
        norms = row['gradient_norms_before_clip']
        require(set(norms) == {'segmentation', 'quality_heads'}
                and all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in norms.values())
                and row['gradient_norm_before_clip'] == norms['segmentation'], 'Gradient group history differs')
        require(type(row['amp_retries']) is int and 0 <= row['amp_retries'] < protocol['amp_max_overflow_retries']
                and math.isfinite(row['amp_scale']) and row['amp_scale'] > 0, 'AMP history differs')
        require(len(row['quality_metadata']) == len(selected), 'Quality history sample count differs')
        for index, metadata in zip(selected, row['quality_metadata']):
            identity = corruption_metadata(index, epoch, records[index])
            require(all(metadata.get(key) == value for key, value in identity.items()), 'Quality condition/seed history differs')
        for sample in expected:
            identity = (sample['scene'], sample['frame'])
            exposures[identity] = exposures.get(identity, 0)+1
    if final:
        require(len(history) == protocol['planned_steps'] and len(exposures) == len(records)
                and set(exposures.values()) == {protocol['equivalent_epochs']}, 'Final exposure budget differs')
    return exposures


class ResourceGuard:
    """Sampled engineering budget; native calls are not OS-preempted."""
    def __init__(self, protocol, directory, device, config_hash, started=None):
        self.protocol, self.directory, self.device = protocol, directory, device
        self.config_hash = config_hash
        self.started = time.perf_counter() if started is None else started
        self.path = directory / 'resource_usage.json'
        self.record = read_json(self.path) if self.path.exists() else {}
        require(not self.record or self.record.get('schema') == 18 and self.record.get('config_fingerprint') == config_hash,
                'Resource history binding differs')
        self.previous = self.record.get('elapsed_training_seconds', 0.)
        require(math.isfinite(self.previous) and self.previous >= 0, 'Invalid cumulative resource record')
        require(all(type(self.record.get(key, 0)) is int and self.record.get(key, 0) >= 0 for key in
            ('peak_rss_bytes', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes')), 'Invalid recorded resource peaks')

    def check(self, persist=False, extra_bytes=0):
        limits = self.protocol['resources']
        elapsed = self.previous + time.perf_counter() - self.started
        files = set()
        for name in self.protocol['budget_paths']:
            path = project_path(name)
            if path.is_dir(): files.update(item for item in path.rglob('*') if item.is_file())
            elif path.is_file(): files.add(path)
        total = sum(item.stat().st_size for item in files)
        allocated = max(self.record.get('peak_gpu_allocated_bytes', 0), torch.cuda.max_memory_allocated() if self.device == 'cuda' else 0)
        reserved = max(self.record.get('peak_gpu_reserved_bytes', 0), torch.cuda.max_memory_reserved() if self.device == 'cuda' else 0)
        rss = max(self.record.get('peak_rss_bytes', 0), peak_rss_bytes())
        result = {'schema': 18, 'config_fingerprint': self.config_hash, 'elapsed_training_seconds': elapsed,
                  'peak_rss_bytes': rss, 'peak_gpu_allocated_bytes': allocated,
                  'peak_gpu_reserved_bytes': reserved, 'current_output_bytes': total,
                  'invocations': self.record.get('invocations', []) + [{'elapsed_seconds': time.perf_counter()-self.started}],
                  'scope': 'Sampled cumulative invocation time; an interrupted in-flight native operation may be unrecorded'}
        resource_bytes = len(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))+1 if persist else 0
        require(shutil.disk_usage(ROOT).free - max(extra_bytes, resource_bytes) >= limits['disk_reserve_bytes'], 'Disk reserve exhausted')
        require(total + max(extra_bytes, resource_bytes) <= limits['output_limit_bytes'], 'Round output budget exceeded')
        if persist:
            atomic_json(self.path, result)
        require(elapsed <= limits['training_seconds_per_arm'], 'Cumulative per-arm training time budget exceeded')
        require(rss <= limits['rss_soft_bytes'], 'RSS soft budget exceeded')
        require(allocated <= limits['gpu_allocated_soft_bytes'] and reserved <= limits['gpu_reserved_soft_bytes'], 'GPU soft budget exceeded')
        return result


def peak_rss_bytes():
    if os.name != 'nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb), 'Cannot inspect process memory')
    return int(counters.PeakWorkingSetSize)


def optimization_groups(model):
    segmentation, quality = [], []
    for name, parameter in model.named_parameters():
        (quality if name.startswith('quality_heads.') else segmentation).append(parameter)
    require(segmentation and quality, 'Both optimizer parameter groups are required')
    require(len({id(p) for p in segmentation+quality}) == len(segmentation)+len(quality), 'Optimizer parameters overlap')
    return [{'name': 'segmentation', 'params': segmentation}, {'name': 'quality_heads', 'params': quality}]


def make_optimization(model, config):
    protocol = config['protocol']
    optimizer = torch.optim.AdamW(optimization_groups(model), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda completed: expanded.learning_rate_for_update(completed+1, protocol) / protocol['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['effective_amp'], init_scale=config['amp_initial_scale'])
    return optimizer, scheduler, scaler


def validate_optimization_groups(model, optimizer):
    expected = optimization_groups(model)
    require(len(optimizer.param_groups) == 2, 'Optimizer group count changed')
    for actual, wanted in zip(optimizer.param_groups, expected):
        require(actual['name'] == wanted['name']
                and [id(p) for p in actual['params']] == [id(p) for p in wanted['params']], 'Optimizer parameter group changed')


def train_update(model, optimizer, scaler, dataset, selected, epoch, config, update_index):
    protocol, device = config['protocol'], torch.device(config['device'])
    require(type(update_index) is int and 1 <= update_index <= protocol['planned_steps'], 'Invalid effective update index')
    require(len(selected) == protocol['accumulation'], 'Incomplete effective batch')
    validate_optimization_groups(model, optimizer)
    strength = gate_strength_for_update(update_index)
    attempts = 0
    retry_rng = _capture_rng()
    while True:
        accumulated, parts, quality_records = 0., {}, []
        for index in selected:
            inputs, all_targets = dataset.get(index, epoch=epoch)
            targets = {key: all_targets[key][None].to(device) for key in ('foreground', 'centers', 'offsets')}
            quality_target = all_targets['quality_target'][None].to(device)
            metadata = dict(all_targets['quality_metadata'])
            identity = corruption_metadata(index, epoch, dataset.records[index])
            require(all(metadata.get(key) == value for key, value in identity.items()), 'Synthetic condition/seed metadata differs')
            quality_records.append(metadata)
            with torch.autocast(device_type=config['device'], dtype=torch.float16, enabled=config['effective_amp']):
                outputs = model(inputs[None].to(device), gate_strength=strength)
                segmentation, components = segmentation_loss(outputs, targets)
                auxiliary = quality_loss(outputs['quality_maps'], quality_target)
                loss = segmentation+protocol['lambda_quality']*auxiliary
            require(bool(torch.isfinite(loss)), 'Nonfinite loss; durable checkpoint preserved')
            scaler.scale(loss / len(selected)).backward()
            accumulated += float(loss.detach().cpu()) / len(selected)
            components = dict(components, segmentation=segmentation.detach(), quality_mse=auxiliary.detach())
            for name, value in components.items():
                parts[name] = parts.get(name, 0.) + float(value.cpu()) / len(selected)
        scaler.unscale_(optimizer)
        norms = {group['name']: torch.nn.utils.clip_grad_norm_(group['params'], protocol['gradient_clip_norm'])
                 for group in optimizer.param_groups}
        old_scale = scaler.get_scale()
        require(config['effective_amp'] or all(bool(torch.isfinite(norm)) for norm in norms.values()),
                'Nonfinite gradient; durable checkpoint preserved')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            require(all(bool(torch.isfinite(norm)) for norm in norms.values()), 'Accepted update has nonfinite gradient norm')
            require(all(bool(torch.isfinite(value).all()) for value in model.parameters()), 'Accepted update has nonfinite parameters')
            return accumulated, parts, {key: float(value.detach().cpu()) for key, value in norms.items()}, attempts, quality_records
        attempts += 1
        require(attempts < config['amp_max_overflow_retries'], 'Repeated AMP overflow; no skipped updates counted')
        _restore_rng(retry_rng)


def run(args):
    started = time.perf_counter()
    config_path, directory = project_path(args.config), project_path(args.run_dir)
    protocol = read_json(config_path)
    validate_protocol(protocol, args)
    if args.resume and (directory/'result.json').exists() and read_json(directory/'result.json').get('status') == 'trained':
        _model, _config, _manifest, result, _evidence = load_final_model(directory, protocol)
        print(json.dumps({'status': 'verified_completed_read_only', 'arm': args.arm, 'optimizer_steps': 800}), flush=True)
        return result
    with resource_probe.storage.lock(ROOT/'runtime/grasp_foreground_v12/generation_gpu_lock') if args.device == 'cuda' else nullcontext():
        return run_training(args, protocol, config_path, directory, started)


def run_training(args, protocol, config_path, directory, started):
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    expanded.ensure_disk_reserve(protocol['free_disk_reserve_gib'])
    manifest, data_root = load_manifest(protocol, args.arm)
    fixed, quality, paired = paired_initialization(protocol['seed'])
    model = fixed if args.arm == 'fixed' else quality
    del fixed, quality
    reset_random_streams(protocol['seed'], args.device)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(min(1., protocol['resources']['gpu_reserved_soft_bytes'] /
            torch.cuda.get_device_properties(0).total_memory), 0)
    model.to(torch.device(args.device))
    config = build_config(protocol, config_path, manifest, paired, args)
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    guard = ResourceGuard(protocol, directory, args.device, config_hash, started)
    guard.check(extra_bytes=1024**3)
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(directory):
        # Reload cumulative observations under the same writer lock as checkpoints.
        guard = ResourceGuard(protocol, directory, args.device, config_hash, started)
        guard.check(extra_bytes=1024**3)
        prepare_run_files(directory, config, manifest, args.resume)
        optimizer, scheduler, scaler = make_optimization(model, config)
        initialization = initialization_evidence(model, optimizer, scheduler, scaler, config)
        store = CheckpointStore(directory/'checkpoints', config_hash, data_hash, keep_last=protocol['keep_last'])
        restored = restore_or_initialize_checkpoint(directory, store, model, optimizer, scheduler, scaler, config)
        epoch, next_batch, global_step = (restored['cursor'][key] for key in ('epoch', 'next_batch', 'global_step'))
        history = restored['extra']['history']
        init_path = directory/'initialization.json'
        if init_path.exists():
            require(read_json(init_path) == initialization, 'Scratch initialization evidence changed')
        else:
            require(global_step == 0, 'Nonzero training state without initialization evidence')
            atomic_json(init_path, initialization)
        require(restored['extra']['config'] == config and restored['extra']['data_fingerprint'] == data_hash,
                'Checkpoint configuration/data identity differs')
        require([row['step'] for row in history] == list(range(1, global_step+1)), 'Checkpoint history/cursor differs')
        if global_step:
            require(restored['extra'].get('initialization') == initialization, 'Checkpoint initialization identity differs')
            require((directory/'resource_usage.json').exists(), 'Resource history missing for an existing trained state')
        else:
            require(model_tensor_fingerprint(model.state_dict()) == initialization['initial_model_tensor_sha256']
                    and not optimizer.state_dict()['state'] and rng_fingerprint() == initialization['initial_rng_fingerprint'],
                    'Committed step-zero state differs')
        dataset = QualityGraspNetFrames(data_root, manifest['records']['train'], size=tuple(protocol['input_size']),
                                 seed=protocol['seed'], augment=protocol['augment'])
        accumulation = protocol['accumulation']
        require(len(dataset) % accumulation == 0 and 0 <= next_batch < len(dataset) and next_batch % accumulation == 0
                and global_step == epoch * (len(dataset)//accumulation) + next_batch//accumulation, 'Checkpoint training cursor differs')
        validate_optimization_groups(model, optimizer)
        expected_lr = protocol['learning_rate'] * (expanded.learning_rate_for_update(global_step+1, protocol) / protocol['learning_rate'])
        require(scheduler.last_epoch == global_step and all(group['lr'] == expected_lr for group in optimizer.param_groups)
                and scheduler.get_last_lr() == [expected_lr, expected_lr], 'Checkpoint scheduler state differs')
        verify_history(history, manifest, protocol, args.arm)
        limit = args.max_steps if args.max_steps is not None else protocol['planned_steps']
        require(global_step <= limit <= protocol['planned_steps'], 'Stopping boundary precedes checkpoint or exceeds plan')
        start_step, checkpoint_path = global_step, restored['checkpoint_path']
        guard.check(persist=True)
        atomic_json(directory/'history.json', history)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while global_step < limit:
            guard.check()
            order = dataset.epoch_order(epoch)
            selected = order[next_batch:next_batch+accumulation]
            used_lr = optimizer.param_groups[0]['lr']
            try:
                loss, parts, norms, retries, quality_records = train_update(model, optimizer, scaler, dataset, selected, epoch, config, global_step+1)
            except (RuntimeError, ValueError):
                guard.check(persist=True)
                raise
            scheduler.step()
            global_step += 1
            finished_epoch = epoch
            next_batch += len(selected)
            if next_batch == len(dataset):
                epoch, next_batch = epoch+1, 0
            row = {'step': global_step, 'epoch': finished_epoch, 'loss': loss, 'components': parts,
                'learning_rate_used': used_lr, 'learning_rate_next_update': scheduler.get_last_lr()[0],
                'next_update_index': global_step+1 if global_step < protocol['planned_steps'] else None,
                'next_update_planned': global_step < protocol['planned_steps'], 'gradient_norm_before_clip': norms['segmentation'], 'gradient_norms_before_clip': norms,
                'gate_strength': gate_strength_for_update(global_step), 'quality_metadata': quality_records,
                'amp_retries': retries, 'amp_scale': scaler.get_scale(),
                'samples': [{key: dataset.records[index][key] for key in ('scene', 'frame')} for index in selected]}
            history.append(row)
            guard.check(persist=True)
            saved_now = global_step % config['save_every_optimizer_steps'] == 0 or global_step == limit
            if saved_now:
                expanded.ensure_disk_reserve(protocol['free_disk_reserve_gib'])
                guard.check(extra_bytes=1024**3)
                saved = store.save(model, optimizer, epoch, next_batch, global_step, True, scheduler=scheduler, scaler=scaler,
                    extra={'history': history, 'config': config, 'data_fingerprint': data_hash, 'initialization': initialization})
                checkpoint_path = str(store.directory/saved['file'])
                atomic_json(directory/'history.json', history)
                atomic_json(directory/'progress.json', {'status': 'training', 'arm': args.arm, 'step': global_step,
                    'cursor': {'epoch': epoch, 'next_batch': next_batch, 'global_step': global_step},
                    'config_fingerprint': config_hash, 'data_fingerprint': data_hash})
            print(json.dumps(dict(row, checkpoint_committed=saved_now)), flush=True)
            if args.crash_after_step == global_step:
                os._exit(37)
        trained = global_step == protocol['planned_steps']
        verify_history(history, manifest, protocol, args.arm, final=trained)
        resources = guard.check(persist=True)
        result = {'schema': 18, 'status': 'trained' if trained else 'step_limit_reached', 'arm': args.arm,
            'config_fingerprint': config_hash, 'data_fingerprint': data_hash, 'optimizer_steps': global_step,
            'planned_steps': protocol['planned_steps'], 'arm_training_frames': len(dataset), 'equivalent_epochs': protocol['equivalent_epochs'],
            'resumed_from_step': start_step, 'updates_this_invocation': global_step-start_step,
            'cursor': {'epoch': epoch, 'next_batch': next_batch, 'global_step': global_step},
            'checkpoint_path': str(Path(checkpoint_path).relative_to(ROOT)), 'checkpoint_sha256': file_sha256(checkpoint_path),
            'history_sha256': file_sha256(directory/'history.json'), 'initialization_sha256': file_sha256(init_path),
            'resource_usage_sha256': file_sha256(directory/'resource_usage.json'),
            'validation_evaluated': False, 'evaluation': None, 'automatic_extension': False,
            'decision': 'fixed_budget_trained_no_efficacy_gate' if trained else 'partial_training_no_validation',
            'resources': resources, 'rejected_checkpoint_generations': restored.get('rejected_generations', []),
            'latest_pointer_problem': restored.get('latest_pointer_problem')}
        guard.check(extra_bytes=2*len(json.dumps(result).encode())+4096)
        atomic_json(directory/'result.json', result)
        atomic_json(directory/'progress.json', {key: result[key] for key in ('status', 'arm', 'optimizer_steps', 'cursor', 'config_fingerprint', 'data_fingerprint')})
        print(json.dumps({'status': result['status'], 'arm': args.arm, 'optimizer_steps': global_step,
                          'updates_this_invocation': global_step-start_step, 'validation_evaluated': False, 'resources': resources}), flush=True)
        return result


def load_final_model(run_directory, protocol):
    directory = Path(run_directory).resolve()
    config, result, manifest = (read_json(directory/name) for name in ('config.json', 'result.json', 'data_manifest.json'))
    require(config['schema'] == 18 and config['arm'] in ARMS and config['protocol'] == protocol
            and result['schema'] == 18 and result['status'] == 'trained' and result['arm'] == config['arm'], 'Require a sealed trained dual-stream run')
    config_path = project_path(config['protocol_file'])
    require(read_json(config_path) == protocol and file_sha256(config_path) == config['protocol_sha256'], 'Final protocol identity changed')
    args = argparse.Namespace(arm=config['arm'], run_dir=str(directory), max_steps=None, crash_after_step=None,
                              checkpoint_interval=config['engineering_checkpoint_override'], device=config['device'])
    validate_protocol(protocol, args)
    current_manifest, _ = load_manifest(protocol, config['arm'])
    require(current_manifest == manifest and fingerprint(manifest) == config['data_fingerprint'], 'Final data identity changed')
    fixed, quality, paired = paired_initialization(protocol['seed'])
    model = fixed if config['arm'] == 'fixed' else quality
    del fixed, quality
    require(build_config(protocol, config_path, manifest, paired, args) == config, 'Final runtime/source/config identity changed')
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    checkpoint_directory = directory/'checkpoints'
    pointer = read_json(checkpoint_directory/'latest.json')
    require(Path(pointer['manifest']).name == pointer['manifest'], 'Unsafe checkpoint manifest path')
    metadata_path = checkpoint_directory/pointer['manifest']
    require(file_sha256(metadata_path) == pointer['manifest_sha256'], 'Final checkpoint manifest checksum differs')
    metadata = read_json(metadata_path)
    require(Path(metadata['file']).name == metadata['file'], 'Unsafe checkpoint payload path')
    checkpoint_path = checkpoint_directory/metadata['file']
    require(file_sha256(checkpoint_path) == metadata['sha256'] == result['checkpoint_sha256']
            and checkpoint_path.stat().st_size == metadata['size_bytes']
            and project_path(result['checkpoint_path']) == checkpoint_path, 'Final checkpoint payload binding differs')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for record in (metadata, payload, result):
        require(record['config_fingerprint'] == config_hash and record['data_fingerprint'] == data_hash, 'Final checkpoint source binding differs')
    wanted_cursor = {'epoch': protocol['equivalent_epochs'], 'next_batch': 0, 'global_step': protocol['planned_steps']}
    require(payload['cursor'] == result['cursor'] == wanted_cursor and metadata['global_step'] == result['optimizer_steps'] == protocol['planned_steps']
            and payload['extra']['config'] == config and payload['extra']['data_fingerprint'] == data_hash, 'Final training budget/cursor differs')
    history, initialization = read_json(directory/'history.json'), read_json(directory/'initialization.json')
    require(payload['extra']['history'] == history and file_sha256(directory/'history.json') == result['history_sha256']
            and payload['extra']['initialization'] == initialization and file_sha256(directory/'initialization.json') == result['initialization_sha256'],
            'Final history/initialization differs')
    verify_history(history, manifest, protocol, config['arm'], final=True)
    optimizer, scheduler, scaler = make_optimization(model, config)
    require(initialization == initialization_evidence(model, optimizer, scheduler, scaler, config), 'Final scratch initialization cannot be reconstructed')
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    validate_optimization_groups(model, optimizer)
    final_lr = expanded.learning_rate_for_update(protocol['planned_steps']+1, protocol)
    require(scheduler.last_epoch == protocol['planned_steps'] and all(group['lr'] == final_lr for group in optimizer.param_groups)
            and scheduler.get_last_lr() == [final_lr, final_lr], 'Final scheduler/learning-rate state differs')
    require(len(optimizer.state) == len(list(model.parameters())), 'Final optimizer state is incomplete')
    for state in optimizer.state.values():
        require(float(state['step']) == protocol['planned_steps'] and
                all(bool(torch.isfinite(state[key]).all()) for key in ('exp_avg', 'exp_avg_sq')), 'Final optimizer step/moments differ')
    require(result['validation_evaluated'] is False and result['evaluation'] is None and result['automatic_extension'] is False,
            'Training result contains unauthorized validation or extension')
    require(result['planned_steps'] == protocol['planned_steps'] and result['arm_training_frames'] == len(manifest['records']['train'])
            and result['equivalent_epochs'] == protocol['equivalent_epochs'], 'Final result exposure declarations differ')
    resource_record = read_json(directory/'resource_usage.json')
    require(file_sha256(directory/'resource_usage.json') == result['resource_usage_sha256'] and resource_record == result['resources']
            and resource_record['config_fingerprint'] == config_hash, 'Final cumulative resource evidence differs')
    model.load_state_dict(payload['model'], strict=True)
    require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), 'Nonfinite final model')
    model.eval()
    evidence = {'arm': config['arm'], 'checkpoint': str(checkpoint_path.relative_to(ROOT)), 'checkpoint_sha256': metadata['sha256'],
        'config_fingerprint': config_hash, 'data_fingerprint': data_hash, 'result_sha256': file_sha256(directory/'result.json'),
        'initialization_sha256': result['initialization_sha256'], 'local_optimizer_steps': protocol['planned_steps'],
        'shared_model_tensor_sha256': paired['shared_model_tensor_sha256'], 'initialization': initialization}
    return model, config, manifest, result, evidence


if __name__ == '__main__':
    try:
        run(arguments())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print('Dual-stream training stopped: {}'.format(error), file=sys.stderr, flush=True)
        sys.exit(2)
