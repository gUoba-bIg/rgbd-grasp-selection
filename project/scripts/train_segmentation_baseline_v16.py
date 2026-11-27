"""Matched RGB/RGB-D scratch baselines; training only, no validation decoding."""
import argparse
import ctypes
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
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet
from src.segmentation_v2 import segmentation_loss

ARMS = ('rgb', 'rgbd')
TRAIN_SCENES = sorted(coverage.OLD_TRAIN + coverage.NEW_TRAIN)
VALIDATION_SCENES = [2, 6, 7, 11, 13, 15, 18, 22, 33, 39]
CONV_KEY = 'stem.0.weight'
project_path = expanded.project_path
model_tensor_fingerprint = coverage.model_tensor_fingerprint
rng_fingerprint = coverage.rng_fingerprint


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/segmentation_baseline_v16.json')
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
        'scripts/train_segmentation_baseline_v16.py', 'src/segmentation.py', 'src/segmentation_v2.py',
        'src/graspnet_frames.py', 'src/checkpoints.py'}
    return {name: file_sha256(project_path(name)) for name in sorted(names)}


def verify_frozen_sources(protocol):
    require(bool(protocol['source_hashes']) and bool(protocol['input_hashes']), 'Unfrozen baseline sources/inputs')
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
    research = ROOT / 'experiments/segmentation_baseline_v16' / arm
    verification = ROOT / 'runtime/verification'
    engineering = directory.is_relative_to(verification) and 'v16' in directory.relative_to(verification).as_posix()
    require(directory == research or engineering, 'Run directory is not an allowed research or v16 verification target')
    require(arm in directory.parts and directory != verification, 'Run directory must identify its arm')
    for ancestor in directory.parents:
        if ancestor == ROOT:
            break
        require(not (ancestor / 'config.json').exists(), 'Run cannot be nested inside another experiment')
    if (directory / 'config.json').exists():
        old = read_json(directory / 'config.json')
        require(old.get('schema') == 16 and old.get('arm') == arm, 'Existing directory belongs to another experiment')


def validate_protocol(protocol, args):
    expected = {'schema': 16, 'seed': 20261022, 'input_size': [240, 320], 'batch_size': 1, 'accumulation': 2,
        'planned_steps': 1600, 'equivalent_epochs': 10, 'total_research_update_budget': 3200,
        'segment_boundaries': [200, 800, 1600], 'augment': True, 'learning_rate': .001, 'weight_decay': .0001,
        'warmup_steps': 80, 'min_learning_rate': .00001, 'amp': True, 'gradient_clip_norm': 1.,
        'save_every_steps': 10, 'keep_last': 3, 'num_workers': 0, 'free_disk_reserve_gib': 60,
        'loss': 'src.segmentation_v2.segmentation_loss', 'center_mse_weight': 200, 'camera': 'realsense',
        'frames': list(range(0, 256, 16)), 'train_scenes': TRAIN_SCENES, 'validation_scenes': VALIDATION_SCENES,
        'data_protocol': 'configs/segmentation_coverage_v8.json'}
    require(all(protocol.get(key) == value for key, value in expected.items()), 'Frozen matched-baseline policy differs')
    resources = {'cpu_threads': 4, 'training_seconds_per_arm': 1800, 'evaluation_seconds_per_invocation': 1800,
        'acceptance_seconds_per_invocation': 1800, 'rss_soft_bytes': 4*1024**3,
        'gpu_allocated_soft_bytes': 14495514624, 'gpu_reserved_soft_bytes': 14495514624,
        'output_limit_bytes': 12*1024**3, 'disk_reserve_bytes': 60*1024**3}
    require(protocol.get('resources') == resources, 'Frozen resource policy differs')
    require(protocol.get('remaining_unopened_validation_scenes') == [52, 59, 75, 79, 80], 'Unopened validation declaration differs')
    require(set(protocol['arms']) == set(ARMS) and args.arm in ARMS, 'Unexpected baseline arm')
    for arm, channels in (('rgb', 3), ('rgbd', 5)):
        require(protocol['arms'][arm]['mode'] == arm and protocol['arms'][arm]['input_channels'] == channels
                and protocol['arms'][arm]['run_directory'] == 'experiments/segmentation_baseline_v16/' + arm,
                'Frozen arm definition differs')
    for name in ('max_steps', 'crash_after_step'):
        value = getattr(args, name)
        require(value is None or type(value) is int and 1 <= value <= 1600, 'Invalid absolute step budget')
    require(args.checkpoint_interval is None or type(args.checkpoint_interval) is int and 1 <= args.checkpoint_interval <= 10, 'Invalid checkpoint interval')
    ensure_run_target(project_path(args.run_dir), args.arm)
    if project_path(args.run_dir) == ROOT / protocol['arms'][args.arm]['run_directory']:
        require(args.device == 'cuda' and args.checkpoint_interval is None and args.crash_after_step is None,
                'Research runs require frozen CUDA/checkpoint policy; interruption probes are separate')
    verify_frozen_sources(protocol)
    if 'runtime' in protocol:
        require(runtime_identity() == protocol['runtime'], 'Frozen training runtime changed')


def load_manifest(protocol, arm=None):
    require(arm is None or arm in ARMS, 'Unknown data arm')
    path = project_path(protocol['data_protocol'])
    require(file_sha256(path) == protocol['data_protocol_sha256'], 'Frozen V8 data protocol changed')
    original = read_json(path)
    require(all(original[key] == protocol[key] for key in ('data_root', 'data_manifest', 'data_audit', 'split', 'frames', 'camera', 'validation_scenes')),
            'Baseline data declarations differ from V8')
    parent, data_root = coverage.load_manifest(original, 'coverage_expanded')
    require(original['arms']['coverage_expanded']['train_scenes'] == protocol['train_scenes'], 'Frozen twenty-scene selection differs')
    manifest = dict(parent)
    manifest.pop('arm')
    manifest.update(schema=16, validation_policy='Only byte hashes checked by trainer; validation decoding belongs to a separate final-evaluation stage')
    return manifest, data_root


def reset_random_streams(seed, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    """Construct both arms from one scratch RGB reference, then reset identical RNG streams."""
    reset_random_streams(seed)
    rgb = PilotInstanceNet(mode='rgb')
    rgbd = PilotInstanceNet(mode='rgbd')
    reference = rgb.state_dict()
    target = {name: value.clone() for name, value in reference.items()}
    target[CONV_KEY] = torch.zeros_like(rgbd.state_dict()[CONV_KEY])
    target[CONV_KEY][:, :3].copy_(reference[CONV_KEY])
    rgbd.load_state_dict(target, strict=True)
    shared = {name: value for name, value in reference.items() if name != CONV_KEY}
    evidence = {'seed': seed, 'pretrained_weights_used': False, 'parent_checkpoint_used': False,
        'initialization_policy': 'Scratch RGB reference; exact shared state and RGB channels; extra depth/validity channels zero',
        'shared_model_tensor_sha256': model_tensor_fingerprint(shared),
        'rgb_initial_model_tensor_sha256': model_tensor_fingerprint(reference),
        'rgbd_initial_model_tensor_sha256': model_tensor_fingerprint(rgbd.state_dict()),
        'rgb_conv_tensor_sha256': model_tensor_fingerprint({CONV_KEY: reference[CONV_KEY]}),
        'extra_input_channels_zero': bool(torch.count_nonzero(rgbd.state_dict()[CONV_KEY][:, 3:]) == 0),
        'shared_rgb_channels_equal': torch.equal(reference[CONV_KEY], rgbd.state_dict()[CONV_KEY][:, :3]),
        'common_tensors_equal': all(torch.equal(value, rgbd.state_dict()[name]) for name, value in shared.items()),
        'initial_output_bitwise_equality_claimed': False}
    require(all(evidence[key] for key in ('extra_input_channels_zero', 'shared_rgb_channels_equal', 'common_tensors_equal')),
            'Matched scratch initialization failed')
    reset_random_streams(seed)
    return rgb, rgbd, evidence


def build_config(protocol, config_path, manifest, paired_evidence, args):
    return {'schema': 16, 'arm': args.arm, 'mode': args.arm, 'protocol': protocol,
        'protocol_sha256': file_sha256(config_path), 'protocol_file': str(config_path.relative_to(ROOT)),
        'data_fingerprint': fingerprint(manifest), 'source_sha256': source_hashes(), 'runtime': runtime_identity(),
        'device': args.device, 'effective_amp': protocol['amp'] and args.device == 'cuda',
        'save_every_optimizer_steps': args.checkpoint_interval or protocol['save_every_steps'],
        'engineering_checkpoint_override': args.checkpoint_interval, 'amp_initial_scale': 128., 'amp_max_overflow_retries': 16,
        'num_workers': 0, 'deterministic_algorithms': True, 'paired_initialization': paired_evidence,
        'scheduler_index_rule': 'LambdaLR index k sets next update k+1; clamps at final floor'}


def initialization_evidence(model, optimizer, scheduler, scaler, config):
    require(not optimizer.state_dict()['state'] and scheduler.last_epoch == 0, 'Optimization state is not fresh')
    actual = model_tensor_fingerprint(model.state_dict())
    require(actual == config['paired_initialization'][config['arm'] + '_initial_model_tensor_sha256'], 'Scratch initial model differs')
    return {'schema': 16, 'arm': config['arm'], 'paired_initialization': config['paired_initialization'],
        'initial_model_tensor_sha256': actual, 'optimizer_history_empty': True, 'optimizer_state_entries': 0,
        'scheduler_last_epoch': 0, 'scheduler_state': scheduler.state_dict(), 'initial_used_lr': optimizer.param_groups[0]['lr'],
        'scaler_state': scaler.state_dict(), 'cursor': {'epoch': 0, 'next_batch': 0, 'global_step': 0}, 'history_length': 0,
        'rng_seed': config['protocol']['seed'], 'initial_rng_fingerprint': rng_fingerprint(), 'run_config_fingerprint': fingerprint(config)}


def verify_history(history, manifest, protocol, arm=None, final=False):
    require([row['step'] for row in history] == list(range(1, len(history)+1)), 'Noncontiguous training history')
    records, exposures, orders = manifest['records']['train'], {}, {}
    per_epoch = len(records) // protocol['accumulation']
    for step, row in enumerate(history, 1):
        epoch, offset = (step-1)//per_epoch, (step-1)%per_epoch * protocol['accumulation']
        if epoch not in orders:
            orders[epoch] = np.random.RandomState((protocol['seed']+epoch) % 2**32).permutation(len(records)).tolist()
        expected = [{key: records[index][key] for key in ('scene', 'frame')} for index in orders[epoch][offset:offset+protocol['accumulation']]]
        require(row['epoch'] == epoch and row['samples'] == expected, 'Training sample order/augmentation epoch changed')
        require(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm_before_clip'])
                and all(math.isfinite(value) for value in row['components'].values()), 'Nonfinite training history')
        require(math.isclose(row['learning_rate_used'], expanded.learning_rate_for_update(step, protocol), rel_tol=1e-12, abs_tol=1e-12),
                'Learning rate history changed')
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
        require(not self.record or self.record.get('schema') == 16 and self.record.get('config_fingerprint') == config_hash,
                'Resource history binding differs')
        self.previous = self.record.get('elapsed_training_seconds', 0.)
        require(math.isfinite(self.previous) and self.previous >= 0, 'Invalid cumulative resource record')
        require(all(type(self.record.get(key, 0)) is int and self.record.get(key, 0) >= 0 for key in
            ('peak_rss_bytes', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes')), 'Invalid recorded resource peaks')

    def check(self, persist=False, extra_bytes=0):
        limits = self.protocol['resources']
        elapsed = self.previous + time.perf_counter() - self.started
        total = 0
        for name in self.protocol['budget_paths']:
            path = project_path(name)
            total += sum(item.stat().st_size for item in path.rglob('*') if item.is_file()) if path.is_dir() else (path.stat().st_size if path.is_file() else 0)
        allocated = max(self.record.get('peak_gpu_allocated_bytes', 0), torch.cuda.max_memory_allocated() if self.device == 'cuda' else 0)
        reserved = max(self.record.get('peak_gpu_reserved_bytes', 0), torch.cuda.max_memory_reserved() if self.device == 'cuda' else 0)
        rss = max(self.record.get('peak_rss_bytes', 0), peak_rss_bytes())
        result = {'schema': 16, 'config_fingerprint': self.config_hash, 'elapsed_training_seconds': elapsed,
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


def make_optimization(model, config):
    protocol = config['protocol']
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda completed: expanded.learning_rate_for_update(completed+1, protocol) / protocol['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['effective_amp'], init_scale=config['amp_initial_scale'])
    return optimizer, scheduler, scaler


def train_update(model, optimizer, scaler, dataset, selected, epoch, config):
    protocol, device = config['protocol'], torch.device(config['device'])
    attempts = 0
    retry_rng = _capture_rng()
    while True:
        accumulated, parts = 0., {}
        for index in selected:
            inputs, targets = dataset.get(index, epoch=epoch)
            targets = {key: value[None].to(device) for key, value in targets.items() if key != 'labels'}
            with torch.autocast(device_type=config['device'], dtype=torch.float16, enabled=config['effective_amp']):
                outputs = model(inputs[None].to(device))
                loss, components = segmentation_loss(outputs, targets)
            require(bool(torch.isfinite(loss)), 'Nonfinite loss; durable checkpoint preserved')
            scaler.scale(loss / len(selected)).backward()
            accumulated += float(loss.detach().cpu()) / len(selected)
            for name, value in components.items():
                parts[name] = parts.get(name, 0.) + float(value.cpu()) / len(selected)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip_norm'])
        old_scale = scaler.get_scale()
        require(config['effective_amp'] or bool(torch.isfinite(norm)), 'Nonfinite gradient; durable checkpoint preserved')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            require(bool(torch.isfinite(norm)), 'Accepted update has nonfinite gradient norm')
            require(all(bool(torch.isfinite(value).all()) for value in model.parameters()), 'Accepted update has nonfinite parameters')
            return accumulated, parts, float(norm.detach().cpu()), attempts
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
        print(json.dumps({'status': 'verified_completed_read_only', 'arm': args.arm, 'optimizer_steps': 1600}), flush=True)
        return result
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    expanded.ensure_disk_reserve(protocol['free_disk_reserve_gib'])
    manifest, data_root = load_manifest(protocol, args.arm)
    rgb, rgbd, paired = paired_initialization(protocol['seed'])
    model = rgb if args.arm == 'rgb' else rgbd
    del rgb, rgbd
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
        dataset = GraspNetFrames(data_root, manifest['records']['train'], size=tuple(protocol['input_size']), mode=args.arm,
                                 seed=protocol['seed'], augment=protocol['augment'])
        accumulation = protocol['accumulation']
        require(len(dataset) % accumulation == 0 and 0 <= next_batch < len(dataset) and next_batch % accumulation == 0
                and global_step == epoch * (len(dataset)//accumulation) + next_batch//accumulation, 'Checkpoint training cursor differs')
        expected_lr = protocol['learning_rate'] * (expanded.learning_rate_for_update(global_step+1, protocol) / protocol['learning_rate'])
        require(scheduler.last_epoch == global_step and optimizer.param_groups[0]['lr'] == expected_lr
                and scheduler.get_last_lr()[0] == expected_lr, 'Checkpoint scheduler state differs')
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
                loss, parts, norm, retries = train_update(model, optimizer, scaler, dataset, selected, epoch, config)
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
                'next_update_planned': global_step < protocol['planned_steps'], 'gradient_norm_before_clip': norm,
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
        result = {'schema': 16, 'status': 'trained' if trained else 'step_limit_reached', 'arm': args.arm,
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
    require(config['schema'] == 16 and config['arm'] in ARMS and config['protocol'] == protocol
            and result['schema'] == 16 and result['status'] == 'trained' and result['arm'] == config['arm'], 'Require a sealed trained baseline')
    config_path = project_path(config['protocol_file'])
    require(read_json(config_path) == protocol and file_sha256(config_path) == config['protocol_sha256'], 'Final protocol identity changed')
    args = argparse.Namespace(arm=config['arm'], run_dir=str(directory), max_steps=None, crash_after_step=None,
                              checkpoint_interval=config['engineering_checkpoint_override'], device=config['device'])
    validate_protocol(protocol, args)
    current_manifest, _ = load_manifest(protocol, config['arm'])
    require(current_manifest == manifest and fingerprint(manifest) == config['data_fingerprint'], 'Final data identity changed')
    rgb, rgbd, paired = paired_initialization(protocol['seed'])
    model = rgb if config['arm'] == 'rgb' else rgbd
    del rgb, rgbd
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
    final_lr = expanded.learning_rate_for_update(protocol['planned_steps']+1, protocol)
    require(scheduler.last_epoch == protocol['planned_steps'] and optimizer.param_groups[0]['lr'] == final_lr
            and scheduler.get_last_lr()[0] == final_lr, 'Final scheduler/learning-rate state differs')
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
        print('Baseline training stopped: {}'.format(error), file=sys.stderr, flush=True)
        sys.exit(2)
