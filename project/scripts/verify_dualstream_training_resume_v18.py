"""Retained CUDA/AMP recovery evidence for the paired dual-stream training arms."""
import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from scripts.verify_gpu_resume import compare_tree
from scripts import multiview_common_v14r1 as storage
from src.checkpoints import atomic_json, file_sha256, fingerprint, _exclusive_lock

ARMS = ('fixed', 'quality')
COMPONENTS = ('model', 'optimizer', 'scheduler', 'scaler', 'cursor', 'rng', 'extra')
PHASES = ('continuous', 'interrupted', 'resumed')
SCRATCH = 'runtime/verification/dualstream_training_v18'
LOGS = 'logs/P4/dualstream_training_v18_resume'
CONTRACT = {'continuous_steps': 82, 'interrupted_after_step': 81, 'durable_step': 80,
    'checkpoint_interval': 10, 'initial_output_mode': 'float32_without_autocast_gate_strength_zero',
    'initial_output_atol': 0., 'initial_output_rtol': 0., 'initial_gate_strength': 0.,
    'initial_sample': 'first_seeded_training_sample_epoch0', 'shared_tensors': 'bitwise_equal',
    'optimizer_parameter_groups': 2, 'updates_per_epoch': 80,
    'gate_zero_through_step': 80, 'gate_full_strength_step': 240,
    'nominal_initial_probe_forward_calls': 2, 'nominal_engineering_updates_per_arm': 165,
    'resumed_payload_components': 'model_optimizer_scheduler_scaler_cursor_rng_extra_bitwise_equal'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(root, name):
    root, name = Path(root).resolve(), Path(name)
    path = name.resolve() if name.is_absolute() else (root / name).resolve()
    require(path != root and path.is_relative_to(root), 'Path escaped project')
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def receipt_path(path):
    return Path(str(path) + '.sha256.json')


def once(path, value, read_only=False):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Immutable proof changed: ' + str(path))
    else:
        require(not read_only, 'Missing completed proof: ' + str(path))
        # Existing atomic writer is retained; semantic receipt also binds exact bytes.
        atomic_json(path, value)


def commit(path, value, binding, read_only=False):
    # src.atomic_json preserves insertion order, so use its exact serialized bytes.
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    receipt = {'schema': 18, 'binding_fingerprint': binding['fingerprint'],
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    once(receipt_path(path), receipt, read_only)
    once(path, value, read_only)
    require(file_sha256(path) == receipt['sha256'], 'Proof bytes differ from committed intent')
    return value


def load_commit(path, binding):
    path = Path(path)
    expected = {'schema': 18, 'binding_fingerprint': binding['fingerprint'],
                'sha256': file_sha256(path), 'bytes': path.stat().st_size}
    require(read(receipt_path(path)) == expected, 'Proof receipt changed: ' + str(path))
    return read(path)


def snapshot(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        require(not path.is_symlink(), 'Link in retained engineering evidence')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {
                'sha256': file_sha256(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    return result


def file_evidence(root, path):
    path = Path(path)
    return {'path': path.relative_to(root).as_posix(), 'sha256': file_sha256(path),
            'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}


def verify_file(root, evidence):
    require(file_evidence(root, safe(root, evidence['path'])) == evidence, 'Retained evidence changed')


def load_checkpoint(root, directory, committed_manifest=False, manifest_name=None):
    cache = Path(directory) / 'checkpoints'
    if manifest_name is not None:
        name = manifest_name
    elif committed_manifest:
        # A manifest is the durable commit; latest.json can lag its publication.
        # Refuse corrupt committed evidence rather than silently restarting.
        paths = sorted(cache.glob('checkpoint-*.json'), reverse=True)
        require(bool(paths), 'No committed checkpoint manifest')
        name = paths[0].name
    else:
        pointer = read(cache / 'latest.json')
        name = pointer['manifest']
    require(isinstance(name, str) and re.fullmatch(r'checkpoint-[0-9]{19,20}-[0-9a-f]{12}\.json', name),
            'Invalid checkpoint pointer')
    meta_path = cache / name
    if not committed_manifest and manifest_name is None:
        require(file_sha256(meta_path) == pointer['manifest_sha256'], 'Checkpoint pointer digest changed')
    meta = read(meta_path)
    require(meta['generation'] == Path(name).stem and meta['file'] == Path(name).stem + '.pt',
            'Checkpoint manifest names differ')
    path = cache / meta['file']
    require(path.stat().st_size == meta['size_bytes'] and file_sha256(path) == meta['sha256'],
            'Checkpoint payload integrity failed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(payload['schema_version'] == meta['schema_version'] == 1
            and payload['cursor']['global_step'] == meta['global_step']
            and all(key in payload for key in COMPONENTS), 'Incomplete checkpoint state')
    for key in ('config_fingerprint', 'data_fingerprint'):
        require(payload[key] == meta[key], 'Checkpoint identity changed')
    require(payload['config_fingerprint'] == fingerprint(read(directory / 'config.json'))
            and payload['data_fingerprint'] == fingerprint(read(directory / 'data_manifest.json')),
            'Checkpoint configuration/data differ')
    return payload, {'manifest': file_evidence(root, meta_path), 'payload': file_evidence(root, path),
                     'global_step': meta['global_step']}


def checkpoint_at_step(root, directory, step):
    candidates = [path for path in sorted((directory / 'checkpoints').glob('checkpoint-*.json'), reverse=True)
                  if read(path).get('global_step') == step]
    require(bool(candidates), 'Missing retained durable-boundary checkpoint')
    return load_checkpoint(root, directory, manifest_name=candidates[0].name)


def current_step(root, directory):
    if not list((directory / 'checkpoints').glob('checkpoint-*.json')):
        require(not (directory / 'checkpoints/latest.json').exists(), 'Checkpoint pointer has no committed manifest')
        return None
    payload, _ = load_checkpoint(root, directory, committed_manifest=True)
    return payload['cursor']['global_step']


def validate_contract(protocol):
    require(protocol['schema'] == 18 and set(protocol['arms']) == set(ARMS)
            and protocol['seed'] == 20261022 and protocol['planned_steps'] == 800
            and protocol['accumulation'] == 4 and protocol['amp'] is True
            and protocol['batch_size'] == 1 and protocol['input_size'] == [480, 640]
            and protocol['total_research_update_budget'] == 1600
            and protocol['warmup_steps'] == 40 and protocol['learning_rate'] == .001,
            'Unexpected dual-stream training protocol')
    require(protocol['recovery_contract'] == CONTRACT, 'Recovery contract changed')


def check_partial(result):
    require(result['status'] == 'step_limit_reached' and result['optimizer_steps'] == 82
            and result['planned_steps'] == 800, 'Engineering trajectory ended at the wrong step')
    require(result.get('validation_evaluated') is False and result.get('evaluation') is None,
            'Engineering probe decoded evaluation data')
    start = result['resumed_from_step']
    require(type(start) is int and 0 <= start <= 82 and result['updates_this_invocation'] == 82 - start,
            'Recorded engineering invocation updates differ')


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_initial_tensors(fixed, quality):
    require(bool(fixed) and set(fixed) == set(quality), 'Initial model state names differ')
    require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                for state in (fixed, quality) for value in state.values()), 'Invalid initial model tensor')
    equal = not compare_tree(fixed, quality)
    require(equal, 'Scratch dual-stream tensors are not bitwise identical')
    return {'all_model_tensors_bitwise_equal': equal,
            'same_parameter_names': True, 'same_parameter_shapes_and_dtypes': True}


def flatten_outputs(outputs):
    require(set(outputs) == {'foreground', 'centers', 'offsets', 'quality_maps', 'effective_gates'},
            'Missing initial output head')
    result = {key: outputs[key] for key in ('foreground', 'centers', 'offsets')}
    for key in ('quality_maps', 'effective_gates'):
        require(isinstance(outputs[key], (tuple, list)) and len(outputs[key]) == 4,
                'Expected all four quality/gate scales')
        for index, value in enumerate(outputs[key]):
            result[key + '/' + str(index)] = value
    return result


def compare_initial_outputs(first, second, contract):
    first, second = flatten_outputs(first), flatten_outputs(second)
    results = {}
    for key in sorted(first):
        left, right = first[key].detach().cpu(), second[key].detach().cpu()
        require(left.dtype == right.dtype == torch.float32 and left.shape == right.shape
                and torch.isfinite(left).all() and torch.isfinite(right).all(), 'Invalid initial float32 output')
        difference = (left.double() - right.double()).abs()
        denominator = torch.maximum(left.double().abs(), right.double().abs()).clamp_min(torch.finfo(torch.float32).tiny)
        results[key] = {'shape': list(left.shape), 'fixed_tensor_sha256': tensor_digest(left),
            'quality_tensor_sha256': tensor_digest(right), 'maximum_absolute_difference': float(difference.max()),
            'maximum_symmetric_relative_difference': float((difference / denominator).max()),
            'bitwise_equal': not compare_tree(left, right),
            'within_frozen_tolerance': bool(torch.allclose(left, right,
                atol=contract['initial_output_atol'], rtol=contract['initial_output_rtol']))}
    require(all(item['within_frozen_tolerance'] and item['bitwise_equal'] for item in results.values()),
            'Initial zero-strength output equivalence failed')
    require(all(bool(torch.all(first[key] == 1)) for key in first if key.startswith('effective_gates/')),
            'Initial zero-strength gates are not one')
    return results


def verify_resource_parent(root, protocol):
    evidence = {}
    for field in ('resource_probe_protocol', 'resource_probe_report', 'resource_probe_resume_report'):
        path = safe(root, protocol[field])
        digest = file_sha256(path)
        require(protocol.get(field + '_sha256') == digest
                and protocol['input_hashes'].get(path.relative_to(root).as_posix()) == digest,
                'Resource parent is not frozen: ' + field)
        evidence[field] = {'path': path.relative_to(root).as_posix(), 'sha256': digest}
    resource_protocol = read(safe(root, protocol['resource_probe_protocol']))
    report = read(safe(root, protocol['resource_probe_report']))
    resume = read(safe(root, protocol['resource_probe_resume_report']))
    require(resource_protocol.get('schema') == 18 and resource_protocol.get('status') == 'frozen',
            'Resource protocol is not frozen')
    require(report.get('status') == 'passed_synthetic_gpu_resource_probe'
            and report.get('schema') == 18 and report.get('optimizer_updates') == 0
            and report.get('real_frames') == 0 and resume.get('status') == 'passed'
            and resume.get('schema') == 18, 'Resource prerequisite did not pass')
    require(resume['binding']['protocol_sha256'] == evidence['resource_probe_protocol']['sha256']
            and resume['binding']['probe_binding_fingerprint'] == report['binding_fingerprint'],
            'Resource recovery proof belongs to a different probe')
    return evidence


class Guard:
    def __init__(self, root, protocol):
        self.root, self.protocol, self.started = Path(root), protocol, time.perf_counter()

    def check(self, extra_bytes=0):
        limits = self.protocol['resources']
        require(shutil.disk_usage(self.root).free - extra_bytes >= limits['disk_reserve_bytes'], 'Disk reserve exhausted')
        total = 0
        for name in self.protocol['budget_paths']:
            path = safe(self.root, name)
            total += sum(item.stat().st_size for item in path.rglob('*') if item.is_file()) if path.is_dir() else (
                path.stat().st_size if path.is_file() else 0)
        require(total + extra_bytes <= limits['output_limit_bytes'], 'Round output budget exceeded')
        rss = peak_rss_bytes()
        require(rss <= limits['rss_soft_bytes'], 'Verifier RSS budget exceeded')
        allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        require(allocated <= limits['gpu_allocated_soft_bytes'] and reserved <= limits['gpu_reserved_soft_bytes'],
                'Verifier GPU budget exceeded')
        return {'elapsed_seconds_this_invocation': time.perf_counter() - self.started,
                'current_output_bytes': total, 'peak_rss_bytes': rss,
                'peak_gpu_allocated_bytes': allocated, 'peak_gpu_reserved_bytes': reserved}


def peak_rss_bytes():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb), 'Cannot inspect RSS')
        return int(value.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))


def make_binding(root, config, protocol, trainer):
    validate_contract(protocol)
    require(protocol.get('status') == 'frozen', 'Research training protocol is not frozen')
    trainer.verify_frozen_sources(protocol)
    trainer.verify_parent_resources(protocol)
    parent = verify_resource_parent(root, protocol)
    require(torch.cuda.is_available(), 'Actual CUDA is required for the engineering recovery proof')
    require(Path(sys.prefix).resolve() == safe(root, protocol['interpreter']).parent.parent,
            'Wrong isolated training interpreter')
    for name in ('scripts/verify_dualstream_training_resume_v18.py', 'scripts/verify_gpu_resume.py',
                 'tests/test_dualstream_training_resume_v18.py', 'scripts/multiview_common_v14r1.py',
                 'src/checkpoints.py', 'src/segmentation_dualstream_v18.py', 'src/segmentation_quality_v18.py',
                 'scripts/train_segmentation_dualstream_v18.py'):
        require(protocol['source_hashes'].get(name) == file_sha256(root / name), 'Verifier source is not frozen: ' + name)
    runtime = trainer.runtime_identity()
    require(protocol['runtime'] == runtime, 'Frozen GPU runtime changed')
    manifest, data_root = trainer.load_manifest(protocol)
    require(len(manifest['records']['train']) == 320, 'Expected the same 320 training frames for both arms')
    value = {'schema': 18, 'phase': 'gpu_resume_verification', 'protocol_sha256': file_sha256(config),
        'input_hashes': protocol['input_hashes'], 'source_hashes': protocol['source_hashes'],
        'interpreter_sha256': file_sha256(sys.executable), 'runtime': runtime,
        'data_fingerprint': fingerprint(manifest), 'recovery_contract': protocol['recovery_contract'],
        'resource_parent': parent}
    value['fingerprint'] = fingerprint(value)
    return value, manifest, data_root


def initial_training_sample(protocol, trainer, manifest, data_root):
    sets = {arm: trainer.QualityGraspNetFrames(data_root, manifest['records']['train'],
        size=tuple(protocol['input_size']), seed=protocol['seed'], augment=protocol['augment']) for arm in ARMS}
    index = sets['fixed'].epoch_order(0)[0]
    require(index == sets['quality'].epoch_order(0)[0] == 74, 'First seeded training index changed')
    first, first_targets = sets['fixed'].get(index, epoch=0)
    second, second_targets = sets['quality'].get(index, epoch=0)
    require(not compare_tree(first, second), 'The first RGBD sample/augmentation differs across arms')
    require(not compare_tree(first_targets, second_targets), 'Paired first training targets differ')
    record = manifest['records']['train'][index]
    expected = trainer.corruption_metadata(index, 0, record=record)
    metadata = first_targets['quality_metadata']
    require(expected['condition'] == 'missing'
            and all(metadata.get(key) == value for key, value in expected.items()),
            'Initial probe does not use the first training corruption condition')
    sample = {'scene': record['scene'], 'frame': record['frame'], 'epoch': 0, 'record_index': index,
        'fixed_input_sha256': tensor_digest(first), 'quality_input_sha256': tensor_digest(second),
        'quality_target_sha256': tensor_digest(first_targets['quality_target']),
        'quality_metadata': metadata, 'quality_target_passed_to_model': False}
    return first, second, sample


def initial_probe(root, protocol, trainer, manifest, data_root, guard):
    fixed, quality, evidence = trainer.paired_initialization(protocol['seed'])
    tensor_checks = compare_initial_tensors(fixed.state_dict(), quality.state_dict())
    first, second, sample = initial_training_sample(protocol, trainer, manifest, data_root)
    trainer.reset_random_streams(protocol['seed'], 'cuda')
    models = {'fixed': fixed.to('cuda').eval(), 'quality': quality.to('cuda').eval()}
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        outputs = {'fixed': models['fixed'](first[None].to('cuda'), gate_strength=0.),
                   'quality': models['quality'](second[None].to('cuda'), gate_strength=0.)}
    comparison = compare_initial_outputs(outputs['fixed'], outputs['quality'], protocol['recovery_contract'])
    result = {'status': 'passed', 'paired_model_evidence': evidence, 'tensor_checks': tensor_checks,
        'sample': sample,
        'mode': protocol['recovery_contract']['initial_output_mode'], 'gate_strength': 0.,
        'device': 'cuda', 'output_comparisons': comparison,
        'atol': protocol['recovery_contract']['initial_output_atol'], 'rtol': protocol['recovery_contract']['initial_output_rtol'],
        'validation_evaluated': False, 'new_optimizer_updates': 0, 'model_forward_calls': 2}
    guard.check()
    del outputs, models, fixed, quality
    gc.collect()
    torch.cuda.empty_cache()
    return result


def initial_attempts(root, binding):
    records = []
    for index, path in enumerate(sorted((root / SCRATCH / 'initial_attempts').glob('*.intent.json'))):
        require(path.name == '%04d.intent.json' % index, 'Initial attempt ledger has gaps')
        intent = load_commit(path, binding)
        require(intent == {'schema': 18, 'attempt_index': index,
            'binding_fingerprint': binding['fingerprint'], 'nominal_model_forward_calls': 2,
            'gate_strength': 0.}, 'Initial attempt identity changed')
        result_path = path.with_name(path.name.replace('.intent.json', '.result.json'))
        result = load_commit(result_path, binding) if result_path.exists() else None
        if result is not None:
            require(result['intent_sha256'] == file_sha256(path)
                    and result['probe']['status'] == 'passed'
                    and result['probe']['model_forward_calls'] == 2
                    and result['probe']['new_optimizer_updates'] == 0,
                    'Initial attempt result differs')
        records.append((path, intent, result))
    return records


def ensure_initial_probe(root, protocol, trainer, manifest, data_root, guard, binding, read_only):
    path = root / SCRATCH / 'initialization_probe.json'
    if path.exists():
        return load_commit(path, binding)
    require(not read_only, 'Missing initialization probe')
    attempts = initial_attempts(root, binding)
    completed = [result for _, _, result in attempts if result is not None]
    if completed:
        initial = completed[-1]['probe']
    else:
        index = len(attempts)
        intent_path = root / SCRATCH / 'initial_attempts' / ('%04d.intent.json' % index)
        intent = {'schema': 18, 'attempt_index': index, 'binding_fingerprint': binding['fingerprint'],
                  'nominal_model_forward_calls': 2, 'gate_strength': 0.}
        guard.check(len(json_bytes(intent)) * 2 + 1024)
        commit(intent_path, intent, binding)
        with storage.lock(root / 'runtime/grasp_foreground_v12/generation_gpu_lock'):
            initial = initial_probe(root, protocol, trainer, manifest, data_root, guard)
        result = {'schema': 18, 'intent_sha256': file_sha256(intent_path), 'probe': initial}
        guard.check(len(json_bytes(result)) * 2 + 1024)
        commit(intent_path.with_name('%04d.result.json' % index), result, binding)
    guard.check(len(json_bytes(initial)) * 2 + 1024)
    commit(path, initial, binding)
    return initial


def initial_forward_accounting(root, binding, initial):
    attempts = initial_attempts(root, binding)
    completed = [result for _, _, result in attempts if result is not None]
    require(bool(completed) and all(result['probe'] == initial for result in completed),
            'Completed initial comparison lacks matching retained attempt evidence')
    unknown = len(attempts) - len(completed)
    return {'nominal_model_forward_calls': 2, 'known_completed_model_forward_calls': len(completed) * 2,
            'attempts': len(attempts), 'unknown_outcome_attempts': unknown,
            'exact_call_count_known': unknown == 0, 'new_optimizer_updates': 0,
            'interpretation': 'An unsealed attempt may have executed zero, one or two calls; never treated as zero work'}


def parse_log(path):
    rows, malformed = [], 0
    for line in Path(path).read_text(encoding='utf-8', errors='strict').splitlines():
        if not line.startswith('{'):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if type(value) is dict and 'step' in value:
            require(type(value['step']) is int and 1 <= value['step'] <= 82, 'Unexpected probe update in log')
            rows.append(value)
    require(all(rows[index]['step'] == rows[index - 1]['step'] + 1 for index in range(1, len(rows))),
            'Noncontiguous engineering invocation log')
    return rows, malformed


def invocation(root, config, protocol, binding, arm, phase, directory, guard):
    attempts = root / SCRATCH / 'attempts'
    logs = root / LOGS
    attempts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    intents = sorted(attempts.glob(arm + '_' + phase + '_*.intent.json'))
    for index, path in enumerate(intents):
        require(path.name == '%s_%s_%04d.intent.json' % (arm, phase, index), 'Attempt ledger has gaps')
        old = load_commit(path, binding)
        require(old['arm'] == arm and old['phase'] == phase, 'Attempt ledger identity differs')
    index = len(intents)
    stem = '%s_%s_%04d' % (arm, phase, index)
    log = logs / (stem + '.log')
    require(not log.exists(), 'Attempt log already exists without matching intent')
    start_step = current_step(root, directory)
    command = [sys.executable, '-B', str(root / 'scripts/train_segmentation_dualstream_v18.py'),
        '--config', str(config), '--arm', arm, '--run-dir', str(directory), '--max-steps', '82',
        '--checkpoint-interval', '10', '--device', 'cuda']
    if start_step is not None or (directory / 'config.json').exists():
        command.append('--resume')
    if phase == 'interrupted':
        command += ['--crash-after-step', '81']
    intent = {'schema': 18, 'arm': arm, 'phase': phase, 'attempt_index': index,
        'binding_fingerprint': binding['fingerprint'], 'start_step': start_step, 'command': command,
        'log': log.relative_to(root).as_posix()}
    guard.check(len(json_bytes(intent)) * 2 + 1024)
    commit(attempts / (stem + '.intent.json'), intent, binding)
    guard.check()
    started, returncode, timed_out = time.perf_counter(), None, False
    from scripts.pipeline import project_environment
    with log.open('x', encoding='utf-8') as stream:
        stream.write('COMMAND ' + json.dumps(command, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        try:
            process = subprocess.run(command, cwd=root, env=project_environment(root), stdout=stream,
                stderr=subprocess.STDOUT, timeout=protocol['resources']['training_seconds_per_arm'])
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(intent, exit_code=returncode, timed_out=timed_out,
                  elapsed_seconds=time.perf_counter() - started, log_evidence=file_evidence(root, log))
    # The observed return and duration cannot be regenerated. Publish one atomic
    # record, then bind it into the deterministic phase proof; never invent a lost return.
    guard.check(len(json_bytes(result)) + 1024)
    once(attempts / (stem + '.result.json'), result)
    expected = 37 if phase == 'interrupted' else 0
    require(not timed_out and returncode == expected, 'Probe process failed; preserved log: ' + str(log))
    if phase == 'interrupted':
        rows, malformed = parse_log(log)
        require(not malformed and rows and rows[-1]['step'] == 81 and rows[-1]['checkpoint_committed'] is False,
                'Forced exit lacks the uncommitted step-81 log')
    guard.check()
    return result


def read_attempt_result(root, path, intent):
    value = read(path)
    require(all(value.get(key) == item for key, item in intent.items()), 'Attempt result identity differs')
    require(type(value['timed_out']) is bool and type(value['elapsed_seconds']) in (float, int)
            and math.isfinite(value['elapsed_seconds']) and value['elapsed_seconds'] >= 0
            and (value['exit_code'] is None or type(value['exit_code']) is int), 'Invalid observed process result')
    verify_file(root, value['log_evidence'])
    return value


def latest_process(root, binding, arm, phase):
    result = None
    for path in sorted((root / SCRATCH / 'attempts').glob(arm + '_' + phase + '_*.intent.json')):
        intent = load_commit(path, binding)
        target = path.with_name(path.name.replace('.intent.json', '.result.json'))
        result = read_attempt_result(root, target, intent) if target.exists() else None
    expected = 37 if phase == 'interrupted' else 0
    return result if result is not None and result['exit_code'] == expected and not result['timed_out'] else None


def verify_phase(root, phase, proof, directory):
    require(proof['status'] == 'passed' and proof['phase'] == phase, 'Wrong phase proof')
    for value in proof['checkpoint'].values():
        if isinstance(value, dict) and 'path' in value:
            verify_file(root, value)
    if proof['process'] is not None:
        verify_file(root, proof['process']['log_evidence'])
    if phase == 'interrupted':
        require(proof['checkpoint']['global_step'] == 80 and proof['process']['exit_code'] == 37,
                'Durable forced-exit evidence differs')
    else:
        require(proof['checkpoint']['global_step'] == 82 and snapshot(directory) == proof['completed_snapshot'],
                'Completed engineering trajectory changed')


def ensure_phase(root, config, protocol, binding, arm, phase, guard, read_only=False):
    base = root / SCRATCH
    directory = base / arm / ('continuous' if phase == 'continuous' else 'interrupted')
    path = base / 'proofs' / (arm + '_' + phase + '.json')
    if path.exists():
        proof = load_commit(path, binding)
        verify_phase(root, phase, proof, directory)
        return proof
    require(not read_only, 'Missing phase proof in read-only mode')
    step = current_step(root, directory)
    process = latest_process(root, binding, arm, phase)
    if phase == 'interrupted':
        require(step is None or 0 <= step <= 80, 'Cannot invent lost forced-exit proof after later updates')
        if step != 80 or process is None:
            process = invocation(root, config, protocol, binding, arm, phase, directory, guard)
        rows, malformed = parse_log(safe(root, process['log']))
        require(not malformed and rows and rows[-1]['step'] == 81 and rows[-1]['checkpoint_committed'] is False,
                'Missing logged forced-exit boundary')
    elif step != 82 or not (directory / 'result.json').exists():
        require(step is None or 0 <= step <= 82, 'Engineering trajectory exceeded its target')
        if phase == 'resumed':
            require(step is not None and 80 <= step <= 82, 'Recovery did not start from the forced boundary')
        process = invocation(root, config, protocol, binding, arm, phase, directory, guard)
    payload, checkpoint = load_checkpoint(root, directory, committed_manifest=True)
    expected = {'epoch': 1, 'next_batch': 0 if phase == 'interrupted' else 8,
                'global_step': 80 if phase == 'interrupted' else 82}
    require(payload['cursor'] == expected, 'Incorrect epoch-boundary checkpoint cursor')
    if phase != 'interrupted':
        check_partial(read(directory / 'result.json'))
    proof = {'schema': 18, 'status': 'passed', 'arm': arm, 'phase': phase,
             'binding_fingerprint': binding['fingerprint'], 'checkpoint': checkpoint, 'process': process,
             'completed_snapshot': snapshot(directory) if phase != 'interrupted' else None}
    guard.check(len(json_bytes(proof)) * 2 + 1024)
    commit(path, proof, binding)
    return proof


def check_initialization(record, paired, protocol, payload):
    arm = record['arm']
    require(record['schema'] == 18 and arm in ARMS and record['paired_initialization'] == paired
            and record['initial_model_tensor_sha256'] == paired[arm + '_initial_model_tensor_sha256']
            and record['optimizer_history_empty'] is True and record['optimizer_state_entries'] == 0
            and record['scheduler_last_epoch'] == 0 and record['history_length'] == 0
            and record['cursor'] == {'epoch': 0, 'next_batch': 0, 'global_step': 0}
            and record['rng_seed'] == protocol['seed'] and record['run_config_fingerprint'] == payload['config_fingerprint'],
            'Invalid shared scratch initialization')
    require(record['initial_used_lr'] == protocol['learning_rate'] / protocol['warmup_steps'], 'Initial warmup differs')
    require(record['scaler_state'] == {'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
            'growth_interval': 2000, '_growth_tracker': 0}, 'Initial AMP scaler differs')
    require(record['scheduler_state']['last_epoch'] == 0 and record['scheduler_state']['_step_count'] == 1
            and record['scheduler_state']['base_lrs'] == [protocol['learning_rate']] * 2
            and record['scheduler_state']['_last_lr'] == [record['initial_used_lr']] * 2,
            'Initial two-group scheduler differs')


def verify_gate_boundary(history):
    require(len(history) == 82, 'Engineering gate history is incomplete')
    expected = [0. if step <= 80 else (step - 80) / 160. for step in range(1, 83)]
    for step, (row, strength) in enumerate(zip(history, expected), 1):
        require(row.get('step') == step and type(row.get('gate_strength')) in (int, float)
                and row['gate_strength'] == strength, 'Gate history changed at update ' + str(step))
        require(row.get('epoch') == (step - 1) // 80, 'Gate history crossed the wrong epoch')
    return expected


def arm_evidence(root, protocol, trainer, manifest, binding, arm, initial, phases):
    base = root / SCRATCH / arm
    reference, reference_file = load_checkpoint(root, base / 'continuous', committed_manifest=True)
    recovered, recovered_file = load_checkpoint(root, base / 'interrupted', committed_manifest=True)
    differences = {key: compare_tree(reference[key], recovered[key], key) for key in COMPONENTS}
    require(all(not value for value in differences.values()), 'Recovery state mismatch: ' + str(differences))
    require(reference['config_fingerprint'] == recovered['config_fingerprint']
            and reference['data_fingerprint'] == recovered['data_fingerprint'], 'Recovery identities differ')
    require(reference['cursor'] == {'epoch': 1, 'next_batch': 8, 'global_step': 82}, 'Wrong final engineering cursor')
    trainer.verify_history(reference['extra']['history'], manifest, protocol, arm, final=False)
    require(len(reference['extra']['history']) == 82, 'Engineering history is incomplete')
    records = {name: read(base / name / 'initialization.json') for name in ('continuous', 'interrupted')}
    for record in records.values():
        check_initialization(record, initial['paired_model_evidence'], protocol, reference)
    require(not compare_tree(records['continuous'], records['interrupted']), 'Within-arm initialization changed')
    history = reference['extra']['history']
    strengths = verify_gate_boundary(history)
    prefix, prefix_file = checkpoint_at_step(root, base / 'continuous', 80)
    durable, durable_file = checkpoint_at_step(root, base / 'interrupted', 80)
    require(all(not compare_tree(prefix[key], durable[key], key) for key in COMPONENTS),
            'Durable step-80 full state differs from the uninterrupted trajectory')
    require(prefix['cursor'] == durable['cursor'] == {'epoch': 1, 'next_batch': 0, 'global_step': 80}
            and not compare_tree(history[:80], durable['extra']['history']),
            'Durable step-80 history prefix changed')
    return {'schema': 18, 'status': 'passed', 'arm': arm, 'binding_fingerprint': binding['fingerprint'],
        'bitwise_equal': {key: True for key in COMPONENTS}, 'final_cursor': recovered['cursor'],
        'config_fingerprint': reference['config_fingerprint'], 'data_fingerprint': reference['data_fingerprint'],
        'initialization': records['continuous'], 'history_sha256': fingerprint(history),
        'samples_sha256': fingerprint([row['samples'] for row in history]),
        'learning_rates': [row['learning_rate_used'] for row in history],
        'gate_strengths': strengths, 'durable_prefix_bitwise_equal': True,
        'continuous_durable_checkpoint': prefix_file, 'interrupted_durable_checkpoint': durable_file,
        'durable_history_sha256': fingerprint(durable['extra']['history']),
        'continuous_checkpoint': reference_file, 'recovered_checkpoint': recovered_file,
        'durable_step_before_forced_exit': 80, 'logical_updates_after_durable_boundary': 2,
        'process_phases': phases, 'validation_evaluated': False,
        'completed_snapshots': {name: snapshot(base / name) for name in ('continuous', 'interrupted')}}


def update_accounting(root, binding):
    attempts = root / SCRATCH / 'attempts'
    records, known, exact = [], 0, True
    for path in sorted(attempts.glob('*.intent.json')):
        intent = load_commit(path, binding)
        log = safe(root, intent['log'])
        result_path = path.with_name(path.name.replace('.intent.json', '.result.json'))
        result = read_attempt_result(root, result_path, intent) if result_path.exists() else None
        rows, malformed = parse_log(log) if log.exists() else ([], 0)
        if result:
            verify_file(root, result['log_evidence'])
        certain = result is not None and result['exit_code'] in (0, 37) and not result['timed_out'] and not malformed
        exact = exact and certain
        known += len(rows)
        records.append({'arm': intent['arm'], 'phase': intent['phase'], 'attempt_index': intent['attempt_index'],
            'logged_accepted_updates': len(rows), 'exact_count_known': certain,
            'observed_exit_code': result['exit_code'] if result else None, 'log': intent['log'],
            'log_sha256': file_sha256(log) if log.exists() else None,
            'intent_sha256': file_sha256(path), 'result_sha256': file_sha256(result_path) if result else None})
    return {'nominal_updates_per_arm': 165, 'nominal_total_updates': 330, 'logged_accepted_updates': known,
        'exact_count_known': exact, 'interpretation': 'Exact when all attempt returns and flushed step logs survive; otherwise a lower bound',
        'excluded_from_research_update_budget': True, 'research_updates_used': 0, 'attempts': records}


def cross_arm_checks(arms):
    first, second = arms['fixed'], arms['quality']
    common_fields = ('optimizer_history_empty', 'optimizer_state_entries', 'scheduler_last_epoch',
        'scheduler_state', 'initial_used_lr', 'scaler_state', 'cursor', 'history_length', 'rng_seed', 'initial_rng_fingerprint')
    checks = {'same_data_fingerprint': first['data_fingerprint'] == second['data_fingerprint'],
        'same_training_sample_trajectory': first['samples_sha256'] == second['samples_sha256'],
        'same_learning_rate_trajectory': first['learning_rates'] == second['learning_rates'],
        'same_explicit_gate_schedule': first['gate_strengths'] == second['gate_strengths'],
        'same_reset_optimization_and_rng_state': all(not compare_tree(first['initialization'][key],
            second['initialization'][key]) for key in common_fields),
        'both_full_state_bitwise_resume': all(all(value['bitwise_equal'].values()) for value in arms.values())}
    require(all(checks.values()), 'Cross-arm engineering contract failed')
    return checks


def verify(config_path, check_only=False, root=ROOT):
    from scripts import train_segmentation_dualstream_v18 as trainer
    root, config = Path(root).resolve(), safe(root, config_path)
    protocol = read(config)
    binding, manifest, data_root = make_binding(root, config, protocol, trainer)
    guard = Guard(root, protocol)
    guard.check()
    base, output = root / SCRATCH, safe(root, protocol['outputs']['training_resume_report'])

    def execute(read_only):
        guard.check(0 if (base / 'binding.json').exists() else len(json_bytes(binding)) + 1024)
        once(base / 'binding.json', binding, read_only)
        initial = ensure_initial_probe(root, protocol, trainer, manifest, data_root, guard, binding, read_only)
        arms = {}
        for arm in ARMS:
            phase_proofs = {phase: ensure_phase(root, config, protocol, binding, arm, phase, guard, read_only)
                            for phase in PHASES}
            value = arm_evidence(root, protocol, trainer, manifest, binding, arm, initial, phase_proofs)
            path = base / 'proofs' / (arm + '_complete.json')
            guard.check(0 if path.exists() else len(json_bytes(value)) * 2 + 1024)
            commit(path, value, binding, read_only)
            arms[arm] = value
            guard.check()
        report = {'schema': 18, 'status': 'passed', 'binding': binding, 'binding_fingerprint': binding['fingerprint'],
            'paired_initialization': initial, 'arms': arms, 'cross_arm_checks': cross_arm_checks(arms),
            'initial_probe_forward_accounting': initial_forward_accounting(root, binding, initial),
            'engineering_update_accounting': update_accounting(root, binding), 'checkpoint_interval': 10,
            'planned_research_updates': 1600, 'excluded_from_research_results': True,
            'validation_evaluated': False, 'real_power_cut_tested': False,
            'engineering_checkpoints_retained': True,
            'scope': 'Actual CUDA/AMP interruption; full state equality at steps 80 and 82 across epoch change and gate activation',
            'initial_output_claim': 'All segmentation, quality and gate outputs bitwise equal in float32 at explicit gate strength zero',
            'limitations': protocol['limitations']}
        guard.check(0 if output.exists() else len(json_bytes(report)) * 2 + 1024)
        commit(output, report, binding, read_only)
        return {'status': 'verified_completed_read_only' if read_only else 'passed',
            'report_sha256': file_sha256(output), 'report': output.relative_to(root).as_posix(),
            'engineering_logged_updates': report['engineering_update_accounting']['logged_accepted_updates'],
            'research_updates_used': 0, 'resources': guard.check()}

    if check_only or output.exists():
        return execute(True)
    base.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(base):
        return execute(output.exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/segmentation_dualstream_v18.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.check_only), ensure_ascii=False, allow_nan=False), flush=True)
