"""Bounded synthetic GPU resource probe; no research frames or optimizer updates."""
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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import multiview_common_v14r1 as storage
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

UNITS = ('fixed_fp32', 'quality_fp32', 'fixed_amp', 'quality_amp')
HEADS = ('foreground', 'centers', 'offsets')
CONFIG = 'configs/dualstream_resources_v18.json'
RUNTIME = {'python': '3.12.14', 'torch': '2.6.0+cu126', 'torchvision': '0.21.0+cu126',
           'numpy': '1.26.4', 'scipy': '1.14.1', 'Pillow': '11.3.0', 'cuda': '12.6',
           'cudnn': 90501, 'gpu': 'NVIDIA GeForce RTX 4060 Ti'}
require = storage.require


def safe(root, name):
    root = Path(root).resolve()
    path = Path(name)
    path = path.resolve() if path.is_absolute() else (root/path).resolve()
    require(path != root and path.is_relative_to(root), 'Probe path escaped project')
    return path


def receipt(file):
    return storage.receipt_path(file)


def commit_bytes(file, payload, binding, readonly=False):
    record = {'schema': 18, 'binding_fingerprint': binding['fingerprint'],
              'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    storage.once(receipt(file), record, readonly)
    storage.publish_once(file, payload, readonly)


def commit(file, value, binding, readonly=False):
    commit_bytes(file, storage.json_bytes(value), binding, readonly)


def verify_bytes(file, binding):
    require(storage.read(receipt(file)) == {'schema': 18, 'binding_fingerprint': binding['fingerprint'],
        'sha256': storage.digest(file), 'bytes': Path(file).stat().st_size}, 'Probe payload receipt differs: '+str(file))


def load_commit(file, binding):
    verify_bytes(file, binding)
    return storage.read(file)


def snapshot(directory):
    directory = Path(directory)
    result = {}
    for file in sorted(directory.rglob('*')):
        require(not file.is_symlink(), 'Link in probe output inventory')
        if file.is_file():
            result[file.relative_to(directory).as_posix()] = {'sha256': storage.digest(file),
                'bytes': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns}
    return result


def validate_protocol(root, config, protocol):
    require(protocol['schema'] == 18 and protocol['phase'] == 'synthetic_gpu_resources'
        and protocol['units'] == list(UNITS) and protocol['input_shape'] == [1, 5, 480, 640]
        and protocol['seed'] == 20261022 and protocol['input_seed'] == 18001
        and protocol['fp32_gate_strength'] == 0 and protocol['amp_gate_strength'] == 1
        and protocol['optimizer_updates'] == protocol['real_frames'] == 0, 'Fixed probe design changed')
    require(protocol['runtime'] == RUNTIME and protocol['interpreter'] == 'runtime/envs/vision/Scripts/python.exe'
        and protocol['driver_interpreter'] == 'runtime/envs/grasp_eval/Scripts/python.exe',
        'Probe requires the pinned vision runtime')
    limits = protocol['resources']
    require(limits == {'cpu_threads': 4, 'worker_timeout_seconds': 180, 'invocation_timeout_seconds': 1800,
        'rss_soft_bytes': 4*1024**3, 'gpu_allocated_soft_bytes': int(13.5*1024**3),
        'gpu_reserved_soft_bytes': int(13.5*1024**3), 'output_limit_bytes': 1024**3,
        'disk_reserve_bytes': 60*1024**3}, 'Probe resource contract changed')
    require(protocol['outputs'] == {'root': 'runtime/verification/dualstream_resources_v18',
        'report': 'results/P4/dualstream_resources_v18.json',
        'resume_report': 'results/P4/dualstream_resources_v18_resume.json'}, 'Probe output scope changed')
    require(protocol['source_hashes'], 'Probe source freeze is required before execution')
    require(protocol.get('status') == 'frozen', 'Probe protocol must be explicitly frozen')
    for name in ('scripts/probe_dualstream_resources_v18.py', 'src/segmentation_dualstream_v18.py',
                 'scripts/multiview_common_v14r1.py'):
        require(protocol['source_hashes'].get(name) == storage.digest(safe(root, name)), 'Probe source is not frozen: '+name)
    for name, checksum in protocol['source_hashes'].items():
        require(isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum)
            and storage.digest(safe(root, name)) == checksum, 'Probe source changed: '+name)
    require(protocol['input_hashes'], 'Probe input freeze is required before execution')
    for name in (protocol['parent_protocol'], protocol['parent_acceptance_report'], protocol['interpreter'],
                 'runtime/envs/vision/pyvenv.cfg', protocol['driver_interpreter'], 'runtime/envs/grasp_eval/pyvenv.cfg'):
        require(protocol['input_hashes'].get(name) == storage.digest(safe(root, name)), 'Probe input is not frozen: '+name)
    for name, checksum in protocol['input_hashes'].items():
        require(isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum)
            and storage.digest(safe(root, name)) == checksum, 'Probe input changed: '+name)
    base = protocol['base_python']; base_file = Path(base['path'])
    require(base_file.is_absolute() and base_file.name.lower() == 'python.exe'
        and isinstance(base['sha256'], str) and re.fullmatch('[0-9a-f]{64}', base['sha256'])
        and storage.digest(base_file) == base['sha256'], 'Vision base interpreter must be frozen')
    for key in ('parent_protocol_sha256', 'parent_acceptance_sha256'):
        require(isinstance(protocol[key], str) and re.fullmatch('[0-9a-f]{64}', protocol[key]), 'Parent acceptance must be frozen')
    parent_config = safe(root, protocol['parent_protocol'])
    parent_report = safe(root, protocol['parent_acceptance_report'])
    require(storage.digest(parent_config) == protocol['parent_protocol_sha256']
        and storage.digest(parent_report) == protocol['parent_acceptance_sha256'], 'Parent AP acceptance changed')
    parent = storage.read(parent_report)
    require(parent['status'] == 'accepted_complete_development_mask_ap'
        and parent['binding']['protocol_sha256'] == protocol['parent_protocol_sha256'], 'Parent AP has not been accepted')
    required_budget_paths = {protocol['outputs']['root'], protocol['outputs']['report'],
        protocol['outputs']['report']+'.sha256.json', protocol['outputs']['resume_report']}
    require(isinstance(protocol['budget_paths'], list) and required_budget_paths <= set(protocol['budget_paths']),
            'Probe budget omits declared outputs')
    for name in protocol['budget_paths']: safe(root, name)
    body = {'schema': 18, 'phase': 'synthetic_gpu_resources', 'protocol_sha256': storage.digest(config),
        'source_hashes': protocol['source_hashes'], 'input_hashes': protocol['input_hashes'], 'base_python': base,
        'parent_protocol_sha256': protocol['parent_protocol_sha256'],
        'parent_acceptance_sha256': protocol['parent_acceptance_sha256'],
        'interpreter_sha256': storage.digest(safe(root, protocol['interpreter'])),
        'driver_interpreter_sha256': storage.digest(safe(root, protocol['driver_interpreter'])), 'units': list(UNITS)}
    return dict(body, fingerprint=storage.fingerprint(body))


class Guard:
    def __init__(self, root, protocol, worker=False):
        self.root, self.protocol, self.worker = Path(root), protocol, worker
        self.started = time.perf_counter()

    def measured(self):
        files = set()
        for name in self.protocol['budget_paths']:
            path = safe(self.root, name)
            if path.is_file(): files.add(path)
            elif path.exists(): files.update(file for file in path.rglob('*') if file.is_file())
        # Interrupted atomic publications outside the cache root also consume budget.
        for key in ('report', 'resume_report'):
            target = safe(self.root, self.protocol['outputs'][key])
            for base in (target, receipt(target)):
                files.update(file for file in base.parent.glob(base.name+'.*.tmp') if file.is_file())
        total = sum(file.stat().st_size for file in files)
        torch = sys.modules.get('torch')
        allocated = reserved = 0
        if self.worker and torch is not None and torch.cuda.is_initialized():
            allocated, reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
        return {'elapsed_seconds': time.perf_counter()-self.started, 'peak_rss_bytes': storage.peak_memory_bytes(),
                'peak_gpu_allocated_bytes': allocated, 'peak_gpu_reserved_bytes': reserved, 'output_bytes': total}

    def check(self, extra=0):
        value = self.measured(); limits = self.protocol['resources']
        require(type(extra) is int and extra >= 0, 'Invalid output byte reservation')
        cap = limits['worker_timeout_seconds' if self.worker else 'invocation_timeout_seconds']
        require(value['elapsed_seconds'] <= cap, 'Probe time budget exceeded')
        require(value['peak_rss_bytes'] <= limits['rss_soft_bytes'], 'Probe RSS budget exceeded')
        require(value['peak_gpu_allocated_bytes'] <= limits['gpu_allocated_soft_bytes']
            and value['peak_gpu_reserved_bytes'] <= limits['gpu_reserved_soft_bytes'], 'Probe GPU allocator budget exceeded')
        require(value['output_bytes']+extra <= limits['output_limit_bytes'], 'Probe output budget exceeded')
        require(shutil.disk_usage(self.root).free-extra >= limits['disk_reserve_bytes'], 'Probe disk reserve exhausted')
        return value


def unit_parts(unit):
    require(unit in UNITS, 'Unknown resource unit')
    return unit.rsplit('_', 1)


def intent_record(unit, index, binding):
    return {'schema': 18, 'unit': unit, 'attempt_index': index, 'binding_fingerprint': binding['fingerprint'],
            'planned_model_calls': 1, 'optimizer_updates': 0, 'real_frames': 0}


def attempt_rows(directory, unit, binding):
    parent = directory/unit/'attempts'
    rows = []
    for attempt in sorted(parent.iterdir()) if parent.exists() else []:
        require(attempt.is_dir() and re.fullmatch('[0-9]{6}', attempt.name), 'Unknown probe attempt directory')
        index = int(attempt.name)
        require(index == len(rows), 'Probe attempt indices are not contiguous')
        payload_names = {'intent.json', 'forward_started.json', 'forward_returned.json', 'heads.npz',
                         'result.json', 'error.json', 'execution.json'}
        allowed = payload_names | {name+'.sha256.json' for name in payload_names} | {'worker.log'}
        for file in attempt.iterdir():
            require(file.is_file() and not file.is_symlink(), 'Non-file in attempt evidence')
            if file.name not in allowed:
                match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', file.name)
                require(match is not None and match.group(1) in allowed, 'Unknown attempt evidence file')
            elif file.name in payload_names:
                verify_bytes(file, binding)
            elif file.name.endswith('.sha256.json'):
                pending = storage.read(file)
                require(set(pending) == {'schema', 'binding_fingerprint', 'sha256', 'bytes'}
                    and pending['schema'] == 18 and pending['binding_fingerprint'] == binding['fingerprint']
                    and isinstance(pending['sha256'], str) and re.fullmatch('[0-9a-f]{64}', pending['sha256'])
                    and type(pending['bytes']) is int and pending['bytes'] >= 0, 'Malformed pending payload receipt')
        intent_file = attempt/'intent.json'
        # The driver can lose power between its intent receipt and payload.
        if not intent_file.exists():
            if receipt(intent_file).exists():
                expected = intent_record(unit, index, binding)
                payload = storage.json_bytes(expected)
                require(storage.read(receipt(intent_file)) == {'schema': 18, 'binding_fingerprint': binding['fingerprint'],
                    'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}, 'Pending intent receipt differs')
            # mkdir or an incomplete atomic intent may precede any durable receipt.
            # Preserve it as unknown; it is never treated as a model result.
            require(all(file.name == receipt(intent_file).name
                or re.fullmatch(r'intent\.json(?:\.sha256\.json)?\.[0-9a-f]{32}\.tmp', file.name)
                for file in attempt.iterdir()), 'Unbound attempt contains model output')
        else:
            require(load_commit(intent_file, binding) == intent_record(unit, index, binding), 'Probe attempt intent differs')
        known, exact, successful, forward_returned = 0, None, None, False
        if (attempt/'forward_started.json').exists():
            require(intent_file.exists() and load_commit(attempt/'forward_started.json', binding) == {
                'schema': 18, 'unit': unit, 'attempt_index': index,
                'scope': 'Intent immediately before one model call; not proof that the call returned'},
                'Model-call intent differs')
        returned_file = attempt/'forward_returned.json'
        if returned_file.exists():
            require((attempt/'forward_started.json').exists() and load_commit(returned_file, binding) == {'schema': 18, 'unit': unit, 'attempt_index': index,
                'model_forward_returned': True, 'model_calls': 1}, 'Forward-return receipt differs')
            known, exact, forward_returned = 1, 1, True
        error_file = attempt/'error.json'
        if error_file.exists():
            error = load_commit(error_file, binding)
            require(error['schema'] == 18 and error['unit'] == unit and error['attempt_index'] == index
                and type(error['model_call_invoked']) is bool, 'Worker error accounting differs')
            error_calls = int(error['model_call_invoked'])
            require(not forward_returned or error_calls == 1, 'Failure contradicts returned forward')
            known, exact = max(known, error_calls), error_calls
        if (attempt/'result.json').exists():
            successful = load_commit(attempt/'result.json', binding)
            require(forward_returned and not error_file.exists(), 'Successful worker lacks a unique forward result')
            known, exact = 1, 1
        if (attempt/'execution.json').exists():
            execution = load_commit(attempt/'execution.json', binding)
            require(execution['schema'] == 18 and execution['unit'] == unit and execution['attempt_index'] == index
                and type(execution['timed_out']) is bool
                and (execution['returncode'] is None if execution['timed_out'] else type(execution['returncode']) is int)
                and type(execution['elapsed_seconds']) in (int, float) and math.isfinite(execution['elapsed_seconds'])
                and execution['elapsed_seconds'] >= 0, 'Driver execution accounting differs')
        rows.append({'directory': attempt, 'known_model_calls': known, 'exact_model_calls': exact,
                     'forward_returned': forward_returned, 'result': successful})
    return rows


def accounting(rows):
    unknown = sum(row['exact_model_calls'] is None for row in rows)
    lower = sum(row['known_model_calls'] for row in rows)
    return {'attempts': len(rows), 'model_calls_returned': sum(row['forward_returned'] for row in rows),
            'known_model_call_lower_bound': lower, 'unknown_model_call_attempts': unknown,
            'actual_model_calls_exact': None if unknown else lower}


def validate_result(result, unit, protocol, attempt, binding):
    arm, mode = unit_parts(unit)
    require(result['schema'] == 18 and result['status'] == 'completed_resource_unit' and result['unit'] == unit
        and result['arm'] == arm and result['mode'] == mode and result['runtime'] == protocol['runtime']
        and result['input_shape'] == protocol['input_shape'] and result['model_calls'] == 1
        and result['optimizer_updates'] == result['real_frames'] == 0
        and result['gate_strength'] == (0 if mode == 'fp32' else 1), 'Resource unit result identity differs')
    for key in ('initial_parameters_sha256', 'input_sha256'):
        require(isinstance(result[key], str) and re.fullmatch('[0-9a-f]{64}', result[key]), 'Missing numerical source digest')
    limits = protocol['resources']; resources = result['resources']
    require(type(result['hardware_gpu_total_memory_bytes']) is int and result['hardware_gpu_total_memory_bytes'] > 0,
            'GPU memory capacity was not recorded')
    for key, cap in (('elapsed_seconds', limits['worker_timeout_seconds']), ('peak_rss_bytes', limits['rss_soft_bytes']),
        ('peak_gpu_allocated_bytes', limits['gpu_allocated_soft_bytes']),
        ('peak_gpu_reserved_bytes', limits['gpu_reserved_soft_bytes']), ('output_bytes', limits['output_limit_bytes'])):
        require(type(resources[key]) in (int, float) and math.isfinite(resources[key]) and 0 <= resources[key] <= cap,
                'Successful resource unit exceeded its fixed budget')
    if mode == 'fp32':
        require(result['finite_heads'] is True and result['backward_calls'] == 0, 'FP32 probe performed unexpected work')
        file = attempt/'heads.npz'; verify_bytes(file, binding)
        require(result['heads_sha256'] == storage.digest(file), 'FP32 output digest differs')
    else:
        require(result['backward_calls'] == 1 and result['all_observed_gradients_finite'] is True
            and result['gradient_scaling_used'] is False and type(result['loss']) in (int, float)
            and math.isfinite(result['loss']), 'AMP numerical smoke test failed')
        require(set(result['nonzero_gradient_elements']) == {'rgb_encoder', 'depth_encoder', 'quality_heads'},
                'Missing required gradient groups')
        require(all(type(value) is int and value > 0 for value in result['nonzero_gradient_elements'].values()),
                'Depth or quality branch has no observed gradient')
    return result


def adopt_success(directory, unit, protocol, binding, readonly=False):
    rows = attempt_rows(directory, unit, binding)
    successful = [row for row in rows if row['result'] is not None]
    require(len(successful) <= 1, 'Multiple successful attempts for one resource unit')
    unit_file = directory/unit/'unit.json'
    if not successful:
        require(not unit_file.exists(), 'Unit receipt has no successful worker')
        return None
    row = successful[0]
    result = validate_result(row['result'], unit, protocol, row['directory'], binding)
    value = {'schema': 18, 'status': 'completed', 'unit': unit,
        'attempt': row['directory'].relative_to(directory).as_posix(),
        'result_sha256': storage.digest(row['directory']/'result.json')}
    commit(unit_file, value, binding, readonly)
    return {'completion': value, 'result': result, 'attempt_accounting': accounting(rows)}


def runtime_identity(torch):
    return {'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': importlib.metadata.version('torchvision'), 'numpy': importlib.metadata.version('numpy'),
        'scipy': importlib.metadata.version('scipy'), 'Pillow': importlib.metadata.version('Pillow'),
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def synthetic_input(torch, protocol):
    shape = protocol['input_shape']; generator = torch.Generator(device='cpu').manual_seed(protocol['input_seed'])
    rgb = torch.randn((shape[0], 3, shape[2], shape[3]), generator=generator)
    depth = torch.rand((shape[0], 1, shape[2], shape[3]), generator=generator)
    valid = (torch.rand(depth.shape, generator=generator) > .15).float()
    return torch.cat((rgb, depth*valid, valid), dim=1)


def worker(root, config, unit, index):
    root, config = Path(root).resolve(), safe(root, config)
    protocol = storage.read(config); binding = validate_protocol(root, config, protocol)
    require(Path(sys.executable).resolve() == safe(root, protocol['interpreter']), 'Wrong probe worker interpreter')
    require(Path(sys._base_executable).resolve() == Path(protocol['base_python']['path']).resolve(), 'Wrong vision base interpreter')
    directory = safe(root, protocol['outputs']['root']); attempt = directory/unit/'attempts'/('%06d' % index)
    require(load_commit(attempt/'intent.json', binding) == intent_record(unit, index, binding), 'Worker has no durable driver intent')
    require(not (attempt/'result.json').exists() and not (attempt/'error.json').exists(), 'Attempt is already concluded')
    guard = Guard(root, protocol, worker=True); invoked = False
    try:
        import torch
        import numpy as np
        from src.segmentation_dualstream_v18 import make_paired_models
        torch.set_num_threads(4); torch.set_num_interop_threads(4)
        require(torch.cuda.is_available(), 'CUDA is required for the synthetic resource worker')
        require(runtime_identity(torch) == protocol['runtime'], 'Pinned worker runtime differs')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        with storage.lock(root/'runtime/grasp_foreground_v12/generation_gpu_lock'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.set_per_process_memory_fraction(min(1.0, protocol['resources']['gpu_reserved_soft_bytes']/torch.cuda.get_device_properties(0).total_memory))
            arm, mode = unit_parts(unit)
            fixed, quality = make_paired_models(protocol['seed'])
            model = fixed if arm == 'fixed' else quality
            del fixed, quality
            initial = hashlib.sha256()
            for name, tensor in sorted(model.state_dict().items()):
                initial.update(name.encode('utf-8')); initial.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            inputs = synthetic_input(torch, protocol)
            input_sha = tensor_digest(inputs)
            model = model.to('cuda'); inputs = inputs.to('cuda')
            model.train(mode == 'amp'); model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); guard.check()
            commit(attempt/'forward_started.json', {'schema': 18, 'unit': unit, 'attempt_index': index,
                'scope': 'Intent immediately before one model call; not proof that the call returned'}, binding)
            strength = 0 if mode == 'fp32' else 1
            invoked = True
            if mode == 'fp32':
                with torch.no_grad():
                    outputs = model(inputs, gate_strength=strength)
            else:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs, gate_strength=strength)
            torch.cuda.synchronize()
            commit(attempt/'forward_returned.json', {'schema': 18, 'unit': unit, 'attempt_index': index,
                'model_forward_returned': True, 'model_calls': 1}, binding)
            guard.check()
            result = {'schema': 18, 'status': 'completed_resource_unit', 'unit': unit, 'arm': arm, 'mode': mode,
                'runtime': runtime_identity(torch), 'input_shape': protocol['input_shape'], 'model_calls': 1,
                'hardware_gpu_total_memory_bytes': int(torch.cuda.get_device_properties(0).total_memory),
                'optimizer_updates': 0, 'real_frames': 0, 'gate_strength': strength,
                'initial_parameters_sha256': initial.hexdigest(), 'input_sha256': input_sha}
            if mode == 'fp32':
                arrays = {name: outputs[name].detach().cpu().numpy() for name in HEADS}
                require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays.values()), 'Nonfinite FP32 head')
                payload = io.BytesIO(); np.savez_compressed(payload, **arrays); data = payload.getvalue()
                guard.check(len(data)+8192); commit_bytes(attempt/'heads.npz', data, binding)
                result.update(finite_heads=True, backward_calls=0, heads_sha256=storage.digest(attempt/'heads.npz'))
            else:
                # These fixed smoke targets have no sensor-quality or research meaning.
                loss = sum((outputs[name].float()-.2).square().mean() for name in HEADS)
                loss = loss+sum((q.float()-.25).square().mean() for q in outputs['quality_maps'])
                require(bool(torch.isfinite(loss)), 'Nonfinite synthetic AMP loss')
                loss.backward(); torch.cuda.synchronize()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'Nonfinite AMP gradient')
                groups = {'rgb_encoder': model.rgb_encoder, 'depth_encoder': model.depth_encoder, 'quality_heads': model.quality_heads}
                counts = {name: sum(int(torch.count_nonzero(p.grad)) for p in component.parameters() if p.grad is not None)
                          for name, component in groups.items()}
                require(all(value > 0 for value in counts.values()), 'Required AMP gradient group is zero')
                result.update(backward_calls=1, loss=float(loss.detach().cpu()), gradient_scaling_used=False,
                    all_observed_gradients_finite=True, nonzero_gradient_elements=counts)
            result['resources'] = guard.check(16384)
            commit(attempt/'result.json', result, binding)
            return result
    except Exception as error:
        record = {'schema': 18, 'unit': unit, 'attempt_index': index, 'model_call_invoked': invoked,
                  'error_type': type(error).__name__, 'message': str(error)}
        try:
            record['resources_at_failure'] = guard.measured()
            commit(attempt/'error.json', record, binding)
        except Exception:
            pass  # A missing outcome stays explicitly unknown on the next invocation.
        raise


def launch_worker(root, config, protocol, unit, index, attempt, binding, guard):
    environment = os.environ.copy()
    environment.update({'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4'})
    environment.pop('PYTHONPATH', None)
    command = [str(safe(root, protocol['interpreter'])), '-B', str(root/'scripts/probe_dualstream_resources_v18.py'),
        '--config', str(config), '--internal-worker', '--unit', unit, '--attempt-index', str(index)]
    started = time.perf_counter(); timed_out = False; code = None
    with (attempt/'worker.log').open('xb', buffering=0) as log:
        log.write(('COMMAND '+json.dumps(command, ensure_ascii=False)+'\n').encode('utf-8'))
        try:
            completed = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=protocol['resources']['worker_timeout_seconds'])
            code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        os.fsync(log.fileno())
    record = {'schema': 18, 'unit': unit, 'attempt_index': index, 'returncode': code,
              'timed_out': timed_out, 'elapsed_seconds': time.perf_counter()-started}
    commit(attempt/'execution.json', record, binding)
    guard.check()
    require(code == 0 and not timed_out, 'Resource worker stopped; retain attempt and inspect '+str(attempt/'worker.log'))
    return record


def compare_fp32(directory, units, binding):
    import numpy as np
    paths = {name: directory/units[name]['completion']['attempt']/'heads.npz' for name in ('fixed_fp32', 'quality_fp32')}
    for file in paths.values(): verify_bytes(file, binding)
    comparisons = {}
    with np.load(paths['fixed_fp32'], allow_pickle=False) as first, np.load(paths['quality_fp32'], allow_pickle=False) as second:
        require(set(first.files) == set(second.files) == set(HEADS), 'Unexpected FP32 head inventory')
        for name in HEADS:
            left, right = first[name], second[name]
            shape = (1, 2 if name == 'offsets' else 1, 480, 640)
            require(left.shape == right.shape == shape and left.dtype == right.dtype == np.float32
                and np.isfinite(left).all() and np.isfinite(right).all(), 'FP32 head geometry/dtype differs')
            require(left.tobytes() == right.tobytes(), 'Initial paired FP32 heads are not bitwise identical')
            comparisons[name] = {'shape': list(shape), 'bitwise_equal': True, 'maximum_absolute_difference': 0.0}
    return comparisons


def final_report(directory, units, protocol, binding):
    require(set(units) == set(UNITS), 'Probe has incomplete unit support')
    initial = {row['result']['initial_parameters_sha256'] for row in units.values()}
    inputs = {row['result']['input_sha256'] for row in units.values()}
    require(len(initial) == len(inputs) == 1, 'Worker initialization or synthetic input differs')
    rows = [row for unit in UNITS for row in attempt_rows(directory, unit, binding)]
    return {'schema': 18, 'status': 'passed_synthetic_gpu_resource_probe', 'binding_fingerprint': binding['fingerprint'],
        'units': units, 'fp32_initial_heads': compare_fp32(directory, units, binding),
        'attempt_accounting': accounting(rows), 'nominal_model_calls': 4, 'optimizer_updates': 0, 'real_frames': 0,
        'successful_unit_resource_peaks': {key: max(row['result']['resources'][key] for row in units.values())
            for key in ('peak_rss_bytes', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes')},
        'is_research_training': False, 'is_training_resume_verification': False,
        'quality_supervision_selected': False, 'research_effectiveness_established': False,
        'limitations': protocol['limitations']}


def output_snapshot(root, protocol):
    directory = safe(root, protocol['outputs']['root'])
    result = {'root/'+name: row for name, row in snapshot(directory).items()}
    for path in external_report_files(root, protocol):
        result[path.relative_to(root).as_posix()] = {'sha256': storage.digest(path),
            'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    return result


def external_report_files(root, protocol):
    target = safe(root, protocol['outputs']['report']); files = set()
    for base in (target, receipt(target)):
        if base.is_file(): files.add(base)
        for file in base.parent.glob(base.name+'.*.tmp'):
            require(file.is_file() and not file.is_symlink()
                and re.fullmatch(re.escape(base.name)+r'\.[0-9a-f]{32}\.tmp', file.name), 'Unknown external report temporary')
            files.add(file)
    require(all(not file.is_symlink() for file in files), 'Linked external report artifact')
    return sorted(files)


def validate_layout(directory, binding):
    root_allowed = {'binding.json', 'binding.json.sha256.json', 'completion.json', 'completion.json.sha256.json'}
    for file in directory.iterdir() if directory.exists() else []:
        require(not file.is_symlink(), 'Link in probe layout')
        if file.name in UNITS:
            require(file.is_dir(), 'Resource unit must be a directory')
        else:
            match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', file.name)
            require(file.is_file() and (file.name in root_allowed or (match and match.group(1) in root_allowed)),
                    'Unexpected top-level probe artifact')
    missing = False
    for unit in UNITS:
        target = directory/unit
        require(not missing or not target.exists(), 'Resource units are not a contiguous prefix')
        if target.exists():
            allowed = {'unit.json', 'unit.json.sha256.json'}
            for file in target.iterdir():
                require(not file.is_symlink(), 'Link in resource unit')
                if file.name == 'attempts':
                    require(file.is_dir(), 'Attempt container must be a directory')
                else:
                    match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', file.name)
                    require(file.is_file() and (file.name in allowed or (match and match.group(1) in allowed)),
                            'Unexpected resource unit artifact')
            attempt_rows(directory, unit, binding)
        missing = missing or not (target/'unit.json').exists()


def execute(root, config, protocol, binding, max_new_units=None, check_only=False):
    directory = safe(root, protocol['outputs']['root']); guard = Guard(root, protocol); guard.check()
    report_file = safe(root, protocol['outputs']['report'])
    complete = (directory/'completion.json').exists(); readonly = check_only or complete
    require(not check_only or complete, 'Read-only check requires a complete resource probe')
    validate_layout(directory, binding)
    before = output_snapshot(root, protocol) if readonly else None
    commit(directory/'binding.json', binding, binding, readonly)
    units = {}; new_units = new_calls = 0
    for unit in UNITS:
        had_completion = (directory/unit/'unit.json').exists()
        if not had_completion:
            require(not readonly, 'Completed probe is missing a unit')
            if max_new_units is not None and new_units >= max_new_units:
                break
        saved = adopt_success(directory, unit, protocol, binding, readonly)
        if saved is None:
            require(not readonly, 'Completed probe is missing a unit')
            rows = attempt_rows(directory, unit, binding); index = len(rows)
            attempt = directory/unit/'attempts'/('%06d' % index)
            guard.check(32768); commit(attempt/'intent.json', intent_record(unit, index, binding), binding)
            launch_worker(root, config, protocol, unit, index, attempt, binding, guard)
            new_calls += 1
            saved = adopt_success(directory, unit, protocol, binding)
            require(saved is not None, 'Worker returned without a committed resource result')
        elif not (directory/unit/'unit.json').exists():
            raise RuntimeError('Missing durable unit completion')
        if not had_completion: new_units += 1
        units[unit] = saved
        guard.check()
    work = {'schema': 18, 'new_units': new_units, 'reused_units': len(units)-new_units, 'committed_units': len(units),
            'expected_units': 4, 'new_model_forward_calls': new_calls, 'optimizer_updates': 0, 'real_frames': 0}
    if len(units) != 4:
        return dict(work, status='partial', resources=guard.check())
    report = final_report(directory, units, protocol, binding)
    guard.check(65536); commit(report_file, report, binding, readonly)
    files = snapshot(directory)
    completion = {'schema': 18, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
        'outputs': {name: row['sha256'] for name, row in files.items()
            if not re.fullmatch(r'completion\.json(?:\.sha256\.json)?(?:\.[0-9a-f]{32}\.tmp)?', name)},
        'external_outputs': {file.relative_to(root).as_posix(): storage.digest(file) for file in external_report_files(root, protocol)},
        'report_sha256': storage.digest(report_file)}
    guard.check(0 if readonly else len(storage.json_bytes(completion))+8192)
    commit(directory/'completion.json', completion, binding, readonly)
    if readonly: require(before == output_snapshot(root, protocol), 'Read-only resource check changed SHA, size or mtime')
    return dict(work, status='verified_completed_read_only' if readonly else report['status'],
        report_sha256=storage.digest(report_file), attempt_accounting=report['attempt_accounting'], resources=guard.check())


def run(root=ROOT, config=None, max_new_units=None, check_only=False):
    root = Path(root).resolve(); config = safe(root, config or CONFIG)
    require(max_new_units is None or type(max_new_units) is int and 0 <= max_new_units <= 4, 'Invalid new-unit limit')
    require(not check_only or max_new_units is None, 'Read-only cannot request a work limit')
    protocol = storage.read(config); binding = validate_protocol(root, config, protocol)
    require(Path(sys.executable).resolve() == safe(root, protocol['driver_interpreter']), 'Wrong CPU probe driver interpreter')
    directory = safe(root, protocol['outputs']['root'])
    if check_only or (directory/'completion.json').exists():
        return execute(root, config, protocol, binding, max_new_units, check_only)
    with storage.lock(root/'runtime/locks/dualstream_resources_v18'):
        return execute(root, config, protocol, binding, max_new_units, check_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/CONFIG)
    parser.add_argument('--max-new-units', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--internal-worker', action='store_true')
    parser.add_argument('--unit', choices=UNITS)
    parser.add_argument('--attempt-index', type=int)
    args = parser.parse_args()
    try:
        if args.internal_worker:
            require(args.unit is not None and type(args.attempt_index) is int and args.attempt_index >= 0
                and not args.check_only and args.max_new_units is None, 'Invalid worker invocation')
            result = worker(ROOT, args.config, args.unit, args.attempt_index)
        else:
            require(args.unit is None and args.attempt_index is None, 'Worker identity outside internal mode')
            result = run(config=args.config, max_new_units=args.max_new_units, check_only=args.check_only)
        if not args.internal_worker:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        print('Dual-stream resource probe stopped: '+str(error), file=sys.stderr, flush=True)
        sys.exit(2)
