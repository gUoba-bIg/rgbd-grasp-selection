"""Prepare tiny synthetic CPU oracle fixtures; no formal diagnostic is executed."""
import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
BASE = 'logs/P4/center_response_diagnostics_v20/preparation/reference_core'
DRAFT = 'docs/第二十轮中心响应诊断草案.txt'
DRAFT_SHA = '859b8a1ae1317255da3567aee834ede9862820499afe5d5dbef2cbd63e708357'
SOURCE = 'scripts/build_center_response_fixtures_v20.py'
LIMIT_BYTES = 1024**3


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')
    with Path(path).open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())


def cases():
    result = []
    def add(name, height, width, objects=(), output_hw=None):
        labels = [[0]*width for _ in range(height)]
        for identity, points in objects:
            for row, col in points:
                if labels[row][col]: raise ValueError('Fixture objects overlap')
                labels[row][col] = identity
        result.append({'name': name, 'original_labels': labels, 'output_hw': list(output_hw or (height, width))})
    add('empty', 7, 9)
    add('one_pixel', 1, 1, [(1, [(0, 0)])])
    add('all_foreground', 7, 9, [(3, [(y, x) for y in range(7) for x in range(9)])])
    add('half_even', 9, 13, [(1, [(2, 2), (3, 3)]), (2, [(4, 7), (5, 8)]), (3, [(3, 10), (4, 11)])])
    add('boundary', 9, 11, [(1, [(0, 0), (0, 1)]), (88, [(8, 10)])])
    add('concave', 9, 9, [(4, [(y, x) for y in range(1, 8) for x in range(1, 8) if x in (1, 7) or y == 7])])
    add('disconnected', 9, 9, [(5, [(1, 1), (7, 7)])])
    add('coincident_centers', 9, 9, [(1, [(2, 4), (6, 4)]), (2, [(4, 2), (4, 6)])])
    add('overlapping_gaussians', 11, 13, [(1, [(4, 4)]), (2, [(4, 6)])])
    add('areas_19_and_20', 8, 10, [(1, [(i//10, i%10) for i in range(19)]), (2, [(i//10, i%10) for i in range(30, 50)])])
    add('resize_disappeared', 8, 10, [(1, [(0, 0)]), (2, [(y, x) for y in range(2, 6) for x in range(2, 8)])], (4, 5))
    add('resize_odd_grid', 9, 13, [(1, [(0, 0), (8, 12)]), (2, [(y, x) for y in range(2, 7) for x in range(4, 9)])], (4, 6))
    add('offset_non_square', 5, 17, [(7, [(0, 0), (4, 16), (2, 8)]), (8, [(1, 14)])])
    return result


def file_record(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def runtime_identity():
    versions = {}
    for name in ('numpy', 'Pillow', 'scipy', 'torch'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {'python': sys.version, 'executable': file_record(sys.executable),
        'base_executable': file_record(sys._base_executable), 'pyvenv': file_record(Path(sys.prefix)/'pyvenv.cfg'),
        'package_versions': versions, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}


def peak_rss():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[(name, ctypes.c_size_t) for name in
                ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        value = Counters(); value.cb = ctypes.sizeof(value)
        process = ctypes.windll.kernel32.GetCurrentProcess; process.restype = wintypes.HANDLE
        get = ctypes.windll.psapi.GetProcessMemoryInfo; get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not get(process(), ctypes.byref(value), value.cb): raise OSError('Cannot measure preparation peak RSS')
        return int(value.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def output_directory(value):
    folder = Path(value).resolve(); base = (ROOT/BASE).resolve()
    if not folder.is_relative_to(base) or folder == base: raise ValueError('Preparation output must be a unique child of reference_core')
    return folder


def oracle(folder):
    started = time.perf_counter(); folder = output_directory(folder)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Oracle requires an empty visible-device setting')
    if Path(sys.executable).resolve() != (ROOT/'runtime/envs/vision/Scripts/python.exe').resolve(): raise ValueError('Use the existing vision CPU interpreter')
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from src.graspnet_frames import instance_targets
    declared = json.loads((folder/'inputs.json').read_text(encoding='utf-8'))
    if declared != {'schema': 20, 'scope': 'fixed_small_synthetic_labels_only', 'cases': cases()}: raise ValueError('Oracle input differs from fixed artificial cases')
    rows = []
    for case in declared['cases']:
        original = np.asarray(case['original_labels'], dtype=np.uint16)
        h, w = case['output_hw']
        resized = np.asarray(Image.fromarray(original).resize((w, h), Image.Resampling.NEAREST), dtype=np.int64)
        tensors = instance_targets(resized)
        if any(tensor.device.type != 'cpu' for tensor in tensors.values()): raise ValueError('Oracle tensor is not CPU')
        arrays = {name: tensor.numpy() for name, tensor in tensors.items()}
        output = folder/(case['name']+'.npz')
        with output.open('xb') as stream: np.savez(stream, original_labels=original, labels_input=resized, **arrays)
        rows.append({'name': case['name'], 'output_hw': case['output_hw'], 'artifact': file_record(output),
            'arrays': {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                'raw_sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                for name, value in dict(original_labels=original, labels_input=resized, **arrays).items()}})
    measured = peak_rss(); elapsed = time.perf_counter()-started
    if measured > LIMIT_BYTES or elapsed > 120: raise RuntimeError('Preparation oracle resource limit exceeded')
    result = {'schema': 20, 'status': 'completed_preparation_oracle_not_formal_diagnostic', 'formal_contract_frozen': False,
        'runtime': runtime_identity(), 'torch_cpu_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'function_source_sha256': hashlib.sha256(inspect.getsource(instance_targets).encode('utf-8')).hexdigest(),
        'repository_source_dependencies': [file_record(ROOT/name) for name in ('src/graspnet_frames.py', 'src/checkpoints.py', SOURCE)],
        'input': file_record(folder/'inputs.json'), 'cases': rows,
        'resources': {'elapsed_seconds': elapsed, 'peak_rss_bytes': measured, 'limit_seconds': 120, 'limit_rss_bytes': LIMIT_BYTES},
        'new_model_forward_calls': 0, 'new_training_updates': 0, 'new_backward_calls': 0, 'new_official_evaluations': 0,
        'new_random_draws': 0, 'real_data_files_opened': 0, 'gpu_interfaces_called': False}
    publish(folder/'oracle.json', result)
    print(json.dumps({'status': result['status'], 'cases': len(rows), 'oracle_sha256': digest(folder/'oracle.json')}), flush=True)


def build():
    if Path(sys.executable).resolve() != (ROOT/'runtime/envs/grasp_eval/Scripts/python.exe').resolve(): raise ValueError('Use the existing grasp CPU interpreter')
    if digest(ROOT/DRAFT) != DRAFT_SHA: raise ValueError('Draft changed; review before preparing fixtures')
    if 'torch' in sys.modules: raise ValueError('Parent preparation process must not import Torch')
    bound_paths = [ROOT/name for name in (DRAFT, SOURCE, 'src/graspnet_frames.py', 'src/checkpoints.py')]
    bound_paths += [ROOT/'runtime/envs'/name/'Scripts/python.exe' for name in ('vision', 'grasp_eval')]
    bound_before = [file_record(path) for path in bound_paths]
    folder = ROOT/BASE/(datetime.datetime.now(datetime.timezone.utc).strftime('oracle_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    publish(folder/'inputs.json', {'schema': 20, 'scope': 'fixed_small_synthetic_labels_only', 'cases': cases()})
    command = [str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-B', str(ROOT/SOURCE), '--oracle-directory', str(folder)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[name] = '1'
    publish(folder/'preparation.json', {'schema': 20, 'status': 'preparation_only', 'formal_contract_frozen': False,
        'draft': file_record(ROOT/DRAFT), 'generator': file_record(ROOT/SOURCE), 'parent_runtime': runtime_identity(), 'command': command,
        'source_and_interpreter_before': bound_before,
        'resource_limits': {'timeout_seconds': 120, 'rss_limit_bytes': LIMIT_BYTES, 'cpu_threads': 1}})
    started = time.perf_counter(); outcome = {'schema': 20, 'returncode': None, 'exception': None}
    try:
        with (folder/'oracle.stdout.log').open('xb') as stdout, (folder/'oracle.stderr.log').open('xb') as stderr:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, timeout=120)
        outcome['returncode'] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        outcome['exception'] = type(error).__name__+': '+str(error)
    outcome['observed_parent_wait_seconds'] = time.perf_counter()-started
    publish(folder/'process_outcome.json', outcome)
    if outcome['exception'] is not None or outcome['returncode'] != 0: raise RuntimeError('Oracle failed; evidence retained in '+str(folder))
    bound_after = [file_record(path) for path in bound_paths]
    if bound_after != bound_before: raise RuntimeError('Source or interpreter changed during preparation; evidence retained')
    value = json.loads((folder/'oracle.json').read_text(encoding='utf-8'))
    publish(folder/'manifest.json', {'schema': 20, 'status': 'prepared_fixtures_not_formal_contract', 'formal_contract_frozen': False,
        'case_count': len(value['cases']), 'artifacts': [file_record(path) for path in sorted(folder.iterdir()) if path.is_file()],
        'source_and_interpreter_before': bound_before, 'source_and_interpreter_after': bound_after,
        'source_and_runtime_hashes_are_preparation_evidence': True})
    print(json.dumps({'status': 'prepared_fixtures_not_formal_contract', 'directory': str(folder),
        'manifest_sha256': digest(folder/'manifest.json'), 'case_count': len(value['cases'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--oracle-directory')
    args = parser.parse_args()
    if args.oracle_directory: oracle(args.oracle_directory)
    else: build()
