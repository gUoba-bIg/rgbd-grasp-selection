"""Immutable storage and label-separated bindings for local observed risk."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import multiview_common_v14r1 as storage
from scripts.multiview_common_v14r1 import (require, read, digest, json_bytes, fingerprint,
    safe_path, atomic_bytes, atomic, publish_once, once, receipt_path, binding_id,
    lock, runtime_identity, peak_memory_bytes, snapshot, Guard)

CONFIG = ROOT / 'configs/grasp_local_risk_v15.json'
FRAMES = ('0016', '0080', '0160', '0240')
SCORE_NAMES = ('global', 'left_finger', 'right_finger', 'bottom', 'approach')
NUMERICAL_NAMES = ('numpy', 'scipy', 'open3d', 'PIL', 'psutil', 'threadpoolctl.py')
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'


def commit_bytes(path, payload, binding, read_only=False):
    receipt = {'schema': 15, 'binding_fingerprint': binding_id(binding),
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    once(receipt_path(path), receipt, read_only)
    publish_once(path, payload, read_only)
    return receipt


def commit(path, value, binding, read_only=False):
    return commit_bytes(path, json_bytes(value), binding, read_only)


def verify_bytes(path, binding):
    path = Path(path)
    receipt = read(receipt_path(path))
    require(receipt == {'schema': 15, 'binding_fingerprint': binding_id(binding),
                        'sha256': digest(path), 'bytes': path.stat().st_size}, 'Commit receipt differs: ' + str(path))
    return receipt


def load_commit(path, binding):
    verify_bytes(path, binding)
    return read(path)


def frame_directory(root, protocol, frame, phase='features'):
    require(frame in protocol['frames'] and phase == 'features', 'Unexpected feature frame/phase')
    return safe_path(root, protocol['outputs']['features_root']) / frame


def validate_protocol(protocol):
    require(protocol['schema'] == 15 and protocol['frames'] == list(FRAMES), 'Wrong frozen frame selection')
    expected = {'block_size': 8, 'voxel_size_m': .01, 'collision_threshold': .01,
                'approach_distance_m': .05, 'empty_threshold': .01, 'maximum_depth_m': 2.0,
                'cpu_threads': 2, 'scene_id': 3, 'camera': 'realsense'}
    require(all(protocol.get(key) == value for key, value in expected.items()), 'Frozen local-risk policy changed')
    require(protocol['metadata_fields_read'] == ['intrinsic_matrix', 'factor_depth']
            and protocol['empty_is_rejection'] is False and protocol['efficacy_gate'] is False,
            'Diagnostic meaning changed')
    require(set(protocol['feature_inputs']) == set(FRAMES) and set(protocol['frame_inputs']) == set(FRAMES),
            'Frame input membership changed')
    for frame in FRAMES:
        paths = protocol['frame_inputs'][frame]
        require(set(paths) == {'depth', 'meta', 'generation_dir'}, 'Feature input fields changed')
        base = 'data/raw/graspnet_coverage_v8/scenes/scene_0003/realsense/'
        require(paths['depth'] == base + 'depth/' + frame + '.png'
                and paths['meta'] == base + 'meta/' + frame + '.mat', 'Raw frame identity changed')
        require(paths['generation_dir'] == 'results/P3/grasp_multiview_v14r1/frames/' + frame + '/generation',
                'Unexpected parent generation')
        files = protocol['feature_inputs'][frame]
        require(paths['depth'] in files and paths['meta'] in files, 'Missing raw calibration input')
        require(all(name in (paths['depth'], paths['meta']) or name.startswith(paths['generation_dir'] + '/')
                    for name in files), 'Label/foreign data in feature input map')
    require(all(name.startswith('scripts/') or name.startswith('tests/') or name.startswith('src/')
                or name.startswith('docs/') or name.startswith('logs/P3/local_risk_tests_v15')
                or name == 'runtime/build/graspnet/utils/collision_detector.py'
                for name in protocol['source_hashes']), 'Unexpected source path')
    allowed_shared = {'configs/grasp_multiview_v14r1.json', 'configs/grasp_observed_collision_v9.json',
        'results/P2/grasp_geometry_v7.json', 'results/P2/grasp_eval_environment_v7.json',
        'results/P2/grasp_environment_acceptance_v7.json'}
    require(set(protocol['shared_inputs']).issubset(allowed_shared), 'Label-bearing shared input')
    require(protocol['outputs'] == {
        'features_root': 'results/P3/grasp_local_risk_v15/features',
        'features_resume_report': 'results/P3/grasp_local_risk_v15_features_resume.json',
        'evaluation_report': 'results/P3/grasp_local_risk_evaluation_v15.json',
        'acceptance_report': 'results/P3/grasp_local_risk_acceptance_v15.json',
        'visuals_dir': 'results/P3/grasp_local_risk_visuals_v15'}, 'Output location contract changed')
    for phase in ('features', 'evaluation', 'acceptance'):
        require(protocol['resources'][phase] == {'cpu_threads': 2, 'invocation_soft_seconds': 300,
                'rss_soft_bytes': 2 * 1024**3, 'output_limit_bytes': 512 * 1024**2,
                'disk_reserve_bytes': 60 * 1024**3}, 'Resource contract changed')


def numerical_inventory(root):
    root = Path(root)
    site = root / 'runtime/envs/grasp_eval/Lib/site-packages'
    api = site / 'graspnetAPI'
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(api.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    for name in NUMERICAL_NAMES:
        target = site / name
        for path in ([target] if target.is_file() else target.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.pyd', '.dll'):
                files[path.relative_to(root).as_posix()] = digest(path)
    binary = site / 'grasp_nms.cp312-win_amd64.pyd'
    require(binary.is_file(), 'Pinned API extension missing')
    files[binary.relative_to(root).as_posix()] = digest(binary)
    return files


def verify_runtime(root, protocol):
    interpreter = safe_path(root, protocol['interpreter'])
    require(Path(sys.prefix).resolve() == interpreter.parent.parent.resolve(), 'Wrong isolated CPU interpreter')
    require(runtime_identity() == protocol['runtime'], 'CPU numerical runtime changed')
    require('torch' not in sys.modules, 'Model runtime must not be loaded')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    require(packages == protocol['expected_packages'], 'CPU package inventory changed')
    numerical = numerical_inventory(root)
    require(numerical == protocol['numerical_inputs'], 'CPU numerical source inventory changed')
    return numerical


def make_binding(root, config, protocol, phase='features', frame=None, extra_inputs=None):
    validate_protocol(protocol)
    require(phase in ('features', 'evaluation', 'acceptance'), 'Unknown binding phase')
    if phase == 'features':
        require(frame in FRAMES, 'Unknown binding frame')
        files = dict(protocol['feature_inputs'][frame])
    else:
        require(frame is None, 'Evaluation is jointly bound to all frames')
        files = dict(protocol['evaluation_inputs'])
        for item in protocol['feature_inputs'].values():
            files.update(item)
    for group in ('shared_inputs', 'source_hashes', 'numerical_inputs'):
        for name, sha in protocol[group].items():
            require(name not in files or files[name] == sha, 'Conflicting source binding')
            files[name] = sha
    for name, sha in (extra_inputs or {}).items():
        require(name not in files or files[name] == sha, 'Conflicting dynamic binding')
        files[name] = sha
    if phase == 'features':
        require(not extra_inputs, 'Feature binding cannot add label dependencies')
    numerical = verify_runtime(root, protocol) or {}
    for name, sha in files.items():
        observed = numerical[name] if name in numerical else digest(safe_path(root, name))
        require(observed == sha, 'Frozen input/source changed: ' + name)
    result = {'schema': 15, 'phase': phase, 'frame': frame, 'protocol_sha256': digest(config),
              'files': files, 'runtime': protocol['runtime'], 'interpreter_sha256': digest(sys.executable),
              'gpu_used': False, 'new_training_steps': 0, 'new_model_forward_calls': 0,
              'new_official_geometry_computations': 0}
    result['fingerprint'] = fingerprint(result)
    return result


def verify_frozen_tree(root, directory, files):
    directory = Path(directory).resolve()
    expected = {name: sha for name, sha in files.items() if safe_path(root, name).is_relative_to(directory)}
    require(bool(expected), 'No pinned parent inventory')
    actual = {}
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'Link in frozen parent inventory')
        if path.is_file():
            actual[path.relative_to(Path(root).resolve()).as_posix()] = digest(path)
    require(actual == expected, 'Frozen parent tree differs')
    return actual


def recover_temporaries(directory, allowed_files, check_only=False):
    directory = Path(directory)
    allowed = set(allowed_files)
    if not directory.exists():
        return
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'Link in output inventory')
        if not path.is_file():
            continue
        name = path.relative_to(directory).as_posix()
        if name in allowed:
            continue
        matched = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', name)
        require(matched is not None and matched.group(1) in allowed and not check_only,
                'Unexpected output inventory: ' + name)
        path.unlink()  # Only an uncommitted temporary of an exact allowed output.
