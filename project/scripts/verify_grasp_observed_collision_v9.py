"""Bounded, resumable observed-cloud collision engineering. No labels or GPU."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/grasp_observed_collision_v9.json'
OUT = ROOT / 'results/P3/grasp_observed_collision_v9'
REPORT = ROOT / 'results/P3/grasp_observed_collision_v9.json'
ENV = ROOT / 'runtime/envs/grasp_eval'
SCORE_NAMES = ('global', 'left_finger', 'right_finger', 'bottom', 'approach')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def once(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Immutable output differs: ' + str(path))
    else:
        atomic(path, value)


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.process.lock'
    stream = path.open('a+b')
    try:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def dependencies():
    import numpy as np
    import open3d as o3d
    from graspnetAPI import GraspGroup
    path = ROOT / 'runtime/build/graspnet/utils/collision_detector.py'
    spec = importlib.util.spec_from_file_location('frozen_collision_v9', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require('torch' not in sys.modules and not hasattr(np, 'int'), 'Unexpected runtime/NumPy alias')
    return np, o3d, GraspGroup, module.ModelFreeCollisionDetector


class Monitor:
    """Sampled soft limits; cannot interrupt an in-flight native call safely."""
    def __init__(self, protocol, out=OUT):
        import psutil
        self.process = psutil.Process()
        self.protocol, self.out = protocol, out
        self.started, self.peak, self.error = time.monotonic(), 0, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def watch(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
                if self.peak > self.protocol['rss_soft_bytes']:
                    self.error = 'RSS soft limit exceeded'
                if time.monotonic() - self.started > self.protocol['invocation_soft_seconds']:
                    self.error = 'Invocation time soft limit exceeded'
            except Exception as exc:
                self.error = 'Resource monitor failed: ' + str(exc)
            self.stop.wait(.25)

    def check(self):
        require(self.error is None, self.error or '')
        require(time.monotonic() - self.started <= self.protocol['invocation_soft_seconds'], 'Time limit exceeded')
        require(shutil.disk_usage(ROOT).free >= self.protocol['disk_reserve_bytes'], 'Disk reserve unavailable')
        size = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) if self.out.exists() else 0
        if REPORT.exists():
            size += REPORT.stat().st_size
        require(size <= self.protocol['output_limit_bytes'], 'Output budget exceeded')

    def close(self):
        self.stop.set()
        self.thread.join(1)


def runtime_binding(protocol):
    require(Path(sys.prefix).resolve() == ENV.resolve(), 'Run only with isolated grasp_eval Python')
    site = ENV / 'Lib/site-packages'
    api = site / 'graspnetAPI'
    tree = {p.relative_to(api).as_posix(): digest(p) for p in sorted(api.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}
    require(fingerprint(tree) == protocol['installed_api_tree_sha256'], 'V7 API source changed')
    inventory = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    require(inventory == read(ROOT / 'results/P2/grasp_geometry_v7.json')['packages'], 'V7 package inventory changed')
    # Bind native implementation and Python code for numerical dependencies, without copying artifacts.
    files = {}
    for name in ('numpy', 'scipy', 'open3d', 'PIL', 'psutil', 'threadpoolctl.py'):
        target = site / name
        paths = [target] if target.is_file() else target.rglob('*')
        for path in paths:
            if path.is_file() and path.suffix in ('.py', '.pyd', '.dll'):
                files[path.relative_to(ROOT).as_posix()] = digest(path)
    return {'python': sys.version, 'prefix': str(ENV.resolve()), 'inventory': inventory,
            'api_files': tree, 'numerical_dependency_files': files,
            'cpu_threads': 2, 'gpu_used': False}


def make_binding(protocol):
    hashes = dict(protocol['input_hashes'], **protocol['source_hashes'])
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), 'Path escaped project')
        require(digest(path) == expected, 'Frozen input/source changed: ' + relative)
    return {'schema': 9, 'protocol_sha256': digest(CONFIG), 'script_sha256': digest(__file__),
            'frozen_hashes': hashes, 'runtime': runtime_binding(protocol)}


def load_inputs(protocol, np):
    from PIL import Image
    from scipy.io import loadmat
    base = ROOT / 'data/raw/graspnet/scenes/scene_0003/realsense'
    # variable_names ensures semantic/pose/class fields are never decoded.
    meta = loadmat(base / 'meta/0000.mat', variable_names=['intrinsic_matrix', 'factor_depth'])
    intrinsic = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
    factor = float(np.asarray(meta['factor_depth']).reshape(-1)[0])
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all() and factor > 0
            and math.isfinite(factor) and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0, 'Invalid camera')
    with Image.open(base / 'depth/0000.png') as image:
        depth = np.asarray(image)
    z = depth.astype(np.float64) / factor
    require(z.ndim == 2, 'Depth must be a 2D image')
    v, u = np.indices(z.shape)
    valid = np.isfinite(z) & (z > 0) & (z <= protocol['maximum_depth_m'])
    points = np.stack(((u[valid] - intrinsic[0, 2]) * z[valid] / intrinsic[0, 0],
                       (v[valid] - intrinsic[1, 2]) * z[valid] / intrinsic[1, 1], z[valid]), axis=1)
    smoke = read(ROOT / 'results/P3/inference_smoke.json')
    require(len(points) == smoke['valid_points_before_sampling'] and depth.shape == tuple(smoke['input_shape_hw']),
            'Source depth/camera reproduction mismatch')
    require(intrinsic.tolist() == smoke['intrinsic_matrix'] and factor == smoke['factor_depth'], 'Camera differs from source')
    grasps = np.load(ROOT / 'results/P3/scene_0003_realsense_0000_candidates.npy', allow_pickle=False)
    require(grasps.shape == (protocol['candidate_count'], 17) and np.isfinite(grasps).all(), 'Candidate shape/nonfinite')
    require((grasps[:, 1:4] > 0).all(), 'Nonpositive grasp dimensions')
    return points, grasps, {'depth_shape_hw': list(depth.shape), 'factor_depth': factor,
                           'intrinsic_matrix': intrinsic.tolist(), 'observed_points': len(points)}


def block_rows(detector, grasps, start, stop, protocol, Group):
    collision, empty, scores = detector.detect(Group(grasps[start:stop].copy()),
        approach_dist=protocol['approach_distance_m'], collision_thresh=protocol['collision_threshold'],
        return_empty_grasp=True, empty_thresh=protocol['empty_threshold'], return_ious=True)
    return [{'original_index': i, 'collision': bool(collision[i-start]), 'empty_diagnostic': bool(empty[i-start]),
             'occupied_volume_scores': {name: float(score[i-start]) for name, score in zip(SCORE_NAMES, scores)}}
            for i in range(start, stop)]


def validate_rows(rows, start, stop, protocol):
    require(isinstance(rows, list) and len(rows) == stop-start, 'Incorrect block rows')
    for i, row in enumerate(rows, start):
        require(row['original_index'] == i and type(row['collision']) is bool
                and type(row['empty_diagnostic']) is bool, 'Invalid index or flag')
        scores = row['occupied_volume_scores']
        require(set(scores) == set(SCORE_NAMES) and all(type(x) in (float, int) and math.isfinite(x) and x >= 0
                for x in scores.values()), 'Invalid occupied-volume score')
        require(row['collision'] == (scores['global'] > protocol['collision_threshold']), 'Collision decision mismatch')


def nms_indices(grasps, indices, protocol, Group, np):
    if not indices:
        return []
    source = grasps[indices].copy()
    # Upstream returns a NEW group. Do not copy demo's unused return value.
    returned = Group(source).nms(protocol['nms_translation_m'], math.radians(protocol['nms_rotation_degrees']))
    rows = returned.grasp_group_array
    available, kept = list(indices), []
    for row in rows:
        matches = [i for i in available if np.array_equal(grasps[i], row)]
        require(matches, 'NMS emitted modified/unknown candidate')
        # Exact duplicates have indistinguishable rows; stable lowest original index is the declared tie convention.
        chosen = min(matches)
        kept.append(chosen)
        available.remove(chosen)
    return kept


def derive_report(rows, grasps, protocol, Group, np, binding_fp, cloud):
    free = [r['original_index'] for r in rows if not r['collision']]
    kept = nms_indices(grasps, free, protocol, Group, np)
    raw_kept = nms_indices(grasps, list(range(len(grasps))), protocol, Group, np)
    result_rows = []
    for row in rows:
        reason = 'observed_collision_rejected' if row['collision'] else (
            'survived_observed_collision_and_nms' if row['original_index'] in kept else 'nms_suppressed_after_collision')
        result_rows.append(dict(row, decision=reason, input_score=float(grasps[row['original_index'], 0])))
    return {'schema': 9, 'status': 'accepted_limited_observed_collision_engineering',
            'binding_fingerprint': binding_fp, 'dataset_ap_computed': False, 'force_closure_computed': False,
            'robot_safety_established': False, 'gpu_used': False, 'uses_ground_truth_labels': False,
            'metadata_fields_read': protocol['metadata_fields_read'], 'cloud': cloud,
            'counts': {'raw_candidates': len(grasps), 'collision_rejected': len(grasps)-len(free),
                       'collision_free_observed': len(free), 'nms_rejected_after_collision': len(free)-len(kept),
                       'survivors': len(kept), 'raw_direct_nms_survivors': len(raw_kept),
                       'empty_diagnostic': sum(r['empty_diagnostic'] for r in rows)},
            'survivor_original_indices': kept, 'raw_direct_nms_original_indices': raw_kept,
            'candidates': result_rows, 'nms_duplicate_index_convention': protocol['nms_duplicate_index_convention'],
            'score_semantics': 'occupied point count / voxelized geometric volume; not set IoU or probability',
            'empty_rejected': False, 'top_k_truncation': None,
            'resource_policy': {'cpu_threads_requested': 2, 'rss_sampled_soft_limit_bytes': protocol['rss_soft_bytes'],
                                'time_soft_seconds_per_invocation': protocol['invocation_soft_seconds'],
                                'native_calls_not_preempted': True, 'sample_interval_seconds': .25},
            'limitations': protocol['limitations']}


def execute_cache(protocol, binding, grasps, compute, derive, monitor, out=OUT, report_path=REPORT,
                  max_new_blocks=None, check_only=False):
    """All receipts are commits; incomplete payload is accepted only after identical recomputation."""
    fp = fingerprint(binding)
    count, size = len(grasps), protocol['block_size']
    expected = math.ceil(count / size)
    completed = (out / 'completion.json').exists()
    require(not check_only or completed, 'Cannot check incomplete run')
    allowed_blocks = {'block_{:04d}{}'.format(i, suffix) for i in range(expected)
                      for suffix in ('.json', '.sha256.json')}
    block_dir = out / 'blocks'
    if block_dir.exists():
        require(all(p.is_file() and (p.name in allowed_blocks or p.name.endswith('.tmp'))
                    for p in block_dir.iterdir()), 'Unexpected block artifact')
        if completed:
            # A power loss may leave an uncommitted atomic-write temporary. Preserve it,
            # but never count it as a payload/receipt or include it in committed results.
            require({p.name for p in block_dir.iterdir() if not p.name.endswith('.tmp')} == allowed_blocks,
                    'Completed block artifact set differs')
    if out.exists() and any(out.iterdir()):
        binding_path = out / 'binding.json'
        if binding_path.exists():
            require(read(binding_path) == binding, 'Cache binding changed/missing')
        else:
            # The first atomic binding write may have lost power before its rename.
            # Only these uncommitted binding temporaries can precede a fresh binding;
            # any block/output without a binding must still fail closed.
            require(all(p.is_file() and p.name.startswith('binding.json.')
                        and p.name.endswith('.tmp') and len(p.name) > len('binding.json..tmp')
                        for p in out.iterdir()), 'Cache binding changed/missing')
    elif check_only:
        raise RuntimeError('Missing complete output')
    if not completed:
        once(out / 'binding.json', binding)
    rows, receipts, new, reused = [], {}, 0, 0
    for block in range(expected):
        monitor.check()
        start, stop = block * size, min(count, (block + 1) * size)
        name = 'block_{:04d}'.format(block)
        path, receipt_path = out / 'blocks' / (name + '.json'), out / 'blocks' / (name + '.sha256.json')
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(path.exists() and receipt == {'file': path.name, 'sha256': digest(path)}, 'Block receipt mismatch')
            payload = read(path)
            reused += 1
        else:
            require(not completed, 'Completed cache has missing block receipt')
            if max_new_blocks is not None and new >= max_new_blocks:
                break
            block_result = compute(start, stop)
            payload = {'binding_fingerprint': fp, 'block_index': block, 'start': start, 'stop': stop, 'rows': block_result}
            validate_rows(block_result, start, stop, protocol)
            once(path, payload)
            receipt = {'file': path.name, 'sha256': digest(path)}
            once(receipt_path, receipt)
            new += 1
        require(payload['binding_fingerprint'] == fp and payload['block_index'] == block
                and payload['start'] == start and payload['stop'] == stop, 'Block binding/range mismatch')
        validate_rows(payload['rows'], start, stop, protocol)
        rows.extend(payload['rows'])
        receipts[str(receipt_path.relative_to(out)).replace('\\', '/')] = digest(receipt_path)
    all_done = len(rows) == count
    invocation = {'status': 'completed' if all_done else 'step_limit_reached', 'new_blocks': new,
                  'reused_blocks': reused, 'committed_blocks': new+reused, 'expected_blocks': expected,
                  'binding_fingerprint': fp}
    if completed:
        completion = read(out / 'completion.json')
        require(completion['status'] == 'completed' and completion['binding_fingerprint'] == fp
                and completion['expected_blocks'] == expected and completion['block_receipts'] == receipts,
                'Completion binding/receipts mismatch')
        require(completion['report_sha256'] == digest(report_path), 'Report hash changed')
        require(completion['last_invocation_sha256'] == digest(out / 'last_invocation.json'), 'Invocation hash changed')
        require(read(report_path) == derive(rows), 'Completed report does not reconstruct from committed rows')
        require(read(out / 'last_invocation.json')['committed_blocks'] == expected, 'Incomplete final invocation')
        require(set(completion) == {'status', 'binding_fingerprint', 'expected_blocks', 'block_receipts',
                                    'report_sha256', 'last_invocation_sha256'}, 'Unexpected completion fields')
        return {'status': 'verified_existing_no_write', 'expected_blocks': expected}
    atomic(out / 'last_invocation.json', invocation)
    if all_done:
        report = derive(rows)
        once(report_path, report)
        monitor.check()
        completion = {'status': 'completed', 'binding_fingerprint': fp, 'expected_blocks': expected,
                      'block_receipts': receipts, 'report_sha256': digest(report_path),
                      'last_invocation_sha256': digest(out / 'last_invocation.json')}
        once(out / 'completion.json', completion)
    return invocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new-blocks', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    require(args.max_new_blocks is None or args.max_new_blocks > 0, 'Block limit must be positive')
    require(not (args.check_only and args.max_new_blocks), 'Conflicting modes')
    protocol = read(CONFIG)
    monitor = Monitor(protocol)
    try:
        with lock(ROOT / 'runtime/grasp_observed_collision_v9'):
            binding = make_binding(protocol)
            monitor.check()
            np, o3d, Group, Detector = dependencies()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=protocol['cpu_threads']):
                points, grasps, cloud = load_inputs(protocol, np)
                detector = Detector(points, voxel_size=protocol['voxel_size_m'])
                cloud['downsampled_points'] = len(detector.scene_points)
                cloud['voxel_size_m'] = protocol['voxel_size_m']
                cloud['downsampled_points_sha256'] = hashlib.sha256(detector.scene_points.tobytes()).hexdigest()
                monitor.check()
                status = execute_cache(protocol, binding, grasps,
                    lambda a, b: block_rows(detector, grasps, a, b, protocol, Group),
                    lambda rows: derive_report(rows, grasps, protocol, Group, np, fingerprint(binding), cloud),
                    monitor, max_new_blocks=args.max_new_blocks, check_only=args.check_only)
                monitor.check()
                print(json.dumps(dict(status, monitored_peak_rss_bytes=monitor.peak,
                                      elapsed_seconds=time.monotonic()-monitor.started), ensure_ascii=False))
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
