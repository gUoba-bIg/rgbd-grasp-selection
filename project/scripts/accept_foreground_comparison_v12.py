"""Independent, exact-arithmetic acceptance for a frozen two-population pilot."""
import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('full_scene', 'predicted_foreground')
FRICTION = (.2, .4, .6, .8, 1., 1.2)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Expected a relative artifact path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact path escaped root')
    return path


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def publish_once(path, payload, read_only=False):
    if path.exists():
        require(path.read_bytes() == payload, 'Immutable comparison artifact changed')
        return
    require(not read_only, 'Read-only comparison artifact is missing')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def writer_lock(root):
    folder = root / 'runtime/grasp_foreground_v12/acceptance_lock'
    folder.mkdir(parents=True, exist_ok=True)
    stream = (folder / '.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def peak_memory_bytes():
    import ctypes
    from ctypes import wintypes
    require(os.name == 'nt', 'Frozen working-set accounting requires Windows')

    class Memory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                    ('PrivateUsage', ctypes.c_size_t)]

    kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.argtypes, kernel.GetCurrentProcess.restype = [], wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    require(bool(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb)), 'Memory query failed')
    return int(memory.PeakWorkingSetSize)


def numeric_guard(root, started, extra_bytes=0, outputs=None):
    require(peak_memory_bytes() <= 8 * 1024 ** 3, 'Acceptance peak working set exceeds 8 GiB soft cap')
    require(time.monotonic() - started <= 300, 'Acceptance exceeds 300 second soft cap')
    require(shutil.disk_usage(root).free >= 60 * 1024 ** 3, 'Disk reserve below 60 GiB')
    paths = list(outputs or []) + [root / 'results/P3' / ('grasp_foreground_' + phase + '_v12_resume.json')
                                  for phase in ('generation', 'geometry')]
    files = set()
    for path in paths:
        if path.is_dir():
            files.update(item.resolve() for item in path.rglob('*') if item.is_file())
        else:
            if path.is_file():
                files.add(path.resolve())
            if path.parent.is_dir():
                files.update(item.resolve() for item in path.parent.glob(path.name + '.*')
                             if item.is_file() and (item.name.endswith('.tmp') or '.tmp-' in item.name))
    require(sum(path.stat().st_size for path in files) + extra_bytes <= 512 * 1024 ** 2, 'Combined round output exceeds 512 MiB')


def source_sample(points, predicted_ids, arm, seed=20261021, count=20000):
    require(arm in ARMS and points.ndim == 2 and points.shape[1] == 3 and points.dtype == np.float32
            and np.isfinite(points).all() and len(predicted_ids) == len(points), 'Invalid frozen source arrays')
    eligible = np.arange(len(points), dtype=np.int64) if arm == 'full_scene' else np.flatnonzero(predicted_ids > 0)
    if len(eligible) < count:
        return {'input_status': 'insufficient_input', 'eligible_valid_indices': eligible,
                'selected_local_indices': np.empty(0, dtype=np.int64), 'selected_valid_indices': np.empty(0, dtype=np.int64),
                'sampled_points': np.empty((0, 3), dtype=np.float32)}
    local = np.random.RandomState(seed).choice(len(eligible), count, replace=False).astype(np.int64)
    selected = eligible[local]
    return {'input_status': 'ready', 'eligible_valid_indices': eligible, 'selected_local_indices': local,
            'selected_valid_indices': selected, 'sampled_points': points[selected]}


def exact_curve(ordered_records):
    scores = [record['score'] for record in ordered_records]
    require(all(score in (-1.,) + FRICTION for score in scores), 'Invalid discrete friction score')
    counts, matrix, total = [], [], Fraction(0)
    for k in range(1, 51):
        row = [sum(0 < score <= mu for score in scores[:k]) for mu in FRICTION]
        counts.append(row)
        matrix.append([number / k for number in row])
        total += sum((Fraction(number, k) for number in row), Fraction(0))
    average = total / 300
    return {'counts': counts, 'matrix': matrix, 'mu04_top50_count': counts[49][1],
            'mu04_top50_fraction': Fraction(counts[49][1], 50), 'mean_fraction': average}


def fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'display_value': float(value)}


def effect_decision(curves, comparable=True):
    first, second = (curves[arm] for arm in ARMS)
    primary = second['mu04_top50_count'] > first['mu04_top50_count']
    secondary = second['mean_fraction'] >= first['mean_fraction']
    positive = bool(comparable and primary and secondary)
    return {'decision': 'incomplete_comparison_insufficient_input' if not comparable else
            ('positive_single_frame_sampling_pilot' if positive else 'valid_negative_do_not_adopt'),
            'both_arms_comparable_20000_point_inputs': bool(comparable), 'positive_effect_gate_passed': positive,
            'adopt_predicted_foreground_in_this_pilot': positive,
            'mu04_top50_strict_integer_increase': primary, 'exact_50x6_mean_non_decrease': secondary,
            'mu04_top50_count_difference': second['mu04_top50_count'] - first['mu04_top50_count'],
            'mu04_top50_fraction_difference': fraction_record(second['mu04_top50_fraction'] - first['mu04_top50_fraction']),
            'exact_matrix_mean_difference': fraction_record(second['mean_fraction'] - first['mean_fraction']),
            'statistical_significance_claimed': False, 'dataset_ap_claimed': False}


def independent_geometry(candidates, preparation, records, oracle, report):
    """Validate provenance, cached selection/scores, 300 fractions and record ordering."""
    total = len(candidates)
    require(candidates.dtype == np.float32 and candidates.shape == (total, 17) and np.isfinite(candidates).all(), 'Invalid generated candidates')
    require(preparation['input_count'] == total and preparation['selected_count'] == len(records), 'Preparation count differs')
    require(preparation['schema'] == 12 and report['schema'] == 12, 'Wrong geometry schema')
    require([record['selection_index'] for record in records] == list(range(len(records))), 'Score receipt indices are not contiguous')
    require(len({record['original_index'] for record in records}) == len(records), 'Duplicate selected candidate index')
    clipped = candidates.copy()
    clipped[:, 1] = np.clip(clipped[:, 1], 0, .1)
    for item, record in zip(preparation['selected'], records):
        require(all(record.get(key) == value for key, value in item.items()), 'Score receipt does not reproduce preparation')
        original = record['original_index']
        require(type(original) is int and 0 <= original < total and record['grasp'] == clipped[original].tolist(), 'Selected row not from its generated candidate')
        require(type(record['collision']) is bool and type(record['empty']) is bool and
                (not record['collision'] or record['score'] == -1.), 'Collision score semantics differ')
    nms_ids = preparation['nms_original_indices']
    require(len(nms_ids) == preparation['nms_count'] and len(nms_ids) == len(set(nms_ids))
            and all(type(i) is int and 0 <= i < total for i in nms_ids), 'NMS provenance invalid')
    require(set(record['original_index'] for record in records).issubset(set(nms_ids)), 'Selected candidate absent from NMS provenance')
    object_ids = preparation['object_ids']
    if total:
        slots = np.asarray(preparation['nms_object_slots'])
        require(slots.shape == (len(nms_ids),) and np.all((slots >= 0) & (slots < len(object_ids))), 'Invalid object assignment slots')
        limited = []
        for slot in range(len(object_ids)):
            ids = np.asarray(nms_ids, dtype=np.int64)[slots == slot]
            ids = ids[np.argsort(clipped[ids, 0])[::-1]][:10]
            limited.append(ids)
        require(preparation['per_object_pre_limit_counts'] == [len(ids) for ids in limited], 'Per-object top-10 counts differ')
        combined = np.concatenate(limited)
        descending = np.argsort(clipped[combined, 0])[::-1]
        cutoff = float(clipped[combined[descending[min(49, len(combined) - 1)]], 0])
        require(preparation['selection_min_confidence'] == cutoff, 'Global selection cutoff differs')
        expected_selected = [int(i) for ids in limited for i in ids[clipped[ids, 0] >= cutoff]]
        require([record['original_index'] for record in records] == expected_selected, 'Per-object/global selection or cutoff ties differ')
    expected_oracle = {'groups': [], 'scores': [], 'collision': []}
    for slot, object_id in enumerate(object_ids):
        group = [record for record in records if record['object_slot'] == slot]
        require(all(record['object_id'] == object_id for record in group), 'Geometry object slot/ID disagreement')
        require([record['object_local_index'] for record in group] == list(range(len(group))), 'Object-local selected order differs')
        expected_oracle['groups'].append([record['grasp'] for record in group])
        expected_oracle['scores'].append([record['score'] for record in group])
        expected_oracle['collision'].append([record['collision'] for record in group])
    require(oracle['result'] == expected_oracle, 'Official function oracle and cached scores differ')
    require(oracle['executed'] is (total > 0), 'Empty-input branch falsely claims official execution')
    require(report['official_oracle_exact_match'] is (True if total else None), 'Official reference claim differs')
    confidence = np.asarray([record['grasp'][0] for record in records], dtype=preparation['input_dtype'])
    ordered = [records[int(i)] for i in np.argsort(-confidence)]
    require(report['ordered_records'] == ordered, 'Global confidence sorting differs')
    curve = exact_curve(ordered)
    require(report['friction_grid_ascending'] == list(FRICTION), 'Friction reporting order differs')
    require(report['single_frame_prefix_pass_fraction_50x6'] == curve['matrix'], 'Independent 300-cell count fractions differ')
    require(report['single_frame_prefix_matrix_mean'] == float(np.asarray(curve['matrix'], dtype=np.float64).mean()), 'Recorded display mean differs')
    selected_ids = {record['original_index'] for record in records}
    tracking = [{'original_index': i, 'reason': 'evaluated' if i in selected_ids else
                 ('official_object_top10_or_global_score_threshold' if i in nms_ids else 'official_nms')} for i in range(total)]
    require(report['all_original_candidate_tracking'] == tracking, 'Original candidate rejection accounting differs')
    expected_counts = {'original_candidate_count': total, 'nms_count': len(nms_ids), 'selected_count': len(records),
                       'collision_including_empty_count': sum(record['collision'] for record in records),
                       'empty_count': sum(record['empty'] for record in records),
                       'positive_friction_score_count': sum(record['score'] > 0 for record in records),
                       'noncollision_contact_search_failure_count': sum(record['contact_diagnostic']['executed'] and
                           not record['contact_diagnostic']['contacts_found'] for record in records),
                       'noncollision_found_contacts_no_grid_force_closure_count': sum(record['score'] < 0 and
                           record['contact_diagnostic']['contacts_found'] is True for record in records)}
    require(all(report[key] == value for key, value in expected_counts.items()), 'Geometry summary counts differ')
    require(report['dataset_ap_computed'] is False and report['robot_success_rate_computed'] is False
            and report['gpu_used'] is False and report['new_training_steps'] == 0, 'Unsupported scoring scope')
    return curve, expected_counts


def snapshot(root, paths):
    return {path.relative_to(root).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns} for path in sorted(set(paths))}


def checked_branch(root, config, directory, arm, phase):
    completion = read(directory / 'completion.json')
    binding = read(directory / 'binding.json')
    report = read(directory / 'report.json')
    require(completion['schema'] == 12 and completion['status'] == 'completed' and completion['arm'] == arm, 'Unsealed branch')
    require(binding['protocol_sha256'] == digest(config) and binding['schema'] == 12 and binding['arm'] == arm, 'Branch protocol/identity differs')
    options = {'sort_keys': True, 'allow_nan': False}
    if phase == 'generation':
        options.update(ensure_ascii=False, separators=(',', ':'))
    binding_id = hashlib.sha256(json.dumps(binding, **options).encode()).hexdigest()
    require(completion['binding_fingerprint'] == binding_id and report['binding_fingerprint'] == binding_id, 'Branch fingerprint differs')
    expected_source = 'scripts/generate_foreground_grasps_v12.py' if phase == 'generation' else 'scripts/evaluate_foreground_grasps_v12.py'
    require(binding['script_sha256' if phase == 'generation' else 'own_source_sha256'] == digest(root / expected_source), 'Producer source changed')
    paths = [directory / 'completion.json']
    for relative, expected in completion['outputs'].items():
        path = safe_path(directory, relative)
        require(path != directory / 'completion.json' and digest(path) == expected, 'Branch output hash differs')
        paths.append(path)
    require({'binding.json', 'report.json'}.issubset(completion['outputs']), 'Branch seal omitted evidence')
    if phase == 'generation':
        artifact_names = ('eligible_indices.npy', 'sampled_eligible_indices.npy', 'sampled_valid_indices.npy',
                          'sampled_points.npy', 'sampled_pixel_indices.npy', 'forward_bundle.npz', 'candidates.npy', 'seed_provenance.npz')
        require(set(completion['outputs']) == set(artifact_names) | {name + '.pending.json' for name in artifact_names}
                | {'binding.json', 'input_commit.json', 'report.json'}, 'Generation seal membership differs')
        require(report['schema'] == 12 and report['status'] == 'completed' and report['arm'] == arm, 'Generation report not complete')
        require(completion['report_sha256'] == digest(directory / 'report.json')
                and completion['candidates_sha256'] == digest(directory / 'candidates.npy'), 'Generation output identity differs')
        require(binding['uses_ground_truth_labels'] is False and binding['uses_object_geometry'] is False
                and binding['new_training_updates'] == 0, 'Generation used forbidden information or training')
    return completion, binding, report, paths


def checked_receipt(path, binding):
    receipt = read(path.with_suffix('.sha256.json'))
    require(receipt == {'sha256': digest(path), 'binding_fingerprint': binding}, 'Geometry receipt differs')
    return read(path)


def checked_generation(root, config, protocol, directory, arm, points, pixels, labels):
    completion, binding, report, paths = checked_branch(root, config, directory, arm, 'generation')
    expected = source_sample(points, labels, arm, protocol['seed'], protocol['num_points'])
    arrays = {'eligible_indices': expected['eligible_valid_indices'], 'sampled_eligible_indices': expected['selected_local_indices'],
              'sampled_valid_indices': expected['selected_valid_indices'], 'sampled_points': expected['sampled_points'],
              'sampled_pixel_indices': pixels[expected['selected_valid_indices']]}
    commit = read(directory / 'input_commit.json')
    require(commit['schema'] == 12 and commit['arm'] == arm and commit['status'] == 'inputs_committed'
            and commit['binding_fingerprint'] == completion['binding_fingerprint'], 'Input commit identity differs')
    for name, value in arrays.items():
        path = directory / (name + '.npy')
        actual = np.load(path, allow_pickle=False)
        require(actual.dtype == value.dtype and np.array_equal(actual, value), 'Independent sampling differs: ' + name)
        require(commit['files'][path.name] == digest(path), 'Input commit array checksum differs')
    info = commit['information']
    expected_info = {'input_status': expected['input_status'], 'valid_point_count': len(points),
                     'eligible_point_count': len(expected['eligible_valid_indices']), 'sampled_point_count': len(expected['sampled_points']),
                     'sampling_with_replacement': False, 'seed': protocol['seed'], 'requested_points': protocol['num_points']}
    require(all(info[key] == value for key, value in expected_info.items()) and report['input_information'] == info, 'Input budget/count evidence differs')
    legacy_sampling_hash = None
    if arm == 'full_scene':
        native = np.random.RandomState(protocol['seed']).choice(len(points), protocol['num_points'], replace=False)
        legacy_sampling_hash = hashlib.sha256(native.tobytes()).hexdigest()
        require(legacy_sampling_hash == protocol['sampling']['full_scene_sampled_indices_sha256'], 'Original platform-native sampling hash differs')
        require(np.array_equal(native, arrays['sampled_valid_indices']), 'New full-scene sample differs from legacy values')
    candidates = np.load(directory / 'candidates.npy', allow_pickle=False)
    require(candidates.dtype == np.float32 and candidates.ndim == 2 and candidates.shape[1] == 17
            and 0 <= len(candidates) <= 1024 and np.isfinite(candidates).all(), 'Malformed generated candidates')
    with np.load(directory / 'forward_bundle.npz', allow_pickle=False) as archive:
        bundle = {key: archive[key].copy() for key in archive.files}
    with np.load(directory / 'seed_provenance.npz', allow_pickle=False) as archive:
        provenance = {key: archive[key].copy() for key in archive.files}
    require(np.array_equal(candidates, bundle['candidates']), 'Candidate export differs from forward seal')
    ready = expected['input_status'] == 'ready'
    forward_calls = 1 if ready else 0
    outcome = 'insufficient_input' if not ready else 'generated' if len(candidates) else 'empty_candidates'
    require(all(completion[key] == report[key] == value for key, value in (
        ('candidate_count', len(candidates)), ('input_status', expected['input_status']), ('outcome', outcome),
        ('forward_calls', forward_calls), ('forward_calls_committed', forward_calls))), 'Generation activity/count differs')
    require(report['uses_ground_truth_labels'] is False and report['uses_object_geometry'] is False
            and report['new_training_updates'] == 0 and report['is_dataset_ap_result'] is False, 'Unsupported generation claims')
    evidence = report['forward_evidence']
    require(evidence['model_forward_calls'] == forward_calls, 'Wrong committed forward budget')
    if ready:
        require(evidence['strict_load'] is True and evidence['weights_only'] is True and evidence['precision'] == 'float32'
                and evidence['autocast'] is False and evidence['checkpoint_sha256'] == protocol['checkpoint']['sha256'], 'Forward identity differs')
        sa1, sa2, selected, fp2 = (bundle[name] for name in ('sa1_indices', 'sa2_indices', 'selected_seed_indices', 'fp2_xyz'))
        require(sa1.dtype == sa2.dtype == selected.dtype == np.int64 and sa1.shape == (2048,) and sa2.shape == (1024,)
                and fp2.dtype == np.float32 and fp2.shape == (1024, 3), 'Malformed sampling hierarchy')
        require(np.all((sa1 >= 0) & (sa1 < 20000)) and np.all((sa2 >= 0) & (sa2 < 2048)), 'Sampling hierarchy index out of range')
        composed = sa1[sa2]
        require(np.array_equal(fp2, arrays['sampled_points'][composed]), 'Saved fp2 coordinates differ from actual composed source')
        require(selected.shape == (len(candidates),) and np.all((selected >= 0) & (selected < 1024))
                and np.all(np.diff(selected) > 0), 'Decoded seed identity/order differs')
        source = composed[selected]
        require(np.array_equal(candidates[:, 13:16], fp2[selected]), 'Generated translations lack exact sampled seed provenance')
    else:
        require(not len(candidates) and all(not len(bundle[name]) for name in ('sa1_indices', 'sa2_indices', 'selected_seed_indices', 'fp2_xyz')), 'Insufficient population still generated candidates')
        source = np.empty(0, dtype=np.int64)
    expected_provenance = {key: value for key, value in bundle.items() if key != 'candidates'}
    expected_provenance.update(candidate_sampled_indices=source, candidate_valid_indices=arrays['sampled_valid_indices'][source],
                               candidate_pixel_indices=arrays['sampled_pixel_indices'][source])
    require(set(provenance) == set(expected_provenance) and all(provenance[key].dtype == value.dtype and
            np.array_equal(provenance[key], value) for key, value in expected_provenance.items()), 'Candidate provenance export differs')
    require(np.all(candidates[:, 16] == -1), 'Predicted instance IDs were substituted into object classes')
    if len(candidates):
        rotation = candidates[:, 4:13].reshape(-1, 3, 3)
        require(np.max(np.abs(rotation.transpose(0, 2, 1) @ rotation - np.eye(3))) < 1e-4
                and np.max(np.abs(np.linalg.det(rotation) - 1)) < 1e-4, 'Generated rotations are invalid')
        require(np.all((candidates[:, 1] >= 0) & (candidates[:, 1] <= .100001)) and np.allclose(candidates[:, 2], .02)
                and np.isclose(candidates[:, 3, None], [.01, .02, .03, .04], atol=1e-6).any(axis=1).all(), 'Generated dimensions invalid')
    legacy = report['legacy_baseline_diagnostic']
    require(legacy['selection_gate'] is False and legacy['repeat_on_mismatch'] is False, 'Old candidate reproduction became a post-hoc selector')
    require(legacy['comparison_applicable'] is (arm == 'full_scene'), 'Wrong legacy comparison scope')
    if arm == 'full_scene':
        old = np.load(root / 'results/P3/scene_0003_realsense_0000_candidates.npy', allow_pickle=False)
        same = old.shape == candidates.shape and old.dtype == candidates.dtype
        require(legacy['old_candidate_count'] == len(old) and legacy['same_shape_dtype'] is same
                and legacy['bitwise_array_equal'] is bool(same and old.tobytes() == candidates.tobytes()), 'Legacy diagnostic arithmetic differs')
    return candidates, {'input_status': expected['input_status'], 'outcome': outcome, 'eligible_points': len(expected['eligible_valid_indices']),
                        'sampled_points': len(expected['sampled_points']), 'candidate_count': len(candidates),
                        'forward_calls_committed': forward_calls, 'legacy_native_sampled_indices_sha256': legacy_sampling_hash,
                        'legacy_baseline_diagnostic': legacy,
                        'candidate_sha256': digest(directory / 'candidates.npy'), 'seed_chain_independently_verified': True}, paths


def checked_recovery(root, config, protocol, results):
    proofs = {}
    for phase, child, environment in (('generation', 'generate_foreground_grasps_v12.py', 'vision'),
                                       ('geometry', 'evaluate_foreground_grasps_v12.py', 'grasp_eval')):
        path = root / 'results/P3' / ('grasp_foreground_' + phase + '_v12_resume.json')
        proof = read(path)
        require(proof['schema'] == 12 and proof['status'] == 'passed' and proof['phase'] == phase
                and set(proof['arms']) == set(ARMS) and proof['new_training_updates'] == 0, 'Incomplete recovery proof')
        binding = proof['binding']
        require(binding['protocol_sha256'] == digest(config) and binding['wrapper_sha256'] == digest(root / 'scripts/run_foreground_comparison_v12.py')
                and binding['child_sha256'] == digest(root / 'scripts' / child)
                and binding['interpreter_sha256'] == digest(root / 'runtime/envs' / environment / 'Scripts/python.exe'), 'Recovery binding changed')
        for relative, checksum in binding['helpers'].items():
            require(digest(safe_path(root, relative)) == checksum, 'Recovery helper changed')
        arm_evidence = {}
        for arm in ARMS:
            record = proof['arms'][arm]
            directory = safe_path(root, protocol['outputs']['root']) / arm / phase
            require(record['status'] == 'passed' and record['byte_and_mtime_identical'] is True
                    and record['read_only_check_passed'] is True and record['prefix_before'] == record['prefix_after'], 'Recovery prefix comparison failed')
            require(bool(record['prefix_before']), 'Recovery prefix was empty')
            for relative, expected in record['prefix_before'].items():
                require(snapshot(root, [safe_path(root, relative)])[relative] == expected, 'Durable prefix changed since recovery proof')
            require(record['completion_sha256'] == digest(directory / 'completion.json')
                    and record['report_sha256'] == digest(directory / 'report.json'), 'Recovery final seal changed')
            probe, complete, readonly = (record[key] for key in ('probe_record', 'completion_record', 'read_only_record'))
            require(readonly['status'] == 'verified_completed_read_only', 'No successful terminal read-only invocation')
            if phase == 'generation':
                require(probe['status'] == 'inputs_committed' and probe['new_model_forward_calls'] == 0
                        and readonly['new_model_forward_calls'] == 0 and complete['new_model_forward_calls'] in (0, 1), 'Input-stage recovery did extra inference')
                require(complete['forward_calls_committed'] == results[arm]['generation']['forward_calls_committed'], 'Committed inference count differs')
            else:
                require(readonly['new_grasps'] == 0 and readonly['geometry_computations'] == 0
                        and readonly['official_oracle_executed_this_invocation'] is False, 'Read-only invocation rescored grasps')
                for event in (probe, complete, readonly):
                    require(event['new_grasps'] + event['reused_grasps'] == event['committed_grasps'], 'Scoring counters do not conserve')
                require(complete['committed_grasps'] == readonly['committed_grasps'] == results[arm]['geometry_counts']['selected_count'], 'Incomplete scoring recovery')
                if probe['status'] == 'partial':
                    require(probe['committed_grasps'] >= 2 and complete['reused_grasps'] >= probe['committed_grasps'], 'Scoring prefix was recomputed')
            arm_evidence[arm] = {'prefix_members': len(record['prefix_before']), 'probe_record': probe,
                                'completion_record': complete, 'read_only_record': readonly, 'prefix_bytes_and_mtime_unchanged': True}
        proofs[phase] = {'path': path.relative_to(root).as_posix(), 'sha256': digest(path), 'arms': arm_evidence}
    return proofs


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', size)


def render_comparison(results, curves, decision):
    canvas = Image.new('RGB', (1380, 620), '#f5f7fa')
    draw = ImageDraw.Draw(canvas)
    colours = {'full_scene': '#2767a8', 'predicted_foreground': '#d76a2b'}
    draw.text((24, 16), 'FIXED-BUDGET FOREGROUND SAMPLING COMPARISON', font=font(27), fill='#17243a')
    draw.text((24, 55), 'Training scene 0003 / frame 0000 | one frozen seed | 20,000 points per executable arm | not dataset AP', font=font(17), fill='#485971')
    for index, arm in enumerate(ARMS):
        x = 24 + index * 280
        draw.rectangle((x, 91, x + 17, 108), fill=colours[arm])
        draw.text((x + 25, 87), arm.replace('_', ' '), font=font(17), fill='#17243a')
    lefts = (65, 520, 975)
    top, bottom, panel_width = 175, 450, 340
    for left, title in zip(lefts, ('A  Candidate counts', 'B  Top-50 pass fraction by friction', 'C  Prefix pass fraction at mu = 0.4')):
        draw.text((left - 32, 135), title, font=font(17), fill='#17243a')
        draw.line((left, top, left, bottom, left + panel_width, bottom), fill='#7d8797', width=2)
    names = ('generated', 'evaluated', 'positive')
    counts = {arm: [results[arm]['generation']['candidate_count'], results[arm]['geometry_counts']['selected_count'],
                    results[arm]['geometry_counts']['positive_friction_score_count']] for arm in ARMS}
    ceiling = max(1, max(value for group in counts.values() for value in group))
    for position, label in enumerate(names):
        x = lefts[0] + 45 + position * 108
        draw.text((x, bottom + 12), label, anchor='mm', font=font(14), fill='#485971')
        for offset, arm in enumerate(ARMS):
            value = counts[arm][position]
            bx = x - 28 + offset * 29
            by = bottom - (bottom - top - 30) * value / ceiling
            draw.rectangle((bx, by, bx + 24, bottom), fill=colours[arm])
            draw.text((bx + 12, by - 13), str(value), anchor='mm', font=font(14), fill=colours[arm])
    for panel_index in (1, 2):
        left = lefts[panel_index]
        for value in (0, .25, .5, .75, 1):
            y = bottom - value * (bottom - top)
            draw.line((left, y, left + panel_width, y), fill='#dce1e8')
            draw.text((left - 8, y), '{:.2g}'.format(value), anchor='rm', font=font(13), fill='#485971')
        for arm in ARMS:
            if panel_index == 1:
                series = curves[arm]['matrix'][49]
                coords = [(left + i * panel_width / 5, bottom - value * (bottom - top)) for i, value in enumerate(series)]
            else:
                series = [row[1] for row in curves[arm]['matrix']]
                coords = [(left + i * panel_width / 49, bottom - value * (bottom - top)) for i, value in enumerate(series)]
            draw.line(coords, fill=colours[arm], width=3)
            if panel_index == 1:
                for x, y in coords:
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colours[arm])
        ticks = [(i / 5, '{:.1f}'.format(mu)) for i, mu in enumerate(FRICTION)] if panel_index == 1 else [(0, '1'), (.184, '10'), (.49, '25'), (1, '50')]
        for fraction, label in ticks:
            draw.text((left + panel_width * fraction, bottom + 14), label, anchor='mm', font=font(14), fill='#485971')
    draw.text((24, 510), 'Effect rule: strictly more top-50 passes at mu=0.4 AND no decrease in the exact mean of all 300 prefix/friction cells.', font=font(17), fill='#17243a')
    draw.text((24, 546), 'Outcome: ' + decision['decision'].replace('_', ' '), font=font(20), fill='#17243a')
    draw.text((24, 581), 'Negative and insufficient-input results are retained. Geometric scoring is not physical robot success or a generalization estimate.', font=font(15), fill='#485971')
    stream = io.BytesIO()
    canvas.save(stream, format='PNG')
    return stream.getvalue()


def execute(root, config, check_only=False):
    started = time.monotonic()
    protocol = read(config)
    require(protocol['schema'] == 12 and protocol['arms'] == list(ARMS) and protocol['frame'] == {'scene': 3, 'camera': 'realsense', 'frame': '0000'}, 'Unsupported protocol/frame')
    require(protocol['seed'] == 20261021 and protocol['num_points'] == 20000 and protocol['maximum_depth_m'] == 2
            and protocol['sampling']['replacement'] is False, 'Frozen matched sampling budget changed')
    rule = protocol['effect_rule']
    require(rule['primary_friction'] == .4 and rule['primary_prefix'] == rule['prefixes'] == 50
            and rule['friction_grid'] == list(FRICTION) and rule['no_posthoc_threshold_or_seed_changes'] is True, 'Frozen scientific gate changed')
    require(protocol['acceptance_resources'] == {'cpu_threads': 2, 'invocation_soft_seconds': 300, 'rss_soft_bytes': 8 * 1024 ** 3,
            'output_limit_bytes': 512 * 1024 ** 2, 'disk_reserve_bytes': 60 * 1024 ** 3}, 'Acceptance resource bounds changed')
    output_root = safe_path(root, protocol['outputs']['root'])
    output = safe_path(root, protocol['outputs']['acceptance_report'])
    visual = safe_path(root, protocol['outputs']['visuals_dir']) / 'scene_0003_0000_comparison.png'
    readonly = check_only or output.exists()
    require(not check_only or output.is_file(), 'Read-only requires completed acceptance')
    budget_paths = [safe_path(root, value) for value in protocol['outputs'].values()]
    numeric_guard(root, started, outputs=budget_paths)
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pillow': importlib.metadata.version('Pillow')}
    require(all(protocol['runtime'][key] == value for key, value in versions.items()), 'Frozen acceptance numerical runtime differs')
    prior = dict(protocol['input_hashes'], **protocol['source_hashes'])
    prior[protocol['v10_protocol']['path']] = protocol['v10_protocol']['sha256']
    for relative, checksum in prior.items():
        require(digest(safe_path(root, relative)) == checksum, 'Frozen source/input differs: ' + relative)
    source = safe_path(root, protocol['association_input_dir'])
    points, pixels, labels = [np.load(source / (name + '.npy'), allow_pickle=False) for name in ('valid_points', 'valid_pixel_indices', 'valid_instance_ids')]
    require(pixels.dtype == np.int64 and pixels.shape == (len(points),) and np.all(np.diff(pixels) > 0)
            and np.all((pixels >= 0) & (pixels < 720 * 1280)) and labels.dtype == np.int32
            and labels.shape == (len(points),) and np.all(labels >= 0) and np.all((points[:, 2] > 0) & (points[:, 2] <= 2)), 'Malformed sealed R11 provenance')
    source_seal = read(source / 'completion.json')
    for name in ('valid_points.npy', 'valid_pixel_indices.npy', 'valid_instance_ids.npy'):
        require(source_seal['files'][name] == digest(source / name), 'R11 source arrays are not sealed')
    results, curves, all_paths, generations = {}, {}, [], {}
    for arm in ARMS:
        generation_dir = output_root / arm / 'generation'
        candidates, generation, paths = checked_generation(root, config, protocol, generation_dir, arm, points, pixels, labels)
        generations[arm] = candidates
        results[arm] = {'generation': generation}
        all_paths.extend(paths)
    for arm in ARMS:
        directory = output_root / arm / 'geometry'
        completion, binding, report, paths = checked_branch(root, config, directory, arm, 'geometry')
        all_paths.extend(paths)
        require(set(binding['all_generation_seals']) == set(ARMS), 'Geometry did not bind both already-completed candidate arms')
        for peer in ARMS:
            peer_path = output_root / peer / 'generation/completion.json'
            require(binding['all_generation_seals'][peer]['completion_sha256'] == digest(peer_path), 'Geometry bound a different generation result')
        preparation = checked_receipt(directory / 'preparation.json', completion['binding_fingerprint'])
        geometry_names = {'binding.json', 'preparation.json', 'preparation.sha256.json', 'oracle.json', 'oracle.sha256.json', 'report.json', 'last_invocation.json'}
        geometry_names.update('grasps/grasp_%04d%s' % (i, suffix) for i in range(preparation['selected_count']) for suffix in ('.json', '.sha256.json'))
        require(set(completion['outputs']) == geometry_names, 'Geometry seal membership differs')
        records = [checked_receipt(directory / 'grasps' / ('grasp_%04d.json' % i), completion['binding_fingerprint'])
                   for i in range(preparation['selected_count'])]
        oracle = checked_receipt(directory / 'oracle.json', completion['binding_fingerprint'])
        curve, counts = independent_geometry(generations[arm], preparation, records, oracle, report)
        require(completion['expected_grasps'] == counts['selected_count'] and report['generation_outcome'] == results[arm]['generation']['outcome'], 'Geometry/generation mismatch')
        expected_status = ('accepted_limited_single_frame_geometry_engineering' if len(generations[arm]) else
                           'skipped_insufficient_generation_input' if results[arm]['generation']['outcome'] == 'insufficient_input' else
                           'completed_empty_candidate_negative_result')
        require(report['status'] == expected_status, 'Incorrect geometry engineering outcome')
        results[arm]['geometry_counts'] = counts
        results[arm]['exact_metrics'] = {'top50_mu04_pass_count': curve['mu04_top50_count'],
                                        'top50_mu04_pass_fraction': fraction_record(curve['mu04_top50_fraction']),
                                        'matrix_mean': fraction_record(curve['mean_fraction']),
                                        'prefix_pass_counts_50x6': curve['counts'], 'prefix_pass_fraction_50x6': curve['matrix']}
        curves[arm] = curve
    sealed_before = snapshot(root, all_paths)
    recovery = checked_recovery(root, config, protocol, results)
    comparable = all(results[arm]['generation']['input_status'] == 'ready' and results[arm]['generation']['sampled_points'] == 20000 for arm in ARMS)
    decision = effect_decision(curves, comparable)
    figure = render_comparison(results, curves, decision)
    report = {'schema': 12, 'status': 'passed_engineering_integrity', 'frame': protocol['frame'], 'protocol_sha256': digest(config),
              'acceptance_source_sha256': digest(Path(__file__)), 'scientific_decision': decision, 'arms': results,
              'engineering_checks': {'matched_input_reconstruction': True, 'candidate_seed_provenance': True,
                  'all_original_candidates_accounted': True, 'official_reference_or_explicit_empty_branch': True,
                  'all_300_cells_independently_recomputed_per_arm': True, 'prespecified_exact_rational_decision': True,
                  'staged_recovery_and_readonly_reentry': True, 'all_frozen_inputs_unchanged': True},
              'recovery_evidence': recovery, 'sealed_branch_files': sealed_before,
              'visual': {'path': visual.relative_to(root).as_posix(), 'sha256': hashlib.sha256(figure).hexdigest(),
                         'font_sha256': digest(Path('C:/Windows/Fonts/segoeui.ttf')), 'panels_fixed_before_results': True},
              'new_training_updates': 0, 'acceptance_model_forward_calls': 0, 'acceptance_geometry_scoring_calls': 0,
              'acceptance_runtime': versions,
              'acceptance_ground_truth_label_reads': 0, 'dataset_ap_computed': False, 'robot_success_rate_computed': False,
              'prior_hashes_verified': prior, 'limitations': protocol['limitations']}
    require(snapshot(root, all_paths) == sealed_before, 'Acceptance modified sealed outputs')
    for relative, checksum in prior.items():
        require(digest(safe_path(root, relative)) == checksum, 'Frozen source/input changed during acceptance')
    payload = json_bytes(report)
    extra = (0 if visual.exists() else len(figure)) + (0 if output.exists() else len(payload))
    numeric_guard(root, started, extra_bytes=extra, outputs=budget_paths)
    publish_once(visual, figure, readonly)
    publish_once(output, payload, readonly)
    print(json.dumps({'status': 'verified_completed_read_only' if readonly else 'passed_engineering_integrity',
                      'decision': decision, 'report': output.relative_to(root).as_posix()}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config.resolve() if args.config else root / 'configs/grasp_foreground_comparison_v12.json'
    with writer_lock(root):
        execute(root, config, args.check_only)


if __name__ == '__main__':
    main()
