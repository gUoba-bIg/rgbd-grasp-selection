"""Bounded per-frame object-geometry baseline on four existing training views."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_grasp_geometry_v10 as legacy
import audit_grasp_geometry_v10 as audit
import multiview_common_v14 as common
import numpy as np

FRAMES = ('0016', '0080', '0160', '0240')
OBJECTS = [2, 21, 22, 41, 43, 44, 52, 60, 62]
FRICTION = [.2, .4, .6, .8, 1., 1.2]
require = legacy.require


def validate_candidates(candidates, count):
    require(type(count) is int and 0 <= count <= 1024 and candidates.dtype == np.float32
            and candidates.shape == (count, 17) and np.isfinite(candidates).all(), 'Invalid sealed candidates')
    require((candidates[:, 0] >= 0).all() and (candidates[:, 16] == -1).all(), 'Invalid confidence or substituted object IDs')
    return candidates


def validate_frame_poses(xml_poses, metadata, camera, align, limits):
    """Pair MAT and XML by dataset ID, never by their potentially different order."""
    require(list(sorted(xml_poses)) == OBJECTS, 'Frame XML object set differs from the nine shared assets')
    pose_audit = audit.pose_comparison(xml_poses, metadata['cls_indexes'], metadata['poses'], OBJECTS, limits)
    camera_metrics = audit.rigid_metrics(camera, limits, 'Frame camera')
    table_metrics = audit.rigid_metrics(align, limits, 'Scene table alignment')
    intrinsic = np.asarray(metadata['intrinsic_matrix'], dtype=np.float64)
    factor = np.asarray(metadata['factor_depth'], dtype=np.float64).reshape(-1)
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
            and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0
            and np.array_equal(intrinsic[2], [0., 0., 1.])
            and factor.size == 1 and np.isfinite(factor).all() and factor[0] > 0, 'Invalid per-frame calibration')
    transform = np.linalg.inv(np.matmul(align, camera))
    transform_metrics = audit.rigid_metrics(transform, limits, 'Camera-relative table transform')
    return {'object_pose_comparison': pose_audit, 'camera_pose': np.asarray(camera).tolist(),
            'cam0_wrt_table': np.asarray(align).tolist(), 'camera_rigid_metrics': camera_metrics,
            'table_alignment_rigid_metrics': table_metrics, 'table_transform_rigid_metrics': transform_metrics,
            'table_transform': transform.tolist(), 'intrinsic_matrix': intrinsic.tolist(), 'factor_depth': float(factor[0]),
            'object_ids_verified': list(xml_poses)}


def empty_preparation(frame, pose_audit):
    return {'schema': 14, 'frame': frame, 'input_count': 0, 'input_dtype': 'float32',
            'object_ids': pose_audit['object_ids_verified'], 'nms_count': 0,
            'nms_original_indices': [], 'nms_object_slots': [],
            'per_object_pre_limit_counts': [0] * len(OBJECTS), 'selection_min_confidence': None,
            'selected': [], 'selected_count': 0, 'geometry_preparation_executed': False,
            'reason': 'zero_generated_candidates', 'frame_geometry_audit': pose_audit}


def empty_oracle(preparation):
    return {'executed': False, 'reason': 'zero_generated_candidates', 'result': {
        'groups': [[] for _ in preparation['object_ids']], 'scores': [[] for _ in preparation['object_ids']],
        'collision': [[] for _ in preparation['object_ids']]}}


def load_frame_geometry(frame, annotation_path, meta_path, camera_path, align_path, limits):
    from scipy.io import loadmat
    poses, quaternions = audit.read_xml_poses(annotation_path)
    metadata = loadmat(meta_path, variable_names=['cls_indexes', 'poses', 'intrinsic_matrix', 'factor_depth'])
    cameras = np.load(camera_path, allow_pickle=False)
    require(cameras.shape == (256, 4, 4), 'Scene camera trajectory does not contain all 256 frames')
    camera = cameras[int(frame)]
    align = np.load(align_path, allow_pickle=False)
    evidence = validate_frame_poses(poses, metadata, camera, align, limits)
    evidence.update(frame=frame, xml_quaternions=quaternions)
    return poses, evidence


def prepare(candidates, frame, frame_geometry, models_root, api):
    """Keep shared model-root independent of new annotations and frame metadata."""
    poses_by_id, evidence = frame_geometry
    if len(candidates) == 0:
        return empty_preparation(frame, evidence), None
    np_api, o3d, Group, official, _, _ = api
    object_ids, poses = list(poses_by_id), list(poses_by_id.values())
    models, raw_counts = [], []
    for object_id in object_ids:
        cloud = o3d.io.read_point_cloud(str(models_root / ('models/%03d/nontextured.ply' % object_id)))
        points = np.asarray(cloud.points)
        require(points.ndim == 2 and points.shape[1] == 3 and len(points) and np.isfinite(points).all(), 'Invalid shared model cloud')
        raw_counts.append(len(points))
        models.append(official.voxel_sample_points(points, .008))
        del cloud, points
    clipped = candidates.copy()
    clipped[clipped[:, 1] < 0, 1] = 0
    clipped[clipped[:, 1] > .1, 1] = .1
    table = official.create_table_points(1., 1., .05, dx=-.5, dy=-.5, dz=-.05, grid_size=.008)
    table = official.transform_points(table, np.asarray(evidence['table_transform']))
    groups, transformed, scene, selection = legacy.select_groups(clipped, models, poses, Group, official, np_api)
    selected = []
    for slot, rows in enumerate(groups):
        originals = legacy.exact_indices(clipped, rows, np_api)
        for local, (row, original) in enumerate(zip(rows, originals)):
            selected.append({'selection_index': len(selected), 'object_slot': slot, 'object_id': object_ids[slot],
                             'object_local_index': local, 'original_index': original,
                             'grasp_dtype': str(rows.dtype), 'grasp': row.tolist()})
    require(len({row['original_index'] for row in selected}) == len(selected), 'Duplicate original candidate provenance')
    preparation = dict(selection, schema=14, frame=frame, input_count=len(candidates), input_dtype=str(candidates.dtype),
        object_ids=object_ids, raw_model_point_counts=raw_counts, sampled_model_point_counts=[len(model) for model in models],
        table_points=len(table), table_transform=evidence['table_transform'], poses=[pose.tolist() for pose in poses],
        selected=selected, selected_count=len(selected), geometry_preparation_executed=True, frame_geometry_audit=evidence)
    state = {'groups': groups, 'models': models, 'transformed': transformed, 'poses': poses,
             'scene': np.concatenate([scene, table]), 'table': table, 'clipped': clipped,
             'obj_ids': object_ids, 'data_root': models_root}
    return preparation, state


def cache_names(count):
    names = {'binding.json'}
    for name in ['preparation.json', 'oracle.json', 'report.json', 'completion.json'] + [
            'grasps/grasp_%04d.json' % index for index in range(count)]:
        names.update((name, name + '.sha256.json'))
    return names


def check_layout(directory, allowed, complete=False, remove_temporaries=False):
    files = set()
    for path in directory.rglob('*'):
        name = path.relative_to(directory).as_posix()
        require(not path.is_symlink(), 'Symlink in immutable cache')
        if path.is_dir():
            require(any(target.startswith(name + '/') for target in allowed), 'Unexpected cache directory')
        elif name in allowed:
            files.add(name)
        else:
            temporary = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', name)
            require(temporary is not None and temporary.group(1) in allowed
                    and remove_temporaries and not complete, 'Unexpected or unsealed cache file: ' + name)
            require(path.resolve().is_relative_to(directory.resolve()), 'Temporary path escaped cache')
            path.unlink()
    require(not complete or files == allowed, 'Incomplete sealed cache membership')


def save_commit(path, value, binding, guard, read_only=False):
    payload = common.json_bytes(value)
    receipt = {'schema': 14, 'binding_fingerprint': binding['fingerprint'],
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    extra = (0 if path.exists() else len(payload))
    extra += 0 if common.receipt_path(path).exists() else len(common.json_bytes(receipt))
    guard.check(extra)
    common.commit(path, value, binding, read_only)
    guard.check()


def load_optional_commit(path, binding):
    # An intent without a payload is recoverable only by identical recomputation.
    return common.load_commit(path, binding) if path.exists() else None


def summarize(preparation, records, oracle, binding_fingerprint, frame, generation_outcome):
    require(preparation['schema'] == 14 and preparation['frame'] == frame, 'Wrong frame preparation')
    require(len(records) == preparation['selected_count'], 'Incomplete dynamic selected support')
    for item, record in zip(preparation['selected'], records):
        require(all(record.get(key) == value for key, value in item.items()), 'Geometry record identity changed')
    if preparation['input_count'] == 0:
        require(not records and oracle == empty_oracle(preparation), 'Invalid zero-candidate branch')
    else:
        require(preparation['selected_count'] > 0 and oracle['executed'] is True, 'Nonempty candidate set requires official oracle')
    # The unchanged official-style summarizer uses 50 prefixes, denominator K,
    # and all selected records; cutoff ties may exceed 50 and are not truncated.
    report = legacy.summarize(preparation, records, oracle['result'], binding_fingerprint)
    status = ('accepted_limited_training_view_geometry_baseline' if preparation['input_count'] else
              'skipped_insufficient_generation_input' if generation_outcome == 'insufficient_input' else
              'completed_empty_candidate_negative_result')
    report.update(schema=14, frame=frame, status=status, generation_outcome=generation_outcome,
        official_oracle_executed=oracle['executed'], official_oracle_exact_match=True if oracle['executed'] else None,
        geometry_preparation_executed=preparation['geometry_preparation_executed'],
        frame_pose_and_calibration_verified=True, candidate_count=preparation['input_count'],
        full_point_cloud_input=True, no_method_comparison=True, soft_ranking_used=False,
        reported_prefix_count=50, prefix_denominator_rule='K even when fewer than K selected records',
        official_cutoff_ties_retained=True, selected_count_may_exceed_50=True)
    report['limitations'] = [
        'Four preselected views of one previously used training scene; not held-out, calibration, test or generalization evidence.',
        'This is a full-point-cloud baseline only; no new training, soft reranking or method-improvement comparison.',
        'Candidates, camera/object poses and labels are frame-specific; no frame-0000 candidate or label substitution.',
        'The official 50th-score threshold retains ties; all selected records are scored, with official K denominators for 50 prefixes.',
        'A zero-candidate result is retained without invoking an official empty-input oracle.',
        'Object-geometry friction labels are not actual robot success or calibrated safety probabilities.',
        'The frozen contact-search false negative and public-mirror provenance limitations remain unresolved.'
    ]
    return report


GENERATION_BINARY = ('valid_points.npy', 'valid_pixel_indices.npy', 'sampled_valid_indices.npy',
                     'sampled_points.npy', 'sampled_pixel_indices.npy', 'forward_bundle.npz',
                     'candidates.npy', 'seed_provenance.npz')
GENERATION_JSON = ('input_commit.json', 'forward_commit.json', 'report.json')


def verify_generation_binding(root, config, protocol, frame, binding):
    base = dict(protocol['generation_shared_inputs'])
    base.update(protocol['generation_inputs'][frame])
    base.update(protocol['source_hashes'])
    numerical = binding['numerical_source_and_extension_hashes']
    prefixes = ('runtime/build/graspnet/', 'runtime/envs/vision/Lib/site-packages/pointnet2/',
                'runtime/envs/vision/Lib/site-packages/knn_pytorch/')
    require(bool(numerical) and all(name.startswith(prefixes) and Path(name).suffix.lower() in
            ('.py', '.pyd', '.dll', '.cu', '.cpp', '.h') for name in numerical), 'Invalid generation numerical inventory')
    actual_numerical_names = {path.relative_to(root).as_posix() for prefix in prefixes
        for path in common.safe_path(root, prefix.rstrip('/')).rglob('*')
        if path.is_file() and path.suffix in ('.py', '.pyd', '.dll', '.cu', '.cpp', '.h')}
    require(set(numerical) == actual_numerical_names, 'Generation numerical inventory is incomplete or changed')
    require(all(name not in base or base[name] == value for name, value in numerical.items()), 'Conflicting numerical source hash')
    expected_files = dict(base, **numerical)
    require(binding['files'] == expected_files, 'Generation source binding membership differs')
    for name, checksum in expected_files.items():
        require(common.digest(common.safe_path(root, name)) == checksum, 'Generation source changed: ' + name)
    extensions = binding['resolved_extensions']
    require(set(extensions) == {'pointnet2._ext', 'knn_pytorch.knn_pytorch'}
            and all(value['path'] in numerical and value['sha256'] == numerical[value['path']]
                    and Path(value['path']).suffix.lower() == '.pyd' for value in extensions.values()),
            'Generation extension binding differs')
    expected = {'schema': 14, 'input_group': 'generation_inputs', 'frame': frame,
        'protocol_sha256': common.digest(config), 'files': expected_files, 'runtime_name': 'vision',
        'runtime': protocol['runtimes']['vision'],
        'interpreter_sha256': common.digest(common.safe_path(root, protocol['interpreters']['vision'])),
        'resolved_extensions': extensions, 'model_runtime': protocol['model_runtime'],
        'numerical_source_and_extension_hashes': numerical, 'precision': 'float32', 'autocast': False,
        'uses_ground_truth_labels': False, 'uses_object_geometry': False, 'training_updates': 0}
    expected['fingerprint'] = common.fingerprint(expected)
    require(binding == expected, 'Generation binding contract/fingerprint differs')


def verify_generation(root, config, protocol, frame):
    directory = common.frame_directory(root, protocol, frame, 'generation')
    binding = common.read(directory / 'binding.json')
    verify_generation_binding(root, config, protocol, frame, binding)
    completion = common.load_commit(directory / 'completion.json', binding)
    require(completion['schema'] == 14 and completion['frame'] == frame and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == binding['fingerprint'], 'All four generations must be sealed first')
    names = {'binding.json'}
    for name in GENERATION_BINARY + GENERATION_JSON:
        names.update((name, name + '.sha256.json'))
    require(set(completion['outputs']) == names, 'Generation manifest membership differs')
    check_layout(directory, names | {'completion.json', 'completion.json.sha256.json'}, complete=True)
    for name, checksum in completion['outputs'].items():
        require(common.digest(directory / name) == checksum, 'Sealed generation output changed: ' + name)
    for name in GENERATION_BINARY:
        common.verify_bytes(directory / name, binding)
    commits = {name: common.load_commit(directory / name, binding) for name in GENERATION_JSON}
    report, forward, inputs = (commits[name] for name in ('report.json', 'forward_commit.json', 'input_commit.json'))
    for value in commits.values():
        require(value['schema'] == 14 and value['frame'] == frame and value['binding_fingerprint'] == binding['fingerprint'],
                'Generation committed identity differs')
    require(report['status'] == 'completed' and inputs['status'] == 'inputs_committed'
            and report['input_information'] == inputs['information'], 'Generation input/report provenance differs')
    require(inputs['files'] == {name: common.digest(directory / name) for name in GENERATION_BINARY[:5]},
            'Generation input commit file inventory differs')
    require(all(completion[key] == report[key] for key in ('candidate_count', 'candidates_sha256', 'outcome',
            'input_status', 'forward_calls_committed')) and completion['report_sha256'] == common.digest(directory / 'report.json'),
            'Generation completion/report differ')
    require(report['outcome'] in ('generated', 'empty_candidates', 'insufficient_input')
            and report['input_status'] in ('ready', 'insufficient_input'), 'Unsupported generation outcome')
    expected_calls = 0 if report['input_status'] == 'insufficient_input' else 1
    require(report['forward_calls_committed'] == forward['forward_calls_committed'] == expected_calls
            and forward['evidence']['model_forward_calls'] == expected_calls
            and forward['forward_bundle_sha256'] == common.digest(directory / 'forward_bundle.npz'), 'Forward seal differs')
    require(report['uses_ground_truth_labels'] is False and report['uses_object_geometry'] is False
            and report['training_updates'] == 0, 'Generation crossed the label/training boundary')
    candidates = validate_candidates(np.load(directory / 'candidates.npy', allow_pickle=False), report['candidate_count'])
    require(report['candidates_sha256'] == common.digest(directory / 'candidates.npy')
            and (len(candidates) > 0) == (report['outcome'] == 'generated')
            and (report['outcome'] == 'insufficient_input') == (expected_calls == 0), 'Generated count/outcome mismatch')
    files = {path.relative_to(root).as_posix(): common.digest(path)
             for path in directory.rglob('*') if path.is_file()}
    return {'completion_sha256': common.digest(directory / 'completion.json'),
            'candidate_count': len(candidates), 'outcome': report['outcome'], 'input_status': report['input_status'],
            'files': files, 'candidate_path': directory / 'candidates.npy', 'input_information': report['input_information']}


def make_binding(root, config, protocol, frame):
    # Do not even hash new object annotations until every predicted view is sealed.
    generations = {}
    for selected_frame in protocol['frames']:
        generations[selected_frame] = verify_generation(root, config, protocol, selected_frame)
    extras = {name: checksum for value in generations.values() for name, checksum in value['files'].items()}
    assets = common.verify_assets(root, config, protocol)
    extras.update(assets)
    runtime = legacy.common.runtime_binding(protocol)
    extras.update(runtime['numerical_dependency_files'])
    extras.update({'runtime/envs/grasp_eval/Lib/site-packages/graspnetAPI/' + name: checksum
                   for name, checksum in runtime['api_files'].items()})
    binding = common.make_binding(root, config, protocol, input_group='geometry_inputs', frame=frame,
                                  runtime_name='grasp_eval', extra_inputs=extras)
    return binding, generations


def validate_protocol(protocol, frame):
    require(protocol['schema'] == 14 and protocol['frames'] == list(FRAMES) and frame in FRAMES, 'Wrong frozen four-view selection')
    require(protocol['models_root'] == 'data/raw/grasp_geometry_v10', 'Shared model root changed')
    require(protocol['scene_id'] == 3 and protocol['camera'] == 'realsense'
            and protocol['object_ids'] == OBJECTS, 'Wrong training scene/camera/object set')
    require(protocol['scoring'] == {key: value for key, value in legacy.SCORING.items() if key != 'gpu_used'},
            'Frozen official scoring semantics changed')
    require(protocol['resources']['geometry'] == {'cpu_threads': 2, 'invocation_soft_seconds': 1800,
            'rss_soft_bytes': 8 * 1024 ** 3, 'output_limit_bytes': 1024 ** 3,
            'disk_reserve_bytes': 60 * 1024 ** 3}, 'Scoring resource contract changed')
    require(protocol['pose_limits'] == protocol['geometry_audit'], 'Pose tolerance contracts disagree')


def frame_geometry(root, protocol, frame):
    paths = protocol['frame_inputs'][frame]
    return load_frame_geometry(frame, common.safe_path(root, paths['annotation']), common.safe_path(root, paths['meta']),
        common.safe_path(root, protocol['camera_poses_path']), common.safe_path(root, protocol['cam0_wrt_table_path']), protocol['pose_limits'])


def completed_report(binding, directory, frame, generation, generations):
    require(common.read(directory / 'binding.json') == binding, 'Completed geometry binding changed')
    completion = common.load_commit(directory / 'completion.json', binding)
    require(completion['schema'] == 14 and completion['frame'] == frame and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == binding['fingerprint'], 'Invalid geometry completion')
    require(completion['all_generation_completion_sha256'] == {
            key: value['completion_sha256'] for key, value in generations.items()}, 'Completed four-generation prerequisite changed')
    prep = common.load_commit(directory / 'preparation.json', binding)
    require(type(prep['selected_count']) is int and prep['selected_count'] >= 0
            and completion['expected_grasps'] == prep['selected_count']
            and prep['input_count'] == generation['candidate_count'], 'Completed candidate counts differ')
    names = cache_names(prep['selected_count'])
    check_layout(directory, names, complete=True)
    require(set(completion['outputs']) == names - {'completion.json', 'completion.json.sha256.json'}, 'Geometry completion membership differs')
    require(all(common.digest(directory / name) == checksum for name, checksum in completion['outputs'].items()),
            'Sealed geometry output changed')
    records = [common.load_commit(directory / ('grasps/grasp_%04d.json' % index), binding) for index in range(prep['selected_count'])]
    oracle = common.load_commit(directory / 'oracle.json', binding)
    report = summarize(prep, records, oracle, binding['fingerprint'], frame, generation['outcome'])
    require(common.load_commit(directory / 'report.json', binding) == report
            and completion['report_sha256'] == common.digest(directory / 'report.json'), 'Completed report is not reproducible')
    return report


def run(root, config, frame, max_new_grasps=None, check_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    require(config.is_relative_to(root), 'Config outside project')
    require(max_new_grasps is None or type(max_new_grasps) is int and max_new_grasps >= 0, 'Invalid new-grasp budget')
    require(not (check_only and max_new_grasps is not None), 'Read-only mode cannot request new scoring')
    protocol = common.read(config)
    validate_protocol(protocol, frame)
    guard = common.Guard(root, protocol, phase='geometry')
    guard.check()
    binding, generations = make_binding(root, config, protocol, frame)
    generation = generations[frame]
    directory = common.frame_directory(root, protocol, frame, 'geometry')
    guard.check()

    def result(status, fresh, reused, committed, expected, oracle_executed=False, preparations=0):
        return {'status': status, 'frame': frame, 'new_grasps': fresh, 'reused_grasps': reused,
                'committed_grasps': committed, 'expected_grasps': expected,
                'new_geometry_computations': fresh, 'new_geometry_preparations': preparations,
                'official_oracle_executed_this_invocation': oracle_executed, 'new_model_forward_calls': 0,
                'resources': guard.check()}

    def verify():
        report = completed_report(binding, directory, frame, generation, generations)
        count = report['selected_count']
        return dict(result('verified_completed_read_only', 0, count, count, count),
                    report_status=report['status'], report_sha256=common.digest(directory / 'report.json'),
                    completion_sha256=common.digest(directory / 'completion.json'))

    if (directory / 'completion.json').exists():
        return verify()
    require(not check_only, 'Check-only requires a complete geometry seal')
    with common.lock(root / 'runtime/locks/grasp_multiview_v14' / (frame + '_geometry')):
        if (directory / 'completion.json').exists():
            return verify()
        prep_path = directory / 'preparation.json'
        if not (directory / 'binding.json').exists():
            check_layout(directory, {'binding.json'}, remove_temporaries=True)
        if not prep_path.exists():
            check_layout(directory, cache_names(0), remove_temporaries=True)
        guard.check(0 if (directory / 'binding.json').exists() else len(common.json_bytes(binding)))
        common.once(directory / 'binding.json', binding)
        prep = load_optional_commit(prep_path, binding)
        api, state, dex = None, None, None
        preparations = 0

        def rebuild():
            nonlocal api, state, dex, preparations
            candidates = validate_candidates(np.load(generation['candidate_path'], allow_pickle=False), generation['candidate_count'])
            observed_geometry = frame_geometry(root, protocol, frame)
            info = generation['input_information']
            require(observed_geometry[1]['intrinsic_matrix'] == info['intrinsic_matrix']
                    and observed_geometry[1]['factor_depth'] == info['factor_depth'], 'Generation and scoring calibration disagree')
            api = legacy.imports() if len(candidates) else None
            rebuilt, state = prepare(candidates, frame, observed_geometry, common.safe_path(root, protocol['models_root']), api)
            if state is not None:
                dex = legacy.LazyModels(state['data_root'], state['obj_ids'], api[3].load_dexnet_model)
            preparations += 1
            guard.check()
            return rebuilt

        def ensure_state():
            if state is None:
                require(rebuild() == prep, 'Rebuilt geometry differs from committed preparation')

        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            if prep is None:
                prep = rebuild()
                save_commit(prep_path, prep, binding, guard)
            require(prep['schema'] == 14 and prep['frame'] == frame
                    and prep['input_count'] == generation['candidate_count'], 'Prepared frame/count changed')
            count = prep['selected_count']
            require(type(count) is int and 0 <= count <= min(generation['candidate_count'], 10 * len(OBJECTS))
                    and len(prep['selected']) == count, 'Invalid dynamic selected support')
            check_layout(directory, cache_names(count), remove_temporaries=True)
            records, fresh, reused = [], 0, 0
            for index, item in enumerate(prep['selected']):
                guard.check()
                require(item['selection_index'] == index, 'Non-contiguous selected indices')
                path = directory / ('grasps/grasp_%04d.json' % index)
                record = load_optional_commit(path, binding)
                if record is None:
                    require(not any((directory / ('grasps/grasp_%04d.json' % later)).exists() for later in range(index + 1, count)),
                            'Non-contiguous committed score prefix')
                    if max_new_grasps is not None and fresh >= max_new_grasps:
                        return result('partial', fresh, reused, len(records), count, preparations=preparations)
                    ensure_state()
                    record = legacy.score_one(item, state, dex, api)
                    guard.check()
                    save_commit(path, record, binding, guard)
                    fresh += 1
                else:
                    reused += 1
                require(all(record.get(key) == value for key, value in item.items()), 'Committed score provenance changed')
                records.append(record)
            oracle_path = directory / 'oracle.json'
            oracle = load_optional_commit(oracle_path, binding)
            oracle_executed = False
            if oracle is None:
                if prep['input_count']:
                    ensure_state()
                    oracle = {'executed': True, 'result': legacy.oracle_result(state, api, dex)}
                    oracle_executed = True
                    guard.check()
                else:
                    oracle = empty_oracle(prep)
                save_commit(oracle_path, oracle, binding, guard)
            report = summarize(prep, records, oracle, binding['fingerprint'], frame, generation['outcome'])
            save_commit(directory / 'report.json', report, binding, guard)
            names = cache_names(count) - {'completion.json', 'completion.json.sha256.json'}
            completion = {'schema': 14, 'status': 'completed', 'frame': frame,
                'binding_fingerprint': binding['fingerprint'], 'expected_grasps': count,
                'report_sha256': common.digest(directory / 'report.json'),
                'all_generation_completion_sha256': {key: value['completion_sha256'] for key, value in generations.items()},
                'outputs': {name: common.digest(directory / name) for name in sorted(names)}}
            save_commit(directory / 'completion.json', completion, binding, guard)
            completed_report(binding, directory, frame, generation, generations)
            return dict(result('completed', fresh, reused, count, count, oracle_executed, preparations),
                        report_status=report['status'], report_sha256=common.digest(directory / 'report.json'),
                        completion_sha256=common.digest(directory / 'completion.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=common.CONFIG)
    parser.add_argument('--frame', required=True, choices=FRAMES)
    parser.add_argument('--max-new-grasps', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.config, args.frame, args.max_new_grasps, args.check_only), sort_keys=True))


if __name__ == '__main__':
    main()
