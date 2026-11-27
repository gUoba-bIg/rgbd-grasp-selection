"""Single-frame, resumable object geometry regression; not dataset performance."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_grasp_observed_collision_v9 as common

read, digest, fingerprint = common.read, common.digest, common.fingerprint
require, atomic, once, lock = common.require, common.atomic, common.once, common.lock
OUT = ROOT / 'results/P3/grasp_geometry_v10'
REPORT = ROOT / 'results/P3/grasp_geometry_v10.json'
CONFIG = ROOT / 'configs/grasp_geometry_v10.json'
SCORING = {
    'preserve_candidate_dtype': True, 'width_clip_m': [0., .1], 'nms_translation_m': .03,
    'nms_rotation_degrees': 30., 'model_voxel_size_m': .008,
    'assignment': 'closest transformed sampled model point; candidate object_id is ignored',
    'per_object_limit': 10, 'global_score_rank': 50, 'retain_cutoff_ties': True,
    'final_sort': 'numpy.argsort(-confidence) with the frozen NumPy version',
    'table': {'size_m': [1., 1., .05], 'offset_m': [-.5, -.5, -.05], 'grid_size_m': .008,
              'transform': 'inverse(cam0_wrt_table @ camera_poses[frame_id])'},
    'collision': {'height_m': .02, 'finger_width_m': .01, 'base_depth_m': .02,
                  'outlier_m': .05, 'empty_threshold_points': 10, 'empty_is_collision': True},
    'friction_scoring_order': [1.2, 1., .8, .6, .4, .2],
    'friction_reporting_order': [.2, .4, .6, .8, 1., 1.2], 'reported_prefixes': 50,
    'contact_diagnostic_check_approach': False,
    'reference': 'Unmodified pinned official eval_grasp; exact per-object row/order, collision and discrete score equality',
    'new_training_updates': 0, 'gpu_used': False}


class Monitor(common.Monitor):
    def check(self):
        require(self.error is None, self.error or '')
        require(time.monotonic() - self.started <= self.protocol['invocation_soft_seconds'], 'Time limit exceeded')
        require(shutil.disk_usage(ROOT).free >= self.protocol['disk_reserve_bytes'], 'Disk reserve unavailable')
        size = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) if self.out.exists() else 0
        if REPORT.is_file():
            size += REPORT.stat().st_size
        require(size <= self.protocol['output_limit_bytes'], 'Output budget exceeded')


def relative(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Path escaped project')
    return path


def validate_audits(manifest, audit, geometry, manifest_hash, audit_hash, protocol_hash):
    require(manifest.get('status') == 'verified_geometry_assets', 'Asset manifest not complete')
    require(audit.get('status') == 'passed_integrity_only', 'Asset integrity audit not passed')
    require(manifest.get('audit_sha256') == audit_hash, 'Manifest audit SHA differs')
    require(audit.get('files') == manifest.get('files') and audit.get('binding') == manifest.get('binding'),
            'Asset audit files/binding differs')
    require(geometry.get('status') == 'passed', 'Geometry audit not passed')
    require(geometry.get('manifest_sha256') == manifest_hash, 'Geometry manifest binding differs')
    require(geometry.get('asset_audit_sha256') == audit_hash, 'Geometry asset audit binding differs')
    require(geometry.get('protocol_sha256') == protocol_hash, 'Geometry protocol binding differs')


def make_binding(protocol, config):
    hashes = dict(protocol['input_hashes'], **protocol['source_hashes'])
    for name, expected in hashes.items():
        require(digest(relative(name)) == expected, 'Frozen source/input changed: ' + name)
    manifest_path = relative(protocol['asset_manifest_path'])
    audit_path = relative(protocol['asset_audit_path'])
    # Asset gate and file metadata are checked against the independently accepted audit.
    audit, manifest = read(audit_path), read(manifest_path)
    geometry_path = relative(protocol['geometry_audit_path'])
    geometry = read(geometry_path)
    validate_audits(manifest, audit, geometry, digest(manifest_path), digest(audit_path), digest(config))
    data_root = relative(protocol['data_root'])
    assets = {}
    for name, item in manifest['files'].items():
        path = (data_root / name).resolve()
        require(path.is_relative_to(data_root.resolve()), 'Asset path escaped root')
        expected = item['sha256']
        require(digest(path) == expected, 'Asset bytes changed: ' + name)
        assets[name] = expected
    expected_assets = {'models/%03d/%s' % (i, filename) for i in protocol['object_ids']
                       for filename in ('nontextured.ply', 'textured.obj', 'textured.sdf')}
    expected_assets.update('scenes/scene_0003/realsense/' + name for name in
                           ('annotations/0000.xml', 'camera_poses.npy', 'cam0_wrt_table.npy', 'meta/0000.mat'))
    require(set(assets) == expected_assets, 'Unexpected fixed-frame asset set')
    own = {name: digest(ROOT / name) for name in
           ('scripts/verify_grasp_geometry_v10.py', 'scripts/verify_grasp_observed_collision_v9.py')}
    return {'schema': 10, 'protocol_sha256': digest(config), 'frozen_hashes': hashes,
            'own_sources': own, 'asset_manifest_sha256': digest(manifest_path),
            'asset_audit_sha256': digest(audit_path), 'geometry_audit_sha256': digest(geometry_path), 'assets': assets,
            'runtime': common.runtime_binding(protocol)}


def imports():
    import numpy as np
    import open3d as o3d
    from graspnetAPI import GraspGroup
    from graspnetAPI.utils import eval_utils as official
    from graspnetAPI.utils.config import get_config
    from graspnetAPI.utils.xmlhandler import xmlReader
    require('torch' not in sys.modules and not hasattr(np, 'int'), 'Unexpected runtime/NumPy alias')
    return np, o3d, GraspGroup, official, get_config, xmlReader


def exact_indices(source, selected, np):
    """Equivalent identical rows are labelled by lowest remaining source index."""
    result, used = [], set()
    for row in selected:
        hits = np.flatnonzero(np.all(source == row, axis=1))
        hits = [int(i) for i in hits if int(i) not in used]
        require(bool(hits), 'Selected row absent from original candidates')
        result.append(hits[0])
        used.add(hits[0])
    return result


def select_groups(clipped, models, poses, Group, official, np):
    """Exact eval_grasp preprocessing, including its hard-coded 50th-score tie rule."""
    nms = Group(clipped.copy()).nms(.03, 30.0 / 180 * np.pi)
    transformed = [official.transform_points(model, pose) for model, pose in zip(models, poses)]
    scene = np.concatenate(transformed, axis=0)
    seg = np.concatenate([i * np.ones(len(m), dtype=np.int32) for i, m in enumerate(transformed)])
    indices = official.compute_closest_points(nms.translations, scene)
    assignments = seg[indices]
    pre = []
    for i in range(len(models)):
        group = nms[assignments == i]
        group.sort_by_score()
        pre.append(group[:10].grasp_group_array)
    all_rows = np.vstack(pre)
    require(len(all_rows) > 0, 'No candidates after official preprocessing')
    order = np.argsort(all_rows[:, 0])[::-1]
    threshold = all_rows[order[min(49, len(order) - 1)], 0]
    groups = [rows[rows[:, 0] >= threshold] for rows in pre]
    return groups, transformed, scene, {'nms_count': len(nms),
        'nms_original_indices': exact_indices(clipped, nms.grasp_group_array, np),
        'nms_object_slots': assignments.tolist(), 'per_object_pre_limit_counts': [len(x) for x in pre],
        'selection_min_confidence': float(threshold)}


def prepare(protocol, api):
    np, o3d, Group, official, get_config, xmlReader = api
    data_root = relative(protocol['data_root'])
    base = data_root / ('scenes/scene_%04d/%s' % (protocol['scene_id'], protocol['camera']))
    vectors = xmlReader(str(base / ('annotations/%04d.xml' % protocol['frame_id']))).getposevectorlist()
    parsed = [official.parse_posevector(v) for v in vectors]
    obj_ids, poses = [x[0] for x in parsed], [x[1] for x in parsed]
    require(len(obj_ids) == 9 and sorted(obj_ids) == sorted(protocol['object_ids']), 'Unexpected fixed-frame objects')
    models, raw_counts = [], []
    for obj_id in obj_ids:
        cloud = o3d.io.read_point_cloud(str(data_root / ('models/%03d/nontextured.ply' % obj_id)))
        points = np.asarray(cloud.points)
        require(points.ndim == 2 and points.shape[1] == 3 and len(points) and np.isfinite(points).all(), 'Invalid model points')
        raw_counts.append(len(points))
        models.append(official.voxel_sample_points(points, .008))
        del cloud, points
    candidates = np.load(relative(protocol['candidate_path']), allow_pickle=False)
    require(candidates.shape == (protocol['candidate_count'], 17) and np.isfinite(candidates).all(), 'Invalid candidates')
    clipped = candidates.copy()  # Do not promote float32 before width comparisons.
    clipped[clipped[:, 1] < 0, 1] = 0
    clipped[clipped[:, 1] > .1, 1] = .1
    camera = np.load(base / 'camera_poses.npy', allow_pickle=False)[protocol['frame_id']]
    align = np.load(base / 'cam0_wrt_table.npy', allow_pickle=False)
    table_transform = np.linalg.inv(np.matmul(align, camera))
    table = official.create_table_points(1., 1., .05, dx=-.5, dy=-.5, dz=-.05, grid_size=.008)
    table = official.transform_points(table, table_transform)
    groups, transformed, scene, selection = select_groups(clipped, models, poses, Group, official, np)
    selected = []
    for slot, rows in enumerate(groups):
        for local, (row, original) in enumerate(zip(rows, exact_indices(clipped, rows, np))):
            selected.append({'selection_index': len(selected), 'object_slot': slot,
                'object_id': obj_ids[slot], 'object_local_index': local,
                'original_index': original, 'grasp_dtype': str(rows.dtype), 'grasp': row.tolist()})
    prep = dict(selection, schema=10, input_count=len(candidates), input_dtype=str(candidates.dtype),
        object_ids=obj_ids, raw_model_point_counts=raw_counts, sampled_model_point_counts=[len(m) for m in models],
        table_points=len(table), table_transform=table_transform.tolist(), poses=[p.tolist() for p in poses],
        selected=selected, selected_count=len(selected))
    state = {'groups': groups, 'models': models, 'transformed': transformed, 'poses': poses,
             'scene': np.concatenate([scene, table]), 'table': table, 'clipped': clipped,
             'obj_ids': obj_ids, 'data_root': data_root}
    return prep, state


class LazyModels:
    """Single-entry cache; upstream callers may still retain the previous model.

    Replacing the entry preserves upstream ordering, but two models can coexist
    temporarily through caller references. RSS monitoring remains authoritative.
    """
    def __init__(self, root, ids, loader):
        self.root, self.ids, self.loader = root, ids, loader
        self.current, self.model = None, None

    def __getitem__(self, slot):
        if self.current != slot:
            self.model = None
            gc.collect()
            self.model = self.loader(str(self.root / ('models/%03d/textured' % self.ids[slot])))
            self.current = slot
        return self.model


def score_configs(api):
    np, _, _, official, get_config, _ = api
    config, values, result = get_config(), np.array([1.2, 1., .8, .6, .4, .2]), {}
    for coefficient in values:
        coefficient = round(coefficient, 2)
        config['metrics']['force_closure']['friction_coef'] = coefficient
        result[coefficient] = official.GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])
    return values, result


def score_one(item, state, dex, api):
    np, _, Group, official, _, _ = api
    slot = item['object_slot']
    row = np.asarray([item['grasp']], dtype=item['grasp_dtype'])
    # dexnet_models are not dereferenced by collision_detection except for local assignment.
    collision, empty, grasps = official.collision_detection([row], [state['transformed'][slot]],
        [None], [state['poses'][slot]], state['scene'], outlier=.05, empty_thresh=10, return_dexgrasps=True)
    collided, emptied, grasp = bool(collision[0][0]), bool(empty[0][0]), grasps[0][0]
    diagnostic = {'executed': False, 'contacts_found': None, 'contacts': []}
    score = -1.
    if not collided and grasp is not None:
        model = dex[slot]
        values, configs = score_configs(api)
        score = float(official.get_grasp_score(grasp, model, values, configs))
        found, contacts = grasp.close_fingers(model, check_approach=False, vis=False)
        diagnostic = {'executed': True, 'contacts_found': bool(found), 'contacts': []}
        for contact in (contacts or []):
            if contact is None:
                diagnostic['contacts'].append(None)
                continue
            point = np.asarray(contact.point)
            normal = None if contact.normal is None else np.asarray(contact.normal)
            grid = model.sdf.transform_pt_obj_to_grid(point)
            value = float(model.sdf[grid])
            require(np.isfinite(point).all() and math.isfinite(value) and
                    (normal is None or np.isfinite(normal).all()), 'Nonfinite contact diagnostic')
            diagnostic['contacts'].append({'point_object_m': point.tolist(),
                'normal': None if normal is None else normal.tolist(), 'sdf_value_m': value})
    require(score in (-1., .2, .4, .6, .8, 1., 1.2), 'Unexpected friction score')
    return dict(item, collision=collided, empty=emptied, score=score, contact_diagnostic=diagnostic)


def receipt_path(path):
    return path.with_suffix('.sha256.json')


def commit(path, value, binding):
    once(path, value)
    once(receipt_path(path), {'sha256': digest(path), 'binding_fingerprint': binding})


def load_commit(path, binding):
    receipt = receipt_path(path)
    require(path.exists() or not receipt.exists(), 'Missing committed cache payload: ' + str(path))
    if not receipt.exists():
        # A payload written before an interrupted receipt is not committed. Recompute it
        # and require exact equality in once(); never trust or delete the orphan payload.
        return None
    record = read(receipt)
    require(record == {'sha256': digest(path), 'binding_fingerprint': binding}, 'Cache receipt differs')
    return read(path)


def oracle_result(state, api, dex):
    np, _, Group, official, get_config, _ = api
    groups, scores, collision = official.eval_grasp(Group(state['clipped'].copy()), state['models'], dex,
        state['poses'], get_config(), table=state['table'], voxel_size=.008, TOP_K=50)
    return {'groups': [x.tolist() for x in groups], 'scores': [x.tolist() for x in scores],
            'collision': [np.asarray(x, dtype=bool).tolist() for x in collision]}


def summarize(prep, records, oracle, binding):
    import numpy as np
    require(len(records) == prep['selected_count'], 'Incomplete selected candidate scores')
    require(len({r['original_index'] for r in records}) == len(records), 'Duplicate evaluated original index')
    require(all(r['score'] in (-1., .2, .4, .6, .8, 1., 1.2) and
                (not r['collision'] or r['score'] == -1.) for r in records), 'Invalid score/collision semantics')
    all_groups, scores, collision = [], [], []
    for slot in range(len(prep['object_ids'])):
        rows = [x for x in records if x['object_slot'] == slot]
        all_groups.append([r['grasp'] for r in rows])
        scores.append([r['score'] for r in rows])
        collision.append([r['collision'] for r in rows])
    require(oracle == {'groups': all_groups, 'scores': scores, 'collision': collision}, 'Official oracle differs')
    confidence = np.asarray([x['grasp'][0] for x in records], dtype=prep['input_dtype'])
    order = np.argsort(-confidence)
    ordered = [records[int(i)] for i in order]
    quality = np.asarray([x['score'] for x in ordered])
    friction = [.2, .4, .6, .8, 1., 1.2]
    matrix = np.zeros((50, 6))
    for j, coefficient in enumerate(friction):
        for k in range(50):
            prefix = quality[:k + 1]
            matrix[k, j] = np.sum((prefix <= coefficient) & (prefix > 0)) / (k + 1)
    selected_ids = {x['original_index'] for x in records}
    nms_ids = set(prep['nms_original_indices'])
    tracking = [{'original_index': i, 'reason': 'evaluated' if i in selected_ids else
                 ('official_object_top10_or_global_score_threshold' if i in nms_ids else 'official_nms')}
                for i in range(prep['input_count'])]
    return {'schema': 10, 'status': 'accepted_limited_single_frame_geometry_engineering',
        'binding_fingerprint': binding, 'dataset_ap_computed': False, 'robot_success_rate_computed': False,
        'gpu_used': False, 'new_training_steps': 0, 'original_candidate_count': prep['input_count'],
        'nms_count': prep['nms_count'], 'selected_count': len(records),
        'collision_including_empty_count': sum(x['collision'] for x in records),
        'empty_count': sum(x['empty'] for x in records),
        'positive_friction_score_count': sum(x['score'] > 0 for x in records),
        'noncollision_contact_search_failure_count': sum(x['contact_diagnostic']['executed'] and
              not x['contact_diagnostic']['contacts_found'] for x in records),
        'noncollision_found_contacts_no_grid_force_closure_count': sum(x['score'] < 0 and
              x['contact_diagnostic']['contacts_found'] is True for x in records),
        'official_oracle_exact_match': True, 'friction_grid_ascending': friction,
        'single_frame_prefix_pass_fraction_50x6': matrix.tolist(),
        'single_frame_prefix_matrix_mean': float(matrix.mean()),
        'ordered_records': ordered, 'all_original_candidate_tracking': tracking,
        'limitations': ['One existing training-scene frame; not scene/split AP or independent test evidence.',
            'V8 segmentation model did not generate these candidates.',
            'Unmodified V7 axial-sphere contact-search false negative remains unresolved.',
            'Object geometry scoring is not robot safety or actual grasp success probability.',
            'Public mirror source lacks a separately published official asset archive digest.']}


def expected_cache_names(count):
    names = {'binding.json', 'preparation.json', 'preparation.sha256.json', 'oracle.json',
             'oracle.sha256.json', 'last_invocation.json', 'completion.json'}
    for i in range(count):
        names.update(('grasps/grasp_%04d.json' % i, 'grasps/grasp_%04d.sha256.json' % i))
    return names


def legitimate_temporary(name, allowed):
    match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', name)
    return bool(match and match.group(1) in allowed)


def check_layout(out, names, complete=False):
    actual = set()
    for path in out.rglob('*'):
        name = path.relative_to(out).as_posix()
        if path.is_dir():
            require(name == 'grasps' and any(n.startswith('grasps/') for n in names), 'Unexpected cache directory')
        elif name in names:
            actual.add(name)
        else:
            require(legitimate_temporary(name, names), 'Unexpected cache file: ' + name)
    require(not complete or actual == names, 'Incomplete completed cache membership')


def check_complete(binding, out=OUT, report=REPORT):
    fp = fingerprint(binding)
    completion = read(out / 'completion.json')
    require(completion['status'] == 'completed', 'Invalid completion status')
    require(completion['binding_fingerprint'] == fp, 'Completed binding differs')
    for name, value in completion['outputs'].items():
        path = relative(name)
        require(digest(path) == value, 'Completed output changed: ' + name)
    prep = load_commit(out / 'preparation.json', fp)
    require(completion['expected_grasps'] == prep['selected_count'], 'Completed candidate count differs')
    names = expected_cache_names(prep['selected_count'])
    check_layout(out, names, complete=True)
    expected_outputs = {(out / name).relative_to(ROOT).as_posix() for name in
                        names - {'binding.json', 'completion.json'}} | {report.relative_to(ROOT).as_posix()}
    require(set(completion['outputs']) == expected_outputs, 'Completed output manifest membership differs')
    records = [load_commit(out / 'grasps' / ('grasp_%04d.json' % i), fp) for i in range(prep['selected_count'])]
    require(all(r is not None for r in records), 'Missing completed candidate')
    for selected, record in zip(prep['selected'], records):
        require(all(record.get(k) == v for k, v in selected.items()), 'Candidate identity changed')
    oracle = load_commit(out / 'oracle.json', fp)
    require(read(report) == summarize(prep, records, oracle, fp), 'Completed report not reproducible')
    return read(report)


def run(args):
    protocol = read(args.config)
    require(protocol['schema'] == 10 and protocol['cpu_threads'] == 2, 'Unsupported protocol')
    require(protocol['scoring'] == SCORING, 'Scoring protocol differs from pinned implementation')
    require((protocol['scene_id'], protocol['camera'], protocol['frame_id'], protocol['candidate_count']) ==
            (3, 'realsense', 0, 236), 'Unexpected frame/input scope')
    binding = make_binding(protocol, args.config)
    fp = fingerprint(binding)
    if args.check_only:
        require((OUT / 'binding.json').is_file() and (OUT / 'completion.json').is_file(),
                'No completed output to check')
        require(read(OUT / 'binding.json') == binding, 'Completed binding differs')
        report = check_complete(binding, OUT, REPORT)
        print(json.dumps({'status': 'verified_completed_read_only', 'selected_count': report['selected_count']}), flush=True)
        return 0
    with lock(ROOT / 'runtime/grasp_geometry_v10/lock'):
        OUT.mkdir(parents=True, exist_ok=True)
        if not (OUT / 'binding.json').exists():
            check_layout(OUT, {'binding.json'})
        once(OUT / 'binding.json', binding)
        if (OUT / 'completion.json').exists():
            report = check_complete(binding, OUT, REPORT)
            print(json.dumps({'status': 'verified_completed_read_only', 'selected_count': report['selected_count']}), flush=True)
            return 0
        api = imports()
        monitor = Monitor(protocol, OUT)
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2):
                prep, state = prepare(protocol, api)
                commit(OUT / 'preparation.json', prep, fp)
                check_layout(OUT, expected_cache_names(prep['selected_count']))
                monitor.check()
                dex = LazyModels(state['data_root'], state['obj_ids'], api[3].load_dexnet_model)
                records, fresh, reused = [], 0, 0
                for item in prep['selected']:
                    path = OUT / 'grasps' / ('grasp_%04d.json' % item['selection_index'])
                    record = load_commit(path, fp)
                    if record is None:
                        if args.max_new_grasps is not None and fresh >= args.max_new_grasps:
                            atomic(OUT / 'last_invocation.json', {'status': 'step_limit_reached',
                                'new_grasps': fresh, 'reused_grasps': reused, 'committed_grasps': len(records),
                                'expected_grasps': prep['selected_count'], 'binding_fingerprint': fp,
                                'peak_rss_bytes': monitor.peak, 'seconds': time.monotonic() - monitor.started})
                            return 0
                        record = score_one(item, state, dex, api)
                        commit(path, record, fp)
                        fresh += 1
                    else:
                        reused += 1
                    require(all(record.get(k) == v for k, v in item.items()), 'Cached grasp identity differs')
                    records.append(record)
                    monitor.check()
                    print(json.dumps({'committed': len(records), 'total': prep['selected_count'],
                                      'new': fresh, 'reused': reused}), flush=True)
                oracle_path = OUT / 'oracle.json'
                oracle = load_commit(oracle_path, fp)
                if oracle is None:
                    oracle = oracle_result(state, api, dex)
                    commit(oracle_path, oracle, fp)
                report = summarize(prep, records, oracle, fp)
                monitor.check()
                once(REPORT, report)
                atomic(OUT / 'last_invocation.json', {'status': 'completed', 'new_grasps': fresh,
                    'reused_grasps': reused, 'committed_grasps': len(records), 'expected_grasps': len(records),
                    'binding_fingerprint': fp, 'peak_rss_bytes': monitor.peak,
                    'seconds': time.monotonic() - monitor.started})
                paths = [REPORT, OUT / 'preparation.json', receipt_path(OUT / 'preparation.json'),
                         oracle_path, receipt_path(oracle_path), OUT / 'last_invocation.json']
                for i in range(len(records)):
                    path = OUT / 'grasps' / ('grasp_%04d.json' % i)
                    paths += [path, receipt_path(path)]
                once(OUT / 'completion.json', {'status': 'completed', 'binding_fingerprint': fp,
                    'expected_grasps': len(records),
                    'outputs': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}})
                check_complete(binding, OUT, REPORT)
                print(json.dumps({'status': report['status'], 'selected_count': len(records),
                    'positive_friction_score_count': report['positive_friction_score_count']}), flush=True)
                return 0
        finally:
            monitor.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--max-new-grasps', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    require(args.max_new_grasps is None or args.max_new_grasps > 0, 'Invalid new-grasp limit')
    require(not (args.check_only and args.max_new_grasps is not None), 'Check-only cannot request new grasps')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
