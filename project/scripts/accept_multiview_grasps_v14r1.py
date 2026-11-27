"""Independent provenance and arithmetic audit of four related training views."""
import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import io as byteio
import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zlib

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import multiview_common_v14r1 as storage
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAMES = ['0016', '0080', '0160', '0240']
FRICTION = [.2, .4, .6, .8, 1., 1.2]
OBJECTS = [2, 21, 22, 41, 43, 44, 52, 60, 62]
ARRAYS = ('valid_points', 'valid_pixel_indices', 'sampled_valid_indices', 'sampled_points', 'sampled_pixel_indices')
FIGURE = 'scene_0003_four_training_views.png'
require = storage.require


def reconstruct_cloud(depth, intrinsic, factor):
    """Flattened original-pixel arithmetic, without importing generation code."""
    require(depth.ndim == 2 and np.issubdtype(depth.dtype, np.number), 'Invalid depth image')
    intrinsic = np.asarray(intrinsic, np.float64)
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all() and intrinsic[0, 0] > 0
            and intrinsic[1, 1] > 0 and math.isfinite(factor) and factor > 0, 'Invalid camera calibration')
    pixel = np.arange(depth.size, dtype=np.int64)
    row, col = pixel // depth.shape[1], pixel % depth.shape[1]
    z = depth.reshape(-1) / factor
    xyz = np.column_stack(((col-intrinsic[0, 2])*z/intrinsic[0, 0],
                           (row-intrinsic[1, 2])*z/intrinsic[1, 1], z))
    good = np.isfinite(xyz).all(axis=1) & (z > 0) & (z <= 2)
    return xyz[good].astype(np.float32), pixel[good]


def independent_sample(points, pixels, seed=20261021, count=20000):
    require(points.dtype == np.float32 and points.shape == (len(pixels), 3)
            and pixels.dtype == np.int64 and np.all(np.diff(pixels) > 0), 'Invalid valid-cloud provenance')
    ready = len(points) >= count
    native = np.random.RandomState(seed).choice(len(points), count, replace=False) if ready else np.empty(0, np.int64)
    indices = native.astype(np.int64)
    arrays = {'valid_points': points, 'valid_pixel_indices': pixels, 'sampled_valid_indices': indices,
              'sampled_points': points[indices], 'sampled_pixel_indices': pixels[indices]}
    info = {'input_status': 'ready' if ready else 'insufficient_input', 'valid_point_count': len(points),
        'sampled_point_count': len(indices), 'sampling_with_replacement': False, 'seed': seed, 'requested_points': count,
        'native_sampling_index_dtype': str(native.dtype), 'native_sampling_indices_sha256': hashlib.sha256(native.tobytes()).hexdigest()}
    return arrays, info


def validate_seed_provenance(candidates, bundle, provenance, arrays, ready):
    require(candidates.dtype == np.float32 and candidates.shape == (len(candidates), 17) and len(candidates) <= 1024
            and np.isfinite(candidates).all(), 'Invalid generated candidate array')
    names = {'candidates', 'sa1_indices', 'sa2_indices', 'fp2_xyz', 'legacy_fp2_indices', 'selected_seed_indices'}
    require(set(bundle) == names and bundle['candidates'].dtype == candidates.dtype
            and np.array_equal(bundle['candidates'], candidates), 'Candidate bundle differs')
    sa1, sa2, selected, fp2, legacy = [bundle[key] for key in ('sa1_indices', 'sa2_indices', 'selected_seed_indices', 'fp2_xyz', 'legacy_fp2_indices')]
    require(sa1.dtype == sa2.dtype == selected.dtype == legacy.dtype == np.int64 and fp2.dtype == np.float32, 'Seed chain dtypes differ')
    if ready:
        require(sa1.shape == (2048,) and sa2.shape == legacy.shape == (1024,) and fp2.shape == (1024, 3)
                and np.all((0 <= sa1) & (sa1 < 20000)) and np.all((0 <= sa2) & (sa2 < 2048)), 'Invalid actual SA sampling chain')
        composed = sa1[sa2]
        require(np.array_equal(fp2, arrays['sampled_points'][composed]), 'FP2 positions differ from actual SA composition')
        require(selected.shape == (len(candidates),) and np.all((0 <= selected) & (selected < 1024))
                and np.all(np.diff(selected) > 0), 'Candidate seed selection differs')
        positions = composed[selected]
        require(np.array_equal(candidates[:, 13:16], fp2[selected]), 'Candidate translation is not its actual FP2 seed')
    else:
        require(len(candidates) == 0 and sa1.shape == sa2.shape == selected.shape == legacy.shape == (0,)
                and fp2.shape == (0, 3), 'Insufficient input unexpectedly executed a forward')
        positions = np.empty(0, np.int64)
    expected = {key: value for key, value in bundle.items() if key != 'candidates'}
    expected.update(candidate_sampled_indices=positions, candidate_valid_indices=arrays['sampled_valid_indices'][positions],
                    candidate_pixel_indices=arrays['sampled_pixel_indices'][positions])
    require(set(provenance) == set(expected) and all(provenance[key].dtype == value.dtype
            and np.array_equal(provenance[key], value) for key, value in expected.items()), 'Actual seed pixel provenance changed')
    if len(candidates):
        rotation = candidates[:, 4:13].reshape(-1, 3, 3)
        require(np.max(np.abs(rotation.transpose(0, 2, 1) @ rotation-np.eye(3))) < 1e-4
                and np.max(np.abs(np.linalg.det(rotation)-1)) < 1e-4, 'Invalid candidate rotation')
        require(np.all((0 <= candidates[:, 1]) & (candidates[:, 1] <= .100001))
                and np.allclose(candidates[:, 2], .02) and np.isclose(candidates[:, 3, None], [.01, .02, .03, .04], atol=1e-6).any(axis=1).all()
                and np.all((candidates[:, 15] > 0) & (candidates[:, 15] <= 2)) and np.all(candidates[:, 16] == -1),
                'Candidate confidence/dimensions/unknown object class differ')


def fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'display_value': float(value)}


def exact_curve(ordered_records):
    counts, matrix = [], []
    for k in range(1, 51):
        scores = [record['score'] for record in ordered_records[:k]]
        require(all(score in [-1.]+FRICTION for score in scores), 'Invalid friction score')
        values = [sum(0 < score <= mu for score in scores) for mu in FRICTION]
        counts.append(values)
        matrix.append([number/k for number in values])
    total = sum((Fraction(value, k) for k, row in enumerate(counts, 1) for value in row), Fraction(0))/300
    return {'counts_50x6': counts, 'fraction_50x6': matrix, 'mean_exact': fraction_record(total),
            'top10_mu04_count': counts[9][1], 'top50_mu04_count': counts[49][1],
            'top50_counts_by_friction': counts[49]}


def independent_geometry(candidates, preparation, records, oracle, report):
    """Independent post-NMS selection and arithmetic, not an independent NMS oracle."""
    total, objects = len(candidates), preparation['object_ids']
    require(sorted(objects) == OBJECTS and preparation['input_count'] == total
            and preparation['input_dtype'] == 'float32' and preparation['selected_count'] == len(records)
            and len(preparation['selected']) == len(records), 'Frame support count/identity differs')
    clipped = candidates.copy()
    clipped[:, 1] = np.clip(clipped[:, 1], 0, .1)
    require([record['selection_index'] for record in records] == list(range(len(records)))
            and len(set(record['original_index'] for record in records)) == len(records), 'Duplicate or missing selected row')
    for item, record in zip(preparation['selected'], records):
        require(all(record.get(key) == value for key, value in item.items()), 'Scored row differs from selection seal')
        index = record['original_index']
        require(type(index) is int and 0 <= index < total and record['grasp'] == clipped[index].tolist(), 'Scored candidate lost original provenance')
        require(type(record['collision']) is bool and type(record['empty']) is bool and record['score'] in [-1.]+FRICTION
                and (not record['collision'] or record['score'] == -1.), 'Collision/friction label semantics differ')
    nms_ids, slots = preparation['nms_original_indices'], preparation['nms_object_slots']
    require(len(nms_ids) == preparation['nms_count'] == len(slots) and len(set(nms_ids)) == len(nms_ids)
            and all(type(index) is int and 0 <= index < total for index in nms_ids)
            and all(type(slot) is int and 0 <= slot < len(objects) for slot in slots), 'Invalid cached official NMS provenance')
    expected_selected = []
    if total:
        require(bool(nms_ids), 'Nonempty candidate pool has no NMS survivors')
        limited = []
        for slot in range(len(objects)):
            indices = np.asarray([index for index, assigned in zip(nms_ids, slots) if assigned == slot], np.int64)
            indices = indices[np.argsort(clipped[indices, 0])[::-1]][:10]
            limited.append(indices)
        require(preparation['per_object_pre_limit_counts'] == [len(group) for group in limited], 'Object top-ten counts differ')
        all_indices = np.concatenate(limited)
        order = np.argsort(clipped[all_indices, 0])[::-1]
        cutoff = float(clipped[all_indices[order[min(49, len(order)-1)]], 0])
        require(preparation['selection_min_confidence'] == cutoff, 'Global fiftieth confidence threshold differs')
        expected_selected = [int(index) for group in limited for index in group[clipped[group, 0] >= cutoff]]
    else:
        require(not nms_ids and not records and preparation['selection_min_confidence'] is None
                and preparation['per_object_pre_limit_counts'] == [0]*len(objects), 'Invalid empty support')
    require([record['original_index'] for record in records] == expected_selected, 'Object top-ten/global cutoff tie selection differs')
    result = {'groups': [], 'scores': [], 'collision': []}
    for slot, object_id in enumerate(objects):
        group = [record for record in records if record['object_slot'] == slot]
        require(all(record['object_id'] == object_id for record in group)
                and [record['object_local_index'] for record in group] == list(range(len(group))), 'Object-local label provenance differs')
        result['groups'].append([record['grasp'] for record in group])
        result['scores'].append([record['score'] for record in group])
        result['collision'].append([record['collision'] for record in group])
    require(oracle['result'] == result and oracle['executed'] is (total > 0), 'Unmodified official oracle disagrees with saved scores')
    require(report['official_oracle_exact_match'] is (True if total else None), 'Invalid official-reference claim')
    confidence = np.asarray([record['grasp'][0] for record in records], np.float32)
    ordered = [records[int(index)] for index in np.argsort(-confidence)]
    require(report['ordered_records'] == ordered, 'Official confidence ordering differs')
    curve = exact_curve(ordered)
    require(report['friction_grid_ascending'] == FRICTION and report['single_frame_prefix_pass_fraction_50x6'] == curve['fraction_50x6']
            and report['single_frame_prefix_matrix_mean'] == float(np.asarray(curve['fraction_50x6']).mean()), 'Independent prefix matrix differs')
    selected_ids, nms_set = set(expected_selected), set(nms_ids)
    tracking = [{'original_index': index, 'reason': 'evaluated' if index in selected_ids else
        'official_object_top10_or_global_score_threshold' if index in nms_set else 'official_nms'} for index in range(total)]
    require(report['all_original_candidate_tracking'] == tracking, 'Not every original candidate is accounted for')
    counts = {'original_candidate_count': total, 'nms_count': len(nms_ids), 'selected_count': len(records),
        'collision_including_empty_count': sum(record['collision'] for record in records),
        'empty_count': sum(record['empty'] for record in records), 'positive_friction_score_count': sum(record['score'] > 0 for record in records),
        'noncollision_contact_search_failure_count': sum(record['contact_diagnostic']['executed'] and not record['contact_diagnostic']['contacts_found'] for record in records),
        'noncollision_found_contacts_no_grid_force_closure_count': sum(record['score'] < 0 and record['contact_diagnostic']['contacts_found'] is True for record in records)}
    require(all(report[key] == value for key, value in counts.items()), 'Geometry count summary differs')
    return curve, counts


def summarize_views(frames):
    require(list(frames) == FRAMES, 'Summary changed requested frame membership/order')
    executable = [frame for frame in FRAMES if frames[frame]['input_status'] == 'ready']
    insufficient = [frame for frame in FRAMES if frames[frame]['input_status'] == 'insufficient_input']
    empty = [frame for frame in executable if frames[frame]['candidate_count'] == 0]
    official = [frame for frame in executable if frames[frame]['candidate_count'] > 0]
    require(len(executable)+len(insufficient) == 4, 'Unknown input coverage state')
    for frame in FRAMES:
        item = frames[frame]
        require(item['forward_calls_committed'] == (1 if frame in executable else 0)
                and item['official_oracle_executed'] is (frame in official), 'Coverage activity differs')
        if frame in insufficient:
            require(item['candidate_count'] == 0, 'Insufficient frame generated candidates')

    def average(selected):
        if not selected:
            return None
        values = [Fraction(frames[frame]['metrics']['mean_exact']['numerator'], frames[frame]['metrics']['mean_exact']['denominator']) for frame in selected]
        per_cell = [[sum((Fraction(frames[frame]['metrics']['counts_50x6'][k-1][j], k) for frame in selected), Fraction(0))/len(selected)
                     for j in range(6)] for k in range(1, 51)]
        return {'frame_count': len(selected), 'frame_ids': selected, 'mean_exact': fraction_record(sum(values, Fraction(0))/len(values)),
                'mean_fraction_50x6': [[float(value) for value in row] for row in per_cell]}
    return {'requested_frames': FRAMES, 'requested_frame_count': 4, 'training_scene_count': 1,
        'executable_generation_frames': executable, 'executable_generation_frame_count': len(executable),
        'official_geometry_executed_frames': official, 'official_geometry_executed_frame_count': len(official),
        'empty_candidate_frames': empty, 'insufficient_input_frames': insufficient,
        'executable_frames_equalweight_mean': average(executable), 'all_requested_zero_filled_engineering_mean': average(FRAMES),
        'frame_0000_included_in_mean': False, 'statistical_independence_claimed': False,
        'comparison_to_segmentation_or_soft_ranking': False}


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, np.float64)
    require(q.shape == (4,) and np.isfinite(q).all() and float(q @ q) > 0, 'Invalid XML quaternion')
    w, x, y, z = q
    scale = 2/float(q @ q)
    return np.asarray([[1-scale*(y*y+z*z), scale*(x*y-z*w), scale*(x*z+y*w)],
                       [scale*(x*y+z*w), 1-scale*(x*x+z*z), scale*(y*z-x*w)],
                       [scale*(x*z-y*w), scale*(y*z+x*w), 1-scale*(x*x+y*y)]], np.float64)


def independent_pose_check(xml_path, metadata, camera, align, preparation, limits):
    """Post-generation only: independent XML/MAT ID join and rigid transforms."""
    raw = Path(xml_path).read_bytes()
    require(b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper(), 'Unsafe XML declaration')
    tree = ET.fromstring(raw)
    require(tree.tag == 'scene', 'Unexpected annotation document')
    xml = {}
    for node in tree:
        object_id = int(node.findtext('obj_id'))
        require(object_id not in xml, 'Duplicate XML object identity')
        translation = [float(value) for value in node.findtext('pos_in_world').split()]
        rotation = quaternion_matrix([float(value) for value in node.findtext('ori_in_world').split()])
        require(len(translation) == 3 and all(math.isfinite(x) for x in translation), 'Invalid XML translation')
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        xml[object_id] = transform
    raw_ids = np.asarray(metadata['cls_indexes'])
    require(np.issubdtype(raw_ids.dtype, np.integer), 'Noninteger MAT object identities')
    ids = raw_ids.reshape(-1).astype(int)-1
    poses = np.asarray(metadata['poses'])
    require(sorted(xml) == sorted(ids.tolist()) == OBJECTS and len(set(ids.tolist())) == 9 and poses.shape == (3, 4, 9)
            and np.isfinite(poses).all(), 'Per-frame MAT/XML object set differs')
    max_translation, max_angle = 0.0, 0.0
    for index, object_id in enumerate(ids):
        mat = np.eye(4)
        mat[:3] = poses[:, :, index]
        expected = xml[int(object_id)]
        delta = float(np.max(np.abs(mat[:3, 3]-expected[:3, 3])))
        distance = float(np.sqrt(np.sum((mat[:3, :3]-expected[:3, :3])**2)))
        angle = 2*math.asin(min(1., distance/math.sqrt(8)))
        require(delta <= limits['translation_abs_tolerance_m'] and angle <= limits['rotation_angle_tolerance_rad'], 'Frame XML/MAT pose mismatch')
        max_translation, max_angle = max(max_translation, delta), max(max_angle, angle)
    object_matrices = []
    for index in range(9):
        matrix = np.eye(4)
        matrix[:3] = poses[:, :, index]
        object_matrices.append(matrix)
    for matrix in (camera, align, *object_matrices):
        matrix = np.asarray(matrix)
        require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
                and np.allclose(matrix[3], [0, 0, 0, 1], atol=limits['homogeneous_row_tolerance'], rtol=0)
                and np.max(np.abs(matrix[:3, :3].T @ matrix[:3, :3]-np.eye(3))) <= limits['rotation_orthogonality_tolerance']
                and abs(np.linalg.det(matrix[:3, :3])-1) <= limits['rotation_determinant_tolerance'], 'Frame camera/table transform is not rigid')
    audit = preparation['frame_geometry_audit']
    require(audit['camera_pose'] == np.asarray(camera).tolist() and audit['cam0_wrt_table'] == np.asarray(align).tolist(), 'Geometry audit substituted camera/table poses')
    transform = np.linalg.inv(np.asarray(align) @ np.asarray(camera))
    require(np.allclose(transform, np.asarray(audit['table_transform']), atol=1e-12, rtol=0), 'Geometry table transform differs')
    require(audit['intrinsic_matrix'] == np.asarray(metadata['intrinsic_matrix'], np.float64).tolist()
            and audit['factor_depth'] == float(np.asarray(metadata['factor_depth']).reshape(-1)[0]), 'Geometry/generation calibration differs')
    if preparation['geometry_preparation_executed']:
        require(preparation['table_transform'] == audit['table_transform'], 'Geometry preparation table differs')
        require(len(preparation['poses']) == len(preparation['object_ids']) == 9, 'Geometry object poses incomplete')
        for object_id, recorded in zip(preparation['object_ids'], preparation['poses']):
            require(np.allclose(recorded, xml[object_id], atol=2e-7, rtol=0), 'Geometry preparation object pose differs from XML')
    return {'object_count': 9, 'maximum_xml_mat_translation_difference_m': max_translation,
            'maximum_xml_mat_rotation_angle_rad': max_angle, 'camera_table_transform_checked': True}


def verify_inventory(directory, names):
    actual = set()
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'Link in sealed result inventory')
        name = path.relative_to(directory).as_posix()
        if path.is_file():
            actual.add(name)
        else:
            require(path.is_dir() and any(item.startswith(name+'/') for item in names), 'Unexpected result subdirectory')
    require(actual == set(names), 'Sealed result membership differs')


def verify_seal(directory, binding, names, frame):
    require(storage.read(directory/'binding.json') == binding, 'Stage binding differs')
    completion = storage.load_commit(directory/'completion.json', binding)
    require(completion['schema'] == 14 and completion['status'] == 'completed' and completion['frame'] == frame
            and completion['binding_fingerprint'] == binding['fingerprint'], 'Incomplete or wrong-frame stage')
    require(set(completion['outputs']) == set(names)
            and all(storage.digest(storage.safe_path(directory, name)) == sha for name, sha in completion['outputs'].items()),
            'Stage manifest/bytes differ')
    verify_inventory(directory, set(names) | {'completion.json', 'completion.json.sha256.json'})
    require(completion['report_sha256'] == storage.digest(directory/'report.json'), 'Stage report hash differs')
    return completion


def expected_binding(root, config, protocol, frame, group, runtime, extras=None):
    files = ({**protocol['generation_shared_inputs'], **protocol['generation_inputs'][frame]} if group == 'generation_inputs'
             else dict(protocol[group]))
    files.update(protocol.get('prior_inputs', {}))
    files.update(protocol['source_hashes'])
    for name, checksum in (extras or {}).items():
        require(name not in files or files[name] == checksum, 'Conflicting dynamic dependency')
        files[name] = checksum
    require(all(storage.digest(storage.safe_path(root, name)) == value for name, value in files.items()), 'Bound source/input changed')
    result = {'schema': 14, 'input_group': group, 'frame': frame, 'protocol_sha256': storage.digest(config),
              'files': files, 'runtime_name': runtime, 'runtime': protocol['runtimes'][runtime],
              'interpreter_sha256': storage.digest(storage.safe_path(root, protocol['interpreters'][runtime]))}
    return result


def generation_binding(root, config, protocol, frame, binding):
    numerical = binding['numerical_source_and_extension_hashes']
    directories = ('runtime/build/graspnet', 'runtime/envs/vision/Lib/site-packages/pointnet2',
                   'runtime/envs/vision/Lib/site-packages/knn_pytorch')
    expected_names = {path.relative_to(root).as_posix() for name in directories for path in storage.safe_path(root, name).rglob('*')
                      if path.is_file() and path.suffix in ('.py', '.pyd', '.dll', '.cu', '.cpp', '.h')}
    require(bool(expected_names) and set(numerical) == expected_names, 'Numerical source inventory membership differs')
    extensions = binding['resolved_extensions']
    require(set(extensions) == {'pointnet2._ext', 'knn_pytorch.knn_pytorch'} and all(value['path'] in numerical
            and value['sha256'] == numerical[value['path']] and Path(value['path']).suffix == '.pyd' for value in extensions.values()),
            'Resolved native extension identity differs')
    result = expected_binding(root, config, protocol, frame, 'generation_inputs', 'vision', numerical)
    result.update(resolved_extensions=extensions, model_runtime=protocol['model_runtime'],
        numerical_source_and_extension_hashes=numerical, precision='float32', autocast=False,
        uses_ground_truth_labels=False, uses_object_geometry=False, training_updates=0)
    result['fingerprint'] = storage.fingerprint(result)
    require(binding == result, 'Generation binding differs')


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def raw_confidence_summary(candidates):
    require(candidates.dtype == np.float32 and candidates.ndim == 2 and candidates.shape[1] == 17
            and np.isfinite(candidates).all(), 'Raw candidate values must be finite float32')
    values = candidates[:, 0]
    return {'minimum': float(values.min()) if len(values) else None,
            'maximum': float(values.max()) if len(values) else None,
            'negative_count': int(np.count_nonzero(values < 0))}


def verify_generation_lineage(root, protocol, frame, directory, inputs, forward, report, completion):
    """Independently validate parent receipts and exact-byte migration, without its algorithms."""
    recovery = protocol['recovery_sources']
    parent_ref, source = recovery['parent_protocol'], recovery['frames'][frame]
    parent_path = storage.safe_path(root, parent_ref['path'])
    require(storage.digest(parent_path) == parent_ref['sha256']
            and protocol['prior_inputs'].get(parent_ref['path']) == parent_ref['sha256'], 'Parent protocol is not frozen')
    parent = storage.read(parent_path)
    for key in ('frames', 'scene_id', 'camera', 'seed', 'num_points', 'maximum_depth_m', 'checkpoint',
                'generation_inputs', 'generation_shared_inputs', 'runtimes', 'model_runtime', 'scoring', 'geometry_audit', 'pose_limits'):
        require(parent[key] == protocol[key], 'Revision changed frozen numerical or scoring policy: '+key)
    lineage = {'mode': source['mode'], 'parent_protocol_path': parent_ref['path'],
        'parent_protocol_sha256': parent_ref['sha256'], 'directory': source['directory'],
        'source_binding_fingerprint': None, 'source_input_commit_sha256': None,
        'source_forward_commit_sha256': None, 'source_completion_sha256': None}
    source_snapshot = None
    copied = []
    if source['mode'] == 'fresh':
        require(source['directory'] is None and source['files'] == {}, 'Fresh frame contains an invented parent output')
    else:
        require(source['mode'] in ('reuse_completed', 'reuse_inputs'), 'Unknown recovery policy')
        old = storage.safe_path(root, source['directory'])
        require(old != directory.resolve(), 'Recovery cannot cite its own outputs')
        source_snapshot = storage.snapshot(old)
        actual = {(old/name).relative_to(root).as_posix(): value['sha256'] for name, value in source_snapshot.items()}
        require(actual == source['files'] and all(protocol['prior_inputs'].get(name) == checksum
                for name, checksum in actual.items()), 'Parent source membership or frozen bytes changed')
        old_binding = storage.read(old/'binding.json')
        generation_binding(root, parent_path, parent, frame, old_binding)
        old_inputs = storage.load_commit(old/'input_commit.json', old_binding)
        require(old_inputs['schema'] == 14 and old_inputs['frame'] == frame and old_inputs['status'] == 'inputs_committed'
                and old_inputs['binding_fingerprint'] == old_binding['fingerprint']
                and old_inputs['information'] == inputs['information'], 'Parent input receipt or sampling information differs')
        copied = [name+'.npy' for name in ARRAYS]
        require(old_inputs['files'] == {name: storage.digest(old/name) for name in copied}, 'Parent input array hashes differ')
        lineage.update(source_binding_fingerprint=old_binding['fingerprint'],
                       source_input_commit_sha256=storage.digest(old/'input_commit.json'))
        if source['mode'] == 'reuse_completed':
            copied += ['forward_bundle.npz', 'candidates.npy', 'seed_provenance.npz']
            names = {'binding.json'} | {value for name in copied+['input_commit.json', 'forward_commit.json', 'report.json']
                                       for value in (name, name+'.sha256.json')}
            old_completion = verify_seal(old, old_binding, names, frame)
            old_forward = storage.load_commit(old/'forward_commit.json', old_binding)
            old_report = storage.load_commit(old/'report.json', old_binding)
            require(old_forward['evidence'] == forward['evidence']
                    and old_forward['forward_calls_committed'] == forward['forward_calls_committed']
                    and old_forward['forward_bundle_sha256'] == storage.digest(old/'forward_bundle.npz'), 'Reused forward evidence changed')
            for key in ('input_status', 'outcome', 'candidate_count', 'candidates_sha256', 'forward_calls_committed'):
                require(old_report[key] == old_completion[key] == report[key], 'Reused completed output identity changed')
            lineage.update(source_forward_commit_sha256=storage.digest(old/'forward_commit.json'),
                           source_completion_sha256=storage.digest(old/'completion.json'))
        else:
            names = {'binding.json', 'input_commit.json', 'input_commit.json.sha256.json'}
            names.update(value for name in copied for value in (name, name+'.sha256.json'))
            verify_inventory(old, names)
        for name in copied:
            storage.verify_bytes(old/name, old_binding)
            require((old/name).read_bytes() == (directory/name).read_bytes(), 'Migrated binary differs from its parent bytes: '+name)
        require(storage.snapshot(old) == source_snapshot, 'Independent lineage verification modified parent outputs')
    failures = source['prior_failed_forward_attempts']
    metadata = {'lineage_source': lineage, 'prior_failed_forward_attempts': failures,
                'reused_completed_generation': source['mode'] == 'reuse_completed'}
    require(type(failures) is int and failures >= 0, 'Invalid failed-forward accounting')
    for item in (inputs, forward, report, completion):
        require(item['lineage_source'] == lineage and item['prior_failed_forward_attempts'] == failures,
                'Generation lineage fields do not match independent source verification')
    require(all(item['reused_completed_generation'] is metadata['reused_completed_generation']
                for item in (forward, report, completion)), 'Reused outputs cannot be relabelled new')
    return {'metadata': metadata, 'copied_binary_files': copied, 'source_snapshot': source_snapshot}


def forward_attempt_ledger(generations, recovery_proof):
    reused = [frame for frame in FRAMES if generations[frame]['lineage']['metadata']['reused_completed_generation']]
    new = [frame for frame in FRAMES if frame not in reused and generations[frame]['identity']['forward_calls_committed'] == 1]
    previous_success = sum(generations[frame]['identity']['forward_calls_committed'] for frame in reused)
    failures = sum(generations[frame]['lineage']['metadata']['prior_failed_forward_attempts'] for frame in FRAMES)
    observed = recovery_proof['completion_invocation_new_model_forward_calls_by_frame']
    require(set(observed) == set(FRAMES) and all(type(observed[frame]) is int and observed[frame] in (0, 1)
            and (frame in new or observed[frame] == 0) for frame in FRAMES), 'Observed forward accounting contradicts provenance')
    return {'reused_successful_frames': reused, 'revision_successful_frames': new,
        'retained_successful_forward_outputs': previous_success, 'newly_committed_forward_outputs': len(new),
        'known_prior_failed_forward_attempts': failures,
        'known_total_forward_attempts': previous_success+len(new)+failures,
        'completion_invocation_new_model_forward_calls_by_frame': observed,
        'completion_invocation_new_model_forward_calls_total': sum(observed.values()),
        'scope': 'Durably committed successful outputs plus explicitly recorded parent failures; final invocation counters can be zero after resume and do not bound unrecorded process-loss attempts',
        'parent_failed_output_raw_values_known': False,
        'revision_negative_values_prove_parent_failure_values': False}


def inspect_generation(root, config, protocol, frame):
    directory = storage.frame_directory(root, protocol, frame, 'generation')
    binding = storage.read(directory/'binding.json')
    generation_binding(root, config, protocol, frame, binding)
    binary = tuple(name+'.npy' for name in ARRAYS)+('forward_bundle.npz', 'candidates.npy', 'seed_provenance.npz')
    json_names = ('input_commit.json', 'forward_commit.json', 'report.json')
    names = {'binding.json'} | {value for name in binary+json_names for value in (name, name+'.sha256.json')}
    completion = verify_seal(directory, binding, names, frame)
    for name in binary:
        storage.verify_bytes(directory/name, binding)
    inputs, forward, report = (storage.load_commit(directory/name, binding) for name in json_names)
    require(all(value['schema'] == 14 and value['frame'] == frame and value['binding_fingerprint'] == binding['fingerprint']
                for value in (inputs, forward, report)), 'Generation commit identity differs')
    paths = protocol['frame_inputs'][frame]
    with Image.open(storage.safe_path(root, paths['rgb'])) as source:
        rgb = np.asarray(source.convert('RGB'))
    with Image.open(storage.safe_path(root, paths['depth'])) as source:
        depth = np.asarray(source)
    require(rgb.shape == (720, 1280, 3) and rgb.dtype == np.uint8 and depth.shape == (720, 1280)
            and depth.dtype == np.uint16, 'Frame RGB/depth shape or dtype differs')
    from scipy.io import loadmat
    # No object IDs or poses may be requested until every generation is sealed.
    metadata = loadmat(storage.safe_path(root, paths['meta']), variable_names=['intrinsic_matrix', 'factor_depth'])
    intrinsic = np.asarray(metadata['intrinsic_matrix'], np.float64)
    require(np.asarray(metadata['factor_depth']).size == 1, 'Depth factor is not scalar')
    factor = float(np.asarray(metadata['factor_depth']).item())
    points, pixels = reconstruct_cloud(depth, intrinsic, factor)
    arrays, information = independent_sample(points, pixels)
    information.update(shape_hw=list(depth.shape), intrinsic_matrix=intrinsic.tolist(), factor_depth=factor,
                       metadata_fields_read=['intrinsic_matrix', 'factor_depth'], model_input='XYZ only; RGB verified, not a model feature')
    require(inputs['status'] == 'inputs_committed' and inputs['information'] == information
            and inputs['files'] == {name+'.npy': storage.digest(directory/(name+'.npy')) for name in ARRAYS}, 'Independent input reconstruction differs')
    for name, value in arrays.items():
        actual = np.load(directory/(name+'.npy'), allow_pickle=False)
        require(actual.dtype == value.dtype and np.array_equal(actual, value), 'Independent point/sample/pixel array differs: '+name)
    candidates = np.load(directory/'candidates.npy', allow_pickle=False)
    ready = information['input_status'] == 'ready'
    validate_seed_provenance(candidates, archive(directory/'forward_bundle.npz'), archive(directory/'seed_provenance.npz'), arrays, ready)
    calls = 1 if ready else 0
    outcome = 'insufficient_input' if not ready else 'generated' if len(candidates) else 'empty_candidates'
    require(forward['forward_calls_committed'] == forward['evidence']['model_forward_calls'] == calls
            and forward['forward_bundle_sha256'] == storage.digest(directory/'forward_bundle.npz'), 'Forward activity/bundle seal differs')
    if ready:
        evidence = forward['evidence']
        require(evidence['strict_load'] is True and evidence['weights_only'] is True and evidence['precision'] == 'float32'
                and evidence['autocast'] is False and evidence['checkpoint_sha256'] == protocol['checkpoint']['sha256']
                and 0 <= evidence['peak_gpu_allocated_bytes'] <= protocol['resources']['generation']['gpu_allocated_soft_bytes']
                and 0 <= evidence['peak_gpu_reserved_bytes'] <= protocol['resources']['generation']['gpu_reserved_soft_bytes'], 'Frozen forward settings/resources differ')
    identity = {'input_status': information['input_status'], 'outcome': outcome, 'candidate_count': len(candidates),
                'forward_calls_committed': calls, 'candidates_sha256': storage.digest(directory/'candidates.npy')}
    require(all(completion[key] == report[key] == value for key, value in identity.items())
            and report['status'] == 'completed' and report['input_information'] == information
            and report['forward_evidence'] == forward['evidence'], 'Generation outcome summary differs')
    require(report['uses_ground_truth_labels'] is False and report['uses_object_geometry'] is False
            and report['training_updates'] == 0 and report['is_dataset_ap_result'] is False, 'Unsupported generation claim')
    expected_geometry = {'candidate_count': len(candidates), 'geometry_checks_applicable': bool(len(candidates))}
    if len(candidates):
        rotation = candidates[:, 4:13].reshape(-1, 3, 3)
        expected_geometry.update(maximum_rotation_orthogonality_error=float(np.max(np.abs(rotation.transpose(0, 2, 1) @ rotation-np.eye(3)))),
                                 maximum_determinant_error=float(np.max(np.abs(np.linalg.det(rotation)-1))))
    require(report['geometry_checks'] == expected_geometry, 'Candidate geometry summary differs')
    confidence = raw_confidence_summary(candidates)
    require(report['raw_confidence_summary'] == confidence, 'Raw confidence diagnostic was filtered or changed')
    lineage = verify_generation_lineage(root, protocol, frame, directory, inputs, forward, report, completion)
    return {'directory': directory, 'binding': binding, 'completion': completion, 'candidates': candidates, 'identity': identity,
            'lineage': lineage, 'raw_confidence_summary': confidence,
            'input_information': information, 'snapshot': storage.snapshot(directory),
            'files': {path.relative_to(root).as_posix(): storage.digest(path) for path in directory.rglob('*') if path.is_file()}}


def geometry_runtime_files(root, protocol):
    site = root/'runtime/envs/grasp_eval/Lib/site-packages'
    api = site/'graspnetAPI'
    tree = {path.relative_to(api).as_posix(): storage.digest(path) for path in sorted(api.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts}
    checksum = hashlib.sha256(json.dumps(tree, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(checksum == protocol['installed_api_tree_sha256'], 'Pinned official API implementation changed')
    packages = {distribution.metadata['Name']: distribution.version for distribution in importlib.metadata.distributions(path=[str(site)])}
    require(packages == storage.read(root/'results/P2/grasp_geometry_v7.json')['packages'], 'Geometry environment packages changed')
    files = {path.relative_to(root).as_posix(): storage.digest(path) for path in sorted(api.rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts}
    for name in ('numpy', 'scipy', 'open3d', 'PIL', 'psutil', 'threadpoolctl.py'):
        directory = site/name
        paths = [directory] if directory.is_file() else directory.rglob('*')
        files.update({path.relative_to(root).as_posix(): storage.digest(path) for path in paths
                      if path.is_file() and path.suffix in ('.py', '.pyd', '.dll')})
    return files


def inspect_geometry(root, config, protocol, frame, generations, asset_files, numerical_files):
    extras = {name: checksum for item in generations.values() for name, checksum in item['files'].items()}
    extras.update(asset_files)
    extras.update(numerical_files)
    binding = expected_binding(root, config, protocol, frame, 'geometry_inputs', 'grasp_eval', extras)
    binding['fingerprint'] = storage.fingerprint(binding)
    directory = storage.frame_directory(root, protocol, frame, 'geometry')
    require(storage.read(directory/'binding.json') == binding, 'Geometry did not bind all four sealed generations')
    preparation = storage.load_commit(directory/'preparation.json', binding)
    require(preparation['schema'] == 14 and preparation['frame'] == frame and type(preparation['selected_count']) is int
            and 0 <= preparation['selected_count'] <= len(generations[frame]['candidates']), 'Invalid dynamic geometry support')
    names = {'binding.json'}
    for name in ['preparation.json', 'oracle.json', 'report.json']+['grasps/grasp_%04d.json' % i for i in range(preparation['selected_count'])]:
        names.update((name, name+'.sha256.json'))
    completion = verify_seal(directory, binding, names, frame)
    require(completion['expected_grasps'] == preparation['selected_count'] and completion['all_generation_completion_sha256'] == {
        key: storage.digest(item['directory']/'completion.json') for key, item in generations.items()}, 'Geometry generation dependency seals differ')
    records = [storage.load_commit(directory/('grasps/grasp_%04d.json' % i), binding) for i in range(preparation['selected_count'])]
    oracle, report = (storage.load_commit(directory/name, binding) for name in ('oracle.json', 'report.json'))
    metrics, counts = independent_geometry(generations[frame]['candidates'], preparation, records, oracle, report)
    outcome = generations[frame]['identity']['outcome']
    expected_status = ('accepted_limited_training_view_geometry_baseline' if outcome == 'generated' else
                       'completed_empty_candidate_negative_result' if outcome == 'empty_candidates' else 'skipped_insufficient_generation_input')
    require(report['schema'] == 14 and report['frame'] == frame and report['binding_fingerprint'] == binding['fingerprint']
            and report['status'] == expected_status and report['generation_outcome'] == outcome
            and report['official_oracle_executed'] is oracle['executed']
            and report['geometry_preparation_executed'] is (outcome == 'generated')
            and report['candidate_count'] == counts['original_candidate_count'], 'Geometry branch status differs')
    for key in ('dataset_ap_computed', 'robot_success_rate_computed', 'gpu_used', 'soft_ranking_used'):
        require(report[key] is False, 'Unsupported geometry claim: '+key)
    require(report['new_training_steps'] == 0 and report['full_point_cloud_input'] is True and report['no_method_comparison'] is True
            and report['reported_prefix_count'] == 50 and report['official_cutoff_ties_retained'] is True
            and report['selected_count_may_exceed_50'] is True
            and report['prefix_denominator_rule'] == 'K even when fewer than K selected records', 'Geometry scope/denominator changed')
    from scipy.io import loadmat
    paths = protocol['frame_inputs'][frame]
    metadata = loadmat(storage.safe_path(root, paths['meta']), variable_names=['intrinsic_matrix', 'factor_depth', 'cls_indexes', 'poses'])
    camera = np.load(storage.safe_path(root, protocol['camera_poses_path']), allow_pickle=False)[int(frame)]
    align = np.load(storage.safe_path(root, protocol['cam0_wrt_table_path']), allow_pickle=False)
    pose = independent_pose_check(storage.safe_path(root, paths['annotation']), metadata, camera, align, preparation, protocol['pose_limits'])
    require(preparation['frame_geometry_audit']['intrinsic_matrix'] == generations[frame]['input_information']['intrinsic_matrix']
            and preparation['frame_geometry_audit']['factor_depth'] == generations[frame]['input_information']['factor_depth'], 'Generation/geometry calibration mismatch')
    return {'directory': directory, 'snapshot': storage.snapshot(directory), 'completion': completion,
        'summary': {**generations[frame]['identity'], 'frame': frame, 'status': expected_status, 'official_oracle_executed': oracle['executed'],
            **generations[frame]['lineage']['metadata'], 'raw_confidence_summary': generations[frame]['raw_confidence_summary'],
            'metrics': metrics, 'counts': counts, 'independent_pose_check': pose,
            'generation_completion_sha256': storage.digest(generations[frame]['directory']/'completion.json'),
            'geometry_completion_sha256': storage.digest(directory/'completion.json')}}


def resource_evidence(record, protocol, phase):
    evidence = record if phase == 'assets' else record['resources']
    for key, limit in (('peak_rss_bytes', 'rss_soft_bytes'), ('elapsed_seconds', 'invocation_soft_seconds'),
                       ('current_output_bytes', 'output_limit_bytes')):
        value = evidence[key]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= protocol['resources'][phase][limit], 'Recorded resource limit violated')


def validate_geometry_probe(probe, count):
    if probe.get('recovered_completed_probe') is True:
        require(0 <= count <= 2 and probe['status'] == 'verified_completed_read_only'
                and probe['expected_grasps'] == probe['committed_grasps'] == probe['reused_grasps'] == count
                and probe['new_grasps'] == probe['new_model_forward_calls'] == probe['new_geometry_computations'] == probe['new_geometry_preparations'] == 0
                and probe['official_oracle_executed_this_invocation'] is False,
                'Recovered small-support probe must be explicitly read-only and complete')
    else:
        require(probe.get('recovered_completed_probe', False) is False
                and probe['status'] == ('partial' if count > 2 else 'completed'), 'Geometry probe stopping boundary differs')


def verify_recovery(root, config, protocol, phase, generations=None, geometries=None):
    proof_path = storage.safe_path(root, protocol['outputs'][phase+'_resume_report'])
    proof = storage.read(proof_path)
    child = {'assets': 'fetch_multiview_assets_v14r1.py', 'generation': 'generate_multiview_grasps_v14r1.py',
             'geometry': 'evaluate_multiview_grasps_v14r1.py'}[phase]
    interpreter = 'grasp_eval' if phase == 'geometry' else 'vision'
    expected_binding = {'protocol_sha256': storage.digest(config), 'wrapper_sha256': storage.digest(root/'scripts/run_multiview_v14r1.py'),
        'child_sha256': storage.digest(root/'scripts'/child), 'common_sha256': storage.digest(root/'scripts/multiview_common_v14r1.py'),
        'interpreter_sha256': storage.digest(storage.safe_path(root, protocol['interpreters'][interpreter])),
        'pipeline_sha256': storage.digest(root/'scripts/pipeline.py')}
    upstream = ('assets', 'generation') if phase == 'geometry' else ('assets',) if phase == 'generation' else ()
    expected_binding['upstream_recovery_proofs'] = {previous: storage.digest(storage.safe_path(root, protocol['outputs'][previous+'_resume_report'])) for previous in upstream}
    expected_frames = ['all'] if phase == 'assets' else FRAMES
    require(proof['schema'] == 14 and proof['phase'] == phase and proof['status'] == 'passed' and proof['binding'] == expected_binding
            and set(proof['frames']) == set(expected_frames), 'Phase recovery proof identity differs')
    observed_forward_calls = {}
    for frame in expected_frames:
        entry = proof['frames'][frame]
        directory = storage.safe_path(root, protocol['outputs']['assets_dir']) if phase == 'assets' else storage.frame_directory(root, protocol, frame, phase)
        current = storage.snapshot(directory)
        prefix = entry['prefix_before']
        require(entry['status'] == 'passed' and entry['byte_and_mtime_identical'] is True and bool(prefix)
                and entry['completed_snapshot'] == current and all(current.get(name) == value for name, value in prefix.items())
                and entry['completion_sha256'] == storage.digest(directory/'completion.json'), 'Actual recovery prefix or completed snapshot changed')
        probe, final, readonly = (entry[name] for name in ('probe_record', 'completion_record', 'read_only_record'))
        require(final['status'] in ('completed', 'verified_completed_read_only') and readonly['status'] == 'verified_completed_read_only', 'Missing completion/read-only evidence')
        for record in (probe, final, readonly):
            resource_evidence(record, protocol, phase)
            if phase != 'assets':
                require(record['frame'] == frame, 'Recovery record frame differs')
        if phase == 'generation':
            expected_prefix = {'binding.json', 'input_commit.json', 'input_commit.json.sha256.json'}
            expected_prefix.update(value for name in ARRAYS for value in (name+'.npy', name+'.npy.sha256.json'))
            require(set(prefix) == expected_prefix and probe['status'] == 'inputs_committed'
                    and probe['new_model_forward_calls'] == readonly['new_model_forward_calls'] == 0,
                    'Generation probe/read-only crossed model boundary')
            item = generations[frame]['identity']
            require(probe['input_status'] == item['input_status'] and probe['input_commit_sha256'] == storage.digest(directory/'input_commit.json'), 'Input-only probe differs')
            for record in (final, readonly):
                require(all(record[key] == item[key] for key in ('input_status', 'outcome', 'candidate_count', 'forward_calls_committed')),
                        'Generation recovery counters/outcome differ')
            require(final['new_model_forward_calls'] in (0, item['forward_calls_committed']), 'Generation forward budget differs')
            metadata = generations[frame]['lineage']['metadata']
            require(all(all(record[key] == value for key, value in metadata.items()) for record in (probe, final, readonly)),
                    'Recovery proof misstates generation lineage')
            require(type(final['new_model_forward_calls']) is int
                    and (not metadata['reused_completed_generation'] or final['new_model_forward_calls'] == 0),
                    'A reused completed frame performed a new forward')
            observed_forward_calls[frame] = final['new_model_forward_calls']
        elif phase == 'geometry':
            count = geometries[frame]['completion']['expected_grasps']
            committed = min(2, count)
            validate_geometry_probe(probe, count)
            for record, expected in ((probe, committed), (final, count), (readonly, count)):
                require(all(type(record[key]) is int and record[key] >= 0 for key in ('new_grasps', 'reused_grasps', 'committed_grasps', 'expected_grasps'))
                        and record['new_grasps']+record['reused_grasps'] == expected == record['committed_grasps']
                        and record['expected_grasps'] == count and record['new_geometry_computations'] == record['new_grasps']
                        and record['new_model_forward_calls'] == 0, 'Geometry per-candidate accounting differs')
            require(probe['new_grasps'] <= 2 and final['reused_grasps'] >= committed
                    and readonly['new_grasps'] == readonly['new_geometry_preparations'] == 0
                    and readonly['official_oracle_executed_this_invocation'] is False, 'Read-only geometry performed new work')
            if count > 2:
                expected_prefix = {'binding.json', 'preparation.json', 'preparation.json.sha256.json'}
                expected_prefix.update(value for index in range(2) for value in ('grasps/grasp_%04d.json' % index, 'grasps/grasp_%04d.json.sha256.json' % index))
                require(set(prefix) == expected_prefix, 'Geometry recovery prefix does not contain exactly first two scores')
            else:
                require(prefix == current, 'Zero/small-support complete probe changed')
        else:
            expected_prefix = {'binding.json'}
            asset_dir = storage.safe_path(root, protocol['outputs']['assets_dir'])
            for selected in FRAMES[:2]:
                name = storage.safe_path(root, protocol['frame_inputs'][selected]['annotation']).relative_to(asset_dir).as_posix()
                expected_prefix.update((name, name+'.sha256.json'))
            require(set(prefix) == expected_prefix and probe['status'] == 'partial' and 0 <= probe['new_files'] <= 2
                    and 0 <= final['new_files'] <= 2 and readonly['new_files'] == readonly['network_bytes'] == 0, 'Asset prefix/network read-only evidence differs')
            require(all(0 <= record['network_bytes'] <= protocol['asset_source']['maximum_network_bytes'] for record in (probe, final, readonly)), 'Asset network budget exceeded')
            if protocol['asset_source'].get('reuse_frozen_assets', False):
                require(all(record['network_bytes'] == 0 for record in (probe, final, readonly)), 'Frozen parent asset reuse accessed network')
    result = {'sha256': storage.digest(proof_path), 'frames_verified': expected_frames,
              'actual_prefix_and_completed_bytes_mtime_checked': True, 'scope': 'Process boundaries, not physical power removal'}
    if phase == 'generation':
        result['completion_invocation_new_model_forward_calls_by_frame'] = observed_forward_calls
    return result


def scientific_png(frames, summary):
    canvas = Image.new('RGB', (1680, 1040), 'white')
    draw = ImageDraw.Draw(canvas)
    large, regular, small = (ImageFont.load_default(size=size) for size in (29, 22, 18))
    draw.text((48, 25), 'Full-cloud grasp baseline | four views of training scene 0003', fill='#172634', font=large)
    draw.text((48, 72), 'Same checkpoint, 20,000 sampled points per executable view. No method comparison.', fill='#455462', font=regular)
    colours = ['#2868ad', '#ca7029', '#32855d', '#855ead']
    for number, frame in enumerate(FRAMES):
        left, top = 70+(number % 2)*820, 150+(number//2)*365
        right, bottom = left+670, top+230
        item = frames[frame]
        draw.text((left, top-30), 'Frame '+frame+' | '+item['outcome'], fill=colours[number], font=regular)
        for value in range(6):
            y = bottom-value/5*230
            draw.line((left, y, right, y), fill='#dfe5e8')
            draw.text((left-40, y-8), '%.1f' % (value/5), fill='#455462', font=small)
        draw.line((left, top, left, bottom, right, bottom), fill='#455462', width=2)
        points = [(left+index*670/49, bottom-row[1]*230) for index, row in enumerate(item['metrics']['fraction_50x6'])]
        draw.line(points, fill=colours[number], width=4)
        for k in (1, 10, 20, 30, 40, 50):
            draw.text((left+(k-1)*670/49-7, bottom+8), str(k), fill='#455462', font=small)
        text = 'Candidates %d | selected %d | friction 0.4 top50 %d/50' % (item['candidate_count'], item['counts']['selected_count'], item['metrics']['top50_mu04_count'])
        draw.text((left, bottom+43), text, fill='#172634', font=small)
        draw.text((left, bottom+70), '50 x 6 prefix mean: %.6f' % item['metrics']['mean_exact']['display_value'], fill='#455462', font=small)
    draw.text((55, 895), 'Curves: passing fraction at friction 0.4 versus prefix size K. Denominator remains K.', fill='#172634', font=regular)
    mean = summary['executable_frames_equalweight_mean']
    value = 'not executable' if mean is None else '%.6f' % mean['mean_exact']['display_value']
    draw.text((55, 934), 'Executable views %d/4 | equal-view mean %s | frame 0000 excluded' % (summary['executable_generation_frame_count'], value), fill='#172634', font=regular)
    draw.text((55, 982), 'Related training views, not independent scenes, dataset AP, calibrated risk, or robot success.', fill='#455462', font=small)
    output = byteio.BytesIO()
    canvas.save(output, format='PNG', optimize=False)
    return output.getvalue()


def validate_recovery_contract(protocol):
    recovery = protocol['recovery_sources']
    require(set(recovery['frames']) == set(FRAMES)
            and recovery['parent_protocol']['path'] == 'configs/grasp_multiview_v14.json'
            and protocol['prior_inputs'].get(recovery['parent_protocol']['path']) == recovery['parent_protocol']['sha256'],
            'Parent protocol or frame scope not frozen')
    failure_log = 'logs/P3/multiview_v14/generation_0160_complete.log'
    require(protocol['prior_inputs'].get(failure_log) == 'f4d989433e15ba77a50ab1298949824c7bdf406decf85c17f6fa1a6793092b02',
            'Original failed-forward evidence missing or changed')
    for frame, mode, failures in zip(FRAMES, ('reuse_completed', 'reuse_completed', 'reuse_inputs', 'fresh'), (0, 0, 1, 0)):
        source = recovery['frames'][frame]
        require(source['mode'] == mode and type(source['prior_failed_forward_attempts']) is int
                and source['prior_failed_forward_attempts'] == failures, 'Recovery mode or known failed-forward count changed')
        if mode == 'fresh':
            require(source['directory'] is None and source['files'] == {}, 'Fresh frame has recovery files')
        else:
            directory = 'results/P3/grasp_multiview_v14/frames/'+frame+'/generation'
            require(source['directory'] == directory and bool(source['files'])
                    and all(name.startswith(directory+'/') and protocol['prior_inputs'].get(name) == value
                            for name, value in source['files'].items()), 'Parent source references not pinned')
    require(protocol['outputs']['frames_root'] == 'results/P3/grasp_multiview_v14r1/frames'
            and protocol['outputs']['assets_dir'] == 'data/raw/grasp_multiview_v14r1'
            and recovery['assets_dir'] == 'data/raw/grasp_multiview_v14'
            and protocol['asset_source']['reuse_frozen_assets'] is True, 'Revision output/source directories differ')


def validate_protocol(protocol):
    require(protocol['schema'] == 14 and protocol['frames'] == FRAMES and protocol['scene_id'] == 3 and protocol['camera'] == 'realsense'
            and protocol['seed'] == 20261021 and protocol['num_points'] == 20000 and protocol['maximum_depth_m'] == 2
            and protocol['object_ids'] == OBJECTS, 'Frozen experiment selection/sampling differs')
    require(protocol['acceptance'] == {'efficacy_gate': False, 'all_generation_before_geometry': True,
        'include_empty_candidates_in_executable_mean': True, 'exclude_insufficient_input_from_executable_mean': True,
        'reference_frame_0000_in_aggregate': False}, 'Descriptive acceptance contract changed')
    require(protocol['resources']['acceptance'] == {'cpu_threads': 2, 'rss_soft_bytes': 8*1024**3,
        'output_limit_bytes': 1024**3, 'disk_reserve_bytes': 60*1024**3, 'invocation_soft_seconds': 300}, 'Acceptance resource bounds changed')
    required_sources = {'scripts/accept_multiview_grasps_v14r1.py', 'scripts/generate_multiview_grasps_v14r1.py',
        'scripts/evaluate_multiview_grasps_v14r1.py', 'scripts/multiview_common_v14r1.py', 'scripts/run_multiview_v14r1.py',
        'scripts/fetch_multiview_assets_v14r1.py', 'scripts/pipeline.py'}
    require(required_sources.issubset(protocol['source_hashes']), 'Implementation not frozen')
    for frame in FRAMES:
        require(set(protocol['generation_inputs'][frame]) == {protocol['frame_inputs'][frame][key] for key in ('rgb', 'depth', 'meta')}, 'Generation input contains a pose annotation')
    require(protocol['pose_limits'] == protocol['geometry_audit'] and storage.runtime_identity() == protocol['runtimes']['vision'], 'Numerical runtime/pose tolerances differ')
    validate_recovery_contract(protocol)


def run(root=ROOT, config=None, check_only=False):
    root = Path(root).resolve()
    config = Path(config).resolve() if config else root/'configs/grasp_multiview_v14r1.json'
    require(config.is_relative_to(root), 'Configuration outside project')
    protocol = storage.read(config)
    validate_protocol(protocol)
    require(Path(sys.prefix).resolve() == storage.safe_path(root, protocol['interpreters']['vision']).parent.parent.resolve(), 'Wrong acceptance interpreter')
    guard = storage.Guard(root, protocol, phase='acceptance')
    guard.check()
    with storage.lock(root/'runtime/locks/grasp_multiview_v14r1/acceptance'):
        return locked_acceptance(root, config, protocol, guard, check_only)


def recover_publication_temporaries(root, protocol, check_only):
    report = storage.safe_path(root, protocol['outputs']['acceptance_report'])
    image = storage.safe_path(root, protocol['outputs']['visuals_dir'])/FIGURE
    for target in (report, image):
        if not target.parent.exists():
            continue
        for path in target.parent.glob(target.name+'.*.tmp'):
            require(not path.is_symlink() and path.is_file() and path.resolve().parent == target.parent.resolve()
                    and re.fullmatch(re.escape(target.name)+r'\.[0-9a-f]{32}\.tmp', path.name), 'Unknown publication temporary')
            require(not check_only and not report.exists(), 'Completed/read-only publication contains an interrupted temporary')
            path.unlink()  # Only an uncommitted temporary for this exact output.


def locked_acceptance(root, config, protocol, guard, check_only):
    recover_publication_temporaries(root, protocol, check_only)
    generations = {}
    for frame in FRAMES:
        generations[frame] = inspect_generation(root, config, protocol, frame)
        guard.check()
    generation_proof = verify_recovery(root, config, protocol, 'generation', generations=generations)
    ledger = forward_attempt_ledger(generations, generation_proof)
    # Object poses, model files and new annotations are not read before all four
    # label-free generation seals and independently reconstructed inputs pass.
    assets = storage.verify_assets(root, config, protocol)
    for frame in FRAMES:
        annotation = storage.safe_path(root, protocol['frame_inputs'][frame]['annotation'])
        require(zlib.crc32(annotation.read_bytes()) & 0xffffffff == protocol['asset_source']['entries'][frame]['crc32'], 'Annotation ZIP member CRC differs')
    asset_proof = verify_recovery(root, config, protocol, 'assets')
    numerical = geometry_runtime_files(root, protocol)
    geometries = {}
    for frame in FRAMES:
        geometries[frame] = inspect_geometry(root, config, protocol, frame, generations, assets, numerical)
        guard.check()
    geometry_proof = verify_recovery(root, config, protocol, 'geometry', generations, geometries)
    frames = {frame: geometries[frame]['summary'] for frame in FRAMES}
    summary = summarize_views(frames)
    figure = scientific_png(frames, summary)
    outputs = {key: storage.safe_path(root, value) for key, value in protocol['outputs'].items()}
    report = {'schema': 14, 'status': 'accepted_descriptive_multiview_baseline_engineering',
        'protocol_sha256': storage.digest(config), 'acceptor_sha256': storage.digest(Path(__file__)),
        'revision': 'v14r1', 'revision_kind': 'engineering_contract_repair_not_method_improvement',
        'parent_protocol': protocol['recovery_sources']['parent_protocol'], 'forward_attempt_accounting': ledger,
        'raw_confidence_policy': 'All finite float32 raw values retained unchanged, including negatives; no sigmoid, clipping, absolute value, or sign-based filtering',
        'frames': frames, 'summary': summary, 'recovery_evidence': {'assets': asset_proof, 'generation': generation_proof, 'geometry': geometry_proof},
        'checks': {'all_four_input_clouds_and_samples_independently_reconstructed': True,
            'actual_sa_seed_and_original_pixel_chains_checked': True, 'all_generation_sealed_before_geometry_access': True,
            'reused_parent_binary_bytes_and_evidence_identical': True, 'parent_failure_history_preserved': True,
            'parent_sampling_scoring_and_pose_policy_unchanged': True, 'raw_confidence_diagnostics_independently_checked': True,
            'post_nms_object_top10_and_global_cutoff_ties_recomputed': True, 'all_scored_records_match_official_oracle': True,
            'all_1200_prefix_count_fraction_cells_recomputed': True, 'exact_equal_view_aggregation': True,
            'all_actual_prefix_and_completed_sha_size_mtime_checked': True, 'per_frame_xml_mat_camera_calibration_checked': True},
        'independence_boundary': {'nms_and_nearest_object_algorithms_reexecuted': False, 'force_closure_recomputed': False,
            'description': 'Independent input/provenance, post-NMS selection, oracle consistency, and arithmetic audit; NMS/object assignment/geometry retain the pinned official implementation'},
        'historical_reference_frame': '0000', 'historical_reference_in_aggregate': False, 'efficacy_gate': False,
        'new_model_forward_calls': 0, 'new_geometry_computations': 0, 'new_training_steps': 0,
        'is_dataset_ap_result': False, 'method_improvement_claimed': False, 'independent_confirmation': False,
        'visual': {'file': protocol['outputs']['visuals_dir']+'/'+FIGURE, 'sha256': hashlib.sha256(figure).hexdigest()},
        'limitations': protocol['limitations']}
    for frame in FRAMES:
        require(storage.snapshot(generations[frame]['directory']) == generations[frame]['snapshot']
                and storage.snapshot(geometries[frame]['directory']) == geometries[frame]['snapshot'], 'Acceptance altered sealed stage bytes or timestamps')
        lineage = generations[frame]['lineage']
        if lineage['source_snapshot'] is not None:
            source = storage.safe_path(root, lineage['metadata']['lineage_source']['directory'])
            require(storage.snapshot(source) == lineage['source_snapshot'], 'Parent generation bytes or timestamps changed during acceptance')
    frozen = {**assets, **numerical, **protocol['geometry_inputs'], **protocol['source_hashes']}
    for item in generations.values():
        frozen.update(item['binding']['files'])
    for name, checksum in frozen.items():
        require(storage.digest(storage.safe_path(root, name)) == checksum, 'Dependency changed during independent acceptance')
    payload = storage.json_bytes(report)
    image_path = outputs['visuals_dir']/FIGURE
    if outputs['visuals_dir'].exists():
        require(all(path == image_path and path.is_file() and not path.is_symlink()
                    for path in outputs['visuals_dir'].rglob('*')), 'Unexpected acceptance visual inventory')
    guard.check((0 if image_path.exists() else len(figure))+(0 if outputs['acceptance_report'].exists() else len(payload)))
    storage.publish_once(image_path, figure, check_only)
    guard.check(0 if outputs['acceptance_report'].exists() else len(payload))
    storage.publish_once(outputs['acceptance_report'], payload, check_only)
    return {'status': 'verified_completed_read_only' if check_only else 'passed', 'report_sha256': storage.digest(outputs['acceptance_report']),
            'figure_sha256': storage.digest(image_path), 'resources': guard.check()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/grasp_multiview_v14r1.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.config, args.check_only), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()


