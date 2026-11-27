"""Frozen single-frame geometry audit; no scoring, rescaling, or pickle loading.

Completed reports are immutable. Reentry verifies all recorded input/code/runtime
hashes without loading geometry. --recompute-check additionally repeats the audit.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Path escaped approved root: ' + str(relative))
    return path


def atomic_once(path, value):
    """Commit once; preserve interrupted temporaries and never replace evidence."""
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Existing report differs; do not overwrite')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    # A parent OS lock excludes competing writers; Windows rename never replaces.
    temporary.rename(path)


class ResourceMonitor:
    """Sample RSS/time; enforce soft limits at explicit geometry boundaries.

    This is not a hard allocation/time sandbox. A long native call can cross a
    limit before control returns. Any sampled breach remains latched, preventing
    a successful commit even when memory use subsequently falls.
    """
    def __init__(self, protocol, process=None):
        import psutil
        self.protocol = protocol
        self.process = process if process is not None else psutil.Process()
        self.started = time.monotonic()
        self.peak_rss_bytes = 0
        self.failure = None
        self.stopped = threading.Event()
        self.thread = None

    def sample(self):
        try:
            rss = int(self.process.memory_info().rss)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            elapsed = time.monotonic() - self.started
            if rss > self.protocol['rss_soft_bytes'] and self.failure is None:
                self.failure = 'Sampled RSS exceeds predeclared soft limit'
            if elapsed > self.protocol['invocation_soft_seconds'] and self.failure is None:
                self.failure = 'Elapsed geometry time exceeds predeclared soft limit'
        except Exception as error:
            self.failure = 'Resource sampling failed: ' + repr(error)

    def check(self, output_bytes=0):
        self.sample()
        require(self.failure is None, self.failure or 'Resource sampling failed')
        require(0 <= output_bytes <= self.protocol['output_limit_bytes'],
                'Geometry report exceeds predeclared output limit')
        require(shutil.disk_usage(ROOT).free >= self.protocol['disk_reserve_bytes'] + output_bytes,
                'Project disk reserve would fall below predeclared limit')

    def _watch(self):
        while not self.stopped.wait(.05):
            self.sample()

    def __enter__(self):
        self.check()
        self.thread = threading.Thread(target=self._watch, daemon=True,
                                       name='geometry-audit-resource-sampler')
        self.thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
        self.sample()
        print(json.dumps({'resource_observation': 'geometry_audit_v10',
                          'sample_interval_seconds': .05,
                          'peak_sampled_rss_bytes': self.peak_rss_bytes,
                          'elapsed_seconds': time.monotonic() - self.started,
                          'soft_limit_breach': self.failure}, ensure_ascii=False), flush=True)
        if error_type is None:
            self.check()


def encoded_report_size(report):
    return len((json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


@contextmanager
def monitored_geometry(protocol):
    from threadpoolctl import threadpool_limits
    with ResourceMonitor(protocol) as monitor:
        with threadpool_limits(limits=protocol['cpu_threads']):
            yield monitor


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def finite(array, label):
    array = np.asarray(array)
    require(np.isfinite(array).all(), label + ' has nonfinite values')
    return array


def rigid_metrics(matrix, limits, label):
    matrix = finite(np.asarray(matrix, dtype=np.float64), label)
    require(matrix.shape == (4, 4), label + ' must be 4x4')
    rotation = matrix[:3, :3]
    orthogonal_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    homogeneous_error = float(np.max(np.abs(matrix[3] - [0., 0., 0., 1.])))
    require(orthogonal_error <= limits['rotation_orthogonality_tolerance'],
            label + ' rotation is not orthogonal')
    require(abs(determinant - 1.) <= limits['rotation_determinant_tolerance'],
            label + ' rotation determinant is not +1')
    require(homogeneous_error <= limits['homogeneous_row_tolerance'],
            label + ' invalid homogeneous row')
    return {'orthogonality_max_abs': orthogonal_error, 'determinant': determinant,
            'homogeneous_row_max_abs': homogeneous_error}


def pose_comparison(xml_poses, mat_ids, mat_poses, expected_ids, limits):
    ids = np.asarray(mat_ids).reshape(-1)
    require(np.issubdtype(ids.dtype, np.integer), 'MAT class indices must be integers')
    ids = [int(item) - 1 for item in ids]
    require(len(set(ids)) == len(ids), 'Duplicate MAT object ID')
    require(set(ids) == set(expected_ids), 'MAT object set mismatch')
    require(len(xml_poses) == len(expected_ids) and set(xml_poses) == set(expected_ids),
            'XML object set mismatch')
    mat_poses = finite(mat_poses, 'MAT poses')
    require(mat_poses.shape == (3, 4, len(ids)), 'Invalid MAT poses shape')
    rows = []
    for object_id, xml_pose in xml_poses.items():
        mat_pose = np.eye(4)
        mat_pose[:3] = mat_poses[:, :, ids.index(object_id)]
        xm = rigid_metrics(xml_pose, limits, 'XML object %d' % object_id)
        mm = rigid_metrics(mat_pose, limits, 'MAT object %d' % object_id)
        delta = np.abs(np.asarray(xml_pose)[:3, 3] - mat_pose[:3, 3])
        # Frobenius distance avoids the arccos(trace) noise floor for float32.
        distance = np.linalg.norm(np.asarray(xml_pose)[:3, :3] - mat_pose[:3, :3])
        angle = float(2. * np.arcsin(np.clip(distance / (2. * np.sqrt(2.)), 0., 1.)))
        require(float(delta.max()) <= limits['translation_abs_tolerance_m'],
                'XML/MAT translation mismatch for %d' % object_id)
        require(angle <= limits['rotation_angle_tolerance_rad'],
                'XML/MAT rotation mismatch for %d' % object_id)
        rows.append({'object_id': object_id, 'translation_abs_difference_m': delta.tolist(),
                     'relative_rotation_angle_rad': angle, 'xml': xm, 'mat': mm,
                     'xml_pose': np.asarray(xml_pose).tolist(), 'mat_pose': mat_pose.tolist()})
    return {'xml_order_0based': list(xml_poses), 'mat_order_0based': ids, 'objects': rows,
            'association': 'XML obj_id equals MAT cls_indexes minus one; never positional pairing'}


def read_xml_poses(path):
    """Validate XML structure, then use the frozen official coordinate parser."""
    raw = Path(path).read_bytes()
    require(b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper(),
            'XML external entity/doctype is not allowed')
    root = ET.fromstring(raw)
    require(root.tag == 'scene', 'Unexpected XML root')
    expected = ['obj_id', 'obj_name', 'obj_path', 'pos_in_world', 'ori_in_world']
    ids = []
    quaternions = {}
    for child in root:
        require(child.tag == 'obj' and [element.tag for element in child] == expected,
                'Unexpected XML object field layout')
        object_id = int(child[0].text)
        require(object_id not in ids, 'Duplicate XML object ID')
        ids.append(object_id)
        translation = finite([float(x) for x in child[3].text.split()], 'XML translation')
        quat = finite([float(x) for x in child[4].text.split()], 'XML quaternion')
        require(translation.shape == (3,) and quat.shape == (4,), 'Invalid XML pose vector length')
        # Four quaternion components rounded to 1e-4 have norm error <= 1e-4.
        require(abs(float(np.linalg.norm(quat)) - 1.) <= .0002,
                'XML quaternion norm inconsistent with four-decimal unit quaternion')
        quaternions[str(object_id)] = quat.tolist()
    from graspnetAPI.utils.xmlhandler import xmlReader
    from graspnetAPI.utils.eval_utils import parse_posevector
    poses = {}
    for vector in xmlReader(str(path)).getposevectorlist():
        object_id, pose = parse_posevector(vector)
        poses[object_id] = pose
    require(list(poses) == ids, 'Official XML parser object order differs')
    return poses, quaternions


def read_sdf(path):
    """Parse the exact official x-fastest layout, with strict payload checks."""
    with Path(path).open('r', encoding='ascii') as stream:
        dims = tuple(int(x) for x in stream.readline().split())
        require(len(dims) == 3 and all(2 <= x <= 1024 for x in dims), 'Invalid SDF dimensions')
        require(int(np.prod(dims)) <= 50_000_000, 'SDF voxel resource limit exceeded')
        origin = finite([float(x) for x in stream.readline().split()], 'SDF origin')
        require(origin.shape == (3,), 'Invalid SDF origin')
        resolution = float(stream.readline())
        require(np.isfinite(resolution) and 0. < resolution <= .1, 'Invalid SDF resolution')
        def strict_lines():
            for line in stream:
                require(bool(line.strip()), 'Blank SDF value line is incompatible with official reader')
                yield line
        data = np.loadtxt(strict_lines(), dtype=np.float64, ndmin=1, comments=None)
    require(data.ndim == 1 and data.size == int(np.prod(dims)), 'SDF payload count/layout mismatch')
    finite(data, 'SDF values')
    require(float(data.min()) < 0. < float(data.max()), 'SDF must contain inside and outside signs')
    require(bool(np.any(np.abs(data) < resolution * np.sqrt(2.) / 2.)), 'SDF has no official near-zero surface band')
    return data.reshape(dims, order='F'), origin, resolution


def vertices_bounds(vertices, limits, label):
    vertices = finite(np.asarray(vertices, dtype=np.float64), label)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 4,
            label + ' invalid vertex array')
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    require(bool(np.all(upper > lower)), label + ' degenerate model bounds')
    diagonal = float(np.linalg.norm(upper - lower))
    require(limits['model_diagonal_min_m'] <= diagonal <= limits['model_diagonal_max_m'],
            label + ' outside predeclared metre-scale plausibility bounds')
    return {'vertices': int(len(vertices)), 'lower_m': lower.tolist(),
            'upper_m': upper.tolist(), 'diagonal_m': diagonal}


def mesh_sdf_audit(ply_vertices, obj_vertices, sdf, origin, resolution, limits):
    from scipy.ndimage import map_coordinates
    ply_vertices = np.asarray(ply_vertices, dtype=np.float64)
    obj_vertices = np.asarray(obj_vertices, dtype=np.float64)
    ply = vertices_bounds(ply_vertices, limits, 'PLY')
    obj = vertices_bounds(obj_vertices, limits, 'OBJ')
    delta = np.abs(np.array([ply['lower_m'], ply['upper_m']]) -
                   np.array([obj['lower_m'], obj['upper_m']]))
    tolerance = max(limits['mesh_bounds_abs_tolerance_m'],
                    limits['mesh_bounds_voxel_tolerance'] * resolution)
    require(float(delta.max()) <= tolerance, 'PLY/OBJ bounds do not align')
    sdf_lower = np.asarray(origin)
    sdf_upper = sdf_lower + (np.array(sdf.shape) - 1) * resolution
    lower, upper = np.array(obj['lower_m']), np.array(obj['upper_m'])
    margin = limits['sdf_bounds_margin_voxels'] * resolution
    require(bool(np.all(lower >= sdf_lower - margin) and np.all(upper <= sdf_upper + margin)),
            'OBJ lies outside SDF grid bounds plus predeclared margin')
    indices = np.unique(np.linspace(0, len(obj_vertices)-1,
                                   min(limits['surface_sample_count'], len(obj_vertices)), dtype=int))
    samples = obj_vertices[indices]
    grid = (samples - origin) / resolution
    # Clipping is only for at-most-one-voxel quantization padding, not rescaling.
    clipping = np.maximum(-grid, grid - (np.array(sdf.shape)-1)).clip(min=0)
    require(float(clipping.max()) <= limits['sdf_bounds_margin_voxels'] + 1e-9,
            'Surface sample lies beyond allowed SDF padding')
    values = map_coordinates(sdf, grid.T, order=1, mode='nearest', prefilter=False)
    absolute_voxels = np.abs(values) / resolution
    q50, q95 = [float(x) for x in np.quantile(absolute_voxels, [.5, .95])]
    require(q95 <= limits['sdf_surface_q95_max_voxels'], 'OBJ surface/SDF distance mismatch')
    return {'ply': ply, 'obj': obj, 'ply_obj_bounds_max_abs_m': float(delta.max()),
            'ply_obj_bounds_tolerance_m': float(tolerance),
            'sdf': {'dimensions': list(sdf.shape), 'origin_m': np.asarray(origin).tolist(),
                    'resolution_m': float(resolution), 'lower_m': sdf_lower.tolist(),
                    'upper_m': sdf_upper.tolist(), 'min_distance_m': float(sdf.min()),
                    'max_distance_m': float(sdf.max()),
                    'negative_voxels': int(np.count_nonzero(sdf < 0)),
                    'positive_voxels': int(np.count_nonzero(sdf > 0)),
                    'surface_sample_count': int(len(indices)),
                    'sample_index_rule': 'unique integer linspace in original OBJ vertex order',
                    'sample_outside_grid_count': int(np.count_nonzero(clipping.max(axis=1) > 0)),
                    'sample_abs_distance_q50_voxels': q50,
                    'sample_abs_distance_q95_voxels': q95,
                    'sample_abs_distance_max_voxels': float(absolute_voxels.max())}}


def expected_data_paths(protocol):
    base = 'scenes/scene_%04d/%s' % (protocol['scene_id'], protocol['camera'])
    frame = '%04d' % protocol['frame_id']
    paths = [base + '/annotations/' + frame + '.xml', base + '/camera_poses.npy',
             base + '/cam0_wrt_table.npy', base + '/meta/' + frame + '.mat']
    for object_id in protocol['object_ids']:
        paths.extend('models/%03d/%s' % (object_id, name) for name in
                     ('nontextured.ply', 'textured.obj', 'textured.sdf'))
    return paths


def runtime_binding(protocol):
    site = ROOT / 'runtime/envs/grasp_eval/Lib/site-packages'
    api = site / 'graspnetAPI'
    api_files = {p.relative_to(api).as_posix(): digest(p) for p in sorted(api.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts}
    require(api_files and fingerprint(api_files) == protocol['installed_api_tree_sha256'],
            'Frozen installed API changed')
    inventory = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    require(inventory == read(ROOT / 'results/P2/grasp_geometry_v7.json')['packages'],
            'Isolated runtime package inventory changed')
    dependencies = {}
    for name in ('numpy', 'scipy', 'open3d', 'transforms3d', 'psutil', 'threadpoolctl.py'):
        directory = site / name
        candidates = [directory] if directory.is_file() else sorted(directory.rglob('*'))
        files = {(p.name if directory.is_file() else p.relative_to(directory).as_posix()): digest(p)
                 for p in candidates
                 if p.is_file() and p.suffix in ('.py', '.pyd', '.dll') and '__pycache__' not in p.parts}
        require(files, 'Missing numerical dependency: ' + name)
        dependencies[name] = {'files': len(files), 'sha256': fingerprint(files)}
    return {'python_version': sys.version, 'python_executable_sha256': digest(sys.executable),
            'inventory': inventory, 'api_tree_sha256': fingerprint(api_files),
            'numerical_dependencies': dependencies}


def make_binding(protocol, config):
    require(protocol['schema'] == SCHEMA and protocol['scene_id'] == 3 and
            protocol['frame_id'] == 0 and protocol['camera'] == 'realsense', 'Unexpected frozen scope')
    require(protocol['object_ids'] == [2, 21, 22, 41, 43, 44, 52, 60, 62], 'Unexpected object inventory')
    frozen = dict(protocol['input_hashes'], **protocol['source_hashes'], **protocol['prior_inputs'])
    frozen[protocol['asset_lock']['path']] = protocol['asset_lock']['sha256']
    for name, expected in frozen.items():
        require(digest(within(ROOT, name)) == expected, 'Frozen dependency changed: ' + name)
    manifest_path = within(ROOT, protocol['data_manifest'])
    asset_audit_path = within(ROOT, protocol['data_audit'])
    manifest, asset_audit = read(manifest_path), read(asset_audit_path)
    require(manifest['status'] == 'verified_geometry_assets', 'Incomplete asset manifest')
    require(asset_audit['status'] == 'passed_integrity_only', 'Incomplete asset integrity audit')
    require(manifest['schema'] == SCHEMA and asset_audit['schema'] == SCHEMA,
            'Unexpected asset evidence schema')
    require(manifest['audit_sha256'] == digest(asset_audit_path), 'Asset audit linkage mismatch')
    require(manifest['binding'] == asset_audit['binding'] and manifest['files'] == asset_audit['files'],
            'Asset evidence differs')
    require(manifest['binding']['protocol_sha256'] == digest(config) and
            manifest['binding']['asset_lock_sha256'] == protocol['asset_lock']['sha256'],
            'Asset evidence protocol/lock mismatch')
    for name, expected in manifest['binding']['source_sha256'].items():
        require(digest(within(ROOT, name)) == expected, 'Asset source code changed: ' + name)
    require(manifest['binding']['prior_inputs'] == protocol['prior_inputs'],
            'Asset prior-input binding differs')
    files = {}
    data_root = within(ROOT, protocol['data_root'])
    require(set(manifest['files']) == set(expected_data_paths(protocol)), 'Asset evidence membership mismatch')
    for name in expected_data_paths(protocol):
        path = within(data_root, name)
        files[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        require(all(manifest['files'][name][key] == value for key, value in files[name].items()),
                'Asset bytes/hash differs from integrity manifest: ' + name)
    return {'protocol_sha256': digest(config), 'manifest_sha256': digest(manifest_path),
            'asset_audit_sha256': digest(asset_audit_path), 'frozen_dependencies': frozen,
            'source_files': {name: digest(ROOT / name) for name in
                             ('scripts/audit_grasp_geometry_v10.py', 'tests/test_grasp_geometry_audit_v10.py')},
            'data_files': files, 'runtime': runtime_binding(protocol)}


def audit_geometry(protocol, monitor):
    monitor.check()
    from scipy.io import loadmat
    import open3d as o3d
    from graspnetAPI.utils.dexnet.grasping.meshpy.obj_file import ObjFile
    from graspnetAPI.utils.eval_utils import create_table_points, transform_points
    data_root = within(ROOT, protocol['data_root'])
    scene = data_root / ('scenes/scene_%04d' % protocol['scene_id']) / protocol['camera']
    frame = '%04d' % protocol['frame_id']
    limits = protocol['geometry_audit']
    poses, quaternions = read_xml_poses(scene / 'annotations' / (frame + '.xml'))
    mat = loadmat(str(scene / 'meta' / (frame + '.mat')))
    comparison = pose_comparison(poses, mat['cls_indexes'], mat['poses'], protocol['object_ids'], limits)
    comparison['raw_xml_quaternions_wxyz'] = quaternions
    camera_poses = np.load(scene / 'camera_poses.npy', allow_pickle=False)
    alignment = np.load(scene / 'cam0_wrt_table.npy', allow_pickle=False)
    require(camera_poses.shape == (256, 4, 4), 'Expected 256 camera poses')
    finite(camera_poses, 'Camera poses')
    all_camera_metrics = [rigid_metrics(pose, limits, 'Camera pose %d' % i)
                          for i, pose in enumerate(camera_poses)]
    camera_pose = camera_poses[protocol['frame_id']]
    alignment_metrics = rigid_metrics(alignment, limits, 'Camera/table alignment')
    table_transform = np.linalg.inv(alignment @ camera_pose)
    table_metrics = rigid_metrics(table_transform, limits, 'Inverse table transform')
    table = transform_points(create_table_points(1., 1., .05, dx=-.5, dy=-.5, dz=-.05, grid_size=.008), table_transform)
    finite(table, 'Transformed table')
    models = []
    for object_id in poses:
        monitor.check()
        directory = data_root / ('models/%03d' % object_id)
        ply = np.asarray(o3d.io.read_point_cloud(str(directory / 'nontextured.ply')).points)
        mesh = ObjFile(str(directory / 'textured.obj')).read()
        obj = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        require(triangles.ndim == 2 and triangles.shape[1] == 3 and len(triangles) > 0,
                'Expected nonempty triangular OBJ faces')
        require(np.issubdtype(triangles.dtype, np.integer) and
                triangles.min() >= 0 and triangles.max() < len(obj), 'OBJ face index out of range')
        sdf, origin, resolution = read_sdf(directory / 'textured.sdf')
        monitor.check()
        result = mesh_sdf_audit(ply, obj, sdf, origin, resolution, limits)
        result.update(object_id=object_id, obj_triangle_count=int(len(triangles)))
        transformed = finite(transform_points(ply, poses[object_id]), 'Camera-space model')
        result['camera_space_bounds_m'] = [transformed.min(axis=0).tolist(), transformed.max(axis=0).tolist()]
        models.append(result)
        monitor.check()
        print('Geometry audit object %03d passed.' % object_id, flush=True)
        del mesh, ply, obj, triangles, sdf, transformed
    return {'pose_comparison': comparison, 'models': models,
            'camera_and_table': {'camera_pose_count': len(camera_poses),
                'max_camera_orthogonality_error': max(x['orthogonality_max_abs'] for x in all_camera_metrics),
                'max_camera_determinant_error': max(abs(x['determinant']-1) for x in all_camera_metrics),
                'alignment': alignment_metrics, 'table_transform': table_metrics,
                'table_to_camera_matrix': table_transform.tolist(), 'table_points': int(len(table)),
                'table_camera_bounds_m': [table.min(axis=0).tolist(), table.max(axis=0).tolist()]}}


def report_for(binding, geometry):
    return {'schema': SCHEMA, 'status': 'passed',
            'protocol_sha256': binding['protocol_sha256'], 'manifest_sha256': binding['manifest_sha256'],
            'asset_audit_sha256': binding['asset_audit_sha256'],
            'binding': binding, 'binding_sha256': fingerprint(binding),
            'geometry': geometry, 'geometry_sha256': fingerprint(geometry),
            'limitations': ['Geometry consistency and metre-scale plausibility are not independent unit certification.',
                'Fixed third-party mirror consistency is not official whole-archive identity verification.',
                'No candidate scoring, dataset AP, segmentation improvement or robot safety tested.',
                'No model rescaling, no remote pickle loading, no modification of official parsers.']}


def verify_report(report, binding):
    require(report.get('schema') == SCHEMA and report.get('status') == 'passed', 'Invalid completed audit status')
    require(report.get('binding') == binding and report.get('binding_sha256') == fingerprint(binding),
            'Completed geometry audit binding changed')
    require(report.get('geometry_sha256') == fingerprint(report['geometry']), 'Geometry report checksum changed')
    require(report == report_for(binding, report['geometry']), 'Unexpected completed audit fields')
    require(len(report['geometry']['models']) == 9 and
            set(row['object_id'] for row in report['geometry']['models']) == {2, 21, 22, 41, 43, 44, 52, 60, 62},
            'Incomplete completed model audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/grasp_geometry_v10.json')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--recompute-check', action='store_true')
    args = parser.parse_args()
    config = within(ROOT, args.config)
    protocol = read(config)
    path = within(ROOT, protocol['geometry_audit_path'])
    require(not (args.check_only and args.recompute_check), 'Choose one check mode')
    require(Path(sys.prefix).resolve() == (ROOT / 'runtime/envs/grasp_eval').resolve(), 'Use frozen isolated geometry environment')
    if args.check_only or args.recompute_check or path.exists():
        require(path.exists(), 'Cannot verify missing geometry report')
        binding = make_binding(protocol, config)
        report = read(path)
        verify_report(report, binding)
        if args.recompute_check:
            with monitored_geometry(protocol) as monitor:
                require(report == report_for(binding, audit_geometry(protocol, monitor)),
                        'Geometry recomputation differs')
                require(binding == make_binding(protocol, config), 'Inputs changed during geometry recomputation')
                monitor.check(output_bytes=encoded_report_size(report))
        print('Completed geometry audit verified unchanged; no output writes.')
        return
    with exclusive_lock(path.with_suffix('.lock')):
        binding = make_binding(protocol, config)
        if path.exists():
            verify_report(read(path), binding)
        else:
            with monitored_geometry(protocol) as monitor:
                geometry = audit_geometry(protocol, monitor)
                require(binding == make_binding(protocol, config), 'Inputs changed during geometry audit')
                report = report_for(binding, geometry)
                verify_report(report, binding)
                monitor.check(output_bytes=encoded_report_size(report))
            # Stop and validate the sampler before the immutable commit. No
            # post-commit resource exception can leave a failed run marked passed.
            monitor.check(output_bytes=encoded_report_size(report))
            atomic_once(path, report)
        print('Geometry audit passed and committed; no scoring performed.')


if __name__ == '__main__':
    main()
