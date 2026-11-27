"""Independent arithmetic and post-seal diagnostics for a single-frame interface."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '2'

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def publish_once(path, data, read_only=False):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, 'Immutable acceptance artifact differs: ' + str(path))
        return
    require(not read_only, 'Missing completed acceptance artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checked_path(root, value):
    require(isinstance(value, str) and not Path(value).is_absolute(), 'Expected a relative project path')
    result = (root / value).resolve()
    require(result.is_relative_to(root.resolve()), 'Path escaped project')
    return result


def independent_input(rgb, depth, factor, size=(240, 320)):
    """Exact frozen numerical preprocessing without labels, targets, or augmentation."""
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and depth.shape == rgb.shape[:2], 'Input shape mismatch')
    require(np.isfinite(factor) and factor > 0, 'Invalid depth scale')
    target = (size[1], size[0])
    colour = np.asarray(Image.fromarray(rgb).resize(target, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    depth_low = np.asarray(Image.fromarray(depth).resize(target, Image.Resampling.NEAREST), dtype=np.float32) / factor
    valid = np.isfinite(depth_low) & (depth_low > 0)
    normalized_depth = np.where(valid, np.minimum(depth_low, 2.0) / 2.0, 0).astype(np.float32)
    colour = (colour - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([colour.transpose(2, 0, 1), normalized_depth[None], valid.astype(np.float32)[None]]), dtype=np.float32)


def independent_nearest(labels, output_shape):
    """Integer half-pixel mapping checked independently of PIL's resize call."""
    height, width = output_shape
    iy = np.minimum(((np.arange(height, dtype=np.int64) * 2 + 1) * labels.shape[0]) // (2 * height), labels.shape[0] - 1)
    ix = np.minimum(((np.arange(width, dtype=np.int64) * 2 + 1) * labels.shape[1]) // (2 * width), labels.shape[1] - 1)
    return labels[iy[:, None], ix[None, :]]


def independent_decode(heads, decoder):
    """Rebuild the decoder with SciPy peak filtering and separate array arithmetic."""
    import torch
    require(set(heads) == {'foreground', 'centers', 'offsets'}, 'Unexpected raw head names')
    height, width = heads['foreground'].shape[-2:]
    for name, channels in (('foreground', 1), ('centers', 1), ('offsets', 2)):
        require(heads[name].shape == (1, channels, height, width) and heads[name].dtype == np.float32
                and np.isfinite(heads[name]).all(), 'Malformed raw head: ' + name)
    # Use the same scalar sigmoid primitive, not the production decode routine.
    foreground = torch.from_numpy(heads['foreground'][0, 0]).sigmoid().numpy()
    centers = torch.from_numpy(heads['centers'][0, 0]).sigmoid().numpy()
    offsets = heads['offsets'][0]
    peaks = (centers == maximum_filter(centers, size=9, mode='constant', cval=-np.inf)) & (centers >= decoder['center_threshold'])
    coordinates = np.argwhere(peaks)
    ranks = np.argsort(-centers[peaks], kind='stable')
    chosen = []
    for index in ranks:
        candidate = coordinates[index]
        if all(np.sum((candidate - other) ** 2) >= 25 for other in chosen):
            chosen.append(candidate)
            if len(chosen) >= decoder['max_centers']:
                break
    output = np.zeros((height, width), dtype=np.int32)
    yy, xx = np.nonzero(foreground >= decoder['foreground_threshold'])
    if not chosen or not len(yy):
        return output, float(centers.max())
    coordinates = np.asarray(chosen)
    destinations = np.column_stack([yy + offsets[1, yy, xx] * height, xx + offsets[0, yy, xx] * width])
    assignment = np.empty(len(yy), dtype=np.int64)
    for start in range(0, len(yy), 2048):
        distances = np.square(destinations[start:start + 2048, None] - coordinates[None]).sum(axis=2)
        assignment[start:start + 2048] = np.argmin(distances, axis=1)
    next_id = 1
    for center_index in range(len(coordinates)):
        membership = assignment == center_index
        if np.count_nonzero(membership) >= decoder['min_pixels']:
            output[yy[membership], xx[membership]] = next_id
            next_id += 1
    return output, float(centers.max())


def independent_cloud(depth, intrinsic, factor, mask):
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all() and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0,
            'Invalid intrinsics')
    require(mask.shape == depth.shape and mask.dtype == np.int32, 'Malformed original instance map')
    yy, xx = np.indices(depth.shape)
    z = depth / factor
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0) & (z <= 2.0)
    pixels = np.flatnonzero(good.ravel()).astype(np.int64)
    points = np.column_stack([x[good], y[good], z[good]]).astype(np.float32)
    labels = mask[good].astype(np.int32)
    projected = np.column_stack([points[:, 0].astype(np.float64) * intrinsic[0, 0] / points[:, 2] + intrinsic[0, 2],
                                 points[:, 1].astype(np.float64) * intrinsic[1, 1] / points[:, 2] + intrinsic[1, 2]])
    expected = np.column_stack([pixels % depth.shape[1], pixels // depth.shape[1]])
    error = float(np.sqrt(np.square(projected - expected).sum(axis=1)).max()) if len(points) else 0.0
    return points, pixels, labels, error


def expected_associations(candidates, sampled, sampled_indices, valid_pixels, valid_labels, width):
    require(candidates.ndim == 2 and candidates.shape[1] == 17 and np.isfinite(candidates).all(), 'Invalid candidates')
    result = []
    for index, candidate in enumerate(candidates):
        matches = np.flatnonzero(np.all(sampled == candidate[13:16], axis=1))
        details = []
        for sample_position in matches:
            valid_index = int(sampled_indices[sample_position])
            pixel = int(valid_pixels[valid_index])
            details.append({'sample_position': int(sample_position), 'valid_point_index': valid_index,
                            'pixel_index': pixel, 'row': pixel // width, 'col': pixel % width,
                            'instance_id': int(valid_labels[valid_index])})
        labels = {item['instance_id'] for item in details}
        if not details:
            status, kind, label = 'unmatched', 'none', None
        elif len(labels) > 1:
            status, kind, label = 'ambiguous', 'duplicate_conflict', None
        else:
            label = next(iter(labels))
            status = 'assigned' if label > 0 else 'background'
            kind = 'unique' if len(details) == 1 else 'duplicate_consensus'
        result.append({'original_index': index, 'candidate_row': candidate.tolist(), 'status': status,
                       'match_kind': kind, 'instance_id': label, 'matches': details,
                       'nearest_distance_m': None if details else float(np.linalg.norm(
                           sampled.astype(np.float64) - candidate[13:16].astype(np.float64), axis=1).min())})
    return result


def independent_metrics(predicted, truth):
    require(predicted.shape == truth.shape, 'Metric shape mismatch')
    pred_ids, true_ids = np.unique(predicted[predicted > 0]), np.unique(truth[truth > 0])
    overlap = np.zeros((len(pred_ids), len(true_ids)), dtype=np.float64)
    for i, pred_id in enumerate(pred_ids):
        for j, true_id in enumerate(true_ids):
            first, second = predicted == pred_id, truth == true_id
            overlap[i, j] = np.count_nonzero(first & second) / max(1, np.count_nonzero(first | second))
    matched = []
    if overlap.size:
        valid = overlap >= .5
        rows, columns = linear_sum_assignment(-(valid.astype(float) * (min(overlap.shape) + 1) + overlap))
        matched = [{'predicted_instance_id': int(pred_ids[i]), 'ground_truth_object_label': int(true_ids[j]),
                    'mask_iou': float(overlap[i, j])} for i, j in zip(rows, columns) if valid[i, j]]
    tp = len(matched)
    fp, fn = len(pred_ids) - tp, len(true_ids) - tp
    intersection = int(np.count_nonzero((predicted > 0) & (truth > 0)))
    union = int(np.count_nonzero((predicted > 0) | (truth > 0)))
    return {'foreground_intersection': intersection, 'foreground_union': union,
            'foreground_pixel_iou': intersection / union if union else 1.0,
            'instances': {'tp': tp, 'fp': fp, 'fn': fn, 'precision_iou50': tp / (tp + fp) if tp + fp else 0.0,
                          'recall_iou50': tp / (tp + fn) if tp + fn else 0.0,
                          'f1_iou50': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0},
            'predicted_instances': len(pred_ids), 'actual_instances': len(true_ids), 'matches': matched}


def colour(identifier):
    import colorsys
    return tuple(int(round(channel * 255)) for channel in colorsys.hsv_to_rgb((identifier * .618033988749895) % 1, .78, .95))


def font(size):
    location = Path('C:/Windows/Fonts/segoeui.ttf')
    return ImageFont.truetype(str(location), size) if location.is_file() else ImageFont.load_default(size=size)


def mask_overlay(rgb, mask, prefix):
    blended = rgb.copy()
    for identifier in np.unique(mask):
        if identifier:
            pixels = mask == identifier
            blended[pixels] = np.round(.48 * rgb[pixels] + .52 * np.asarray(colour(int(identifier)))).astype(np.uint8)
    panel = Image.fromarray(blended).resize((640, 360), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(panel)
    for identifier in np.unique(mask):
        if not identifier:
            continue
        y, x = np.nonzero(mask == identifier)
        cx, cy = int(np.median(x) * 640 / mask.shape[1]), int(np.median(y) * 360 / mask.shape[0])
        text = prefix + str(identifier)
        bounds = draw.textbbox((cx, cy), text, font=font(15), anchor='mm')
        draw.rectangle((bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2), fill=(20, 25, 31))
        draw.text((cx, cy), text, fill='white', font=font(15), anchor='mm')
    return panel


def render(rgb, predicted, truth, associations, metrics):
    canvas = Image.new('RGB', (1320, 930), '#f5f7fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), 'SEGMENTATION TO GRASP-SEED INTERFACE', fill='#162238', font=font(27))
    draw.text((20, 50), 'Training scene 0003 | RealSense | frame 0000 | frozen V8 final model | engineering check only', fill='#40506b', font=font(17))
    raw = Image.fromarray(rgb).resize((640, 360), Image.Resampling.BILINEAR)
    seed_panel = raw.copy()
    seed_draw = ImageDraw.Draw(seed_panel)
    for row in associations:
        for match in row['matches']:
            x, y = match['col'] * 640 / rgb.shape[1], match['row'] * 360 / rgb.shape[0]
            ink = colour(row['instance_id']) if row['status'] == 'assigned' else (240, 240, 240)
            seed_draw.ellipse((x - 2.2, y - 2.2, x + 2.2, y + 2.2), fill=ink, outline=(10, 10, 10), width=1)
    panels = [('A  Original RGB', raw), ('B  Prediction IDs P1..Pn (category-agnostic)', mask_overlay(rgb, predicted, 'P')),
              ('C  Post-seal training-label diagnostic (L = object label)', mask_overlay(rgb, truth, 'L')),
              ('D  Original candidate seed pixels (not gripper poses)', seed_panel)]
    for index, (title, panel) in enumerate(panels):
        x, y = 20 + (index % 2) * 660, 88 + (index // 2) * 404
        draw.text((x, y), title, fill='#162238', font=font(17))
        canvas.paste(panel, (x, y + 28))
    counts = {name: sum(row['status'] == name for row in associations) for name in ('assigned', 'background', 'ambiguous', 'unmatched')}
    draw.text((20, 899), '{} original candidates | assigned {} | background {} | ambiguous {} | unmatched {} | no AP / robot claim'.format(
        len(associations), counts['assigned'], counts['background'], counts['ambiguous'], counts['unmatched']), fill='#40506b', font=font(16))
    buffer = io.BytesIO()
    canvas.save(buffer, format='PNG')
    return buffer.getvalue()


def record_snapshot(paths, root):
    return {path.relative_to(root).as_posix(): {'sha256': digest(path), 'size_bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns} for path in sorted(set(paths))}


def peak_memory_bytes():
    """Windows lifetime working-set peak; checked at soft publication boundaries."""
    import ctypes
    from ctypes import wintypes
    require(os.name == 'nt', 'Frozen acceptance resource accounting requires Windows')

    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                    ('PrivateUsage', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    process = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.argtypes, kernel.GetCurrentProcess.restype = [], wintypes.HANDLE
    process.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    process.GetProcessMemoryInfo.restype = wintypes.BOOL
    memory = ProcessMemory()
    memory.cb = ctypes.sizeof(memory)
    require(bool(process.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb)), 'Working-set peak query failed')
    return int(memory.PeakWorkingSetSize)


def check_memory_limit(limit):
    require(peak_memory_bytes() <= limit, 'Acceptance RSS soft cap exceeded; no completed report may be published')


@contextmanager
def acceptance_lock(root):
    directory = root / 'runtime/instance_grasp_interface_v11/acceptance_lock'
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / '.acceptance.lock').open('a+b')
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


def budget_files(root, outputs):
    """Include both durable recovery proofs and exact output-file pending siblings."""
    paths = [checked_path(root, value) for value in outputs.values()]
    paths += [root / 'results/P3' / name for name in ('instance_interface_v11_resume.json', 'grasp_instance_association_v11_resume.json')]
    result = set()
    for path in paths:
        if path.is_dir():
            result.update(member.resolve() for member in path.rglob('*') if member.is_file())
        else:
            if path.is_file():
                result.add(path.resolve())
            if path.parent.is_dir():
                result.update(member.resolve() for member in path.parent.glob(path.name + '.*') if member.is_file()
                              and (member.name.endswith('.tmp') or '.tmp-' in member.name))
    return result


def label_free_source_audit(path):
    """Inspect allowed loader calls; not an operating-system access trace."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    require(not names.intersection({'GraspNetFrames', 'instance_targets'}), 'Deployment references a target-bearing dataset API')
    imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    require(not any('train_segmentation' in name or 'graspnet_frames' in name for name in imports), 'Deployment imports training data code')
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             ((isinstance(node.func, ast.Name) and node.func.id == 'loadmat') or
              (isinstance(node.func, ast.Attribute) and node.func.attr == 'loadmat'))]
    require(len(calls) == 1, 'Expected one explicit calibration loader call')
    keywords = {item.arg: item.value for item in calls[0].keywords}
    require('variable_names' in keywords and ast.literal_eval(keywords['variable_names']) == ['intrinsic_matrix', 'factor_depth'],
            'MAT variable allowlist differs')
    return {'source_sha256': digest(path), 'restricted_metadata_variables': ['intrinsic_matrix', 'factor_depth'],
            'target_bearing_dataset_api_absent': True, 'training_data_import_absent': True,
            'evidence_kind': 'Bound source inspection and independent arithmetic, not OS-level file access telemetry'}


def checked_seals(root, protocol, config):
    paths = []
    directories = {name: checked_path(root, protocol['outputs'][name + '_dir']) for name in ('perception', 'association')}
    for name, directory in directories.items():
        completion = read(directory / 'completion.json')
        require(completion['schema'] == 11 and completion['status'] == ('completed' if name == 'perception' else 'passed'),
                'Incomplete sealed branch: ' + name)
        report = checked_path(root, protocol['outputs'][name + '_report'])
        require(digest(report) == completion['report_sha256'], 'Sealed report hash differs')
        paths.extend([directory / 'completion.json', report])
        manifest = completion['outputs' if name == 'perception' else 'files']
        for relative, expected in manifest.items():
            path = checked_path(directory, relative)
            require(digest(path) == expected, 'Sealed output hash differs: ' + str(path))
            paths.append(path)
        binding = read(directory / 'binding.json')
        if name == 'perception':
            encoded = json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            require(binding['protocol_sha256'] == digest(config), 'Prediction protocol differs')
            require(binding['ground_truth_labels_read'] is False and binding['object_poses_or_ids_read'] is False
                    and binding['metadata_variables_read'] == ['intrinsic_matrix', 'factor_depth'], 'Prediction input policy differs')
            require(binding['runtime_identity'] == protocol['runtime'], 'Frozen runtime differs')
            for source, expected in binding['sources'].items():
                require(digest(checked_path(root, source)) == expected, 'Prediction source changed')
            require(completion['ground_truth_labels_read'] is False and completion['training_performed'] is False
                    and completion['model_forward_calls_committed'] == 1, 'Prediction activity policy differs')
        else:
            encoded = json.dumps(binding, sort_keys=True, allow_nan=False).encode()
            require(binding['config_sha256'] == digest(config), 'Association protocol differs')
            require(binding['script_sha256'] == digest(root / 'scripts/associate_grasp_instances_v11.py'), 'Association source differs')
            for source, expected in binding['input_hashes'].items():
                require(digest(checked_path(root, source)) == expected, 'Association source/input differs')
        require(hashlib.sha256(encoded).hexdigest() == completion['binding_fingerprint'], 'Branch binding fingerprint differs')
    return directories, record_snapshot(paths, root)


def checked_recovery(root, protocol, config):
    result = {}
    for phase, filename in (('perception', 'instance_interface_v11_resume.json'), ('association', 'grasp_instance_association_v11_resume.json')):
        path = root / 'results/P3' / filename
        proof = read(path)
        require(proof['schema'] == 11 and proof['status'] == 'passed' and proof['phase'] == phase, 'Wrong staged recovery proof')
        binding = proof['binding']
        child_name = 'infer_instance_interface_v11.py' if phase == 'perception' else 'associate_grasp_instances_v11.py'
        wrapper_name = 'run_instance_interface_v11.py' if phase == 'perception' else 'run_grasp_association_v11.py'
        require(binding['protocol_sha256'] == digest(config)
                and binding['child_sha256'] == digest(root / 'scripts' / child_name)
                and binding['wrapper_sha256'] == digest(root / 'scripts' / wrapper_name)
                and binding['recovery_helper_sha256'] == digest(root / 'scripts/interface_recovery_v11.py')
                and binding['interpreter_sha256'] == digest(root / 'runtime/envs/vision/Scripts/python.exe'), 'Recovery binding changed')
        for name, expected in binding['helpers'].items():
            require(digest(checked_path(root, name)) == expected, 'Recovery helper source changed')
        require(proof['byte_and_mtime_identical'] is True and proof['read_only_check_passed'] is True
                and proof['new_training_updates'] == 0, 'Recovery checks incomplete')
        require(proof['prefix_before'] == proof['prefix_after'] and len(proof['prefix_before']) > 0, 'Prefix was not preserved')
        for relative, expected in proof['prefix_before'].items():
            item = checked_path(root, relative)
            require({'sha256': digest(item), 'bytes': item.stat().st_size, 'mtime_ns': item.stat().st_mtime_ns} == expected,
                    'Committed recovery prefix differs from current state')
        require(proof['completion_sha256'] == digest(checked_path(root, protocol['outputs'][phase + '_dir']) / 'completion.json')
                and proof['report_sha256'] == digest(checked_path(root, protocol['outputs'][phase + '_report'])), 'Recovery final seal differs')
        probe, complete, readonly = (proof[key] for key in ('probe_record', 'completion_record', 'read_only_record'))
        if phase == 'perception':
            require(probe['status'] == 'heads_committed' and probe['new_model_forward_calls'] in (0, 1)
                    and complete['status'] in ('completed', 'verified_completed_read_only') and complete['new_model_forward_calls'] == 0
                    and readonly['status'] == 'verified_completed_read_only' and readonly['new_model_forward_calls'] == 0,
                    'Head commit/resume reran inference or was not staged')
            prefix_names = {Path(name).name for name in proof['prefix_before']}
            require({'heads.npz', 'input_tensor.npy', 'heads_commit.json'}.issubset(prefix_names), 'Head prefix omitted durable arrays/commit')
        else:
            require(probe['status'] == 'partial' and probe['completed_candidates'] >= 32
                    and probe['new_blocks'] >= 0 and probe['reused_blocks'] >= 0
                    and probe['new_blocks'] + probe['reused_blocks'] >= 2
                    and complete['status'] == 'passed' and complete['new_blocks'] >= 0 and complete['reused_blocks'] >= 2
                    and complete['new_blocks'] + complete['reused_blocks'] == 15
                    and readonly['status'] == 'passed' and readonly['new_blocks'] == 0 and readonly['reused_blocks'] == 15
                    and readonly['points_reused'] is True and readonly['check_only'] is True, 'Association 2-to-15 recovery differs')
            require(all(any(name.endswith('blocks/block_{:04d}.json'.format(i)) for name in proof['prefix_before']) for i in range(2)),
                    'Association prefix omitted initial blocks')
        result[phase] = {'path': path.relative_to(root).as_posix(), 'sha256': digest(path),
                         'prefix_members': len(proof['prefix_before']), 'prefix_bytes_and_mtime_unchanged': True,
                         'completed_read_only': True, 'probe_record': probe, 'completion_record': complete,
                         'read_only_record': readonly}
    return result


def execute(root, config, check_only=False):
    started = time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import PIL
    import scipy
    import torch
    torch.set_num_threads(2)
    protocol = read(config)
    require(protocol['schema'] == 11 and protocol['frame'] == {'scene': 3, 'camera': 'realsense', 'frame': '0000'}, 'Only frozen frame permitted')
    require(protocol['audit']['ground_truth_read_only_after_prediction_and_association_commits'] is True, 'Missing diagnostic separation rule')
    require(shutil.disk_usage(root).free >= protocol['resources']['disk_reserve_bytes'], 'Disk reserve unavailable')
    check_memory_limit(protocol['resources']['rss_soft_bytes'])
    actual_versions = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
                       'pillow': PIL.__version__, 'torch': str(torch.__version__)}
    require(all(protocol['runtime'][key] == value for key, value in actual_versions.items()), 'Acceptance numerical runtime differs')
    output = checked_path(root, protocol['outputs']['acceptance_report'])
    readonly = check_only or output.exists()
    if check_only:
        require(output.is_file(), 'Read-only acceptance requires completed report')
    old_hashes = dict(protocol['input_hashes'], **protocol['source_hashes'])
    old_hashes.update({item['path']: item['sha256'] for item in protocol['inputs'].values()})
    old_hashes[protocol['checkpoint']['path']] = protocol['checkpoint']['sha256']
    old_hashes[protocol['checkpoint']['manifest_path']] = protocol['checkpoint']['manifest_sha256']
    for relative, expected in old_hashes.items():
        require(digest(checked_path(root, relative)) == expected, 'Frozen old input changed: ' + relative)
    dirs, initial_seals = checked_seals(root, protocol, config)
    recovery = checked_recovery(root, protocol, config)
    source_policy = {name: label_free_source_audit(root / 'scripts' / filename) for name, filename in (
        ('prediction', 'infer_instance_interface_v11.py'), ('association', 'associate_grasp_instances_v11.py'))}
    pdir, adir = dirs['perception'], dirs['association']
    with Image.open(checked_path(root, protocol['inputs']['rgb']['path'])) as stream:
        rgb = np.asarray(stream.convert('RGB'))
    with Image.open(checked_path(root, protocol['inputs']['depth']['path'])) as stream:
        depth = np.asarray(stream).copy()
    meta = loadmat(checked_path(root, protocol['inputs']['meta']['path']), variable_names=['intrinsic_matrix', 'factor_depth'])
    factor, intrinsic = float(np.asarray(meta['factor_depth']).item()), np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
    require(list(depth.shape) == protocol['preprocessing']['original_size'], 'Original input shape differs')
    expected_input = independent_input(rgb, depth, factor, tuple(protocol['preprocessing']['size']))[None]
    actual_input = np.load(pdir / 'input_tensor.npy', allow_pickle=False)
    require(actual_input.dtype == np.float32 and np.array_equal(expected_input, actual_input), 'Frozen preprocessing does not reproduce')
    with np.load(pdir / 'heads.npz', allow_pickle=False) as archive:
        heads = {key: archive[key].copy() for key in archive.files}
    predicted, center_max = independent_decode(heads, protocol['decoder'])
    actual_low = np.load(pdir / 'instances_low.npy', allow_pickle=False)
    require(actual_low.dtype == np.int32 and np.array_equal(actual_low, predicted), 'Independent decoder differs')
    original = independent_nearest(predicted, depth.shape)
    actual_original = np.load(pdir / 'instances_original.npy', allow_pickle=False)
    require(actual_original.dtype == np.int32 and np.array_equal(actual_original, original), 'Original-resolution nearest mask differs')
    ids = np.unique(predicted[predicted > 0]).tolist()
    prediction_counts = {'predicted_instances': len(ids), 'instance_ids': ids,
                         'low_foreground_pixels': int(np.count_nonzero(predicted)), 'original_foreground_pixels': int(np.count_nonzero(original)),
                         'low_pixels_by_instance': {str(i): int(np.count_nonzero(predicted == i)) for i in ids},
                         'original_pixels_by_instance': {str(i): int(np.count_nonzero(original == i)) for i in ids}}
    preport = read(checked_path(root, protocol['outputs']['perception_report']))
    require(preport['prediction_counts'] == prediction_counts, 'Prediction count arithmetic differs')
    points, pixels, labels, projection_error = independent_cloud(depth, intrinsic, factor, original)
    require(len(points) == protocol['association']['valid_point_count'], 'Frozen cloud size differs')
    require(projection_error <= protocol['association']['reprojection_tolerance_pixels'], 'Pixel roundtrip tolerance exceeded')
    indices = np.random.RandomState(protocol['association']['sample_seed']).choice(len(points), protocol['association']['num_points'], replace=False)
    sampling_hash = hashlib.sha256(indices.tobytes()).hexdigest()
    require(sampling_hash == protocol['association']['sampled_indices_sha256'], 'Sampling identity differs')
    expected_arrays = {'valid_points': points, 'valid_pixel_indices': pixels, 'valid_instance_ids': labels,
                       'sampled_indices': indices, 'sampled_points': points[indices], 'sampled_pixel_indices': pixels[indices]}
    for name, expected in expected_arrays.items():
        actual = np.load(adir / (name + '.npy'), allow_pickle=False)
        require(actual.dtype == expected.dtype and np.array_equal(actual, expected), 'Independent point/sample arithmetic differs: ' + name)
    candidates = np.load(checked_path(root, protocol['candidate']['path']), allow_pickle=False)
    require(candidates.dtype == np.float32 and candidates.shape == (236, 17) and np.all(candidates[:, 16] == -1), 'Frozen candidate array differs')
    rows = expected_associations(candidates, points[indices], indices, pixels, labels, depth.shape[1])
    association = read(checked_path(root, protocol['outputs']['association_report']))
    require(association['rows'] == rows and association['uses_ground_truth_labels'] is False and association['uses_object_geometry'] is False
            and association['candidate_scores_modified'] is False, 'Independent candidate tracing differs')
    require(all(row['matches'] for row in rows), 'Original candidate sampled source is missing')
    count_ids, counts = np.unique(labels, return_counts=True)
    per_instance_points = {str(int(i)): int(count) for i, count in zip(count_ids, counts)}
    statuses = {name: sum(row['status'] == name for row in rows) for name in ('assigned', 'background', 'ambiguous', 'unmatched')}
    per_instance_candidates = {str(i): sum(row['status'] == 'assigned' and row['instance_id'] == i for row in rows) for i in ids}
    summary = {'candidate_count': len(rows), 'status_counts': statuses,
               'exact_matched_candidates': sum(bool(row['matches']) for row in rows),
               'duplicate_xyz_candidates': sum(len(row['matches']) > 1 for row in rows),
               'per_instance_candidate_count': {key: value for key, value in per_instance_candidates.items() if value},
               'valid_point_count': len(points), 'per_instance_valid_points': per_instance_points,
               'sampled_indices_sha256': sampling_hash}
    require(association['summary'] == summary and sum(per_instance_points.values()) == len(points), 'Independent summary/count conservation differs')
    # GT is opened only here, after both commits, recovery evidence, and label-free reconstruction passed.
    require(record_snapshot([checked_path(root, name) for name in initial_seals], root) == initial_seals, 'Sealed branches changed before diagnosis')
    audit = protocol['audit']
    label_path, reference_path = (checked_path(root, audit[key]) for key in ('label_path', 'reference_frame_path'))
    require(digest(label_path) == audit['label_sha256'] and digest(reference_path) == audit['reference_frame_sha256'], 'Frozen training diagnostic source differs')
    with Image.open(label_path) as stream:
        truth = np.asarray(stream).copy()
        truth_low = np.asarray(stream.resize((predicted.shape[1], predicted.shape[0]), Image.Resampling.NEAREST), dtype=np.int64)
    require(truth.shape == depth.shape and np.all((truth >= 0) & (truth <= 88)), 'Invalid training object labels')
    low_metrics = independent_metrics(predicted, truth_low)
    foreground_low = torch.from_numpy(heads['foreground'][0, 0]).sigmoid().numpy() >= protocol['decoder']['foreground_threshold']
    fg_intersection = int(np.count_nonzero(foreground_low & (truth_low > 0)))
    fg_union = int(np.count_nonzero(foreground_low | (truth_low > 0)))
    low_metrics.update(foreground_intersection=fg_intersection, foreground_union=fg_union,
                       foreground_pixel_iou=fg_intersection / fg_union if fg_union else 1.0)
    reference = read(reference_path)
    require(reference['identity'] == {'arm': 'coverage_expanded', 'split': 'train', 'scene': 3, 'frame': '0000'}, 'Wrong legacy diagnostic frame')
    expected_metric = dict(low_metrics)
    expected_metric.pop('matches')
    expected_metric.update(scene=3, frame='0000')
    reference_metric = dict(reference['metrics'])
    reference_center = reference_metric.pop('max_center_probability')
    require(reference_metric == expected_metric, 'Frozen V8 training-frame count/rate metrics did not reproduce')
    center_error = abs(reference_center - center_max)
    require(center_error <= audit['center_probability_abs_tolerance'], 'CPU/GPU sigmoid display-value tolerance exceeded')
    original_metrics = independent_metrics(original, truth)
    picture = render(rgb, original, truth, rows, low_metrics)
    picture_path = checked_path(root, protocol['outputs']['visuals_dir']) / 'scene_0003_0000_interface.png'
    final_seals = record_snapshot([checked_path(root, name) for name in initial_seals], root)
    require(final_seals == initial_seals, 'Diagnostic branch modified deployment outputs')
    for relative, expected in old_hashes.items():
        require(digest(checked_path(root, relative)) == expected, 'Old source/input changed during acceptance')
    require(time.monotonic() - started <= protocol['resources']['invocation_soft_seconds'], 'Acceptance time cap exceeded; no report published')
    require(shutil.disk_usage(root).free >= protocol['resources']['disk_reserve_bytes'], 'Disk reserve unavailable before report publication')
    check_memory_limit(protocol['resources']['rss_soft_bytes'])
    report = {'schema': 11, 'status': 'passed', 'decision': 'accepted_limited_single_frame_interface_engineering',
              'frame': protocol['frame'], 'protocol_sha256': digest(config), 'acceptance_script_sha256': digest(Path(__file__)),
              'checks': {'frozen_old_inputs_unchanged': True, 'label_free_input_tensor_exact': True,
                         'independent_raw_head_decode_exact': True, 'original_nearest_mapping_exact': True,
                         'all_point_and_sample_arrays_exact': True, 'all_236_candidates_accounted': True,
                         'sampled_sources_exact_without_nearest_fallback': True, 'point_count_conservation': True,
                         'coordinate_roundtrip_within_tolerance': True, 'staged_resume_and_readonly_verified': True,
                         'training_diagnostic_after_sealed_deployment_outputs': True, 'legacy_v8_training_metrics_exact': True,
                         'deployment_artifacts_unchanged_by_diagnostics': True},
              'source_policy_audit': source_policy, 'prediction_counts': prediction_counts, 'association_summary': summary,
              'maximum_coordinate_error_pixels': projection_error, 'reprojection_tolerance_pixels': protocol['association']['reprojection_tolerance_pixels'],
              'reprojection_error_norm': 'Euclidean L2 distance in original image pixels',
              'post_seal_training_diagnostic': {'label_path': audit['label_path'], 'label_sha256': audit['label_sha256'],
                  'legacy_reference_path': audit['reference_frame_path'], 'legacy_reference_sha256': audit['reference_frame_sha256'],
                  'input_resolution_metrics': low_metrics, 'original_resolution_nearest_metrics': original_metrics,
                  'foreground_metric_definition': 'V8 input resolution uses raw foreground head threshold, not pruned instance mask; original-resolution diagnostic uses resized instance foreground',
                  'max_center_probability': center_max, 'legacy_gpu_max_center_probability': reference_center,
                  'center_probability_cpu_gpu_absolute_error': center_error,
                  'center_probability_abs_tolerance': audit['center_probability_abs_tolerance'],
                  'prediction_and_label_ids_have_different_semantics': True,
                  'used_to_change_predictions_or_associations': False, 'positive_performance_gain_required': False},
              'deployment_seals_before_and_after_diagnostic': initial_seals, 'recovery_evidence': recovery,
              'prior_sha256_verified': old_hashes,
              'visual': {'path': picture_path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(picture).hexdigest(),
                         'candidate_glyph_meaning': 'surface seed pixel only, never gripper pose',
                         'panels_fixed_before_results': True, 'font_sha256': digest(Path('C:/Windows/Fonts/segoeui.ttf'))},
              'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
                          'pillow': PIL.__version__, 'torch': str(torch.__version__), 'model_loaded': False, 'gpu_used': False},
              'new_training_updates': 0, 'new_grasp_generation_calls': 0, 'is_dataset_ap_result': False,
              'resource_guard': 'Windows lifetime PeakWorkingSetSize and elapsed-time soft checks before work/publication; no hard native-call interruption',
              'limitations': protocol['limitations']}
    report_bytes = json_bytes(report)
    existing_paths = budget_files(root, protocol['outputs'])
    existing_total = sum(path.stat().st_size for path in existing_paths)
    incremental_bytes = (0 if picture_path.exists() else len(picture)) + (0 if output.exists() else len(report_bytes))
    require(existing_total + incremental_bytes <= protocol['resources']['output_limit_bytes'], 'Combined interface output cap exceeded')
    publish_once(picture_path, picture, readonly)
    publish_once(output, report_bytes, readonly)
    print(json.dumps({'status': 'verified_completed_read_only' if readonly else 'passed',
                      'prediction_instances': len(ids), 'association_summary': summary,
                      'low_resolution_training_diagnostic': low_metrics['instances'],
                      'report': output.relative_to(root).as_posix()}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config.resolve() if args.config else root / 'configs/instance_grasp_interface_v11.json'
    with acceptance_lock(root):
        execute(root, config, args.check_only)


if __name__ == '__main__':
    main()
