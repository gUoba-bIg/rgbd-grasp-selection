"""Durable, final-checkpoint RGB/RGB-D development evaluation."""
import argparse
from contextlib import contextmanager
import ctypes
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image

ARMS = ('rgb', 'rgbd')
PARTITIONS = ('train', 'development')
CONFIG = 'configs/segmentation_baseline_v16.json'
LIMITATIONS = [
    'All ten nontraining scenes are previously inspected development data; none is independent confirmation.',
    'One paired seed and 1600 optimizer updates per arm do not establish method superiority or generalization.',
    'Instance F1 uses maximum-cardinality mask IoU >= 0.50 matching and is not mask AP.',
    'Metrics use 240 by 320 prediction and nearest-resized reference labels; scene macro pools counts within each scene first.',
    'Only the final checkpoint is evaluated; no best checkpoint, scene, threshold or seed selection.',
    'This bounded baseline does not complete the full P4 research stage and has no efficacy acceptance gate.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(root, value):
    root = Path(root).resolve()
    target = (root / value).resolve()
    require(target.is_relative_to(root) and target != root, 'Path escapes the project: ' + str(value))
    return target


def read(file):
    return json.loads(Path(file).read_text(encoding='utf-8'))


def digest(file):
    value = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def publish(file, payload, read_only=False):
    file = Path(file)
    if file.exists():
        require(file.read_bytes() == payload, 'Immutable output differs: ' + str(file))
        return
    require(not read_only, 'Missing immutable output: ' + str(file))
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file)


def receipt_path(file):
    return Path(str(file)+'.sha256.json')


def commit_bytes(file, payload, binding, read_only=False):
    receipt = {'schema': 16, 'binding_fingerprint': binding['fingerprint'],
        'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    publish(receipt_path(file), json_bytes(receipt), read_only)
    publish(file, payload, read_only)
    return receipt


def commit(file, value, binding, read_only=False):
    return commit_bytes(file, json_bytes(value), binding, read_only)


def verify_bytes(file, binding):
    file = Path(file)
    receipt = read(receipt_path(file))
    require(receipt == {'schema': 16, 'binding_fingerprint': binding['fingerprint'],
        'sha256': digest(file), 'bytes': file.stat().st_size}, 'Output receipt differs: ' + str(file))
    return receipt


def load_commit(file, binding):
    verify_bytes(file, binding)
    return read(file)


def snapshot(root, files):
    root = Path(root).resolve()
    return {file.resolve().relative_to(root).as_posix(): {'sha256': digest(file),
        'bytes': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns} for file in sorted(files)}


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.lock').open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
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


def peak_rss():
    if os.name != 'nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    require(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),
        ctypes.byref(counters), counters.cb), 'Cannot measure native process peak memory')
    return counters.PeakWorkingSetSize


def rational(numerator, denominator, empty=0):
    value = Fraction(numerator, denominator) if denominator else Fraction(empty)
    return {'numerator': value.numerator, 'denominator': value.denominator, 'value': float(value)}


def metrics_from_counts(tp, fp, fn, intersection, union):
    return {'instances': {'tp': tp, 'fp': fp, 'fn': fn,
        'precision_iou50': rational(tp, tp+fp), 'recall_iou50': rational(tp, tp+fn),
        'f1_iou50': rational(2*tp, 2*tp+fp+fn)},
        'foreground_intersection': intersection, 'foreground_union': union,
        'foreground_pixel_iou': rational(intersection, union, empty=1)}


def frame_metrics(instances, foreground, labels):
    from src.segmentation import instance_counts
    counts = instance_counts(instances, labels, iou_threshold=.5)
    actual = labels > 0
    return dict(metrics_from_counts(*(counts[name] for name in ('tp', 'fp', 'fn')),
        int(np.count_nonzero(foreground & actual)), int(np.count_nonzero(foreground | actual))),
        predicted_instances=int(np.unique(instances[instances > 0]).size),
        actual_instances=int(np.unique(labels[labels > 0]).size))


def aggregate(rows):
    require(rows, 'Cannot summarize an empty selected partition')
    def pooled(items):
        return metrics_from_counts(*(sum(row['metrics']['instances'][key] for row in items)
            for key in ('tp', 'fp', 'fn')), sum(row['metrics']['foreground_intersection'] for row in items),
            sum(row['metrics']['foreground_union'] for row in items))
    scenes = [dict(scene=scene, frame_count=len(items), **pooled(items)) for scene in sorted({row['identity']['scene'] for row in rows})
        for items in [[row for row in rows if row['identity']['scene'] == scene]]]
    macro = {}
    for name in ('instance_f1_iou50', 'foreground_pixel_iou'):
        values = [item['instances']['f1_iou50'] if name == 'instance_f1_iou50' else item[name] for item in scenes]
        value = sum((Fraction(item['numerator'], item['denominator']) for item in values), Fraction()) / len(scenes)
        macro[name] = rational(value.numerator, value.denominator)
    return {'frame_count': len(rows), 'scene_count': len(scenes), 'scenes': scenes,
        'micro': pooled(rows), 'scene_macro': macro}


def paired_comparison(arms):
    output = {}
    for split in PARTITIONS:
        rgb, rgbd = (arms[arm]['partitions'][split] for arm in ARMS)
        require([item['scene'] for item in rgb['scenes']] == [item['scene'] for item in rgbd['scenes']],
            'Paired scene selection differs')
        delta = lambda a, b: rational((Fraction(b['numerator'], b['denominator'])-Fraction(a['numerator'], a['denominator'])).numerator,
            (Fraction(b['numerator'], b['denominator'])-Fraction(a['numerator'], a['denominator'])).denominator)
        output[split] = {'scene_macro_delta_rgbd_minus_rgb': {key: delta(rgb['scene_macro'][key], rgbd['scene_macro'][key])
            for key in ('instance_f1_iou50', 'foreground_pixel_iou')}, 'scenes': [
                {'scene': a['scene'], 'instance_f1_iou50_delta': delta(a['instances']['f1_iou50'], b['instances']['f1_iou50']),
                    'foreground_pixel_iou_delta': delta(a['foreground_pixel_iou'], b['foreground_pixel_iou'])}
                for a, b in zip(rgb['scenes'], rgbd['scenes'])]}
    return output


def frame_identity(split, record):
    return {'partition': split, 'scene': record['scene'], 'frame': record['frame'], 'camera': 'realsense'}


def frame_directory(directory, identity):
    require(identity['partition'] in PARTITIONS and type(identity['scene']) is int
        and re.fullmatch(r'[0-9]{4}', identity['frame']), 'Invalid evaluation frame identity')
    return Path(directory)/'frames'/identity['partition']/('scene_%04d_%s' % (identity['scene'], identity['frame']))


def prediction_bytes(instances, foreground):
    stream = io.BytesIO()
    np.savez_compressed(stream, instances=instances, foreground=foreground)
    return stream.getvalue()


def load_prediction(file, binding, size=(240, 320)):
    verify_bytes(file, binding)
    with np.load(file, allow_pickle=False) as arrays:
        require(set(arrays.files) == {'instances', 'foreground'}, 'Unexpected prediction arrays')
        instances, foreground = arrays['instances'], arrays['foreground']
    require(instances.dtype == np.int32 and instances.shape == tuple(size) and foreground.dtype == np.bool_
        and foreground.shape == tuple(size), 'Prediction dtype/shape differs')
    require(np.all(instances >= 0) and np.max(instances, initial=0) <= 64
        and np.all(foreground[instances > 0]), 'Invalid decoded prediction labels or foreground support')
    require(np.array_equal(np.unique(instances[instances > 0]), np.arange(1, np.max(instances, initial=0)+1)),
        'Decoded instance IDs are not contiguous')
    return instances, foreground


def load_labels(root, protocol, record):
    source = path(root, protocol['data_root']) / record['files']['label']
    require(digest(source) == record['sha256']['label'], 'Reference label checksum differs')
    with Image.open(source) as image:
        values = np.asarray(image)
        require(values.ndim == 2 and values.min() >= 0 and values.max() <= 88, 'Invalid original reference labels')
        labels = np.asarray(image.resize(tuple(reversed(protocol['input_size'])), Image.Resampling.NEAREST), np.int64)
    return labels


def prepare_input(root, protocol, record, arm):
    """Frozen nonaugmented preprocessing; inference never receives reference labels."""
    from scipy.io import loadmat
    directory = path(root, protocol['data_root'])
    shape = tuple(reversed(protocol['input_size']))
    with Image.open(directory/record['files']['rgb']) as image:
        original_size = image.size
        rgb = np.asarray(image.convert('RGB').resize(shape, Image.Resampling.BILINEAR), np.float32)/255.
    rgb = (rgb-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
    channels = [rgb.transpose(2,0,1)]
    if arm == 'rgbd':
        with Image.open(directory/record['files']['depth']) as image:
            require(image.size == original_size, 'Source RGB and depth sizes differ')
            depth = np.asarray(image.resize(shape, Image.Resampling.NEAREST), np.float32)
        metadata = loadmat(directory/record['files']['meta'], variable_names=['factor_depth'])
        factor = float(np.asarray(metadata['factor_depth']).reshape(-1)[0])
        require(math.isfinite(factor) and factor > 0, 'Invalid depth scale')
        depth = depth/factor
        valid = np.isfinite(depth) & (depth > 0)
        normalized = np.where(valid, np.minimum(depth, 2.)/2., 0).astype(np.float32)
        channels.extend([normalized[None], valid.astype(np.float32)[None]])
    return np.ascontiguousarray(np.concatenate(channels), dtype=np.float32)


def predict(model, inputs, protocol):
    import torch
    from src.segmentation import decode_instances
    with torch.inference_mode():
        outputs = model(torch.from_numpy(inputs)[None].to('cuda'))
        require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in outputs.values()),
            'Nonfinite or non-FP32 evaluation heads')
        instances = decode_instances(outputs, **protocol['decoder'])
        foreground = outputs['foreground'][0,0].float().sigmoid().cpu().numpy() >= protocol['decoder']['foreground_threshold']
    return instances, foreground


class Guard:
    """Boundary-sampled budgets, not an operating-system hard timeout."""
    def __init__(self, root, protocol, phase):
        self.root, self.protocol, self.phase = Path(root), protocol, phase
        self.started = time.perf_counter()
        self.gpu = False

    def check(self, extra_bytes=0):
        limits = self.protocol['resources']
        elapsed = time.perf_counter()-self.started
        require(elapsed <= limits[self.phase+'_seconds_per_invocation'], 'Invocation time budget exceeded')
        peak = peak_rss()
        require(peak <= limits['rss_soft_bytes'], 'Process peak memory budget exceeded')
        files = set()
        for name in self.protocol['budget_paths']:
            target = path(self.root, name)
            if target.is_dir():
                files.update(p.resolve() for p in target.rglob('*') if p.is_file())
            elif target.exists():
                files.add(target)
        total = sum(item.stat().st_size for item in files)
        require(total+extra_bytes <= limits['output_limit_bytes'], 'Round output budget exceeded')
        require(shutil.disk_usage(self.root).free-extra_bytes >= limits['disk_reserve_bytes'], 'Disk reserve exhausted')
        allocated = reserved = 0
        if self.gpu:
            import torch
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            require(allocated <= limits['gpu_allocated_soft_bytes'] and reserved <= limits['gpu_reserved_soft_bytes'],
                'GPU memory budget exceeded')
        return {'elapsed_seconds': elapsed, 'peak_rss_bytes': peak, 'peak_gpu_allocated_bytes': allocated,
            'peak_gpu_reserved_bytes': reserved, 'output_bytes': total,
            'scope': 'Boundary-sampled soft limits; an in-flight native call is not preempted'}


def verify_protocol(root, protocol, config_path):
    require(protocol['schema'] == 16 and tuple(protocol['arms']) == ARMS and protocol['input_size'] == [240,320]
        and protocol['seed'] == 20261022 and protocol['planned_steps'] == 1600 and protocol['equivalent_epochs'] == 10,
        'Frozen baseline identity changed')
    require(protocol['decoder'] == {'foreground_threshold': .5, 'center_threshold': .35, 'min_pixels': 20, 'max_centers': 64}
        and protocol['evaluation'] == {'partitions': list(PARTITIONS), 'mode': 'float32_without_autocast',
            'checkpoint': 'final_step_1600_only', 'main_metrics': ['scene_macro_instance_f1_iou50', 'scene_macro_foreground_pixel_iou'],
            'pair_direction': 'rgbd_minus_rgb', 'efficacy_gate': False, 'frame_probe_count': 2}, 'Evaluation rule changed')
    require(protocol['frames'] == list(range(0,256,16)) and len(protocol['train_scenes']) == 20
        and len(protocol['validation_scenes']) == 10 and not set(protocol['train_scenes']) & set(protocol['validation_scenes']),
        'Data scope changed')
    require(protocol['source_hashes'] and protocol['input_hashes'], 'Protocol has not been frozen')
    for name, expected in {**protocol['input_hashes'], **protocol['source_hashes']}.items():
        require(digest(path(root, name)) == expected, 'Frozen input/source changed: '+name)
    require(path(root, protocol['interpreter']).resolve() == Path(sys.executable).resolve(), 'Wrong isolated interpreter')
    require(read(config_path) == protocol, 'Loaded protocol changed')
    if 'runtime' in protocol:
        import importlib.metadata
        runtime = protocol['runtime']
        for package, key in (('torch','torch'),('torchvision','torchvision'),('numpy','numpy'),('scipy','scipy'),('Pillow','Pillow')):
            require(importlib.metadata.version(package) == runtime[key], 'Numerical package version changed: '+package)
        require(sys.version.split()[0] == runtime['python'], 'Python version changed')


def verified_manifest(root, protocol):
    manifest = read(path(root, protocol['data_manifest']))
    audit = read(path(root, protocol['data_audit']))
    require(manifest['schema'] == 8 and manifest['status'] == 'verified_coverage_sample'
        and audit['status'] == 'passed' and audit['files_verified'] == 2040 and not audit['errors'] and not audit['hash_failures'],
        'Frozen data audit is incomplete')
    require(audit['manifest_sha256'] == digest(path(root, protocol['data_manifest']))
        and audit['protocol_sha256'] == manifest['protocol_sha256'] == protocol['data_protocol_sha256']
        and audit['split_sha256'] == manifest['split_sha256'] == digest(path(root, protocol['split'])), 'Data audit binding differs')
    require(set(manifest['records']) == {'train','validation'} and len(manifest['files']) == 2040, 'Unexpected source partitions')
    split = read(path(root, protocol['split']))['groups']
    require(set(protocol['train_scenes']).issubset(split['train']) and set(protocol['validation_scenes']).issubset(split['validation'])
        and not set(protocol['validation_scenes']) & set(protocol['remaining_unopened_validation_scenes']), 'Forbidden scene selection')
    records = {}
    expected_files = set()
    for partition, source, scenes in (('train','train',protocol['train_scenes']), ('development','validation',protocol['validation_scenes'])):
        selected = manifest['records'][source]
        require([(r['scene'],r['frame']) for r in selected] == [(scene,'%04d'%frame) for scene in scenes for frame in protocol['frames']],
            'Missing, extra or reordered evaluation frame')
        for row in selected:
            require(row['camera'] == 'realsense' and set(row['files']) == set(row['sha256']) == {'rgb','depth','meta','label'}, 'Frame modalities differ')
            for kind, relative in row['files'].items():
                wanted = 'scenes/scene_%04d/realsense/%s/%s%s' % (row['scene'],kind,row['frame'],'.mat' if kind=='meta' else '.png')
                require(relative.replace('\\','/') == wanted and row['sha256'][kind] == manifest['files'][wanted]['sha256'],
                    'Frame source membership differs')
                expected_files.add(wanted)
        for scene in scenes:
            prefix = 'scenes/scene_%04d/' % scene
            expected_files.add(prefix+'object_id_list.txt')
            expected_files.update(prefix+'realsense/'+name for name in ('camK.npy','camera_poses.npy','cam0_wrt_table.npy'))
        records[partition] = selected
    require(set(manifest['files']) == expected_files, 'Foreign data in frozen manifest')
    directory = path(root, protocol['data_root'])
    for relative, receipt in manifest['files'].items():
        file = path(directory, relative)
        require(file.stat().st_size == receipt['bytes'] and digest(file) == receipt['sha256'], 'Data bytes changed: '+relative)
    return manifest, records


def training_sources(root, protocol, config_path, arm, manifest):
    """JSON and digest gate for a complete arm; this does not execute a model."""
    directory = path(root, protocol['arms'][arm]['run_directory'])
    filenames = ('config.json','data_manifest.json','initialization.json','history.json','progress.json','result.json','resource_usage.json')
    values = {name: read(directory/name) for name in filenames}
    config, run_data, result = (values[name] for name in ('config.json','data_manifest.json','result.json'))
    require(config['schema'] == 16 and config['arm'] == config['mode'] == arm and config['protocol'] == protocol
        and config['protocol_sha256'] == digest(config_path), 'Training config differs')
    require(run_data['source_manifest'] == manifest and run_data['source_manifest_sha256'] == digest(path(root,protocol['data_manifest']))
        and run_data['records'] == manifest['records'] and run_data['schema'] == 16, 'Training data identity differs')
    config_hash, data_hash = fingerprint(config), fingerprint(run_data)
    require(result['status'] == 'trained' and result['arm'] == arm and result['optimizer_steps'] == result['planned_steps'] == 1600
        and result['cursor'] == {'epoch':10,'next_batch':0,'global_step':1600}
        and result['validation_evaluated'] is False and result['evaluation'] is None
        and result['config_fingerprint'] == config_hash and result['data_fingerprint'] == data_hash, 'Arm is not a sealed final training result')
    require(result['resource_usage_sha256'] == digest(directory/'resource_usage.json')
        and result['resources'] == values['resource_usage.json'], 'Training resource seal differs')
    pointer_file = directory/'checkpoints/latest.json'
    pointer = read(pointer_file)
    metadata_file = path(directory/'checkpoints', pointer['manifest'])
    require(digest(metadata_file) == pointer['manifest_sha256'], 'Checkpoint pointer digest differs')
    metadata = read(metadata_file)
    checkpoint = path(directory/'checkpoints',metadata['file'])
    require(metadata['global_step'] == 1600 and metadata['config_fingerprint'] == config_hash and metadata['data_fingerprint'] == data_hash
        and checkpoint.stat().st_size == metadata['size_bytes'] and digest(checkpoint) == metadata['sha256'] == result['checkpoint_sha256']
        and checkpoint == path(root,result['checkpoint_path']), 'Final model checksum/cursor differs')
    files = [directory/name for name in filenames]+[pointer_file,metadata_file,checkpoint]
    return {'directory': directory, 'values': values, 'checkpoint': checkpoint, 'metadata': metadata,
        'files': {file.relative_to(root).as_posix(): digest(file) for file in files},
        'identity': {'checkpoint_sha256': metadata['sha256'], 'config_fingerprint': config_hash, 'data_fingerprint': data_hash,
            'result_sha256': digest(directory/'result.json'), 'initialization_sha256': digest(directory/'initialization.json'),
            'optimizer_steps': 1600}}


def make_binding(root, protocol, config_path, arm, sources):
    body = {'schema':16,'phase':'evaluation','arm':arm,'protocol_sha256':digest(config_path),
        'source_hashes':protocol['source_hashes'],'input_hashes':protocol['input_hashes'],
        'interpreter_sha256':digest(path(root,protocol['interpreter'])),
        'training_files': {name: sha for source in sources.values() for name,sha in source['files'].items()},
        'final_checkpoints':{key:value['identity'] for key,value in sources.items()},
        'precision':'float32_without_autocast','decoder':protocol['decoder'],'metric_size_hw':protocol['input_size'],
        'evaluation_order':['train','development'],'training_updates':0,'runtime':protocol.get('runtime')}
    return dict(body,fingerprint=fingerprint(body))


def evaluation_targets(directory, records):
    files = {Path(directory)/name for name in ('binding.json','report.json','completion.json')}
    files.update(receipt_path(file) for file in list(files) if file.name != 'binding.json')
    for split in PARTITIONS:
        for record in records[split]:
            parent = frame_directory(directory,frame_identity(split,record))
            for name in ('prediction.npz','record.json'):
                files.update((parent/name,receipt_path(parent/name)))
    return files


def check_inventory(directory, expected, read_only=False):
    directory = Path(directory)
    completed = (directory/'completion.json').exists()
    for file in sorted(directory.rglob('*')) if directory.exists() else []:
        if not file.is_file() or file in expected:
            continue
        match = re.fullmatch(r'(.+)\.([0-9a-f]{32})\.tmp',file.name)
        require(match is not None and file.with_name(match.group(1)) in expected and not read_only and not completed,
            'Unexpected or post-completion temporary file: '+str(file))
        file.unlink()  # Only an exact known sibling temporary below this arm, before completion.


def validate_frame_prefix(directory,records):
    gap = False
    for split in PARTITIONS:
        for record in records[split]:
            parent = frame_directory(directory,frame_identity(split,record))
            prediction,metrics = parent/'prediction.npz',parent/'record.json'
            present = [p.exists() for p in (receipt_path(prediction),prediction,receipt_path(metrics),metrics)]
            require(not gap or not any(present), 'Existing evaluation artifacts follow an incomplete frame')
            require(not present[3] or all(present), 'Committed frame has missing prediction or receipt')
            require(not present[2] or present[0] and present[1], 'Metric intent precedes prediction commitment')
            require(not present[1] or present[0], 'Prediction lacks its prior immutable intent')
            if not all(present):
                gap = True


def arm_report(arm,binding,rows):
    return {'schema':16,'status':'completed_final_checkpoint_development_evaluation','arm':arm,
        'binding_fingerprint':binding['fingerprint'],'checkpoint':binding['final_checkpoints'][arm],
        'frame_count':len(rows),'partitions':{split:aggregate([row for row in rows if row['identity']['partition']==split]) for split in PARTITIONS},
        'model_forward_calls_committed':len(rows),'new_training_updates':0,'checkpoint_selection':'final_step_1600_only',
        'efficacy_gate':False,'limitations':LIMITATIONS}


def verify_completion(directory,binding,records):
    expected = evaluation_targets(directory,records)
    check_inventory(directory,expected,read_only=True)
    require({p for p in directory.rglob('*') if p.is_file()} == expected, 'Completed evaluation inventory differs')
    completion = load_commit(directory/'completion.json',binding)
    files = {p.relative_to(directory).as_posix():digest(p) for p in expected
        if p not in {directory/'completion.json',receipt_path(directory/'completion.json')}}
    require(completion == {'schema':16,'status':'completed','arm':binding['arm'],'binding_fingerprint':binding['fingerprint'],
        'expected_frames':sum(map(len,records.values())),'outputs':files,'report_sha256':digest(directory/'report.json'),
        'model_forward_calls_committed':sum(map(len,records.values()))}, 'Evaluation completion seal differs')
    return completion


def locked_evaluate(root,protocol,config_path,arm,max_new_frames,check_only,guard):
    manifest,records = verified_manifest(root,protocol)
    sources = {name:training_sources(root,protocol,config_path,name,manifest) for name in ARMS}
    binding = make_binding(root,protocol,config_path,arm,sources)
    directory = path(root,protocol['outputs']['evaluation_root'])/arm
    check_inventory(directory,evaluation_targets(directory,records),check_only)
    validate_frame_prefix(directory,records)
    publish(directory/'binding.json',json_bytes(binding),check_only)
    completed = (directory/'completion.json').exists()
    if completed:
        verify_completion(directory,binding,records)
    require(not check_only or completed, 'Read-only evaluation requires a completed arm')
    before = snapshot(root,[p for p in directory.rglob('*') if p.is_file()]) if completed else None
    rows, model, new_frames, forwards, reused, encountered_missing = [],None,0,0,0,False
    total = sum(map(len,records.values()))
    for split in PARTITIONS:
        for source_record in records[split]:
            identity = frame_identity(split,source_record)
            frame = frame_directory(directory,identity)
            prediction_file,record_file = frame/'prediction.npz',frame/'record.json'
            exists = record_file.exists()
            if exists:
                require(not encountered_missing, 'Evaluation frame commits are not a continuous prefix')
                reused += 1
            else:
                encountered_missing = True
                if max_new_frames is not None and new_frames >= max_new_frames:
                    require(not prediction_file.exists() and not receipt_path(prediction_file).exists(), 'Prediction exists beyond stopped prefix')
                    continue
                require(not check_only, 'Missing completed frame record')
            if not prediction_file.exists():
                require(not exists and not completed, 'Missing prediction for committed frame')
                if model is None:
                    import torch
                    from scripts import train_segmentation_baseline_v16 as trainer
                    require(torch.cuda.is_available(), 'CUDA required for a new model evaluation')
                    trainer.reset_random_streams(protocol['seed'],'cuda')
                    model,*_ = trainer.load_final_model(sources[arm]['directory'],protocol)
                    model = model.to('cuda').eval()
                    guard.gpu = True
                    guard.check()
                predicted,foreground = predict(model,prepare_input(root,protocol,source_record,arm),protocol)
                payload = prediction_bytes(predicted,foreground)
                guard.check(len(payload))
                commit_bytes(prediction_file,payload,binding)
                forwards += 1
            instances,foreground = load_prediction(prediction_file,binding,protocol['input_size'])
            # Dataset labels were used for training; here only input imagery enters the model.
            metrics = frame_metrics(instances,foreground,load_labels(root,protocol,source_record))
            row = {'schema':16,'binding_fingerprint':binding['fingerprint'],'arm':arm,'identity':identity,
                'source_sha256':source_record['sha256'],'prediction_sha256':digest(prediction_file),
                'checkpoint_sha256':sources[arm]['identity']['checkpoint_sha256'],
                'prediction_committed_before_metric_label_decoding':True,'model_forward_calls_committed':1,'metrics':metrics}
            commit(record_file,row,binding,read_only=completed or check_only)
            rows.append(row)
            if not exists:
                new_frames += 1
            if len(rows)%16 == 0:
                guard.check()
    done = len(rows) == total
    if done:
        report = arm_report(arm,binding,rows)
        guard.check(len(json_bytes(report)))
        commit(directory/'report.json',report,binding,read_only=completed or check_only)
        expected = evaluation_targets(directory,records)
        files = {p.relative_to(directory).as_posix():digest(p) for p in expected
            if p not in {directory/'completion.json',receipt_path(directory/'completion.json')}}
        completion = {'schema':16,'status':'completed','arm':arm,'binding_fingerprint':binding['fingerprint'],
            'expected_frames':total,'outputs':files,'report_sha256':digest(directory/'report.json'),'model_forward_calls_committed':total}
        guard.check(len(json_bytes(completion)))
        commit(directory/'completion.json',completion,binding,read_only=completed or check_only)
        verify_completion(directory,binding,records)
    for source in sources.values():
        require(all(digest(path(root,name)) == sha for name,sha in source['files'].items()), 'Training source changed during evaluation')
    if before is not None:
        require(before == snapshot(root,[p for p in directory.rglob('*') if p.is_file()]), 'Read-only reentry modified evaluation files')
    return {'schema':16,'status':'verified_completed_read_only' if completed else ('completed' if done else 'partial'),
        'arm':arm,'new_frames':new_frames,'reused_frames':reused,'committed_frames':len(rows),'expected_frames':total,
        'new_model_forward_calls':forwards,'new_training_updates':0,
        'completion_sha256':digest(directory/'completion.json') if done else None,
        'report_sha256':digest(directory/'report.json') if done else None,'resources':guard.check()}


def aggregate_completed(root,protocol,config_path,check_only,guard):
    manifest,records = verified_manifest(root,protocol)
    sources = {arm:training_sources(root,protocol,config_path,arm,manifest) for arm in ARMS}
    reports,seals = {},{}
    for arm in ARMS:
        binding = make_binding(root,protocol,config_path,arm,sources)
        directory = path(root,protocol['outputs']['evaluation_root'])/arm
        require(read(directory/'binding.json') == binding, 'Evaluation binding changed')
        verify_completion(directory,binding,records)
        reports[arm] = load_commit(directory/'report.json',binding)
        seals[arm] = {'binding_fingerprint':binding['fingerprint'],'completion_sha256':digest(directory/'completion.json'),
            'report_sha256':digest(directory/'report.json')}
    report = {'schema':16,'status':'completed_paired_development_baseline','protocol_sha256':digest(config_path),
        'arms':reports,'evaluation_seals':seals,'paired':paired_comparison(reports),
        'model_frames_committed':960,'new_training_updates':0,'efficacy_gate':False,'limitations':LIMITATIONS}
    guard.check(len(json_bytes(report)))
    output = path(root,protocol['outputs']['evaluation_report'])
    publish(output,json_bytes(report),check_only)
    return {'schema':16,'status':'verified_completed_read_only' if check_only else 'completed',
        'aggregate_report_sha256':digest(output),'new_model_forward_calls':0,'new_training_updates':0,'resources':guard.check()}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=CONFIG)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--max-new-frames',type=int)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    return parser.parse_args(argv)


def run(args):
    root = ROOT
    config_path = path(root,args.config)
    protocol = read(config_path)
    guard = Guard(root,protocol,'evaluation')
    guard.check()
    verify_protocol(root,protocol,config_path)
    require(args.aggregate != bool(args.arm), 'Select either one arm or the aggregate')
    require(args.max_new_frames is None or type(args.max_new_frames) is int and 0 <= args.max_new_frames <= 480,
        'Invalid additional-frame budget')
    require(not args.aggregate or args.max_new_frames is None, 'Aggregate cannot generate frames')
    with lock(root/'runtime/locks/segmentation_baseline_evaluation_v16'):
        result = aggregate_completed(root,protocol,config_path,args.check_only,guard) if args.aggregate else locked_evaluate(
            root,protocol,config_path,args.arm,args.max_new_frames,args.check_only,guard)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result


if __name__ == '__main__':
    try:
        run(arguments())
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print('Baseline evaluation stopped: '+str(error),file=sys.stderr,flush=True)
        sys.exit(2)
