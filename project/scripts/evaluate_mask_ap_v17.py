"""Official visible-mask AP from sealed center-score predictions, in 22 blocks."""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scripts import mask_ap_common_v17 as common
import numpy as np

require = common.require
ARMS = ('rgb', 'rgbd')
SCENES = (2, 6, 7, 11, 13, 15, 18, 22, 33, 39)
METRICS = ('AP', 'AP50', 'AP75')
PAYLOADS = ('inputs.json', 'official.json', 'report.json')


def coco_parameters():
    return {'iouType': 'segm', 'iouThrs': np.linspace(.5, .95, 10).tolist(),
        'recThrs': np.linspace(0., 1., 101).tolist(), 'maxDets': [1, 10, 100],
        'areaRng': [[0, 10000000000], [0, 1024], [1024, 9216], [9216, 10000000000]],
        'areaRngLbl': ['all', 'small', 'medium', 'large'], 'useCats': 1, 'catIds': [1]}


def validate_protocol(protocol):
    require(protocol['schema'] == 17 and tuple(protocol['arms']) == ARMS
        and tuple(protocol['validation_scenes']) == SCENES and protocol['frames'] == list(range(0, 256, 16)), 'Fixed AP cohort changed')
    require(protocol['input_size'] == [240, 320] and protocol['original_size'] == [720, 1280]
        and protocol['decoder'] == {'foreground_threshold': .5, 'center_threshold': .35, 'min_pixels': 20, 'max_centers': 64},
        'Prediction geometry or decoder changed')
    expected = coco_parameters(); policy = protocol['coco_policy']
    require(policy['iou_type'] == 'segm' and policy['iou_thresholds'] == expected['iouThrs']
        and policy['recall_threshold_count'] == 101 and policy['max_dets'] == expected['maxDets']
        and policy['area_ranges'] == expected['areaRng'] and policy['area_names'] == expected['areaRngLbl']
        and policy['metrics'] == list(METRICS), 'Official AP parameters changed')
    require(protocol['mask_policy'] == {'resize': 'integer_repeat_rows_3_columns_4_equivalent_to_nearest',
        'gt': 'all_original_visible_label_ids_greater_than_zero_one_instance_per_id', 'category_id': 1, 'useCats': 1,
        'iscrowd': 0, 'gt_minimum_area_filter': False, 'gt_area': 'original_visible_mask_pixel_count', 'dt_bbox_in_loadRes': False},
        'Visible-mask interpretation changed')


def frames(protocol):
    return [{'scene': scene, 'frame': '%04d' % frame, 'image_id': scene*1000+frame}
        for scene in protocol['validation_scenes'] for frame in protocol['frames']]


def frame_path(directory, item):
    return Path(directory)/'frames'/('scene_%04d_%s' % (item['scene'], item['frame']))


def check_fingerprint(binding):
    body = dict(binding); actual = body.pop('fingerprint')
    require(actual == common.fingerprint(body), 'Binding fingerprint differs')


def check_environment_fingerprint(binding):
    body = dict(binding); actual = body.pop('fingerprint')
    expected = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    require(actual == expected, 'Environment binding fingerprint differs')


def verify_scores(root, config, protocol):
    """All completion inventories are checked before any research label read."""
    seals, files = {}, {}
    expected_ids = [(item['scene'], item['frame']) for item in frames(protocol)]
    for arm in ARMS:
        directory = common.safe(root, protocol['outputs']['scores_root'])/arm
        binding = common.read(directory/'binding.json'); check_fingerprint(binding)
        require(binding['schema'] == 17 and binding['phase'] == 'scores' and binding['arm'] == arm
            and binding['protocol_sha256'] == common.digest(config) and binding['source_hashes'] == protocol['source_hashes']
            and binding['input_hashes'] == protocol['input_hashes'] and binding['expected_frames'] == 160
            and binding['parent_protocol_sha256'] == protocol['parent_protocol_sha256']
            and binding['parent_acceptance_sha256'] == protocol['parent_acceptance_sha256']
            and binding['runtime'] == protocol['runtime'] and binding['decoder'] == protocol['decoder']
            and binding['input_size'] == protocol['input_size'] and binding['precision'] == 'float32_without_autocast',
            'Score source contract differs')
        completion = common.load_commit(directory/'completion.json', binding)
        actual = common.snapshot(directory)
        outputs = {name: row['sha256'] for name, row in actual.items()
            if name not in ('completion.json', 'completion.json.sha256.json')}
        require(completion == {'schema': 17, 'status': 'completed', 'arm': arm,
            'binding_fingerprint': binding['fingerprint'], 'expected_frames': 160, 'outputs': outputs,
            'report_sha256': common.digest(directory/'report.json')}, 'Incomplete or changed score inventory')
        require(all(not name.endswith('.tmp') and '..' not in Path(name).parts for name in outputs), 'Unsealed score temporary')
        report = common.load_commit(directory/'report.json', binding)
        require(report['schema'] == 17 and report['status'] == 'completed_development_mask_scores'
            and report['arm'] == arm and report['expected_frames'] == report['frame_count'] == 160
            and report['binding_fingerprint'] == binding['fingerprint'] and report['parent_predictions_bitwise_equal'] is True
            and report['ground_truth_metric_evaluation_performed'] is False
            and [(row['identity']['scene'], row['identity']['frame']) for row in report['rows']] == expected_ids,
            'Sealed score report differs')
        for item, row in zip(frames(protocol), report['rows']):
            base = frame_path(directory, item)
            require(common.load_commit(base/'record.json', binding) == row and row['identity']['partition'] == 'development'
                and row['identity']['camera'] == 'realsense' and row['arm'] == arm
                and row['parent_instances_and_foreground_bitwise_equal'] is True
                and row['model_inputs_and_scores_use_ground_truth'] is False, 'Score frame record changed')
            for name in ('heads.npz', 'prediction.npz'):
                common.verify_bytes(base/name, binding)
            require(row['heads_sha256'] == common.digest(base/'heads.npz')
                and row['prediction_sha256'] == common.digest(base/'prediction.npz'), 'Frame payload digest differs')
        files.update({(directory/name).relative_to(root).as_posix(): value['sha256'] for name, value in actual.items()})
        seals[arm] = {'directory': directory, 'binding': binding, 'completion': completion, 'rows': report['rows'],
                     'completion_sha256': common.digest(directory/'completion.json')}
    return seals, files


def verify_base_executable(specification):
    base = Path(specification['path']).resolve()
    require(base.is_file() and common.digest(base) == specification['sha256'], 'Base interpreter changed')


def verify_environment(root, protocol):
    require(common.safe(root, protocol['evaluation_interpreter']) == Path(sys.executable).resolve(), 'Wrong isolated AP interpreter')
    report_file = common.safe(root, protocol['environment_report'])
    require(common.digest(report_file) == protocol['environment_report_sha256'], 'Unaccepted evaluator environment')
    report = common.read(report_file); binding = report['binding']; check_environment_fingerprint(binding)
    common.verify_bytes(report_file, binding)
    env_config = common.safe(root, protocol['environment_protocol']); env_protocol = common.read(env_config)
    require(binding['protocol_sha256'] == common.digest(env_config)
        and report['status'] == 'passed' and report['probe']['status'] == 'passed'
        and report['research_frames_read'] == 0, 'Environment did not pass synthetic acceptance')
    require(binding['base_python'] == env_protocol['base_python'], 'Environment base interpreter binding differs')
    verify_base_executable(env_protocol['base_python'])
    directory = common.safe(root, report['environment'])
    actual = {name: {key: row[key] for key in ('sha256', 'bytes')} for name, row in common.snapshot(directory).items()}
    require(actual == report['environment_inventory'], 'Sealed environment inventory changed')
    versions = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
    require(versions == report['package_inventory'], 'AP package inventory changed')
    import PIL
    import pycocotools._mask as compiled_mask
    site = directory/'Lib/site-packages'
    require(all(Path(module.__file__).resolve().is_relative_to(site.resolve()) for module in (np, PIL, compiled_mask)),
        'AP package imported outside sealed environment')
    runtime = {'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': PIL.__version__,
        'pycocotools': importlib.metadata.version('pycocotools')}
    require(runtime == protocol['expected_evaluation_runtime'] and 'torch' not in sys.modules, 'Unexpected AP numerical runtime')
    return runtime


def make_binding(root, config, protocol, score_files, seals):
    common.verify_source_maps(root, protocol)
    for name in ('scripts/evaluate_mask_ap_v17.py', 'scripts/mask_ap_common_v17.py'):
        require(protocol['source_hashes'].get(name) == common.digest(common.safe(root, name)), 'Metric source not frozen')
    runtime = verify_environment(root, protocol)
    parent_file = common.safe(root, protocol['parent_protocol'])
    acceptance = common.safe(root, protocol['parent_acceptance_report'])
    require(common.digest(parent_file) == protocol['parent_protocol_sha256']
        and common.digest(acceptance) == protocol['parent_acceptance_sha256'], 'Parent acceptance changed')
    require(common.read(acceptance)['status'] == 'accepted_complete_paired_development_baseline', 'Parent baseline incomplete')
    parent = common.read(parent_file)
    # Validate source/initial checkpoint maps without importing the model runtime.
    for arm in ARMS:
        binding = seals[arm]['binding']
        require(binding['interpreter_sha256'] == common.digest(common.safe(root, protocol['interpreter'])), 'Extraction interpreter changed')
        for name, sha in binding['parent_training_files'].items():
            require(common.digest(common.safe(root, name)) == sha, 'Parent training source changed')
    value = {'schema': 17, 'phase': 'metrics', 'protocol_sha256': common.digest(config),
        'source_hashes': protocol['source_hashes'], 'input_hashes': protocol['input_hashes'],
        'score_files': score_files, 'score_completion_sha256': {arm: seals[arm]['completion_sha256'] for arm in ARMS},
        'environment_report_sha256': protocol['environment_report_sha256'], 'runtime': runtime,
        'interpreter_sha256': common.digest(sys.executable), 'parameters': coco_parameters(),
        'new_training_updates': 0, 'new_model_forward_calls': 0}
    value['fingerprint'] = common.fingerprint(value)
    return value, parent


def encode_mask(mask):
    from pycocotools import mask as mask_api
    require(mask.ndim == 2 and mask.dtype == np.bool_, 'RLE requires a two-dimensional boolean mask')
    result = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
    require(int(mask_api.area(result)) == int(mask.sum()), 'RLE mask area differs')
    return {'size': [int(v) for v in result['size']], 'counts': result['counts'].decode('ascii')}


def upsample(instances):
    require(instances.dtype == np.int32 and instances.shape == (240, 320) and np.all(instances >= 0), 'Invalid original decoder mask')
    return np.repeat(np.repeat(instances, 3, axis=0), 4, axis=1)


def validate_center_mapping(row, selected_centers, centers):
    require(type(row['selected_center_index']) is int and 0 <= row['selected_center_index'] < len(selected_centers)
        and selected_centers[row['selected_center_index']] == {key: row[key] for key in ('selected_center_index', 'row', 'col', 'score')},
        'Selected-center provenance differs')
    require(type(row['row']) is int and type(row['col']) is int and 0 <= row['row'] < 240 and 0 <= row['col'] < 320
        and type(row['score']) is float and math.isfinite(row['score']) and float(np.float32(.35)) <= row['score'] <= 1
        and row['score'] == float(centers[row['row'], row['col']]), 'Center confidence differs')


def global_ground_truth(root, protocol, parent, guard):
    from PIL import Image
    images, annotations, source = [], [], {}
    for item in frames(protocol):
        file = common.safe(root, parent['data_root'])/'scenes'/('scene_%04d' % item['scene'])/'realsense/label'/(item['frame']+'.png')
        relative = file.relative_to(root).as_posix()
        require(protocol['input_hashes'].get(relative) == common.digest(file), 'Original label must be directly frozen: '+relative)
        with Image.open(file) as image:
            labels = np.array(image)
        require(labels.shape == (720, 1280) and np.issubdtype(labels.dtype, np.integer) and np.all(labels >= 0), 'Invalid original visible labels')
        images.append({'id': item['image_id'], 'height': 720, 'width': 1280})
        for original_id in sorted(int(x) for x in np.unique(labels) if x > 0):
            binary = labels == original_id
            ys, xs = np.nonzero(binary)
            annotation_id = len(annotations)+1
            annotations.append({'id': annotation_id, 'image_id': item['image_id'], 'category_id': 1,
                'segmentation': encode_mask(binary), 'area': int(binary.sum()), 'iscrowd': 0,
                'bbox': [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]})
            source[str(annotation_id)] = dict(item, original_label_id=original_id, label_path=relative, label_sha256=protocol['input_hashes'][relative])
        guard.check()
    return {'schema': 17, 'gt': {'info': {}, 'images': images, 'categories': [{'id': 1, 'name': 'object'}], 'annotations': annotations},
            'source_map': source}


def prediction_rows(root, protocol, parent, seals, arm, guard):
    rows = []
    for item, record in zip(frames(protocol), seals[arm]['rows']):
        file = frame_path(seals[arm]['directory'], item)/'prediction.npz'
        with np.load(file, allow_pickle=False) as archive:
            require(set(archive.files) == {'instances', 'foreground'}, 'Unexpected prediction cache')
            instances, foreground = archive['instances'], archive['foreground']
        require(foreground.dtype == np.bool_ and foreground.shape == (240, 320) and np.all(foreground[instances > 0]), 'Invalid foreground cache')
        old_file = common.safe(root, parent['outputs']['evaluation_root'])/arm/'frames/development'/('scene_%04d_%s' % (item['scene'], item['frame']))/'prediction.npz'
        require(common.digest(old_file) == record['parent_prediction_sha256'], 'Frozen parent mask digest differs')
        with np.load(old_file, allow_pickle=False) as archive:
            require(np.array_equal(instances, archive['instances']) and np.array_equal(foreground, archive['foreground']), 'Decoded parent mask changed')
        with np.load(file.parent/'heads.npz', allow_pickle=False) as archive:
            centers = archive['center_probability']
            require(centers.dtype == np.float32 and centers.shape == (240, 320) and np.isfinite(centers).all(), 'Invalid retained center scores')
        mapping = record['instances']
        require([row['instance_id'] for row in mapping] == list(range(1, len(mapping)+1))
            and np.array_equal(np.unique(instances[instances > 0]), np.arange(1, len(mapping)+1))
            and len(mapping) <= 64, 'Instance-score mapping differs')
        high = upsample(instances)
        for row in mapping:
            validate_center_mapping(row, record['selected_centers'], centers)
            detection = {'image_id': item['image_id'], 'category_id': 1,
                'segmentation': encode_mask(high == row['instance_id']), 'score': row['score']}
            provenance = dict(item, arm=arm, **row, score_record_sha256=common.digest(file.parent/'record.json'),
                parent_prediction_sha256=record['parent_prediction_sha256'])
            rows.append({'detection': detection, 'source': provenance})
        guard.check()
    return rows


def scope_inputs(arm, scope, global_gt, predictions, binding):
    selected = global_gt['gt']['images'] if scope == 'pooled' else [image for image in global_gt['gt']['images'] if image['id']//1000 == int(scope[6:])]
    ids = {image['id'] for image in selected}
    annotations = [row for row in global_gt['gt']['annotations'] if row['image_id'] in ids]
    predictions = [row for row in predictions if row['detection']['image_id'] in ids]
    require([row['detection']['image_id'] for row in predictions] == sorted(row['detection']['image_id'] for row in predictions), 'Detection order changed')
    return {'schema': 17, 'arm': arm, 'scope': scope, 'binding_fingerprint': binding['fingerprint'],
        'gt': dict(global_gt['gt'], images=selected, annotations=annotations),
        'detections': [row['detection'] for row in predictions],
        'source_map': {'gt_annotation_id': {str(row['id']): global_gt['source_map'][str(row['id'])] for row in annotations},
            'dt_list_index': {str(index): dict(row['source'], scope_detection_id=index+1) for index, row in enumerate(predictions)}}}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def official_evaluation(inputs):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(all(set(row) == {'image_id', 'category_id', 'segmentation', 'score'} for row in inputs['detections']), 'Detection bbox or foreign fields are forbidden')
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = copy.deepcopy(inputs['gt']); gt.createIndex()
        if inputs['detections']:
            dt = gt.loadRes(copy.deepcopy(inputs['detections']))
        else:
            dt = COCO(); dt.dataset = {'info': {}, 'images': copy.deepcopy(gt.dataset['images']),
                'categories': copy.deepcopy(gt.dataset['categories']), 'annotations': []}; dt.createIndex()
        evaluation = COCOeval(gt, dt, 'segm')
        evaluation.params.imgIds = [row['id'] for row in inputs['gt']['images']]
        parameters = coco_parameters()
        require(all(json_value(getattr(evaluation.params, key)) == value for key, value in parameters.items()), 'Installed official defaults differ')
        evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
    return {'schema': 17, 'parameters': json_value(dict(parameters, imgIds=evaluation.params.imgIds)),
        'evalImgs': json_value(evaluation.evalImgs), 'precision': json_value(evaluation.eval['precision']),
        'recall': json_value(evaluation.eval['recall']), 'scores': json_value(evaluation.eval['scores']),
        'counts': evaluation.eval['counts'], 'stats': json_value(evaluation.stats)}


def metric_value(values):
    values = np.asarray(values, np.float64)
    require(np.isfinite(values).all() and np.all((values == -1) | ((values >= 0) & (values <= 1))), 'Invalid official precision values')
    available = values[values > -1]
    return {'value': float(available.mean()) if available.size else None,
            'reason': None if available.size else 'no_valid_ground_truth'}


def block_report(inputs, official, binding):
    precision = np.asarray(official['precision'], np.float64)
    require(precision.shape == (10, 101, 1, 4, 3) and official['counts'] == [10, 101, 1, 4, 3]
        and official['parameters'] == dict(coco_parameters(), imgIds=[row['id'] for row in inputs['gt']['images']]), 'Official output geometry differs')
    metrics = {'AP': metric_value(precision[:, :, 0, 0, 2]), 'AP50': metric_value(precision[0, :, 0, 0, 2]),
               'AP75': metric_value(precision[5, :, 0, 0, 2])}
    require(len(official['stats']) == 12 and all(abs(official['stats'][index] - (metrics[key]['value'] if metrics[key]['value'] is not None else -1)) <= 1e-12
        for index, key in enumerate(METRICS)), 'Official summary differs from stored precision')
    return {'schema': 17, 'status': 'completed', 'arm': inputs['arm'], 'scope': inputs['scope'],
        'binding_fingerprint': binding['fingerprint'], 'image_count': len(inputs['gt']['images']),
        'ground_truth_count': len(inputs['gt']['annotations']), 'detection_count': len(inputs['detections']),
        'metrics': metrics, 'metric_scale': 'fraction_0_1', 'all_images_preserved': True, 'official_evaluator_unmodified': True}


def scope_order(protocol):
    return [(arm, scope) for arm in ARMS for scope in ['scene_%04d' % scene for scene in protocol['validation_scenes']]+['pooled']]


def mean_scenes(items):
    available = {scene: row['value'] for scene, row in items.items() if row['value'] is not None}
    return {'value': float(np.mean(list(available.values()))) if available else None,
        'reason': None if available else 'no_eligible_scenes', 'eligible_scenes': [int(scene) for scene in available],
        'excluded_scenes': {scene: row['reason'] for scene, row in items.items() if row['value'] is None}}


def difference(first, second):
    # Named direction is RGBD minus RGB; do not zero-fill undefined metrics.
    valid = first['value'] is not None and second['value'] is not None
    return {'value': second['value']-first['value'] if valid else None, 'reason': None if valid else 'one_or_both_arms_undefined'}


def summarize(protocol, reports, binding):
    arms = {}
    for arm in ARMS:
        scenes = {str(scene): reports[(arm, 'scene_%04d' % scene)]['metrics'] for scene in protocol['validation_scenes']}
        arms[arm] = {'scenes': scenes, 'scene_macro': {key: mean_scenes({scene: row[key] for scene, row in scenes.items()}) for key in METRICS},
                     'pooled': reports[(arm, 'pooled')]['metrics']}
    paired = {'by_scene': {str(scene): {key: difference(arms['rgb']['scenes'][str(scene)][key], arms['rgbd']['scenes'][str(scene)][key])
        for key in METRICS} for scene in protocol['validation_scenes']}}
    for scope in ('scene_macro', 'pooled'):
        paired[scope] = {key: difference(arms['rgb'][scope][key], arms['rgbd'][scope][key]) for key in METRICS}
    return {'schema': 17, 'status': 'completed_development_mask_ap', 'binding': binding,
        'expected_blocks': 22, 'completed_blocks': len(reports), 'arms': arms, 'paired_rgbd_minus_rgb': paired,
        'metric_scale': 'fraction_0_1', 'new_model_forward_calls': 0, 'new_training_updates': 0,
        'held_out_data_used': False, 'thresholds_or_checkpoints_selected': False, 'limitations': protocol['limitations']}


def commit_value(path, value, binding, guard, read_only=False):
    guard.check(0 if Path(path).exists() else len(common.json_bytes(value))+4096)
    common.commit(path, value, binding, read_only)


def verify_block(directory, binding):
    completion = common.load_commit(directory/'completion.json', binding)
    allowed = set(PAYLOADS)|{name+'.sha256.json' for name in PAYLOADS}
    actual = common.snapshot(directory)
    require(set(actual) == allowed|{'completion.json', 'completion.json.sha256.json'}, 'Metric block inventory differs')
    require(completion['schema'] == 17 and completion['status'] == 'completed'
        and completion['binding_fingerprint'] == binding['fingerprint']
        and completion['outputs'] == {name: actual[name]['sha256'] for name in allowed}, 'Metric block seal differs')
    inputs, official, report = (common.load_commit(directory/name, binding) for name in PAYLOADS)
    require(completion['arm'] == inputs['arm'] and completion['scope'] == inputs['scope'], 'Metric completion scope differs')
    require(report == block_report(inputs, official, binding), 'Metric report changed')
    return report


def check_layout(directory, protocol, read_only):
    allowed = {'binding.json', 'completion.json', 'completion.json.sha256.json'}
    directories = set(ARMS)
    stopped = False
    for arm, scope in scope_order(protocol):
        relative = arm+'/'+scope
        directories.add(relative)
        names = set(PAYLOADS)|{name+'.sha256.json' for name in PAYLOADS}|{'completion.json', 'completion.json.sha256.json'}
        allowed.update(relative+'/'+name for name in names)
        block = directory/arm/scope
        complete = (block/'completion.json').exists()
        require(not (complete and stopped), 'Noncontiguous completed AP block prefix')
        if not complete:
            if stopped:
                require(not block.exists() or not any(block.rglob('*')), 'Files beyond the first unfinished AP block')
            stopped = True
    if directory.exists():
        for path in directory.rglob('*'):
            require(not path.is_symlink(), 'Link in metric inventory')
            name = path.relative_to(directory).as_posix()
            if path.is_dir():
                require(name in directories, 'Unknown metric directory')
            elif name not in allowed:
                match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', name)
                require(not read_only and match is not None and match.group(1) in allowed, 'Unexpected metric file')
                path.unlink()  # Exact uncommitted temporary of this revision only.


def execute(root, config, protocol, binding, parent, seals, guard, max_new_blocks=None, check_only=False):
    directory = common.safe(root, protocol['outputs']['metrics_root'])
    output = common.safe(root, protocol['outputs']['metrics_report'])
    complete = (directory/'completion.json').exists()
    read_only = check_only or complete
    require(not check_only or complete, 'Read-only metric check requires completion')
    check_layout(directory, protocol, read_only)
    guard.check(0 if (directory/'binding.json').exists() else len(common.json_bytes(binding))+4096)
    common.once(directory/'binding.json', binding, read_only)
    work = {'schema': 17, 'new_blocks': 0, 'reused_blocks': 0, 'committed_blocks': 0, 'expected_blocks': 22,
        'new_official_evaluations': 0, 'new_model_forward_calls': 0, 'new_training_updates': 0}
    reports, global_gt, predictions = {}, None, {}
    for arm, scope in scope_order(protocol):
        block = directory/arm/scope
        if (block/'completion.json').exists():
            report = verify_block(block, binding); work['reused_blocks'] += 1
        else:
            require(not read_only, 'Missing completed metric block')
            if max_new_blocks is not None and work['new_blocks'] >= max_new_blocks:
                break
            if not (block/'inputs.json').exists():
                if global_gt is None:
                    global_gt = global_ground_truth(root, protocol, parent, guard)
                if arm not in predictions:
                    predictions[arm] = prediction_rows(root, protocol, parent, seals, arm, guard)
                inputs = scope_inputs(arm, scope, global_gt, predictions[arm], binding)
                commit_value(block/'inputs.json', inputs, binding, guard)
            inputs = common.load_commit(block/'inputs.json', binding)
            require(inputs['arm'] == arm and inputs['scope'] == scope and inputs['binding_fingerprint'] == binding['fingerprint'], 'Metric input scope changed')
            if not (block/'official.json').exists():
                guard.check(32*1024**2)
                official = official_evaluation(inputs); work['new_official_evaluations'] += 1
                commit_value(block/'official.json', official, binding, guard)
            official = common.load_commit(block/'official.json', binding)
            report = block_report(inputs, official, binding)
            commit_value(block/'report.json', report, binding, guard)
            names = set(PAYLOADS)|{name+'.sha256.json' for name in PAYLOADS}
            completion = {'schema': 17, 'status': 'completed', 'arm': arm, 'scope': scope,
                'binding_fingerprint': binding['fingerprint'], 'outputs': {name: common.digest(block/name) for name in sorted(names)}}
            commit_value(block/'completion.json', completion, binding, guard)
            verify_block(block, binding); work['new_blocks'] += 1
        require(report['arm'] == arm and report['scope'] == scope, 'Reused metric scope differs')
        reports[(arm, scope)] = report
        work['committed_blocks'] += 1
        guard.check()
    if len(reports) != 22:
        return dict(work, status='partial', resources=guard.check())
    report = summarize(protocol, reports, binding)
    commit_value(output, report, binding, guard, read_only)
    snapshot = common.snapshot(directory)
    completion = {'schema': 17, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
        'expected_blocks': 22, 'outputs': {name: row['sha256'] for name, row in snapshot.items()
            if name not in ('completion.json', 'completion.json.sha256.json')},
        'report_sha256': common.digest(output)}
    commit_value(directory/'completion.json', completion, binding, guard, read_only)
    return dict(work, status='verified_completed_read_only' if read_only else 'completed',
        report_sha256=common.digest(output), resources=guard.check())


def run(root=ROOT, config=None, max_new_blocks=None, check_only=False):
    root = Path(root).resolve(); config = common.safe(root, config or 'configs/mask_ap_v17.json')
    protocol = common.read(config); validate_protocol(protocol)
    require(max_new_blocks is None or type(max_new_blocks) is int and max_new_blocks >= 0, 'Invalid block limit')
    require(not check_only or max_new_blocks is None, 'Read-only check cannot request work')
    guard = common.Guard(root, protocol, 'metrics'); guard.check()
    seals, score_files = verify_scores(root, config, protocol)
    binding, parent = make_binding(root, config, protocol, score_files, seals)
    directory = common.safe(root, protocol['outputs']['metrics_root'])
    if check_only or (directory/'completion.json').exists():
        return execute(root, config, protocol, binding, parent, seals, guard, max_new_blocks, check_only)
    with common.lock(root/'runtime/locks/mask_ap_v17/metrics'):
        return execute(root, config, protocol, binding, parent, seals, guard, max_new_blocks, check_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/mask_ap_v17.json')
    parser.add_argument('--max-new-blocks', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.config, args.max_new_blocks, args.check_only), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
