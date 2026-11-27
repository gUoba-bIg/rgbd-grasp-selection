"""Official visible-mask AP for four sealed dual-stream conditions, in 88 blocks."""
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
from scripts import evaluate_mask_ap_v17 as official_v17
from scripts import probe_dualstream_resources_v18 as durable
from scripts import extract_dualstream_scores_v18 as extractor
import numpy as np

require = common.require
ARMS = ('fixed', 'quality')
CONDITIONS = ('clean', 'noise', 'missing', 'joint')
CONFIG = 'configs/dualstream_evaluation_v18.json'
SCENES = (2, 6, 7, 11, 13, 15, 18, 22, 33, 39)
METRICS = ('AP', 'AP50', 'AP75')
PAYLOADS = ('inputs.json', 'official.json', 'report.json')


coco_parameters = official_v17.coco_parameters
encode_mask = official_v17.encode_mask
metric_value = official_v17.metric_value
mean_scenes = official_v17.mean_scenes
difference = official_v17.difference
verify_environment = official_v17.verify_environment


def validate_protocol(protocol):
    require(protocol['schema'] == 18 and protocol['status'] == 'frozen'
        and protocol['phase'] == 'development_dualstream_evaluation'
        and tuple(protocol['arms']) == ARMS and tuple(protocol['conditions']) == CONDITIONS
        and tuple(protocol['validation_scenes']) == SCENES and protocol['frames'] == list(range(0, 256, 16)), 'Fixed AP cohort changed')
    require(protocol['input_size'] == [480, 640] and protocol['original_size'] == [720, 1280]
        and protocol['decoder'] == {'foreground_threshold': .5, 'center_threshold': .35, 'min_pixels': 20, 'max_centers': 64}
        and protocol['peak_window'] == 9 and protocol['center_suppression_distance'] == 5
        and protocol['gate_strength'] == 1 and protocol['precision'] == 'float32_without_autocast'
        and protocol['expected_frames_per_arm_condition'] == 160, 'Prediction geometry or decoder changed')
    expected = coco_parameters(); policy = protocol['coco_policy']
    require(policy['iou_type'] == 'segm' and policy['iou_thresholds'] == expected['iouThrs']
        and policy['recall_threshold_count'] == 101 and policy['max_dets'] == expected['maxDets']
        and policy['area_ranges'] == expected['areaRng'] and policy['area_names'] == expected['areaRngLbl']
        and policy['metrics'] == list(METRICS) and policy['pair_direction'] == 'quality_minus_fixed', 'Official AP parameters changed')
    require(protocol['mask_policy'] == {'resize': 'exact_pixel_center_nearest_integer_indices',
        'index_formula': 'floor((2*out_index+1)*source_size/(2*target_size))',
        'gt': 'all_original_visible_label_ids_greater_than_zero_one_instance_per_id', 'category_id': 1, 'useCats': 1,
        'iscrowd': 0, 'gt_minimum_area_filter': False, 'gt_area': 'original_visible_mask_pixel_count', 'dt_bbox_in_loadRes': False},
        'Visible-mask interpretation changed')
    require(protocol['resources'] == {'cpu_threads': 4, 'invocation_soft_seconds': 1800,
        'rss_soft_bytes': 4*1024**3, 'gpu_allocated_soft_bytes': int(13.5*1024**3),
        'gpu_reserved_soft_bytes': int(13.5*1024**3), 'output_limit_bytes': 12*1024**3,
        'disk_reserve_bytes': 60*1024**3}, 'Metric resource contract changed')
    require(protocol['outputs']['metrics_root'] == 'results/P4/dualstream_evaluation_v18/metrics'
        and protocol['outputs']['metrics_report'] == 'results/P4/dualstream_evaluation_v18_metrics.json', 'Metric output scope changed')
    require({protocol['outputs']['metrics_root'], protocol['outputs']['metrics_report'],
        protocol['outputs']['metrics_report']+'.sha256.json'} <= set(protocol['budget_paths']), 'Metric outputs omitted from budget')


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
    """Require all 1,280 sealed score frames before any ground-truth decoding."""
    result = extractor.verify_scores_seals(root, config, protocol)
    seals, files = {}, {}
    for condition in CONDITIONS:
        for arm in ARMS:
            directory = common.safe(root, protocol['outputs']['scores_root'])/condition/arm
            binding = result['bindings'][condition][arm]
            report = result['reports'][condition][arm]
            actual = common.snapshot(directory)
            files.update({(directory/name).relative_to(root).as_posix(): row['sha256'] for name, row in actual.items()})
            seals[(condition, arm)] = {'directory': directory, 'binding': binding, 'rows': report['rows'],
                'completion_sha256': result['seals'][condition][arm]['completion_sha256']}
    return seals, files, result['context']['protocol']


def make_binding(root, config, protocol, score_files, seals):
    common.verify_source_maps(root, protocol)
    for name in ('scripts/evaluate_dualstream_mask_ap_v18.py', 'scripts/evaluate_mask_ap_v17.py',
                 'scripts/mask_ap_common_v17.py', 'scripts/probe_dualstream_resources_v18.py',
                 'scripts/extract_dualstream_scores_v18.py', 'scripts/multiview_common_v14r1.py',
                 'scripts/evaluate_segmentation_baseline_v16.py'):
        require(protocol['source_hashes'].get(name) == common.digest(common.safe(root, name)), 'Metric source not frozen: '+name)
    runtime = verify_environment(root, protocol)
    parent_file = common.safe(root, protocol['parent_training_protocol'])
    report_file = common.safe(root, protocol['parent_training_report'])
    require(common.digest(parent_file) == protocol['parent_training_protocol_sha256']
        and common.digest(report_file) == protocol['parent_training_report_sha256'], 'Training parent changed')
    parent_report = common.read(report_file)
    require(parent_report['schema'] == 18 and parent_report['status'] == 'completed'
        and parent_report['research_optimizer_steps_completed'] == 1600, 'Paired training has not completed')
    for seal in seals.values():
        score = seal['binding']
        require(score['base_python'] == protocol['base_python']
            and score['interpreter_sha256'] == common.digest(common.safe(root, protocol['interpreter'])), 'Score runtime changed')
        for name, checksum in score['parent_training_files'].items():
            require(common.digest(common.safe(root, name)) == checksum, 'Sealed training state changed: '+name)
    official_v17.verify_base_executable(protocol['base_python'])
    value = {'schema': 18, 'phase': 'metrics', 'protocol_sha256': common.digest(config),
        'source_hashes': protocol['source_hashes'], 'input_hashes': protocol['input_hashes'],
        'score_files': score_files, 'score_completion_sha256': {condition: {arm: seals[(condition, arm)]['completion_sha256']
            for arm in ARMS} for condition in CONDITIONS},
        'environment_report_sha256': protocol['environment_report_sha256'], 'runtime': runtime,
        'interpreter_sha256': common.digest(sys.executable), 'parameters': coco_parameters(),
        'new_training_updates': 0, 'new_model_forward_calls': 0}
    value['fingerprint'] = common.fingerprint(value)
    return value


def upsample(instances):
    require(instances.dtype == np.int32 and instances.shape == (480, 640) and np.all(instances >= 0), 'Invalid decoder instance mask')
    rows = ((2*np.arange(720, dtype=np.int64)+1)*480)//(2*720)
    columns = ((2*np.arange(1280, dtype=np.int64)+1)*640)//(2*1280)
    return instances[rows[:, None], columns[None, :]]


def validate_center_mapping(row, selected_centers, centers):
    require(type(row['selected_center_index']) is int and 0 <= row['selected_center_index'] < len(selected_centers)
        and selected_centers[row['selected_center_index']] == {key: row[key] for key in ('selected_center_index', 'row', 'col', 'score')},
        'Selected-center provenance differs')
    require(type(row['row']) is int and type(row['col']) is int and 0 <= row['row'] < 480 and 0 <= row['col'] < 640
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
    return {'schema': 18, 'gt': {'info': {}, 'images': images, 'categories': [{'id': 1, 'name': 'object'}], 'annotations': annotations},
            'source_map': source}


def prediction_rows(root, protocol, parent, seals, condition, arm, guard):
    rows = []
    sealed = seals[(condition, arm)]
    for item, record in zip(frames(protocol), sealed['rows']):
        file = frame_path(sealed['directory'], item)/'prediction.npz'
        durable.verify_bytes(file, sealed['binding'])
        with np.load(file, allow_pickle=False) as archive:
            require(set(archive.files) == {'instances', 'foreground'}, 'Unexpected prediction cache')
            instances, foreground = archive['instances'], archive['foreground']
        require(foreground.dtype == np.bool_ and foreground.shape == (480, 640)
            and instances.dtype == np.int32 and instances.shape == (480, 640)
            and np.all(instances >= 0) and np.all(foreground[instances > 0]), 'Invalid prediction geometry')
        durable.verify_bytes(file.parent/'heads.npz', sealed['binding'])
        with np.load(file.parent/'heads.npz', allow_pickle=False) as archive:
            centers = archive['center_probability']
            require(centers.dtype == np.float32 and centers.shape == (480, 640) and np.isfinite(centers).all(), 'Invalid retained center scores')
        mapping = record['instances']
        require([row['instance_id'] for row in mapping] == list(range(1, len(mapping)+1))
            and np.array_equal(np.unique(instances[instances > 0]), np.arange(1, len(mapping)+1))
            and len(mapping) <= 64, 'Instance-score mapping differs')
        high = upsample(instances)
        for row in mapping:
            validate_center_mapping(row, record['selected_centers'], centers)
            detection = {'image_id': item['image_id'], 'category_id': 1,
                'segmentation': encode_mask(high == row['instance_id']), 'score': row['score']}
            provenance = dict(item, arm=arm, condition=condition, **row,
                score_record_sha256=common.digest(file.parent/'record.json'), prediction_sha256=record['prediction_sha256'])
            rows.append({'detection': detection, 'source': provenance})
        guard.check()
    return rows


def scope_inputs(condition, arm, scope, global_gt, predictions, binding):
    selected = global_gt['gt']['images'] if scope == 'pooled' else [image for image in global_gt['gt']['images'] if image['id']//1000 == int(scope[6:])]
    ids = {image['id'] for image in selected}
    annotations = [row for row in global_gt['gt']['annotations'] if row['image_id'] in ids]
    predictions = [row for row in predictions if row['detection']['image_id'] in ids]
    require([row['detection']['image_id'] for row in predictions] == sorted(row['detection']['image_id'] for row in predictions), 'Detection order changed')
    return {'schema': 18, 'condition': condition, 'arm': arm, 'scope': scope, 'binding_fingerprint': binding['fingerprint'],
        'gt': dict(global_gt['gt'], images=selected, annotations=annotations),
        'detections': [row['detection'] for row in predictions],
        'source_map': {'gt_annotation_id': {str(row['id']): global_gt['source_map'][str(row['id'])] for row in annotations},
            'dt_list_index': {str(index): dict(row['source'], scope_detection_id=index+1) for index, row in enumerate(predictions)}}}


def official_evaluation(inputs):
    return dict(official_v17.official_evaluation(inputs), schema=18)


def block_report(inputs, official, binding):
    require(inputs['schema'] == official['schema'] == 18 and inputs['condition'] in CONDITIONS,
            'Wrong metric payload schema or condition')
    return dict(official_v17.block_report(inputs, official, binding), schema=18, condition=inputs['condition'])


def scope_order(protocol):
    scopes = ['scene_%04d' % scene for scene in protocol['validation_scenes']]+['pooled']
    return [(condition, arm, scope) for condition in CONDITIONS for arm in ARMS for scope in scopes]


def summarize(protocol, reports, binding):
    conditions = {}
    for condition in CONDITIONS:
        arms = {}
        for arm in ARMS:
            scenes = {str(scene): reports[(condition, arm, 'scene_%04d' % scene)]['metrics'] for scene in protocol['validation_scenes']}
            arms[arm] = {'scenes': scenes,
                'scene_macro': {key: mean_scenes({scene: row[key] for scene, row in scenes.items()}) for key in METRICS},
                'pooled': reports[(condition, arm, 'pooled')]['metrics']}
        paired = {'by_scene': {str(scene): {key: difference(arms['fixed']['scenes'][str(scene)][key], arms['quality']['scenes'][str(scene)][key])
            for key in METRICS} for scene in protocol['validation_scenes']}}
        for scope in ('scene_macro', 'pooled'):
            paired[scope] = {key: difference(arms['fixed'][scope][key], arms['quality'][scope][key]) for key in METRICS}
        conditions[condition] = {'arms': arms, 'paired_quality_minus_fixed': paired}
    robustness = {}
    for arm in ARMS:
        values = {condition: conditions[condition]['arms'][arm]['scene_macro']['AP']['value'] for condition in CONDITIONS[1:]}
        available = all(value is not None for value in values.values())
        robustness[arm] = {'value': float(np.mean(list(values.values()))) if available else None,
            'reason': None if available else 'one_or_more_conditions_undefined', 'condition_values': values,
            'conditions': list(CONDITIONS[1:]), 'condition_weight': 1/3}
    return {'schema': 18, 'status': 'completed_development_dualstream_mask_ap', 'binding': binding,
        'expected_blocks': 88, 'completed_blocks': len(reports), 'conditions': conditions,
        'robustness_scene_macro_AP': dict(robustness, quality_minus_fixed=difference(robustness['fixed'], robustness['quality'])),
        'clean_delta_scene_macro_AP': conditions['clean']['paired_quality_minus_fixed']['scene_macro']['AP'],
        'metric_scale': 'fraction_0_1', 'new_model_forward_calls': 0, 'new_training_updates': 0,
        'held_out_data_used': False, 'thresholds_or_checkpoints_selected': False, 'efficacy_gate': False,
        'limitations': protocol['limitations']}


def commit_value(path, value, binding, guard, read_only=False):
    guard.check(0 if Path(path).exists() else len(common.json_bytes(value))+4096)
    durable.commit(path, value, binding, read_only)


def verify_block(directory, binding):
    completion = durable.load_commit(directory/'completion.json', binding)
    allowed = set(PAYLOADS)|{name+'.sha256.json' for name in PAYLOADS}
    actual = common.snapshot(directory)
    require(set(actual) == allowed|{'completion.json', 'completion.json.sha256.json'}, 'Metric block inventory differs')
    require(completion['schema'] == 18 and completion['status'] == 'completed'
        and completion['binding_fingerprint'] == binding['fingerprint']
        and completion['outputs'] == {name: actual[name]['sha256'] for name in allowed}, 'Metric block seal differs')
    inputs, official, report = (durable.load_commit(directory/name, binding) for name in PAYLOADS)
    require(completion['condition'] == inputs['condition'] and completion['arm'] == inputs['arm'] and completion['scope'] == inputs['scope'], 'Metric completion scope differs')
    require(report == block_report(inputs, official, binding), 'Metric report changed')
    return report


def check_layout(directory, protocol, read_only):
    allowed = {'binding.json', 'completion.json', 'completion.json.sha256.json'}
    directories = set(CONDITIONS)|{condition+'/'+arm for condition in CONDITIONS for arm in ARMS}
    stopped = False
    for condition, arm, scope in scope_order(protocol):
        relative = condition+'/'+arm+'/'+scope
        directories.add(relative)
        names = set(PAYLOADS)|{name+'.sha256.json' for name in PAYLOADS}|{'completion.json', 'completion.json.sha256.json'}
        allowed.update(relative+'/'+name for name in names)
        block = directory/condition/arm/scope
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


def metric_snapshot(root, protocol):
    directory = common.safe(root, protocol['outputs']['metrics_root'])
    result = {'metrics/'+name: row for name, row in common.snapshot(directory).items()}
    report = common.safe(root, protocol['outputs']['metrics_report'])
    for file in (report, common.receipt_path(report)):
        if file.exists():
            result[file.relative_to(root).as_posix()] = {'sha256': common.digest(file),
                'bytes': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns}
    return result


def execute(root, config, protocol, binding, parent, seals, guard, max_new_blocks=None, check_only=False):
    directory = common.safe(root, protocol['outputs']['metrics_root'])
    output = common.safe(root, protocol['outputs']['metrics_report'])
    complete = (directory/'completion.json').exists()
    read_only = check_only or complete
    require(not check_only or complete, 'Read-only metric check requires completion')
    before = metric_snapshot(root, protocol) if read_only else None
    check_layout(directory, protocol, read_only)
    guard.check(0 if (directory/'binding.json').exists() else len(common.json_bytes(binding))+4096)
    common.once(directory/'binding.json', binding, read_only)
    work = {'schema': 18, 'new_blocks': 0, 'reused_blocks': 0, 'committed_blocks': 0, 'expected_blocks': 88,
        'new_official_evaluations': 0, 'new_model_forward_calls': 0, 'new_training_updates': 0}
    reports, global_gt, predictions = {}, None, {}
    for condition, arm, scope in scope_order(protocol):
        block = directory/condition/arm/scope
        if (block/'completion.json').exists():
            report = verify_block(block, binding); work['reused_blocks'] += 1
        else:
            require(not read_only, 'Missing completed metric block')
            if max_new_blocks is not None and work['new_blocks'] >= max_new_blocks:
                break
            if not (block/'inputs.json').exists():
                if global_gt is None:
                    global_gt = global_ground_truth(root, protocol, parent, guard)
                if (condition, arm) not in predictions:
                    predictions.clear()  # Keep one condition/arm cohort in memory at a time.
                    predictions[(condition, arm)] = prediction_rows(root, protocol, parent, seals, condition, arm, guard)
                inputs = scope_inputs(condition, arm, scope, global_gt, predictions[(condition, arm)], binding)
                commit_value(block/'inputs.json', inputs, binding, guard)
            inputs = durable.load_commit(block/'inputs.json', binding)
            require(inputs['condition'] == condition and inputs['arm'] == arm and inputs['scope'] == scope and inputs['binding_fingerprint'] == binding['fingerprint'], 'Metric input scope changed')
            if not (block/'official.json').exists():
                guard.check(32*1024**2)
                official = official_evaluation(inputs); work['new_official_evaluations'] += 1
                commit_value(block/'official.json', official, binding, guard)
            official = durable.load_commit(block/'official.json', binding)
            report = block_report(inputs, official, binding)
            commit_value(block/'report.json', report, binding, guard)
            names = set(PAYLOADS)|{name+'.sha256.json' for name in PAYLOADS}
            completion = {'schema': 18, 'status': 'completed', 'condition': condition, 'arm': arm, 'scope': scope,
                'binding_fingerprint': binding['fingerprint'], 'outputs': {name: common.digest(block/name) for name in sorted(names)}}
            commit_value(block/'completion.json', completion, binding, guard)
            verify_block(block, binding); work['new_blocks'] += 1
        require(report['condition'] == condition and report['arm'] == arm and report['scope'] == scope, 'Reused metric scope differs')
        reports[(condition, arm, scope)] = report
        work['committed_blocks'] += 1
        guard.check()
    if len(reports) != 88:
        return dict(work, status='partial', resources=guard.check())
    report = summarize(protocol, reports, binding)
    commit_value(output, report, binding, guard, read_only)
    snapshot = common.snapshot(directory)
    completion = {'schema': 18, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
        'expected_blocks': 88, 'outputs': {name: row['sha256'] for name, row in snapshot.items()
            if name not in ('completion.json', 'completion.json.sha256.json')},
        'report_sha256': common.digest(output)}
    commit_value(directory/'completion.json', completion, binding, guard, read_only)
    if read_only:
        require(before == metric_snapshot(root, protocol), 'Read-only AP verification changed SHA, size or mtime')
    return dict(work, status='verified_completed_read_only' if read_only else 'completed',
        report_sha256=common.digest(output), resources=guard.check())


def run(root=ROOT, config=None, max_new_blocks=None, check_only=False):
    root = Path(root).resolve(); config = common.safe(root, config or CONFIG)
    protocol = common.read(config); validate_protocol(protocol)
    require(max_new_blocks is None or type(max_new_blocks) is int and 0 <= max_new_blocks <= 88, 'Invalid block limit')
    require(not check_only or max_new_blocks is None, 'Read-only check cannot request work')
    guard = extractor.Guard(root, protocol); guard.check()
    seals, score_files, parent = verify_scores(root, config, protocol)
    binding = make_binding(root, config, protocol, score_files, seals)
    directory = common.safe(root, protocol['outputs']['metrics_root'])
    if check_only or (directory/'completion.json').exists():
        return execute(root, config, protocol, binding, parent, seals, guard, max_new_blocks, check_only)
    with common.lock(root/'runtime/locks/dualstream_evaluation_v18/metrics'):
        return execute(root, config, protocol, binding, parent, seals, guard, max_new_blocks, check_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--max-new-blocks', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.config, args.max_new_blocks, args.check_only), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
