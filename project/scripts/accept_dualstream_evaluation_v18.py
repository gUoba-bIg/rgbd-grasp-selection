"""Independent CPU audit of paired dual-stream development predictions and metrics."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
from scripts import accept_mask_ap_v17 as arithmetic
from scripts import mask_ap_common_v17 as storage
from scripts import probe_dualstream_resources_v18 as durable
from scripts import extract_dualstream_scores_v18 as extractor

ARMS = ('fixed', 'quality')
CONDITIONS = ('clean', 'noise', 'missing', 'joint')
QUALITY_SIZES = ((120, 160), (60, 80), (30, 40), (15, 20))
TOLERANCE = 1e-12
QUALITY_DIAGNOSTIC_TOLERANCE = 1e-6
CONFIG = 'configs/dualstream_evaluation_v18.json'
FIGURE = 'paired_dualstream_development_comparison.png'
require = arithmetic.require
independent_decode = arithmetic.independent_decode
decode_rle = arithmetic.decode_rle
dense_overlap = arithmetic.dense_overlap
independent_image_matching = arithmetic.independent_image_matching
independent_ap = arithmetic.independent_ap
compare_metrics = arithmetic.compare_metrics
close_arrays = arithmetic.close_arrays
expected_parameters = arithmetic.expected_parameters
source_labels = arithmetic.source_labels
reference_identity = arithmetic.reference_identity
file_state = arithmetic.file_state


def common():
    return SimpleNamespace(safe=storage.safe, read=storage.read, digest=storage.digest,
        load_commit=durable.load_commit, verify_bytes=durable.verify_bytes,
        receipt_path=durable.receipt, snapshot=storage.snapshot, fingerprint=storage.fingerprint)


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def original_resolution_instances(low):
    require(low.dtype == np.int32 and low.shape == (480, 640) and np.all(low >= 0), 'Invalid prediction grid')
    rows = ((2*np.arange(720, dtype=np.int64)+1)*480)//1440
    columns = ((2*np.arange(1280, dtype=np.int64)+1)*640)//2560
    return low[rows[:, None], columns[None, :]]


def resize_foreground(low):
    require(low.dtype == np.bool_ and low.shape == (480, 640), 'Invalid foreground grid')
    return original_resolution_instances(low.astype(np.int32)).astype(bool)


def preprocess_arrays(rgb, depth, factor):
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
        and depth.shape == rgb.shape[:2] and math.isfinite(factor) and factor > 0,
        'Invalid RGB, depth or calibration input')
    rgb = np.asarray(Image.fromarray(rgb).resize((640, 480), Image.Resampling.BILINEAR), np.float32)/255.0
    depth = np.asarray(Image.fromarray(depth).resize((640, 480), Image.Resampling.NEAREST), np.float32)/factor
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, np.minimum(depth, 2.0)/2.0, 0).astype(np.float32)
    rgb = (rgb-np.asarray([.485, .456, .406], np.float32))/np.asarray([.229, .224, .225], np.float32)
    return np.ascontiguousarray(np.concatenate((rgb.transpose(2, 0, 1), z[None], valid.astype(np.float32)[None])))


def independent_corruption(clean, index, record, condition):
    require(condition in CONDITIONS and clean.dtype == np.float32 and clean.shape == (5, 480, 640)
        and np.isfinite(clean).all(), 'Invalid clean preprocessed input')
    require(np.all((clean[4] == 0) | (clean[4] == 1))
        and np.all((clean[3] >= 0) & (clean[3] <= 1)), 'Invalid clean depth/validity')
    valid = clean[4] == 1
    require(np.all(clean[3][~valid] == 0) and np.all(clean[3][valid] > 0), 'Depth and validity disagree')
    scene, frame = int(record['scene']), int(record['frame'])
    key = 'R18|20261022|0|{}|{}|{}'.format(scene, frame, condition)
    seed = int(hashlib.sha256(key.encode('ascii')).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    z0 = clean[3]*np.float32(2.)
    z1 = z0.copy()
    clipped_zero, clipped_upper = np.zeros_like(valid), np.zeros_like(valid)
    if condition in ('noise', 'joint'):
        candidate = z0+rng.normal(0., .02, size=valid.shape).astype(np.float32)
        clipped_zero = valid & (candidate <= 0)
        clipped_upper = valid & (candidate > 2.)
        z1 = np.where(valid, np.clip(candidate, 0., 2.), 0).astype(np.float32)
    removed = np.zeros_like(valid)
    rectangle = None
    if condition in ('missing', 'joint'):
        top, left = int(rng.randint(0, 361)), int(rng.randint(0, 481))
        removed[top:top+120, left:left+160] = valid[top:top+120, left:left+160]
        rectangle = [top, left, 120, 160]
    final_valid = valid & ~removed & ~clipped_zero
    z1[~final_valid] = 0.
    target = np.zeros(valid.shape, np.float32)
    target[final_valid] = np.exp(-np.abs(z1[final_valid]-z0[final_valid])/np.float32(.02))
    modified = clean.copy()
    modified[3] = z1/np.float32(2.)
    modified[4] = final_valid.astype(np.float32)
    metadata = {'condition': condition, 'seed': seed, 'index': index, 'epoch': 0, 'scene': scene, 'frame': frame,
        'height': 480, 'width': 640, 'missing_rectangle': rectangle,
        'noise_sigma_m': .02 if condition in ('noise', 'joint') else 0., 'quality_decay_m': .02,
        'source_valid_pixels': int(valid.sum()), 'original_invalid_pixels': int((~valid).sum()),
        'added_missing_valid_pixels': int(removed.sum()), 'noise_clipped_zero_pixels': int((clipped_zero & ~removed).sum()),
        'noise_clipped_zero_before_missing_pixels': int(clipped_zero.sum()), 'noise_clipped_upper_pixels': int(clipped_upper.sum()),
        'final_valid_pixels': int(final_valid.sum()), 'mean_quality': float(target.mean(dtype=np.float64))}
    return modified, target[None], metadata


def reconstruct_input(root, context, source, index, condition):
    directory = storage.safe(root, context['protocol']['data_root'])
    paths = {name: storage.safe(directory, source['files'][name]) for name in ('rgb', 'depth', 'meta')}
    require(all(storage.digest(path) == source['sha256'][name] for name, path in paths.items()), 'Input source bytes changed')
    with Image.open(paths['rgb']) as image:
        rgb = np.array(image.convert('RGB'), copy=True)
    with Image.open(paths['depth']) as image:
        depth = np.array(image, copy=True)
    calibration = loadmat(paths['meta'], variable_names=['factor_depth'])
    factor = float(np.asarray(calibration['factor_depth']).reshape(-1)[0])
    clean = preprocess_arrays(rgb, depth, factor)
    inputs, target, metadata = independent_corruption(clean, index, source, condition)
    return inputs, target, metadata, list(rgb.shape[:2])


def maximum_cardinality(iou, threshold=.5):
    overlap = np.asarray(iou, np.float64)
    require(overlap.ndim == 2 and np.isfinite(overlap).all() and np.all((overlap >= 0) & (overlap <= 1)), 'Invalid F1 IoU matrix')
    assigned = [-1]*overlap.shape[1]
    def visit(prediction, seen):
        for truth in np.flatnonzero(overlap[prediction] >= threshold):
            truth = int(truth)
            if truth in seen:
                continue
            seen.add(truth)
            if assigned[truth] < 0 or visit(assigned[truth], seen):
                assigned[truth] = prediction
                return True
        return False
    return sum(int(visit(index, set())) for index in range(overlap.shape[0]))


def auxiliary_segmentation(instances, foreground, labels):
    ids = sorted(int(x) for x in np.unique(instances) if x > 0)
    truths = sorted(int(x) for x in np.unique(labels) if x > 0)
    overlap = dense_overlap(instances, labels, ids, truths)
    tp = maximum_cardinality(overlap['iou'])
    fp, fn = len(ids)-tp, len(truths)-tp
    intersection = int(np.count_nonzero(foreground & (labels > 0)))
    union = int(np.count_nonzero(foreground | (labels > 0)))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'foreground_intersection': intersection, 'foreground_union': union}


def quality_diagnostics(heads, target):
    require(target.dtype == np.float32 and target.shape == (1, 480, 640)
        and np.isfinite(target).all() and np.all((target >= 0) & (target <= 1)), 'Invalid independent quality target')
    means, losses, target_means = [], [], []
    for index, (height, width) in enumerate(QUALITY_SIZES):
        quality = heads['quality_'+str(index)]
        require(quality.dtype == np.float32 and quality.shape == (height, width)
            and np.isfinite(quality).all() and np.all((quality >= 0) & (quality <= 1)), 'Invalid quality prediction')
        area = target[0].reshape(height, 480//height, width, 640//width).mean(axis=(1, 3), dtype=np.float64)
        means.append(float(quality.mean(dtype=np.float64)))
        losses.append(float(np.mean((quality.astype(np.float64)-area)**2, dtype=np.float64)))
        target_means.append(float(area.mean(dtype=np.float64)))
    return {'scale_mse': losses, 'mean_four_scale_mse': math.fsum(losses)/4,
            'scale_quality_mean': means, 'mean_four_scale_quality': math.fsum(means)/4,
            'scale_target_mean': target_means, 'target_mean': float(target.mean(dtype=np.float64)),
            'is_calibration_probability': False, 'mse_independently_computed_without_production_mse': True}


def inspect_scores(root, protocol, context, sealed, guard):
    outputs = {c: {a: {'frames': {}, 'report': sealed['reports'][c][a], 'binding': sealed['bindings'][c][a]}
                   for a in ARMS} for c in CONDITIONS}
    auxiliary = {c: {a: [] for a in ARMS} for c in CONDITIONS}
    for condition in CONDITIONS:
        for index, source in enumerate(context['records']):
            inputs, target, metadata, shape = reconstruct_input(root, context, source, index, condition)
            require(shape == [720, 1280], 'Original input geometry differs')
            for arm in ARMS:
                binding = sealed['bindings'][condition][arm]
                directory = storage.safe(root, protocol['outputs']['scores_root'])/condition/arm
                frame = extractor.frame_directory(directory, source)
                info = durable.load_commit(frame/'input.json', binding)
                require(info['input_sha256'] == array_digest(inputs) and info['quality_target_sha256'] == array_digest(target)
                    and info['corruption'] == metadata and info['original_size_hw'] == shape
                    and info['quality_target_is_model_input'] is False and info['reference_labels_decoded'] is False,
                    'Independent RGB/depth corruption input or quality target differs')
                durable.verify_bytes(frame/'heads.npz', binding)
                with np.load(frame/'heads.npz', allow_pickle=False) as archive:
                    heads = {name: archive[name] for name in archive.files}
                require(set(heads) == extractor.HEAD_KEYS and heads['foreground_probability'].shape == (480, 640), 'Wrong cached heads')
                rebuilt = independent_decode(heads['foreground_probability'], heads['center_probability'], heads['offsets'], protocol['decoder'])
                durable.verify_bytes(frame/'prediction.npz', binding)
                with np.load(frame/'prediction.npz', allow_pickle=False) as prediction:
                    require(set(prediction.files) == {'instances', 'foreground'} and prediction['instances'].dtype == np.int32
                        and prediction['foreground'].dtype == np.bool_ and np.array_equal(prediction['instances'], rebuilt['instances'])
                        and np.array_equal(prediction['foreground'], rebuilt['foreground']), 'Independent decoder differs')
                row = durable.load_commit(frame/'record.json', binding)
                centers = [{'selected_center_index': n, 'row': point[0], 'col': point[1],
                    'score': float(heads['center_probability'][point[0], point[1]])} for n, point in enumerate(rebuilt['selected_centers'])]
                require(row['instances'] == rebuilt['retained_instances'] and row['selected_centers'] == centers,
                        'Selected center scores or instance IDs differ')
                quality = quality_diagnostics(heads, target)
                close_arrays(row['quality_means'], quality['scale_quality_mean'], 'Stored quality means differ')
                image_id = source['scene']*1000+int(source['frame'])
                outputs[condition][arm]['frames'][image_id] = {'source': source, 'row': row, 'prediction_file': frame/'prediction.npz'}
                # Every cohort was sealed before this first reference-label read.
                labels, _ = source_labels(root, protocol, context, source)
                counts = auxiliary_segmentation(original_resolution_instances(rebuilt['instances']), resize_foreground(rebuilt['foreground']), labels)
                auxiliary[condition][arm].append({'scene': source['scene'], 'frame': source['frame'], **counts, 'quality': quality})
            guard.check()
    return outputs, auxiliary


def metric_delta(first, second):
    defined = first['value'] is not None and second['value'] is not None
    return {'value': second['value']-first['value'] if defined else None,
            'reason': None if defined else 'one_or_both_arms_undefined'}


def independent_aggregate(protocol, reports, binding):
    metrics = ('AP', 'AP50', 'AP75')
    conditions = {}
    for condition in CONDITIONS:
        arms = {}
        for arm in ARMS:
            scenes = {str(scene): reports[(condition, arm, 'scene_%04d'%scene)]['metrics'] for scene in protocol['validation_scenes']}
            macro = {}
            for name in metrics:
                available = {scene: row[name]['value'] for scene, row in scenes.items() if row[name]['value'] is not None}
                macro[name] = {'value': math.fsum(available.values())/len(available) if available else None,
                    'reason': None if available else 'no_eligible_scenes', 'eligible_scenes': [int(s) for s in available],
                    'excluded_scenes': {scene: row[name]['reason'] for scene, row in scenes.items() if row[name]['value'] is None}}
            arms[arm] = {'scenes': scenes, 'scene_macro': macro, 'pooled': reports[(condition, arm, 'pooled')]['metrics']}
        paired = {'by_scene': {str(scene): {name: metric_delta(arms['fixed']['scenes'][str(scene)][name],
                    arms['quality']['scenes'][str(scene)][name]) for name in metrics} for scene in protocol['validation_scenes']}}
        for scope in ('scene_macro', 'pooled'):
            paired[scope] = {name: metric_delta(arms['fixed'][scope][name], arms['quality'][scope][name]) for name in metrics}
        conditions[condition] = {'arms': arms, 'paired_quality_minus_fixed': paired}
    robustness = {}
    for arm in ARMS:
        values = {c: conditions[c]['arms'][arm]['scene_macro']['AP']['value'] for c in CONDITIONS[1:]}
        defined = all(v is not None for v in values.values())
        robustness[arm] = {'value': math.fsum(values.values())/3 if defined else None,
            'reason': None if defined else 'one_or_more_conditions_undefined', 'condition_values': values,
            'conditions': list(CONDITIONS[1:]), 'condition_weight': 1/3}
    return {'schema': 18, 'status': 'completed_development_dualstream_mask_ap', 'binding': binding,
        'expected_blocks': 88, 'completed_blocks': len(reports), 'conditions': conditions,
        'robustness_scene_macro_AP': dict(robustness, quality_minus_fixed=metric_delta(robustness['fixed'], robustness['quality'])),
        'clean_delta_scene_macro_AP': conditions['clean']['paired_quality_minus_fixed']['scene_macro']['AP'],
        'metric_scale': 'fraction_0_1', 'new_model_forward_calls': 0, 'new_training_updates': 0,
        'held_out_data_used': False, 'thresholds_or_checkpoints_selected': False, 'efficacy_gate': False,
        'limitations': protocol['limitations']}


def auxiliary_aggregate(protocol, rows):
    def summarize(items):
        counts = {key: sum(row[key] for row in items) for key in ('tp', 'fp', 'fn', 'foreground_intersection', 'foreground_union')}
        denominator = 2*counts['tp']+counts['fp']+counts['fn']
        values = {'instance_f1': 2*counts['tp']/denominator if denominator else None,
            'foreground_iou': counts['foreground_intersection']/counts['foreground_union'] if counts['foreground_union'] else None}
        values.update({name: math.fsum(row['quality'][key] for row in items)/len(items) for name, key in
            (('quality_mse', 'mean_four_scale_mse'), ('quality_mean', 'mean_four_scale_quality'), ('target_mean', 'target_mean'))})
        values['quality_scale_mse'] = [math.fsum(row['quality']['scale_mse'][i] for row in items)/len(items) for i in range(4)]
        values['quality_scale_mean'] = [math.fsum(row['quality']['scale_quality_mean'][i] for row in items)/len(items) for i in range(4)]
        return {'frame_count': len(items), 'counts': counts, 'metrics': values}
    result = {}
    names = ('instance_f1', 'foreground_iou', 'quality_mse', 'quality_mean', 'target_mean')
    for condition in CONDITIONS:
        result[condition] = {}
        for arm in ARMS:
            entries = rows[condition][arm]
            scenes = {str(s): summarize([row for row in entries if row['scene'] == s]) for s in protocol['validation_scenes']}
            macro = {}
            for name in names:
                eligible = {s: row['metrics'][name] for s, row in scenes.items() if row['metrics'][name] is not None}
                macro[name] = {'value': math.fsum(eligible.values())/len(eligible) if eligible else None,
                    'eligible_scenes': [int(s) for s in eligible], 'excluded_scenes': [int(s) for s in scenes if s not in eligible],
                    'reason': None if eligible else 'no_defined_scene_values'}
            result[condition][arm] = {'scenes': scenes, 'scene_macro': macro, 'pooled_frame_counts': summarize(entries), 'frames': entries}
        result[condition]['paired_scene_macro_quality_minus_fixed'] = {name: metric_delta(result[condition]['fixed']['scene_macro'][name],
            result[condition]['quality']['scene_macro'][name]) for name in names}
    return {'conditions': result, 'instance_matching': 'maximum_cardinality_bipartite_IoU_at_least_0.50',
        'segmentation_resolution': [720, 1280], 'scene_f1_and_iou': 'pooled_counts_within_scene_then_equal_scene_mean',
        'empty_count_metric': 'null_when_F1_denominator_or_foreground_union_is_zero',
        'quality_reduction': 'float64_integer_area_target_means_then_pixel_mean_then_equal_four_scales_then_equal_frames',
        'quality_scope': 'availability_and_fidelity_to_preprocessed_depth_reference_not_calibrated_probability',
        'quality_mse_production_comparison_performed': False, 'quality_mse_production_values_present': False,
        'quality_mean_production_comparison_absolute_tolerance': TOLERANCE,
        'optional_torch_diagnostic_absolute_tolerance': QUALITY_DIAGNOSTIC_TOLERANCE}


def verify_environment(root, protocol):
    from scripts import setup_coco_eval_environment_v17 as setup
    path = storage.safe(root, protocol['environment_protocol'])
    result = setup.run(root, path, check_only=True)
    require(result['status'] == 'verified_completed_read_only'
        and result['report_sha256'] == protocol['environment_report_sha256'], 'Frozen COCO environment verification failed')
    report = storage.read(storage.safe(root, protocol['environment_report']))
    require(all(report['probe']['runtime'][key] == value for key, value in protocol['expected_evaluation_runtime'].items()),
            'Frozen official evaluation runtime differs')
    return report


def metrics_binding(root, protocol, config, seals):
    directory = storage.safe(root, protocol['outputs']['scores_root'])
    files = {name: row['sha256'] for name, row in storage.snapshot(directory, root).items()}
    body = {'schema': 18, 'phase': 'metrics', 'protocol_sha256': storage.digest(config),
        'source_hashes': protocol['source_hashes'], 'input_hashes': protocol['input_hashes'], 'score_files': files,
        'score_completion_sha256': {c: {a: seals[c][a]['completion_sha256'] for a in ARMS} for c in CONDITIONS},
        'environment_report_sha256': protocol['environment_report_sha256'], 'runtime': protocol['expected_evaluation_runtime'],
        'interpreter_sha256': storage.digest(storage.safe(root, protocol['evaluation_interpreter'])),
        'parameters': expected_parameters(), 'new_training_updates': 0, 'new_model_forward_calls': 0}
    return dict(body, fingerprint=storage.fingerprint(body))


def inspect_metrics(root, protocol, config, outputs, seals, context, guard):
    binding = metrics_binding(root, protocol, config, seals)
    directory = storage.safe(root, protocol['outputs']['metrics_root'])
    require(storage.read(directory/'binding.json') == binding, 'Metric binding differs from frozen sources')
    expected = {directory/'binding.json', directory/'completion.json', durable.receipt(directory/'completion.json')}
    references = reference_identity(root, protocol, context, guard)
    reports, audits = {}, []
    for condition in CONDITIONS:
        for arm in ARMS:
            for scope in ['scene_%04d'%s for s in protocol['validation_scenes']]+['pooled']:
                block = directory/condition/arm/scope
                for name in ('inputs.json', 'official.json', 'report.json', 'completion.json'):
                    expected.update((block/name, durable.receipt(block/name)))
                report, audit = inspect_block(root, protocol, condition, arm, scope, block, binding,
                    outputs[condition][arm]['frames'], references, context, guard)
                reports[(condition, arm, scope)] = report
                audits.append(audit)
    require({p for p in directory.rglob('*') if p.is_file()} == expected, 'Wrong fixed 88-block metric inventory')
    completion = durable.load_commit(directory/'completion.json', binding)
    report_file = storage.safe(root, protocol['outputs']['metrics_report'])
    excluded = {directory/'completion.json', durable.receipt(directory/'completion.json')}
    require(completion == {'schema': 18, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
        'expected_blocks': 88, 'outputs': {p.relative_to(directory).as_posix(): storage.digest(p) for p in expected-excluded},
        'report_sha256': storage.digest(report_file)}, 'Metric aggregate completion differs')
    aggregate = independent_aggregate(protocol, reports, binding)
    error = compare_metrics(durable.load_commit(report_file, binding), aggregate, 'Independent condition/scene/paired aggregate differs')
    return aggregate, audits, error


def inspect_block(root,protocol,condition,arm,scope,directory,binding,score_frames,references,context,guard):
    store=common()
    expected_files={directory/name for name in ('inputs.json','official.json','report.json','completion.json')}
    expected_files.update(store.receipt_path(file) for file in list(expected_files))
    require({p for p in directory.rglob('*') if p.is_file()}==expected_files,'Metric block inventory differs')
    for file in expected_files:
        require(not file.is_symlink(),'Link in metric block')
    completion=store.load_commit(directory/'completion.json',binding)
    payload_files=expected_files-{directory/'completion.json',store.receipt_path(directory/'completion.json')}
    require(completion=={'schema':18,'status':'completed','condition':condition,'arm':arm,'scope':scope,'binding_fingerprint':binding['fingerprint'],
        'outputs':{file.name:store.digest(file) for file in payload_files}},'Metric block completion differs')
    inputs,official,reported=(store.load_commit(directory/name,binding) for name in ('inputs.json','official.json','report.json'))
    selected=[image_id for image_id in references if scope=='pooled' or image_id//1000==int(scope[6:])]
    require(len(selected)==(160 if scope=='pooled' else 16),'Wrong fixed AP block support')
    require(set(inputs)=={'schema','condition','arm','scope','binding_fingerprint','gt','detections','source_map'}
        and inputs['schema']==18 and inputs['condition']==condition and inputs['arm']==arm and inputs['scope']==scope
        and inputs['binding_fingerprint']==binding['fingerprint'],'Metric input identity differs')
    truth=inputs['gt']
    require(set(truth)=={'info','images','categories','annotations'} and truth['info']=={}
        and truth['categories']==[{'id':1,'name':'object'}]
        and truth['images']==[references[index]['image'] for index in selected],'Image cohort or category interpretation differs')
    expected_gt=[annotation for image_id in selected for annotation in references[image_id]['annotations']]
    require(len(truth['annotations'])==len(expected_gt)
        and [ann['id'] for ann in truth['annotations']]==[ann['id'] for ann in expected_gt],'Original visible instances were lost or reordered')
    require(set(inputs['source_map'])=={'gt_annotation_id','dt_list_index'}
        and inputs['source_map']['gt_annotation_id']=={str(ann['id']):ann['source'] for ann in expected_gt},'GT annotation provenance differs')
    gt_by_image={image_id:[] for image_id in selected}
    for actual,expected in zip(truth['annotations'],expected_gt):
        require(set(actual)=={'id','image_id','category_id','segmentation','area','iscrowd','bbox'}
            and {key:value for key,value in actual.items() if key!='segmentation'}=={key:value for key,value in expected.items() if key!='source'},
            'GT area, bbox, ID, or crowd policy differs')
        gt_by_image[expected['image_id']].append(actual)
    detections=inputs['detections']
    expected_dt=[(image_id,item) for image_id in selected for item in score_frames[image_id]['row']['instances']]
    require(len(detections)==len(expected_dt) and set(inputs['source_map']['dt_list_index'])=={str(i) for i in range(len(detections))},
        'Detection-score support changed')
    dt_by_image={image_id:[] for image_id in selected}
    for index,(actual,(image_id,mapping)) in enumerate(zip(detections,expected_dt)):
        source=score_frames[image_id]; record=source['row']; item=source['source']
        require(set(actual)=={'image_id','category_id','segmentation','score'} and actual['image_id']==image_id
            and actual['category_id']==1 and type(actual['score']) is float and actual['score']==mapping['score'],
            'Detection score/order differs or a bbox entered official loadRes')
        expected_source={'scene':item['scene'],'frame':item['frame'],'image_id':image_id,'arm':arm,'condition':condition,**mapping,
            'score_record_sha256':store.digest(source['prediction_file'].parent/'record.json'),
            'prediction_sha256':record['prediction_sha256'],'scope_detection_id':index+1}
        require(inputs['source_map']['dt_list_index'][str(index)]==expected_source,'Detection selected-center provenance differs')
        dt_by_image[image_id].append((index+1,actual,mapping['instance_id']))
    results=[]; checked_masks=0
    for image_id in selected:
        reference=references[image_id]
        labels,_=source_labels(root,protocol,context,reference['source'])
        # One dense predicted map and one reference map are resident for this image only.
        with np.load(score_frames[image_id]['prediction_file'],allow_pickle=False) as cache:
            high=original_resolution_instances(cache['instances'])
        gt_ids=[]; labels_ids=[]
        for annotation,original in zip(gt_by_image[image_id],reference['annotations']):
            binary=decode_rle(annotation['segmentation'])
            require(binary.shape==(720,1280) and np.array_equal(binary,labels==original['source']['original_label_id'])
                and int(np.count_nonzero(binary))==annotation['area'],'GT compressed RLE differs from original visible mask')
            gt_ids.append(annotation['id']); labels_ids.append(original['source']['original_label_id']); checked_masks+=1
        dt_ids=[]; instance_ids=[]; confidence=[]
        for detection_id,annotation,instance_id in dt_by_image[image_id]:
            binary=decode_rle(annotation['segmentation'])
            require(binary.shape==(720,1280) and np.array_equal(binary,high==instance_id),'Detection RLE differs from nearest-resized immutable prediction')
            dt_ids.append(detection_id); instance_ids.append(instance_id); confidence.append(annotation['score']); checked_masks+=1
        overlap=dense_overlap(high,labels,instance_ids,labels_ids)
        results.append(independent_image_matching(image_id,gt_ids,dt_ids,confidence,overlap['iou']))
        if len(results)%16==0:
            guard.check()
    require(official['schema']==18 and official['parameters']==dict(expected_parameters(),imgIds=selected)
        and official['counts']==[10,101,1,4,3] and len(official['evalImgs'])==4*len(selected),'Official parameter/array support changed')
    require(official['evalImgs'][:len(selected)]==results,'Independent noncrowd all-area greedy matching differs')
    precision=np.asarray(official['precision'],np.float64)
    recall=np.asarray(official['recall'],np.float64)
    scores=np.asarray(official['scores'],np.float64)
    require(precision.shape==scores.shape==(10,101,1,4,3) and recall.shape==(10,1,4,3)
        and all(np.isfinite(value).all() for value in (precision,recall,scores)),'Official accumulation cache has invalid shape or values')
    expected=independent_ap(results)
    close_arrays(precision[:,:,0,0,2],expected['precision'],'Independent all-area primary precision differs')
    close_arrays(recall[:,0,0,2],expected['recall'],'Independent all-area maxDet100 recall differs')
    close_arrays(scores[:,:,0,0,2],expected['scores'],'Independent all-area sampled confidence differs')
    metric={key:{'value':expected['summary'][key],'reason':None if expected['ground_truth_count'] else 'no_valid_ground_truth'}
        for key in ('AP','AP50','AP75')}
    require(len(official['stats'])==12,'Official summary size differs')
    close_arrays(official['stats'][:3],[metric[key]['value'] if metric[key]['value'] is not None else -1 for key in ('AP','AP50','AP75')],
        'Independent primary AP summary differs')
    report={'schema':18,'status':'completed','condition':condition,'arm':arm,'scope':scope,'binding_fingerprint':binding['fingerprint'],
        'image_count':len(selected),'ground_truth_count':len(expected_gt),'detection_count':len(expected_dt),
        'metrics':metric,'metric_scale':'fraction_0_1','all_images_preserved':True,'official_evaluator_unmodified':True}
    difference=compare_metrics(reported,report,'Block primary metrics differ')
    return report,{'condition':condition,'arm':arm,'scope':scope,'image_count':len(selected),'rle_masks_checked':checked_masks,
        'maximum_primary_precision_difference':float(np.max(np.abs(precision[:,:,0,0,2]-expected['precision']))),
        'maximum_summary_difference':difference,'greedy_matching_exact':True,'all_area_maxdet100_audited':True,
        'auxiliary_area_and_lower_maxdets_independently_recomputed':False,'report_sha256':store.digest(directory/'report.json')}



def resource_evidence(record,protocol):
    value=record['resources']; limits=protocol['resources']
    for name,maximum in (('elapsed_seconds',limits['invocation_soft_seconds']),('peak_rss_bytes',limits['rss_soft_bytes']),
            ('peak_gpu_allocated_bytes',limits['gpu_allocated_soft_bytes']),('peak_gpu_reserved_bytes',limits['gpu_reserved_soft_bytes']),
            ('current_output_bytes',limits['output_limit_bytes'])):
        require(type(value[name]) in (int,float) and math.isfinite(value[name]) and 0<=value[name]<=maximum,
            'Recorded child exceeded a resource budget')


def work_record(record,phase,key,operation,protocol):
    unit='frames' if phase=='scores' else 'blocks'; total=160 if phase=='scores' else 88
    require(record['schema']==18 and record['expected_'+unit]==total,'Wrong recovery work identity')
    if phase=='scores':
        require(record['condition']+'/'+record['arm']==key,'Wrong recovery condition/arm')
    for name in ('new_'+unit,'reused_'+unit,'committed_'+unit,'new_model_forward_calls','new_training_updates'):
        require(type(record[name]) is int and record[name]>=0,'Invalid child work accounting')
    require(record['new_'+unit]+record['reused_'+unit]==record['committed_'+unit]
        and record['new_training_updates']==0,'Inconsistent child work accounting')
    if operation=='probe':
        require(record['status']=='partial' and record['committed_'+unit]==2,'Recovery probe was not two units')
    else:
        require(record['status'] in ('completed','verified_completed_read_only') and record['committed_'+unit]==total,
            'Recovery did not cover the complete cohort')
    readonly=operation=='read_only' or record['status']=='verified_completed_read_only'
    if readonly:
        require(record['status']=='verified_completed_read_only' and record['new_'+unit]==record['new_model_forward_calls']==0,
            'Read-only recovery performed new work')
    if phase=='scores':
        require(record['new_model_forward_calls']<=record['new_frames'],'Extra forward calls within a completed invocation')
    else:
        require(record['new_model_forward_calls']==0 and type(record['new_official_evaluations']) is int
            and 0<=record['new_official_evaluations']<=record['new_blocks'],'Unexpected work in official metric recovery')
        if readonly:
            require(record['new_official_evaluations']==0,'Read-only recovery reran official evaluation')
    resource_evidence(record,protocol)


def inspect_recovery(root,protocol,config):
    store=common(); result={}
    for phase in ('scores','metrics'):
        path=store.safe(root,protocol['outputs']['score_resume_report' if phase=='scores' else 'metrics_resume_report'])
        proof=store.read(path)
        child='scripts/extract_dualstream_scores_v18.py' if phase=='scores' else 'scripts/evaluate_dualstream_mask_ap_v18.py'
        interpreter=protocol['interpreter'] if phase=='scores' else protocol['evaluation_interpreter']
        expected_binding={'protocol_sha256':store.digest(config),'wrapper_sha256':store.digest(root/'scripts/run_dualstream_evaluation_v18.py'),
            'child_sha256':store.digest(root/child),'interpreter_sha256':store.digest(store.safe(root,interpreter)),
            'pipeline_sha256':store.digest(root/'scripts/pipeline.py')}
        keys=[c+'/'+a for c in CONDITIONS for a in ARMS] if phase=='scores' else ['all']
        require(proof['schema']==18 and proof['phase']==phase and proof['status']=='passed' and proof['binding']==expected_binding
            and set(proof['entries'])==set(keys)
            and proof['scope']=='Actual process boundaries with two units first; no physical power-cut claim','Recovery proof identity differs')
        entries={}
        for key in keys:
            directory=store.safe(root,protocol['outputs']['scores_root'])/key if phase=='scores' else store.safe(root,protocol['outputs']['metrics_root'])
            report=directory/'report.json' if phase=='scores' else store.safe(root,protocol['outputs']['metrics_report'])
            entry=proof['entries'][key]; current=store.snapshot(directory,root)
            if phase=='metrics':
                current.update({file.relative_to(root).as_posix():file_state(file) for file in (report,store.receipt_path(report))})
            require(entry['status']=='passed' and entry['byte_and_mtime_identical'] is True and entry['read_only_check_passed'] is True
                and entry['completed_snapshot']==current and entry['completion_sha256']==store.digest(directory/'completion.json')
                and entry['report_sha256']==store.digest(report),'Completed recovery cache SHA, size or mtime changed')
            prefix=entry['prefix_before']; require(prefix==entry['prefix_after'],'Prefix changed across restart')
            relative=directory.relative_to(root).as_posix()
            if phase=='scores':
                first=[relative+'/frames/scene_%04d_%04d/'%(scene,frame)
                    for scene in protocol['validation_scenes'] for frame in protocol['frames']][:2]
            else:
                first=[relative+'/'+condition+'/'+arm+'/'+scope+'/' for condition in CONDITIONS for arm in ARMS
                    for scope in ['scene_%04d'%scene for scene in protocol['validation_scenes']]+['pooled']][:2]
            binding_files={relative+'/binding.json'}
            if phase=='scores': binding_files.add(relative+'/binding.json.sha256.json')
            expected_prefix={name:row for name,row in current.items()
                if name in binding_files or any(name.startswith(start) for start in first)}
            require(prefix==expected_prefix and bool(prefix),'Probe prefix omits a sealed artifact or exceeds two units')
            for operation in ('probe','completion','read_only'):
                work_record(entry[operation+'_record'],phase,key,operation,protocol)
            if phase=='metrics':
                require(entry['completion_record']['report_sha256']==entry['read_only_record']['report_sha256']==store.digest(report),
                    'Official summary receipt differs across recovery')
            entries[key]={'prefix_files_checked':len(prefix),'completed_files_checked':len(current),
                'byte_and_mtime_identical':True,'new_units_at_probe':entry['probe_record']['new_frames' if phase=='scores' else 'new_blocks'],
                'new_units_at_completion':entry['completion_record']['new_frames' if phase=='scores' else 'new_blocks']}
        result[phase]={'proof_sha256':store.digest(path),'entries':entries,'physical_power_loss_tested':False}
    return result


def validate_protocol(protocol):
    extractor.verify_design(protocol)
    require(protocol['validation_scenes'] == [2, 6, 7, 11, 13, 15, 18, 22, 33, 39]
        and protocol['frames'] == list(range(0, 256, 16)) and protocol['original_size'] == [720, 1280], 'Development support changed')
    require(protocol['mask_policy'] == {'category_id': 1, 'dt_bbox_in_loadRes': False,
        'gt': 'all_original_visible_label_ids_greater_than_zero_one_instance_per_id',
        'gt_area': 'original_visible_mask_pixel_count', 'gt_minimum_area_filter': False,
        'index_formula': 'floor((2*out_index+1)*source_size/(2*target_size))', 'iscrowd': 0,
        'resize': 'exact_pixel_center_nearest_integer_indices', 'useCats': 1}, 'Fixed integer mask geometry changed')
    policy, expected = protocol['coco_policy'], expected_parameters()
    require(policy['iou_type'] == 'segm' and policy['iou_thresholds'] == expected['iouThrs']
        and policy['recall_threshold_count'] == 101 and policy['max_dets'] == expected['maxDets']
        and policy['area_ranges'] == expected['areaRng'] and policy['area_names'] == expected['areaRngLbl']
        and policy['metrics'] == ['AP', 'AP50', 'AP75'] and policy['summaries'] == ['equal_scene_macro', 'pooled']
        and policy['pair_direction'] == 'quality_minus_fixed' and policy['no_valid_ground_truth'] == 'null_with_reason'
        and policy['empty_detections'] == 'retain_all_images_with_empty_annotation_list', 'AP policy changed')
    require(protocol['independent_audit'] == {'scope': 'all_area_max_dets_100_matching_precision_AP_AP50_AP75',
        'absolute_tolerance': TOLERANCE, 'relative_tolerance': 0,
        'auxiliary_area_and_smaller_max_dets': 'official_outputs_only_not_independently_reimplemented'}, 'Audit tolerance or scope changed')
    require(protocol['acceptance_interpreter'] == 'runtime/envs/grasp_eval/Scripts/python.exe', 'Wrong independent interpreter policy')


def watched_snapshot(root, protocol, config, context):
    files = {config}
    for field in ('parent_training_protocol', 'parent_training_report', 'environment_protocol', 'environment_report'):
        files.add(storage.safe(root, protocol[field]))
    for key in ('score_resume_report', 'metrics_resume_report', 'metrics_report'):
        file = storage.safe(root, protocol['outputs'][key]); files.add(file)
        if key == 'metrics_report': files.add(durable.receipt(file))
    for p in (protocol, context['protocol']):
        for group in ('source_hashes', 'input_hashes'):
            files.update(storage.safe(root, name) for name in p[group])
    for source in context['sources'].values():
        files.update(storage.safe(root, name) for name in source['files'])
    for key in ('scores_root', 'metrics_root'):
        files.update(file for file in storage.safe(root, protocol['outputs'][key]).rglob('*') if file.is_file())
    result = {file.relative_to(root).as_posix(): file_state(file) for file in sorted(files)}
    result['external_base_python'] = file_state(Path(protocol['base_python']['path']))
    return result


def figure_bytes(summary):
    canvas = Image.new('RGB', (1520, 1200), 'white'); draw = ImageDraw.Draw(canvas)
    title, font, small = ImageFont.load_default(size=27), ImageFont.load_default(size=18), ImageFont.load_default(size=15)
    colors = {'fixed': '#2775a5', 'quality': '#bd6236'}
    draw.text((35, 20), 'Dual-stream visible-mask AP: four fixed development conditions', font=title, fill='#152b3b')
    draw.text((35, 61), '10 previously used scenes | 160 frames per arm and condition | final step 800 | one seed', font=font, fill='#44525c')
    for panel, condition in enumerate(CONDITIONS):
        x, y = 60+(panel%2)*760, 135+(panel//2)*320
        draw.text((x, y-24), condition.upper()+' / equal-scene mean', font=font, fill='#152b3b')
        baseline = y+190
        for tick in range(6):
            yy = baseline-180*tick/5
            draw.line((x+25, yy, x+650, yy), fill='#dce3e7')
            draw.text((x-14, yy-8), '%.1f'%(tick/5), font=small, fill='#53616b')
        values = summary['conditions'][condition]
        for n, metric in enumerate(('AP', 'AP50', 'AP75')):
            cx = x+100+n*205
            for offset, arm in ((0, 'fixed'), (60, 'quality')):
                value = values['arms'][arm]['scene_macro'][metric]['value']
                if value is None:
                    draw.text((cx+offset, baseline-25), 'NA', font=small, fill=colors[arm])
                else:
                    draw.rectangle((cx+offset, baseline-180*value, cx+offset+44, baseline), fill=colors[arm])
                    draw.text((cx+offset-5, baseline-180*value-22), '%.3f'%value, font=small, fill=colors[arm])
            draw.text((cx+20, baseline+10), metric, font=font, fill='#152b3b')
            delta = values['paired_quality_minus_fixed']['scene_macro'][metric]['value']
            draw.text((cx-5, baseline+38), 'Delta '+('NA' if delta is None else '%+.4f'%delta), font=small,
                      fill='#a63b30' if delta is not None and delta < 0 else '#374b58')
        draw.text((x+35, baseline+67), 'Fixed', font=small, fill=colors['fixed'])
        draw.text((x+120, baseline+67), 'Quality', font=small, fill=colors['quality'])
    draw.text((45, 790), 'Every scene AP difference (quality minus fixed); negative results retained', font=font, fill='#152b3b')
    for n, c in enumerate(CONDITIONS): draw.text((270+n*280, 827), c.upper(), font=font, fill='#152b3b')
    for i, scene in enumerate(summary['conditions']['clean']['arms']['fixed']['scenes']):
        y = 861+i*25
        draw.text((60, y), 'Scene %04d'%int(scene), font=small, fill='#152b3b')
        for n, c in enumerate(CONDITIONS):
            value = summary['conditions'][c]['paired_quality_minus_fixed']['by_scene'][scene]['AP']['value']
            draw.text((270+n*280, y), 'NA' if value is None else '%+.6f'%value, font=small,
                      fill='#a63b30' if value is not None and value < 0 else '#374b58')
    draw.text((45, 1133), 'All-area maxDet100; exact integer pixel-center mask resize 480x640 to 720x1280.', font=small, fill='#44525c')
    draw.text((45, 1161), 'Development evidence only. No held-out/generalization or full-method claim. Quality is a depth-fidelity proxy.', font=small, fill='#44525c')
    stream = io.BytesIO(); canvas.save(stream, format='PNG'); return stream.getvalue()


def prepare_own_outputs(root, report, visuals, readonly):
    figure = visuals/FIGURE
    for destination in (report, figure):
        require(destination.is_relative_to(root) and not destination.is_symlink(), 'Acceptance destination escaped project')
        for file in destination.parent.glob(destination.name+'.*.tmp'):
            require(re.fullmatch(re.escape(destination.name)+r'(?:\.sha256\.json)?\.[0-9a-f]{32}\.tmp', file.name)
                and not file.is_symlink() and not readonly and not report.exists(), 'Unexpected acceptance temporary')
            file.unlink()
    require(not visuals.exists() or set(visuals.iterdir()) <= {figure, durable.receipt(figure)}, 'Unexpected figure artifact')
    return figure


def execute(root, config, protocol, check_only=False):
    validate_protocol(protocol)
    require(Path(sys.executable).resolve() == storage.safe(root, protocol['acceptance_interpreter']), 'Wrong independent acceptance interpreter')
    import scipy
    import PIL
    runtime = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__, 'Pillow': PIL.__version__}
    require(all(protocol['runtime'][k] == v for k, v in runtime.items()), 'Independent numeric runtime differs')
    storage.verify_source_maps(root, protocol)
    for name in ('scripts/accept_dualstream_evaluation_v18.py', 'tests/test_dualstream_acceptance_v18.py',
                 'scripts/accept_mask_ap_v17.py', 'scripts/setup_coco_eval_environment_v17.py'):
        require(protocol['source_hashes'].get(name) == storage.digest(storage.safe(root, name)), 'Independent audit source not frozen')
    guard = storage.Guard(root, protocol, 'acceptance'); guard.check()
    verify_environment(root, protocol)
    sealed = extractor.verify_scores_seals(root, config, protocol)
    context = sealed['context']
    before = watched_snapshot(root, protocol, config, context)
    outputs, auxiliary = inspect_scores(root, protocol, context, sealed, guard)
    aggregate, diagnostics, error = inspect_metrics(root, protocol, config, outputs, sealed['seals'], context, guard)
    recovery = inspect_recovery(root, protocol, config)
    require(watched_snapshot(root, protocol, config, context) == before, 'Sealed source or result changed during independent audit')
    storage.verify_source_maps(root, protocol); guard.check()
    body = {'schema': 18, 'phase': 'acceptance', 'protocol_sha256': storage.digest(config),
        'source_hashes': protocol['source_hashes'], 'input_hashes': protocol['input_hashes'],
        'parent_training_report_sha256': protocol['parent_training_report_sha256'],
        'environment_report_sha256': protocol['environment_report_sha256'], 'score_seals': sealed['seals'],
        'metrics_completion_sha256': storage.digest(storage.safe(root, protocol['outputs']['metrics_root'])/'completion.json'),
        'metrics_report_sha256': storage.digest(storage.safe(root, protocol['outputs']['metrics_report'])),
        'recovery_proof_sha256': {phase: row['proof_sha256'] for phase, row in recovery.items()},
        'interpreter_sha256': storage.digest(sys.executable), 'runtime': runtime}
    binding = dict(body, fingerprint=storage.fingerprint(body))
    png = figure_bytes(aggregate)
    report = {'schema': 18, 'status': 'accepted_complete_dualstream_development_comparison', 'binding': binding,
        'decision': 'engineering_and_independent_metric_consistency_without_winning_gate',
        'conditions': aggregate['conditions'], 'robustness_scene_macro_AP': aggregate['robustness_scene_macro_AP'],
        'clean_delta_scene_macro_AP': aggregate['clean_delta_scene_macro_AP'], 'auxiliary': auxiliary_aggregate(protocol, auxiliary),
        'fixed_support': {'scenes': protocol['validation_scenes'], 'frames_per_condition_arm': 160,
            'conditions': list(CONDITIONS), 'arms': list(ARMS), 'model_frames': 1280, 'metric_blocks': 88},
        'score_forward_attempt_accounting': {c: {a: sealed['reports'][c][a]['forward_attempt_accounting'] for a in ARMS} for c in CONDITIONS},
        'independent_checks': {'decoded_frames': 1280, 'input_and_quality_target_sha_reconstructed': True,
            'original_label_masks_and_exact_integer_resize_checked': True, 'blocks': diagnostics,
            'maximum_aggregate_difference': error, 'absolute_tolerance': TOLERANCE, 'relative_tolerance': 0,
            'source_file_count_checked': len(before), 'source_files_unchanged': True,
            'neural_forward_independently_repeated': False, 'auxiliary_area_and_lower_maxdets_independently_recomputed': False},
        'recovery': recovery, 'figure': {'file': protocol['outputs']['visuals_dir']+'/'+FIGURE, 'sha256': hashlib.sha256(png).hexdigest()},
        'new_model_forward_calls': 0, 'new_official_evaluations': 0, 'new_training_updates': 0,
        'held_out_data_used': False, 'thresholds_or_checkpoints_selected': False, 'is_full_p4_method_completion': False,
        'limitations': protocol['limitations']}
    output = storage.safe(root, protocol['outputs']['acceptance_report'])
    readonly = check_only or output.exists()
    figure = prepare_own_outputs(root, output, storage.safe(root, protocol['outputs']['visuals_dir']), readonly)
    files = (output, durable.receipt(output), figure, durable.receipt(figure))
    old = {str(p): file_state(p) for p in files} if readonly else None
    guard.check(0 if readonly else len(png)+len(storage.json_bytes(report))+8192)
    durable.commit_bytes(figure, png, binding, readonly)
    durable.commit(output, report, binding, readonly)
    if readonly: require(old == {str(p): file_state(p) for p in files}, 'Read-only audit changed own artifacts')
    return {'schema': 18, 'status': 'verified_completed_read_only' if readonly else report['status'],
        'report_sha256': storage.digest(output), 'figure_sha256': storage.digest(figure),
        'new_model_forward_calls': 0, 'new_official_evaluations': 0, 'new_training_updates': 0, 'resources': guard.check()}


def run(root=ROOT, config=None, check_only=False):
    root = Path(root).resolve(); config = storage.safe(root, config or CONFIG)
    # Completed read-only reentry does not touch a lock file.
    protocol = storage.read(config)
    if check_only or storage.safe(root, protocol['outputs']['acceptance_report']).exists():
        return execute(root, config, protocol, check_only)
    with storage.lock(root/'runtime/locks/dualstream_evaluation_v18_acceptance'):
        return execute(root, config, protocol, check_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(config=args.config, check_only=args.check_only), ensure_ascii=False, allow_nan=False), flush=True)

