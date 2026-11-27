"""Frozen coverage comparison acceptance with durable per-frame inference receipts."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import PIL
from PIL import Image, ImageDraw
import scipy
import torch
import torchvision
from scripts import train_segmentation_coverage_v8 as trainer
from scripts.accept_segmentation_expanded_v3 import verify_partition
from scripts.render_segmentation_expanded_v3 import instance_colors
from scripts.train_segmentation_expanded_v3 import learning_rate_for_update, ensure_disk_reserve
from scripts.verify_segmentation_repair_v5_resume import check_initialization
from src.checkpoints import atomic_json, atomic_write, file_sha256, fingerprint, _exclusive_lock
from src.graspnet_frames import GraspNetFrames
from src.segmentation import decode_instances, instance_counts, summarize_counts
from src.segmentation_diagnostics_v3 import selected_centers, center_matches, distribution

ARMS = ('coverage_control', 'coverage_expanded')
PROTOCOL = ROOT / 'configs/segmentation_coverage_v8.json'
AGGREGATE = ROOT / 'results/P2/segmentation_coverage_v8.json'
PROOF = ROOT / 'results/P2/segmentation_coverage_v8_resume.json'
DIRECTORY = ROOT / 'results/P2/segmentation_coverage_v8_acceptance'
HEADS = ROOT / 'results/P2/segmentation_coverage_v8_heads.json'
OUTPUT = ROOT / 'results/第八轮训练覆盖对照验收.json'
COUNT_KEYS = ('tp', 'fp', 'fn')
STATE_KEYS = {'model', 'optimizer', 'scheduler', 'scaler', 'cursor', 'rng', 'extra'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_close(actual, expected):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            'Recorded arithmetic differs from independent recomputation')


def gate(partition, protocol):
    rule = protocol['acceptance']
    checks = {
        'scene_macro_foreground_iou': partition['scene_macro']['foreground_pixel_iou'] >= rule['scene_macro_foreground_iou_min'],
        'scene_macro_instance_f1': partition['scene_macro']['instance_f1_iou50'] >= rule['scene_macro_instance_f1_min'],
        'every_scene_has_true_positive': {row['scene'] for row in partition['scenes']} == set(protocol['validation_scenes']) and all(
            row['instances']['tp'] >= rule['minimum_tp_per_scene'] for row in partition['scenes'])}
    return {'passed': all(checks.values()), 'checks': checks, 'thresholds': rule}


def adoption(checked, protocol):
    first, second = (checked[arm]['development_validation'] for arm in ARMS)
    gates = {arm: gate(checked[arm]['development_validation'], protocol)['passed'] for arm in ARMS}
    f1_gain = second['scene_macro']['instance_f1_iou50'] - first['scene_macro']['instance_f1_iou50']
    fg_drop = first['scene_macro']['foreground_pixel_iou'] - second['scene_macro']['foreground_pixel_iou']
    prior = {row['scene']: row for row in first['scenes']}
    require(set(prior) == {row['scene'] for row in second['scenes']} == set(protocol['validation_scenes']), 'Adoption scene identities differ')
    deltas = {str(row['scene']): row['instances']['f1_iou50'] - prior[row['scene']]['instances']['f1_iou50'] for row in second['scenes']}
    rule = protocol['candidate_adoption']
    checks = {'candidate_passes_absolute_gate': gates[ARMS[1]],
              'development_macro_f1_gain': f1_gain >= rule['scene_macro_f1_gain_over_control_min'] or math.isclose(
                  f1_gain, rule['scene_macro_f1_gain_over_control_min'], rel_tol=0, abs_tol=1e-12),
              'development_macro_foreground_guard': fg_drop <= rule['scene_macro_foreground_iou_drop_max'] or math.isclose(
                  fg_drop, rule['scene_macro_foreground_iou_drop_max'], rel_tol=0, abs_tol=1e-12),
              'scene_consistency': sum(value > 0 for value in deltas.values()) >= rule['minimum_scenes_with_positive_f1_gain']}
    selected = ARMS[1] if all(checks.values()) else ARMS[0] if gates[ARMS[0]] else None
    return {'engineering_gate_by_arm': gates, 'candidate_adoption_checks': checks,
            'development_macro_f1_gain_candidate_minus_control': f1_gain,
            'development_macro_foreground_drop_control_minus_candidate': fg_drop,
            'scene_f1_gain_candidate_minus_control': deltas,
            'positive_scene_count': sum(value > 0 for value in deltas.values()), 'selected_arm': selected,
            'decision': 'provisional_engineering_candidate' if selected else 'neither_arm_accepted',
            'interpretation': 'Prespecified development-only cost rule; not statistical significance or independent confirmation'}


def independent_partition(partition, scenes, frames):
    """Recalculate pooled per-scene counts before an equal-weight scene mean."""
    verify_partition(partition, scenes, frames)
    rows = partition['frames']
    for row in rows:
        require(type(row['foreground_intersection']) is int and type(row['foreground_union']) is int, 'Pixel areas must be integers')
        require(row['predicted_instances'] == row['instances']['tp'] + row['instances']['fp'] and
                row['actual_instances'] == row['instances']['tp'] + row['instances']['fn'], 'Per-frame instance totals disagree')
        require(math.isfinite(row['max_center_probability']) and 0 <= row['max_center_probability'] <= 1, 'Invalid center probability')
    recomputed = []
    for scene in sorted(scenes):
        members = [row for row in rows if row['scene'] == scene]
        intersection = sum(row['foreground_intersection'] for row in members)
        union = sum(row['foreground_union'] for row in members)
        counts = summarize_counts({key: sum(row['instances'][key] for row in members) for key in COUNT_KEYS})
        value = {'scene': scene, 'frame_count': len(members), 'foreground_intersection': intersection,
                 'foreground_union': union, 'foreground_pixel_iou': intersection / union if union else 1., 'instances': counts}
        require(value == next(row for row in partition['scenes'] if row['scene'] == scene), 'Scene pooled counts or rates differ')
        recomputed.append(value)
    intersection = sum(row['foreground_intersection'] for row in rows)
    union = sum(row['foreground_union'] for row in rows)
    value = {'frame_count': len(rows), 'scene_count': len(recomputed), 'foreground_intersection': intersection,
             'foreground_union': union, 'foreground_pixel_iou': intersection / union if union else 1.,
             'instances': summarize_counts({key: sum(row['instances'][key] for row in rows) for key in COUNT_KEYS}),
             'scene_macro': {'foreground_pixel_iou': sum(row['foreground_pixel_iou'] for row in recomputed) / len(recomputed),
                             'instance_f1_iou50': sum(row['instances']['f1_iou50'] for row in recomputed) / len(recomputed)},
             'scenes': recomputed}
    require(all(value[key] == partition[key] for key in value), 'Partition pooled or macro arithmetic differs')
    return value


def checked_evaluation(result, protocol, arm):
    evaluation = result['evaluation']
    partitions = {'train': protocol['arms'][arm]['train_scenes'], 'development_validation': protocol['validation_scenes'],
                  'historical_validation': protocol['historical_validation_scenes'], 'round6_validation': protocol['round6_validation_scenes']}
    require(set(evaluation) == set(partitions) | {'acceptance_gate'}, 'Unexpected final evaluation partitions')
    checked = {name: independent_partition(evaluation[name], scenes, protocol['frames']) for name, scenes in partitions.items()}
    for name in ('historical_validation', 'round6_validation'):
        require(evaluation[name]['frames'] == [row for row in evaluation['development_validation']['frames'] if row['scene'] in partitions[name]],
                'Validation subgroup differs from development rows')
    computed_gate = gate(checked['development_validation'], protocol)
    require(all(evaluation['acceptance_gate'][key] == value for key, value in computed_gate.items()), 'Recorded acceptance gate differs')
    require(result['decision'] == ('engineering_gate_passed' if computed_gate['passed'] else 'engineering_gate_not_met'), 'Result decision differs')
    return checked


def check_history(history, manifest, protocol, arm):
    exposures = trainer.verify_history(history, manifest, protocol, arm, final=True)
    for row in history:
        step = row['step']
        expected_next = protocol['learning_rate'] * (learning_rate_for_update(step + 1, protocol) / protocol['learning_rate'])
        require(row['learning_rate_next_update'] == expected_next and row['next_update_index'] == (step + 1 if step < 1600 else None)
                and row['next_update_planned'] is (step < 1600), 'History scheduler next-update boundary differs')
        require(type(row['amp_retries']) is int and 0 <= row['amp_retries'] < 16
                and math.isfinite(row['amp_scale']) and row['amp_scale'] > 0, 'Invalid AMP update history')
        require(all(math.isfinite(value) for value in row['components'].values()), 'Nonfinite component history')
    return {'committed_updates': len(history), 'training_frames': len(exposures), 'total_sample_exposures': sum(exposures.values()),
            'exposures_per_frame': sorted(set(exposures.values())), 'within_arm_order_verified': True,
            'learning_rate_used_and_next_verified': True}


def check_starts(initializations, evidence, results, protocol):
    for arm in ARMS:
        parent = {'parent_checkpoint_sha256': protocol['parent_checkpoint_sha256'],
                  'parent_config_fingerprint': protocol['parent_config_fingerprint'],
                  'parent_data_fingerprint': protocol['parent_data_fingerprint'],
                  'parent_model_tensor_sha256': evidence[arm]['parent_evidence']['parent_model_tensor_sha256']}
        check_initialization(initializations[arm], parent, protocol, results[arm]['config_fingerprint'])
        require(initializations[arm]['schema'] == 8, 'Wrong actual initialization schema')
    comparable = [{key: value for key, value in initializations[arm].items() if key != 'run_config_fingerprint'} for arm in ARMS]
    require(comparable[0] == comparable[1], 'Actual model/optimizer/scheduler/scaler/RNG/reset state differs across arms')
    return True


def actual_runtime_identity():
    require(torch.cuda.is_available(), 'The strong final-model loader and runtime verification require the recorded GPU to remain visible')
    return {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'pillow': PIL.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}


def verify_inference_runtime(expected, configs=None):
    actual = actual_runtime_identity()
    require(actual == expected, 'Actual acceptance runtime/GPU differs from frozen GPU recovery proof')
    if configs is not None:
        require(all(configs[arm]['device_name'] == actual['gpu'] for arm in ARMS), 'Actual acceptance GPU differs from training device')
    return actual


def verify_recovery(proof, protocol):
    require(proof['status'] == 'passed' and proof['config_sha256'] == file_sha256(PROTOCOL)
            and proof['data_manifest_sha256'] == file_sha256(ROOT / protocol['data_manifest']), 'GPU proof does not bind current data/protocol')
    expected_cross = {'same_model_optimizer_scheduler_scaler_rng_cursor_at_step0', 'same_learning_rate_first82', 'each_arm_all_state_bitwise_recovery'}
    require(set(proof['cross_arm_checks']) == expected_cross and all(value is True for value in proof['cross_arm_checks'].values()), 'Incomplete cross-arm recovery proof')
    require(proof['excluded_from_research_results'] is True and proof['validation_evaluated'] is False and proof['real_power_cut_tested'] is False,
            'GPU proof scope differs')
    verify_inference_runtime(proof['runtime_identity'])
    for arm in ARMS:
        item, boundary = proof['arms'][arm], protocol['arms'][arm]['train_frames'] // protocol['accumulation']
        require(item['status'] == 'passed' and set(item['bitwise_equal']) == STATE_KEYS
                and all(value is True for value in item['bitwise_equal'].values()), 'Incomplete full-state GPU recovery proof')
        require(item['resumed_from_step'] == boundary and item['updates_on_resume'] == 2
                and item['final_cursor'] == {'epoch': 1, 'next_batch': 4, 'global_step': boundary + 2}
                and item['validation_evaluated'] is False, 'Recovery did not cover the arm-specific epoch boundary')
    for name, checksum in proof['source_sha256'].items():
        require(file_sha256(trainer.project_path(name)) == checksum, 'Recovery source changed: ' + name)


def acceptance_sources():
    names = set(trainer.source_hashes()) | {'scripts/accept_segmentation_coverage_v8.py',
        'scripts/run_segmentation_coverage_v8.py', 'scripts/run_coverage_acceptance_v8.py',
        'scripts/verify_segmentation_coverage_v8_resume.py',
        'scripts/verify_segmentation_repair_v5_resume.py', 'scripts/verify_gpu_resume.py',
        'scripts/render_segmentation_expanded_v3.py', 'src/segmentation_diagnostics_v3.py'}
    return {name: file_sha256(ROOT / name) for name in sorted(names)}


def make_binding(configs, manifests, evidence):
    versions = {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                'numpy': np.__version__, 'scipy': scipy.__version__, 'Pillow': PIL.__version__}
    require(all(configs[arm]['versions'] == versions for arm in ARMS), 'Acceptance and training library versions differ')
    actual_runtime = verify_inference_runtime(read(PROOF)['runtime_identity'], configs)
    return {'schema': 8, 'protocol_sha256': file_sha256(PROTOCOL), 'aggregate_sha256': file_sha256(AGGREGATE),
            'gpu_recovery_sha256': file_sha256(PROOF), 'sources': acceptance_sources(), 'versions': versions,
            'cuda_runtime': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
            'devices': {arm: actual_runtime['gpu'] for arm in ARMS}, 'actual_inference_runtime': actual_runtime,
            'run_evidence': evidence, 'arm_manifest_fingerprints': {arm: fingerprint(manifests[arm]) for arm in ARMS},
            'evaluation': 'FP32 final-checkpoint inference; fixed decoder; deterministic algorithms; TF32 disabled',
            'selection_endpoint': 'development scene macro F1 with the predeclared absolute/relative engineering rules'}


def cache_key(identity):
    arm, split, scene, frame = (identity[name] for name in ('arm', 'split', 'scene', 'frame'))
    require(arm in ARMS and split in ('train', 'development_validation') and type(scene) is int and 0 <= scene < 10000
            and isinstance(frame, str) and re.fullmatch(r'\d{4}', frame) is not None, 'Invalid cache frame identity')
    return '{}_{}_scene_{:04d}_{}'.format(arm, split, scene, frame)


def cache_paths(directory, identity):
    key = cache_key(identity)
    return directory / 'frames' / (key + '.json'), directory / 'frames' / (key + '.sha256.json')


def image_required(identity, protocol):
    return identity['split'] == 'development_validation' and identity['scene'] in protocol['visualization_scenes'] and identity['frame'] == protocol['visualization_frame']


def verify_frame(row, binding, identity, reference, need_image=False):
    require(row['schema'] == 8 and row['binding_fingerprint'] == binding and row['identity'] == identity, 'Cached frame binding or identity changed')
    require(row['metrics'] == reference, 'Cached per-frame metrics differ from frozen final result')
    gt, predicted = row['gt_instance_count'], row['selected_center_count']
    require(type(gt) is int and gt == reference['actual_instances'] and type(predicted) is int and 0 <= predicted <= 64, 'Invalid diagnostic instance/peak counts')
    require(len(row['offset_instance_mean_epe_pixels']) == gt and all(math.isfinite(value) and value >= 0 for value in row['offset_instance_mean_epe_pixels']), 'Invalid offset EPE diagnostic')
    for radius in ('5', '10'):
        match = row['center_matching'][radius]
        tp, fp, fn = (match[key] for key in COUNT_KEYS)
        require(all(type(value) is int and value >= 0 for value in (tp, fp, fn)) and tp + fp == predicted and tp + fn == gt,
                'Invalid center diagnostic counts')
        require(match['radius_pixels'] == int(radius) and len(match['pairs']) == tp, 'Invalid center matching pairs')
        require(len({pair['prediction_index'] for pair in match['pairs']}) == tp and len({pair['gt_index'] for pair in match['pairs']}) == tp,
                'Center matches are not one-to-one')
        require(all(0 <= pair['prediction_index'] < predicted and 0 <= pair['gt_index'] < gt and math.isfinite(pair['distance_pixels'])
                    and 0 <= pair['distance_pixels'] <= int(radius) for pair in match['pairs']), 'Center pair out of range')
        for name, expected in (('precision', tp / (tp + fp) if tp + fp else 0.), ('recall', tp / (tp + fn) if tp + fn else 0.),
                               ('f1', 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)):
            check_close(match[name], expected)
    require(row['center_matching']['5']['tp'] <= row['center_matching']['10']['tp'], 'Center cardinality decreased with larger radius')
    require(bool(row.get('prediction_mask')) == need_image, 'Fixed-visualization cache membership differs')


def load_cached(directory, binding, identity, reference, need_image=False):
    payload, receipt_path = cache_paths(directory, identity)
    if not receipt_path.exists():
        if payload.exists():
            orphan = read(payload)
            require(orphan.get('binding_fingerprint') == binding and orphan.get('identity') == identity,
                    'Uncommitted cache payload belongs to another request')
        return None
    receipt = read(receipt_path)
    require(receipt['schema'] == 8 and receipt['binding_fingerprint'] == binding and receipt['identity'] == identity
            and receipt['payload_file'] == payload.name, 'Cache receipt identity or binding mismatch')
    require(payload.is_file() and file_sha256(payload) == receipt['payload_sha256'], 'Committed frame payload missing or corrupt')
    row = read(payload)
    verify_frame(row, binding, identity, reference, need_image)
    if need_image:
        image = row['prediction_mask']
        expected = 'predictions/' + cache_key(identity) + '.png'
        require(image['file'] == expected, 'Prediction mask path differs from fixed identity')
        path = directory / expected
        require(path.is_file() and file_sha256(path) == image['sha256'], 'Committed prediction mask missing or corrupt')
        with Image.open(path) as value:
            require(value.mode == 'L' and value.size == (320, 240), 'Prediction mask format changed')
            labels = np.asarray(value)
            require(int(labels.max()) <= 64, 'Prediction mask labels out of decoder bounds')
    return row


def commit_frame(directory, row):
    payload, receipt = cache_paths(directory, row['identity'])
    require(not receipt.exists(), 'Never overwrite a committed frame receipt')
    atomic_json(payload, row)
    atomic_json(receipt, {'schema': 8, 'binding_fingerprint': row['binding_fingerprint'], 'identity': row['identity'],
                          'payload_file': payload.name, 'payload_sha256': file_sha256(payload)})


def infer_frame(model, dataset, index, identity, reference, protocol, binding, directory):
    inputs, targets = dataset.get(index)
    labels = targets['labels'].numpy()
    with torch.no_grad():
        outputs = model(inputs[None].to('cuda'))
        predicted = decode_instances(outputs, **protocol['decoder'])
        foreground = outputs['foreground'][0, 0].sigmoid().cpu().numpy() >= protocol['decoder']['foreground_threshold']
        probability = outputs['centers'][0, 0].float().sigmoid().cpu()
        max_center = float(outputs['centers'].sigmoid().max().cpu())
    counts = summarize_counts(instance_counts(predicted, labels))
    intersection, union = int((foreground & (labels > 0)).sum()), int((foreground | (labels > 0)).sum())
    metrics = {'scene': identity['scene'], 'frame': identity['frame'], 'foreground_intersection': intersection,
               'foreground_union': union, 'foreground_pixel_iou': intersection / union if union else 1., 'instances': counts,
               'predicted_instances': counts['tp'] + counts['fp'], 'actual_instances': counts['tp'] + counts['fn'],
               'max_center_probability': max_center}
    require(metrics == reference, 'Independent inference differs from final frozen frame metrics: ' + cache_key(identity))
    peaks = selected_centers(probability, protocol['decoder']['center_threshold'], protocol['decoder']['max_centers'])
    height, width = labels.shape
    difference = outputs['offsets'][0].detach().cpu().numpy() - targets['offsets'].numpy()
    difference[0] *= width
    difference[1] *= height
    epe = np.sqrt((difference ** 2).sum(axis=0))
    centers, errors = [], []
    for label in np.unique(labels[labels > 0]):
        yy, xx = np.nonzero(labels == label)
        centers.append([int(np.round(yy.astype(np.float32).mean())), int(np.round(xx.astype(np.float32).mean()))])
        errors.append(float(epe[yy, xx].mean()))
    row = {'schema': 8, 'binding_fingerprint': binding, 'identity': identity, 'metrics': metrics,
           'center_matching': {str(radius): center_matches(peaks, np.asarray(centers).reshape(-1, 2), radius) for radius in (5, 10)},
           'selected_center_count': len(peaks), 'gt_instance_count': len(centers), 'offset_instance_mean_epe_pixels': errors}
    if image_required(identity, protocol):
        require(predicted.shape == (240, 320) and int(predicted.max()) <= 64, 'Unexpected predicted label dimensions/range')
        relative = 'predictions/' + cache_key(identity) + '.png'
        buffer = io.BytesIO()
        Image.fromarray(predicted.astype(np.uint8)).save(buffer, format='PNG')
        atomic_write(directory / relative, lambda stream: stream.write(buffer.getvalue()))
        row['prediction_mask'] = {'file': relative, 'sha256': file_sha256(directory / relative)}
    verify_frame(row, binding, identity, reference, image_required(identity, protocol))
    return row


def summarize_heads(rows):
    matching = {}
    for radius in ('5', '10'):
        totals = {key: sum(row['center_matching'][radius][key] for row in rows) for key in COUNT_KEYS}
        tp, fp, fn = (totals[key] for key in COUNT_KEYS)
        matching[radius] = dict(totals, recall=tp / (tp + fn) if tp + fn else 0., precision=tp / (tp + fp) if tp + fp else 0.,
                               f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return {'frames': len(rows), 'center_matching': matching, 'selected_center_count': sum(row['selected_center_count'] for row in rows),
            'gt_instance_count': sum(row['gt_instance_count'] for row in rows),
            'offset_instance_mean_epe_pixels': distribution([value for row in rows for value in row['offset_instance_mean_epe_pixels']])}


def grouped_heads(rows, protocol):
    groups = {'train': [row for row in rows if row['identity']['split'] == 'train'],
              'development_validation': [row for row in rows if row['identity']['split'] == 'development_validation']}
    for name in ('historical_validation', 'round6_validation'):
        groups[name] = [row for row in groups['development_validation'] if row['identity']['scene'] in protocol[name + '_scenes']]
    return {'groups': {name: summarize_heads(members) for name, members in groups.items()},
            'scenes': {name: [{'scene': scene, **summarize_heads([row for row in members if row['identity']['scene'] == scene])}
                              for scene in sorted({row['identity']['scene'] for row in members})] for name, members in groups.items()},
            'frames': rows}


def publish_json_once(path, value, read_only=False):
    if path.exists():
        require(read(path) == value, 'Existing completed artifact differs: ' + str(path))
    else:
        require(not read_only, 'Completed artifact missing: ' + str(path))
        atomic_json(path, value)


def publish_bytes_once(path, value, read_only=False):
    if path.exists():
        require(path.read_bytes() == value, 'Existing visualization bytes differ: ' + str(path))
    else:
        require(not read_only, 'Completed visualization missing: ' + str(path))
        atomic_write(path, lambda stream: stream.write(value))


def render_fixed(manifest, rows, protocol, binding, read_only=False):
    require(protocol['visualization_scenes'] == protocol['validation_scenes'] and protocol['visualization_frame'] == '0000',
            'Visualization must use all ten frozen development scene0000 frames')
    records = [row for row in manifest['records']['validation'] if row['frame'] == '0000']
    require([row['scene'] for row in records] == protocol['visualization_scenes'], 'Fixed visualization identities differ')
    dataset = GraspNetFrames(ROOT / protocol['data_root'], records, tuple(protocol['input_size']), protocol['mode'], protocol['seed'], augment=False)
    lookup = {arm: {(row['identity']['scene'], row['identity']['frame']): row for row in rows[arm]
                    if row['identity']['split'] == 'development_validation'} for arm in ARMS}
    images = []
    for index, record in enumerate(records):
        _, targets = dataset.get(index)
        labels = targets['labels'].numpy()
        with Image.open(ROOT / protocol['data_root'] / record['files']['rgb']) as source:
            rgb = source.convert('RGB').resize((320, 240), Image.Resampling.BILINEAR)
        panels = [('RGB input', rgb), ('GT reference only', instance_colors(labels))]
        counts = {}
        for arm in ARMS:
            cached = lookup[arm][(record['scene'], record['frame'])]
            with Image.open(DIRECTORY / cached['prediction_mask']['file']) as source:
                prediction = np.asarray(source).copy()
            counts[arm] = summarize_counts(instance_counts(prediction, labels))
            require(counts[arm] == cached['metrics']['instances'], 'Stored visualization mask differs from verified metrics')
            panels.append((arm + ' | F1 {:.3f}'.format(counts[arm]['f1_iou50']), instance_colors(prediction)))
        canvas = Image.new('RGB', (1280, 340), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'DEVELOPMENT VALIDATION | scene {:04d} frame 0000 | fixed before results; independent label colors'.format(record['scene']), fill='black')
        draw.text((8, 26), 'Same frozen v5 parent +1600 new updates each | control:160 frames/20 epochs | expanded:320 frames/10 epochs', fill='black')
        draw.text((8, 44), 'GT is reference only; fixed decoder .50/.35/area20/max64. These examples do not replace all-scene evaluation.', fill='black')
        for column, (caption, panel) in enumerate(panels):
            draw.text((column * 320 + 8, 70), caption, fill='black')
            canvas.paste(panel, (column * 320, 94))
        path = DIRECTORY / 'visuals' / 'development_scene_{:04d}_0000.png'.format(record['scene'])
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG')
        publish_bytes_once(path, buffer.getvalue(), read_only)
        images.append({'scene': record['scene'], 'frame': '0000', 'file': str(path.relative_to(ROOT)),
                       'sha256': file_sha256(path), 'counts': counts})
    manifest_path = DIRECTORY / 'visuals/manifest.json'
    publish_json_once(manifest_path, {'schema': 8, 'binding_fingerprint': binding, 'frames': images,
                      'fixed_before_training': True, 'all_frame_metrics_match': True, 'extra_inference_for_rendering': False}, read_only)
    return images


def set_determinism():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_evidence():
    protocol, aggregate, proof = read(PROTOCOL), read(AGGREGATE), read(PROOF)
    require(aggregate['schema'] == 8 and aggregate['status'] == 'completed' and aggregate['protocol_sha256'] == file_sha256(PROTOCOL)
            and set(aggregate['arms']) == set(ARMS), 'Completed two-arm aggregate does not match this protocol')
    require(aggregate['new_research_updates'] == 3200 and aggregate['automatic_extension'] is False
            and aggregate['thresholds_changed'] is False, 'Research budget or fixed threshold policy changed')
    verify_recovery(proof, protocol)
    models, configs, manifests, results, evidence, checked, histories, initializations = ({ } for _ in range(8))
    for arm in ARMS:
        directory = trainer.project_path(protocol['arms'][arm]['run_directory'])
        model, config, manifest, result, bound = trainer.load_final_model(directory, protocol)
        item = aggregate['arms'][arm]
        require(trainer.project_path(item['run_directory']) == directory and item['result'] == result and config['arm'] == arm,
                'Aggregate, result and arm identities differ')
        require(config['device'] == 'cuda' and config['effective_amp'] is True, 'Research arm did not use the frozen GPU/AMP configuration')
        for key, name in (('result_sha256', 'result.json'), ('config_sha256', 'config.json'),
                          ('history_sha256', 'history.json'), ('initialization_sha256', 'initialization.json')):
            require(item[key] == file_sha256(directory / name), 'Aggregate file checksum differs: ' + name)
        histories[arm] = check_history(read(directory / 'history.json'), manifest, protocol, arm)
        initializations[arm] = read(directory / 'initialization.json')
        checked[arm] = checked_evaluation(result, protocol, arm)
        require(proof['arms'][arm]['data_fingerprint'] == bound['data_fingerprint'], 'Recovery probe used a different per-arm dataset')
        proof_initial = {key: value for key, value in proof['arms'][arm]['initialization'].items() if key != 'run_config_fingerprint'}
        actual_initial = {key: value for key, value in initializations[arm].items() if key != 'run_config_fingerprint'}
        require(proof_initial == actual_initial, 'Research initialization differs from recovery-tested state')
        models[arm], configs[arm], manifests[arm], results[arm], evidence[arm] = model, config, manifest, result, bound
    check_starts(initializations, evidence, results, protocol)
    require(manifests[ARMS[0]]['records']['validation'] == manifests[ARMS[1]]['records']['validation'], 'Arms do not share exactly the same development records')
    expected_sources = set(trainer.source_hashes()) | {'scripts/verify_segmentation_coverage_v8_resume.py',
        'scripts/verify_segmentation_repair_v5_resume.py', 'scripts/verify_gpu_resume.py', 'scripts/pipeline.py'}
    require(set(proof['source_sha256']) == expected_sources, 'Recovery proof source inventory incomplete')
    return protocol, models, configs, manifests, results, evidence, checked, histories, initializations


def collect_frames(models, manifests, results, protocol, binding, max_new_frames=None, completed=False, runtime_identity=None):
    rows, receipts, new_frames, reused_frames = {arm: [] for arm in ARMS}, [], 0, 0
    expected_total = sum(protocol['arms'][arm]['train_frames'] + len(protocol['validation_scenes']) * len(protocol['frames']) for arm in ARMS)
    require(expected_total == 800, 'Frozen model-frame budget changed')
    for arm in ARMS:
        on_device = False
        for split, name in (('train', 'train'), ('validation', 'development_validation')):
            records = manifests[arm]['records'][split]
            dataset = GraspNetFrames(ROOT / protocol['data_root'], records, tuple(protocol['input_size']), protocol['mode'], protocol['seed'], augment=False)
            references = {(row['scene'], row['frame']): row for row in results[arm]['evaluation'][name]['frames']}
            for index, record in enumerate(records):
                identity = {'arm': arm, 'split': name, 'scene': record['scene'], 'frame': record['frame']}
                reference = references[(record['scene'], record['frame'])]
                cached = load_cached(DIRECTORY, binding, identity, reference, image_required(identity, protocol))
                if cached is None:
                    require(not completed, 'Completed acceptance contains an uncommitted frame')
                    if max_new_frames is not None and new_frames >= max_new_frames:
                        return rows, receipts, {'status': 'step_limit_reached', 'binding_fingerprint': binding,
                            'new_frames': new_frames, 'reused_frames': reused_frames, 'committed_frames': len(receipts), 'expected_frames': expected_total}
                    ensure_disk_reserve(protocol['free_disk_reserve_gib'])
                    if not on_device:
                        require(runtime_identity is not None, 'Frozen actual inference runtime binding is missing')
                        verify_inference_runtime(runtime_identity)
                        models[arm].to('cuda').eval()
                        on_device = True
                    cached = infer_frame(models[arm], dataset, index, identity, reference, protocol, binding, DIRECTORY)
                    commit_frame(DIRECTORY, cached)
                    new_frames += 1
                    print(json.dumps({'committed': cache_key(identity), 'new_frames': new_frames, 'reused_frames': reused_frames}, ensure_ascii=False), flush=True)
                else:
                    reused_frames += 1
                rows[arm].append(cached)
                _, receipt_path = cache_paths(DIRECTORY, identity)
                receipts.append({'file': str(receipt_path.relative_to(DIRECTORY)), 'sha256': file_sha256(receipt_path)})
        if on_device:
            models[arm].to('cpu')
    require(len(receipts) == expected_total, 'Incomplete independent inference evidence')
    expected_payloads = {cache_paths(DIRECTORY, row['identity'])[0].name for values in rows.values() for row in values}
    expected_receipts = {Path(row['file']).name for row in receipts}
    actual = {path.name for path in (DIRECTORY / 'frames').glob('*.json')}
    require(actual == expected_payloads | expected_receipts, 'Unexpected or duplicate frame cache artifacts')
    return rows, receipts, {'status': 'completed', 'binding_fingerprint': binding, 'new_frames': new_frames,
                           'reused_frames': reused_frames, 'committed_frames': len(receipts), 'expected_frames': expected_total}


def finish(protocol, manifests, rows, receipts, evidence, checked, histories, initializations, binding, read_only=False, invocation=None):
    decision = adoption(checked, protocol)
    heads = {arm: grouped_heads(rows[arm], protocol) for arm in ARMS}
    heads_report = {'schema': 8, 'binding_fingerprint': binding, 'arms': heads, 'thresholds_changed': False,
                    'primary_endpoint_unchanged': True, 'all_800_model_frame_metrics_reproduced': True,
                    'checkpoints': evidence,
                    'metric_note': 'Center diagnostics use fixed decoder peaks with maximum-cardinality 5/10px matching. Offset EPE aggregates per-instance mean foreground pixel distances at320x240. Diagnostics are not adoption endpoints.'}
    publish_json_once(HEADS, heads_report, read_only)
    visuals = render_fixed(manifests[ARMS[0]], rows, protocol, binding, read_only)
    report = {'轮次': '第八轮同更新预算的训练覆盖对照', '执行状态': '完成', 'schema': 8, 'binding_fingerprint': binding,
              'decision': decision, '指标': checked, '每臂新增更新': 1600, '新增研究更新总数': 3200, '共同父模型累计更新': 3200,
              '训练覆盖与曝光': histories, '同父模型及重置初始状态': True, '跨臂样本顺序要求': '不要求相同；各自数据集内顺序及恢复已核验',
              '逐帧汇总独立复核': True, '全部800模型帧独立重推理一致': True, '解码阈值改变': False,
              '中心与偏移诊断': {arm: heads[arm]['groups'] for arm in ARMS}, '固定可视化': visuals,
              '运行证据': evidence, '初始化证据': initializations, '配置SHA256': file_sha256(PROTOCOL), '总报告SHA256': file_sha256(AGGREGATE),
              'GPU恢复证明SHA256': file_sha256(PROOF), '实际拉闸断电测试': False,
              '解释边界': ['十个验证场景均为开发验证，不是盲测、校准集或官方测试',
                         '新增训练场景、样本数与每帧重复次数共同改变；不能单独归因于物体多样性或某一因素',
                         '实例F1采用掩膜IoU≥0.50匹配，不是mask AP；中心与偏移诊断不是采纳终点',
                         '共同父模型暖启动，单种子、固定预算工程判断，不证明从零训练优势或统计显著性',
                         '未事后挑选检查点、调阈值或延长训练；完成负结果不等于效果达标',
                         '未读取剩余保留验证、校准或官方测试；没有抓取AP或机器人成功率结论']}
    publish_json_once(OUTPUT, report, read_only)
    if not read_only:
        require(invocation is not None and invocation['status'] == 'completed', 'Final invocation evidence missing')
        atomic_json(DIRECTORY / 'last_invocation.json', invocation)
    paths = [HEADS, OUTPUT, DIRECTORY / 'visuals/manifest.json', DIRECTORY / 'last_invocation.json'] + [ROOT / row['file'] for row in visuals]
    completion = {'schema': 8, 'status': 'completed', 'binding_fingerprint': binding, 'expected_model_frames': 800,
                  'frame_receipts': receipts, 'outputs': {str(path.relative_to(ROOT)): file_sha256(path) for path in paths},
                  'thresholds_changed': False, 'decision': decision}
    publish_json_once(DIRECTORY / 'completion.json', completion, read_only)
    return report


def run(max_new_frames=None):
    require(max_new_frames is None or type(max_new_frames) is int and max_new_frames > 0, 'max-new-frames must be positive')
    state = load_evidence()
    protocol, models, configs, manifests, results, evidence, checked, histories, initializations = state
    binding_record = make_binding(configs, manifests, evidence)
    binding = fingerprint(binding_record)
    complete_path = DIRECTORY / 'completion.json'
    completed = complete_path.exists()
    if completed:
        require(read(complete_path)['status'] == 'completed' and read(complete_path)['binding_fingerprint'] == binding,
                'Completed acceptance binding changed')
        require((DIRECTORY / 'binding.json').exists() and read(DIRECTORY / 'binding.json') == binding_record,
                'Completed acceptance binding document changed')
        rows, receipts, invocation = collect_frames(models, manifests, results, protocol, binding, completed=True)
        report = finish(protocol, manifests, rows, receipts, evidence, checked, histories, initializations, binding, read_only=True)
        print(json.dumps({'status': 'verified_completed_read_only', 'model_frames': 800, 'decision': report['decision']}, ensure_ascii=False), flush=True)
        return report
    ensure_disk_reserve(protocol['free_disk_reserve_gib'])
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(DIRECTORY):
        require(not complete_path.exists(), 'Another invocation completed acceptance; rerun read-only verification')
        if not (DIRECTORY / 'binding.json').exists():
            require(not (DIRECTORY / 'frames').exists(), 'Frame cache exists without experiment binding')
        publish_json_once(DIRECTORY / 'binding.json', binding_record)
        set_determinism()
        rows, receipts, invocation = collect_frames(models, manifests, results, protocol, binding, max_new_frames=max_new_frames,
                                                     runtime_identity=binding_record['actual_inference_runtime'])
        if invocation['status'] == 'step_limit_reached':
            atomic_json(DIRECTORY / 'last_invocation.json', invocation)
            print(json.dumps(invocation), flush=True)
            return invocation
        report = finish(protocol, manifests, rows, receipts, evidence, checked, histories, initializations, binding, invocation=invocation)
        print(json.dumps({'status': 'completed', 'invocation': invocation, 'decision': report['decision']}, ensure_ascii=False), flush=True)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new-frames', type=int, help='Engineering interruption boundary, not an evaluation subset or selection endpoint')
    args = parser.parse_args()
    run(args.max_new_frames)
