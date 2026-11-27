"""Join sealed local observations to previously committed geometry labels only."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_risk_common_v15 as common
import numpy as np

require = common.require
FRICTION = (-1., .2, .4, .6, .8, 1., 1.2)
METRICS = ('tpr', 'fpr', 'auroc')
FEATURE_ARRAYS = ('observed_points_float64.npy', 'valid_pixel_indices.npy', 'voxel_points.npy')
METRIC_CONTRACT = {'reference_event': 'official_collision_including_empty',
    'positive_direction': 'higher_global_occupied_volume_score_means_collision', 'threshold_comparison': '>',
    'auroc_tie_credit': 'one_half_exact', 'zero_denominator': 'null_with_reason',
    'aggregation': 'equal_frame_over_defined_metrics_no_zero_fill',
    'unknown_labels': 'not_scored_and_not_in_metric_denominator'}


def exact_record(value=None, reason=None):
    if value is None:
        require(isinstance(reason, str) and reason, 'Undefined metric needs a reason')
        return {'numerator': None, 'denominator': None, 'value': None, 'reason': reason}
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'value': float(value), 'reason': None}


def binary_metrics(trace):
    """Exact arithmetic on a fixed support; larger observation means collision."""
    confusion = dict(tp=0, fp=0, tn=0, fn=0)
    positives, negatives = [], []
    for row in trace:
        actual, predicted, score = row['collision'], row['predicted_collision'], row['global_iou']
        require(type(actual) is bool and type(predicted) is bool
                and type(score) in (float, int) and math.isfinite(score) and score >= 0,
                'Invalid binary label or observation')
        require(predicted == (score > .01), 'Observation threshold changed')
        confusion['tp' if actual and predicted else 'fn' if actual else 'fp' if predicted else 'tn'] += 1
        (positives if actual else negatives).append(score)
    positive_count, negative_count = len(positives), len(negatives)
    tpr = exact_record(Fraction(confusion['tp'], positive_count)) if positive_count else exact_record(
        reason='no_reference_support' if not trace else 'no_positive_reference')
    fpr = exact_record(Fraction(confusion['fp'], negative_count)) if negative_count else exact_record(
        reason='no_reference_support' if not trace else 'no_negative_reference')
    wins = sum(pos > neg for pos in positives for neg in negatives)
    ties = sum(pos == neg for pos in positives for neg in negatives)
    pairs = positive_count * negative_count
    reason = ('no_reference_support' if not trace else
              'no_positive_reference' if not positives else 'no_negative_reference')
    auroc = exact_record(Fraction(2 * wins + ties, 2 * pairs)) if pairs else exact_record(reason=reason)
    auroc.update(pair_count=pairs, wins=wins, ties=ties, losses=pairs - wins - ties)
    return {'support_count': len(trace), 'positive_reference_count': positive_count,
            'negative_reference_count': negative_count, 'confusion': confusion,
            'metrics': {'tpr': tpr, 'fpr': fpr, 'auroc': auroc}}


def frame_equal_metrics(frames):
    summary = {}
    for metric in METRICS:
        values, eligible, excluded = [], [], {}
        for frame, report in frames.items():
            value = report['metrics'][metric]
            if value['value'] is None:
                excluded[frame] = value['reason']
            else:
                values.append(Fraction(value['numerator'], value['denominator']))
                eligible.append(frame)
        mean = exact_record(sum(values, Fraction(0)) / len(values)) if values else exact_record(
            reason='no_eligible_frames')
        summary[metric] = {'eligible_frames': eligible, 'excluded_frames': excluded,
                           'eligible_frame_count': len(eligible), 'mean': mean}
    return summary


def validate_candidates(candidates):
    require(candidates.dtype == np.float32 and candidates.ndim == 2 and candidates.shape[1] == 17
            and 0 <= len(candidates) <= 1024 and np.isfinite(candidates).all()
            and (candidates[:, 16] == -1).all() and (candidates[:, 1] >= 0).all()
            and (candidates[:, 2:4] > 0).all(), 'Invalid original candidate array')
    return candidates


def validate_feature_rows(frame, candidates, rows):
    validate_candidates(candidates)
    require(len(rows) == len(candidates), 'Feature computation dropped candidates')
    for index, row in enumerate(rows):
        require(type(row['original_index']) is int and row['original_index'] == index
                and row['frame'] == frame, 'Feature frame/index differs')
        require(row['raw_confidence'] == float(candidates[index, 0]), 'Raw confidence changed')
        require(row['candidate_row_sha256'] == hashlib.sha256(candidates[index].tobytes(order='C')).hexdigest(),
                'Feature candidate bytes differ')
        scores = row['occupied_volume_scores']
        require(set(scores) == set(common.SCORE_NAMES) and all(type(score) in (int, float)
                and math.isfinite(score) and score >= 0 for score in scores.values()), 'Invalid occupied-volume scores')
        require(type(row['collision']) is bool and row['collision'] == (scores['global'] > .01),
                'Fixed observation collision rule differs')
        applicable = bool(candidates[index, 1] > 0)
        require(type(row['empty_diagnostic_applicable']) is bool
                and row['empty_diagnostic_applicable'] == applicable, 'Empty diagnostic applicability differs')
        require((type(row['empty_diagnostic']) is bool and row['empty_diagnostic_reason'] is None) if applicable else
                (row['empty_diagnostic'] is None and row['empty_diagnostic_reason'] == 'zero_width'),
                'Undefined zero-width diagnostic was not retained')
    return rows


def feature_names(block_count):
    names = {'binding.json'}
    for name in list(FEATURE_ARRAYS) + ['preparation.json', 'report.json', 'completion.json'] + [
            'blocks/block_%04d.json' % index for index in range(block_count)]:
        names.update((name, name + '.sha256.json'))
    return names


def verify_inventory(directory, names):
    actual = set()
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'Link in immutable inventory')
        name = path.relative_to(directory).as_posix()
        if path.is_dir():
            require(any(target.startswith(name + '/') for target in names), 'Unexpected sealed directory')
        else:
            actual.add(name)
    require(actual == set(names), 'Incomplete or unsealed output inventory')


def verify_features(root, config, protocol, guard):
    """Finish all four label-free seals before any evaluation input is opened."""
    frames, dynamic_inputs = {}, {}
    for frame in protocol['frames']:
        binding = common.make_binding(root, config, protocol, phase='features', frame=frame)
        directory = common.frame_directory(root, protocol, frame)
        require(common.read(directory / 'binding.json') == binding, 'Feature binding differs')
        completion = common.load_commit(directory / 'completion.json', binding)
        count = completion['candidate_count']
        require(type(count) is int and 0 <= count <= 1024, 'Invalid feature candidate count')
        blocks = (count + protocol['block_size'] - 1) // protocol['block_size']
        require(completion['schema'] == 15 and completion['status'] == 'completed'
                and completion['frame'] == frame and completion['binding_fingerprint'] == binding['fingerprint']
                and completion['expected_blocks'] == blocks, 'Complete frame features required before label access')
        names = feature_names(blocks)
        require(set(completion['outputs']) == names - {'completion.json', 'completion.json.sha256.json'},
                'Feature completion inventory differs')
        verify_inventory(directory, names)
        for name in sorted(names):
            path = common.safe_path(directory, name)
            sha = common.digest(path)
            if name in completion['outputs']:
                require(sha == completion['outputs'][name], 'Sealed feature file changed')
            dynamic_inputs[path.relative_to(root).as_posix()] = sha
        for name in FEATURE_ARRAYS:
            common.verify_bytes(directory / name, binding)
        preparation = common.load_commit(directory / 'preparation.json', binding)
        report = common.load_commit(directory / 'report.json', binding)
        require(report['schema'] == 15 and report['status'] == 'completed'
                and report['frame'] == frame and report['candidate_count'] == count and report['expected_blocks'] == blocks
                and report['binding_fingerprint'] == binding['fingerprint']
                and completion['report_sha256'] == common.digest(directory / 'report.json')
                and report['preparation_sha256'] == common.digest(directory / 'preparation.json'),
                'Feature report identity differs')
        require(all(report[key] is False for key in ('uses_ground_truth_labels', 'uses_object_geometry',
                    'gpu_used', 'nms_applied', 'empty_is_rejection', 'unobserved_surfaces_ruled_safe'))
                and report['all_candidates_preserved'] is True
                and all(type(report[key]) is int and report[key] == 0 for key in
                        ('new_training_steps', 'new_model_forward_calls', 'new_official_geometry_computations')),
                'Feature isolation or computation contract differs')
        require(preparation['schema'] == 15 and preparation['status'] == 'prepared' and preparation['frame'] == frame
                and preparation['binding_fingerprint'] == binding['fingerprint'] and preparation['candidate_count'] == count
                and preparation['point_precision'] == 'float64' and preparation['candidate_precision'] == 'float32'
                and preparation['metadata_fields_read'] == ['intrinsic_matrix', 'factor_depth']
                and preparation['files'] == {name: common.digest(directory / name) for name in FEATURE_ARRAYS},
                'Feature preparation source/precision differs')
        rows = []
        for index in range(blocks):
            block = common.load_commit(directory / ('blocks/block_%04d.json' % index), binding)
            start, stop = index * protocol['block_size'], min(count, (index + 1) * protocol['block_size'])
            require(block['schema'] == 15 and block['frame'] == frame
                    and block['binding_fingerprint'] == binding['fingerprint']
                    and block['block_index'] == index and block['start'] == start and block['stop'] == stop,
                    'Feature block partition differs')
            require([row['original_index'] for row in block['rows']] == list(range(start, stop)),
                    'Feature block indices differ')
            rows.extend(block['rows'])
        require(rows == report['rows'], 'Feature report rows differ from blocks')
        require(report['counts'] == {'collision': sum(row['collision'] for row in rows),
            'empty_diagnostic': sum(row['empty_diagnostic'] is True for row in rows),
            'empty_not_applicable': sum(not row['empty_diagnostic_applicable'] for row in rows)},
            'Feature summary counts differ')
        generation = common.safe_path(root, protocol['frame_inputs'][frame]['generation_dir'])
        candidates = validate_candidates(np.load(generation / 'candidates.npy', allow_pickle=False))
        require(len(candidates) == count and report['candidates_sha256'] == common.digest(generation / 'candidates.npy')
                and report['source_generation_completion_sha256'] == common.digest(generation / 'completion.json'),
                'Feature source candidates differ')
        generation_report = common.read(generation / 'report.json')
        require(generation_report['frame'] == frame and generation_report['candidate_count'] == count
                and generation_report['input_status'] in ('ready', 'insufficient_input')
                and preparation['input_status'] == generation_report['input_status']
                and preparation['candidates_sha256'] == report['candidates_sha256']
                and preparation['source_generation_completion_sha256'] == report['source_generation_completion_sha256'],
                'Generation frame/count differs')
        validate_feature_rows(frame, candidates, rows)
        frames[frame] = {'candidates': candidates, 'rows': rows,
            'input_status': generation_report['input_status'],
            'seal': {'binding_fingerprint': binding['fingerprint'],
                     'completion_sha256': dynamic_inputs[(directory / 'completion.json').relative_to(root).as_posix()],
                     'report_sha256': completion['report_sha256'],
                     'candidates_sha256': report['candidates_sha256']}}
        guard.check()
    return frames, dynamic_inputs


def parent_binding(binding, frame, protocol):
    value = dict(binding)
    checksum = value.pop('fingerprint', None)
    require(common.fingerprint(value) == checksum and binding['schema'] == 14
            and binding['input_group'] == 'geometry_inputs' and binding['frame'] == frame
            and binding['runtime_name'] == 'grasp_eval'
            and binding['protocol_sha256'] == protocol['shared_inputs']['configs/grasp_multiview_v14r1.json'],
            'Frozen parent geometry binding differs')


def geometry_names(count):
    names = {'binding.json'}
    for name in ['preparation.json', 'report.json', 'oracle.json', 'completion.json'] + [
            'grasps/grasp_%04d.json' % index for index in range(count)]:
        names.update((name, name + '.sha256.json'))
    return names


def load_geometry(root, protocol, frame):
    """Read cached labels without importing a detector, scorer, model or NMS."""
    directory = common.safe_path(root, protocol['parent_geometry_dirs'][frame])
    common.verify_frozen_tree(root, directory, protocol['evaluation_inputs'])
    binding = common.read(directory / 'binding.json')
    parent_binding(binding, frame, protocol)
    completion = common.storage.load_commit(directory / 'completion.json', binding)
    preparation = common.storage.load_commit(directory / 'preparation.json', binding)
    count = preparation['selected_count']
    require(type(count) is int and 0 <= count <= preparation['input_count'] <= 1024,
            'Invalid dynamic parent support count')
    require(completion['schema'] == preparation['schema'] == 14
            and completion['frame'] == preparation['frame'] == frame and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == binding['fingerprint']
            and completion['expected_grasps'] == count, 'Frozen geometry is not complete')
    names = geometry_names(count)
    verify_inventory(directory, names)
    require(set(completion['outputs']) == names - {'completion.json', 'completion.json.sha256.json'},
            'Geometry completion inventory differs')
    for name, sha in completion['outputs'].items():
        require(common.digest(directory / name) == sha, 'Frozen geometry file changed')
    records, hashes = [], []
    for index in range(count):
        path = directory / ('grasps/grasp_%04d.json' % index)
        records.append(common.storage.load_commit(path, binding))
        hashes.append(common.digest(path))
    oracle = common.storage.load_commit(directory / 'oracle.json', binding)
    report = common.storage.load_commit(directory / 'report.json', binding)
    require(report['schema'] == 14 and report['frame'] == frame
            and report['binding_fingerprint'] == binding['fingerprint']
            and completion['report_sha256'] == common.digest(directory / 'report.json'), 'Parent report source differs')
    source = {name.replace('.json', '') + '_sha256': common.digest(directory / name)
              for name in ('binding.json', 'completion.json', 'preparation.json', 'oracle.json', 'report.json')}
    return preparation, report, records, hashes, oracle, source


def join_support(frame, candidates, features, geometry):
    preparation, report, records, hashes, oracle, source = geometry
    validate_feature_rows(frame, candidates, features)
    count, support = len(candidates), len(records)
    require(preparation['input_dtype'] == 'float32' and preparation['input_count'] == count
            and preparation['selected_count'] == support == len(preparation['selected']) == len(hashes),
            'Parent selection/candidate count differs')
    require(report['candidate_count'] == report['original_candidate_count'] == count
            and report['selected_count'] == support, 'Parent report support differs')
    clipped = candidates.copy()
    clipped[clipped[:, 1] < 0, 1] = 0
    clipped[clipped[:, 1] > .1, 1] = .1
    original_ids, trace = [], []
    for selection, (item, record, checksum) in enumerate(zip(preparation['selected'], records, hashes)):
        original, slot = item['original_index'], item['object_slot']
        require(type(original) is int and 0 <= original < count and type(slot) is int
                and 0 <= slot < len(preparation['object_ids']) and item['selection_index'] == selection,
                'Invalid frozen support provenance')
        require(item['object_id'] == preparation['object_ids'][slot]
                and all(record.get(key) == value for key, value in item.items()), 'Parent record identity differs')
        require(record['grasp_dtype'] in ('float32', 'float64') and record['grasp'] == clipped[original].tolist(),
                'Geometry row differs from exact float32-width-clipped candidate')
        require(type(record['collision']) is bool and type(record['empty']) is bool
                and (not record['empty'] or record['collision']) and type(record['score']) in (int, float)
                and record['score'] in FRICTION
                and (not record['collision'] or record['score'] == -1.), 'Invalid official collision/empty semantics')
        require(isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum) is not None,
                'Missing immutable record checksum')
        feature = features[original]
        original_ids.append(original)
        trace.append({'frame': frame, 'selection_index': selection, 'original_index': original,
            'object_slot': slot, 'object_id': record['object_id'], 'object_local_index': record['object_local_index'],
            'grasp_record_sha256': checksum, 'candidate_row_sha256': feature['candidate_row_sha256'],
            'global_iou': feature['occupied_volume_scores']['global'], 'predicted_collision': feature['collision'],
            'collision': record['collision'], 'empty': record['empty'], 'friction_score': record['score']})
    require(len(set(original_ids)) == support, 'Duplicate original index in parent support')
    groups, scores, collisions = [], [], []
    for slot in range(len(preparation['object_ids'])):
        selected = [record for record in records if record['object_slot'] == slot]
        require([record['object_local_index'] for record in selected] == list(range(len(selected))),
                'Parent object-local indices differ')
        groups.append([record['grasp'] for record in selected])
        scores.append([record['score'] for record in selected])
        collisions.append([record['collision'] for record in selected])
    require(oracle['result'] == {'groups': groups, 'scores': scores, 'collision': collisions},
            'Cached official oracle differs from immutable records')
    require((count == 0 and support == 0 and oracle['executed'] is False
             and report['official_oracle_executed'] is False and report['official_oracle_exact_match'] is None)
            or (count > 0 and support > 0 and oracle['executed'] is True
                and report['official_oracle_executed'] is True and report['official_oracle_exact_match'] is True),
            'Official empty/nonempty path differs')
    confidence = np.asarray([record['grasp'][0] for record in records], dtype=np.float32)
    ordered = [records[int(index)] for index in np.argsort(-confidence)]
    require(report['ordered_records'] == ordered, 'Original official confidence order differs')
    require(report['collision_including_empty_count'] == sum(row['collision'] for row in records)
            and report['empty_count'] == sum(row['empty'] for row in records)
            and report['positive_friction_score_count'] == sum(row['score'] > 0 for row in records),
            'Parent label counts differ')
    return trace, source


def build_report(protocol, binding, features, geometries, parent_report_sha256):
    frames = {}
    for frame in protocol['frames']:
        feature = features[frame]
        trace, source = join_support(frame, feature['candidates'], feature['rows'], geometries[frame])
        frames[frame] = dict(binary_metrics(trace), frame=frame, candidate_count=len(feature['candidates']),
            selected_count=len(trace), unscored_candidate_count=len(feature['candidates']) - len(trace),
            input_status=feature['input_status'], label_trace=trace, sources=source)
    return {'schema': 15, 'status': 'completed_conditional_collision_diagnostic', 'binding': binding,
        'binding_fingerprint': binding['fingerprint'], 'protocol_sha256': binding['protocol_sha256'],
        'feature_seals': {frame: features[frame]['seal'] for frame in protocol['frames']},
        'parent_acceptance_report_sha256': parent_report_sha256, 'frames': frames,
        'frame_equal_metrics': frame_equal_metrics(frames),
        'metric_policy': {'reference_event': 'official_collision_including_empty',
            'observed_collision_threshold': .01, 'threshold_comparison': 'strictly_greater',
            'risk_score': 'global_occupied_volume_score', 'higher_score_means': 'collision',
            'auroc_ties': 'half_credit_exact_pairs', 'metrics': list(METRICS),
            'aggregation': 'equal_frame_mean_over_eligible_frames_only'},
        'score_semantics': 'Observed point count / voxelized geometric volume; not set IoU or calibrated probability',
        'all_features_sealed_before_label_access': True, 'unscored_labels_are_unknown': True,
        'new_feature_computations': 0, 'new_geometry_computations': 0, 'new_official_geometry_computations': 0,
        'new_model_forward_calls': 0, 'new_training_steps': 0, 'new_nms_or_support_selection': False,
        'gpu_used': False, 'efficacy_gate': False, 'method_improvement_claimed': False,
        'dataset_ap_computed': False, 'calibrated_probability_claimed': False,
        'independent_confirmation': False, 'limitations': protocol['limitations']}


def recover_report_temporaries(path, check_only):
    """Only this report and its receipt own temporary names in shared P3 storage."""
    for target in (path, common.receipt_path(path)):
        for candidate in target.parent.glob(target.name + '.*.tmp'):
            require(not candidate.is_symlink() and re.fullmatch(re.escape(target.name) + r'\.[0-9a-f]{32}\.tmp',
                    candidate.name) is not None and not check_only, 'Unsealed evaluation temporary')
            candidate.unlink()


def publish_or_verify(path, report, binding, guard, read_only):
    recover_report_temporaries(path, read_only)
    payload = common.json_bytes(report)
    receipt = {'schema': 15, 'binding_fingerprint': binding['fingerprint'],
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    extra = (0 if path.exists() else len(payload)) + (
        0 if common.receipt_path(path).exists() else len(common.json_bytes(receipt)))
    guard.check(extra)
    common.commit(path, report, binding, read_only)
    require(common.load_commit(path, binding) == report, 'Completed evaluation does not reproduce')
    return {'status': 'verified_completed_read_only' if read_only else 'completed',
        'report_sha256': common.digest(path), 'report_receipt_sha256': common.digest(common.receipt_path(path)),
        'evaluated_frames': len(report['frames']),
        'evaluated_candidates': sum(frame['selected_count'] for frame in report['frames'].values()),
        'new_feature_computations': 0, 'new_geometry_computations': 0, 'new_official_geometry_computations': 0,
        'new_model_forward_calls': 0, 'new_training_steps': 0, 'resources': guard.check()}


def run(root, config, check_only=False):
    root = Path(root).resolve()
    config = Path(config)
    config = config.resolve() if config.is_absolute() else (root / config).resolve()
    require(config.is_relative_to(root), 'Configuration outside project')
    protocol = common.read(config)
    common.validate_protocol(protocol)
    require(protocol['metrics'] == METRIC_CONTRACT, 'Prespecified evaluation metric policy changed')
    guard = common.Guard(root, protocol, phase='evaluation')
    guard.check()
    # No geometry read or hash is permitted until all frame feature seals pass.
    features, dynamic_inputs = verify_features(root, config, protocol, guard)
    binding = common.make_binding(root, config, protocol, phase='evaluation', extra_inputs=dynamic_inputs)
    parent_path = common.safe_path(root, protocol['parent_acceptance_report'])
    require(parent_path.relative_to(root).as_posix() in protocol['evaluation_inputs'], 'Parent acceptance is unbound')
    parent = common.read(parent_path)
    require(parent['schema'] == 14 and parent['status'] == 'accepted_descriptive_multiview_baseline_engineering'
            and list(parent['frames']) == protocol['frames'], 'Parent multiview acceptance is incomplete')
    geometries = {}
    for frame in protocol['frames']:
        geometries[frame] = load_geometry(root, protocol, frame)
        guard.check()
    report = build_report(protocol, binding, features, geometries, common.digest(parent_path))
    path = common.safe_path(root, protocol['outputs']['evaluation_report'])
    if check_only or path.exists():
        return publish_or_verify(path, report, binding, guard, True)
    with common.lock(root / 'runtime/locks/local_risk_evaluation_v15'):
        return publish_or_verify(path, report, binding, guard, path.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(common.CONFIG))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(common.ROOT, args.config, args.check_only), ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
