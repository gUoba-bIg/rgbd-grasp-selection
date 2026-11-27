"""Evaluate one frozen soft order on the same previously scored candidate set."""
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
import soft_ranking_common_v13 as common

import numpy as np

FRICTION = (.2, .4, .6, .8, 1., 1.2)
CANDIDATES = 236
SUPPORT = 50


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_feature_ranking(candidates, report):
    """Check the complete label-free order before allowing label-source decoding."""
    require(candidates.dtype == np.float32 and candidates.shape == (CANDIDATES, 17)
            and np.isfinite(candidates).all(), 'Frozen candidate array differs')
    rows = report['rows']
    require(len(rows) == CANDIDATES and [row['original_index'] for row in rows] == list(range(CANDIDATES)),
            'Feature rows must preserve every original candidate in order')
    for index, row in enumerate(rows):
        require(row['raw_confidence'] == float(candidates[index, 0]), 'Raw candidate confidence changed')
        require(all(math.isfinite(row[key]) for key in ('p', 'h', 'v', 'c', 'q', 'multiplier', 'soft_score')), 'Nonfinite feature/rank value')
        require(all(0 <= row[key] <= 1 for key in ('p', 'h', 'v', 'c', 'q')), 'Feature quality outside unit interval')
        quality = (row['p'] * (1 - row['h']) + row['v'] * row['c']) / 2
        require(row['q'] == quality and row['multiplier'] == .5 + .5 * quality
                and row['soft_score'] == row['raw_confidence'] * row['multiplier'], 'Frozen soft-score formula differs')
    expected = sorted(range(CANDIDATES), key=lambda i: (-rows[i]['soft_score'], i))
    require(report['rank_original_indices'] == expected, 'Full soft order or tie policy differs')
    return rows, expected


def extract_support(candidates, preparation, geometry_report, records, record_hashes):
    """Use frozen original IDs, never nearest geometry or a new selection pass."""
    require(preparation['input_count'] == CANDIDATES and preparation['selected_count'] == SUPPORT
            and len(preparation['selected']) == len(records) == len(record_hashes) == SUPPORT,
            'The fixed shared support must contain exactly 50 of the same 236 candidates')
    clipped = candidates.copy()
    clipped[clipped[:, 1] < 0, 1] = 0
    clipped[clipped[:, 1] > .1, 1] = .1
    indices = []
    trace = []
    for selection_index, (item, record, checksum) in enumerate(zip(preparation['selected'], records, record_hashes)):
        original = item['original_index']
        require(type(original) is int and 0 <= original < CANDIDATES and item['selection_index'] == selection_index,
                'Invalid frozen support provenance')
        require(all(record.get(key) == value for key, value in item.items()), 'Geometry score identity differs from preparation')
        require(record['grasp'] == clipped[original].tolist(), 'Geometry record is not the exact float32-width-clipped original row')
        require(record['score'] in (-1.,) + FRICTION and type(record['collision']) is bool
                and (not record['collision'] or record['score'] == -1.), 'Invalid frozen geometry label')
        require(isinstance(checksum, str) and len(checksum) == 64, 'Missing immutable score-record digest')
        indices.append(original)
        trace.append({'selection_index': selection_index, 'original_index': original, 'score': record['score'],
                      'collision': record['collision'], 'grasp_record_sha256': checksum})
    require(len(set(indices)) == SUPPORT, 'Duplicate original candidate in shared support')
    confidence = np.asarray([record['grasp'][0] for record in records], dtype=preparation['input_dtype'])
    require(preparation['input_dtype'] == 'float32' and len(np.unique(confidence)) == SUPPORT,
            'Baseline confidence tie requires a separately specified fair tie policy; no silent substitution')
    baseline_records = [records[int(i)] for i in np.argsort(-confidence)]
    require(geometry_report['ordered_records'] == baseline_records, 'Official baseline order is not reproduced exactly')
    require(geometry_report['original_candidate_count'] == CANDIDATES and geometry_report['selected_count'] == SUPPORT,
            'Official report candidate/support count differs')
    return indices, [record['original_index'] for record in baseline_records], trace


def support_projection(full_rank, support_indices):
    require(len(full_rank) == CANDIDATES and set(full_rank) == set(range(CANDIDATES)), 'Full rank is not a permutation')
    require(len(support_indices) == SUPPORT and len(set(support_indices)) == SUPPORT, 'Shared support is not fixed 50')
    support = set(support_indices)
    projected = [original for original in full_rank if original in support]
    require(len(projected) == SUPPORT and set(projected) == support, 'Projection changed support membership')
    return projected


def exact_metrics(order, label_trace):
    labels = {item['original_index']: item['score'] for item in label_trace}
    require(len(order) == SUPPORT and len(set(order)) == SUPPORT and set(order) == set(labels), 'Ranking/label join changed support')
    scores = [labels[original] for original in order]
    require(all(score in (-1.,) + FRICTION for score in scores), 'Unsupported geometry score')
    counts, fractions = [], []
    total = Fraction(0)
    for prefix in range(1, SUPPORT + 1):
        count_row = [sum(0 < score <= mu for score in scores[:prefix]) for mu in FRICTION]
        counts.append(count_row)
        fractions.append([number / prefix for number in count_row])
        total += sum((Fraction(number, prefix) for number in count_row), Fraction(0))
    return {'counts_50x6': counts, 'fraction_50x6': fractions, 'top10_mu04_count': counts[9][1],
            'top50_counts_by_friction': counts[-1], 'mean_exact': total / (SUPPORT * len(FRICTION))}


def fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'display_value': float(value)}


def metrics_record(metrics):
    return dict(metrics, mean_exact=fraction_record(metrics['mean_exact']))


def decision(baseline, soft):
    conserved = baseline['top50_counts_by_friction'] == soft['top50_counts_by_friction']
    require(conserved, 'Pure reordering changed full-support labels; do not report an effect')
    primary = soft['top10_mu04_count'] > baseline['top10_mu04_count']
    secondary = soft['mean_exact'] >= baseline['mean_exact']
    positive = primary and secondary
    return {'decision': 'positive_exploratory_fixed_support_ranking' if positive else 'valid_negative_do_not_adopt',
            'positive_effect_gate_passed': positive, 'top10_strict_increase': primary,
            'exact_mean_non_decrease': secondary, 'all_friction_top50_conserved': conserved,
            'top10_mu04_count_difference': soft['top10_mu04_count'] - baseline['top10_mu04_count'],
            'mean_exact_difference': fraction_record(soft['mean_exact'] - baseline['mean_exact']),
            'dataset_ap_claimed': False, 'statistical_significance_claimed': False}


def validate_protocol(protocol):
    require(protocol['schema'] == 13 and protocol['candidate_count'] == CANDIDATES
            and protocol['support_count'] == SUPPORT and protocol['block_size'] == 16,
            'Frozen candidate/support definition changed')
    evaluation = protocol['evaluation']
    require(evaluation['friction_grid'] == list(FRICTION) and evaluation['primary_top_k'] == 10
            and evaluation['primary_friction'] == .4
            and evaluation['positive_decision'] == 'positive_exploratory_fixed_support_ranking'
            and evaluation['negative_decision'] == 'valid_negative_do_not_adopt', 'Frozen evaluation gate changed')
    require(protocol['feature_definition']['quality'] == '(p*(1-h)+v*c)/2'
            and protocol['feature_definition']['sort'] == 'descending soft_score then ascending original_index',
            'Frozen feature/rank definition changed')
    require(not evaluation['dataset_ap'] and not evaluation['independent_confirmation']
            and not evaluation['calibrated_risk_probability'], 'Unsupported scientific claim')


def check_layout(directory, allowed, complete=False, remove_temporaries=False):
    actual = set()
    directories = {str(Path(name).parent).replace('\\', '/') for name in allowed if '/' in name}
    for path in directory.rglob('*'):
        name = path.relative_to(directory).as_posix()
        if path.is_dir():
            require(name in directories, 'Unexpected cache directory: ' + name)
        elif name in allowed:
            actual.add(name)
        else:
            temporary = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', name)
            require(temporary is not None and temporary.group(1) in allowed, 'Unexpected cache file: ' + name)
            require(remove_temporaries and not complete, 'Unsealed temporary requires writable recovery: ' + name)
            # Only this exact bounded cache target and UUID pattern may be removed.
            require(path.resolve().is_relative_to(directory.resolve()), 'Temporary path escaped cache')
            path.unlink()
    require(not complete or actual == allowed, 'Incomplete cache membership')


def verify_manifest(directory, outputs, expected_names):
    require(set(outputs) == expected_names, 'Completion manifest membership differs')
    for name, expected in outputs.items():
        require(common.digest(common.safe_path(directory, name)) == expected, 'Sealed output changed: ' + name)


def verify_features(root, protocol, binding):
    """Complete all label-free verification before opening any geometry source."""
    directory = common.safe_path(root, protocol['outputs']['features_dir'])
    require(common.read(directory / 'binding.json') == binding, 'Feature binding differs from current frozen sources')
    completion = common.load_commit(directory / 'completion.json', binding)
    block_count = (CANDIDATES + protocol['block_size'] - 1) // protocol['block_size']
    require(completion['schema'] == 13 and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == binding['fingerprint']
            and completion['candidate_count'] == CANDIDATES and completion['block_count'] == block_count,
            'Full feature seal required before geometry access')
    names = {'binding.json', 'report.json', 'report.json.sha256.json'}
    for index in range(block_count):
        names.update(('blocks/block_%04d.json' % index, 'blocks/block_%04d.json.sha256.json' % index))
    verify_manifest(directory, completion['outputs'], names)
    check_layout(directory, names | {'completion.json', 'completion.json.sha256.json'}, complete=True)
    report = common.load_commit(directory / 'report.json', binding)
    require(completion['report_sha256'] == common.digest(directory / 'report.json'), 'Feature report seal differs')
    require(report['schema'] == 13 and report['status'] == 'completed'
            and report['binding_fingerprint'] == binding['fingerprint']
            and report['candidate_count'] == CANDIDATES and report['block_count'] == block_count
            and report['model_forward_calls'] == 0 and report['training_steps'] == 0
            and report['geometry_labels_read'] is False
            and report['feature_definition'] == protocol['feature_definition']
            and report['source_hashes'] == protocol['source_hashes']
            and report['candidate_sha256'] == protocol['feature_inputs'][protocol['candidate_path']]
            and report['protocol_sha256'] == binding['protocol_sha256']
            and report['feature_inputs'] == protocol['feature_inputs'], 'Feature report source provenance differs')
    rows = []
    for index in range(block_count):
        block = common.load_commit(directory / 'blocks' / ('block_%04d.json' % index), binding)
        block_rows = block['rows']
        start, stop = index * protocol['block_size'], min(CANDIDATES, (index + 1) * protocol['block_size'])
        require(block['schema'] == 13 and block['binding_fingerprint'] == binding['fingerprint']
                and block['start'] == start and block['stop'] == stop
                and [row['original_index'] for row in block_rows] == list(range(start, stop)), 'Feature block partition differs')
        rows.extend(block_rows)
    require(rows == report['rows'], 'Feature report rows differ from committed blocks')
    candidates = np.load(common.safe_path(root, protocol['candidate_path']), allow_pickle=False)
    _, order = validate_feature_ranking(candidates, report)
    return candidates, report, order, common.digest(directory / 'completion.json')


def legacy_fingerprint(binding):
    # R12 scoring intentionally used default json separators/ASCII, not R13 JSON.
    return hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_legacy_commit(path, binding_fingerprint):
    receipt = common.read(path.with_suffix('.sha256.json'))
    require(receipt == {'sha256': common.digest(path), 'binding_fingerprint': binding_fingerprint},
            'Frozen geometry receipt differs: ' + str(path))
    return common.read(path)


def load_geometry(root, protocol):
    """Decode only sealed cached labels; this module has no geometry/model import."""
    directory = common.safe_path(root, protocol['geometry_dir'])
    binding = common.read(directory / 'binding.json')
    fingerprint = legacy_fingerprint(binding)
    completion = common.read(directory / 'completion.json')
    require(binding['schema'] == 12 and binding['arm'] == 'full_scene'
            and completion['schema'] == 12 and completion['arm'] == 'full_scene'
            and completion['status'] == 'completed' and completion['binding_fingerprint'] == fingerprint
            and completion['expected_grasps'] == SUPPORT, 'Frozen full-scene geometry seal differs')
    names = {'binding.json', 'preparation.json', 'preparation.sha256.json', 'oracle.json',
             'oracle.sha256.json', 'report.json', 'last_invocation.json'}
    for index in range(SUPPORT):
        names.update(('grasps/grasp_%04d.json' % index, 'grasps/grasp_%04d.sha256.json' % index))
    verify_manifest(directory, completion['outputs'], names)
    check_layout(directory, names | {'completion.json'}, complete=True)
    require(all((directory / name).relative_to(root).as_posix() in protocol['evaluation_inputs']
                for name in names | {'completion.json'}), 'Geometry seal member is not protocol-bound')
    preparation = load_legacy_commit(directory / 'preparation.json', fingerprint)
    require(preparation['schema'] == 12 and preparation['arm'] == 'full_scene', 'Wrong preparation source')
    records, hashes = [], []
    for index in range(SUPPORT):
        path = directory / 'grasps' / ('grasp_%04d.json' % index)
        records.append(load_legacy_commit(path, fingerprint))
        hashes.append(common.digest(path))
    oracle = load_legacy_commit(directory / 'oracle.json', fingerprint)
    groups, scores, collisions = [], [], []
    for slot in range(len(preparation['object_ids'])):
        selected = [record for record in records if record['object_slot'] == slot]
        groups.append([record['grasp'] for record in selected])
        scores.append([record['score'] for record in selected])
        collisions.append([record['collision'] for record in selected])
    require(oracle == {'executed': True, 'result': {'groups': groups, 'scores': scores, 'collision': collisions}},
            'Cached official oracle differs from all 50 immutable labels')
    report = common.read(directory / 'report.json')
    require(report['schema'] == 12 and report['arm'] == 'full_scene'
            and report['binding_fingerprint'] == fingerprint
            and report['official_oracle_executed'] is True and report['official_oracle_exact_match'] is True,
            'Cached official report provenance differs')
    source = {'source_geometry_binding_sha256': common.digest(directory / 'binding.json'),
              'source_geometry_completion_sha256': common.digest(directory / 'completion.json'),
              'source_geometry_preparation_sha256': common.digest(directory / 'preparation.json'),
              'source_geometry_oracle_sha256': common.digest(directory / 'oracle.json'),
              'source_geometry_report_sha256': common.digest(directory / 'report.json')}
    return preparation, report, records, hashes, source


def build_report(candidates, features, full_rank, geometry, binding, feature_binding, feature_hash, protocol):
    preparation, geometry_report, records, hashes, source = geometry
    support, baseline_order, trace = extract_support(candidates, preparation, geometry_report, records, hashes)
    soft_order = support_projection(full_rank, support)
    baseline, soft = exact_metrics(baseline_order, trace), exact_metrics(soft_order, trace)
    require(geometry_report['friction_grid_ascending'] == list(FRICTION)
            and geometry_report['single_frame_prefix_pass_fraction_50x6'] == baseline['fraction_50x6']
            and geometry_report['single_frame_prefix_matrix_mean'] == float(np.asarray(baseline['fraction_50x6']).mean()),
            'Official baseline prefix matrix/mean does not reproduce')
    scores = [row['soft_score'] for row in features['rows']]
    support_scores = [scores[index] for index in support]
    return {'schema': 13, 'status': 'completed_common_support_diagnostic',
            'binding_fingerprint': binding['fingerprint'], 'feature_binding_fingerprint': feature_binding['fingerprint'],
            'feature_completion_sha256': feature_hash, 'protocol_sha256': binding['protocol_sha256'],
            'support': dict(candidate_count=CANDIDATES, selected_count=SUPPORT,
                            original_indices_by_selection=support, baseline_original_indices=baseline_order,
                            soft_original_indices=soft_order, baseline_confidence_ties=0,
                            full_soft_duplicate_score_count=len(scores) - len(set(scores)),
                            support_soft_duplicate_score_count=len(support_scores) - len(set(support_scores)),
                            soft_tie_policy='descending soft_score then ascending original_index', **source),
            'label_trace': trace, 'metrics': {'baseline': metrics_record(baseline), 'soft': metrics_record(soft)},
            'decision': decision(baseline, soft), 'friction_grid': list(FRICTION),
            'new_geometry_computations': 0, 'new_model_forward_calls': 0, 'new_training_steps': 0,
            'feature_sealed_before_label_access': True, 'new_nms_or_support_selection': False,
            'dataset_ap_computed': False, 'independent_confirmation': False,
            'limitations': protocol['limitations']}


def run(root, config, check_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    require(config.is_relative_to(root), 'Configuration is outside project')
    protocol = common.read(config)
    validate_protocol(protocol)
    guard = common.Guard(root, protocol)
    guard.check()
    # This ordering is part of the experiment: no geometry digest/read before the
    # entire source-bound feature rank and its receipts have passed verification.
    feature_binding = common.make_binding(root, config, protocol, 'feature_inputs')
    candidates, features, full_rank, feature_hash = verify_features(root, protocol, feature_binding)
    binding = common.make_binding(root, config, protocol, 'evaluation_inputs')
    geometry = load_geometry(root, protocol)
    report = build_report(candidates, features, full_rank, geometry, binding, feature_binding, feature_hash, protocol)
    directory = common.safe_path(root, protocol['outputs']['evaluation_dir'])
    allowed = {'binding.json', 'report.json', 'report.json.sha256.json', 'completion.json', 'completion.json.sha256.json'}

    def publish_or_verify(read_only):
        check_layout(directory, allowed, remove_temporaries=not read_only)
        guard.check(0 if (directory / 'binding.json').exists() else len(common.json_bytes(binding)))
        common.once(directory / 'binding.json', binding, read_only)
        guard.check()

        def guarded_commit(path, value):
            payload = common.json_bytes(value)
            receipt = {'schema': 13, 'binding_fingerprint': binding['fingerprint'],
                       'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
            extra = (0 if path.exists() else len(payload))
            extra += 0 if common.receipt_path(path).exists() else len(common.json_bytes(receipt))
            guard.check(extra)
            common.commit(path, value, binding, read_only)
            guard.check()

        guarded_commit(directory / 'report.json', report)
        names = {'binding.json', 'report.json', 'report.json.sha256.json'}
        completion = {'schema': 13, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
                      'feature_completion_sha256': feature_hash, 'report_sha256': common.digest(directory / 'report.json'),
                      'outputs': {name: common.digest(directory / name) for name in sorted(names)}}
        guarded_commit(directory / 'completion.json', completion)
        require(common.load_commit(directory / 'completion.json', binding) == completion, 'Evaluation completion differs')
        check_layout(directory, allowed, complete=True)
        return {'status': 'verified_completed_read_only' if read_only else 'completed',
                'new_geometry_computations': 0, 'new_model_forward_calls': 0,
                'decision': report['decision']['decision'], 'support_count': SUPPORT,
                'report_sha256': common.digest(directory / 'report.json'),
                'completion_sha256': common.digest(directory / 'completion.json'),
                'resources': guard.check()}

    if check_only or (directory / 'completion.json').exists():
        return publish_or_verify(True)
    # A partially committed intent is recovered by exact recomputation, never by
    # replacing an existing payload. Locks live outside immutable result storage.
    with common.lock(root / 'runtime/locks/soft_ranking_evaluation_v13'):
        return publish_or_verify((directory / 'completion.json').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(common.CONFIG))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(common.ROOT, args.config, args.check_only), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
