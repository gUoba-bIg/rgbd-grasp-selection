"""Independent acceptance of one exploratory, fixed-support soft ranking rule."""
import argparse
from fractions import Fraction
import hashlib
import io as byteio
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import soft_ranking_common_v13 as storage
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRICTION = (.2, .4, .6, .8, 1., 1.2)
COUNT, SUPPORT = 236, 50
FIGURE = 'scene_0003_0000_fixed_support_soft_ranking.png'
require = storage.require


def probability_entropy(logit):
    """Independent scalar computation; no producer/evaluator arithmetic imported."""
    require(math.isfinite(logit), 'Nonfinite foreground logit')
    if logit < 0:
        numerator = math.exp(logit)
        p = numerator / (1 + numerator)
    else:
        p = 1 / (1 + math.exp(-logit))
    h = 0.0 if p == 0 or p == 1 else -math.fsum((p * math.log2(p), (1-p) * math.log2(1-p)))
    require(-1e-15 <= h <= 1+1e-15, 'Entropy outside theoretical bounds')
    return p, max(0.0, min(1.0, h))


def low_coordinate(row, col, height, width, low_height, low_width):
    require(all(type(x) is int for x in (row, col, height, width, low_height, low_width))
            and 0 <= row < height and 0 <= col < width and min(low_height, low_width) > 0,
            'Invalid half-pixel coordinate')
    return (math.floor((row + .5) * low_height / height),
            math.floor((col + .5) * low_width / width))


def depth_observations(depth, row, col):
    require(depth.dtype == np.float64 and depth.ndim == 2, 'Expected float64 meter depth')
    height, width = depth.shape
    require(0 <= row < height and 0 <= col < width, 'Seed outside depth image')
    top, bottom, left, right = max(0, row-2), min(height, row+3), max(0, col-2), min(width, col+3)
    values = [float(depth[y, x]) for y in range(top, bottom) for x in range(left, right)]
    valid = [z for z in values if math.isfinite(z) and 0 < z <= 2]
    seed = float(depth[row, col])
    seed_valid = math.isfinite(seed) and 0 < seed <= 2
    c = math.fsum(max(0.0, 1.0-abs(z-seed)/.02) for z in valid)/len(valid) if valid and seed_valid else 0.0
    return {'zseed_m': seed if math.isfinite(seed) else None, 'seed_depth_valid': seed_valid,
            'patch_bounds_top_bottom_left_right_exclusive': [top, bottom, left, right],
            'patch_total_count': len(values), 'patch_valid_count': len(valid), 'v': len(valid)/len(values), 'c': c}


def quality_score(confidence, p, h, v, c):
    require(math.isfinite(confidence) and confidence >= 0 and
            all(math.isfinite(x) and 0 <= x <= 1 for x in (p, h, v, c)), 'Invalid quality inputs')
    q = (p * (1-h) + v*c) / 2
    multiplier = (1+q) / 2
    return {'q': q, 'multiplier': multiplier, 'soft_score': confidence * multiplier}


def independent_order(rows):
    require([row['original_index'] for row in rows] == list(range(len(rows))), 'Missing/reordered original feature rows')
    require(all(math.isfinite(row['soft_score']) and row['soft_score'] >= 0 for row in rows), 'Nonfinite score')
    return sorted(range(len(rows)), key=lambda index: (-rows[index]['soft_score'], index))


def compare_rows(expected, actual, tolerance):
    require(type(tolerance) is float and tolerance == 1e-12, 'Scalar tolerance differs from frozen rule')
    require(len(expected) == len(actual), 'Feature row count changed')
    maximum = 0.0
    for reference, observed in zip(expected, actual):
        require(set(reference) == set(observed), 'Feature row schema differs')
        for key, value in reference.items():
            other = observed[key]
            if type(value) is float:
                require(type(other) is float and math.isfinite(other), 'Nonfinite or mistyped scalar: ' + key)
                delta = abs(value-other)
                if key in ('raw_confidence', 'foreground_logit', 'zseed_m'):
                    require(value == other, 'Frozen source scalar differs: ' + key)
                else:
                    require(delta <= tolerance, 'Independent feature scalar differs: ' + key)
                maximum = max(maximum, delta)
            else:
                require(type(value) is type(other) and value == other, 'Independent feature identity/count differs: ' + key)
    return maximum


def load_array(path):
    return np.load(path, allow_pickle=False)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def verify_seed_chain(candidates, bundle, provenance, generation, association):
    """Reconstruct actual network sampling provenance, not legacy FP2 shortcuts."""
    require(candidates.dtype == np.float32 and candidates.shape == (COUNT, 17)
            and np.isfinite(candidates).all() and np.all(candidates[:, 0] >= 0), 'Candidate pool differs')
    require(np.array_equal(bundle['candidates'], candidates), 'Forward/export candidate difference')
    sa1, sa2, selected = [bundle[key] for key in ('sa1_indices', 'sa2_indices', 'selected_seed_indices')]
    require(sa1.dtype == sa2.dtype == selected.dtype == np.int64 and sa1.shape == (2048,)
            and sa2.shape == (1024,) and selected.shape == (COUNT,), 'Sampling chain shape/type differs')
    require(np.all((0 <= sa1) & (sa1 < 20000)) and np.all((0 <= sa2) & (sa2 < 2048))
            and np.all((0 <= selected) & (selected < 1024)) and np.all(np.diff(selected) > 0), 'Sampling chain index invalid')
    composed = sa1[sa2]
    positions = composed[selected]
    require(bundle['fp2_xyz'].dtype == np.float32 and bundle['fp2_xyz'].shape == (1024, 3)
            and np.array_equal(bundle['fp2_xyz'], generation['sampled_points'][composed])
            and np.array_equal(candidates[:, 13:16], generation['sampled_points'][positions]), 'Actual FP2 seed coordinates differ')
    expected = {key: value for key, value in bundle.items() if key != 'candidates'}
    expected.update(candidate_sampled_indices=positions,
                    candidate_valid_indices=generation['sampled_valid_indices'][positions],
                    candidate_pixel_indices=generation['sampled_pixel_indices'][positions])
    require(set(expected) == set(provenance) and all(expected[key].dtype == provenance[key].dtype
            and np.array_equal(value, provenance[key]) for key, value in expected.items()), 'Saved seed provenance differs')
    require(np.array_equal(generation['sampled_valid_indices'], association['sampled_indices'])
            and np.array_equal(generation['sampled_pixel_indices'], association['sampled_pixel_indices'])
            and np.array_equal(generation['sampled_points'], association['sampled_points']), 'R12 sample does not match frozen R11 sample')
    return positions, expected['candidate_valid_indices'], expected['candidate_pixel_indices'], selected


def reconstruct_features(root, protocol):
    path = lambda key: storage.safe_path(root, protocol[key])
    require(storage.digest(path('candidate_path')) == storage.digest(path('legacy_candidate_path')), 'Original candidate bytes changed')
    candidates = load_array(path('candidate_path'))
    generation_dir, association_dir = path('generation_dir'), path('association_dir')
    generation = {name: load_array(generation_dir / (name+'.npy')) for name in
                  ('sampled_points', 'sampled_valid_indices', 'sampled_pixel_indices')}
    association = {name: load_array(association_dir / (name+'.npy')) for name in
                   ('sampled_points', 'sampled_indices', 'sampled_pixel_indices', 'valid_points', 'valid_pixel_indices', 'valid_instance_ids')}
    positions, valid_indices, pixels, selected = verify_seed_chain(candidates, load_archive(generation_dir/'forward_bundle.npz'),
        load_archive(generation_dir/'seed_provenance.npz'), generation, association)
    source_rows = []
    for block in range(15):
        payload = storage.read(association_dir / 'blocks' / ('block_%04d.json' % block))
        require(payload['start'] == 16*block and payload['stop'] == min(COUNT, 16*(block+1)), 'R11 block boundaries differ')
        source_rows.extend(payload['rows'])
    require(len(source_rows) == COUNT, 'R11 source rows incomplete')
    heads = load_archive(path('heads_path'))
    foreground = heads['foreground']
    require(foreground.dtype == np.float32 and foreground.shape == (1, 1, 240, 320)
            and np.isfinite(foreground).all(), 'Foreground head differs')
    foreground = foreground[0, 0]
    instances = load_array(path('instances_path'))
    require(instances.dtype == np.int32 and instances.shape == (720, 1280), 'Instance frame differs')
    from scipy.io import loadmat
    # Read only calibration. This branch does not request object classes/poses.
    calibration = loadmat(path('meta_path'), variable_names=['factor_depth'])
    factor = float(np.asarray(calibration['factor_depth']).reshape(-1)[0])
    require(math.isfinite(factor) and factor > 0, 'Invalid depth calibration')
    with Image.open(path('depth_path')) as source:
        depth = np.asarray(source, dtype=np.float64) / factor
    require(depth.shape == (720, 1280), 'Depth resolution differs')
    rows = []
    for original, candidate in enumerate(candidates):
        position, valid_index, pixel, seed_index = (int(values[original]) for values in (positions, valid_indices, pixels, selected))
        require(0 <= valid_index < len(association['valid_points']) and 0 <= pixel < depth.size, 'Seed index out of bounds')
        source = source_rows[original]
        row, col = divmod(pixel, depth.shape[1])
        instance = int(instances[row, col])
        expected_match = {'sample_position': position, 'valid_point_index': valid_index, 'pixel_index': pixel,
                          'row': row, 'col': col, 'instance_id': instance}
        require(source['original_index'] == original and source['candidate_row'] == candidate.tolist()
                and source['match_kind'] == 'unique' and source['matches'] == [expected_match]
                and source['instance_id'] == instance and source['status'] == ('assigned' if instance > 0 else 'background'),
                'R11 unique source does not match actual R12 chain')
        require(int(association['valid_pixel_indices'][valid_index]) == pixel
                and int(association['valid_instance_ids'][valid_index]) == instance
                and np.array_equal(association['valid_points'][valid_index], candidate[13:16]), 'Valid-cloud source differs')
        require(np.flatnonzero(np.all(association['sampled_points'] == candidate[13:16], axis=1)).tolist() == [position],
                'Candidate source no longer unique')
        low_row, low_col = low_coordinate(row, col, 720, 1280, 240, 320)
        logit = float(foreground[low_row, low_col])
        p, h = probability_entropy(logit)
        local = depth_observations(depth, row, col)
        require(local['seed_depth_valid'] and np.float32(local['zseed_m']) == candidate[15], 'Candidate seed/depth mismatch')
        rows.append(dict(original_index=original, raw_confidence=float(candidate[0]), seed_pixel_index=pixel,
            seed_sample_position=position, seed_valid_point_index=valid_index, selected_seed_index=seed_index,
            seed_row=row, seed_col=col, low_row=low_row, low_col=low_col, foreground_logit=logit,
            p=p, h=h, instance_id_diagnostic_only=instance, **local,
            **quality_score(float(candidate[0]), p, h, local['v'], local['c'])))
    return candidates, rows, independent_order(rows)


def fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'display_value': float(value)}


def prefix_metrics(order, labels):
    require(len(order) == SUPPORT and len(set(order)) == SUPPORT and set(order) == set(labels), 'Fixed support changed')
    counts, fractions, cumulative = [], [], [0]*6
    total = Fraction(0, 1)
    for k, original in enumerate(order, 1):
        score = labels[original]
        require(score in (-1.,)+FRICTION, 'Invalid frozen geometry score')
        for j, mu in enumerate(FRICTION):
            cumulative[j] += int(0 < score <= mu)
        counts.append(cumulative.copy())
        fractions.append([value/k for value in cumulative])
        total += sum((Fraction(value, k) for value in cumulative), Fraction(0, 1))
    return {'counts_50x6': counts, 'fraction_50x6': fractions, 'top10_mu04_count': counts[9][1],
            'top50_counts_by_friction': counts[-1], 'mean_exact': fraction_record(total/300)}


def rational(record):
    require(type(record['numerator']) is int and type(record['denominator']) is int and record['denominator'] > 0,
            'Invalid exact fraction')
    value = Fraction(record['numerator'], record['denominator'])
    require(record == fraction_record(value), 'Unreduced/inconsistent fraction record')
    return value


def effect_decision(baseline, soft):
    require(baseline['top50_counts_by_friction'] == soft['top50_counts_by_friction'], 'Reordering changed full-support labels')
    difference = soft['top10_mu04_count']-baseline['top10_mu04_count']
    mean_difference = rational(soft['mean_exact'])-rational(baseline['mean_exact'])
    positive = difference > 0 and mean_difference >= 0
    return {'decision': 'positive_exploratory_fixed_support_ranking' if positive else 'valid_negative_do_not_adopt',
            'positive_effect_gate_passed': positive, 'top10_strict_increase': difference > 0,
            'exact_mean_non_decrease': mean_difference >= 0, 'all_friction_top50_conserved': True,
            'top10_mu04_count_difference': difference, 'mean_exact_difference': fraction_record(mean_difference),
            'dataset_ap_claimed': False, 'statistical_significance_claimed': False}


def fixed_support(candidates, preparation, geometry_report, records, hashes, full_order):
    require(preparation['input_count'] == COUNT and preparation['input_dtype'] == 'float32'
            and preparation['selected_count'] == SUPPORT and len(preparation['selected']) == len(records) == len(hashes) == SUPPORT,
            'Fixed support must be exactly fifty original candidates')
    require(len(full_order) == COUNT and set(full_order) == set(range(COUNT)), 'Global rank is not a permutation')
    trace, originals = [], []
    for index, (selected, record, checksum) in enumerate(zip(preparation['selected'], records, hashes)):
        original = selected['original_index']
        require(type(original) is int and 0 <= original < COUNT and selected['selection_index'] == index,
                'Invalid official selection identity')
        require(all(record.get(key) == value for key, value in selected.items()), 'Geometry selection identity changed')
        candidate = candidates[original].copy()
        candidate[1] = min(np.float32(.1), max(np.float32(0), candidate[1]))
        require(record['grasp'] == candidate.tolist(), 'Scored row is not the frozen width-clipped candidate')
        score, collision = record['score'], record['collision']
        require(score in (-1.,)+FRICTION and type(collision) is bool and (not collision or score == -1.), 'Invalid collision/score label')
        originals.append(original)
        trace.append({'selection_index': index, 'original_index': original, 'score': score,
                      'collision': collision, 'grasp_record_sha256': checksum})
    require(len(set(originals)) == SUPPORT, 'Repeated official support original ID')
    require(len(set(float(candidates[index, 0]) for index in originals)) == SUPPORT, 'Baseline confidence tie is out of protocol')
    baseline = sorted(originals, key=lambda original: -float(candidates[original, 0]))
    by_id = {record['original_index']: record for record in records}
    require(geometry_report['ordered_records'] == [by_id[index] for index in baseline]
            and geometry_report['original_candidate_count'] == COUNT and geometry_report['selected_count'] == SUPPORT,
            'Official baseline ordering differs')
    soft = [original for original in full_order if original in set(originals)]
    labels = {item['original_index']: item['score'] for item in trace}
    metrics = {'baseline': prefix_metrics(baseline, labels), 'soft': prefix_metrics(soft, labels)}
    require(geometry_report['single_frame_prefix_pass_fraction_50x6'] == metrics['baseline']['fraction_50x6'],
            'Old official prefix matrix differs')
    return originals, baseline, soft, trace, metrics, effect_decision(metrics['baseline'], metrics['soft'])


def verify_inventory(directory, expected):
    files = set()
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'Symlink in immutable result inventory')
        if path.is_file():
            files.add(path.relative_to(directory).as_posix())
        else:
            require(path.is_dir() and any(name.startswith(path.relative_to(directory).as_posix()+'/') for name in expected),
                    'Unexpected result directory')
    require(files == set(expected), 'Immutable output inventory differs')


def verify_manifest(directory, manifest, names):
    require(set(manifest) == set(names), 'Sealed output membership differs')
    require(all(storage.digest(storage.safe_path(directory, name)) == manifest[name] for name in names),
            'Sealed output hash changed')


def verify_new_seal(directory, binding, phase):
    require(storage.read(directory/'binding.json') == binding, 'Source-bound stage identity differs')
    completion = storage.load_commit(directory/'completion.json', binding)
    require(completion['schema'] == 13 and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == binding['fingerprint'], 'Stage is not completely sealed')
    names = {'binding.json', 'report.json', 'report.json.sha256.json'}
    if phase == 'features':
        for block in range(15):
            name = 'blocks/block_%04d.json' % block
            names.update((name, name+'.sha256.json'))
        require(completion['candidate_count'] == COUNT and completion['block_count'] == 15
                and completion['model_forward_calls'] == 0, 'Feature completion counters differ')
    verify_manifest(directory, completion['outputs'], names)
    verify_inventory(directory, names | {'completion.json', 'completion.json.sha256.json'})
    report = storage.load_commit(directory/'report.json', binding)
    require(completion['report_sha256'] == storage.digest(directory/'report.json'), 'Completion report hash differs')
    return report, completion


def verify_feature_report(directory, report, binding, protocol):
    rows = []
    for block in range(15):
        value = storage.load_commit(directory / ('blocks/block_%04d.json' % block), binding)
        start, stop = 16*block, min(COUNT, 16*(block+1))
        require(value == {'schema': 13, 'binding_fingerprint': binding['fingerprint'], 'start': start, 'stop': stop,
                          'rows': report['rows'][start:stop]}, 'Feature block/report partition differs')
        rows.extend(value['rows'])
    expected = {'schema': 13, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
        'protocol_sha256': binding['protocol_sha256'], 'candidate_count': COUNT, 'block_count': 15,
        'candidate_sha256': protocol['feature_inputs'][protocol['candidate_path']],
        'feature_inputs': protocol['feature_inputs'], 'source_hashes': protocol['source_hashes'],
        'feature_definition': protocol['feature_definition'], 'rows': rows,
        'rank_original_indices': independent_order(rows), 'model_forward_calls': 0, 'training_steps': 0,
        'geometry_labels_read': False, 'calibrated_risk_probability': False, 'limitations': protocol['limitations']}
    require(report == expected, 'Feature report claims/content differ')


def verify_resources(record, protocol):
    evidence, limits = record['resources'], protocol['resources']
    for key, maximum in (('peak_rss_bytes', limits['rss_soft_bytes']),
                         ('elapsed_seconds', limits['invocation_soft_seconds']),
                         ('current_output_bytes', limits['output_limit_bytes'])):
        value = evidence[key]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum,
                'Recorded child resource bound exceeded: ' + key)


def verify_phase_proof(root, config, protocol, phase, directory):
    name = 'feature_resume_report' if phase == 'features' else 'resume_report'
    proof = storage.read(storage.safe_path(root, protocol['outputs'][name]))
    child = 'extract_soft_ranking_features_v13.py' if phase == 'features' else 'evaluate_soft_ranking_v13.py'
    hashes = {'protocol_sha256': storage.digest(config),
        'wrapper_sha256': storage.digest(root/'scripts/run_soft_ranking_v13.py'),
        'child_sha256': storage.digest(root/'scripts'/child),
        'common_sha256': storage.digest(root/'scripts/soft_ranking_common_v13.py'),
        'pipeline_sha256': storage.digest(root/'scripts/pipeline.py'),
        'interpreter_sha256': storage.digest(storage.safe_path(root, protocol['interpreter']))}
    require(proof['schema'] == 13 and proof['phase'] == phase and proof['status'] == 'passed'
            and proof['binding'] == hashes and proof['read_only_check_passed'] is True, 'Recovery proof identity/status differs')
    current = storage.snapshot(directory)
    require(proof['completed_snapshot'] == current and proof['completion_sha256'] == storage.digest(directory/'completion.json')
            and proof['report_sha256'] == storage.digest(directory/'report.json'), 'Recovery completed snapshot changed')
    final, readonly = proof['completion_record'], proof['read_only_record']
    require(final['status'] in ('completed', 'verified_completed_read_only') and readonly['status'] == 'verified_completed_read_only',
            'Missing completed/read-only child evidence')
    verify_resources(final, protocol)
    verify_resources(readonly, protocol)
    if phase == 'features':
        prefix, after, probe = proof['prefix_before'], proof['prefix_after'], proof['probe_record']
        require(prefix == after and proof['byte_and_mtime_identical'] is True
                and all(current.get(name) == value for name, value in prefix.items()), 'Durable feature prefix changed')
        expected = {'binding.json'}
        for block in range(2):
            path = 'blocks/block_%04d.json' % block
            expected.update((path, path+'.sha256.json'))
        require(set(prefix) == expected, 'Recovery probe did not seal exactly two blocks')
        for record, total in ((probe, 2), (final, 15), (readonly, 15)):
            require(all(type(record[key]) is int and record[key] >= 0 for key in
                    ('new_blocks', 'reused_blocks', 'committed_blocks', 'expected_blocks', 'new_feature_rows'))
                    and record['expected_blocks'] == 15 and record['committed_blocks'] == total
                    and record['new_blocks'] + record['reused_blocks'] == total, 'Feature block accounting differs')
        require(probe['status'] == 'partial' and probe['new_feature_rows'] == probe['new_blocks']*16,
                'Missing two-block process boundary')
        require(final['reused_blocks'] >= 2 and final['new_blocks'] <= 13
                and final['new_feature_rows'] == (0 if final['new_blocks'] == 0 else final['new_blocks']*16-4),
                'Resume did not preserve the two-block prefix')
        require(readonly['new_blocks'] == readonly['new_feature_rows'] == 0 and readonly['reused_blocks'] == 15,
                'Read-only feature invocation recomputed rows')
        verify_resources(probe, protocol)
    else:
        require(proof['feature_resume_sha256'] == storage.digest(storage.safe_path(root, protocol['outputs']['feature_resume_report'])),
                'Evaluation did not bind the passed feature recovery proof')
        for record in (final, readonly):
            require(record['new_geometry_computations'] == record['new_model_forward_calls'] == 0,
                    'Evaluation reran model or geometry')
            require(record['completion_sha256'] == proof['completion_sha256'] and record['report_sha256'] == proof['report_sha256'],
                    'Evaluation child evidence has different seals')
    return proof


def read_old_geometry(root, protocol):
    directory = storage.safe_path(root, protocol['geometry_dir'])
    binding = storage.read(directory/'binding.json')
    binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    completion = storage.read(directory/'completion.json')
    require(binding['schema'] == completion['schema'] == 12 and binding['arm'] == completion['arm'] == 'full_scene'
            and completion['status'] == 'completed' and completion['binding_fingerprint'] == binding_hash
            and completion['expected_grasps'] == SUPPORT, 'Legacy geometry seal differs')
    names = {'binding.json', 'preparation.json', 'preparation.sha256.json', 'oracle.json', 'oracle.sha256.json',
             'report.json', 'last_invocation.json'}
    for index in range(SUPPORT):
        names.update(('grasps/grasp_%04d.json' % index, 'grasps/grasp_%04d.sha256.json' % index))
    verify_manifest(directory, completion['outputs'], names)
    verify_inventory(directory, names | {'completion.json'})
    require(all((directory/name).relative_to(root).as_posix() in protocol['evaluation_inputs'] for name in names | {'completion.json'}),
            'An old geometry artifact is not frozen by protocol')

    def committed(name):
        path = directory/name
        require(storage.read(path.with_suffix('.sha256.json')) == {'sha256': storage.digest(path), 'binding_fingerprint': binding_hash},
                'Legacy score receipt differs')
        return storage.read(path)

    preparation = committed('preparation.json')
    require(preparation['schema'] == 12 and preparation['arm'] == 'full_scene', 'Legacy support source differs')
    records = [committed('grasps/grasp_%04d.json' % index) for index in range(SUPPORT)]
    hashes = [storage.digest(directory/('grasps/grasp_%04d.json' % index)) for index in range(SUPPORT)]
    oracle = committed('oracle.json')
    expected = {'groups': [], 'scores': [], 'collision': []}
    for slot in range(len(preparation['object_ids'])):
        group = [record for record in records if record['object_slot'] == slot]
        expected['groups'].append([record['grasp'] for record in group])
        expected['scores'].append([record['score'] for record in group])
        expected['collision'].append([record['collision'] for record in group])
    require(oracle == {'executed': True, 'result': expected}, 'Cached official oracle labels differ')
    report = storage.read(directory/'report.json')
    require(report['schema'] == 12 and report['arm'] == 'full_scene' and report['binding_fingerprint'] == binding_hash
            and report['official_oracle_executed'] is True and report['official_oracle_exact_match'] is True
            and report['friction_grid_ascending'] == list(FRICTION), 'Legacy geometry report provenance differs')
    source = {'source_geometry_'+key+'_sha256': storage.digest(directory/(key+'.json'))
              for key in ('binding', 'completion', 'preparation', 'oracle', 'report')}
    return preparation, report, records, hashes, source


def scientific_png(metrics, decision):
    """A deterministic scientific plot; no generated or altered scene imagery."""
    canvas = Image.new('RGB', (1600, 900), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    title_font, text_font, small = (ImageFont.load_default(size=size) for size in (29, 22, 18))
    draw.text((52, 32), 'Fixed-support soft ranking | scene 0003 / frame 0000', fill='#172634', font=title_font)
    draw.text((52, 80), 'Existing training frame. Same 50 officially selected candidates. One frozen rule.', fill='#455462', font=text_font)
    x0, y0, x1, y1 = 90, 180, 850, 680
    for tick in range(6):
        y = y1-(y1-y0)*tick/5
        draw.line((x0, y, x1, y), fill='#dce2e6', width=1)
        draw.text((40, y-10), '%.1f' % (tick/5), fill='#455462', font=small)
    draw.line((x0, y0, x0, y1, x1, y1), fill='#455462', width=2)
    for tick in (1, 10, 20, 30, 40, 50):
        x = x0+(tick-1)*(x1-x0)/49
        draw.text((x-10, y1+12), str(tick), fill='#455462', font=small)
    draw.text((310, 733), 'Prefix size K (same support)', fill='#172634', font=text_font)
    draw.text((90, 140), 'Pass fraction at friction coefficient 0.4', fill='#172634', font=text_font)
    for index, (key, colour, label) in enumerate((('baseline', '#2868ad', 'Original confidence'), ('soft', '#c26725', 'Frozen soft score'))):
        values = [row[1] for row in metrics[key]['fraction_50x6']]
        points = [(x0+i*(x1-x0)/49, y1-value*(y1-y0)) for i, value in enumerate(values)]
        draw.line(points, fill=colour, width=4)
        draw.line((95, 785+index*30, 135, 785+index*30), fill=colour, width=4)
        draw.text((145, 772+index*30), label, fill=colour, font=small)
    draw.line((900, 145, 900, 835), fill='#dce2e6', width=2)
    baseline, soft = metrics['baseline'], metrics['soft']
    rows = [
        ('Predeclared bounded effect gates', '#172634'),
        ('Top10 passing count at friction 0.4', '#455462'),
        ('%d -> %d   (%+d)' % (baseline['top10_mu04_count'], soft['top10_mu04_count'], decision['top10_mu04_count_difference']), '#172634'),
        ('Strict increase: '+str(decision['top10_strict_increase']), '#172634'),
        ('Exact mean of 50 x 6 prefix fractions', '#455462'),
        ('%.8f -> %.8f' % (baseline['mean_exact']['display_value'], soft['mean_exact']['display_value']), '#172634'),
        ('Exact non-decrease: '+str(decision['exact_mean_non_decrease']), '#172634'),
        ('All six top50 counts conserved: True', '#172634'),
        ('Positive exploratory gate: '+str(decision['positive_effect_gate_passed']), '#172634'),
        ('No adoption' if not decision['positive_effect_gate_passed'] else 'Exploratory signal only', '#c26725'),
        ('Not dataset AP or generalization evidence.', '#455462'),
        ('Support was previously selected using', '#455462'),
        ('official object geometry; labels are reused.', '#455462'),
        ('Entropy/depth are heuristic observations,', '#455462'),
        ('not calibrated risk or robot success.', '#455462')]
    for index, (text, colour) in enumerate(rows):
        draw.text((940, 160+index*43), text, fill=colour, font=text_font if index < 10 else small)
    buffer = byteio.BytesIO()
    canvas.save(buffer, format='PNG', optimize=False)
    return buffer.getvalue()


def validate_protocol(protocol):
    require(protocol['schema'] == 13 and protocol['candidate_count'] == COUNT and protocol['support_count'] == SUPPORT
            and protocol['block_size'] == 16 and protocol['independent_scalar_abs_tolerance'] == 1e-12,
            'Frozen experiment dimensions differ')
    require(protocol['frame'] == {'scene': 3, 'camera': 'realsense', 'frame': '0000'}, 'Out-of-scope frame')
    definition, evaluation = protocol['feature_definition'], protocol['evaluation']
    require(definition['roi_radius_pixels'] == 2 and definition['maximum_depth_m'] == 2
            and definition['maximum_depth_inclusive'] is True and definition['depth_scale_m'] == .02
            and definition['quality'] == '(p*(1-h)+v*c)/2' and definition['soft_multiplier'] == '0.5+0.5*quality'
            and definition['instance_id_used_in_score'] is False, 'Feature rule changed')
    require(evaluation['friction_grid'] == list(FRICTION) and evaluation['primary_top_k'] == 10
            and evaluation['primary_friction'] == .4 and evaluation['primary_rule'] == 'strict integer increase'
            and evaluation['positive_decision'] == 'positive_exploratory_fixed_support_ranking'
            and evaluation['negative_decision'] == 'valid_negative_do_not_adopt'
            and evaluation['calibrated_risk_probability'] is False and evaluation['dataset_ap'] is False
            and evaluation['independent_confirmation'] is False, 'Effect interpretation changed')
    geometry_prefix = protocol['geometry_dir'].rstrip('/')+'/'
    require(not any(name.startswith(geometry_prefix) for name in protocol['feature_inputs'])
            and set(protocol['feature_inputs']).isdisjoint(protocol['evaluation_inputs']), 'Labels entered feature input group')
    source_names = {'scripts/extract_soft_ranking_features_v13.py', 'scripts/evaluate_soft_ranking_v13.py',
                    'scripts/accept_soft_ranking_v13.py', 'scripts/soft_ranking_common_v13.py',
                    'scripts/run_soft_ranking_v13.py', 'scripts/pipeline.py'}
    require(source_names.issubset(protocol['source_hashes']), 'Incomplete frozen source implementation')
    require(protocol['resources'] == {'cpu_threads': 2, 'rss_soft_bytes': 4*1024**3, 'invocation_soft_seconds': 300,
            'output_limit_bytes': 64*1024**2, 'disk_reserve_bytes': 60*1024**3}, 'Resource contract differs')
    require(set(protocol['outputs'].values()).issubset(set(protocol['budget_paths']) | {
            'results/P3/grasp_soft_ranking_v13/features', 'results/P3/grasp_soft_ranking_v13/evaluation'}), 'Unbudgeted output path')


def run(root=ROOT, config=None, check_only=False):
    root = Path(root).resolve()
    config = Path(config).resolve() if config else root/'configs/grasp_soft_ranking_v13.json'
    require(config.is_relative_to(root), 'Configuration outside project')
    protocol = storage.read(config)
    validate_protocol(protocol)
    guard = storage.Guard(root, protocol)
    guard.check()
    with storage.lock(root/'runtime/locks/soft_ranking_acceptance_v13'):
        return locked_acceptance(root, config, protocol, guard, check_only)


def locked_acceptance(root, config, protocol, guard, check_only):
    outputs = {key: storage.safe_path(root, name) for key, name in protocol['outputs'].items()}
    feature_binding = storage.make_binding(root, config, protocol, 'feature_inputs')
    features, feature_completion = verify_new_seal(outputs['features_dir'], feature_binding, 'features')
    verify_feature_report(outputs['features_dir'], features, feature_binding, protocol)
    feature_snapshot = storage.snapshot(outputs['features_dir'])
    feature_proof = verify_phase_proof(root, config, protocol, 'features', outputs['features_dir'])
    candidates, independent_rows, independent_rank = reconstruct_features(root, protocol)
    difference = compare_rows(independent_rows, features['rows'], protocol['independent_scalar_abs_tolerance'])
    require(independent_rank == features['rank_original_indices'], 'Independent global soft ranking differs')
    require(storage.snapshot(outputs['features_dir']) == feature_snapshot, 'Feature seal changed during independent reconstruction')
    guard.check()
    # Only here may any cached geometry labels be hashed or decoded. No feature
    # scalar or ordering is adjusted in response to the following diagnostics.
    evaluation_binding = storage.make_binding(root, config, protocol, 'evaluation_inputs')
    preparation, geometry_report, records, hashes, source = read_old_geometry(root, protocol)
    originals, baseline, soft, trace, metrics, decision = fixed_support(candidates, preparation, geometry_report,
                                                                       records, hashes, independent_rank)
    evaluation, completion = verify_new_seal(outputs['evaluation_dir'], evaluation_binding, 'evaluation')
    evaluation_snapshot = storage.snapshot(outputs['evaluation_dir'])
    evaluation_proof = verify_phase_proof(root, config, protocol, 'evaluation', outputs['evaluation_dir'])
    feature_hash = storage.digest(outputs['features_dir']/'completion.json')
    require(completion['feature_completion_sha256'] == feature_hash, 'Evaluation feature seal changed')
    scores = [row['soft_score'] for row in features['rows']]
    support = dict(candidate_count=COUNT, selected_count=SUPPORT, original_indices_by_selection=originals,
        baseline_original_indices=baseline, soft_original_indices=soft, baseline_confidence_ties=0,
        full_soft_duplicate_score_count=len(scores)-len(set(scores)),
        support_soft_duplicate_score_count=SUPPORT-len(set(scores[index] for index in originals)),
        soft_tie_policy='descending soft_score then ascending original_index', **source)
    expected_evaluation = {'schema': 13, 'status': 'completed_common_support_diagnostic',
        'binding_fingerprint': evaluation_binding['fingerprint'], 'feature_binding_fingerprint': feature_binding['fingerprint'],
        'feature_completion_sha256': feature_hash, 'protocol_sha256': storage.digest(config), 'support': support,
        'label_trace': trace, 'metrics': metrics, 'decision': decision, 'friction_grid': list(FRICTION),
        'new_geometry_computations': 0, 'new_model_forward_calls': 0, 'new_training_steps': 0,
        'feature_sealed_before_label_access': True, 'new_nms_or_support_selection': False,
        'dataset_ap_computed': False, 'independent_confirmation': False, 'limitations': protocol['limitations']}
    require(evaluation == expected_evaluation, 'Independent support/labels/matrices/exact decision differ from evaluator')
    figure = scientific_png(metrics, decision)
    report = {'schema': 13, 'status': 'accepted_exploratory_fixed_support_engineering',
        'protocol_sha256': storage.digest(config), 'acceptor_sha256': storage.digest(Path(__file__)),
        'feature_binding_fingerprint': feature_binding['fingerprint'], 'evaluation_binding_fingerprint': evaluation_binding['fingerprint'],
        'feature_completion_sha256': feature_hash, 'evaluation_completion_sha256': storage.digest(outputs['evaluation_dir']/'completion.json'),
        'feature_resume_sha256': storage.digest(outputs['feature_resume_report']), 'evaluation_resume_sha256': storage.digest(outputs['resume_report']),
        'candidate_sha256': storage.digest(storage.safe_path(root, protocol['candidate_path'])),
        'support': support, 'metrics': metrics, 'decision': decision,
        'independent_maximum_scalar_absolute_difference': difference,
        'engineering_checks': {'all_236_candidate_bytes_unchanged': True, 'actual_seed_chain_and_unique_source_crosschecked': True,
            'all_feature_rows_independently_reconstructed': True, 'all_236_rank_positions_exact': True,
            'features_verified_before_cached_label_access': True, 'fixed_50_support_and_label_join_exact': True,
            'all_600_prefix_counts_and_fractions_exact': True, 'rational_effect_decision_exact': True,
            'two_block_prefix_bytes_and_mtime_unchanged': True, 'both_stage_readonly_snapshots_unchanged': True},
        'recovery_evidence': {'feature_probe': feature_proof['probe_record'], 'feature_completion': feature_proof['completion_record'],
            'feature_read_only': feature_proof['read_only_record'], 'evaluation_read_only': evaluation_proof['read_only_record'],
            'scope': 'Process-boundary recovery; not a physical power-interruption test'},
        'label_access_evidence': 'Frozen source review, disjoint input groups, feature seals, and ordered independent reconstruction; not operating-system access tracing',
        'new_model_forward_calls': 0, 'new_training_steps': 0, 'new_geometry_computations': 0,
        'is_dataset_ap_result': False, 'independent_confirmation': False,
        'visual': {'file': protocol['outputs']['visuals_dir']+'/'+FIGURE, 'sha256': hashlib.sha256(figure).hexdigest()},
        'limitations': protocol['limitations']}
    # Child resource observations remain in the frozen wrapper evidence. They are
    # referenced by hash, not copied into this immutable scientific report.
    for value in report['recovery_evidence'].values():
        if isinstance(value, dict):
            value.pop('resources', None)
    for group in ('feature_inputs', 'evaluation_inputs', 'source_hashes'):
        require(all(storage.digest(storage.safe_path(root, name)) == expected for name, expected in protocol[group].items()),
                'Frozen source/input changed during acceptance')
    require(feature_snapshot == storage.snapshot(outputs['features_dir']) and evaluation_snapshot == storage.snapshot(outputs['evaluation_dir']),
            'Read-only acceptance changed sealed stage artifacts')
    payload = storage.json_bytes(report)
    figure_path = outputs['visuals_dir']/FIGURE
    if outputs['visuals_dir'].exists():
        for path in outputs['visuals_dir'].rglob('*'):
            require(path.is_file() and not path.is_symlink() and (path == figure_path or
                re.fullmatch(re.escape(FIGURE)+r'\.[0-9a-f]{32}\.tmp', path.name)), 'Unexpected acceptance visual output')
    extra = (0 if outputs['acceptance_report'].exists() else len(payload)) + (0 if figure_path.exists() else len(figure))
    guard.check(extra)
    storage.publish_once(figure_path, figure, check_only)
    guard.check(0 if outputs['acceptance_report'].exists() else len(payload))
    storage.publish_once(outputs['acceptance_report'], payload, check_only)
    resources = guard.check()
    return {'status': 'verified_completed_read_only' if check_only else 'passed', 'decision': decision['decision'],
            'report_sha256': storage.digest(outputs['acceptance_report']), 'figure_sha256': storage.digest(figure_path),
            'new_model_forward_calls': 0, 'new_geometry_computations': 0, 'resources': resources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/grasp_soft_ranking_v13.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.config, args.check_only), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
