"""Label-free local observations for one frozen candidate pool; no model inference."""
import argparse
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
import numpy as np
from PIL import Image
from scipy.io import loadmat

if __package__:
    from . import soft_ranking_common_v13 as common
else:
    import soft_ranking_common_v13 as common

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/grasp_soft_ranking_v13.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def stable_foreground(logit):
    """Stable float64 sigmoid and normalized binary entropy, not calibrated risk."""
    value = float(logit)
    require(math.isfinite(value), 'Foreground logit must be finite')
    if value >= 0:
        probability = 1. / (1. + math.exp(-value))
    else:
        exponent = math.exp(value)
        probability = exponent / (1. + exponent)
    entropy = 0. if probability in (0., 1.) else -(
        probability * math.log(probability) + (1. - probability) * math.log1p(-probability)) / math.log(2.)
    require(math.isfinite(entropy) and -1e-15 <= entropy <= 1. + 1e-15, 'Invalid binary entropy')
    return probability, min(1., max(0., entropy))


def nearest_low_pixel(row, col, original_shape, low_shape):
    require(len(original_shape) == len(low_shape) == 2 and all(type(n) is int and n > 0 for n in (*original_shape, *low_shape)),
            'Invalid image dimensions')
    require(type(row) is int and type(col) is int and 0 <= row < original_shape[0] and 0 <= col < original_shape[1],
            'Seed pixel out of bounds')
    return ((2 * row + 1) * low_shape[0] // (2 * original_shape[0]),
            (2 * col + 1) * low_shape[1] // (2 * original_shape[1]))


def local_depth_features(depth_m, row, col):
    """Clipped 5x5 observed-depth continuity, inclusive of the seed pixel."""
    require(isinstance(depth_m, np.ndarray) and depth_m.ndim == 2 and depth_m.dtype == np.float64, 'Depth must be a 2D float64 meter array')
    height, width = depth_m.shape
    require(type(row) is int and type(col) is int and 0 <= row < height and 0 <= col < width, 'Seed pixel out of bounds')
    top, bottom, left, right = max(0, row - 2), min(height, row + 3), max(0, col - 2), min(width, col + 3)
    patch = depth_m[top:bottom, left:right]
    valid = np.isfinite(patch) & (patch > 0) & (patch <= 2.)
    count = int(np.count_nonzero(valid))
    total = int(patch.size)
    seed = float(depth_m[row, col])
    seed_valid = math.isfinite(seed) and 0 < seed <= 2.
    # A valid candidate seed guarantees at least its own observation. The explicit
    # empty fallback remains tested for standalone diagnostics and corrupted input.
    continuity = float(np.maximum(0., 1. - np.abs(patch[valid] - seed) / .02).mean()) if count and seed_valid else 0.
    return {'zseed_m': seed if math.isfinite(seed) else None, 'seed_depth_valid': seed_valid,
        'patch_bounds_top_bottom_left_right_exclusive': [top, bottom, left, right],
        'patch_total_count': total, 'patch_valid_count': count, 'v': count / total, 'c': continuity}


def combine_features(raw_confidence, probability, entropy, valid_fraction, continuity):
    values = [float(value) for value in (raw_confidence, probability, entropy, valid_fraction, continuity)]
    require(all(math.isfinite(value) for value in values) and values[0] >= 0, 'Invalid feature/confidence value')
    require(all(0. <= value <= 1. for value in values[1:]), 'Feature outside unit interval')
    confidence, p, h, v, c = values
    quality = (p * (1. - h) + v * c) / 2.
    multiplier = .5 + .5 * quality
    score = confidence * multiplier
    require(math.isfinite(score) and .5 <= multiplier <= 1. and 0 <= quality <= 1., 'Invalid soft ranking arithmetic')
    return {'q': quality, 'multiplier': multiplier, 'soft_score': score}


def candidate_feature(original_index, candidate, source_row, arrays, foreground_logits, depth_m, actual_seed):
    require(type(original_index) is int and original_index >= 0 and candidate.dtype == np.float32 and candidate.shape == (17,)
            and np.isfinite(candidate).all(), 'Malformed original candidate')
    require(source_row['original_index'] == original_index and source_row['candidate_row'] == candidate.tolist(), 'Association candidate identity differs')
    matches = source_row['matches']
    require(len(matches) == 1 and source_row['match_kind'] == 'unique', 'Candidate must have exactly one exact sampled source')
    source = matches[0]
    position, valid_index, pixel = (actual_seed[key] for key in ('sample_position', 'valid_point_index', 'pixel_index'))
    require(all(source[key] == actual_seed[key] for key in ('sample_position', 'valid_point_index', 'pixel_index')),
            'Actual R12 seed chain differs from unique R11 exact source')
    require(all(type(value) is int for value in (position, valid_index, pixel)), 'Noninteger source indices')
    require(0 <= position < len(arrays['sampled_indices']) and 0 <= valid_index < len(arrays['valid_points']), 'Invalid source index')
    require(valid_index == int(arrays['sampled_indices'][position]) and pixel == int(arrays['valid_pixel_indices'][valid_index])
            and pixel == int(arrays['sampled_pixel_indices'][position]), 'Source index chain differs')
    require(np.array_equal(candidate[13:16], arrays['valid_points'][valid_index])
            and np.array_equal(candidate[13:16], arrays['sampled_points'][position]), 'Seed XYZ does not match original sampled source')
    # Confirm uniqueness from the sealed point sample, not only a report field.
    exact_positions = np.flatnonzero(np.all(arrays['sampled_points'] == candidate[13:16], axis=1))
    require(exact_positions.tolist() == [position], 'Exact source is absent or ambiguous')
    require(depth_m.shape == (720, 1280) and foreground_logits.dtype == np.float32 and foreground_logits.shape == (240, 320)
            and np.isfinite(foreground_logits).all(), 'Frozen frame/head shape differs')
    row, col = divmod(pixel, depth_m.shape[1])
    require(source['row'] == row and source['col'] == col, 'Source pixel coordinates differ')
    expected_instance = int(arrays['valid_instance_ids'][valid_index])
    require(source['instance_id'] == expected_instance and source_row['instance_id'] == expected_instance,
            'Predicted instance diagnostic differs')
    low_row, low_col = nearest_low_pixel(row, col, tuple(depth_m.shape), tuple(foreground_logits.shape))
    logit = float(foreground_logits[low_row, low_col])
    p, h = stable_foreground(logit)
    local = local_depth_features(depth_m, row, col)
    require(local['seed_depth_valid'] and np.float32(local['zseed_m']) == candidate[15], 'Raw seed depth does not match frozen candidate')
    score = combine_features(float(candidate[0]), p, h, local['v'], local['c'])
    return dict(original_index=original_index, raw_confidence=float(candidate[0]), seed_pixel_index=pixel,
        seed_sample_position=position, seed_valid_point_index=valid_index,
        selected_seed_index=actual_seed['selected_seed_index'],
        seed_row=row, seed_col=col, low_row=low_row, low_col=low_col, foreground_logit=logit,
        p=p, h=h, instance_id_diagnostic_only=expected_instance, **local, **score)


def rank_rows(rows):
    require(isinstance(rows, list) and [row['original_index'] for row in rows] == list(range(len(rows))), 'Rows must retain every original index in original order')
    require(all(math.isfinite(row['soft_score']) and row['soft_score'] >= 0 for row in rows), 'Invalid ranking score')
    return [row['original_index'] for row in sorted(rows, key=lambda row: (-row['soft_score'], row['original_index']))]


def validate_actual_seed_chain(candidates, provenance, generation, association):
    """Follow saved network indices, never infer their origin by a nearest search."""
    sa1, sa2, selected = (provenance[key] for key in ('sa1_indices', 'sa2_indices', 'selected_seed_indices'))
    fp2 = provenance['fp2_xyz']
    require(sa1.dtype == sa2.dtype == selected.dtype == np.int64 and sa1.shape == (2048,)
            and sa2.shape == (1024,) and selected.shape == (len(candidates),), 'Invalid actual SA index arrays')
    require(fp2.dtype == np.float32 and fp2.shape == (1024, 3), 'Invalid actual FP2 coordinates')
    sample = generation['sampled_points']
    require(sample.dtype == np.float32 and sample.ndim == 2 and sample.shape[1] == 3
            and np.isfinite(sample).all(), 'Invalid original network input sample')
    require(np.all((sa1 >= 0) & (sa1 < len(sample))) and np.all((sa2 >= 0) & (sa2 < len(sa1)))
            and np.all((selected >= 0) & (selected < len(sa2))) and np.all(np.diff(selected) > 0),
            'Actual SA/selected index out of range or not ordered')
    composed = sa1[sa2]
    require(np.array_equal(sample[composed], fp2) and np.array_equal(candidates[:, 13:16], fp2[selected]),
            'Actual SA composition does not reproduce FP2 and candidate XYZ')
    position = composed[selected]
    for key, expected in (('candidate_sampled_indices', position),
                          ('candidate_valid_indices', generation['sampled_valid_indices'][position]),
                          ('candidate_pixel_indices', generation['sampled_pixel_indices'][position])):
        require(provenance[key].dtype == np.int64 and np.array_equal(provenance[key], expected),
                'Saved actual candidate source differs: ' + key)
    for new, old in (('sampled_points', 'sampled_points'), ('sampled_valid_indices', 'sampled_indices'),
                     ('sampled_pixel_indices', 'sampled_pixel_indices')):
        require(np.array_equal(generation[new], association[old]), 'R12 full-scene sample differs from sealed R11 source')
    return [{'sample_position': int(position[i]), 'valid_point_index': int(provenance['candidate_valid_indices'][i]),
             'pixel_index': int(provenance['candidate_pixel_indices'][i]), 'selected_seed_index': int(selected[i])}
            for i in range(len(candidates))]


def validate_protocol(protocol):
    require(protocol['schema'] == 13 and protocol['candidate_count'] == 236 and protocol['block_size'] == 16,
            'Frozen candidate/block contract changed')
    definition = protocol['feature_definition']
    for key, expected in {'roi_radius_pixels': 2, 'maximum_depth_m': 2, 'maximum_depth_inclusive': True,
                          'depth_scale_m': .02, 'quality': '(p*(1-h)+v*c)/2',
                          'soft_multiplier': '0.5+0.5*quality', 'instance_id_used_in_score': False}.items():
        require(definition[key] == expected, 'Frozen feature definition differs: ' + key)
    inputs = protocol['feature_inputs']
    require(not set(inputs).intersection(protocol['evaluation_inputs']), 'Feature inputs include evaluation artifacts')
    allowed = {protocol[key] for key in ('candidate_path', 'legacy_candidate_path', 'heads_path', 'instances_path', 'depth_path', 'meta_path')}
    heads_dir = str(Path(protocol['heads_path']).parent).replace('\\', '/')
    allowed.update(heads_dir + '/' + name for name in ('completion.json', 'heads_commit.json'))
    assoc = protocol['association_dir'].rstrip('/')
    allowed.update(assoc + '/' + name for name in ('binding.json', 'completion.json', 'points_commit.json',
        'sampled_indices.npy', 'sampled_pixel_indices.npy', 'sampled_points.npy', 'valid_instance_ids.npy',
        'valid_pixel_indices.npy', 'valid_points.npy'))
    for block in range(15):
        allowed.update(assoc + '/blocks/block_%04d%s' % (block, suffix) for suffix in ('.json', '.sha256.json'))
    generation = protocol['generation_dir'].rstrip('/')
    allowed.update(generation + '/' + name for name in ('binding.json', 'completion.json', 'input_commit.json', 'report.json'))
    for name in ('candidates.npy', 'eligible_indices.npy', 'forward_bundle.npz', 'sampled_eligible_indices.npy',
                 'sampled_pixel_indices.npy', 'sampled_points.npy', 'sampled_valid_indices.npy', 'seed_provenance.npz'):
        allowed.update((generation + '/' + name, generation + '/' + name + '.pending.json'))
    require(set(inputs) == allowed, 'Feature input allowlist changed or includes an unrelated source')
    require(all('/geometry/' not in path and '/label/' not in path for path in inputs), 'Forbidden feature input')


def legacy_association_fingerprint(binding):
    """V11 JSON serialization is deliberately distinct from the V13 receipt format."""
    return hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_feature_inputs(root, protocol):
    """Only bound prediction/source data; object poses and score geometry are never opened."""
    def input_path(name):
        require(name in protocol['feature_inputs'], 'Unbound feature input: ' + name)
        return common.safe_path(root, name)
    def array(name):
        return np.load(input_path(name), allow_pickle=False)
    candidates = array(protocol['candidate_path'])
    legacy = array(protocol['legacy_candidate_path'])
    require(candidates.dtype == legacy.dtype == np.float32 and candidates.shape == legacy.shape == (236, 17)
            and np.isfinite(candidates).all() and np.array_equal(candidates, legacy), 'The fixed 236-candidate pool changed')
    association = {name: array(protocol['association_dir'] + '/' + name + '.npy') for name in (
        'sampled_indices', 'sampled_pixel_indices', 'sampled_points', 'valid_instance_ids', 'valid_pixel_indices', 'valid_points')}
    generation = {name: array(protocol['generation_dir'] + '/' + name + '.npy') for name in (
        'sampled_points', 'sampled_valid_indices', 'sampled_pixel_indices')}
    for name in ('sampled_valid_indices', 'sampled_pixel_indices'):
        require(generation[name].dtype == np.int64 and generation[name].shape == (20000,), 'Invalid R12 source indices')
    require(generation['sampled_points'].shape == (20000, 3), 'Frozen network input size differs')
    with np.load(input_path(protocol['generation_dir'] + '/seed_provenance.npz'), allow_pickle=False) as archive:
        provenance = {name: archive[name].copy() for name in archive.files}
    traces = validate_actual_seed_chain(candidates, provenance, generation, association)
    with np.load(input_path(protocol['heads_path']), allow_pickle=False) as archive:
        require(set(archive.files) == {'foreground', 'centers', 'offsets'}, 'Frozen head archive differs')
        foreground = archive['foreground'].copy()
    require(foreground.dtype == np.float32 and foreground.shape == (1, 1, 240, 320)
            and np.isfinite(foreground).all(), 'Invalid frozen foreground logits')
    with Image.open(input_path(protocol['depth_path'])) as image:
        raw_depth = np.array(image)
    calibration = loadmat(input_path(protocol['meta_path']), variable_names=['intrinsic_matrix', 'factor_depth'])
    factor = np.asarray(calibration['factor_depth']).reshape(-1)
    require(raw_depth.shape == (720, 1280) and raw_depth.dtype == np.uint16 and factor.shape == (1,)
            and float(factor[0]) == 1000., 'Frozen depth calibration differs')
    intrinsic = np.asarray(calibration['intrinsic_matrix'])
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all(), 'Invalid intrinsic calibration')
    depth_m = raw_depth.astype(np.float64) / float(factor[0])
    instances = array(protocol['instances_path'])
    require(instances.dtype == np.int32 and instances.shape == (720, 1280), 'Invalid predicted instance frame')
    pixels = association['valid_pixel_indices']
    require(pixels.dtype == np.int64 and pixels.ndim == 1 and np.all((pixels >= 0) & (pixels < instances.size))
            and np.all(np.diff(pixels) > 0), 'Invalid original pixel lookup')
    require(np.array_equal(instances.ravel()[pixels], association['valid_instance_ids']), 'Predicted ID lookup differs')
    source_binding = legacy_association_fingerprint(common.read(input_path(protocol['association_dir'] + '/binding.json')))
    rows = []
    for block in range(15):
        name = protocol['association_dir'] + '/blocks/block_%04d.json' % block
        payload = common.read(input_path(name))
        receipt = common.read(input_path(name.replace('.json', '.sha256.json')))
        require(payload['schema'] == 11 and payload['binding_fingerprint'] == source_binding
                and receipt['binding_fingerprint'] == source_binding
                and receipt['sha256'] == protocol['feature_inputs'][name]
                and payload['start'] == block * 16 and payload['stop'] == min(236, (block + 1) * 16),
                'Frozen R11 source block differs')
        rows.extend(payload['rows'])
    require([row['original_index'] for row in rows] == list(range(236)), 'Frozen R11 source rows incomplete')
    return {'candidates': candidates, 'source_rows': rows, 'association': association,
            'foreground': foreground[0, 0], 'depth_m': depth_m, 'actual_seeds': traces}


def block_value(index, rows, binding):
    return {'schema': 13, 'binding_fingerprint': binding['fingerprint'], 'start': index * 16,
            'stop': index * 16 + len(rows), 'rows': rows}


def payload_names():
    return ['blocks/block_%04d.json' % block for block in range(15)] + ['report.json', 'completion.json']


def allowed_output_names():
    names = {'binding.json'}
    for name in payload_names():
        names.update((name, name + '.sha256.json'))
    return names


def check_inventory(directory, remove_temporaries=False):
    allowed = allowed_output_names()
    if not directory.exists():
        return
    for path in sorted(directory.rglob('*')):
        name = path.relative_to(directory).as_posix()
        require(not path.is_symlink(), 'Unexpected output symbolic link: ' + name)
        if path.is_dir():
            require(name == 'blocks', 'Unexpected feature output directory: ' + name)
            continue
        require(path.is_file() and not path.is_symlink(), 'Unexpected output filesystem object: ' + name)
        if name in allowed:
            continue
        match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', name)
        require(match is not None and match[1] in allowed, 'Unexpected feature output: ' + name)
        require(remove_temporaries, 'Interrupted temporary output requires writable recovery: ' + name)
        path.unlink()


def save_commit(path, value, binding, guard):
    # Reserve both the JSON and intent receipt before either becomes durable.
    guard.check(2 * len(common.json_bytes(value)) + 1024)
    common.commit(path, value, binding)
    guard.check()


def validate_cached_block(value, block, binding):
    require(value['schema'] == 13 and value['binding_fingerprint'] == binding['fingerprint']
            and value['start'] == block * 16 and value['stop'] == min(236, (block + 1) * 16)
            and [row['original_index'] for row in value['rows']] == list(range(value['start'], value['stop'])),
            'Cached feature block identity differs')
    return value['rows']


def make_report(rows, binding, protocol):
    require(len(rows) == 236, 'Full ranking requires every candidate')
    return {'schema': 13, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
            'protocol_sha256': binding['protocol_sha256'], 'candidate_count': 236, 'block_count': 15,
            'candidate_sha256': protocol['feature_inputs'][protocol['candidate_path']],
            'feature_inputs': protocol['feature_inputs'], 'source_hashes': protocol['source_hashes'],
            'feature_definition': protocol['feature_definition'], 'rows': rows,
            'rank_original_indices': rank_rows(rows), 'model_forward_calls': 0, 'training_steps': 0,
            'geometry_labels_read': False, 'calibrated_risk_probability': False,
            'limitations': protocol['limitations']}


def completion_inventory(directory):
    names = ['binding.json']
    for name in payload_names()[:-1]:
        names.extend((name, name + '.sha256.json'))
    return {name: common.digest(directory / name) for name in names}


def verify_completed(directory, binding, protocol, guard):
    check_inventory(directory)
    require(common.read(directory / 'binding.json') == binding, 'Feature binding changed')
    completion = common.load_commit(directory / 'completion.json', binding)
    require(completion['schema'] == 13 and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == binding['fingerprint']
            and completion['candidate_count'] == 236 and completion['block_count'] == 15
            and completion['outputs'] == completion_inventory(directory), 'Completion inventory differs')
    rows = []
    for block in range(15):
        value = common.load_commit(directory / ('blocks/block_%04d.json' % block), binding)
        rows.extend(validate_cached_block(value, block, binding))
        guard.check()
    report = common.load_commit(directory / 'report.json', binding)
    require(report == make_report(rows, binding, protocol)
            and completion['report_sha256'] == common.digest(directory / 'report.json'), 'Completed report differs')
    return {'status': 'verified_completed_read_only', 'new_blocks': 0, 'new_feature_rows': 0,
            'reused_blocks': 15, 'committed_blocks': 15, 'expected_blocks': 15, 'resources': guard.check()}


def run(root=ROOT, config=CONFIG, max_new_blocks=None, check_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    require(max_new_blocks is None or type(max_new_blocks) is int and max_new_blocks >= 0, 'Invalid block limit')
    protocol = common.read(config)
    validate_protocol(protocol)
    guard = common.Guard(root, protocol)
    guard.check()
    binding = common.make_binding(root, config, protocol, input_group='feature_inputs')
    directory = common.safe_path(root, protocol['outputs']['features_dir'])
    guard.check()
    if (directory / 'completion.json').exists():
        return verify_completed(directory, binding, protocol, guard)
    require(not check_only, 'Feature extraction incomplete; check-only cannot publish or recompute')
    with common.lock(root / 'runtime/locks/soft_ranking_features_v13'):
        check_inventory(directory, remove_temporaries=True)
        guard.check(2 * len(common.json_bytes(binding)) + 1024)
        common.once(directory / 'binding.json', binding)
        rows, reused, added, new_rows, inputs = [], 0, 0, 0, None
        for block in range(15):
            path = directory / ('blocks/block_%04d.json' % block)
            if path.exists():
                rows.extend(validate_cached_block(common.load_commit(path, binding), block, binding))
                reused += 1
            else:
                require(not any((directory / ('blocks/block_%04d.json' % later)).exists() for later in range(block + 1, 15)),
                        'Non-contiguous committed block prefix')
                if max_new_blocks is not None and added >= max_new_blocks:
                    return {'status': 'partial', 'new_blocks': added, 'new_feature_rows': new_rows,
                            'reused_blocks': reused, 'committed_blocks': reused + added, 'expected_blocks': 15,
                            'resources': guard.check()}
                if inputs is None:
                    inputs = load_feature_inputs(root, protocol)
                    guard.check()
                current = [candidate_feature(i, inputs['candidates'][i], inputs['source_rows'][i], inputs['association'],
                            inputs['foreground'], inputs['depth_m'], inputs['actual_seeds'][i])
                           for i in range(block * 16, min(236, (block + 1) * 16))]
                save_commit(path, block_value(block, current, binding), binding, guard)
                rows.extend(current)
                added += 1
                new_rows += len(current)
            guard.check()
        require(common.make_binding(root, config, protocol, input_group='feature_inputs') == binding,
                'Input/source/runtime identity changed during feature extraction')
        report = make_report(rows, binding, protocol)
        save_commit(directory / 'report.json', report, binding, guard)
        completion = {'schema': 13, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
            'candidate_count': 236, 'block_count': 15, 'outputs': completion_inventory(directory),
            'report_sha256': common.digest(directory / 'report.json'), 'model_forward_calls': 0}
        save_commit(directory / 'completion.json', completion, binding, guard)
        return {'status': 'completed', 'new_blocks': added, 'new_feature_rows': new_rows,
                'reused_blocks': reused, 'committed_blocks': reused + added, 'expected_blocks': 15,
                'resources': guard.check()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--max-new-blocks', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.config, args.max_new_blocks, args.check_only), sort_keys=True))
