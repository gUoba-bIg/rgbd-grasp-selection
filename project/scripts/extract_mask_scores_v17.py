"""Retain center scores for fixed final-checkpoint development instance masks."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
from scipy.ndimage import maximum_filter
from scripts import evaluate_segmentation_baseline_v16 as parent_storage

CONFIG = 'configs/mask_ap_v17.json'
ARMS = ('rgb', 'rgbd')
DECODER = {'foreground_threshold': .5, 'center_threshold': .35, 'min_pixels': 20, 'max_centers': 64}
HEAD_KEYS = {'foreground_probability', 'center_probability', 'offsets'}
SCORE_POLICY = {'name': 'retained_selected_center_sigmoid_float32', 'ties': 'stable_preserve_original_order_no_perturbation',
    'maps_required': ['instance_id', 'selected_center_index', 'row', 'col', 'score'],
    'raw_prediction_cache_must_match_parent': True, 'maximum_instances': 64}


def require(value, message):
    if not value:
        raise ValueError(message)


def check_heads(heads, size=None):
    require(set(heads) == HEAD_KEYS, 'Unexpected cached head names')
    foreground, centers, offsets = (heads[key] for key in ('foreground_probability', 'center_probability', 'offsets'))
    shape = tuple(size) if size is not None else foreground.shape
    require(len(shape) == 2 and all(type(v) is int and v > 0 for v in shape), 'Invalid head dimensions')
    require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in heads.values()), 'Heads must be finite float32')
    require(foreground.shape == centers.shape == shape and offsets.shape == (2,)+shape, 'Cached head shapes differ')
    require(np.all((foreground >= 0) & (foreground <= 1)) and np.all((centers >= 0) & (centers <= 1)), 'Cached probabilities outside [0, 1]')


def decode_with_scores(heads, foreground_threshold=.5, center_threshold=.35, min_pixels=20, max_centers=64):
    """Original decode ordering, including IDs after small-instance rejection."""
    check_heads(heads)
    require(0 <= foreground_threshold <= 1 and 0 <= center_threshold <= 1
            and type(min_pixels) is int and min_pixels >= 1 and type(max_centers) is int and max_centers >= 1,
            'Invalid decoder parameters')
    foreground, centers, offsets = (heads[key] for key in ('foreground_probability', 'center_probability', 'offsets'))
    height, width = foreground.shape
    instance_map = np.zeros((height, width), dtype=np.int32)
    foreground_mask = foreground >= foreground_threshold
    maxima = maximum_filter(centers, size=9, mode='constant', cval=-np.inf)
    coordinates = np.argwhere((centers == maxima) & (centers >= center_threshold))
    scores = centers[coordinates[:, 0], coordinates[:, 1]]
    order = np.argsort(-scores, kind='stable')
    selected = []
    for index in order:
        candidate = coordinates[index]
        if all(np.linalg.norm(candidate-other) >= 5 for other in selected):
            selected.append(candidate)
            if len(selected) == max_centers:
                break
    selected_centers = [{'selected_center_index': index, 'row': int(point[0]), 'col': int(point[1]),
                         'score': float(centers[tuple(point)])} for index, point in enumerate(selected)]
    yy, xx = np.nonzero(foreground_mask)
    if not selected or not len(yy):
        return instance_map, foreground_mask, [], selected_centers
    coordinates = np.asarray(selected)
    destinations = np.stack([yy+offsets[1, yy, xx]*height, xx+offsets[0, yy, xx]*width], axis=1)
    nearest = np.empty(len(yy), dtype=np.int32)
    for start in range(0, len(yy), 4096):
        distance = ((destinations[start:start+4096, None]-coordinates[None])**2).sum(axis=2)
        nearest[start:start+4096] = distance.argmin(axis=1)
    instances = []
    for center_index in range(len(coordinates)):
        mask = nearest == center_index
        if int(mask.sum()) >= min_pixels:
            label = len(instances)+1
            instance_map[yy[mask], xx[mask]] = label
            instances.append(dict(instance_id=label, **selected_centers[center_index]))
    return instance_map, foreground_mask, instances, selected_centers


def array_payload(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def read_arrays(file):
    with np.load(file, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def identity(record):
    return parent_storage.frame_identity('development', record)


def frame_directory(directory, record):
    frame = identity(record)
    require(type(frame['scene']) is int and re.fullmatch(r'[0-9]{4}', frame['frame']), 'Invalid frame identity')
    return Path(directory)/'frames'/('scene_%04d_%s' % (frame['scene'], frame['frame']))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--max-new-frames', type=int)
    parser.add_argument('--check-only', action='store_true')
    return parser.parse_args(argv)


def storage():
    from scripts import mask_ap_common_v17
    return mask_ap_common_v17


def read_parent(root, protocol):
    common = storage()
    parent_file = common.safe(root, protocol['parent_protocol'])
    require(common.digest(parent_file) == protocol['parent_protocol_sha256'], 'Frozen parent protocol changed')
    acceptance_file = common.safe(root, protocol['parent_acceptance_report'])
    require(isinstance(protocol['parent_acceptance_sha256'], str) and len(protocol['parent_acceptance_sha256']) == 64,
            'Parent acceptance is not frozen')
    require(common.digest(acceptance_file) == protocol['parent_acceptance_sha256'], 'Parent acceptance changed')
    acceptance = common.read(acceptance_file)
    require(acceptance['schema'] == 16 and acceptance['status'] == 'accepted_complete_paired_development_baseline'
            and acceptance['binding']['protocol_sha256'] == protocol['parent_protocol_sha256'],
            'Require completed paired parent acceptance')
    parent = common.read(parent_file)
    parent_storage.verify_protocol(root, parent, parent_file)
    require(protocol['runtime'] == parent['runtime'] and protocol['validation_scenes'] == parent['validation_scenes'] and protocol['frames'] == parent['frames']
            and protocol['input_size'] == parent['input_size'] and protocol['decoder'] == parent['decoder'], 'Parent inference policy differs')
    # Hashing the frozen parent snapshot includes label bytes; none is decoded here.
    manifest, records = parent_storage.verified_manifest(root, parent)
    sources = {arm: parent_storage.training_sources(root, parent, parent_file, arm, manifest) for arm in ARMS}
    bindings, seals = {}, {}
    for arm in ARMS:
        directory = common.safe(root, parent['outputs']['evaluation_root'])/arm
        binding = parent_storage.make_binding(root, parent, parent_file, arm, sources)
        require(parent_storage.read(directory/'binding.json') == binding, 'Parent evaluation binding differs')
        parent_storage.verify_completion(directory, binding, records)
        bindings[arm] = binding
        seals[arm] = {'binding_fingerprint': binding['fingerprint'], 'completion_sha256': common.digest(directory/'completion.json'),
                      'report_sha256': common.digest(directory/'report.json')}
    require(len(records['development']) == 160 and len(parent['validation_scenes']) == 10, 'Require all 160 frozen development frames')
    return {'protocol': parent, 'protocol_file': parent_file, 'records': records['development'],
            'sources': sources, 'evaluation_bindings': bindings, 'evaluation_seals': seals}


def make_binding(root, protocol, config_path, arm, context):
    common = storage()
    body = {'schema': 17, 'phase': 'scores', 'arm': arm, 'protocol_sha256': common.digest(config_path),
        'source_hashes': protocol['source_hashes'], 'input_hashes': protocol['input_hashes'],
        'parent_protocol_sha256': protocol['parent_protocol_sha256'], 'parent_acceptance_sha256': protocol['parent_acceptance_sha256'],
        'parent_training_files': {name: sha for source in context['sources'].values() for name, sha in source['files'].items()},
        'parent_evaluation_seals': context['evaluation_seals'], 'final_checkpoints': {
            name: source['identity'] for name, source in context['sources'].items()},
        'runtime': protocol['runtime'], 'interpreter_sha256': common.digest(common.safe(root, protocol['interpreter'])),
        'decoder': protocol['decoder'], 'input_size': protocol['input_size'], 'precision': 'float32_without_autocast',
        'expected_frames': 160, 'score_definition': 'Float32 sigmoid at the selected center, retained through small-mask ID reassignment'}
    return dict(body, fingerprint=common.fingerprint(body))


def parent_prediction(root, context, arm, record):
    common = storage()
    directory = common.safe(root, context['protocol']['outputs']['evaluation_root'])/arm
    file = parent_storage.frame_directory(directory, identity(record))/'prediction.npz'
    instances, foreground = parent_storage.load_prediction(file, context['evaluation_bindings'][arm], context['protocol']['input_size'])
    return instances, foreground, common.digest(file)


def load_model(root, context, arm, guard):
    import torch
    from scripts import train_segmentation_baseline_v16 as trainer
    require(torch.cuda.is_available(), 'CUDA required for a new score forward')
    trainer.reset_random_streams(context['protocol']['seed'], 'cuda')
    model, *_ = trainer.load_final_model(context['sources'][arm]['directory'], context['protocol'])
    limit = guard.protocol['resources']['gpu_reserved_soft_bytes']
    torch.cuda.set_per_process_memory_fraction(min(1., limit/torch.cuda.get_device_properties(0).total_memory), 0)
    model = model.to('cuda').eval()
    guard.check()
    return model


def prepare_device_input(inputs):
    import torch
    return torch.from_numpy(inputs)[None].to('cuda')


def forward_once(model, inputs):
    import torch
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        outputs = model(inputs)
        require(set(outputs) == {'foreground', 'centers', 'offsets'} and
                all(value.dtype == torch.float32 for value in outputs.values()), 'Model returned non-FP32 heads')
        return {'foreground_probability': outputs['foreground'][0, 0].float().sigmoid().cpu().numpy(),
                'center_probability': outputs['centers'][0, 0].float().sigmoid().cpu().numpy(),
                'offsets': outputs['offsets'][0].float().cpu().numpy()}


def committed(common, file, value, binding, guard, read_only=False):
    guard.check(len(common.json_bytes(value))+4096)
    common.commit(file, value, binding, read_only=read_only)


def commit_arrays(common, file, payload, binding, guard, read_only=False):
    guard.check(len(payload)+4096)
    common.commit_bytes(file, payload, binding, read_only=read_only)


def attempts(frame, binding):
    common = storage()
    result = []
    directory = Path(frame)/'attempts'
    intents = sorted(directory.glob('attempt_*.intent.json')) if directory.exists() else []
    for index, file in enumerate(intents):
        require(file.name == 'attempt_%04d.intent.json' % index, 'Noncontiguous score attempt ledger')
        intent = common.load_commit(file, binding)
        require(intent['schema'] == 17 and intent['phase'] == 'model_forward' and intent['arm'] == binding['arm']
                and intent['binding_fingerprint'] == binding['fingerprint'] and intent['attempt_index'] == index,
                'Forward intent identity differs')
        declared = intent['identity']
        require(declared['partition'] == 'development' and declared['camera'] == 'realsense' and
                frame_directory(Path(frame).parent.parent, declared) == Path(frame), 'Forward intent belongs to another frame')
        outcome_file = file.with_name(file.name.replace('.intent.json', '.outcome.json'))
        outcome = common.load_commit(outcome_file, binding) if outcome_file.exists() else None
        if outcome is not None:
            require(all(outcome.get(key) == value for key, value in intent.items()) and
                    outcome['status'] in ('model_returned', 'model_exception'), 'Forward outcome identity differs')
        elif common.receipt_path(outcome_file).exists():
            pending = common.read(common.receipt_path(outcome_file))
            require(set(pending) == {'schema', 'binding_fingerprint', 'sha256', 'bytes'} and pending['schema'] == 17
                    and pending['binding_fingerprint'] == binding['fingerprint'] and type(pending['bytes']) is int and pending['bytes'] > 0
                    and re.fullmatch('[0-9a-f]{64}', pending['sha256']), 'Malformed interrupted outcome receipt')
        result.append({'attempt_index': index, 'intent': intent, 'outcome': outcome})
    return result


def attempt_summary(frame, binding):
    rows = attempts(frame, binding)
    returned = sum(row['outcome'] is not None and row['outcome']['status'] == 'model_returned' for row in rows)
    exceptional = sum(row['outcome'] is not None and row['outcome']['status'] == 'model_exception' for row in rows)
    unknown = sum(row['outcome'] is None for row in rows)
    return {'intents_committed': len(rows), 'model_calls_returned': returned, 'model_exceptions_observed': exceptional,
            'unknown_outcome_intents': unknown, 'known_model_call_lower_bound': returned+exceptional,
            'exact_call_count_known': unknown == 0}


def new_heads(root, context, arm, record, frame, binding, guard, model, work):
    common = storage()
    inputs = parent_storage.prepare_input(root, context['protocol'], record, arm)
    require(inputs.dtype == np.float32 and inputs.shape == ((3 if arm == 'rgb' else 5),)+tuple(context['protocol']['input_size'])
            and np.isfinite(inputs).all(), 'Model input differs')
    device_input = prepare_device_input(inputs)
    previous = attempts(frame, binding)
    index = len(previous)
    intent = {'schema': 17, 'phase': 'model_forward', 'arm': arm, 'identity': identity(record),
        'attempt_index': index, 'binding_fingerprint': binding['fingerprint'],
        'input_tensor_sha256': hashlib.sha256(inputs.tobytes(order='C')).hexdigest(),
        'checkpoint_sha256': context['sources'][arm]['identity']['checkpoint_sha256']}
    file = frame/'attempts'/('attempt_%04d.intent.json' % index)
    committed(common, file, intent, binding, guard)
    guard.check(4*1024**2)
    work['new_model_forward_calls'] += 1
    try:
        heads = forward_once(model, device_input)
    except Exception as error:
        outcome = dict(intent, status='model_exception', error_type=type(error).__name__, error=str(error))
        committed(common, file.with_name(file.name.replace('.intent.json', '.outcome.json')), outcome, binding, guard)
        raise
    payload = array_payload(**heads)
    outcome = dict(intent, status='model_returned', heads_sha256=hashlib.sha256(payload).hexdigest(), heads_bytes=len(payload))
    committed(common, file.with_name(file.name.replace('.intent.json', '.outcome.json')), outcome, binding, guard)
    # Preserve the first payload before validating numeric values or parent-mask equality.
    commit_arrays(common, frame/'heads.npz', payload, binding, guard)
    return heads


def expected_record(root, context, arm, source, frame, binding, heads):
    common = storage()
    check_heads(heads, context['protocol']['input_size'])
    instances, foreground, scores, centers = decode_with_scores(heads, **context['protocol']['decoder'])
    old_instances, old_foreground, old_sha = parent_prediction(root, context, arm, source)
    require(np.array_equal(instances, old_instances) and np.array_equal(foreground, old_foreground),
            'New decoded masks differ bitwise from frozen parent development prediction')
    require(any(item['outcome'] is not None and item['outcome']['status'] == 'model_returned'
                and item['outcome']['heads_sha256'] == common.digest(frame/'heads.npz')
                for item in attempts(frame, binding)), 'Raw heads have no matching returned-forward evidence')
    record = {'schema': 17, 'arm': arm, 'identity': identity(source), 'binding_fingerprint': binding['fingerprint'],
        'heads_sha256': common.digest(frame/'heads.npz'), 'parent_prediction_sha256': old_sha,
        'checkpoint_sha256': context['sources'][arm]['identity']['checkpoint_sha256'],
        'instances': scores, 'selected_centers': centers, 'predicted_instance_count': len(scores),
        'parent_instances_and_foreground_bitwise_equal': True, 'model_inputs_and_scores_use_ground_truth': False,
        'training_updates': 0, 'forward_attempt_accounting': attempt_summary(frame, binding)}
    return instances, foreground, record


def output_inventory(directory, records, complete=False, cleanup=False):
    common = storage()
    directory = Path(directory)
    bases = [directory/'binding.json', directory/'report.json', directory/'completion.json']
    names = set(bases)
    names.update(common.receipt_path(file) for file in bases if file.name != 'binding.json')
    directories = {directory/'frames'}
    for source in records:
        frame = frame_directory(directory, source)
        directories.update((frame, frame/'attempts'))
        for basename in ('heads.npz', 'prediction.npz', 'record.json'):
            file = frame/basename
            names.update((file, common.receipt_path(file)))
        ledger = frame/'attempts'
        if ledger.exists():
            for file in ledger.iterdir():
                stem = re.sub(r'\.[0-9a-f]{32}\.tmp$', '', file.name)
                require(file.is_file() and re.fullmatch(r'attempt_[0-9]{4}\.(intent|outcome)\.json(\.sha256\.json)?', stem),
                        'Unexpected forward ledger file')
                names.add(file.with_name(stem))
    if directory.exists():
        for file in directory.rglob('*'):
            require(not file.is_symlink(), 'Link in score output tree')
            if file.is_file() and re.search(r'\.[0-9a-f]{32}\.tmp$', file.name):
                base = file.with_name(re.sub(r'\.[0-9a-f]{32}\.tmp$', '', file.name))
                require(base in names and not complete, 'Unexpected temporary score artifact')
                if cleanup:
                    file.unlink()
                continue
            require(file in names if file.is_file() else file in directories, 'Unexpected score output: '+str(file))
    if complete:
        require(all(file.exists() for file in names), 'Incomplete score output inventory')
    return names


def make_report(arm, binding, rows):
    accounting = {key: sum(row['forward_attempt_accounting'][key] for row in rows) for key in
        ('intents_committed', 'model_calls_returned', 'model_exceptions_observed', 'unknown_outcome_intents', 'known_model_call_lower_bound')}
    accounting['exact_call_count_known'] = accounting['unknown_outcome_intents'] == 0
    return {'schema': 17, 'status': 'completed_development_mask_scores', 'arm': arm,
        'binding_fingerprint': binding['fingerprint'], 'expected_frames': binding['expected_frames'], 'frame_count': len(rows),
        'predicted_instance_count': sum(row['predicted_instance_count'] for row in rows), 'rows': rows,
        'forward_attempt_accounting': accounting, 'nominal_model_forward_calls': binding['expected_frames'],
        'parent_predictions_bitwise_equal': True, 'ground_truth_metric_evaluation_performed': False,
        'new_training_updates': 0, 'scores_are_calibrated_probabilities': False}


def pending_accounting(directory, records, binding):
    common = storage()
    result = {'heads_without_frame_record': 0, 'heads_receipt_without_payload': 0, 'unknown_outcome_intents': 0}
    for source in records:
        frame = frame_directory(directory, source)
        result['heads_without_frame_record'] += int((frame/'heads.npz').exists() and not (frame/'record.json').exists())
        result['heads_receipt_without_payload'] += int(common.receipt_path(frame/'heads.npz').exists() and not (frame/'heads.npz').exists())
        result['unknown_outcome_intents'] += attempt_summary(frame, binding)['unknown_outcome_intents']
    return result


def verify_completed(directory, binding, context, root):
    common = storage()
    names = output_inventory(directory, context['records'], complete=True)
    completion = common.load_commit(directory/'completion.json', binding)
    outputs = {file.relative_to(directory).as_posix(): common.digest(file) for file in names
               if file not in (directory/'completion.json', common.receipt_path(directory/'completion.json'))}
    require(completion == {'schema': 17, 'status': 'completed', 'arm': binding['arm'], 'binding_fingerprint': binding['fingerprint'],
        'expected_frames': len(context['records']), 'outputs': outputs, 'report_sha256': common.digest(directory/'report.json')},
        'Score completion seal differs')
    rows = []
    for source in context['records']:
        frame = frame_directory(directory, source)
        common.verify_bytes(frame/'heads.npz', binding)
        common.verify_bytes(frame/'prediction.npz', binding)
        row = common.load_commit(frame/'record.json', binding)
        require(row['identity'] == identity(source) and row['arm'] == binding['arm'] and row['binding_fingerprint'] == binding['fingerprint']
                and row['heads_sha256'] == common.digest(frame/'heads.npz') and row['prediction_sha256'] == common.digest(frame/'prediction.npz'),
                'Score frame identity differs')
        require(row['forward_attempt_accounting'] == attempt_summary(frame, binding), 'Attempt ledger differs')
        rows.append(row)
    require(common.load_commit(directory/'report.json', binding) == make_report(binding['arm'], binding, rows), 'Score report differs')
    return rows


def run(root=ROOT, config=CONFIG, arm=None, max_new_frames=None, check_only=False):
    common = storage()
    root = Path(root).resolve()
    config_path = common.safe(root, config)
    protocol = common.read(config_path)
    require(protocol['schema'] == 17 and tuple(protocol['arms']) == ARMS and arm in ARMS
            and protocol['decoder'] == DECODER and protocol['input_size'] == [240, 320]
            and protocol['precision'] == 'float32_without_autocast' and protocol['score_policy'] == SCORE_POLICY
            and protocol['camera'] == 'realsense', 'Frozen score policy differs')
    require(max_new_frames is None or type(max_new_frames) is int and max_new_frames >= 0, 'Invalid new-frame budget')
    require(not check_only or max_new_frames is None, 'Read-only mode cannot request work budget')
    common.verify_source_maps(root, protocol)
    require(common.safe(root, protocol['interpreter']) == Path(sys.executable).resolve(), 'Wrong score interpreter')
    guard = common.Guard(root, protocol, phase='scores')
    guard.check()
    context = read_parent(root, protocol)
    binding = make_binding(root, protocol, config_path, arm, context)
    directory = common.safe(root, protocol['outputs']['scores_root'])/arm
    work = {'schema': 17, 'arm': arm, 'new_frames': 0, 'new_model_forward_calls': 0,
            'new_training_updates': 0, 'reused_frames': 0, 'committed_frames': 0, 'expected_frames': len(context['records'])}
    if (directory/'completion.json').exists() or check_only:
        require((directory/'completion.json').exists(), 'Read-only scores require completion')
        before = common.snapshot(directory)
        require(common.read(directory/'binding.json') == binding, 'Score binding differs')
        rows = verify_completed(directory, binding, context, root)
        require(common.snapshot(directory) == before, 'Read-only score verification changed outputs')
        return dict(work, status='verified_completed_read_only', reused_frames=len(rows), committed_frames=len(rows),
                    pending_payload_accounting=pending_accounting(directory, context['records'], binding), resources=guard.check())
    with common.lock(root/'runtime/locks/mask_scores_v17'/arm):
        output_inventory(directory, context['records'], cleanup=True)
        guard.check(len(common.json_bytes(binding))+4096)
        # Binding has no receipt; its fingerprint is repeated by every committed artifact.
        if (directory/'binding.json').exists():
            require(common.read(directory/'binding.json') == binding, 'Existing score binding differs')
        else:
            common.publish_once(directory/'binding.json', common.json_bytes(binding))
        rows, model, missing_seen = [], None, False
        for source in context['records']:
            frame = frame_directory(directory, source)
            exists = (frame/'record.json').exists()
            if exists:
                require(not missing_seen, 'Score records are not a contiguous frame prefix')
                work['reused_frames'] += 1
            else:
                missing_seen = True
                if max_new_frames is not None and work['new_frames'] >= max_new_frames:
                    continue
            if not (frame/'heads.npz').exists():
                require(not exists, 'Committed frame is missing raw heads')
                with common.lock(root/'runtime/grasp_foreground_v12/generation_gpu_lock'):
                    if model is None:
                        model = load_model(root, context, arm, guard)
                    new_heads(root, context, arm, source, frame, binding, guard, model, work)
            common.verify_bytes(frame/'heads.npz', binding)
            heads = read_arrays(frame/'heads.npz')
            instances, foreground, row = expected_record(root, context, arm, source, frame, binding, heads)
            payload = array_payload(instances=instances, foreground=foreground)
            commit_arrays(common, frame/'prediction.npz', payload, binding, guard, read_only=exists)
            row['prediction_sha256'] = common.digest(frame/'prediction.npz')
            committed(common, frame/'record.json', row, binding, guard, read_only=exists)
            rows.append(row)
            if not exists:
                work['new_frames'] += 1
            guard.check()
        work['committed_frames'] = len(rows)
        done = len(rows) == len(context['records'])
        if done:
            report = make_report(arm, binding, rows)
            committed(common, directory/'report.json', report, binding, guard)
            names = output_inventory(directory, context['records'])
            outputs = {file.relative_to(directory).as_posix(): common.digest(file) for file in names
                if file not in (directory/'completion.json', common.receipt_path(directory/'completion.json'))}
            completion = {'schema': 17, 'status': 'completed', 'arm': arm, 'binding_fingerprint': binding['fingerprint'],
                'expected_frames': len(rows), 'outputs': outputs, 'report_sha256': common.digest(directory/'report.json')}
            committed(common, directory/'completion.json', completion, binding, guard)
            verify_completed(directory, binding, context, root)
        return dict(work, status='completed' if done else 'partial',
                    pending_payload_accounting=pending_accounting(directory, context['records'], binding), resources=guard.check())


if __name__ == '__main__':
    try:
        args = arguments()
        print(json.dumps(run(config=args.config, arm=args.arm, max_new_frames=args.max_new_frames, check_only=args.check_only),
                         sort_keys=True, allow_nan=False), flush=True)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print('Mask score extraction stopped: '+str(error), file=sys.stderr, flush=True)
        sys.exit(2)
