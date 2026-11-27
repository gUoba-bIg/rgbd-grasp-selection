"""Resumable pinned object-geometry scoring for two frozen-input candidate arms."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_grasp_geometry_v10 as legacy

read, digest, fingerprint = legacy.read, legacy.digest, legacy.fingerprint
require = legacy.require
ARMS = ('full_scene', 'predicted_foreground')
OUTPUT_RELATIVE = 'results/P3/grasp_foreground_v12'
RESOURCES = {'cpu_threads': 2, 'invocation_soft_seconds': 1800,
             'rss_soft_bytes': 8 * 1024 ** 3, 'output_limit_bytes': 512 * 1024 ** 2,
             'disk_reserve_bytes': 60 * 1024 ** 3}


def relative(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Path escaped project')
    return path


def output_bytes(directory):
    if directory.is_file():
        return directory.stat().st_size
    return sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) if directory.exists() else 0


def combined_output_bytes():
    names = (OUTPUT_RELATIVE, 'results/P3/grasp_foreground_comparison_v12.json',
             'results/P3/grasp_foreground_visuals_v12',
             'results/P3/grasp_foreground_generation_v12_resume.json',
             'results/P3/grasp_foreground_geometry_v12_resume.json')
    return sum(output_bytes(relative(name)) for name in names)


class Monitor(legacy.common.Monitor):
    """All-arm output budget; 2-thread/8-GiB/30-minute per-invocation soft limits."""
    def __init__(self, resources):
        super().__init__(resources, relative(OUTPUT_RELATIVE))

    def check(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        require(self.error is None, self.error or '')
        require(self.peak <= self.protocol['rss_soft_bytes'], 'RSS soft limit exceeded')
        require(time.monotonic() - self.started <= self.protocol['invocation_soft_seconds'], 'Time limit exceeded')
        require(shutil.disk_usage(ROOT).free >= self.protocol['disk_reserve_bytes'], 'Disk reserve unavailable')
        require(combined_output_bytes() <= self.protocol['output_limit_bytes'], 'Combined R12 output budget exceeded')


def publish(path, value, resources, immutable=True):
    if path.exists() and immutable:
        require(read(path) == value, 'Immutable geometry output differs: ' + str(path))
        return
    payload_size = len((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
    require(combined_output_bytes() + payload_size <= resources['output_limit_bytes'],
            'Atomic publication exceeds combined R12 output budget')
    require(shutil.disk_usage(ROOT).free - payload_size >= resources['disk_reserve_bytes'], 'Publication disk reserve unavailable')
    legacy.atomic(path, value)


def commit(path, value, binding, resources):
    publish(path, value, resources)
    publish(legacy.receipt_path(path), {'sha256': digest(path), 'binding_fingerprint': binding}, resources)


def validate_protocol(protocol, arm):
    require(protocol['schema'] == 12 and protocol['arms'] == list(ARMS) and arm in ARMS, 'Unsupported R12 arm/schema')
    require(protocol['resources'] == RESOURCES, 'Frozen scoring resource limits changed')
    require(protocol['outputs']['root'] == OUTPUT_RELATIVE, 'R12 output root changed')
    for key, expected in (('acceptance_report', 'results/P3/grasp_foreground_comparison_v12.json'),
                          ('visuals_dir', 'results/P3/grasp_foreground_visuals_v12')):
        require(protocol['outputs'].get(key, expected) == expected, 'R12 ancillary output path changed')
    require(protocol['v10_protocol']['path'] == 'configs/grasp_geometry_v10.json', 'Wrong reference protocol')


def generation_binding(directory, arm, config_sha256=None, generator_sha256=None):
    import numpy as np
    completion = read(directory / 'completion.json')
    report = read(directory / 'report.json')
    binding = read(directory / 'binding.json')
    if config_sha256 is not None:
        require(binding['protocol_sha256'] == config_sha256, 'Generation used a different frozen protocol')
    if generator_sha256 is not None:
        require(binding['script_sha256'] == generator_sha256, 'Generation used a different script version')
    require(completion['schema'] == 12 and completion['status'] == 'completed'
            and completion['arm'] == arm, 'Generation is not committed for this arm')
    require(completion['outcome'] in ('generated', 'empty_candidates', 'insufficient_input'), 'Unknown generation outcome')
    canonical = json.dumps(binding, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    require(completion['binding_fingerprint'] == hashlib.sha256(canonical).hexdigest(), 'Generation binding fingerprint changed')
    require(type(completion['candidate_count']) is int and completion['candidate_count'] >= 0, 'Invalid generated count')
    require(report['status'] == 'completed' and report['outcome'] == completion['outcome'], 'Generation report differs')
    require(digest(directory / 'report.json') == completion['report_sha256'], 'Generation report hash differs')
    require(digest(directory / 'candidates.npy') == completion['candidates_sha256'], 'Generated candidate bytes differ')
    outputs = completion['outputs']
    require({'binding.json', 'report.json', 'candidates.npy'} <= set(outputs), 'Incomplete generation manifest')
    actual = {}
    for name, expected in outputs.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()) and path != directory / 'completion.json', 'Invalid generation output path')
        require(digest(path) == expected, 'Generated output changed: ' + name)
        actual[name] = expected
    candidates = np.load(directory / 'candidates.npy', allow_pickle=False)
    require(candidates.dtype == np.float32 and candidates.shape == (completion['candidate_count'], 17)
            and np.isfinite(candidates).all(), 'Invalid generated candidates')
    require((len(candidates) > 0) == (completion['outcome'] == 'generated'), 'Generation outcome/count conflict')
    require((candidates[:, 16] == -1).all(), 'Generated object IDs should remain unspecified')
    if len(candidates):
        require((candidates[:, 1:4] >= 0).all(), 'Invalid generated grasp dimensions')
    return {'arm': arm, 'completion_sha256': digest(directory / 'completion.json'),
            'binding_sha256': digest(directory / 'binding.json'), 'report_sha256': digest(directory / 'report.json'),
            'candidate_count': len(candidates), 'candidate_dtype': str(candidates.dtype),
            'outcome': completion['outcome'], 'output_hashes': actual}


def make_binding(protocol, config, arm):
    # Both arms must be immutable before any object-geometry evaluation begins.
    # Freeze the peer as well as the selected arm, so a scored arm cannot inform regeneration.
    config_hash = digest(config)
    generator_hash = digest(ROOT / 'scripts/generate_foreground_grasps_v12.py')
    all_generations = {name: generation_binding(relative(OUTPUT_RELATIVE) / name / 'generation', name,
                                              config_hash, generator_hash)
                       for name in ARMS}
    frozen_hashes = dict(protocol['input_hashes'], **protocol.get('source_hashes', {}))
    for name, expected in frozen_hashes.items():
        require(digest(relative(name)) == expected, 'Frozen R12 input changed: ' + name)
    old_path = relative(protocol['v10_protocol']['path'])
    require(digest(old_path) == protocol['v10_protocol']['sha256'], 'Reference geometry protocol changed')
    old = read(old_path)
    require(old['schema'] == 10 and old['scoring'] == legacy.SCORING, 'Reference scoring semantics differ')
    # This is verification only: the old runner/main/output globals are never used or modified.
    old_binding = legacy.make_binding(old, old_path)
    return {'schema': 12, 'arm': arm, 'protocol_sha256': digest(config), 'own_source_sha256': digest(__file__),
            'v10_verified_binding': old_binding, 'generation': all_generations[arm],
            'all_generation_seals': all_generations,
            'frozen_input_and_source_hashes': frozen_hashes, 'scoring': legacy.SCORING}, old


def empty_preparation(old, arm):
    return {'schema': 12, 'arm': arm, 'input_count': 0, 'input_dtype': 'float32',
            'object_ids': old['object_ids'], 'nms_count': 0, 'nms_original_indices': [],
            'nms_object_slots': [], 'per_object_pre_limit_counts': [0] * len(old['object_ids']),
            'selection_min_confidence': None, 'selected': [], 'selected_count': 0,
            'geometry_preparation_executed': False, 'reason': 'zero_generated_candidates'}


def prepare(old, candidate_path, count, arm, api):
    updated = dict(old, candidate_path=candidate_path.relative_to(ROOT).as_posix(), candidate_count=count)
    preparation, state = legacy.prepare(updated, api)
    preparation = dict(preparation, schema=12, arm=arm, geometry_preparation_executed=True)
    return preparation, state


def empty_oracle(prep):
    return {'executed': False, 'reason': 'zero_generated_candidates',
            'result': {'groups': [[] for _ in prep['object_ids']], 'scores': [[] for _ in prep['object_ids']],
                       'collision': [[] for _ in prep['object_ids']]}}


def summarize(prep, records, oracle, binding, arm, generation_outcome):
    require(prep['arm'] == arm and prep['schema'] == 12, 'Preparation arm/version mismatch')
    if prep['input_count'] == 0:
        require(prep['selected_count'] == 0 and not records and oracle == empty_oracle(prep), 'Invalid empty-input scoring state')
    else:
        require(prep['selected_count'] > 0 and oracle['executed'] is True, 'Nonempty scoring needs executed oracle')
    result = legacy.summarize(prep, records, oracle['result'], binding)
    report_status = ('accepted_limited_single_frame_geometry_engineering' if prep['input_count'] else
                     'skipped_insufficient_generation_input' if generation_outcome == 'insufficient_input' else
                     'completed_empty_candidate_negative_result')
    result.update(schema=12, arm=arm, generation_outcome=generation_outcome,
                  status=report_status,
                  official_oracle_executed=oracle['executed'],
                  official_oracle_exact_match=True if oracle['executed'] else None,
                  geometry_preparation_executed=prep['geometry_preparation_executed'],
                  predicted_foreground_input=(arm == 'predicted_foreground'),
                  no_predicted_instance_object_id_substitution=True,
                  ground_truth_geometry_used_only_for_evaluation=bool(prep['input_count']))
    result['limitations'] = [
        'One reused training-scene frame; this is not scene/split AP, independent test evidence or a generalization estimate.',
        'The foreground arm uses the frozen V8 predicted foreground union; this is not a new trained grasp model.',
        'Different point sets can alter sampling and predictions; any observed difference is a single-frame diagnostic, not causal proof.',
        'Predicted instance IDs are not substituted for dataset object IDs in official geometry assignment.',
        'The frozen V7 axial-sphere contact-search false negative remains unresolved.',
        'Object geometry scoring is not robot safety, actual grasp success, or a calibrated probability.',
        'The public mirror source lacks a separately published official asset archive digest.',
        'An empty candidate arm has an all-zero 50x6 prefix matrix by definition; no official empty-input oracle was run.'
    ]
    return result


def cache_names(count):
    return legacy.expected_cache_names(count) | {'report.json'}


def validate_records(prep, records):
    require(len(records) == prep['selected_count'], 'Missing score records')
    for selected, record in zip(prep['selected'], records):
        require(record is not None and all(record.get(k) == v for k, v in selected.items()), 'Cached grasp identity changed')


def check_complete(binding, directory, arm):
    fp = fingerprint(binding)
    require(read(directory / 'binding.json') == binding, 'Completed geometry binding changed')
    completion = read(directory / 'completion.json')
    require(completion['schema'] == 12 and completion['arm'] == arm and completion['status'] == 'completed'
            and completion['binding_fingerprint'] == fp, 'Invalid completed geometry binding')
    prep = legacy.load_commit(directory / 'preparation.json', fp)
    require(prep is not None and completion['expected_grasps'] == prep['selected_count'], 'Completed selected count differs')
    names = cache_names(prep['selected_count'])
    legacy.check_layout(directory, names, complete=True)
    require(set(completion['outputs']) == names - {'completion.json'}, 'Completed geometry manifest membership differs')
    for name, expected in completion['outputs'].items():
        require(digest(directory / name) == expected, 'Completed geometry output changed: ' + name)
    records = [legacy.load_commit(directory / 'grasps' / ('grasp_%04d.json' % i), fp) for i in range(prep['selected_count'])]
    validate_records(prep, records)
    oracle = legacy.load_commit(directory / 'oracle.json', fp)
    expected = summarize(prep, records, oracle, fp, arm, binding['generation']['outcome'])
    require(read(directory / 'report.json') == expected, 'Completed geometry report is not reproducible')
    return expected


def run(args):
    config = args.config.resolve()
    protocol = read(config)
    validate_protocol(protocol, args.arm)
    directory = relative(OUTPUT_RELATIVE) / args.arm / 'geometry'
    monitor = Monitor(protocol['resources'])
    try:
        monitor.check()
        binding, old = make_binding(protocol, config, args.arm)
        fp = fingerprint(binding)

        def verified():
            report = check_complete(binding, directory, args.arm)
            result = {'status': 'verified_completed_read_only', 'arm': args.arm,
                      'selected_count': report['selected_count'], 'new_grasps': 0,
                      'reused_grasps': report['selected_count'], 'geometry_computations': 0,
                      'committed_grasps': report['selected_count'], 'expected_grasps': report['selected_count'],
                      'official_oracle_executed_this_invocation': False}
            print(json.dumps(result), flush=True)
            monitor.check()
            return result

        if args.check_only:
            require((directory / 'completion.json').is_file(), 'No committed geometry completion')
            return verified()
        with legacy.lock(ROOT / 'runtime/grasp_foreground_v12' / (args.arm + '_geometry_lock')):
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / 'binding.json').exists():
                legacy.check_layout(directory, {'binding.json'})
            publish(directory / 'binding.json', binding, protocol['resources'])
            if (directory / 'completion.json').exists():
                return verified()
            generation = binding['generation']
            candidate_path = directory.parent / 'generation/candidates.npy'
            prep_path = directory / 'preparation.json'
            prep = legacy.load_commit(prep_path, fp)
            api, state, dex = None, None, None

            def ensure_state():
                nonlocal api, state, dex
                if state is None:
                    api = legacy.imports()
                    rebuilt, state = prepare(old, candidate_path, generation['candidate_count'], args.arm, api)
                    require(rebuilt == prep, 'Rebuilt preparation differs from committed prefix')
                    dex = legacy.LazyModels(state['data_root'], state['obj_ids'], api[3].load_dexnet_model)
                return state

            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2):
                if prep is None:
                    if generation['candidate_count'] == 0:
                        prep = empty_preparation(old, args.arm)
                    else:
                        api = legacy.imports()
                        prep, state = prepare(old, candidate_path, generation['candidate_count'], args.arm, api)
                        dex = legacy.LazyModels(state['data_root'], state['obj_ids'], api[3].load_dexnet_model)
                    commit(prep_path, prep, fp, protocol['resources'])
                require(prep['input_count'] == generation['candidate_count'] and prep['arm'] == args.arm, 'Prepared count/arm mismatch')
                legacy.check_layout(directory, cache_names(prep['selected_count']))
                records, fresh, reused, oracle_executed = [], 0, 0, False
                for item in prep['selected']:
                    monitor.check()
                    path = directory / 'grasps' / ('grasp_%04d.json' % item['selection_index'])
                    record = legacy.load_commit(path, fp)
                    if record is None:
                        if args.max_new_grasps is not None and fresh >= args.max_new_grasps:
                            result = {'status': 'partial', 'arm': args.arm, 'new_grasps': fresh,
                                      'reused_grasps': reused, 'committed_grasps': len(records),
                                      'expected_grasps': prep['selected_count'], 'binding_fingerprint': fp,
                                      'peak_rss_bytes': monitor.peak, 'seconds': time.monotonic() - monitor.started}
                            publish(directory / 'last_invocation.json', result, protocol['resources'], immutable=False)
                            print(json.dumps(result), flush=True)
                            return result
                        ensure_state()
                        record = legacy.score_one(item, state, dex, api)
                        commit(path, record, fp, protocol['resources'])
                        fresh += 1
                    else:
                        reused += 1
                    require(all(record.get(k) == v for k, v in item.items()), 'Cached candidate identity differs')
                    records.append(record)
                oracle_path = directory / 'oracle.json'
                oracle = legacy.load_commit(oracle_path, fp)
                if oracle is None:
                    if prep['input_count']:
                        ensure_state()
                        oracle = {'executed': True, 'result': legacy.oracle_result(state, api, dex)}
                        oracle_executed = True
                    else:
                        oracle = empty_oracle(prep)
                    commit(oracle_path, oracle, fp, protocol['resources'])
                report = summarize(prep, records, oracle, fp, args.arm, generation['outcome'])
                publish(directory / 'report.json', report, protocol['resources'])
                result = {'status': 'completed', 'arm': args.arm, 'new_grasps': fresh, 'reused_grasps': reused,
                          'committed_grasps': len(records), 'expected_grasps': prep['selected_count'],
                          'binding_fingerprint': fp, 'peak_rss_bytes': monitor.peak,
                          'seconds': time.monotonic() - monitor.started,
                          'official_oracle_executed_this_invocation': oracle_executed}
                publish(directory / 'last_invocation.json', result, protocol['resources'], immutable=False)
                outputs = {name: digest(directory / name) for name in sorted(cache_names(len(records)) - {'completion.json'})}
                publish(directory / 'completion.json', {'schema': 12, 'status': 'completed', 'arm': args.arm,
                        'binding_fingerprint': fp, 'expected_grasps': len(records), 'outputs': outputs}, protocol['resources'])
                check_complete(binding, directory, args.arm)
                monitor.check()
                result.update(report_status=report['status'], selected_count=report['selected_count'],
                              positive_friction_score_count=report['positive_friction_score_count'])
                print(json.dumps(result), flush=True)
                return result
    finally:
        monitor.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grasp_foreground_comparison_v12.json')
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--max-new-grasps', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    require(args.max_new_grasps is None or args.max_new_grasps > 0, 'Invalid new-grasp limit')
    require(not (args.check_only and args.max_new_grasps is not None), 'Check-only cannot request new grasps')
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
