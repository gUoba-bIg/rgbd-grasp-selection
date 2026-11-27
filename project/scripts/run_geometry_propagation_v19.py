"""Bounded case workers and immutable restart evidence for synthetic geometry."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import geometry_storage_v19 as store
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from src import geometry_propagation_v19 as geometry

CONFIG = 'configs/geometry_propagation_v19.json'
SOURCE = 'scripts/run_geometry_propagation_v19.py'
MODES = ('full', 'diagonal')
OUTPUTS = {'root': 'runtime/verification/geometry_propagation_v19',
           'report': 'results/P6/geometry_propagation_v19.json',
           'logs': 'logs/P6/geometry_propagation_v19',
           'acceptance_report': 'results/P6/geometry_propagation_v19_acceptance.json',
           'visuals_dir': 'results/P6/geometry_propagation_v19_visuals',
           'resume_report': 'results/P6/geometry_propagation_v19_resume.json',
           'readonly_report': 'results/P6/geometry_propagation_v19_readonly_verification.json'}
LIMITS = {'cpu_threads': 4, 'worker_timeout_seconds': 120, 'driver_soft_seconds': 1800,
          'rss_soft_bytes': 2*1024**3, 'output_limit_bytes': 1024**3, 'disk_reserve_bytes': 60*1024**3}
require = store.require


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    return value


def validate(root, config, protocol):
    require(protocol['schema'] == 19 and protocol['status'] == 'frozen'
        and protocol['phase'] == 'synthetic_geometry_propagation', 'Geometry protocol must be frozen before execution')
    require(protocol['cases'] == geometry.enumerate_cases() and protocol['seed_inventory'] == geometry.seed_inventory(), 'Fixed case or seed design changed')
    require(protocol['sampling'] == {'generator': 'numpy.RandomState.standard_normal', 'master_seed': 20261027,
        'case_count': 72, 'batch_count': 16, 'batch_size': 8192, 'samples_per_case': 131072, 'dtype': 'float64'}, 'Sampling contract changed')
    require(protocol['resources'] == LIMITS and protocol['outputs'] == OUTPUTS, 'Geometry resources or output locations changed')
    require(protocol['interpreter'] == 'runtime/envs/grasp_eval/Scripts/python.exe'
        and store.safe(root, protocol['interpreter']) == Path(sys.executable).resolve(), 'Wrong synthetic CPU interpreter')
    require(protocol['runtime'] == {'python': sys.version.split()[0], 'numpy': np.__version__,
        'scipy': importlib.metadata.version('scipy')}, 'Synthetic numerical runtime changed')
    require('torch' not in sys.modules, 'Geometry must not import Torch')
    required_sources = {SOURCE, 'scripts/geometry_storage_v19.py', 'src/geometry_propagation_v19.py',
        'tests/test_geometry_propagation_storage_v19.py', 'tests/test_geometry_propagation_v19.py',
        'scripts/multiview_common_v14r1.py', 'docs/第十九轮三维传播合成验证草案.txt'}
    require(required_sources <= set(protocol['source_hashes']) and protocol['input_hashes'], 'Incomplete geometry source/input freeze')
    for field in ('source_hashes', 'input_hashes'):
        for name, checksum in protocol[field].items():
            require(isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum)
                and store.digest(store.safe(root, name)) == checksum, 'Frozen geometry source/input changed: '+name)
    for name in (protocol['interpreter'], 'runtime/envs/grasp_eval/pyvenv.cfg'):
        require(name in protocol['input_hashes'], 'Interpreter omitted from frozen inputs')
    base = Path(protocol['base_python']['path'])
    require(base.is_absolute() and base == Path(sys._base_executable).resolve()
        and store.digest(base) == protocol['base_python']['sha256'], 'External base interpreter changed')
    require(all(name in protocol['budget_paths'] for name in (OUTPUTS['root'], OUTPUTS['report'], OUTPUTS['report']+'.sha256.json', OUTPUTS['logs'])), 'Geometry output budget omissions')
    require(store.read(config) == protocol, 'Geometry protocol changed during read')
    body = {'schema': 19, 'protocol_sha256': store.digest(config), 'source_hashes': protocol['source_hashes'],
            'input_hashes': protocol['input_hashes'], 'runtime': protocol['runtime'], 'base_python': protocol['base_python']}
    return dict(body, fingerprint=store.fingerprint(body))


def numerical_batch(case, normals):
    result = {'modes': {}, 'mismatch': None}
    values = {mode: geometry.sample_diagnostics(case['nominal'], case['intrinsic'], case['input_covariance'],
                case['standard_deviation'], normals, mode) for mode in MODES}
    for mode, sampled in values.items():
        reference = sampled['reference']; row = {'nonpositive_depth_count': sampled['nonpositive_depth_count'],
            'depth_above_two_metres_count': sampled['depth_above_two_metres_count']}
        for form, prefix in (('linear', 'first_order'), ('exact', 'exact')):
            row[form] = {'statistics': sampled[form+'_statistics'], 'coverage': geometry.coverage_diagnostics(
                sampled[form+'_points'], reference[prefix+'_mean'], reference[prefix+'_covariance'], role=form)}
        result['modes'][mode] = row
    diagonal = values['diagonal']['reference']
    result['mismatch'] = geometry.coverage_diagnostics(values['full']['linear_points'],
        diagonal['first_order_mean'], diagonal['first_order_covariance'], role='mismatch')
    return plain(result)


def combined_coverage(rows, role):
    require(len({row['numerical_rank'] for row in rows}) == 1, 'Coverage ranks differ between batches')
    counts = [sum(row['checks'][i]['interval']['successes'] for row in rows) for i in range(len(rows[0]['checks']))]
    return geometry.coverage_from_counts(counts, sum(row['sample_count'] for row in rows), rows[0]['numerical_rank'],
        sum(row['off_support_count'] for row in rows), max(row['maximum_nullspace_residual_metres'] for row in rows), role=role)


def summarize_case(case, batches):
    require(len(batches) == 16 and [row['batch_index'] for row in batches] == list(range(16)), 'Case requires all sixteen ordered batches')
    result = {'schema': 19, 'case_id': case['case_id'], 'modes': {}, 'passed': True,
              'production_samples_required': 131072, 'nonlinear_and_mismatch_are_diagnostics_only': True}
    for mode in MODES:
        reference = geometry.propagate_gaussian(case['nominal'], case['intrinsic'], case['input_covariance'], case['standard_deviation'], mode)
        rows = [batch['values']['modes'][mode] for batch in batches]
        value = {'reference': reference, 'nonpositive_depth_count': sum(row['nonpositive_depth_count'] for row in rows),
                 'depth_above_two_metres_count': sum(row['depth_above_two_metres_count'] for row in rows)}
        for form, prefix in (('linear', 'first_order'), ('exact', 'exact')):
            summary = geometry.aggregate_batches([row[form]['statistics'] for row in rows])
            moments = geometry.moment_tolerance(summary, reference[prefix+'_mean'], reference[prefix+'_covariance'])
            coverage = combined_coverage([row[form]['coverage'] for row in rows], form)
            value[form] = {'summary': summary, 'moments': moments, 'coverage': coverage}
            result['passed'] = result['passed'] and moments['contract_gate_passed'] is True
            if form == 'linear': result['passed'] = result['passed'] and coverage['contract_gate_passed'] is True
        result['passed'] = result['passed'] and value['nonpositive_depth_count'] == 0
        result['modes'][mode] = value
    result['mismatch_full_linear_with_diagonal_covariance'] = combined_coverage([row['values']['mismatch'] for row in batches], 'mismatch')
    return plain(result)


def deterministic_checks(cases):
    nodes = np.array([-np.sqrt(3.), 0., np.sqrt(3.)]); weights = np.array([1/6, 2/3, 1/6])
    choices = list(itertools.product(range(3), repeat=3))
    normals = np.array([[nodes[i] for i in indices] for indices in choices])
    mass = np.array([np.prod(weights[list(indices)]) for indices in choices])
    rows = []
    for case in cases:
        jacobian = geometry.jacobian_diagnostics(case['nominal'], case['intrinsic'])
        row = {'case_id': case['case_id'], 'jacobian': jacobian, 'moments': {}, 'passed': jacobian['passed']}
        for mode in MODES:
            sample = geometry.sample_diagnostics(case['nominal'], case['intrinsic'], case['input_covariance'], case['standard_deviation'], normals, mode)
            checks = {}
            for form, prefix in (('linear', 'first_order'), ('exact', 'exact')):
                points = sample[form+'_points']; mean = mass @ points; centered = points-mean
                covariance = np.einsum('n,ni,nj->ij', mass, centered, centered)
                reference = sample['reference']; delta_mean = mean-reference[prefix+'_mean']; delta_covariance = covariance-reference[prefix+'_covariance']
                passed = np.all(np.abs(delta_mean) <= 1e-12+1e-12*np.abs(reference[prefix+'_mean'])) and np.all(
                    np.abs(delta_covariance) <= 1e-14+1e-10*np.abs(reference[prefix+'_covariance']))
                checks[form] = {'mean_difference': delta_mean, 'covariance_difference': delta_covariance, 'passed': bool(passed)}
                row['passed'] = row['passed'] and bool(passed)
            row['moments'][mode] = checks
        rows.append(row)
    intrinsic = geometry.enumerate_cases()[0]['intrinsic']; nominal = [319.5, 239.5, 1.]
    ranks = []
    for vector, expected in (([0, 0, 0], 0), ([1, 0, 0], 1), ([0, 1, 0], 1), ([0, 0, .01], 1), ([1, 0, .01], 2), ([0, 1, .01], 2)):
        vector = np.array(vector, dtype=np.float64); sigma = np.outer(vector, vector)
        reference = geometry.propagate_gaussian(nominal, intrinsic, sigma, np.abs(vector))
        delta = np.array([-3., -1., 0., 1., 3.])[:, None]*vector
        linear = reference['nominal_output']+delta @ reference['jacobian'].T
        exact = geometry.backproject(np.array(nominal)+delta, intrinsic)
        supported = all(np.all(geometry.support_distance(points, reference[prefix+'_mean'], reference[prefix+'_covariance'])['in_support'])
            for points, prefix in ((linear, 'first_order'), (exact, 'exact')))
        ranks.append({'vector': vector, 'expected_exact_rank': expected, 'actual_exact_rank': reference['exact_spectrum']['numerical_rank'],
            'passed': bool(supported and reference['exact_spectrum']['numerical_rank'] == expected)})
    return plain({'schema': 19, 'case_checks': rows, 'rank_checks': ranks, 'passed': all(row['passed'] for row in rows+ranks),
        'new_random_draw_calls': 0, 'scope': 'deterministic_degree_four_Gaussian_quadrature_and_support_checks'})


def case_directory(root, protocol, case):
    return store.safe(root, protocol['outputs']['root'])/'cases'/case['case_id']


def attempts(directory):
    result = sorted((directory/'attempts').iterdir()) if (directory/'attempts').exists() else []
    require(all(path.is_dir() and re.fullmatch(r'attempt_[0-9]{6}', path.name) for path in result), 'Unknown worker attempt')
    require([int(path.name[8:]) for path in result] == list(range(1, len(result)+1)), 'Worker attempt sequence has gaps')
    return result


def case_payloads(directory, protocol):
    paths = [directory/name for name in ('summary.json', 'report.json', 'completion.json', 'failure.json')]
    for index in range(protocol['sampling']['batch_count']):
        paths += [directory/('batches/batch_%02d'%index)/name for name in ('z.npy', 'statistics.json')]
    for attempt in attempts(directory):
        paths += [attempt/name for name in ('intent.json', 'worker_started.json', 'process_outcome.json', 'worker_result.json')]
        for index in range(protocol['sampling']['batch_count']):
            paths += [attempt/('draws/batch_%02d'%index)/name for name in ('intent.json', 'outcome.json')]
    return paths


def validate_worker_result(row, protocol, case_id):
    require(row['schema'] == 19 and row['status'] == 'case_worker_completed' and row['case_id'] == case_id,
            'Worker result identity differs')
    count = row['new_random_draw_calls']
    require(type(count) is int and 0 <= count <= protocol['sampling']['batch_count']
        and row['new_standard_normal_triples'] == count*protocol['sampling']['batch_size']
        and type(row['new_model_forward_calls']) is int and row['new_model_forward_calls'] == 0
        and type(row['new_training_updates']) is int and row['new_training_updates'] == 0, 'Worker work accounting differs')
    value = row['resources']; limits = protocol['resources']
    for name, cap in (('elapsed_seconds', limits['worker_timeout_seconds']), ('peak_rss_bytes', limits['rss_soft_bytes']),
                      ('current_output_bytes', limits['output_limit_bytes'])):
        require(type(value[name]) in (int, float) and np.isfinite(value[name]) and 0 <= value[name] <= cap, 'Worker exceeded resource contract')
    require(value['peak_gpu_allocated_bytes'] == value['peak_gpu_reserved_bytes'] == 0, 'Worker reported GPU allocation')


def draw_accounting(root, directory, protocol, binding):
    returned = []; unknown = 0; unobserved_processes = 0
    for attempt in attempts(directory):
        ordinal = int(attempt.name[8:]); case_id = directory.name; attempt_returned = 0; attempt_unknown = 0
        if (attempt/'intent.json').exists():
            intent = store.load(attempt/'intent.json', binding)
            require(set(intent) == {'schema', 'case_id', 'attempt', 'command', 'stdout', 'stderr'} and intent['schema'] == 19
                and intent['case_id'] == case_id and intent['attempt'] == ordinal
                and isinstance(intent['command'], list) and all(isinstance(part, str) for part in intent['command'])
                and intent['command'][-4:] == ['--worker-case', case_id, '--attempt', str(ordinal)], 'Parent worker intent differs')
            for kind in ('stdout', 'stderr'):
                require(intent[kind] == protocol['outputs']['logs']+'/'+case_id+'_'+attempt.name+'.'+kind+'.log', 'Worker log path differs')
        if (attempt/'worker_started.json').exists():
            require(store.load(attempt/'worker_started.json', binding) == {'schema': 19, 'case_id': case_id, 'attempt': ordinal}, 'Worker-start identity differs')
        if (attempt/'process_outcome.json').exists():
            outcome = store.load(attempt/'process_outcome.json', binding)
            require(outcome['schema'] == 19 and outcome['case_id'] == case_id and outcome['attempt'] == ordinal
                and outcome['status'] in ('exit_observed', 'timeout_killed', 'process_error')
                and (type(outcome['returncode']) is int if outcome['status'] == 'exit_observed' else outcome['returncode'] is None), 'Worker exit identity differs')
            if outcome['status'] != 'exit_observed': unobserved_processes += 1
            if outcome['status'] in ('exit_observed', 'timeout_killed'):
                require(set(outcome['logs']) == {'stdout', 'stderr'}, 'Observed process logs are missing')
                for kind, record in outcome['logs'].items():
                    require(record['path'] == intent[kind] and store.state(store.safe(root, record['path'])) == record['state'], 'Completed process log changed')
            else: require(outcome['logs'] is None, 'Unobserved process logs must not be falsely sealed')
        else: unobserved_processes += 1
        for index in range(protocol['sampling']['batch_count']):
            path = attempt/('draws/batch_%02d'%index)
            if not path.exists(): continue
            expected = {'schema': 19, 'case_id': case_id, 'batch_index': index,
                'seed': geometry.batch_seed(case_id, index), 'sample_count': protocol['sampling']['batch_size'],
                'generator': 'numpy.RandomState.standard_normal'}
            if (path/'intent.json').exists(): require(store.load(path/'intent.json', binding) == expected, 'Draw intent differs')
            if (path/'outcome.json').exists():
                row = store.load(path/'outcome.json', binding)
                require((path/'intent.json').exists() and set(row) == set(expected)|{'status', 'z_sha256', 'z_bytes'}
                    and all(row[key] == value for key, value in expected.items()) and row['status'] == 'returned'
                    and isinstance(row['z_sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['z_sha256'])
                    and type(row['z_bytes']) is int and row['z_bytes'] > 24*row['sample_count'], 'Invalid returned draw evidence')
                returned.append(row); attempt_returned += 1
            else: unknown += 1; attempt_unknown += 1
        if (attempt/'worker_result.json').exists():
            row = store.load(attempt/'worker_result.json', binding); validate_worker_result(row, protocol, case_id)
            require(row['new_random_draw_calls'] == attempt_returned and attempt_unknown == 0, 'Worker counters differ from returned draw ledger')
    uncertain = unknown+unobserved_processes > 0
    return {'worker_attempts': len(attempts(directory)), 'returned_random_draw_calls': len(returned),
        'unknown_random_draw_calls': unknown, 'unobserved_worker_exits': unobserved_processes,
        'exact_random_draw_calls': None if uncertain else len(returned),
        'known_standard_normal_triples_lower_bound': sum(row['sample_count'] for row in returned),
        'exact_standard_normal_triples': None if uncertain else sum(row['sample_count'] for row in returned)}, returned


def load_normals(path, binding, size):
    store.verify_bytes(path, binding)
    with Path(path).open('rb') as stream: values = np.load(stream, allow_pickle=False)
    require(values.dtype == np.float64 and values.shape == (size, 3) and values.flags.c_contiguous and np.isfinite(values).all(), 'Invalid cached Z')
    return values


def batch_record(case, index, path, normals):
    return {'schema': 19, 'case_id': case['case_id'], 'batch_index': index, 'seed': geometry.batch_seed(case['case_id'], index),
            'z_sha256': store.digest(path), 'sample_count': len(normals), 'values': numerical_batch(case, normals)}


def verify_case(root, directory, case, protocol, binding, recompute=False):
    store.validate_layout(directory, case_payloads(directory, protocol), binding, readonly=True)
    completion = store.load(directory/'completion.json', binding)
    files = store.snapshot(directory)
    expected = {name: row['sha256'] for name, row in files.items() if name not in ('completion.json', 'completion.json.sha256.json')}
    require(completion == {'schema': 19, 'case_id': case['case_id'], 'binding_fingerprint': binding['fingerprint'], 'outputs': expected}, 'Case completion inventory changed')
    accounting, returned = draw_accounting(root, directory, protocol, binding); batches = []
    for index in range(protocol['sampling']['batch_count']):
        path = directory/('batches/batch_%02d'%index)/'z.npy'
        normals = load_normals(path, binding, protocol['sampling']['batch_size'])
        seed = geometry.batch_seed(case['case_id'], index)
        require(any(row['batch_index'] == index and row['seed'] == seed and row['z_sha256'] == store.digest(path) for row in returned), 'Z lacks returned-call provenance')
        row = store.load(path.parent/'statistics.json', binding)
        require(row['schema'] == 19 and row['case_id'] == case['case_id'] and row['batch_index'] == index
            and row['seed'] == seed and row['z_sha256'] == store.digest(path) and row['sample_count'] == len(normals), 'Batch statistics identity differs')
        if recompute: require(row == batch_record(case, index, path, normals), 'Cached-Z statistical recount differs')
        batches.append(row)
    summary = summarize_case(case, batches)
    require(store.load(directory/'summary.json', binding) == summary, 'Case summary differs')
    report = store.load(directory/'report.json', binding)
    require(report == {'schema': 19, 'case_id': case['case_id'], 'passed': summary['passed'],
        'summary_sha256': store.digest(directory/'summary.json'), 'draw_accounting': accounting}, 'Case accounting report differs')
    require(store.snapshot(directory) == files, 'Read-only case verification changed SHA, size or mtime')
    return report


def finalize_case(root, directory, case, protocol, binding):
    # A live orphan worker must release its lock before the parent seals this case.
    with store.lock(root/'runtime/locks/geometry_propagation_v19_workers'/case['case_id']):
        store.validate_layout(directory, case_payloads(directory, protocol), binding)
        summary = store.load(directory/'summary.json', binding)
        accounting, _ = draw_accounting(root, directory, protocol, binding)
        store.commit(directory/'report.json', {'schema': 19, 'case_id': case['case_id'], 'passed': summary['passed'],
            'summary_sha256': store.digest(directory/'summary.json'), 'draw_accounting': accounting}, binding)
        files = store.snapshot(directory)
        store.commit(directory/'completion.json', {'schema': 19, 'case_id': case['case_id'], 'binding_fingerprint': binding['fingerprint'],
            'outputs': {name: row['sha256'] for name, row in files.items() if name not in ('completion.json', 'completion.json.sha256.json')}}, binding)
    return verify_case(root, directory, case, protocol, binding)


def worker(root, protocol, binding, case, attempt_index):
    directory = case_directory(root, protocol, case); attempt = directory/('attempts/attempt_%06d'%attempt_index)
    guard = store.Guard(root, protocol, worker=True)
    with store.lock(root/'runtime/locks/geometry_propagation_v19_workers'/case['case_id']):
        require(attempts(directory) and attempt == attempts(directory)[-1] and (attempt/'intent.json').exists()
            and not (directory/'completion.json').exists(), 'Worker lacks the latest pending parent intent')
        intent = store.load(attempt/'intent.json', binding)
        require(intent['schema'] == 19 and intent['case_id'] == case['case_id'] and intent['attempt'] == attempt_index
            and intent['command'][-4:] == ['--worker-case', case['case_id'], '--attempt', str(attempt_index)], 'Worker parent intent differs')
        store.validate_layout(directory, case_payloads(directory, protocol), binding)
        require(not any(path.exists() for path in (attempt/'worker_started.json', store.receipt(attempt/'worker_started.json'),
            attempt/'draws', attempt/'worker_result.json', attempt/'process_outcome.json')), 'Worker attempt cannot be executed twice')
        store.commit(attempt/'worker_started.json', {'schema': 19, 'case_id': case['case_id'], 'attempt': attempt_index}, binding)
        require(not (directory/'failure.json').exists(), 'Prior range failure must be retained without new sampling')
        batches = []; new_calls = 0
        for index in range(protocol['sampling']['batch_count']):
            guard.check(); path = directory/('batches/batch_%02d'%index)/'z.npy'; seed = geometry.batch_seed(case['case_id'], index)
            if not path.exists():
                draw = attempt/('draws/batch_%02d'%index)
                intent = {'schema': 19, 'case_id': case['case_id'], 'batch_index': index, 'seed': seed,
                          'sample_count': protocol['sampling']['batch_size'], 'generator': 'numpy.RandomState.standard_normal'}
                store.commit(draw/'intent.json', intent, binding)
                normals = np.random.RandomState(seed).standard_normal((protocol['sampling']['batch_size'], 3))
                new_calls += 1
                stream = io.BytesIO(); np.save(stream, normals, allow_pickle=False); payload = stream.getvalue()
                outcome = dict(intent, status='returned', z_sha256=hashlib.sha256(payload).hexdigest(), z_bytes=len(payload))
                store.commit(draw/'outcome.json', outcome, binding); guard.check(len(payload)+4096)
                store.commit_bytes(path, payload, binding)
            normals = load_normals(path, binding, protocol['sampling']['batch_size'])
            row = batch_record(case, index, path, normals)
            guard.check(len(store.json_bytes(row))+4096); store.commit(path.parent/'statistics.json', row, binding)
            batches.append(row)
            if any(row['values']['modes'][mode]['nonpositive_depth_count'] for mode in MODES):
                store.commit(directory/'failure.json', {'schema': 19, 'reason': 'nonpositive_depth_range_anomaly', 'batch_index': index}, binding)
                raise ValueError('Nonpositive depth sample retained; no subsequent batch may be drawn')
        summary = summarize_case(case, batches)
        guard.check(len(store.json_bytes(summary))+4096); store.commit(directory/'summary.json', summary, binding)
        result = {'schema': 19, 'status': 'case_worker_completed', 'case_id': case['case_id'],
            'new_random_draw_calls': new_calls, 'new_standard_normal_triples': new_calls*protocol['sampling']['batch_size'],
            'new_model_forward_calls': 0, 'new_training_updates': 0, 'resources': guard.check()}
        store.commit(attempt/'worker_result.json', result, binding)
        return result


def launch(root, config, protocol, binding, case, guard):
    directory = case_directory(root, protocol, case)
    with store.lock(root/'runtime/locks/geometry_propagation_v19_workers'/case['case_id']):
        # An orphan may have completed since the driver's earlier inspection.
        if (directory/'summary.json').exists() and any((path/'worker_result.json').exists() for path in attempts(directory)):
            return {'status': 'adopt_completed_worker', 'new_random_draw_calls': 0}
        index = len(attempts(directory))+1; attempt = directory/('attempts/attempt_%06d'%index)
        command = [str(store.safe(root, protocol['interpreter'])), '-B', str(root/SOURCE), '--config', str(config),
                   '--worker-case', case['case_id'], '--attempt', str(index)]
        logs = {kind: protocol['outputs']['logs']+'/'+case['case_id']+'_'+attempt.name+'.'+kind+'.log' for kind in ('stdout', 'stderr')}
        store.commit(attempt/'intent.json', dict(schema=19, case_id=case['case_id'], attempt=index, command=command, **logs), binding)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): environment[key] = '4'
    guard.check(65536)
    outcome = {'schema': 19, 'case_id': case['case_id'], 'attempt': index, 'status': 'exit_observed', 'returncode': None, 'logs': None}
    try:
        store.safe(root, protocol['outputs']['logs']).mkdir(parents=True, exist_ok=True)
        with store.safe(root, logs['stdout']).open('xb', buffering=0) as stdout, store.safe(root, logs['stderr']).open('xb', buffering=0) as stderr:
            completed = subprocess.run(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, timeout=protocol['resources']['worker_timeout_seconds'])
            os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
        outcome['returncode'] = completed.returncode
    except subprocess.TimeoutExpired:
        outcome['status'] = 'timeout_killed'
    except OSError as error:
        outcome.update(status='process_error', error=str(error))
    if outcome['status'] in ('exit_observed', 'timeout_killed'):
        outcome['logs'] = {kind: {'path': name, 'state': store.state(store.safe(root, name))} for kind, name in logs.items()}
    store.commit(attempt/'process_outcome.json', outcome, binding)
    require(outcome['status'] == 'exit_observed' and outcome['returncode'] == 0, 'Case worker stopped; retain attempt and caches')
    result = store.load(attempt/'worker_result.json', binding)
    validate_worker_result(result, protocol, case['case_id'])
    guard.check()
    return result


def driver(root, config, protocol, binding, max_new_cases=None, check_only=False):
    directory = store.safe(root, protocol['outputs']['root']); output = store.safe(root, protocol['outputs']['report'])
    readonly = check_only or (directory/'completion.json').exists()
    require(not check_only or (directory/'completion.json').exists(), 'Check-only requires a completed experiment')
    guard = store.Guard(root, protocol); guard.check()
    canonical = [directory/name for name in ('binding.json', 'deterministic.json', 'completion.json')]
    for case in protocol['cases']: canonical += case_payloads(case_directory(root, protocol, case), protocol)
    # Never adopt or remove a temporary while an orphan worker can be writing it.
    # Locks are released before launch; creating each new intent takes its case lock again.
    with ExitStack() as locks:
        if not readonly:
            for case in protocol['cases']:
                locks.enter_context(store.lock(root/'runtime/locks/geometry_propagation_v19_workers'/case['case_id']))
        store.validate_layout(directory, canonical, binding, readonly)
        store.recover(output, binding, readonly)
    before = store.snapshot(directory) if readonly else None
    external_before = {str(path): store.state(path) for path in (output, store.receipt(output))} if readonly else None
    store.commit(directory/'binding.json', binding, binding, readonly)
    deterministic = deterministic_checks(protocol['cases'])
    store.commit(directory/'deterministic.json', deterministic, binding, readonly)
    require(deterministic['passed'], 'Deterministic geometry checks failed')
    reports = []; new_cases = 0; new_calls = 0
    for case in protocol['cases']:
        guard.check(); target = case_directory(root, protocol, case)
        if (target/'completion.json').exists(): report = verify_case(root, target, case, protocol, binding, recompute=readonly)
        else:
            require(not readonly, 'Completed experiment is missing a case')
            if max_new_cases is not None and new_cases >= max_new_cases: break
            require(not (target/'failure.json').exists(), 'Stored range failure prevents automatic continuation')
            ready = (target/'summary.json').exists() and any((path/'worker_result.json').exists() for path in attempts(target))
            if not ready:
                result = launch(root, config, protocol, binding, case, guard); new_calls += result['new_random_draw_calls']
            report = finalize_case(root, target, case, protocol, binding); new_cases += 1
        reports.append(report)
        require(report['passed'], 'Case statistical gate failed; retain result and stop without resampling')
    result = {'schema': 19, 'new_cases': new_cases, 'committed_cases': len(reports), 'expected_cases': len(protocol['cases']),
        'new_random_draw_calls': new_calls, 'new_model_forward_calls': 0, 'new_training_updates': 0}
    if len(reports) != len(protocol['cases']): return dict(result, status='partial', resources=guard.check())
    summary = {'schema': 19, 'status': 'completed_synthetic_geometry_cases', 'binding': binding,
        'deterministic_checks_passed': True, 'statistical_checks_passed': True, 'case_count': len(reports), 'cases': reports,
        'full_p6_completion': False, 'independent_acceptance_pending': True, 'physical_power_loss_tested': False}
    guard.check(len(store.json_bytes(summary))+4096); store.commit(output, summary, binding, readonly)
    files = store.snapshot(directory)
    completion = {'schema': 19, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
        'outputs': {name: row['sha256'] for name, row in files.items() if name not in ('completion.json', 'completion.json.sha256.json')},
        'report_sha256': store.digest(output)}
    store.commit(directory/'completion.json', completion, binding, readonly)
    if readonly:
        require(store.snapshot(directory) == before and {str(path): store.state(path) for path in (output, store.receipt(output))} == external_before,
                'Completed geometry check changed SHA, size or mtime')
    return dict(result, status='verified_completed_read_only' if readonly else 'completed', report_sha256=store.digest(output), resources=guard.check())


def run(root=ROOT, config=CONFIG, max_new_cases=None, check_only=False, worker_case=None, attempt=None):
    root = Path(root).resolve(); config = store.safe(root, config); protocol = store.read(config)
    binding = validate(root, config, protocol)
    require(max_new_cases is None or type(max_new_cases) is int and 0 <= max_new_cases <= 72, 'Invalid new-case limit')
    require(not check_only or max_new_cases is None, 'Read-only cannot request new cases')
    if worker_case is not None:
        require(not check_only and max_new_cases is None and type(attempt) is int and attempt > 0, 'Invalid worker invocation')
        selected = [case for case in protocol['cases'] if case['case_id'] == worker_case]
        require(len(selected) == 1, 'Unknown worker case')
        return worker(root, protocol, binding, selected[0], attempt)
    require(attempt is None, 'Attempt index requires one worker case')
    if check_only or (store.safe(root, protocol['outputs']['root'])/'completion.json').exists():
        return driver(root, config, protocol, binding, max_new_cases, check_only)
    with store.lock(root/'runtime/locks/geometry_propagation_v19_driver'):
        return driver(root, config, protocol, binding, max_new_cases, check_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--max-new-cases', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker-case')
    parser.add_argument('--attempt', type=int)
    args = parser.parse_args()
    print(json.dumps(run(config=args.config, max_new_cases=args.max_new_cases, check_only=args.check_only,
        worker_case=args.worker_case, attempt=args.attempt), ensure_ascii=False, allow_nan=False), flush=True)
