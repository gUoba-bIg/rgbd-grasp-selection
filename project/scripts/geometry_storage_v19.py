"""Schema-19 immutable receipts and bounded storage for synthetic geometry."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

from scripts import multiview_common_v14r1 as legacy

require = legacy.require
read = legacy.read
digest = legacy.digest
json_bytes = legacy.json_bytes
fingerprint = legacy.fingerprint
lock = legacy.lock


def safe(root, name):
    root = Path(root).resolve(); value = Path(name)
    value = value.resolve() if value.is_absolute() else (root/value).resolve()
    require(value != root and value.is_relative_to(root), 'Geometry path escaped project')
    return value


def receipt(path):
    return Path(str(path)+'.sha256.json')


def state(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing file or link in geometry evidence')
    return {'sha256': digest(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}


def snapshot(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'Link in geometry inventory')
        if path.is_file(): result[path.relative_to(directory).as_posix()] = state(path)
    return result


def verify_bytes(path, binding):
    path = Path(path)
    expected = {'schema': 19, 'binding_fingerprint': binding['fingerprint'],
                'sha256': digest(path), 'bytes': path.stat().st_size}
    require(read(receipt(path)) == expected, 'Geometry receipt differs: '+str(path))
    return expected


def load(path, binding):
    verify_bytes(path, binding)
    return read(path)


def recover(path, binding, readonly=False):
    """Adopt an intact receipt-bound temporary; only exact owned names may be removed."""
    path = Path(path)
    for target in (receipt(path), path):
        candidates = list(target.parent.glob(target.name+'.*.tmp'))
        for temporary in candidates:
            require(re.fullmatch(re.escape(target.name)+r'\.[0-9a-f]{32}\.tmp', temporary.name)
                    and not temporary.is_symlink() and not readonly, 'Unknown or read-only temporary: '+str(temporary))
        if not target.exists() and candidates:
            for temporary in candidates:
                if target == receipt(path):
                    try: value = read(temporary)
                    except (ValueError, OSError): continue
                    if (set(value) == {'schema', 'binding_fingerprint', 'sha256', 'bytes'}
                        and value['schema'] == 19 and value['binding_fingerprint'] == binding['fingerprint']
                        and isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256'])
                        and type(value['bytes']) is int and value['bytes'] >= 0):
                        os.replace(temporary, target); break
                elif receipt(path).exists():
                    expected = read(receipt(path))
                    require(expected['schema'] == 19 and expected['binding_fingerprint'] == binding['fingerprint'], 'Temporary binding differs')
                    if temporary.stat().st_size == expected['bytes'] and digest(temporary) == expected['sha256']:
                        os.replace(temporary, target); break
        for temporary in candidates:
            if temporary.exists(): temporary.unlink()


def commit_bytes(path, payload, binding, readonly=False):
    path = Path(path)
    record = {'schema': 19, 'binding_fingerprint': binding['fingerprint'],
              'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    legacy.once(receipt(path), record, readonly)
    legacy.publish_once(path, payload, readonly)


def commit(path, value, binding, readonly=False):
    commit_bytes(path, json_bytes(value), binding, readonly)


def validate_layout(directory, canonical, binding, readonly=False):
    """Only listed payloads, their receipts and their UUID atomic temporaries exist."""
    directory = Path(directory)
    canonical = {Path(path) for path in canonical}
    allowed = canonical | {receipt(path) for path in canonical}
    parents = {directory}
    for path in allowed:
        parents.update(parent for parent in path.parents if parent == directory or directory in parent.parents)
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'Link in geometry output')
        if path.is_dir():
            require(path in parents, 'Unknown geometry directory: '+str(path))
        elif path not in allowed:
            require(any(path.parent == target.parent and re.fullmatch(re.escape(target.name)+r'\.[0-9a-f]{32}\.tmp', path.name)
                        for target in allowed), 'Unknown geometry file: '+str(path))
    for path in sorted(canonical): recover(path, binding, readonly)


class Guard:
    def __init__(self, root, protocol, worker=False):
        self.root, self.protocol, self.worker = Path(root), protocol, worker
        self.started = time.perf_counter()

    def check(self, extra=0):
        limits = self.protocol['resources']; files = set()
        require(type(extra) is int and extra >= 0, 'Invalid output reservation')
        for name in self.protocol['budget_paths']:
            path = safe(self.root, name)
            if path.is_file(): files.add(path)
            elif path.exists(): files.update(p for p in path.rglob('*') if p.is_file())
            if not path.is_dir(): files.update(p for p in path.parent.glob(path.name+'.*.tmp') if p.is_file())
        value = {'elapsed_seconds': time.perf_counter()-self.started, 'peak_rss_bytes': legacy.peak_memory_bytes(),
                 'current_output_bytes': sum(path.stat().st_size for path in files),
                 'peak_gpu_allocated_bytes': 0, 'peak_gpu_reserved_bytes': 0}
        require('torch' not in sys.modules, 'Geometry process must not import Torch')
        require(value['elapsed_seconds'] <= limits['worker_timeout_seconds' if self.worker else 'driver_soft_seconds'], 'Geometry time budget exceeded')
        require(value['peak_rss_bytes'] <= limits['rss_soft_bytes'], 'Geometry RSS budget exceeded')
        require(value['current_output_bytes']+extra <= limits['output_limit_bytes'], 'Geometry output budget exceeded')
        require(shutil.disk_usage(self.root).free-extra >= limits['disk_reserve_bytes'], 'Geometry disk reserve exhausted')
        return value


def completed_context(root, config):
    """Read-only provenance/seal verification; no numerical core or RNG import."""
    root = Path(root).resolve(); config = safe(root, config); protocol = read(config)
    require(protocol['schema'] == 19 and protocol['status'] == 'frozen'
        and protocol['phase'] == 'synthetic_geometry_propagation', 'Require frozen synthetic geometry protocol')
    require(protocol['sampling'] == {'generator': 'numpy.RandomState.standard_normal', 'master_seed': 20261027,
        'case_count': 72, 'batch_count': 16, 'batch_size': 8192, 'samples_per_case': 131072, 'dtype': 'float64'}, 'Frozen sampling support differs')
    require([case['case_id'] for case in protocol['cases']] == ['case_%03d'%i for i in range(72)], 'Wrong ordered case cohort')
    require(safe(root, protocol['interpreter']) == Path(sys.executable).resolve(), 'Wrong synthetic verification interpreter')
    require(protocol['runtime'] == {'python': sys.version.split()[0], 'numpy': importlib.metadata.version('numpy'),
        'scipy': importlib.metadata.version('scipy')}, 'Frozen numerical runtime changed')
    source_state = {config.relative_to(root).as_posix(): state(config)}
    for field in ('source_hashes', 'input_hashes'):
        require(protocol[field], 'Geometry maps have not been frozen')
        for name, checksum in protocol[field].items():
            source_state[name] = state(safe(root, name))
            require(source_state[name]['sha256'] == checksum, 'Geometry source/input changed: '+name)
    base = Path(protocol['base_python']['path'])
    source_state['external_base_python'] = state(base)
    require(base.is_absolute() and base == Path(sys._base_executable).resolve()
        and source_state['external_base_python']['sha256'] == protocol['base_python']['sha256'], 'Base executable changed')
    body = {'schema': 19, 'protocol_sha256': digest(config), 'source_hashes': protocol['source_hashes'],
        'input_hashes': protocol['input_hashes'], 'runtime': protocol['runtime'], 'base_python': protocol['base_python']}
    binding = dict(body, fingerprint=fingerprint(body))
    directory = safe(root, protocol['outputs']['root']); tree = snapshot(directory)
    require(load(directory/'binding.json', binding) == binding, 'Geometry binding changed')
    output = safe(root, protocol['outputs']['report'])
    completion = load(directory/'completion.json', binding)
    require(completion == {'schema': 19, 'status': 'completed', 'binding_fingerprint': binding['fingerprint'],
        'outputs': {name: row['sha256'] for name, row in tree.items() if name not in ('completion.json', 'completion.json.sha256.json')},
        'report_sha256': digest(output)}, 'Geometry root seal differs')
    report = load(output, binding)
    require(report['schema'] == 19 and report['status'] == 'completed_synthetic_geometry_cases'
        and report['binding'] == binding and report['case_count'] == 72 and len(report['cases']) == 72
        and report['deterministic_checks_passed'] is True and report['statistical_checks_passed'] is True
        and [row['case_id'] for row in report['cases']] == [case['case_id'] for case in protocol['cases']]
        and all(row['passed'] is True for row in report['cases']), 'Geometry case reports are incomplete')
    log_state = {}; unsealed_logs = []
    for case in protocol['cases']:
        attempts = directory/'cases'/case['case_id']/'attempts'
        for attempt in sorted(attempts.iterdir()):
            if not (attempt/'intent.json').exists(): continue
            intent = load(attempt/'intent.json', binding)
            outcome = load(attempt/'process_outcome.json', binding) if (attempt/'process_outcome.json').exists() else None
            for kind in ('stdout', 'stderr'):
                name = intent[kind]; path = safe(root, name)
                require(path.parent == safe(root, protocol['outputs']['logs']), 'Child log escaped declared directory')
                if outcome is not None and outcome['status'] in ('exit_observed', 'timeout_killed'):
                    record = outcome['logs'][kind]
                    require(record['path'] == name and state(path) == record['state'], 'Sealed worker log changed')
                    log_state[name] = record['state']
                else:
                    unsealed_logs.append(name)
    require(snapshot(directory) == tree, 'Geometry seal verification changed files')
    return {'protocol': protocol, 'config': config, 'binding': binding, 'directory': directory,
        'report': report, 'sealedtree': tree, 'source_state': source_state, 'log_state': log_state,
        'unsealed_log_paths': sorted(set(unsealed_logs)),
        'report_state': {path.relative_to(root).as_posix(): state(path) for path in (output, receipt(output))}}
