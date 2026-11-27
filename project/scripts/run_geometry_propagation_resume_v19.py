"""Record actual two-case restart and completed read-only process boundaries."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import geometry_storage_v19 as store
from scripts import run_geometry_propagation_v19 as driver
from scripts.pipeline import project_environment

CONFIG = driver.CONFIG
SOURCE = 'scripts/run_geometry_propagation_resume_v19.py'
TEST = 'tests/test_geometry_propagation_wrapper_v19.py'
SCOPE = 'Actual process boundaries with two cases first; no physical power-cut claim'
require = store.require


def inventory(root, directory, report=None):
    result = {(directory/relative).relative_to(root).as_posix(): row
              for relative, row in store.snapshot(directory).items()}
    if report is not None:
        for path in (report, store.receipt(report)):
            result[path.relative_to(root).as_posix()] = store.state(path)
    return result


def prefix_unchanged(root, directory, prefix):
    current = inventory(root, directory)
    require(bool(prefix) and all(current.get(name) == row for name, row in prefix.items()),
            'Saved two-case prefix changed SHA, bytes or mtime')
    return {name: current[name] for name in prefix}


def unrecorded_probe_count(directory, report):
    require(not any(path.exists() for path in (directory/'completion.json',
        store.receipt(directory/'completion.json'), report, store.receipt(report))),
        'Terminal geometry evidence lacks a recorded two-case process boundary')
    allowed = {directory/'cases'/'case_000', directory/'cases'/'case_001'}
    candidates = list((directory/'cases').iterdir()) if (directory/'cases').exists() else []
    require(all(path in allowed and path.is_dir() and not path.is_symlink() for path in candidates),
            'Unrecorded case evidence extends beyond the two-case boundary')
    return sum((path/'completion.json').is_file() for path in allowed)


def validate_record(row, operation, protocol):
    require(row['schema'] == 19 and row['expected_cases'] == 72, 'Wrong geometry work identity')
    for key in ('new_cases', 'committed_cases', 'new_random_draw_calls', 'new_model_forward_calls', 'new_training_updates'):
        require(type(row[key]) is int and row[key] >= 0, 'Invalid geometry work counter')
    require(row['new_cases'] <= row['committed_cases'] <= 72
        and row['new_random_draw_calls'] <= row['new_cases']*16
        and row['new_model_forward_calls'] == row['new_training_updates'] == 0, 'Inconsistent geometry work accounting')
    if operation == 'normalization':
        require(row['status'] == 'partial' and row['committed_cases'] <= 2
            and row['new_cases'] == row['new_random_draw_calls'] == 0,
            'Probe normalization must not start a new case or random draw')
    elif operation == 'probe':
        require(row['status'] == 'partial' and row['committed_cases'] == 2, 'Probe must stop after exactly two cases')
    else:
        require(row['committed_cases'] == 72 and row['status'] in ('completed', 'verified_completed_read_only'),
                'Geometry cohort is incomplete')
    if operation == 'read_only' or row['status'] == 'verified_completed_read_only':
        require(row['status'] == 'verified_completed_read_only'
            and row['new_cases'] == row['new_random_draw_calls'] == 0, 'Read-only verification performed new work')
    value, limits = row['resources'], protocol['resources']
    for field, maximum in (('elapsed_seconds', limits['driver_soft_seconds']),
        ('peak_rss_bytes', limits['rss_soft_bytes']), ('current_output_bytes', limits['output_limit_bytes'])):
        require(isinstance(value[field], (int, float)) and 0 <= value[field] <= maximum, 'Geometry resource limit exceeded')
    require(value['peak_gpu_allocated_bytes'] == value['peak_gpu_reserved_bytes'] == 0, 'Geometry child used GPU memory')


def invoke(root, config, protocol, operation, flags):
    command = [str(store.safe(root, protocol['interpreter'])), '-B', str(root/driver.SOURCE), '--config', str(config)]+flags
    log = store.safe(root, protocol['outputs']['logs'])/('resume_'+operation+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    guard = store.Guard(root, protocol); guard.check(65536)
    environment = project_environment(root); environment['CUDA_VISIBLE_DEVICES'] = ''
    print('Synthetic geometry: '+operation, flush=True)
    with log.open('ab', buffering=0) as stream:
        stream.write(('\nCOMMAND '+json.dumps(command, ensure_ascii=False)+'\n').encode('utf-8'))
        offset = stream.tell()
        result = subprocess.run(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=protocol['resources']['driver_soft_seconds']+60)
        os.fsync(stream.fileno())
    require(result.returncode == 0, 'Geometry child stopped; inspect '+str(log))
    guard.check()
    with log.open('rb') as stream:
        stream.seek(offset); lines = stream.read().decode('utf-8').splitlines()
    for line in reversed(lines):
        try: row = json.loads(line)
        except ValueError: continue
        if isinstance(row, dict) and 'status' in row:
            validate_record(row, operation, protocol)
            return row
    raise RuntimeError('Geometry child has no current completion record')


def execute(root, config, protocol):
    directory = store.safe(root, protocol['outputs']['root'])
    report = store.safe(root, protocol['outputs']['report'])
    proof_file = store.safe(root, protocol['outputs']['resume_report'])
    binding = {'protocol_sha256': store.digest(config), 'wrapper_sha256': store.digest(root/SOURCE),
        'child_sha256': store.digest(root/driver.SOURCE),
        'interpreter_sha256': store.digest(store.safe(root, protocol['interpreter'])),
        'pipeline_sha256': store.digest(root/'scripts/pipeline.py')}
    proof = store.read(proof_file) if proof_file.exists() else {
        'schema': 19, 'status': 'in_progress', 'binding': binding, 'scope': SCOPE, 'entry': {}}
    require(proof['schema'] == 19 and proof['status'] in ('in_progress', 'passed')
        and proof['binding'] == binding and proof['scope'] == SCOPE, 'Geometry recovery proof binding changed')
    for temporary in proof_file.parent.glob(proof_file.name+'.*.tmp'):
        require(re.fullmatch(re.escape(proof_file.name)+r'\.[0-9a-f]{32}\.tmp', temporary.name)
            and not temporary.is_symlink() and proof['status'] != 'passed', 'Unexpected recovery-proof temporary')
        temporary.unlink()

    def save():
        store.Guard(root, protocol).check(len(store.json_bytes(proof))+1024)
        store.legacy.atomic(proof_file, proof)

    if not proof_file.exists(): save()
    entry = proof['entry']
    if proof['status'] == 'passed':
        before = inventory(root, directory, report)
        require(before == entry['completed_snapshot'], 'Completed geometry cache changed')
        invoke(root, config, protocol, 'read_only', ['--check-only'])
        require(inventory(root, directory, report) == before, 'Read-only geometry reentry changed files')
        return proof
    if 'prefix_before' not in entry:
        count = unrecorded_probe_count(directory, report)
        # The child first adopts complete atomic temporaries. Count only after that
        # adoption, otherwise a two-case probe could accidentally create case 003.
        pending = [directory/'cases'/('case_%03d'%index)/'completion.json' for index in range(2)]
        if any(not path.exists() and (store.receipt(path).exists()
            or any(path.parent.glob(path.name+'.*.tmp'))) for path in pending):
            entry['normalization_record'] = invoke(root, config, protocol, 'normalization', ['--max-new-cases', '0'])
            save()
            count = unrecorded_probe_count(directory, report)
        row = invoke(root, config, protocol, 'probe', ['--max-new-cases', str(2-count)])
        entry.update(probe_record=row, prefix_before=inventory(root, directory)); save()
    prefix_unchanged(root, directory, entry['prefix_before'])
    completed = invoke(root, config, protocol, 'completion', [])
    after = prefix_unchanged(root, directory, entry['prefix_before'])
    before = inventory(root, directory, report)
    readonly = invoke(root, config, protocol, 'read_only', ['--check-only'])
    require(inventory(root, directory, report) == before, 'Read-only geometry verification changed files')
    require(completed['report_sha256'] == readonly['report_sha256'] == store.digest(report), 'Geometry report SHA differs')
    entry.update(prefix_after=after, completion_record=completed, read_only_record=readonly,
        completed_snapshot=before, report_sha256=store.digest(report),
        completion_sha256=store.digest(directory/'completion.json'), byte_and_mtime_identical=True, read_only_check_passed=True)
    proof['status'] = 'passed'; save()
    return proof


def run(root=ROOT, config=CONFIG):
    root = Path(root).resolve(); config = store.safe(root, config); protocol = store.read(config)
    driver.validate(root, config, protocol)
    for name in (SOURCE, TEST, 'scripts/pipeline.py'):
        require(protocol['source_hashes'].get(name) == store.digest(root/name), 'Wrapper source is not frozen: '+name)
    require(protocol['outputs']['resume_report'] in protocol['budget_paths'], 'Recovery proof is outside the output budget')
    with store.lock(root/'runtime/locks/geometry_propagation_v19_wrapper'):
        proof = execute(root, config, protocol)
    return {'schema': 19, 'status': proof['status'], 'proof_sha256': store.digest(store.safe(root, protocol['outputs']['resume_report']))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    print(json.dumps(run(config=parser.parse_args().config), ensure_ascii=False), flush=True)
