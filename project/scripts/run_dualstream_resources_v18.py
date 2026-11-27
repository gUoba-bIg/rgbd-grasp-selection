"""Demonstrate one-to-four process restart of a synthetic GPU resource probe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import probe_dualstream_resources_v18 as probe
from scripts.pipeline import project_environment

s = probe.storage


def last_record(path, offset):
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        for line in reversed(stream.read().decode('utf-8').splitlines()):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and 'status' in value:
                return value
    raise RuntimeError('Probe child has no current result record')


def validate_record(row, operation):
    s.require(row.get('schema') == 18 and row.get('expected_units') == 4,
              'Wrong child resource contract')
    for key in ('new_units', 'reused_units', 'committed_units', 'new_model_forward_calls',
                'optimizer_updates', 'real_frames'):
        s.require(type(row.get(key)) is int and row[key] >= 0, 'Invalid child work accounting')
    s.require(row['new_units'] + row['reused_units'] == row['committed_units'] <= 4
              and row['new_model_forward_calls'] <= row['new_units']
              and row['optimizer_updates'] == row['real_frames'] == 0, 'Unexpected child work')
    if operation == 'probe':
        s.require(row['status'] == 'partial' and row['committed_units'] == 1,
                  'Restart probe must stop at exactly one unit')
    else:
        s.require(row['status'] in ('passed_synthetic_gpu_resource_probe', 'verified_completed_read_only')
                  and row['committed_units'] == 4, 'Resource probe is incomplete')
    if operation == 'read_only' or row['status'] == 'verified_completed_read_only':
        s.require(row['status'] == 'verified_completed_read_only' and row['new_units'] == 0
                  and row['new_model_forward_calls'] == 0, 'Read-only child performed new work')


def inventory(root, directory):
    prefix = directory.relative_to(root).as_posix() + '/'
    return {prefix + name: row for name, row in probe.snapshot(directory).items()}


def completed_snapshot(root, p):
    directory = probe.safe(root, p['outputs']['root'])
    result = inventory(root, directory)
    report = probe.safe(root, p['outputs']['report'])
    for file in (report, probe.receipt(report)):
        s.require(file.is_file(), 'Missing external probe report or receipt')
        result[file.relative_to(root).as_posix()] = {'sha256': s.digest(file),
            'bytes': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns}
    return result


def checked_prefix(root, directory, prefix):
    observed = inventory(root, directory)
    s.require(bool(prefix) and all(observed.get(name) == value for name, value in prefix.items()),
              'Saved one-unit prefix changed SHA, size or mtime')
    return {name: observed[name] for name in prefix}


def unrecorded_probe_count(root, p, directory):
    # Even an unfinished later attempt invalidates an unrecorded one-unit boundary.
    for unit in probe.UNITS[1:]:
        s.require(not (directory/unit).exists(), 'Unrecorded work exceeds the one-unit probe')
    report = probe.safe(root, p['outputs']['report'])
    s.require(not report.exists() and not probe.receipt(report).exists()
              and not (directory/'completion.json').exists()
              and not probe.receipt(directory/'completion.json').exists(),
              'Completed outputs lack recorded restart history')
    return int((directory/probe.UNITS[0]/'unit.json').is_file())


def invoke(root, config, p, operation, flags, guard):
    guard.check(65536)
    command = [str(probe.safe(root, p['driver_interpreter'])), '-B',
               str(root/'scripts/probe_dualstream_resources_v18.py'), '--config', str(config)] + flags
    file = root/'logs/P4/dualstream_resources_v18'/('wrapper_' + operation + '.log')
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('ab', buffering=0) as stream:
        stream.write(('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n').encode('utf-8'))
        offset = stream.tell()
        result = subprocess.run(command, cwd=root, env=project_environment(root), stdout=stream,
            stderr=subprocess.STDOUT, timeout=p['resources']['invocation_timeout_seconds'] + 60)
        os.fsync(stream.fileno())
    s.require(result.returncode == 0, 'Resource probe stopped; saved units retained. Inspect ' + str(file))
    guard.check()
    row = last_record(file, offset)
    validate_record(row, operation)
    return row


def execute(root, config, p):
    frozen = probe.validate_protocol(root, config, p)
    driver = probe.safe(root, p['driver_interpreter'])
    s.require(p['driver_interpreter'] == 'runtime/envs/grasp_eval/Scripts/python.exe'
              and Path(sys.executable).resolve() == driver
              and p['input_hashes'].get(p['driver_interpreter']) == s.digest(driver),
              'Resource wrapper requires its frozen CPU driver')
    directory = probe.safe(root, p['outputs']['root'])
    proof_file = probe.safe(root, p['outputs']['resume_report'])
    guard = probe.Guard(root, p)
    binding = {'protocol_sha256': s.digest(config), 'probe_binding_fingerprint': frozen['fingerprint'],
               'wrapper_sha256': s.digest(Path(__file__)), 'driver_sha256': s.digest(driver)}
    proof = s.read(proof_file) if proof_file.exists() else {
        'schema': 18, 'status': 'in_progress', 'binding': binding,
        'scope': 'Actual process boundaries with one then four committed synthetic units',
        'physical_power_loss_tested': False, 'is_training_resume_verification': False}
    s.require(proof.get('schema') == 18 and proof.get('binding') == binding
              and proof.get('status') in ('in_progress', 'passed'), 'Resource recovery proof binding differs')

    def save():
        guard.check(len(s.json_bytes(proof)) + 1024)
        s.atomic(proof_file, proof)

    def call(operation, flags):
        print('Synthetic resource probe: ' + operation, flush=True)
        return invoke(root, config, p, operation, flags, guard)

    if proof['status'] == 'passed':
        before = completed_snapshot(root, p)
        s.require(before == proof['completed_snapshot'], 'Sealed probe outputs changed')
        call('read_only', ['--check-only'])
        s.require(completed_snapshot(root, p) == before, 'Read-only probe changed output files')
    else:
        if not proof_file.exists():
            save()
        if 'prefix_before' not in proof:
            count = unrecorded_probe_count(root, p, directory)
            row = call('probe', ['--max-new-units', str(1-count)])
            proof.update(probe_record=row, prefix_before=inventory(root, directory))
            save()
        checked_prefix(root, directory, proof['prefix_before'])
        complete = call('completion', [])
        after = checked_prefix(root, directory, proof['prefix_before'])
        before = completed_snapshot(root, p)
        readonly = call('read_only', ['--check-only'])
        s.require(completed_snapshot(root, p) == before, 'Read-only probe changed SHA, size or mtime')
        report = probe.safe(root, p['outputs']['report'])
        s.require(complete['report_sha256'] == readonly['report_sha256'] == s.digest(report),
                  'Resource aggregate report SHA differs')
        proof.update(status='passed', prefix_after=after, completion_record=complete, read_only_record=readonly,
            completed_snapshot=before, report_sha256=s.digest(report), byte_and_mtime_identical=True,
            completion_sha256=s.digest(directory/'completion.json'))
        save()
    result = {'schema': 18, 'status': 'passed', 'phase': 'synthetic_resource_restart',
              'proof_sha256': s.digest(proof_file), 'resources': guard.check()}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return proof


def run(root=ROOT, config=None):
    root = Path(root).resolve()
    config = probe.safe(root, config or probe.CONFIG)
    with s.lock(root/'runtime/locks/dualstream_resources_v18_wrapper'):
        return execute(root, config, s.read(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/probe.CONFIG)
    run(config=parser.parse_args().config)
