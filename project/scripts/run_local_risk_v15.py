"""Run fixed feature blocks across process boundaries, preserving real prefixes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import local_risk_common_v15 as common
from scripts.pipeline import project_environment


def last_record(path, offset):
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        for line in reversed(stream.read().decode('utf-8').splitlines()):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and 'status' in record:
                return record
    raise RuntimeError('Child did not emit a current status record')


def validate_record(operation, record, frame):
    common.require(record.get('frame') == frame, 'Child frame mismatch')
    statuses = {'probe': ('partial', 'completed'), 'complete': ('completed', 'verified_completed_read_only'),
                'read_only': ('verified_completed_read_only',)}
    recovered = record.get('recovered_completed_probe', False)
    if recovered:
        common.require(operation == 'probe' and recovered is True and record.get('expected_blocks', 3) <= 2,
                       'Invalid small completed-probe recovery')
        statuses['probe'] = ('verified_completed_read_only',)
    common.require(record.get('status') in statuses[operation], 'Unexpected child status')
    fields = ('new_blocks', 'reused_blocks', 'committed_blocks', 'expected_blocks',
              'new_feature_rows', 'new_voxelizations', 'new_detector_calls')
    common.require(all(type(record.get(key)) is int and record[key] >= 0 for key in fields), 'Invalid work counters')
    common.require(record['new_blocks'] + record['reused_blocks'] == record['committed_blocks'] <= record['expected_blocks'],
                   'Block accounting differs')
    if record['status'] != 'partial':
        common.require(record['committed_blocks'] == record['expected_blocks'], 'Missing completed blocks')
    for key in ('new_training_steps', 'new_model_forward_calls', 'new_official_geometry_computations'):
        common.require(record.get(key) == 0, 'Forbidden new computation: ' + key)
    if operation == 'probe':
        common.require(record['committed_blocks'] <= 2, 'Probe exceeded two-block prefix')
    if operation == 'read_only' or recovered or record['status'] == 'verified_completed_read_only':
        common.require(all(record[key] == 0 for key in ('new_blocks', 'new_feature_rows', 'new_voxelizations', 'new_detector_calls')),
                       'Read-only performed new feature work')


def verify_prefix(directory, prefix):
    current = common.snapshot(directory)
    common.require(bool(prefix) and all(current.get(name) == item for name, item in prefix.items()),
                   'Committed prefix bytes or modification times changed')


def probe_flags(directory):
    directory = Path(directory)
    count = len([p for p in (directory / 'blocks').glob('block_*.json') if not p.name.endswith('.sha256.json')])
    common.require(count <= 2, 'Unrecorded prefix exceeds the frozen probe')
    if (directory / 'completion.json').exists():
        common.require(common.read(directory / 'completion.json')['expected_blocks'] <= 2,
                       'Unrecorded completion exceeds two blocks')
        return ['--check-only'], True
    return ['--max-new-blocks', str(2 - count)], False


def run(root=ROOT, config=common.CONFIG):
    root, config = Path(root).resolve(), Path(config).resolve()
    protocol = common.read(config)
    common.validate_protocol(protocol)
    common.require(bool(protocol['source_hashes']), 'Source freeze is required')
    for name, sha in protocol['source_hashes'].items():
        common.require(common.digest(common.safe_path(root, name)) == sha, 'Frozen source changed: ' + name)
    script = root / 'scripts/extract_local_risk_v15.py'
    interpreter = common.safe_path(root, protocol['interpreter'])
    proof_path = common.safe_path(root, protocol['outputs']['features_resume_report'])
    binding = {'protocol_sha256': common.digest(config), 'wrapper_sha256': common.digest(Path(__file__)),
        'child_sha256': common.digest(script), 'common_sha256': common.digest(Path(common.__file__)),
        'interpreter_sha256': common.digest(interpreter), 'pipeline_sha256': common.digest(root / 'scripts/pipeline.py')}
    outer = {**protocol, 'resources': {**protocol['resources'], 'features': {
        **protocol['resources']['features'], 'invocation_soft_seconds':
        (protocol['resources']['features']['invocation_soft_seconds'] + 60) * (len(protocol['frames']) * 3 + 1)}}}
    guard = common.Guard(root, outer, 'features')
    def save_proof(value):
        guard.check(len(common.json_bytes(value)) + 1024)
        common.atomic(proof_path, value)
    guard.check()
    with common.lock(root / 'runtime/grasp_local_risk_v15/features_wrapper'):
        if proof_path.exists():
            proof = common.read(proof_path)
            common.require(proof['schema'] == 15 and proof['phase'] == 'features' and proof['binding'] == binding,
                           'Recovery proof binding changed')
        else:
            proof = {'schema': 15, 'phase': 'features', 'status': 'in_progress', 'binding': binding,
                     'frames': {}, 'scope': 'Actual process boundaries; not physical power removal'}
            save_proof(proof)
        common.require(set(proof['frames']).issubset(protocol['frames']), 'Unexpected proof frame')
        for frame in protocol['frames']:
            directory = common.frame_directory(root, protocol, frame)
            entry = proof['frames'].get(frame)
            if entry is None:
                common.require(not (directory / 'completion.json').exists(), 'Completed feature lacks recovery history')
                entry = {'status': 'in_progress'}
                proof['frames'][frame] = entry
                save_proof(proof)

            def invoke(operation, flags, recovered=False):
                guard.check(65536)
                command = [str(interpreter), '-B', str(script), '--config', str(config), '--frame', frame] + flags
                logs = root / 'logs/P3/local_risk_v15'
                logs.mkdir(parents=True, exist_ok=True)
                log = logs / ('features_' + frame + '_' + operation + '.log')
                print('Local risk %s: %s' % (frame, operation), flush=True)
                with log.open('ab', buffering=0) as stream:
                    stream.write(('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n').encode('utf-8'))
                    offset = stream.tell()
                    result = subprocess.run(command, cwd=root, env=project_environment(root), stdout=stream,
                        stderr=subprocess.STDOUT, timeout=protocol['resources']['features']['invocation_soft_seconds'] + 60)
                    os.fsync(stream.fileno())
                common.require(result.returncode == 0, 'Feature phase stopped; checkpoints retained. Inspect ' + str(log))
                record = last_record(log, offset)
                if recovered:
                    record['recovered_completed_probe'] = True
                validate_record(operation, record, frame)
                guard.check()
                return record

            if entry['status'] == 'passed':
                common.require(entry['completed_snapshot'] == common.snapshot(directory), 'Completed feature tree changed')
                verify_prefix(directory, entry['prefix_before'])
                invoke('read_only', ['--check-only'])
                common.require(entry['completed_snapshot'] == common.snapshot(directory), 'Read-only changed completed tree')
                continue
            common.require(entry['status'] == 'in_progress', 'Unknown proof state')
            if 'prefix_before' not in entry:
                flags, recovered = probe_flags(directory)
                record = invoke('probe', flags, recovered)
                entry.update(probe_record=record, prefix_before=common.snapshot(directory))
                save_proof(proof)
            verify_prefix(directory, entry['prefix_before'])
            final = invoke('complete', [])
            verify_prefix(directory, entry['prefix_before'])
            before = common.snapshot(directory)
            readonly = invoke('read_only', ['--check-only'])
            common.require(before == common.snapshot(directory), 'Read-only changed bytes or mtime')
            entry.update(status='passed', completion_record=final, read_only_record=readonly,
                byte_and_mtime_identical=True, completed_snapshot=before,
                completion_sha256=common.digest(directory / 'completion.json'))
            save_proof(proof)
        if proof['status'] != 'passed':
            proof['status'] = 'passed'
            save_proof(proof)
        print(json.dumps({'status': 'passed', 'phase': 'features', 'frames': protocol['frames']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=common.CONFIG)
    run(config=parser.parse_args().config)
