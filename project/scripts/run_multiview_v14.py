"""Bounded serial phases with real subprocess-boundary recovery proofs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import multiview_common_v14 as common
from scripts.pipeline import project_environment


def last_record(path, offset):
    with path.open('rb') as stream:
        stream.seek(offset)
        for line in reversed(stream.read().decode('utf-8').splitlines()):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and 'status' in value:
                return value
    raise RuntimeError('Child did not emit a current status record')


def verify_prefix(directory, prefix):
    current = common.snapshot(directory)
    common.require(bool(prefix) and all(current.get(name) == item for name, item in prefix.items()),
                   'Durable prefix bytes or modification times changed')


def validate_record(phase, operation, record):
    expected = {'probe': ('partial', 'completed'), 'complete': ('completed', 'verified_completed_read_only'),
                'read_only': ('verified_completed_read_only',)}
    if phase == 'generation':
        expected['probe'] = ('inputs_committed',)
    recovered_probe = record.get('recovered_completed_probe', False)
    if recovered_probe:
        common.require(phase == 'geometry' and operation == 'probe' and recovered_probe is True
                       and record.get('expected_grasps', 3) <= 2, 'Invalid recovered completed probe')
        expected['probe'] = ('verified_completed_read_only',)
    common.require(record['status'] in expected[operation], 'Unexpected child phase status')
    if phase == 'geometry':
        common.require(record.get('new_geometry_computations') == record.get('new_grasps')
                       and record.get('new_model_forward_calls') == 0, 'Geometry work accounting differs')
        values = [record.get(key) for key in ('new_grasps', 'reused_grasps', 'committed_grasps', 'expected_grasps')]
        common.require(all(type(value) is int and value >= 0 for value in values)
                       and values[0] + values[1] == values[2] <= values[3], 'Geometry count accounting differs')
        if record['status'] != 'partial':
            common.require(values[2] == values[3], 'Completed geometry has missing scores')
    if operation == 'read_only' or recovered_probe:
        keys = {'assets': ('new_files', 'network_bytes'), 'generation': ('new_model_forward_calls',),
                'geometry': ('new_grasps',)}[phase]
        for key in keys:
            common.require(record.get(key) == 0, 'Read-only performed new work: ' + key)
        if phase == 'geometry':
            common.require(record.get('official_oracle_executed_this_invocation') is False
                           and record.get('new_geometry_preparations') == 0
                           and record.get('new_geometry_computations') == 0
                           and record.get('new_model_forward_calls') == 0, 'Read-only recomputed geometry')
    if phase == 'generation' and operation == 'probe':
        common.require(record.get('new_model_forward_calls') == 0, 'Input probe performed model inference')
    if phase == 'geometry' and operation == 'probe':
        common.require(0 <= record['new_grasps'] <= 2, 'Geometry probe exceeded two new candidates')
    if phase == 'assets' and operation == 'probe':
        common.require(0 <= record['new_files'] <= 2, 'Asset probe exceeded two files')


def run(phase, root=ROOT, config=common.CONFIG):
    root, config = Path(root).resolve(), Path(config).resolve()
    protocol = common.read(config)
    common.require(phase in ('assets', 'generation', 'geometry') and bool(protocol['source_hashes']), 'Unfrozen phase')
    for name, sha in protocol['source_hashes'].items():
        common.require(common.digest(common.safe_path(root, name)) == sha, 'Frozen source changed: ' + name)
    scripts = {'assets': 'fetch_multiview_assets_v14.py', 'generation': 'generate_multiview_grasps_v14.py',
               'geometry': 'evaluate_multiview_grasps_v14.py'}
    script = root / 'scripts' / scripts[phase]
    interpreter = common.safe_path(root, protocol['interpreters']['grasp_eval' if phase == 'geometry' else 'vision'])
    proof_path = common.safe_path(root, protocol['outputs'][phase + '_resume_report'])
    binding = {'protocol_sha256': common.digest(config), 'wrapper_sha256': common.digest(Path(__file__)),
               'child_sha256': common.digest(script), 'common_sha256': common.digest(Path(common.__file__)),
               'interpreter_sha256': common.digest(interpreter), 'pipeline_sha256': common.digest(root / 'scripts/pipeline.py')}
    frames = ['all'] if phase == 'assets' else protocol['frames']
    # Each child has its own time budget. The wrapper spans multiple children.
    wrapper_protocol = {**protocol, 'resources': {**protocol['resources'], phase: {
        **protocol['resources'][phase], 'invocation_soft_seconds':
        (protocol['resources'][phase]['invocation_soft_seconds'] + 60) * (len(frames) * 3 + 1)}}}
    guard = common.Guard(root, wrapper_protocol, phase)
    def save_proof(value):
        guard.check(len(common.json_bytes(value)) + 1024)
        common.atomic(proof_path, value)
    guard.check()
    upstream = {}
    for previous in (('assets', 'generation') if phase == 'geometry' else ('assets',) if phase == 'generation' else ()):
        previous_path = common.safe_path(root, protocol['outputs'][previous + '_resume_report'])
        record = common.read(previous_path)
        common.require(record['schema'] == 14 and record['phase'] == previous and record['status'] == 'passed'
                       and record['binding']['protocol_sha256'] == common.digest(config), 'Upstream recovery proof incomplete')
        required = ['all'] if previous == 'assets' else protocol['frames']
        common.require(set(record['frames']) == set(required), 'Upstream recovery frame selection differs')
        for item in required:
            saved = record['frames'][item]
            folder = common.safe_path(root, protocol['outputs']['assets_dir']) if previous == 'assets' else common.frame_directory(root, protocol, item, previous)
            common.require(saved['status'] == 'passed' and saved['completed_snapshot'] == common.snapshot(folder),
                           'Upstream recovery output changed')
        upstream[previous] = common.digest(previous_path)
    binding['upstream_recovery_proofs'] = upstream
    with common.lock(root / 'runtime/grasp_multiview_v14' / (phase + '_wrapper')):
        if proof_path.exists():
            proof = common.read(proof_path)
            common.require(proof['schema'] == 14 and proof['phase'] == phase and proof['binding'] == binding,
                           'Recovery proof binding differs')
        else:
            proof = {'schema': 14, 'phase': phase, 'status': 'in_progress', 'binding': binding, 'frames': {},
                     'scope': 'Actual subprocess boundaries, not physical power removal'}
            save_proof(proof)
        common.require(set(proof['frames']).issubset(frames), 'Unexpected recovery frame')
        for frame in frames:
            directory = common.safe_path(root, protocol['outputs']['assets_dir']) if phase == 'assets' else common.frame_directory(root, protocol, frame, phase)
            entry = proof['frames'].get(frame)
            if entry is None:
                common.require(not (directory / 'completion.json').exists(), 'Completed frame lacks recovery proof; history cannot be invented')
                entry = {'status': 'in_progress'}
                proof['frames'][frame] = entry
                save_proof(proof)

            def invoke(operation, flags):
                guard.check(65536)
                recovered_probe = phase == 'geometry' and operation == 'probe' and flags == ['--check-only']
                command = [str(interpreter), '-B', str(script), '--config', str(config)]
                if phase != 'assets':
                    command += ['--frame', frame]
                command += flags
                logs = root / 'logs/P3/multiview_v14'
                logs.mkdir(parents=True, exist_ok=True)
                log = logs / (phase + '_' + frame + '_' + operation + '.log')
                print('Multiview %s %s: %s' % (phase, frame, operation), flush=True)
                with log.open('ab', buffering=0) as stream:
                    stream.write(('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n').encode('utf-8'))
                    offset = stream.tell()
                    result = subprocess.run(command, cwd=root, env=project_environment(root), stdout=stream,
                        stderr=subprocess.STDOUT, timeout=protocol['resources'][phase]['invocation_soft_seconds'] + 60)
                    os.fsync(stream.fileno())
                common.require(result.returncode == 0, 'Phase stopped; retained checkpoints. Inspect ' + str(log))
                guard.check()
                record = last_record(log, offset)
                if recovered_probe:
                    common.require((directory / 'completion.json').exists(), 'Missing recovered probe completion')
                    record['recovered_completed_probe'] = True
                validate_record(phase, operation, record)
                if phase != 'assets':
                    common.require(record.get('frame') == frame, 'Wrong child frame')
                return record

            if entry['status'] == 'passed':
                common.require(entry['completed_snapshot'] == common.snapshot(directory), 'Completed output changed')
                verify_prefix(directory, entry['prefix_before'])
                invoke('read_only', ['--check-only'])
                common.require(entry['completed_snapshot'] == common.snapshot(directory), 'Read-only altered completed output')
                continue
            common.require(entry['status'] == 'in_progress', 'Invalid recovery status')
            if 'prefix_before' not in entry:
                if phase == 'generation':
                    common.require(not (directory / 'forward_commit.json').exists(), 'Unrecorded forward cannot be claimed as input-only probe')
                    flags = ['--stop-after-inputs']
                elif phase == 'geometry':
                    count = len([p for p in (directory / 'grasps').glob('grasp_*.json') if not p.name.endswith('.sha256.json')])
                    common.require(count <= 2, 'Unrecorded scoring prefix exceeds probe')
                    if (directory / 'completion.json').exists():
                        common.require(common.read(directory / 'completion.json')['expected_grasps'] <= 2,
                                       'Completed unrecorded geometry exceeds the probe budget')
                        flags = ['--check-only']
                    else:
                        flags = ['--max-new-grasps', str(2 - count)]
                else:
                    count = sum(common.safe_path(root, protocol['frame_inputs'][f]['annotation']).exists() for f in protocol['frames'])
                    common.require(count <= 2, 'Unrecorded asset prefix exceeds probe')
                    flags = ['--max-new-files', str(2 - count)]
                entry.update(probe_record=invoke('probe', flags), prefix_before=common.snapshot(directory))
                save_proof(proof)
            verify_prefix(directory, entry['prefix_before'])
            final = invoke('complete', [])
            verify_prefix(directory, entry['prefix_before'])
            before = common.snapshot(directory)
            readonly = invoke('read_only', ['--check-only'])
            common.require(before == common.snapshot(directory), 'Read-only changed output bytes or modification times')
            entry.update(status='passed', completion_record=final, read_only_record=readonly,
                         byte_and_mtime_identical=True, completed_snapshot=before,
                         completion_sha256=common.digest(directory / 'completion.json'))
            save_proof(proof)
        if proof['status'] != 'passed':
            proof['status'] = 'passed'
            save_proof(proof)
        print(json.dumps({'status': 'passed', 'phase': phase, 'frames': frames}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('assets', 'generation', 'geometry'), required=True)
    run(parser.parse_args().phase)
