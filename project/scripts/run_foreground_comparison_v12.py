"""Persist process-boundary recovery evidence for two fixed sampling strategies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json, file_sha256, _exclusive_lock

ARMS = ('full_scene', 'predicted_foreground')
CONFIG = 'configs/grasp_foreground_comparison_v12.json'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def snapshot(directory):
    return {p.relative_to(ROOT).as_posix(): {'sha256': file_sha256(p), 'bytes': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns} for p in sorted(directory.rglob('*'))
            if p.is_file() and not p.name.endswith('.lock')}


def last_record(path, offset):
    with path.open('rb') as stream:
        stream.seek(offset)
        lines = stream.read().decode('utf-8').splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'status' in value:
            return value
    raise RuntimeError('Current child invocation did not publish a JSON status')


def validate_record(phase, record, operation):
    if phase == 'generation':
        count = record['new_model_forward_calls']
        require(type(count) is int and count in (0, 1), 'Invalid forward counter')
        if operation == 'probe':
            require(record['status'] == 'inputs_committed' and count == 0, 'Inputs-only boundary did inference')
        elif operation == 'read_only':
            require(record['status'] == 'verified_completed_read_only' and count == 0, 'Read-only generation did inference')
        else:
            require(record['status'] in ('completed', 'verified_completed_read_only'), 'Generation not completed')
            require(record['status'] != 'verified_completed_read_only' or count == 0, 'Completed cache reran inference')
    else:
        keys = ('new_grasps', 'reused_grasps', 'committed_grasps', 'expected_grasps')
        require(all(type(record[key]) is int and record[key] >= 0 for key in keys), 'Invalid scoring counters')
        require(record['new_grasps'] + record['reused_grasps'] == record['committed_grasps'], 'Scoring count mismatch')
        if operation == 'probe' and record['status'] == 'partial':
            require(record['committed_grasps'] >= 2 and record['committed_grasps'] < record['expected_grasps'],
                    'Missing two-grasp scoring prefix')
        else:
            require(record['status'] in ('completed', 'verified_completed_read_only')
                    and record['committed_grasps'] == record['expected_grasps'], 'Scoring did not complete')
        if operation == 'read_only':
            require(record['status'] == 'verified_completed_read_only' and record['new_grasps'] == 0
                    and record['geometry_computations'] == 0
                    and record['official_oracle_executed_this_invocation'] is False, 'Read-only scored geometry')


def prefix_snapshot(directory, phase):
    all_files = snapshot(directory)
    if phase == 'generation':
        require((directory / 'input_commit.json').is_file(), 'Input commit missing')
        return all_files
    require((directory / 'preparation.json').is_file(), 'Geometry preparation missing')
    return {name: value for name, value in all_files.items()
            if '/grasps/' in name or Path(name).name in ('preparation.json', 'preparation.sha256.json')}


def verify_prefix(prefix):
    require(bool(prefix), 'Empty durable prefix')
    for name, expected in prefix.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT.resolve()), 'Prefix path escaped project')
        require(path.is_file() and {'sha256': file_sha256(path), 'bytes': path.stat().st_size,
                'mtime_ns': path.stat().st_mtime_ns} == expected, 'Durable prefix changed: ' + name)


def run_locked(phase):
    require(phase in ('generation', 'geometry'), 'Unknown comparison phase')
    script = ROOT / ('scripts/generate_foreground_grasps_v12.py' if phase == 'generation'
                     else 'scripts/evaluate_foreground_grasps_v12.py')
    interpreter = ROOT / ('runtime/envs/vision/Scripts/python.exe' if phase == 'generation'
                          else 'runtime/envs/grasp_eval/Scripts/python.exe')
    proof_path = ROOT / ('results/P3/grasp_foreground_' + phase + '_v12_resume.json')
    binding = {'protocol_sha256': file_sha256(ROOT / CONFIG), 'wrapper_sha256': file_sha256(Path(__file__)),
               'child_sha256': file_sha256(script), 'interpreter_sha256': file_sha256(interpreter),
               'helpers': {name: file_sha256(ROOT / name) for name in ('src/checkpoints.py', 'scripts/pipeline.py')}}
    if proof_path.exists():
        proof = read(proof_path)
        require(proof['schema'] == 12 and proof['phase'] == phase and proof['binding'] == binding,
                'Comparison recovery binding changed')
        require(proof['status'] in ('in_progress', 'passed'), 'Unknown recovery proof state')
    else:
        proof = {'schema': 12, 'phase': phase, 'status': 'in_progress', 'binding': binding, 'arms': {},
                 'new_training_updates': 0, 'scope': 'Process-boundary recovery; not a physical power interruption'}
        atomic_json(proof_path, proof)
    logs = ROOT / 'logs/P3/foreground_v12'
    logs.mkdir(parents=True, exist_ok=True)

    def invoke(arm, operation, flags):
        command = [str(interpreter), '-B', str(script), '--config', CONFIG, '--arm', arm] + flags
        log = logs / ('%s_%s_%s.log' % (phase, arm, operation))
        print('Comparison %s / %s: %s' % (phase, arm, operation), flush=True)
        with log.open('ab', buffering=0) as stream:
            stream.write(('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n').encode('utf-8'))
            offset = stream.tell()
            result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT), stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        require(result.returncode == 0, 'Comparison stopped; durable results retained: ' + str(log))
        record = last_record(log, offset)
        validate_record(phase, record, operation)
        return record

    for arm in ARMS:
        directory = ROOT / 'results/P3/grasp_foreground_v12' / arm / phase
        complete_path, report_path = directory / 'completion.json', directory / 'report.json'
        previous = proof['arms'].get(arm)

        def read_only():
            before = snapshot(directory)
            value = invoke(arm, 'read_only', ['--check-only'])
            require(snapshot(directory) == before, 'Read-only invocation modified results or timestamps')
            return value

        if previous:
            verify_prefix(previous['prefix_before'])
            if previous.get('status') == 'passed':
                require(previous['completion_sha256'] == file_sha256(complete_path)
                        and previous['report_sha256'] == file_sha256(report_path), 'Final comparison evidence changed')
                read_only()
                continue
        else:
            require(not complete_path.exists(), 'Completed arm without saved probe evidence; no invented recovery test')
            probe = invoke(arm, 'probe', ['--stop-after-inputs'] if phase == 'generation' else ['--max-new-grasps', '2'])
            previous = {'status': 'prefix_committed', 'probe_record': probe,
                        'prefix_before': prefix_snapshot(directory, phase)}
            proof['arms'][arm] = previous
            atomic_json(proof_path, proof)
        final = invoke(arm, 'complete', [])
        verify_prefix(previous['prefix_before'])
        require(complete_path.is_file() and report_path.is_file(), 'Final arm output missing')
        readonly = read_only()
        result = dict(previous, status='passed', prefix_after=previous['prefix_before'],
                      byte_and_mtime_identical=True, completion_record=final, read_only_record=readonly,
                      read_only_check_passed=True, completion_sha256=file_sha256(complete_path),
                      report_sha256=file_sha256(report_path))
        if phase == 'geometry':
            result['recovery_demonstration'] = ('two_grasp_prefix_then_remaining' if previous['probe_record']['status'] == 'partial'
                                                else 'all_available_grasps_in_probe_then_read_only')
        proof['arms'][arm] = result
        atomic_json(proof_path, proof)
    if proof['status'] != 'passed':
        proof['status'] = 'passed'
        atomic_json(proof_path, proof)
    print(json.dumps({'status': 'passed', 'phase': phase, 'arms': list(ARMS)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('generation', 'geometry'), required=True)
    args = parser.parse_args()
    directory = ROOT / ('runtime/grasp_foreground_v12/' + args.phase + '_wrapper_lock')
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(directory):
        run_locked(args.phase)


if __name__ == '__main__':
    main()
