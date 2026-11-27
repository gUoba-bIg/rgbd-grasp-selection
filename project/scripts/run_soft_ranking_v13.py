"""Execute bounded feature/label phases with durable recovery evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from scripts import soft_ranking_common_v13 as io


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
    raise RuntimeError('Child did not emit a current JSON status')


def verify_prefix(directory, prefix):
    io.require(bool(prefix), 'Empty durable prefix')
    current = io.snapshot(directory)
    io.require(all(current.get(name) == info for name, info in prefix.items()), 'Durable prefix bytes/mtime changed')


def validate_record(phase, record, operation):
    io.require(record['status'] in ('partial', 'completed', 'verified_completed_read_only'), 'Invalid child status')
    if phase == 'features':
        for key in ('new_blocks', 'reused_blocks', 'committed_blocks', 'expected_blocks', 'new_feature_rows'):
            io.require(type(record[key]) is int and record[key] >= 0, 'Invalid feature counter: ' + key)
        io.require(record['expected_blocks'] == 15 and record['new_blocks'] + record['reused_blocks'] == record['committed_blocks'], 'Feature block accounting differs')
        if operation == 'probe':
            io.require(record['status'] == 'partial' and record['committed_blocks'] == 2, 'Two-block probe boundary missing')
        else:
            io.require(record['committed_blocks'] == 15 and record['status'] != 'partial', 'Features incomplete')
        if operation == 'read_only':
            io.require(record['new_blocks'] == 0 and record['new_feature_rows'] == 0, 'Read-only recomputed features')
    else:
        io.require(record['status'] != 'partial', 'Evaluation incomplete')
        io.require(record['new_geometry_computations'] == 0 and record['new_model_forward_calls'] == 0, 'Evaluation recomputed models or geometry')
    if operation == 'read_only':
        io.require(record['status'] == 'verified_completed_read_only', 'Missing read-only verification')


def run(phase):
    protocol = io.read(io.CONFIG)
    io.require(phase in ('features', 'evaluation'), 'Unknown phase')
    io.require(bool(protocol['source_hashes']), 'Source implementation has not been frozen')
    for name, expected in protocol['source_hashes'].items():
        io.require(io.digest(io.safe_path(ROOT, name)) == expected, 'Frozen source changed before launch: ' + name)
    script_name = 'extract_soft_ranking_features_v13.py' if phase == 'features' else 'evaluate_soft_ranking_v13.py'
    script = ROOT / 'scripts' / script_name
    interpreter = io.safe_path(ROOT, protocol['interpreter'])
    directory = io.safe_path(ROOT, protocol['outputs'][phase + '_dir'])
    proof_path = io.safe_path(ROOT, protocol['outputs']['feature_resume_report' if phase == 'features' else 'resume_report'])
    binding = {'protocol_sha256': io.digest(io.CONFIG), 'wrapper_sha256': io.digest(Path(__file__)),
               'child_sha256': io.digest(script), 'common_sha256': io.digest(Path(io.__file__)),
               'pipeline_sha256': io.digest(ROOT / 'scripts/pipeline.py'), 'interpreter_sha256': io.digest(interpreter)}
    guard = io.Guard(ROOT, protocol)
    guard.check()
    with io.lock(ROOT / 'runtime/grasp_soft_ranking_v13' / (phase + '_wrapper')):
        if proof_path.exists():
            proof = io.read(proof_path)
            io.require(proof['schema'] == 13 and proof['phase'] == phase and proof['binding'] == binding, 'Recovery proof binding changed')
        else:
            io.require(not (directory / 'completion.json').exists(), 'Completed results lack recovery proof; do not invent recovery history')
            proof = {'schema': 13, 'phase': phase, 'status': 'in_progress', 'binding': binding,
                     'scope': 'Process-boundary recovery, not physical power interruption'}
            guard.check(len(io.json_bytes(proof)))
            io.atomic(proof_path, proof)
        logs = ROOT / 'logs/P3/soft_ranking_v13'
        logs.mkdir(parents=True, exist_ok=True)

        def invoke(operation, flags):
            guard.check()
            command = [str(interpreter), '-B', str(script), '--config', str(io.CONFIG)] + flags
            log = logs / (phase + '_' + operation + '.log')
            print('Soft ranking %s: %s' % (phase, operation), flush=True)
            with log.open('ab', buffering=0) as stream:
                stream.write(('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n').encode('utf-8'))
                offset = stream.tell()
                result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT), stdout=stream, stderr=subprocess.STDOUT, timeout=330)
                os.fsync(stream.fileno())
            io.require(result.returncode == 0, 'Phase stopped; checkpoints retained. Inspect ' + str(log))
            guard.check()
            value = last_record(log, offset)
            validate_record(phase, value, operation)
            return value

        def read_only():
            before = io.snapshot(directory)
            value = invoke('read_only', ['--check-only'])
            io.require(before == io.snapshot(directory), 'Read-only changed file bytes or timestamps')
            return value, before

        if proof['status'] == 'passed':
            io.require(proof['completion_sha256'] == io.digest(directory / 'completion.json') and
                       proof['report_sha256'] == io.digest(directory / 'report.json'), 'Final artifacts changed')
            io.require(proof['completed_snapshot'] == io.snapshot(directory), 'Completed file bytes or timestamps changed')
            if phase == 'features':
                verify_prefix(directory, proof['prefix_before'])
            else:
                io.require(proof['feature_resume_sha256'] == io.digest(io.safe_path(ROOT, protocol['outputs']['feature_resume_report'])),
                           'Completed evaluation upstream recovery proof changed')
            read_only()
            print(json.dumps({'status': 'passed', 'phase': phase, 'reused_completed_proof': True}), flush=True)
            return
        io.require(proof['status'] == 'in_progress', 'Unknown recovery status')
        if phase == 'features':
            if 'prefix_before' not in proof:
                existing = list((directory / 'blocks').glob('block_*.json')) if (directory / 'blocks').exists() else []
                existing = [p for p in existing if not p.name.endswith('.sha256.json')]
                io.require(len(existing) <= 2, 'Unexpected unrecorded prefix; refusing fabricated recovery evidence')
                probe = invoke('probe', ['--max-new-blocks', str(2 - len(existing))])
                proof.update(probe_record=probe, prefix_before=io.snapshot(directory))
                guard.check(len(io.json_bytes(proof)))
                io.atomic(proof_path, proof)
            verify_prefix(directory, proof['prefix_before'])
        else:
            previous = io.read(io.safe_path(ROOT, protocol['outputs']['feature_resume_report']))
            feature_directory = io.safe_path(ROOT, protocol['outputs']['features_dir'])
            io.require(previous['status'] == 'passed' and previous['schema'] == 13 and previous['phase'] == 'features'
                       and previous['binding']['protocol_sha256'] == io.digest(io.CONFIG), 'Features have not passed current recovery verification')
            io.require(previous['completion_sha256'] == io.digest(feature_directory / 'completion.json')
                       and previous['report_sha256'] == io.digest(feature_directory / 'report.json')
                       and previous['completed_snapshot'] == io.snapshot(feature_directory), 'Feature recovery evidence no longer matches sealed results')
            proof['feature_resume_sha256'] = io.digest(io.safe_path(ROOT, protocol['outputs']['feature_resume_report']))
        final = invoke('complete', [])
        if phase == 'features':
            verify_prefix(directory, proof['prefix_before'])
            proof['prefix_after'] = proof['prefix_before']
            proof['byte_and_mtime_identical'] = True
        readonly, frozen = read_only()
        proof.update(status='passed', completion_record=final, read_only_record=readonly,
                     read_only_check_passed=True, completed_snapshot=frozen,
                     completion_sha256=io.digest(directory / 'completion.json'),
                     report_sha256=io.digest(directory / 'report.json'))
        guard.check(len(io.json_bytes(proof)))
        io.atomic(proof_path, proof)
        print(json.dumps({'status': 'passed', 'phase': phase}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('features', 'evaluation'), required=True)
    run(parser.parse_args().phase)
