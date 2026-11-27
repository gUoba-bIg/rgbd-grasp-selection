"""Versioned, single-writer recovery evidence for the two V11 interface phases."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json, file_sha256, _exclusive_lock


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def state(path):
    return {'sha256': file_sha256(path), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns}


def snapshot(directory, report):
    result = {p.relative_to(ROOT).as_posix(): state(p) for p in sorted(directory.rglob('*'))
              if p.is_file() and p.name not in ('.checkpoint.lock', '.process.lock')}
    if report.is_file():
        result[report.relative_to(ROOT).as_posix()] = state(report)
    return result


def last_json(path, offset=0):
    with path.open('rb') as stream:
        stream.seek(offset)
        current = stream.read().decode('utf-8')
    for line in reversed(current.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'status' in value:
            return value
    raise RuntimeError('Child invocation did not return a JSON status')


def phase_spec(phase):
    if phase == 'perception':
        return {'script': 'scripts/infer_instance_interface_v11.py',
                'directory': 'results/P3/instance_interface_v11',
                'report': 'results/P3/instance_interface_v11.json',
                'proof': 'results/P3/instance_interface_v11_resume.json',
                'probe_flags': ['--stop-after-heads'],
                'prefix': ['input_tensor.npy', 'input_tensor.npy.pending.json',
                           'heads.npz', 'heads.npz.pending.json', 'heads_commit.json']}
    require(phase == 'association', 'Unknown interface phase')
    return {'script': 'scripts/associate_grasp_instances_v11.py',
            'directory': 'results/P3/grasp_instance_association_v11',
            'report': 'results/P3/grasp_instance_association_v11.json',
            'proof': 'results/P3/grasp_instance_association_v11_resume.json',
            'probe_flags': ['--max-new-blocks', '2'],
            'prefix': ['points_commit.json'] + ['blocks/block_%04d%s' % (i, suffix)
                for i in (0, 1) for suffix in ('.json', '.sha256.json')]}


def validate_invocation(phase, record, completed, check_only=False):
    if phase == 'perception':
        require(record['new_model_forward_calls'] in (0, 1), 'Unexpected forward count')
        if completed:
            require(record['new_model_forward_calls'] == 0, 'Committed heads were recomputed')
            require(record['status'] == ('verified_completed_read_only' if check_only else 'completed'),
                    'Unexpected prediction completion status')
        else:
            require(record['status'] == 'heads_committed', 'Heads were not committed')
    else:
        if completed:
            require(record['status'] == 'passed' and type(record['new_blocks']) is int
                    and type(record['reused_blocks']) is int and record['new_blocks'] >= 0
                    and record['reused_blocks'] >= 2 and record['new_blocks'] + record['reused_blocks'] == 15,
                    'Expected15 blocks including reused prefix')
            require(record['points_reused'] is True, 'Committed point cache was rebuilt')
            if check_only:
                require(record['new_blocks'] == 0 and record['reused_blocks'] == 15 and record['check_only'] is True,
                        'Read-only association recomputed blocks')
        else:
            require(record['status'] == 'partial' and record['completed_candidates'] >= 32,
                    'Two association blocks were not committed')


def run_phase_locked(phase, entry):
    spec = phase_spec(phase)
    config = ROOT / 'configs/instance_grasp_interface_v11.json'
    script = ROOT / spec['script']
    executable = ROOT / 'runtime/envs/vision/Scripts/python.exe'
    directory, report, proof_path = [ROOT / spec[k] for k in ('directory', 'report', 'proof')]
    completion = directory / 'completion.json'
    prefix = [directory / name for name in spec['prefix']]
    logs = ROOT / ('logs/P3/interface_' + phase + '_v11_wrapper')
    logs.mkdir(parents=True, exist_ok=True)
    binding = {'protocol_sha256': file_sha256(config), 'child_sha256': file_sha256(script),
               'wrapper_sha256': file_sha256(entry), 'recovery_helper_sha256': file_sha256(Path(__file__)),
               'interpreter_sha256': file_sha256(executable),
               'helpers': {n: file_sha256(ROOT / n) for n in ('src/checkpoints.py', 'scripts/pipeline.py')}}

    def prefix_state():
        return {p.relative_to(ROOT).as_posix(): state(p) for p in prefix}

    def invoke(name, flags):
        command = [str(executable), '-B', str(script), '--config', config.relative_to(ROOT).as_posix()] + flags
        log = logs / (name + '.log')
        print('Interface recovery %s: %s' % (phase, name), flush=True)
        with log.open('ab', buffering=0) as stream:
            stream.write(('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n').encode('utf-8'))
            output_offset = stream.tell()
            child = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                   stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        require(child.returncode == 0, 'Interface phase stopped; committed prefix retained. See ' + str(log))
        return last_json(log, output_offset)

    def read_only():
        before = snapshot(directory, report)
        record = invoke('read_only_check', ['--check-only'])
        validate_invocation(phase, record, True, True)
        require(snapshot(directory, report) == before, 'Read-only check changed bytes, membership or timestamps')
        return record

    prior = read(proof_path) if proof_path.exists() else None
    if prior:
        require(prior['binding'] == binding and prior['phase'] == phase and prior['schema'] == 11,
                'Interface recovery binding changed')
        require(prior['status'] in ('prefix_committed', 'passed'), 'Unknown recovery proof state')
        require(prefix_state() == prior['prefix_before'], 'Committed prefix changed')
        if prior['status'] == 'passed':
            require(file_sha256(completion) == prior['completion_sha256'] and
                    file_sha256(report) == prior['report_sha256'], 'Completed interface evidence changed')
            read_only()
            print('Verified immutable completed interface recovery: ' + phase, flush=True)
            return
    if prior is None:
        require(not completion.exists(), 'Complete cache without wrapper recovery proof; do not invent a probe')
        probe = invoke('prefix_probe', spec['probe_flags'])
        validate_invocation(phase, probe, False)
        require(all(p.is_file() for p in prefix), 'Incomplete durable prefix')
        prior = {'schema': 11, 'status': 'prefix_committed', 'phase': phase, 'binding': binding,
                 'probe_record': probe, 'prefix_before': prefix_state(), 'new_training_updates': 0,
                 'scope': 'Process-stage recovery only; not a physical power-cut test'}
        atomic_json(proof_path, prior)
    final = invoke('complete_from_prefix', [])
    # An interruption after child completion but before wrapper completion is also resumable.
    if phase == 'perception' and final.get('status') == 'verified_completed_read_only':
        validate_invocation(phase, final, True, True)
    else:
        validate_invocation(phase, final, True)
    require(completion.is_file() and report.is_file(), 'Completion/report not committed')
    after = prefix_state()
    require(after == prior['prefix_before'], 'Prefix changed during resume')
    readonly_record = read_only()
    proof = dict(prior, status='passed', prefix_after=after, byte_and_mtime_identical=True,
                 completion_record=final, read_only_record=readonly_record, read_only_check_passed=True,
                 completion_sha256=file_sha256(completion), report_sha256=file_sha256(report))
    atomic_json(proof_path, proof)
    print('Interface stage recovery passed: ' + phase, flush=True)


def run_phase(phase, entry):
    lock_directory = ROOT / ('runtime/instance_grasp_interface_v11/' + phase + '_wrapper_lock')
    lock_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(lock_directory):
        run_phase_locked(phase, Path(entry))
