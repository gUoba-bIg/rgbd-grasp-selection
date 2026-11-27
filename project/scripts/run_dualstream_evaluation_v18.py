"""Retain actual two-unit restart evidence for all fixed evaluation cohorts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import mask_ap_common_v17 as common
from scripts import extract_dualstream_scores_v18 as scores
from scripts.pipeline import project_environment


def last_record(file, offset):
    with file.open('rb') as stream:
        stream.seek(offset)
        for line in reversed(stream.read().decode('utf-8').splitlines()):
            try: row = json.loads(line)
            except ValueError: continue
            if isinstance(row, dict) and 'status' in row: return row
    raise RuntimeError('Evaluation child has no current result record')


def validate_record(row, phase, key, operation):
    unit = 'frames' if phase == 'scores' else 'blocks'
    expected = 160 if phase == 'scores' else 88
    common.require(row.get('schema') == 18 and row.get('expected_'+unit) == expected,
                   'Wrong evaluation child identity')
    if phase == 'scores':
        condition, arm = key.split('/')
        common.require(row.get('condition') == condition and row.get('arm') == arm, 'Wrong score cohort')
    for name in ('new_'+unit, 'reused_'+unit, 'committed_'+unit, 'new_model_forward_calls', 'new_training_updates'):
        common.require(type(row.get(name)) is int and row[name] >= 0, 'Invalid evaluation work accounting')
    common.require(row['new_'+unit]+row['reused_'+unit] == row['committed_'+unit] <= expected
        and row['new_training_updates'] == 0, 'Unexpected training or evaluation work')
    if phase == 'scores':
        common.require(row['new_model_forward_calls'] <= row['new_frames'], 'Extra forward calls in one successful invocation')
    else:
        common.require(row['new_model_forward_calls'] == 0 and type(row.get('new_official_evaluations')) is int
            and 0 <= row['new_official_evaluations'] <= row['new_blocks'], 'Metric invocation performed unexpected work')
    if operation == 'probe':
        common.require(row['status'] == 'partial' and row['committed_'+unit] == 2, 'Probe must stop at two units')
    else:
        common.require(row['status'] in ('completed', 'verified_completed_read_only')
            and row['committed_'+unit] == expected, 'Evaluation is incomplete')
    if operation == 'read_only' or row['status'] == 'verified_completed_read_only':
        common.require(row['status'] == 'verified_completed_read_only'
            and row['new_'+unit] == row['new_model_forward_calls'] == 0, 'Read-only performed new work')
        if phase == 'metrics': common.require(row['new_official_evaluations'] == 0, 'Read-only reran official evaluation')


def invoke(root, p, command, name, guard):
    guard.check(65536)
    log = root/'logs/P4/dualstream_evaluation_v18'/name
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab', buffering=0) as stream:
        stream.write(('\nCOMMAND '+json.dumps(command, ensure_ascii=False)+'\n').encode('utf-8'))
        offset = stream.tell()
        result = subprocess.run(command, cwd=root, env=project_environment(root), stdout=stream,
            stderr=subprocess.STDOUT, timeout=p['resources']['invocation_soft_seconds']+60)
        os.fsync(stream.fileno())
    common.require(result.returncode == 0, 'Evaluation stopped; saved work retained. Inspect '+str(log))
    guard.check()
    return last_record(log, offset)


def layout(root, p, phase, key):
    if phase == 'scores':
        directory = common.safe(root, p['outputs']['scores_root'])/key
        return directory, directory/'report.json'
    return common.safe(root, p['outputs']['metrics_root']), common.safe(root, p['outputs']['metrics_report'])


def completed_snapshot(root, directory, report):
    result = common.snapshot(directory, root)
    if not report.is_relative_to(directory):
        for file in (report, common.receipt_path(report)):
            common.require(file.is_file(), 'Missing external metric report or receipt')
            result[file.relative_to(root).as_posix()] = {'sha256': common.digest(file),
                'bytes': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns}
    return result


def checked_prefix(root, directory, prefix):
    observed = common.snapshot(directory, root)
    common.require(bool(prefix) and all(observed.get(k) == v for k, v in prefix.items()),
                   'Saved prefix changed SHA, size or mtime')
    return {k: observed[k] for k in prefix}


def unrecorded_probe_count(root, directory, p, phase):
    report = directory/'report.json' if phase == 'scores' else common.safe(root, p['outputs']['metrics_report'])
    common.require(not any(file.exists() for file in (report, common.receipt_path(report),
        directory/'completion.json', common.receipt_path(directory/'completion.json'))),
        'Terminal artifacts lack recorded two-unit restart history')
    if phase == 'scores':
        permitted = [directory/'frames'/('scene_%04d_%04d' % (scene, frame))
                     for scene in p['validation_scenes'] for frame in p['frames']][:2]
        candidates = list((directory/'frames').iterdir()) if (directory/'frames').exists() else []
        marker = 'record.json'
    else:
        permitted = [directory/condition/arm/scope for condition in p['conditions'] for arm in p['arms']
                     for scope in ['scene_%04d' % scene for scene in p['validation_scenes']]+['pooled']][:2]
        candidates = [x for condition in p['conditions'] for arm in p['arms']
                      if (directory/condition/arm).exists() for x in (directory/condition/arm).iterdir()]
        marker = 'completion.json'
    common.require(all(x in permitted and x.is_dir() for x in candidates),
                   'Unrecorded payload or attempt extends beyond the two-unit boundary')
    return sum((x/marker).is_file() for x in permitted)


def execute(root, config, p, phase):
    common.require(phase in ('scores', 'metrics'), 'Unknown evaluation phase')
    scores.verify_design(p); scores.verify_maps(root, p)
    interpreter = common.safe(root, p['interpreter'] if phase == 'scores' else p['evaluation_interpreter'])
    child = root/('scripts/extract_dualstream_scores_v18.py' if phase == 'scores'
                  else 'scripts/evaluate_dualstream_mask_ap_v18.py')
    proof_file = common.safe(root, p['outputs']['score_resume_report' if phase == 'scores' else 'metrics_resume_report'])
    binding = {'protocol_sha256': common.digest(config), 'wrapper_sha256': common.digest(Path(__file__)),
        'child_sha256': common.digest(child), 'interpreter_sha256': common.digest(interpreter),
        'pipeline_sha256': common.digest(root/'scripts/pipeline.py')}
    keys = [condition+'/'+arm for condition in p['conditions'] for arm in p['arms']] if phase == 'scores' else ['all']
    proof = common.read(proof_file) if proof_file.exists() else {'schema': 18, 'phase': phase,
        'status': 'in_progress', 'binding': binding, 'entries': {},
        'scope': 'Actual process boundaries with two units first; no physical power-cut claim'}
    common.require(proof['schema'] == 18 and proof['phase'] == phase and proof['binding'] == binding
        and proof['status'] in ('in_progress', 'passed') and set(proof['entries']).issubset(keys), 'Recovery proof binding differs')

    def save():
        scores.Guard(root, p).check(len(common.json_bytes(proof))+1024)
        common.atomic(proof_file, proof)

    if not proof_file.exists(): save()
    for key in keys:
        directory, report = layout(root, p, phase, key)

        def call(operation, flags):
            command = [str(interpreter), '-B', str(child), '--config', str(config)]
            if phase == 'scores':
                condition, arm = key.split('/')
                command += ['--condition', condition, '--arm', arm]
            command += flags
            print('Dual-stream evaluation %s %s: %s' % (phase, key, operation), flush=True)
            row = invoke(root, p, command, phase+'_'+key.replace('/', '_')+'_'+operation+'.log', scores.Guard(root, p))
            validate_record(row, phase, key, operation)
            return row

        entry = proof['entries'].get(key)
        if entry is None:
            common.require(proof['status'] == 'in_progress' and not (directory/'completion.json').exists(),
                           'Completed cohort lacks actual restart history')
            entry = {'status': 'in_progress'}; proof['entries'][key] = entry; save()
        if entry['status'] == 'passed':
            common.require(completed_snapshot(root, directory, report) == entry['completed_snapshot'], 'Sealed evaluation changed')
            call('read_only', ['--check-only'])
            common.require(completed_snapshot(root, directory, report) == entry['completed_snapshot'], 'Read-only changed evaluation')
            continue
        common.require(entry['status'] == 'in_progress' and proof['status'] == 'in_progress', 'Unknown recovery state')
        if 'prefix_before' not in entry:
            count = unrecorded_probe_count(root, directory, p, phase)
            flag = '--max-new-frames' if phase == 'scores' else '--max-new-blocks'
            row = call('probe', [flag, str(2-count)])
            entry.update(probe_record=row, prefix_before=common.snapshot(directory, root)); save()
        checked_prefix(root, directory, entry['prefix_before'])
        complete = call('completion', [])
        after = checked_prefix(root, directory, entry['prefix_before'])
        before = completed_snapshot(root, directory, report)
        readonly = call('read_only', ['--check-only'])
        common.require(completed_snapshot(root, directory, report) == before, 'Read-only changed SHA, size or mtime')
        common.require(complete['report_sha256'] == readonly['report_sha256'] == common.digest(report), 'Report SHA differs')
        entry.update(status='passed', prefix_after=after, completion_record=complete, read_only_record=readonly,
            byte_and_mtime_identical=True, read_only_check_passed=True, completed_snapshot=before,
            completion_sha256=common.digest(directory/'completion.json'), report_sha256=common.digest(report))
        save()
    if proof['status'] != 'passed': proof['status'] = 'passed'; save()
    print(json.dumps({'schema': 18, 'status': 'passed', 'phase': phase, 'proof_sha256': common.digest(proof_file)}), flush=True)
    return proof


def run(root=ROOT, config=None, phase=None):
    root = Path(root).resolve(); config = common.safe(root, config or scores.CONFIG)
    common.require(phase in ('scores', 'metrics'), 'Unknown evaluation phase')
    with common.lock(root/'runtime/locks/dualstream_evaluation_v18_wrapper'/phase):
        return execute(root, config, common.read(config), phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/scores.CONFIG)
    parser.add_argument('--phase', choices=('scores', 'metrics'), required=True)
    args = parser.parse_args(); run(config=args.config, phase=args.phase)
