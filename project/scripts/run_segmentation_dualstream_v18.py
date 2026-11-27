"""Execute the fixed three-boundary paired dual-stream training budget."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import multiview_common_v14r1 as s
from scripts.pipeline import project_environment

CONFIG = 'configs/segmentation_dualstream_v18.json'
ARMS = ('fixed', 'quality')
BOUNDARIES = (100, 400, 800)


def snapshot(root, directory):
    result = {}
    for file in sorted(directory.rglob('*')):
        s.require(not file.is_symlink(), 'Link in training artifact inventory')
        if file.is_file():
            result[file.relative_to(root).as_posix()] = {'sha256': s.digest(file),
                'bytes': file.stat().st_size, 'mtime_ns': file.stat().st_mtime_ns}
    return result


def budget(root, p, extra=0):
    files = set()
    for name in p['budget_paths']:
        path = s.safe_path(root, name)
        if path.is_file(): files.add(path)
        elif path.exists(): files.update(x for x in path.rglob('*') if x.is_file())
    total = sum(x.stat().st_size for x in files)
    s.require(total+extra <= p['resources']['output_limit_bytes'], 'Dual-stream output budget exceeded')
    s.require(shutil.disk_usage(root).free-extra >= p['resources']['disk_reserve_bytes'], 'Disk reserve exhausted')
    return total


def remaining_boundaries(step):
    s.require(type(step) is int and 0 <= step <= 800, 'Invalid committed research cursor')
    return [limit for limit in BOUNDARIES if limit > step] or [800]


def current_step(directory):
    if not (directory/'config.json').exists():
        s.require(not directory.exists() or not any(directory.iterdir()), 'Unbound training directory')
        return 0
    from src.checkpoints import CheckpointStore, fingerprint
    config, manifest = (s.read(directory/name) for name in ('config.json', 'data_manifest.json'))
    store = CheckpointStore(directory/'checkpoints', fingerprint(config), fingerprint(manifest), keep_last=3)
    artifacts = [x for x in store.directory.iterdir() if x.name != '.checkpoint.lock']
    if not artifacts:
        from scripts.train_segmentation_pilot import assert_no_training_evidence
        assert_no_training_evidence(directory)
        return 0
    for path in store._manifest_paths():
        try:
            metadata, _ = store._read_manifest(path)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
        step = metadata['global_step']
        remaining_boundaries(step)
        return step
    raise RuntimeError('Existing experiment has no verified checkpoint; refuse reinitialization')


def last_record(file, offset):
    with file.open('rb') as stream:
        stream.seek(offset)
        for line in reversed(stream.read().decode('utf-8').splitlines()):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and 'status' in row: return row
    raise RuntimeError('Training child has no current completion record')


def validate_boundary(row, arm, limit):
    s.require(row.get('arm') == arm and row.get('optimizer_steps') == limit
        and row.get('status') == ('trained' if limit == 800 else 'step_limit_reached')
        and row.get('validation_evaluated') is False, 'Unexpected training boundary or evaluation')
    updates = row.get('updates_this_invocation')
    s.require(type(updates) is int and 0 <= updates <= limit, 'Invalid invocation update accounting')


def invoke(root, config, p, arm, directory, limit):
    budget(root, p, 65536)
    command = [str(s.safe_path(root, p['interpreter'])), '-B',
        str(root/'scripts/train_segmentation_dualstream_v18.py'), '--config', str(config),
        '--arm', arm, '--run-dir', str(directory), '--max-steps', str(limit)]
    if (directory/'config.json').exists(): command.append('--resume')
    log = root/'logs/P4/segmentation_dualstream_v18'/('training_%s_through_%d.log' % (arm, limit))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab', buffering=0) as stream:
        stream.write(('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n').encode('utf-8'))
        offset = stream.tell()
        completed = subprocess.run(command, cwd=root, env=project_environment(root), stdout=stream,
            stderr=subprocess.STDOUT, timeout=p['resources']['training_seconds_per_arm']+60)
        os.fsync(stream.fileno())
    s.require(completed.returncode == 0, 'Training stopped; durable checkpoints retained. Inspect ' + str(log))
    budget(root, p)
    row = last_record(log, offset)
    validate_boundary(row, arm, limit)
    return row


def verify_contract(root, config, p):
    s.require(p['schema'] == 18 and p['status'] == 'frozen' and tuple(p['arms']) == ARMS
        and p['planned_steps'] == 800 and p['total_research_update_budget'] == 1600
        and p['segment_boundaries'] == list(BOUNDARIES), 'Unfrozen or unexpected training contract')
    s.require(bool(p['source_hashes']) and bool(p['input_hashes']), 'Training source/input freeze required')
    for group in ('source_hashes', 'input_hashes'):
        for name, checksum in p[group].items():
            s.require(s.digest(s.safe_path(root, name)) == checksum, 'Frozen source/input changed: ' + name)
    budget(root, p)


def execute(root, config, p):
    from scripts import train_segmentation_dualstream_v18 as trainer
    from scripts import verify_dualstream_training_resume_v18 as recovery
    recovery.verify(config, check_only=True, root=root)
    proof_file = s.safe_path(root, p['outputs']['training_resume_report'])
    proof_sha = s.digest(proof_file)
    arms = {}; updates_now = 0
    for arm in ARMS:
        directory = s.safe_path(root, p['arms'][arm]['run_directory'])
        existing = s.read(directory/'result.json') if (directory/'result.json').exists() else {}
        if existing.get('status') != 'trained':
            for limit in remaining_boundaries(current_step(directory)):
                print('Dual-stream %s: through %d of 800' % (arm, limit), flush=True)
                row = invoke(root, config, p, arm, directory, limit)
                updates_now += row['updates_this_invocation']
        before = snapshot(root, directory)
        model, run_config, manifest, result, evidence = trainer.load_final_model(directory, p)
        del model
        s.require(snapshot(root, directory) == before, 'Training verification modified saved files')
        s.require(result['status'] == 'trained' and result['optimizer_steps'] == 800
            and result['validation_evaluated'] is False, 'Incomplete paired research arm')
        arms[arm] = {'run_directory': directory.relative_to(root).as_posix(), 'result': result,
            'result_sha256': s.digest(directory/'result.json'), 'config_sha256': s.digest(directory/'config.json'),
            'history_sha256': s.digest(directory/'history.json'),
            'initialization_sha256': s.digest(directory/'initialization.json'),
            'checkpoint_sha256': evidence['checkpoint_sha256']}
    report = {'schema': 18, 'status': 'completed', 'protocol_sha256': s.digest(config),
        'resume_proof_sha256': proof_sha, 'research_optimizer_steps_completed': 1600,
        'arms': arms, 'automatic_extension': False, 'validation_evaluated': False,
        'efficacy_gate': False, 'scope': 'Fixed-budget matched dual-stream development training',
        'physical_update_count_scope': 'Committed logical steps; interrupted uncommitted work may have been replayed and remains in append-only child logs'}
    file = s.safe_path(root, p['outputs']['training_report']); payload = s.json_bytes(report)
    budget(root, p, 0 if file.exists() else len(payload))
    s.publish_once(file, payload)
    print(json.dumps({'schema': 18, 'status': 'completed', 'phase': 'training',
        'research_optimizer_steps_completed': 1600, 'accepted_updates_this_wrapper_invocation': updates_now,
        'report_sha256': s.digest(file)}), flush=True)
    return report


def run(root=ROOT, config=None):
    root = Path(root).resolve(); config = s.safe_path(root, config or CONFIG)
    p = s.read(config); verify_contract(root, config, p)
    with s.lock(root/'runtime/locks/segmentation_dualstream_v18_wrapper'):
        return execute(root, config, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/CONFIG)
    run(config=parser.parse_args().config)
