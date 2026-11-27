"""Prove that a completed independent audit leaves every protected artifact unchanged."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import accept_dualstream_evaluation_v18 as audit
from scripts import mask_ap_common_v17 as storage
from scripts.pipeline import Pipeline, project_environment

CONFIG = 'configs/dualstream_evaluation_v18.json'
SOURCE = 'scripts/verify_dualstream_evaluation_readonly_v18.py'
TEST = 'tests/test_dualstream_readonly_v18.py'
CHILD = 'scripts/accept_dualstream_evaluation_v18.py'
PROOF = 'results/P4/dualstream_evaluation_v18_readonly_verification.json'
LOGS = 'logs/P4/dualstream_evaluation_v18'
SCOPE = 'Independent recount after completed P4E18; protected SHA, bytes and mtime unchanged; no physical power-cut claim'
require = audit.require


def last_record(path):
    for line in reversed(Path(path).read_text(encoding='utf-8').splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'status' in value:
            return value
    raise ValueError('Read-only child returned no result record')


def prepare(root, config):
    protocol = storage.read(config)
    audit.validate_protocol(protocol)
    require(Path(sys.executable).resolve() == storage.safe(root, protocol['acceptance_interpreter']),
            'Use the frozen independent acceptance interpreter')
    storage.verify_source_maps(root, protocol)
    for name in (SOURCE, TEST, CHILD):
        require(protocol['source_hashes'].get(name) == storage.digest(root/name), 'Read-only source is not frozen: '+name)
    require(PROOF in protocol['budget_paths'] and LOGS in protocol['budget_paths'], 'Read-only evidence is outside the output budget')
    pipeline = Pipeline(root)
    require(pipeline.state['stages'].get('P4E18', {}).get('status') == 'complete', 'P4E18 must complete before final read-only verification')
    stage_fingerprint = pipeline.verify('P4E18')
    proof = root/PROOF
    declarations = list(protocol['source_hashes'])+list(protocol['input_hashes'])+pipeline.stages['P4E18']['inputs']
    require(all(not proof.is_relative_to(storage.safe(root, name)) for name in declarations),
            'Read-only proof must not be an input to its own evaluation')
    report = storage.safe(root, protocol['outputs']['acceptance_report'])
    accepted = storage.read(report)
    require(accepted['schema'] == 18 and accepted['status'] == 'accepted_complete_dualstream_development_comparison'
        and accepted['binding']['protocol_sha256'] == storage.digest(config), 'Completed acceptance report is required')
    audit.durable.verify_bytes(report, accepted['binding'])
    context = audit.extractor.read_parent(root, protocol)
    body = {'schema': 18, 'protocol_sha256': storage.digest(config), 'source_sha256': storage.digest(root/SOURCE),
        'child_sha256': storage.digest(root/CHILD), 'interpreter_sha256': storage.digest(sys.executable),
        'acceptance_report_sha256': storage.digest(report), 'stage_fingerprint': stage_fingerprint,
        'command': [str(storage.safe(root, protocol['acceptance_interpreter'])), '-B', str(root/CHILD), '--config', str(config), '--check-only']}
    return protocol, context, dict(body, fingerprint=storage.fingerprint(body))


def protected_snapshot(root, protocol, config, context):
    result = audit.watched_snapshot(root, protocol, config, context)
    paths = {storage.safe(root, protocol['outputs']['acceptance_report'])}
    for row in context['records']:
        paths.update(storage.safe(root, Path(context['protocol']['data_root'])/name) for name in row['files'].values())
    directories = [storage.safe(root, protocol['outputs']['visuals_dir'])]
    directories += [storage.safe(root, item['run_directory']) for item in context['protocol']['arms'].values()]
    for directory in directories:
        require(directory.is_dir(), 'Missing protected completed directory')
        result.update(storage.snapshot(directory, root))
    for path in paths:
        require(not path.is_symlink(), 'Link in protected inputs or reports')
        result[path.relative_to(root).as_posix()] = audit.file_state(path)
    for name in list(result):
        if name == 'external_base_python':
            continue
        receipt = audit.durable.receipt(root/name)
        if receipt.exists():
            require(not receipt.is_symlink(), 'Link in protected receipt')
            result[receipt.relative_to(root).as_posix()] = audit.file_state(receipt)
    return dict(sorted(result.items()))


def validate_record(record, root, protocol):
    require(record['schema'] == 18 and record['status'] == 'verified_completed_read_only', 'Child did not verify a completed read-only audit')
    for key in ('new_model_forward_calls', 'new_official_evaluations', 'new_training_updates'):
        require(type(record[key]) is int and record[key] == 0, 'Read-only child performed new work: '+key)
    audit.resource_evidence(record, protocol)
    require(record['resources']['peak_gpu_allocated_bytes'] == record['resources']['peak_gpu_reserved_bytes'] == 0,
            'Read-only audit reported GPU allocation')
    report = storage.safe(root, protocol['outputs']['acceptance_report'])
    figure = storage.safe(root, protocol['outputs']['visuals_dir'])/audit.FIGURE
    require(record['report_sha256'] == storage.digest(report) and record['figure_sha256'] == storage.digest(figure),
            'Read-only child report or figure identity differs')


def verify_proof(root, protocol, binding, proof, current):
    require(set(proof) == {'schema', 'status', 'scope', 'binding', 'child_record', 'child_logs', 'files_before', 'files_after',
                          'bytes_and_mtime_unchanged', 'physical_power_loss_tested'}, 'Unexpected read-only proof structure')
    require(proof['schema'] == 18 and proof['status'] == 'passed' and proof['scope'] == SCOPE and proof['binding'] == binding
        and proof['bytes_and_mtime_unchanged'] is True and proof['physical_power_loss_tested'] is False
        and proof['files_before'] == proof['files_after'] == current, 'Completed read-only proof or protected artifact changed')
    validate_record(proof['child_record'], root, protocol)
    require(set(proof['child_logs']) == {'stdout', 'stderr'}, 'Read-only proof must retain both child logs')
    for kind, row in proof['child_logs'].items():
        path = storage.safe(root, row['path'])
        require(path.parent == root/LOGS and re.fullmatch(r'readonly_[0-9a-f]{32}\.'+kind+r'\.log', path.name)
            and not path.is_symlink() and audit.file_state(path) == row['state'], 'Original read-only child log changed')
    require(last_record(root/proof['child_logs']['stdout']['path']) == proof['child_record'], 'Original child record differs from retained stdout')


def invoke(root, protocol, binding, guard):
    guard.check(65536)
    token = uuid.uuid4().hex
    paths = {kind: root/LOGS/('readonly_'+token+'.'+kind+'.log') for kind in ('stdout', 'stderr')}
    (root/LOGS).mkdir(parents=True, exist_ok=True)
    environment = project_environment(root); environment['CUDA_VISIBLE_DEVICES'] = ''
    with paths['stdout'].open('xb', buffering=0) as stdout, paths['stderr'].open('xb', buffering=0) as stderr:
        result = subprocess.run(binding['command'], cwd=root, env=environment, stdout=stdout, stderr=stderr,
                                timeout=protocol['resources']['invocation_soft_seconds']+60)
        os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    require(result.returncode == 0, 'Read-only child stopped; inspect '+str(paths['stderr']))
    record = last_record(paths['stdout'])
    validate_record(record, root, protocol); guard.check()
    return record, {kind: {'path': path.relative_to(root).as_posix(), 'state': audit.file_state(path)} for kind, path in paths.items()}


def execute(root, config, protocol, context, binding):
    output = root/PROOF
    guard = storage.Guard(root, protocol, 'acceptance'); guard.check()
    for path in output.parent.glob(output.name+'.*.tmp'):
        require(re.fullmatch(re.escape(output.name)+r'\.[0-9a-f]{32}\.tmp', path.name)
            and not path.is_symlink() and not output.exists(), 'Unexpected temporary beside completed read-only proof')
        path.unlink()
    before = protected_snapshot(root, protocol, config, context)
    existing = storage.read(output) if output.exists() else None
    own_before = audit.file_state(output) if existing is not None else None
    if existing is not None:
        verify_proof(root, protocol, binding, existing, before)
    record, logs = invoke(root, protocol, binding, guard)
    after = protected_snapshot(root, protocol, config, context)
    require(after == before, 'Read-only audit changed protected SHA, bytes or mtime')
    if existing is None:
        proof = {'schema': 18, 'status': 'passed', 'scope': SCOPE, 'binding': binding, 'child_record': record,
            'child_logs': logs, 'files_before': before, 'files_after': after, 'bytes_and_mtime_unchanged': True,
            'physical_power_loss_tested': False}
        guard.check(len(storage.json_bytes(proof))+4096)
        storage.publish_once(output, storage.json_bytes(proof))
    else:
        require(audit.file_state(output) == own_before, 'Completed read-only proof changed during revalidation')
        verify_proof(root, protocol, binding, storage.read(output), after)
    return {'schema': 18, 'status': 'verified_completed_read_only' if existing is not None else 'passed',
        'proof_sha256': storage.digest(output), 'protected_file_count': len(after), 'child_record': record,
        'child_logs': logs, 'new_model_forward_calls': 0, 'new_official_evaluations': 0, 'new_training_updates': 0}


def run(root=ROOT, config=CONFIG):
    root = Path(root).resolve(); config = storage.safe(root, config)
    protocol, context, binding = prepare(root, config)
    with storage.lock(root/'runtime/locks/dualstream_evaluation_v18_readonly'):
        return execute(root, config, protocol, context, binding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(config=args.config), ensure_ascii=False, allow_nan=False), flush=True)
