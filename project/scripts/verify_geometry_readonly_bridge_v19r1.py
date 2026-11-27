"""Supplemental synthetic-geometry entry retaining the original statistical read-only audit."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import verify_geometry_propagation_readonly_v19 as original
from scripts import verify_pipeline_readonly_identity_v19r1 as identity
from scripts.pipeline import Pipeline, digest_json, project_environment, project_lock

store, audit = original.store, original.audit
require = original.require
CONTRACT = identity.CONTRACT
SOURCE = identity.BRIDGE


def expected_stage_fingerprint(pipeline):
    record = pipeline.state['stages'].get(identity.STAGE, {})
    require(record.get('status') == 'complete', 'P6E19 must complete before final read-only verification')
    return digest_json({'input_fingerprint': record['input_fingerprint'], 'outputs': record['outputs']})


def validate_identity(row, root, contract, expected, controls):
    fields = {'schema', 'status', 'stage', 'stage_fingerprint', 'completed_record_fingerprint',
              'interpreter_path', 'python_version', 'interpreter_sha256', 'contract_sha256',
              'control_files_before', 'control_files_after', 'control_files_unchanged'}
    require(set(row) == fields, 'Unexpected original-runtime verification record')
    require(type(row['schema']) is int and row['schema'] == 19
            and row['status'] == 'verified_original_scheduler_dependencies'
            and row['stage'] == identity.STAGE, 'Original-runtime helper did not verify P6E19')
    require(row['stage_fingerprint'] == row['completed_record_fingerprint'] == expected,
            'Real dependency verification differs from completed stage fingerprint')
    require(Path(row['interpreter_path']).resolve() == Path(identity.PIPELINE_INTERPRETER).resolve()
            and row['python_version'] == identity.PIPELINE_VERSION
            and row['interpreter_sha256'] == contract['interpreter_sha256']['pipeline'],
            'Wrong original-runtime helper interpreter')
    require(row['contract_sha256'] == store.digest(root/CONTRACT), 'Helper used another supplemental contract')
    require(row['control_files_unchanged'] is True
            and row['control_files_before'] == row['control_files_after'] == controls,
            'Helper stage configuration or state differs')


def invoke_identity(root, contract, expected, controls):
    directory = store.safe(root, identity.LOGS); directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    paths = {kind: directory/('pipeline_'+token+'.'+kind+'.log') for kind in ('stdout', 'stderr')}
    command = [identity.PIPELINE_INTERPRETER, '-B', str(root/identity.SOURCE), '--root', str(root),
               '--contract', CONTRACT, '--contract-sha256', store.digest(root/CONTRACT)]
    with paths['stdout'].open('xb', buffering=0) as out, paths['stderr'].open('xb', buffering=0) as err:
        result = subprocess.run(command, cwd=root, env=project_environment(root), stdout=out, stderr=err,
                                timeout=contract['resources']['driver_soft_seconds']+60)
        os.fsync(out.fileno()); os.fsync(err.fileno())
    require(result.returncode == 0, 'Original-runtime dependency verification failed; inspect '+str(paths['stderr']))
    require(paths['stdout'].stat().st_size <= 65536, 'Unexpectedly large helper result')
    row = original.last_record(paths['stdout'])
    validate_identity(row, root, contract, expected, controls)
    require(identity.control_snapshot(root) == controls, 'Control files changed during helper execution')
    return {'record': row, 'command': command, 'returncode': result.returncode,
            'logs': {kind: {'path': path.relative_to(root).as_posix(), 'state': store.state(path)}
                     for kind, path in paths.items()}}


def prepare(root):
    contract = identity.load_contract(root)
    config = root/original.CONFIG; protocol = store.read(config)
    require(store.safe(root, protocol['interpreter']) == Path(sys.executable).resolve(), 'Use the frozen synthetic interpreter')
    require(protocol['interpreter'] == identity.ACCEPTANCE_INTERPRETER, 'Original synthetic interpreter path differs')
    for name in (original.SOURCE, original.TEST, original.CHILD, 'scripts/pipeline.py'):
        require(protocol['source_hashes'].get(name) == store.digest(root/name), 'Read-only source is not frozen: '+name)
    old_output = store.safe(root, protocol['outputs']['readonly_report'])
    logs = store.safe(root, protocol['outputs']['logs'])
    proof = store.safe(root, identity.PROOF)
    require(all(protocol['outputs'][key] in protocol['budget_paths'] for key in ('readonly_report', 'logs'))
            and proof.is_relative_to(logs), 'Supplemental evidence is outside the original output budget')
    require(not old_output.exists(), 'Original proof unexpectedly exists; preserve and review it before supplemental execution')
    pipeline = Pipeline(root); expected = expected_stage_fingerprint(pipeline)
    declarations = list(protocol['source_hashes'])+list(protocol['input_hashes'])+pipeline.stages[identity.STAGE]['inputs']
    declarations += [protocol['outputs'][key] for key in ('root', 'report', 'resume_report', 'acceptance_report', 'visuals_dir')]
    declarations += list(contract['source_hashes'])+list(contract['input_hashes'])+list(contract['failure_evidence'])+[CONTRACT]
    require(all(not old_output.is_relative_to(store.safe(root, name)) and not proof.is_relative_to(store.safe(root, name))
                for name in declarations) and not old_output.is_relative_to(logs),
            'Read-only proof must not be an input or part of protected results')
    controls = identity.control_snapshot(root)
    helper = invoke_identity(root, contract, expected, controls)
    context = store.completed_context(root, config)
    report, visuals = audit.acceptance_paths(root, protocol)
    accepted = store.load(report, context['binding'])
    require(accepted['schema'] == 19 and accepted['status'] == 'accepted_limited_synthetic_geometry'
            and accepted['binding'] == context['binding'], 'Completed independent acceptance is required')
    store.recover(report, context['binding'], readonly=True)
    store.validate_layout(visuals, [visuals/name for name in audit.FIGURES], context['binding'], readonly=True)
    for name in audit.FIGURES: store.verify_bytes(visuals/name, context['binding'])
    body = {'schema': 19, 'protocol_sha256': store.digest(config), 'source_sha256': store.digest(root/original.SOURCE),
            'child_sha256': store.digest(root/original.CHILD), 'interpreter_sha256': store.digest(sys.executable),
            'acceptance_report_sha256': store.digest(report), 'stage_fingerprint': expected,
            'supplement_revision': identity.REVISION, 'supplement_contract_sha256': store.digest(root/CONTRACT),
            'supplement_source_sha256': store.digest(root/SOURCE),
            'pipeline_interpreter_sha256': contract['interpreter_sha256']['pipeline'],
            'pipeline_helper_sha256': store.digest(root/identity.SOURCE),
            'command': [str(store.safe(root, protocol['interpreter'])), '-B', str(root/original.CHILD),
                        '--config', str(config), '--check-only']}
    return contract, protocol, context, dict(body, fingerprint=store.fingerprint(body)), helper, controls


def protected_snapshot(root, protocol, context, contract):
    result = original.protected_snapshot(root, protocol, context)
    result['supplemental_sources_inputs_and_evidence'] = {
        name: store.state(store.safe(root, name))
        for name in list(contract['source_hashes'])+list(contract['input_hashes'])+list(contract['failure_evidence'])+[CONTRACT]}
    result['external_pipeline_python_v19r1'] = store.state(Path(identity.PIPELINE_INTERPRETER))
    require(not store.safe(root, protocol['outputs']['readonly_report']).exists(), 'Original proof appeared during supplemental verification')
    return result


def validate_helper_evidence(evidence, root, contract, expected):
    require(set(evidence) == {'record', 'command', 'returncode', 'logs'} and type(evidence['returncode']) is int
            and evidence['returncode'] == 0, 'Helper evidence is incomplete')
    row = evidence['record']
    validate_identity(row, root, contract, expected, row['control_files_before'])
    expected_command = [identity.PIPELINE_INTERPRETER, '-B', str(root/identity.SOURCE), '--root', str(root),
                        '--contract', CONTRACT, '--contract-sha256', store.digest(root/CONTRACT)]
    require(evidence['command'] == expected_command and set(evidence['logs']) == {'stdout', 'stderr'}, 'Helper command/log inventory changed')
    for kind, entry in evidence['logs'].items():
        path = store.safe(root, entry['path'])
        require(path.parent == root/identity.LOGS and re.fullmatch(r'pipeline_[0-9a-f]{32}\.'+kind+r'\.log', path.name)
                and not path.is_symlink() and store.state(path) == entry['state'], 'Original helper log changed')
    require(original.last_record(root/evidence['logs']['stdout']['path']) == row, 'Helper record differs from original stdout')


def verify_proof(proof, root, contract, protocol, binding, current):
    require(set(proof) == {'schema', 'revision', 'status', 'acceptance_proof', 'pipeline_verification',
                          'control_files_before', 'control_files_after', 'control_files_unchanged'}, 'Unexpected supplemental proof fields')
    require(type(proof['schema']) is int and proof['schema'] == 19 and proof['revision'] == identity.REVISION
            and proof['status'] == 'passed' and proof['control_files_unchanged'] is True
            and proof['control_files_before'] == proof['control_files_after'], 'Supplemental proof is invalid')
    original.verify_proof(root, protocol, binding, proof['acceptance_proof'], current)
    validate_helper_evidence(proof['pipeline_verification'], root, contract, binding['stage_fingerprint'])
    require(proof['pipeline_verification']['record']['control_files_before'] == proof['control_files_before'],
            'Proof control files differ from helper control files')


def execute(root, contract, protocol, context, binding, helper, controls):
    output = root/identity.PROOF
    guard = store.Guard(root, protocol); guard.check()
    require(not list(output.parent.glob(output.name+'.*.tmp')), 'Retain and inspect interrupted supplemental proof files before retrying')
    before = protected_snapshot(root, protocol, context, contract)
    existing = store.read(output) if output.exists() else None
    own_before = store.state(output) if existing is not None else None
    if existing is not None: verify_proof(existing, root, contract, protocol, binding, before)
    record, logs = original.invoke(root, protocol, binding, guard)
    after = protected_snapshot(root, protocol, context, contract)
    require(after == before, 'Read-only audit changed protected SHA, bytes or mtime')
    after_controls = identity.control_snapshot(root)
    require(after_controls == controls, 'Stage configuration or pipeline state changed during verification')
    if existing is None:
        inner = {'schema': 19, 'status': 'passed', 'scope': original.SCOPE, 'binding': binding,
                 'child_record': record, 'child_logs': logs, 'files_before': before, 'files_after': after,
                 'bytes_and_mtime_unchanged': True, 'physical_power_loss_tested': False}
        proof = {'schema': 19, 'revision': identity.REVISION, 'status': 'passed', 'acceptance_proof': inner,
                 'pipeline_verification': helper, 'control_files_before': controls,
                 'control_files_after': after_controls, 'control_files_unchanged': True}
        verify_proof(proof, root, contract, protocol, binding, after)
        guard.check(len(store.json_bytes(proof))+4096)
        store.legacy.publish_once(output, store.json_bytes(proof))
    else:
        require(store.state(output) == own_before, 'Supplemental proof changed during revalidation')
        verify_proof(store.read(output), root, contract, protocol, binding, after)
    return {'schema': 19, 'revision': identity.REVISION, 'status': 'verified_read_only' if existing is not None else 'passed',
            'proof_path': identity.PROOF, 'proof_sha256': store.digest(output), 'child_record': record,
            'child_logs': logs, 'pipeline_verification': helper, 'new_random_draw_calls': 0,
            'new_production_samples': 0, 'new_model_forward_calls': 0, 'new_training_updates': 0}


def run(root=ROOT):
    root = Path(root).resolve()
    with project_lock(root), store.lock(root/'runtime/locks/geometry_propagation_v19_readonly'):
        return execute(root, *prepare(root))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    print(json.dumps(run(), ensure_ascii=True, allow_nan=False), flush=True)
