"""Retain a final read-only recount after the synthetic acceptance stage."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts import accept_geometry_propagation_v19 as audit
from scripts import geometry_storage_v19 as store
from scripts.pipeline import Pipeline, project_environment

CONFIG = audit.CONFIG
SOURCE = 'scripts/verify_geometry_propagation_readonly_v19.py'
TEST = 'tests/test_geometry_propagation_readonly_v19.py'
CHILD = audit.SOURCE
STAGE = 'P6E19'
SCOPE = 'Independent recount after completed P6E19; SHA, bytes and mtime unchanged; no physical power-cut claim'
require = store.require


def last_record(path):
    for line in reversed(Path(path).read_text(encoding='utf-8').splitlines()):
        try: value = json.loads(line)
        except ValueError: continue
        if isinstance(value, dict) and 'status' in value: return value
    raise RuntimeError('Read-only child returned no result record')


def prepare(root, config):
    protocol = store.read(config)
    require(store.safe(root, protocol['interpreter']) == Path(sys.executable).resolve(), 'Use the frozen synthetic interpreter')
    for name in (SOURCE, TEST, CHILD, 'scripts/pipeline.py'):
        require(protocol['source_hashes'].get(name) == store.digest(root/name), 'Read-only source is not frozen: '+name)
    output = store.safe(root, protocol['outputs']['readonly_report'])
    logs = store.safe(root, protocol['outputs']['logs'])
    require(all(protocol['outputs'][key] in protocol['budget_paths'] for key in ('readonly_report', 'logs')),
            'Read-only evidence is outside the output budget')
    pipeline = Pipeline(root)
    require(pipeline.state['stages'].get(STAGE, {}).get('status') == 'complete', 'P6E19 must complete before final read-only verification')
    declarations = list(protocol['source_hashes'])+list(protocol['input_hashes'])+pipeline.stages[STAGE]['inputs']
    declarations += [protocol['outputs'][key] for key in ('root', 'report', 'resume_report', 'acceptance_report', 'visuals_dir')]
    require(all(not output.is_relative_to(store.safe(root, name)) for name in declarations)
            and not output.is_relative_to(logs), 'Read-only proof must not be an input or part of protected results')
    stage_fingerprint = pipeline.verify(STAGE)
    context = store.completed_context(root, config)
    report, visuals = audit.acceptance_paths(root, protocol)
    accepted = store.load(report, context['binding'])
    require(accepted['schema'] == 19 and accepted['status'] == 'accepted_limited_synthetic_geometry'
            and accepted['binding'] == context['binding'], 'Completed independent acceptance is required')
    store.recover(report, context['binding'], readonly=True)
    store.validate_layout(visuals, [visuals/name for name in audit.FIGURES], context['binding'], readonly=True)
    for name in audit.FIGURES: store.verify_bytes(visuals/name, context['binding'])
    body = {'schema': 19, 'protocol_sha256': store.digest(config), 'source_sha256': store.digest(root/SOURCE),
        'child_sha256': store.digest(root/CHILD), 'interpreter_sha256': store.digest(sys.executable),
        'acceptance_report_sha256': store.digest(report), 'stage_fingerprint': stage_fingerprint,
        'command': [str(store.safe(root, protocol['interpreter'])), '-B', str(root/CHILD), '--config', str(config), '--check-only']}
    return protocol, context, dict(body, fingerprint=store.fingerprint(body))


def protected_snapshot(root, protocol, context):
    report, visuals = audit.acceptance_paths(root, protocol)
    return {'evidence': audit.evidence_state(root, context),
        'acceptance_report': {p.relative_to(root).as_posix(): store.state(p) for p in (report, store.receipt(report))},
        'figures': store.snapshot(visuals)}


def validate_record(row, root, protocol):
    require(row['schema'] == 19 and row['status'] == 'verified_read_only' and type(row['case_count']) is int
            and row['case_count'] == 72, 'Child did not verify a completed read-only audit')
    for key in ('new_random_draw_calls', 'new_production_samples', 'new_model_forward_calls', 'new_training_updates'):
        require(type(row[key]) is int and row[key] == 0, 'Read-only child performed new work: '+key)
    limits, resources = protocol['resources'], row['resources']
    for key, cap in (('elapsed_seconds', limits['driver_soft_seconds']), ('peak_rss_bytes', limits['rss_soft_bytes']),
                     ('current_output_bytes', limits['output_limit_bytes'])):
        require(type(resources[key]) in (int, float) and math.isfinite(resources[key]) and 0 <= resources[key] <= cap,
                'Read-only child exceeded fixed resource limits')
    for key in ('peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'):
        require(type(resources[key]) is int and resources[key] == 0, 'Read-only child reported GPU allocation')
    require(row['report_sha256'] == store.digest(store.safe(root, protocol['outputs']['acceptance_report'])), 'Read-only report SHA differs')


def verify_proof(root, protocol, binding, proof, current):
    require(set(proof) == {'schema', 'status', 'scope', 'binding', 'child_record', 'child_logs', 'files_before', 'files_after',
        'bytes_and_mtime_unchanged', 'physical_power_loss_tested'}, 'Unexpected read-only proof structure')
    require(proof['schema'] == 19 and proof['status'] == 'passed' and proof['scope'] == SCOPE and proof['binding'] == binding
        and proof['files_before'] == proof['files_after'] == current and proof['bytes_and_mtime_unchanged'] is True
        and proof['physical_power_loss_tested'] is False, 'Read-only proof or protected artifact changed')
    validate_record(proof['child_record'], root, protocol)
    require(set(proof['child_logs']) == {'stdout', 'stderr'}, 'Read-only proof requires both child logs')
    for kind, row in proof['child_logs'].items():
        path = store.safe(root, row['path'])
        require(path.parent == store.safe(root, protocol['outputs']['logs'])
            and re.fullmatch(r'readonly_[0-9a-f]{32}\.'+kind+r'\.log', path.name) and store.state(path) == row['state'],
            'Original read-only child log changed')
    require(last_record(root/proof['child_logs']['stdout']['path']) == proof['child_record'], 'Original stdout record differs')


def invoke(root, protocol, binding, guard):
    guard.check(65536)
    folder = store.safe(root, protocol['outputs']['logs']); folder.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    paths = {kind: folder/('readonly_'+token+'.'+kind+'.log') for kind in ('stdout', 'stderr')}
    environment = project_environment(root); environment['CUDA_VISIBLE_DEVICES'] = ''
    with paths['stdout'].open('xb', buffering=0) as stdout, paths['stderr'].open('xb', buffering=0) as stderr:
        result = subprocess.run(binding['command'], cwd=root, env=environment, stdout=stdout, stderr=stderr,
                                timeout=protocol['resources']['driver_soft_seconds']+60)
        os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    require(result.returncode == 0, 'Read-only child stopped; inspect '+str(paths['stderr']))
    row = last_record(paths['stdout']); validate_record(row, root, protocol); guard.check()
    return row, {kind: {'path': path.relative_to(root).as_posix(), 'state': store.state(path)} for kind, path in paths.items()}


def execute(root, protocol, context, binding):
    output = store.safe(root, protocol['outputs']['readonly_report'])
    guard = store.Guard(root, protocol); guard.check()
    for path in output.parent.glob(output.name+'.*.tmp'):
        require(re.fullmatch(re.escape(output.name)+r'\.[0-9a-f]{32}\.tmp', path.name)
                and not path.is_symlink() and not output.exists(), 'Unexpected temporary beside read-only proof')
        path.unlink()
    before = protected_snapshot(root, protocol, context)
    existing = store.read(output) if output.exists() else None
    own_before = store.state(output) if existing is not None else None
    if existing is not None: verify_proof(root, protocol, binding, existing, before)
    row, logs = invoke(root, protocol, binding, guard)
    after = protected_snapshot(root, protocol, context)
    require(after == before, 'Read-only audit changed protected SHA, bytes or mtime')
    if existing is None:
        proof = {'schema': 19, 'status': 'passed', 'scope': SCOPE, 'binding': binding, 'child_record': row,
            'child_logs': logs, 'files_before': before, 'files_after': after,
            'bytes_and_mtime_unchanged': True, 'physical_power_loss_tested': False}
        guard.check(len(store.json_bytes(proof))+4096); store.legacy.publish_once(output, store.json_bytes(proof))
    else:
        require(store.state(output) == own_before, 'Completed read-only proof changed')
        verify_proof(root, protocol, binding, store.read(output), after)
    return {'schema': 19, 'status': 'verified_read_only' if existing is not None else 'passed',
        'proof_sha256': store.digest(output), 'child_record': row, 'child_logs': logs,
        'new_random_draw_calls': 0, 'new_production_samples': 0, 'new_model_forward_calls': 0, 'new_training_updates': 0}


def run(root=ROOT, config=CONFIG):
    root = Path(root).resolve(); config = store.safe(root, config)
    protocol, context, binding = prepare(root, config)
    with store.lock(root/'runtime/locks/geometry_propagation_v19_readonly'):
        return execute(root, protocol, context, binding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', default=CONFIG)
    print(json.dumps(run(config=parser.parse_args().config), ensure_ascii=False, allow_nan=False), flush=True)
