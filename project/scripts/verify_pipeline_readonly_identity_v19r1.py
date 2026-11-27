"""Verify the original completed stage under its original scheduler interpreter."""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

CONTRACT = 'configs/geometry_readonly_bridge_v19r1.json'
REVISION = 'geometry_readonly_bridge_v19r1'
STAGE = 'P6E19'
PROTOCOL = 'configs/geometry_propagation_v19.json'
ACCEPTANCE_INTERPRETER = 'runtime/envs/grasp_eval/Scripts/python.exe'
PIPELINE_INTERPRETER = r'D:\python\python.exe'
PIPELINE_VERSION = '3.7.3'
BRIDGE = 'scripts/verify_geometry_readonly_bridge_v19r1.py'
SOURCE = 'scripts/verify_pipeline_readonly_identity_v19r1.py'
TEST = 'tests/test_geometry_readonly_bridge_v19r1.py'
LOGS = 'logs/P6/geometry_propagation_v19/readonly_bridge_v19r1'
PROOF = LOGS + '/proof.json'
SOURCES = (BRIDGE, SOURCE, TEST, 'scripts/pipeline.py',
           'scripts/verify_geometry_propagation_readonly_v19.py',
           'tests/test_geometry_propagation_readonly_v19.py', 'scripts/accept_geometry_propagation_v19.py')
PINNED_INPUTS = (PROTOCOL, 'results/P6/geometry_propagation_v19_acceptance.json',
                 'results/P6/geometry_propagation_v19_acceptance.json.sha256.json')
CONTROL_FILES = ('configs/stages.json', 'runtime/pipeline/state.json', CONTRACT)


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), 'Expected project-relative path')
    root = Path(root).resolve()
    path = (root/name).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError('Path escaped project')
    require(str(relative) != '.', 'Project root cannot be an artifact')
    require(not (root/name).is_symlink(), 'Artifact link is forbidden')
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def file_state(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked evidence file')
    stat = path.stat()
    return {'sha256': digest(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def control_snapshot(root):
    return {name: file_state(safe(root, name)) for name in CONTROL_FILES}


def load_contract(root, contract=CONTRACT, expected_sha=None):
    require(contract == CONTRACT, 'Only the declared supplemental contract is supported')
    path = safe(root, contract)
    if expected_sha is not None:
        require(digest(path) == expected_sha, 'Supplemental contract changed before helper entry')
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    keys = {'schema', 'revision', 'status', 'stage', 'protocol', 'acceptance_interpreter',
            'pipeline_interpreter', 'pipeline_python_version', 'interpreter_sha256',
            'source_hashes', 'input_hashes', 'failure_evidence', 'proof', 'logs', 'resources'}
    require(set(value) == keys, 'Unexpected supplemental contract fields')
    for key, expected in (('schema', 19), ('revision', REVISION), ('status', 'frozen'),
                          ('stage', STAGE), ('protocol', PROTOCOL),
                          ('acceptance_interpreter', ACCEPTANCE_INTERPRETER),
                          ('pipeline_interpreter', PIPELINE_INTERPRETER),
                          ('pipeline_python_version', PIPELINE_VERSION), ('proof', PROOF), ('logs', LOGS)):
        require(value[key] == expected, 'Supplemental contract changed: '+key)
    require(set(value['source_hashes']) == set(SOURCES), 'Supplemental source inventory differs')
    require(set(value['input_hashes']) == set(PINNED_INPUTS), 'Supplemental input inventory differs')
    require(isinstance(value['failure_evidence'], dict) and bool(value['failure_evidence']), 'Failure evidence is required')
    for group in ('source_hashes', 'input_hashes', 'failure_evidence'):
        for name, sha in value[group].items():
            require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) is not None,
                    'Invalid SHA in supplemental contract')
            require(digest(safe(root, name)) == sha, 'Supplemental source/input/evidence changed: '+name)
    require(set(value['interpreter_sha256']) == {'acceptance', 'pipeline'}, 'Interpreter inventory differs')
    for key, interpreter in (('acceptance', safe(root, ACCEPTANCE_INTERPRETER)),
                              ('pipeline', Path(PIPELINE_INTERPRETER))):
        sha = value['interpreter_sha256'][key]
        require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) is not None,
                'Invalid interpreter SHA')
        require(digest(interpreter) == sha, 'Frozen interpreter changed: '+key)
    with safe(root, PROTOCOL).open(encoding='utf-8') as stream:
        protocol = json.load(stream)
    require(value['resources'] == protocol['resources'], 'Original resource limits must remain identical')
    return value


def run(root=ROOT, contract=CONTRACT, expected_sha=None):
    root = Path(root).resolve()
    frozen = load_contract(root, contract, expected_sha)
    require(Path(sys.executable).resolve() == Path(PIPELINE_INTERPRETER).resolve(),
            'Dependency verification requires the original scheduler interpreter')
    version = '.'.join(str(x) for x in sys.version_info[:3])
    require(version == PIPELINE_VERSION, 'Original scheduler Python version differs')
    from scripts.pipeline import Pipeline, digest_json
    before = control_snapshot(root)
    pipeline = Pipeline(root)
    record = pipeline.state['stages'].get(STAGE, {})
    require(record.get('status') == 'complete', 'P6E19 is not complete')
    expected = digest_json({'input_fingerprint': record['input_fingerprint'], 'outputs': record['outputs']})
    actual = pipeline.verify(STAGE)
    require(actual == expected, 'Verified stage fingerprint differs from completed record')
    after = control_snapshot(root)
    require(before == after, 'Dependency verification changed stage configuration or state')
    return {'schema': 19, 'status': 'verified_original_scheduler_dependencies', 'stage': STAGE,
            'stage_fingerprint': actual, 'completed_record_fingerprint': expected,
            'interpreter_path': str(Path(sys.executable).resolve()), 'python_version': version,
            'interpreter_sha256': digest(sys.executable), 'contract_sha256': digest(safe(root, contract)),
            'control_files_before': before, 'control_files_after': after, 'control_files_unchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--contract', default=CONTRACT)
    parser.add_argument('--contract-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.contract, args.contract_sha256), ensure_ascii=True, allow_nan=False), flush=True)
