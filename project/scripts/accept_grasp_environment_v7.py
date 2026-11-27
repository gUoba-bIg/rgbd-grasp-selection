"""Offline, read-only environment verification; publish a separate immutable acceptance."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/P2/grasp_environment_acceptance_v7.json'
STATUS = 'limited_engineering_ready_nondegenerate_paths_passed'
SETUP_PATH = ROOT / 'scripts/setup_grasp_evaluation_environment_v7.py'
SPEC = importlib.util.spec_from_file_location('grasp_environment_v7_setup', SETUP_PATH)
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def within_root(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Report path escaped project')
    return path


def validate_package_inventory(actual, expected, lock):
    require(actual == expected, 'Installed inventory changed since numerical check')
    normalized = {canonical(name): version for name, version in actual.items()}
    require(len(normalized) == len(actual), 'Duplicate normalized package names')
    locked = {canonical(row['name']): row['version'] for row in lock}
    require(len(locked) == len(lock), 'Duplicate dependency lock names')
    require(normalized == dict(locked, pip='25.3'), 'Installed versions differ from full dependency lock plus bootstrap pip')
    require(not any(name.startswith(('torch', 'nvidia-', 'cupy', 'tensorflow')) for name in normalized),
            'Unexpected accelerator dependency')


def validate_geometry_report(report):
    require(report['status'] == STATUS and not report['failures'], 'Numerical checks did not complete with the limited status')
    require(report['dataset_ap_computed'] is False and report['dataset_frames_read'] == 0
            and report['gpu_used'] is False and report['all_geometry_reliable'] is False,
            'Synthetic-only scientific boundary changed')
    expected = ('official_graspgroup_compiled_nms', 'official_collision_and_dex_conversion',
                'rigid_transform_meters', 'sdf_mesh_alias_compatibility',
                'official_force_closure_contacts', 'official_quality_contact_search_and_score',
                'official_cvxopt_force_closure_qp')
    for name in expected:
        require(report['checks'][name]['passed'] is True, 'Required numerical check not passed: ' + name)
    edge = report['checks']['automatic_contact_search_diagnostic']
    require(edge['known_upstream_limitation'] is True and edge['explicit_contact_quality'] == 1
            and edge['searched_contact_quality'] == 0 and edge['contacts_found'] is False
            and edge['linear_zero_crossing_returned_none'] is True,
            'Known axial-sphere limitation was removed or changed')


def metadata_inventory(python):
    command = 'import importlib.metadata,json,sys;print(json.dumps({"prefix":sys.prefix,"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}}))'
    return json.loads(subprocess.check_output([str(python), '-I', '-B', '-c', command], text=True))


def verify():
    result_path = ROOT / 'results/P2/grasp_eval_environment_v7.json'
    geometry_path = ROOT / 'results/P2/grasp_geometry_v7.json'
    result, geometry = read(result_path), read(geometry_path)
    require(result['status'] == STATUS and not result['failures'], 'Environment preparation is incomplete')
    require(result['setup_script_sha256'] == SETUP.digest(SETUP_PATH), 'Setup source changed')
    require(result['geometry_report_sha256'] == SETUP.digest(geometry_path), 'Numerical report changed')
    require(result['geometry_script_sha256'] == geometry['script_sha256'] == SETUP.digest(ROOT / 'scripts/verify_grasp_geometry_v7.py'),
            'Numerical test source changed')
    require(result['patch_sha256'] == SETUP.digest(SETUP.PATCH), 'Compatibility patch changed')
    require(result['source_commit'] == SETUP.COMMIT == SETUP.git('rev-parse', 'HEAD') and not SETUP.git('status', '--porcelain'),
            'Frozen third-party source changed')
    require(SETUP.tree_hash(SETUP.UPSTREAM) == result['upstream_tree_sha256_before'] == result['upstream_tree_sha256_after'],
            'Frozen third-party source bytes changed')
    require(SETUP.audit_clone_source() == result['runtime_clone_audit'], 'Runtime clone audit changed')
    require(SETUP.audit_installed_source() == result['installed_source_audit'], 'Installed API source changed')
    require('include-system-site-packages = false' in (SETUP.ENV / 'pyvenv.cfg').read_text(), 'Environment is not isolated')
    compiled = [{'file': str(path.relative_to(ROOT)), 'sha256': SETUP.digest(path)}
                for path in (SETUP.ENV / 'Lib/site-packages').glob('grasp_nms*.pyd')]
    require(len(compiled) == 1 and compiled == result['compiled_nms'], 'Compiled NMS changed')
    for key, filename in (('resolution_sha256', 'resolution.json'), ('dependency_lock_sha256', 'resolved_dependency_lock.json')):
        require(result[key] == SETUP.digest(SETUP.DOWNLOADS / filename), 'Frozen dependency evidence changed: ' + filename)
    resolution_binding = read(SETUP.DOWNLOADS / 'resolution_binding.json')
    require(resolution_binding['requirements_sha256'] == SETUP.digest(ROOT / 'configs/grasp_eval_v7_requirements.txt')
            and resolution_binding['source_commit'] == SETUP.COMMIT
            and resolution_binding['patch_sha256'] == SETUP.digest(SETUP.PATCH), 'Dependency resolution inputs changed')
    for item in read(SETUP.DOWNLOADS / 'archive_audit.json')['archives']:
        require(SETUP.digest(within_root(item['path'])) == item['sha256'], 'Downloaded archive changed: ' + item['name'])
    inventory = metadata_inventory(SETUP.ENV / 'Scripts/python.exe')
    require(Path(inventory['prefix']).resolve() == SETUP.ENV.resolve(), 'Evaluation Python resolved to another environment')
    validate_package_inventory(inventory['packages'], geometry['packages'], read(SETUP.DOWNLOADS / 'resolved_dependency_lock.json'))
    caller = ROOT / 'runtime/envs/vision'
    require(Path(result['caller_environment']).resolve() == caller.resolve(), 'Unexpected source training environment')
    caller_packages = metadata_inventory(caller / 'Scripts/python.exe')['packages']
    caller_hash = hashlib.sha256(json.dumps(caller_packages, sort_keys=True).encode()).hexdigest()
    require(caller_hash == result['caller_packages_sha256_before'] == result['caller_packages_sha256_after'],
            'Training environment package inventory changed')
    validate_geometry_report(geometry)
    for attempt in geometry['fixture_revision']['prior_attempts']:
        require(SETUP.digest(within_root(attempt['file'])) == attempt['sha256'], 'Historical failed attempt changed')
    check = subprocess.run([str(SETUP.ENV / 'Scripts/python.exe'), '-I', '-B', '-m', 'pip', 'check'], capture_output=True, text=True)
    require(check.returncode == 0, 'Current package dependency consistency check failed: ' + check.stdout + check.stderr)
    footprint = SETUP.check_budget()
    bindings = {str(path.relative_to(ROOT)): SETUP.digest(path) for path in (
        Path(__file__), SETUP_PATH, ROOT / 'scripts/verify_grasp_geometry_v7.py', result_path, geometry_path,
        SETUP.PATCH, ROOT / 'configs/grasp_eval_v7_requirements.txt', SETUP.DOWNLOADS / 'resolved_dependency_lock.json',
        SETUP.DOWNLOADS / 'resolution.json', SETUP.DOWNLOADS / 'resolution_binding.json', SETUP.DOWNLOADS / 'archive_audit.json')}
    return {'schema': 7, 'status': 'accepted_limited_engineering_readiness', 'bindings': bindings,
            'dataset_ap_computed': False, 'dataset_frames_read': 0, 'gpu_used': False,
            'numerical_tests_rerun': False, 'packages_installed_or_downloaded': False,
            'environment_report_unchanged': True, 'source_checks_passed': True, 'compiled_nms': compiled,
            'dependency_packages': len(inventory['packages']), 'footprint_bytes': footprint,
            'pip_check': check.stdout.strip(), 'known_limitation': result['known_upstream_limitation'],
            'all_geometry_reliable': False, 'remaining': result['remaining']}


def publish_once(path, report):
    if path.exists():
        require(read(path) == report, 'Existing acceptance differs; inspect rather than overwrite')
        return 'verified_existing_no_write'
    SETUP.atomic_json(path, report)
    return 'written'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify without writing any result')
    args = parser.parse_args()
    report = verify()
    disposition = 'checked_no_write' if args.check_only else publish_once(OUTPUT, report)
    print(json.dumps({'status': report['status'], 'output': str(OUTPUT), 'disposition': disposition,
                      'footprint_bytes': report['footprint_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
