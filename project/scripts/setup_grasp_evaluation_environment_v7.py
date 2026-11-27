"""Bounded, isolated CPU geometry environment; no dataset evaluation or GPU work."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / 'runtime/envs/grasp_eval'
BUILD = ROOT / 'runtime/build/graspnetAPI_eval_v7'
DOWNLOADS = ROOT / 'runtime/downloads/grasp_eval_v7'
TEMP = ROOT / 'runtime/temp/grasp_eval_v7'
LOGS = ROOT / 'logs/P2/grasp_eval_v7'
RESULT = ROOT / 'results/P2/grasp_eval_environment_v7.json'
UPSTREAM = ROOT / 'third_party/graspnetAPI'
COMMIT = 'eb57dd2092d8dbe05312a29c3d0c22f3226efbfc'
CAP = 2 * 2**30
BOOTSTRAP = ['pip==25.3', 'setuptools==75.8.0', 'wheel==0.45.1', 'numpy==1.26.4', 'Cython==3.0.12']
PATCH = ROOT / 'configs/graspnetAPI_eval_v7_numpy.patch'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def footprint():
    return sum(p.stat().st_size for root in (ENV, BUILD, DOWNLOADS, TEMP) if root.exists()
               for p in root.rglob('*') if p.is_file())


def check_budget(extra=0):
    used = footprint()
    if used + extra > CAP or shutil.disk_usage(ROOT).free - extra < 60 * 2**30:
        raise RuntimeError('V7 2 GiB total footprint or 60 GiB reserve would be exceeded')
    return used


def run(command, name):
    check_budget()
    LOGS.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                        'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PIP_NO_CACHE_DIR': '1',
                        'TEMP': str(TEMP), 'TMP': str(TEMP), 'MPLBACKEND': 'Agg',
                        'CUDA_VISIBLE_DEVICES': ''})
    TEMP.mkdir(parents=True, exist_ok=True)
    print(name, flush=True)
    with (LOGS / (name + '.log')).open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND ' + json.dumps([str(v) for v in command], ensure_ascii=False) + '\n')
        stream.flush()
        process = subprocess.Popen([str(v) for v in command], cwd=ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(.5)
            try:
                check_budget()
            except Exception:
                process.terminate()
                process.wait(timeout=15)
                raise
        stream.flush()
        os.fsync(stream.fileno())
    check_budget()
    if process.returncode:
        raise RuntimeError('{} failed: code {}; inspect {}'.format(name, process.returncode, LOGS / (name + '.log')))


def git(*args):
    return subprocess.check_output(['git', '-C', str(UPSTREAM), *args], text=True).strip()


def tree_hash(root):
    values = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and '.git' not in path.relative_to(root).parts and '__pycache__' not in path.parts:
            values[path.relative_to(root).as_posix()] = digest(path)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def audit_installed_source():
    source = BUILD / 'graspnetAPI'
    installed = ENV / 'Lib/site-packages/graspnetAPI'
    def files(root):
        return {path.relative_to(root).as_posix(): digest(path)
                for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    original, actual = files(source), files(installed)
    omitted = sorted(original.keys() - actual.keys())
    allowed_omissions = ['utils/dexnet/LICENSE', 'utils/dexnet/grasping/meshpy/LICENSE']
    if omitted != allowed_omissions or actual.keys() - original.keys() or any(
            actual[name] != original[name] for name in actual):
        raise RuntimeError('Installed API differs from patched clone beyond the two upstream wheel-omitted license files')
    return {'source_sha256': tree_hash(source), 'installed_sha256': tree_hash(installed),
            'omitted_noncode_files': omitted, 'matched_files': len(actual)}


def audit_clone_source():
    changes = {
        'graspnetAPI/utils/dexnet/grasping/meshpy/sdf.py': [
            ('dtype=np.int)', 'dtype=int)'), ('.astype(np.int)', '.astype(int)')],
        'graspnetAPI/utils/dexnet/grasping/meshpy/mesh.py': [('.astype(np.int)', '.astype(int)')]}
    tracked = subprocess.check_output(['git', '-C', str(UPSTREAM), 'ls-files', '-z']).decode().split('\0')
    for name in filter(None, tracked):
        expected = (UPSTREAM / name).read_bytes()
        for old, new in changes.get(name, []):
            if expected.count(old.encode()) != 1:
                raise RuntimeError('Alias patch is not uniquely applicable: ' + name)
            expected = expected.replace(old.encode(), new.encode())
        if not (BUILD / name).is_file() or (BUILD / name).read_bytes() != expected:
            raise RuntimeError('Runtime source has an unapproved change: ' + name)
    return {'tracked_files': len(list(filter(None, tracked))), 'allowed_changed_files': sorted(changes)}


def prepare():
    if git('rev-parse', 'HEAD') != COMMIT or git('status', '--porcelain'):
        raise RuntimeError('Frozen upstream source is not the expected clean commit')
    for path in (DOWNLOADS, LOGS, TEMP):
        path.mkdir(parents=True, exist_ok=True)
    if not (ENV / 'Scripts/python.exe').exists():
        if ENV.exists() and any(ENV.iterdir()):
            raise RuntimeError('Incomplete environment must be inspected rather than overwritten')
        run([sys.executable, '-m', 'venv', str(ENV)], 'create_environment')
    if 'include-system-site-packages = false' not in (ENV / 'pyvenv.cfg').read_text():
        raise RuntimeError('Independent environment must not inherit system packages')
    if not BUILD.exists():
        run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(UPSTREAM), str(BUILD)], 'clone_source')
        run(['git', '-C', str(BUILD), 'checkout', '--detach', COMMIT], 'checkout_source')
    if subprocess.check_output(['git', '-C', str(BUILD), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise RuntimeError('Runtime source clone is not the pinned commit')
    # Only this explicit patch may modify the cloned source; old source stays untouched.
    reverse = subprocess.run(['git', '-C', str(BUILD), 'apply', '--unidiff-zero', '--reverse', '--check', str(PATCH)], capture_output=True)
    if reverse.returncode:
        run(['git', '-C', str(BUILD), 'apply', '--unidiff-zero', '--check', str(PATCH)], 'check_patch')
        run(['git', '-C', str(BUILD), 'apply', '--unidiff-zero', str(PATCH)], 'apply_patch')
    audit_clone_source()


def verify_pins():
    pins = BOOTSTRAP + [line.strip() for line in (ROOT / 'configs/grasp_eval_v7_requirements.txt').read_text().splitlines()
                        if '==' in line and not line.startswith('#')]
    records = {}
    for pin in sorted(set(pins)):
        name, version = pin.split('==')
        with urllib.request.urlopen('https://pypi.org/pypi/{}/{}/json'.format(name, version), timeout=30) as response:
            metadata = json.load(response)
        if metadata['info']['version'] != version or not metadata['urls']:
            raise RuntimeError('Version does not exist on PyPI: ' + pin)
        records[pin] = {'requires_python': metadata['info']['requires_python'],
                        'files': [{'filename': row['filename'], 'size': row['size'], 'sha256': row['digests']['sha256']}
                                  for row in metadata['urls']]}
    atomic_json(DOWNLOADS / 'pypi_direct_pin_metadata.json', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Install frozen resolved archives after size audit')
    args = parser.parse_args()
    if RESULT.exists():
        archive = RESULT.parent / 'grasp_eval_v7_attempts'
        archive.mkdir(exist_ok=True)
        index = len(list(archive.glob('environment_*.json'))) + 1
        atomic_json(archive / ('environment_{:03d}.json'.format(index)), json.loads(RESULT.read_text(encoding='utf-8')))
    before = tree_hash(UPSTREAM)
    caller_packages = {item.metadata['Name']: item.version for item in importlib.metadata.distributions()}
    caller_packages_hash = hashlib.sha256(json.dumps(caller_packages, sort_keys=True).encode()).hexdigest()
    report = {'schema': 7, 'status': 'running', 'scope': 'CPU synthetic geometry compatibility only',
              'dataset_ap_computed': False, 'dataset_read': False, 'gpu_used': False,
              'environment': str(ENV.relative_to(ROOT)), 'source_clone': str(BUILD.relative_to(ROOT)),
              'source_commit': COMMIT, 'upstream_tree_sha256_before': before,
              'caller_environment': sys.prefix, 'caller_packages_sha256_before': caller_packages_hash,
              'budget_bytes': CAP, 'reserve_bytes': 60 * 2**30, 'failures': []}
    report['setup_script_sha256'] = digest(__file__)
    atomic_json(RESULT, report)
    try:
        check_budget()
        prepare()
        verify_pins()
        python = ENV / 'Scripts/python.exe'
        run([python, '-m', 'pip', 'install', '--no-cache-dir', *BOOTSTRAP], 'bootstrap')
        plan = DOWNLOADS / 'resolution.json'
        binding = {'requirements_sha256': digest(ROOT / 'configs/grasp_eval_v7_requirements.txt'),
                   'source_commit': COMMIT, 'patch_sha256': digest(PATCH), 'python': sys.version.split()[0]}
        binding_path = DOWNLOADS / 'resolution_binding.json'
        if plan.exists() and (not binding_path.exists() or json.loads(binding_path.read_text()) != binding):
            raise RuntimeError('Existing dependency resolution belongs to another frozen request')
        if not plan.exists():
            run([python, '-m', 'pip', 'install', '--dry-run', '--ignore-installed', '--no-build-isolation',
                 '--report', plan, '-r', ROOT / 'configs/grasp_eval_v7_requirements.txt', BUILD], 'resolve')
            atomic_json(binding_path, binding)
        resolved = json.loads(plan.read_text(encoding='utf-8'))
        packages = [{'name': row['metadata']['name'], 'version': row['metadata']['version'],
                     'download': row['download_info']} for row in resolved['install']]
        atomic_json(DOWNLOADS / 'resolved_dependency_lock.json', packages)
        archives = DOWNLOADS / 'archives'
        archives.mkdir(exist_ok=True)
        rows = []
        for package in packages:
            url = package['download']['url']
            if url.startswith('file:'):
                continue
            from urllib.parse import urlparse, unquote
            destination = archives / Path(unquote(urlparse(url).path)).name
            expected = package['download']['archive_info']['hashes']['sha256']
            if not destination.exists():
                with urllib.request.urlopen(url, timeout=60) as response:
                    length = int(response.headers.get('Content-Length', '0'))
                    check_budget(length)
                    temporary = destination.with_suffix(destination.suffix + '.part')
                    with temporary.open('wb') as stream:
                        for chunk in iter(lambda: response.read(1024 * 1024), b''):
                            stream.write(chunk)
                            check_budget()
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
            if digest(destination) != expected:
                raise RuntimeError('Archive hash mismatch: ' + destination.name)
            unpacked = sum(item.file_size for item in zipfile.ZipFile(destination).infolist()) if destination.suffix == '.whl' else destination.stat().st_size * 12
            rows.append({'name': package['name'], 'version': package['version'], 'path': str(destination.relative_to(ROOT)),
                         'sha256': expected, 'bytes': destination.stat().st_size, 'estimated_installed_bytes': unpacked})
        # Conservative double counting of bootstrap protects the cap; 128 MiB compiler/temp margin.
        estimate = sum(row['estimated_installed_bytes'] for row in rows) + 128 * 2**20
        installed = json.loads(subprocess.check_output([str(python), '-c',
            'import importlib.metadata,json;print(json.dumps({d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}))'], text=True))
        canonical = lambda name: re.sub(r'[-_.]+', '-', name).lower()
        installed_normalized = {canonical(name): version for name, version in installed.items()}
        already_installed = all(installed_normalized.get(canonical(package['name'])) == package['version'] for package in packages)
        check_budget(0 if already_installed else estimate)
        atomic_json(DOWNLOADS / 'archive_audit.json', {'archives': rows, 'conservative_additional_install_bytes': estimate})
        report.update({'resolved_packages': len(packages), 'resolution_sha256': digest(plan),
                       'dependency_lock_sha256': digest(DOWNLOADS / 'resolved_dependency_lock.json'),
                       'patch_sha256': digest(PATCH), 'size_audit_passed': True})
        if args.install:
            if not already_installed:
                requirements = [str(ROOT / row['path']) for row in rows]
                run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation', *requirements], 'install_dependencies')
                run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation', BUILD], 'install_api')
            report['runtime_clone_audit'] = audit_clone_source()
            report['installed_source_audit'] = audit_installed_source()
            report['installed_api_source_sha256'] = report['installed_source_audit']['installed_sha256']
            report['compiled_nms'] = [{'file': str(path.relative_to(ROOT)), 'sha256': digest(path)}
                                      for path in (ENV / 'Lib/site-packages').glob('grasp_nms*.pyd')]
            if len(report['compiled_nms']) != 1:
                raise RuntimeError('Expected one compiled CPU NMS extension')
            run([python, '-m', 'pip', 'check'], 'pip_check')
            run([python, '-B', ROOT / 'scripts/verify_grasp_geometry_v7.py'], 'synthetic_geometry')
            geometry_path = ROOT / 'results/P2/grasp_geometry_v7.json'
            geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
            if geometry['status'] != 'limited_engineering_ready_nondegenerate_paths_passed':
                raise RuntimeError('Unexpected synthetic geometry status')
            report['status'] = 'limited_engineering_ready_nondegenerate_paths_passed'
            report['geometry_report_sha256'] = digest(geometry_path)
            report['geometry_script_sha256'] = digest(ROOT / 'scripts/verify_grasp_geometry_v7.py')
            report['known_upstream_limitation'] = 'Axial sphere automatic contact search is a false negative; nondegenerate offset sphere passes; scoring algorithm unchanged'
            report['all_geometry_reliable'] = False
        else:
            report['status'] = 'dependencies_resolved_not_installed'
        report['remaining'] = ['Known axial-sphere upstream contact-search degeneracy remains unresolved',
                               'Dataset XML/model completeness and real-frame geometric scoring not yet verified',
                               'No dataset AP, independent baseline reproduction, calibration or test evaluation',
                               'Synthetic numerical success does not establish equivalence of all upstream evaluation paths']
    except Exception as error:
        report['status'] = 'blocked_or_failed'
        report['failures'].append(type(error).__name__ + ': ' + str(error))
        raise
    finally:
        report['footprint_bytes'] = footprint()
        report['upstream_tree_sha256_after'] = tree_hash(UPSTREAM)
        report['frozen_upstream_unchanged'] = report['upstream_tree_sha256_after'] == before
        packages_after = {item.metadata['Name']: item.version for item in importlib.metadata.distributions()}
        report['caller_packages_sha256_after'] = hashlib.sha256(json.dumps(packages_after, sort_keys=True).encode()).hexdigest()
        report['caller_environment_packages_unchanged'] = report['caller_packages_sha256_after'] == caller_packages_hash
        atomic_json(RESULT, report)


if __name__ == '__main__':
    main()
