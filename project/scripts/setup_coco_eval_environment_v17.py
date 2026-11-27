"""Build and seal a separate wheel-only COCO evaluator; synthetic data only."""
import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = 'pycocotools-2.0.11-cp312-abi3-win_amd64.whl'
WHEEL_URL = 'https://files.pythonhosted.org/packages/83/b4/f6708404ff494706b80e714b919f76dc4ec9845a4007affd6d6b0843f928/' + WHEEL_NAME
WHEEL_SHA256 = 'ffe806ce535f5996445188f9a35643791dc54beabc61bd81e2b03367356d604f'
VERSIONS = {'numpy': '1.26.4', 'pillow': '11.3.0', 'pycocotools': '2.0.11'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe(root, value):
    root = Path(root).resolve()
    path = (root/Path(value)).resolve()
    require(path != root and path.is_relative_to(root), 'Path escaped project')
    return path


def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def once(path, value, readonly=False):
    payload = encoded(value)
    if Path(path).exists():
        require(Path(path).read_bytes() == payload, 'Immutable environment record differs: '+str(path))
    else:
        require(not readonly, 'Missing sealed environment record')
        atomic(path, payload)


def commit(path, value, binding, readonly=False):
    payload = encoded(value)
    once(str(path)+'.sha256.json', {'schema': 17, 'binding_fingerprint': binding['fingerprint'],
        'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}, readonly)
    once(path, value, readonly)


def load_commit(path, binding):
    path = Path(path)
    require(read(str(path)+'.sha256.json') == {'schema': 17, 'binding_fingerprint': binding['fingerprint'],
        'sha256': digest(path), 'bytes': path.stat().st_size}, 'Environment receipt changed')
    return read(path)


def inventory(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'Link in isolated environment')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return result


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def process_rss(pid):
    if os.name != 'nt':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x410, False, pid)
    require(bool(handle), 'Cannot inspect process memory')
    try:
        value = Counters(); value.cb = ctypes.sizeof(value)
        require(bool(psapi.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb)), 'Cannot read process memory')
        return int(value.PeakWorkingSetSize)
    finally:
        kernel.CloseHandle(handle)


class Guard:
    def __init__(self, root, protocol):
        self.root, self.protocol, self.started, self.peak = Path(root), protocol, time.monotonic(), 0

    def check(self, extra=0, pid=None):
        limits = self.protocol['resources']
        total = 0
        for name in self.protocol['budget_paths']:
            path = safe(self.root, name)
            total += sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else (path.stat().st_size if path.is_file() else 0)
        require(total+extra <= limits['output_limit_bytes'], 'Environment output budget exceeded')
        require(shutil.disk_usage(self.root).free-extra >= limits['disk_reserve_bytes'], 'Disk reserve exhausted')
        self.peak = max(self.peak, process_rss(os.getpid()), process_rss(pid) if pid else 0)
        elapsed = time.monotonic()-self.started
        require(self.peak <= limits['rss_soft_bytes'] and elapsed <= limits['invocation_soft_seconds'], 'Environment resource limit exceeded')
        return {'elapsed_seconds': elapsed, 'peak_rss_bytes': self.peak, 'current_output_bytes': total}


def validate_protocol(root, protocol):
    require(protocol['schema'] == 17 and protocol['env_dir'] == 'runtime/envs/coco_eval_v17', 'Wrong isolated environment target')
    require({row['name'].lower(): row['version'] for row in protocol['wheels']} == VERSIONS, 'Wheel package/version selection changed')
    require(len(protocol['wheels']) == 3, 'Unexpected additional wheels')
    for row in protocol['wheels']:
        path = safe(root, row['path'])
        require(path.suffix == '.whl', 'Only binary wheels are allowed')
        if row['name'].lower() == 'pycocotools':
            require(path.name == WHEEL_NAME and row['sha256'] == WHEEL_SHA256 and row['url'] == WHEEL_URL,
                    'COCO wheel source changed')
            require(path.is_relative_to(safe(root, 'runtime/downloads/coco_eval_v17')), 'Download escaped new wheel directory')
        else:
            require(path.is_file() and digest(path) == row['sha256'], 'Reusable wheel changed')
    expected = {k.lower(): v for k, v in protocol['expected_packages'].items()}
    require(set(expected) == set(VERSIONS)|{'pip'} and all(expected[k] == v for k, v in VERSIONS.items()), 'Unpinned dependency inventory')
    require(protocol['outputs'] == {'report': 'results/P4/mask_ap_environment_v17.json',
        'work_dir': 'runtime/coco_eval_setup_v17', 'logs_dir': 'logs/P4/mask_ap_v17/environment'}, 'Unsafe environment output directory')


def binding_for(root, config, protocol):
    validate_protocol(root, protocol)
    base = Path(protocol['base_python']['path']).resolve()
    require(base.is_file() and digest(base) == protocol['base_python']['sha256'], 'Base interpreter changed')
    require(protocol['source_hashes'].get('scripts/setup_coco_eval_environment_v17.py') == digest(Path(__file__)), 'Environment setup source not frozen')
    for name, sha in protocol['source_hashes'].items():
        require(digest(safe(root, name)) == sha, 'Frozen environment source changed')
    value = {'schema': 17, 'phase': 'coco_environment', 'protocol_sha256': digest(config),
        'base_python': protocol['base_python'], 'source_hashes': protocol['source_hashes'], 'wheels': protocol['wheels']}
    value['fingerprint'] = fingerprint(value)
    return value


def wheel_members_match(archive, site):
    """Audit importable package payload bytes directly against the frozen wheel."""
    with zipfile.ZipFile(archive) as wheel:
        checked = 0
        for name in wheel.namelist():
            if name.endswith('/') or '.dist-info/' in name or '.data/' in name:
                continue
            path = (Path(site)/name).resolve()
            require(path.is_relative_to(Path(site).resolve()), 'Wheel member escaped site-packages')
            require(path.is_file() and path.read_bytes() == wheel.read(name), 'Installed wheel payload differs: '+name)
            checked += 1
    require(checked > 0, 'Wheel has no checked package payload')
    return checked


def child(root, protocol, command, name, guard):
    logs = safe(root, protocol['outputs']['logs_dir']); logs.mkdir(parents=True, exist_ok=True)
    log = logs/(name+'_'+uuid.uuid4().hex+'.log')
    temp = safe(root, protocol['outputs']['work_dir'])/'temp'; temp.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1',
        'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PIP_NO_CACHE_DIR': '1', 'PIP_CONFIG_FILE': os.devnull,
        'CUDA_VISIBLE_DEVICES': '', 'TEMP': str(temp), 'TMP': str(temp), 'OMP_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4'})
    environment.pop('PYTHONPATH', None)
    guard.check(1024**2)
    with log.open('x', encoding='utf-8') as stream:
        stream.write('COMMAND '+json.dumps([str(x) for x in command], ensure_ascii=False)+'\n'); stream.flush()
        process = subprocess.Popen([str(x) for x in command], cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    guard.check(pid=process.pid)
                except RuntimeError as error:
                    if process.poll() is None or not str(error).startswith('Cannot '):
                        raise
                    guard.check()
                time.sleep(.2)
        except BaseException:
            process.kill(); process.wait(timeout=15)
            raise
        stream.flush(); os.fsync(stream.fileno())
    require(process.returncode == 0, 'Environment child failed; retained log '+str(log))
    guard.check()
    return log


def fetch_wheel(root, row, guard):
    path = safe(root, row['path'])
    if path.exists():
        require(digest(path) == row['sha256'], 'Existing wheel differs')
        return False
    require(row['url'] == WHEEL_URL, 'Only the frozen COCO wheel may be fetched')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+'.part')
    guard.check(1024**2)
    with urllib.request.urlopen(row['url'], timeout=30) as response, partial.open('wb') as stream:
        require(response.geturl() == row['url'], 'Unexpected download redirect')
        count = 0
        for data in iter(lambda: response.read(65536), b''):
            count += len(data)
            require(count <= 1024**2, 'Frozen wheel download exceeded one MiB')
            guard.check(len(data)); stream.write(data)
        stream.flush(); os.fsync(stream.fileno())
    require(digest(partial) == row['sha256'], 'Downloaded wheel digest differs')
    os.replace(partial, path)
    return True


def empty_results(coco):
    from pycocotools.coco import COCO
    import copy
    result = COCO()
    result.dataset = {'info': copy.deepcopy(coco.dataset.get('info', {})),
        'images': copy.deepcopy(coco.dataset['images']), 'categories': copy.deepcopy(coco.dataset['categories']), 'annotations': []}
    result.createIndex()
    return result


def synthetic_probe():
    import numpy as np
    import PIL
    from pycocotools import mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools._mask as compiled_mask
    site = Path(sys.prefix)/'Lib/site-packages'
    require(all(Path(module.__file__).resolve().is_relative_to(site.resolve()) for module in (np, PIL, compiled_mask)),
            'Synthetic probe imported a package outside its isolated environment')
    shape = np.zeros((5, 7), np.uint8); shape[1:4, 2:5] = 1
    rle = mask.encode(np.asfortranarray(shape))
    require(np.array_equal(mask.decode(rle), shape) and int(mask.area(rle)) == 9, 'Non-square RLE round-trip failed')
    other = np.zeros_like(shape); other[0, 0] = 1
    wrong = mask.encode(np.asfortranarray(other))
    def metric(detections, with_gt=True):
        coco = COCO(); coco.dataset = {'info': {}, 'images': [{'id': 1, 'height': 5, 'width': 7}],
            'categories': [{'id': 1, 'name': 'object'}], 'annotations': []}
        if with_gt:
            coco.dataset['annotations'] = [{'id': 1, 'image_id': 1, 'category_id': 1, 'segmentation': rle,
                'area': 9, 'bbox': [2, 1, 3, 3], 'iscrowd': 0}]
        coco.createIndex()
        dt = coco.loadRes(detections) if detections else empty_results(coco)
        evaluator = COCOeval(coco, dt, 'segm'); evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        return float(evaluator.stats[0])
    good = {'image_id': 1, 'category_id': 1, 'segmentation': rle, 'score': .5}
    bad = {'image_id': 1, 'category_id': 1, 'segmentation': wrong, 'score': .5}
    with contextlib.redirect_stdout(io.StringIO()):
        values = {'perfect': metric([dict(good)]), 'empty_predictions': metric([]),
            'false_before_true_tie': metric([dict(bad), dict(good)]),
            'true_before_false_tie': metric([dict(good), dict(bad)]), 'no_ground_truth': metric([], False)}
    expected = {'perfect': 1., 'empty_predictions': 0., 'false_before_true_tie': .5, 'true_before_false_tie': 1., 'no_ground_truth': -1.}
    require(all(abs(values[k]-v) <= 1e-12 for k, v in expected.items()), 'Synthetic official AP differs')
    packages = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
    return {'schema': 17, 'status': 'passed', 'synthetic_only': True, 'research_frames_read': 0,
        'gpu_used': False, 'new_model_forward_calls': 0, 'rle_non_square_round_trip': True,
        'official_synthetic_ap': values, 'packages': packages, 'runtime': {'python': platform.python_version(),
            'implementation': platform.python_implementation(), 'architecture': platform.machine(),
            'numpy': np.__version__, 'Pillow': PIL.__version__, 'pycocotools': importlib.metadata.version('pycocotools')}}


def _run(root, config, check_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    protocol = read(config); binding = binding_for(root, config, protocol)
    guard = Guard(root, protocol); guard.check()
    directory = safe(root, protocol['env_dir']); work = safe(root, protocol['outputs']['work_dir'])
    report_path = safe(root, protocol['outputs']['report'])
    if check_only or report_path.exists():
        report = load_commit(report_path, binding)
        require(report['binding'] == binding and report['status'] == 'passed' and report['environment_inventory'] == inventory(directory),
                'Completed environment changed')
        for row in protocol['wheels']:
            require(digest(safe(root, row['path'])) == row['sha256'], 'Wheel changed after environment seal')
        return {'status': 'verified_completed_read_only', 'report_sha256': digest(report_path), 'resources': guard.check()}
    require(not directory.exists() or not any(directory.iterdir()) or (work/'binding.json').exists(), 'Unbound partial environment')
    work.mkdir(parents=True, exist_ok=True)
    guard.check(len(encoded(binding))*2+1024); once(work/'binding.json', binding)
    new_downloads = sum(int(fetch_wheel(root, row, guard)) for row in protocol['wheels'] if row['name'].lower() == 'pycocotools')
    # Only this new directory is touched; a bound partial construction can resume.
    if not (work/'created.json').exists():
        child(root, protocol, [protocol['base_python']['path'], '-B', '-m', 'venv', str(directory)], 'create', guard)
        commit(work/'created.json', {'schema': 17, 'env_dir': protocol['env_dir'], 'binding_fingerprint': binding['fingerprint']}, binding)
    else:
        load_commit(work/'created.json', binding)
    require('include-system-site-packages = false' in (directory/'pyvenv.cfg').read_text(), 'Environment inherits external packages')
    python = directory/'Scripts/python.exe'
    if not (work/'installed.json').exists():
        command = [python, '-B', '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', '--no-cache-dir', '--force-reinstall']
        command += [safe(root, row['path']) for row in protocol['wheels']]
        child(root, protocol, command, 'install', guard)
        checked = {row['name']: wheel_members_match(safe(root, row['path']), directory/'Lib/site-packages') for row in protocol['wheels']}
        commit(work/'installed.json', {'schema': 17, 'checked_payload_files': checked, 'binding_fingerprint': binding['fingerprint']}, binding)
    else:
        load_commit(work/'installed.json', binding)
    for row in protocol['wheels']:
        wheel_members_match(safe(root, row['path']), directory/'Lib/site-packages')
    if not (work/'probe.json').exists():
        log = child(root, protocol, [python, '-B', Path(__file__).resolve(), '--internal-probe', '--env-dir', directory], 'synthetic_probe', guard)
        values = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('{')]
        require(bool(values), 'Missing synthetic probe output')
        probe = values[-1]
        require(probe['packages'] == {k.lower(): v for k, v in protocol['expected_packages'].items()}, 'Unexpected installed packages')
        commit(work/'probe.json', probe, binding)
    probe = load_commit(work/'probe.json', binding)
    require(probe['status'] == 'passed' and probe['synthetic_only'] is True
        and probe['packages'] == {k.lower(): v for k, v in protocol['expected_packages'].items()}
        and probe['runtime']['python'] == protocol['python_version'], 'Saved synthetic environment evidence differs')
    report = {'schema': 17, 'status': 'passed', 'binding': binding, 'environment': protocol['env_dir'],
        'probe': probe, 'environment_inventory': inventory(directory), 'package_inventory': probe['packages'],
        'wheels': protocol['wheels'], 'research_frames_read': 0, 'new_model_forward_calls': 0, 'gpu_used': False,
        'old_environments_modified': False, 'source_compilation_performed': False}
    guard.check(len(encoded(report))*2+2048); commit(report_path, report, binding)
    return {'status': 'passed', 'report_sha256': digest(report_path), 'new_downloads': new_downloads, 'resources': guard.check()}


def run(root, config, check_only=False):
    root = Path(root).resolve()
    protocol = read(config)
    if check_only or safe(root, protocol['outputs']['report']).exists():
        return _run(root, config, check_only)
    with exclusive(root/'runtime/locks/coco_eval_v17.lock'):
        return _run(root, config, check_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/coco_eval_environment_v17.json')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--internal-probe', action='store_true')
    parser.add_argument('--env-dir', type=Path)
    args = parser.parse_args()
    if args.internal_probe:
        require(args.env_dir is not None and Path(sys.prefix).resolve() == args.env_dir.resolve(), 'Probe requires the isolated target interpreter')
        value = synthetic_probe()
    else:
        value = run(ROOT, args.config, args.check_only)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
