"""Bounded immutable storage for the four-view baseline experiment."""
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import uuid
import zlib

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/grasp_multiview_v14.json'
sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def safe_path(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), 'Expected project-relative file')
    root, path = Path(root).resolve(), (Path(root) / name).resolve()
    require(path != root and path.is_relative_to(root), 'Path escaped project')
    return path


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic(path, value):
    atomic_bytes(path, json_bytes(value))


def publish_once(path, payload, read_only=False):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, 'Immutable bytes changed: ' + str(path))
    else:
        require(not read_only, 'Missing completed output: ' + str(path))
        atomic_bytes(path, payload)


def once(path, value, read_only=False):
    publish_once(path, json_bytes(value), read_only)


def receipt_path(path):
    return Path(path).with_name(Path(path).name + '.sha256.json')


def binding_id(binding):
    return binding['fingerprint'] if isinstance(binding, dict) else binding


def commit(path, value, binding, read_only=False):
    payload = json_bytes(value)
    receipt = {'schema': 14, 'binding_fingerprint': binding_id(binding),
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    # Intent is durable first; a missing payload may be recomputed, never silently replaced.
    once(receipt_path(path), receipt, read_only)
    publish_once(path, payload, read_only)
    return receipt


def load_commit(path, binding):
    path = Path(path)
    receipt = read(receipt_path(path))
    require(receipt == {'schema': 14, 'binding_fingerprint': binding_id(binding),
                        'sha256': digest(path), 'bytes': path.stat().st_size}, 'Commit receipt differs: ' + str(path))
    return read(path)


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.process.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def runtime_identity():
    return {'python': platform.python_version(), 'numpy': importlib.metadata.version('numpy'),
            'scipy': importlib.metadata.version('scipy'), 'pillow': importlib.metadata.version('Pillow')}


def make_binding(root, config, protocol, input_group='generation_inputs', frame=None, runtime_name='vision', extra_inputs=None):
    require(protocol['schema'] == 14, 'Wrong protocol schema')
    require(runtime_identity() == protocol['runtimes'][runtime_name], 'Frozen numerical runtime changed')
    interpreter = safe_path(root, protocol['interpreters'][runtime_name])
    require(Path(sys.prefix).resolve() == interpreter.parent.parent.resolve(), 'Wrong isolated interpreter')
    if input_group == 'generation_inputs':
        require(frame in protocol['frames'], 'Frame is outside the frozen selection')
        files = dict(protocol['generation_shared_inputs'])
        files.update(protocol[input_group][frame])
    else:
        files = dict(protocol[input_group])
    files.update(protocol['source_hashes'])
    for name, expected in files.items():
        require(digest(safe_path(root, name)) == expected, 'Frozen input/source changed: ' + name)
    for name, expected in (extra_inputs or {}).items():
        require(name not in files or files[name] == expected, 'Additional binding conflicts')
        require(digest(safe_path(root, name)) == expected, 'Dynamic sealed dependency changed: ' + name)
        files[name] = expected
    value = {'schema': 14, 'input_group': input_group, 'frame': frame, 'protocol_sha256': digest(config),
             'files': files, 'runtime_name': runtime_name, 'runtime': runtime_identity(),
             'interpreter_sha256': digest(sys.executable)}
    value['fingerprint'] = fingerprint(value)
    return value


def peak_memory_bytes():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb), 'Cannot inspect process memory')
        return int(value.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))


def snapshot(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns} for p in sorted(directory.rglob('*')) if p.is_file()}


class Guard:
    """Soft checks between bounded operations, not OS-enforced resource limits."""
    def __init__(self, root, protocol, phase='generation'):
        self.root, self.protocol, self.started = Path(root), protocol, time.monotonic()
        self.phase = phase
        self.peak = 0

    def check(self, extra_bytes=0):
        limits = self.protocol['resources'][self.phase]
        self.peak = max(self.peak, peak_memory_bytes())
        require(self.peak <= limits['rss_soft_bytes'], 'RSS budget exceeded')
        require(time.monotonic() - self.started <= limits['invocation_soft_seconds'], 'Invocation time budget exceeded')
        require(shutil.disk_usage(self.root).free - extra_bytes >= limits['disk_reserve_bytes'], 'Disk reserve exhausted')
        size = 0
        for name in self.protocol['budget_paths']:
            path = safe_path(self.root, name)
            size += sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else (path.stat().st_size if path.is_file() else 0)
            if not path.is_dir() and path.parent.exists():
                size += sum(p.stat().st_size for p in path.parent.glob(path.name + '.*.tmp')
                            if p.is_file() and re.fullmatch(re.escape(path.name) + r'\.[0-9a-f]{32}\.tmp', p.name))
        require(size + extra_bytes <= limits['output_limit_bytes'], 'Round output budget exceeded')
        return {'peak_rss_bytes': self.peak, 'elapsed_seconds': time.monotonic() - self.started,
                'current_output_bytes': size}



def commit_bytes(path, payload, binding, read_only=False):
    receipt = {'schema': 14, 'binding_fingerprint': binding_id(binding),
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    once(receipt_path(path), receipt, read_only)
    publish_once(path, payload, read_only)
    return receipt


def verify_bytes(path, binding):
    path = Path(path)
    receipt = read(receipt_path(path))
    require(receipt == {'schema': 14, 'binding_fingerprint': binding_id(binding),
                        'sha256': digest(path), 'bytes': path.stat().st_size}, 'Binary commit receipt differs')
    return receipt


def frame_directory(root, protocol, frame, phase):
    require(frame in protocol['frames'] and phase in ('generation', 'geometry'), 'Out-of-scope frame/phase')
    return safe_path(root, protocol['outputs']['frames_root']) / frame / phase


def verify_assets(root, config, protocol):
    """Verify a completed asset seal without decoding poses or loading models."""
    directory = safe_path(root, protocol['outputs']['assets_dir'])
    binding = read(directory / 'binding.json')
    expected = {'schema': 14, 'input_group': 'asset_inputs', 'frame': None,
                'protocol_sha256': digest(config), 'files': {**protocol['asset_inputs'], **protocol['source_hashes']},
                'runtime_name': 'vision', 'runtime': protocol['runtimes']['vision'],
                'interpreter_sha256': digest(safe_path(root, protocol['interpreters']['vision']))}
    expected['fingerprint'] = fingerprint(expected)
    require(binding == expected, 'Asset binding differs')
    for name, sha in expected['files'].items():
        require(digest(safe_path(root, name)) == sha, 'Asset input/source changed: ' + name)
    report = load_commit(directory / 'report.json', binding)
    completion = load_commit(directory / 'completion.json', binding)
    require(report['schema'] == 14 and report['status'] == 'verified_assets' and
            list(report['frames']) == protocol['frames'], 'Asset report scope differs')
    expected_outputs = {'report.json': digest(directory / 'report.json')}
    files = {'binding.json', 'report.json', 'report.json.sha256.json', 'completion.json', 'completion.json.sha256.json'}
    for frame in protocol['frames']:
        path = safe_path(root, protocol['frame_inputs'][frame]['annotation'])
        require(path.is_relative_to(directory), 'Annotation escaped asset directory')
        receipt = verify_bytes(path, binding)
        entry = protocol['asset_source']['entries'][frame]
        require(report['frames'][frame] == {'path': path.relative_to(root).as_posix(),
                'sha256': receipt['sha256'], 'bytes': entry['bytes'], 'crc32': entry['crc32']}, 'Annotation manifest differs')
        require(path.stat().st_size == entry['bytes'], 'Annotation size differs')
        require(zlib.crc32(path.read_bytes()) & 0xffffffff == entry['crc32'], 'Frozen annotation CRC differs')
        name = path.relative_to(directory).as_posix()
        expected_outputs[name] = receipt['sha256']
        files.update((name, name + '.sha256.json'))
    source = protocol['asset_source']
    require(report == {'schema': 14, 'status': 'verified_assets', 'frames': report['frames'],
        'source_revision': source['revision'], 'archive_bytes': source['archive_bytes'], 'archive_etag': source['etag'],
        'new_models_downloaded': 0, 'pose_values_decoded': False, 'official_whole_archive_checksum_verified': False},
        'Asset source or scope declaration differs')
    require(completion == {'schema': 14, 'status': 'completed', 'outputs': expected_outputs}, 'Asset completion differs')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual == files, 'Asset output tree differs')
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}
