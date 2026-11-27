"""Bounded immutable storage for the fixed-support ranking diagnostic."""
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/grasp_soft_ranking_v13.json'
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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
    receipt = {'schema': 13, 'binding_fingerprint': binding_id(binding),
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    # Intent is durable first; a missing payload may be recomputed, never silently replaced.
    once(receipt_path(path), receipt, read_only)
    publish_once(path, payload, read_only)
    return receipt


def load_commit(path, binding):
    path = Path(path)
    receipt = read(receipt_path(path))
    require(receipt == {'schema': 13, 'binding_fingerprint': binding_id(binding),
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


def make_binding(root, config, protocol, input_group='feature_inputs'):
    require(protocol['schema'] == 13, 'Wrong protocol schema')
    require(runtime_identity() == protocol['runtime'], 'Frozen numerical runtime changed')
    expected_prefix = safe_path(root, protocol['interpreter']).parent.parent
    require(Path(sys.prefix).resolve() == expected_prefix.resolve(), 'Wrong isolated interpreter')
    files = dict(protocol[input_group])
    files.update(protocol['source_hashes'])
    for name, expected in files.items():
        require(digest(safe_path(root, name)) == expected, 'Frozen input/source changed: ' + name)
    value = {'schema': 13, 'input_group': input_group, 'protocol_sha256': digest(config),
             'files': files, 'runtime': runtime_identity(), 'interpreter_sha256': digest(sys.executable)}
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
    def __init__(self, root, protocol):
        self.root, self.protocol, self.started = Path(root), protocol, time.monotonic()
        self.peak = 0

    def check(self, extra_bytes=0):
        limits = self.protocol['resources']
        self.peak = max(self.peak, peak_memory_bytes())
        require(self.peak <= limits['rss_soft_bytes'], 'RSS budget exceeded')
        require(time.monotonic() - self.started <= limits['invocation_soft_seconds'], 'Invocation time budget exceeded')
        require(shutil.disk_usage(self.root).free - extra_bytes >= limits['disk_reserve_bytes'], 'Disk reserve exhausted')
        size = 0
        for name in self.protocol['budget_paths']:
            path = safe_path(self.root, name)
            size += sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else (path.stat().st_size if path.is_file() else 0)
        require(size + extra_bytes <= limits['output_limit_bytes'], 'Round output budget exceeded')
        return {'peak_rss_bytes': self.peak, 'elapsed_seconds': time.monotonic() - self.started,
                'current_output_bytes': size}
