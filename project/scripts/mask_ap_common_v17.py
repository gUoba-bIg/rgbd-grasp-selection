"""Bounded immutable storage for category-agnostic visible-mask evaluation."""
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from scripts import multiview_common_v14r1 as storage
from scripts.multiview_common_v14r1 import require,read,digest,fingerprint,json_bytes,receipt_path,lock,atomic,atomic_bytes,publish_once,once
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='4'


def safe(root, name):
    root=Path(root).resolve()
    candidate=Path(name)
    candidate=candidate.resolve() if candidate.is_absolute() else (root/candidate).resolve()
    require(candidate.is_relative_to(root) and candidate != root,'Path is outside the experiment project')
    return candidate


path=safe


def commit_bytes(file, payload, binding, read_only=False):
    import hashlib
    record={'schema':17,'binding_fingerprint':storage.binding_id(binding),
            'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
    once(receipt_path(file),record,read_only)
    publish_once(file,payload,read_only)
    return record


def verify_bytes(file, binding):
    file=Path(file)
    expected={'schema':17,'binding_fingerprint':storage.binding_id(binding),
              'sha256':digest(file),'bytes':file.stat().st_size}
    require(read(receipt_path(file)) == expected,'Committed payload or source binding differs: '+str(file))
    return expected


def load_bytes(file, binding):
    verify_bytes(file,binding)
    return Path(file).read_bytes()


def commit(file, value, binding, read_only=False):
    return commit_bytes(file,json_bytes(value),binding,read_only)


commit_json=commit


def load_commit(file, binding):
    verify_bytes(file,binding)
    return read(file)


def snapshot(directory, root=None):
    directory=Path(directory).resolve()
    root=Path(root).resolve() if root is not None else directory
    result={}
    for file in sorted(directory.rglob('*')):
        require(not file.is_symlink(),'Link in experiment inventory')
        if file.is_file():
            require(file.is_relative_to(root),'Snapshot path escaped the declared root')
            result[file.relative_to(root).as_posix()]={'sha256':digest(file),'bytes':file.stat().st_size,
                                                     'mtime_ns':file.stat().st_mtime_ns}
    return result


def verify_source_maps(root, protocol):
    require(bool(protocol['source_hashes']) and bool(protocol['input_hashes']),'Protocol inputs and source must be frozen')
    for group in ('source_hashes','input_hashes'):
        for name,checksum in protocol[group].items():
            require(isinstance(checksum,str) and len(checksum)==64 and digest(safe(root,name))==checksum,
                    'Frozen source/input changed: '+name)


class Guard:
    def __init__(self,root,protocol,phase='scores'):
        require(phase in ('scores','metrics','acceptance'),'Unknown mask-evaluation phase')
        self.root=Path(root).resolve()
        self.protocol=protocol
        self.phase=phase
        self.started=time.perf_counter()

    def check(self,extra_bytes=0):
        require(type(extra_bytes) is int and extra_bytes>=0,'Invalid publication byte count')
        limits=self.protocol['resources']
        total=0
        for name in self.protocol['budget_paths']:
            target=safe(self.root,name)
            if target.is_file():
                total+=target.stat().st_size
            elif target.exists():
                total+=sum(file.stat().st_size for file in target.rglob('*') if file.is_file())
        elapsed=time.perf_counter()-self.started
        rss=storage.peak_memory_bytes()
        allocated=reserved=0
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            allocated=int(torch.cuda.max_memory_allocated())
            reserved=int(torch.cuda.max_memory_reserved())
        require(shutil.disk_usage(self.root).free-extra_bytes>=limits['disk_reserve_bytes'],'Disk reserve exhausted')
        require(total+extra_bytes<=limits['output_limit_bytes'],'Mask-evaluation output budget exhausted')
        require(elapsed<=limits['invocation_soft_seconds'],'Mask-evaluation invocation exceeded its time budget')
        require(rss<=limits['rss_soft_bytes'],'Mask-evaluation RSS budget exhausted')
        require(allocated<=limits['gpu_allocated_soft_bytes'] and reserved<=limits['gpu_reserved_soft_bytes'],
                'Mask-evaluation GPU allocator budget exhausted')
        return {'elapsed_seconds':elapsed,'peak_rss_bytes':rss,'peak_gpu_allocated_bytes':allocated,
                'peak_gpu_reserved_bytes':reserved,'current_output_bytes':total}
