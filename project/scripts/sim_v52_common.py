"""Isolated v52 bookkeeping. Never updates the historical pipeline state."""
import contextlib
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/simulation_v52'
LOG = ROOT / 'logs/simulation_v52'
STATE = ROOT / 'runtime/pipeline/simulation_v52.json'
CONFIG = ROOT / 'configs/simulation_v52.json'

def now():
    return dt.datetime.now().astimezone().isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, data, once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if once:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    else:
        temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        temporary.write_text(payload, encoding='utf-8')
        os.replace(temporary, path)

def append(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

def events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()] if Path(path).exists() else []

def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    stat = path.stat()
    return dict(sha256=digest.hexdigest(), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')

@contextlib.contextmanager
def lock(path):
    import msvcrt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

def status(phase, **values):
    write(STATE, dict(at=now(), version=52, phase=phase,
        all_tasks_percent_legacy=45, experiment_percent_legacy=56.25,
        progress_scope='Existing pre-simulation deliverable weights; new scope not yet reweighted',
        P5_original_scientific_passed=False, P6_original_scientific_passed=False,
        R4_authorized=False, automatic_expansion=False, recurring_automation=False, **values))

def prepare():
    with lock(ROOT/'runtime/pipeline/execution.lock'), lock(ROOT/'runtime/pipeline/simulation_v52.lock'):
        snapshot = LOG/'historical_snapshot.json'
        if snapshot.exists():
            verify_history()
            print(json.dumps({'already_prepared': True}))
            return
        paths = set()
        for pattern in ('src/*v51*.py', 'scripts/*v51*.py', 'configs/*v51*.json',
                        'runtime/pipeline/*v51*.json', 'src/*v50*.py', 'scripts/*v50*.py',
                        'configs/*v50*.json', 'runtime/pipeline/*v50*.json',
                        'docs/*v39*.txt', 'docs/*v50*.txt', 'docs/*v51*.txt'):
            paths.update(ROOT.glob(pattern))
        paths.add(ROOT/'runtime/pipeline/state.json')
        for name in ('results/joint_validation_v51', 'results/joint_validation_v51r1',
                     'results/joint_validation_v51r2', 'logs/joint_validation_v51',
                     'logs/joint_validation_v51r1', 'logs/joint_validation_v51r2'):
            paths.update(p for p in (ROOT/name).rglob('*') if p.is_file())
        write(snapshot, dict(at=now(), protected={relative(p): identity(p) for p in sorted(paths)},
            authorization='User R1/R2 and at most48 engineering attempts; stop for review before R4',
            historical_unknown_calls=2, active_research_processes_observed_before_prepare=0), once=True)
        status('R1_R2_preparing', engineering_attempts_started=0, engineering_attempts_cap=48)
        print(json.dumps({'prepared': True, 'protected': len(paths)}))

def verify_history():
    values = read(LOG/'historical_snapshot.json')['protected']
    changed = [name for name, value in values.items() if not (ROOT/name).is_file() or identity(ROOT/name) != value]
    if changed:
        raise RuntimeError('Historical artifacts changed: ' + str(changed[:20]))
    return len(values)

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='verify':
        print(json.dumps({'unchanged': verify_history()}))
    else:
        prepare()
