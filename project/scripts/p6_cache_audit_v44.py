"""Authorized finite fit/calibration-cache audit; no image replay or fitting."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
import numpy as np
from src import p6_cache_audit_v44 as core

LOG = 'logs/P6/cache_audit_v44'
OUT = 'results/P6/cache_audit_v44'
CONFIG = 'configs/p6_cache_audit_v44.json'
INPUTS = dict(generated_array='results/P6/mixture_v43/generated.npz', generated_metadata='results/P6/mixture_v43/generated.json',
              model='results/P6/mixture_v43/model.json', posterior='results/P6/mixture_v43/posterior.npz',
              aggregate='results/P6/mixture_v43/result.json')
SOURCES = ('src/p6_cache_audit_v44.py', 'scripts/p6_cache_audit_v44.py', 'tests/test_p6_cache_audit_v44.py',
           'src/uncertainty_mixture_v43.py', 'scripts/p6_mixture_v43.py')
BLOCKS = ('training', 'calibration', 'assessment')
need = core.need


def dependency_paths():
    """Path inventory only; root can freeze without running any statistics."""
    return [*INPUTS.values(), *SOURCES, 'configs/p6_mixture_v43.json', 'logs/P6/mixture_v43/freeze.json']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def identity(path):
    p = Path(path); s = p.stat()
    need(not p.is_symlink() and p.is_file(), 'Expected ordinary bound file')
    return dict(sha256=sha(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def safe(root, name):
    path = (Path(root)/name).resolve()
    need(path.is_relative_to(Path(root).resolve()) and path != Path(root).resolve(), 'Path escapes project')
    return path


def once(path, value):
    p = Path(path); payload = encoded(value)
    if p.exists():
        need(p.read_bytes() == payload, 'Immutable record differs: '+str(p)); return
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, p)


def receipt(path):
    return Path(str(path)+'.sha256.json')


def commit(path, value, binding, guard):
    payload = encoded(value); guard.check(0 if Path(path).exists() else len(payload)+2048)
    once(receipt(path), dict(schema=44, binding=binding, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
    once(path, value)
    return value


def load_commit(path, binding):
    p = Path(path)
    need(read(receipt(p)) == dict(schema=44, binding=binding, sha256=sha(p), bytes=p.stat().st_size), 'Receipt differs: '+str(p))
    return read(p)


def rss_bytes():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)]+[(name, ctypes.c_size_t) for name in
            ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True); psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    data = Counters(); data.cb = ctypes.sizeof(data)
    need(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb), 'Cannot read RSS')
    return int(data.PeakWorkingSetSize)


class Guard:
    def __init__(self, root, phase):
        self.root = Path(root); self.phase = phase; self.start = time.monotonic()
    def check(self, extra=0):
        elapsed = time.monotonic()-self.start; peak = rss_bytes()
        size = sum(p.stat().st_size for folder in (self.root/LOG, self.root/OUT) if folder.exists() for p in folder.rglob('*') if p.is_file())
        need(elapsed <= dict(prepare=120, run=300, review=300, cold=60)[self.phase], 'Phase time budget exceeded')
        need(peak <= 2*1024**3 and size+extra <= 64*1024**2, 'Memory/output budget exceeded')
        need(shutil.disk_usage(self.root).free-extra >= 60*1024**3, 'Disk reserve exhausted')
        return dict(elapsed_seconds=elapsed, peak_rss_bytes=peak, output_bytes=size)


def authorization(root):
    root = Path(root); auth_file = root/LOG/'authorization.json'; auth = read(auth_file)
    need(auth['authorized'] is True and auth['cache_analysis_only'] is True
         and auth['new_model_calls'] == auth['new_AP_calls'] == auth['new_fit_calls'] == 0
         and auth['cumulative_seconds'] == 600, 'Missing bounded cache-only authorization')
    if 'protocol' in auth: need(auth['protocol'] == core.PROTOCOL, 'Predeclared audit rules differ')
    need(set(dependency_paths()).issubset(auth['protected']), 'Authorization omits required dependencies')
    for name, wanted in auth['protected'].items():
        need(identity(safe(root, name)) == wanted, 'Frozen dependency changed: '+name)
    root_config = root/'configs/research_cache_v44.json'
    frozen = read(root/'logs/research_v44/freeze.json')
    need(frozen['config'] == identity(root_config), 'Root protocol is not frozen')
    root_protocol = read(root_config)
    relative_auth = LOG+'/authorization.json'
    need(root_protocol['protected'][relative_auth] == identity(auth_file), 'Authorization not bound by root freeze')
    return auth, identity(auth_file)


def prepare(root=ROOT, config=CONFIG):
    root = Path(root).resolve(); guard = Guard(root, 'prepare'); guard.check()
    auth, auth_id = authorization(root)
    body = dict(schema=44, protocol=core.PROTOCOL, authorization=auth_id,
                inputs={key: dict(path=name, **identity(root/name)) for key, name in INPUTS.items()},
                protected=auth['protected'], packages=dict(numpy=np.__version__, scipy=__import__('scipy').__version__),
                runtime_python=sys.version.split()[0], interpreter_sha256=sha(sys.executable),
                new_parameter_fits=0, new_neural_forward=0, new_official_AP=0, raw_replay=0)
    config_file = safe(root, config); once(config_file, body)
    binding = sha(config_file)
    result = dict(schema=44, passed=True, config=str(config_file.relative_to(root)).replace('\\', '/'),
                  config_sha256=binding, authorization_sha256=auth_id['sha256'], numerical_statistics_performed=False)
    commit(root/LOG/'prepare.json', result, binding, guard)
    return dict(status='prepared', resources=guard.check(), **result)


def verified_config(root, config):
    auth, auth_id = authorization(root)
    path = safe(root, config); c = read(path); binding = sha(path)
    need(c['schema'] == 44 and c['protocol'] == core.PROTOCOL and c['authorization'] == auth_id
         and c['protected'] == auth['protected'] and c['interpreter_sha256'] == sha(sys.executable)
         and c['packages'] == dict(numpy=np.__version__, scipy=__import__('scipy').__version__)
         and c['runtime_python'] == sys.version.split()[0], 'Prepared audit identity differs')
    need(set(c['inputs']) == set(INPUTS), 'Input inventory differs')
    for key, row in c['inputs'].items():
        need(row['path'] == INPUTS[key] and identity(root/row['path']) == {k:row[k] for k in ('sha256', 'bytes', 'mtime_ns')}, 'Prepared input changed')
    prepared = load_commit(root/LOG/'prepare.json', binding)
    need(prepared['config_sha256'] == binding and prepared['numerical_statistics_performed'] is False, 'Preparation seal differs')
    return c, binding


def load_allowed_cache(root, c):
    metadata = read(root/c['inputs']['generated_metadata']['path'])
    need(metadata['array_sha256'] == c['inputs']['generated_array']['sha256'], 'Generated array metadata differs')
    rows = metadata['rows']; need(len(rows) == 2160 and len({r['key'] for r in rows}) == 2160, 'Wrong frozen case inventory')
    fit = [r for r in rows if r['split'] == 'fit']; cal_indices = np.array([i for i,r in enumerate(rows) if r['split'] == 'calibration'], np.int64)
    need(len(fit) == 1296 and len(cal_indices) == 432, 'Wrong fit/calibration support')
    with np.load(root/c['inputs']['generated_array']['path'], allow_pickle=False) as archive:
        # The compressed container contains evaluation bytes. No evaluation
        # feature/label subset is passed to diagnostic or selection functions.
        x, y, w = archive['depth_X'], archive['depth_y'], archive['depth_weights']
        cal_targets = archive['y'][cal_indices]
        fit_indices = np.array([i for i,r in enumerate(rows) if r['split'] == 'fit'], np.int64)
        support = np.unique(archive['y'][fit_indices])
    with np.load(root/c['inputs']['posterior']['path'], allow_pickle=False) as archive:
        need(set(archive.files) == {'probability'}, 'Unexpected posterior archive')
        p = archive['probability'][cal_indices]
    cal = [dict(rows[int(i)], target_index=int(target)) for i,target in zip(cal_indices, cal_targets)]
    full = read(root/c['inputs']['aggregate']['path'])
    selected = {r['key']: r for r in full['rows'] if r['split'] == 'calibration'}
    need(len(selected) == 432, 'Duplicate or missing calibration aggregates')
    cal_results = [selected[r['key']] for r in cal]
    model = read(root/c['inputs']['model']['path'])
    need(model['generated_sha256'] == c['inputs']['generated_array']['sha256'] and full['model_sha256'] == c['inputs']['model']['sha256'], 'Frozen model/result binding differs')
    del full, selected
    return dict(depth_X=x, depth_y=y, weights=w, fit_case_rows=fit, depth_model=model['depth'],
                probability=p, cal_rows=cal, result_cal_rows=cal_results, prior=model['empirical_prior'], supported_indices=support)


def block_value(name, data, completed):
    if name == 'training':
        return core.analyze_training(**{k:data[k] for k in ('depth_X', 'depth_y', 'weights', 'fit_case_rows', 'depth_model')})
    if name == 'calibration':
        return core.analyze_calibration(**{k:data[k] for k in ('probability', 'cal_rows', 'result_cal_rows', 'prior', 'supported_indices')})
    return core.mechanism_assessment(completed['training'], completed['calibration'])


def compute_committed(root, phase, name, output, function, binding, guard):
    """Unknown numerical calls cannot be silently retried, unlike verified data."""
    prefix = root/LOG/'calls'/(phase+'_'+name)
    intent_path, returned_path = Path(str(prefix)+'.intent.json'), Path(str(prefix)+'.returned.json')
    intent = dict(schema=44, phase=phase, block=name, binding=binding,
                  output=output.relative_to(root).as_posix(), numerical_cache_analysis=True,
                  neural_forward=False, parameter_fit=False, raw_replay=False)
    if intent_path.exists():
        need(load_commit(intent_path, binding) == intent, 'Analysis intent changed')
        if output.exists():
            value = load_commit(output, binding)
            returned = dict(schema=44, phase=phase, block=name, binding=binding,
                            output_sha256=sha(output), output_bytes=output.stat().st_size)
            need(returned_path.exists(), 'Unknown numerical return acknowledgement requires explicit audit: '+phase+'/'+name)
            need(load_commit(returned_path, binding) == returned, 'Analysis return changed')
            return value, False
        need(False, 'Unknown numerical analysis call requires explicit audit before retry: '+phase+'/'+name)
    need(not output.exists() and not returned_path.exists(), 'Unbound numerical artifact')
    commit(intent_path, intent, binding, guard)
    value = function(); guard.check()
    commit(output, value, binding, guard)
    commit(returned_path, dict(schema=44, phase=phase, block=name, binding=binding,
                              output_sha256=sha(output), output_bytes=output.stat().st_size), binding, guard)
    return value, True


def verify_calls(root, binding, phases):
    calls=[]
    allowed={phase+'_'+name+'.'+event+'.json'+suffix for phase in ('run','review') for name in BLOCKS
             for event in ('intent','returned') for suffix in ('','.sha256.json')}
    directory=root/LOG/'calls'
    need(directory.is_dir() and all(p.is_file() and not p.is_symlink() and p.name in allowed for p in directory.iterdir()),
         'Unexpected numerical analysis ledger artifact')
    for phase in phases:
        for name in BLOCKS:
            prefix=root/LOG/'calls'/(phase+'_'+name)
            intent=load_commit(Path(str(prefix)+'.intent.json'),binding)
            returned=load_commit(Path(str(prefix)+'.returned.json'),binding)
            output=root/OUT/'blocks'/(name+'.json') if phase=='run' else root/LOG/'review_blocks'/(name+'.json')
            need(intent==dict(schema=44,phase=phase,block=name,binding=binding,output=output.relative_to(root).as_posix(),
                              numerical_cache_analysis=True,neural_forward=False,parameter_fit=False,raw_replay=False), 'Intent identity differs')
            load_commit(output,binding)
            need(returned==dict(schema=44,phase=phase,block=name,binding=binding,output_sha256=sha(output),
                                output_bytes=output.stat().st_size), 'Returned analysis output differs')
            calls.append(phase+'/'+name)
    return dict(completed_numerical_cache_analyses=calls, unknown_calls=0, new_parameter_fits=0, new_neural_forward=0)


def completed_report(root, binding):
    directory = root/OUT
    report = load_commit(directory/'result.json', binding)
    manifest = load_commit(directory/'completion.json', binding)
    names = ['result.json']+['blocks/'+name+'.json' for name in BLOCKS]
    expected = set(names+[name+'.sha256.json' for name in names]+['completion.json', 'completion.json.sha256.json'])
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    need(actual == expected and all(not p.is_symlink() for p in directory.rglob('*')), 'Unexpected audit output inventory')
    values={name:load_commit(directory/name,binding) for name in names}
    need(report['training']==values['blocks/training.json'] and report['calibration']==values['blocks/calibration.json']
         and report['decision']==values['blocks/assessment.json'],'Summary does not match committed diagnostic blocks')
    hashes = {name: sha(directory/name) for name in sorted(expected-{'completion.json', 'completion.json.sha256.json'})}
    need(manifest == dict(schema=44, binding=binding, outputs=hashes), 'Completion inventory differs')
    verify_calls(root,binding,('run',))
    return report


def run(root=ROOT, config=CONFIG):
    root = Path(root).resolve(); guard = Guard(root, 'run'); guard.check(); c, binding = verified_config(root, config)
    if (root/OUT/'completion.json').exists():
        completed_report(root, binding)
        return dict(status='verified_completed_read_only', new_blocks=0, reused_blocks=3, resources=guard.check())
    data = None; completed = {}; new = 0
    for name in BLOCKS:
        path = root/OUT/'blocks'/(name+'.json')
        def calculate():
            nonlocal data
            if data is None: data=load_allowed_cache(root,c)
            return block_value(name,data,completed)
        completed[name], calculated = compute_committed(root,'run',name,path,calculate,binding,guard)
        new += int(calculated)
    report = dict(schema=44, passed=True, engineering_cache_audit_complete=True, scientific_adoption=False,
                  binding=binding, protocol=core.PROTOCOL, training=completed['training'], calibration=completed['calibration'],
                  decision=completed['assessment'], new_parameter_fits=0, new_neural_forward=0, new_official_AP=0,
                  raw_replay=0, mixed_split_container_bytes_loaded=True, evaluation_rows_used_for_diagnosis=False)
    commit(root/OUT/'result.json', report, binding, guard)
    names = ['result.json']+['blocks/'+name+'.json' for name in BLOCKS]
    files = names+[name+'.sha256.json' for name in names]
    commit(root/OUT/'completion.json', dict(schema=44, binding=binding, outputs={name:sha(root/OUT/name) for name in sorted(files)}), binding, guard)
    completed_report(root, binding)
    return dict(status='completed', new_blocks=new, reused_blocks=3-new, resources=guard.check())


def review(root=ROOT, config=CONFIG):
    root = Path(root).resolve(); guard = Guard(root, 'review'); guard.check(); c, binding = verified_config(root, config)
    report = completed_report(root, binding); proof = root/LOG/'review.json'
    expected = dict(schema=44, passed=True, binding=binding, result_sha256=sha(root/OUT/'result.json'),
                    completion_sha256=sha(root/OUT/'completion.json'), rechecked_blocks=list(BLOCKS),
                    same_implementation_cache_recomputation=True, independent_statistical_implementation=False,
                    new_parameter_fits=0, new_neural_forward=0, raw_replay=0)
    if proof.exists():
        need(load_commit(proof, binding) == expected, 'Saved review differs')
        verify_calls(root,binding,('run','review'))
        return dict(status='verified_review_read_only', new_blocks=0, resources=guard.check())
    data = None; completed = {}; new = 0
    for name in BLOCKS:
        marker = root/LOG/'review_blocks'/(name+'.json')
        original = load_commit(root/OUT/'blocks'/(name+'.json'), binding)
        check = dict(schema=44, binding=binding, name=name, original_sha256=sha(root/OUT/'blocks'/(name+'.json')), equal=True)
        def calculate():
            nonlocal data
            if data is None: data=load_allowed_cache(root,c)
            rebuilt=block_value(name,data,completed)
            need(encoded(rebuilt)==encoded(original),'Cache recomputation differs: '+name)
            return check
        value,calculated=compute_committed(root,'review',name,marker,calculate,binding,guard)
        need(value==check,'Review block differs');new+=int(calculated)
        completed[name] = original
    need(report['decision'] == completed['assessment'], 'Report decision differs from assessment')
    commit(proof, expected, binding, guard)
    verify_calls(root,binding,('run','review'))
    return dict(status='reviewed', new_blocks=new, resources=guard.check())


def cold(root=ROOT, config=CONFIG):
    root = Path(root).resolve(); guard = Guard(root, 'cold'); guard.check(); c, binding = verified_config(root, config)
    completed_report(root, binding); reviewed = load_commit(root/LOG/'review.json', binding)
    need(reviewed['passed'] and reviewed['result_sha256'] == sha(root/OUT/'result.json'), 'Missing completed review')
    verify_calls(root,binding,('run','review'))
    value = dict(schema=44, passed=True, binding=binding, result_sha256=sha(root/OUT/'result.json'),
                 review_sha256=sha(root/LOG/'review.json'), completion_sha256=sha(root/OUT/'completion.json'),
                 cached_numeric_blocks_recomputed=0, new_parameter_fits=0, new_neural_forward=0, raw_replay=0)
    path = root/LOG/'cold.json'; existed = path.exists()
    if existed: need(load_commit(path, binding) == value, 'Cold proof changed')
    else: commit(path, value, binding, guard)
    return dict(status='verified_completed_read_only' if existed else 'cold_verified', resources=guard.check())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'review', 'cold'))
    parser.add_argument('--config', default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(globals()[args.mode](ROOT, args.config), sort_keys=True, allow_nan=False), flush=True)
