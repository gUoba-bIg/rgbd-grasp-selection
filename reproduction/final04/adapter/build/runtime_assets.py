from seal_io import *
import sys, shutil
from collections import Counter
frozen=read(ROOT/R6/'preregistration/frozen_inputs.json')
files={}
def add(p,expected=None):
    p=Path(p);check(p.is_file(),p);key=p.relative_to(ROOT).as_posix()
    if key not in files or expected:files[key]=(p,expected)
def tree(path):
    for p in walk(ROOT/path):add(p)
for rel,h in frozen.items():add(ROOT/rel,h)
for d in ['src','scripts','configs','data/simulation_v52/robot','audit_package_v52f/source_diff','third_party/patches','runtime/build/graspnet','runtime/build/graspnetAPI_eval_v7','third_party/graspnet-baseline','third_party/graspnetAPI',
          'IP140_R1_smoke10_slot07_nominal021_01/static_precheck']:
    tree(d)
for n in (P0,R6,R7,R4,PL):
    for q in (ROOT/n).glob('*.py'):add(q)
    for q in (ROOT/n).glob('*.json'):
        if q.stat().st_size < 200_000_000:add(q)
    for d in ('code','implementation','preregistration','model','protocol'):
        if (ROOT/n/d).is_dir():tree(n+'/'+d)
for n in (P0,R6):
    for d in (ROOT/n/'scenes').iterdir():
        for f in ['observed_rgbd.npz','settled_state.json','receipt.json']:add(d/f)
for d in (ROOT/R6/'trials').iterdir():add(d/'intent.json')
for p in (ROOT/PL/'slots/10/zero_step').glob('*.json'):add(p)
for q in [ROOT/'data/simulation_v52/objects/voxel_repair/014',ROOT/'data/simulation_v52/objects/voxel_repair/017',ROOT/'data/simulation_v52/objects/final_static_audit/021']:
    for f in walk(q):add(f)
for q in [ROOT/'runtime/simulation_v52e/src/pybullet-3.2.7/examples/SharedMemory/PhysicsServerCommandProcessor.cpp',ROOT/'runtime/simulation_v52e/src/pybullet-3.2.7/examples/pybullet/pybullet.c',ROOT/'runtime/simulation_v52e/src/pybullet-3.2.7/examples/SharedMemory/PhysicsClientC_API.cpp']:add(q)
for p in (ROOT/'audit_package_v52f/source_diff/patched_source').rglob('*'):
    if p.is_file():
        real=ROOT/'runtime/simulation_v52e/src/pybullet-3.2.7'/p.relative_to(ROOT/'audit_package_v52f/source_diff/patched_source')
        if real.exists():add(real)
proof=read(ROOT/'IP140_R1_smoke10_slot07_nominal021_01/static_precheck/damping_readback_blocker.json')
for e in proof['source_evidence']:
    source=ROOT/e['path']
    if source.is_file():add(source,e['sha256'])
for n in ['results/P6/local_correspondence_v50','runtime/downloads/grasp_eval_v7']:
    for q in (ROOT/n).glob('*.json'):add(q)
# Include historical source, configuration and templates reached by older engineering imports.
for d in ROOT.iterdir():
    if d.is_dir() and d.name.startswith(('IP140_R1_smoke','IP140_R2_smoke')):
        for pat in ['*.py','*.json','model/*.json','model/*.urdf']:
            for p in d.glob(pat):
                if p.is_file() and p.stat().st_size<20_000_000:add(p)
for q in (ROOT/'docs').glob('*'):
    if q.is_file() and any(x in q.name.lower() for x in ['v52e','pybullet','点云','cuda','扩展','环境','构建','v52f']):add(q)
for q in (ROOT/'runtime/simulation_v52e').glob('*.json'):add(q)
for q in (ROOT/'runtime/wheels').glob('*.json'):add(q)
source=Path(os.environ['ProgramFiles'])/'Git/mingw64/bin/libzstd.dll'
files['vendor/ProgramFiles/Git/mingw64/bin/libzstd.dll']=(source,None)
# Ship the imported DLL's local dependencies where present, exactly as installed with Git.
for n in ['libwinpthread-1.dll','libgcc_s_seh-1.dll','libstdc++-6.dll']:
    f=source.with_name(n)
    if f.is_file():files['vendor/ProgramFiles/Git/mingw64/bin/'+n]=(f,None)
put(REPO/'metadata/binary_bindings.json',{'pybullet':frozen['runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd'],
    'zstd_sha256':sha(source),'windows_runtime_and_gpu_driver':'existing system drivers; ABI pinned by capture and checked on setup',
    'exact_frozen_inputs_count':len(frozen),'upstream_H_inputs':R6+'/preregistration/frozen_inputs.json'})
pack('execution_project',files)
base=Path(sys.base_prefix);rt={}
for p in walk(base):
    rel=p.relative_to(base)
    if 'site-packages' in {x.lower() for x in rel.parts} or rel.parts[0]=='Scripts':continue
    rt[rel.as_posix()]=(p,None)
check('python.exe' in rt and 'python312.dll' in rt,'CPython installation')
pack('python312',rt)
