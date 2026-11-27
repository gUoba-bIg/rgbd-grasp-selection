"""Create a new interface-repair workspace and record preservation; no simulator imports."""
import sys,json,tarfile,shutil
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sim_v52_common import read,write,identity,relative,now,lock
NEW=ROOT/'runtime/simulation_v52e'
LOG=ROOT/'logs/simulation_v52e'
ARCHIVE=ROOT/'runtime/simulation_v52/cache/http-v2/4/5/7/b/d/457bdc0f13fc3174ca2e66c339d420a683bd3fc6fb4f34259b382951.body'

def main():
    assert Path.cwd().resolve()==ROOT
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52e.lock'):
        if (LOG/'authorization.json').exists():raise RuntimeError('Already prepared; inspect rather than repeat')
        from scripts.sim_v52d_control import historical_verify
        historical_verify()
        p=dict(read(ROOT/'logs/simulation_v52d/historical_snapshot.json')['protected'])
        for name,value in read(ROOT/'results/simulation_v52d/delivery_closeout.json')['artifacts'].items():
            if name=='docs/实验自动续接说明.txt':continue
            assert identity(ROOT/name)==value,name
            p[name]=value
        for folder in ['runtime/simulation_v52/venv','results/simulation_v52d','logs/simulation_v52d']:
            for f in (ROOT/folder).rglob('*'):
                if f.is_file():p[relative(f)]=identity(f)
        for pattern in ['scripts/sim_v52*.py','docs/v52d*']:
            for f in ROOT.glob(pattern):
                if f.is_file():p[relative(f)]=identity(f)
        for name in ['runtime/pipeline/simulation_v52_dynamics_run.json',relative(ARCHIVE)]:p[name]=identity(ROOT/name)
        assert identity(ARCHIVE)['sha256']=='042879db8d101ac7590dee475fc6aded508b85fe1273fdbbfde1d88bd200e14f'
        LOG.mkdir(parents=True,exist_ok=True)
        write(LOG/'historical_snapshot.json',dict(at=now(),protected=p,live_handoff_mutable_with_archive=True),once=True)
        write(LOG/'authorization.json',dict(at=now(),task='v52e interface repair and zero-step actual parameter acceptance',
            local_source_patch_authorized=True,isolated_build_authorized=True,zero_step_acceptance_authorized=True,
            allowed_physics_steps=0,physics_motion_authorized=False,new_12_authorized=False,M1_authorized=False,
            new48_authorized=False,R4_authorized=False,automatic_retry_after_parameter_failure=False,
            targets=dict(lateralFriction=1.,collisionMargin=.0005,restitution=0.),master_cap_Nm=10.,gear_cap_Nm=80.,mimic_gate_rad=.03),once=True)
        for directory in ['src','temp','cache','wheels','build','prep']:(NEW/directory).mkdir(parents=True,exist_ok=True)
        dst=NEW/'src'
        with tarfile.open(ARCHIVE,'r:gz') as archive:
            members=archive.getmembers()
            for m in members:
                target=(dst/m.name).resolve();target.relative_to(dst.resolve())
                if m.issym() or m.islnk():raise ValueError('Unexpected archive link: '+m.name)
            archive.extractall(dst,members=members,filter='data')
        src=dst/'pybullet-3.2.7'
        assert (src/'setup.py').exists()
        before={str(f.relative_to(src)).replace('\\','/'):identity(f) for f in src.rglob('*') if f.is_file()}
        write(LOG/'source_before.json',dict(at=now(),archive=identity(ARCHIVE),source_root=relative(src),files=before),once=True)
        shutil.copy2(ROOT/'docs/实验自动续接说明.txt',LOG/'resume_note_before_v52e.txt')
        write(ROOT/'runtime/pipeline/simulation_v52e.json',dict(at=now(),phase='source_review_and_isolated_build',
            physics_steps=0,acceptance_starts=0,M1_implemented=False,R4_authorized=False,
            old_scope_all_tasks_percent=45,old_scope_experiment_percent=56.25,expanded_scope_percent=None))
        print(json.dumps(dict(protected_files=len(p),source_files=len(before),new_source=relative(src),new_physics_steps=0)))

if __name__=='__main__':main()
