"""Audited rank-only A/B using both original expected scientific outputs and two fresh isolated processes per unit."""
from pathlib import Path
import gzip,hashlib,json,os,shutil,subprocess,sys,time,traceback
V=Path(__file__).resolve().parents[1];B=V/'adapter';T=V/'cold';P=T/'p';OLD=V.parent/'V1_repro_dependency_closure03'
R=P/'IP140_R2_R6formal720_static_distance_uncertified_resume01';SM='IP140_R2_smoke22_miss_negative_controls_full_execution01'
def read(f):return json.loads(f.read_bytes())
def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def out(p,d):
    with p.open('x',encoding='utf8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
def gate():
    for role in ('vision','rank','physics'):
        pre=read(T/'setup_logs'/('preflight_'+role+'.json'))
        assert pre['model_forwards']==pre['physical_steps']==0 and read(T/'setup_logs'/('fonts_'+role+'.json'))['accepted']
    cases=read(B/'metadata/cold_preregistration.json')['cases']
    fixtures=read(OLD/'audit/preflight_fixture_manifest.json')['files']
    keys={f['registered_path']for f in fixtures}
    with gzip.open(B/'locks/source_R6_final.jsonl.gz','rt',encoding='utf8')as z:
        locks={p:(n,h) for p,n,h in map(json.loads,z) if p in keys}
    assert set(locks)==keys and len(keys)==12
    lane=R/'workspaces/rank_gate04';lane.mkdir(parents=True,exist_ok=False)
    system=Path(os.environ['SystemRoot']);shell=system/'System32/WindowsPowerShell/v1.0/powershell.exe'
    for name in ('data',SM):
        e=os.environ.copy();e['LINK_DST']=str(lane/name);e['LINK_SRC']=str(P/name)
        subprocess.run([str(shell),'-NoProfile','-NonInteractive','-Command',"$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:LINK_DST -Target $env:LINK_SRC | Out-Null"],check=True,env=e)
        assert (lane/name).is_junction() and (lane/name).resolve()==(P/name).resolve()
    out(lane/'workspace.json',{'purpose':'isolated zero-step A/B rank qualification','no_method_trials':True})
    result=[]
    for arm in ('A','B'):
        for case in cases:
            tid,uid=case['trial_id'],case['unit'];dest=V/'gates'/arm/'units'/uid;dest.mkdir(parents=True,exist_ok=False)
            for f in fixtures:
                if f['trial']!=tid:continue
                src=Path(f['actual']);rel=f['registered_path']
                assert Path(rel).parent.name==uid and (src.stat().st_size,sha(src))==locks[rel]==(f['bytes'],f['sha256'])
                if src.name in ('network.npz','network.json'):
                    shutil.copyfile(src,dest/src.name);assert sha(dest/src.name)==f['sha256']
            e={k:v for k,v in os.environ.items() if k.upper() in ('WINDIR','SYSTEMROOT','COMSPEC','SYSTEMDRIVE','NUMBER_OF_PROCESSORS','PROCESSOR_ARCHITECTURE','PATHEXT','COMPUTERNAME','USERNAME')}
            tmp=dest/'scratch';tmp.mkdir()
            e.update(PATH=os.pathsep.join(map(str,[T/'env/rank/Scripts',T/'bin',T/'bin/DLLs',system/'System32',system])),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONHASHSEED=str(read(B/'metadata/stage_seeds.json')[tid]['rank']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TEMP=str(tmp),TMP=str(tmp),TMPDIR=str(tmp),CUDA_VISIBLE_DEVICES='',HOME=str(tmp),USERPROFILE=str(tmp),LOCALAPPDATA=str(tmp),APPDATA=str(tmp))
            cmd=[str(T/'env/rank/Scripts/python.exe'),'-s','-B',str(B/'adapters/ranking_fixture.py'),arm,tid]
            out(dest/'invocation.json',{'cwd':str(lane),'command':cmd,'case':tid,'unit':uid,'fresh':True,'utc_start':time.time()})
            with (dest/'stdout.log').open('xb')as so,(dest/'stderr.log').open('xb')as se:c=subprocess.run(cmd,env=e,cwd=lane,stdout=so,stderr=se)
            if c.returncode:raise RuntimeError((tid,arm,c.returncode,(dest/'stderr.log').read_text(encoding='utf8',errors='replace')[-1500:]))
            r=read(dest/'probe.json');fonts=read(dest/'font_source.json');assert r['passed'] and fonts['accepted']
            r['font_entry_count']=len(fonts['resolved_font_entries']);r['font_sha256']=sha(dest/'font_source.json');result.append(r)
    for i in range(3):
        a,b=result[i],result[i+3]
        for k in ('case','unit','pool_sha256','ledger_sha256','all_score_fields_sha256','selection_core_sha256','full_eligible_sha256','observed_E','pool_count','methods'):
            assert a[k]==b[k],k
        assert a['font_entry_count']>b['font_entry_count']==2
    out(V/'audit/font_AB_and_rank_gate.json',dict(passed=True,arms={'A':'fresh cache using all locked Matplotlib fonts','B':'separate cache using two wheel fonts'},historical_baseline='sealed R6 expected score/selection outputs; old 03 logged incidental host-font reads',three_registered_units=3,models=0,physics_steps=0,processes=result,classified_host_fonts='incidental non-scientific dependencies; not redistributed'))
if __name__=='__main__':
    try:gate()
    except BaseException:
        out(V/'audit/pre_cold_STOP.json',dict(traceback=traceback.format_exc(),cold_started=False));raise
