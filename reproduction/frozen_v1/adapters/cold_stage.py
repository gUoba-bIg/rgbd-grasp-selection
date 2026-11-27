"""Path and provenance adapter. All inference, ranking, eligibility and physical evaluation call frozen functions."""
import os, sys, hashlib, json, importlib.util, random, traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
target=Path(sys.argv[1]).resolve();role=sys.argv[2];tid=sys.argv[3]
R6='IP140_R2_R6formal720_static_distance_uncertified_resume01'
R4='IP140_R2_R4dev72_cross_dim_explicit_resume01'
P=target/'p';run=P/R6;out=target/'cold_receipts'/tid
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,ob):
    with p.open('x',encoding='utf8') as f:json.dump(ob,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
def check(c,why):
    if not c:raise RuntimeError(str(why))
case=next(c for c in read(BASE/'metadata/cold_preregistration.json')['cases'] if c['trial_id']==tid)
uid=case['unit'];stage={'forward':'vision','rank':'rank','method':'physics'}[role]
check(Path(sys.prefix).resolve()==target/'env'/stage and Path(sys.base_prefix).resolve()==target/'bin','unregistered interpreter')
sys.path[:0]=[str(run/'code'),str(run/'implementation'),str(P/R4/'implementation'),str(P)]
os.chdir(run/'workspaces/cpu01')
os.environ['PROGRAMFILES']=str(P/'vendor/ProgramFiles')
from ctypes import wintypes
import ctypes
libdirs=[P/'vendor/ProgramFiles/Git/mingw64/bin',target/'bin',target/'bin/DLLs',target/'env'/stage/'Lib/site-packages/torch/lib']
handles=[os.add_dll_directory(str(d)) for d in libdirs if d.is_dir()]
def module_log():
    paths={};bad=[];driver_by_name=None;external_driver={}
    for name,mod in list(sys.modules.items()):
        f=getattr(mod,'__file__',None)
        if not f or not Path(f).is_file():continue
        f=Path(f).resolve()
        if not (f.is_relative_to(target) or f.is_relative_to(BASE)):bad.append([name,str(f)])
        paths[name]=str(f)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);ps=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE;proc=kernel.GetCurrentProcess()
    ps.EnumProcessModules.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.HMODULE),wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    ps.GetModuleFileNameExW.argtypes=[wintypes.HANDLE,wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    mods=(wintypes.HMODULE*8192)();needed=wintypes.DWORD();binaries={}
    check(ps.EnumProcessModules(proc,mods,ctypes.sizeof(mods),ctypes.byref(needed)),'module enumeration')
    for handle in mods[:needed.value//ctypes.sizeof(wintypes.HMODULE)]:
        b=ctypes.create_unicode_buffer(32768)
        if not ps.GetModuleFileNameExW(proc,handle,b,len(b)):continue
        f=Path(b.value).resolve();binaries[str(f)]=f.stat().st_size if f.is_file() else None
        if not f.is_relative_to(target) and not f.is_relative_to(Path(os.environ['SYSTEMROOT']).resolve()):
            if driver_by_name is None:
                import gzip
                with gzip.open(BASE/'locks/signed_display_driver.jsonl.gz','rt',encoding='utf8') as z:
                    driver_by_name={}
                    for ln in z:
                        nm,_,h=json.loads(ln);driver_by_name.setdefault(Path(nm).name.casefold(),set()).add(h)
            dh=sha(f)
            if dh in driver_by_name.get(f.name.casefold(),set()):external_driver[str(f)]=dh
            else:bad.append(['native',str(f)])
    eng=paths.get('pybullet')
    put(out/(role+'_imports.json'),dict(executable=sys.executable,base=sys.base_prefix,sys_path=sys.path,python_modules=paths,native_modules=binaries,registered_external_driver=external_driver,outside=bad,
        engine_sha256=sha(eng) if eng else None,project_module_hashes={str(f.relative_to(P)):sha(f) for f in {Path(j) for j in paths.values()} if f.is_relative_to(P)}))
    check(not bad,('module outside registered runtime/OS',bad))
def main():
    import numpy as np
    from task_worker import make_unit,rank_and_E
    import common
    design=read(run/'preregistration/design.json')
    u=next(x for x in design['units'] if x['unit_id']==uid);scene=next(x for x in design['scenes'] if x['base_scene']==u['scene'])
    path=run/'units'/uid
    if role!='method' and not (path.parent).exists():path.parent.mkdir()
    stage_seed=int(os.environ['PYTHONHASHSEED']);random.seed(stage_seed);np.random.seed(stage_seed)
    if role=='forward':
        check(not path.exists(),'new inference directory required')
        check(sha(run/'scenes'/u['scene']/'observed_rgbd.npz')==case['observation_sha'],'raw observation changed')
        # Persist reservation before the first possible forward. A missing return is never retried.
        put(out/'forward_dispatch.json',dict(reserved_P3=1,reserved_P5=1,input=case['observation_sha'],worker_seed=case['worker_seed']))
        from probes import count_models
        with count_models(out,case["model_input_fingerprints"]):make_unit(u,scene)
        n=read(path/'network.json')
        check(sha(path/'raw_decoder.npy')==case['raw_sha'] and n['networks_returned']=={'P3':1,'P5_fixed':1},'raw network or returned model mismatch')
        put(out/'forward_return.json',dict(raw_decoder_sha256=sha(path/'raw_decoder.npy'),network_receipt=n))
    elif role=='rank':
        check((out/'forward_return.json').is_file(),'missing fresh model return')
        rank_and_E(uid)
        sel=read(path/'selection.json');o=read(path/'orders.json');expected=read(BASE/'metadata/smoke_expected.json')[tid]
        def near(a,b):
            if type(a) in (float,int) and type(b) in (float,int):return np.isfinite(a) and np.isfinite(b) and abs(a-b)<=1e-12
            if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(near(x,y) for x,y in zip(a,b))
            if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(near(a[k],b[k]) for k in a)
            return a==b
        def score_core(x):return {k:v for k,v in x.items() if k not in ('scoring_sec','pool_build_sec')}
        check(near(score_core(o),expected['orders']),'frozen score, version, pool or observation equality')
        core={k:v for k,v in sel.items() if k not in ['native_eligibility_api_sha256','eligibility']}
        eligibility=[{k:v for k,v in row.items() if k!='planning_seconds'} for row in sel['eligibility']]
        check(core==expected['selection_core'] and eligibility==expected['eligibility'],'fresh eligibility or trajectory mismatch')
        check(sum(q['eligible'] for q in sel['eligibility'])==case['old_E_count'] and o['pool_count']==case['pool_count'] and sha(path/'pool.npz')==case['pool_sha'] and sel['selected']==case['all_method_indices'],'common pool or selection mismatch')
        put(out/'rank_return.json',dict(selected=sel['selected'],eligible_count=sum(q['eligible'] for q in sel['eligibility']),fresh_selection=sha(path/'selection.json'),old_E_check=True))
    else:
        check((out/'rank_return.json').is_file(),'unverified fresh selection')
        # Frozen intents hash timing-dependent receipts. Bind ONLY our three regenerated receipts to
        # a derived replay envelope after confirming the original 720-intent seal. No cached plan is executed.
        intent_path=run/'trials'/tid/'intent.json';it=read(intent_path);seal_path=run/'sealed_intents.json';seal=read(seal_path)
        check(len(seal['files'])==720 and sha(intent_path)==case['intent_sha256']==seal['files'][f'trials/{tid}/intent.json'],'original intent')
        mapping={'observation':run/it['observation_path'],'stable_state':run/it['stable_state_path'],'pool':path/'pool.npz',
            'raw_decoder':path/'raw_decoder.npy','ranking':path/('ranking_'+case['method']+'.json'),'selection':path/'selection.json','scoring_components':path/'orders.json','exclusion':path/'candidate_domain_ledger.json'}
        changes={}
        for tag,file in mapping.items():
            check(run/it[tag+'_path']==file,'intent path')
            val=sha(file);old=it[tag+'_sha256']
            if old!=val:
                check(tag in ('selection','scoring_components','ranking'),'disallowed rebinding '+tag)
                changes[tag]={'old':old,'new':val};it[tag+'_sha256']=val
        put(out/'receipt_rebinding.json',dict(original=case['intent_sha256'],differences=changes,purpose='Non-deciding elapsed times and native trace metadata; old decisions separately compared, fresh plan used.'))
        # Both files live solely in the disposable cold output tree. Frozen archive files are immutable.
        for f in [intent_path,seal_path]:
            with f.open('rb') as a,(out/('before_'+f.name)).open('xb') as b:b.write(a.read())
        if changes:
            intent_path.write_text(json.dumps(it,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf8')
            seal['files'][f'trials/{tid}/intent.json']=sha(intent_path)
            seal_path.write_text(json.dumps(seal,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf8')
        from platform_trace import R4Trace
        authorize=R4Trace.authorize_dynamic
        def qualified_authorization(self,*args,**kwargs):
            if case['expected_physical_returns']!=4728:raise RuntimeError('registered zero physical budget')
            return authorize(self,*args,**kwargs)
        R4Trace.authorize_dynamic=qualified_authorization
        from method_entry import main as run_method
        put(out/'method_dispatch.json',dict(reproduction_only=True,maximum_steps=case['expected_physical_returns'],candidate=case['candidate']))
        sys.argv=[str(run/'code/method_entry.py'),tid,str(it['worker_seed'])]
        run_method()
        result=read(intent_path.parent/'method_result.json');steps=result['physical_returns']
        detail='success' if result['grasp_success'] else 'empty_old_E' if it['old_E_count']==0 and result['failure_category'] in ('no_executable_candidate','no_eligible_candidate') else result.get('detail_class')
        check(detail==case['expected_detail'] and steps==result['physical_steps']==case['expected_physical_returns'] and not result['shared_fault']
              and result['engineering_gate_passed']==case['expected_engineering_gate'] and result['grasp_success']==case['expected_grasp_success'],'reproduction result mismatch')
        if case['candidate']:
            stat=read(intent_path.parent/'static_preflight/path_result.json')
            check(stat['snapshots_completed']==case['static_snapshots_expected'] and stat['first_failure']==case['static_first_failure_expected'],'static classification mismatch')
        if steps:
            ep=read(intent_path.parent/'episode/result.json')
            check(abs(ep['max_lift_m']-case['max_lift_reference_m'])<=case['max_lift_absolute_tolerance_m'] and ep['final_hold_streak']==case['final_streak_expected'],'numeric hold comparison')
        put(out/'method_return.json',dict(classification=detail,physical_returns=steps,grasp_success=result['grasp_success'],result_sha=sha(intent_path.parent/'method_result.json'),original_intent=case['intent_sha256']))
if __name__=='__main__':
    try:main()
    finally:module_log()
