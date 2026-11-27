from seal_io import *
import sys, platform, subprocess, re, shutil
from collections import Counter
git=shutil.which('git')
def command(args,cwd=ROOT):
    r=subprocess.run(args,cwd=cwd,capture_output=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
    return {'args':[str(x) for x in args],'exit':r.returncode,'stdout':r.stdout.decode('utf8','replace').strip(),'stderr':r.stderr.decode('utf8','replace').strip()}
repos={}
for k in ('.',R6,R7,R8,'成文','third_party/graspnet-baseline','third_party/graspnetAPI','runtime/build/graspnet','runtime/build/graspnetAPI_eval_v7'):
    top=command([git,'rev-parse','--show-toplevel'],ROOT/k);v={'toplevel':top}
    if top['exit']==0:
        v.update(head=command([git,'rev-parse','HEAD'],ROOT/k),status=command([git,'status','--porcelain=v1','--untracked-files=all'],ROOT/k),
                 tags=command([git,'tag','--list','paper_v1_r6r7_frozen'],ROOT/k))
        patch=subprocess.check_output([git,'diff','--binary','HEAD'],cwd=ROOT/k,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
        if patch:(REPO/'metadata'/('upstream_'+k.replace('/','_')+'.diff')).write_bytes(patch)
    repos[k]=v
put(REPO/'metadata/git_before.json',dict(observed_utc=utc(),repositories=repos,root_historical_repository=False))
r=ROOT/R6;orig=ROOT/P0;ledger=read(r/'analysis/formal_ledger.json')['rows']
cases=[]
for target in ['success','empty_old_E','static_distance_uncertified']:
    row=min((x for x in ledger if x['detail_class']==target),key=lambda x:x['trial_id'])
    tid=row['trial_id'];f=r/'trials'/tid/'intent.json';it=read(f);u=it['unit_id'];unit=orig/'units'/u
    result=read(f.parent/'method_result.json')
    rec=dict(trial_id=tid,unit=u,method=it['method'],scene=it['scene_key'],candidate=it['selected_candidate_id'],intent_sha256=sha(f),
             expected_detail=target,expected_grasp_success=result['grasp_success'],expected_engineering_gate=result['engineering_gate_passed'],
             expected_physical_returns=result['physical_returns'],old_E_count=it['old_E_count'],pool_count=it['common_pool_count'],
             expected_raw_indices=read(unit/'candidate_domain_ledger.json')['top20_original_indices'],expected_order=read(unit/('ranking_'+it['method']+'.json'))['complete_order'],
             all_method_indices=read(unit/'selection.json')['selected'],pool_sha=it['pool_sha256'],raw_sha=it['raw_decoder_sha256'],
             observation_sha=it['observation_sha256'],stable_sha=it['stable_state_sha256'],worker_seed=it['worker_seed'],
             registered_sampling_seed=it['sampling_seed'],registered_condition_seed=it['condition_seed'],
             inputs={k:it[k] for k in it if k.endswith('_sha256') or k.endswith('_path')},model_input_fingerprints=it['model_input_fingerprints'])
    rec['ranking_components_absolute_tolerance']=1e-12
    if target=='success':
        jr=read(f.parent/'episode/result.json')
        rec.update(max_lift_reference_m=jr['max_lift_m'],max_lift_absolute_tolerance_m=1e-6,
                   final_streak_expected=jr['final_hold_streak'],minimum_terminal_consecutive_returns=480,min_terminal_lift_m=.1)
    if (f.parent/'static_preflight/path_result.json').is_file():
        pa=read(f.parent/'static_preflight/path_result.json');rec['static_snapshots_expected']=pa['snapshots_completed'];rec['static_first_failure_expected']=pa['first_failure']
    cases.append(rec)
put(REPO/'metadata/cold_preregistration.json',dict(created_utc=utc(),status='fixed_before_reproduction',rule='lexically first R6 trial in each of the three requested frozen outcome classes',
    cases=cases,order=[x['trial_id'] for x in cases],max_unique_cases=3,nominal_model_calls={'P3':3,'P5':3},
    physical_step_cap=4728,max_total_models_including_packaging_restarts={'P3':3,'P5':3},max_packaging_retries=2,
    max_attempts_per_physical_case=1,allow_resume_completed_stage=False,
    retry_scope='Only a packaging/path pre-call fault with exact proof of no model or physics invocation; keep all failure receipts. Unknown starts consume the corresponding cap; never repeat after model/step invocation.',
    static_checks_per_selected_case_max=4728,minimum_success_hold_s=2.0,
    tolerances='source IDs, 17-vector dtype/bytes, object and eligibility, selection indices, scores, result class and step counts checked; all native safety gates unmodified',
    stop='stop V2 on any unsuccessful cold gate; zero extra experiments',independent_confirmation=False))
put(REPO/'metadata/platform_capture.json',dict(utc=utc(),system=platform.platform(),interpreter=sys.version,base=str(Path(sys.base_prefix).resolve()),
    git=command([git,'--version']),gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,compute_cap','--format=csv,noheader']),
    cuda_driver=command(['nvidia-smi']),cpu=command(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress']),
    machine=command(['powershell','-NoProfile','-Command','Get-CimInstance Win32_ComputerSystem | Select Model,TotalPhysicalMemory | ConvertTo-Json -Compress']),
    windows=command(['powershell','-NoProfile','-Command','Get-CimInstance Win32_OperatingSystem | Select Caption,Version,BuildNumber | ConvertTo-Json -Compress']),
    toolchain=command(['where','nvcc']),active_env={k:os.environ.get(k) for k in ['CUDA_PATH','CUDA_VISIBLE_DEVICES','PYTHONHASHSEED','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBER_OF_PROCESSORS']},
    recorded_time='capture time, not historical hardware telemetry',source_execution_env_origin=P0+'/code/dispatch.py; '+R6+'/code/formal_dispatch.py; '+R7+'/code/dispatch_ablation.py'))
print('frozen cases',[(c['trial_id'],c['expected_detail']) for c in cases],flush=True)
