"""Summarize newly returned scientific output and evidence; do not modify its classifications."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,os,sys
V=Path(__file__).resolve().parents[1];T=V/'cold';P=T/'p';B=V/'adapter'
R=P/'IP140_R2_R6formal720_static_distance_uncertified_resume01';C=T/'cold_receipts';OLD=V.parent/'V1_repro_dependency_closure03'
def read(p):return json.loads(p.read_bytes()) if p.is_file() else None
def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,d):
    with p.open('x',encoding='utf8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
sys.path.insert(0,str(B/'adapters'))
from summarize import main as actual_summary
assert (C/'STOP.json').is_file() or (C/'complete.json').is_file()
d=actual_summary(T);old=read(OLD/'audit/three_case_gate.json')
d.update(history_before_final04={'case_outcomes':'0/3 in each prior round; preserved as filed','prior_receipts_as_filed':old['historical_counters_as_filed'],'dependency03_cold_started':old['final_cold_started'],'dependency03_final_cold_P3':old['final_cold_P3'],'dependency03_final_cold_P5':old['final_cold_P5']})
calendars=list(map(json.loads,(C/'ledger.jsonl').read_bytes().splitlines()));d['stage_ledger']=calendars
cases=read(B/'metadata/cold_preregistration.json')['cases']
identity=[];files={};artifacts=[];blocked=[]
seps={os.path.normcase(str((P/r['registered_path']).resolve())) for r in read(V/'provenance_evidence_appendix/manifest.json')['files']}
for q,r in zip(cases,d['cases']):
    tid=q['trial_id'];unit=q['unit'];o=C/tid;trial=R/'trials'/tid
    r['has_been_dispatched']=any(z.get('trial')==tid and z['event']=='stage_dispatch' for z in calendars)
    r['accepted_method_return']=r['cold_gates']['method'] and any(z.get('trial')==tid and z.get('role')=='method' and z['event']=='stage_return' and z['code']==0 for z in calendars)
    r['api']={}
    for k,apiroot in [('eligibility',R/'units'/unit/'eligibility_api'),('static',trial/'static_api'),('dynamic',trial/'dynamic_api'),('empty',trial/'empty_api'),('empty_old_E',trial/'no_candidate_zero_api')]:
        s=read(apiroot/'summary.json')
        if s is not None:r['api'][k]={'connections_requested':s['calls'].get('connect',0),'connections_returned':s['returns'].get('connect',0),'native_step_calls':s['actual_native_step_calls'],'native_step_returns':s['returns'].get('stepSimulation',0),'closest_queries':s['calls'].get('getClosestPoints',0),'closest_returns':s['returns'].get('getClosestPoints',0),'exceptions':s['exceptions'],'summary_sha256':sha(apiroot/'summary.json')}
    result=read(trial/'method_result.json')
    r['engine_or_method_observations']={k:result[k] for k in ('shared_fault','failure_detail','failure_category','detail_class','physical_steps','physical_returns','engineering_gate_passed','grasp_success','trial_completed') if (result is not None and k in result)}
    ep=read(trial/'episode/result.json')
    if ep:r['native_result_selected_fields']={k:ep[k]for k in ('max_lift_m','final_hold_streak','physical_steps','completed_physical_steps','grasp_success','failure_category','failure_reason','failure_stage','attempted_physical_steps','engineering_gate_passed','episode_completed','violations') if k in ep}
    for role in ('forward','rank','method'):
        e=read(o/(role+'_data_scope.json'));font=read(o/(role+'_font_scope.json'));mods=read(o/(role+'_imports.json'))
        if mods:identity.append({'case':tid,'stage':role,'engine_sha256':mods.get('engine_sha256'),'unregistered_modules':mods.get('outside'),'font_isolation_return':font and font['accepted'],'project_reads':len(e['unique_open_request_counts']) if e else None})
        if e:
            blocked.extend(e['blocked_old_project_access'])
            for path,count in e['unique_open_request_counts'].items():
                if path in seps:raise ValueError('provenance mounted')
                f=Path(path)
                if f.is_relative_to(P) and f.is_file():
                    rec=files.setdefault(path,dict(cases_and_stages=[],bytes=f.stat().st_size,whole_sha256=sha(f) if path not in files else files[path]['whole_sha256']))
                    rec['cases_and_stages'].append([tid,role,count])
d['case_acceptances']=sum(r['accepted_method_return'] for r in d['cases'])
for label,key in [('physical_returns','native_returns'),('physical_requests','native_calls')]:
    counts=[r[key] for r in d['cases']]
    d[label]=sum(counts) if all(x is not None for x in counts) else None
d['unknown_native_step_counts']=any(x['native_calls'] is None or x['native_returns'] is None for x in d['cases'])
d['models']={n:{k:sum(r['model_calls'][n][k] for r in d['cases'])for k in ('entered','returned')}for n in ('P3','P5')}
d['source_scopes']=identity;d['forbidden_project_open_attempts']=blocked
d['independent_cold_passed']=d['gate_passed'] and d['case_acceptances']==3 and not blocked and not d['unknown_native_step_counts']
d['verdict']='V1可独立复现（三个冻结案例）' if d['independent_cold_passed'] else 'V1独立冷启动未通过'
put(V/'audit/final_cold_outcome.json',d)
with gzip.open(V/'audit/actually_opened_cold_project_inputs.jsonl.gz','xt',encoding='utf8') as f:
    for path,x in sorted(files.items()):f.write(json.dumps(dict(path=path,**x),ensure_ascii=False)+'\n')
print(d['verdict'],d['case_acceptances'],d['models'],d['physical_requests'],d['physical_returns'])
