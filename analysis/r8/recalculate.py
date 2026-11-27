"""Offline R8 manuscript statistics. Run with Python and NumPy; no project or simulator imports.
Default output is a new derived directory. It will not overwrite an earlier analysis."""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import argparse, csv, gzip, hashlib, itertools, json, math, sys, statistics
import numpy as np

R6='IP140_R2_R6formal720_static_distance_uncertified_resume01'
R7='IP140_R2_R7ablation240_mechanism01'
ARM6=('Raw','A','C0')
ARM7=('C0','C0_no_instance','C0_depth_only','C0_warp_only')
HASH={R6:('e39fa2d1bd03f79b7732569b7307044aba1303f4686cb0eb63edb1d575f2510f','aed378d1487c6be5895978e895106be375c73d964be2c29cf62104297bca234d'),
      R7:('17ef16090537dda292a1767ce8128a6031ae2965a370871e2300d1caaf1f1baf','367436e9c1d9f457f7d691f5c64239ddbc3ec6cbdcbf8c02306c51ac2834cc1c')}
def insist(b,n):
    if not b: raise ValueError(n)
def bad(v):raise ValueError('non-finite JSON number '+v)
def finitefloat(v):
    x=float(v);return x if math.isfinite(x) else bad(v)
def loads(b):return json.loads(b,parse_constant=bad,parse_float=finitefloat)
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def statclass(result,static):
    if result.get('detail_class')=='static_distance_uncertified':return result['detail_class']
    cat=result['failure_category']
    if cat is None:
        insist(result['grasp_success'],'success category without success');return 'success'
    if cat=='no_executable_candidate':return 'empty_old_E'
    if cat=='no_eligible_candidate':return 'empty_shared_pool'
    if cat!='path_failure':return cat
    if not static:return 'static_other'
    if static.get('exception')=="ValueError('both_registered_entries_required_before_lift')":return 'static_missing_bilateral_entry'
    why='|'.join(static['violations'])
    families=[('registered_surface',['registered_surface','registered_face','inner_face','witness_face']),
              ('forbidden_collision',['nonpad_target_collision','robot_table_collision','unregistered_self_collision','robot_target_collision']),
              ('early_target_contact',['before_close','before_legal_entry']),('penetration',['penetration'])]
    return 'static_'+('+'.join(k for k,words in families if any(w in why for w in words)) or 'other')

class Archive:
    def __init__(self,path):
        self.p=path;self.name=path.name;self.reads={};self.json_cache={}
        manifest=path/'SHA256SUMS.txt';self.manifest_sha=digest(manifest)
        insist(self.manifest_sha==HASH[self.name][0], 'manifest SHA '+self.name)
        self.files={}
        for line in manifest.read_text(encoding='utf-8-sig').splitlines():
            h,k=line.split('  ',1)
            insist(k not in self.files and len(h)==64,'manifest entry');self.files[k]=h
        receipt=loads((path/'delivery/verification.json').read_bytes())
        self.zip=receipt.get('core_path',receipt.get('path'))
        self.zip_sha=digest(path/self.zip)
        insist(self.zip_sha==HASH[self.name][1],'frozen package identity')
    def check(self,k,blob=None):
        if k not in self.reads:
            h=hashlib.sha256(blob).hexdigest() if blob is not None else digest(self.p/k)
            insist(h==self.files[k],self.name+'/'+k+' SHA mismatch');self.reads[k]=h
        return self.reads[k]
    def read(self,k,cache=False):
        if k in self.json_cache:return self.json_cache[k]
        b=(self.p/k).read_bytes();self.check(k,b);v=loads(b)
        if cache:self.json_cache[k]=v
        return v
    def path(self,k):return self.name+'/'+k

def per_trial(ar,phase,d,t,led,seal):
    tid=t['trial_id'];pre='trials/'+tid+'/';job='run/jobs/method/'+tid+'/'
    it=ar.read(pre+'intent.json');rs=ar.read(pre+'method_result.json');rc=ar.read(job+'return.json')
    ih,rh,jh=[ar.reads[k] for k in [pre+'intent.json',pre+'method_result.json',job+'return.json']]
    insist(seal['files'][pre+'intent.json']==ih==rc['intent_sha256']==led['intent_sha256'] and rh==rc['method_result_sha256']==led['result_sha256'] and jh==led.get('worker_receipt_sha256',led.get('receipt_sha256')),'intent/result/return '+tid)
    insist(rc['completed'] and rc['id']==rs['trial_id']==it['trial_id']==tid and it['unit_id']==rs['unit_id']==t['unit_id']==led['unit_id'] and it['method']==rs['method']==t['method']==led['method'] and it['worker_seed']==t['worker_seed'],'method/seed '+tid)
    insist(it['attempt_limit']==1 and not it['ranking_uses_ground_truth'] and it['no_postlock_fallback'] and not it['r5_interface_in_chain'],'policy scope')
    ind=it['selected_pool_index'];chosen=it['selected_candidate_id'];selection=ar.read(it['selection_path'],True);rank=ar.read(it['ranking_path'],True);poolhash=ar.check(it['pool_path'])
    insist(poolhash==it['pool_sha256']==led['pool_sha256']==rank['shared_pool_sha256'] and ar.reads[it['ranking_path']]==it['ranking_sha256'] and ar.reads[it['selection_path']]==it['selection_sha256'] and ar.check(it['stable_state_path'])==it['stable_state_sha256'],'frozen input links')
    elig=selection['eligibility'];order=rank['complete_order']
    insist(sorted(order)==list(range(len(elig))) and sum(q['eligible'] for q in elig)==it['old_E_count'] and ind==next((k for k in order if elig[k]['eligible']),None),'shared E selection '+tid)
    insist((ind is None and chosen is None) or ind is not None and elig[ind]['candidate_id']==chosen,'candidate mapping')
    insist(chosen==led['selected_candidate_id']==rc['candidate_id'] and rs['selected_pool_index']==ind and not rs['shared_fault'],'candidate identity')
    phys=rs['physical_returns'];selected=chosen is not None
    insist(phys==rs['physical_steps']==led['physical_returns'] and all(type(rs[k]) is bool and rs[k]==led[k] for k in ('trial_completed','engineering_gate_passed','grasp_success')),'receipt status '+tid)
    static=ar.read(pre+'static_preflight/path_result.json') if selected else {}
    admitted=bool(static.get('passed'));insist(admitted==(phys>0),'path/physics boundary')
    if selected:
        insist(static['candidate_id']==chosen and static['physical_steps']==0 and static['trial_id']==tid,'static owner')
    cat=statclass(rs,static);ok=rs['grasp_success']
    insist(cat==led['detail_class'],'failure route '+tid)
    if phys:
        ep=ar.read(pre+'episode/result.json');api=ar.read(pre+'dynamic_api/summary.json');ar.check(pre+'episode/telemetry.jsonl.gz')
        insist(ep['telemetry_sha256']==ar.reads[pre+'episode/telemetry.jsonl.gz'] and ep['step_returns']==phys==api['actual_native_step_calls'] and api['calls']==api['returns'] and not api['exceptions'] and ep['unknown_steps']==0 and ep['grasp_success']==ok,'dynamic return '+tid)
        window=ar.read(pre+'episode/final_480_steps.json')['observed_rows']
        if ok:
            insist(len(window)==480 and [r['step'] for r in window]==list(range(4249,4729)) and all(r['effective_left_target_contact'] and r['effective_right_target_contact'] and r['lift_m']>=.1 and not r['aggregate_table_support'] and not r['slipped'] and not r['engineering_violations'] for r in window),'hold window '+tid)
    else:insist(not ok,'no physical success possible '+tid)
    u=d[it['unit_id']]
    return dict(phase=phase,trial_id=tid,unit_id=u['unit_id'],scene=u['scene'],condition=u['condition'],object_id=u['object_id'],position=u['position'],repeat=u['repeat'],method=t['method'],candidate=chosen,old_E=it['old_E_count'],pool_size=it['common_pool_count'],static_pass=int(admitted),physical_returns=phys,success=int(ok),detail=cat,completed=int(rs['trial_completed']),engineering_gate=int(rs['engineering_gate_passed']),pool_sha256=poolhash,observation_sha256=it['observation_sha256'],state_sha256=it['stable_state_sha256'],intent_sha256=ih,result_sha256=rh,return_sha256=jh,method_result=ar.path(pre+'method_result.json'))

def wilson(k,n):
    z=statistics.NormalDist().inv_cdf(.975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-radius),min(1,center+radius)]
def exact(b,c):
    m=b+c
    if not m:return 1.
    return float(min(Fraction(1),2*Fraction(sum(math.comb(m,j) for j in range(min(b,c)+1)),2**m)))
def effects(units,methods,comparisons,seed,reps):
    grouped=defaultdict(lambda:defaultdict(list))
    for uid,rr in units.items():
        r=rr[methods[0]];grouped[r['object_id'],r['position']][r['scene']].append([rr[m]['success'] for m in methods])
        insist(len({(x['pool_sha256'],x['state_sha256'],x['observation_sha256'],x['old_E']) for x in rr.values()})==1,'pair input mismatch')
    n=len(units);rng=np.random.default_rng(seed);boot=np.zeros((reps,len(methods)),int)
    for k in sorted(grouped):
        ss=grouped[k];sc=np.array([np.array(ss[i]).sum(axis=0) for i in sorted(ss)])
        boot+=sc[rng.integers(0,len(sc),(reps,len(sc)))].sum(axis=1)
    out=dict(n=n,n_scenes=sum(len(z) for z in grouped.values()),arms={},pairs={})
    for i,m in enumerate(methods):
        rr=[r[m] for r in units.values()];k=sum(x['success'] for x in rr);dv=sum(x['physical_returns']>0 for x in rr);sp=sum(x['static_pass'] for x in rr);chosen=sum(x['candidate'] is not None for x in rr)
        out['arms'][m]=dict(n=n,success=k,rate=k/n,wilson95=wilson(k,n),chosen=chosen,empty=n-chosen,static_fail=chosen-sp,static_pass=sp,static_rate=sp/n,static_rate_if_selected=sp/chosen if chosen else None,dynamic=dv,conditional=k/dv if dv else None,dynamic_hard_fail=dv-k,native_returns=sum(x['physical_returns'] for x in rr),detail=dict(Counter(x['detail'] for x in rr)))
    for first,second in comparisons:
        win=[];lose=[];both=0;different=[];both_empty=0
        for uid,rr in units.items():
            a,b=rr[first],rr[second]
            if a['success']>b['success']:win.append(uid)
            if a['success']<b['success']:lose.append(uid)
            both+=a['success']*b['success']
            if a['candidate']!=b['candidate']:different.append(uid)
            both_empty+=a['candidate'] is None and b['candidate'] is None
        j=methods.index(first);i=methods.index(second);w,l=len(win),len(lose)
        out['pairs'][first+'-'+second]=dict(contrast=[first,second],n=n,delta_pp=100*(w-l)/n,paired95_pp=np.quantile((boot[:,j]-boot[:,i])*100/n,[.025,.975]).tolist(),both_success=both,both_fail=n-w-l-both,first_only=w,second_only=l,exact_mcnemar=exact(w,l),discordant_n=w+l,first_only_units=sorted(win),second_only_units=sorted(lose),different=len(different),different_units=sorted(different),both_empty=both_empty,same_nonempty=n-both_empty-len(different),different_rate=len(different)/n,first_only_opponent_fail=dict(Counter(units[u][second]['detail'] for u in win)),second_only_opponent_fail=dict(Counter(units[u][first]['detail'] for u in lose)))
    return out

def run(project,out):
    out.mkdir(parents=True,exist_ok=True);ar6=Archive(project/R6);ar7=Archive(project/R7);allrows=[];pairs_by_phase={};evidence=[];archives={}
    for phase,ar,arms,lf,count in [('R6',ar6,ARM6,'analysis/formal_ledger.json',720),('R7',ar7,ARM7,'analysis/ablation_ledger.json',240)]:
        d=ar.read('preregistration/design.json');reg={u['unit_id']:u for u in d['units']};s=ar.read('sealed_intents.json')
        l=ar.read(lf);ledger={r['trial_id']:r for r in l['rows']};insist(l['adjudicated']==len(ledger)==len(s['files'])==count,'denominator')
        rows=[per_trial(ar,phase,reg,t,ledger[t['trial_id']],s) for t in d['trials']]
        g=defaultdict(dict)
        for r in rows:g[r['unit_id']][r['method']]=r
        if phase=='R7':
            rf=ar.read('preregistration/full_C0_reference_index.json')['units']
            for u,rr in g.items():
                c=pairs_by_phase['R6'][u]['C0'];a=rf[u]
                insist(c['trial_id']==a['r6_trial_id'] and [a['source_hashes'][k] for k in ['intent','result','receipt']]==[c[k] for k in ['intent_sha256','result_sha256','return_sha256']],'C0 reference '+u)
                rr['C0']=c
        ps=ar.read('analysis/unit_pairs.json')
        for u in ps['units']:
            for m,x in u['methods'].items():
                row=g[u['unit_id']][m]
                insist(x['trial_id']==row['trial_id'] and x['grasp_success']==bool(row['success']) and x['selected_candidate_id']==row['candidate'] and x['detail_class']==row['detail'] and x['result_sha256']==row['result_sha256'],'existing paired file')
        allrows.extend(rows);pairs_by_phase[phase]=g
        comparisons=[('C0','A'),('A','Raw'),('C0','Raw')] if phase=='R6' else [(m,'C0') for m in arms[1:]]
        published=ar.read('analysis/paired_statistics.json')
        for label,criteria in [('all',{}),('clean',{'condition':'clean'}),('joint',{'condition':'joint'}),*[(k,{'object_id':k}) for k in ('014','017','021')]]:
            subset={u:r for u,r in g.items() if all(r[arms[0]][k]==want for k,want in criteria.items())}
            res=effects(subset,arms,comparisons,d['bootstrap_seed'],d['bootstrap_reps']);archives.setdefault(phase,{})[label]=res
            ref=published['overall' if label=='all' else 'strata'] if phase=='R6' else published['strata']['overall' if label=='all' else label]
            if phase=='R6' and label!='all':ref=ref[label]
            for m,r in res['arms'].items():
                refm=ref['rates'][m] if phase=='R6' else ref['methods'][m]
                insist(r['success']==refm['success'] and np.isclose(r['rate'],refm['rate'],rtol=0,atol=1e-14),'arm statistical mismatch')
                if phase=='R6':insist(np.allclose(r['wilson95'],refm['wilson95'],rtol=0,atol=1e-13),'Wilson')
                else:insist(r['dynamic']==refm['dynamics_entered'] and r['detail']==refm['outcomes'],'ablation admissions')
            for key,p in res['pairs'].items():
                rp=ref['paired'][key] if phase=='R6' else ref['paired_vs_full_C0'][p['contrast'][0]]
                lo=rp['cluster_CI95_percentage_points'] if phase=='R6' else rp['interval95_paired_pp']
                delta=rp['difference_percentage_points'] if phase=='R6' else rp['ablation_minus_C0_pp']
                insist(np.allclose(p['paired95_pp'],lo,rtol=0,atol=1e-11) and abs(p['delta_pp']-delta)<1e-11,'CI mismatch '+key)
                if phase=='R7':insist(set(rp['different_selection_units'])==set(p['different_units']) and set(rp['ablation_only_success'])==set(p['first_only_units']) and set(rp['full_C0_only_success'])==set(p['second_only_units']),'ablation paired subunits')
            if phase=='R6':
                sel=res['pairs']['C0-A'];r=ref['selection']['counts'];insist((sel['different'],sel['both_empty'],sel['same_nonempty'])==(r.get('different_candidates',0),r.get('both_abstain',0),r.get('same_candidate',0)),'selection count')
        if phase=='R6':
            br=ar.read('analysis/outcome_breakdown.json');insist(Counter(x['detail'] for x in rows)==br['adjudicated_detail'] and sum(x['physical_returns'] for x in rows)==br['returned_dynamics_steps'],'R6 route rollup')
        else:
            br=ar.read('analysis/failure_routes.json')
            for m in arms[1:]:
                p=archives[phase]['all']['pairs'][m+'-C0'];t=br['arms'][m]
                insist(set(p['first_only_units'])=={r['unit_id'] for r in t['ablation_only']} and set(p['second_only_units'])=={r['unit_id'] for r in t['full_C0_only']},'R7 failure paths')
                insist(all(g[r['unit_id']]['C0']['detail']==r['C0_failure'] for r in t['ablation_only']) and all(g[r['unit_id']][m]['detail']==r['ablation_failure'] for r in t['full_C0_only']),'R7 mismatch destinations')
    csvout(out/'trial_rows.csv',allrows)
    routes=[];tables=[];pairtable=[];selection=[]
    for phase,groups in archives.items():
        for stratum,q in groups.items():
            for m,a in q['arms'].items():
                tables.append(dict(phase=phase,stratum=stratum,method=m,assigned=a['n'],success=a['success'],rate_pct=100*a['rate'],wilson_low_pct=100*a['wilson95'][0],wilson_high_pct=100*a['wilson95'][1],empty=a['empty'],static_rejected=a['static_fail'],static_admitted=a['static_pass'],static_pass_all_pct=100*a['static_rate'],dynamics_entered=a['dynamic'],conditional_success_pct=None if a['conditional'] is None else 100*a['conditional'],native_returned_steps=a['native_returns']))
                for cat,c in a['detail'].items():routes.append(dict(phase=phase,stratum=stratum,method=m,detail=cat,n=c))
            for k,p in q['pairs'].items():
                pairtable.append(dict(phase=phase,stratum=stratum,contrast=k,assigned=p['n'],first_only=p['first_only'],second_only=p['second_only'],both_success=p['both_success'],both_failure=p['both_fail'],delta_pp=p['delta_pp'],CI_low_pp=p['paired95_pp'][0],CI_high_pp=p['paired95_pp'][1],exact_mcnemar_two_sided=p['exact_mcnemar'],discordant=p['discordant_n'],different_candidates=p['different'],different_pct=100*p['different_rate'],both_empty=p['both_empty'],same_nonempty=p['same_nonempty']))
                if stratum=='all':
                    for flag in ['first_only_units','second_only_units','different_units']:
                        for uid in p[flag]:
                            a,b=p['contrast'];ua=pairs_by_phase[phase][uid][a];ub=pairs_by_phase[phase][uid][b]
                            selection.append(dict(phase=phase,contrast=k,unit_id=uid,census=flag,first_trial=ua['trial_id'],second_trial=ub['trial_id'],first_candidate=ua['candidate'],second_candidate=ub['candidate'],first_outcome=ua['detail'],second_outcome=ub['detail']))
    csvout(out/'rates.csv',tables);csvout(out/'paired_effects.csv',pairtable);csvout(out/'failure_counts.csv',routes);csvout(out/'discordance_units.csv',selection)
    save(out/'statistics.json',archives)
    for ar in (ar6,ar7):
        evidence.extend(dict(version=ar.name,relative_path=k,sha256=h,bytes=(ar.p/k).stat().st_size) for k,h in sorted(ar.reads.items()))
    with gzip.open(out/'source_files.csv.gz','xt',encoding='utf8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(evidence[0]));wr.writeheader();wr.writerows(evidence)
    proof=dict(source_versions={ar.name:dict(directory=ar.name,manifest_sha256=ar.manifest_sha,archive_path=ar.zip,archive_sha256=ar.zip_sha,checked_manifest_files=len(ar.reads)) for ar in (ar6,ar7)},historical_results_read=len(allrows),reference_links_only=80,new_method_trials=0,physics_steps=0,recomputed=True,existing_paired_outputs_reconciled=True,bootstrap_seed=1262275604,bootstrap_draws=20000,exact_test='two-sided binomial on discordant matched units; unadjusted supplementary analysis')
    save(out/'audit.json',proof);print(json.dumps(proof))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();run(a.root.resolve(),a.output.resolve())
