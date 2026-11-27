"""Read immutable receipts. No estimator consumes historical trial rows."""
import json,math,datetime
from collections import Counter,defaultdict
import numpy as np
from common import *
def ci_wilson(k,n):
    z=1.959963984540054;v=k/n;den=1+z*z/n
    mid=(v+z*z/(2*n))/den;half=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
    return [max(0.,mid-half),min(1.,mid+half)]
def category(r,static):
    c=r['failure_category']
    if c is None:return 'success'
    if c=='no_executable_candidate':return 'empty_old_E'
    if c=='no_eligible_candidate':return 'empty_shared_pool'
    if c!='path_failure':return c
    if not static:return 'static_other'
    if static.get('exception')=="ValueError('both_registered_entries_required_before_lift')":return 'static_missing_bilateral_entry'
    why='|'.join(static.get('violations',[]));tags=[]
    for name,words in [('registered_surface',('registered_surface','registered_face','inner_face','witness_face')),('forbidden_collision',('nonpad_target_collision','robot_table_collision','unregistered_self_collision','robot_target_collision')),('early_target_contact',('before_close','before_legal_entry')),('penetration',('penetration',))]:
        if any(w in why for w in words):tags.append(name)
    return 'static_'+('+'.join(tags) if tags else 'other')
def cluster_summary(units,criteria,seed,reps):
    filtered=[u for u in units if all(u[k]==v for k,v in criteria.items())];M=['Raw','A','C0'];n=len(filtered)
    blocks=defaultdict(lambda:defaultdict(list))
    for u in filtered:blocks[(u['object_id'],u['position'])][u['repeat']].append([int(u['methods'][m]['grasp_success']) for m in M])
    rng=np.random.default_rng(seed);draws=np.zeros((reps,3),dtype=float)
    for key,rows in sorted(blocks.items()):
        require(sorted(rows)==list(range(1,11)),'incomplete physical repeat block')
        group=np.asarray([np.sum(rows[i],axis=0) for i in range(1,11)])
        draws+=group[rng.integers(0,10,size=(reps,10))].sum(axis=1)
    draws/=n
    rates={};paired={};select=[]
    for i,m in enumerate(M):
        k=sum(u['methods'][m]['grasp_success'] for u in filtered)
        rates[m]=dict(success=k,assigned=n,rate=k/n,wilson95=ci_wilson(k,n),cluster_percentile95=np.quantile(draws[:,i],[.025,.975]).tolist(),failure_categories=dict(Counter(u['methods'][m]['failure_category'] or 'success' for u in filtered)))
    for hi,lo in [('C0','A'),('A','Raw'),('C0','Raw')]:
        hi_only=sum(u['methods'][hi]['grasp_success'] and not u['methods'][lo]['grasp_success'] for u in filtered)
        lo_only=sum(u['methods'][lo]['grasp_success'] and not u['methods'][hi]['grasp_success'] for u in filtered)
        both=sum(u['methods'][hi]['grasp_success'] and u['methods'][lo]['grasp_success'] for u in filtered)
        paired[hi+'-'+lo]=dict(higher_label=hi,reference_label=lo,unit_count=n,hi_only=hi_only,lo_only=lo_only,both_success=both,both_failure=n-both-hi_only-lo_only,discordant=lo_only+hi_only,difference=(hi_only-lo_only)/n,difference_percentage_points=100*(hi_only-lo_only)/n,cluster_CI95_percentage_points=(100*np.quantile(draws[:,M.index(hi)]-draws[:,M.index(lo)],[.025,.975])).tolist())
    for u in filtered:
        a=u['methods']['A']['selected_candidate_id'];c=u['methods']['C0']['selected_candidate_id']
        select.append('both_abstain' if a is None and c is None else 'one_abstains' if a is None or c is None else 'same_candidate' if a==c else 'different_candidates')
    cnt=Counter(select)
    return dict(assigned_units=n,rates=rates,paired=paired,selection=dict(counts=dict(cnt),distinct_executing_candidate_rate_all_units=cnt['different_candidates']/n,any_distinct_choice_rate_all_units=(cnt['different_candidates']+cnt['one_abstains'])/n,distinct_rate_conditional_both_selected=cnt['different_candidates']/(cnt['same_candidate']+cnt['different_candidates']) if cnt['same_candidate']+cnt['different_candidates'] else None))
def elapsed(values):
    if not values:return {'n':0,'total_seconds':0,'mean_seconds':None}
    return dict(n=len(values),total_seconds=float(sum(values)),mean_seconds=float(np.mean(values)),median_seconds=float(np.median(values)),p95_seconds=float(np.quantile(values,.95)))
def main():
    d=read(HERE/'preregistration/design.json');ledger=[];detail=[];timing=defaultdict(list);dataset={u['unit_id']:{**u,'methods':{}} for u in d['units']}
    for t in d['trials']:
        k=t['trial_id'];dest=HERE/'trials'/k;job=HERE/'run/jobs/method'/k
        rfile=dest/'method_result.json';receipt=job/'return.json';i=dest/'intent.json';intent=read(i) if i.exists() else {}
        r=read(rfile) if rfile.exists() else {};known=receipt.is_file() and rfile.is_file() and not r['shared_fault'];dispatched=(job/'dispatch.json').is_file()
        status='adjudicated' if known else 'unknown_or_shared_failure' if dispatched else 'unstarted'
        if known:
            proof=read(receipt);require(proof['method_result_sha256']==sha(rfile) and proof['intent_sha256']==sha(i) and r['trial_id']==k,'analysis identity')
            require(r['physical_steps']==r['physical_returns'] and type(r['grasp_success']) is bool,'uninterpretable native result')
        sr=read(dest/'static_preflight/path_result.json') if (dest/'static_preflight/path_result.json').is_file() else {}
        ep=read(dest/'episode/result.json') if (dest/'episode/result.json').is_file() else {}
        row=dict(trial_id=k,unit_id=t['unit_id'],method=t['method'],registered_order=t['order'],status=status,worker_seed=t['worker_seed'],intent_sha256=sha(i) if i.exists() else None,result_sha256=sha(rfile) if rfile.exists() else None,worker_receipt_sha256=sha(receipt) if receipt.exists() else None,selected_candidate_id=intent.get('selected_candidate_id'),selected_native_index=(intent.get('selected_p3') or {}).get('native_source_index'),old_E_count=intent.get('old_E_count'),eligible_pool_size=intent.get('common_pool_count'),observation_sha256=intent.get('observation_sha256'),pool_sha256=intent.get('pool_sha256'),ranking_sha256=intent.get('ranking_sha256'),stable_state_sha256=intent.get('stable_state_sha256'),trial_completed=r.get('trial_completed') if known else None,engineering_gate_passed=r.get('engineering_gate_passed') if known else None,grasp_success=r.get('grasp_success') if known else None,negative_control_passed=r.get('negative_control_passed'),failure_category=r.get('failure_category'),detail_class=category(r,sr) if known else None,failure_detail=r.get('failure_detail'),physical_step_calls=r.get('physical_steps'),physical_returns=r.get('physical_returns'),forbidden_collision=r.get('forbidden_collision'),static_snapshots=sr.get('snapshots_completed'),static_first_failure=sr.get('first_failure'),engineering_wall_s=read(receipt)['elapsed_sec'] if known else None,max_lift_m=ep.get('max_lift_m'),final_hold_streak=ep.get('final_hold_streak'),slipped=ep.get('slipped'),terminal_counts=ep.get('terminal_observation_counts'))
        if ep and known:
            require(ep['grasp_success']==r['grasp_success'] and ep['telemetry_sha256']==sha(dest/'episode/telemetry.jsonl.gz'),'native evidence')
            if ep['grasp_success']:
                w=read(dest/'episode/final_480_steps.json')['observed_rows']
                require(len(w)==480 and [x['step'] for x in w]==list(range(4249,4729)) and all(x['lift_m']>=.1 and x['effective_left_target_contact'] and x['effective_right_target_contact'] and not x['aggregate_table_support'] and not x['slipped'] and not x['engineering_violations'] for x in w),'success not supported by observed 480')
        ledger.append(row);dataset[t['unit_id']]['methods'][t['method']]=row
        if known:
            for field in ['static_sec','execution_sec']:
                if r.get(field) is not None:timing[t['method']+'.'+field].append(r[field])
    known=sum(r['status']=='adjudicated' for r in ledger);unknown=sum(r['status']=='unknown_or_shared_failure' for r in ledger)
    for u in d['units']:
        base=HERE/'units'/u['unit_id']
        for f,keys in [('network.json',['segmentation_sec','candidate_generation_sec']),('orders.json',['pool_build_sec','scoring_sec'])]:
            if not (base/f).exists():continue
            record=read(base/f)
            for key in keys:timing['shared.'+key].append(record[key])
        if (base/'selection.json').exists():timing['shared.old_E_planning_sec'].append(sum(x['planning_seconds'] for x in read(base/'selection.json')['eligibility']))
    for p in (HERE/'scenes').glob('*/receipt.json'):timing['shared.scene_settle_acquisition_sec'].append(read(p)['wall_seconds'])
    put(HERE/'analysis/development_exclusion.json',dict(formal_denominator_only=True,included_R4_trials=0,included_R5_trials=0,history=d['history'],protocol_deviation=d['prior_protocol_deviation']))
    put(HERE/'analysis/formal_ledger.json',dict(assigned=720,adjudicated=known,unknown_or_shared_failure=unknown,not_dispatched=720-known-unknown,rows=ledger))
    put(HERE/'analysis/unit_pairs.json',dict(method_order=['Raw','A','C0'],units=list(dataset.values())))
    outcome=Counter(r['detail_class'] for r in ledger if r['status']=='adjudicated');native=[r for r in ledger if r['status']=='adjudicated' and r['physical_returns']]
    put(HERE/'analysis/outcome_breakdown.json',dict(adjudicated_detail=dict(outcome),by_method={m:dict(Counter(r['detail_class'] for r in ledger if r['method']==m and r['status']=='adjudicated')) for m in ('Raw','A','C0')},dynamics_entered=len(native),returned_dynamics_steps=sum(r['physical_returns'] for r in native),counts_by_status=dict(Counter(r['status'] for r in ledger)),unknown_physics_not_assumed_zero=True))
    put(HERE/'analysis/timing.json',{k:elapsed(v) for k,v in timing.items()})
    result=dict(complete=known==720,adjudicated=known,unknown_or_shared_failure=unknown,assigned_per_method=240,stats_spec=d['statistics'],uncertainty='Wilson intervals are marginal descriptive summaries. Bootstrap resamples physical repeats jointly across both conditions; intervals with no observed discordance may collapse and do not prove equivalence.')
    if known==720:
        allunits=list(dataset.values());require(all(set(u['methods'])=={'Raw','A','C0'} and len({v['pool_sha256'] for v in u['methods'].values()})==len({v['stable_state_sha256'] for v in u['methods'].values()})==1 for u in allunits),'pairing integrity')
        result['overall']=cluster_summary(allunits,{},d['bootstrap_seed'],d['bootstrap_reps'])
        result['strata']={k:cluster_summary(allunits,crit,d['bootstrap_seed'],d['bootstrap_reps']) for k,crit in [(v,{'condition':v}) for v in ('clean','joint')]+[(v,{'object_id':v}) for v in ('014','017','021')]}
    else:
        result['withheld_reason']='Incomplete formal method outcomes. Unstarted and unknown trials are not scored as zero.'
        result['observed_success_counts_only']={m:sum(r['grasp_success'] is True for r in ledger if r['method']==m) for m in ('Raw','A','C0')}
    put(HERE/'analysis/paired_statistics.json',result)
if __name__=='__main__':main()
