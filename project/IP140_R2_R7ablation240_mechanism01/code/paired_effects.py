"""Pre-outcome paired inference. Cluster bootstrap resamples whole repeats within object/position."""
from collections import Counter,defaultdict
from itertools import combinations
import numpy as np
from freeze_sample import ARMS
from common import require
METHODS=('C0',)+ARMS

def compute(units, filters, seed, reps):
    chosen=[u for u in units if all(u[k]==v for k,v in filters.items())]
    n=len(chosen);blocks=defaultdict(lambda:defaultdict(list))
    for u in chosen:
        blocks[u['object_id'],u['position']][u['scene']].append(
            [int(u['methods'][m]['grasp_success']) for m in METHODS])
    require(n>0 and reps==20000,'empty stratum or bootstrap change')
    rng=np.random.default_rng(seed);draws=np.zeros((reps,len(METHODS)),dtype=np.int64)
    for key in sorted(blocks):
        clusters=blocks[key]
        require(len({len(x) for x in clusters.values()})==1,'unequal condition coverage')
        arr=np.array([np.sum(clusters[k],axis=0) for k in sorted(clusters)])
        k=len(arr);draws+=arr[rng.integers(0,k,size=(reps,k))].sum(axis=1)
    out=dict(assigned=n,base_scenes=sum(len(b) for b in blocks.values()),methods={},paired_vs_full_C0={},unique_among_all_four={})
    for i,m in enumerate(METHODS):
        rows=[u['methods'][m] for u in chosen];success=sum(r['grasp_success'] for r in rows)
        dyn=sum(r['physical_returns']>0 for r in rows);stat=sum(r['static_admitted'] for r in rows)
        selected=sum(r['selected_candidate_id'] is not None for r in rows)
        out['methods'][m]=dict(attempts=n,success=success,rate=success/n,
            interval95_cluster=(np.quantile(draws[:,i],[.025,.975])/n).tolist(),
            selected=selected,static_admitted=stat,static_admission_rate_all=stat/n,static_admission_rate_if_selected=stat/selected if selected else None,
            dynamics_entered=dyn,successful_if_dynamic=success,conditional_success_rate=success/dyn if dyn else None,
            outcomes=dict(Counter(r['detail_class'] for r in rows)))
        others=[w for w in METHODS if w!=m]
        out['unique_among_all_four'][m]=dict(
            success=[u['unit_id'] for u in chosen if u['methods'][m]['grasp_success'] and all(not u['methods'][w]['grasp_success'] for w in others)],
            failure=[u['unit_id'] for u in chosen if not u['methods'][m]['grasp_success'] and all(u['methods'][w]['grasp_success'] for w in others)])
        if i==0:continue
        gained=[];lost=[];changed=[];selected_both=0;both_abstained=0;both_success=0
        for u in chosen:
            a=u['methods'][m];c=u['methods']['C0'];uid=u['unit_id']
            if a['grasp_success'] and not c['grasp_success']:gained.append(uid)
            if c['grasp_success'] and not a['grasp_success']:lost.append(uid)
            both_success+=a['grasp_success'] and c['grasp_success']
            selected_both+=a['selected_candidate_id'] is not None and c['selected_candidate_id'] is not None
            both_abstained+=a['selected_candidate_id'] is None and c['selected_candidate_id'] is None
            if a['selected_candidate_id']!=c['selected_candidate_id']:changed.append(uid)
        out['paired_vs_full_C0'][m]=dict(ablation_only_success=gained,full_C0_only_success=lost,
            both_success=both_success,both_failure=n-both_success-len(gained)-len(lost),ablation_minus_C0_pp=100*(len(gained)-len(lost))/n,
            interval95_paired_pp=(100*np.quantile(draws[:,i]-draws[:,0],[.025,.975])/n).tolist(),
            different_selection_units=changed,selection_divergence_rate_all=len(changed)/n,both_selected=selected_both,both_abstained=both_abstained)
    return out
