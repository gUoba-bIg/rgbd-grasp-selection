"""One fixed selective P6 correspondence development test; zero new fit/global calls."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[_k]='2'
import sys
sys.dont_write_bytecode=True
import argparse
import inspect
import json
import time
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import correspondence_v40 as old
from src import correspondence_v42 as core
from scripts import p6_correspondence_v40 as base
from scripts.pipeline import project_lock

LOG=ROOT/'logs/P6/selective_v42'
OUT=ROOT/'results/P6/selective_v42'
OLD_LOG=ROOT/'logs/P6/correspondence_v40'
OLD_OUT=ROOT/'results/P6/correspondence_v40'
SCENES=base.SCENES
LIMITS=dict(seconds=1800,rss_bytes=2*1024**3,output_bytes=256*1024**2,
    local_alignment_generate=720,local_alignment_review=720,ridge_fit=0,neural_forward=0,ap=0)
INHERITED=['src/correspondence_v42.py','scripts/p6_selective_v42.py','src/correspondence_v40.py','scripts/p6_correspondence_v40.py','scripts/pipeline.py']
read,sha,now,rel,identity=base.read,base.sha,base.now,base.rel,base.identity
once,replace_json,append=base.once,base.replace_json,base.append
runtime_identity,peak_rss=base.runtime_identity,base.peak_rss
array_sha,close_numeric=base.array_sha,base.close_numeric


def audit_ledger():
    starts={}; returns={}; keys=set(); path=LOG/'calls.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e['event']=='started':
                old.need(e['id'] not in starts and e['key'] not in keys,'Duplicate research call')
                starts[e['id']]=e; keys.add(e['key'])
            else:
                old.need(e['event']=='returned' and e['id'] in starts and e['id'] not in returns,'Unpaired research return')
                s=starts[e['id']]; old.need(s['key']==e['key'] and s['kind']==e['kind'],'Call identity differs')
                returns[e['id']]=e
    kinds=['local_alignment_generate','local_alignment_review','ridge_fit','neural_forward','ap']
    old.need(all(s['kind'] in kinds for s in starts.values()),'Undeclared research call kind')
    return dict(counts={k:sum(e['kind']==k for e in returns.values()) for k in kinds},
        started_counts={k:sum(e['kind']==k for e in starts.values()) for k in kinds},
        unknown_calls=[e for i,e in starts.items() if i not in returns],returned_by_key={e['key']:e['result'] for e in returns.values()})


class Budget:
    def __init__(self,stage):
        self.stage=stage; self.started=time.perf_counter(); self.last=-1e9
        previous=read(LOG/'usage.json') if (LOG/'usage.json').exists() else {}
        self.prior=float(previous.get('seconds',0)); self.maxrss=int(previous.get('peak_rss_bytes',0))
        self.files=set(p for folder in (LOG,OUT) if folder.exists() for p in folder.rglob('*') if p.is_file())
        self.ledger=audit_ledger(); old.need(not self.ledger['unknown_calls'],'Unknown prior call retained; no automatic retry')
        self.sample(True)
    def register(self,p): self.files.add(p)
    def sample(self,force=False):
        elapsed=time.perf_counter()-self.started
        if not force and elapsed-self.last<2: return
        self.last=elapsed; self.maxrss=max(self.maxrss,peak_rss())
        total=sum(p.stat().st_size for p in self.files if p.exists())
        value=dict(seconds=self.prior+elapsed,peak_rss_bytes=self.maxrss,registered_output_bytes=total,stage=self.stage,at=now(),
            counts=self.ledger['counts'],unknown_calls=self.ledger['unknown_calls'],sample_interval_seconds=2)
        replace_json(LOG/'usage.json',value); self.register(LOG/'usage.json')
        old.need(value['seconds']<=LIMITS['seconds'],'Cumulative component wall time exceeded')
        old.need(self.maxrss<=LIMITS['rss_bytes'] and total<=LIMITS['output_bytes'],'RSS/output budget exceeded')
    def invoke(self,kind,key,fn):
        self.sample()
        if key in self.ledger['returned_by_key']: return self.ledger['returned_by_key'][key]
        old.need(self.ledger['started_counts'][kind]<LIMITS[kind],'Research call cap: '+kind)
        cid=uuid.uuid4().hex; event=dict(event='started',id=cid,key=key,kind=kind,at=now())
        append(LOG/'calls.jsonl',event); self.register(LOG/'calls.jsonl')
        self.ledger['started_counts'][kind]+=1; self.ledger['unknown_calls'].append(event)
        value=fn()
        append(LOG/'calls.jsonl',dict(event='returned',id=cid,key=key,kind=kind,at=now(),result=value))
        self.ledger['unknown_calls'].remove(event); self.ledger['counts'][kind]+=1; self.ledger['returned_by_key'][key]=value
        self.sample(); return value
    def close(self): self.sample(True)


def parent_data():
    data=read(OLD_LOG/'data_audit.json'); generated=read(OLD_OUT/'generated.json'); model=read(OLD_OUT/'model.json')
    review=read(OLD_LOG/'review.json'); cold=read(OLD_LOG/'cold.json')
    old.need(data['passed'] and data['scenes']==SCENES and review['passed'] and cold['passed'],'Parent data/review incomplete')
    old.need(model['generated_sha256']==sha(OLD_OUT/'generated.json'),'Parent model source changed')
    base.verify_identities(review['reviewed_outputs'])
    rows=generated['rows']; indexed={r['key']:r for r in rows}
    old.need(len(rows)==len(indexed)==720,'Parent global cache incomplete/duplicated')
    return data,indexed,model


def prepare():
    data,_,_=parent_data(); base.verify_identities(data['protected'])
    source=INHERITED+[rel(OLD_LOG/p) for p in ('data_audit.json','proposal.json','review.json','cold.json')]+[rel(OLD_OUT/p) for p in ('generated.json','model.json','result.json')]
    p=dict(version='v42',component='P6_selective_correspondence',created=now(),runtime=runtime_identity(),inherited_sources=INHERITED,
        protected={k:identity(ROOT/k) for k in source},scenes=SCENES,records=data['records'],warps=[list(x) for x in old.WARPS],
        cases=dict(fit=432,calibration=144,evaluation=144),limits=LIMITS,
        global_cache=rel(OLD_OUT/'generated.json'),frozen_affine_map=rel(OLD_OUT/'model.json'),
        scope='One exposed-development selective action/identifiability candidate. Does not disambiguate native bias from added warp in general; never full P6/uvz/grasp.',
        exposure='All20 scenes trained upstream, all old development exposed; no independent confirmation; no evaluation-set parameter selection.',
        local=dict(grid_columns=3,grid_rows=2,assignment='floor(u*3/width)+3*floor(v*2/height)',
            sampling='The same fixed observable8px interior and3x3 valid erosion, global deterministic <=4096 depth edges partitioned into6 cells; no per-cell extra sampling.',
            min_edges_per_cell=32,unique_peak_gap_min=1e-12,candidate_count_per_cell=81,candidate_uv=[-4,4],
            score='v40 weighted absolute directional gradient dot product; fixed cell denominator; out-of-support hypothesis samples zero.',
            affine='Reuse v40 fitted six-parameter global map on local raw integer peaks; then clip[-4,4]/rint half-to-even. New local feature distribution, not preserved local calibration.'),
        decision=dict(zero_vote='usable local corrected integer uv equals[0,0] exactly',
            move_vote='usable local corrected uv nonzero, global corrected uv nonzero, Chebyshev distance(local,global)<=1',
            required_votes=4,required_distinct_rows=2,required_distinct_columns=2,
            coverage='Across all supporting cells;3+1 row split allowed; not2 columns within each row.',
            supported_zero='Only Z support passes => action[0,0],usable true; prediction, not ground truth.',
            supported_move='Only M support passes => old global corrected action,usable true.',
            unknown='No support or unusable global => unknown_abstain,action[0,0],usable false; no_rejected gate fails.',
            disjoint='Z/M supporting cells are disjoint, so both cannot obtain4of6; assert.',
            no_reference='No source-id/GT/condition/scene/clean-reference passed to estimator or decision.'),
        fitting=dict(new_fit=0,new_calibration=0,threshold_search=0,global_estimators_recomputed=0,
            fit_split_role='Descriptive same fixed prediction only; old global fit remains untouched.',
            calibration_split_role='Descriptive fixed prediction only; not threshold fitting or model selection.'),
        evaluation=dict(parent_gate_outcomes=read(OLD_OUT/'result.json')['gates'],gate_values_source='v40 proposal unchanged numerical thresholds, NOT parent pass/fail selection',
            fixed_thresholds=read(OLD_LOG/'proposal.json')['gates'],same_original_main_and_boundary_pixels=True,
            unknown='All retained; action error includes zero fallback but unknown remains rejected and identity successes=0.',
            unchanged_geometry='Original files/XYZ untouched; only separate hypothetical action XYZ for error measurement.',
            full_uvz=False),
        synthetic_preparation=dict(max_check_cases=12,max_estimator_invocations=24,limit_scope='per_self_test_invocation',real_data_calls=0,
            planned_check_cases=11,planned_primary=11,planned_independent=11,planned_total=22,logical_decision_table_separate=True),
        stop_rule='Generate720local+review720local, zero fit/global/NN/AP; retain all failures, report and stop. No search, expansion, P7/P10.',
        stages=['generate','evaluate','review','cold'])
    path=LOG/'proposal.json'
    if path.exists():
        previous=read(path); p['created']=previous['created']; old.need(p==previous,'Changed proposal; archive old explicitly before revision')
    else: once(path,p)
    return dict(passed=True,proposal_sha256=sha(path))


def authorize():
    p=read(LOG/'proposal.json'); a=read(LOG/'authorization.json')
    old.need(a.get('frozen') is True and a['proposal_sha256']==sha(LOG/'proposal.json'),'Missing root freeze/changed proposal')
    old.need(p['runtime']==runtime_identity() and p['limits']==LIMITS,'Runtime/budget drift')
    base.verify_identities(a['protected']); base.verify_identities(p['protected'])
    data,indexed,model=parent_data(); base.verify_identities(data['protected'])
    return data,indexed,model


def pack(value):
    answer={k:v for k,v in value.items() if k!='edge_indices'}
    answer['edge_indices_sha256']=array_sha(value['edge_indices'])
    return answer


def global_action(parent,model):
    _,action=old.predict(model,np.array([parent['estimate']['raw_uv']]),np.array([parent['estimate']['usable']]))
    return action[0]


def decision_counts(rows):
    names=('supported_zero','supported_move','unknown_abstain')
    return {name:sum(r['local']['decision']['status']==name for r in rows) for name in names}


def generate(data,indexed,model,budget):
    rows=[]
    for split,records in data['records'].items():
        for r in records:
            frame=base.load_frame(data,r)
            for delta in old.WARPS:
                key=base.key_for(split,r,delta); parent=indexed[key]; global_uv=global_action(parent,model)
                z,valid,sid=old.warp_integer(frame['depth'],frame['valid'],delta)
                value=budget.invoke('local_alignment_generate','generate/'+key,
                    lambda:pack(core.estimate_local(frame['rgb'],z,valid,model,global_uv,parent['estimate']['usable'])))
                old.need(value['edge_indices_sha256']==parent['estimate']['edge_indices_sha256'],'Global observable sampling differs from parent')
                old.need(array_sha(sid)==parent['source_id_sha256'],'Parent source-id mismatch')
                rows.append(dict(key=key,split=split,scene=r['scene'],frame=r['frame'],warp_uv=list(delta),local=value,
                    source_id_sha256=array_sha(sid),lost_source_valid_count=int(frame['valid'].sum()-valid.sum())))
            budget.sample()
    old.need(len(rows)==720,'Local generated scope differs')
    result=dict(version='v42',rows=rows,proposal_sha256=sha(LOG/'proposal.json'),parent_generated_sha256=sha(OLD_OUT/'generated.json'),
        parent_model_sha256=sha(OLD_OUT/'model.json'),decision_counts=decision_counts(rows),
        decision_counts_by_split={s:decision_counts([r for r in rows if r['split']==s]) for s in SCENES},new_fit=0,new_global_estimates=0)
    target=OUT/'generated.json'
    if target.exists(): old.need(read(target)==result,'Generated cache differs')
    else: once(target,result)
    budget.register(target)


def generated():
    g=read(OUT/'generated.json')
    old.need(g['proposal_sha256']==sha(LOG/'proposal.json') and g['parent_generated_sha256']==sha(OLD_OUT/'generated.json') and g['parent_model_sha256']==sha(OLD_OUT/'model.json'),'Generated provenance differs')
    old.need(len(g['rows'])==len({r['key'] for r in g['rows']})==720,'Generated case identity differs')
    return g


def evaluate(data,indexed,model,budget):
    g=generated(); eval_rows={r['key']:r for r in g['rows'] if r['split']=='evaluation'}
    rows=[]; denominators=[]
    for r in data['records']['evaluation']:
        frame=base.load_frame(data,r,labels=True)
        ref=old.select_reference(frame['depth'],frame['valid'],frame['eligible'],frame['labels'])
        denominators.append(dict(scene=r['scene'],frame=r['frame'],selected_points=len(ref['indices']),main_points=int(ref['main'].sum()),
            boundary_points=int(ref['boundary'].sum()),fixed_domain_count=ref['fixed_domain_count'],
            excluded_valid_from_fixed8px_border=ref['excluded_valid_from_fixed8px_border'],selected_indices_sha256=array_sha(ref['indices'])))
        for delta in old.WARPS:
            key=base.key_for('evaluation',r,delta); item=eval_rows[key]; d=item['local']['decision']; action=np.array(d['action_uv'])
            _,_,sid=old.warp_integer(frame['depth'],frame['valid'],delta)
            metric=old.evaluate_case(frame,ref,delta,action,d['usable'],sid)
            rows.append(dict(scene=r['scene'],frame=r['frame'],key=key,warp_uv=list(delta),applied_uv=action.tolist(),decision_status=d['status'],
                unknown_reason=d['unknown_reason'],lost_source_valid_count=item['lost_source_valid_count'],**metric))
        budget.sample()
    summary,gates=old.summarize(rows)
    summary['decision_counts']=decision_counts(list(eval_rows.values()))
    summary['decision_counts_by_split']=g['decision_counts_by_split']
    summary['unknown_reasons']={reason:sum(r['unknown_reason']==reason for r in rows) for reason in sorted({r['unknown_reason'] for r in rows})}
    supported=all(gates.values())
    result=dict(version='v42',engineering_computation_completed=True,engineering_passed=False,independent_review_pending=True,
        identifiability_supported=supported,scientific_usable=supported,P6_full_passed=False,uvz_calibrated=False,P7_started=False,
        summary=summary,gates=gates,evaluation_rows=rows,frame_denominators=denominators,
        scientific_scope='Exposed-development selective correspondence action only. Global-vs-native-bias decomposition not guaranteed; no fulluvz, probability calibration, P6 or grasp claims.',
        error_definition='Error of applied action, including zero for abstentions. An abstention is never a correct-zero identification; rejected count/gate and zero identity-success counts retained.',
        parent_v40_summary=read(OLD_OUT/'result.json')['summary'],generated_sha256=sha(OUT/'generated.json'),
        parent_model_sha256=sha(OLD_OUT/'model.json'),proposal_sha256=sha(LOG/'proposal.json'),new_fit=0,new_global_estimates=0,
        next_action='Stop after review/cold. No automatic change of threshold/model or next stage.',
        progress=dict(all_tasks_percent=45,experiments_percent=56.25,changed=False))
    target=OUT/'result.json'
    if target.exists(): old.need(read(target)==result,'Result differs on reentry')
    else: once(target,result)
    budget.register(target)


def independent_decide(local,valid,global_uv,global_valid):
    zero=[]; move=[]
    for c in range(6):
        if not valid[c]: continue
        if list(local[c])==[0,0]: zero.append(c)
        elif any(global_uv) and max(abs(int(local[c][j])-int(global_uv[j])) for j in range(2))<=1: move.append(c)
    def sufficient(cells):
        return len(cells)>=4 and {c//3 for c in cells}=={0,1} and len({c%3 for c in cells})>=2
    z=bool(global_valid and sufficient(zero)); m=bool(global_valid and sufficient(move))
    old.need(not set(zero)&set(move) and not(z and m),'Independent support disjointness')
    n=sum(bool(x) for x in valid)
    if z: status='supported_zero'; action=[0,0]; ok=True; reason='none'
    elif m: status='supported_move'; action=[int(x) for x in global_uv]; ok=True; reason='none'
    else:
        status='unknown_abstain'; action=[0,0]; ok=False
        reason='global_unusable' if not global_valid else ('fewer_than_four_usable_cells' if n<4 else 'no_four_cell_consensus')
    return dict(status=status,action_uv=action,usable=ok,unknown_reason=reason,zero_support_cells=zero,move_support_cells=move,
        zero_supported=z,move_supported=m,valid_cells=n,original_xyz_overwritten=False,zero_status_is_prediction_not_truth=True)


def independent_local(rgb,z,valid,model,global_uv,global_valid):
    gray=.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2]
    rg=np.gradient(gray); dg=np.gradient(z.astype(float))
    rm=np.sqrt(rg[0]**2+rg[1]**2); dm=np.sqrt(dg[0]**2+dg[1]**2)
    support=np.zeros(z.shape,bool)
    support[1:-1,1:-1]=np.logical_and.reduce([valid[dy:valid.shape[0]-2+dy,dx:valid.shape[1]-2+dx] for dy in range(3) for dx in range(3)])
    support[:8]=False; support[-8:]=False; support[:,:8]=False; support[:,-8:]=False
    allidx=np.flatnonzero(support&(dm>.01)); n=min(4096,len(allidx)); idx=allidx[np.arange(n)*len(allidx)//n] if n else np.empty(0,np.int64)
    h,w=z.shape; yy,xx=np.divmod(idx,w)
    cells=np.floor(xx*3/w).astype(int)+3*np.floor(yy*2/h).astype(int)
    candidates=sorted([(x,y) for y in range(-4,5) for x in range(-4,5)],key=lambda t:(t[0]**2+t[1]**2,t[1],t[0]))
    rows=[]; corrected=[]; usability=[]; B=np.array(model['coefficients'])
    for cell in range(6):
        sy=yy[cells==cell]; sx=xx[cells==cell]; scores=[]
        for dx,dy in candidates:
            ty=sy-dy; tx=sx-dx; inside=(ty>=0)&(ty<h)&(tx>=0)&(tx<w)
            values=np.zeros(len(sx),float)
            if inside.any():
                a,b=sy[inside],sx[inside]; c,d=ty[inside],tx[inside]
                values[inside]=np.abs(dg[0][a,b]*rg[0][c,d]+dg[1][a,b]*rg[1][c,d])/((dm[a,b]+.02)*(rm[c,d]+.02))
            scores.append(float(values.sum()/max(1,len(values))))
        ordered=sorted(scores); gap=ordered[-1]-ordered[-2]; best=int(np.argmax(scores)); ok=len(sx)>=32 and gap>1e-12
        raw=list(candidates[best]) if ok else [0,0]
        continuous=np.array(raw+[1],float)@B
        action=np.around(np.maximum(-4,np.minimum(4,continuous))).astype(int) if ok else np.array([0,0])
        rows.append(dict(cell=cell,row=cell//3,column=cell%3,edge_count=len(sx),scores=scores,raw_uv=raw,usable=bool(ok),peak_margin=gap,
            best_score=scores[best],rejection='none' if ok else ('fewer_than32_edges' if len(sx)<32 else 'nonunique_peak'),
            continuous_uv=continuous.tolist(),corrected_uv=action.tolist()))
        corrected.append(action.tolist()); usability.append(bool(ok))
    return dict(cells=rows,edge_count=len(idx),edge_indices_sha256=array_sha(idx),
        decision=independent_decide(corrected,usability,global_uv,global_valid),global_corrected_uv=[int(x) for x in global_uv],
        candidate_scores_written=486,nonempty_candidate_dot_products=81*sum(r['edge_count']>0 for r in rows))


def independent_counts(rows):
    counts=dict(supported_zero=0,supported_move=0,unknown_abstain=0)
    for row in rows: counts[row['local']['decision']['status']]+=1
    return counts


def review(data,indexed,model,budget):
    g=generated(); result=read(OUT/'result.json'); originals={r['key']:r for r in g['rows']}
    old.need(result['generated_sha256']==sha(OUT/'generated.json') and result['parent_model_sha256']==sha(OLD_OUT/'model.json') and result['proposal_sha256']==sha(LOG/'proposal.json'),'Result provenance differs')
    result_rows={r['key']:r for r in result['evaluation_rows']}; denominators={(r['scene'],r['frame']):r for r in result['frame_denominators']}
    reviewed=[]; reviewed_evaluation=[]; B=np.array(model['coefficients'])
    for split,records in data['records'].items():
        for r in records:
            frame=base.load_frame(data,r,labels=split=='evaluation')
            for delta in old.WARPS:
                key=base.key_for(split,r,delta); parent=indexed[key]
                continuous=np.array(parent['estimate']['raw_uv']+[1],float)@B
                global_uv=np.around(np.maximum(-4,np.minimum(4,continuous))).astype(int) if parent['estimate']['usable'] else np.array([0,0])
                z,valid,sid=base.independent_warp(frame['depth'],frame['valid'],delta)
                value=budget.invoke('local_alignment_review','review/'+key,lambda:independent_local(frame['rgb'],z,valid,model,global_uv,parent['estimate']['usable']))
                close_numeric(originals[key]['local'],value,key+'/local')
                old.need(value['edge_indices_sha256']==parent['estimate']['edge_indices_sha256'] and array_sha(sid)==originals[key]['source_id_sha256']==parent['source_id_sha256'],'Independent observable/source identities differ')
                item=dict(key=key,split=split,scene=r['scene'],frame=r['frame'],warp_uv=list(delta),local=value,source_id_sha256=array_sha(sid),
                    lost_source_valid_count=int(frame['valid'].sum()-valid.sum()))
                close_numeric(originals[key],item,key+'/cache'); reviewed.append(item)
                if split=='evaluation':
                    d=value['decision']; action=np.array(d['action_uv'])
                    metrics,den=base.independent_evaluation(frame,delta,action,d['usable'],sid)
                    row=dict(scene=r['scene'],frame=r['frame'],key=key,warp_uv=list(delta),applied_uv=action.tolist(),decision_status=d['status'],
                        unknown_reason=d['unknown_reason'],lost_source_valid_count=item['lost_source_valid_count'],**metrics)
                    close_numeric(result_rows[key],row,key+'/evaluation')
                    close_numeric(denominators[(r['scene'],r['frame'])],dict(scene=r['scene'],frame=r['frame'],**den),key+'/denominator')
                    reviewed_evaluation.append(row)
            budget.sample()
    by_split={s:independent_counts([r for r in reviewed if r['split']==s]) for s in SCENES}
    close_numeric(g['decision_counts'],independent_counts(reviewed)); close_numeric(g['decision_counts_by_split'],by_split)
    summary,gates=base.independent_summary(reviewed_evaluation)
    summary['decision_counts']=independent_counts([r for r in reviewed if r['split']=='evaluation']); summary['decision_counts_by_split']=by_split
    reasons={}
    for r in reviewed_evaluation: reasons[r['unknown_reason']]=reasons.get(r['unknown_reason'],0)+1
    summary['unknown_reasons']=reasons
    close_numeric(summary,result['summary']); close_numeric(gates,result['gates'])
    supported=all(gates.values())
    old.need(result['identifiability_supported']==supported and result['scientific_usable']==supported and result['P6_full_passed'] is False and result['uvz_calibrated'] is False,'Scientific conclusion differs')
    audit=audit_ledger(); old.need(audit['counts']=={k:LIMITS[k] for k in audit['counts']} and not audit['unknown_calls'],'Actual research call accounting incomplete')
    base.verify_identities(read(LOG/'authorization.json')['protected'])
    record=dict(passed=True,engineering_passed=True,identifiability_supported=supported,scientific_usable=supported,P6_full_passed=False,uvz_calibrated=False,
        reviewed_outputs={rel(OUT/n):identity(OUT/n) for n in ('generated.json','result.json')},counts=audit['counts'],unknown_calls=[],
        checks=['independent six-cell81-score replay720','frozen global cache and affine map reused;zero global estimators andfit',
            'disjoint exact-zero/nonzero support and explicit abstention','source-id/denominator/XYZ/GT-proxy replay144',
            'independent all-original gates and summary including unknown','all frozen inputs unchanged'],at=now())
    target=LOG/'review.json'
    if target.exists():
        previous=read(target); record['at']=previous['at']; old.need(previous==record,'Review changed')
    else: once(target,record)
    budget.register(target)


def cold(data,indexed,model,budget):
    reviewed=read(LOG/'review.json'); result=read(OUT/'result.json')
    old.need(reviewed['passed'] and reviewed['engineering_passed'],'Independent review did not pass')
    base.verify_identities(reviewed['reviewed_outputs'])
    audit=audit_ledger(); old.need(audit['counts']=={k:LIMITS[k] for k in audit['counts']} and not audit['unknown_calls'],'Cold research ledger differs')
    old.need(result['identifiability_supported']==reviewed['identifiability_supported'] and result['P6_full_passed'] is False and result['uvz_calibrated'] is False,'Cold scientific scope drift')
    record=dict(passed=True,engineering_passed=True,identifiability_supported=result['identifiability_supported'],scientific_usable=result['scientific_usable'],
        P6_full_passed=False,uvz_calibrated=False,review_sha256=sha(LOG/'review.json'),reviewed_outputs=reviewed['reviewed_outputs'],
        counts=audit['counts'],unknown_calls=[],new_research_calls=0,process_id=os.getpid(),at=now())
    target=LOG/'cold.json'
    if target.exists():
        previous=read(target); record['at']=previous['at']; record['process_id']=previous['process_id']; old.need(previous==record,'Cold marker changed')
    else: once(target,record)
    budget.register(target)


def self_test():
    rng=np.random.RandomState(42); h,w=96,112; raw=rng.uniform(.3,1.5,(h,w)); valid=np.ones((h,w),bool); valid[44:48,54:58]=False
    rgb=np.repeat(((raw-.3)/1.2)[:,:,None],3,axis=2)
    model=dict(coefficients=[[1.,0.],[0.,1.],[0.,0.]])
    tests=[]; primary=0; independent=0
    for delta in old.WARPS:
        z,ok,_=old.warp_integer(raw,valid,delta)
        a=pack(core.estimate_local(rgb,z,ok,model,np.array(delta),True)); primary+=1
        b=independent_local(rgb,z,ok,model,np.array(delta),True); independent+=1
        close_numeric(a,b,'synthetic/local')
        old.need(a['decision']['usable'] and a['decision']['action_uv']==list(delta),'Synthetic positive control correspondence failed')
    tests.append('nine-direction synthetic coedges, six-cell independent scores and decision')
    for kind in ('blank','padding'):
        z=np.ones((h,w)); ok=np.ones((h,w),bool)
        if kind=='padding': z[:,:3]=0; ok=z>0
        a=pack(core.estimate_local(rgb,z,ok,model,np.array([0,0]),True)); primary+=1
        b=independent_local(rgb,z,ok,model,np.array([0,0]),True); independent+=1
        close_numeric(a,b,'synthetic/'+kind)
        old.need(a['decision']['status']=='unknown_abstain' and not a['decision']['usable'],'No evidence disguised as identified zero')
    tests.append('blank/padding produce explicit unknown, no artificial border leak')
    logical=[
        ([[0,0]]*6,[True]*6,[3,0],True,'supported_zero'),
        ([[3,0]]*6,[True]*6,[3,0],True,'supported_move'),
        ([[0,0]]*3+[[3,0]]*3,[True]*6,[3,0],True,'unknown_abstain'),
        ([[0,0]]*6,[True]*3+[False]*3,[0,0],True,'unknown_abstain'),
        ([[0,0]]*6,[True]*6,[0,0],False,'unknown_abstain'),
        ([[1,0]]*6,[True]*6,[1,0],True,'supported_move'),
        ([[0,0]]*4+[[3,0]]*2,[True]*6,[3,0],True,'supported_zero'),
        ([[2,1]]*6,[True]*6,[0,0],True,'unknown_abstain')]
    for local,usable,global_uv,global_valid,status in logical:
        a=core.decide(local,usable,global_uv,global_valid); b=independent_decide(local,usable,global_uv,global_valid)
        close_numeric(a,b,'logical/decision'); old.need(a['status']==status,'Incorrect decision truth table')
    tests.append('exact-zero only;3+1 coverage allowed;disjoint3vs3 abstain;global unavailable;one-pixel nonzero preserved')
    # The old evaluator must retain unknown in denominators and fail its rejection gate.
    fake=[]
    for s in SCENES['evaluation']:
        for frame in ('0000','0064','0128','0192'):
            for d in old.WARPS:
                fake.append(dict(scene=s,frame=frame,warp_uv=list(d),euclidean_error_px=0.,no_correction_error_px=float(np.linalg.norm(d)),
                    usable=True,same_source_correspondence_exact=True,oracle_source_identity_exact=True,oracle_error_px=0.,
                    metrics=dict(boundary=dict(n=10,baseline_instance_identity_matches=5,estimated_instance_identity_matches=10))))
    fake[0]['usable']=False; fake[0]['same_source_correspondence_exact']=False; fake[0]['metrics']['boundary']['estimated_instance_identity_matches']=0
    aa,ag=old.summarize(fake); bb,bg=base.independent_summary(fake); close_numeric(aa,bb); close_numeric(ag,bg)
    old.need(not ag['no_rejected_case'] and aa['rejection_count']==1,'Rejection gate relaxed')
    tests.append('unknown kept in144-case denominators and original no-rejected gate fails')
    old.need(list(inspect.signature(core.estimate_local).parameters)==['rgb','observed_depth','observed_valid','model','global_uv','global_usable'],'Forbidden predictor feature')
    old.need(primary==independent==11 and primary+independent<=24,'Synthetic budget changed')
    result=dict(passed=True,checks=tests,synthetic_check_cases=11,synthetic_primary_estimator_calls=primary,synthetic_independent_estimator_calls=independent,
        synthetic_estimator_calls=primary+independent,logical_decision_cases=len(logical),synthetic_fit_calls=0,formal_research_calls=0,real_image_decodes=0,
        new_global_estimator_calls=0,neural_calls=0,positive_control_scope='Synthetic known global shift supplied only as test fixture, never a formal input.',
        runtime=runtime_identity(),source_sha256={k:sha(ROOT/k) for k in INHERITED},peak_rss_bytes=peak_rss(),at=now())
    target=LOG/'component_tests.json'
    if target.exists(): os.replace(target,target.with_name('component_tests.previous.'+uuid.uuid4().hex+'.json'))
    once(target,result)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','self-test','generate','evaluate','review','cold']); args=p.parse_args()
    old.need(Path.cwd().resolve()==ROOT,'Run explicitly from project ROOT')
    LOG.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); budget=None; success=False; answer={}
    try:
        with project_lock(ROOT):
            if args.stage in ('prepare','self-test'):
                answer=prepare() if args.stage=='prepare' else self_test(); success=True
                print(json.dumps(dict(stage=args.stage,passed=answer['passed'])))
            else:
                data,indexed,model=authorize(); budget=Budget(args.stage)
                globals()[args.stage](data,indexed,model,budget); budget.close(); success=True
                print(json.dumps(dict(stage=args.stage,completed=True,counts=budget.ledger['counts'])))
    finally:
        if args.stage in ('prepare','self-test'):
            path=LOG/'preparation_usage.json'; usage=read(path) if path.exists() else dict(seconds=0.,attempts=[])
            elapsed=time.perf_counter()-start; usage['seconds']+=elapsed
            usage['attempts'].append(dict(stage=args.stage,seconds=elapsed,passed=success,at=now(),peak_rss_bytes=peak_rss(),
                synthetic_estimator_calls=answer.get('synthetic_estimator_calls',0),logical_decision_cases=answer.get('logical_decision_cases',0)))
            replace_json(path,usage)
        elif budget is not None and not success: budget.close()


if __name__=='__main__': main()
