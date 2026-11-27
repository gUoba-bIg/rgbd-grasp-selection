"""One bounded engineering repair of v44's independent resize verifier.

The original run, failed verifier, old unknown operation and all old artifacts
remain unchanged. This entry point cannot run models, AP, fit, or new samples.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='2'
import sys,time,json,uuid,hashlib,argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts import p5_complementarity_v44 as old
from src import p5_complementarity_v44 as core
from src import p6_image_geometry_v24 as reference
LOG=ROOT/'logs/P5/complementarity_v44r1';OUT=ROOT/'results/P5/complementarity_v44r1'
AUTH=LOG/'authorization.json';PROPOSAL=LOG/'proposal.json'
read,sha,identity,relative,need,once,atomic=old.read,old.sha,old.identity,old.relative,old.need,old.once,old.atomic
FAILED_CASE='clean_scene_0002_0080'
RESOURCE_CAPS=dict(GT_decode=1600,depth_decode=2400,MAT_decode=2400,valid_replay=2400)
KINDS=('repair_review_case','GT_decode','depth_decode','MAT_decode','valid_replay')
_STATE=None
_STATE_SIGNATURE=None


def name(case):return f"{case['condition']}_scene_{case['scene']:04d}_{case['frame']}"
def dependency_paths():
    paths=set(old.dependency_paths()+[relative(old.AUTH),relative(old.MANIFEST),relative(old.LOG/'prepare.json'),
        relative(old.LOG/'calls.jsonl'),'scripts/p5_complementarity_review_v44r1.py',relative(PROPOSAL)])
    for directory in (old.OUT,old.LOG/'review_cases'):
        paths.update(relative(p) for p in directory.rglob('*') if p.is_file())
    paths.update(relative(p) for p in old.LOG.glob('usage_*.json'))
    paths.update(relative(p) for p in old.LOG.glob('failure_*.json'))
    return sorted(paths)


def old_state():
    state=old.accounting();need(state['run_cache_case']['intents']==state['run_cache_case']['known_returns']==800,'Original run incomplete')
    r=state['review_cache_case'];need((r['intents'],r['known_returns'],r['unknown'])==(26,25,1),'Original interrupted review identity changed')
    need(r['unknown_operations'][0]['case']==FAILED_CASE,'Unexpected original unknown case')
    for k in RESOURCE_CAPS:need(state[k]['unknown']==0,'Unexpected original unknown suboperation')
    return state


def proposed():
    return dict(version='44r1',repair='Independent verifier exact rational resize; original P5 primary results unchanged',
        reason='Pillow480-to720 floating accumulator differs from frozen exact rational tie-to-upper convention; original P4 protocol already avoids this issue.',
        independent_resize='480x640->720x1280: each input row repeated [1,2] alternately240times; each column repeated2times. Original GT stays720x1280.',
        evidence_reuse='25 completed original review receipts by SHA; original26th unknown remains unknown;775 unique remaining cases independently reconstructed.',
        authorization_exception='One necessary failed-case engineering recheck under user-authorized internal repair:801 cumulative review attempts;800 unique verified cases. Original800-intent review is not reported as normally complete.',
        old_accounting=old_state(),independent_valid='One call to frozen v24.source per remaining case; no redundant call to primary core replay. GT decoded once per frame in a process and shared across its conditions.',
        expected_new=dict(repair_review_case=775,GT_decode=155,depth_decode=775,MAT_decode=775,valid_replay=775),
        resources=dict(cumulative_P5_seconds=1800,cumulative_whole_chain_seconds=2700,rss_bytes=2*1024**3,combined_P5_output_bytes=64*1024**2,
            original_raw_operation_caps=RESOURCE_CAPS,repair_review_case_intent_cap=775,new_model_calls=0,new_AP_calls=0,new_fit_calls=0),
        scope='No extra scene/frame/condition/seed, no changed primary row, threshold or decision. Any new unknown stops. No second repair or training is authorized by this component.',
        frozen_original_scientific_adoption_supported=False)


def independent_resize(a):
    need(a.shape==(480,640),'Repair only accepts the frozen480x640 grid')
    return np.repeat(np.repeat(a,np.tile(np.array([1,2]),240),axis=0),2,axis=1)


def independent_case(rgb,depth,gt,rgb_scores,depth_scores,q,valid,case):
    rm,gids=core.match(independent_resize(rgb),gt,rgb_scores,True)
    dm,dgids=core.match(independent_resize(depth),gt,depth_scores,True)
    need(gids==dgids and len(gids)>0,'Ground truth scope mismatch')
    rg=set(rm.values());dg=set(dm.values());rescue=dg-rg;lost=rg-dg
    sigma=q['sigma'];anomaly=q['anomaly_probability']
    need(sigma.shape==depth.shape==valid.shape==anomaly.shape,'Quality/mask grid mismatch')
    need(np.isfinite(sigma).all() and np.all(sigma>0) and np.isfinite(anomaly).all() and np.all((anomaly>=0)&(anomaly<=1)),'Invalid quality array')
    extra=np.exp(-sigma.astype(np.float64)/.005)*(1-.5*anomaly.astype(np.float64));candidates=[]
    for item in sorted(depth_scores,key=lambda r:int(r['instance_id'])):
        pid=int(item['instance_id']);mask=depth==pid;support=mask&valid
        label='rescue' if dm.get(pid) in rescue else ('false_positive' if pid not in dm else 'redundant_true_positive')
        candidates.append(dict(instance_id=pid,kind=label,matched_gt=dm.get(pid),pixels=int(mask.sum()),
            observed_valid_fraction=float(np.mean(valid[mask])),valid_pixels=int(support.sum()),
            negative_sigma=float(-np.mean(sigma[support],dtype=np.float64)) if support.any() else None,
            negative_anomaly=float(-np.mean(anomaly[support],dtype=np.float64)) if support.any() else None,
            extra=float(np.mean(extra[support])) if support.any() else None,score=float(item['score'])))
    return dict(**{k:case[k] for k in ('scene','frame','condition')},ground_truth=len(gids),rgb_tp=len(rg),rgb_fp=len(rgb_scores)-len(rg),
        depth_tp=len(dg),depth_fp=len(depth_scores)-len(dg),rgb_fn=len(gids)-len(rg),rescued=len(rescue),lost=len(lost),both_missed=len(gids-(rg|dg)),
        rescued_gt=sorted(rescue),lost_gt=sorted(lost),candidates=candidates)


def _accept(state,e):
    suffix='_intent' if e['kind'].endswith('_intent') else '_returned';kind=e['kind'][:-len(suffix)]
    need(kind in KINDS,'Unknown repair ledger kind');s=state[kind];cid=e['id']
    if suffix=='_intent':
        need(cid not in s['intents'] and e['case'] not in s['cases'],'Duplicate repair intent/case')
        s['intents'][cid]=e;s['cases'][e['case']]=cid
    else:
        need(cid in s['intents'] and cid not in s['returns'],'Unpaired/duplicate repair return');s['returns'].add(cid)
def ledger():
    global _STATE,_STATE_SIGNATURE
    path=LOG/'calls.jsonl';stat=path.stat() if path.exists() else None
    signature=(stat.st_size,stat.st_mtime_ns) if stat else None
    if _STATE is None or signature!=_STATE_SIGNATURE:
        _STATE={k:dict(intents={},returns=set(),cases={}) for k in KINDS}
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():_accept(_STATE,json.loads(line))
        _STATE_SIGNATURE=signature
    return _STATE
def event(kind,**fields):
    global _STATE_SIGNATURE
    state=ledger();e=dict(kind=kind,**fields);LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(e,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    _accept(state,e)
    stat=(LOG/'calls.jsonl').stat();_STATE_SIGNATURE=(stat.st_size,stat.st_mtime_ns)
def accounting():
    result={}
    for k,s in ledger().items():
        unknown=set(s['intents'])-s['returns']
        result[k]=dict(intents=len(s['intents']),known_returns=len(s['returns']),unknown=len(unknown),unknown_operations=[dict(id=x,case=s['intents'][x]['case']) for x in sorted(unknown)])
    old_counts=old_state()
    return dict(new=result,old=old_counts,cumulative_raw={k:dict(intents=old_counts[k]['intents']+result[k]['intents'],known_returns=old_counts[k]['known_returns']+result[k]['known_returns'],unknown=old_counts[k]['unknown']+result[k]['unknown']) for k in RESOURCE_CAPS},
        cumulative_review_attempts=old_counts['review_cache_case']['intents']+result['repair_review_case']['intents'],old_unknown_preserved=True,new_model_calls=0,new_AP_calls=0,new_fit_calls=0)
def usage():return old.usage()+sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'))


class Session:
    def __init__(self,phase):
        self.phase=phase;self.start=time.monotonic();self.prior=usage();self.token=uuid.uuid4().hex;self.path=LOG/('usage_'+self.token+'.json');self.save('running')
        self.gt_cache={};self.last_storage_check=0.
    def save(self,state):atomic(self.path,dict(phase=self.phase,seconds=time.monotonic()-self.start,state=state,peak_rss_bytes=old.memory()))
    def guard(self):
        elapsed=time.monotonic()-self.start;self.save('running');need(self.prior+elapsed<1800,'Original cumulative P5 time budget exceeded')
        need(old.memory()<2*1024**3,'Original RSS budget exceeded')
        if elapsed-self.last_storage_check>1 or self.last_storage_check==0:
            size=sum(p.stat().st_size for d in (old.LOG,old.OUT,LOG,OUT) for p in d.rglob('*') if p.is_file())
            need(size<64*1024**2,'Original combined output budget exceeded');self.last_storage_check=elapsed
    def intent(self,kind,case):
        self.guard();entry=ledger()[kind];need(case not in entry['cases'],'Previously attempted repair operation cannot repeat')
        cap=775 if kind=='repair_review_case' else RESOURCE_CAPS[kind]-old_state()[kind]['intents']
        need(len(entry['intents'])<cap,'Original operation limit exceeded: '+kind)
        cid=uuid.uuid4().hex;event(kind+'_intent',id=cid,case=case);return cid
    def returned(self,kind,cid):event(kind+'_returned',id=cid)


def authorize():
    a=read(AUTH);need(a.get('authorized') is True and a.get('cache_analysis_only') is True,'Repair authorization missing')
    exact=dict(old_expected_unknown=1,extra_failed_case_engineering_verification=1,old_review_attempts=26,new_review_attempts_cap=775,
        total_unique_reviewed_cases=800,total_review_attempts_expected=801,cumulative_seconds=1800)
    need(all(a.get(k)==v for k,v in exact.items()),'Repair exception/coverage/budget authorization differs')
    need(all(a.get(k)==0 for k in ('new_model_calls','new_AP_calls','new_fit_calls')),'Repair computation scope differs')
    need(set(dependency_paths())<=set(a['protected']),'Repair authorization omits dependencies')
    for p,v in a['protected'].items():need(identity(ROOT/p)==v,'Repair protected source drift: '+p)
    need(read(PROPOSAL)==proposed(),'Repair proposal differs');old.authorize()
    need(old_state()==read(PROPOSAL)['old_accounting'],'Original ledger changed')
    return a


def original_completed(manifest):
    entries=old.ledger()['review_cache_case'];passed={}
    for case in manifest['cases']:
        n=name(case);cid=entries['cases'].get(n)
        if cid is not None and cid in entries['returns']:
            receipt=old.LOG/'review_cases'/(n+'.json');row=old.OUT/'cases'/(n+'.json')
            need(read(receipt)['case_sha256']==sha(row),'Old completed review row drift')
            passed[n]=dict(original_review_receipt=relative(receipt),receipt_sha256=sha(receipt),case_sha256=sha(row))
    need(len(passed)==25,'Expected25 completed old review cases');return passed


def compute(case,record,manifest,session):
    c=read(old.CONFIG);arrays={};receipts={};n=name(case)
    for role,p in case['paths'].items():
        path=old.checked(p,manifest);receipts[role]=read(old.checked(path.with_suffix('.json'),manifest))
        with np.load(path,allow_pickle=False) as f:arrays[role]={k:f[k].copy() for k in (('sigma','valid_probability','anomaly_probability') if role=='quality' else (('instances','gate') if role=='depth' else ('instances',)))}
    label=old.checked(ROOT/c['data_root']/record['files']['label'],manifest);key=(record['scene'],record['frame'])
    if key not in session.gt_cache:
        cid=session.intent('GT_decode',session.token+f":{key[0]}:{key[1]}")
        with Image.open(label) as im:gt=np.asarray(im).copy()
        session.returned('GT_decode',cid);session.gt_cache={key:gt}
    gt=session.gt_cache[key]
    for asset in ('depth','meta'):old.checked(ROOT/c['data_root']/record['files'][asset],manifest)
    ids={k:session.intent(k,n) for k in ('depth_decode','MAT_decode','valid_replay')}
    source=reference.source(c,record,case['condition'])
    for kind,cid in ids.items():session.returned(kind,cid)
    q=arrays['quality'];qr=receipts['quality'];dr=receipts['depth']
    need(dr['corruption']['seed']==source['seed'],'Corruption seed differs')
    if 'corruption' in qr:need(qr['corruption']==dr['corruption'],'Quality input identity differs')
    if 'observed_depth_sha256' in qr:need(hashlib.sha256(source['observed'].tobytes()).hexdigest()==qr['observed_depth_sha256'],'Independent source replay hash differs')
    need(np.isfinite(q['valid_probability']).all() and np.all((q['valid_probability']>=0)&(q['valid_probability']<=1)),'Invalid predicted validity')
    gate=q['valid_probability']*np.exp(-q['sigma']/.005)*(1-.5*q['anomaly_probability'])
    need(np.allclose(gate,arrays['depth']['gate'],rtol=2e-6,atol=2e-7),'Frozen quality/gate identity differs')
    return independent_case(arrays['rgb']['instances'],arrays['depth']['instances'],gt,receipts['rgb']['scores'],dr['scores'],q,source['observed_valid'],case)


def inventory():return {relative(p):identity(p) for p in OUT.rglob('*') if p.is_file() and p!=OUT/'closeout.json'}
def verify_complete():
    authorize();review=read(LOG/'review.json');need(review['passed'] is True,'Repair review not complete')
    need(review['authorization_sha256']==sha(AUTH),'Repair authorization drift')
    need(review['old_output_inventory']==old.output_inventory(),'Original run results changed')
    need(review['reviewed_outputs']==inventory(),'Repair output set/identity differs')
    need(review['accounting']==accounting(),'Repair/old accounting differs')
    manifest=read(old.MANIFEST);prior=original_completed(manifest);covered=set(prior)
    for p in (OUT/'review_cases').glob('*.json'):
        proof=read(p);n=proof['case'];need(n not in covered and proof['passed'],'Duplicate/failed repair proof')
        need(proof['case_sha256']==sha(old.OUT/'cases'/(n+'.json')),'Repair case hash drift')
        entry=ledger()['repair_review_case'];need(n in entry['cases'] and entry['cases'][n] in entry['returns'],'Repair receipt without returned ledger')
        covered.add(n)
    need(len(covered)==800 and covered=={name(c) for c in manifest['cases']},'Unique full coverage incomplete')
    counts=accounting();need(counts['cumulative_review_attempts']==801,'Actual review attempt count differs')
    need(counts['new']['repair_review_case']['intents']==counts['new']['repair_review_case']['known_returns']==775,'Repair775 incomplete')
    need(all(counts['new'][k]['unknown']==0 for k in KINDS),'Unknown new repair operation')
    need(all(v['intents']<=RESOURCE_CAPS[k] for k,v in counts['cumulative_raw'].items()),'Original raw budget exceeded')
    return dict(passed=True,unique_cases_verified=800,actual_review_attempts=801,old_unknown_preserved=True,scientific_adoption_supported=False,
        training_candidate_worthwhile=read(old.OUT/'summary.json')['training_candidate_worthwhile'],accounting=counts)


def review(session):
    authorize()
    if (LOG/'review.json').exists():verify_complete();return
    need(all(accounting()['new'][k]['unknown']==0 for k in KINDS),'New unknown operations remain; automatic retry prohibited')
    manifest=read(old.MANIFEST);need(sha(old.MANIFEST)==read(old.LOG/'prepare.json')['manifest_sha256'],'Original manifest drift')
    prior=original_completed(manifest);records={(r['scene'],r['frame']):r for r in manifest['records']};before=old.output_inventory();rows=[]
    for i,case in enumerate(manifest['cases']):
        n=name(case);path=old.OUT/'cases'/(n+'.json')
        if n in prior:rows.append(read(path));continue
        receipt=OUT/'review_cases'/(n+'.json');entry=ledger()['repair_review_case']
        if n in entry['cases']:
            need(entry['cases'][n] in entry['returns'] and read(receipt)['case_sha256']==sha(path),'Previously returned repair proof missing/drifted')
            rows.append(read(path));continue
        need(not receipt.exists(),'Uncommitted repair proof preserved')
        cid=session.intent('repair_review_case',n);row=compute(case,records[(case['scene'],case['frame'])],manifest,session)
        need(row==read(path),'Independent repaired arithmetic differs: '+n)
        once(receipt,dict(passed=True,case=n,case_sha256=sha(path),replaces_failed_unknown=n==FAILED_CASE,old_unknown_overwritten=False))
        session.returned('repair_review_case',cid);rows.append(row)
        if (i+1)%100==0:print(json.dumps(dict(phase='repair_review',unique_covered=i+1,total=800)),flush=True)
    summary=core.summarize(rows);summary['raw_reconstruction']=old.proposed()['raw_reconstruction']
    need(summary==read(old.OUT/'summary.json'),'Original aggregate/decision differs')
    need(before==old.output_inventory(),'Review changed original result set/bytes')
    once(OUT/'validation.json',dict(passed=True,original_summary_sha256=sha(old.OUT/'summary.json'),old_completed_reused=25,new_repair_completed=775,
        actual_review_attempts=801,unique_cases_verified=800,old_unknown_preserved=True,old_failed_review_completed=False,
        training_candidate_worthwhile=summary['training_candidate_worthwhile'],scientific_adoption_supported=False))
    once(LOG/'review.json',dict(passed=True,authorization_sha256=sha(AUTH),old_output_inventory=before,reviewed_outputs=inventory(),accounting=accounting(),
        old_completed_reused=prior,original_summary_sha256=sha(old.OUT/'summary.json'),unique_cases_verified=800,actual_review_attempts=801,
        old_unknown_preserved=True,scientific_adoption_supported=False,training_candidate_worthwhile=summary['training_candidate_worthwhile']))
    verify_complete()


def cold(session):
    result=verify_complete()
    for i,(p,v) in enumerate(read(old.MANIFEST)['files'].items()):
        if i%50==0:session.guard()
        need(identity(ROOT/p)==v,'Cold original input changed: '+p)
    once(LOG/'cold.json',dict(**result,review_sha256=sha(LOG/'review.json'),new_cache_scoring=0,original_inputs_unchanged=True))


def toy(session):
    ramp=np.arange(480*640,dtype=np.int32).reshape(480,640);a=core.nearest(ramp,720,1280);b=independent_resize(ramp)
    pil=np.asarray(Image.fromarray(ramp).resize((1280,720),Image.Resampling.NEAREST));different=a!=pil
    checks=dict(exact_repeat_equals_primary=bool(np.array_equal(a,b)),PIL_bug_reproduced=bool(different.any()))
    # Matching equivalence on synthetic masks at the actual deployment size.
    low=np.zeros((480,640),np.int32);low[30:180,20:220]=1;low[190:350,200:450]=2
    gt=independent_resize(low);scores=[dict(instance_id=1,score=.8),dict(instance_id=2,score=.5)]
    q=dict(sigma=np.full(low.shape,.005,np.float32),anomaly_probability=np.zeros(low.shape,np.float32));valid=np.ones(low.shape,bool)
    case=dict(scene=0,frame='toy',condition='clean')
    primary=core.analyze_case(low,low,gt,scores,scores,q,valid,case,False)
    rebuilt=independent_case(low,low,gt,scores,scores,q,valid,case)
    checks['synthetic_case_all_fields_equal']=primary==rebuilt
    checks['prototype_raw_counts_within_original_caps']=all(old_state()[k]['intents']+(155 if k=='GT_decode' else 775)<=v for k,v in RESOURCE_CAPS.items())
    checks['old_unknown_kept']=old_state()['review_cache_case']['unknown']==1
    need(all(checks.values()),'Repair synthetic proof failed')
    once(LOG/'toy_tests.json',dict(passed=True,checks=checks,ramp_shape=[480,640],target_shape=[720,1280],PIL_different_pixels=int(different.sum()),
        PIL_different_rows=np.flatnonzero(different.any(1)).tolist(),real_raw_decodes=0,real_cache_scoring=0,new_model_calls=0,new_AP_calls=0,new_fit_calls=0,
        historical_exact_geometry='scripts/p4_complete_evaluation_v22r2.py:32-35,81',old_source_sha256=sha(ROOT/'src/p5_complementarity_v44.py')))
    once(PROPOSAL,proposed())


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('toy','review','cold'));args=p.parse_args();session=Session(args.phase)
    try:{'toy':toy,'review':review,'cold':cold}[args.phase](session);session.guard();session.save('passed')
    except BaseException as exc:
        session.save('failed');once(LOG/('failure_'+uuid.uuid4().hex+'.json'),dict(phase=args.phase,error=repr(exc),accounting=accounting()));raise


if __name__=='__main__':main()
