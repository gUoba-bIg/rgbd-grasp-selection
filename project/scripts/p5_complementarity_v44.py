"""Bounded CPU cache audit. Never imports a neural network or official AP tool."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='2'
import sys,json,time,hashlib,uuid,argparse,ctypes
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.io import loadmat
from src import p5_complementarity_v44 as core

LOG=ROOT/'logs/P5/complementarity_v44'
OUT=ROOT/'results/P5/complementarity_v44'
AUTH=LOG/'authorization.json'
PROPOSAL=LOG/'proposal.json'
MANIFEST=LOG/'input_manifest.json'
OLD=ROOT/'results/P5/completion_v39/full'
CONFIG=ROOT/'configs/p5_quality_gate_v23.json'
QUALITY_CONFIG=ROOT/'configs/p6_image_geometry_v24.json'
QUALITY_INDEX=ROOT/'results/P6/image_geometry_v24/quality_complete.json'
need=core.need
CAPS=dict(run_cache_case=800,review_cache_case=800,GT_decode=1600,depth_decode=2400,MAT_decode=2400,valid_replay=2400)
_LEDGER=None


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity(path):
    path=Path(path);s=path.stat()
    return dict(sha256=sha(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def relative(path):return Path(path).relative_to(ROOT).as_posix()
def once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():need(read(path)==value,'Committed output differs: '+str(path));return
    payload=json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)+'\n'
    with path.open('x',encoding='utf-8') as f:f.write(payload);f.flush();os.fsync(f.fileno())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    for attempt in range(21):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==20:raise
            time.sleep(.1)


def dependency_paths():
    return ['src/p5_complementarity_v44.py','scripts/p5_complementarity_v44.py',relative(PROPOSAL),relative(CONFIG),relative(QUALITY_CONFIG),relative(QUALITY_INDEX),
        'results/P5/completion_v39/full/exports_complete.json','results/P5/completion_v39/full/decision.json',
        'logs/P5/completion_v39/root_final_review.json','logs/P5/completion_v39/cold.json',
        'logs/P6/image_geometry_v24/quality_identity.json','logs/P6/image_geometry_v24/final_readonly.json',
        'experiments/p5_quality_gate_v23/quality_final.pt','logs/P5/fusion_v38/proposal.json',
        'experiments/p5_fusion_v38/quality/final.json','experiments/p4_full_baseline_v22r2/selected/epoch_33_rgb.pt',
        'src/p5_quality_v23.py','src/p6_image_geometry_v24.py','scripts/p6_common_v24.py','src/modal_fusion_v38.py']


def proposed():
    return dict(version=44,scope='P5 cache-only complementarity and observable quality association; all repeatedly exposed development',
        frames=160,scenes=[2,6,7,11,13,15,18,22,33,39],conditions=list(core.CONDITIONS),
        inputs='RGB160 and qualityRGBD800 v39 caches; original labels; frozen v24 sigma/anomaly800; raw depth+meta only for known-valid replay',
        matching='Original720x1280 grid; cached scores descending then predictionID; greedy one-to-one IoU>=.5, GT ties smallestID. No official AP.',
        quality='D predicted region; means over observed-valid pixels; extra=mean(exp(-sigma/.005)*(1-.5*anomaly)); predicted validity excluded.',
        quality_pairing='Rescue vs false-positive; within scene and condition; ascending positive valid-fraction/frame/ID, nearest unused negative valid fraction <=.01; tie frame/ID. No fitting.',
        continue_rule=dict(degraded_scene_macro_absolute_GT_recall_rescue_min=.02,positive_complementarity_scenes_min=6,
            degraded_valid_matched_pairs_min=100,quality_identifiable_scenes_min=6,extra_matched_concordance_min=.55,positive_quality_scenes='strict majority of identifiable scenes'),
        metric_note='Matched-pair concordance, not ordinary global AUROC; sigma/anomaly and residual known-valid concordance separately reported. GT labels cannot become router training labels from this development audit.',
        resources=dict(total_seconds=1800,rss_bytes=2*1024**3,output_bytes=64*1024**2,run_cache_cases=800,review_cache_cases=800,operation_intent_caps=CAPS,new_model_calls=0,new_AP_calls=0,new_fit_calls=0),
        raw_reconstruction='Finite frozen known-valid reconstruction: GT1600/depth2400/MAT2400 decodes and2400 CPU corruption replays incl independent old-source review. No new scene, corruption seed, RGB decode, neural inference or AP.',
        final='Scientific adoption always false; screening true only permits proposing a separate train-split trial. False or insufficient evidence means no new training. No P6/P7 launch.',
        limitations=['Label-informed union coverage is neither a realizable merger nor a full-method upper bound.','Known-valid matching leaves area, position, confidence, object and within-scene dependence confounding.','No independent test data, causal effect, AP improvement, calibrated quality probability, or training guarantee is established.'])


def _accept_event(state,event):
    suffix='_intent' if event['kind'].endswith('_intent') else '_returned'
    kind=event['kind'][:-len(suffix)];need(kind in CAPS,'Unknown operation kind')
    entry=state[kind];cid=event['id']
    if suffix=='_intent':
        need(cid not in entry['intents'] and event['case'] not in entry['cases'],'Duplicate intent ID/case')
        entry['intents'][cid]=event;entry['cases'][event['case']]=cid
    else:
        need(cid in entry['intents'] and cid not in entry['returns'],'Unpaired/duplicate returned operation')
        entry['returns'].add(cid)


def ledger():
    global _LEDGER
    if _LEDGER is None:
        _LEDGER={k:dict(intents={},returns=set(),cases={}) for k in CAPS}
        path=LOG/'calls.jsonl'
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():_accept_event(_LEDGER,json.loads(line))
    return _LEDGER


def accounting():
    result={}
    for kind,entry in ledger().items():
        unknown=set(entry['intents'])-entry['returns']
        result[kind]=dict(intents=len(entry['intents']),known_returns=len(entry['returns']),unknown=len(unknown),
            unknown_operations=[dict(id=k,case=entry['intents'][k]['case']) for k in sorted(unknown)])
    result.update(new_model_calls=0,new_AP_calls=0,new_fit_calls=0)
    return result


def append(kind,**fields):
    state=ledger();event=dict(kind=kind,**fields)
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    _accept_event(state,event)


def memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    c=Counters();c.cb=ctypes.sizeof(c);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    need(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(c),c.cb),'RSS read failed')
    return int(c.PeakWorkingSetSize)


def usage():return sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'))
class Session:
    def __init__(self,phase):
        self.phase=phase;self.start=time.monotonic();self.prior=usage();self.path=LOG/('usage_'+uuid.uuid4().hex+'.json');self.save('running')
    def save(self,state):atomic(self.path,dict(phase=self.phase,seconds=time.monotonic()-self.start,state=state,peak_rss_bytes=memory()))
    def guard(self):
        self.save('running');need(self.prior+time.monotonic()-self.start<1800,'P5 component cumulative wall cap')
        need(memory()<2*1024**3,'P5 component RSS cap')
        need(sum(p.stat().st_size for d in (LOG,OUT) for p in d.rglob('*') if p.is_file())<64*1024**2,'P5 component output cap')
    def intent(self,kind,case):
        self.guard();entry=ledger()[kind];need(len(entry['intents'])<CAPS[kind],'Finite operation budget exhausted: '+kind)
        need(case not in entry['cases'],'Previously attempted operation cannot be repeated: '+kind+'/'+case)
        cid=uuid.uuid4().hex;append(kind+'_intent',id=cid,case=case);return cid
    def returned(self,kind,cid):append(kind+'_returned',id=cid)
    def call(self,kind,case,operation):
        cid=self.intent(kind,case);value=operation();self.returned(kind,cid);return value


def authorize(prepared=True):
    a=read(AUTH)
    need(a.get('authorized') is True and a.get('cache_analysis_only') is True,'Explicit cache-only authorization missing')
    need(all(a.get(k)==0 for k in ('new_model_calls','new_AP_calls','new_fit_calls')),'Unauthorized computation scope')
    need(set(dependency_paths())<=set(a['protected']),'Authorization omits source metadata dependency')
    for name,expected in a['protected'].items():need(identity(ROOT/name)==expected,'Authorized source changed: '+name)
    need(read(PROPOSAL)==proposed(),'Proposal differs from fixed source rules')
    if prepared:need(read(LOG/'prepare.json')['authorization_sha256']==sha(AUTH),'Authorization changed after preparation')
    return a


def case_paths(record,condition,qindex):
    stem=f"scene_{record['scene']:04d}_{record['frame']}"
    return dict(rgb=OLD/'exports/baseline_rgb_clean'/(stem+'.npz'),depth=OLD/'exports'/('core_quality_'+condition)/(stem+'.npz'),
        quality=ROOT/qindex[condition+'_'+stem]['cache'])


def prepare(session):
    authorize(False)
    if (LOG/'prepare.json').exists():
        need(read(LOG/'prepare.json')['authorization_sha256']==sha(AUTH),'Preparation authorization drift');return
    c=read(CONFIG);qc=read(QUALITY_CONFIG)
    need(qc['quality_checkpoint']==read(ROOT/'logs/P5/fusion_v38/proposal.json')['quality_parent'],'P5/P6 frozen quality checkpoint differs')
    quality_model_sha=sha(ROOT/qc['quality_checkpoint'])
    quality_sealed=read(ROOT/'logs/P6/image_geometry_v24/final_readonly.json')['files']
    for key in ('development_records','seed','quality_screen_epoch','input_size','core_conditions','data_root'):
        need(c[key]==qc[key],'P5/P6 input definition differs: '+key)
    need(len(c['development_records'])==160 and c['core_conditions']==list(core.CONDITIONS),'Wrong scope')
    need(sorted({r['scene'] for r in c['development_records']})==proposed()['scenes'],'Wrong scenes')
    index=read(QUALITY_INDEX);need(index['total']==800 and len(index['items'])==800,'Missing full quality caches')
    qi={x['case']:x for x in index['items']};need(len(qi)==800,'Duplicate quality identities')
    sealed=read(OLD/'exports_complete.json')['inventory'];paths={};cases=[]
    for r in c['development_records']:
        for key in ('rgb','depth','label','meta'):
            p=ROOT/c['data_root']/r['files'][key];v=identity(p);need(v['sha256']==r['sha256'][key],'Raw record changed');paths[relative(p)]=v
        for condition in core.CONDITIONS:
            session.guard();p=case_paths(r,condition,qi)
            for role in ('rgb','depth'):
                receipt=p[role].with_suffix('.json')
                for file in (p[role],receipt):
                    name=relative(file)
                    if name not in paths:
                        v=identity(file);need(v==sealed[name],'v39 accepted export changed: '+name);paths[name]=v
                row=read(receipt);need(row['scene']==r['scene'] and row['frame']==r['frame'] and row['cache_sha256']==paths[relative(p[role])]['sha256'],'Prediction receipt differs')
            qrow=read(p['quality'].with_suffix('.json'));qv=identity(p['quality'])
            need(qv['sha256']==qi[condition+f"_scene_{r['scene']:04d}_{r['frame']}"]['sha256']==qrow['cache_sha256'],'Quality index hash differs')
            need((qrow['scene'],qrow['frame'],qrow['condition'])==(r['scene'],r['frame'],condition),'Quality record differs')
            if 'model_sha256' in qrow:need(qrow['model_sha256']==quality_model_sha,'Quality checkpoint receipt differs')
            for file in (p['quality'],p['quality'].with_suffix('.json')):
                name=relative(file);v=identity(file);need(v==quality_sealed[name],'Previously accepted quality artifact changed: '+name);paths[name]=v
            cases.append(dict(scene=r['scene'],frame=r['frame'],condition=condition,paths={k:relative(v) for k,v in p.items()}))
    once(MANIFEST,dict(files=paths,cases=cases,records=c['development_records'],source_seal_sha256=sha(OLD/'exports_complete.json'),quality_index_sha256=sha(QUALITY_INDEX)))
    once(LOG/'prepare.json',dict(passed=True,authorization_sha256=sha(AUTH),manifest_sha256=sha(MANIFEST),files=len(paths),cases=800,real_cache_numeric_scoring=0,raw_reconstruction=proposed()['raw_reconstruction'],operation_intent_caps=CAPS,new_model_calls=0,new_AP_calls=0,new_fit_calls=0))


def checked(path,manifest):
    path=ROOT/path if isinstance(path,str) else path
    need(identity(path)==manifest['files'][relative(path)],'Bound source changed: '+relative(path));return path


def compute_case(case,record,manifest,independent,session):
    c=read(CONFIG);arrays={};receipts={}
    for role,name in case['paths'].items():
        path=checked(name,manifest);receipts[role]=read(checked(path.with_suffix('.json'),manifest))
        with np.load(path,allow_pickle=False) as f:
            arrays[role]={k:f[k].copy() for k in (('sigma','valid_probability','anomaly_probability') if role=='quality' else (('instances','gate') if role=='depth' else ('instances',)))}
    label=checked(ROOT/c['data_root']/record['files']['label'],manifest)
    opname=f"{session.phase}:{case['condition']}_scene_{case['scene']:04d}_{case['frame']}"
    def decode_image(path,resize=False):
        with Image.open(path) as im:return np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32) if resize else np.asarray(im).copy()
    gt=session.call('GT_decode',opname,lambda:decode_image(label))
    dp=checked(ROOT/c['data_root']/record['files']['depth'],manifest)
    raw=session.call('depth_decode',opname,lambda:decode_image(dp,True))
    mp=checked(ROOT/c['data_root']/record['files']['meta'],manifest)
    meta=session.call('MAT_decode',opname,lambda:loadmat(mp,variable_names=['factor_depth']))
    factor=float(np.asarray(meta['factor_depth']).reshape(-1)[0]);need(np.isfinite(factor) and factor>0,'Invalid depth units')
    observed,valid=session.call('valid_replay',opname,lambda:core.observed_depth(raw,factor,record,case['condition'],c['seed'],c['quality_screen_epoch']))
    if independent:
        from src import p6_image_geometry_v24 as old_replay
        reference_name=opname+':frozen_reference'
        reference_ids={kind:session.intent(kind,reference_name) for kind in ('depth_decode','MAT_decode','valid_replay')}
        old=old_replay.source(c,record,case['condition'])
        for kind,cid in reference_ids.items():session.returned(kind,cid)
        need(np.array_equal(observed,old['observed']) and np.array_equal(valid,old['observed_valid']),'Independent frozen corruption replay differs')
    q=arrays['quality'];qr=receipts['quality'];dr=receipts['depth']
    seed=int(hashlib.sha256(f"P5v23|{c['seed']}|{c['quality_screen_epoch']}|{record['scene']}|{record['frame']}|{case['condition']}".encode()).hexdigest()[:8],16)
    need(dr['corruption']['seed']==seed,'Prediction corruption identity differs')
    if 'corruption' in qr:need(qr['corruption']==dr['corruption'],'Quality/prediction corruption metadata differs')
    if 'observed_depth_sha256' in qr:need(hashlib.sha256(observed.tobytes()).hexdigest()==qr['observed_depth_sha256'],'Observed-depth replay differs')
    need(np.isfinite(q['valid_probability']).all() and np.all((q['valid_probability']>=0)&(q['valid_probability']<=1)),'Invalid predicted valid')
    gate=q['valid_probability']*np.exp(-q['sigma']/.005)*(1-.5*q['anomaly_probability'])
    need(np.allclose(gate,arrays['depth']['gate'],rtol=2e-6,atol=2e-7),'Frozen quality outputs do not reproduce cached gate')
    row=core.analyze_case(arrays['rgb']['instances'],arrays['depth']['instances'],gt,receipts['rgb']['scores'],dr['scores'],q,valid,
        {k:case[k] for k in ('scene','frame','condition')},independent)
    return row


def output_inventory():
    return {relative(p):identity(p) for p in OUT.rglob('*') if p.is_file() and p!=OUT/'closeout.json'}


def run(session,independent=False):
    authorize();manifest=read(MANIFEST);need(sha(MANIFEST)==read(LOG/'prepare.json')['manifest_sha256'],'Manifest changed')
    need(all(accounting()[k]['unknown']==0 for k in CAPS),'Unknown prior operations preserved; automatic retry prohibited')
    marker=LOG/'review.json' if independent else OUT/'run.json'
    if marker.exists():
        if independent:need(read(marker)['reviewed_outputs']==output_inventory(),'Completed review output drift')
        return
    if independent:need(read(OUT/'run.json')['passed'] is True,'Run not complete')
    before=output_inventory() if independent else None
    records={(r['scene'],r['frame']):r for r in manifest['records']};rows=[]
    for n,case in enumerate(manifest['cases']):
        name=f"{case['condition']}_scene_{case['scene']:04d}_{case['frame']}"
        path=OUT/'cases'/(name+'.json')
        kind='review_cache_case' if independent else 'run_cache_case';entry=ledger()[kind]
        if name in entry['cases']:
            need(entry['cases'][name] in entry['returns'] and path.exists(),'Committed case lacks returned ledger/receipt')
            if independent:
                proof=read(LOG/'review_cases'/(name+'.json'));need(proof['case_sha256']==sha(path),'Partial review receipt changed')
            rows.append(read(path));continue
        need(independent or not path.exists(),'Uncommitted cache receipt preserved; no automatic reuse')
        cid=session.intent(kind,name)
        row=compute_case(case,records[(case['scene'],case['frame'])],manifest,independent,session)
        if independent:need(read(path)==row,'Independent case arithmetic differs: '+name)
        else:once(path,row)
        if independent:once(LOG/'review_cases'/(name+'.json'),dict(case_sha256=sha(path),passed=True))
        rows.append(row);session.returned(kind,cid)
        if (n+1)%100==0:print(json.dumps(dict(phase=session.phase,completed=n+1,total=800)),flush=True)
    summary=core.summarize(rows)
    summary['raw_reconstruction']=proposed()['raw_reconstruction']
    if independent:
        need(read(OUT/'summary.json')==summary,'Independent summary differs')
        need(before==output_inventory(),'Review mutated output path set/identity')
        counts=accounting();need(all(counts[k]['intents']==counts[k]['known_returns']==CAPS[k] and counts[k]['unknown']==0 for k in CAPS),'Finite cache accounting incomplete/unknown')
        once(marker,dict(passed=True,reviewed_outputs=before,accounting=counts,manifest_sha256=sha(MANIFEST),authorization_sha256=sha(AUTH),scientific_adoption_supported=False,
            training_candidate_worthwhile=summary['training_candidate_worthwhile']))
    else:
        once(OUT/'summary.json',summary)
        once(marker,dict(passed=True,case_count=800,manifest_sha256=sha(MANIFEST),summary_sha256=sha(OUT/'summary.json'),
            scientific_adoption_supported=False,training_candidate_worthwhile=summary['training_candidate_worthwhile'],operation_intent_caps=CAPS,accounting_at_run=accounting(),raw_reconstruction=proposed()['raw_reconstruction']))


def cold(session):
    authorize();review=read(LOG/'review.json');need(review['passed'],'Review incomplete')
    need(review['manifest_sha256']==sha(MANIFEST) and review['authorization_sha256']==sha(AUTH),'Cold input seal changed')
    need(review['reviewed_outputs']==output_inventory(),'Cold output set/identity changed')
    need(review['accounting']==accounting(),'Cold ledger changed')
    # Hash-only original input verification; no cache scoring or old writes.
    for n,(path,expected) in enumerate(read(MANIFEST)['files'].items()):
        if n%50==0:session.guard()
        need(identity(ROOT/path)==expected,'Cold source drift: '+path)
    once(LOG/'cold.json',dict(passed=True,review_sha256=sha(LOG/'review.json'),accounting=accounting(),new_cache_scoring=0,old_sources_unchanged=True,
        training_candidate_worthwhile=review['training_candidate_worthwhile'],scientific_adoption_supported=False))


def toy(session):
    gt=np.array([[1,1,0,2,2],[1,1,0,2,2]],np.int32);rgb=np.where(gt==1,1,0).astype(np.int32);dep=np.where(gt==2,1,0).astype(np.int32)
    scores=[dict(instance_id=1,score=.8)];q=dict(sigma=np.full(gt.shape,.005,np.float32),anomaly_probability=np.zeros(gt.shape,np.float32));valid=np.ones(gt.shape,bool)
    a=core.analyze_case(rgb,dep,gt,scores,scores,q,valid,dict(scene=2,frame='0000',condition='clean'))
    b=core.analyze_case(rgb,dep,gt,scores,scores,q,valid,dict(scene=2,frame='0000',condition='clean'),True)
    checks=dict(independent_match=a==b,rescue_and_loss=a['rescued']==a['lost']==1,union_not_merger=a['both_missed']==0,
        extra_without_predvalid=abs(a['candidates'][0]['extra']-np.exp(-1))<1e-7,
        no_valid_no_quality=core.analyze_case(rgb,dep,gt,scores,scores,q,~valid,dict(scene=2,frame='0000',condition='clean'))['candidates'][0]['extra'] is None)
    tiegt=np.array([[1,1,2,2]],np.int32);tiepred=np.ones_like(tiegt)
    checks['matching_tie_smallest_GT']=core.match(tiepred,tiegt,scores)[0]=={1:1}
    low=np.arange(12,dtype=np.int32).reshape(3,4)
    checks['nonuniform_resize']=np.array_equal(core.nearest(low,7,9),np.asarray(Image.fromarray(low).resize((9,7),Image.Resampling.NEAREST)))
    rows=[dict(scene=2,frame='0000',condition='joint',candidates=[dict(instance_id=1,kind='rescue',observed_valid_fraction=.8,extra=.8,negative_sigma=-.002,negative_anomaly=-.1),dict(instance_id=2,kind='false_positive',observed_valid_fraction=.795,extra=.2,negative_sigma=-.008,negative_anomaly=-.5)])]
    pairs=core.matched_quality(rows);checks['valid_matched_positive']=len(pairs)==1 and pairs[0]['outcomes']['extra']==1
    rows[0]['candidates'][1]['observed_valid_fraction']=.7;checks['unmatched_preserved_no_claim']=core.matched_quality(rows)==[]
    state={k:dict(intents={},returns=set(),cases={}) for k in CAPS}
    _accept_event(state,dict(kind='run_cache_case_intent',id='toyA',case='toycase'))
    checks['unknown_kept']=len(state['run_cache_case']['intents'])==1 and not state['run_cache_case']['returns']
    try:_accept_event(state,dict(kind='run_cache_case_intent',id='toyB',case='toycase'))
    except RuntimeError:checks['duplicate_case_rejected']=True
    else:checks['duplicate_case_rejected']=False
    _accept_event(state,dict(kind='run_cache_case_returned',id='toyA'))
    checks['returned_bound_to_intent']=state['run_cache_case']['returns']=={'toyA'}
    staged=[]
    for sc in proposed()['scenes']:
        for condition in core.CONDITIONS:
            for frame in range(16):
                item=dict(a,scene=sc,condition=condition,frame=f'{frame*16:04d}')
                item['candidates']=[dict(instance_id=1,kind='rescue',observed_valid_fraction=.8,extra=.8,negative_sigma=-.002,negative_anomaly=-.1),dict(instance_id=2,kind='false_positive',observed_valid_fraction=.795,extra=.2,negative_sigma=-.008,negative_anomaly=-.5)]
                staged.append(item)
    summary=core.summarize(staged)
    checks['screen_can_pass_without_scientific_claim']=summary['training_candidate_worthwhile'] is True and summary['scientific_adoption_supported'] is False
    checks['rescue_absolute_and_miss_denominators_distinct']=summary['conditions']['clean']['scenes'][0]['rescue_recall_gain']==.5 and summary['conditions']['clean']['scenes'][0]['rescued_fraction_of_rgb_misses']==1.
    need(all(checks.values()),'Synthetic test failed')
    once(LOG/'toy_tests.json',dict(passed=True,checks={k:bool(v) for k,v in checks.items()},real_records_read=0,new_model_calls=0,new_AP_calls=0,new_fit_calls=0))
    once(PROPOSAL,proposed())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('toy','prepare','run','review','cold'));args=parser.parse_args()
    s=Session(args.phase)
    try:
        {'toy':toy,'prepare':prepare,'run':lambda s:run(s,False),'review':lambda s:run(s,True),'cold':cold}[args.phase](s);s.guard();s.save('passed')
    except BaseException as exc:
        s.save('failed');once(LOG/('failure_'+uuid.uuid4().hex+'.json'),dict(phase=args.phase,error=repr(exc),accounting=accounting()));raise


if __name__=='__main__':main()
