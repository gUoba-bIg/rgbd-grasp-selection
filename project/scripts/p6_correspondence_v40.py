"""One bounded, exposed-development global translation identifiability study.

No neural calls. Real estimators/fit require root authorization. Formal stages
are generate, fit, evaluate, review, cold. Cold performs no research calls.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import inspect
import json
import time
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import uniform_filter
from src import correspondence_v40 as core
from scripts.pipeline import project_lock

LOG = ROOT / 'logs/P6/correspondence_v40'
OUT = ROOT / 'results/P6/correspondence_v40'
LIMITS = dict(seconds=1800, rss_bytes=2*1024**3, output_bytes=256*1024**2,
              alignment_generate=720, alignment_review=720, ridge_fit=1, neural_forward=0, ap=0)
SCENES = dict(fit=[3,4,5,8,9,10,12,14,16,17,19,20], calibration=[30,31,34,38], evaluation=[42,43,57,93])
INHERITED = ['src/correspondence_v40.py', 'scripts/p6_correspondence_v40.py', 'scripts/pipeline.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def identity(path):
    st = path.stat()
    return dict(sha256=sha(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def replace_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.writing')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def append(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
        f.flush(); os.fsync(f.fileno())


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def runtime_identity():
    import scipy, PIL
    return dict(python=sys.version.split()[0], executable=str(Path(sys.executable).resolve()),
                numpy=np.__version__, scipy=scipy.__version__, Pillow=PIL.__version__,
                cpu_threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})


def peak_rss():
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('peak','working','a','b','c','d','e','f')]
    value= Counters(); value.cb=ctypes.sizeof(value)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    core.need(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(value),value.cb), 'Memory accounting unavailable')
    return int(value.peak)


def audit_ledger():
    """Pure read: pair actual call events, retaining missing returns as unknown."""
    starts, returned, keys = {}, {}, {}
    path = LOG/'calls.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            e = json.loads(line)
            if e['event']=='started':
                core.need(e['id'] not in starts and e['key'] not in keys, 'Duplicate research call identity')
                starts[e['id']]=e; keys[e['key']]=e['id']
            else:
                core.need(e['event']=='returned' and e['id'] in starts and e['id'] not in returned, 'Unmatched research return')
                s=starts[e['id']]
                core.need(e['kind']==s['kind'] and e['key']==s['key'], 'Changed call return identity')
                returned[e['id']]=e
    counts={k:0 for k in ('alignment_generate','alignment_review','ridge_fit','neural_forward','ap')}
    for e in returned.values():
        core.need(e['kind'] in counts, 'Undeclared research call kind')
        counts[e['kind']]+=1
    return dict(counts=counts, unknown_calls=[s for i,s in starts.items() if i not in returned],
                returned_by_key={e['key']:e['result'] for e in returned.values()}, started_counts={k:sum(s['kind']==k for s in starts.values()) for k in counts})


class Budget:
    def __init__(self, stage):
        self.stage=stage; self.started=time.perf_counter(); self.last=-1e9
        self.previous=read(LOG/'usage.json') if (LOG/'usage.json').exists() else {}
        self.prior=float(self.previous.get('seconds',0))
        self.maxrss=int(self.previous.get('peak_rss_bytes',0))
        self.files=set(p for folder in (LOG,OUT) if folder.exists() for p in folder.rglob('*') if p.is_file())
        self.ledger=audit_ledger()
        core.need(not self.ledger['unknown_calls'], 'Unknown prior research call: preserve evidence; no automatic retry')
        self.sample(force=True)

    def register(self,path):
        self.files.add(path)

    def sample(self,force=False):
        elapsed=time.perf_counter()-self.started
        if not force and elapsed-self.last<2: return
        self.last=elapsed; self.maxrss=max(self.maxrss,peak_rss())
        total=sum(p.stat().st_size for p in self.files if p.exists())
        data=dict(seconds=self.prior+elapsed,peak_rss_bytes=self.maxrss,registered_output_bytes=total,
            stage=self.stage,at=now(),counts=self.ledger['counts'],unknown_calls=self.ledger['unknown_calls'],sample_interval_seconds=2)
        replace_json(LOG/'usage.json',data); self.register(LOG/'usage.json')
        core.need(data['seconds']<=LIMITS['seconds'], 'Cumulative stage wall-time limit')
        core.need(self.maxrss<=LIMITS['rss_bytes'], 'RSS limit')
        core.need(total<=LIMITS['output_bytes'], 'Output byte limit')

    def invoke(self,kind,key,function):
        self.sample()
        if key in self.ledger['returned_by_key']: return self.ledger['returned_by_key'][key]
        core.need(self.ledger['started_counts'][kind]<LIMITS[kind], 'Research call budget exceeded: '+kind)
        cid=uuid.uuid4().hex
        s=dict(event='started',id=cid,kind=kind,key=key,at=now())
        append(LOG/'calls.jsonl',s); self.register(LOG/'calls.jsonl')
        self.ledger['started_counts'][kind]+=1; self.ledger['unknown_calls'].append(s)
        answer=function()
        append(LOG/'calls.jsonl',dict(event='returned',id=cid,kind=kind,key=key,at=now(),result=answer))
        self.ledger['unknown_calls'].remove(s); self.ledger['counts'][kind]+=1
        self.ledger['returned_by_key'][key]=answer
        self.sample()
        return answer

    def close(self):
        self.sample(force=True)


def verify_identities(mapping,full=True):
    for name, expected in mapping.items():
        p=ROOT/name
        core.need(p.resolve().is_relative_to(ROOT) and p.is_file(), 'Protected file missing/outside root: '+name)
        st=p.stat()
        core.need(st.st_size==expected['bytes'] and st.st_mtime_ns==expected['mtime_ns'], 'Protected file stat changed: '+name)
        if full: core.need(sha(p)==expected['sha256'], 'Protected content changed: '+name)


def prepare():
    data=read(LOG/'data_audit.json')
    core.need(data['passed'] and data['scenes']==SCENES, 'Audited split differs')
    core.need([len(data['records'][k]) for k in SCENES]==[48,16,16], 'Wrong frame counts')
    verify_identities(data['protected'])
    protected={k:identity(ROOT/k) for k in INHERITED+['logs/P6/correspondence_v40/data_audit.json']}
    proposal=dict(version='v40',component='P6_correspondence_identifiability',created=now(),runtime=runtime_identity(),
        inherited_sources=INHERITED,protected=protected,data_audit_sha256=sha(LOG/'data_audit.json'),
        records=data['records'],scenes=SCENES,warps=[list(x) for x in core.WARPS],cases=dict(fit=432,calibration=144,evaluation=144),
        scope='Global whole-image integer rigid translation on exposed development scenes only. Not nonrigid/real calibration/full uvz/P6 full pass.',
        exposure='All20 scenes used in upstream training; previous development results exposed; split disjoint only for this estimator.',
        estimator=dict(observable_inputs=['RGB','observed_depth','observed_validity'],resolution=[640,480],candidate_uv=[-4,4],candidate_count=81,
            depth_edge_threshold_m=.01,max_edges=4096,min_edges=64,fixed_observed_border=8,valid_erosion_size=3,
            weights='gradient magnitude/(magnitude+.02) for RGB and depth; absolute unit-gradient dot product',
            score='sum weighted directional agreement / original selected count; candidate out-of-image samples zero',
            tie='minimum squared displacement then dy then dx; ambiguous top gap<=1e-12 rejected',
            rejected='retained with zero shift; counts failure for identity; no_rejected_case must pass',
            padding_defense='Fixed observed8px interior and3x3 observed-valid erosion, never oracle support'),
        fit=dict(family='six_parameter_affine_ridge',input_features=['raw_dx','raw_dy','one'],ridge=1.,intercept_penalty=0.,fit_calls=1,
            output='clip[-4,4] then np.rint half-to-even integer; rejected forcedzero',search=False),
        calibration='Calibration4 scenes describe empirical residual L-infinity95% radius only; no fit/tuning, probability or conditional coverage guarantee.',
        evaluation=dict(main_points=4096,extra_gt_boundary_points=1024,fixed_source_border=8,
            eligible='valid unclipped originaldepth<=2m; report excluded border and lost sourcevalid separately',
            source_identity='integer nearest warp source-id; no interpolation; observed_uv=source_uv+warp; corrected_uv=observed_uv-estimate',
            labels='decode evaluation only for proxy strata and boundary instance identity, never estimator/fit',
            geometry='Separate raw/corrected/oracle XYZ; original source files and observations never overwritten',
            weighting='Euclidean pixels: equal scene/frame/warp; clean excluded from improvement denominator',
            boundary='GT instance identity retention is a correspondence proxy, not proof of same physical surface'),
        gates=dict(nonzero_mean_error_px_max=1.5,improvement_fraction_min=.5,each_direction_mean_px_max=2.,worst_scene_mean_px_max=2.,
            clean_mean_px_max=.5,clean_max_px_max=1.,boundary_identity_delta_min=0.,rejected_cases_max=0,oracle_source_and_xyz_exact=True),
        limits=LIMITS,synthetic_preparation=dict(max_check_cases=12,max_estimator_invocations=24,real_data_estimator_calls=0,neural_calls=0),
        stop_rule='One fit, no search/automatic repair/expansion. Report finite identifiability result, P6_full_passed always false.',
        stages=['generate','fit','evaluate','review','cold'])
    path=LOG/'proposal.json'
    if path.exists():
        old=read(path); proposal['created']=old['created']
        core.need(proposal==old,'Existing proposal differs; preserve old before preparing a revision')
    else: once(path,proposal)
    return dict(passed=True,proposal_sha256=sha(path))


def authorized():
    p=read(LOG/'proposal.json'); a=read(LOG/'authorization.json')
    core.need(a.get('frozen') is True and a['proposal_sha256']==sha(LOG/'proposal.json'),'Formal authorization/proposal mismatch')
    core.need(p['runtime']==runtime_identity() and p['limits']==LIMITS,'Numerical runtime or budget changed')
    verify_identities(a['protected']); verify_identities(p['protected'])
    data=read(LOG/'data_audit.json'); verify_identities(data['protected'])
    return p,data


def load_frame(data,record,labels=False):
    base=ROOT/data['data_root']
    m=loadmat(base/record['files']['meta'],variable_names=['intrinsic_matrix','factor_depth'])
    k=np.asarray(m['intrinsic_matrix'],np.float64); factor=float(m['factor_depth'].reshape(-1)[0])
    core.need(k.shape==(3,3) and np.isfinite(k).all() and factor>0 and k[0,0]>0 and k[1,1]>0,'Invalid camera')
    with Image.open(base/record['files']['depth']) as im:
        size=im.size; raw=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32)/factor
    valid=np.isfinite(raw)&(raw>0)
    depth=np.where(valid,np.minimum(raw,2.),0).astype(np.float32)
    with Image.open(base/record['files']['rgb']) as im:
        core.need(im.size==size,'RGB/depth dimensions differ')
        rgb=np.asarray(im.convert('RGB').resize((640,480),Image.Resampling.BILINEAR),np.float64)/255.
    frame=dict(rgb=rgb,depth=depth,valid=valid,eligible=valid&(raw<=2.),K=k,original_size=size)
    if labels:
        with Image.open(base/record['files']['label']) as im:
            core.need(im.size==size,'GT/depth dimensions differ')
            frame['labels']=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.int64)
    return frame


def key_for(split,record,delta):
    return f"{split}/{record['scene']:04d}/{record['frame']}/{delta[0]:+d}_{delta[1]:+d}"


def pack_estimate(value):
    return dict(raw_uv=value['raw_uv'].tolist(),scores=value['scores'].tolist(),usable=value['usable'],
        edge_count=value['edge_count'],edge_indices_sha256=array_sha(value['edge_indices']),
        peak_margin=value['peak_margin'],best_score=value['best_score'],rejection=value['rejection'])


def generate(data,budget):
    rows=[]
    for split,records in data['records'].items():
        for r in records:
            frame=load_frame(data,r)
            for delta in core.WARPS:
                obs,ok,source=core.warp_integer(frame['depth'],frame['valid'],delta)
                key=key_for(split,r,delta)
                value=budget.invoke('alignment_generate','generate/'+key,lambda:pack_estimate(core.estimate(frame['rgb'],obs,ok)))
                rows.append(dict(key=key,split=split,scene=r['scene'],frame=r['frame'],warp_uv=list(delta),estimate=value,
                    observed_valid_count=int(ok.sum()),original_valid_count=int(frame['valid'].sum()),
                    lost_source_valid_count=int(frame['valid'].sum()-ok.sum()),source_id_sha256=array_sha(source)))
            budget.sample()
    core.need(len(rows)==720,'Generate case count differs')
    target=OUT/'generated.json'
    result=dict(version='v40',rows=rows,counts={s:sum(r['split']==s for r in rows) for s in SCENES},
                proposal_sha256=sha(LOG/'proposal.json'),data_audit_sha256=sha(LOG/'data_audit.json'))
    if target.exists(): core.need(read(target)==result,'Generated cache differs')
    else: once(target,result)
    budget.register(target)


def generated():
    value=read(OUT/'generated.json')
    core.need(value['proposal_sha256']==sha(LOG/'proposal.json'),'Generated proposal identity differs')
    return value['rows']


def fit(data,budget):
    rows=generated(); fr=[r for r in rows if r['split']=='fit']; cr=[r for r in rows if r['split']=='calibration']
    raw=np.array([r['estimate']['raw_uv'] for r in fr]); y=np.array([r['warp_uv'] for r in fr])
    model=budget.invoke('ridge_fit','fit/one_affine_ridge',lambda:core.ridge_fit(raw,y))
    calraw=np.array([r['estimate']['raw_uv'] for r in cr]); usable=np.array([r['estimate']['usable'] for r in cr])
    continuous,applied=core.predict(model,calraw,usable)
    residual=applied-np.array([r['warp_uv'] for r in cr]); linf=np.max(np.abs(residual),axis=1)
    radius=float(np.sort(linf)[int(np.ceil(.95*len(linf)))-1])
    model=dict(model,proposal_sha256=sha(LOG/'proposal.json'),generated_sha256=sha(OUT/'generated.json'),
        calibration=dict(cases=len(cr),predicted_continuous_uv=continuous.tolist(),applied_uv=applied.tolist(),residual_uv=residual.tolist(),
            empirical_linf95_radius_px=radius,rejected_cases=int((~usable).sum()),
            role='Description only; no second fit, model selection, probability/coverage calibration or covariance.'),
        fitted_scene_ids=SCENES['fit'],calibration_scene_ids=SCENES['calibration'])
    target=OUT/'model.json'
    if target.exists(): core.need(read(target)==model,'Model artifact differs')
    else: once(target,model)
    budget.register(target)


def evaluate(data,budget):
    allrows=generated(); er=[r for r in allrows if r['split']=='evaluation']; model=read(OUT/'model.json')
    core.need(model['generated_sha256']==sha(OUT/'generated.json'),'Model source changed')
    raw=np.array([r['estimate']['raw_uv'] for r in er]); usable=np.array([r['estimate']['usable'] for r in er])
    continuous,applied=core.predict(model,raw,usable)
    indexed={r['key']:(r,c,a) for r,c,a in zip(er,continuous,applied)}
    rows=[]; frames=[]
    for r in data['records']['evaluation']:
        frame=load_frame(data,r,labels=True)
        ref=core.select_reference(frame['depth'],frame['valid'],frame['eligible'],frame['labels'])
        frames.append(dict(scene=r['scene'],frame=r['frame'],selected_points=len(ref['indices']),main_points=int(ref['main'].sum()),
            boundary_points=int(ref['boundary'].sum()),fixed_domain_count=ref['fixed_domain_count'],
            excluded_valid_from_fixed8px_border=ref['excluded_valid_from_fixed8px_border'],selected_indices_sha256=array_sha(ref['indices'])))
        for delta in core.WARPS:
            item,c,a=indexed[key_for('evaluation',r,delta)]
            _,_,source=core.warp_integer(frame['depth'],frame['valid'],delta)
            metrics=core.evaluate_case(frame,ref,delta,a,item['estimate']['usable'],source)
            rows.append(dict(scene=r['scene'],frame=r['frame'],key=item['key'],warp_uv=list(delta),continuous_uv=c.tolist(),applied_uv=a.tolist(),
                lost_source_valid_count=item['lost_source_valid_count'],**metrics))
        budget.sample()
    summary,gates=core.summarize(rows)
    result=dict(version='v40',engineering_computation_completed=True,engineering_passed=False,independent_review_pending=True,
        identifiability_supported=all(gates.values()),scientific_usable=all(gates.values()),P6_full_passed=False,uvz_calibrated=False,P7_started=False,
        summary=summary,gates=gates,evaluation_rows=rows,frame_denominators=frames,
        scientific_scope='Exposed-development whole-image integer rigid translation only; no fulluvz/no real sensor/no grasp risk validation.',
        boundary_identity_scope='GT instance identity proxy only; physical synthetic correspondence is source-id exactness and UV coordinate error.',
        calibrated_probability_model=False,model_sha256=sha(OUT/'model.json'),generated_sha256=sha(OUT/'generated.json'),
        proposal_sha256=sha(LOG/'proposal.json'),next_action='Stop after independent review/cold; no automatic expansion or further P6/P7/P10.',
        progress=dict(all_tasks_percent=45,experiments_percent=56.25,changed=False))
    target=OUT/'result.json'
    if target.exists(): core.need(read(target)==result,'Result artifact differs')
    else: once(target,result)
    budget.register(target)


def independent_warp(z,valid,delta):
    h,w=z.shape; yy,xx=np.indices(z.shape); sx=xx-int(delta[0]); sy=yy-int(delta[1])
    inside=(sx>=0)&(sx<w)&(sy>=0)&(sy<h)
    index=np.zeros(z.shape,np.int64); index[inside]=sy[inside]*w+sx[inside]
    ok=inside & valid.ravel()[index]
    out=np.zeros_like(z); out[ok]=z.ravel()[index[ok]]
    sid=np.full(z.shape,-1,np.int64); sid[ok]=index[ok]
    return out,ok,sid


def independent_estimate(rgb,z,valid):
    """Separate score/candidate/sample implementation for read-only replay."""
    gray=.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2]
    rg=np.gradient(gray); dg=np.gradient(z.astype(np.float64))
    rm=np.sqrt(rg[0]*rg[0]+rg[1]*rg[1]); dm=np.sqrt(dg[0]*dg[0]+dg[1]*dg[1])
    support=np.zeros(z.shape,bool)
    support[1:-1,1:-1]=np.logical_and.reduce([valid[dy:valid.shape[0]-2+dy,dx:valid.shape[1]-2+dx] for dy in range(3) for dx in range(3)])
    support[:8]=False; support[-8:]=False; support[:,:8]=False; support[:,-8:]=False
    allidx=np.flatnonzero(support & (dm>.01)); n=min(4096,len(allidx))
    idx=allidx[(np.arange(n)*len(allidx)//n)] if n else np.empty(0,np.int64)
    yy,xx=np.divmod(idx,z.shape[1]); candidates=sorted([(x,y) for y in range(-4,5) for x in range(-4,5)],key=lambda t:(t[0]*t[0]+t[1]*t[1],t[1],t[0]))
    scores=[]
    for dx,dy in candidates:
        u=xx-dx; v=yy-dy; inside=(u>=0)&(u<z.shape[1])&(v>=0)&(v<z.shape[0])
        value=np.zeros(n,float)
        if inside.any():
            sy,sx=yy[inside],xx[inside]; ty,tx=v[inside],u[inside]
            # Product of gradient normalization and saturating magnitudes cancels.
            dot=np.abs(dg[0][sy,sx]*rg[0][ty,tx]+dg[1][sy,sx]*rg[1][ty,tx])
            value[inside]=dot/((dm[sy,sx]+.02)*(rm[ty,tx]+.02))
        scores.append(float(value.sum()/max(1,n)))
    scores=np.array(scores); best=int(scores.argmax()); ordered=np.sort(scores); margin=float(ordered[-1]-ordered[-2])
    usable=n>=64 and margin>1e-12
    return dict(raw_uv=list(candidates[best]) if usable else [0,0],scores=scores.tolist(),usable=bool(usable),edge_count=n,
        edge_indices_sha256=array_sha(idx),peak_margin=margin,best_score=float(scores[best]),
        rejection='none' if usable else ('fewer_than64_edges' if n<64 else 'nonunique_peak'))


def close_numeric(a,b,path='value'):
    if isinstance(a,dict):
        core.need(isinstance(b,dict) and set(a)==set(b),'Keys differ: '+path)
        for k in a: close_numeric(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        core.need(isinstance(b,list) and len(a)==len(b),'Length differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)): close_numeric(x,y,path+'/'+str(i))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        core.need(np.isclose(a,b,rtol=1e-11,atol=1e-12),'Numeric mismatch: '+path)
    else: core.need(a==b,'Mismatch: '+path)


def independent_evaluation(frame,delta,applied,usable,source):
    """Independent source/stratum/geometry replay; never invokes core evaluation."""
    z=frame['depth']; labels=frame['labels']; h,w=z.shape
    domain=frame['eligible'].copy(); domain[:8]=False; domain[-8:]=False; domain[:,:8]=False; domain[:,-8:]=False
    b=np.zeros(z.shape,bool)
    b[1:]|=labels[1:]!=labels[:-1]; b[:-1]|=labels[:-1]!=labels[1:]
    b[:,1:]|=labels[:,1:]!=labels[:,:-1]; b[:,:-1]|=labels[:,:-1]!=labels[:,1:]
    boundary=b.copy(); boundary[1:]|=b[:-1]; boundary[:-1]|=b[1:]; boundary[:,1:]|=b[:,:-1]; boundary[:,:-1]|=b[:,1:]
    sets=[]
    for mask,limit in ((domain,4096),(domain&boundary,1024)):
        flat=np.flatnonzero(mask); n=min(limit,len(flat)); core.need(n>0,'Independent empty evaluation stratum')
        sets.append(flat[np.arange(n)*len(flat)//n])
    idx=np.array(sorted(set(sets[0].tolist()+sets[1].tolist())),np.int64)
    sv,su=idx//w,idx%w; ov=sv+delta[1]; ou=su+delta[0]; cv=ov-applied[1]; cu=ou-applied[0]
    core.need(np.array_equal(source[ov,ou],idx),'Independent source-id mismatch')
    core.need(np.all((cu>=0)&(cu<w)&(cv>=0)&(cv<h)),'Independent evaluation out of fixed support')
    depth=z.ravel()[idx].astype(np.float64); K=frame['K']; width,height=frame['original_size']
    def projection(u,v):
        original_u=np.floor((u+.5)*width/w); original_v=np.floor((v+.5)*height/h)
        return np.stack((depth*(original_u-K[0,2])/K[0,0],depth*(original_v-K[1,2])/K[1,1],depth),axis=1)
    truth=projection(su,sv); baseline=projection(ou,ov); corrected=projection(cu,cv); oracle=projection(ou-delta[0],ov-delta[1])
    core.need(np.array_equal(truth,oracle),'Independent oracle geometry mismatch')
    original=labels.ravel()[idx]; raw=labels[ov,ou]; corr=labels[cv,cu]; rawvalid=frame['valid'][ov,ou]
    groups=dict(main=np.isin(idx,sets[0]),boundary=boundary.ravel()[idx],unregistered_same_instance=rawvalid&(raw==original),
        unregistered_cross_instance=raw!=original,unregistered_depth_jump_proxy=rawvalid&(np.abs(z[ov,ou]-z.ravel()[idx])>.02),unregistered_reference_missing=~rawvalid)
    metrics={}
    for name,mask in groups.items():
        metrics[name]=dict(n=int(mask.sum()),baseline_instance_identity_matches=int(np.sum((raw==original)&mask)),
            estimated_instance_identity_matches=int(np.sum((corr==original)&mask)) if usable else 0,
            baseline_squared_error3d_sum=float(np.square(baseline[mask]-truth[mask]).sum()),estimated_squared_error3d_sum=float(np.square(corrected[mask]-truth[mask]).sum()))
    error=np.array(applied)-delta
    row=dict(error_uv_px=error.tolist(),euclidean_error_px=float(np.sqrt(np.dot(error,error))),no_correction_error_px=float(np.sqrt(np.dot(delta,delta))),
        usable=usable,same_source_correspondence_exact=bool(usable and list(applied)==list(delta)),metrics=metrics,oracle_source_identity_exact=True,oracle_error_px=0.,
        original_xyz_overwritten=False,uncertainty_scope='global integer image translation only; no depth-noise/fulluvz/collision calibration')
    denominator=dict(selected_points=len(idx),main_points=int(groups['main'].sum()),boundary_points=int(groups['boundary'].sum()),
        fixed_domain_count=int(domain.sum()),excluded_valid_from_fixed8px_border=int(frame['eligible'].sum()-domain.sum()),selected_indices_sha256=array_sha(idx))
    return row,denominator


def independent_summary(rows):
    scenes=sorted({x['scene'] for x in rows}); changed=[x for x in rows if any(x['warp_uv'])]; clean=[x for x in rows if not any(x['warp_uv'])]
    scene={str(s):sum(x['euclidean_error_px'] for x in changed if x['scene']==s)/32 for s in scenes}
    dirs={str(tuple(d)):sum(x['euclidean_error_px'] for x in changed if x['warp_uv']==list(d))/16 for d in core.WARPS[1:]}
    mean=sum(scene.values())/4; baseline=sum(x['no_correction_error_px'] for x in changed)/128
    boundary={}
    for s in scenes:
        cases=[x for x in changed if x['scene']==s]
        fixed=sum(x['metrics']['boundary']['baseline_instance_identity_matches']/x['metrics']['boundary']['n'] for x in cases)/32
        estimated=sum(x['metrics']['boundary']['estimated_instance_identity_matches']/x['metrics']['boundary']['n'] for x in cases)/32
        boundary[str(s)]=dict(fixed=fixed,estimated=estimated,delta=estimated-fixed)
    summary=dict(nonzero_mean_error_px=mean,no_correction_mean_error_px=baseline,improvement_fraction=1-mean/baseline,
        clean_mean_error_px=sum(x['euclidean_error_px'] for x in clean)/16,clean_max_error_px=max(x['euclidean_error_px'] for x in clean),
        worst_scene_error_px=max(scene.values()),direction_errors_px=dirs,scene_errors_px=scene,
        boundary_retention_fixed_vs_estimated=dict(fixed=sum(x['fixed'] for x in boundary.values())/4,estimated=sum(x['estimated'] for x in boundary.values())/4,by_scene=boundary),
        rejection_count=sum(not x['usable'] for x in rows),total_cases=len(rows),exact_source_correspondence_case_fraction=sum(x['same_source_correspondence_exact'] for x in rows)/144,
        oracle_all_exact=all(x['oracle_source_identity_exact'] and x['oracle_error_px']==0 for x in rows))
    gate=dict(mean_error=mean<=1.5,improvement=summary['improvement_fraction']>=.5,each_direction=max(dirs.values())<=2.,worst_scene=max(scene.values())<=2.,
        clean_mean=summary['clean_mean_error_px']<=.5,clean_max=summary['clean_max_error_px']<=1.,
        boundary_identity_noninferior=summary['boundary_retention_fixed_vs_estimated']['estimated']>=summary['boundary_retention_fixed_vs_estimated']['fixed'],
        no_rejected_case=summary['rejection_count']==0,oracle_positive_control=summary['oracle_all_exact'])
    return summary,gate


def review(data,budget):
    rows=generated(); source_rows={x['key']:x for x in rows}; model=read(OUT/'model.json'); result=read(OUT/'result.json')
    core.need(len(source_rows)==720 and len(rows)==720,'Generated case identity duplication')
    core.need(model['generated_sha256']==sha(OUT/'generated.json') and model['proposal_sha256']==sha(LOG/'proposal.json'),'Model provenance differs')
    core.need(result['generated_sha256']==sha(OUT/'generated.json') and result['model_sha256']==sha(OUT/'model.json') and result['proposal_sha256']==sha(LOG/'proposal.json'),'Result provenance differs')
    fr=[x for x in rows if x['split']=='fit']; X=np.array([x['estimate']['raw_uv']+[1] for x in fr],float); Y=np.array([x['warp_uv'] for x in fr],float)
    B=np.array(model['coefficients']); lhs=(X.T@X+np.diag([1.,1.,0.]))@B; rhs=X.T@Y
    core.need(np.allclose(lhs,rhs,rtol=1e-12,atol=1e-10),'Ridge normal-equation residual; no refit performed')
    eval_rows={x['key']:x for x in result['evaluation_rows']}; denominators={(x['scene'],x['frame']):x for x in result['frame_denominators']}
    independently_evaluated=[]; cal_cont=[]; cal_applied=[]; cal_res=[]; cal_rejected=0
    for split,records in data['records'].items():
        for r in records:
            frame=load_frame(data,r,labels=split=='evaluation')
            for delta in core.WARPS:
                key=key_for(split,r,delta); original=source_rows[key]
                obs,valid,sid=independent_warp(frame['depth'],frame['valid'],delta)
                value=budget.invoke('alignment_review','review/'+key,lambda:independent_estimate(frame['rgb'],obs,valid))
                close_numeric(original['estimate'],value,key+'/estimator')
                core.need(array_sha(sid)==original['source_id_sha256'],'Independent source-map hash mismatch')
                core.need(int(valid.sum())==original['observed_valid_count'] and int(frame['valid'].sum()-valid.sum())==original['lost_source_valid_count'],'Independent visibility denominator mismatch')
                continuous=np.array(value['raw_uv']+[1],float)@B
                applied=np.around(np.maximum(-4,np.minimum(4,continuous))).astype(np.int64) if value['usable'] else np.array([0,0])
                if split=='calibration':
                    cal_cont.append(continuous.tolist()); cal_applied.append(applied.tolist()); cal_res.append((applied-delta).tolist()); cal_rejected+=not value['usable']
                if split=='evaluation':
                    metrics,den=independent_evaluation(frame,delta,applied,value['usable'],sid)
                    row=dict(scene=r['scene'],frame=r['frame'],key=key,warp_uv=list(delta),continuous_uv=continuous.tolist(),applied_uv=applied.tolist(),
                        lost_source_valid_count=int(frame['valid'].sum()-valid.sum()),**metrics)
                    close_numeric(eval_rows[key],row,key+'/evaluation')
                    close_numeric(denominators[(r['scene'],r['frame'])],dict(scene=r['scene'],frame=r['frame'],**den),key+'/denominator')
                    independently_evaluated.append(row)
            budget.sample()
    cal=model['calibration']; close_numeric(cal_cont,cal['predicted_continuous_uv']); close_numeric(cal_applied,cal['applied_uv']); close_numeric(cal_res,cal['residual_uv'])
    radius=sorted(max(abs(x),abs(y)) for x,y in cal_res)[136]
    core.need(radius==cal['empirical_linf95_radius_px'] and cal_rejected==cal['rejected_cases'],'Calibration description mismatch')
    summary,gates=independent_summary(independently_evaluated); close_numeric(summary,result['summary']); close_numeric(gates,result['gates'])
    supported=all(gates.values())
    core.need(result['identifiability_supported']==supported and result['scientific_usable']==supported and result['P6_full_passed'] is False and result['uvz_calibrated'] is False,'Incorrect scientific conclusion')
    audit=audit_ledger(); core.need(audit['counts']=={k:LIMITS[k] for k in audit['counts']} and not audit['unknown_calls'],'Actual research ledger incomplete')
    verify_identities(read(LOG/'authorization.json')['protected'])
    outputs={rel(OUT/name):identity(OUT/name) for name in ('generated.json','model.json','result.json')}
    record=dict(passed=True,engineering_passed=True,identifiability_supported=supported,scientific_usable=supported,P6_full_passed=False,
        reviewed_outputs=outputs,counts=audit['counts'],unknown_calls=audit['unknown_calls'],
        checks=['independent observable81-score replay720','raw-source-map/visibility replay720','ridge normal equations without refit',
                'calibration description without tuning','independent evaluation144 and strata/denominators','separate summary/gates','frozen inputs unchanged'],
        actual_research_counts_note='720 generate +720 independent read-only estimator replay; one fit; no second fit/neural/AP.',at=now())
    target=LOG/'review.json'
    if target.exists():
        old=read(target); record['at']=old['at']; core.need(old==record,'Review record differs')
    else: once(target,record)
    budget.register(target)


def cold(data,budget):
    review_record=read(LOG/'review.json'); result=read(OUT/'result.json')
    core.need(review_record['passed'] and review_record['engineering_passed'],'Independent review not passed')
    verify_identities(review_record['reviewed_outputs'])
    audit=audit_ledger(); core.need(audit['counts']=={k:LIMITS[k] for k in audit['counts']} and not audit['unknown_calls'],'Cold ledger incomplete')
    core.need(result['identifiability_supported']==review_record['identifiability_supported'] and result['P6_full_passed'] is False,'Cold scientific scope differs')
    record=dict(passed=True,engineering_passed=True,identifiability_supported=result['identifiability_supported'],scientific_usable=result['scientific_usable'],
        P6_full_passed=False,review_sha256=sha(LOG/'review.json'),reviewed_outputs=review_record['reviewed_outputs'],
        counts=audit['counts'],unknown_calls=[],new_research_calls=0,process_id=os.getpid(),at=now())
    target=LOG/'cold.json'
    if target.exists():
        old=read(target); record['at']=old['at']; record['process_id']=old['process_id']; core.need(old==record,'Cold marker differs')
    else: once(target,record)
    budget.register(target)


def self_test():
    checks=[]; count=0; independent_count=0
    rng=np.random.RandomState(40); h,w=96,112
    raw=rng.uniform(.3,1.5,(h,w)); valid=np.ones((h,w),bool); valid[45:48,60:63]=False
    rgb=np.repeat(((raw-.3)/1.2)[:,:,None],3,axis=2)
    for delta in core.WARPS:
        z,ok,sid=core.warp_integer(raw,valid,delta); iz,iv,isid=independent_warp(raw,valid,delta)
        core.need(np.array_equal(z,iz) and np.array_equal(ok,iv) and np.array_equal(sid,isid),'Synthetic warp/source mismatch')
        e=core.estimate(rgb,z,ok); count+=1; independent=independent_estimate(rgb,z,ok); independent_count+=1
        close_numeric(pack_estimate(e),independent,'synthetic/estimator')
        core.need(e['usable'] and e['raw_uv'].tolist()==list(delta),'Synthetic directional alignment failed')
    checks.append('nine direction signs, exact source-id, no interpolation, independent score agreement')
    blank=core.estimate(np.zeros((h,w,3)),np.ones((h,w)),valid); count+=1
    core.need(not blank['usable'] and blank['raw_uv'].tolist()==[0,0],'Blank observation not rejected')
    artificial=np.ones((h,w)); artificial[:,0:3]=0
    pad=core.estimate(rgb,artificial,artificial>0); count+=1
    core.need(not pad['usable'] and pad['edge_count']==0,'Artificial image padding leaks into estimator')
    checks+=['blank rejection retained','fixed8px interior excludes artificial padding']
    rawshift=np.tile(np.array(core.WARPS), (3,1)); model=core.ridge_fit(rawshift,rawshift)
    design=np.column_stack([rawshift,np.ones(len(rawshift))]); B=np.array(model['coefficients'])
    core.need(np.allclose((design.T@design+np.diag([1.,1.,0.]))@B,design.T@rawshift),'Synthetic ridge normal equation')
    continuous,applied=core.predict(model,rawshift,np.ones(len(rawshift),bool)); core.need(np.array_equal(rawshift,applied),'Synthetic affine prediction')
    custom=dict(coefficients=[[0.,0.],[0.,0.],[.5,1.5]])
    _,rounding=core.predict(custom,np.zeros((2,2)),np.array([True,False])); core.need(rounding.tolist()==[[0,2],[0,0]],'Half-to-even or reject rule')
    checks+=['one six-parameter synthetic ridge check','clip/half-even/rejected prediction']
    labels=np.zeros((h,w),np.int64); labels[:,w//2:]=1; labels[h//2:]+=2
    frame=dict(depth=raw,valid=valid,eligible=valid,labels=labels,K=np.array([[100.,0.,55.],[0.,100.,47.],[0.,0.,1.]]),original_size=(w,h))
    ref=core.select_reference(raw,valid,valid,labels)
    for usable in (True,False):
        delta=(3,-3); _,_,sid=core.warp_integer(raw,valid,delta); applied=np.array(delta) if usable else np.array([0,0])
        first=core.evaluate_case(frame,ref,delta,applied,usable,sid); second,den=independent_evaluation(frame,delta,applied,usable,sid)
        close_numeric(first,second,'synthetic/evaluation')
        core.need(first['oracle_source_identity_exact'] and (usable or first['metrics']['boundary']['estimated_instance_identity_matches']==0),'Oracle/reject boundary semantics')
    checks.append('independent source correspondence/XYZ/GT-strata and reject denominator')
    synthetic_rows=[]
    for scene in SCENES['evaluation']:
        for frame_id in ('0000','0064','0128','0192'):
            for delta in core.WARPS:
                _,_,sid=core.warp_integer(raw,valid,delta)
                metrics=core.evaluate_case(frame,ref,delta,np.array(delta),True,sid)
                synthetic_rows.append(dict(scene=scene,frame=frame_id,warp_uv=list(delta),**metrics))
    ss,sg=core.summarize(synthetic_rows); ins,ing=independent_summary(synthetic_rows)
    close_numeric(ss,ins,'synthetic/summary'); close_numeric(sg,ing,'synthetic/gates')
    core.need(all(sg.values()) and ss['improvement_fraction']==1.,'Perfect synthetic correspondence must pass finite gates')
    synthetic_rows[0]['usable']=False
    core.need(not core.summarize(synthetic_rows)[1]['no_rejected_case'],'Rejected case silently dropped')
    checks.append('balanced144-case summary, Euclidean error gates, clean excluded from gain, rejection fails')
    core.need(list(inspect.signature(core.estimate).parameters)==['rgb','observed_depth','observed_valid'],'Forbidden estimator inputs')
    core.need(count<=12,'Synthetic estimator-check budget')
    result=dict(passed=True,checks=checks,synthetic_check_cases=count,synthetic_estimator_calls=count+independent_count,
        synthetic_primary_estimator_calls=count,synthetic_independent_estimator_calls=independent_count,synthetic_ridge_checks=1,formal_research_calls=0,
        real_image_decodes=0,neural_calls=0,runtime=runtime_identity(),source_sha256={k:sha(ROOT/k) for k in INHERITED},peak_rss_bytes=peak_rss(),at=now())
    target=LOG/'component_tests.json'
    if target.exists():
        backup=LOG/('component_tests.previous.'+uuid.uuid4().hex+'.json'); os.replace(target,backup)
    once(target,result)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','self-test','generate','fit','evaluate','review','cold']); args=ap.parse_args()
    core.need(Path.cwd().resolve()==ROOT,'Execute explicitly from project ROOT')
    LOG.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); budget=None; success=False
    try:
        with project_lock(ROOT):
            if args.stage in ('prepare','self-test'):
                answer=prepare() if args.stage=='prepare' else self_test()
                print(json.dumps(dict(stage=args.stage,passed=answer['passed']),ensure_ascii=False)); success=True
            else:
                _,data=authorized(); budget=Budget(args.stage)
                globals()[args.stage](data,budget); budget.close(); success=True
                print(json.dumps(dict(stage=args.stage,completed=True,counts=budget.ledger['counts']),ensure_ascii=False))
    finally:
        if args.stage in ('prepare','self-test'):
            path=LOG/'preparation_usage.json'; old=read(path) if path.exists() else dict(seconds=0.,attempts=[])
            elapsed=time.perf_counter()-started; old['seconds']+=elapsed
            old['attempts'].append(dict(stage=args.stage,seconds=elapsed,passed=success,at=now(),peak_rss_bytes=peak_rss()))
            replace_json(path,old)
        elif budget is not None and not success:
            budget.close()


if __name__=='__main__': main()
