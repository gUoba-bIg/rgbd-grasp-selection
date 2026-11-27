"""CPU-only independent raw-depth target reconstruction and quality checks."""
import sys,math,hashlib,time,uuid,os
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23 import *
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.stats import rankdata

def reference(c,r,condition):
    path=ROOT/c['data_root'];meta=loadmat(path/r['files']['meta']);factor=float(meta['factor_depth'].reshape(-1)[0])
    with Image.open(path/r['files']['depth']) as im:z=np.array(im.resize((640,480),Image.Resampling.NEAREST),dtype=np.float32)/factor
    v=np.isfinite(z)&(z>0);z=np.where(v,np.minimum(z,2.)/2.,0).astype(np.float32)*2;out=z.copy();ok=v.copy();h,w=z.shape
    seed=int(hashlib.sha256(f"P5v23|{c['seed']}|{c['quality_screen_epoch']}|{r['scene']}|{r['frame']}|{condition}".encode()).hexdigest()[:8],16);rng=np.random.RandomState(seed)
    if condition in ['shift3','joint']:out=np.concatenate([np.zeros((h,3),np.float32),z[:,:-3]],axis=1);ok=np.concatenate([np.zeros((h,3),bool),v[:,:-3]],axis=1)
    if condition in ['noise5','joint']:
        n=rng.normal(0,.005,(h,w)).astype(np.float32);out=np.where(ok,np.maximum(0,np.minimum(2,out+n)),0).astype(np.float32);ok=ok&(out>0)
    if condition in ['missing15','joint']:
        missing=rng.uniform(size=(h,w))<.15;ok=ok&~missing;y=rng.randint(0,h-h//4+1);x=rng.randint(0,w-w//4+1);ok[y:y+h//4,x:x+w//4]=False
    out[~ok]=0;res=out-z;mask=v&ok;edge=np.zeros((h,w),bool)
    for axis in [0,1]:
        left=[slice(None)]*2;right=[slice(None)]*2;left[axis]=slice(None,-1);right[axis]=slice(1,None);a,b=tuple(left),tuple(right);jump=(np.abs(z[a]-z[b])>.02)&v[a]&v[b];edge[a]|=jump;edge[b]|=jump
    return dict(q_valid=ok.astype(np.float32)[None],q_residual=res[None],q_supervised=mask.astype(np.float32)[None],q_anomaly=(edge&mask&(np.abs(res)>.01)).astype(np.float32)[None])

def tie_ap(y,s):
    y=np.asarray(y,dtype=int);s=np.asarray(s);n=y.sum()
    if not n:return None
    # Group exact score ties before descending cumulative counts, independent of producer's ranking path.
    vals,idx=np.unique(s,return_inverse=True);count=np.bincount(idx);positive=np.bincount(idx,weights=y);tp=np.cumsum(positive[::-1]);total=np.cumsum(count[::-1]);return float(np.sum(positive[::-1]/n*tp/total))

def run():
    c=verify();original=read(BASE/'quality_screen.json');by={};snap={};checked=0
    for condition in c['core_conditions']:
        for r in c['monitor_records']:
            stem=f"{condition}_{r['scene']:04d}_{r['frame']}";cache=BASE/'quality_screen'/(stem+'.npz');rp=cache.with_suffix('.json');row=read(rp);need(sha(cache)==row['cache_sha256'],'Cache digest failed');snap[cache.relative_to(ROOT).as_posix()]=stat(cache);snap[rp.relative_to(ROOT).as_posix()]=stat(rp)
            targets=reference(c,r,condition)
            with np.load(cache) as a:
                need(all(np.array_equal(a[k],v) for k,v in targets.items()),'Independent target reconstruction differs '+stem)
                vp=a['valid_probability'].astype(np.float64);sg=a['sigma'].astype(np.float64);ap=a['anomaly_probability'];valid=targets['q_valid'][0];mask=targets['q_supervised'][0].astype(bool);res=targets['q_residual'][0].astype(np.float64);label=targets['q_anomaly'][0].astype(bool)
                need(np.isfinite(vp).all() and np.isfinite(sg).all() and np.isfinite(ap).all() and (sg>0).all(),'Nonfinite cached quality')
                d=by.setdefault(r['scene'],dict(brier=0.,pixels=0,nll=0.,constant=0.,n=0,x=[],y=[],scores=[],labels=[]));d['brier']+=float(((vp-valid)**2).sum());d['pixels']+=valid.size;d['nll']+=float((.5*(res[mask]/sg[mask])**2+np.log(sg[mask]/.005)).sum());d['constant']+=float((.5*(res[mask]/.005)**2).sum());d['n']+=int(mask.sum())
                for y in range(0,480,16):
                    for x in range(0,640,16):
                        m=mask[y:y+16,x:x+16]
                        if m.sum()>=32:d['x'].append(float(sg[y:y+16,x:x+16][m].mean()*math.sqrt(2/math.pi)));d['y'].append(float(np.abs(res[y:y+16,x:x+16][m]).mean()))
                d['scores'].extend(ap[::4,::4].ravel().tolist());d['labels'].extend(label[::4,::4].ravel().tolist())
            checked+=1
    scenes=[]
    for sc,d in sorted(by.items()):
        rho=float(np.corrcoef(rankdata(d['x']),rankdata(d['y']))[0,1]) if np.ptp(d['x']) and np.ptp(d['y']) else None
        scenes.append(dict(scene=sc,valid_brier=d['brier']/d['pixels'],mean_nll=d['nll']/d['n'],constant_nll=d['constant']/d['n'],patch_spearman=rho,anomaly_AP_fixed_stride4=tie_ap(d['labels'],d['scores']),anomaly_prevalence=sum(d['labels'])/len(d['labels'])))
    macro={k:float(np.mean([s[k] for s in scenes])) if all(s[k] is not None for s in scenes) else None for k in original['scene_macro']}
    # Float32 producer and float64 independent arithmetic can move tied patch ranks slightly.
    tolerances={k:(.001 if k=='patch_spearman' else 1e-5) for k in macro}
    for k,v in macro.items():need((v is None and original['scene_macro'][k] is None) or (v is not None and abs(v-original['scene_macro'][k])<=tolerances[k]*max(1,abs(v))),'Independent metric differs '+k)
    gates=dict(validity=macro['valid_brier']<=.05,error_correlation=macro['patch_spearman'] is not None and macro['patch_spearman']>=.2,error_nll=macro['constant_nll']-macro['mean_nll']>=.05,boundary_anomaly=macro['anomaly_AP_fixed_stride4'] is not None and macro['anomaly_AP_fixed_stride4']>max(.02,1.2*macro['anomaly_prevalence']))
    need(gates==original['gates'],'Independent adoption decisions differ')
    for p,s in snap.items():need(stat(ROOT/p)==s,'Review mutated input '+p)
    out=dict(passed=True,reviewed_frames=checked,independent_macro=macro,gates=gates,quality_usable=all(gates.values()),all_source_bytes_sizes_mtimes_unchanged=True,new_model_calls=0,review_source=stat(Path(__file__)),tolerances=tolerances)
    once(LOG/'quality_review.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':
    start=time.monotonic();uid=uuid.uuid4().hex;outcome='failed'
    try:run();outcome='passed'
    finally:once(LOG/('usage_'+uid+'.json'),dict(at=now(),seconds=time.monotonic()-start,state=outcome,phase='quality_independent_review'))
