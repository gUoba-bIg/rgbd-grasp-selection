"""Observed-image depth-only uncertainty: no invented pixel covariance."""
import hashlib
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from scipy.stats import chi2
from scripts.p6_common_v24 import ROOT,OLD,BASE,read,need,sha,name,array_sha
LEVELS=(.5,.8,.95)

def source(c,r,condition):
    meta=loadmat(ROOT/c['data_root']/r['files']['meta']);K=np.asarray(meta['intrinsic_matrix'],np.float64);factor=float(meta['factor_depth'].reshape(-1)[0]);need(K.shape==(3,3) and factor>0 and np.isfinite(K).all(),'Invalid camera')
    with Image.open(ROOT/c['data_root']/r['files']['depth']) as im:
        width,height=im.size;raw=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32)/factor
    original_valid=np.isfinite(raw)&(raw>0);z=np.where(original_valid,np.minimum(raw,2.)/2.,0).astype(np.float32)*2;out=z.copy();ok=original_valid.copy();h,w=z.shape
    seed=int(hashlib.sha256(f"P5v23|{c['seed']}|{c['quality_screen_epoch']}|{r['scene']}|{r['frame']}|{condition}".encode()).hexdigest()[:8],16);rng=np.random.RandomState(seed)
    if condition in ['shift3','joint']:
        out[:,3:]=z[:,:-3];out[:,:3]=0;ok[:,3:]=original_valid[:,:-3];ok[:,:3]=False
    if condition in ['noise5','joint']:
        out=np.where(ok,np.clip(out+rng.normal(0,.005,(h,w)).astype(np.float32),0,2),0).astype(np.float32);ok&=out>0
    if condition in ['missing15','joint']:
        ok&=rng.uniform(size=(h,w))>=.15;y=int(rng.randint(0,h-h//4+1));x=int(rng.randint(0,w-w//4+1));ok[y:y+h//4,x:x+w//4]=False
    out[~ok]=0;eligible=original_valid&ok&(raw<=2)
    return dict(reference=z,observed=out,eligible=eligible,original_valid=original_valid,observed_valid=ok,raw=raw,K=K,width=width,height=height,factor=factor,seed=seed)

def fixed(r,d):return OLD/'exports'/f'core_fixed_learned_{d}'/f"scene_{r['scene']:04d}_{r['frame']}.npz"
def quality(r,d):
    p=OLD/'quality_screen'/f"{d}_{r['scene']:04d}_{r['frame']}.npz"
    return p if p.exists() else BASE/'quality'/(name(r,d)+'.npz')
def load(c,r,d):
    mp=fixed(r,d);qp=quality(r,d);need(sha(mp)==read(mp.with_suffix('.json'))['cache_sha256'],'Fixed mask changed');need(sha(qp)==read(qp.with_suffix('.json'))['cache_sha256'],'Quality changed')
    with np.load(mp,allow_pickle=False) as m:instances=m['instances'].copy()
    with np.load(qp,allow_pickle=False) as q:heads={k:q[k].copy() for k in ['sigma','valid_probability','anomaly_probability']}
    need(all(v.shape==(480,640) and np.isfinite(v).all() for v in heads.values()),'Invalid quality maps');need(np.all(heads['sigma']>0),'Invalid positive sigma')
    for k in ['valid_probability','anomaly_probability']:need(np.all((heads[k]>=0)&(heads[k]<=1)),'Invalid probability')
    return instances,heads

def boundary(instances):
    out=np.zeros(instances.shape,bool)
    for axis in [0,1]:
        a=[slice(None)]*2;b=list(a);a[axis]=slice(None,-1);b[axis]=slice(1,None);a,b=tuple(a),tuple(b);change=instances[a]!=instances[b];out[a]|=change;out[b]|=change
    return binary_dilation(out,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool),iterations=1)

def select(mask,limit):
    candidates=np.flatnonzero(mask)
    return candidates[np.arange(min(limit,len(candidates)),dtype=np.int64)*len(candidates)//min(limit,len(candidates))] if len(candidates) else np.empty(0,np.int64)

def build(c,s,instances,q):
    edge=boundary(instances);main=select(s['eligible'],4096);extra=select(s['eligible']&edge,1024);idx=np.union1d(main,extra);need(len(main)>0,'No evaluable pixels');v,u=np.divmod(idx,640);oy=((2*v+1)*s['height'])//960;ox=((2*u+1)*s['width'])//1280;K=s['K'];ray=np.column_stack(((ox-K[0,2])/K[0,0],(oy-K[1,2])/K[1,1],np.ones(len(idx))));z=s['observed'].ravel()[idx].astype(np.float64);ref=s['reference'].ravel()[idx].astype(np.float64);sg=q['sigma'].ravel()[idx].astype(np.float64);cov=np.einsum('ni,nj,n->nij',ray,ray,sg**2)
    return dict(flat_index=idx,main=np.isin(idx,main),boundary=edge.ravel()[idx],instance=instances.ravel()[idx],original_uv=np.column_stack((ox,oy)),rays=ray,observed_xyz=z[:,None]*ray,reference_xyz=ref[:,None]*ray,covariance=cov,sigma=sg,weight=(q['valid_probability'].astype(np.float64)*(1-q['anomaly_probability'].astype(np.float64))).ravel()[idx],depth_observed=z,depth_reference=ref)

def stats(a):
    residual=a['depth_observed']-a['depth_reference'];sg=a['sigma'];ratio=(residual/sg)**2;xyz=(a['observed_xyz']-a['reference_xyz']);axis_var=np.diagonal(a['covariance'],axis1=1,axis2=2);active=axis_var>1e-18;diagonal=np.divide(xyz**2,axis_var,out=np.zeros_like(xyz),where=active).sum(1);rank=active.sum(1);result={}
    groups=dict(main=a['main'],boundary=a['boundary'],interior=a['main']&~a['boundary']&(a['instance']>0),background=a['main']&(a['instance']==0),oracle_noise_control=a['main']&(a['depth_reference']>.025)&(a['depth_reference']<1.975))
    for label,m in groups.items():
        rr=residual[m];ss=sg[m];n=int(m.sum());weights=a['weight'][m]
        result[label]=dict(n=n,covered=[int(np.count_nonzero(ratio[m]<=chi2.ppf(p,1))) for p in LEVELS],constant_covered=[int(np.count_nonzero((rr/.005)**2<=chi2.ppf(p,1))) for p in LEVELS],diagonal_covered95=int(np.count_nonzero(diagonal[m]<=chi2.ppf(.95,rank[m]))),squared_error3d=float((xyz[m]**2).sum()),signed_standardized=float((rr/ss).sum()),nll=float((.5*(rr/ss)**2+np.log(ss/.005)).sum()),constant_nll=float((.5*(rr/.005)**2).sum()),low_weight_count=int((weights<.5).sum()),weight_sum=float(weights.sum()),weighted_squared_error3d=float(((xyz[m]**2).sum(1)*weights).sum()))
    return result

def aggregate(rows):
    summaries={}
    for condition in ['clean','noise5','missing15','shift3','joint']:
        scenes=[]
        for sc in sorted({r['scene'] for r in rows}):
            rr=[r for r in rows if r['scene']==sc and r['condition']==condition];need(len(rr)==16,'Incomplete scene condition');groups={}
            for label in rr[0]['metrics']:
                vals=[r['metrics'][label] for r in rr];n=sum(x['n'] for x in vals)
                groups[label]=dict(n=n,coverage=[sum(x['covered'][j] for x in vals)/n if n else None for j in range(3)],constant_coverage=[sum(x['constant_covered'][j] for x in vals)/n if n else None for j in range(3)],diagonal_coverage95=sum(x['diagonal_covered95'] for x in vals)/n if n else None,rmse3d=(sum(x['squared_error3d'] for x in vals)/n)**.5 if n else None,standardized_bias=sum(x['signed_standardized'] for x in vals)/n if n else None,nll=sum(x['nll'] for x in vals)/n if n else None,constant_nll=sum(x['constant_nll'] for x in vals)/n if n else None,would_remove_fraction=sum(x['low_weight_count'] for x in vals)/n if n else None,actually_removed=0)
            scenes.append(dict(scene=sc,groups=groups,eligible_pixels=sum(x['coverage_counts']['eligible'] for x in rr),original_valid_pixels=sum(x['coverage_counts']['original_valid'] for x in rr),observed_valid_pixels=sum(x['coverage_counts']['observed_valid'] for x in rr),clipped_original_pixels=sum(x['coverage_counts']['original_clipped'] for x in rr)))
        macro={}
        for label in scenes[0]['groups']:
            vv=[s['groups'][label] for s in scenes];need(all(v['n']>0 for v in vv),'Empty diagnostic scene stratum '+label)
            macro[label]={key:( [float(np.mean([v[key][j] for v in vv])) for j in range(3)] if key in ['coverage','constant_coverage'] else float(np.mean([v[key] for v in vv]))) for key in ['coverage','constant_coverage','diagonal_coverage95','rmse3d','standardized_bias','nll','constant_nll','would_remove_fraction']}
        summaries[condition]=dict(scenes=scenes,macro=macro)
    noisy=['noise5','shift3','joint'];main=[summaries[d]['macro']['main']['coverage'][2] for d in noisy];edge=[summaries[d]['macro']['boundary']['coverage'][2] for d in noisy];worst=min(s['groups']['main']['coverage'][2] for d in noisy for s in summaries[d]['scenes']);gain=float(np.mean([summaries[d]['macro']['main']['constant_nll']-summaries[d]['macro']['main']['nll'] for d in summaries if d!='clean']))
    engineering=all(.90<=s['groups']['oracle_noise_control']['constant_coverage'][2]<=.98 for s in summaries['noise5']['scenes'])
    gates=dict(noisy_95_coverage=all(.90<=v<=.98 for v in main),boundary_95_coverage=all(v>=.85 for v in edge),worst_scene_95_coverage=worst>=.80,nll_improvement=gain>=.05)
    return dict(conditions=summaries,positive_control_passed=engineering,scientific_gates=gates,raw_sigma_ready=all(gates.values()),mean_degraded_nll_improvement=gain,worst_scene_coverage95=worst,scope='Rank1 depth-only known perturbation on exposed images; not real sensor calibration or full pixel/depth uncertainty')
