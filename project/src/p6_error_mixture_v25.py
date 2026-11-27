"""Conditional signed depth-error mixture; inference consumes observations only."""
import numpy as np
from scipy.ndimage import median_filter, maximum_filter, minimum_filter, uniform_filter
from scipy.special import ndtr, logsumexp

def observable_cells(z, valid, sigma):
    median = median_filter(z, size=3, mode='nearest')
    hi = maximum_filter(np.where(valid, z, -np.inf), size=7, mode='nearest')
    lo = minimum_filter(np.where(valid, z, np.inf), size=7, mode='nearest')
    spread = np.where(valid, hi-lo, 0.)
    dense = uniform_filter(valid.astype(float), size=3, mode='nearest') > .999
    sample = dense & valid & (spread < .02)
    noise = float(np.median(np.abs(z[sample]-median[sample]))) if sample.any() else 0.
    rough = np.searchsorted([.005,.02,.08], spread, side='right')
    quality = np.searchsorted([.003,.01], sigma, side='right')
    sign = np.searchsorted([-.002,.002], z-median, side='right')
    cells = (((int(noise>.0015)*4+rough)*3+quality)*3+sign).astype(np.int16)
    return cells, noise

def fit_one(y, iterations=30):
    y = np.asarray(y, dtype=np.float64)
    means = np.array([np.median(y),np.quantile(y,.05),np.quantile(y,.95)])
    scales = np.array([.002,.02,.02]); weights=np.array([.90,.05,.05])
    trace=[]
    for _ in range(iterations):
        ll = np.log(weights)[None,:]-np.log(scales)[None,:]-.5*((y[:,None]-means)/scales)**2-.5*np.log(2*np.pi)
        total=logsumexp(ll,axis=1); trace.append(float(total.mean()))
        responsibility=np.exp(ll-total[:,None]); n=responsibility.sum(0)+1e-12
        means=(responsibility*y[:,None]).sum(0)/n
        scales=np.sqrt((responsibility*(y[:,None]-means)**2).sum(0)/n).clip(.0001,.5)
        weights=np.maximum(n/n.sum(),1e-6);weights/=weights.sum()
    order=np.argsort(means)
    return dict(weights=weights[order].tolist(),means=means[order].tolist(),scales=scales[order].tolist(),n=len(y),iterations=iterations,log_likelihood_trace=trace)

def fit(cells, residual, guard=lambda:None):
    pools={}
    for k in [0,1]:
        guard()
        if np.any(cells//36==k):pools[str(k)]=fit_one(residual[cells//36==k])
    guard()
    global_pool=fit_one(residual)
    models={}
    for k in range(72):
        guard()
        y=residual[cells==k]
        if len(y)>=256: models[str(k)]=dict(fit_one(y),fallback=None)
        else: models[str(k)]=dict(pools.get(str(k//36),global_pool),fallback='noise_pool' if str(k//36) in pools else 'global_pool',cell_n=len(y))
    return dict(family='three_component_signed_error_Gaussian_mixture',models=models,pooled=pools,global_pool=global_pool,feature_count=72,condition_name_used_as_feature=False,iterations_per_fit=30)

def predict(model,cells,residual):
    table=model['models'];w=np.array([table[str(k)]['weights'] for k in range(72)])[cells]
    m=np.array([table[str(k)]['means'] for k in range(72)])[cells]
    s=np.array([table[str(k)]['scales'] for k in range(72)])[cells]
    standardized=(residual[:,None]-m)/s
    pit=np.sum(w*ndtr(standardized),axis=1)
    logpdf=logsumexp(np.log(w)-np.log(s)-.5*standardized**2-.5*np.log(2*np.pi),axis=1)
    mean=np.sum(w*m,axis=1);variance=np.sum(w*(s*s+(m-mean[:,None])**2),axis=1)
    return dict(pit=pit,logpdf=logpdf,mean=mean,variance=variance)

def metrics(a,p):
    err=a['observed']-a['reference'];corrected=err-p['mean'];out={}
    for label,mask in dict(main=a['main'],boundary=a['boundary']).items():
        t=p['pit'][mask];r=err[mask];n=int(mask.sum())
        out[label]=dict(n=n,covered=[int(np.sum((t>=(1-q)/2)&(t<=(1+q)/2))) for q in [.5,.8,.95]],nll=float(np.sum(-p['logpdf'][mask]-np.log(.005*np.sqrt(2*np.pi)))),constant_nll=float(np.sum(.5*(r/.005)**2)),squared_error3d=float(np.sum((corrected[mask,None]*a['rays'][mask])**2)),raw_squared_error3d=float(np.sum((r[:,None]*a['rays'][mask])**2)),mean_error=float(corrected[mask].sum()))
    return out

def aggregate(rows):
    conditions={}
    for d in ['clean','noise5','missing15','shift3','joint']:
        scenes=[]
        for sc in sorted({x['scene'] for x in rows}):
            subset=[x for x in rows if x['scene']==sc and x['condition']==d];groups={}
            if not subset:raise ValueError('Missing scene condition')
            for group in ['main','boundary']:
                rr=[x['metrics'][group] for x in subset];n=sum(x['n'] for x in rr)
                if n==0:raise ValueError('Empty stratum')
                groups[group]=dict(n=n,coverage=[sum(x['covered'][j] for x in rr)/n for j in range(3)],nll=sum(x['nll'] for x in rr)/n,constant_nll=sum(x['constant_nll'] for x in rr)/n,rmse3d=np.sqrt(sum(x['squared_error3d'] for x in rr)/n),raw_rmse3d=np.sqrt(sum(x['raw_squared_error3d'] for x in rr)/n),bias=sum(x['mean_error'] for x in rr)/n)
            scenes.append(dict(scene=sc,groups=groups))
        macro={}
        for group in ['main','boundary']:
            macro[group]={k:np.mean([s['groups'][group][k] for s in scenes],axis=0).tolist() for k in ['coverage','nll','constant_nll','rmse3d','raw_rmse3d','bias']}
        conditions[d]=dict(scenes=scenes,macro=macro)
    noisy=['noise5','shift3','joint'];worst=min(s['groups']['main']['coverage'][2] for d in noisy for s in conditions[d]['scenes'])
    gain=float(np.mean([conditions[d]['macro']['main']['constant_nll']-conditions[d]['macro']['main']['nll'] for d in conditions if d!='clean']))
    gates=dict(noisy_95_coverage=all(.90<=conditions[d]['macro']['main']['coverage'][2]<=.98 for d in noisy),boundary_95_coverage=all(conditions[d]['macro']['boundary']['coverage'][2]>=.85 for d in noisy),worst_scene_95_coverage=worst>=.80,nll_improvement=gain>=.05)
    return dict(conditions=conditions,scientific_gates=gates,ready=all(gates.values()),worst_scene_coverage95=worst,mean_degraded_nll_improvement=gain,coverage_definition='central predictive mixture probability; not a Gaussian covariance ellipsoid',cases=len(rows))
