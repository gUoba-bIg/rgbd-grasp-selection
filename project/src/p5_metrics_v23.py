"""Pure CPU quality summaries: scene macro statistics and fixed-grid anomaly AP."""
import math
import numpy as np
from scipy.stats import spearmanr

def ap_binary(labels,scores):
    y=np.asarray(labels,dtype=bool);s=np.asarray(scores,dtype=np.float64);positive=int(y.sum())
    if not positive:return None
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order];ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(y)[ends].astype(float);fp=ends+1-tp;recall=tp/positive;precision=tp/(tp+fp)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))

def diagnostics(valid_prob,sigma,anomaly_prob,t):
    valid=np.asarray(t['q_valid']).squeeze();mask=np.asarray(t['q_supervised']).squeeze().astype(bool);r=np.asarray(t['q_residual']).squeeze();y=np.asarray(t['q_anomaly']).squeeze().astype(bool)
    vp=np.asarray(valid_prob).squeeze();sg=np.asarray(sigma).squeeze();ap=np.asarray(anomaly_prob).squeeze()
    assert all(np.isfinite(a).all() for a in [vp,sg,ap,r]) and (sg>0).all()
    h,w=mask.shape;assert h%16==w%16==0
    def blocks(a):return a.reshape(h//16,16,w//16,16).sum((1,3))
    n=blocks(mask);eligible=n>=32
    predicted=blocks(sg*math.sqrt(2/math.pi)*mask)/np.maximum(n,1);observed=blocks(np.abs(r)*mask)/np.maximum(n,1)
    nll=.5*np.square(r[mask]/sg[mask])+np.log(sg[mask]/.005);constant=.5*np.square(r[mask]/.005)
    return dict(valid_brier_sum=float(np.square(vp-valid,dtype=np.float64).sum()),pixels=int(valid.size),supervised=int(mask.sum()),nll_sum=float(nll.sum(dtype=np.float64)),constant_nll_sum=float(constant.sum(dtype=np.float64)),patch_pred=predicted[eligible].astype(float).tolist(),patch_error=observed[eligible].astype(float).tolist(),anomaly_scores=ap[::4,::4].flatten().astype(float).tolist(),anomaly_labels=y[::4,::4].flatten().astype(int).tolist())

def summary(rows):
    scenes=[]
    for scene in sorted({r['scene'] for r in rows}):
        sub=[r['diagnostics'] for r in rows if r['scene']==scene];n=sum(d['supervised'] for d in sub);p=sum(d['pixels'] for d in sub)
        x=[x for d in sub for x in d['patch_pred']];y=[x for d in sub for x in d['patch_error']]
        rho=float(spearmanr(x,y).statistic) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None
        scores=[x for d in sub for x in d['anomaly_scores']];labels=[x for d in sub for x in d['anomaly_labels']]
        scenes.append(dict(scene=scene,valid_brier=sum(d['valid_brier_sum'] for d in sub)/p,mean_nll=sum(d['nll_sum'] for d in sub)/n if n else None,constant_nll=sum(d['constant_nll_sum'] for d in sub)/n if n else None,patch_spearman=rho,anomaly_AP_fixed_stride4=ap_binary(labels,scores),anomaly_prevalence=sum(labels)/len(labels)))
    def mean(k):return float(np.mean([s[k] for s in scenes])) if all(s[k] is not None for s in scenes) else None
    macro={k:mean(k) for k in ['valid_brier','mean_nll','constant_nll','patch_spearman','anomaly_AP_fixed_stride4','anomaly_prevalence']}
    gates=dict(validity=macro['valid_brier']<=.05,error_correlation=macro['patch_spearman'] is not None and macro['patch_spearman']>=.2,error_nll=macro['mean_nll'] is not None and macro['constant_nll']-macro['mean_nll']>=.05,boundary_anomaly=macro['anomaly_AP_fixed_stride4'] is not None and macro['anomaly_AP_fixed_stride4']>max(.02,1.2*macro['anomaly_prevalence']))
    return dict(frames=len(rows),scenes=scenes,scene_macro=macro,gates=gates,quality_usable=all(gates.values()),scope='Known synthetic perturbation relative to original valid depth; no true sensor-error calibration; anomaly AP uses fixed stride4 grid, ties grouped.')
