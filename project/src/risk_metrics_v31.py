"""Fixed offline-event metrics, preserving empty frames and equal scene weight."""
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

def probability_metrics(y,logits):
 y=np.asarray(y,np.float64);z=np.asarray(logits,np.float64);p=expit(z)
 if not len(y):return dict(n=0,positives=0,nll=None,brier=None,ece=None,auroc=None,pr_auc=None,bins=[])
 bins=[];ece=0.
 for i in range(10):
  mask=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1);n=int(mask.sum());conf=float(p[mask].mean()) if n else None;acc=float(y[mask].mean()) if n else None
  if n:ece+=n/len(y)*abs(acc-conf)
  bins.append(dict(lower=i/10,upper=(i+1)/10,n=n,confidence=conf,frequency=acc))
 pos=int(y.sum());neg=len(y)-pos;auc=float((rankdata(z)[y==1].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
 order=np.argsort(-z,kind='stable');ys=y[order];zs=z[order];ends=np.r_[np.flatnonzero(np.diff(zs)!=0),len(zs)-1];tp=np.cumsum(ys)[ends];rec=tp/pos if pos else np.zeros(len(tp));ap=float(np.sum(np.diff(np.r_[0.,rec])*tp/(ends+1))) if pos else None
 return dict(n=len(y),positives=pos,nll=float(np.mean(np.logaddexp(0,z)-y*z)),brier=float(np.mean((p-y)**2)),ece=float(ece),auroc=auc,pr_auc=ap,bins=bins)

def ranking(y,score,spans):
 frame=[]
 for s in spans:
  a,b=s['start'],s['end'];order=np.argsort(-score[a:b],kind='stable');labels=y[a:b][order];frame.append(dict(scene=s['scene'],condition=s['condition'],**{f'p{k}':float(labels[:k].sum()/k) for k in [1,5,10]}))
 scenes=sorted({r['scene'] for r in frame});by_scene=[]
 for scene in scenes:
  rs=[r for r in frame if r['scene']==scene];clean=[r for r in rs if r['condition']=='clean'];by_scene.append(dict(scene=scene,**{f'p{k}':float(np.mean([r[f'p{k}'] for r in rs])) for k in [1,5,10]},clean_p5=float(np.mean([r['p5'] for r in clean]))))
 return dict(**{f'p{k}':float(np.mean([r[f'p{k}'] for r in by_scene])) for k in [1,5,10]},clean_p5=float(np.mean([r['clean_p5'] for r in by_scene])),scenes=by_scene,frames=frame)
