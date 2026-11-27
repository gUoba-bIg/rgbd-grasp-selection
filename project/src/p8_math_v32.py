"""Label-free ranking and explicitly scoped GraspNet AP / ECE15 metrics."""
import numpy as np
from scipy.special import expit
REMOVED={'no_quality':[5,6,7,8],'no_instance':[6,9,10,11],'full':[]}
def logits(x,weights,mean,scale,removed=()):
 a=((x-mean)/scale).astype(np.float64);a[:,list(removed)]=0
 for k in ['0','2','4']:
  a=a@weights[k+'.weight'].astype(np.float64).T+weights[k+'.bias']
  if k!='4':a=np.maximum(a,0)
 return a.reshape(-1)
def ranked_candidates(candidates,eligible,scores):
 indices=np.flatnonzero(eligible);order=np.lexsort((indices,-np.asarray(scores)[indices]));indices=indices[order];a=candidates[indices].copy()
 # Exact unique ordinal confidence preserves method ordering and deterministic ties.
 a[:,0]=np.arange(len(a),0,-1,dtype=np.float32);return a,indices
def grasp_ap(scores,collisions):
 s=np.asarray(scores);co=np.asarray(collisions,bool);success=(s[:,None]>0)&(s[:,None]<=np.array([.2,.4,.6,.8,1.,1.2])[None])&~co[:,None]
 prefix=np.zeros((50,6),np.float64);n=min(len(s),50)
 if n:prefix[:n]=np.cumsum(success[:n],axis=0);prefix[n:]=prefix[n-1]
 prefix/=np.arange(1,51)[:,None]
 return dict(AP=float(prefix.mean()),AP_04=float(prefix[:,1].mean()),prefix=prefix.tolist(),selected=n)
def probability(y,z):
 y=np.asarray(y,np.float64);z=np.asarray(z,np.float64);n=len(y)
 if not n:return dict(n=0,positives=0,nll=None,brier=None,ece15=None,bins=[],aurc=None)
 p=expit(z);order=np.argsort(p,kind='stable');bins=[];ece=0.
 for ix in np.array_split(order,15):
  nn=len(ix);conf=float(p[ix].mean()) if nn else None;freq=float(y[ix].mean()) if nn else None
  if nn:ece+=nn/n*abs(conf-freq)
  bins.append(dict(n=nn,confidence=conf,frequency=freq))
 risk_order=np.argsort(-p,kind='stable');coverage=np.cumsum(1-y[risk_order])/np.arange(1,n+1)
 return dict(n=n,positives=int(y.sum()),nll=float((np.logaddexp(0,z)-y*z).mean()),brier=float(np.mean((p-y)**2)),ece15=float(ece),bins=bins,aurc=float(coverage.mean()))
def input_sample(a,seed,scene):
 from src.risk_observations_v31 import points
 cloud,pixels=points(a['observed_depth'],a['observed_valid'],a['K'],a['original_size']);fg=a['instances'].reshape(-1)[pixels]>0;cloud=cloud[fg];pixels=pixels[fg]
 if len(cloud)<1024:return cloud[:0],pixels[:0],len(cloud)
 ix=np.random.RandomState(seed+scene*1000+64).choice(len(cloud),20000,replace=len(cloud)<20000)
 return cloud[ix],pixels[ix],len(cloud)
