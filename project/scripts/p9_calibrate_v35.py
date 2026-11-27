"""Identical float64 temperature arithmetic for each prespecified new seed."""
import os,sys,hashlib,math
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p9_common_v35 import *
from scripts.p8_temperature_experiment_v34 import objective,independent,stationary
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def fit(seed,name,z,y,scenes,b):
 import numpy as np
 from scipy.optimize import minimize_scalar
 p=seedpath(seed)/'temperature_fits'/(name+'.json');binding=dict(z=array_sha(z),y=array_sha(y),scenes=array_sha(scenes));need(enough(y),'Insufficient temperature class support')
 if p.exists():r=load(p);need(r['binding']==binding and r['success'],'Prior temperature fit mismatch');return r
 def solve():
  trace=[]
  def f(t):
   b.guard();need(len(trace)<100,'Per-fit temperature budget');v,cid=invoke('temperature_objective',lambda:objective(t,z,y),seed=seed,fit=name,logT=float(t));trace.append(dict(logT=float(t),value=v,call_id=cid));return v
  r=minimize_scalar(f,bounds=(np.log(.05),np.log(20.)),method='bounded',options=dict(maxiter=100,xatol=1e-6));return dict(logT=float(r.x),T=float(np.exp(r.x)),value=float(r.fun),success=bool(r.success),nfev=int(r.nfev),trace=trace)
 r,cid=invoke('temperature_fit',solve,seed=seed,fit=name);r.update(call_id=cid,binding=binding,training_scenes=sorted(set(scenes.tolist())),support=support(y));sealed(p,r);need(r['success'] and r['nfev']==len(r['trace']),'Temperature solver failure');return r
def apply(seed,name,T,z):
 import numpy as np
 p=seedpath(seed)/'temperature_applications'/(name+'.json')
 if p.exists():r=load(p);need(r['T']==T and r['input_sha256']==array_sha(z),'Existing temperature input mismatch');return np.asarray(r['logits'])
 q,cid=invoke('temperature_apply',lambda:z/T,seed=seed,application=name,n=len(z));need(np.isfinite(q).all(),'Invalid calibrated logits');sealed(p,dict(T=T,input_sha256=array_sha(z),logits=q.tolist(),call_id=cid));return q
def run():
 import numpy as np
 from src.p8_math_v32 import probability
 c=verify();b=Budget(c,'metrics');load(BASE/'training_complete.json');cal=pool('calibration');val=pool('validation');need(set(cal['scenes']).isdisjoint(val['scenes']),'Calibration validation overlap')
 for seed in c['seeds']:
  out=seedpath(seed)
  if (out/'calibration.json').exists():load(out/'calibration.json');continue
  with np.load(artifact(out/'predictions/calibration.npz')) as a:z=a['logits'].astype(float)
  y=cal['labels'].astype(float);q=np.empty_like(z);folds=[]
  for scene in c['groups']['calibration']:
   held=cal['scenes']==scene;train=~held;name=f'held_{scene:04d}';r=fit(seed,name,z[train],y[train],cal['scenes'][train],b);need(scene not in r['training_scenes'],'Held-out leakage');q[held]=apply(seed,name,r['T'],z[held]);folds.append(dict(scene=scene,n=int(held.sum()),T=r['T']));status('temperature_holdout',seed=seed,completed=len(folds),total=15)
  raw=probability(y,z);oof=probability(y,q);passed=enough(y) and oof['nll']<=raw['nll']+.001;prob={};T=None
  if passed:
   T=fit(seed,'final',z,y,cal['scenes'],b)['T']
   for split,d in [('calibration',cal),('validation',val)]:
    with np.load(artifact(out/'predictions'/f'{split}.npz')) as a:zz=a['logits'].astype(float)
    qq=apply(seed,split,T,zz);need(np.array_equal(np.argsort(zz,kind='stable'),np.argsort(qq,kind='stable')),'Positive temperature changed ranking');prob[split]=dict(raw=probability(d['labels'],zz),calibrated=probability(d['labels'],qq))
  sealed(out/'calibration.json',dict(at=now(),seed=seed,screen_passed=passed,raw=raw,held_out=oof,folds=folds,held_logits=q.tolist(),temperature=T,probability=prob,validation_used_in_calibration=False))
 b.save('complete');sealed(BASE/'calibration_complete.json',dict(at=now(),seeds=c['seeds']))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
