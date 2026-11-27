"""Independent raw coordinates, feature cells, mixture CDF/density and geometry."""
import os,sys,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v25 import *
from scripts.pipeline import project_lock
from scripts.p6_accept_v24 import reference,ref_boundary,ref_select,compare
from scripts.accept_mask_ap_v17 import independent_decode
from src import p6_image_geometry_v24 as oldgeo
from scipy.ndimage import median_filter,maximum_filter,minimum_filter,uniform_filter
from scipy.special import erf
import numpy as np

def cells_reference(z,ok,sigma):
 md=median_filter(z,3,mode='nearest');lo=minimum_filter(np.where(ok,z,np.inf),7,mode='nearest');hi=maximum_filter(np.where(ok,z,-np.inf),7,mode='nearest');spread=np.where(ok,hi-lo,0)
 interior=uniform_filter(ok.astype(np.float64),3,mode='nearest')>.999;eligible=interior&ok&(spread<.02);noise=float(np.quantile(np.abs(z-md)[eligible],.5)) if eligible.any() else 0.
 spread=spread.astype(np.float64);sigma=sigma.astype(np.float64);rough=(spread>=.005).astype(int)+(spread>=.02)+(spread>=.08);q=(sigma>=.003).astype(int)+(sigma>=.01);delta=(z-md).astype(np.float64);sgn=(delta>=-.002).astype(int)+(delta>=.002)
 return (36*int(noise>.0015)+9*rough+3*q+sgn).astype(np.int16),noise

def source_reference(c,r,d,split):
 z,obs,valid,ok,raw,K,xx,yy=invoke('corruption_replay',lambda:reference(c,r,d),case=name(r,d),phase='independent_'+split)
 if split=='development':
  with np.load(oldgeo.quality(r,d)) as f:sigma=f['sigma'].copy()
  with np.load(oldgeo.fixed(r,d)) as f:mask=f['instances'].copy()
 else:
  op=BASE/'observations'/split/(name(r,d)+'.npz');data,row=load_sample(op);need(row['observed_sha256']==array_sha(obs),'Observed raw reconstruction differs');sigma=data['sigma'];mask=data['instances']
  if split=='holdout':
   m=independent_decode(**{k:data[k] for k in ['foreground_probability','center_probability','offsets']},decoder=c['decoder'])['instances'];need(np.array_equal(m,mask),'Independent mask decoder differs')
 edge=ref_boundary(mask);eligible=valid&ok&(raw<=2);main=ref_select(eligible,4096);extra=ref_select(eligible&edge,1024);idx=np.array(sorted(set(main)|set(extra)),np.int64);v,u=divmod(idx,640);rays=np.column_stack(((xx[u]-K[0,2])/K[0,0],(yy[v]-K[1,2])/K[1,1],np.ones(len(idx))));cell,noise=cells_reference(obs,ok,sigma)
 return dict(flat_index=idx,cell=cell.ravel()[idx],main=np.isin(idx,main),boundary=edge.ravel()[idx],instance=mask.ravel()[idx],rays=rays,observed=obs.ravel()[idx].astype(float),reference=z.ravel()[idx].astype(float),sigma_old=sigma.ravel()[idx].astype(float)),noise

def density_reference(model,a):
 r=a['observed']-a['reference'];n=len(r);cdf=np.zeros(n);mean=np.zeros(n);terms=[];second=np.zeros(n)
 for j in range(3):
  weight=np.array([model['models'][str(k)]['weights'][j] for k in a['cell']]);mu=np.array([model['models'][str(k)]['means'][j] for k in a['cell']]);s=np.array([model['models'][str(k)]['scales'][j] for k in a['cell']]);t=(r-mu)/s
  cdf+=weight*(1+erf(t/np.sqrt(2)))/2;terms.append(np.log(weight)-np.log(s)-np.log(2*np.pi)/2-t*t/2);mean+=weight*mu;second+=weight*(s*s+mu*mu)
 stack=np.array(terms);mx=stack.max(0);logpdf=mx+np.log(np.exp(stack-mx).sum(0));variance=second-mean*mean
 covariance=variance[:,None,None]*a['rays'][:,:,None]*a['rays'][:,None,:];eigen=np.linalg.eigvalsh(covariance);need(np.all(eigen[:,0]>=-1e-12*eigen[:,-1]) and np.all(eigen[:,-1]>0),'Invalid covariance PSD');need(np.all(np.sum(eigen>np.maximum(eigen[:,-1,None]*1e-12,1e-18),axis=1)==1),'Covariance rank not one')
 pred=dict(pit=cdf,logpdf=logpdf,mean=mean,variance=variance,corrected_xyz=(a['observed']-mean)[:,None]*a['rays'],covariance=covariance)
 metrics={}
 for group in ['main','boundary']:
  mask=a[group];res=r[mask];corrected=res-mean[mask];t=cdf[mask]
  metrics[group]=dict(n=int(mask.sum()),covered=[int(np.count_nonzero(np.logical_and(t>=lo,t<=hi))) for lo,hi in [(.25,.75),(.1,.9),(.025,.975)]],nll=float(np.sum(-logpdf[mask]-np.log(.005)-.5*np.log(2*np.pi))),constant_nll=float(np.sum(res**2/(2*.005**2))),squared_error3d=float(np.sum(corrected**2*np.einsum('ij,ij->i',a['rays'][mask],a['rays'][mask]))),raw_squared_error3d=float(np.sum(res**2*np.einsum('ij,ij->i',a['rays'][mask],a['rays'][mask]))),mean_error=float(np.sum(corrected)))
 return pred,metrics

def run(mode):
 c=verify();endpoint=LOG/(mode+'_readonly.json')
 if endpoint.exists():print('SKIP readonly '+mode);return
 b=Budget(c,'review_'+mode);outcome='failed';count=0;rows=[]
 try:
  if mode!='data':model=read(BASE/'model.json');need(sha(BASE/'model.json')==read(LOG/'model_frozen.json')['sha256'],'Error model changed')
  splits=['fit','holdout'] if mode=='data' else [mode]
  for split in splits:
   records=c[split+'_records']
   for r in records:
    for d in c['core_conditions']:
     b.guard();stem=name(r,d);receipt=LOG/('review_'+mode)/(stem+'.json')
     if receipt.exists():
      row=read(receipt);need(all(stat(ROOT/p)==v for p,v in row['files'].items()),'Reviewed source changed');count+=1
      if mode!='data':rows.append(read(BASE/'predictions'/split/(stem+'.json')))
      continue
     a,noise=source_reference(c,r,d,split);sample=BASE/'samples'/split/(stem+'.npz');saved,row=load_sample(sample)
     for k in a:need(np.array_equal(a[k],saved[k]) if k!='rays' else np.allclose(a[k],saved[k],rtol=0,atol=1e-14),'Independent sample differs '+k)
     need(abs(noise-row['noise_proxy'])<1e-12,'Noise proxy differs');paths=[sample,sample.with_suffix('.json')]
     if mode!='data':
      p=BASE/'predictions'/split/(stem+'.npz');actual,rr=load_sample(p);expected,metrics=invoke('mixture_prediction',lambda:density_reference(model,a),case=stem,phase='independent_'+split)
      for k in expected:need(np.allclose(expected[k],actual[k],rtol=1e-9,atol=1e-12),'Independent mixture/geometry differs '+k)
      compare(metrics,rr['metrics']);rows.append(rr);paths += [p,p.with_suffix('.json')]
     once(receipt,dict(passed=True,files={p.relative_to(ROOT).as_posix():stat(p) for p in paths}));count+=1
     if count%80==0:print(f'independent {mode} {count}',flush=True)
  if mode!='data':
   summary=read(BASE/(mode+'_summary.json'));gates={};macros={};worst=1.;gain=[]
   for d in c['core_conditions']:
    cc={}
    for group in ['main','boundary']:
     cv=[];ng=[]
     for scene in sorted({r['scene'] for r in records}):
      subset=[r['metrics'][group] for r in rows if r['scene']==scene and r['condition']==d];n=sum(r['n'] for r in subset);coverage=sum(r['covered'][2] for r in subset)/n;cv.append(coverage);ng.append(sum(r['constant_nll']-r['nll'] for r in subset)/n)
      if group=='main' and d in ['noise5','shift3','joint']:worst=min(worst,coverage)
     cc[group]=sum(cv)/len(cv);need(abs(cc[group]-summary['conditions'][d]['macro'][group]['coverage'][2])<1e-12,'Macro differs')
     if group=='main' and d!='clean':gain.append(sum(ng)/len(ng))
    macros[d]=cc
   gates=dict(noisy_95_coverage=all(.9<=macros[d]['main']<=.98 for d in ['noise5','shift3','joint']),boundary_95_coverage=all(macros[d]['boundary']>=.85 for d in ['noise5','shift3','joint']),worst_scene_95_coverage=worst>=.8,nll_improvement=sum(gain)/4>=.05)
   need(gates==summary['scientific_gates'] and all(gates.values())==summary['ready'],'Scientific decision differs')
  once(endpoint,dict(passed=True,cases=count,full_scope=mode,independent_source_geometry_and_metrics=True));outcome='passed'
 finally:b.save(outcome)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['data','holdout','development']);a=p.parse_args()
 with project_lock(ROOT):run(a.mode)
