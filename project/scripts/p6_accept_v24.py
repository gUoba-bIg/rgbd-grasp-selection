"""Independent CPU reconstruction of every saved P6 point and score."""
import os,sys,argparse,hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import *
from scripts.pipeline import project_lock
from src import p6_image_geometry_v24 as g
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.stats import chi2

def reference(c,r,d):
 meta=loadmat(ROOT/c['data_root']/r['files']['meta']);factor=float(meta['factor_depth'].reshape(-1)[0]);K=meta['intrinsic_matrix'].astype(np.float64)
 with Image.open(ROOT/c['data_root']/r['files']['depth']) as im:raw0=np.asarray(im);H,W=raw0.shape
 yy=np.array([int((j+.5)*H/480) for j in range(480)]);xx=np.array([int((i+.5)*W/640) for i in range(640)]);raw=raw0[np.ix_(yy,xx)].astype(np.float32)/factor;valid=np.isfinite(raw)&(raw>0);z=np.where(valid,np.minimum(raw,2.)/2.,0).astype(np.float32)*2;out=z.copy();ok=valid.copy()
 seed=int(hashlib.sha256(f"P5v23|{c['seed']}|{c['quality_screen_epoch']}|{r['scene']}|{r['frame']}|{d}".encode()).hexdigest()[:8],16);rng=np.random.RandomState(seed)
 if d in ['shift3','joint']:out=np.concatenate([np.zeros((480,3),np.float32),z[:,:-3]],1);ok=np.concatenate([np.zeros((480,3),bool),valid[:,:-3]],1)
 if d in ['noise5','joint']:
  out=np.where(ok,np.maximum(0,np.minimum(2,out+rng.normal(0,.005,z.shape).astype(np.float32))),0).astype(np.float32);ok&=out>0
 if d in ['missing15','joint']:
  ok&=~(rng.uniform(size=z.shape)<.15);y=rng.randint(0,361);x=rng.randint(0,481);ok[y:y+120,x:x+160]=False
 out[~ok]=0;return z,out,valid,ok,raw,K,xx,yy

def ref_boundary(m):
 edge=np.zeros_like(m,dtype=bool);edge[1:]|=m[1:]!=m[:-1];edge[:-1]|=m[1:]!=m[:-1];edge[:,1:]|=m[:,1:]!=m[:,:-1];edge[:,:-1]|=m[:,1:]!=m[:,:-1]
 dil=edge.copy();dil[1:]|=edge[:-1];dil[:-1]|=edge[1:];dil[:,1:]|=edge[:,:-1];dil[:,:-1]|=edge[:,1:];return dil

def ref_select(mask,n):
 ids=np.flatnonzero(mask);k=min(n,len(ids));return np.array([ids[j*len(ids)//k] for j in range(k)],dtype=np.int64)

def compare(a,b,path=''):
 if isinstance(a,dict):need(a.keys()==b.keys(),'Key mismatch '+path);[compare(a[k],b[k],path+'/'+k) for k in a]
 elif isinstance(a,list):need(len(a)==len(b),'Length mismatch '+path);[compare(x,y,path) for x,y in zip(a,b)]
 elif isinstance(a,(float,np.floating)):need(np.isclose(a,b,rtol=1e-10,atol=1e-8),'Metric mismatch '+path)
 else:need(a==b,'Value mismatch '+path)

def check(c,r,d,a,row):
 z,obs,valid,ok,raw,K,xx,yy=reference(c,r,d);eligible=valid&ok&(raw<=2);mp=g.fixed(r,d);qp=g.quality(r,d)
 with np.load(mp,allow_pickle=False) as f:m=f['instances'].copy()
 with np.load(qp,allow_pickle=False) as f:q={k:f[k].copy() for k in ['sigma','valid_probability','anomaly_probability']}
 edge=ref_boundary(m);base=ref_select(eligible,4096);extra=ref_select(eligible&edge,1024);ids=np.array(sorted(set(base)|set(extra)),np.int64);need(np.array_equal(a['flat_index'],ids),'Selected point mismatch');v,u=divmod(ids,640);uv=np.column_stack((xx[u],yy[v]));sg=q['sigma'].ravel()[ids].astype(float);rays=np.column_stack(((uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(len(ids))));ref=z.ravel()[ids].astype(float);depth=obs.ravel()[ids].astype(float)
 base_set=set(base);expected=dict(main=np.array([i in base_set for i in ids]),boundary=edge.ravel()[ids],instance=m.ravel()[ids],original_uv=uv,sigma=sg,depth_observed=depth,depth_reference=ref)
 for k,value in expected.items():need(np.array_equal(a[k],value),'Source point differs '+k)
 need(np.allclose(a['rays'],rays,rtol=0,atol=1e-14),'Ray mismatch');need(np.allclose(a['reference_xyz'],ref[:,None]*rays,rtol=0,atol=1e-14) and np.allclose(a['observed_xyz'],depth[:,None]*rays,rtol=0,atol=1e-14),'Metric coordinates mismatch')
 J=np.zeros((len(ids),3,3),np.float64);J[:,0,0]=depth/K[0,0];J[:,1,1]=depth/K[1,1];J[:,:,2]=rays;S=np.zeros_like(J);S[:,2,2]=sg**2;cov=J@S@J.transpose(0,2,1);need(np.allclose(cov,a['covariance'],rtol=1e-12,atol=1e-16),'Covariance not J S Jt')
 eig,vec=np.linalg.eigh(a['covariance']);need(np.all(eig[:,0]>=-1e-12*eig[:,-1]),'PSD mismatch');active=eig>np.maximum(eig[:,-1,None]*1e-12,1e-18);need(np.all(active.sum(1)==1),'Rank mismatch');err=a['observed_xyz']-a['reference_xyz'];coord=np.einsum('ni,nij->nj',err,vec);need(float(np.max(np.linalg.norm(np.where(active,0,coord),axis=1)))<=1e-10,'Off ray support');distance=np.divide(coord**2,eig,out=np.zeros_like(coord),where=active).sum(1)
 weight=q['valid_probability'].ravel()[ids].astype(float)*(1-q['anomaly_probability'].ravel()[ids].astype(float));need(np.array_equal(a['weight'],weight),'Weight differs');res=depth-ref;axis=np.diagonal(cov,axis1=1,axis2=2);axes=axis>1e-18;diagdist=np.divide(err**2,axis,out=np.zeros_like(err),where=axes).sum(1);metrics={}
 groups=dict(main=a['main'],boundary=a['boundary'],interior=a['main']&~a['boundary']&(a['instance']>0),background=a['main']&(a['instance']==0),oracle_noise_control=a['main']&(ref>.025)&(ref<1.975))
 for label,sel in groups.items():
  rr=res[sel];ss=sg[sel];w=weight[sel];metrics[label]=dict(n=int(sel.sum()),covered=[int((distance[sel]<=chi2.ppf(p,1)).sum()) for p in [.5,.8,.95]],constant_covered=[int((rr**2<=chi2.ppf(p,1)*.005**2).sum()) for p in [.5,.8,.95]],diagonal_covered95=int((diagdist[sel]<=chi2.ppf(.95,axes[sel].sum(1))).sum()),squared_error3d=float(np.sum(err[sel]*err[sel])),signed_standardized=float(np.sum(rr/ss)),nll=float(np.sum(rr**2/(2*ss**2)+np.log(ss)-np.log(.005))),constant_nll=float(np.sum(rr**2/(2*.005**2))),low_weight_count=int((w<.5).sum()),weight_sum=float(w.sum()),weighted_squared_error3d=float(np.sum(np.einsum('ni,ni->n',err[sel],err[sel])*w)))
 compare(metrics,row['metrics']);counts=dict(original_valid=int(valid.sum()),observed_valid=int(ok.sum()),eligible=int(eligible.sum()),original_clipped=int((raw>2).sum()),selected_main=len(base),selected_with_boundary=len(ids));compare(counts,row['coverage_counts']);need(array_sha(obs)==row['observed_depth_sha256'],'Independent depth replay differs');need(sha(qp)==row['quality_sha256'] and sha(mp)==row['mask_sha256'],'Source cache differs');return dict(row,metrics=metrics)

def run(screen=False):
 c=verify();endpoint=LOG/('preflight_readonly.json' if screen else 'readonly.json');b=Budget(c,'independent_preflight' if screen else 'independent_full');rows=[];outcome='failed'
 try:
  for r,d in case_order(c)[:100 if screen else 800]:
   stem=name(r,d);p=BASE/'geometry'/(stem+'.npz');rp=p.with_suffix('.json');receipt=LOG/'readonly_cases'/(stem+'.json');row=read(rp);need(sha(p)==row['cache_sha256'],'Output cache changed')
   if receipt.exists():
    saved=read(receipt);need(stat(p)==saved['cache'] and stat(rp)==saved['row'],'Previously verified output changed');rows.append(row);continue
   b.guard();before={x.relative_to(ROOT).as_posix():stat(x) for x in [p,rp,g.quality(r,d),g.fixed(r,d)]};cid=b.intent('corruption_replay',case=stem)
   with np.load(p,allow_pickle=False) as v:a={k:v[k] for k in v.files}
   verified=check(c,r,d,a,row);b.returned('corruption_replay',cid,case=stem);need(all(stat(ROOT/f)==v for f,v in before.items()),'Readonly modified inputs');once(receipt,dict(passed=True,cache=stat(p),row=stat(rp),point_count=len(a['flat_index']),independent_eigendecomposition=True));rows.append(verified)
   if len(rows)%100==0:print('readonly '+str(len(rows))+'/800',flush=True)
  if not screen:
   summary=g.aggregate(rows);compare(summary,read(BASE/'summary.json'));need(summary['positive_control_passed'],'Known noise positive control failed')
   # Separate recomputation of macro coverage and final continuation gates.
   for d in c['core_conditions']:
    ss=summary['conditions'][d]['scenes'];macro95=sum(x['groups']['main']['coverage'][2] for x in ss)/10;need(abs(macro95-summary['conditions'][d]['macro']['main']['coverage'][2])<1e-12,'Scene macro failed')
   noise=['noise5','shift3','joint'];checkg=dict(noisy_95_coverage=all(.90<=summary['conditions'][d]['macro']['main']['coverage'][2]<=.98 for d in noise),boundary_95_coverage=all(summary['conditions'][d]['macro']['boundary']['coverage'][2]>=.85 for d in noise),worst_scene_95_coverage=min(x['groups']['main']['coverage'][2] for d in noise for x in summary['conditions'][d]['scenes'])>=.8,nll_improvement=summary['mean_degraded_nll_improvement']>=.05);need(checkg==summary['scientific_gates'] and all(checkg.values())==summary['raw_sigma_ready'],'Science decision differs')
  once(endpoint,dict(passed=True,frames=len(rows),new_model_calls=0,new_training=0,independent_raw_coordinates_depth_covariance_metrics=True,all_sources_unchanged=True));outcome='passed'
 finally:b.save(outcome)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--screen',action='store_true');a=ap.parse_args()
 with project_lock(ROOT):run(a.screen)
