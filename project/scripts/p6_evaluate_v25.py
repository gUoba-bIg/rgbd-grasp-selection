"""One error-model fit, then finite holdout and conditional full regression."""
import os,sys,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v25 import *
from scripts.pipeline import project_lock
from src import p6_error_mixture_v25 as mix
from src import p6_image_geometry_v24 as geo
import numpy as np

def run(mode):
 c=verify();endpoint=BASE/('model.json' if mode=='fit' else mode+'_summary.json')
 if endpoint.exists():print('SKIP complete '+mode,flush=True);return
 b=Budget(c,mode);outcome='failed'
 try:
  need(read(LOG/'data_readonly.json')['passed'],'Repair data not reviewed')
  if mode=='fit':
   cells=[];errors=[];binding={}
   for r in c['fit_records']:
    for d in c['core_conditions']:
     p=BASE/'samples'/'fit'/(name(r,d)+'.npz');a,row=load_sample(p);need(np.all(a['main']),'Fit must use uniform points only');cells.append(a['cell']);errors.append(a['observed']-a['reference']);binding[p.relative_to(ROOT).as_posix()]=stat(p)
   model=invoke('error_model_fit',lambda:mix.fit(np.concatenate(cells),np.concatenate(errors),b.guard),cases=320,hyperparameter_search=False);once(endpoint,model);once(LOG/'model_frozen.json',dict(sha256=sha(endpoint),fit_sources=binding,fit_scenes=c['fit_scenes'],holdout_scenes=c['holdout_scenes'],holdout_used_for_fit=False));outcome='passed';return
  model=read(BASE/'model.json');need(sha(BASE/'model.json')==read(LOG/'model_frozen.json')['sha256'],'Model changed')
  if mode=='development':need(read(BASE/'holdout_summary.json')['ready'] and read(LOG/'holdout_readonly.json')['passed'],'Holdout gate not passed')
  rows=[];records=c[mode+'_records']
  for r in records:
   for d in c['core_conditions']:
    b.guard();stem=name(r,d);p=BASE/'predictions'/mode/(stem+'.npz')
    if p.with_suffix('.json').exists():a,row=load_sample(p);rows.append(row);continue
    sp=BASE/'samples'/mode/(stem+'.npz')
    if mode=='development' and not sp.exists():
     s=invoke('corruption_replay',lambda:geo.source(c,r,d),case=stem,phase=mode)
     with np.load(geo.BASE/'geometry'/(stem+'.npz')) as f:a={k:f[k].copy() for k in f.files}
     _,q=geo.load(c,r,d);cells,noise=mix.observable_cells(s['observed'],s['observed_valid'],q['sigma']);idx=a['flat_index'];need(np.array_equal(s['observed'].ravel()[idx],a['depth_observed']) and np.array_equal(s['reference'].ravel()[idx],a['depth_reference']),'Old selected geometry identity differs')
     sample=dict(flat_index=idx,cell=cells.ravel()[idx],main=a['main'],boundary=a['boundary'],instance=a['instance'],rays=a['rays'],observed=a['depth_observed'],reference=a['depth_reference'],sigma_old=a['sigma'])
     save_sample(sp,sample,dict(scene=r['scene'],frame=r['frame'],condition=d,noise_proxy=noise,reused_v24_geometry=True))
    sample,row=load_sample(sp);pred=invoke('mixture_prediction',lambda:mix.predict(model,sample['cell'],sample['observed']-sample['reference']),case=stem,phase=mode)
    pred['corrected_xyz']=(sample['observed']-pred['mean'])[:,None]*sample['rays'];pred['covariance']=np.einsum('ni,nj,n->nij',sample['rays'],sample['rays'],pred['variance']);metrics=mix.metrics(sample,pred)
    save_sample(p,pred,dict(scene=r['scene'],frame=r['frame'],condition=d,metrics=metrics,sample_sha256=sha(sp),model_sha256=sha(BASE/'model.json')));rows.append(read(p.with_suffix('.json')))
    if len(rows)%80==0:status(mode,completed=len(rows),total=len(records)*5);print(f'{mode} {len(rows)}/{len(records)*5}',flush=True)
  once(endpoint,mix.aggregate(rows));outcome='passed';print(json.dumps(read(endpoint)['scientific_gates']),flush=True)
 finally:b.save(outcome)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','holdout','development']);a=p.parse_args()
 with project_lock(ROOT):run(a.mode)
