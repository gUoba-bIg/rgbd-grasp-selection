"""Complete independent numerical review, resumable at whole-seed/case boundaries."""
import os,sys,math
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p9_common_v35 import *
from scripts.p9_calibrate_v35 import array_sha,independent,stationary
def audit_fit(seed,name,z,y,scenes,returned,b):
 import numpy as np
 r=load(seedpath(seed)/'temperature_fits'/(name+'.json'));need(r['success'] and r['call_id'] in returned and r['nfev']==len(r['trace'])<=100,'Fit lacks completed trace');need(r['binding']==dict(z=array_sha(z),y=array_sha(y),scenes=array_sha(scenes)) and enough(y) and r['training_scenes']==sorted(set(scenes.tolist())),'Temperature training data mismatch')
 for step in r['trace']:
  b.guard();need(step['call_id'] in returned,'Unknown temperature objective used');vg,_=invoke('independent_objective',lambda:independent(step['logT'],z,y),seed=seed,fit=name,role='trace');need(abs(vg[0]-step['value'])<1e-11,'Independent NLL trajectory mismatch')
 vg,_=invoke('independent_objective',lambda:independent(r['logT'],z,y),seed=seed,fit=name,role='solution');need(abs(vg[0]-r['value'])<1e-11 and stationary(r['logT'],vg[1]),'Temperature stationary proof failed');vp,_=invoke('independent_objective',lambda:independent(r['logT']+1e-5,z,y),seed=seed,fit=name,role='plus');vm,_=invoke('independent_objective',lambda:independent(r['logT']-1e-5,z,y),seed=seed,fit=name,role='minus');need(abs((vp[0]-vm[0])/2e-5-vg[1])<1e-7,'Independent temperature derivative mismatch');need(.05<=r['T']<=20 and abs(math.exp(r['logT'])-r['T'])<1e-12,'Temperature bounds mismatch');return r
def audit_apply(seed,name,T,z,returned):
 import numpy as np
 r=load(seedpath(seed)/'temperature_applications'/(name+'.json'));need(r['call_id'] in returned and r['T']==T and r['input_sha256']==array_sha(z),'Application binding mismatch');q,_=invoke('independent_apply',lambda:np.asarray([float(v)/T for v in z]),seed=seed,application=name);need(np.array_equal(q,r['logits']),'Calibrated input differs');return q
def saved_unit(path):
 if not path.exists():return False
 for p,s in load(path)['files'].items():need(stat(ROOT/p)==s,'Previously reviewed unit changed '+p)
 return True
def review(cold=False):
 import numpy as np,torch
 from src.p8_math_v32 import logits,ranked_candidates
 from scripts.p7_review_v31 import close_values
 from scripts.p8_review_v32 import independent_probability,independent_selection
 from scripts.p8_geometry_v32 import Geometry,observations,check_oracle
 from scripts.p9_metrics_v35 import calculate
 c=verify();b=Budget(c,'cold' if cold else 'review')
 if cold:
  ro=load(LOG/'readonly.json');need(ro['passed'],'Full acceptance missing')
  for p,s in ro['files'].items():b.guard();need(stat(ROOT/p)==s,'Cold artifact changed '+p)
  parents(c,b.guard);sealed(LOG/'cold.json',dict(at=now(),passed=True,files=ro['files'],new_training=0,new_forwards=0,new_official_scores=0));b.save('complete');return
 rows=events();returned={x['call_id'] for x in rows if x['kind'].endswith('_returned')};intents={x['call_id']:x for x in rows if x['kind'].endswith('_intent')};data={s:pool(s) for s in ['train','calibration','validation']};mean=data['train']['features'].mean(0,dtype=np.float64).astype(np.float32);scale=np.maximum(data['train']['features'].std(0,dtype=np.float64),1e-6).astype(np.float32);weights={}
 # Reuse accepted source features/outcomes only after verifying exact pool labels and scenes.
 for split,d in data.items():
  need(set(d['scenes'])==set(c['groups'][split]),'Split mismatch');need(np.array_equal(mean,d['mean']) and np.array_equal(scale,d['scale']),'Standardization not training-only')
  for span in d['spans']:
   case=P7/'cases'/split/key(span);labs=load(case/'labels.json');sl=slice(span['start'],span['end'])
   with np.load(artifact(case/'features.npz')) as f:need(np.array_equal(d['features'][sl],f['features']) and f['indices'].tolist()==labs['indices'],'Pool feature binding differs')
   need(np.array_equal(d['labels'][sl],[int(v['positive']) for v in labs['records']]),'Pool outcome binding differs')
 for seed in c['seeds']:
  out=seedpath(seed);fit=load(out/'fit.json');rp=LOG/'reviewed_seeds'/f'{seed}.json'
  with np.load(artifact(out/'weights.npz')) as a:weights[seed]={k:a[k].copy() for k in a.files}
  if saved_unit(rp):continue
  best=float('inf');chosen=0;stale=0;training_ids=[]
  for epoch,h in enumerate(fit['history'],1):
   b.guard();need(h['epoch']==epoch<=30 and stale<5,'Early-stop or epoch order violated');r=load(out/'epochs'/f'{epoch:02d}.json');p=EXP/str(seed)/f'epoch_{epoch:02d}.pt';need(stat(p)==r['checkpoint'],'Checkpoint hash mismatch');v=torch.load(p,map_location='cpu',weights_only=False);need(v['config_sha256']==sha(CONF) and v['next_epoch']==epoch+1 and v['history']==fit['history'][:epoch] and v['source']==fit['source'],'Recovery checkpoint contract differs');need(np.array_equal(v['mean'],mean) and np.array_equal(v['scale'],scale),'Checkpoint normalization differs');need(len(r['training_calls'])==28 and r['validation_call'] in returned,'Epoch invocations incomplete')
   for batch,cid in enumerate(r['training_calls']):need(cid in returned and intents[cid]['seed']==seed and intents[cid]['epoch']==epoch and intents[cid]['batch']==batch,'Training invocation provenance differs')
   training_ids+=r['training_calls'];w={k:t.detach().numpy() for k,t in v['model'].items()};z,_=invoke('risk_numpy_audit',lambda:logits(data['validation']['features'],w,mean,scale),seed=seed,role='every_epoch_validation',epoch=epoch);nll=float(np.mean(np.logaddexp(0,z)-data['validation']['labels']*z));need(abs(nll-r['nll'])<1e-7 and r['nll']==h['validation_nll'],'Epoch validation NLL differs')
   if h['validation_nll']<best:best=h['validation_nll'];chosen=epoch;stale=0
   else:stale+=1
  need(len(set(training_ids))==len(training_ids)==fit['unique_updates'] and chosen==fit['selected_epoch'] and (len(fit['history'])==30 or stale>=5),'Training stop/selection differs');selected=torch.load(EXP/str(seed)/f'epoch_{chosen:02d}.pt',map_location='cpu',weights_only=False);need(all(np.array_equal(weights[seed][k],v.detach().numpy()) for k,v in selected['model'].items()),'Export not selected model')
  zz={}
  for split,d in data.items():
   p=out/'predictions'/f'{split}.npz';meta=load(p.with_suffix('.calls.json'));need(meta['call_id'] in returned and meta['input']==stat(P7/'predictions'/f'{split}.npz') and meta['selected_checkpoint']==fit['checkpoint'],'Fixed-pool forward binding differs')
   with np.load(artifact(p)) as a:zz[split]=a['logits'].astype(float)
   z,_=invoke('risk_numpy_audit',lambda:logits(d['features'],weights[seed],mean,scale),seed=seed,role='selected_pool',split=split);need(np.allclose(z,zz[split],rtol=2e-5,atol=2e-5),'Independent pool logits differ')
  cal=load(out/'calibration.json');d=data['calibration'];y=d['labels'].astype(float);z=zz['calibration'];q=np.empty_like(z);folds=[]
  for scene in c['groups']['calibration']:
   held=d['scenes']==scene;name=f'held_{scene:04d}';r=audit_fit(seed,name,z[~held],y[~held],d['scenes'][~held],returned,b);need(scene not in r['training_scenes'],'Held scene leakage');q[held]=audit_apply(seed,name,r['T'],z[held],returned);folds.append(dict(scene=scene,n=int(held.sum()),T=r['T']))
  raw=independent_probability(y,z);oof=independent_probability(y,q);close_values(cal['raw'],raw);close_values(cal['held_out'],oof);close_values(cal['folds'],folds);need(np.array_equal(q,cal['held_logits']) and cal['screen_passed']==(enough(y) and oof['nll']<=raw['nll']+.001),'Holdout gate mismatch');prob={}
  if cal['screen_passed']:
   T=audit_fit(seed,'final',z,y,d['scenes'],returned,b)['T'];need(T==cal['temperature'],'Final T differs')
   for split in ['calibration','validation']:
    z=zz[split];q=audit_apply(seed,split,T,z,returned);need(np.array_equal(np.argsort(z,kind='stable'),np.argsort(q,kind='stable')),'Temperature changes risk ordering');prob[split]=dict(raw=independent_probability(data[split]['labels'],z),calibrated=independent_probability(data[split]['labels'],q))
  else:need(cal['temperature'] is None,'Failed holdout fitted final T')
  close_values(prob,cal['probability']);sealed(rp,dict(passed=True,files={p.relative_to(ROOT).as_posix():stat(p) for base in [out,EXP/str(seed)] for p in base.rglob('*') if p.is_file()}));status('full_review_seed',seed=seed)
 geo=Geometry(c);oldreturns=set()
 for dirname in ['P7/risk_v31','P8/ablation_v32']:oldreturns.update(x['call_id'] for x in map(json.loads,(ROOT/'logs'/dirname/'calls.jsonl').read_text().splitlines()) if x['kind'].endswith('_returned'))
 for number,r in enumerate(cases(c),1):
  rp=LOG/'reviewed_cases'/(key(r)+'.json')
  if saved_unit(rp):continue
  b.guard();d=BASE/'cases'/key(r);sel=load(d/'selection.json');labs=load(d/'labels.json');a=observations(r,'whole');cand=a['candidates'];state=geo.scene(r);fp=P8/'cases/validation'/key(r)/'whole/features.npz';need(sel['input_features']==stat(fp),'Candidate input binding differs')
  with np.load(artifact(fp)) as f:features=f['features'].copy();keep=f['keep'].copy()
  with np.load(artifact(d/'predictions.npz')) as pred:
   for seed in c['seeds']:
    need(sel['call_ids'][str(seed)] in returned,'Unknown risk forward used');z,_=invoke('risk_numpy_audit',lambda:logits(features,weights[seed],mean,scale),seed=seed,case=r,role='all_candidates');need(np.allclose(z,pred[str(seed)],rtol=2e-5,atol=2e-5),'Candidate risk scores differ');expected=independent_selection(cand,keep,pred[str(seed)],geo,state);need(expected==sel['methods'][str(seed)],'Independent official selection differs')
    op=d/f'oracle_{seed}.json'
    if op.exists():o=load(op);need(o['call_id'] in returned,'Unknown oracle used');norm,original=ranked_candidates(cand,keep,pred[str(seed)]);check_oracle(o['value'],norm,original,{x['index']:x for x in labs['records']},expected)
  required=sorted(set().union(*(set(v['indices']) for v in sel['methods'].values())));need(required==sel['required_indices']==labs['indices'] and geo.assignments(cand,required,state)==sel['required_slots'],'Label scope/assignment mismatch')
  for ix,slot,rec in zip(required,sel['required_slots'],labs['records']):
   need(rec==load(d/'labels'/f'{ix:04d}.json') and rec['index']==ix and rec['grasp']==cand[ix].tolist() and rec['object_slot']==slot and rec['positive']==bool(0<rec['score']<=.4 and not rec['collision']),'Physical label contract differs')
   if rec['origin']=='accepted_P8':prior=load(ROOT/rec['origin_path']);need(rec['source_receipt']==stat(ROOT/rec['origin_path']) and rec['call_id'] in oldreturns and all(rec[k]==prior[k] for k in ['index','object_slot','object_id','score','collision','empty','positive','call_id']),'Inherited score evidence differs')
   else:need(rec['call_id'] in returned and intents[rec['call_id']]['kind']=='official_score_intent','Unknown P9 physical score used')
  sealed(rp,dict(passed=True,files={p.relative_to(ROOT).as_posix():stat(p) for p in d.rglob('*') if p.is_file()}));status('full_review_cases',completed=number,total=75)
 result=load(BASE/'metrics.json');close_values(result,calculate(c,independent=True));parents(c,b.guard);ledger=counts();need(all(not x['unknown'] for x in ledger.values()),'Unknown new calls retained; cannot accept');need(ledger['upstream_oracle']['known_returns']==3,'Missing three upstream comparisons');files={p.relative_to(ROOT).as_posix():stat(p) for base in [BASE,EXP,LOG/'reviewed_seeds',LOG/'reviewed_cases'] for p in base.rglob('*') if p.is_file()};files.update({p.relative_to(ROOT).as_posix():stat(p) for p in [CONF,LOG/'calls.jsonl',ROOT/c['parent_binding']]});sealed(LOG/'readonly.json',dict(at=now(),passed=True,files=files,scientific_usable=result['scientific_usable'],calls=ledger,all_parent_files_unchanged=True,all_seed_models_and_metrics_verified=True));b.save('complete');print('P9 complete numerical review passed',flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):review('--cold' in sys.argv)
