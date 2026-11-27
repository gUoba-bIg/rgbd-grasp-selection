"""One fixed whole-cloud repair; no network training or repeated official scoring."""
import os,sys,hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_repair_common_v33 import *
def vector_hash(a):return hashlib.sha256(a.tobytes()).hexdigest()
def fit_model(c,b,name,data,mask,held_scene=None):
 import numpy as np
 from scipy.optimize import minimize
 from src.p8_affine_calibration_v33 import objective,projected_gradient
 p=BASE/'fits'/(name+'.json');z=data['z'][mask];y=data['y'][mask];binding=dict(logits=vector_hash(z),labels=vector_hash(y),scene_ids=vector_hash(data['scenes'][mask]))
 if p.exists():r=load(p);need(r['binding']==binding,'Fit data changed');return r
 need(enough(y),'Insufficient training class support in calibration fold')
 def solve():
  trace=[]
  def fun(x):
   b.guard();need(len(trace)<200,'Per-fit objective budget');(value,gradient),cid=invoke('calibration_objective',lambda:objective(x,z,y),fit=name,parameters=x.tolist());trace.append(dict(parameters=x.tolist(),value=value,gradient=gradient.tolist(),call_id=cid));return value,gradient
  result=minimize(fun,np.array([1.,0.]),jac=True,method='L-BFGS-B',bounds=[(.05,20.),(-3.,3.)],options=dict(maxiter=100,maxfun=200,ftol=1e-12,gtol=1e-8));return dict(parameters=result.x.tolist(),objective=float(result.fun),gradient=result.jac.tolist(),success=bool(result.success),message=str(result.message),nfev=int(result.nfev),iterations=int(result.nit),trace=trace)
 r,cid=invoke('calibration_fit',solve,fit=name,held_scene=held_scene);r.update(call_id=cid,binding=binding,held_scene=held_scene,training_scenes=sorted(set(data['scenes'][mask].tolist())),support=support(y));sealed(p,r)
 need(r['success'] and projected_gradient(r['parameters'],r['gradient'])<2e-6,'Calibration optimizer not converged');need(r['nfev']==len(r['trace']),'Objective count mismatch');return r
def apply_once(name,parameters,z):
 import numpy as np
 from src.p8_affine_calibration_v33 import transform
 p=BASE/'applications'/(name+'.json');binding=vector_hash(z)
 if p.exists():r=load(p);need(r['parameters']==parameters and r['input_sha256']==binding,'Existing application differs');return np.asarray(r['logits'],float)
 q,cid=invoke('calibration_apply',lambda:transform(parameters,z),application=name,samples=len(z));need(np.isfinite(q).all(),'Invalid calibrated logits');sealed(p,dict(parameters=parameters,input_sha256=binding,logits=q.tolist(),call_id=cid));return q
def ranking_evidence(c,independent=False):
 import numpy as np
 from src.p8_math_v32 import grasp_ap
 inherited=load(P8/'metrics.json');rows=[];cache={}
 for row in inherited['rows']:
  if row['method'] not in ['D0','D1','D2','D6']:continue
  source='whole' if row['method'] in ['D0','D6'] else 'foreground';d=P8/'cases/validation'/f"scene_{row['scene']:04d}_0064_{row['condition']}"/source
  if d not in cache:
   selection=load(d/'selection.json');ls=load(d/'labels.json');cache[d]=(selection,{r['index']:r for r in ls['records']})
  sel,labs=cache[d];indices=sel['methods'][row['method']]['indices'];scores=[labs[i]['score'] for i in indices];collisions=[labs[i]['collision'] for i in indices]
  if independent:
   prefix=[[sum(0<s<=mu and not co for s,co in list(zip(scores,collisions))[:k])/k for mu in [.2,.4,.6,.8,1.,1.2]] for k in range(1,51)];ap=float(np.mean(prefix))
  else:metrics=grasp_ap(scores,collisions);prefix=metrics['prefix'];ap=metrics['AP']
  need(np.array_equal(prefix,row['prefix']) and abs(ap-row['AP'])<1e-14,'Inherited AP differs from cached physical outcomes');rows.append(dict(scene=row['scene'],condition=row['condition'],method=row['method'],AP=ap))
 need(len(rows)==300,'Incomplete fixed baseline/risk AP coverage');agg={m:{cond:float(np.mean([r['AP'] for r in rows if r['method']==m and r['condition']==cond])) for cond in c['conditions']} for m in ['D0','D1','D2','D6']};degraded={m:float(np.mean([v for k,v in a.items() if k!='clean'])) for m,a in agg.items()};best=max(['D0','D1','D2'],key=lambda m:degraded[m]);gain=degraded['D6']-degraded[best];clean_loss=max(agg[m]['clean'] for m in ['D0','D1','D2'])-agg['D6']['clean'];mask=load(P8/'mask_summary.json');need(mask['maps_shared_identically_by_all_eight_groups'],'Shared mask binding absent')
 return dict(rows=rows,condition_AP=agg,degraded_AP=degraded,best_simple=best,mean_degraded_AP_gain=gain,clean_AP_loss=clean_loss,clean_mask_AP_loss=0.,new_official_calls=0,source='P8 v32 cached accepted labels and masks; E1/E2/E3 retain D6 ordering')
def decision(rank,prob,screen,validated):
 raw=prob.get('validation',{}).get('E1');fixed=prob.get('validation',{}).get('E3');relative=1-fixed['ece15']/raw['ece15'] if validated and raw['ece15']>0 else None
 gates=dict(scene_holdout_screen=screen['passed'],degraded_grasp_AP_gain=rank['mean_degraded_AP_gain']>=.01,clean_grasp_AP_preserved=rank['clean_AP_loss']<=.01,clean_mask_AP_preserved=rank['clean_mask_AP_loss']<=.01,validation_calibration_support=validated and raw['positives']>=30 and raw['n']-raw['positives']>=30,calibration_ECE15_reduction=relative is not None and relative>=.10,calibration_NLL_noninferior=validated and fixed['nll']<=raw['nll']+.001)
 return dict(gates=gates,scientific_usable=all(gates.values()),relative_ECE15_reduction=relative,validation_executed=validated)
def run():
 import numpy as np
 from src.p8_affine_calibration_v33 import holdout_masks
 from src.p8_math_v32 import probability
 c=verify();b=Budget(c,'calibration');data=dataset('calibration');need(sorted(set(data['scenes'].tolist()))==sorted(c['groups']['calibration']),'Calibration scenes differ');q=np.empty_like(data['z']);coverage=np.zeros(len(q),bool);folds=[]
 for scene in c['groups']['calibration']:
  b.guard();train,held=holdout_masks(data['scenes'],scene);fit=fit_model(c,b,f'held_scene_{scene:04d}',data,train,scene);need(scene not in fit['training_scenes'],'Held-out label leaked into fit');q[held]=apply_once(f'held_scene_{scene:04d}',fit['parameters'],data['z'][held]);coverage[held]=True;folds.append(dict(scene=scene,training_scenes=fit['training_scenes'],held_count=int(held.sum()),parameters=fit['parameters']));status('calibration_scene_holdout',completed=len(folds),total=15)
 need(coverage.all(),'Missing held-out predictions');raw=probability(data['y'],data['z']);held_metrics=probability(data['y'],q);screen=dict(passed=enough(data['y']) and held_metrics['nll']<=raw['nll']+.001,raw=raw,held_out=held_metrics,folds=folds,all_logits=q.tolist(),support=support(data['y']),validation_used=False);sealed(BASE/'holdout_screen.json',screen)
 rank=ranking_evidence(c);sealed(BASE/'ranking_evidence.json',rank);reports={};final=None
 if screen['passed']:
  final=fit_model(c,b,'final_all_calibration',data,np.ones(len(q),bool));val=dataset('validation');need(set(data['scenes']).isdisjoint(val['scenes']),'Calibration/validation overlap');T=c['prior_temperature']
  for split,d in [('calibration',data),('validation',val)]:
   b.guard();affine=apply_once(split+'_E3',final['parameters'],d['z']);temperature=apply_once(split+'_E2',[1./T,0.],d['z']);reports[split]=dict(E1=probability(d['y'],d['z']),E2=probability(d['y'],temperature),E3=probability(d['y'],affine));need(np.array_equal(np.argsort(d['z'],kind='stable'),np.argsort(affine,kind='stable')),'Calibrated logit ranking differs')
 result=dict(at=now(),**decision(rank,reports,screen,screen['passed']),probability=reports,final_parameters=final['parameters'] if final else None,ranking=rank,scope=c['scope'],old_P8_scientific_result=False,repair_selected_after_old_development_results=True,new_network_training=0,new_neural_forwards=0,new_official_scores=0,progress_earned=0);sealed(BASE/'result.json',result);b.save('complete');status('repair_computation_complete',scientific_usable=result['scientific_usable'],scene_holdout_passed=screen['passed'])
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
