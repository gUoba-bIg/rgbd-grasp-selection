"""Run and independently audit the fixed single-temperature experiment."""
import os,sys,math,hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_temperature_common_v34 import *
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def objective(logT,z,y):
 import numpy as np
 q=z/np.exp(logT);return float(np.mean(np.logaddexp(0,q)-y*q))
def independent(logT,z,y):
 temperature=math.exp(float(logT));values=[];derivatives=[]
 for v,t in zip(z,y):
  q=float(v)/temperature;p=1/(1+math.exp(-q)) if q>=0 else math.exp(q)/(1+math.exp(q));values.append(max(q,0)+math.log1p(math.exp(-abs(q)))-float(t)*q);derivatives.append(-(p-float(t))*q)
 return math.fsum(values)/len(y),math.fsum(derivatives)/len(y)
def stationary(logT,derivative):
 lo,hi=math.log(.05),math.log(20.)
 projected=min(0.,derivative) if logT-lo<2e-6 else max(0.,derivative) if hi-logT<2e-6 else derivative
 return abs(projected)<2e-6
def fit_fold(scene,data,train,budget):
 import numpy as np
 from scipy.optimize import minimize_scalar
 p=BASE/'fits'/f'scene_{scene:04d}.json';z=data['z'][train];y=data['y'][train];binding=dict(z=array_sha(z),y=array_sha(y),scenes=array_sha(data['scenes'][train]))
 if p.exists():r=load(p);need(r['binding']==binding,'Fit inputs changed');return r
 need(enough(y),'Insufficient calibration fold class support')
 def solve():
  trace=[]
  def fun(logT):
   budget.guard();need(len(trace)<100,'Per-fold objective limit');value,cid=invoke('temperature_objective',lambda:objective(logT,z,y),held_scene=scene,logT=float(logT));trace.append(dict(logT=float(logT),value=value,call_id=cid));return value
  r=minimize_scalar(fun,bounds=(np.log(.05),np.log(20.)),method='bounded',options=dict(maxiter=100,xatol=1e-6));return dict(logT=float(r.x),temperature=float(np.exp(r.x)),objective=float(r.fun),success=bool(r.success),message=str(r.message),nfev=int(r.nfev),trace=trace)
 r,cid=invoke('temperature_fit',solve,held_scene=scene);r.update(call_id=cid,held_scene=scene,training_scenes=sorted(set(data['scenes'][train].tolist())),support=support(y),binding=binding);sealed(p,r);need(r['success'] and r['nfev']==len(r['trace']),'Temperature solver failed');return r
def apply_once(name,T,z):
 import numpy as np
 p=BASE/'applications'/(name+'.json');binding=array_sha(z)
 if p.exists():r=load(p);need(r['T']==T and r['input_sha256']==binding,'Application input differs');return np.asarray(r['logits'])
 q,cid=invoke('temperature_apply',lambda:z/T,application=name,samples=len(z));need(np.isfinite(q).all() and T>0,'Invalid temperature output');sealed(p,dict(T=T,input_sha256=binding,logits=q.tolist(),call_id=cid));return q
def decision(rank,screen,prob):
 executed='validation' in prob;raw=prob.get('validation',{}).get('uncalibrated');cal=prob.get('validation',{}).get('temperature');relative=1-cal['ece15']/raw['ece15'] if executed and raw['ece15']>0 else None
 gates=dict(scene_holdout_screen=screen['passed'],degraded_grasp_AP_gain=rank['mean_degraded_AP_gain']>=.01,clean_grasp_AP_preserved=rank['clean_AP_loss']<=.01,clean_mask_AP_preserved=rank['clean_mask_AP_loss']<=.01,validation_calibration_support=(raw['positives']>=30 and raw['n']-raw['positives']>=30) if executed else None,calibration_ECE15_reduction=(relative is not None and relative>=.10) if executed else None,calibration_NLL_noninferior=(cal['nll']<=raw['nll']+.001) if executed else None)
 return dict(gates=gates,scientific_usable=all(v is True for v in gates.values()),validation_executed=executed,relative_ECE15_reduction=relative)
def run():
 import numpy as np
 from src.p8_math_v32 import probability
 c=verify();b=Budget(c,'experiment');data=dataset('calibration');need(sorted(set(data['scenes']))==sorted(c['groups']['calibration']),'Calibration split differs');q=np.empty_like(data['z']);covered=np.zeros(len(q),bool);folds=[]
 for scene in c['groups']['calibration']:
  held=data['scenes']==scene;train=~held;need(held.any() and train.any(),'Empty scene fold');fit=fit_fold(scene,data,train,b);need(scene not in fit['training_scenes'],'Held-out scene leakage');q[held]=apply_once(f'held_scene_{scene:04d}',fit['temperature'],data['z'][held]);covered[held]=True;folds.append(dict(scene=scene,training_scenes=fit['training_scenes'],n=int(held.sum()),temperature=fit['temperature']));status('scene_holdout',completed=len(folds),total=15)
 need(covered.all(),'Incomplete holdout');raw=probability(data['y'],data['z']);oof=probability(data['y'],q);screen=dict(passed=enough(data['y']) and oof['nll']<=raw['nll']+.001,raw=raw,held_out=oof,folds=folds,all_logits=q.tolist(),support=support(data['y']),validation_used=False);sealed(BASE/'holdout_screen.json',screen)
 rank=load(P33/'ranking_evidence.json');sealed(BASE/'ranking_evidence.json',dict(evidence=rank,source=stat(P33/'ranking_evidence.json'),reused_accepted_AP=True));reports={}
 if screen['passed']:
  val=dataset('validation');need(set(data['scenes']).isdisjoint(val['scenes']),'Calibration/validation overlap');T=c['prior_temperature']
  for split,d in [('calibration',data),('validation',val)]:
   qf=apply_once(split+'_frozen_P7_temperature',T,d['z']);need(np.array_equal(np.argsort(d['z'],kind='stable'),np.argsort(qf,kind='stable')),'Temperature changed logit ranking');reports[split]=dict(uncalibrated=probability(d['y'],d['z']),temperature=probability(d['y'],qf))
 result=dict(at=now(),**decision(rank,screen,reports),probability=reports,ranking=rank,final_temperature=c['prior_temperature'] if screen['passed'] else None,full_calibration_temperature_refitted=False,scope=c['scope'],original_v32_scientific_result=False,v33_scientific_result=False,new_neural_forwards=0,new_training=0,new_official_scores=0,progress_points=0);sealed(BASE/'result.json',result);b.save('complete');status('computation_complete',screen_passed=screen['passed'],scientific_usable=result['scientific_usable'])
def audit_objective(logT,z,y,**kw):return invoke('independent_objective',lambda:independent(logT,z,y),logT=float(logT),**kw)[0]
def audit_fit(scene,data,train,returned,budget):
 import numpy as np
 r=load(BASE/'fits'/f'scene_{scene:04d}.json');z=data['z'][train];y=data['y'][train];need(r['success'] and r['call_id'] in returned,'Fit not known successful');need(r['held_scene']==scene and r['training_scenes']==sorted(set(data['scenes'][train].tolist())) and scene not in r['training_scenes'],'Scene leakage');need(r['binding']==dict(z=array_sha(z),y=array_sha(y),scenes=array_sha(data['scenes'][train])) and enough(y),'Fold binding/support differs');need(r['nfev']==len(r['trace'])<=100 and all(t['call_id'] in returned for t in r['trace']),'Optimization call provenance differs');need(.05<=r['temperature']<=20 and np.isclose(r['temperature'],math.exp(r['logT']),rtol=1e-15),'Temperature bounds/transformation differs')
 for step in r['trace']:
  budget.guard();v,g=audit_objective(step['logT'],z,y,held_scene=scene,role='trace');need(np.isclose(v,step['value'],rtol=1e-11,atol=1e-12),'Independent objective trace differs')
 v,g=audit_objective(r['logT'],z,y,held_scene=scene,role='solution');need(np.isclose(v,r['objective'],rtol=1e-11,atol=1e-12) and stationary(r['logT'],g),'Constrained optimum failed');vp,_=audit_objective(r['logT']+1e-5,z,y,held_scene=scene,role='difference_plus');vm,_=audit_objective(r['logT']-1e-5,z,y,held_scene=scene,role='difference_minus');need(abs((vp-vm)/2e-5-g)<1e-7,'Independent finite-difference derivative differs');return r
def audit_apply(name,T,z,returned):
 import numpy as np
 r=load(BASE/'applications'/(name+'.json'));need(r['T']==T and r['input_sha256']==array_sha(z) and r['call_id'] in returned,'Application provenance differs');q,cid=invoke('independent_apply',lambda:np.asarray([float(x)/float(T) for x in z]),application=name,samples=len(z));need(np.array_equal(q,r['logits']),'Independent probability-input vector differs');return q
def review(cold=False):
 import numpy as np
 from scripts.p7_review_v31 import close_values
 from scripts.p8_review_v32 import independent_probability
 from scripts.p8_repair_run_v33 import ranking_evidence
 c=verify();b=Budget(c,'cold' if cold else 'review')
 if cold:
  r=load(LOG/'readonly.json');need(r['passed'],'Full review missing')
  for p,s in r['files'].items():b.guard();need(stat(ROOT/p)==s,'Cold artifact changed '+p)
  verify_parents(c,b.guard);sealed(LOG/'cold.json',dict(at=now(),passed=True,files=r['files'],new_neural_forwards=0,new_training=0,new_official_scores=0));b.save('complete');return
 data=dataset('calibration');screen=load(BASE/'holdout_screen.json');result=load(BASE/'result.json');returned={r['call_id'] for r in events() if r['kind'].endswith('_returned')};q=np.empty_like(data['z']);folds=[]
 for scene in c['groups']['calibration']:
  held=data['scenes']==scene;train=~held;r=audit_fit(scene,data,train,returned,b);q[held]=audit_apply(f'held_scene_{scene:04d}',r['temperature'],data['z'][held],returned);folds.append(dict(scene=scene,training_scenes=r['training_scenes'],n=int(held.sum()),temperature=r['temperature']));status('full_readonly',verified_folds=len(folds),total=15)
 raw=independent_probability(data['y'],data['z']);oof=independent_probability(data['y'],q);expected_screen=dict(passed=enough(data['y']) and oof['nll']<=raw['nll']+.001,raw=raw,held_out=oof,folds=folds,all_logits=q.tolist(),support=support(data['y']),validation_used=False);close_values(screen,expected_screen);prob={}
 if screen['passed']:
  prior=load(P7/'fit_result.json');T=c['prior_temperature'];need(T==prior['temperature']==result['final_temperature'] and not result['full_calibration_temperature_refitted'],'Frozen P7 temperature not reused');v,g=audit_objective(math.log(T),data['z'],data['y'],role='frozen_P7_full_temperature');need(stationary(math.log(T),g),'P7 full temperature optimum no longer matches data');vp,_=audit_objective(math.log(T)+1e-5,data['z'],data['y'],role='frozen_P7_difference_plus');vm,_=audit_objective(math.log(T)-1e-5,data['z'],data['y'],role='frozen_P7_difference_minus');need(abs((vp-vm)/2e-5-g)<1e-7,'Frozen temperature derivative differs');val=dataset('validation');need(set(data['scenes']).isdisjoint(val['scenes']),'Split overlap')
  for split,d in [('calibration',data),('validation',val)]:
   qf=audit_apply(split+'_frozen_P7_temperature',T,d['z'],returned);prob[split]=dict(uncalibrated=independent_probability(d['y'],d['z']),temperature=independent_probability(d['y'],qf));need(np.array_equal(np.argsort(d['z'],kind='stable'),np.argsort(qf,kind='stable')),'Monotonic ranking identity failed')
  # P7 fixed-pool probabilities and P8 D6 ranking must use the same accepted risk features/model.
  with np.load(artifact(P7/'predictions/validation.npz')) as old:
   for r in val['spans']:
    stem=f"scene_{r['scene']:04d}_0064_{r['condition']}";ix=load(P7/'cases/validation'/stem/'labels.json')['indices'];sl=slice(r['start'],r['end'])
    with np.load(artifact(P8/'cases/validation'/stem/'whole/features.npz')) as f:need(np.allclose(old['features'][sl],f['features'][ix],rtol=2e-6,atol=2e-7) and np.allclose(old['logits'][sl],f['D6'][ix],rtol=2e-5,atol=2e-5),'Inherited whole-cloud model contract differs')
 else:need(not result['validation_executed'] and result['final_temperature'] is None,'Failed screen executed final validation')
 close_values(prob,result['probability']);rank=ranking_evidence(c,independent=True);close_values(rank,result['ranking']);close_values(rank,load(BASE/'ranking_evidence.json')['evidence']);need(load(BASE/'ranking_evidence.json')['source']==stat(P33/'ranking_evidence.json'),'Inherited AP receipt differs')
 for k,v in decision(rank,expected_screen,prob).items():close_values(v,result[k],k)
 verify_parents(c,b.guard);ledger=counts();need(all(not r['unknown'] for r in ledger.values()),'Unknown new call cannot pass');need(ledger['temperature_fit']['known_returns']==15 and ledger['temperature_apply']['known_returns']==(17 if screen['passed'] else 15),'Unexpected branch invocation count');files={p.relative_to(ROOT).as_posix():stat(p) for p in BASE.rglob('*') if p.is_file()};files.update({p.relative_to(ROOT).as_posix():stat(p) for p in [CONF,LOG/'calls.jsonl',ROOT/c['parent_binding']]});sealed(LOG/'readonly.json',dict(at=now(),passed=True,files=files,scientific_usable=result['scientific_usable'],all_parent_files_unchanged=True,complete_fold_probability_AP_gate_review=True,calls=ledger,new_neural_forwards=0,new_training=0,new_official_scores=0));b.save('complete')
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run() if sys.argv[1]=='run' else review(sys.argv[1]=='cold')
