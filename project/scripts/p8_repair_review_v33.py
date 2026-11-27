"""Numerical acceptance of all new scalars/predictions and inherited AP bindings."""
import os,sys,math
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_repair_common_v33 import *
def independent_objective(parameters,z,y):
 import numpy as np
 a,b=map(float,parameters);values=[];ga=[];gb=[]
 for v,t in zip(z,y):
  q=a*float(v)+b;p=1/(1+math.exp(-q)) if q>=0 else math.exp(q)/(1+math.exp(q));error=p-float(t);values.append(max(q,0)+math.log1p(math.exp(-abs(q)))-float(t)*q);ga.append(error*float(v));gb.append(error)
 value=math.fsum(values)/len(y)+.0005*((a-1)**2+b*b);gradient=np.array([math.fsum(ga)/len(y)+.001*(a-1),math.fsum(gb)/len(y)+.001*b]);return value,gradient
def check_fit(name,data,mask,returned,budget):
 import numpy as np
 from src.p8_affine_calibration_v33 import projected_gradient
 from scripts.p8_repair_run_v33 import vector_hash
 r=load(BASE/'fits'/(name+'.json'));z=data['z'][mask];y=data['y'][mask];params=np.asarray(r['parameters']);need(r['call_id'] in returned and r['success'],'Fit lacks successful actual return');need(.05<=params[0]<=20 and -3<=params[1]<=3,'Parameters outside preregistered bounds');need(r['binding']==dict(logits=vector_hash(z),labels=vector_hash(y),scene_ids=vector_hash(data['scenes'][mask])),'Fit source binding differs');need(r['training_scenes']==sorted(set(data['scenes'][mask].tolist())),'Training scene coverage differs');need(r['nfev']==len(r['trace'])<=200 and all(x['call_id'] in returned for x in r['trace']),'Objective ledger coverage differs')
 budget.guard();(value,gradient),cid=invoke('independent_objective',lambda:independent_objective(params,z,y),fit=name,kind_detail='value_and_gradient');need(np.isclose(value,r['objective'],rtol=1e-11,atol=1e-12) and np.allclose(gradient,r['gradient'],rtol=1e-7,atol=1e-10),'Independent optimized objective/gradient differs');need(projected_gradient(params,gradient)<2e-6,'Constrained optimality failure')
 # Independent finite differences, with no optimizer or network execution.
 for j in range(2):
  plus=params.copy();minus=params.copy();plus[j]+=1e-6;minus[j]-=1e-6
  vp,_=invoke('independent_objective',lambda:independent_objective(plus,z,y)[0],fit=name,axis=j,side=1);vm,_=invoke('independent_objective',lambda:independent_objective(minus,z,y)[0],fit=name,axis=j,side=-1);need(abs((vp-vm)/2e-6-gradient[j])<1e-7,'Finite difference gradient differs')
 # Every logged objective is recomputed without fitting or invoking any model.
 for step in r['trace']:
  (v,g),_=invoke('independent_objective',lambda:independent_objective(step['parameters'],z,y),fit=name,kind_detail='optimizer_trace');need(np.isclose(v,step['value'],rtol=1e-11,atol=1e-12) and np.allclose(g,step['gradient'],rtol=1e-7,atol=1e-10),'Optimizer trace differs')
 return r
def check_application(name,parameters,z,returned):
 import numpy as np
 from scripts.p8_repair_run_v33 import vector_hash
 r=load(BASE/'applications'/(name+'.json'));need(r['parameters']==parameters and r['input_sha256']==vector_hash(z) and r['call_id'] in returned,'Application provenance differs')
 q,cid=invoke('independent_apply',lambda:np.asarray([float(parameters[0])*float(v)+float(parameters[1]) for v in z]),application=name,samples=len(z));need(np.array_equal(q,r['logits']),'Independent affine application differs');return q
def run(cold=False):
 import numpy as np
 from scripts.p7_review_v31 import close_values
 from scripts.p8_review_v32 import independent_probability
 from scripts.p8_repair_run_v33 import ranking_evidence,decision
 c=verify();b=Budget(c,'cold' if cold else 'review')
 if cold:
  r=load(LOG/'readonly.json');need(r['passed'],'Missing complete numerical review')
  for p,s in r['files'].items():b.guard();need(stat(ROOT/p)==s,'Cold artifact changed '+p)
  verify_parents(c,b.guard);sealed(LOG/'cold.json',dict(at=now(),passed=True,files=r['files'],new_neural_forwards=0,new_training=0,new_official_scores=0));b.save('complete');return
 data=dataset('calibration');result=load(BASE/'result.json');screen=load(BASE/'holdout_screen.json');returned={r['call_id'] for r in events() if r['kind'].endswith('_returned')};q=np.empty_like(data['z']);folds=[]
 for scene in c['groups']['calibration']:
  b.guard();held=data['scenes']==scene;train=~held;name=f'held_scene_{scene:04d}';r=check_fit(name,data,train,returned,b);need(r['held_scene']==scene and scene not in r['training_scenes'] and enough(data['y'][train]),'Held-out scene leaked or fold unsupported');q[held]=check_application(name,r['parameters'],data['z'][held],returned);folds.append(dict(scene=scene,training_scenes=r['training_scenes'],held_count=int(held.sum()),parameters=r['parameters']));status('full_readonly',folds_verified=len(folds),total_folds=15)
 raw=independent_probability(data['y'],data['z']);held=independent_probability(data['y'],q);expected_screen=dict(passed=enough(data['y']) and held['nll']<=raw['nll']+.001,raw=raw,held_out=held,folds=folds,all_logits=q.tolist(),support=support(data['y']),validation_used=False);close_values(screen,expected_screen)
 reports={}
 if screen['passed']:
  fit=check_fit('final_all_calibration',data,np.ones(len(q),bool),returned,b);need(fit['held_scene'] is None and fit['parameters']==result['final_parameters'],'Final calibrator differs');val=dataset('validation');need(set(data['scenes']).isdisjoint(val['scenes']),'Scene leakage');T=c['prior_temperature']
  for split,d in [('calibration',data),('validation',val)]:
   affine=check_application(split+'_E3',fit['parameters'],d['z'],returned);temp=check_application(split+'_E2',[1/T,0.],d['z'],returned);reports[split]=dict(E1=independent_probability(d['y'],d['z']),E2=independent_probability(d['y'],temp),E3=independent_probability(d['y'],affine));need(np.array_equal(np.argsort(d['z'],kind='stable'),np.argsort(affine,kind='stable')),'Monotone calibration rank differs')
  # Full-cloud P8 ranking and P7 fixed-pool risk predictions describe the same frozen model.
  p7meta=load(P7/'predictions/validation.json');p7data=np.load(artifact(P7/'predictions/validation.npz'))
  for r in p7meta['spans']:
   b.guard();stem=f"scene_{r['scene']:04d}_0064_{r['condition']}";p7folder=P7/'cases/validation'/stem;ix=load(p7folder/'labels.json')['indices'];p8features=np.load(artifact(P8/'cases/validation'/stem/'whole/features.npz'));sl=slice(r['start'],r['end']);need(np.allclose(p7data['features'][sl],p8features['features'][ix],rtol=2e-6,atol=2e-7),'Whole-cloud feature contract differs');need(np.allclose(p7data['logits'][sl],p8features['D6'][ix],rtol=2e-5,atol=2e-5),'Full-cloud risk model identity differs');p8features.close()
  p7data.close()
 else:need(result['final_parameters'] is None and not (BASE/'fits/final_all_calibration.json').exists(),'Failed screen ran final fitting')
 close_values(reports,result['probability']);rank=ranking_evidence(c,independent=True);close_values(rank,load(BASE/'ranking_evidence.json'));close_values(rank,result['ranking']);expected_decision=decision(rank,reports,expected_screen,screen['passed'])
 for k,v in expected_decision.items():close_values(v,result[k],k)
 # Explicitly validate immutable physical-label coverage, including old call returns.
 old_events=[json.loads(s) for s in (ROOT/'logs/P7/risk_v31/calls.jsonl').read_text().splitlines()];old_returns={r['call_id'] for r in old_events if r['kind']=='official_score_returned'}
 for split in ['calibration','validation']:
  for r in load(P7/'predictions'/f'{split}.json')['spans']:
   d=P7/'cases'/split/f"scene_{r['scene']:04d}_0064_{r['condition']}"
   for label in load(d/'labels.json')['records']:need(label['call_id'] in old_returns and label==load(d/'labels'/f"{label['index']:04d}.json"),'Used label has no known actual return')
 verify_parents(c,b.guard);ledger=counts();need(all(not r['unknown'] for r in ledger.values()),'Unresolved new call cannot pass full acceptance');need(ledger['calibration_fit']['known_returns']==(16 if screen['passed'] else 15),'Fit count differs from finite branch');need(ledger['calibration_apply']['known_returns']==(19 if screen['passed'] else 15),'Application count differs');files={p.relative_to(ROOT).as_posix():stat(p) for p in BASE.rglob('*') if p.is_file()};files.update({p.relative_to(ROOT).as_posix():stat(p) for p in [CONF,LOG/'calls.jsonl',ROOT/c['parent_binding']]});sealed(LOG/'readonly.json',dict(at=now(),passed=True,files=files,scientific_usable=result['scientific_usable'],new_neural_forwards=0,new_official_scores=0,complete_scalar_fit_and_prediction_and_AP_review=True,all_parent_files_unchanged=True,calls=ledger));b.save('complete')
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run('--cold' in sys.argv)
