"""Freeze a bounded repair before calculating its calibration outcomes."""
import os,sys,ast,platform
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_repair_common_v33 import *
def tests():
 import numpy as np
 from scipy.optimize import minimize
 from src.p8_affine_calibration_v33 import objective,transform,holdout_masks,projected_gradient
 from scripts.p8_repair_review_v33 import independent_objective
 from src.p8_math_v32 import probability
 from scripts.p8_review_v32 import independent_probability
 from scripts.p7_review_v31 import close_values
 checks=[]
 for parameters in [[1.,0.],[.05,-3.],[20.,3.],[.9,.3]]:
  z=np.array([-60.,-5.,-1.,0.,.5,3.,60.]);y=np.array([0.,0.,1.,0.,1.,1.,1.]);v,g=objective(parameters,z,y);vv,gg=independent_objective(parameters,z,y);need(np.isclose(v,vv,rtol=1e-12,atol=1e-12) and np.allclose(g,gg,rtol=1e-12,atol=1e-12),'Independent toy objective failed')
  for j in range(2):
   a=np.array(parameters);d=np.array(parameters);a[j]+=1e-6;d[j]-=1e-6;fd=(objective(a,z,y)[0]-objective(d,z,y)[0])/2e-6;need(abs(fd-g[j])<2e-7,'Toy gradient failed')
  need(np.array_equal(np.argsort(z,kind='stable'),np.argsort(transform(parameters,z),kind='stable')),'Monotone calibrator changes order');checks.append('extreme/constraint objective-gradient and monotonicity '+str(parameters))
 ids=np.array([1,1,2,2,3,3]);seen=np.zeros(6,int)
 for scene in [1,2,3]:
  train,held=holdout_masks(ids,scene);need(set(ids[train]).isdisjoint(ids[held]) and train.sum()==4 and held.sum()==2,'Scene holdout leak');seen+=held
 need(np.all(seen==1),'Incomplete scene OOF coverage');checks.append('scene-isolated complete holdout coverage')
 try:holdout_masks(ids,4)
 except ValueError:checks.append('missing held-out scene rejected')
 else:raise RuntimeError('Invalid held-out scene accepted')
 need(projected_gradient([.05,3.],[1.,-2.])==0 and projected_gradient([.05,3.],[-1.,2.])==2,'Constraint optimality sign failed');checks.append('constraint optimality direction')
 for n in [0,14,15,16,151]:
  y=np.arange(n)%2;z=np.round(np.linspace(-4,4,n));close_values(probability(y,z),independent_probability(y,z));checks.append('ECE15 independent stable ties '+str(n))
 # Synthetic optimizer ABI check only, never project calibration/validation results.
 z=np.array([-3.,-2.,-1.,0.,1.,2.,3.,4.]);y=np.array([0.,0.,0.,1.,0.,1.,1.,1.]);history=[]
 def f(p):history.append(p.tolist());return objective(p,z,y)
 fit=minimize(f,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.05,20.),(-3.,3.)],options=dict(maxiter=100,maxfun=200,ftol=1e-12,gtol=1e-8));need(fit.success and fit.nfev==len(history) and projected_gradient(fit.x,fit.jac)<2e-6,'Synthetic optimizer ABI failure');checks.append('synthetic constrained solver and exact objective count')
 for p in list((ROOT/'scripts').glob('p8_repair_*_v33.py'))+[ROOT/'src/p8_affine_calibration_v33.py']:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 checks.append('all repair source syntax');return checks,dict(synthetic_optimizer_calls=1,synthetic_objective_calls=len(history),project_data_fits=0,new_neural_forwards=0,new_official_scores=0)
def run():
 import numpy,scipy
 need(not CONF.exists(),'Already frozen; do not replace protocol');LOG.mkdir(parents=True,exist_ok=True)
 if (LOG/'observed_precheck.json').exists():
  previous=load(LOG/'observed_precheck.json');need(previous['passed'] and previous['observed_exit_code']==0,'Precheck did not pass')
  for p,s in previous['sources'].items():
   # Only this entrypoint changed to consume the just-completed precheck.
   if p!='scripts/p8_repair_prepare_v33.py':need(stat(ROOT/p)==s,'Tested implementation changed '+p)
  ast.parse(Path(__file__).read_text(encoding='utf-8'));checks,toy=previous['checks'],previous['toy']
 else:checks,toy=tests()
 from scripts.p8_common_v32 import verify as verify_p8
 verify_p8();need(load(P8/'closeout.json')['engineering_passed'],'Parent P8 not accepted');need(not load(P8/'closeout.json')['scientific_usable'],'Unexpected parent route result')
 binding={}
 for p in [ROOT/'logs/P7/risk_v31/cold.json',ROOT/'logs/P8/ablation_v32/cold.json']:
  r=read(p);need(r.get('passed',r.get('engineering_passed',False)),'Parent cold review missing')
  for key,s in r['files'].items():
   if key in binding:need(binding[key]==s,'Parent snapshots disagree')
   binding[key]=s
 for p in [P7/'closeout.json',P8/'closeout.json',ROOT/'docs/P8整体消融结果_v32.txt',ROOT/'runtime/pipeline/state.json',ROOT/'logs/P8/ablation_v32/readonly.json',ROOT/'logs/P8/ablation_v32/cold.json',ROOT/'logs/P7/risk_v31/readonly.json',ROOT/'logs/P7/risk_v31/cold.json']:
  binding[p.relative_to(ROOT).as_posix()]=stat(p)
 for p,s in binding.items():need(stat(ROOT/p)==s,'Parent immutable file differs '+p)
 sealed(LOG/'parent_binding.json',binding);parent=read(ROOT/'configs/risk_calibration_v31.json');protected={}
 sources=list((ROOT/'scripts').glob('p8_repair_*_v33.py'))+[ROOT/p for p in ['src/p8_affine_calibration_v33.py','src/p8_math_v32.py','scripts/p8_review_v32.py','scripts/p7_review_v31.py','scripts/p6_common_v24.py','scripts/p3_train_v27.py','scripts/pipeline.py','scripts/p8_common_v32.py','docs/P8完整点云校准修复协议_v33.txt','docs/总进度固定工作包_监督版20261027.txt','configs/risk_calibration_v31.json','configs/p8_ablation_v32.json']]+[LOG/'parent_binding.json']
 for p in sources:protected[p.relative_to(ROOT).as_posix()]=stat(p)
 for p in (ROOT/'runtime/envs/vision/Lib/site-packages/scipy/optimize').glob('*lbfgsb*'):
  if p.is_file():protected[p.relative_to(ROOT).as_posix()]=stat(p)
 for tail in ['_minimize.py','_optimize.py']:
  p=ROOT/'runtime/envs/vision/Lib/site-packages/scipy/optimize'/tail;protected[p.relative_to(ROOT).as_posix()]=stat(p)
 oldlog=ROOT/'logs/P8/ablation_v32';oldwall=sum(read(p)['seconds'] for p in oldlog.glob('controller_usage_*.json'));parent_calls=load(P8/'closeout.json')['calls'];oldsize=sum(p.stat().st_size for d in [P8,oldlog,ROOT/'experiments/p8_ablation_v32'] for p in d.rglob('*') if p.is_file())
 c=dict(version=33,at=now(),groups=parent['groups'],conditions=parent['conditions'],prior_temperature=load(P7/'fit_result.json')['temperature'],parent_binding=(LOG/'parent_binding.json').relative_to(ROOT).as_posix(),protected=protected,resources=dict(seconds=1800,rss_bytes=4*1024**3,disk_reserve_bytes=60*1024**3,output_bytes=64*1024**2),call_limits=dict(calibration_fit=16,calibration_objective=3200,calibration_apply=19,independent_objective=3300,independent_apply=19,network_training_update=0,neural_forward=0,official_score=0,mask_AP=0),scope='single-seed exposed-development repair on unchanged scenes/view/conditions; calibrated probabilities limited to original observable raw-top20 pools; no independent90-scene or robot claims',inherited_P8_calls=parent_calls,inherited_P8_wall_seconds=oldwall,inherited_P8_output_bytes=oldsize,P8_total_wall_cap=oldwall+1800,P8_total_output_cap=oldsize+64*1024**2,environment=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__),primary='E3_positive_affine_whole_cloud',protocol='docs/P8完整点云校准修复协议_v33.txt')
 sealed(LOG/'preflight.json',dict(at=now(),passed=True,checks=checks,toy=toy,parent_files_verified=len(binding),real_calibration_fit_before_freeze=False));once(CONF,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONF)));verify();status('frozen_ready',preflight_checks=len(checks));print('P8 v33 frozen; preflight passed',len(checks),'; parent files',len(binding),flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
