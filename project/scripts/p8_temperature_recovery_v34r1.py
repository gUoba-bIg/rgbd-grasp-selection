"""Finish the immutable v34 experiment under its unchanged resource/science limits."""
import os,sys,math,uuid,subprocess,time,traceback
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_temperature_common_v34 import *
from scripts.p8_temperature_experiment_v34 import audit_fit,audit_apply,independent,decision,array_sha
REC=LOG/'engineering_recovery_v34r1';RC=ROOT/'configs/p8_temperature_acceptance_v34r1.json'
FAIL=LOG/'full_readonly_084fb15da0824fafa9222aa909df95aa.stderr.log'
def verify_recovery():
 c=verify();r=read(RC);need(sha(RC)==read(REC/'freeze.json')['sha256'],'Recovery config changed')
 for p,s in r['protected'].items():need(stat(ROOT/p)==s,'Recovery evidence changed '+p)
 need((LOG/'calls.jsonl').read_bytes().startswith((REC/'calls.jsonl').read_bytes()),'Original call prefix changed');return c
def float32_point(z,y,logT):
 import numpy as np
 from src.risk_metrics_v31 import probability_metrics
 T=np.exp(logT);q=(z/T).astype(np.float64);scalar=np.asarray([float(np.float32(float(v)/float(np.float32(T)))) for v in z]);need(np.array_equal(q,scalar),'Original float32 division contract differs')
 value,_=independent(0.,scalar,y);old=probability_metrics(y,z/T)['nll'];need(abs(value-old)<1e-13,'Scalar legacy NLL differs');return value
def inverse_check(q,T,z):
 import numpy as np
 need(T>0 and np.isfinite(q).all() and np.isfinite(z).all(),'Nonfinite inverse identity input')
 need(np.all(np.abs(q*T-z)<=2*np.spacing(np.abs(z))),'Stored application inverse identity failed')
def prepare():
 import numpy as np
 from scripts.p8_temperature_experiment_v34 import stationary
 need(not RC.exists(),'Recovery already frozen');c=verify();checks=[]
 z=np.array([-60.,-3.,-1.,0.,1.,3.,60.],dtype=np.float32);y=np.array([0.,1.,0.,0.,1.,0.,1.])
 for T in [.05,1.1201578256177873,20.]:float32_point(z,y,math.log(T));q=z.astype(float)/T;inverse_check(q,T,z.astype(float));checks.append('legacy arithmetic/inverse identity '+str(T))
 bad=z.astype(float)/1.2;bad[1]+=.0001
 try:inverse_check(bad,1.2,z.astype(float))
 except RuntimeError:checks.append('altered application rejected')
 else:raise RuntimeError('Inverse identity test failed to detect corruption')
 need(not stationary(math.log(c['prior_temperature']),load(REC/'failure_diagnosis.json')['full_float64_logT_gradient']),'Original failure no longer reproducible');checks.append('preserved original failure assertion')
 ledger=[json.loads(x) for x in (REC/'calls.jsonl').read_text().splitlines()];apps=[x for x in ledger if x['kind']=='independent_apply_intent'];returns={x['call_id'] for x in ledger if x['kind']=='independent_apply_returned'};need(len(apps)==15 and all(x['call_id'] in returns for x in apps),'Cannot reuse incomplete independent applications');checks.append('15 known completed independent applications')
 text=FAIL.read_text(errors='replace');need('P7 full temperature optimum no longer matches data' in text and "line 79, in review" in text,'Unexpected failure boundary')
 control=[json.loads(x) for x in (LOG/'controller.jsonl').read_text().splitlines()];pending={x['call_id'] for x in control if x['kind']=='child_intent'}-{x['call_id'] for x in control if x['kind']=='child_returned'};need(not pending,'Unknown previous child outcome; cannot resume')
 sources=[Path(__file__),ROOT/'docs/P8温度复验修正说明_v34r1.txt',FAIL,ROOT/'scripts/p7_fit_v31.py',ROOT/'scripts/p7_review_v31.py',ROOT/'src/risk_metrics_v31.py',ROOT/'logs/P7/risk_v31/calls.jsonl',ROOT/'logs/P7/risk_v31/readonly.json',ROOT/'logs/P7/risk_v31/cold.json']+list(REC.glob('*.json'))+[REC/'calls.jsonl',REC/'controller.jsonl']+[p for p in BASE.rglob('*') if p.is_file()]
 protected={p.relative_to(ROOT).as_posix():stat(p) for p in sources};once(RC,dict(at=now(),protected=protected,original_config=stat(CONF),resource_caps_unchanged=True,scientific_gates_unchanged=True,new_parameter_fits=0,legacy_NLL_optimality_tolerance=1e-6,scope='supplemental engineering acceptance; original erroneous float64 legacy assertion remains a recorded failure'))
 once(REC/'freeze.json',dict(at=now(),sha256=sha(RC)));sealed(REC/'preflight.json',dict(at=now(),checks=checks,passed=True,formal_data_optimizer_calls=0));verify_recovery();print('v34r1 acceptance frozen:',len(checks),'checks',flush=True)
def prior_application(name,T,z,returned):
 import numpy as np
 r=load(BASE/'applications'/(name+'.json'));need(r['T']==T and r['input_sha256']==array_sha(z) and r['call_id'] in returned,'Original application provenance differs')
 old=[json.loads(x) for x in (REC/'calls.jsonl').read_text().splitlines()];calls=[x for x in old if x['kind']=='independent_apply_intent' and x['application']==name];done={x['call_id'] for x in old if x['kind']=='independent_apply_returned'}
 need(len(calls)==1 and calls[0]['call_id'] in done and calls[0]['samples']==len(z),'Missing prior completed independent application');q=np.asarray(r['logits']);inverse_check(q,T,z);return q
def legacy_temperature(b):
 import numpy as np
 c=read(CONF);T=c['prior_temperature'];prior=load(P7/'fit_result.json');diag=load(REC/'failure_diagnosis.json');trace=diag['old_temperature_trace'];need(len(trace)==prior['temperature_iterations']==21,'P7 temperature trace differs');old=[json.loads(x) for x in (ROOT/'logs/P7/risk_v31/calls.jsonl').read_text().splitlines()];returned={x['call_id'] for x in old if x['kind']=='temperature_objective_returned'};need(all(x['call_id'] in returned for x in trace),'Unknown old temperature objective')
 with np.load(artifact(P7/'predictions/calibration.npz')) as a:z=a['logits'].copy();y=a['labels'].astype(float)
 need(z.dtype==np.float32,'Original P7 logit precision changed');rows=[]
 for step in trace:
  b.guard();x=step['log_temperature'];oldn,cid=invoke('independent_objective',lambda:float32_point(z,y,x),role='legacy_fixed_trace_float32',source_call=step['call_id']);vg,did=invoke('independent_objective',lambda:independent(x,z.astype(float),y),role='legacy_fixed_trace_float64_certificate',source_call=step['call_id']);temp=float(np.exp(x));rows.append(dict(logT=x,T=temp,beta=1/temp,legacy_NLL=oldn,NLL64=vg[0],gradient_beta=-temp*vg[1],audit_call=cid,certificate_call=did,source_call=step['call_id']))
 selected=[r for r in rows if r['T']==T];need(len(selected)==1,'Frozen P7 temperature absent/ambiguous in old trace');s=selected[0];need(s['legacy_NLL']<=min(r['legacy_NLL'] for r in rows)+1e-13,'P7 selected temperature not best original trace');need(abs(s['legacy_NLL']-prior['reports']['calibration']['calibrated']['nll'])<1e-13,'Original P7 NLL not reproduced')
 # L(beta)=mean(log(1+exp(beta*z))-y*beta*z) is convex. An old negative/positive
 # derivative bracket contains the minimizer. Its supporting line bounds regret.
 left=[r for r in rows if r['beta']<=s['beta'] and r['gradient_beta']<=0];right=[r for r in rows if r['beta']>=s['beta'] and r['gradient_beta']>=0];need(left and right,'No existing trace bracket proves optimum')
 lo=max(left,key=lambda r:r['beta']);hi=min(right,key=lambda r:r['beta']);bound=max(s['gradient_beta']*(s['beta']-lo['beta']),s['gradient_beta']*(s['beta']-hi['beta']),0.);need(bound<=read(RC)['legacy_NLL_optimality_tolerance'],'Legacy temperature NLL optimality not certified')
 cert=dict(at=now(),passed=True,temperature=T,source_precision='float32 division then float64 NLL',rows=rows,bracket=[lo['beta'],hi['beta']],convex_NLL_suboptimality_bound=bound,unchanged_original_P7_NLL_tolerance=1e-6,optimizer_calls=0,new_temperature_fits=0);sealed(REC/'legacy_temperature_certificate.json',cert);return cert
def review():
 import numpy as np
 from scripts.p7_review_v31 import close_values
 from scripts.p8_review_v32 import independent_probability
 from scripts.p8_repair_run_v33 import ranking_evidence
 c=verify_recovery();b=Budget(c,'supplemental_review');data=dataset('calibration');screen=load(BASE/'holdout_screen.json');result=load(BASE/'result.json');returned={r['call_id'] for r in events() if r['kind'].endswith('_returned')};q=np.empty_like(data['z']);folds=[]
 for scene in c['groups']['calibration']:
  held=data['scenes']==scene;train=~held;r=audit_fit(scene,data,train,returned,b);q[held]=prior_application(f'held_scene_{scene:04d}',r['temperature'],data['z'][held],returned);folds.append(dict(scene=scene,training_scenes=r['training_scenes'],n=int(held.sum()),temperature=r['temperature']));status('supplemental_readonly',verified_folds=len(folds),total=15)
 raw=independent_probability(data['y'],data['z']);oof=independent_probability(data['y'],q);expected=dict(passed=enough(data['y']) and oof['nll']<=raw['nll']+.001,raw=raw,held_out=oof,folds=folds,all_logits=q.tolist(),support=support(data['y']),validation_used=False);close_values(expected,screen);need(screen['passed'],'Recovery only applies to observed completed branch');certificate=legacy_temperature(b);T=c['prior_temperature'];need(result['final_temperature']==T and not result['full_calibration_temperature_refitted'],'Reused temperature changed');val=dataset('validation');need(set(data['scenes']).isdisjoint(val['scenes']),'Split overlap');prob={}
 for split,d in [('calibration',data),('validation',val)]:
  qf=audit_apply(split+'_frozen_P7_temperature',T,d['z'],returned);prob[split]=dict(uncalibrated=independent_probability(d['y'],d['z']),temperature=independent_probability(d['y'],qf));need(np.array_equal(np.argsort(d['z'],kind='stable'),np.argsort(qf,kind='stable')),'Temperature ranking identity failed')
 with np.load(artifact(P7/'predictions/validation.npz')) as old:
  for r in val['spans']:
   stem=f"scene_{r['scene']:04d}_0064_{r['condition']}";ix=load(P7/'cases/validation'/stem/'labels.json')['indices'];sl=slice(r['start'],r['end'])
   with np.load(artifact(P8/'cases/validation'/stem/'whole/features.npz')) as f:need(np.allclose(old['features'][sl],f['features'][ix],rtol=2e-6,atol=2e-7) and np.allclose(old['logits'][sl],f['D6'][ix],rtol=2e-5,atol=2e-5),'Inherited full-cloud model contract differs')
 close_values(prob,result['probability']);rank=ranking_evidence(c,independent=True);close_values(rank,result['ranking']);close_values(rank,load(BASE/'ranking_evidence.json')['evidence']);need(load(BASE/'ranking_evidence.json')['source']==stat(P33/'ranking_evidence.json'),'Inherited AP receipt differs')
 for k,v in decision(rank,expected,prob).items():close_values(v,result[k],k)
 verify_parents(c,b.guard);verify_recovery();ledger=counts();need(all(not r['unknown'] for r in ledger.values()),'Unknown new invocation');need(ledger['temperature_fit']['known_returns']==15 and ledger['temperature_apply']['known_returns']==17 and ledger['independent_apply']['known_returns']==17,'Original invocation budget/branch differs');need(all(v['attempts']<=c['call_limits'][k] for k,v in ledger.items()),'Original cap exceeded')
 files={p.relative_to(ROOT).as_posix():stat(p) for p in BASE.rglob('*') if p.is_file()};files.update({p.relative_to(ROOT).as_posix():stat(p) for p in [CONF,RC,LOG/'calls.jsonl',ROOT/c['parent_binding'],REC/'legacy_temperature_certificate.json',REC/'legacy_temperature_certificate.json.receipt.json',REC/'freeze.json',Path(__file__),ROOT/'docs/P8温度复验修正说明_v34r1.txt']});sealed(LOG/'readonly.json',dict(at=now(),passed=True,acceptance_version='v34r1',original_v34_failure_preserved=True,files=files,scientific_usable=result['scientific_usable'],all_parent_files_unchanged=True,complete_fold_probability_AP_gate_review=True,calls=ledger,new_neural_forwards=0,new_training=0,new_official_scores=0,legacy_temperature_certificate=stat(REC/'legacy_temperature_certificate.json'),original_caps_and_scientific_gates_unchanged=True));b.save('complete');print('Supplemental full review passed',flush=True)
def deliver():
 from scripts.pipeline import project_lock
 from scripts.p8_temperature_lifecycle_v34 import control_event
 c=verify_recovery();prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));start=time.monotonic();usage=LOG/('controller_usage_recovery_'+uuid.uuid4().hex+'.json')
 def guard():atomic(usage,dict(at=now(),seconds=time.monotonic()-start));need(prior+time.monotonic()-start<900,'Original cumulative controller budget exceeded')
 steps=[('supplemental_full_readonly','scripts/p8_temperature_recovery_v34r1.py','review',LOG/'readonly.json'),('cold_reentry','scripts/p8_temperature_experiment_v34.py','cold',LOG/'cold.json'),('finish','scripts/p8_temperature_lifecycle_v34.py','finish',BASE/'closeout.json')]
 for phase,script,mode,done in steps:
  if done.exists():load(done);continue
  with project_lock(ROOT):pass
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B',script,mode];control_event('child_intent',call_id=cid,phase=phase,command=cmd)
  with (LOG/f'{phase}_{cid}.stdout.log').open('xb') as out,(LOG/f'{phase}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);atomic(ROOT/'runtime/pipeline/p8_temperature_delivery_v34.json',dict(at=now(),phase=phase,controller_pid=os.getpid(),child_pid=proc.pid,all_tasks_percent=64,experiment_percent=80.,P9_started=False,recovery='v34r1'))
   try:
    while proc.poll() is None:guard();time.sleep(2)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
  control_event('child_returned',call_id=cid,phase=phase,exit_code=proc.returncode);need(proc.returncode==0,'Recovery stage failed, logs retained: '+phase)
 guard();r=load(BASE/'closeout.json');control_event('completed_stopped',recovery='v34r1',scientific_usable=r['scientific_usable']);atomic(ROOT/'runtime/pipeline/p8_temperature_delivery_v34.json',dict(at=now(),phase='P8_temperature_completed_stopped',scientific_usable=r['scientific_usable'],all_tasks_percent=64,experiment_percent=80.,P9_started=False,recovery='v34r1'))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 mode=sys.argv[1]
 if mode in ['prepare','review']:
  with project_lock(ROOT):prepare() if mode=='prepare' else review()
 elif mode=='deliver':
  import msvcrt
  with (ROOT/'runtime/pipeline/p8_temperature_v34.lock').open('r+b') as f:
   msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   try:deliver()
   except Exception as e:
    from scripts.p8_temperature_lifecycle_v34 import control_event
    control_event('failure_preserved',recovery='v34r1',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e),recovery='v34r1');raise
   finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
 else:raise RuntimeError('Unknown recovery mode')
