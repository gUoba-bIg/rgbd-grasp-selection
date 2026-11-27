"""Prepare, freeze, deliver and stop the final bounded P8 calibration repair."""
import os,sys,time,math,uuid,shutil,subprocess,traceback,ast,platform
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_temperature_common_v34 import *
def preflight():
 import numpy as np
 from scipy.optimize import minimize_scalar
 from scripts.p8_temperature_experiment_v34 import objective,independent,stationary
 from scripts.p7_review_v31 import close_values
 from scripts.p8_review_v32 import independent_probability
 from src.p8_math_v32 import probability
 checks=[];z=np.array([-60.,-3.,-1.,0.,1.,3.,60.]);y=np.array([0.,1.,0.,0.,1.,0.,1.])
 for T in [.05,1.,1.1201578256177873,20.]:
  t=np.log(T);v=objective(t,z,y);u,g=independent(t,z,y);need(np.isclose(v,u,rtol=1e-12,atol=1e-12),'Independent NLL mismatch');fd=(objective(t+1e-5,z,y)-objective(t-1e-5,z,y))/2e-5;need(abs(fd-g)<1e-7,'Temperature gradient mismatch');need(np.array_equal(np.argsort(z,kind='stable'),np.argsort(z/T,kind='stable')),'Temperature order mismatch');checks.append('extreme/boundary value-gradient-order '+str(T))
 need(stationary(math.log(.05),.5) and not stationary(math.log(.05),-.5) and stationary(math.log(20),-.5) and not stationary(math.log(20),.5),'Projected derivative direction mismatch');checks.append('constraint optimality direction')
 scene_ids=np.array([0,0,1,1,2,2]);seen=np.zeros(6,int)
 for scene in [0,1,2]:
  held=scene_ids==scene;train=~held;need(set(scene_ids[held]).isdisjoint(scene_ids[train]),'Scene leakage');seen+=held
 need((seen==1).all(),'Incomplete holdout coverage');checks.append('scene isolation and exhaustive holdout')
 for n in [0,14,15,16,151]:
  logits=np.round(np.linspace(-4,4,n));labels=np.arange(n)%2;close_values(probability(labels,logits),independent_probability(labels,logits));checks.append('ECE15 stable ties '+str(n))
 # One synthetic solver ABI test, separately recorded from project-data fitting.
 toy_z=np.array([-2.,-1.,0.,1.,2.,3.]);toy_y=np.array([0.,0.,1.,0.,1.,1.]);trace=[]
 def f(x):trace.append(float(x));return objective(x,toy_z,toy_y)
 fit=minimize_scalar(f,bounds=(np.log(.05),np.log(20.)),method='bounded',options=dict(maxiter=100,xatol=1e-6));need(fit.success and fit.nfev==len(trace) and stationary(float(fit.x),independent(fit.x,toy_z,toy_y)[1]),'Bounded scalar solver ABI failed');checks.append('synthetic solver and real invocation count')
 for p in (ROOT/'scripts').glob('p8_temperature_*_v34.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 checks.append('all new source syntax');return checks,dict(synthetic_solver_calls=1,synthetic_objective_calls=len(trace),project_data_fits=0,new_neural_forwards=0,new_official_scores=0)
def prepare():
 import numpy,scipy
 need(not CONF.exists(),'Already frozen; do not overwrite');LOG.mkdir(parents=True,exist_ok=True);checks,toy=preflight()
 from scripts.p8_repair_common_v33 import verify as verify33
 old=verify33();need(load(P33/'closeout.json')['engineering_passed'] and not load(P33/'closeout.json')['scientific_usable'],'Prior repair state differs');binding=read(ROOT/old['parent_binding'])
 for p,s in load(ROOT/'logs/P8/whole_cloud_repair_v33/cold.json')['files'].items():
  if p in binding:need(binding[p]==s,'Inherited bindings disagree')
  binding[p]=s
 for dirname in ['results/P8/whole_cloud_repair_v33','logs/P8/whole_cloud_repair_v33']:
  for p in (ROOT/dirname).rglob('*'):
   if p.is_file():binding[p.relative_to(ROOT).as_posix()]=stat(p)
 for key in ['docs/P8完整点云修复结果_v33.txt','docs/P8修复失败分析与下一步_v33_20261031.txt','configs/p8_whole_cloud_repair_v33.json','runtime/pipeline/state.json']:
  binding[key]=stat(ROOT/key)
 for p,s in binding.items():need(stat(ROOT/p)==s,'Historical artifact changed '+p)
 sealed(LOG/'parent_binding.json',binding);sources=list((ROOT/'scripts').glob('p8_temperature_*_v34.py'))+[ROOT/p for p in ['scripts/p8_repair_common_v33.py','scripts/p8_repair_run_v33.py','scripts/p8_review_v32.py','scripts/p7_review_v31.py','scripts/p6_common_v24.py','scripts/p3_train_v27.py','scripts/pipeline.py','src/p8_math_v32.py','docs/P8单参数温度修复协议_v34.txt','docs/总进度固定工作包_监督版20261027.txt','configs/risk_calibration_v31.json','configs/p8_whole_cloud_repair_v33.json']]+[LOG/'parent_binding.json'];protected={p.relative_to(ROOT).as_posix():stat(p) for p in sources}
 for tail in ['_minimize.py','_optimize.py']:
  p=ROOT/'runtime/envs/vision/Lib/site-packages/scipy/optimize'/tail;protected[p.relative_to(ROOT).as_posix()]=stat(p)
 priorwall={version:sum(read(p)['seconds'] for p in (ROOT/f'logs/P8/{dirname}').glob('controller_usage_*.json')) for version,dirname in [('v32','ablation_v32'),('v33','whole_cloud_repair_v33')]};oldsize=sum(p.stat().st_size for d in [P8,P33,ROOT/'logs/P8/ablation_v32',ROOT/'logs/P8/whole_cloud_repair_v33',ROOT/'experiments/p8_ablation_v32'] for p in d.rglob('*') if p.is_file());p7=read(ROOT/'configs/risk_calibration_v31.json');T=load(P7/'fit_result.json')['temperature'];need(T==1.1201578256177873,'Frozen prior temperature differs')
 c=dict(version=34,at=now(),groups=p7['groups'],conditions=p7['conditions'],prior_temperature=T,parent_binding=(LOG/'parent_binding.json').relative_to(ROOT).as_posix(),protected=protected,resources=dict(seconds=900,rss_bytes=4*1024**3,disk_reserve_bytes=60*1024**3,output_bytes=64*1024**2),call_limits=dict(temperature_fit=15,temperature_objective=1500,temperature_apply=17,independent_objective=1600,independent_apply=17,network_training_update=0,neural_forward=0,official_score=0,mask_AP=0),inherited_wall_seconds=priorwall,total_P8_wall_cap=sum(priorwall.values())+900,inherited_output_bytes=oldsize,total_P8_output_cap=oldsize+64*1024**2,inherited_calls=dict(v32=load(P8/'closeout.json')['calls'],v33=load(P33/'closeout.json')['calls']),scope='one final temperature-only repair on exposed single-seed development scenes; probabilities limited to accepted observable raw-top20 pools; no independent90-scene or physical-robot claim',environment=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__),no_further_calibrator_search_if_failed=True)
 sealed(LOG/'preflight.json',dict(at=now(),passed=True,checks=checks,toy=toy,parent_files_verified=len(binding),project_outcomes_examined_before_freeze=False));once(CONF,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONF)));verify();status('frozen_ready',checks=len(checks));print('P8 v34 frozen; checks passed:',len(checks),'parent files:',len(binding),flush=True)
def finish():
 c=verify();b=Budget(c,'finish');ro=load(LOG/'readonly.json');cold=load(LOG/'cold.json');need(ro['passed'] and cold['passed'] and ro['files']==cold['files'],'Acceptance/cold reentry missing')
 for p,s in cold['files'].items():b.guard();need(stat(ROOT/p)==s,'Final artifact changed '+p)
 r=load(BASE/'result.json');s=load(BASE/'holdout_screen.json');ledger=counts();need(all(not x['unknown'] for x in ledger.values()),'Unknown new invocation');ok=r['scientific_usable'];rank=r['ranking'];report=ROOT/'docs/P8单参数温度修复结果_v34.txt';lines=['P8单参数温度修复结果 v34',now(),'',f'本次有限修复完成并停止：工程完整只读及真实冷重入通过，科学关口通过={ok}。','全部任务总进度64%；实验总进度80%。内部修复不重复计分；这不是成功率。','原v32前景裁剪失败和v33仿射校准失败均保留；P9未启动，周期任务未恢复。','',c['scope'],'路线根据已暴露开发结果调整，不是新的独立泛化验证。原90独立测试场景未打开。','','复用路线：完整观测点云＋原风险排序；新增只拟合15个留一温度，不重新拟合P7全校准温度。',f"15场景留一关口通过={s['passed']}：未校准NLL={s['raw']['nll']:.9f}，留一温度NLL={s['held_out']['nll']:.9f}，变化={s['held_out']['nll']-s['raw']['nll']:+.9f}，允许上限+.001。",f"抓取AP继承已验收结果：四退化平均增益{rank['mean_degraded_AP_gain']*100:+.4f}个百分点，干净条件损失{rank['clean_AP_loss']*100:+.4f}个百分点；共享mask差值0只表示分割未改。"]
 if r['validation_executed']:
  raw=r['probability']['validation']['uncalibrated'];cal=r['probability']['validation']['temperature'];lines += [f"最终直接复用P7温度T={r['final_temperature']}，没有重复全校准集拟合。",f"验证15等频ECE：{raw['ece15']:.9f}→{cal['ece15']:.9f}；相对改善{r['relative_ECE15_reduction']*100:.4f}%（门槛>=10%）。",f"验证NLL：{raw['nll']:.9f}→{cal['nll']:.9f}；Brier：{raw['brier']:.9f}→{cal['brier']:.9f}。",'没有用P7原10等宽ECE替代上述15等频门槛。']
 else:lines+=['留一关口未通过，按协议未执行修复版最终验证；未把未运行的验证指标称为测量失败。']
 lines+=['','科学门槛（None表示按停止规则未评价）：'+str(r['gates']),'概率仅针对原始置信度top20观测候选池中的离线标签；不能直接解释为真实机器人成功率。正温度保持原风险排序，不能把继承的抓取AP增益归功于校准。','','可用性与后续：']
 if ok:lines+=['本次完整点云＋单温度路线在约定开发范围通过修复关口，可以作为P9多随机种子验证候选；等用户看完确认后再启动P9。尚不能证明独立测试、机器人效果或多种子稳定性。']
 else:lines+=['当前含“校准改善”主张的完整方案仍未通过P8，不能直接进入P9。停止继续换校准器、调阈值或追加训练。','保留已通过开发抓取门槛的完整点云风险排序。下一步应与用户讨论是否收缩概率改善主张、把校准作为已验证的负结果/限制；若改变研究主张，必须另行明确科学目标和验收条件，不能事后将本轮改判成功。']
 lines+=['','正式新增真实调用：'+str(ledger),'新增网络更新、神经前向、候选生成、官方抓取评分和mask AP均为0。准备阶段另有一次合成数据求解器接口检查，详见preflight.json，不作为正式实验成果。','既有P8控制链墙时：'+str(c['inherited_wall_seconds'])+'；本次累计上限900秒，旧账本不重置。','旧P7未知调用2990905cbfa24838bb5e7b932827f9d7仍保留为未知。','证据：results/P8/temperature_repair_v34/result.json、holdout_screen.json；logs/P8/temperature_repair_v34/readonly.json、cold.json、calls.jsonl。']
 need(not report.exists(),'Unclosed report exists; preserve before recovery');report.write_text('\n'.join(lines)+'\n',encoding='utf-8');sealed(BASE/'closeout.json',dict(at=now(),engineering_passed=True,scientific_usable=ok,all_tasks_percent=64,experiment_percent=80.,new_progress_points=0,report=stat(report),calls=ledger,inherited_calls=c['inherited_calls'],original_v32_scientific_result=False,v33_scientific_result=False,P9_started=False,scope=c['scope']));sealed(LOG/'progress.json',dict(at=now(),all_tasks_percent=64,experiment_percent=80.,earned_points=0,reason='Final bounded repair inside the already completed P8 work package',scientific_usable=ok))
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/('handoff_before_close_'+uuid.uuid4().hex+'.txt'));hand.write_text(f'最新：{now()} P8单参数温度修复v34完成并停止；15场景留一关口通过={s["passed"]}，最终科学关口通过={ok}，完整只读/真实冷重入及旧P7/v32/v33不变性通过。全部任务总进度64%、实验总进度80%，内部修复不重复计分。报告docs/P8单参数温度修复结果_v34.txt。不得自动进入P9或恢复周期任务；若未通过，停止继续更换校准器和扩量，等待用户讨论主张范围。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 p=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(p,LOG/('supervision_before_close_'+uuid.uuid4().hex+'.json'));v=read(p);v.update(recorded_at=now(),current_authorized_part='P8单参数温度修复v34',current_execution_state='P8_temperature_completed_stopped',active_internal_repair=None,current_execution_evidence='results/P8/temperature_repair_v34/closeout.json',current_progress_evidence='logs/P8/temperature_repair_v34/progress.json',waiting_for_user_next_part_confirmation=True,summary_report='docs/P8单参数温度修复结果_v34.txt',final_readonly='logs/P8/temperature_repair_v34/readonly.json',unchanged_progress_reason='P8内部修复不重复计分',stop_after='P8_temperature_repair',stop_after_part='P8');atomic(p,v);status('P8_temperature_completed_stopped',scientific_usable=ok,summary_report='docs/P8单参数温度修复结果_v34.txt');b.save('complete');print('\n'.join(lines),flush=True)
def control_event(kind,**kw):
 with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def deliver():
 c=verify();prior_events=[json.loads(s) for s in (LOG/'controller.jsonl').read_text().splitlines()] if (LOG/'controller.jsonl').exists() else [];pending={r['call_id'] for r in prior_events if r['kind']=='child_intent'}-{r['call_id'] for r in prior_events if r['kind']=='child_returned'};need(not pending,'Unknown previous child return; check orphan process first');start=time.monotonic();prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
 def guard():atomic(usage,dict(at=now(),seconds=time.monotonic()-start));need(prior+time.monotonic()-start<900,'Controller cumulative time limit')
 steps=[('experiment','scripts/p8_temperature_experiment_v34.py','run',BASE/'result.json'),('full_readonly','scripts/p8_temperature_experiment_v34.py','review',LOG/'readonly.json'),('cold_reentry','scripts/p8_temperature_experiment_v34.py','cold',LOG/'cold.json'),('finish','scripts/p8_temperature_lifecycle_v34.py','finish',BASE/'closeout.json')]
 for phase,script,mode,done in steps:
  if done.exists():load(done);continue
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B',script,mode]
  from scripts.pipeline import project_lock
  with project_lock(ROOT):pass
  control_event('child_intent',call_id=cid,phase=phase,command=cmd)
  with (LOG/f'{phase}_{cid}.stdout.log').open('xb') as out,(LOG/f'{phase}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);atomic(ROOT/'runtime/pipeline/p8_temperature_delivery_v34.json',dict(at=now(),phase=phase,controller_pid=os.getpid(),child_pid=proc.pid,all_tasks_percent=64,experiment_percent=80.,P9_started=False))
   try:
    while proc.poll() is None:guard();time.sleep(2)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
   code=proc.returncode
  control_event('child_returned',call_id=cid,phase=phase,exit_code=code);need(code==0,'Temperature stage failed; evidence retained: '+phase)
 r=load(BASE/'closeout.json');guard();control_event('completed_stopped',scientific_usable=r['scientific_usable']);atomic(ROOT/'runtime/pipeline/p8_temperature_delivery_v34.json',dict(at=now(),phase='P8_temperature_completed_stopped',scientific_usable=r['scientific_usable'],all_tasks_percent=64,experiment_percent=80.,P9_started=False))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 mode=sys.argv[1]
 if mode in ['prepare','finish']:
  with project_lock(ROOT):prepare() if mode=='prepare' else finish()
 elif mode=='deliver':
  import msvcrt
  LOG.mkdir(parents=True,exist_ok=True)
  with (ROOT/'runtime/pipeline/p8_temperature_v34.lock').open('a+b') as lock:
   lock.seek(0,2)
   if lock.tell()==0:lock.write(b'0');lock.flush()
   lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
   try:deliver()
   except Exception as e:control_event('failure_preserved',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e));atomic(ROOT/'runtime/pipeline/p8_temperature_delivery_v34.json',dict(at=now(),phase='engineering_failure_preserved',error=repr(e),all_tasks_percent=64,experiment_percent=80.,P9_started=False));raise
   finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
 else:raise RuntimeError('Unknown lifecycle mode')
