"""Freeze, run the finite P9 chain, review, cold-reenter, report, and stop."""
import os,sys,time,uuid,shutil,random,subprocess,traceback,ast,platform
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p9_common_v35 import *
def preflight():
 import numpy as np,torch
 from scripts.p7_fit_v31 import model
 from scripts.p9_train_v35 import snapshot
 from src.p8_math_v32 import probability,grasp_ap,logits,ranked_candidates
 from scripts.p8_review_v32 import independent_probability
 from scripts.p7_review_v31 import close_values
 from scripts.p9_calibrate_v35 import objective,independent,stationary
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);checks=[];x=torch.linspace(-2,2,19*32).reshape(32,19);y=(torch.arange(32)%3==0).float()
 def init():
  random.seed(91);np.random.seed(91);torch.manual_seed(91);n=model();return n,torch.optim.Adam(n.parameters(),lr=.001)
 def steps(n,opt,count):
  for _ in range(count):
   ix=torch.randperm(32)[:16];ix=ix[np.random.permutation(16)];shift=random.randrange(16);ix=torch.roll(ix,shift);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(n(x[ix]).reshape(-1),y[ix]);loss.backward();opt.step()
 def equal(a,b):
  if isinstance(a,torch.Tensor):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(u,v) for u,v in zip(a,b))
  return a==b
 n1,o1=init();steps(n1,o1,8);end_rng=(random.random(),np.random.random(),torch.rand(1));n2,o2=init();steps(n2,o2,3);p=LOG/'synthetic_resume.pt';snapshot(p,n2,o2,3,np.zeros(19,np.float32),np.ones(19,np.float32),[],{'synthetic':True},config_hash='preflight_only')
 n3,o3=init();v=torch.load(p,map_location='cpu',weights_only=False);n3.load_state_dict(v['model']);o3.load_state_dict(v['optimizer']);random.setstate(v['python_rng']);np.random.set_state(v['numpy_rng']);torch.set_rng_state(v['torch_rng']);steps(n3,o3,5);need(equal(n1.state_dict(),n3.state_dict()) and equal(o1.state_dict(),o3.state_dict()),'Interrupted model/optimizer not bitwise identical');need(end_rng[0]==random.random() and end_rng[1]==np.random.random() and torch.equal(end_rng[2],torch.rand(1)),'Restored RNG streams differ');checks+=['bitwise interrupted model and Adam recovery','Python NumPy Torch RNG recovery']
 with torch.inference_mode():expected=n3(x).reshape(-1).numpy()
 w={k:t.detach().numpy() for k,t in n3.state_dict().items()};actual=logits(x.numpy(),w,np.zeros(19,np.float32),np.ones(19,np.float32));need(np.allclose(actual,expected,rtol=2e-5,atol=2e-5),'Independent network export differs');checks.append('independent NumPy model identity')
 z=np.array([-60.,-3.,-1.,0.,1.,3.,60.]);labels=np.array([0.,1.,0.,0.,1.,0.,1.])
 for T in [.05,1.,20.]:
  t=np.log(T);value,gradient=independent(t,z,labels);need(abs(value-objective(t,z,labels))<1e-12 and abs((objective(t+1e-5,z,labels)-objective(t-1e-5,z,labels))/2e-5-gradient)<1e-7,'Temperature objective/derivative error');checks.append('temperature scalar check '+str(T))
 need(stationary(np.log(.05),.5) and stationary(np.log(20),-.5) and not stationary(np.log(.05),-.5),'Bounded temperature stationarity direction');checks.append('projected boundary derivatives')
 for size in [0,14,15,16,151]:
  z=np.round(np.linspace(-5,5,size));y=np.arange(size)%2;close_values(probability(y,z),independent_probability(y,z));checks.append('ECE15 ties/empty bin '+str(size))
 for scores,collisions in [([],[]),([.2,.4,-1.],[False,True,False])]:
  p=[[sum(0<s<=mu and not co for s,co in list(zip(scores,collisions))[:k])/k for mu in [.2,.4,.6,.8,1.,1.2]] for k in range(1,51)];need(grasp_ap(scores,collisions)['prefix']==p,'AP incomplete candidate denominator mismatch');checks.append('AP fixed denominator '+str(len(scores)))
 a=np.zeros((4,17));norm,ix=ranked_candidates(a,np.array([True,False,True,True]),np.array([1.,5.,1.,0.]));need(ix.tolist()==[0,2,3] and norm[:,0].tolist()==[3,2,1],'Stable ranking/filter mismatch');checks.append('stable observed ranking ties')
 from scripts.p9_geometry_v35 import Geometry
 for p in (ROOT/'scripts').glob('p9*_v35.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 checks.append('source syntax and official adapter imports');return checks,dict(synthetic_training_updates=16,synthetic_training_forwards=16,synthetic_identity_forward=1,synthetic_independent_forward=1,formal_project_updates=0,official_scores=0)
def prepare():
 import numpy,torch,scipy
 from scripts.p8_temperature_recovery_v34r1 import verify_recovery
 need(not CONF.exists(),'P9 already frozen; do not overwrite');LOG.mkdir(parents=True,exist_ok=True);checks,toy=preflight();verify_recovery();close=load(P34/'closeout.json');need(close['engineering_passed'] and close['scientific_usable'],'P8 gate not passed');ro=load(ROOT/'logs/P8/temperature_repair_v34/readonly.json');cold=load(ROOT/'logs/P8/temperature_repair_v34/cold.json');need(ro['passed'] and cold['passed'] and ro['files']==cold['files'],'P8 complete review/cold absent')
 p7=read(ROOT/'configs/risk_calibration_v31.json');p8=read(ROOT/'configs/p8_ablation_v32.json');p34=read(ROOT/'configs/p8_temperature_repair_v34.json');groups=p7['groups'];need(list(map(len,[groups[k] for k in ['train','calibration','validation']]))==[70,15,15] and len(set().union(*map(set,groups.values())))==100,'Frozen scene partition differs')
 binding=read(ROOT/p34['parent_binding']);binding.update(cold['files'])
 for dirname in ['results/P8/temperature_repair_v34','logs/P8/temperature_repair_v34']:
  for p in (ROOT/dirname).rglob('*'):
   if p.is_file():binding[p.relative_to(ROOT).as_posix()]=stat(p)
 for p,s in p8['reference_inputs'].items():
  if p in binding:need(binding[p]==s,'Reference bindings disagree')
  binding[p]=s
 for path in ['configs/p8_temperature_repair_v34.json','configs/p8_temperature_acceptance_v34r1.json','docs/P8单参数温度修复结果_v34.txt','docs/P8温度复验修正说明_v34r1.txt','runtime/pipeline/state.json']:binding[path]=stat(ROOT/path)
 for number,(p,s) in enumerate(binding.items(),1):
  need(stat(ROOT/p)==s,'Parent file changed '+p)
  if number%10000==0:print('P9 parent preflight:',number,'/',len(binding),flush=True)
 sealed(LOG/'parent_binding.json',binding);paths=list((ROOT/'scripts').glob('p9*_v35.py'))+[ROOT/p for p in ['docs/P9三种子风险模型正式协议_v35.txt','docs/总进度固定工作包_监督版20261027.txt','scripts/p6_common_v24.py','scripts/p7_fit_v31.py','scripts/p7_review_v31.py','scripts/p8_geometry_v32.py','scripts/p8_common_v32.py','scripts/p8_review_v32.py','scripts/p8_repair_common_v33.py','scripts/p8_temperature_experiment_v34.py','scripts/p8_temperature_common_v34.py','scripts/p3_train_v27.py','scripts/verify_grasp_geometry_v10.py','scripts/pipeline.py','src/p8_math_v32.py','configs/risk_calibration_v31.json','configs/p8_ablation_v32.json']]+[LOG/'parent_binding.json'];protected={p.relative_to(ROOT).as_posix():stat(p) for p in paths}
 protected.update({p:s for p,s in p8['protected'].items() if p.startswith('runtime/build/')});inherited_wall={d:sum(read(p)['seconds'] for p in (ROOT/'logs/P8'/d).glob('controller_usage_*.json')) for d in ['ablation_v32','whole_cloud_repair_v33','temperature_repair_v34']}
 c=dict(version=35,at=now(),seeds=[20261031,20261101,20261102],groups=groups,conditions=p7['conditions'],data_root=p7['data_root'],candidate_checkpoint=p7['candidate_checkpoint'],reference_inputs=p8['reference_inputs'],parent_binding=(LOG/'parent_binding.json').relative_to(ROOT).as_posix(),protected=protected,phase_seconds=dict(training=1800,prediction=1800,scoring=7200,metrics=1800,review=1800,cold=900,finish=300),wall_seconds=14400,limits=dict(rss_bytes=8*1024**3,output_bytes=2*1024**3,disk_reserve_bytes=60*1024**3),call_limits=dict(risk_update=2700,risk_forward=334,risk_numpy_audit=324,temperature_fit=48,temperature_objective=4800,temperature_apply=51,independent_objective=5000,independent_apply=51,official_score=11250,upstream_oracle=3,candidate_forward=0,segmentation_forward=0,quality_forward=0,mask_AP=0),inherited_calls=dict(P7=load(P7/'closeout.json')['compute'],P8_v32=load(P8/'closeout.json')['calls'],P8_v33=load(ROOT/'results/P8/whole_cloud_repair_v33/closeout.json')['calls'],P8_v34=close['calls']),inherited_P8_controller_seconds=inherited_wall,cumulative_P8_P9_controller_cap=sum(inherited_wall.values())+14400,scope='Three new risk-network seeds conditional on frozen candidate/segmentation/quality backbones; original exposed 15 development scenes x 5 conditions x frame64; no reserved90 test or physical robot claim',environment=dict(python=platform.python_version(),numpy=numpy.__version__,torch=torch.__version__,scipy=scipy.__version__),new_P9_points_only_after_closeout=8)
 sealed(LOG/'preflight.json',dict(at=now(),passed=True,checks=checks,synthetic_calls=toy,parent_files_verified=len(binding),P8_scientific_gate_passed=True,formal_seeds_trained=0));once(CONF,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONF)));verify();status('frozen_ready',preflight_checks=len(checks));print('P9 frozen; preflight checks:',len(checks),flush=True)
def finish():
 c=verify();b=Budget(c,'finish');ro=load(LOG/'readonly.json');cold=load(LOG/'cold.json');need(ro['passed'] and cold['passed'] and ro['files']==cold['files'],'P9 complete acceptance missing')
 for p,s in cold['files'].items():b.guard();need(stat(ROOT/p)==s,'Final file changed '+p)
 r=load(BASE/'metrics.json');ledger=counts();need(all(not x['unknown'] for x in ledger.values()),'New unknown calls not resolved');usable=r['scientific_usable'];path=ROOT/'docs/P9三种子风险模型结果_v35.txt';lines=['P9三随机种子风险模型结果 v35',now(),'',f'三个正式种子、完整数值验收和实际冷重入已完成并停止。稳定性科学关口通过={usable}。','全部任务总进度72%（72/100）；实验总进度90%（72/80）。本次P9交付新增8点，科学负结果同样计完成工作；不是成功率或剩余耗时比例。','P10未启动，周期任务未恢复。',c['scope'],'冻结候选器和分割/质量模型，只改变风险模型初始化/训练顺序；不是所有上游网络的多种子重训。','验证集曾用于P7/P8开发和本次选轮，不是独立泛化验证。原90保留测试场景本次未打开。','','种子 | 训练轮/选定轮 | 四退化AP | 相对最好简单基线增益(pp) | 干净AP损失(pp) | ECE15相对改善 | 科学关口']
 for s in r['per_seed']:
  ece=f"{s['relative_ECE15_reduction']*100:.4f}%" if s['relative_ECE15_reduction'] is not None else '未评价（留一关口未通过）';lines.append(f"{s['seed']} | {s['epochs']}/{s['selected_epoch']} | {s['degraded_AP']:.8f} | {s['degraded_AP_gain']*100:+.4f} | {s['clean_AP_loss']*100:+.4f} | {ece} | {s['scientific_usable']}");lines.append('  全部门槛：'+str(s['gates']))
 lines+=['',f"原最好简单基线{r['best_simple']}四退化AP={r['baseline_degraded_AP']:.8f}；三种子平均增益={r['summary']['degraded_AP_gain']['mean']*100:+.4f}pp，最差种子增益={r['summary']['degraded_AP_gain']['min']*100:+.4f}pp。",'三种子汇总：'+str(r['summary']),'逐场景配对差值保存于metrics.json，统计单位15场景；本阶段未执行P11正式统计检验。','正温度不改变排序；AP增益来自风险排序，校准收益单独按固定top20离线事件池评价，不能解释为真实机器人成功率。','共享mask AP差0只说明分割未改。P5/P6及旧P8负结果全部保留。','','可用性：']
 if usable:lines+=['三个新种子均通过预设开发门槛，可作为后续P10锁定独立测试的候选；须用户看完确认后才进入P10，不能提前声称独立测试通过。']
 else:lines+=['三种子未全部通过预设稳定性关口，完整方案当前不能宣称稳定可用。全部结果保留，不挑最好种子，不再自动换校准器、阈值或追加种子。','下一步应根据逐种子失败项讨论保留风险排序、收缩校准改进主张或另设独立训练修复协议；不对本次验证结果继续调参后回写通过。']
 lines+=['','实际调用（意图/已返回/未知）：'+str(ledger),'正式最大唯一更新2520，实际唯一更新='+str(sum(s['unique_updates'] for s in r['per_seed']))+'；实际尝试含未提交轮恢复，另见账本。','旧P7未知上游调用2990905cbfa24838bb5e7b932827f9d7仍保留未知。准备阶段合成恢复测试另记，不属于研究更新。','证据：results/P9/risk_seeds_v35/metrics.json；logs/P9/risk_seeds_v35/readonly.json、cold.json、calls.jsonl。']
 need(not path.exists(),'Unclosed report exists; preserve before recovery');path.write_text('\n'.join(lines)+'\n',encoding='utf-8');sealed(LOG/'progress.json',dict(at=now(),previous_all=64,previous_experiment=80.,earned_P9_points=8,all_tasks_percent=72,experiment_percent=90.,basis='Three fixed new risk seeds, all scientific results, complete numerical review and cold reentry',scientific_usable=usable));sealed(BASE/'closeout.json',dict(at=now(),engineering_passed=True,scientific_usable=usable,all_tasks_percent=72,experiment_percent=90.,report=stat(path),calls=ledger,scope=c['scope'],P10_started=False,stopped=True))
 h=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(h,LOG/('handoff_before_close_'+uuid.uuid4().hex+'.txt'));h.write_text(f'最新：{now()} P9三个新风险种子已完成全部训练、校准、官方评价、完整数值只读、实际冷重入及父文件不变性核验并停止；稳定性科学关口通过={usable}。全部任务总进度72%、实验总进度90%。范围限定固定上游模型下风险网络种子稳定性，原15暴露开发场景frame64五条件，不是独立测试。报告docs/P9三种子风险模型结果_v35.txt；未启动P10或周期任务，等待用户检查。\n\n历史：\n'+h.read_text(encoding='utf-8-sig'),encoding='utf-8')
 p=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(p,LOG/('supervision_before_close_'+uuid.uuid4().hex+'.json'));v=read(p);v.update(recorded_at=now(),current_authorized_part='P9',current_execution_state='P9_completed_stopped',all_tasks_percent_planning_estimate=72,experiment_percent_planning_estimate=90.,current_progress_evidence='logs/P9/risk_seeds_v35/progress.json',current_execution_evidence='results/P9/risk_seeds_v35/closeout.json',waiting_for_user_next_part_confirmation=True,summary_report='docs/P9三种子风险模型结果_v35.txt',final_readonly='logs/P9/risk_seeds_v35/readonly.json',stop_after_part='P9',stop_after='P9',active_internal_repair=None,unchanged_progress_reason=None);atomic(p,v);atomic(ROOT/'runtime/pipeline/p9_part_v35.json',dict(at=now(),phase='P9_completed_stopped',all_tasks_percent=72,experiment_percent=90.,scientific_usable=usable,P10_started=False));b.save('complete');print('\n'.join(lines),flush=True)
def control(kind,**kw):
 with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def deliver():
 from scripts.pipeline import project_lock
 c=verify();previous=[json.loads(x) for x in (LOG/'controller.jsonl').read_text().splitlines()] if (LOG/'controller.jsonl').exists() else [];pending={x['call_id'] for x in previous if x['kind']=='child_intent'}-{x['call_id'] for x in previous if x['kind']=='child_returned'};need(not pending,'Unknown prior child return; check orphan process before resume');prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));start=time.monotonic();usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
 def guard():atomic(usage,dict(at=now(),seconds=time.monotonic()-start));need(prior+time.monotonic()-start<c['wall_seconds'],'P9 cumulative controller wall limit')
 steps=[('training','p9_train_v35.py',[],BASE/'training_complete.json'),('calibration','p9_calibrate_v35.py',[],BASE/'calibration_complete.json'),('prediction','p9_geometry_v35.py',['predict'],BASE/'selection_complete.json'),('scoring','p9_geometry_v35.py',['score'],BASE/'scoring_complete.json'),('metrics','p9_metrics_v35.py',[],BASE/'metrics.json'),('full_readonly','p9_review_v35.py',[],LOG/'readonly.json'),('cold_reentry','p9_review_v35.py',['--cold'],LOG/'cold.json'),('finish','p9_lifecycle_v35.py',['finish'],BASE/'closeout.json')]
 for phase,script,args,done in steps:
  if done.exists():load(done);continue
  with project_lock(ROOT):pass
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B','scripts/'+script]+args;control('child_intent',call_id=cid,phase=phase,command=cmd)
  with (LOG/f'{phase}_{cid}.stdout.log').open('xb') as out,(LOG/f'{phase}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase=phase,controller_pid=os.getpid(),child_pid=proc.pid,all_tasks_percent=64,experiment_percent=80.,P10_started=False))
   try:
    while proc.poll() is None:guard();time.sleep(2)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
  control('child_returned',call_id=cid,phase=phase,exit_code=proc.returncode);need(proc.returncode==0,'P9 stage failed; evidence retained: '+phase)
 guard();r=load(BASE/'closeout.json');control('completed_stopped',scientific_usable=r['scientific_usable']);atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase='P9_completed_stopped',scientific_usable=r['scientific_usable'],all_tasks_percent=72,experiment_percent=90.,P10_started=False))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 mode=sys.argv[1]
 if mode in ['prepare','finish']:
  with project_lock(ROOT):prepare() if mode=='prepare' else finish()
 elif mode=='deliver':
  import msvcrt
  with (ROOT/'runtime/pipeline/p9_v35.lock').open('a+b') as f:
   f.seek(0,2)
   if f.tell()==0:f.write(b'0');f.flush()
   f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   try:deliver()
   except Exception as e:control('failure_preserved',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e));atomic(ROOT/'runtime/pipeline/p9_delivery_v35.json',dict(at=now(),phase='engineering_failure_preserved',error=repr(e),all_tasks_percent=64,experiment_percent=80.,P10_started=False));raise
   finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
 else:raise RuntimeError('Unknown lifecycle mode')
