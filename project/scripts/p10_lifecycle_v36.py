"""One bounded P10 chain; stops after whole-part scientific reporting."""
import os,sys,ast,time,uuid,subprocess,traceback,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p10_common_v36 import *
def control(kind,**kw):
 LOG.mkdir(parents=True,exist_ok=True)
 with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def prepare():
 import numpy as np
 from scripts.p9_close_recovery_v35r2 import verify_close
 from scripts.p10_assets_v36 import acquisition_config
 need(not CONF.exists(),'P10 already frozen; never overwrite');verify_close();root=load(ROOT/'logs/P9/risk_seeds_v35/root_final_review.json');need(root['passed'] and root['stopped'] and root['scientific_usable']==False,'Expected preserved P9 conclusion differs');checks=load(LOG/'preflight/checks.json');need(checks['passed'] and checks['official_test_images_opened']==0,'Preparation not passed')
 ac=read(ACONF) if ACONF.exists() else acquisition_config();old=read(ROOT/'configs/risk_calibration_v31.json');p9=read(ROOT/'configs/p9_risk_seeds_v35.json');split=read(ROOT/'data/manifests/scene_splits.json')['groups'];groups={k:split[k] for k in ['test_seen','test_similar','test_novel']};need(list(map(len,groups.values()))==[30,30,30] and set(sum(groups.values(),[]))==set(range(100,190)) and not set(sum(old['groups'].values(),[]))&set(sum(groups.values(),[])),'Test split isolation failed')
 binding=read(ROOT/p9['parent_binding']);binding.update(load(ROOT/'logs/P9/risk_seeds_v35/cold.json')['files'])
 for base in ['results/P9/risk_seeds_v35','logs/P9/risk_seeds_v35']:
  for p in (ROOT/base).rglob('*'):
   if p.is_file():binding[p.relative_to(ROOT).as_posix()]=stat(p)
 for name in ['configs/p9_risk_seeds_v35.json','configs/p9_environment_recovery_v35r1.json','configs/p9_close_recovery_v35r2.json','docs/P9三种子风险模型结果_v35.txt','docs/P9结果解释与后续建议_v35.txt','runtime/pipeline/state.json','data/manifests/scene_splits.json']:
  p=ROOT/name;binding[name]=stat(p)
 for p,s in binding.items():need(stat(ROOT/p)==s,'Historical file changed '+p)
 once(LOG/'parent_binding.json',binding)
 conditions=['clean','noise5','missing15','shift3','joint','noise1','noise3','missing5','missing30','shift1','shift5','block'];temps={str(s):load(ROOT/f'results/P9/risk_seeds_v35/seeds/{s}/calibration.json')['temperature'] for s in p9['seeds']}
 paths=list((ROOT/'scripts').glob('p10*_v36.py'))+[ROOT/'docs/P10独立测试与路线修订协议_v36.txt',ACONF,LOG/'preflight/checks.json',ROOT/'src/remote_zip.py',ROOT/'scripts/p23_fetch_v26.py',ROOT/'scripts/p23_common_v26.py']
 paths += [ROOT/p for p in old['protected'] if Path(p).suffix in ['.py','.pyd','.pt','.json'] and not p.startswith('logs/')]
 paths += [ROOT/p for p in p9['protected'] if Path(p).suffix=='.py']
 paths += [ROOT/'scripts/p8_geometry_v32.py',ROOT/'scripts/p8_review_v32.py',ROOT/'src/p8_math_v32.py',ROOT/'src/grasp_observed_filter_v29.py',ROOT/'scripts/p7_review_v31.py',ROOT/'scripts/p7_fit_v31.py',ROOT/'scripts/p3_train_v27.py',ROOT/'scripts/p9_common_v35.py',ROOT/'scripts/p8_repair_common_v33.py',ROOT/'src/risk_observations_v31.py',ROOT/'src/p5_quality_v23.py',ROOT/'results/P7/risk_v31/predictions/train.npz']
 for s in p9['seeds']:
  fit=load(ROOT/f'results/P9/risk_seeds_v35/seeds/{s}/fit.json');paths += [ROOT/f'results/P9/risk_seeds_v35/seeds/{s}/weights.npz',ROOT/f"experiments/risk_seeds_v35/{s}/epoch_{fit['selected_epoch']:02d}.pt",ROOT/f'results/P9/risk_seeds_v35/seeds/{s}/calibration.json']
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 protected={p.relative_to(ROOT).as_posix():stat(p) for p in paths};rates=[];unions=[]
 for dirname in ['P7/risk_v31','P9/risk_seeds_v35']:
  ts=[x['seconds'] for x in map(json.loads,(ROOT/'logs'/dirname/'calls.jsonl').read_text().splitlines()) if x['kind']=='official_score_returned'];rates.append(float(np.mean(ts)))
 for p in (ROOT/'results/P9/risk_seeds_v35/cases').glob('*/selection.json'):
  r=load(p);oldsel=load(ROOT/'results/P8/ablation_v32/cases/validation'/p.parent.name/'whole/selection.json');unions.append(len(set(r['required_indices'])|set(oldsel['methods']['D0']['indices']))+20)
 projected=float(max(rates)*np.mean(unions)*1080*1.25+14400);need(projected<129600,'Empirical resource projection exceeds finite controller cap');need(shutil.disk_usage(ROOT).free>98*1024**3,'Insufficient reserve for 6GiB data,32GiB output and60GiB free')
 c=dict(version=36,at=now(),seed=old['seed'],seeds=p9['seeds'],data_root=DATA.relative_to(ROOT).as_posix(),groups=groups,conditions=conditions,primary_degraded=['noise5','missing15','shift3','joint'],extra_conditions=conditions[5:],temperatures=temps,quality_screen_epoch=old['quality_screen_epoch'],decoder=old['decoder'],candidate_checkpoint=old['candidate_checkpoint'],quality_checkpoint=old['quality_checkpoint'],segmentation_checkpoint=old['segmentation_checkpoint'],scope='90 held-out scenes in three official groups, one fixed RealSense frame64 each,12 frozen corruption conditions; fixed upstream backbones and three retained risk seeds; not full256-frame benchmark; no calibration-improvement claim',parent_binding=(LOG/'parent_binding.json').relative_to(ROOT).as_posix(),asset_config=stat(ACONF),protected=protected,call_limits=dict(grasp_forward=1092,quality_forward=1092,segmentation_forward=1092,corruption=1090,corruption_audit=1090,risk_forward=3270,risk_numpy_audit=3270,official_score=237600,upstream_oracle=12),phase_seconds=dict(generation=7200,features=21600,scoring=108000,metrics=1800,review=21600,cold=1800,finish=600),wall_seconds=129600,limits=dict(gpu_bytes=int(13.5*1024**3),rss_bytes=16*1024**3,output_bytes=32*1024**3,disk_reserve_bytes=60*1024**3),inherited_calls=p9['inherited_calls'],P9_calls=root['calls'],preflight_calls=checks,baseline_chosen_before_test='D0',new_P10_points_after_closeout=8)
 sealed(LOG/'resource_feasibility.json',dict(at=now(),observed_score_mean_seconds=rates,observed_required_union_plus20_mean=float(np.mean(unions)),projected_seconds=projected,cap_seconds=129600,formula='max(P7/P9 known score means) * old observed D0/3seed union+20 pool *1080*1.25 +14400 overhead',forecast_not_guarantee=True));once(CONF,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONF)));verify();status('frozen_ready',conditions_completed=0,total_conditions=1080,formal_test_images_opened=0)
 for name in ['docs/实验自动续接说明.txt','runtime/pipeline/part_level_supervision.json']:shutil.copy2(ROOT/name,LOG/('before_P10_'+Path(name).name))
 h=ROOT/'docs/实验自动续接说明.txt';h.write_text(f'最新：{now()} 用户已同意保留风险排序、收缩校准改善主张并继续P10。v36已通过旧开发数据准备检查、P9父文件核验并正式冻结；仅90保留场景frame64×12条件、固定D0与三个种子，全部1080选取封存后官方评分，不按测试调参，不重训/重拟合。源码/数据范围/预算见configs/p10_heldout_v36.json及docs/P10独立测试与路线修订协议_v36.txt。入口scripts/p10_lifecycle_v36.py deliver；逐场景有限分段、完整复验/真实冷重入/真实汇总后停止，不进P11或周期任务。全部任务总进度72%、实验总进度90%；P10完成后方计80%/100%。原P9科学失败、旧未知调用及state.json不改。\n\n历史：\n'+h.read_text(encoding='utf-8-sig'),encoding='utf-8')
 p=ROOT/'runtime/pipeline/part_level_supervision.json';s=read(p);s.update(recorded_at=now(),current_authorized_part='P10',current_execution_state='P10_frozen_ready',current_runner='scripts/p10_lifecycle_v36.py deliver',current_execution_evidence=CONF.relative_to(ROOT).as_posix(),stop_after_part='P10',stop_after='P10',waiting_for_user_next_part_confirmation=False,summary_report=None,remaining_research_updates_cap=0,active_internal_repair=None,unchanged_progress_reason='P10未完成整体验收，不预记8点');atomic(p,s);print('P10 frozen after',len(checks['checks']),'preflight checks;',len(binding),'historical files;',len(cases(c)),'conditions;',round(projected/3600,2),'hours conservative planning forecast',flush=True)
def finish():
 c=verify();b=Budget(c,'finish');ro=load(LOG/'readonly.json');cold=load(LOG/'cold.json');need(ro['passed'] and cold['passed'] and ro['files']==cold['files'],'Full acceptance/cold mismatch')
 for n,(p,s) in enumerate(cold['files'].items()):
  if n%100==0:b.guard()
  need(stat(ROOT/p)==s,'Final artifact changed '+p)
 rows=[json.loads(x) for x in (LOG/'controller.jsonl').read_text().splitlines()];intents={x['call_id']:x for x in rows if x['kind']=='child_intent'};returned={x['call_id']:x for x in rows if x['kind']=='child_returned'};pending=set(intents)-set(returned);need(len(pending)==1 and all(intents[k]['phase']=='finish' for k in pending),'Unknown previous child');coldcalls=[x for x in returned.values() if x['phase']=='cold' and x['exit_code']==0];need(len(coldcalls)==1,'Actual cold child not returned');ledger=counts();need(ledger==ro['calls'] and all(not x['unknown'] for x in ledger.values()),'Research ledger unknown/changed');r=load(BASE/'metrics.json');usable=r['ranking_scientific_usable'];path=ROOT/'docs/P10独立测试与鲁棒性结果_v36.txt';need(not path.exists(),'Existing unclosed report; preserve')
 lines=['P10 独立测试与鲁棒性结果 v36',now(),'',f'全部既定计算、完整复验及实际新进程冷重入已完成。缩减后的风险排序科学门槛通过={usable}。','全部任务总进度80%（80/100），实验总进度100%（80/80）。固定有限交付权重，含负结果；不是未收缩原方案全部成功、全256帧覆盖或剩余时间比例。','P11统计图表8点及P12论文复现12点仍待执行；本阶段完成后停止，未启动P11或周期任务。',c['scope'],'只使用每场景固定第64帧，不是官方全256帧基准；质量尺度非可信概率，校准不作稳定改善主张。P9原失败及其它负结果保持不变。','missing15实际含随机缺失+块状空洞，不能当纯单因素缺失曲线。边界异常未作为单独压力条件。','', '种子 | 主退化平均AP增益(pp) | 干净AP损失(pp) | 额外7条件AP增益(pp) | 科学可用']
 for s in r['per_seed']:
  lines.append(f"{s['seed']} | {s['core_AP_gain']*100:+.4f} | {s['clean_AP_loss']*100:+.4f} | {s['extra_AP_gain']*100:+.4f} | {s['ranking_usable']}");lines.append('关口：'+str(s['gates']));lines.append('各组：'+str(s['groups']))
 lines+=['','三组各条件完整AP：'+json.dumps(r['condition_AP'],ensure_ascii=False),'','原始/冻结温度概率描述（固定top20离线0.4摩擦无碰撞事件池；非机器人成功率）：'+json.dumps(r['probability'],ensure_ascii=False),'','主张判断：'+('固定候选条件下的风险排序路线通过本次预设独立测试数值门槛；正式显著性与区间须P11另做。' if usable else '排序路线未全部通过预设测试门槛。结果有效，保留各组/条件损失，不根据测试集重训、换种子或放宽条件；后续需讨论主张范围或使用新的独立开发/确认数据，不能把本测试集反复用于选方案。'),'实际研究调用（意图/返回/未知）：'+str(ledger),'准备验证另见logs/P10/heldout_v36/preflight/checks.json；1080条件或4320方法单元不是精确网络前向次数。','旧P7未知调用2990905cbfa24838bb5e7b932827f9d7仍保持未知。','数据来源为固定第三方镜像修订；CRC32和本地SHA验证不等于官方全包认证。','证据：results/P10/heldout_v36/metrics.json、logs/P10/heldout_v36/readonly.json与cold.json、calls.jsonl。']
 path.write_text('\n'.join(lines)+'\n',encoding='utf-8');sealed(LOG/'progress.json',dict(at=now(),previous_all=72,previous_experiment=90.,P10_earned_points=8,all_tasks_percent=80,experiment_percent=100.,ranking_scientific_usable=usable,basis='Complete predeclared 90scene fixed-frame 12condition heldout evaluation, independent full acceptance, cold reentry and true report'))
 sealed(BASE/'closeout.json',dict(at=now(),engineering_passed=True,ranking_scientific_usable=usable,all_tasks_percent=80,experiment_percent=100.,report=stat(path),calls=ledger,cold_child_call_id=coldcalls[0]['call_id'],scope=c['scope'],P11_started=False,stopped=True))
 h=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(h,LOG/('handoff_before_close_'+uuid.uuid4().hex+'.txt'));h.write_text(f'最新：{now()} P10 v36固定90场景frame64×12条件、D0与三种子全部完成，完整独立复验与真实冷重入通过并停止。排序路线科学门槛通过={usable}；原P9校准失败保留。全部任务总进度80%、实验总进度100%（有限交付权重，非全部科学成功或全256帧基准）。报告docs/P10独立测试与鲁棒性结果_v36.txt。未进入P11或周期任务，等待用户检查。\n\n历史：\n'+h.read_text(encoding='utf-8-sig'),encoding='utf-8');p=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(p,LOG/('supervision_before_close_'+uuid.uuid4().hex+'.json'));v=read(p);v.update(recorded_at=now(),current_execution_state='P10_completed_stopped',all_tasks_percent_planning_estimate=80,experiment_percent_planning_estimate=100.,current_progress_evidence='logs/P10/heldout_v36/progress.json',current_execution_evidence='results/P10/heldout_v36/closeout.json',summary_report=path.relative_to(ROOT).as_posix(),waiting_for_user_next_part_confirmation=True,unchanged_progress_reason=None);atomic(p,v);atomic(ROOT/'runtime/pipeline/p10_part_v36.json',dict(at=now(),phase='P10_completed_stopped',all_tasks_percent=80,experiment_percent=100.,ranking_scientific_usable=usable,P11_started=False));b.save('complete');print('P10 complete and stopped; ranking usable',usable,flush=True)
def deliver():
 from scripts.pipeline import project_lock
 c=verify();rows=[json.loads(x) for x in (LOG/'controller.jsonl').read_text().splitlines()] if (LOG/'controller.jsonl').exists() else [];need({x['call_id'] for x in rows if x['kind']=='child_intent'}=={x['call_id'] for x in rows if x['kind']=='child_returned'},'Unknown child; inspect actual processes before recovery');prior=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'));start=time.monotonic();usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
 def guard():atomic(usage,dict(at=now(),seconds=time.monotonic()-start));need(prior+time.monotonic()-start<c['wall_seconds'],'Cumulative P10 controller cap')
 def stage(phase,script,args,done):
  if done.exists():load(done);return
  with project_lock(ROOT):pass
  guard();cid=uuid.uuid4().hex;cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B','scripts/'+script]+args;control('child_intent',call_id=cid,phase=phase,command=cmd)
  with (LOG/f'{phase}_{cid}.stdout.log').open('xb') as out,(LOG/f'{phase}_{cid}.stderr.log').open('xb') as err:
   proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);control('child_started',call_id=cid,phase=phase,controller_pid=os.getpid(),child_pid=proc.pid);atomic(ROOT/'runtime/pipeline/p10_delivery_v36.json',dict(at=now(),phase=phase,controller_pid=os.getpid(),child_pid=proc.pid,args=args,all_tasks_percent=72,experiment_percent=90.,P11_started=False))
   try:
    while proc.poll() is None:guard();time.sleep(2)
   except BaseException:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],cwd=str(ROOT),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False);raise
  control('child_returned',call_id=cid,phase=phase,exit_code=proc.returncode);need(proc.returncode==0,'P10 stage failed with evidence retained: '+phase);need(done.exists(),'Child returned without sealed stage completion');load(done)
 stage('assets','p10_assets_v36.py',['fetch'],BASE/'assets.json');stage('generation','p10_generate_v36.py',[],BASE/'generation_complete.json')
 for scene in sum(c['groups'].values(),[]):stage('features','p10_geometry_v36.py',['features',str(scene)],BASE/'features_done'/f'{scene:04d}.json')
 if not (BASE/'selection_complete.json').exists():sealed(BASE/'selection_complete.json',dict(at=now(),scene_conditions=1080,all_selections_before_labels=True))
 for scene in sum(c['groups'].values(),[]):stage('scoring','p10_geometry_v36.py',['score',str(scene)],BASE/'scoring_done'/f'{scene:04d}.json')
 if not (BASE/'scoring_complete.json').exists():sealed(BASE/'scoring_complete.json',dict(at=now(),scene_conditions=1080))
 stage('metrics','p10_metrics_v36.py',[],BASE/'metrics.json')
 for scene in sum(c['groups'].values(),[]):stage('review','p10_review_v36.py',['scene',str(scene)],LOG/'reviewed_scenes'/f'{scene:04d}.json')
 stage('full_readonly','p10_review_v36.py',['final'],LOG/'readonly.json');stage('cold','p10_review_v36.py',['cold'],LOG/'cold.json');stage('finish','p10_lifecycle_v36.py',['finish'],BASE/'closeout.json');guard();r=load(BASE/'closeout.json');control('completed_stopped',ranking_scientific_usable=r['ranking_scientific_usable']);atomic(ROOT/'runtime/pipeline/p10_delivery_v36.json',dict(at=now(),phase='P10_completed_stopped',all_tasks_percent=80,experiment_percent=100.,ranking_scientific_usable=r['ranking_scientific_usable'],P11_started=False));print('P10 whole part finished and stopped',flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 mode=sys.argv[1]
 if mode in ['prepare','finish']:
  with project_lock(ROOT):prepare() if mode=='prepare' else finish()
 elif mode=='deliver':
  import msvcrt
  p=ROOT/'runtime/pipeline/p10_v36.lock'
  with p.open('a+b') as f:
   f.seek(0,2)
   if f.tell()==0:f.write(b'0');f.flush()
   f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   try:deliver()
   except Exception as e:control('failure_preserved',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e));atomic(ROOT/'runtime/pipeline/p10_delivery_v36.json',dict(at=now(),phase='engineering_failure_preserved',error=repr(e),all_tasks_percent=72,experiment_percent=90.,P11_started=False));raise
   finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
 else:raise RuntimeError('Unknown lifecycle mode')
