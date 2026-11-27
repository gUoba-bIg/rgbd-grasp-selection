"""True repair outcome, unchanged earned progress, then stop for user review."""
import sys,shutil,uuid
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_repair_common_v33 import *
def run():
 c=verify();b=Budget(c,'finish');ro=load(LOG/'readonly.json');cold=load(LOG/'cold.json');need(ro['passed'] and cold['passed'] and ro['files']==cold['files'],'Repair acceptance missing')
 for p,s in cold['files'].items():b.guard();need(stat(ROOT/p)==s,'Repair final output changed '+p)
 result=load(BASE/'result.json');screen=load(BASE/'holdout_screen.json');ledger=counts();need(all(not v['unknown'] for v in ledger.values()),'Unresolved new call');rank=result['ranking'];ok=result['scientific_usable'];report=ROOT/'docs/P8完整点云修复结果_v33.txt';lines=['P8完整点云＋校准修复结果 v33',now(),'',f'本次修复已完成并停止；工程完整只读和真实新进程冷重入通过；修复科学门槛通过={ok}。','全部任务总进度64%；实验总进度80%。固定工作包内部修复不重复记分，不是成功率。','旧P8 v32前景裁剪方案仍为负结果，没有回写或改称成功。P9未启动，周期任务未恢复。','',c['scope'],'本次修复受已公开的开发结果启发；未证明新场景独立泛化或机器人成功。','','输入路线：取消前景硬裁剪，使用已验收的完整点云候选与完整风险模型；抓取AP和分割图复用后经独立核对，不重训/重评已完成计算。',f"四退化平均抓取AP：原始排序D0={rank['degraded_AP']['D0']:.8f}，风险排序D6={rank['degraded_AP']['D6']:.8f}；对固定简单基线{rank['best_simple']}提高{rank['mean_degraded_AP_gain']*100:.4f}个百分点。",f"干净抓取AP损失={rank['clean_AP_loss']*100:.4f}个百分点（负数表示提升）；mask AP差值0仅为分割未改。",'概率校准仅适用于原始置信度top20观测候选池的离线mu<=.4标签，不能泛化为所有风险筛选后候选的真实成功概率。','',f"15场景留一关口：通过={screen['passed']}；未校准NLL={screen['raw']['nll']:.9f}，留出校准NLL={screen['held_out']['nll']:.9f}。"]
 if result['validation_executed']:
  lines+=['最终正斜率仿射参数[a,b]：'+str(result['final_parameters']),'验证场景概率结果（15等频ECE、NLL、Brier）：']
  for method,m in result['probability']['validation'].items():lines.append(f"{method}: ECE15={m['ece15']:.9f}, NLL={m['nll']:.9f}, Brier={m['brier']:.9f}")
  lines+=[f"主方案E3相对未校准E1的ECE15改善：{result['relative_ECE15_reduction']*100:.4f}%（要求至少10%）。",'E2旧温度仅为固定参照；没有根据验证结果更换主方案。E1/E2/E3使用同一原风险排序，因此没有把校准当成AP提升来源。']
 else:lines+=['留一关口未通过，按冻结协议未拟合最终参数、未执行修复版验证指标；不增加第二种校准器或调参。']
 lines+=['','科学门槛：'+str(result['gates']),'','可用性与下一步：']
 if ok:lines+=['本次修复在约定开发范围通过，可作为P9多随机种子验证的候选路线；待用户查看确认后再进入P9。前景裁剪和旧未达标概率方案继续退出主路线。']
 else:
  lines+=['当前修复主方案仍不能直接进入P9。完整点云风险排序的开发收益保留；概率校准未通过的部分不能宣称可用。']
  if result['validation_executed']:
   v=result['probability']['validation'];old_ok=v['E1']['ece15']>0 and v['E2']['ece15']<=.9*v['E1']['ece15'] and v['E2']['nll']<=v['E1']['nll']+.001
   if old_ok:lines+=['固定参照E2旧温度在这组开发数据达到概率门槛，可作为下一次最小复用方案的候选；本次主方案E3失败不能被事后替换，需另行确认协议。']
   else:lines+=['后续建议：保留完整点云风险排序，暂停扩量训练；如继续追求概率声明，先明确新的校准假设和固定留出关口，或经用户同意缩减概率声明。不能靠放宽当前阈值通过。']
  else:lines+=['后续建议：先分析各留出场景误差及样本支持，固定一个有明确理由的新校准假设；本次不自动追加试验。']
 lines+=['','新增真实调用：'+str(ledger),'新增神经网络更新、神经前向、候选生成、官方抓取评分和mask AP均为0；标量拟合与应用单独列账。',f"继承P8 v32控制链墙时={c['inherited_P8_wall_seconds']:.3f}秒；本次固定累计上限1800秒，旧调用账本和未知历史均保留。",'P7历史未知上游调用2990905cbfa24838bb5e7b932827f9d7未改写为返回。','证据：results/P8/whole_cloud_repair_v33/result.json、holdout_screen.json；logs/P8/whole_cloud_repair_v33/readonly.json、cold.json、calls.jsonl。']
 need(not report.exists(),'Existing unclosed repair report must be preserved');report.write_text('\n'.join(lines)+'\n',encoding='utf-8');sealed(BASE/'closeout.json',dict(at=now(),engineering_passed=True,scientific_usable=ok,all_tasks_percent=64,experiment_percent=80.,new_progress_points=0,report=stat(report),calls=ledger,inherited_P8_calls=c['inherited_P8_calls'],old_P8_scientific_result=False,P9_started=False,scope=c['scope']));sealed(LOG/'progress.json',dict(at=now(),previous_all=64,previous_experiment=80.,all_tasks_percent=64,experiment_percent=80.,earned_points=0,reason='repair within already completed P8 work package; no double counting',scientific_usable=ok))
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/('handoff_before_close_'+uuid.uuid4().hex+'.txt'));hand.write_text(f'最新：{now()} P8完整点云修复v33已完成并停止；15场景留一通过={screen["passed"]}，修复科学关口通过={ok}，完整只读/冷重入和旧P7/P8不变性通过。全部任务总进度64%、实验总进度80%，内部修复不重复计分。报告docs/P8完整点云修复结果_v33.txt；原v32负结果保留，不自动进入P9或周期任务，等待用户检查。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 p=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(p,LOG/('supervision_before_close_'+uuid.uuid4().hex+'.json'));s=read(p);s.update(recorded_at=now(),current_authorized_part='P8修复v33',current_execution_state='P8_repair_completed_stopped',active_internal_repair=None,current_execution_evidence='results/P8/whole_cloud_repair_v33/closeout.json',current_progress_evidence='logs/P8/whole_cloud_repair_v33/progress.json',waiting_for_user_next_part_confirmation=True,summary_report='docs/P8完整点云修复结果_v33.txt',final_readonly='logs/P8/whole_cloud_repair_v33/readonly.json',unchanged_progress_reason='P8内部修复不重复计分',stop_after='P8_repair',stop_after_part='P8');atomic(p,s);status('P8_repair_completed_stopped',scientific_usable=ok,summary_report='docs/P8完整点云修复结果_v33.txt');b.save('complete');print('\n'.join(lines),flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
