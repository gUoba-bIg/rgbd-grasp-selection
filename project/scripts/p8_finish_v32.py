"""Report real P8 outcomes and close the authorized part without entering P9."""
import sys,shutil,uuid
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_common_v32 import *
def run():
 c=verify();ro=load(LOG/'readonly.json');cold=load(LOG/'cold.json');need(ro['passed'] and cold['passed'] and ro['files']==cold['files'],'P8 final read-only/cold missing')
 for p,s in cold['files'].items():need(stat(ROOT/p)==s,'Final immutable artifact changed '+p)
 result=load(BASE/'metrics.json');ledger=counts();usable=result['scientific_usable'];report=ROOT/'docs/P8整体消融结果_v32.txt';lines=['P8整体开发消融结果 v32',now(),'',f'P8固定工作包已完成，科学关口通过={usable}；完整数值只读、真实新进程冷重入、P7原文件不变性均通过。',f'全部任务总进度64%（64/100）；实验总进度80%（64/80）。这是固定交付权重点数，不是成功率或剩余时间比例。','当前部分：P8完成并停止；没有启动P9，没有周期任务。','',c['scope'],'验证15场景、每场景第64视角和5条件；同一单种子，验证集参与风险模型epoch选择，不能称独立测试结果。','','各组场景平均抓取AP（比例0至1；每行clean/noise5/missing15/shift3/joint/四退化平均）：']
 for m in c['groups_definition']:
  values=[result['condition_AP'][m][d] for d in c['conditions']]+[result['degraded_mean_AP'][m]];lines.append(m+' '+c['groups_definition'][m]+' '+', '.join(f'{v:.8f}' for v in values))
 lines+=['',f"主要对照：D5相对简单基线{result['best_preregistered_simple']}，四退化平均AP差值{result['mean_degraded_AP_gain']*100:+.4f}个百分点；干净抓取AP损失{result['clean_grasp_AP_loss']*100:.4f}个百分点。",f"同候选源风险排序D6-D0：{result['fixed_pool_risk_gain']*100:+.4f}个百分点；前景输入链路D5-D6：{result['foreground_chain_gain']*100:+.4f}个百分点。",f"温度T={result['temperature']}；验证15等频ECE相对改善={result['relative_ECE15_reduction']}。D4/D5正温度缩放不改变排序，抓取AP相同。",'八组使用同一分割图，因此干净mask AP组间差值0；这是不变性，不是分割改善证据。','科学关口：'+str(result['gates']),'','可用性与后续：']
 if usable:lines+=['本次开发范围内核心科学关口通过，可以据此拟定P9三个随机种子训练；P9仍须用户查看并确认后启动。不能把该结果外推为90独立测试场景或机器人成功。']
 else:
  lines+=['当前完整D5方案不满足P8全部科学关口，不能直接进入P9。负结果与已完成工程交付分别记录，未达标项不改写为成功。','有限修复建议（只提出，未启动）：']
  if result['foreground_chain_gain']<0:lines+=['前景裁剪降低抓取AP时，优先保留完整观测云及D6风险排序；下一版只检验这一输入路线，避免再做整套昂贵前景评价。']
  if result['ablation_gains']['D3']<0 or result['ablation_gains']['D7']<0:lines+=['若删特征组优于完整模型，先按本次已公开结果提出单次删特征协议，并用未用于选轮的验证范围确认；不能直接事后替换D5宣布本轮通过。']
  if not result['gates']['calibration_ECE_reduction']:lines+=['温度校准未达到预设ECE改善时，保留未校准排序，不宣传可信概率；下一版先做固定小范围校准有效性关口，再决定是否评价完整链路。']
  lines+=['修复须保留本次所有负结果、事先固定范围与累计预算；用户确认后再运行，不继续追加诊断。']
 lines+=['','实际新增调用账本（意图/已返回/未知ID）：',str(ledger),'旧P7未知上游调用2990905cbfa24838bb5e7b932827f9d7仍保留，不计作已返回。','完整证据：results/P8/ablation_v32/metrics.json、logs/P8/ablation_v32/readonly.json、cold.json及calls.jsonl。','后续未完成工作：按科学关口决定P9多种子训练，P10锁定后独立测试，P11统计图表与检验，P12论文及复现交付。']
 need(not report.exists(),'P8 report exists without closeout; preserve before recovery');report.write_text('\n'.join(lines)+'\n',encoding='utf-8');sealed(LOG/'progress.json',dict(at=now(),previous_all=56,previous_experiment=70.,earned_P8_points=8,all_tasks_percent=64,experiment_percent=80.,basis='fixed bounded P8 eight-group deliverable, full numerical review, cold reentry and true scientific report; negative result counts as completed work',scientific_gate_passed=usable))
 sealed(BASE/'closeout.json',dict(at=now(),engineering_passed=True,scientific_usable=usable,all_tasks_percent=64,experiment_percent=80.,report=stat(report),calls=ledger,P9_started=False,recurring_automation=False,scope=c['scope']))
 hand=ROOT/'docs/实验自动续接说明.txt';backup=LOG/('handoff_before_close_'+uuid.uuid4().hex+'.txt');shutil.copy2(hand,backup);hand.write_text(f'最新：{now()} P8固定8组开发消融及完整只读、冷重入、P7不变性已完成并停止；科学关口通过={usable}。全部任务总进度64%、实验总进度80%（固定交付权重点数）。报告docs/P8整体消融结果_v32.txt；不得自动启动P9或周期任务，等待用户查看。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 p=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(p,LOG/('supervision_before_close_'+uuid.uuid4().hex+'.json'));s=read(p);s.update(recorded_at=now(),current_authorized_part='P8',current_execution_state='P8_completed_stopped',all_tasks_percent_planning_estimate=64,experiment_percent_planning_estimate=80.,current_progress_evidence='logs/P8/ablation_v32/progress.json',current_execution_evidence='results/P8/ablation_v32/closeout.json',waiting_for_user_next_part_confirmation=True,summary_report='docs/P8整体消融结果_v32.txt',final_readonly='logs/P8/ablation_v32/readonly.json',stop_after_part='P8',stop_after='P8',unchanged_progress_reason=None);atomic(p,s)
 atomic(ROOT/'runtime/pipeline/p8_part_v32.json',dict(at=now(),phase='P8_completed_stopped',all_tasks_percent=64,experiment_percent=80.,scientific_usable=usable,P9_started=False,recurring_automation=False,result='results/P8/ablation_v32/closeout.json'));print('\n'.join(lines),flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()
