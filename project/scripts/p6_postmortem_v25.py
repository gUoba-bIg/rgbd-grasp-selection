"""Post-run paired diagnostic only: reads accepted samples, no model calls or tuning."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v25 import *
from scripts.pipeline import project_lock
import numpy as np
from scipy.stats import chi2

def run():
 c=verify();proof=read(LOG/'final_readonly.json');need(proof['passed'],'No final pass');need(all(stat(ROOT/p)==v for p,v in proof['files'].items()),'Accepted artifact changed');before=counts();result=read(BASE/'closeout.json');need(before==result['actual_calls'],'Ledger differs')
 need(read(LOG/'delivery_complete.json')['stopped'],'Controller not stopped');summary=read(BASE/'holdout_summary.json');rows=[]
 for r in c['holdout_records']:
  for d in c['core_conditions']:
   a,row=load_sample(BASE/'samples/holdout'/(name(r,d)+'.npz'));res=a['observed']-a['reference'];z=(res/a['sigma_old'])**2;groups={}
   for label in ['main','boundary']:
    mask=a[label];groups[label]=dict(n=int(mask.sum()),covered95=int(np.sum(z[mask]<=chi2.ppf(.95,1))),squared_error=float(np.sum((res[mask,None]*a['rays'][mask])**2)))
   rows.append(dict(scene=r['scene'],condition=d,groups=groups))
 paired={}
 for d in c['core_conditions']:
  old={}
  for group in ['main','boundary']:
   values=[]
   for sc in c['holdout_scenes']:
    rr=[x['groups'][group] for x in rows if x['scene']==sc and x['condition']==d];n=sum(x['n'] for x in rr);values.append([sum(x['covered95'] for x in rr)/n,np.sqrt(sum(x['squared_error'] for x in rr)/n)])
   old[group]=dict(coverage95=float(np.mean([x[0] for x in values])),raw_rmse3d=float(np.mean([x[1] for x in values])))
  paired[d]=dict(old_gaussian=old,new_mixture=summary['conditions'][d]['macro'])
 once(BASE/'paired_holdout_diagnostic.json',dict(posthoc_descriptive_only=True,changes_to_frozen_gates=False,same_80_cases_and_same_points=True,new_model_calls=0,conditions=paired))
 report=ROOT/'docs/P6修复失败分析与下一步_20261030.txt';need(not report.exists(),'Postmortem exists')
 lines=['P6 v25修复失败分析（只读补充，未新增拟合/前向）',now(),'','在相同80项留出案例、相同点和场景等权口径比较，避免把v24旧10开发场景与本轮4留出场景直接当配对改善。旧模型以零均值高斯95%区域评价；新模型以混合中央95%区域评价，分布定义分别保留。','条件 | 旧主95覆盖 | 新主95覆盖 | 旧边界95覆盖 | 新边界95覆盖']
 for d,v in paired.items():lines.append(f'{d} | {v["old_gaussian"]["main"]["coverage95"]:.6f} | {v["new_mixture"]["main"]["coverage"][2]:.6f} | {v["old_gaussian"]["boundary"]["coverage95"]:.6f} | {v["new_mixture"]["boundary"]["coverage"][2]:.6f}')
 lines+=['','已观察到的问题：新模型错位条件边界95覆盖80.15%，未达85%；clean相对合成参考的三维RMSE从原始0变为均值修正后的1.842mm，missing有效点从0变为1.667mm。后者是事后描述性副作用，不是事后新增或替换的准入门槛。','判断：仅按噪声代理、局部范围、原sigma和局部残差分格的误差分布，尚不足以表达实例边界的方向性错位；这是依据误差分布和输入信息作出的解释，不是已证实唯一原因。不能靠总体NLL改善掩盖边界欠覆盖，也不能将拟合成功等同科学可用。','','建议的下一修复路线（未运行）：','1. 保留未修正的观测坐标与已验收的传播接口，暂不把混合均值修正写入抓取主链路。','2. 若继续概率路线，增加可观测的RGB—深度边缘错位、有符号水平/垂直梯度和预测实例边界信息，专门建模空间错位及异常尾部；训练只用独立训练场景。','3. 运行前另冻结数据、单次预算和停止条件；保留当前原覆盖门槛，另明确clean几何偏差不可退化的保护条件。原80项已暴露，只作回归检查；新的确认场景需与拟合、调参隔离，不打开最终测试集。','4. 同时必须补齐P2最终数据覆盖和P3完整抓取基线，再判断是否有必要投入正式P7。当前不自动启动下一修复或P7。','','全部任务总进度41%，实验总进度51.25%，修复不重复计分。']
 report.write_text('\n'.join(lines)+'\n',encoding='utf-8');need(counts()==before,'Diagnostic added model calls')
 need(all(stat(ROOT/p)==v for p,v in proof['files'].items()),'Postmortem changed frozen files')
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before_postmortem.txt');hand.write_text('补充：'+now()+' v25根只读复核完成；同80项配对诊断及clean均值修正副作用见docs/P6修复失败分析与下一步_20261030.txt。错位边界仍不合格，P6修复已停止；不运行800开发回归/P7。全部41%、实验51.25%不变。\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 once(LOG/'root_final_review.json',dict(at=now(),passed=True,final_artifacts_rehashed=len(proof['files']),old_sources_unchanged=True,project_lock_acquired=True,actual_calls=before,engineering_passed=True,scientific_adoption=False,full_development_not_run=True,P7_started=False,visual_review='repair_coverage.png inspected: readable titles/axes/legend; no clipping; values match summary',postmortem_files={p.relative_to(ROOT).as_posix():stat(p) for p in [Path(__file__),report,BASE/'paired_holdout_diagnostic.json']},progress=dict(all_tasks=41,experiments=51.25)))
 print(json.dumps(dict(paired=paired,final_rehashed=len(proof['files'])),ensure_ascii=False))
if __name__=='__main__':
 with project_lock(ROOT):run()
