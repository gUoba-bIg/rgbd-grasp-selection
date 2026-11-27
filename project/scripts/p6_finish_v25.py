"""Honest positive/negative closeout and real zero-call cold reentry."""
import sys,subprocess
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v25 import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def snapshot(c):
 paths={ROOT/p for p in c['protected']};paths.update(p for folder in [BASE,LOG] for p in folder.rglob('*') if p.is_file() and p.suffix not in ['.log','.jsonl']);paths.add(CONFIG)
 return {p.relative_to(ROOT).as_posix():stat(p) for p in sorted(paths)}

def run():
 c=verify();hold=read(BASE/'holdout_summary.json');need(read(LOG/'holdout_readonly.json')['passed'],'Holdout review absent');develop=read(BASE/'development_summary.json') if hold['ready'] else None
 if develop is not None:need(read(LOG/'development_readonly.json')['passed'],'Full regression review absent')
 selected=develop or hold;ready=hold['ready'] and bool(develop and develop['ready']);out=BASE/'figures';out.mkdir(parents=True,exist_ok=True)
 ds=c['core_conditions'];fig,ax=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
 for group in ['main','boundary']:ax[0].plot(ds,[selected['conditions'][d]['macro'][group]['coverage'][2] for d in ds],'-o',label=group)
 ax[0].axhline(.95,color='black',ls='--');ax[0].set_ylim(0,1.05);ax[0].set_ylabel('Central 95% mixture predictive coverage');ax[0].legend()
 ax[1].bar(ds,[selected['conditions'][d]['macro']['main']['constant_nll']-selected['conditions'][d]['macro']['main']['nll'] for d in ds]);ax[1].axhline(0,color='black');ax[1].set_ylabel('Fixed 5mm NLL minus mixture NLL')
 fig.suptitle('P6 one-shot error-model repair: '+('exposed development regression' if develop else 'error-model-only scene holdout')+'\nKnown synthetic perturbations; no grasp or real-sensor calibration claim',fontsize=11);fig.savefig(out/'repair_coverage.png',dpi=170);fig.savefig(out/'repair_coverage.pdf');plt.close(fig)
 before=snapshot(c);once(LOG/'cold_before.json',before);calls=counts();commands=[('vision','scripts/p6_data_v25.py',[]),('p4_eval_v22','scripts/p6_evaluate_v25.py',['fit']),('p4_eval_v22','scripts/p6_evaluate_v25.py',['holdout'])]
 if develop:commands.append(('p4_eval_v22','scripts/p6_evaluate_v25.py',['development']))
 cold=[]
 for i,(env,script,args) in enumerate(commands):
  command=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script]+args
  with (LOG/f'cold_{i}.stdout.log').open('x',encoding='utf-8') as so,(LOG/f'cold_{i}.stderr.log').open('x',encoding='utf-8') as se:p=subprocess.run(command,cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
  need(p.returncode==0,'Cold process failed');cold.append(dict(command=command,returncode=p.returncode))
 need(counts()==calls,'Cold calls added');need(all(stat(ROOT/p)==v for p,v in before.items()),'Cold mutated artifacts');once(LOG/'cold_reentry.json',dict(passed=True,commands=cold,new_calls=0,unchanged_files=len(before)))
 expected_replays=2482 if develop else 882;expected_predictions=1760 if develop else 160
 expected=dict(quality_forward=402,segmentation_forward=81,corruption_replay=expected_replays,error_model_fit=1,mixture_prediction=expected_predictions)
 need(all(calls[k]['known_returns']==v and calls[k]['unknown']==0 for k,v in expected.items()),'Actual call scope differs')
 need(usage()<c['limits']['seconds'] and storage()<c['limits']['bytes'],'Resource budget exceeded')
 result=dict(at=now(),status='P6_v25_complete_stopped',engineering_passed=True,scientifically_ready_on_declared_scope=ready,holdout_passed=hold['ready'],holdout_cases=80,full_regression_cases=800 if develop else 0,fit_cases=320,new_neural_updates=0,error_model_fits=1,new_official_AP=0,new_MC=0,actual_calls=calls,recorded_compute_seconds=usage(),old_v24_unchanged=True,full_original_P6_main_node_complete=False,P7_started=False,progress=dict(all_tasks=41,experiments=51.25,additional_points=0,reason='Same P6 work package repair; no double credit'))
 once(BASE/'closeout.json',result)
 lines=['P6单次误差模型修复结果 v25',now(),'','本次采用独立于误差拟合的4个train场景进行有限关口检查；它们曾参与旧质量网络训练，因此不是完整网络独立测试。16场景320条件项拟合一次，4场景80条件项关口。',('留出全部通过，已完成旧10开发场景800项回归评价。旧开发集已暴露，不是新的独立确认集。' if develop else '留出科学门槛未全部通过，按冻结协议停止；完整800项开发回归未运行，没有修改门槛或再调参。'),f'留出科学门槛：{hold["scientific_gates"]}']
 if develop:lines.append(f'完整开发科学门槛：{develop["scientific_gates"]}')
 lines+=['','结论：'+('模型通过本次有界合成扰动准入，仍无真实传感器校准或抓取提升证据。' if ready else '几何与概率计算接口通过工程验收，但修复模型未通过科学准入，不能作为可信碰撞概率。'),'混合CDF中央95%覆盖与混合协方差分别计算；未把混合协方差椭球误称95%概率区域。','条件 | 中央95%主覆盖 | 边界95%覆盖 | 修正3D RMSE(mm)']
 for d in ds:
  m=selected['conditions'][d]['macro'];lines.append(f'{d} | {m["main"]["coverage"][2]:.6f} | {m["boundary"]["coverage"][2]:.6f} | {m["main"]["rmse3d"]*1000:.6f}')
 lines += ['',f'四退化NLL平均改善={selected["mean_degraded_nll_improvement"]:.6f}，最差场景95覆盖={selected["worst_scene_coverage95"]:.6f}。','实际调用：'+str(calls),f'新记录计算{usage():.3f}秒；v24旧记录620.735秒另列，不包含全部编写和等待时间。','全部400项数据来源和所执行评价均经独立实现完整复验；真实冷重入0新增调用；旧冻结成果未变。','','后续建议：'+('先补齐P2数据覆盖和P3完整抓取基线，再决定P7；当前不自动启动。' if ready else '停止继续扩大同类sigma/混合分布调参。保留普通RGBD与未校准质量特征路线，优先补齐P2/P3真实抓取基线；若仍需可信概率，应建立有独立参考的深度/错位误差训练与验证，而非在已暴露开发集反复缩放。'),'旧门控负结果与v24负结果保留；P8科学关口通过才进入三种子正式实验。','全部任务总进度41%，实验总进度51.25%，固定工作包估算；本次修复不重复加分。','P6单次修复整体停止，未进入P7，未恢复自动任务。']
 report=ROOT/'docs/P6单次误差模型修复结果_20261030.txt';need(not report.exists(),'Report already exists');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before_closeout.txt');hand.write_text('最新：'+now()+f' P6 v25单次修复已完成并停止；留出通过={hold["ready"]}，全开发回归={800 if develop else 0}项，科学可用={ready}；工程复验及冷重入通过，旧v24不变。全部41%、实验51.25%，修复不重复计分。报告docs/P6单次误差模型修复结果_20261030.txt。P2/P3仍缺口；不启动P7/周期任务，等用户下一部分确认。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_closeout.json');row=read(ctrl);row.update(recorded_at=now(),current_execution_state='P6_v25_complete_stopped',current_execution_evidence='results/P6/error_mixture_v25/closeout.json',current_runner=None,active_internal_repair=None,waiting_for_user_next_part_confirmation=True,summary_report=report.relative_to(ROOT).as_posix(),final_readonly='logs/P6/error_mixture_v25/final_readonly.json');atomic(ctrl,row)
 status('P6_v25_complete_stopped',scientifically_ready=ready,waiting_for_user=True)
 final=snapshot(c);final[report.relative_to(ROOT).as_posix()]=stat(report);once(LOG/'final_readonly.json',dict(passed=True,files=final,cold_reentry_passed=True,P7_started=False));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
