"""Figures, actual cold entry and scoped P6 delivery; no next-part execution."""
import sys,subprocess,time
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def snapshot(c):
 paths={ROOT/p for p in c['protected']};paths.update(p for p in BASE.rglob('*') if p.is_file());paths.update([CONFIG,LOG/'readonly.json',LOG/'preflight_readonly.json',LOG/'quality_identity.json']);return {p.relative_to(ROOT).as_posix():stat(p) for p in sorted(paths)}
def figures(summary):
 out=BASE/'figures';out.mkdir(parents=True,exist_ok=True);ds=['clean','noise5','missing15','shift3','joint'];x=np.arange(5);fig,ax=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
 for key,label in [('coverage','Quality sigma'),('constant_coverage','Fixed5mm')]:ax[0].plot(x,[summary['conditions'][d]['macro']['main'][key][2] for d in ds],'-o',label=label)
 ax[0].axhline(.95,color='black',ls='--',lw=1);ax[0].set_xticks(x,ds);ax[0].set_ylim(0,1.05);ax[0].set_ylabel('Scene-macro nominal95% coverage');ax[0].legend()
 ax[1].bar(x,[summary['conditions'][d]['macro']['main']['constant_nll']-summary['conditions'][d]['macro']['main']['nll'] for d in ds],color='#436A9A');ax[1].set_xticks(x,ds);ax[1].axhline(0,color='black',lw=1);ax[1].set_ylabel('Fixed5mm NLL minus quality NLL');ax[1].margins(y=.2)
 fig.suptitle('P6: depth-only rank1 uncertainty on exposed image development data\nKnown perturbation reference; no real-sensor calibration or grasp-AP claim',fontsize=12);fig.savefig(out/'p6_coverage.png',dpi=170);fig.savefig(out/'p6_coverage.pdf');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
 for group,label in [('main','Uniform valid-point sample'),('boundary','Predicted instance boundary')]:ax[0].plot(x,[summary['conditions'][d]['macro'][group]['coverage'][2] for d in ds],'-o',label=label)
 ax[0].set_xticks(x,ds);ax[0].set_ylim(0,1.05);ax[0].axhline(.95,ls='--',color='black');ax[0].legend();ax[0].set_ylabel('Nominal95% quality covariance coverage')
 for group in ['main','boundary']:ax[1].plot(x,[1000*summary['conditions'][d]['macro'][group]['rmse3d'] for d in ds],'-o',label=group)
 ax[1].set_xticks(x,ds);ax[1].set_ylabel('3D error RMSE (mm)');ax[1].set_ylim(bottom=0);ax[1].legend();fig.suptitle('P6: boundary diagnosis; every selected point retained');fig.savefig(out/'p6_boundary.png',dpi=170);fig.savefig(out/'p6_boundary.pdf');plt.close(fig)
 once(out/'receipt.json',dict(files={p.relative_to(ROOT).as_posix():stat(p) for p in out.iterdir() if p.is_file()},new_model_calls=0))
def run():
 c=verify();need(read(LOG/'readonly.json')['passed'],'Full P6 readonly not passed');summary=read(BASE/'summary.json');need(summary['positive_control_passed'],'Noise positive control failed');figures(summary);before=snapshot(c);once(LOG/'cold_before.json',before);calls=counts();results=[]
 for tag,script in [('quality','scripts/p6_quality_v24.py'),('geometry','scripts/p6_compute_v24.py')]:
  out=LOG/f'cold_{tag}.stdout.log';err=LOG/f'cold_{tag}.stderr.log';need(not out.exists() and not err.exists(),'Existing cold attempt needs review');env='vision' if tag=='quality' else 'p4_eval_v22';command=[str(ROOT/f'runtime/envs/{env}/Scripts/python.exe'),'-B',script];start=time.monotonic()
  with out.open('x',encoding='utf-8') as so,err.open('x',encoding='utf-8') as se:p=subprocess.run(command,cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
  results.append(dict(command=command,returncode=p.returncode,seconds=time.monotonic()-start));need(p.returncode==0,'Actual cold entry failed')
 need(counts()==calls,'Cold entry added calls');need(all(stat(ROOT/p)==v for p,v in before.items()),'Cold entry changed files');once(LOG/'cold_reentry.json',dict(passed=True,commands=results,new_model_calls=0,new_corruption_replays=0,protected_files_unchanged=len(before)))
 need(calls['forward']['known_returns']==704 and calls['corruption_replay']['known_returns']==2304,'Actual execution scope differs');need(all(v['unknown']==0 for v in calls.values()),'Unknown outcomes remain');need(usage()<c['resources']['seconds'] and storage()<c['resources']['new_output_bytes'],'Recorded budget exceeded')
 ready=summary['raw_sigma_ready'];progress=dict(before_all=37,before_experiment=46.25,earned_points=4,after_all=41,after_experiment=51.25,work_packages=['Real-image instance/boundary interface2','Complete known-perturbation coverage/error evaluation and decision2'],denominators=dict(all=100,experiment=80),not_scientific_success_percent=True)
 once(LOG/'progress_after_P6.json',progress);result=dict(status='P6_image_development_complete_stopped',scope=c['scope'],original_full_P6_main_node_complete=False,engineering_passed=True,raw_quality_sigma_ready=ready,scientific_gates=summary['scientific_gates'],case_count=800,new_quality_inference_cases=700,reused_quality_cases=100,new_training=0,new_mask_or_grasp_AP=0,new_MC_samples=0,actual_calls=calls,recorded_compute_seconds=usage(),progress=progress,old_sources_unchanged=True,cold_reentry_passed=True,next_part_started=False,route='docs/P6开发协议与后续路线_v24.txt');once(BASE/'closeout.json',result)
 lines=['P6图像与实例传播开发结果及后续路线',now(),'','本阶段完成160个已暴露开发帧×5条件共800项；真实图像接口及已知合成扰动覆盖/误差完整评价已验收。原正式主P6依赖图未标complete，P2/P3仍须补齐。','固定RGBD采用P5原固定融合第9轮（源自P4父权重）；质量网络不重训，门控不采用。复用R19数学/采样验收，未重跑72案例。','三维传播是给定像素/内参的深度方向秩1协方差，不包含未估计的像素定位方差；原深度仅为合成扰动参考，不是真实传感器真值。','',('原始质量尺度通过本次事前开发准入门槛，仍不构成独立真实传感器校准。' if ready else '原始质量尺度未通过全部覆盖/误差准入门槛，不能直接解释为可信三维概率。工程接口可用，质量输出只保留为未校准特征，不自动扩大尺度或追加调参。'),'科学门槛：'+str(summary['scientific_gates']),'','条件 | 质量95覆盖 | 固定5mm95覆盖 | 预测边界95覆盖 | 平均3D RMSE(mm)']
 for d in c['core_conditions']:
  m=summary['conditions'][d]['macro'];lines.append(f"{d} | {m['main']['coverage'][2]:.6f} | {m['main']['constant_coverage'][2]:.6f} | {m['boundary']['coverage'][2]:.6f} | {1000*m['main']['rmse3d']:.6f}")
 lines += ['',f"四退化平均NLL改善={summary['mean_degraded_nll_improvement']:.6f}；最差场景95覆盖={summary['worst_scene_coverage95']:.6f}。",'所有选择点、背景实例0均保留；未通过删除低置信点制造低碰撞率。没有新抓取评分，不能声称抓取提高。','noise5固定5mm正对照通过；独立CPU逐点重建原像素/深度、JΣJᵀ、秩/支撑、覆盖/误差及决定，全800项通过；真实冷重入新增前向/重建0。',f'实际调用：{calls}。新研究更新0，官方AP0，蒙特卡洛新样本0；重建调用是同seed既有扰动重放，不是新增独立抽样。',f'记录计算{usage():.3f}秒，尚未包含全部方案编写/人工检查等待。','','后续路线：C0普通RGBD；C1简单质量惩罚；C2三维传播；C3分区风险；C4独立校准。门控退出主候选，旧负结果保留；AP/ECE门槛不改。',('下一步先补P2/P3完整抓取依赖，再考虑P7风险训练与独立校准。' if ready else '修复建议：在独立训练参考数据上建立含偏差/异常值的误差模型，检验边界和错位条件；之后在未用于拟合的场景验证覆盖。不能用本轮开发集调尺度后自称确认。P2/P3完整基线仍须补齐，P7须另经用户确认。'),'P8科学关口通过才进入P9；P10保留测试、P11统计、P12论文复现仍保留。','','全部任务总进度41%，实验总进度51.25%，固定工作包估算；本阶段新增4点交付，负结果同样计完成工作，不表示方法成功率。','P6本次开发整体已停止，不进入P7，不恢复自动任务。']
 report=ROOT/'docs/P6开发结果与后续路线_20261030.txt';need(not report.exists(),'Report exists');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,LOG/'handoff_before_closeout.txt');old=handoff.read_text(encoding='utf-8-sig');handoff.write_text('最新：'+now()+' P6 v24已完成800项开发评价、独立验收及真实冷重入并停止。原始质量sigma准入='+str(ready)+'；工程接口通过，真实传感器校准/抓取优势未建立。全部41%、实验51.25%（新增4点交付，非成功率）。报告docs/P6开发结果与后续路线_20261030.txt；后续门控退出主候选，P2/P3正式缺口保留。等用户确认，不进入P7或恢复自动任务，原state.json不改。\n\n历史：\n'+old,encoding='utf-8')
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_closeout.json');r=read(ctrl);r.update(recorded_at=now(),current_execution_state='P6_v24_complete_stopped',current_execution_evidence='results/P6/image_geometry_v24/closeout.json',current_runner=None,waiting_for_user_next_part_confirmation=True,all_tasks_percent_planning_estimate=41,experiment_percent_planning_estimate=51.25,current_progress_evidence='logs/P6/image_geometry_v24/progress_after_P6.json',summary_report=report.relative_to(ROOT).as_posix(),final_readonly='logs/P6/image_geometry_v24/final_readonly.json',unchanged_progress_reason='P6实际交付接口2点与完整覆盖/误差取舍2点');atomic(ctrl,r);atomic(ROOT/'runtime/pipeline/p6_part_v24.json',dict(at=now(),phase='P6_v24_complete_stopped',all_tasks_percent=41,experiment_percent=51.25,raw_sigma_ready=ready,waiting_for_user=True))
 final=snapshot(c);final[report.relative_to(ROOT).as_posix()]=stat(report);once(LOG/'final_readonly.json',dict(passed=True,files=final,actual_cold_reentry=True,no_next_part_started=True));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
