"""Single-repair reporting after actual independent verification and cold reentry."""
import sys,time,subprocess
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23r1 import *

def snapshot():
    c=verify();paths={ROOT/p for p in c['protected']};paths.update(p for d in [BASE,RUN] for p in d.rglob('*') if p.is_file())
    paths.update([CONFIG,ROOT/'configs/p5_evaluation_v23r1.json',LOG/'evaluation_freeze.json',LOG/'screen_readonly.json',LOG/'delivery_freeze.json'])
    for p,v in read(LOG/'delivery_freeze.json')['protected'].items():need(stat(ROOT/p)==v,'Delivery source changed '+p);paths.add(ROOT/p)
    return {p.relative_to(ROOT).as_posix():stat(p) for p in sorted(paths)}

def run():
    if (BASE/'closeout.json').exists():
        need(all(stat(ROOT/p)==v for p,v in read(LOG/'final_readonly.json')['files'].items()),'Closed result changed');print('Repair already closed; no new computation');return
    need(read(LOG/'screen_readonly.json')['passed'],'Independent200-frame screen acceptance missing');early=(BASE/'finite_stop.json').exists()
    if not early:need(read(BASE/'evaluation/readonly.json')['passed'],'Full readonly missing')
    before=snapshot();once(LOG/'cold_reentry_before.json',before);prior=counts(prior=True);cmd=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B','scripts/p5_repair_v23r1.py','run'];out=LOG/'cold_reentry.stdout.log';err=LOG/'cold_reentry.stderr.log'
    if not (LOG/'cold_reentry.json').exists():
        need(not out.exists() and not err.exists(),'Existing cold attempt must be inspected before recovery');start=time.monotonic()
        with out.open('x',encoding='utf-8') as so,err.open('x',encoding='utf-8') as se:p=subprocess.run(cmd,cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        once(LOG/'cold_attempt.json',dict(command=cmd,exit_code=p.returncode,seconds=time.monotonic()-start));need(p.returncode==0,'Cold entry failed; logs preserved');need(counts(prior=True)==prior,'Cold entry added calls');need(all(stat(ROOT/p)==v for p,v in before.items()),'Cold entry changed research files');once(LOG/'cold_reentry.json',dict(passed=True,actual_entry=True,new_forward=0,new_optimizer=0,new_official_AP=0,files_unchanged=len(before)))
    account=counts();cumulative=counts(prior=True);c=read(CONFIG);r=c['resources'];need(account['optimizer']['known_returns']==(400 if early else 800),'Research update count differs');need(all(account[k]['unknown']==0 for k in ['forward','optimizer','official_AP']),'Unknown call outcomes require explicit review')
    for kind,nkey,tkey in [('forward','max_forward_intents','cumulative_forward_intents'),('optimizer','max_optimizer_returns','cumulative_optimizer_returns'),('official_AP','max_official_ap_calls','cumulative_official_ap_calls')]:
        need(account[kind]['intents']<=r[nkey] and cumulative[kind]['intents']<=r[tkey],'Budget exceeded '+kind)
    need(usage()<r['wall_seconds'] and usage(prior=True)<r['cumulative_wall_seconds'] and output_bytes()<r['output_bytes'],'Time/storage budget exceeded')
    screen=read(BASE/'first_five_gate.json');decision=None if early else read(BASE/'evaluation/decision.json');adopted=False if early else decision['scientifically_adopt'];result=dict(status='P5_single_repair_complete_stopped',quality_usable_reused=True,gate_adopted=adopted,finite_screen_stopped=early,full160_AP_run=not early,screen_mean_degraded_F1_gain=screen['mean_degraded_F1_gain'],decision=decision,accounting_new=account,accounting_cumulative=cumulative,compute_seconds_new=usage(),compute_seconds_cumulative=usage(prior=True),output_bytes_cumulative=output_bytes(),progress=dict(all_tasks_percent=37,experiment_percent=46.25,earned_new_points=0,reason='Repair of already credited P5 deliverable'),old_results_unchanged=True,cold_reentry_passed=True,next_part_started=False,automatic_schedule_recreated=False,scope='Synthetic perturbations on exposed development data; not sensor calibration or independent generalization',next_decision='retain quality branch; cease gate tuning' if not adopted else 'development adoption only; wait for user before next part')
    once(BASE/'closeout.json',result)
    lines=['P5唯一一次门控修复结果与停止记录',now(),'','最大抑制强度0.5→0.25；质量头及固定臂复用，新门控从同一P4父权重开始。旧结果完整保留。']
    if early:
        lines+=['本次在预先冻结的第5轮继续价值关口停止。两臂clean有效性门槛：'+str(screen['gates'])+'。',f"固定20帧四退化平均宏F1差值={screen['mean_degraded_F1_gain']:.9f}（{100*screen['mean_degraded_F1_gain']:.6f}个百分点），未满足严格正增益条件。",'实际新增400次研究优化；200个缓存帧项已全部独立重解码和匹配验收。完整160帧/条件官方AP未运行，不声称已测得修复版全量AP或泛化结论。']
        for d,g in screen['by_condition_F1_gain'].items():lines.append(f'{d}：宏F1差值{100*g:+.6f}个百分点')
    else:
        lines += [f"完成800新增优化、28集合2520帧项与154官方AP块的直接计算或同权重旧控制引用及独立审计。共同选择第{read(BASE/'selection.json')['epoch']}轮。",f"四退化平均AP差值{100*decision['mean_degraded_AP_gain']:+.6f}个百分点，clean AP损失{100*decision['clean_AP_loss']:.6f}个百分点。",'采用门槛：'+str(decision['gates'])]
    lines += ['',('本次受限开发采用门槛通过；仍不能声称独立泛化或抓取优势。' if adopted else '本次修复不采用门控，按预设有限决策停止该方向调参。质量网络已有合成扰动验证继续保留；固定RGBD实例基线可作后续接口基础。'),'该诊断只决定是否值得继续，F1不能替代完整AP，早停也不能证明所有其他门控方案无效。' if early else '工程验收与方法收益分别记录。','真实冷重入已通过：新增前向/优化/AP均0。旧冻结证据及原主state.json保持不变。',f"新增真实调用账本：{account}；累计：{cumulative}。",f"记录计算新增{usage():.3f}秒，累计{usage(prior=True):.3f}秒（不含全部编写与等待）。",'','全部任务总进度37%，实验总进度46.25%，固定工作包估计。同一P5修复不重复增加进度。','已停止；未启动P6、第二次参数搜索或周期自动任务。后续须用户确认。','剩余主任务包括P2最终数据覆盖、P3完整抓取基线、P6真实几何传播、P7校准、P8全链路、通过P8后P9多种子、P10保留测试、P11统计及P12论文复现。']
    report=ROOT/'docs/P5单次门控修复结果_20261027.txt';need(not report.exists(),'Report already exists');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,LOG/'handoff_before_closeout.txt');old=handoff.read_text(encoding='utf-8-sig');handoff.write_text('最新：'+now()+' P5唯一一次0.25门控修复已完成并停止；阶段='+('第5轮有限继续筛查未通过，完整160/AP未跑' if early else '完整评价已收尾')+'；门控采用='+str(adopted)+'，质量分支保留。完整相应范围只读及实际冷重入通过，旧证据不变。全部任务37%，实验46.25%，修复不重复记分。报告docs/P5单次门控修复结果_20261027.txt。不得自行启动下一修复/P6/自动任务，等用户确认。\n\n历史：\n'+old,encoding='utf-8')
    ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_closeout.json');row=read(ctrl);row.update(recorded_at=now(),current_execution_state='P5_single_repair_complete_stopped',current_execution_evidence='results/P5/quality_gate_v23r1/closeout.json',current_runner=None,active_internal_repair=None,remaining_research_updates_cap=0,waiting_for_user_next_part_confirmation=True,summary_report=report.relative_to(ROOT).as_posix(),final_readonly='logs/P5/quality_gate_v23r1/final_readonly.json');atomic(ctrl,row);status('P5_single_repair_complete_stopped',gate_adopted=adopted,early_screen_stopped=early,waiting_for_user=True)
    files=snapshot();files[report.relative_to(ROOT).as_posix()]=stat(report);once(LOG/'final_readonly.json',dict(passed=True,files=files,actual_cold_reentry=True,appropriate_scope='200screen' if early else '200screen+2520full',next_part_started=False));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
