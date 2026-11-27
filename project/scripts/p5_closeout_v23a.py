"""Finite P5 reporting and actual cold reentry; stops after this part."""
import sys,time,subprocess,uuid,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p5_common_v23 import *

def call_identity(row):
    if 'call_id' in row:
        value=row['call_id'];need('id' not in row or row['id']==value,'Conflicting ID fields')
    elif row.get('phase')=='engineering_tests' and row['kind'] in ('forward_intent','forward_returned'):
        value=row.get('id')
    else:raise RuntimeError('Missing recorded call identity; cannot infer an outcome')
    need(isinstance(value,str) and len(value)>0,'Invalid call identity');return value

def ledger_counts():
    e=events();out={}
    for k,prefix in [('forward','forward'),('optimizer','optimizer'),('official_AP','official_AP')]:
        intents=[call_identity(x) for x in e if x['kind']==prefix+'_intent'];returns=[call_identity(x) for x in e if x['kind']==prefix+'_returned'];need(len(intents)==len(set(intents)) and len(returns)==len(set(returns)),'Duplicate '+k+' IDs');need(set(returns).issubset(intents),'Return without intent');out[k]=dict(intents=len(intents),known_returns=len(returns),unknown=len(set(intents)-set(returns)))
    out['optimizer_returns_by_phase']={p:sum(x['kind']=='optimizer_returned' and x.get('phase')==p for x in e) for p in sorted({x.get('phase','unspecified') for x in e if x['kind']=='optimizer_returned'})}
    out['forward_intents_by_phase']={p:sum(x['kind']=='forward_intent' and x.get('phase')==p for x in e) for p in sorted({x.get('phase','unspecified') for x in e if x['kind']=='forward_intent'})}
    return out

def snapshot():
    c=verify();paths={ROOT/p for p in c['protected']};paths.update([Path(__file__),ROOT/'scripts/p5_closeout_v23.py',LOG/'closeout_v23a_freeze.json']);paths.update(p for d in [BASE,RUN] for p in d.rglob('*') if p.is_file());paths.update([CONFIG,ROOT/'configs/p5_evaluation_v23.json',LOG/'segmentation_freeze.json',LOG/'quality_review.json',LOG/'evaluation_freeze.json'])
    for f in ['segmentation_freeze.json']:
        for p,v in read(LOG/f)['protected'].items():need(stat(ROOT/p)==v,'Source changed '+p);paths.add(ROOT/p)
    return {p.relative_to(ROOT).as_posix():stat(p) for p in sorted(paths)}

def freeze():
    once(LOG/'closeout_v23a_freeze.json',dict(source=stat(Path(__file__)),plot_source=stat(ROOT/'scripts/p5_figures_v23.py'),protocol_sha256=sha(CONFIG),stop_after_P5=True))

def run():
    need(stat(Path(__file__))==read(LOG/'closeout_v23a_freeze.json')['source'],'Closeout source changed')
    need(stat(ROOT/'scripts/p5_figures_v23.py')==read(LOG/'closeout_v23a_freeze.json')['plot_source'],'Plot source changed')
    need(read(BASE/'evaluation/readonly.json')['passed'],'P5 complete readonly acceptance missing')
    if (BASE/'closeout.json').exists():
        final=read(LOG/'final_readonly.json');need(all(stat(ROOT/p)==v for p,v in final['files'].items()),'Closed result changed');print('P5 already closed; no additional computation');return
    before=snapshot();once(LOG/'cold_reentry_v23a_before.json',before);prior=ledger_counts();results=[]
    for script in ['scripts/p5_run_v23.py','scripts/p5_segment_v23.py']:
        mode='quality' if 'p5_run_' in script else 'run';tag='quality' if mode=='quality' else 'segmentation';out=LOG/f'cold_{tag}.stdout.log';err=LOG/f'cold_{tag}.stderr.log';need(not out.exists() and not err.exists(),'Cold entry already attempted; inspect saved attempt before recovery')
        start=time.monotonic()
        with out.open('x',encoding='utf-8') as stdout,err.open('x',encoding='utf-8') as stderr:
            p=subprocess.run([str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-B',script,mode],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        row=dict(script=script,mode=mode,exit_code=p.returncode,wall_seconds=time.monotonic()-start);results.append(row);once(LOG/f'cold_{tag}.json',row);need(p.returncode==0,'Cold reentry failed; preserve its logs')
    need(ledger_counts()==prior,'Cold reentry added model/optimizer/AP calls');need(all(stat(ROOT/p)==v for p,v in before.items()),'Cold reentry changed original artifacts');once(LOG/'cold_reentry.json',dict(passed=True,commands=results,new_forward=0,new_optimizer=0,new_official_AP=0,source_files_unchanged=len(before)))
    dec=read(BASE/'evaluation/decision.json');quality=read(BASE/'quality_screen.json');account=ledger_counts();resources=read(CONFIG)['resources'];need(account['forward']['intents']<=resources['max_forward_intents'] and account['optimizer']['known_returns']<=resources['max_optimizer_returns'] and account['official_AP']['intents']<=resources['max_official_ap_calls'],'Budget exceeded')
    need(account['optimizer_returns_by_phase'].get('quality_training')==240 and account['optimizer_returns_by_phase'].get('segmentation_training')==1600,'Research budget/coverage differs');need(account['official_AP']['known_returns']==154,'Official AP scope incomplete')
    # Unknowns are preserved, never converted to exact successful counts.
    need(all(account[k]['unknown']==0 for k in ['forward','optimizer','official_AP']),'Unresolved execution outcomes require explicit review')
    seconds=sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'));need(seconds<resources['wall_seconds'],'Cumulative compute budget exceeded')
    progress=dict(before_all=29,before_experiment=36.25,earned_points=8,completed_work_packages=[dict(points=4,name='有效实例基线下质量预测实现',evidence='100 fixed development cases; raw-depth independent reconstruction and quality gates'),dict(points=4,name='门控有效性对照和取舍',evidence='paired 800 updates/arm;2520 exports;154 official AP blocks; spatial ablations; independent readonly; real cold reentry')],after_all=37,after_experiment=46.25,denominators=dict(all=100,experiment=80),work_completion_not_scientific_superiority=True)
    once(LOG/'progress_after_P5.json',progress)
    result=dict(status='P5_development_part_complete_stopped_for_user',scope='Exposed-development synthetic depth-quality prediction and instance segmentation gating; not real sensor calibration, not grasp efficacy, not P8',quality_usable=quality['quality_usable'],gate_adopted=dec['scientifically_adopt'],decision=dec,accounting=account,recorded_compute_seconds=seconds,progress=progress,automatic_scheduler_recreated=False,next_part_started=False,cold_reentry='logs/P5/quality_gate_v23/cold_reentry.json',remaining_main_plan=['P2 final data coverage','P3 full grasp baseline closure','P6 real geometric propagation and fusion','P7 calibration','P8 full main comparison scientific gate','P9 only after P8 passes','P10/P11/P12 final statistics, manuscript and reproduction'])
    once(BASE/'closeout.json',result)
    details=[]
    for name in ['core_fixed_learned_clean','core_quality_learned_clean','core_fixed_learned_joint','core_quality_learned_joint']:
        m=read(BASE/'evaluation/sets'/(name+'.json'));details.append(f"{name}: 宏F1={m['scene_macro_f1']:.6f}，宏mask AP={m['scene_macro_AP']:.6f}")
    conclusion='满足本次开发采用条件；仍需后续P8真实抓取与泛化关口。' if dec['scientifically_adopt'] else '质量预测可用于已知合成扰动开发；质量门控未满足全部采用条件，保留有效负结果，不推荐将它作为已验证优势进入后续效果主张。'
    repair='后续修复候选：先依据已保存clean/joint、单位门控和空间平均门控对照定位收益损失，再在新协议中只修改门控位置或强度之一，预先固定有限预算与失败停止条件；不复用本次开发集作为独立确认，不改本次冻结结果。不自动追加训练。' if not dec['scientifically_adopt'] else '当前无须为追求更高指标重复本次实验；下一主部分待用户检查后明确授权。'
    text='P5整体开发结果与停止记录\n'+now()+'\n\n'+conclusion+'\n\n'+'\n'.join(details)+f"\n四退化平均mask AP差值={dec['mean_degraded_AP_gain']:.6f}；clean AP损失={dec['clean_AP_loss']:.6f}。\n质量门槛：{quality['gates']}\n门控采用门槛：{dec['gates']}\n\n质量240更新、两臂各800更新；2520帧项完整导出，14个全量条件/消融集、14个20帧分级子集。官方AP154块均已独立核对主precision/recall/scores数组；完整只读及真实冷重入通过，冷重入新增模型调用/优化/AP均0。\n\n全部任务总进度37%，实验总进度46.25%，固定工作包加权估算；本次两个4点工作包完成，科学负结果同样计入已完成工作，不能解释为方法成功率。\n\n"+repair+'\n\nP5已停止，未启动P6或自动任务。原主流水线state.json未手工改为complete。P3完整抓取基线、后续P6/P7/P8、条件性P9以及统计图表、论文与复现仍待完成。\n'
    report=ROOT/'docs/P5整体结果总览_20261027.txt';need(not report.exists(),'Report already exists');report.write_text(text,encoding='utf-8')
    handoff=ROOT/'docs/实验自动续接说明.txt';old=handoff.read_text(encoding='utf-8-sig');shutil.copy2(handoff,LOG/'handoff_before_P5_closeout.txt');handoff.write_text('最新：'+now()+' P5开发部分已完成并停止等待用户检查下一主部分；全部任务37%、实验46.25%（固定工作包估算）。质量预测通过，门控采用='+str(dec['scientifically_adopt'])+'；科学结论、适用范围见docs/P5整体结果总览_20261027.txt。完整只读、实际冷重入已通过；原state.json未修改；不启动P6、不恢复自动任务。\n\n历史：\n'+old,encoding='utf-8')
    ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_P5_closeout.json');s=read(ctrl);s.update(recorded_at=now(),current_execution_state='P5_development_part_complete_stopped',current_execution_evidence='results/P5/quality_gate_v23/closeout.json',current_progress_evidence='logs/P5/quality_gate_v23/progress_after_P5.json',all_tasks_percent_planning_estimate=37,experiment_percent_planning_estimate=46.25,waiting_for_user_next_part_confirmation=True,current_runner=None,remaining_research_updates_cap=0,summary_report='docs/P5整体结果总览_20261027.txt',unchanged_progress_reason='P5两个预先固定4点工作包实际交付，新增8点；科学采用与工作完成分别记录。');ctrl.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    statepath=ROOT/'runtime/pipeline/p5_part_v23.json';statepath.write_text(json.dumps(dict(at=now(),phase='P5_complete_stopped',all_tasks_percent=37,experiment_percent=46.25,progress_after=progress,gate_adopted=dec['scientifically_adopt'],waiting_for_next_part_confirmation=True),ensure_ascii=False,indent=2),encoding='utf-8')
    files=snapshot();files[report.relative_to(ROOT).as_posix()]=stat(report);once(LOG/'final_readonly.json',dict(passed=True,files=files,actual_cold_reentry=True,no_new_experiment_started=True));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    if sys.argv[1:] == ['freeze']:freeze()
    else:run()
