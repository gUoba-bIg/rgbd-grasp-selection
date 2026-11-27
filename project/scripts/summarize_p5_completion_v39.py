"""Root read-only audit, full-range decision and fixed delivery credit; no research."""
import sys,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p5_completion_delivery_v39 import LOG,OUT,STATE,read,stat,atomic,need,now,verify,validate_completion,ledger
def main():
    from scripts.pipeline import project_lock
    with project_lock(ROOT):
        c=verify();d,calls=validate_completion(c);close=read(OUT/'closeout.json')
        need(read(STATE)['phase']=='completed_stopped' and close['decision']==d,'Completion state inconsistent')
        need(close['engineering_passed'] and close['full160_evaluation_completed'],'Full delivery incomplete')
        for path,value in close['reviews'].items():need(stat(ROOT/path)==value,'Accepted receipt changed')
        metrics={p.stem:read(p) for p in (OUT/'full/metrics').glob('*.json')}
        failed=[k for k,v in d['scientific_gates'].items() if not v]
        adopted=d['scientific_usable'];rgb=d['RGB_parent_macro_AP']
        duration=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))+c['preflight_compute_seconds']
        need(duration<3600,'v39 control-chain budget exceeded')
        old=read(ROOT/'logs/P5/fusion_v38/root_final_review.json')
        combined={k:dict(intents=old['actual_calls'][k]['intents']+v['intents'],
                        known_returns=old['actual_calls'][k]['returned']+v['known_returns'],
                        unknown_ids=old['actual_calls'][k]['unknown_ids']+v['unknown_ids']) for k,v in calls.items()}
        lines=['P5完整开发评价及原方案取舍 v39','用户客户端日期2026-11-01；实际主机记录时间：'+now(),'',
            '全部任务总进度45%；实验总进度56.25%。固定交付权重100/80；不是耗时比例、科学成功率或发表概率。',
            '本次P5完整范围已交付并停止：固定两臂同一final800权重，160开发帧、五条件、空间消融、官方mask AP、完整独立只读、真实新进程冷重入和根审计。0新增训练。',
            '工程验收通过；完整科学采用支持='+str(adopted)+'；未通过科学条件='+(', '.join(failed) if failed else '无')+'。',
            '旧v38的20帧关口失败和当时full未执行保持不变。v39是看到该结果后新授权的补评，不是独立盲测确认，不反向改变旧决策。',
            '本次不启动新教师训练、P6、P10或周期任务，等待用户检查整部分。','',
            '一、完整160帧结果（AP 0至1；差值为百分点；每场景16帧聚合，10场景等权）',
            '条件\tRGB父AP\tfixed AP\tquality AP\tquality−fixed(pp)\tquality−RGB(pp)']
        qrgb=[]
        for condition in ('clean','noise5','missing15','shift3','joint'):
            f=metrics['core_fixed_'+condition]['macro_AP'];q=metrics['core_quality_'+condition]['macro_AP']
            lines.append(f'{condition}\t{rgb:.8f}\t{f:.8f}\t{q:.8f}\t{100*(q-f):+.6f}\t{100*(q-rgb):+.6f}')
            if condition!='clean':qrgb.append(q-rgb)
        lines += ['',f'四退化quality−fixed平均：{d["mean_degraded_mask_AP_gain"]*100:+.6f}pp；原要求>=+1pp。',
            f'四退化quality−RGB平均：{sum(qrgb)/4*100:+.6f}pp；RGB不使用深度，深度单独扰动不改变它的输入，直接复用父RGB。',
            f'改善场景：{sum(v>0 for v in d["by_scene_gain"].values())}/10；原要求>=6。',
            'clean相对RGB损失(pp)：'+json.dumps({k:v*100 for k,v in d['clean_both_vs_RGB_loss'].items()},ensure_ascii=False),
            'joint质量相对unit/mean消融增益(pp)：'+json.dumps({k:v*100 for k,v in d['joint_ablation_gain'].items()},ensure_ascii=False),
            '末三监测配对F1跨度：'+str(d['last_three_span'])+'，使用既有监测缓存，未重复前向。',
            '完整判定：'+json.dumps(d['scientific_gates'],ensure_ascii=False),
            '仅在当前暴露开发集上的固定seed和训练预算内有效；不把百分点评估差值直接称统计显著，不声称机器人成功、完整方法多种子或保留测试优势。',
            'unit消融同时改变整体深度贡献强度，不能把unit差异全部当空间质量收益；mean是更接近空间分布作用的控制。','',
            '二、原方案可行性与根判断',
            '保留原研究主线：质量感知RGB-D实例分割→有定义且可校验的不确定性传播→实例相关局部几何风险→6DoF候选选择。原计划本来允许固定6DoF候选上的风险排序，不需要为了题目新造端到端位姿生成器。',
            '文献和已实现模块支持数学/工程可行性，但不能保证本项目的两项核心创新成立；分割门控增益与几何不确定性抓取收益必须分别得到证据。',
            'P6需先明确像素定位/配准残差的物理对应与单位；只有固定射线深度残差时不能宣称完整uvz。边界表面切换可能是离散错配，不能靠全局扩方差同时修复覆盖和锐度。合成参考不是传感器真值。',
            '实例信息应在完整场景点云上作软目标关联；背景/其他物体仍参与障碍检查。实例熵、坐标方差、抓取失败概率分别建模，不能互换。',
            '后续风险阶段保持固定候选，比较简单质量惩罚与有传播的局部几何风险；校准官方离线事件不等于真实机器人成功概率。',
            '依据与细节：docs/原方案可行性独立审查_v39.txt，包含ESANet、EMSANet、vMF-Contact原始论文链接和适用边界。','',
            '三、是否可用及对应建议']
        if adopted:
            lines += ['当前质量融合可作为下一步受控开发的候选输入，尚不代表整篇方法已成功。建议用户检查后再处理P6误差定义、可辨识性与覆盖/锐度，而非跳到昂贵P10。']
        else:
            lines += ['当前实现可用于复现、对照和负结果分析；未满足原方向正式采用门槛，不能直接写作已证实的方法优势，也不应据此启动整链大算量。']
            if not d['scientific_gates'].get('clean_both_vs_RGB',False):
                lines += ['完整范围仍有clean保留问题：若继续一个有界修复，优先保留RGB编码器，允许融合后的轻量refine/head适配，以GT实例损失为主，并加入训练clean上弱教师保持。仅作为待用户检查的候选，不在本次执行；必须匹配两臂、唯一系数/预算，不扫参数或追加长训。']
            else:
                lines += ['完整范围clean保持已过，因此不把“20帧clean失败”当作必须增加RGB教师的充分理由。应根据剩余失败条件聚焦质量表征/空间门控，而非为了修复小样本门槛机械重训。']
            if not d['scientific_gates'].get('spatial_quality',False):
                lines += ['空间质量消融未通过：当前质量代理没有证明优于保留平均强度的简单门控；优先检查质量监督与实例边界/配准退化的对应，不能只调小整体深度权重或删掉shift条件后宣称成功。']
            if not d['scientific_gates'].get('degraded_AP_gain',False):
                lines += ['四退化平均增益未达到原1pp目标：应保留所有条件和matched fixed/RGB对照，决定是否存在有监督依据的最小结构改动；缺少依据时停止重复训练，而非保证下一轮能变阳性。']
            lines += ['若最终采用教师/头适配，须增加学生融合q=0与冻结RGB父对照，防止仅退回RGB被误当深度/质量有效；解冻头后q=0也不再等于原教师。具体方案在本次完整结果基础上另行固定后才能运行。']
        lines += ['', '四、真实资源与剩余交付',
            'v39实际新增调用：'+json.dumps(calls,ensure_ascii=False),
            'v38正式+v39累计调用：'+json.dumps(combined,ensure_ascii=False),
            'v38另有5准备模型入口，全部有返回；未并入正式模型入口数。入口含fusion wrapper和内部quality，不能当作相同数的独立帧或底层精确算子次数。',
            f'v39控制链加准备计算实际累计墙时：{duration:.6f}秒；v38+v39同口径计算累计：{old["recorded_compute_seconds"]+duration:.6f}秒。实现与文献/代码人工审查另计，不冒充GPU计算时；更早版本时间未在此合并。',
            '旧P7未知2990905cbfa24838bb5e7b932827f9d7、旧P10未知91176b8f3d544cf7b98ff793e364d497仍为未知。',
            '进度新增仅P5F-B2点+C1点；v38已记A1点不重复。P5调查交付完成不等于科学方法通过。',
            '按原固定范围余55点：实验35点（P6目标误差2、P7恢复输入9、P8原核心全链路8、P9完整方法多种子8、P10独立确认8），统计图表8、论文复现12。若用户批准新的修复，它属于既有主项返工，不机械叠加总分。',
            '证据：configs/p5_completion_v39.json、results/P5/completion_v39/full/decision.json、logs/P5/completion_v39/review.json、cold.json、root_final_review.json。']
        report=ROOT/'docs/P5完整评价与原方案取舍结果_v39.txt';need(not report.exists(),'Preserve existing report')
        report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        audit=dict(at=now(),passed=True,engineering_passed=True,scientific_adoption_supported=adopted,
            full160_evaluation_completed=True,new_calls=calls,combined_v38_v39_calls=combined,
            controller_seconds=duration,combined_v38_v39_seconds=old['recorded_compute_seconds']+duration,
            protected_files=len(c['protected']),reviewed_outputs=len(read(LOG/'review.json')['reviewed_outputs']),
            original_state_unchanged=True,old_screen_unchanged_false=True,report=stat(report),
            all_tasks_percent=45,experiment_percent=56.25,P6_started=False,P10_still_stopped=True,stopped=True)
        atomic(LOG/'root_final_review.json',audit)
        atomic(LOG/'progress_after_part.json',dict(baseline_points=42,new_credit=3,completed=['P5F-B','P5F-C'],
            no_duplicate_credit_for='P5F-A',all_tasks_percent=45,experiment_percent=56.25,
            basis='Frozen v38 fixed-delivery subitems; negative investigation completion distinct from scientific adoption',
            scientific_adoption_supported=adopted,evidence='logs/P5/completion_v39/root_final_review.json'))
        handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,LOG/'handoff_before_root_close.txt')
        handoff.write_text(f'最新：{now()} P5 v39同权重完整160帧五条件/消融/AP、完整独立复验、真实冷重入和根审计已完成并停止，0新增训练。科学采用={adopted}。报告docs/P5完整评价与原方案取舍结果_v39.txt。全部任务总进度45%、实验总进度56.25%（固定交付权重；科学通过另判）。旧v38失败原样保留。用户检查前不启动修复训练、P6、P10或周期任务。\n\n历史：\n'+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
        metadata=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(metadata,LOG/'supervision_before_root_close.json');v=read(metadata)
        v.update(recorded_at=now(),current_execution_state='P5_v39_completed_stopped',current_runner=None,
            waiting_for_user_next_part_confirmation=True,automatic_resume_authorized=False,remaining_research_updates_cap=0,
            summary_report=report.relative_to(ROOT).as_posix(),final_readonly='logs/P5/completion_v39/root_final_review.json',
            current_progress_evidence='logs/P5/completion_v39/progress_after_part.json',all_tasks_percent_planning_estimate=45,
            experiment_percent_planning_estimate=56.25,unchanged_progress_reason='P5F-B2+C1 actual full completion; A not duplicated')
        atomic(metadata,v)
        st=read(STATE);st.update(all_tasks_percent=45,experiment_percent=56.25,root_final_review='logs/P5/completion_v39/root_final_review.json');atomic(STATE,st)
        print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
