"""Read-only final audit and truthful P5 report; never launches research."""
import sys,json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p5_fusion_delivery_v38 import LOG,OUT,STATE,read,sha,stat,atomic,need,now,verify,ledger

def calls(path):
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    result={}
    for family in ('forward','optimizer','official_AP'):
        aa=[r['call_id'] for r in rows if r['kind']==family+'_intent']
        bb=[r['call_id'] for r in rows if r['kind']==family+'_returned']
        need(len(aa)==len(set(aa)) and len(bb)==len(set(bb)) and set(bb)<=set(aa),'Invalid actual call ledger')
        result[family]=dict(intents=len(aa),returned=len(bb),unknown_ids=sorted(set(aa)-set(bb)))
    return result

def main():
    from scripts.pipeline import project_lock
    with project_lock(ROOT):
        c=verify();proof=read(OUT/'closeout.json');review=read(LOG/'review.json');cold=read(LOG/'cold.json')
        need(read(STATE)['phase']=='completed_stopped','Controller not finished')
        need(proof['engineering_passed'] and review['passed'] and cold['passed'],'Review chain incomplete')
        for path,identity in proof['reviews'].items():need(stat(ROOT/path)==identity,'Review identity changed')
        for path,identity in review['reviewed_outputs'].items():need(stat(ROOT/path)==identity,'Reviewed output changed')
        full=(OUT/'full/decision.json').exists()
        scopes=['screen','full'] if full else ['screen'];need(review['completed_scopes']==scopes==cold['completed_scopes'],'Scope chain differs')
        decision=read(OUT/('full' if full else 'screen')/'decision.json')
        need(decision==proof['full' if full else 'screen'],'Final decision changed')
        need(decision['scientific_adoption_supported']==proof['scientific_adoption_supported']==cold['scientific_usable'],'Scientific chain differs')
        observed=calls(LOG/'calls.jsonl');prep=calls(LOG/'preflight_calls.jsonl')
        for value in list(observed.values())+list(prep.values()):need(not value['unknown_ids'],'New unknown research call requires explicit investigation')
        need(observed['optimizer']['returned']==1600,'Incomplete training')
        need(observed['forward']['returned']==(17920 if full else 14000),'Actual model-entry count differs from expected completed scopes')
        need(observed['official_AP']['returned']==(290 if full else 150),'Official AP count differs from completed scopes')
        need(prep['forward']['returned']==5 and prep['optimizer']['intents']==prep['official_AP']['intents']==0,'Preparation accounting differs')
        need(not ledger()[2],'Unresolved controller child')
        points=45 if full else 42;percent=points/80*100
        metrics={p.stem:read(p) for p in (OUT/('full' if full else 'screen')/'metrics').glob('*.json')}
        adopted=decision['scientific_adoption_supported']
        failed=[k for k,v in decision['scientific_gates' if full else 'validity_gates'].items() if not v]
        usage=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))+c['preflight_compute_seconds']
        lines=['P5 原方向模态融合整体修复 v38：结果与停止记录',now(),'',
            '全部任务总进度：'+str(points)+'%；实验总进度：'+str(percent)+'%。固定100/80交付权重，不是工时比例、科学成功率或发表概率。',
            '当前交付：'+('两臂各800更新、完整160帧五条件及空间消融、独立只读复验和真实新进程冷重入均已完成。' if full else '两臂各800更新及20帧有效性检查已完成并复验；有效性准入失败，未运行完整160帧评价。'),
            ('工程验收通过；完整开发科学采用支持='+str(adopted)+'。失败科学门槛='+', '.join(failed)+'。' if full else '工程验收通过；20帧有效性准入失败，尚未进行完整160帧科学判定。失败有效性门槛='+', '.join(failed)+'。'),
            '本部分已停止。未启动P6/P10/P11，也没有恢复周期任务。旧科学负结果和未知调用保留。',
            '', '结构与训练：冻结已验收P4 RGB主路及质量头，独立D/valid支路在FPN融合处接入；末投影零初始化。固定融合q=1与质量融合同结构同起点，新增支路LR1e-3、各800更新、同样本/退化顺序，最终800步唯一取值。',
            '当前是单种子、已暴露开发集结论；不能作为独立泛化、完整方法多种子或机器人抓取成功证据。',
            '', '条件指标（AP以0至1表示；差值是百分点；场景等权）：',
            '条件\t固定融合AP\t质量融合AP\t质量减固定(pp)\tRGB父模型AP']
        rgb=decision['RGB_parent_macro_AP']
        for condition in ('clean','noise5','missing15','shift3','joint'):
            f=metrics['core_fixed_'+condition]['macro_AP'];q=metrics['core_quality_'+condition]['macro_AP']
            lines.append(f'{condition}\t{f:.8f}\t{q:.8f}\t{(q-f)*100:+.6f}\t{rgb:.8f}')
        lines += ['', 'RGB父模型不消费深度；在仅深度变化的条件下其输入和结果相同，表中不重复运行RGB前向。',
            ('以下为完整160帧科学门槛判定。' if full else '以下仅描述20帧结果及最终科学门槛，不用于本次有效性准入；完整160帧科学门槛尚未判定。'),
            '旧P4 RGBD clean AP仅描述：'+str(decision['historical_RGBD_clean']['AP'])+'，它不代替本次同结构固定融合对照。',
            '四退化平均AP增益：'+str(decision['mean_degraded_mask_AP_gain']*100)+'个百分点；要求至少+1个百分点。',
            '改善场景：'+str(sum(v>0 for v in decision['by_scene_gain'].values()))+'/10；要求至少6。',
            'joint空间质量减单位/均值消融：'+json.dumps({k:v*100 for k,v in decision['joint_ablation_gain'].items()},ensure_ascii=False)+'（百分点）。',
            '末三监测点配对宏F1跨度：'+str(decision['last_three_span'])+'；上限0.05。',
            '完整决策：'+json.dumps(decision,ensure_ascii=False),
            '', '实际资源与证据',
            '生产实际调用：'+json.dumps(observed,ensure_ascii=False),
            '准备实际调用：'+json.dumps(prep,ensure_ascii=False),
            '模型入口数包含融合包装模型与内部质量模型，不能写成相同数量的独立帧。',
            '控制器墙时加准备记录计算合计：'+str(usage)+'秒；实现及审查人工墙时不包含在该计算值。',
            '旧P7未知调用2990905cbfa24838bb5e7b932827f9d7、旧P10未知调用91176b8f3d544cf7b98ff793e364d497仍保持未知。',
            '进度依据：docs/P5整体恢复进度口径与剩余交付_v38.txt；'+('本次P5原剩余4点完成，正负结果均计完成工作，科学准入单列。' if full else '本次只记P5F-A 1点；完整评价及整部分取舍3点仍未完成。'),
            '后续需要：'+('用户检查后才决定P6恢复；不能用分割开发优势替代几何不确定性或抓取证据。' if adopted else '当前不将该质量融合实现用于后续原方向正式方法；须先结合失败指标制定有界修复或范围取舍，再由用户决定。'),
            '完整原方向仍需可辨识的(u,v,z)不确定性、恢复输入后的风险模型和校准、整体消融、完整方法多种子、结果盲独立确认、统计图表、论文与复现材料。',
            '冻结协议configs/p5_fusion_v38.json；完整复验logs/P5/fusion_v38/review.json；冷重入logs/P5/fusion_v38/cold.json。']
        report=ROOT/'docs/P5模态融合整体修复结果_v38.txt';need(not report.exists(),'Existing report must be preserved')
        report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        audit=dict(at=now(),passed=True,protected_files=len(c['protected']),reviewed_outputs=len(review['reviewed_outputs']),
            full160_evaluation_completed=full,scientific_adoption_supported=adopted,actual_calls=observed,preflight_calls=prep,
            actual_child_returns=len(ledger()[1]),unknown_new_children=ledger()[2],recorded_compute_seconds=usage,
            original_pipeline_state_unchanged=True,report=stat(report),all_tasks_percent=points,experiment_percent=percent,
            P6_started=False,P10_still_stopped=True,stopped=True)
        atomic(LOG/'root_final_review.json',audit)
        atomic(LOG/'progress_after_part.json',dict(at=now(),baseline_points=41,credit=points-41,
            completed=['P5F-A','P5F-B','P5F-C'] if full else ['P5F-A'],all_tasks_percent=points,experiment_percent=percent,
            basis='Predeclared delivery criteria; scientific decision separately recorded',scientific_adoption_supported=adopted))
        h=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(h,LOG/'handoff_before_root_close.txt')
        h.write_text(f'最新：{now()} P5 v38已完成本次有界交付并停止。完整160帧评价完成={full}，科学采用={adopted}；工程数值复验、实际冷重入及根最终不变性/调用审计通过。报告docs/P5模态融合整体修复结果_v38.txt。全部任务总进度{points}%，实验总进度{percent}%，固定恢复范围交付权重，正负结果与科学通过分开。等待用户检查下一部分；不启动P6/P10/P11或周期任务。\n\n历史：\n'+h.read_text(encoding='utf-8-sig'),encoding='utf-8')
        p=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(p,LOG/'supervision_before_root_close.json');v=read(p)
        v.update(recorded_at=now(),current_execution_state='P5_v38_completed_stopped',current_runner=None,
            waiting_for_user_next_part_confirmation=True,summary_report=report.relative_to(ROOT).as_posix(),
            final_readonly='logs/P5/fusion_v38/root_final_review.json',current_progress_evidence='logs/P5/fusion_v38/progress_after_part.json',
            all_tasks_percent_planning_estimate=points,experiment_percent_planning_estimate=percent,
            unchanged_progress_reason='Credit only for actual completed predeclared P5 deliverables',remaining_research_updates_cap=0)
        atomic(p,v)
        state=read(STATE);state.update(all_tasks_percent=points,experiment_percent=percent,root_final_review='logs/P5/fusion_v38/root_final_review.json')
        atomic(STATE,state);print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
