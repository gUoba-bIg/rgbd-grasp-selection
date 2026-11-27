"""Close the completed supervised interface round and leave all research stopped."""
import sys,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_baseline_grasp_link_v21 as trial
from scripts.pipeline import project_lock
with project_lock(ROOT):
    r=trial.read(trial.OUT+'/result.json');v=trial.read(trial.OUT+'/readonly.json');c=trial.read(trial.CONFIG)
    trial.need(v['status']=='passed' and r['completed_frames']==v['verified_frames']==4 and r['candidate_count']==v['verified_candidates']==719,'Required full trial/acceptance missing')
    trial.need(r['new_model_forward_calls']==r['new_training_updates']==r['new_official_scoring_calls']==0,'Unexpected work accounting')
    for p,b in c['protected'].items():trial.need(trial.identity(p)['sha256']==b['sha256'],'Protected evidence changed: '+p)
    before=trial.read(trial.LOG+'/progress_before_trial.json');current=json.loads(json.dumps(before))
    found=0
    for p in current['packages']:
        for s in p['subdeliverables']:
            if s['status']=='pending_v21':s['status']='done';found+=1
        p['earned']=sum(s['weight'] for s in p['subdeliverables'] if s['status']=='done')
        if p['id']=='P4':p['evidence'].extend([trial.OUT+'/result.json',trial.OUT+'/readonly.json'])
    earned=sum(p['earned'] for p in current['packages']);trial.need(found==1 and earned==26,'Unexpected fixed planning credit')
    current.update(updated_at=trial.now(),earned_points_current=26,all_percent_current=26,experiment_percent_current=32.5,change_reason='Predeclared R21 cached interface deliverable completed for all four views/719 candidates with independent full readonly acceptance. Added 1 of the existing 2 pending P4 linkage points. Representative formal linkage, full P4 and scientific benefit remain open.',weights_changed_after_outcome=False)
    trial.save(trial.LOG+'/progress_after_trial.json',current)
    closed=dict(recorded_at=trial.now(),status='stopped_awaiting_user_confirmation',round='R21',usable_limited_interface=r['usable_limited_interface'],full_P4_complete=False,full_method_usable_established=False,R18_dualstream_repaired=False,frames_completed=4,candidates_verified=719,associated=643,background_retained=76,new_model_forward_calls=0,new_training_updates=0,new_official_scoring_calls=0,all_tasks_percent_planning_estimate=26,experiment_percent_planning_estimate=32.5,automatic_resume_allowed=False,automation_recreated=False,next_experiment_started=False,requires_user_confirmation_before_retry_or_next=True,evidence={p:trial.identity(p)['sha256'] for p in [trial.CONFIG,trial.OUT+'/result.json',trial.OUT+'/readonly.json',trial.OUT+'/historical_support_diagnostic.json',trial.LOG+'/progress_before_trial.json',trial.LOG+'/progress_after_trial.json']})
    trial.save(trial.OUT+'/closeout.json',closed)
    trial.save('runtime/pipeline/supervised_v21_control.json',dict(closed,original_state_json_unchanged=True))
    lines=['第二十一轮R16基线抓取接口结果及停止记录',closed['recorded_at'],'',
      '结论：有限开发接口可用。四个固定训练视角的719个候选均可通过真实SA采样链追溯到原始深度像素，并与R16预测实例对接；没有改动原候选姿态、置信分数、排列顺序或object_id=-1字段。独立完整只读复验实际通过。',
      '范围：场景3的0016/0080/0160/0240四个相关训练视角。643个候选具有预测实例种子归属，76个属于预测背景，全部保留；无未追溯或歧义项。实例ID是逐帧种子像素标签，不是物体类别或整个夹爪接触归属。',
      '逐帧：0016为118候选/101归属/17背景；0080为187/166/21；0160为219/204/15；0240为195/172/23。',
      '先封存无真值部署关联结果，再连接已有官方几何支持项：200项已知，519项未知，未将未知当作负例。已有μ=0.4通过项共31个，其中29个具有实例归属，2个落入预测背景。该有限支持集表明直接删除背景候选会丢弃已有合格候选，因此维持全部候选，仅添加实例侧表。不是重新评分，也不是数据集AP、抓取成功率、泛化证据或新方法提升。',
      '验收：独立过程从原始深度重构四幅完整点云，检查全部真实SA1/SA2来源、候选种子原像素、逐候选整数标签映射与坐标投影；719个原17列候选逐字节保留，139项旧文件与绑定文件在运行和复验中保持不变。',
      f"资源：实际新增模型前向0、训练0、官方评分0、随机采样0。关联运行记录{r['resources']['seconds']:.3f}秒，独立复验{v['resources']['seconds']:.3f}秒（不含准备和解释器启动），各自预算120秒。峰值RSS约{max(r['resources']['peak_rss_bytes'],v['resources']['peak_rss_bytes'])/1024**2:.1f}MiB，未使用GPU。",
      '流程状态：该单轮已结束，运行和复验退出码均为0，未启动下一轮。自动续接保持删除，等待用户审阅确认。任何失败后重试或下一实验都要重新获得用户确认。',
      '全部任务总进度26%；实验总进度32.5%（口头可约33%）。仍是固定计划工作包加权估算，不是精确工时测量。全部100点、实验80点分母不变；本轮开始前将原P4未完成的2点接入工作细分为缓存接口1点及代表性正式链路验收1点，实际完成前者，25点增至26点。没有按719条记录算总体百分比，也未按新结果改变权重。',
      '主计划位置：P4实例基线的复用、深度接口、缓存抓取候选接口已形成有限可用链路；代表性范围的正式抓取链路验收及最终稳定基线闭环仍待完成。P5双流质量门控仍未解决，R18负结果原样保留；P6合成传播结果不等于真实实例不确定性链路闭环；P7至P12的正式训练、消融、多种子、独立测试、统计及论文交付仍有待办。',
      '下一步建议（未启动）：完成代表性范围的正式抓取链路验收，事前固定数据范围和预算；继续保留背景候选，不因本轮归属结果改变采样或排序。若仍追求R18双流质量门控，另需新协议的小范围有效性训练验证，不能把本轮接口通过称为R18修复。',
      '依据：results/P4/baseline_grasp_link_v21/{result,readonly,closeout,historical_support_diagnostic}.json；logs/P4/baseline_grasp_link_v21/{progress_before_trial,progress_after_trial}.json；configs/baseline_grasp_link_v21.json。','']
    trial.text('docs/第二十一轮基线抓取接口结果与停止记录.txt','\n'.join(lines))
    handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,ROOT/trial.LOG/'handoff_before_closeout.txt')
    head='当前状态：'+closed['recorded_at']+' R21单轮已完成并停止等待用户确认。4/4视角、719/719候选完整关联及独立只读验收通过，有限开发接口可用；643归属、76背景全部保留，背景中有2个既有μ=0.4合格支持项，不能硬删。新增前向/训练/评分/随机采样均0。全部任务总进度26%，实验总进度32.5%（固定工作包估算）。完整P4、R18双流和完整方法尚未通过。\n用户仅授权单轮继续，自动任务仍删除；下一轮或重试必须用户再次明确确认。原state.json未修改；独立控制记录runtime/pipeline/supervised_v21_control.json。结果见docs/第二十一轮基线抓取接口结果与停止记录.txt。\n\n以下仅为历史状态，不构成继续授权：\n'
    handoff.write_text(head+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(closed,ensure_ascii=False))
