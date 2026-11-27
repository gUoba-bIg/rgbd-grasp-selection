"""Write truthful single-trial closeout; never starts or queues another experiment."""
import sys,json,shutil
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_lock
from scripts import run_baseline_usability_v20b as run
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def save(p,v):run.save(ROOT/p,v)
with project_lock(ROOT):
    now=datetime.now().astimezone().isoformat()
    out='results/P4/baseline_usability_v20b';r=read(out+'/result.json');v=read(out+'/readonly.json');c=read('configs/baseline_usability_v20b.json')
    assert r['completed_frames']==v['verified_frames']==30 and v['status']=='passed'
    assert r['new_model_forward_calls']==30 and r['unknown_forward_calls']==0 and v['new_model_forward_calls']==0
    assert all(run.state(ROOT/p)['sha256']==x['sha256'] for p,x in c['protected_files'].items())
    baseline=read('logs/progress_audit/supervised_v1_20261027/weights_before_trial.json')
    freeze=read('logs/P4/baseline_usability_v20b/freeze.json')
    assert run.state(ROOT/'logs/progress_audit/supervised_v1_20261027/weights_before_trial.json')['sha256']==freeze['planning_weights_sha256']
    current=json.loads(json.dumps(baseline));current['updated_at']=now
    for p in current['packages']:
        for s in p['subdeliverables']:
            if s['status']=='pending_current_trial':s['status']='done'
        p['earned']=sum(s['weight'] for s in p['subdeliverables'] if s['status']=='done')
        if p['id']=='P4':p['evidence'].extend([out+'/result.json',out+'/readonly.json'])
    earned=sum(p['earned'] for p in current['packages']);assert earned==25
    current.update(earned_points_current=earned,all_percent_current=25,experiment_percent_current=31.25,change_reason='The two predefined P4 investigation deliverables completed: bound R16 reuse decision and 30-frame instance/depth interface trial with full zero-forward readonly verification. No main scientific gate was marked complete.',weights_changed_after_outcome=False)
    save('logs/progress_audit/supervised_v1_20261027/after_trial.json',current)
    dev=r['summary']['development'];train=r['summary']['train'];maxpixel=max(x['assessment']['geometry']['projection_error_pixels'] for x in r['frames']);maxmetre=max(x['assessment']['geometry']['inverse_formula_error_metres'] for x in r['frames'])
    receipt=dict(recorded_at=now,status='stopped_awaiting_user_confirmation',usable_for_limited_baseline_reuse=r['usable_for_limited_baseline_reuse'],full_P4_completed=False,full_method_validated=False,R18_dualstream_repaired=False,automation_recreated=False,next_experiment_started=False,requires_user_confirmation_before_retry_or_next=True,all_tasks_percent_planning_estimate=25,experiment_percent_planning_estimate=31.25,evidence={p:run.state(ROOT/p)['sha256'] for p in [out+'/result.json',out+'/readonly.json','configs/baseline_usability_v20b.json','logs/progress_audit/supervised_v1_20261027/after_trial.json']})
    save(out+'/closeout.json',receipt)
    save('runtime/pipeline/supervised_v20b_control.json',dict(receipt,automatic_resume_allowed=False,research_state_file_changed=False,scope='Independent supervised trial record; original state.json unchanged'))
    content='\n'.join([
      '第二十轮B单轮验证：已完成并停止，等待用户确认',now,
      '本轮结论：第十六轮RGBD实例基线可在限定的已用训练/开发范围内复用，实例到米制三维坐标接口通过预声明检查。没有修复或美化第十八轮的双流零实例结果，也没有证明完整方法、泛化或抓取性能达标。',
      '执行：30/30帧完成（训练20、已暴露开发10，每场景0000一帧）；实际模型前向30次，未知0；新增训练0，新增官方AP0。逐帧预测和前景图均精确复现原R16冻结缓存。',
      '验收：30帧全部有至少一个IoU>=0.5实例匹配及有效深度，30帧完整只读复验通过且新增前向0；255项原文件与本轮绑定文件在运行及复验中保持不变。',
      f"本次子集指标：训练场景宏F1={train['macro_f1']:.6f}；开发场景宏F1={dev['macro_f1']:.6f}，TP={dev['tp']}、FP={dev['fp']}、FN={dev['fn']}。每场景只有一帧，不能与原160帧开发宏F1作提升比较，也不能称独立测试。",
      f'共6816个确定性有效深度点，未发现无有效深度的预测实例；最大重投影误差{maxpixel:.3e}像素，独立反投影公式差{maxmetre:.3e}米。仅验证接口算术与坐标单位，不证明真实相机标定或不确定性校准。',
      f"模型验证记录用时{r['resources']['elapsed_seconds']:.3f}秒，只读复验{v['elapsed_seconds']:.3f}秒，不含准备和解释器启动；预算分别600/180秒。峰值RSS {r['resources']['peak_rss_bytes']/1024**3:.3f}GiB，GPU预留峰值{r['resources']['peak_gpu_reserved_bytes']/1024**2:.1f}MiB。",
      '流程修复：恢复既有有效实例基线，先用固定小范围检查能否继续使用；执行只有单轮预算及唯一结果目录，已存在证据会拒绝重跑；科学失败与技术失败都停止，下一步需要用户确认。没有恢复自动任务或旧1280单元诊断。',
      '全部任务总进度：25%；实验总进度：31.25%（对外约31%）。固定计划工作包加权估算，非精确工时、测量完成率或置信区间。分母在本轮前固定为全部100点、实验80点，基线为23点，本轮完成预声明的两个P4交付物增加2点。旧30%/40%缺少固定分母，已撤回为历史粗估，不能直接比较涨跌。',
      '主方案位置：P0/P1完成；P2/P3仍有主计划范围闭环；P4已有小规模有效RGB/RGBD实例基线，本轮解决复用与深度接口问题，完整P4仍未签收。P5有已完成负结果但门控机制未有效；P6仅完成受限合成验证；P7有局部风险特征诊断。P8全链路、P9三种子、P10独立测试鲁棒性、P11正式统计图表、P12论文复现仍待完成。',
      '后续建议（仅计划，当前不运行）：下一轮沿主计划完成有效实例基线到既定抓取候选链路的接入与验收，先冻结有限范围及预算。R18若仍要保留双流门控，必须另设新协议验证实例学习有效性，再决定重训，不能放宽旧阈值或覆盖旧负结果。',
      '停止状态：运行和只读进程均已退出0；项目锁可重新获取；自动续接未恢复；等待用户审阅并确认。不得把旧心跳或历史授权视为下一轮许可。',
      '依据：results/P4/baseline_usability_v20b/result.json、readonly.json、closeout.json；configs/baseline_usability_v20b.json；logs/progress_audit/supervised_v1_20261027/weights_before_trial.json、after_trial.json；docs/总进度固定工作包_监督版20261027.txt。',''])
    report=ROOT/'docs/第二十轮B基线复用验证结果与停止记录.txt'
    with report.open('x',encoding='utf-8') as f:f.write(content)
    handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,ROOT/'logs/P4/baseline_usability_v20b/handoff_before_final_stop.txt')
    head='当前状态：'+now+' 已按用户单轮监督要求完成R20B并自动停止，等待用户确认。30/30帧及完整只读复验通过，R16基线仅限开发复用可用，R18双流仍不可用，完整P4/P6/整套方法未完成。新增前向30、未知0、训练0。全部任务总进度25%，实验总进度31.25%（约31%），按本轮前固定工作包权重估算；详见docs/第二十轮B基线复用验证结果与停止记录.txt。\n自动续接仍删除，不得继续下一轮、重试、冻结新实验或恢复旧诊断；必须先获得用户明确确认。原state.json未修改，独立停止状态见runtime/pipeline/supervised_v20b_control.json。\n\n以下仅为旧状态历史，不构成继续授权：\n'
    handoff.write_text(head+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))
