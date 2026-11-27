"""Publish a stopped-stage receipt; no simulation and no historical state edits."""
import os
import shutil
from sim_v52_common import ROOT, OUT, LOG, STATE, now, read, write, identity, relative, lock, verify_history


def main():
    destination = OUT/'stage_closeout.json'
    if destination.exists():
        raise RuntimeError('Closeout already exists; do not overwrite')
    with lock(ROOT/'runtime/pipeline/execution.lock'), lock(ROOT/'runtime/pipeline/simulation_v52.lock'):
        state = read(STATE)
        assert state['phase'] == 'completed_stopped_for_review'
        assert state['engineering_attempts_started'] == 48 and not state['engineering_passed']
        assert state['R4_authorized'] is False and state['automatic_expansion'] is False
        process_path = LOG/'processes_after_completion.json'
        process_check = read(process_path)
        assert not process_check['matching_experiment_python_processes']
        review = read(LOG/'final_review.json')
        assert review['evidence_review_passed'] and not review['engineering_admission_passed']
        changed = [p for p, v in review['artifact_identities'].items() if identity(ROOT/p) != v]
        assert not changed, changed
        historical_count = verify_history()
        handoff = ROOT/'docs/实验自动续接说明.txt'
        old_handoff = identity(handoff)
        backup = LOG/'handoff_history/experiment_resume_before_v52_closeout.txt'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise RuntimeError('Handoff backup already exists')
        shutil.copy2(handoff, backup)
        assert identity(backup) == old_handoff
        prefix = f'''最新：{now()} v52本阶段已完成记录并停止等待研究审核，禁止自动续跑。R1模块/公式审计和Raw缓存对比完成；R2 PyBullet+UR5e+2F140环境/静态接口完成；R3工程48启动/48返回/0新unknown，正向合格0/24，错位与桌碰负例各12/12正确。23回合受载夹爪耦合失败，engineering_passed=false，不能进入72回合R4；720/240也未启动。2450个受保护历史文件及冻结结果身份保持不变；旧P5/P6科学失败与历史2个官方未知不改。已完成独立只读复核，证据一致性通过不代表工程准入通过。报告docs/P5-P6模块调整与机器人仿真工程验证结果_v52.txt；故障依据与下一步未执行修复建议docs/机器人耦合工程缺口与后续修复建议_v52.txt；正式状态runtime/pipeline/simulation_v52.json保持completed_stopped_for_review。全部任务总进度45%、实验总进度56.25%仅旧主交付范围；新增仿真范围总百分比待重新加权，当前单列R1/R2完成、R3 48/48已记录但未过。无新训练/神经前向，不改P3旧候选，不创建周期任务。后续须用户研究审核后明确新机械修复预算；不能以旧心跳或既往授权越过停止点。

历史：
'''
        temporary = handoff.with_name(handoff.name + '.v52closeout.tmp')
        with temporary.open('xb') as stream:
            stream.write(prefix.encode('utf-8'))
            stream.write(handoff.read_bytes())
        os.replace(temporary, handoff)
        final_paths = [
            ROOT/'scripts/sim_v52_delivery.py', ROOT/'scripts/sim_v52_closeout.py',
            ROOT/'docs/P5-P6模块调整与机器人仿真工程验证结果_v52.txt',
            ROOT/'docs/机器人耦合工程缺口与后续修复建议_v52.txt',
            OUT/'delivery_summary.json', OUT/'delivery_case_index.json',
            OUT/'admission/report_048.json', LOG/'final_review.json',
            LOG/'episodes.jsonl', process_path, STATE, handoff, backup]
        write(destination, dict(at=now(), stage='R1_R2_R3_stopped_for_research_review',
            attempts_started=48, attempts_returned=48, new_unknown_attempts=0,
            historical_official_unknown_calls=2, historical_files_unchanged=historical_count,
            independent_evidence_review_passed=True, engineering_admission_passed=False,
            R4_started=False, automatic_expansion=False, needs_research_review=True,
            legacy_main_state_unchanged=identity(ROOT/'runtime/pipeline/state.json'),
            original_handoff_identity=old_handoff,
            no_matching_running_experiment_at_snapshot=True, both_project_locks_acquired=True,
            artifacts={relative(p): identity(p) for p in final_paths},
            publisher_physics_steps=0, publisher_neural_forwards=0), once=True)
        print('Stage closeout written; experiment remains stopped; R4 not started.')


if __name__ == '__main__':
    main()
