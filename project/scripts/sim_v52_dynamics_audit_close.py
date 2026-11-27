"""Close the zero-physics audit and preserve its review boundary."""
import ast
import os
import shutil
from sim_v52_common import ROOT, read, write, identity, relative, lock, now
from sim_v52_dynamics_audit_guard import verify, STATE


def main():
    output = ROOT/'results/simulation_v52_dynamics_audit'
    logs = ROOT/'logs/simulation_v52_dynamics_audit'
    with lock(ROOT/'runtime/pipeline/execution.lock'), lock(ROOT/'runtime/pipeline/simulation_v52_dynamics_audit.lock'):
        assert not (output/'closeout.json').exists()
        peer = read(output/'final_peer_review.json')
        assert peer['review_passed'] and not peer['execution_authorized']
        assert not peer['blocking_issues'] and peer['new_physics_steps'] == 0
        for path, value in peer['bindings'].items():
            assert identity(ROOT/path) == value, path
        n = verify()
        plan = read(output/'proposed_diagnostic_plan.json')
        assert plan['execution_authorized'] is False
        assert plan['current_new_dynamics_runs'] == plan['current_new_physics_steps'] == 0
        assert len(plan['combinations']) == 12
        process_path = logs/'physical_workers_at_close.json'
        assert not read(process_path)['matching_physical_workers']
        static_scripts = [ROOT/'scripts'/name for name in (
            'sim_v52_mechanism_audit_static.py','sim_v52_observation_audit_static.py',
            'sim_v52_dynamics_solver_audit.py','sim_v52_dynamics_plan_static.py',
            'sim_v52_dynamics_audit_guard.py','sim_v52_dynamics_audit_close.py')]
        syntax_checks = {}
        for path in static_scripts:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            direct_physics_imports = []
            physics_step_calls = []
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    direct_physics_imports.extend(x.name for x in node.names if 'pybullet' in x.name)
                elif isinstance(node,ast.ImportFrom) and 'pybullet' in (node.module or ''):
                    direct_physics_imports.append(node.module)
                elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr == 'stepSimulation':
                    physics_step_calls.append(node.lineno)
            assert not direct_physics_imports and not physics_step_calls, path
            syntax_checks[relative(path)] = dict(identity=identity(path),
                direct_pybullet_imports=direct_physics_imports, stepSimulation_calls=physics_step_calls)
        state_backup = logs/'state_before_close.json'
        assert not state_backup.exists()
        shutil.copy2(STATE, state_backup)
        write(STATE, dict(at=now(), phase='completed_static_audit_waiting_research_review',
            report='docs/v52后续_夹爪动力学专项审计与有限诊断方案.txt',
            historical_artifacts_verified=n, historical_v52_grasp_attempts=48,
            new_physics_connections=0, new_physics_steps=0, new_dynamics_tests=0,
            mechanical_repair_implemented=False, original_engineering_admission_passed=False,
            proposed_diagnostic_units=12, diagnostic_plan_execution_authorized=False,
            R4_authorized=False, automatic_expansion=False, recurring_automation=False,
            all_tasks_percent_legacy=45, experiment_percent_legacy=56.25,
            expanded_scope_percent=None))
        handoff = ROOT/'docs/实验自动续接说明.txt'
        handoff_backup = logs/'handoff_before_static_audit_close.txt'
        assert not handoff_backup.exists()
        shutil.copy2(handoff,handoff_backup)
        prefix = ('最新：'+now()+' v52后续夹爪专项已完成零物理步机构/源码/旧48日志审计及有限诊断设计，并停止等待研究审核。新增连接/物理步/动力学测试均0，原3219受保护历史文件未变，原48失败和0.03rad门槛不改。原五gear秩5不冗余；官方140闭环几何已核，受载失配及接触前自碰须分别诊断，唯一根因未证明。报告docs/v52后续_夹爪动力学专项审计与有限诊断方案.txt；计划results/simulation_v52_dynamics_audit/proposed_diagnostic_plan.json仅提议12单元（8无物体、4固定球接触），execution_authorized=false，未实现/启动动力学runner。不得依据历史心跳启动新测试、重跑48或进入R4/72/720/240。状态runtime/pipeline/simulation_v52_dynamics_audit.json；旧simulation_v52.json保持原停止状态。全部任务总进度45%、实验总进度56.25%仅旧范围；新增范围仍待重新加权。待用户审核后才可执行另行登记的有限诊断。\n\n历史：\n')
        temporary = handoff.with_name(handoff.name+'.dynamics_static.tmp')
        with temporary.open('xb') as stream:
            stream.write(prefix.encode('utf-8'))
            stream.write(handoff.read_bytes())
        os.replace(temporary,handoff)
        artifacts = [ROOT/'docs/v52后续_夹爪动力学专项审计与有限诊断方案.txt',
            ROOT/'docs/v52专项机构结构与闭环审计.txt',
            ROOT/'docs/v52专项动力学源码与控制审计.txt',
            ROOT/'docs/v52专项既有48回合受载证据审计.txt',
            STATE, handoff, handoff_backup, state_backup, process_path] + static_scripts
        artifacts += [p for p in output.iterdir() if p.is_file()]
        write(output/'closeout.json',dict(at=now(), stage='zero_physics_audit_complete_stopped',
            historical_artifacts_unchanged=n, old_grasp_attempts=48,
            new_physics_connections=0,new_physics_steps=0,new_dynamics_tests=0,
            old_simulation_state_unchanged=True, old_results_not_reclassified=True,
            independent_document_review_passed=True,
            engineering_readmission_ready=False, R4_started=False,
            diagnostic_execution_authorized=False, proposed_units_only=12,
            clarification='L is a defined opposite pad-center-line calibration load. During pad orientation drift it need not align with either local surface normal; never interpret it as measured normal contact force.',
            static_script_syntax_checks=syntax_checks,
            artifacts={relative(p):identity(p) for p in artifacts}),once=True)
        print(f'Zero-physics audit closed; {n} historical files unchanged; 12-unit plan remains unexecuted.')


if __name__=='__main__':
    main()
