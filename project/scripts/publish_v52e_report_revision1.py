"""Preserve v52e delivery; correct two report phrases without engine work."""
import os
import sys
import shutil
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sim_v52_common import read, write, identity, relative, now, lock
from scripts.sim_v52e_control import status


def main():
    assert Path.cwd().resolve() == ROOT
    with lock(ROOT / 'runtime/pipeline/execution.lock'), lock(ROOT / 'runtime/pipeline/simulation_v52e.lock'):
        old_report = ROOT / 'docs/v52e_参数接口修复与零物理步验收报告.txt'
        report = ROOT / 'docs/v52e_参数接口修复与零物理步验收报告_校订1.txt'
        old_closeout = ROOT / 'results/simulation_v52e/delivery_closeout.json'
        closeout = ROOT / 'results/simulation_v52e/delivery_closeout_revision1.json'
        state = ROOT / 'runtime/pipeline/simulation_v52e.json'
        handoff = ROOT / 'docs/实验自动续接说明.txt'
        prior = read(old_closeout)
        assert not report.exists() and not closeout.exists()
        for path, expected in prior['artifacts'].items():
            assert identity(ROOT / path) == expected, path
        text = old_report.read_text(encoding='utf-8')
        replacements = {
            '不创建电机/gear/物体': '不发送电机控制命令，不额外创建 gear 约束或被抓取物体（loadURDF 内部创建的默认关节电机仍存在）',
            '准备阶段原源码审查把编译生成的 build/egg-info 当成额外源文件': '准备阶段原源码审查把编译生成的 build 等生成物当成额外源文件',
        }
        for before, after in replacements.items():
            assert text.count(before) == 1
            text = text.replace(before, after)
        text = text.replace('交付清单：results/simulation_v52e/delivery_closeout.json',
                            '交付清单：results/simulation_v52e/delivery_closeout_revision1.json（保留原 delivery_closeout.json）')
        text += ('\n校订1说明：仅澄清 loadURDF 内部默认关节电机与用户控制命令的区别，以及首次源码审查的生成物措辞。'
                 '原报告与原交付清单完整保留；参数、验收结果、物理步数和停止状态均不变，未新增任何引擎连接。\n')
        with report.open('x', encoding='utf-8') as stream:
            stream.write(text)
        folder = ROOT / 'logs/simulation_v52e/delivery_revision1'
        folder.mkdir(exist_ok=False)
        shutil.copy2(state, folder / 'state_before_revision.json')
        shutil.copy2(handoff, folder / 'resume_before_revision.txt')
        write(folder / 'revision_basis.json', dict(at=now(), reason='Independent delivery wording review',
              replacements=replacements, prior_report=identity(old_report), prior_closeout=identity(old_closeout),
              new_engine_connections=0, new_physics_steps=0, results_modified=False), once=True)
        prefix = (f'最新：{now()} v52e已完成参数接口修复与零物理步验收并停止待审核。'
                  '正式报告docs/v52e_参数接口修复与零物理步验收报告_校订1.txt；仅校订默认关节电机和生成物措辞，原报告保留，无新增实验。'
                  '1连接、2参数设置、0物理步、0未知；两指垫1.0/0.0005m/0，base及其余link未改。'
                  '只证明参数路由，不证明动力学耦合已解决。M1暂停；不得启动新12、48、R4/72/720/240，等待用户审核。'
                  '全部任务总进度旧范围45%，实验总进度旧范围56.25%，新增范围待重算。\n\n历史：\n')
        old_bytes = handoff.read_bytes()
        temporary = handoff.with_name(handoff.name + '.v52e_revision1.tmp')
        with temporary.open('xb') as stream:
            stream.write(prefix.encode('utf-8') + old_bytes)
        os.replace(temporary, handoff)
        status('delivered_zero_step_passed_stopped_for_review', step_status='known_zero', acceptance_starts=1,
               parameter_admission_passed=True, known_result=True,
               result_path=read(ROOT / 'results/simulation_v52e/execution_summary.json')['result_path'],
               report_path=relative(report), report_revision=1, review_required=True, protected_files_unchanged=14494)
        artifacts = dict(prior['artifacts'])
        for path in [report, state, handoff, old_closeout, Path(__file__), *folder.iterdir()]:
            artifacts[relative(path)] = identity(path)
        write(closeout, dict(at=now(), passed=True, report=relative(report), previous_delivery=identity(old_closeout),
              report_only_revision=True, new_engine_connections=0, actual_physics_steps=0, charged_starts=1,
              review_required=True, artifacts=artifacts), once=True)
        print('Report wording revision published; prior report/results preserved; no engine calls.')


if __name__ == '__main__':
    main()
