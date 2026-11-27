"""Publish existing v52e evidence only. Never imports or connects to PyBullet."""
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sim_v52_common import read, write, identity, relative, now, lock
from scripts.sim_v52e_control import history, verify, status


def main():
    assert Path.cwd().resolve() == ROOT
    log = ROOT / 'logs/simulation_v52e'
    out = ROOT / 'results/simulation_v52e'
    report = ROOT / 'docs/v52e_参数接口修复与零物理步验收报告.txt'
    handoff = ROOT / 'docs/实验自动续接说明.txt'
    state = ROOT / 'runtime/pipeline/simulation_v52e.json'
    with lock(ROOT / 'runtime/pipeline/execution.lock'), lock(ROOT / 'runtime/pipeline/simulation_v52e.lock'):
        if report.exists() or (out / 'delivery_closeout.json').exists():
            raise RuntimeError('Delivery already exists; no overwrite or rerun')
        import psutil
        active = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            command = ' '.join(proc.info.get('cmdline') or [])
            if proc.pid != os.getpid() and (proc.info.get('name') or '').lower() in ('python.exe', 'pythonw.exe'):
                if any(name in command for name in ('sim_v52e_zero_step.py', 'sim_v52d_episode.py', 'sim_v52e_build.py')):
                    active.append(proc.info)
        assert not active, active
        cfg = read(ROOT / 'configs/simulation_v52e.json')
        verify(cfg)
        protected = history()
        summary = read(out / 'execution_summary.json')
        result = read(ROOT / summary['result_path'])
        review = read(log / 'independent_readback_review_v2.json')
        patch = read(log / 'patch/patch_manifest.json')
        build = read(log / 'build_result.json')
        install = read(log / 'install_result.json')
        compiler = read(log / 'compiler_info.json')
        assert summary['parameter_admission_passed'] and summary['charged_starts'] == 1
        assert result['admission_passed'] and result['actual_physics_steps'] == 0 and not result['unknown_calls']
        assert review['passed'] and all(review['checks'].values())
        assert build['exit_code'] == install['exit_code'] == 0
        snapshots = {name: read(ROOT / row['path']) for name, row in result['snapshots'].items()}
        for row in result['snapshots'].values():
            assert identity(ROOT / row['path']) == row['identity']
        maps = {name: {x['link_index']: x for x in data['links']} for name, data in snapshots.items()}
        assert all(len(value) == 25 for value in maps.values())
        table = []
        for index, baseline in sorted(maps['baseline'].items()):
            entries = [maps[name][index] for name in ('baseline', 'afterL', 'afterR')]
            values = [[e['dynamics_raw'][j] for j in (1, 11, 5)] for e in entries]
            final = entries[-1]['dynamics_raw']
            if index in (16, 21):
                assert values[-1] == [1.0, 0.0005, 0.0]
                assert all(final[j] == baseline['dynamics_raw'][j] for j in range(12) if j not in (1, 5, 11))
            else:
                assert all(e['dynamics_raw'] == baseline['dynamics_raw'] for e in entries)
            table.append(dict(link_index=index, link_name=baseline['link_name'], baseline=values[0], afterL=values[1], afterR=values[2],
                              mass_kg=final[0], inertia_diagonal=final[2], non_target_unchanged=index not in (16, 21)))
        table_path = out / 'readback_table.json'
        write(table_path, dict(at=now(), columns=['lateral_friction', 'collision_margin_m', 'restitution'],
                              source_snapshots=result['snapshots'], rows=table), once=True)
        old_binary = ROOT / 'runtime/simulation_v52/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd'
        old_binary_identity = identity(old_binary)
        new_binary_identity = result['engine']['extension_identity']
        wheel_path, wheel_id = next(iter(build['wheels'].items()))
        lines = [
            'v52e 参数接口修复与零物理步验收报告',
            '生成时间：' + now(),
            '项目根目录：' + str(ROOT),
            '',
            '一、结论与停止点',
            '独立版本 PyBullet 3.2.7+v52e1 已完成 C API 与 Python 绑定修复，并通过当前 UR5e + Robotiq 2F-140 两个指垫的实际参数读回验收。',
            '唯一已登记会话：1 次 DIRECT 连接、1 次原机器人加载、2 次 changeDynamics 设置；全部 328 次引擎调用均有返回，0 未知，0 重试，0 物理步。',
            '左、右指垫最终实际读回均为：lateralFriction=1.0、collisionMargin=0.0005 m、restitution=0.0（原冻结配置）。base 与其余 link 未被误改。',
            '本结论仅限参数路由和零步读回。未证明受载耦合误差已经降至 0.03 rad，未证明抓取成功或机械模型正确。M1 未实施，新 12 次、新 48 回合、R4/72/720/240 均未启动。',
            '本轮交付后停止，等待用户审核；不得依照历史心跳、旧授权或本报告自行启动后续动力学。',
            '',
            '二、根因和独立修复',
            '原 3.2.7 Python changeDynamics 解析了 linkIndex，但 collisionMargin 分支调用旧 C setter 时未传入该值。旧 setter 把共享命令的 m_linkIndex 置为 -1；同一命令此前设置的摩擦/恢复系数标记仍然保留，最终作用于 base。仅拆分 Python 调用不能解决单独 margin 仍指向 base 的问题。',
            '修改 1：PhysicsClientC_API.h 新增导出函数 b3ChangeDynamicsInfoSetCollisionMargin2(handle, bodyUniqueId, linkIndex, collisionMargin)。',
            '修改 2：PhysicsClientC_API.cpp 新函数将传入的 linkIndex 写入命令，继续使用原 collision-margin 更新标记。旧 setter 的签名和函数体保持原样，保留原 C ABI；直接调用旧 C setter 的客户端仍保留原 base-only 行为，须显式采用新函数。',
            '修改 3：examples/pybullet/pybullet.c 将 margin 分支改为调用上述新函数，并传入 Python 已解析的目标 linkIndex。',
            '原服务端已有非负 linkIndex 的 margin 处理路径，本次未改服务端、共享命令布局、碰撞形状实现或动力学求解器。当前指垫为恒等局部变换的单 box 子形状；此验收不扩展到任意复杂 compound 形状或所有 body-global/link 参数混合调用。',
            'setup.py 仅增加本地版本号 3.2.7+v52e1 和编译并行数限制（PYBULLET_BUILD_JOBS=2），不改物理代码及编译选项。原 sdist 的 5936 个源文件中，仅上述 4 个源/构建元数据文件发生预期修改；build/egg-info 为编译生成物。',
            '源码差异：logs/simulation_v52e/patch/parameter_route.patch',
            '构建元数据差异：logs/simulation_v52e/patch/build_metadata.patch',
            '完整修改清单及原文件备份：logs/simulation_v52e/patch/patch_manifest.json 与 patch/original_from_sdist/',
            '',
            '三、环境、编译与资源记录',
            '旧环境：runtime/simulation_v52/venv，保留原 PyBullet 3.2.7，未覆盖。',
            '新环境：runtime/simulation_v52e/venv；独立源码：runtime/simulation_v52e/src/pybullet-3.2.7。',
            '实际加载版本：' + result['engine']['distribution_version'] + '；API 版本：' + str(result['engine']['api_version']),
            '实际解释器：' + result['engine']['python_version'],
            '注意：上行 MSC 标识属于 Python 自身构建；本轮扩展编译器信息如下。',
            '扩展编译器：' + compiler['path'],
            '编译器 FileVersion=' + compiler['file_version'] + '；ProductVersion=' + compiler['product_version'],
            '编译命令与环境完整记录：logs/simulation_v52e/build_intent.json；依赖安装记录：build_dependencies.log。',
            '构建/安装均返回 exit_code=0；wheel=' + wheel_path,
            f"编译墙钟 {build['wall_seconds']:.3f} s，采样进程树峰值 RSS {build['peak_sampled_tree_rss_bytes'] / 2**30:.3f} GiB。",
            f"零步验收受控子进程墙钟 {summary['resources']['wall_seconds']:.3f} s，采样进程树峰值 RSS {summary['resources']['peak_sampled_tree_rss_bytes'] / 2**20:.2f} MiB。",
            '上述 RSS 为采样值，非连续峰值测量；本轮无 GPU 训练、无神经网络前向、无动力学积分。',
            '编译日志含警告（包括 Py_DEBUG 配置未设置的 ABI-tag 警告），未隐藏或删去；完整日志 compile.log、安装日志 install.log 均保留。二进制实际导入、执行本次零步读回成功。',
            '',
            '四、零物理步方法与实际调用',
            '配置 configs/simulation_v52e.json 以及 53 个源码、机器人、原配置和构建证据身份在执行前冻结。尝试在进程启动前记账；入口发现已有账本即拒绝第二次运行。',
            '加载原 data/simulation_v52/robot/ur5e_robotiq_2f140.urdf，fixedBase；沿用原惯量、自碰撞及父子排除加载标志。不 reset 机器人、不设重力或时间步、不创建电机/gear/物体、不执行实时或离散物理步。',
            '先读取 base(-1) 和全部 24 个 link 的 dynamics、碰撞形状、位姿/速度及关节状态，形成 baseline。',
            '左指垫调用：changeDynamics(bodyUniqueId=0, linkIndex=16, lateralFriction=1.0, restitution=0.0, collisionMargin=0.0005, physicsClientId=0)。',
            '立即重读全部 base/link，验证 afterL；通过后才执行同样参数的右指垫 linkIndex=21 调用，再重读形成 afterR。每个指垫仍用同一次调用同时设摩擦、恢复系数、margin，不以拆分调用规避问题。',
            '摩擦、恢复系数和 margin 必须实际读回与目标一致；任一不一致将停止，禁止下一参数修改或物理运动。此次两次均一致。调用参数与返回逐条保存在 events.jsonl。',
            '非目标 dynamics/geometry 使用完整 tuple 一致检查。目标 dynamics 仅允许摩擦、恢复系数、margin 三字段变化，质量、惯量等保持不变；目标外形数值仅采用 1e-12 绝对 / 1e-10 相对舍入容差。该容差仅用于未变外形的浮点比较，不涉及 0.03 rad 耦合验收门槛。',
            '原 10 Nm 主动驱动上限、80 Nm 约束上限、0.03 rad 门槛由原配置身份校验保持；本轮没有实际驱动/约束受载测量，不能将配置未变写成力矩实测通过。',
            '',
            '五、全部 base/link 实际读回表',
            '每个三元组依次为：摩擦系数 / collisionMargin(m) / restitution。',
            'baseline 是本次新加载后、修改前的引擎读数；不是把历史 v52d 的错误 base 参数当成新的基准。',
            'linkIndex | link名称 | baseline | afterL | afterR',
        ]
        for row in table:
            triple = lambda values: '/'.join(format(x, '.9g') for x in values)
            lines.append(f"{row['link_index']} | {row['link_name']} | {triple(row['baseline'])} | {triple(row['afterL'])} | {triple(row['afterR'])}")
        lines += [
            '',
            '逐次修改时，24 个非目标 base/link 的完整 dynamics 记录均不变。总体最终只有两指垫的摩擦与 margin 数值改变；两指垫 restitution 原为 0，设置后仍为 0。base 保持 0.5 / 0.001 / 0，其他 22 个非指垫 link 保持各自原值。所有 link 的质量、惯量、其余动力学字段及加载状态未变。',
            '机器可读表：results/simulation_v52e/readback_table.json；全部原始 12 项 dynamics 与碰撞形状、位姿状态见 baseline.json、afterL.json、afterR.json。',
            '',
            '六、修改前后 SHA256 与构建身份',
            '以下源码 before 来自保留的原 3.2.7 sdist，after 来自独立本地源码；不是修改旧安装环境。',
        ]
        for row in patch['modified_files']:
            lines += [row['path'], '  before: ' + row['before']['sha256'], '  after : ' + row['after']['sha256']]
        lines += [
            '旧安装 pyd（交付前复核未变）：' + old_binary_identity['sha256'],
            '新独立 pyd：' + new_binary_identity['sha256'],
            '新 wheel：' + wheel_id['sha256'],
            '原源码归档：' + patch['source_archive']['sha256'],
            '冻结配置：' + result['config_identity']['sha256'],
            '实际结果：' + summary['resources']['result']['sha256'],
        ]
        for name, row in result['snapshots'].items():
            lines.append(name + '：' + row['identity']['sha256'])
        lines += [
            f'原 PyBullet 环境及历史实验的 {protected} 个受保护文件已按 SHA256、字节数、mtime_ns 复核无变化。旧 P5/P6 科学未通过、旧 48 回合未通过和 v52d 参数审查失败均保持原结论。',
            '',
            '七、独立复核、失败证据与修订记录',
            '源码复核：independent_source_review_v2.json；执行控制复核：independent_control_review.json；最终读回复核：independent_readback_review_v2.json（28 项通过）。',
            '最终复核只读取 JSON、源码、ZIP 和哈希，不启动新引擎连接。核对 328 个调用均有对应返回、三份快照与原始返回逐值一致、新 pyd 与已构建 wheel 内成员一致。',
            '准备阶段原源码审查把编译生成的 build/egg-info 当成额外源文件，原失败审查保留；v2 区分原 sdist 源文件与生成物，确认只有 4 个预期源/元数据修改。',
            '控制脚本冻结前修正了未知结果不能写成已知零步的状态记录，旧脚本保存在 preparation/control_before_unknown_status_fix.py；正式调用未出现未知。',
            '最终读回复核初稿用 LF 字面匹配 wheel METADATA 的版本行，因 CRLF 格式误判；原 independent_readback_review.json 保留。v2 使用标准 header parser 读出 3.2.7+v52e1，未改任何实验结果、快照或二进制。',
            '以上均为准备/只读审查修订，不是新增或重试物理实验。',
            '',
            '八、后续边界与待审核事项',
            '已满足本轮“参数路由及零物理步读回”目标。下一步是否开展任何有限动力学诊断，必须由用户另行审核和授权；此报告不自动启动。',
            '正确指垫参数与历史错误路由下的实际参数不同，未来诊断必须独立登记版本与预算，不能将新参数结果追认成旧 M0 同条件复验，也不能反向覆盖旧失败。',
            'M1 机械闭环改造继续暂停。新 12 次诊断、新 48 回合工程准入、R4 算法比较、720 正式和 240 消融均未启动。',
            '交付前未发现活动的 v52e/v52d 物理工作进程；项目执行锁和本版本锁均正常取得，交付退出时释放。',
            '全部任务总进度：旧主交付范围 45%；实验总进度：旧主交付范围 56.25%。新增机器人物理仿真范围尚未建立固定权重，两个扩展后总百分比仍待重算；本次接口修复不机械上调旧比例。',
            '当前阶段单列：v52e 参数修复完成，1 次零步验收通过，0 物理步；已停止等待审核。',
            '',
            '九、证据入口',
            '独立状态：runtime/pipeline/simulation_v52e.json',
            '单次账本：logs/simulation_v52e/acceptance_attempts.jsonl',
            '执行汇总：results/simulation_v52e/execution_summary.json',
            '完整验收目录：' + str(Path(summary['result_path']).parent).replace('\\', '/'),
            '编译/安装：logs/simulation_v52e/build_result.json、install_result.json、compile.log、install.log',
            '交付清单：results/simulation_v52e/delivery_closeout.json',
        ]
        with report.open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
        delivery = log / 'delivery'
        delivery.mkdir(exist_ok=False)
        shutil.copy2(state, delivery / 'state_before_delivery.json')
        shutil.copy2(handoff, delivery / 'resume_before_delivery.txt')
        old_handoff = handoff.read_bytes()
        prefix = (f'最新：{now()} v52e独立参数接口修复及零物理步验收已交付并停止待审核。'
                  '本地PyBullet3.2.7+v52e1采用link-aware C setter和Python绑定，原3.2.7环境保留。'
                  '唯一1连接/1加载/2参数设置，328调用均返回、0未知、0重试、0物理步；'
                  '左右指垫实际friction=1.0、margin=0.0005m、restitution=0，base及其他link未改。'
                  '14494历史文件保持，原10Nm/80Nm/.03rad配置与旧48/v52d/P5/P6失败结论不改。'
                  '只证明参数路由，不证明受载耦合或抓取通过；M1暂停，禁止启动新12、48、R4/72/720/240，等待用户新审核。'
                  '报告docs/v52e_参数接口修复与零物理步验收报告.txt；状态runtime/pipeline/simulation_v52e.json。'
                  '全部任务总进度旧范围45%，实验总进度旧范围56.25%，新增范围待重算。\n\n历史：\n')
        temporary = handoff.with_name(handoff.name + '.v52e.tmp')
        with temporary.open('xb') as stream:
            stream.write(prefix.encode('utf-8') + old_handoff)
        os.replace(temporary, handoff)
        status('delivered_zero_step_passed_stopped_for_review', step_status='known_zero', acceptance_starts=1,
               parameter_admission_passed=True, known_result=True, result_path=summary['result_path'],
               report_path=relative(report), review_required=True, protected_files_unchanged=protected)
        artifacts = {report, handoff, state, table_path, Path(__file__), ROOT / 'configs/simulation_v52e.json',
                     ROOT / wheel_path, ROOT / result['engine']['extension_path'], out / 'execution_summary.json'}
        for directory in (log, ROOT / Path(summary['result_path']).parent):
            artifacts.update(p for p in directory.rglob('*') if p.is_file())
        artifacts.update(ROOT / 'runtime/simulation_v52e/src/pybullet-3.2.7' / row['path'] for row in patch['modified_files'])
        write(out / 'delivery_closeout.json', dict(at=now(), passed=True, scope='parameter routing and zero physics step readback only',
              charged_starts=1, engine_connections=1, actual_physics_steps=0, unknown_calls=0,
              historical_files_verified_unchanged=protected, review_required=True, active_workers_before_delivery=active,
              M1_implemented=False, new12_started=False, new48_started=False, R4_started=False,
              report=relative(report), artifacts={relative(p): identity(p) for p in sorted(artifacts)}), once=True)
        print(json.dumps(dict(delivered=True, report=relative(report), preserved=protected, physics_steps=0, stopped_for_review=True)))


if __name__ == '__main__':
    main()
