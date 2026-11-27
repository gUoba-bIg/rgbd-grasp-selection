"""Read-only experiment summarization; creates new delivery artifacts only."""
import collections
import datetime as dt
import json
from pathlib import Path
from sim_v52_common import ROOT, LOG, OUT, read, write, identity, relative, now


def main():
    report_path = OUT / 'admission/report_048.json'
    review_path = LOG / 'final_review.json'
    report = read(report_path)
    review = read(review_path)
    assert review['evidence_review_passed'] and not review['engineering_admission_passed']
    assert report['complete'] and report['attempts_started'] == 48
    assert not report['engineering_passed'] and not report['R4_started']
    rows = report['rows']
    results = [row['result'] for row in rows]
    audit_path = OUT / 'audit/summary.json'
    audit = read(audit_path)
    events = [json.loads(line) for line in (LOG/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line]
    first = min(dt.datetime.fromisoformat(e['at']) for e in events if e['event'] == 'intent')
    last = max(dt.datetime.fromisoformat(e['at']) for e in events if e['event'] == 'returned')
    kinds = ('nominal', 'miss', 'table_collision')
    by_kind = {}
    for kind in kinds:
        part = [r for r in results if r['slot']['kind'] == kind]
        by_kind[kind] = dict(attempts=len(part),
            protocol_successes=sum(r['success'] for r in part),
            collision_count=sum(r['forbidden_collision'] for r in part),
            slip_count=sum(r['slipped'] for r in part),
            infeasible_count=sum(r.get('infeasible', False) for r in part))
    diagnostic = dict(
        grasp_formed=sum(r['grasp_formed'] for r in results),
        ever_lifted_10cm=sum(r.get('max_lift_m', 0) >= .1 for r in results),
        ever_held_2s=sum(r.get('max_continuous_hold_seconds', 0) >= 2 for r in results),
        final_held_2s=sum(r.get('final_continuous_hold_seconds', 0) >= 2 for r in results),
        max_mimic_error_rad=max(r.get('max_mimic_error_rad', 0) for r in results),
        physics_steps=sum(r['physics_steps'] for r in results),
        episode_window_wall_seconds=(last-first).total_seconds(),
        no_attachment_constraints=all(r['new_object_attachment_constraints'] == 0 for r in results),
        no_object_teleport_after_initialization=all(r['object_teleports_after_initialization'] == 0 for r in results),
        interpretation='Diagnostics only. These counts are not valid grasp success or algorithm evidence.')
    vision = []
    vision_sources = {}
    for arm in ('core_fixed', 'core_quality'):
        for condition in ('clean', 'noise5', 'missing15', 'shift3', 'joint'):
            path = ROOT/f'results/P5/completion_v39/full/metrics/{arm}_{condition}.json'
            d = read(path)
            values = list(d['scenes'].values())
            vision.append(dict(arm=arm, condition=condition, AP=d['macro_AP'],
                AP50=sum(x['AP50'] for x in values)/len(values), scene_count=len(values)))
            vision_sources[relative(path)] = identity(path)
    case_index = []
    for row in rows:
        directory = ROOT/row['directory']
        r = row['result']
        files = {p.name: dict(path=relative(p), identity=identity(p))
            for p in directory.iterdir() if p.is_file()}
        case_index.append(dict(slot_id=row['slot_id'], directory=row['directory'],
            object_id=r['slot']['object_id'], kind=r['slot']['kind'],
            success=r['success'], outcome=r['outcome'],
            max_lift_m=r.get('max_lift_m'),
            final_continuous_hold_seconds=r.get('final_continuous_hold_seconds'),
            max_mimic_error_rad=r.get('max_mimic_error_rad'),
            forbidden_collision=r['forbidden_collision'], slipped=r['slipped'], files=files))
    write(OUT/'delivery_case_index.json', dict(at=now(), scope='All 48 cases, without filtering', cases=case_index), once=True)
    write(OUT/'delivery_summary.json', dict(at=now(), experiment_report=relative(report_path),
        independent_review=relative(review_path), by_kind=by_kind, diagnostics_only=diagnostic,
        original_vision_cached_summary=vision, original_vision_sources=vision_sources,
        no_new_vision_inference=True, scientific_comparison_ready=False,
        R4_started=False, requires_research_review=True,
        all_tasks_percent_legacy=45, experiment_percent_legacy=56.25,
        expanded_scope_percent=None), once=True)

    text = f'''P5-P6模块调整与机器人仿真工程验证阶段交付 v52
生成时间：{now()}
项目目录：{ROOT}

一、明确结论
已完成R1审计、R2环境与静态接口配置、全部48回合R3工程记录及独立只读复核，实验进程已结束。本批工程准入未通过，不能进入R4的72回合算法比较。720回合正式比较及240回合消融未启动。
当前平台能产生关节驱动、接触和物体抬升，但受载夹爪内指耦合不满足冻结门槛，因此尚不能作为可信的算法仿真比较平台。不能以本批不合格机器人结果判定C0算法有效或无效。
冻结判定：正向0/24合格；错位负对照12/12正确；桌碰负对照12/12正确。证据一致性审查通过，不等于工程验收通过。
没有新训练，没有新神经网络前向，没有用物体固定约束、吸附或初始化后的物体位置重置制造成功。旧P5/P6科学失败、v51r2结论及历史2次官方未知调用保持原样。

二、P5/P6的研究作用与资产复用
P5保留RGB、普通RGB-D、质量融合模型和权重、训练工具、原评价。后续主视觉前端使用普通RGB-D；暂停当前质量融合模块的新开发，不宣称其独立创新已验证。质量融合原收益门槛仍未通过。
P6保留共享对应关系、几何/深度误差表示、传播代码、训练工具、权重与日志。暂停当前未正式校准的学习式概率头；不将其输出当作已有效风险概率。后续候选方法先以固定几何误差模型C0为审查对象。
R1确认预测实例关联确实进入接触支持和风险排序，不是只出现在可视化图中。但现有A/Raw差异同时包含实例和确定性几何项，C0/A差异同时包含warp和深度分布；尚不能分离各机制的独立贡献。
需要后续封存场景和去实例关联、warp-only、depth-only对照后才能提出独立贡献或交互结论。本轮未启动这些新实验。
注意：本R3使用从仿真观测深度形成的工程候选验证机械链路，未运行P5预测或P3网络，也未执行Raw/A/C0机器人物理对比。不能把R3写成完整视觉算法已经落地成功。

三、C0/D0公式、参数及公平性审计
统一评分：score = raw_score × contact_support × (1 - collision_proxy)。
contact_support = Σ_g p(g) × association(g) × min(left(g), right(g))。
collision_proxy = clip(expected_occupied_ratio / 0.01, 0, 1)。
C0使用普通RGB-D实例预测与固定几何误差分布；D0使用质量融合实例预测与同一固定分布。D0的存在不改变P5原科学失败。
固定warp先验来自v50八个训练场景的864个人工warp案例，81个共享整数warp，Dirichlet alpha=1/81。固定深度模型是自由度3的Student-t、尺度5mm；5mm不是标准差，也不是已校准真实传感器参数。
观察域0<z<2m，障碍体素1cm，原代理指长6cm/厚1cm/接近5cm，每侧最多32个固定几何锚点。参数来自冻结v51及更早v43/v45/v50，不在本R1中按结果调参。
Raw与A/C0使用同一原20候选池、观察过滤、NMS和官方选择规则；Raw只按原始生成分数排序。先从完整候选singleton物理标签逐项复现既有350结果，再计算50个Raw案例。新增官方eval_grasp/get_grasp_score调用均0，属于完整缓存支持下的协议重算。
动作输入来自观测深度、预测实例及固定训练先验，未发现将本例GT物体归属/成功标签作为排序输入。环境真值仅用于场景构建和结果评价。
数据暴露边界：十场景虽不与网络训练场景重叠，但已用于既往模型选择/准入与反复研究开发，不是独立盲测。固定5mm与已知人工noise5量级相同；当前不能证明真实域风险校准或泛化优势。

四、Raw公平对比结果（既有开发池，缓存官方协议统计）
四退化AP为10场景等权、每场景四退化等权；不是全量GraspNet测试AP。Top-K以下列出mu=0.4。
方法 | clean AP% | 四退化AP% | Top1% | Top5% | Top10%
'''
    for arm in ('Raw', 'A', 'C0'):
        a = audit['aggregates'][arm]
        top = a['degraded_TopK']
        text += f"{arm} | {a['clean_AP']*100:.6f} | {a['degraded_AP']*100:.6f} | {top['1']['mu04']*100:.2f} | {top['5']['mu04']*100:.2f} | {top['10']['mu04']*100:.2f}\n"
    text += '''C0相对Raw四退化AP高2.409068个百分点；相对A高0.948527个百分点。该正向开发结果支持保留研究价值，不构成已校准概率、独立确认、仿真收益或统计显著性结论。
完整公式、逐场景结果、选择支持和来源：results/simulation_v52/audit/summary.json；50个隔离策略输入位于同目录policy_inputs。
R1独立文本中“Panda”是沿用的适配提示；本阶段实际执行平台为UR5e+Robotiq 2F-140，下面的数据及机器人manifest为准。保留原文本并在此更正，不覆盖原审计记录。

五、视觉指标保留（原v39缓存结果，本轮未重跑）
AP50为原10场景AP50等权汇总；与原macro_AP使用相同场景分母。以下数值不能解释为新的分割收益。
模型 | 条件 | Mask AP% | AP50%
'''
    for r in vision:
        text += f"{r['arm']} | {r['condition']} | {r['AP']*100:.6f} | {r['AP50']*100:.6f}\n"
    text += '''原始来源：results/P5/completion_v39/full/metrics；本次汇总源身份保存在delivery_summary.json。
现有深度退化结果保留；本轮没有新增遮挡分层评价或新视觉数据。普通RGB-D作为后续前端是本次研究作用调整，不表示它已在这些开发数据上优于RGB。

六、仿真环境与静态接口
独立Python3.12.14、PyBullet3.2.7、本地CPU TinyRenderer；numpy1.26.4、scipy1.15.3、trimesh4.9.0、Pillow11.3.0、psutil7.0.0、xacro2.1.1。环境与构建日志保留在runtime/simulation_v52及logs/simulation_v52。
UR5e与2F-140来自官方模型仓库固定commit，转换后的网格、许可证和SHA见third_party/simulation_v52/provenance.json及data/simulation_v52/robot/robot_assets.json。
UR5e六个受控机械臂关节；夹爪一个主动关节与五个机械耦合关节。实际模型内指垫最大开口128.534627mm，不以名义140mm冒充真实可用开口，不静默夹紧候选宽度。
坐标约定：^A T_B把B坐标映射到A；沿用相机到台面的标定。GraspNet接触中心=t+R[:,0]×depth；夹爪+Z为接近轴、+Y为闭合轴；旋转映射及开口相关TCP偏移见sim_v52_math.py。
1000个已有P3候选的坐标往返、接触中心和轴向检查通过；最大往返误差1.862e-16m；相机平面反投影检查误差5.972e-8m。仅为数学/静态检查，不等于1000次物理执行。
固定3个GraspNet物体：021球、014桃子、017梅子。021凸代理通过；014/017使用原网格内点的<=1.5mm占据网格、无填洞，精确合并819/640碰撞盒，边界误差1.275/1.261mm。源模型、原尺寸、失败的单凸包/VHACD记录均保留。
这是三种小型紧凑物体的受限工程范围，不能外推到复杂凹物体、拥挤场景或实机。物体质量/摩擦和机器人惯量采用冻结配置，不宣称已完成真实系统辨识。
几何静态准入、导入与逆运动学接口完成；接触受载动力学尚未合格，两者分开报告。

七、48回合完整结果
固定3物体×4桌面位置×（2个正向姿态+1错位负例+1桌碰负例），共48槽位；所有失败和修复后运行均计入同一48上限，未重跑被占用槽位。
每回合先登记意图，再启动进程；48启动/48返回/0新unknown/0超时或强杀，旧历史2个未知官方调用没有被抹除。
成功冻结标准包括抬升>=0.10m、最后连续保持>=2s、双侧指垫接触、无台面支持/滑落/禁止碰撞、全过程耦合误差<=0.03rad。正向门槛20/24且各物体至少6/8。本批阈值未放宽。
物体 | 正向合格/分母 | 错位正确/分母 | 桌碰正确/分母
021 | 0/8 | 4/4 | 4/4
014 | 0/8 | 4/4 | 4/4
017 | 0/8 | 4/4 | 4/4
合计 | 0/24 | 12/12 | 12/12
全部48次的协议成功0/48。含负对照的比率仅用于完整计账，不与后续只含有效抓取任务的算法成功率混用。
结果分类：13次计划桌碰拒绝（12负例+原正向slot0）、23次dynamic_coupling_failed、12次错位抓取/抬升失败。
实际禁止碰撞8/48=16.67%（正向8/24=33.33%）；计划拒绝与真实碰撞分别记录。
不可执行13/48=27.08%，含12个刻意构造的桌碰负例；正向不可执行1/24=4.17%。
滑落4/48=8.33%；在形成抓持的23次中为4/23=17.39%。所有滑落/碰撞均保留。
'''
    text += f"物理动作诊断：{diagnostic['grasp_formed']}次形成抓持，{diagnostic['ever_lifted_10cm']}次曾抬升>=10cm，{diagnostic['ever_held_2s']}次曾连续保持>=2s，{diagnostic['final_held_2s']}次最终保持>=2s；均不得绕过耦合/碰撞失败。全过程最大耦合偏差{diagnostic['max_mimic_error_rad']:.6f}rad。\n"
    text += '''八、工程失败、修复及版本变化
v52：slot0实际占用第1回合，因工程候选接触高度过低导致计划夹爪扫桌。原失败和源代码保留。
v52r1：用已知机器人完整闭合包络及固定台面净空调整工程候选高度；不修改任何已有P3候选位置/朝向。slot1形成真实抬升，但耦合误差0.181311rad且出现禁止自碰，协议失败。
v52r2：经过零物理步的API/源码和机构矩阵检查，将5条gear换为同解空间的局部树，未增加独立从动电机、驱动力、摩擦或门槛。有限力的动力学不保证等价，故仍按原门槛验证。slot2至47共46次占用剩余预算；负载漂移仍存在，不再追加版本或回合。
来源修复记录：docs/机器人仿真工程适配修复_v52r1.txt、docs/机器人仿真机械耦合诊断与等价表示_v52r2.txt；原/恢复配置及freeze分别保留。
本批版本与场景不同，不能从单个前后回合宣称某修复改善成功率；所有分母包含各版本失败。

九、案例及完整日志
合格成功案例：无。不得把抬升画面标成协议通过案例。
slot4（005_slot_04_4f097762）：最大抬升0.147278m、最终保持3s、无滑落及禁止碰撞，但耦合误差0.193643rad，明确记失败。
slot0：规划桌碰失败；slot2：错位负例正常执行后失败；slot3：桌碰负例在规划阶段拒绝。
slot32、40、41、44：形成抓持后滑落，全部保留，不从分母排除。
每个目录含intent、result、resources、stdout/stderr；实际执行回合另含trace、collision_events、观察RGB-D和初末RGB。规划早退可能无运动轨迹，明确为空而非缺造轨迹。
传感器相机初末图用于输入留档，末态可能被机械臂遮挡，不能只凭图片判断成功；以接触/高度/约束日志为准。本轮未补拍、重放或生成替代成功图。
全48案例及逐文件hash：results/simulation_v52/delivery_case_index.json；账本：logs/simulation_v52/episodes.jsonl；全结果：results/simulation_v52/admission/report_048.json。

十、资源统计与实际计算量
'''
    text += f"48回合子进程累计墙钟{report['total_worker_wall_seconds']:.3f}s，累计CPU时间{report['total_worker_cpu_seconds']:.4f}s，进程树RSS采样峰值{report['peak_tree_rss_bytes']}B（{report['peak_tree_rss_bytes']/1024**2:.2f}MiB）。\n"
    text += f"实际物理步{diagnostic['physics_steps']}，dt=1/240s，累计模拟时间{diagnostic['physics_steps']/240:.1f}s。第一次启动至最后返回的现实窗口{diagnostic['episode_window_wall_seconds']:.3f}s，包括诊断、静态修复、审核和人工调度间隔，不等于持续占用计算资源。\n"
    text += f"R1缓存审计计算墙钟{audit['elapsed_seconds']:.3f}s；独立复核与安装/源编译不计入上述R3子进程数字。安装编译全过程CPU/RSS峰值未完整监测，保持未知。\n"
    text += '''机器：6物理/12逻辑CPU，约31.81GiB内存，RTX4060Ti 16GB。R3使用CPU TinyRenderer，无模型前向或GPU计算分配；GPU快照为系统级数据，不能冒称任务专属显存峰值为0。
各回合0.25秒采样资源文件完整保留；RSS采样峰值不声称捕获所有瞬时峰值。无资源越界或超时中断，16GB显存不是本批阻塞原因。

十一、下一步修复方案（未执行，等待研究审核）
先解决夹爪机械表示与受载一致性，再恢复算法比较。优先从官方机械几何推导平行指尖模式的真实闭环、自由度及单执行器传动；用有依据的闭环点/铰链约束替代冗余gear，不能随意叠加约束，也不能给每根从动指独立无限电机。
在零步几何与自由度核验后，只申请一次预注册小预算的受载/收敛诊断，例如dt两档×迭代数两档；质量、摩擦、扭矩、场景和原成功标准固定。所有动力学调试计账，不能在已耗尽的48预算外偷偷补跑。
逐tick记录q/dq、闭环残差、实际电机扭矩、约束反力、指垫接触、输入功率，区分机构错误、数值未收敛和输入饱和。当前证据不能唯一归因成某个PyBullet bug；不盲目加力、加摩擦或放宽0.03rad门槛。
若闭环实现仍受接口限制，才考虑带传动与输入功率约束的受限等效平行执行器，并明确模型边界。不能冒称完整2F-140真实自适应机制。
工程链合格且研究审核后，再接入原P3固定候选及P5普通RGB-D预测，冻结Raw/A/C0同场景同池对比，完整记不可执行与所有失败；仍须单独批准R4，不能由本报告自动触发。
详细依据：docs/机器人耦合工程缺口与后续修复建议_v52.txt。

十二、完整性、进度与停止状态
独立复核通过：48意图/48返回/0新未知；所有报告重算一致；2450个受保护历史文件及冻结源身份不变。原主runtime/pipeline/state.json未修改，原P5/P6科学结论不改。
当前状态runtime/pipeline/simulation_v52.json由正式controller写为completed_stopped_for_review，R3_complete=true、engineering_passed=false、R4_authorized=false。本交付器只生成新文档/索引，不改该状态。
全部任务总进度：旧范围45%（原45/100权重点）。实验总进度：旧范围56.25%（原45/80权重点）。这是旧主交付范围的完成记账，不是成功率或当前新增仿真范围的精确百分比。
新增范围：R1审计完成；R2环境与静态接口完成；R3实际48/48且已停止、工程未通过；R4未启动。新增范围尚未建立重新加权总分母，当前完整扩展项目百分比待重算，不以48/48冒称全部完成。
本批没有更改论文题目，没有形成机器人成功率优势或P5/P6通过的新科学结论。保留原研究方向及资产，但进入论文方法对比前必须补齐受载夹爪工程可信性。

附：全48回合摘要（slot从0开始；NA为计划拒绝前未测，不填0冒充）
slot | object | kind | outcome | lift_m | final_hold_s | max_mimic_rad | collision | slip
'''
    def fmt(value):
        return 'NA' if value is None else f'{value:.6f}'
    for c in case_index:
        text += f"{c['slot_id']} | {c['object_id']} | {c['kind']} | {c['outcome']} | {fmt(c['max_lift_m'])} | {fmt(c['final_continuous_hold_seconds'])} | {fmt(c['max_mimic_error_rad'])} | {c['forbidden_collision']} | {c['slipped']}\n"
    text += '\n交付索引\n'
    for path in (
        'docs/P5-P6模块与Raw基线审计_v52.txt',
        'docs/机器人仿真工程准入协议_v52.txt',
        'docs/机器人耦合工程缺口与后续修复建议_v52.txt',
        'results/simulation_v52/audit/summary.json',
        'results/simulation_v52/coordinate_checks.json',
        'results/simulation_v52/object_assets_voxel_repair.json',
        'data/simulation_v52/robot/robot_assets.json',
        'logs/simulation_v52/environment_before_physics.json',
        'results/simulation_v52/admission/report_048.json',
        'logs/simulation_v52/final_review.json',
        'results/simulation_v52/delivery_summary.json',
        'results/simulation_v52/delivery_case_index.json',
        'runtime/pipeline/simulation_v52.json'):
        text += str(ROOT/path) + '\n'
    destination = ROOT/'docs/P5-P6模块调整与机器人仿真工程验证结果_v52.txt'
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(text)
    print(json.dumps(dict(delivery=relative(destination), cases=len(case_index),
        historical_review_passed=True, engineering_passed=False, R4_started=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
