"""Publish existing v52d evidence and stop. Never launches/connects/steps physics."""
import json,sys,shutil
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sim_v52d_control import read,identity,relative,now,once_sync,verify_frozen,historical_verify,status,lock

def main():
    assert Path.cwd().resolve()==ROOT
    import psutil
    active=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid==__import__('os').getpid():continue
        cmd=' '.join(p.info.get('cmdline') or [])
        if (p.info.get('name') or '').lower().startswith('python') and ('sim_v52d_episode.py' in cmd or ('sim_v52d_control.py' in cmd and ' run' in cmd)):
            active.append(dict(pid=p.pid,command=cmd))
    assert not active,active
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52d.lock'):
        cfg=read(ROOT/'configs/simulation_v52d.json');verify_frozen(cfg)
        protected=historical_verify()
        execution_path=ROOT/'results/simulation_v52d/execution_report_12_aa98f08a.json'
        review_path=ROOT/'logs/simulation_v52d/reviews/final_20261103_145256_1793688776378611100.json'
        analysis_path=ROOT/'results/simulation_v52d/analysis/dynamics_analysis.json'
        e,r,a=map(read,[execution_path,review_path,analysis_path])
        assert e['diagnostic_complete'] and e['charged_attempts']==e['results_returned']==12
        assert e['actual_physics_steps_returned']==79920 and e['exact_total_steps_known'] and not e['unknown_attempts']
        assert not r['review_passed'] and r['checks']['failure_counts']=={'actual_pad_friction':24}
        assert r['all_12_completed'] and r['actual_recorded_diagnostic_steps']==79920
        assert a['analyzed_returned_steps']==79920
        requirements=['docs/v52d_参数读回差异专项核查.txt','docs/v52d_动力学根因分析_校订版.txt','docs/v52d_M1修复设计草案.txt']
        for name in requirements:assert (ROOT/name).is_file(),name
        analysis_units={x['unit']['id']:x for x in a['units']}
        lines=[
            'v52d：十二次M0有限动力学诊断结果、根因分析与待审修复方案',
            '交付时间：'+now(),
            '',
            '结论：本次12/12启动并完整返回，0重试、0异常、0未知，已自动停止。记录完整性成立；参数一致性未通过，M0接触受载耦合也未通过。M0仍不具备重启工程准入或算法比较的条件。历史48回合和旧P5/P6结论不变。M1只给设计，未实施。',
            '',
            '一、执行范围与账本',
            '严格复用冻结M0=v52r2及原slot4动作；按N/L/C × dt(1/240、1/480秒) × 迭代(150、300)，每格1次、原顺序执行。8格无物体、4格固定021球体接触；没有新候选选择、标签参与动作、附着约束、物体后续位姿重置、新训练或神经前向。',
            '6格各4440步、6格各8880步，完整逐tick记录共79920步，累计222仿真秒。这个计数来自返回账本与全量日志复核，不是将名义帧数当实际步骤。物理进程已退出，发布时也核查无活跃诊断worker。',
            '历史48加本次12共60次有登记运行；其中本次仅4个含物体接触单元，不能把8个N/L称为抓取试验或合并成成功率。',
            '',
            '二、12格结果（原post_settle评价窗口；原0.03rad门槛不变）',
            '单元 | 工况 | dt | 迭代 | 原窗口最大耦合误差rad | master峰值Nm(全tick) | gear峰值Nm(全tick) | 原接触结论',
        ]
        for row in e['records']:
            result=row['result'];u=row['intent']['unit'];detail=analysis_units[u['id']]
            lines.append(f"{u['id']} | {u['scenario']} | 1/{round(1/u['time_step_s'])} | {u['solver_iterations']} | {result['max_mimic_error_rad']:.9g} | {detail['overall']['master_torque_max_abs_Nm']:.6g} | {result['summary']['max_abs_gear_reaction_Nm']:.6g} | "+('耦合失败' if u['scenario']=='C' else '抓取不适用；耦合低于门槛'))
        lines += [
            '',
            'C四格均抬升约14.6–14.9cm、连续保持3秒，无滑落和禁止碰撞，但耦合误差0.0962175–0.193643rad（门槛的3.21–6.45倍），原判定仍失败。不能用抬升成功替代整体验收。',
            'N/L全tick最大值（额外包含settle）为约5.91231e-5或2.92170e-5rad，略高于上表旧窗口值；两套统计均保留，没有更换验收窗口。',
            '本12格未出现禁止碰撞，不反向消除旧48中的自碰证据；单个已暴露球场景不能代表全部物体和姿态。',
            '',
            '三、完整记录审查与新发现的参数下发缺陷',
            f"冻结独审共{r['checks']['checks']:,}项计算检查，24项actual_pad_friction失败，其余{r['checks']['checks']-24:,}项通过。检查条数不是研究工作完成百分比，也不是统计样本量。原失败报告完整保留，不改为通过。",
            '已逐tick复算q/dq、实际电机扭矩、5gear残差及反力、接触法向/切向合力矩、外载方向/作用点/COM力矩/材料点位移功、12DOF实际质量矩阵及原接触判定。所有预期字段和步号连续，无缺测补造。',
            '实际读回：12格中两指垫摩擦均0.5、碰撞margin均0.001m；机器人base(-1)摩擦均1.0、margin均0.0005m。代码原希望把1.0/.0005设置给指垫。物体质量0.04kg、摩擦0.6，关节惯量等另有实际读回与复核。',
            '原因已有具体源码与观测闭环：本机PyBullet3.2.7的changeDynamics组合调用在最后处理collisionMargin；b3ChangeDynamicsInfoSetCollisionMargin不接收linkIndex并把共享命令的m_linkIndex设为-1，先前lateralFriction/restitution标志仍在，所以整组命令最终发给base。十二份base/pad读回精确符合此行为，不是遥测字段错位。',
            '这是参数路由缺陷的直接证据；它不等于证明物理引擎求解器整体有Bug，也不证明它是耦合漂移唯一根因。此前只确认代码命令而未充分读回实际属性，验证存在缺口，本轮已将其暴露并单独记录。',
            '本轮没有人为改变这些物理参数，全部格沿同一冻结M0路径执行；但不能再把命令值1.0/.0005当实际生效值。本矩阵可用于实际M0状态下的故障比较，不能称为预期指垫摩擦1.0条件下的验收。旧48缺少同样逐次读回，不能为其补造实际参数。',
            '只拆分kwargs并不能完整修复：独立的collisionMargin调用仍会定位到base。box的setMargin还会调整隐式半尺寸以保持外尺寸，因此不能把0.001与0.0005的差直接解释成指垫膨胀0.5mm或自碰唯一原因。',
            '',
            '四、载荷、驱动力、接触与数值因素的区分',
            'L在close/lift/hold逐步按两指垫实际碰撞盒中心连线施加左-1Ne、右+1Ne；世界坐标作用点、COM力矩r×F和同一材料点离散功均独立复核。两外力合力为0、各力模误差不超过2.23e-16N；外载累计离散功约-0.06815J，说明加载没有遗漏。1N仅是外部标定载荷，绝不是实测夹持力。',
            'C每侧接触法向峰值约12–51N（依组合而变），作用点、接触摩擦和对象约束也不同。L通过只能说明这个轻载工况可维持耦合，不能排除更强外载问题，不能把它当作与C等效载荷。',
            'C首次双指接触后约0.067–0.088秒出现>0.03rad失配，均在close阶段；最大误差通常在lift。误差主要集中inner_knuckle—inner_finger两条同侧关系，跨侧同步最大残差约0.00271rad。闭环锚点最大分离约2.18–4.27mm。这支持接触受载时约束/驱动/接触联合响应不可靠，而不是已经证明关节符号或秩设置错误。',
            '所有逐步记录未达到预定99%上限观察线：master峰值2.430/4.857/4.868/9.679Nm，gear最高4.797Nm相对80Nm。DYN012 master已达96.79%，不能称远离上限；单凭跟踪误差不能断言扭矩限幅，现有证据也不支持直接增力。99%仅诊断计数线，不是新验收门槛。',
            '固定dt时150→300迭代使最大耦合误差仅下降约0.70%/0.26%，同时master扭矩与接触力接近翻倍；不能称求解已收敛、反力已稳定。dt减半使误差约减半但仍失败；固定维度增益/ERP时dt变化同时影响伺服和约束修正尺度，不能单独归因于纯数值精度。',
            '实际全q质量矩阵得到d_full/d_src约0.894509–1.132996；这是同体行归一化差异的证据，尚不是接触联合求解漂移的唯一因果证明。',
            '功率和功均明确为离散估计：tau_post*dq_post、tau_post*Δq、gear反力*Δ残差、Fpre·材料点位移。不能据此声称精确能量守恒或真实硬件夹持力。',
            '',
            '五、M1修复方案（仅设计，未实施）',
            '优先顺序需要调整：先解决参数路由和实际读回一致性，再隔离评估机构约束表示；不能一边换闭环，一边增加摩擦、驱动力或修改碰撞形状。',
            'A. 参数接口层：在独立版本中修复或替代会重置linkIndex的调用路径，并对base、每个pad、对象做启动读回及差异即停。单纯拆开现有调用不是完整margin修复。具体后端接口实现、单位和对象索引必须审查；本轮不安装/编译/更换后端。',
            'B. 公平基线：若后续只研究闭环M1，应在新协议中明确锁定本次实际M0的pad摩擦0.5、margin0.001及base状态，使结构比较不混入材料参数改变。若研究审核要求恢复原声称的1.0/.0005，那应先作为单独参数修复对照，明确它相对本次实际M0是物理条件变化，不能与结构修复同时开展或称为原条件不变。两条分支均尚未获执行授权。',
            'C. 机构层：保留平行四杆范围、原几何/惯量、唯一主动关节10Nm及左右同步；用每侧两个独立的闭环位置条件替换四条同侧gear关系，保留跨侧同步一式。旧五行本来秩5，不因“冗余”而删除；新旧约束不能叠加。',
            'D. 锚点：复用已核官方140模型的inner_knuckle局部点[0,.0999753876967673,-.002218525382070201]和inner_finger局部点[0,.006515,-.017901]，按实际link/COM惯性变换转换。所有闭合行应验证有效rank5、剩余1自由度和原装配支路。',
            'E. 后端与反力：stock P2P每侧有3个world行但平面几何只有2秩，不能固定删某个world轴或未经核查直接叠加。还需核同体交叉项、相关行处理、反力及离散功。gear80Nm与P2P80N不能视为相同上限；用J^Tλ与旧C^Tμ映射审核被动反力，不能任选大数、增加从动电机或通过每步重设关节实现伪闭环。',
            'F. 审核后实施时仍保持0.03rad、原碰撞/抬升/保持标准；先静态接口与参数一致性，再单独登记有限受载诊断预算。加载读回即使零物理步也应在未来启动账本登记。本轮12预算已用尽，不复用旧slot或自动追加。是否重新开展48准入，应在下一版实际过关后另审。',
            '详细机械依据、公式、后端风险与差异表见docs/v52d_M1修复设计草案.txt；其中未来方案须叠加本报告新发现的参数路由前置关口。',
            '',
            '六、资源及历史完整性',
            f"12个worker累计wall {e['total_worker_wall_seconds']:.3f}秒（约{e['total_worker_wall_seconds']/60:.2f}分钟）；会话记录{e['session_wall_seconds']:.3f}秒。单次最大{max(x['returned']['wall_seconds'] for x in e['records']):.3f}秒，低于120秒上限；总worker低于1440秒、会话低于1800秒。",
            f"采样CPU累计下界{e['total_polled_cpu_seconds_lower_bound']:.3f} CPU秒；采样进程树RSS峰值{e['peak_sampled_tree_rss_bytes']/1024**2:.2f}MiB，低于6GiB。GPU计算0、训练0、神经前向0。",
            f"独审统计新结果及日志{r['output_bytes']:,}字节，低于2GiB。CPU/RSS按0.25秒轮询，不能当连续测量精确峰值；只读分析与文档整理不计为新增物理运行，未制造未计账的物理预热。",
            f"{protected}个受保护旧文件的SHA256/大小/mtime保持一致，含原48、旧主state及冻结资产。新的结果、失败审查、准备期旧版本、源码核查均另存；实际参数差异未通过倒改配置来抹除。",
            '',
            '七、交付与停止点',
            '主结论：当前M0不可用于R4算法比较；本次逐步日志可用作故障定位证据。已确认参数下发缺陷；已确认接触受载耦合不合格；尚未证明唯一机械/数值根因。不得将本次“诊断完成”改写为“工程准入通过”。',
            '全部任务总进度：旧主交付范围45%；实验总进度：旧主交付范围56.25%。新增仿真范围仍未重新加权，扩展项目总体百分比待重算。当前专项12/12完成、记录完整；参数一致性及接触机械门槛未过。',
            '已停止等待研究审核。未实现M1、未启动新的48、R4/72、720或240；没有周期任务或自动重试。',
            '',
            '证据索引（均在项目根目录下）',
            relative(execution_path),relative(review_path),relative(analysis_path),
            'docs/v52d_动力学根因分析_校订版.txt','docs/v52d_参数读回差异专项核查.txt','docs/v52d_M1修复设计草案.txt',
            'configs/simulation_v52d.json','logs/simulation_v52d/attempts.jsonl','results/simulation_v52d/attempts/*/telemetry.jsonl.gz',
            '每个attempt目录还包含逐次result、telemetry_readback、资源、stdout/stderr、precharge和claim。',
            '',
            '一级来源与适用范围',
            '机械描述：https://github.com/robotiq/robotiq-ros/blob/721dbbae8377477a5ab729d579ce9cd92927e824/grippers/robotiq_description/urdf/robotiq_2f_140.xacro',
            '闭环模型参考：https://github.com/robotiq/isaacsim_assets/blob/27ae16b7d2c7d7e9011544521d644a0f38c4b501/grippers/Robotiq_2F_140/payloads/Robotiq_2F_140_newton_fourbar_physics.usda',
            '具体参数路由依据为本机PyBullet3.2.7 sdist SHA256 042879db8d101ac7590dee475fc6aded508b85fe1273fdbbfde1d88bd200e14f及十二份实际读回，不以其它版本或通用文档代替。',
        ]
        report_path=ROOT/'docs/v52d_十二次动力学诊断结果与修复建议.txt'
        with report_path.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
        archive=ROOT/'logs/simulation_v52d/delivery';archive.mkdir(exist_ok=False)
        shutil.copy2(ROOT/'runtime/pipeline/simulation_v52_dynamics_run.json',archive/'state_before_delivery.json')
        handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,archive/'resume_note_before_delivery.txt')
        old=handoff.read_bytes()
        prefix=f"最新：{now()} v52d本轮12/12单次M0诊断完整完成并停止，79920实际返回步，0异常/未知/重试。逐tick完整性核对完成，但正式只读审查actual_pad_friction24项失败（其它10489335项通过）：pad实际.5/.001，base1/.0005，已定位3.2.7 collisionMargin setter把共享命令linkIndex置-1的参数路由缺陷。四C耦合.0962-.1936rad均超过.03，不具备新48/R4准入；旧48失败不改。主报告docs/v52d_十二次动力学诊断结果与修复建议.txt；状态runtime/pipeline/simulation_v52_dynamics_run.json。M1仅草案，先审接口/参数一致性再隔离闭环修复；不得启动任何补跑、M1、48、72、720或240，必须等待用户新审核。全部任务总进度旧范围45%，实验总进度旧范围56.25%，扩展范围待重算。本次{protected}历史文件完整性保持。\n\n历史：\n"
        handoff.write_bytes(prefix.encode('utf-8')+old)
        artifacts={}
        for folder in ['results/simulation_v52d','logs/simulation_v52d']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file():artifacts[relative(p)]=identity(p)
        for name in requirements+[relative(report_path),'docs/实验自动续接说明.txt','scripts/publish_v52d_diagnostic_report.py']:
            artifacts[name]=identity(ROOT/name)
        closeout=ROOT/'results/simulation_v52d/delivery_closeout.json'
        once_sync(closeout,dict(at=now(),phase='delivered_stopped_for_review',diagnostic_complete=True,
            data_completeness_passed=True,parameter_consistency_passed=False,formal_review_passed=False,
            original_contact_gate_passed=False,approved_starts=12,charged_starts=12,returned_complete=12,
            actual_physics_steps=79920,unknown=0,retries=0,historical_files_unchanged=protected,
            active_physics_workers=active,new_physics_in_publication=0,M1_implemented=False,new_48_started=False,R4_started=False,
            report=relative(report_path),review=relative(review_path),review_failures=r['checks']['failure_counts'],artifacts=artifacts))
        status('completed_stopped_for_review',diagnostic_complete=True,attempts_charged=12,results_returned=12,
            actual_physics_steps_returned=79920,unknown_attempts=[],report=relative(report_path),
            execution_report=relative(execution_path),closeout=relative(closeout),
            data_completeness_passed=True,parameter_consistency_passed=False,formal_review_passed=False,
            M0_ready_for_admission=False,review_required=True)
        print(json.dumps(dict(report=relative(report_path),closeout=relative(closeout),protected=protected,
            new_physics=0,parameter_consistency_passed=False),ensure_ascii=False))

if __name__=='__main__':main()
