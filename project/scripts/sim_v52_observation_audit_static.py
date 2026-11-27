"""Read existing v52 logs only. No simulator import, client, physics or trials."""
import sys,os,json,math,hashlib,statistics,collections,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/simulation_v52_dynamics_audit/observations.json'
DOC=ROOT/'docs/v52专项既有48回合受载证据审计.txt'
EPISODES=ROOT/'results/simulation_v52/episodes'
BINDINGS={}

def identity(path):
    s=path.stat()
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def rel(path):return path.relative_to(ROOT).as_posix()
def read(path):
    path=Path(path);BINDINGS[rel(path)]=identity(path)
    return json.loads(path.read_text('utf-8-sig'))
def textread(path):
    BINDINGS[rel(path)]=identity(path)
    return path.read_text('utf-8-sig')
def stats(values):
    values=[float(x) for x in values if x is not None]
    return dict(n=len(values),min=min(values),max=max(values),mean=statistics.fmean(values),median=statistics.median(values),max_abs=max(map(abs,values))) if values else dict(n=0,min=None,max=None,mean=None,median=None,max_abs=None)
def once(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def phase_bounds(cfg):
    dt=cfg['physics']['time_step'];last=round(1/dt);out=[dict(phase='settling_uninstrumented',first_step=1,last_step=last)]
    durations=[('opening',.8),('transit',cfg['motion']['transit_seconds']+.3),('pregrasp',cfg['motion']['pregrasp_seconds']+.3),('approach',cfg['motion']['approach_seconds']+.3),('close',cfg['motion']['close_seconds']),('lift',cfg['motion']['lift_seconds']+.3),('hold',cfg['motion']['hold_seconds'])]
    for name,d in durations:
        count=round(d/dt);out.append(dict(phase=name,first_step=last+1,last_step=last+count));last+=count
    return out
def phase_at(step,bounds):
    return next((b['phase'] for b in bounds if b['first_step']<=step<=b['last_step']),'outside_known_schedule')

def extract_urdf(path):
    xml=ET.fromstring(textread(path));links=[];joints=[]
    for link in xml.findall('link'):
        inertial=link.find('inertial')
        links.append(dict(name=link.attrib['name'],mass_kg=float(inertial.find('mass').attrib['value']) if inertial is not None else None,
            inertial_origin=dict(inertial.find('origin').attrib) if inertial is not None and inertial.find('origin') is not None else None,
            inertia=dict(inertial.find('inertia').attrib) if inertial is not None else None,
            collision_count=len(link.findall('collision')),contact_elements=[dict(e.attrib) for e in link.findall('contact')]))
    for j in xml.findall('joint'):
        joints.append(dict(name=j.attrib['name'],type=j.attrib['type'],limit=dict(j.find('limit').attrib) if j.find('limit') is not None else None,
            dynamics=dict(j.find('dynamics').attrib) if j.find('dynamics') is not None else None))
    return dict(path=rel(path),links=links,joints=joints)

def run():
    assert Path.cwd().resolve()==ROOT
    assert not OUT.exists() and not DOC.exists(),'No overwrite'
    configs={}
    for name in ('simulation_v52','simulation_v52r1','simulation_v52r2'):
        path=ROOT/'configs'/f'{name}.json';cfg=read(path);configs[identity(path)['sha256']]=(path,cfg)
    manifest=read(ROOT/'data/simulation_v52/robot/robot_assets.json')
    master=manifest['master_joint'];followers=manifest['mimic_joints'];groups=collections.defaultdict(list)
    episode_rows=[];collision_groups=collections.defaultdict(list);trace_fields=collections.Counter();gear_unavailable=0
    files=sorted(p for p in EPISODES.iterdir() if p.is_dir())
    assert len(files)==48,'Expected precisely48 historical episodes'
    for directory in files:
        result=read(directory/'result.json');trace=read(directory/'trace.json');collisions=read(directory/'collision_events.json')
        resources=read(directory/'resources.json');intent=read(directory/'intent.json');policy=read(directory/'policy_input.json')
        stdout=textread(directory/'stdout.log');stderr=textread(directory/'stderr.log')
        path,cfg=configs[result['config_identity']['sha256']];version=str(cfg['version'])
        assert result['config_identity']==identity(path)==intent['config']
        assert resources['result']==identity(directory/'result.json') and resources['result_status']=='known_result'
        assert resources['exit_code']==0 and resources['termination'] is None
        assert result['collision_events_count']==len(collisions)
        slot=result['slot'];oid=slot['object_id'];kind=slot['kind'];dt=cfg['physics']['time_step'];bounds=phase_bounds(cfg)
        sampled=[];last=None;computed_error_checks=[]
        for row in trace:
            trace_fields.update(row.keys());assert row['phase']==phase_at(row['step'],bounds)
            q={master:row['driver_rad'],**row.get('follower_joint_positions',{})}
            residuals={r['joint']:q[r['joint']]-(q[master]*r['multiplier']+r['offset']) for r in followers if r['joint'] in q}
            if residuals:computed_error_checks.append(abs(max(map(abs,residuals.values()))-row['mimic_error_rad'])<1e-12)
            gear_residuals={}
            for index,edge in enumerate(cfg.get('coupling_repair',{}).get('tree',[])):
                if edge['a'] in q and edge['b'] in q:gear_residuals[str(index)]=q[edge['a']]+edge['ratio']*q[edge['b']]
            reactions={}
            for index,value in enumerate(row.get('gear_constraint_states',[])):
                if isinstance(value,list) and len(value)==1 and isinstance(value[0],(int,float)):reactions[str(index)]=value[0]
                else:gear_unavailable+=1
            dq_estimate={}
            if last is not None and row['phase']==last['phase']:
                interval=(row['step']-last['step'])*dt
                for name,value in q.items():
                    if name in last['q']:dq_estimate[name]=(value-last['q'][name])/interval
            sample=dict(step=row['step'],sim_time_s=row['step']*dt,phase=row['phase'],q_rad=q,
                sampled_master_mimic_residual_rad=residuals,sampled_local_gear_residual_rad=gear_residuals,
                sampled_generalized_gear_reaction_Nm=reactions,
                estimated_interval_average_dq_rad_s=dq_estimate,velocity_is_measured=False,
                bilateral_contact=row['bilateral_contact'],supported=row['supported'],
                mimic_error_rad=row['mimic_error_rad'],lift_m=row['lift_m'])
            groups[(version,oid,kind,row['phase'])].append((slot['slot_id'],sample));sampled.append(sample)
            last=dict(phase=row['phase'],step=row['step'],q=q)
        assert all(computed_error_checks),'Recorded mimic error disagrees with sampled actual positions'
        for event in collisions:
            phase=phase_at(event['step'],bounds);collision_groups[(version,oid,kind,phase,event['kind'])].append(event)
        phase_detail={}
        for phase in dict.fromkeys(s['phase'] for s in sampled):
            rows=[s for s in sampled if s['phase']==phase]
            phase_detail[phase]=dict(sample_count=len(rows),sampled_mimic_error_rad=stats(s['mimic_error_rad'] for s in rows),
                sampled_bilateral_count=sum(s['bilateral_contact'] for s in rows),sampled_supported_count=sum(s['supported'] for s in rows),
                sampled_q_rad={n:stats(s['q_rad'].get(n) for s in rows) for n in set().union(*(s['q_rad'] for s in rows))},
                sampled_master_mimic_residual_rad={n:stats(s['sampled_master_mimic_residual_rad'].get(n) for s in rows) for n in set().union(*(s['sampled_master_mimic_residual_rad'] for s in rows))},
                sampled_local_gear_residual_rad={n:stats(s['sampled_local_gear_residual_rad'].get(n) for s in rows) for n in set().union(*(s['sampled_local_gear_residual_rad'] for s in rows))},
                sampled_generalized_gear_reaction_Nm={n:stats(s['sampled_generalized_gear_reaction_Nm'].get(n) for s in rows) for n in set().union(*(s['sampled_generalized_gear_reaction_Nm'] for s in rows))},
                estimated_interval_average_dq_rad_s={n:stats(s['estimated_interval_average_dq_rad_s'].get(n) for s in rows) for n in set().union(*(s['estimated_interval_average_dq_rad_s'] for s in rows))})
        row=dict(directory=rel(directory),version=version,config=rel(path),slot=slot,
            result={k:result.get(k) for k in ('success','outcome','grasp_formed','slipped','forbidden_collision','physics_steps','simulation_seconds','max_lift_m','final_continuous_hold_seconds','max_mimic_error_rad','dynamic_coupling_passed','final_bilateral_contact')},
            resources={k:resources[k] for k in ('wall_seconds','peak_tree_rss_bytes','cpu_seconds','new_neural_forwards','gpu_allocation_by_design')},
            trace_sample_count=len(trace),trace_intervals_in_steps=sorted(set(trace[i]['step']-trace[i-1]['step'] for i in range(1,len(trace)))),
            phases=phase_detail,collision_events_count=len(collisions),collision_event_kinds=dict(collections.Counter(e['kind'] for e in collisions)),
            collision_first_step=min((e['step'] for e in collisions),default=None),
            first_sampled_bilateral_step=next((s['step'] for s in sampled if s['bilateral_contact']),None),
            first_sampled_mimic_over_003_step=next((s['step'] for s in sampled if s['mimic_error_rad']>.03),None),
            stdout_nonempty=bool(stdout.strip()),stderr_nonempty=bool(stderr.strip()),
            no_direct_velocity_motor_torque_or_fingertip_force_record=True,sampled_state=sized_sampled(sampled))
        episode_rows.append(row)
    aggregate=[]
    for (version,oid,kind,phase),sample_pairs in sorted(groups.items()):
        rows=[s for _,s in sample_pairs]
        aggregate.append(dict(version=version,object_id=oid,condition=kind,phase=phase,
            episode_count=len(set(i for i,_ in sample_pairs)),sample_count=len(rows),
            sampled_mimic_error_rad=stats(s['mimic_error_rad'] for s in rows),
            sampled_bilateral_count=sum(s['bilateral_contact'] for s in rows),sampled_supported_count=sum(s['supported'] for s in rows),
            sampled_q_rad={n:stats(s['q_rad'].get(n) for s in rows) for n in set().union(*(s['q_rad'] for s in rows))},
            sampled_master_mimic_residual_rad={n:stats(s['sampled_master_mimic_residual_rad'].get(n) for s in rows) for n in set().union(*(s['sampled_master_mimic_residual_rad'] for s in rows))},
            sampled_local_gear_residual_rad={n:stats(s['sampled_local_gear_residual_rad'].get(n) for s in rows) for n in set().union(*(s['sampled_local_gear_residual_rad'] for s in rows))},
            sampled_generalized_gear_reaction_Nm={n:stats(s['sampled_generalized_gear_reaction_Nm'].get(n) for s in rows) for n in set().union(*(s['sampled_generalized_gear_reaction_Nm'] for s in rows))},
            estimated_interval_average_dq_rad_s={n:stats(s['estimated_interval_average_dq_rad_s'].get(n) for s in rows) for n in set().union(*(s['estimated_interval_average_dq_rad_s'] for s in rows))}))
    collisions_summary=[dict(version=v,object_id=o,condition=c,phase=p,kind=k,event_count=len(es),
        triggered_contact_normal_force_N=stats(e['force'] for e in es),penetration_distance_m=stats(e['distance'] for e in es),
        links=dict(collections.Counter(str(e.get('links',e.get('link'))) for e in es)),
        scope='Threshold-triggered forbidden contacts scanned each executiontick; excludes allowed pad-object contacts and subthreshold contacts')
        for (v,o,c,p,k),es in sorted(collision_groups.items())]
    robot=extract_urdf(ROOT/'data/simulation_v52/robot/ur5e_robotiq_2f140.urdf')
    asset_parameters=[]
    cfg=next(c for _,c in configs.values() if str(c['version'])=='52r2')
    for oid,asset in cfg['objects'].items():
        meta=read(ROOT/asset['metadata']);u=extract_urdf(ROOT/asset['urdf'])
        asset_parameters.append(dict(object_id=oid,metadata=asset['metadata'],urdf=u,
            declared_physical_assumptions=meta['physical_assumptions'],
            runtime_explicit_settings=dict(lateral_friction=cfg['physics']['object_friction'],restitution=0,linear_damping=.04,angular_damping=.04),
            not_runtime_readback=True,rolling_spinning_friction='Not explicitly set in episode; metadata0.001 is not proof it was applied; no runtime getDynamicsInfo record'))
    sources={}
    for name in ('sim_v52_episode.py','sim_v52r1_episode.py','sim_v52r2_episode.py','sim_v52_robot_assets.py'):
        p=ROOT/'scripts'/name;body=textread(p)
        sources[name]=dict(path=rel(p),relevant_lines=[dict(line=i,text=line) for i,line in enumerate(body.splitlines(),1) if any(x in line for x in ('getJointState(', 'trace.append(', '%24', 'getConstraintState(', 'changeDynamics(', 'command_gripper(', 'force=force', 'max_mimic_error_rad', 'contacts=p.getContactPoints', 'collision', 'URDF_USE_INERTIA'))])
    missing=[
        'Every-tick actual six gripper joint positions and getJointState velocity; existing10Hz position samples only.',
        'Actual applied motor torque (getJointState element3), motor target/rate/gains and saturation flags; configured10Nm is only cap.',
        'Every-tick per-constraint reaction, coordinate residual and peak timestamp; recorded r2 reaction is10Hz only.',
        'Separate left/right fingerpad contact points, normals, normal/tangent forces and total pad wrench; bilateral boolean is not force.',
        'Closed-loop anchor separation/orientation and closure Jacobian residual; current gear residual is a coordinate relationship only.',
        'Runtime getDynamicsInfo and physics parameter readback including all masses/inertias/frictions/damping, dt/substeps/globalERP.',
        'Object full pose and linear/angular velocity, support force and slip/contact-relative motion; only sampledposition/booleans and outcome flags exist.',
        'Input/output power, cumulative work/energy and actuator/constraint force balance; cannot reconstruct without actual torque and measuredvelocity.',
        'Settling240ticks have no q/dq/contact trace or coupling extrema; execution max_mimic excludes that firstsecond.',
        'A paired fixed initial scene/version diagnostic: current v52,v52r1,v52r2 have different slots, so are not controlled repair comparisons.'
    ]
    field_inventory=dict(total_episode_count=48,known_results=48,trace_episode_count=sum(bool(r['trace_sample_count']) for r in episode_rows),
        empty_trace_count=sum(not r['trace_sample_count'] for r in episode_rows),sample_rows=sum(r['trace_sample_count'] for r in episode_rows),
        field_occurrences_in_sample_rows=dict(trace_fields),gear_unavailable_sample_count=gear_unavailable,
        measured_joint_velocity=0,actual_motor_torque=0,numeric_allowed_fingerpad_contact_force=0,
        runtime_dynamics_parameter_readback=0,closed_loop_geometric_residual=0,
        existing_sample_period_s=.1,physics_step_s=1/240,
        execution_tick_max_mimic_present=sum(r['result']['max_mimic_error_rad'] is not None for r in episode_rows))
    summary=[]
    for v,o,c in sorted(set((r['version'],r['slot']['object_id'],r['slot']['kind']) for r in episode_rows)):
        rows=[r for r in episode_rows if (r['version'],r['slot']['object_id'],r['slot']['kind'])==(v,o,c)]
        summary.append(dict(version=v,object_id=o,condition=c,episodes=len(rows),success=sum(r['result']['success'] for r in rows),
            grasp_formed=sum(r['result']['grasp_formed'] for r in rows),forbidden_collision=sum(r['result']['forbidden_collision'] for r in rows),
            max_mimic_per_execution_tick_rad=stats(r['result']['max_mimic_error_rad'] for r in rows),
            max_lift_m=stats(r['result']['max_lift_m'] for r in rows),outcomes=dict(collections.Counter(r['result']['outcome'] for r in rows))))
    result=dict(at=datetime.now().astimezone().isoformat(),scope='Read-only existing48episode observation audit',
        new_physics_connections=0,new_physics_steps=0,new_grasp_episodes=0,simulator_imported=False,
        overall_progress='待固定主交付清单后重算',experiment_progress='待固定主交付清单后重算',
        field_inventory=field_inventory,summary_by_version_object_condition=summary,phase_aggregates=aggregate,
        forbidden_collision_aggregates=collisions_summary,episodes=episode_rows,
        robot_urdf_parameters=robot,robot_inertial_assumptions=manifest['inertial_assumptions'],object_parameters=asset_parameters,
        configs=[dict(path=rel(path),version=cfg['version'],physics=cfg['physics'],motion=cfg['motion'],criteria=cfg['criteria'],phase_bounds=phase_bounds(cfg),coupling_tree=cfg.get('coupling_repair',{}).get('tree')) for path,cfg in configs.values()],
        source_evidence=sources,missing_fields=missing,
        interpretation=['Sampled extrema are lower bounds on unrecorded per-tick extrema; no transient saturation exclusion.',
            'Rotational gear generalized reaction is interpreted inNm, never as fingertip normalforce; actual motor torque unrecorded.',
            'Difference quotients are interval-average estimates from0.1s samples withinphase, not measured instantaneousvelocities.',
            'Positive and miss controls are descriptive load-state evidence, not paired mechanism interventions or a unique solver-bug proof.',
            'No cross-version causal improvement claim: the versions did not execute identicalslots.'],
        input_bindings=BINDINGS,source_identity=identity(Path(__file__)))
    for name,ref in BINDINGS.items():assert identity(ROOT/name)==ref,'Historical input changed duringaudit: '+name
    result['all_audited_inputs_unchanged']=True
    once(OUT,result)
    write_doc(result)
    print(json.dumps(dict(output=rel(OUT),document=rel(DOC),inventory=field_inventory,summary=summary),ensure_ascii=False,indent=2))

def sized_sampled(rows):
    # Retain actual sampled observations and explicit inferred residual/velocity
    # fields; no interpolation or fabricated missingticks.
    return rows

def fmt(value):return '缺测' if value is None else f'{value:.8g}'
def write_doc(report):
    f=report['field_inventory'];lines=[
        'v52专项：既有48回合受载证据审计',
        '审查范围：仅解析既有日志、配置、URDF和物体metadata；未导入模拟器、未建立物理连接、未step、未新实验。',
        '全部任务总进度：待固定主交付清单后重算。实验总进度：待固定主交付清单后重算。',
        '当前专项进度：48/48既有回合只读审计完成；新动力学回合0。',
        '',
        '一、明确结论',
        '23个实际进入接触执行的正例均违反原0.03rad机械耦合门槛；另1个正例规划阶段被桌碰检查拒绝。12个错位负例与12个桌碰负例均正确，不代表正例工程通过。',
        '空闭合与接触后的耦合误差明显不同，支持受载内指机构约束存在问题；现有记录不足以唯一定位实际电机饱和、gear瞬时饱和、闭环机构缺失或求解器误差。',
        'v52仅slot0、v52r1仅slot1、v52r2执行slot2..47，位置/姿态/物体不同；不得把版本间数值差异称为受控修复收益。',
        '',
        '二、记录覆盖与口径',
        f'已核48个result/trace/collision_events/resources/intent/policy_input及stdout/stderr；全部known_result、exit0。35回合各175条trace，共{f["sample_rows"]}条；13个规划拦截回合trace为空，不能当作零误差。',
        '物理dt=1/240s；trace每24tick一条，即0.1s采样。最初240tick落桌静置无trace和耦合极值监测。',
        '35个执行回合有master实际位置与执行tick期间总max_mimic；r1不保存分关节位置。r2的34个执行回合共5950条采样保存5个follower位置及5条gear状态。',
        'driver_rad来自getJointState()[0]，是实际角度；planned joints/开口目标则是计划值，不能混用。',
        '分关节/分阶段误差、gear反力均仅采样值；result.max_mimic_error_rad是执行tick期间聚合最大值，但没有最大值发生tick或对应分关节完整状态。',
        '速度没有实测记录。本审计仅用同阶段相邻0.1s实际q差分得到区间平均dq估计，不是瞬时速度，不据此证明无尖峰或计算真实输入功率。',
        '实际电机扭矩、指垫法向力、切向力/位置、闭环铰点几何残差、实际getDynamicsInfo回读均缺失。',
        'gear_constraint_states为转动约束广义反作用扭矩（Nm解释依据既有本机3.2.7源码审查）；不是指尖夹持力，也不是电机实际扭矩。',
        'collision_events.force只保存阈值触发的禁用自碰、非指垫物体接触等；合法指垫接触只转成双侧布尔，数值未落盘。禁用接触事件峰值不等于全部接触的逐tick峰值。',
        '',
        '三、按版本、物体和条件汇总（总max来自执行tick聚合；缺测不能置零）',
        '版本 | 物体 | 条件 | 回合 | 形成抓取 | 禁用碰撞回合 | 执行tick总max_mimic范围rad']
    for r in report['summary_by_version_object_condition']:
        s=r['max_mimic_per_execution_tick_rad'];lines.append(f'{r["version"]} | {r["object_id"]} | {r["condition"]} | {r["episodes"]} | {r["grasp_formed"]} | {r["forbidden_collision"]} | {fmt(s["min"])} .. {fmt(s["max"])}')
    lines+=['','四、阶段比较（以下全部为10Hz采样峰值，不是逐tick峰值）','版本 | 物体 | 条件 | 阶段 | 采样数 | mimic最大rad | 双侧接触采样数 | gear采样最大绝对Nm']
    for r in report['phase_aggregates']:
        if r['phase'] not in ('approach','close','lift','hold'):continue
        reactions=[s['max_abs'] for s in r['sampled_generalized_gear_reaction_Nm'].values() if s['n']]
        lines.append(f'{r["version"]} | {r["object_id"]} | {r["condition"]} | {r["phase"]} | {r["sample_count"]} | {fmt(r["sampled_mimic_error_rad"]["max"])} | {r["sampled_bilateral_count"]} | {fmt(max(reactions) if reactions else None)}')
    lines+=['','逐关节q范围、主从残差、局部gear残差、反力及差分速度估计完整保存在observations.json；均按episode/版本/物体/条件/阶段区分。',
        '局部gear残差按配置qA+ratio*qB计算，不把名义主从残差与真实闭环位置/方向误差等同。',
        '',
        '五、质量、摩擦、惯量与驱动来源',
        '014桃子和017梅子各0.1kg，021球0.04kg；物体质心取模型AABB中心，惯量用同尺寸均匀盒公式，均为工程假设而非GraspNet测量值。014/017碰撞为819/640盒并集，021为320顶点凸包。',
        '机器人URDF来自固定commit官方描述；新增两侧pad各5g是假设。URDF每个link质量、完整惯量张量、inertial origin及joint dynamics已逐项写入JSON。',
        '加载脚本指定URDF_USE_INERTIA_FROM_FILE；但没有各回合运行时getDynamicsInfo回读，不能宣称已测得模拟器实际质量/惯量。',
        '显式运行配置：物体lateralFriction0.6，pad1.0，restitution0；物体linear/angular damping0.04，pad collisionMargin0.0005m。',
        '物体metadata写rolling/spinning friction0.001，但URDF和episode未显式应用这两个值，也无运行回读，因此只能视为静态建议，不能当作已执行摩擦参数。桌面及其他link未显式设置的摩擦/阻尼也未回读。',
        '主驱动位置电机force上限10（转动关节按Nm解释），最大速度配置0.5rad/s；gear maxForce80，ERP0.2；这些是配置上限，不是实际输出或实机额定力。follower电机关闭，未新增独立夹持驱动。',
        '',
        '六、有限诊断必补字段（本审计不执行动力学）']
    for i,line in enumerate(report['missing_fields'],1):lines.append(f'{i}. {line}')
    lines+=['','七、结论边界与证据位置',
        '建议下一有限诊断固定机械模式、单驱动约定和初始条件，先完整观测再验证原因；本文件不授权新回合、参数扫描、扩大力矩或放宽0.03rad。',
        'results/simulation_v52_dynamics_audit/observations.json：全48回合汇总、逐阶段/分关节数据、缺测字段及输入SHA/bytes/mtime。',
        'scripts/sim_v52r2_episode.py:199–253：执行tick监测、合法pad接触布尔化、每24tick降采样；27–31：follower关闭及gear设置；123–125：对象加载与显式动力学设置。',
        'scripts/sim_v52_robot_assets.py:236–240：主驱动位置电机命令，不记录实际电机扭矩。',
        'docs/机器人耦合工程缺口与后续修复建议_v52.txt：既有本机3.2.7反力单位/API源码审查；本轮未重新运行其中任何动力学。',
        f'审计脚本SHA256：{report["source_identity"]["sha256"]}',
        f'只读输入数量：{len(report["input_bindings"])}；审计前后SHA/bytes/mtime一致。',
        '所有历史失败和原协议均保留。工程抓取形成不等于合格成功，现有数据不支持方法层优势声明。']
    DOC.parent.mkdir(parents=True,exist_ok=True)
    with DOC.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')

if __name__=='__main__':run()
