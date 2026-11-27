"""Streaming, read-only analysis of the completed frozen twelve-unit matrix.

Never imports experiment modules or a physics engine. Refuses incomplete matrices
and refuses to replace prior analysis. Preparation does not read telemetry.
"""
from pathlib import Path
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
import sys
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/simulation_v52d/analysis'
DOC=ROOT/'docs/v52d_动力学根因分析.txt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ident(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=h.hexdigest(),bytes=path.stat().st_size)


def norm(v):return float(np.linalg.norm(v))


def fresh_phase():
    return dict(ticks=0,master_torque_max_abs_Nm=0.,master_at_99pct_cap_ticks=0,
        master_cap_streak=0,master_max_cap_streak_ticks=0,master_tracking_max_abs_rad=0.,
        master_tracking_sumsq_rad2=0.,master_position_error_end_rad=None,
        max_mimic_error_rad=0.,mimic_over_003_ticks=0,forbidden_ticks=0,
        contact_force_normal_point_max_N=0.,contact_force_tangent_point_max_N=0.,
        motor_power_max_abs_W=0.,gear_power_max_abs_W=0.,external_power_max_abs_W=0.,
        motor_work_discrete_J=0.,gear_work_discrete_J=0.,external_work_discrete_J=0.,
        gear_rows={},pad_object={},mass_ratio_min=None,mass_ratio_max=None)


def update_gear(stats,g,step):
    key=g['a']+'|'+g['b']
    s=stats.setdefault(key,dict(a=g['a'],b=g['b'],ratio=g['ratio'],ticks=0,
        residual_max_abs_rad=0.,residual_sumsq_rad2=0.,residual_peak_step=None,
        velocity_residual_max_abs_rad_s=0.,reaction_max_abs_Nm=0.,at_99pct_cap_ticks=0,
        reaction_at_residual_peak_Nm=None,work_discrete_J=0.,power_max_abs_W=0.))
    s['ticks']+=1;e=abs(g['e_post_rad'])
    if e>s['residual_max_abs_rad']:
        s.update(residual_max_abs_rad=e,residual_peak_step=step,reaction_at_residual_peak_Nm=g['reaction_Nm'])
    s['residual_sumsq_rad2']+=g['e_post_rad']**2
    s['velocity_residual_max_abs_rad_s']=max(s['velocity_residual_max_abs_rad_s'],abs(g['de_post_rad_s']))
    s['reaction_max_abs_Nm']=max(s['reaction_max_abs_Nm'],abs(g['reaction_Nm']))
    s['at_99pct_cap_ticks']+=abs(g['reaction_Nm'])>=.99*g['cap_Nm']
    s['work_discrete_J']+=g['reaction_Nm']*(g['e_post_rad']-g['e_pre_rad'])
    s['power_max_abs_W']=max(s['power_max_abs_W'],abs(g['reaction_Nm']*g['de_post_rad_s']))


def finish_phase(p,dt):
    p['seconds']=p['ticks']*dt
    p['master_tracking_RMS_rad']=math.sqrt(p.pop('master_tracking_sumsq_rad2')/max(p['ticks'],1))
    p['master_saturation_observed_fraction']=p['master_at_99pct_cap_ticks']/max(p['ticks'],1)
    p['master_max_cap_streak_seconds']=p['master_max_cap_streak_ticks']*dt
    p.pop('master_cap_streak')
    for g in p['gear_rows'].values():
        g['residual_RMS_rad']=math.sqrt(g.pop('residual_sumsq_rad2')/max(g['ticks'],1))
        g['saturation_observed_fraction']=g['at_99pct_cap_ticks']/max(g['ticks'],1)
    return p


def analyze_attempt(row,manifest):
    intent=row['intent'];result=row['result'];directory=ROOT/intent['directory']
    if row['unknown'] or result is None or result['status']!='completed':raise ValueError('Incomplete attempt')
    unit=intent['unit'];dt=unit['time_step_s'];readback=read(directory/'telemetry_readback.json')
    robot=int(readback['constraints'][0]['body_A']);pads=[int(x) for x in readback['collision_shapes']]
    links_names={x['index']:x['value'][12] for x in readback['joint_info_raw']}
    phase_map=collections.defaultdict(fresh_phase);overall=fresh_phase()
    first={k:None for k in ['robot_object_positive_force','pad_object_above_02N','mimic_above_003','forbidden_contact','master_at_99pct_cap','gear_at_99pct_cap']}
    load_check=dict(loaded_ticks=0,load_records=0,max_force_magnitude_error_N=0.,max_pair_force_balance_N=0.,
        max_outward_direction_error_N=0.,max_com_moment_error_Nm=0.,max_material_displacement_error_m=0.,
        max_discrete_work_error_J=0.,max_power_pre_error_W=0.,unexpected_load_ticks=0)
    consistency=dict(max_mimic_aggregate_error=0.,max_gear_position_error=0.,max_gear_velocity_error=0.,
        max_motor_work_error_J=0.,max_gear_work_error_J=0.,max_contact_force_reconstruction_error_N=0.)
    peaks={};steps=0;mass_rows={};mimic_peak=None;final_state=None
    def event(rec):return dict(step=rec['step'],phase=rec['phase'],time_s=rec['time_s'])
    with gzip.open(directory/'telemetry.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            rec=json.loads(line);steps+=1
            if rec.get('capture_error') or rec['step']!=steps:raise ValueError('Missing or failed telemetry tick')
            if not math.isclose(rec['time_s'],steps*dt,rel_tol=0.,abs_tol=1e-9):raise ValueError('Time mismatch')
            phase=rec['phase'];pp=phase_map[phase];groups=[pp,overall]
            js={x['name']:x for x in rec['joints']};master=js[manifest['master_joint']]
            qmaster=master['q_post_rad'];target=master['command']['position_rad'];tracking=target-qmaster
            mastercap=master['command']['force_cap_Nm'];mastersat=abs(master['applied_motor_torque_Nm'])>=.99*mastercap
            mimic=max(abs(js[x['joint']]['q_post_rad']-(qmaster*x['multiplier']+x['offset'])) for x in manifest['mimic_joints'])
            consistency['max_mimic_aggregate_error']=max(consistency['max_mimic_aggregate_error'],abs(mimic-rec['flags']['max_mimic_error_rad']))
            if mimic_peak is None or mimic>mimic_peak['error_rad']:
                mimic_peak=dict(**event(rec),error_rad=mimic,master_actual_torque_Nm=master['applied_motor_torque_Nm'],
                    master_tracking_error_rad=tracking,gears=[{k:g[k] for k in ['a','b','e_post_rad','reaction_Nm']} for g in rec['gears']])
            if mimic>.03 and first['mimic_above_003'] is None:first['mimic_above_003']=event(rec)
            if mastersat and first['master_at_99pct_cap'] is None:first['master_at_99pct_cap']=event(rec)
            if rec['flags']['forbidden_collision'] and first['forbidden_contact'] is None:first['forbidden_contact']=event(rec)
            for j in rec['joints']:
                w=j['applied_motor_torque_Nm']*(j['q_post_rad']-j['q_pre_rad'])
                consistency['max_motor_work_error_J']=max(consistency['max_motor_work_error_J'],abs(w-j['work_discrete_J']))
            for g in rec['gears']:
                ja,jb=js[g['a']],js[g['b']];r=g['ratio']
                ee=ja['q_post_rad']+r*jb['q_post_rad'];dd=ja['dq_post_rad_s']+r*jb['dq_post_rad_s']
                consistency['max_gear_position_error']=max(consistency['max_gear_position_error'],abs(ee-g['e_post_rad']))
                consistency['max_gear_velocity_error']=max(consistency['max_gear_velocity_error'],abs(dd-g['de_post_rad_s']))
                consistency['max_gear_work_error_J']=max(consistency['max_gear_work_error_J'],abs(g['reaction_Nm']*(g['e_post_rad']-g['e_pre_rad'])-g['work_discrete_J']))
                if abs(g['reaction_Nm'])>=.99*g['cap_Nm'] and first['gear_at_99pct_cap'] is None:first['gear_at_99pct_cap']=event(rec)
                for p in groups:update_gear(p['gear_rows'],g,steps)
            linked={x['index']:x for x in rec['links']};pad_sums={pad:dict(normal=0.,tangent_sum=0.,force=np.zeros(3),moment=np.zeros(3),contacts=0) for pad in pads}
            normal_peak=tangent_peak=0.
            for c in rec['contacts']:
                F=c['normal_force_N']*np.array(c['normal_on_B_world'])+c['lateral_force_1_N']*np.array(c['lateral_direction_1_world'])+c['lateral_force_2_N']*np.array(c['lateral_direction_2_world'])
                consistency['max_contact_force_reconstruction_error_N']=max(consistency['max_contact_force_reconstruction_error_N'],norm(F-np.array(c['force_on_A_world_N'])))
                if c['category']!='robot_object':continue
                if c['normal_force_N']>0 and first['robot_object_positive_force'] is None:first['robot_object_positive_force']=event(rec)
                pad=c['link_A'] if c['body_A']==robot else c['link_B']
                if pad not in pad_sums:continue
                if c['normal_force_N']>.2 and first['pad_object_above_02N'] is None:first['pad_object_above_02N']=event(rec)
                point=np.array(c['point_A_world_m'] if c['body_A']==robot else c['point_B_world_m']);F=F if c['body_A']==robot else -F
                tang=math.hypot(c['lateral_force_1_N'],c['lateral_force_2_N'])
                normal_peak=max(normal_peak,c['normal_force_N']);tangent_peak=max(tangent_peak,tang)
                x=pad_sums[pad];x['normal']+=c['normal_force_N'];x['tangent_sum']+=tang;x['force']+=F
                x['moment']+=np.cross(point-np.array(linked[pad]['com_world_m']),F);x['contacts']+=1
            for p in groups:
                p['ticks']+=1;p['master_torque_max_abs_Nm']=max(p['master_torque_max_abs_Nm'],abs(master['applied_motor_torque_Nm']))
                p['master_at_99pct_cap_ticks']+=mastersat;p['master_cap_streak']=p['master_cap_streak']+1 if mastersat else 0
                p['master_max_cap_streak_ticks']=max(p['master_max_cap_streak_ticks'],p['master_cap_streak'])
                p['master_tracking_max_abs_rad']=max(p['master_tracking_max_abs_rad'],abs(tracking))
                p['master_tracking_sumsq_rad2']+=tracking**2;p['master_position_error_end_rad']=tracking
                p['max_mimic_error_rad']=max(p['max_mimic_error_rad'],mimic);p['mimic_over_003_ticks']+=mimic>.03
                p['forbidden_ticks']+=rec['flags']['forbidden_collision']
                p['contact_force_normal_point_max_N']=max(p['contact_force_normal_point_max_N'],normal_peak)
                p['contact_force_tangent_point_max_N']=max(p['contact_force_tangent_point_max_N'],tangent_peak)
                for kind in ['motor','gear','external']:
                    p[kind+'_power_max_abs_W']=max(p[kind+'_power_max_abs_W'],abs(rec['power_work'][kind+'_power_post_W']))
                    p[kind+'_work_discrete_J']+=rec['power_work'][kind+'_work_discrete_J']
                for pad,x in pad_sums.items():
                    s=p['pad_object'].setdefault(str(pad),dict(link_name=links_names[pad],ticks_with_contact=0,
                        normal_sum_peak_N=0.,tangent_magnitude_sum_peak_N=0.,resultant_force_peak_N=0.,resultant_moment_peak_Nm=0.,normal_sum_time_integral_Ns=0.))
                    s['ticks_with_contact']+=x['contacts']>0;s['normal_sum_peak_N']=max(s['normal_sum_peak_N'],x['normal'])
                    s['tangent_magnitude_sum_peak_N']=max(s['tangent_magnitude_sum_peak_N'],x['tangent_sum'])
                    s['resultant_force_peak_N']=max(s['resultant_force_peak_N'],norm(x['force']))
                    s['resultant_moment_peak_Nm']=max(s['resultant_moment_peak_Nm'],norm(x['moment']))
                    s['normal_sum_time_integral_Ns']+=x['normal']*dt
                vals=[x['d_full_over_d_source'] for x in rec['mass_matrix']['gear_rows']]
                p['mass_ratio_min']=min(vals) if p['mass_ratio_min'] is None else min(p['mass_ratio_min'],min(vals))
                p['mass_ratio_max']=max(vals) if p['mass_ratio_max'] is None else max(p['mass_ratio_max'],max(vals))
            for mr in rec['mass_matrix']['gear_rows']:
                key=mr['a']+'|'+mr['b'];m=mass_rows.setdefault(key,dict(min=None,max=None))
                ratio=mr['d_full_over_d_source'];m['min']=ratio if m['min'] is None else min(m['min'],ratio);m['max']=ratio if m['max'] is None else max(m['max'],ratio)
            loads=rec['external_loads'];expected=unit['scenario']=='L' and phase in ('close','lift','hold')
            if len(loads)!=(2 if expected else 0):load_check['unexpected_load_ticks']+=1
            if loads:
                load_check['loaded_ticks']+=1;load_check['load_records']+=len(loads)
                bypad={x['link_index']:x for x in loads}
                if len(loads)==2 and all(pad in bypad for pad in pads):
                    v=np.array(bypad[pads[1]]['point_pre_world_m'])-np.array(bypad[pads[0]]['point_pre_world_m']);e=v/norm(v)
                    load_check['max_pair_force_balance_N']=max(load_check['max_pair_force_balance_N'],norm(np.array(loads[0]['force_pre_world_N'])+np.array(loads[1]['force_pre_world_N'])))
                    for sign,pad in [(-1,pads[0]),(1,pads[1])]:load_check['max_outward_direction_error_N']=max(load_check['max_outward_direction_error_N'],norm(np.array(bypad[pad]['force_pre_world_N'])-sign*e))
                for x in loads:
                    F=np.array(x['force_pre_world_N']);point=np.array(x['point_pre_world_m']);com=np.array(x['com_pre_world_m'])
                    dx=np.array(x['same_material_point_post_world_m'])-point
                    load_check['max_force_magnitude_error_N']=max(load_check['max_force_magnitude_error_N'],abs(norm(F)-1))
                    load_check['max_com_moment_error_Nm']=max(load_check['max_com_moment_error_Nm'],norm(np.cross(point-com,F)-np.array(x['torque_about_com_pre_Nm'])))
                    load_check['max_material_displacement_error_m']=max(load_check['max_material_displacement_error_m'],norm(dx-np.array(x['material_point_displacement_world_m'])))
                    load_check['max_discrete_work_error_J']=max(load_check['max_discrete_work_error_J'],abs(float(F@dx)-x['work_discrete_J']))
                    load_check['max_power_pre_error_W']=max(load_check['max_power_pre_error_W'],abs(float(F@np.array(x['point_velocity_pre_world_m_s']))-x['power_pre_W']))
            final_state=rec['object_state']
    if steps!=unit['maximum_step_count'] or steps!=result['actual_physics_steps']:raise ValueError('Step count mismatch')
    if max(consistency.values())>1e-8 or load_check['unexpected_load_ticks']:raise ValueError('Telemetry reconstruction disagreement')
    details=dict(unit=unit,source_directory=intent['directory'],samples=steps,
        overall=finish_phase(overall,dt),phases={k:finish_phase(v,dt) for k,v in phase_map.items()},
        first_events=first,mimic_peak=mimic_peak,mass_ratio_by_gear=mass_rows,
        load_checks=load_check,reconstruction_checks=consistency,final_object_state=final_state,
        original_contact_gate=dict(grasp_success=result.get('grasp_success'),dynamic_coupling_passed=result.get('dynamic_coupling_passed'),
            max_mimic_error_rad=result.get('max_mimic_error_rad'),forbidden_collision=result.get('forbidden_collision'),
            max_lift_m=result.get('max_lift_m'),final_continuous_hold_seconds=result.get('final_continuous_hold_seconds')),
        inputs=[ident(directory/n) for n in ['result.json','telemetry_readback.json','telemetry.jsonl.gz','telemetry_summary.json']])
    return details


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report');args=parser.parse_args()
    assert Path.cwd().resolve()==ROOT
    if OUT.exists() or DOC.exists():raise FileExistsError('Preserve prior analysis')
    state=read(ROOT/'runtime/pipeline/simulation_v52_dynamics_run.json')
    if state.get('phase') not in (None,'completed_stopped_for_review') and state.get('status')!='completed_stopped_for_review':
        raise RuntimeError('Controller has not stopped after completion')
    report_path=ROOT/(args.report or state['report']);report=read(report_path)
    if not report['diagnostic_complete'] or report['charged_attempts']!=12 or report['unknown_attempts'] or not report['exact_total_steps_known']:
        raise RuntimeError('Require complete twelve-return matrix with no unknown calls')
    manifest=read(ROOT/'data/simulation_v52/robot/robot_assets.json')
    units=[]
    for row in sorted(report['records'],key=lambda x:x['intent']['unit']['id']):
        units.append(analyze_attempt(row,manifest))
        print('Analyzed '+units[-1]['unit']['id'],flush=True)
    if sum(x['samples'] for x in units)!=79920:raise ValueError('Expected all approved returned steps')
    comparisons=[]
    lookup={(x['unit']['scenario'],x['unit']['time_step_s'],x['unit']['solver_iterations']):x for x in units}
    for scenario in ['N','L','C']:
        for dt in [1/240,1/480]:
            a=lookup[(scenario,dt,150)];b=lookup[(scenario,dt,300)]
            comparisons.append(dict(kind='iterations_at_fixed_dt',scenario=scenario,time_step_s=dt,
                units=[a['unit']['id'],b['unit']['id']],max_mimic_150=a['overall']['max_mimic_error_rad'],
                max_mimic_300=b['overall']['max_mimic_error_rad'],
                ratio_300_over_150=b['overall']['max_mimic_error_rad']/max(a['overall']['max_mimic_error_rad'],1e-30)))
        for it in [150,300]:
            a=lookup[(scenario,1/240,it)];b=lookup[(scenario,1/480,it)]
            comparisons.append(dict(kind='dt_and_controller_discretization',scenario=scenario,iterations=it,
                units=[a['unit']['id'],b['unit']['id']],max_mimic_dt240=a['overall']['max_mimic_error_rad'],
                max_mimic_dt480=b['overall']['max_mimic_error_rad'],
                ratio_480_over_240=b['overall']['max_mimic_error_rad']/max(a['overall']['max_mimic_error_rad'],1e-30)))
    N=[x for x in units if x['unit']['scenario']=='N'];L=[x for x in units if x['unit']['scenario']=='L'];C=[x for x in units if x['unit']['scenario']=='C']
    findings=dict(N_max_mimic_rad=max(x['overall']['max_mimic_error_rad'] for x in N),
        L_max_mimic_rad=max(x['overall']['max_mimic_error_rad'] for x in L),
        C_max_mimic_rad=max(x['overall']['max_mimic_error_rad'] for x in C),
        C_min_mimic_rad=min(x['overall']['max_mimic_error_rad'] for x in C),
        master_saturation_ticks_total=sum(x['overall']['master_at_99pct_cap_ticks'] for x in units),
        gear_saturation_row_ticks_total=sum(g['at_99pct_cap_ticks'] for x in units for g in x['overall']['gear_rows'].values()),
        mass_ratio_min=min(x['overall']['mass_ratio_min'] for x in units),
        mass_ratio_max=max(x['overall']['mass_ratio_max'] for x in units),
        all_NL_under_old_mimic_gate=all(x['overall']['max_mimic_error_rad']<=.03 for x in N+L),
        all_C_fail_old_mimic_gate=all(x['overall']['max_mimic_error_rad']>.03 for x in C))
    output=dict(schema='v52d_readonly_dynamics_analysis_v1',at=datetime.datetime.now().astimezone().isoformat(),
        new_physics_steps=0,source_report=ident(report_path),source_script=ident(__file__),
        analyzed_returned_steps=79920,original_48_unchanged=True,no_experiment_result_reclassification=True,
        findings=findings,units=units,comparisons=comparisons,
        limits=['One run per cell; exposed diagnostic scene, not generalization or success-rate evidence',
                'Full-q mass ratios are stored actual calculateMassMatrix telemetry; independent final reviewer separately verifies matrix reconstruction',
                'Motor and gear cap observation threshold .99 is a diagnostic convention, not a new success gate',
                'Per-step forces are solver-reported average reactions; work/power formulas are diagnostic estimates, not continuous exact work',
                'Time-step comparison also changes ERP/dt, kp/dt and impulse caps; cannot identify a unique solver cause',
                'No source-level engine bug is established by these runs; no automatic M1 implementation or admission authorized'])
    OUT.mkdir(parents=True)
    with (OUT/'dynamics_analysis.json').open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    lines=['v52d 动力学根因分析（只读）','',
        '本分析只读取已完成的12格原始逐tick日志；新增物理步0，不改变原48回合或本12回合结果。',
        '每格单次，作用是有限故障诊断，不代表成功率、泛化结论或新的工程准入。','',
        '一、12格完整结果与驱动力观察',
        '单元 | 工况 | dt | 迭代 | max mimic(rad) | master最大(Nm) | master≥99%cap步 | gear最大(Nm) | gear≥99%cap行步']
    for x in units:
        u=x['unit'];p=x['overall'];gs=p['gear_rows'].values()
        lines.append(f"{u['id']} | {u['scenario']} | 1/{round(1/u['time_step_s'])} | {u['solver_iterations']} | {p['max_mimic_error_rad']:.9g} | {p['master_torque_max_abs_Nm']:.9g} | {p['master_at_99pct_cap_ticks']} | {max(g['reaction_max_abs_Nm'] for g in gs):.9g} | {sum(g['at_99pct_cap_ticks'] for g in gs)}")
    lines += ['',f"N最大误差 {findings['N_max_mimic_rad']:.9g} rad；L最大 {findings['L_max_mimic_rad']:.9g} rad。",
        f"C最大误差范围 {findings['C_min_mimic_rad']:.9g}–{findings['C_max_mimic_rad']:.9g} rad；原门槛保持0.03 rad。",
        '≥99% cap是审计约定，用于观察限幅，不改变任何原实验接受门槛。','',
        '二、接触、失配与驱动的时序（完整分阶段/5行细项见JSON）']
    for x in C:
        lines.append(x['unit']['id']+'：'+json.dumps(x['first_events'],ensure_ascii=False))
        for phase in ['close','lift','hold']:
            p=x['phases'][phase]
            lines.append(f"  {phase}: master torque max={p['master_torque_max_abs_Nm']:.6g}Nm, cap ticks={p['master_at_99pct_cap_ticks']}, tracking end={p['master_position_error_end_rad']:.6g}rad, mimic max={p['max_mimic_error_rad']:.6g}rad; pad-object={json.dumps(p['pad_object'],ensure_ascii=False)}")
    lines += ['', '三、固定dt迭代敏感性与dt/控制离散敏感性']
    for c in comparisons:
        if c['scenario']=='C':lines.append(json.dumps(c,ensure_ascii=False))
    lines += ['', '四、加载惯量、1N外载及功率/功',
        f"实际full-q惯量遥测的d_full/d_src总体范围为 {findings['mass_ratio_min']:.9g}–{findings['mass_ratio_max']:.9g}。", 
        '该比例描述当前载入模型广义惯量的行归一化差异，不证明接触联合求解的唯一根因。',
        'L的力向、世界施力点、COM力矩、材料点位移与Fpre·dx逐tick复核；统计见每unit.load_checks。',
        '功率tau_post*dq_post；motor离散功tau_post*(qpost-qpre)；gear离散功reaction_post*(e_post-e_pre)；外载离散功Fpre·材料点位移。均为诊断估计。']
    for x in units:
        p=x['overall'];lines.append(f"{x['unit']['id']}: motor={p['motor_work_discrete_J']:.9g}J, gear={p['gear_work_discrete_J']:.9g}J, external={p['external_work_discrete_J']:.9g}J; load={json.dumps(x['load_checks'],ensure_ascii=False)}")
    lines += ['', '五、根因边界和M1建议']
    if findings['all_NL_under_old_mimic_gate'] and findings['all_C_fail_old_mimic_gate']:
        lines.append('本有限矩阵中空夹及每侧1N已知外载均保持原耦合门槛，四个球体接触格均未保持；支持重点排查物体接触/摩擦与单驱动及约束的联合响应。1N外载通过不证明所有实际抓取负载都可承受。')
    else:
        lines.append('各工况是否超限应按上述逐格结果解释；不能把部分改善概括成全部受载问题已解决。')
    if findings['gear_saturation_row_ticks_total']==0:
        lines.append('本12格逐tick未观察到gear接近80Nm上限；直接增大gear力矩缺乏依据。')
    if findings['master_saturation_ticks_total']==0:
        lines.append('本12格逐tick未观察到master接近10Nm上限；主驱动力上限不足没有直接限幅证据。位置跟踪差本身不能等同力矩饱和。')
    else:
        lines.append('master存在接近10Nm上限的记录；必须按工况、阶段和持续时间区分，仅此不能证明唯一根因为实机驱动力不足，更不能自动加力。')
    lines += ['改变迭代/时间步的结果只能提供对应敏感性，不能宣布PyBullet bug或证明单一交叉项是故障根因。',
        '建议在研究审核后考虑M1有来源的平行闭环表示：保留同一URDF几何/质量/摩擦、单master、左右同步和原0.03门槛；以两侧各两个独立闭环位置条件替代四条同侧gear，不叠加、不新增五个从关节驱动。',
        '实现前先核实际link/COM锚点转换、闭环行秩和后端依赖行处理；静态等价不是受载有效性证明。M1只能作为待审下一步，必须另有有限预算和停止点，不能从本分析自动进入新48或R4。',
        '原控制接触前自碰需单独保留和排查；不得通过扩大白名单、放宽阈值、把漂移当真实自适应或忽略旧失败取得通过。','',
        '数据定位：results/simulation_v52d/analysis/dynamics_analysis.json，内含各unit/phase/gear的统计、峰值时刻及全部输入哈希。',
        '本分析完成后停止；不启动训练、物理或新的工程验收。']
    with DOC.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(OUT),doc=str(DOC),findings=findings,new_physics_steps=0),ensure_ascii=False))


if __name__=='__main__':main()
