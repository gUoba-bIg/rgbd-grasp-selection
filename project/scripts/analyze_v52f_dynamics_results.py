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
import os
import sys
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_thread_variable]='2'
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/simulation_v52f/analysis'
DOC=ROOT/'docs/v52f_修复后M0逐格比较分析.txt'


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


def condition_from_readback(directory):
    rb=read(directory/'telemetry_readback.json')
    robot=rb['constraints'][0]['body_A']
    names={x['index']:x['value'][12] for x in rb['joint_info_raw']}
    out={}
    for row in rb['dynamics']:
        if row['body_id']!=robot:continue
        name='BASE' if row['link_index']==-1 else names[row['link_index']]
        if name in ('BASE','rq_left_inner_finger_pad','rq_right_inner_finger_pad'):
            out[name]=dict(lateral_friction=row['lateral_friction'],collision_margin_m=row['collision_margin_m'],restitution=row['restitution'])
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report');parser.add_argument('--review',required=True)
    parser.add_argument('--config',default='configs/simulation_v52fr1.json');args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Explicit project workdir required')
    if OUT.exists() or DOC.exists():raise FileExistsError('Prior analysis must remain unchanged')
    state=read(ROOT/'runtime/pipeline/simulation_v52f.json')
    if state.get('phase') not in ('completed_stopped_for_review','failure_or_unknown_preserved_stopped','budget_exhausted_stopped',
        'insufficient_disk_stopped','output_budget_stopped','stop_requested','partial_stopped_for_review'):
        raise RuntimeError('Controller must have stopped before analysis')
    report_path=ROOT/(args.report or state['report']);report=read(report_path)
    review_path=(ROOT/args.review).resolve();review_path.relative_to(ROOT);review=read(review_path)
    if review.get('mode')!='final':raise RuntimeError('Require an existing independent final read-only review, including partial/failed outcomes')
    config_path=(ROOT/args.config).resolve();config_path.relative_to(ROOT)
    config_stat=config_path.stat();config_identity=dict(sha256=ident(config_path)['sha256'],bytes=config_stat.st_size,mtime_ns=config_stat.st_mtime_ns)
    if review['config_identity']!=config_identity:raise RuntimeError('Review is not bound to the supplied execution config')
    if report['charged_attempts']>12:raise RuntimeError('Attempt count exceeds authorization')
    manifest=read(ROOT/'data/simulation_v52/robot/robot_assets.json')
    prior_path=ROOT/'results/simulation_v52d/analysis/dynamics_analysis.json';prior=read(prior_path)
    prior_units={x['unit']['id']:x for x in prior['units']};units=[];paired=[];availability=[]
    records={row['intent']['unit']['id']:row for row in report['records']}
    combinations=read(config_path)['combinations']
    for planned in combinations:
        row=records.get(planned['id'])
        if row is None:
            availability.append(dict(unit=planned,status='not_started',actual_returned_steps=None,reason='No charged intent; not imputed as zero measured execution'))
            continue
        result=row.get('result')
        if row.get('unknown') or result is None:
            availability.append(dict(unit=planned,status='unknown',actual_returned_steps=None,reason='No bound known result; retain unknown, do not infer intended steps'))
            continue
        if result.get('status')!='completed' or not result.get('completion') or result.get('unknown_step_call') is not None:
            availability.append(dict(unit=planned,status='failed_or_incomplete_preserved',actual_returned_steps=result.get('actual_physics_steps'),
                step_calls_started=result.get('physics_step_calls_started'),step_calls_returned=result.get('physics_step_calls_returned'),
                unknown_step_call=result.get('unknown_step_call'),exception=result.get('exception'),
                source_result=ident(ROOT/row['intent']['directory']/'result.json')))
            continue
        unit_review=next((x for x in review.get('independently_recomputed_units',[]) if x['unit']['id']==planned['id']),None)
        if not review.get('review_passed') or unit_review is None or unit_review.get('additional_integrity_failures'):
            availability.append(dict(unit=planned,status='completed_but_review_integrity_failed',actual_returned_steps=result['actual_physics_steps']))
            continue
        new=analyze_attempt(row,manifest);units.append(new);old=prior_units[new['unit']['id']]
        availability.append(dict(unit=planned,status='completed_analyzed',actual_returned_steps=new['samples']))
        if old['unit']!=new['unit']:raise RuntimeError('Not the same registered diagnostic cell')
        # Historical cache is used only when all source identities still match.
        for entry in old['inputs']:
            actual=ident(ROOT/entry['path'])
            if actual!=entry:raise RuntimeError('Historical analyzed input identity changed')
        a,b=old['overall'],new['overall']
        metrics={name:dict(old=a[name],repaired=b[name],delta_repaired_minus_old=b[name]-a[name]) for name in [
            'max_mimic_error_rad','master_torque_max_abs_Nm','master_at_99pct_cap_ticks','forbidden_ticks',
            'contact_force_normal_point_max_N','contact_force_tangent_point_max_N','motor_work_discrete_J','gear_work_discrete_J','external_work_discrete_J']}
        paired.append(dict(unit=new['unit'],old_actual_conditions=condition_from_readback(ROOT/old['source_directory']),
            repaired_actual_conditions=condition_from_readback(ROOT/new['source_directory']),metrics=metrics,
            old_contact_gate=old['original_contact_gate'],repaired_contact_gate=new['original_contact_gate'],
            old_first_events=old['first_events'],repaired_first_events=new['first_events']))
        print('Analyzed repaired '+new['unit']['id'],flush=True)
    analyzed_steps=sum(x['samples'] for x in units)
    if len(units)==12 and analyzed_steps!=79920:raise ValueError('Complete twelve-cell step total is inconsistent')
    expected_old={'BASE':dict(lateral_friction=1.,collision_margin_m=.0005,restitution=0.),
        **{n:dict(lateral_friction=.5,collision_margin_m=.001,restitution=0.) for n in manifest['pad_links']}}
    expected_new={'BASE':dict(lateral_friction=.5,collision_margin_m=.001,restitution=0.),
        **{n:dict(lateral_friction=1.,collision_margin_m=.0005,restitution=0.) for n in manifest['pad_links']}}
    if any(x['old_actual_conditions']!=expected_old or x['repaired_actual_conditions']!=expected_new for x in paired):
        raise RuntimeError('Actual old/new contact conditions differ from declared comparison')
    lookup={(x['unit']['scenario'],x['unit']['time_step_s'],x['unit']['solver_iterations']):x for x in units};sensitivity=[]
    for scenario in 'NLC':
        for dt in (1/240,1/480):
            if any((scenario,dt,i) not in lookup for i in (150,300)):continue
            a,b=(lookup[(scenario,dt,i)] for i in (150,300))
            sensitivity.append(dict(kind='iterations_at_fixed_dt',scenario=scenario,dt=dt,units=[a['unit']['id'],b['unit']['id']],
                mimic_150=a['overall']['max_mimic_error_rad'],mimic_300=b['overall']['max_mimic_error_rad']))
        for it in (150,300):
            if any((scenario,dt,it) not in lookup for dt in (1/240,1/480)):continue
            a,b=(lookup[(scenario,dt,it)] for dt in (1/240,1/480))
            sensitivity.append(dict(kind='dt_and_controller_discretization',scenario=scenario,iterations=it,units=[a['unit']['id'],b['unit']['id']],
                mimic_240=a['overall']['max_mimic_error_rad'],mimic_480=b['overall']['max_mimic_error_rad']))
    limits=['Old pad .5/.001 and base 1/.0005 differ jointly from repaired pad1/.0005 and base.5/.001; not a one-factor friction or margin ablation.',
        'Old v52d parameter consistency remains failed. Old diagnostics are retained observations, not equal-condition confirmed controls.',
        'One run per diagnostic cell, fixed exposed ball scene: no generalization or population success-rate claim.',
        'Parameter integrity, post-settle coupling/contact gates, and overall project scientific conclusions are separate.',
        '99% torque cap is a diagnostic convention only. Absence above99% cannot be called far from the cap.',
        'Discrete power/work and reported contact forces are diagnostic estimates; no exact continuous energy claim.',
        'Changing dt changes ERP/dt, kp/dt and impulse caps. Iteration contrasts must be interpreted at fixed dt.',
        'No automatic M1, new48, R4, formal720, ablation240 or other experiment. Stop after reporting.']
    output=dict(schema='v52f_repaired_M0_comparison_v1',source_script=ident(__file__),source_config=ident(config_path),source_report=ident(report_path),
        source_independent_review=ident(review_path),source_historical_analysis=ident(prior_path),new_physics_steps=0,new_connections=0,
        independent_integrity_passed=review.get('review_passed'),all_twelve_completed=review.get('all_12_completed'),
        analyzed_returned_steps=analyzed_steps,availability=availability,old_results_reclassified=False,
        units=units,paired_cells=paired,sensitivity=sensitivity,limits=limits)
    OUT.mkdir(parents=True)
    with (OUT/'dynamics_comparison.json').open('x',encoding='utf8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['v52f 修复后M0有限逐格比较（只读）','',
        '本分析复用旧v52d汇总及其已绑定原始证据，仅对已完整返回且经过独立数值复核的新格读取全部逐步日志。新增连接和物理步均为0。',
        '旧条件：pad摩擦.5、margin .001m，base摩擦1、margin .0005m。新条件：pad摩擦1、margin .0005m，base摩擦.5、margin .001m。',
        '上述接触参数共同改变，不能归因为单一摩擦或margin消融；旧v52d参数验收失败结论保持。',
        f"本次完整分析{len(units)}/12格、{analyzed_steps}已返回步；完整性独审结论={review.get('review_passed')}。未运行、未知、失败或未完成单独保留，不补成成功或0测量。",'',
        '单元 | 工况 | 原max mimic(rad) | 新max mimic(rad) | 新master峰值(Nm) | 新master≥99%cap步 | 新gear峰值(Nm) | 新C抓取判据']
    for x,pair in zip(units,paired):
        u=x['unit'];s=x['overall'];gear=max(v['reaction_max_abs_Nm'] for v in s['gear_rows'].values())
        lines.append(f"{u['id']} | {u['scenario']} | {pair['metrics']['max_mimic_error_rad']['old']:.9g} | {s['max_mimic_error_rad']:.9g} | {s['master_torque_max_abs_Nm']:.9g} | {s['master_at_99pct_cap_ticks']} | {gear:.9g} | {x['original_contact_gate']['grasp_success']}")
    for item in availability:
        if item['status']!='completed_analyzed':lines.append(item['unit']['id']+'：'+json.dumps(item,ensure_ascii=False))
    lines+=['','原0.03rad门槛、10Nm/80Nm、动作和评价窗口均保持；数值改善不自动等于工程准入。N/L不定义抓取成功。',
        '逐阶段跟踪、全部5条gear残差/反力、接触峰值、首接触与首次超限、惯量交叉项、外载和功统计见JSON；不删去异常或不利结果。','']
    lines+=limits
    with DOC.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(OUT),doc=str(DOC),new_connections=0,new_physics_steps=0)))


if __name__=='__main__':main()

