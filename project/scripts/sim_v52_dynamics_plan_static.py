"""Write a proposed, NOT executable/authorized, twelve-unit diagnostic plan."""
from sim_v52_common import ROOT, read, write, identity, relative, now


def main():
    out = ROOT/'results/simulation_v52_dynamics_audit'
    cfg_path = ROOT/'configs/simulation_v52r2.json'
    cfg = read(cfg_path)
    source_path = ROOT/'results/simulation_v52/episodes/005_slot_04_4f097762/result.json'
    source = read(source_path)
    # Deliberately export action fields only: no evaluation pose, label, or success flag.
    action = {key:source[key] for key in (
        'candidate_world', 'goal_tool_world', 'requested_opening_m',
        'open_driver_rad', 'estimated_contact_driver_rad')}
    action['planning_joint_targets'] = [dict(name=x['name'], joints=x['joints']) for x in source['planning']]
    action['source_identity'] = identity(source_path)
    action['scope'] = 'Cached observation-derived engineering action; not P3 or a GT-selected grasp.'
    action['not_authorized_for_execution'] = True
    action_path = out/'proposed_diagnostic_action.json'
    write(action_path, action, once=True)
    scenarios = [
        dict(id='N', name='no_object_reference', object_present=False,
             external_load=False, reason='Matched no-contact reference with complete telemetry; old miss cases already demonstrated free closure.'),
        dict(id='L', name='no_object_known_outward_pad_load', object_present=False,
             external_load=True, load_force_each_pad_N=1.0,
             load_phases=['close','lift','hold'],
             load_definition='Equal opposite forces at the two fixed pad-collision-box centers. e=(p_right-p_left)/norm. F_left=-1N*e; F_right=+1N*e. Use robot FK only, every tick. No follower motor and no desired follower joint state.',
             reason='Known mechanical calibration load without an object/contact/friction solver; record external work separately. Not a calibrated real-object load or grasp success.'),
        dict(id='C', name='preserved_slot4_ball_contact', object_present=True,
             external_load=False, object_id='021', object_mass_config_kg=.04,
             scene_xy_m=[.48,.06], engineering_yaw_rad=0.0,
             object_urdf=cfg['objects']['021']['urdf'],
             reason='Reproduce one exposed failed contact case with cached action; not an independent generalization or efficacy test.')]
    combinations = []
    for dt in (1/240,1/480):
        for iterations in (150,300):
            for scenario in scenarios:
                combinations.append(dict(id=f'DYN{len(combinations)+1:03d}', scenario=scenario['id'],
                    time_step_s=dt, solver_iterations=iterations, repetitions=1,
                    maximum_sim_seconds=18.5, maximum_step_count=round(18.5/dt)))
    assert len(combinations)==12 and sum(x['maximum_step_count'] for x in combinations)==79920
    plan = dict(at=now(), status='proposed_waiting_research_review',
        execution_authorized=False, implemented_dynamics_runner=False,
        selected_model='unchanged simulation_v52r2 URDF and five-gear tree',
        model_replacement_included=False,
        current_new_physics_steps=0, current_new_dynamics_runs=0,
        earlier_v52_grasp_attempts=48, requested_new_diagnostic_units=12,
        requested_no_object_units=8, requested_object_contact_units=4,
        cumulative_if_fully_approved_and_run=dict(all_v52_and_new_runs=60, object_grasp_attempts=52),
        scenarios=scenarios, combinations=combinations,
        order='Listed order; every started unit charged, no retry or best-case selection. Interrupted/unreturned run remains unknown and blocks automatic continuation.',
        fixed=dict(original_physics=cfg['physics'], original_motion=cfg['motion'],
            original_criteria=cfg['criteria'], gripper_ERP=.2,
            master_position_gain=.1, master_velocity_gain=1.0,
            master_gains_source='Arguments omitted by current helper; defaults kp=.1/kd=1 verified in installed PyBullet3.2.7 source. Keep original call semantics.',
            arm_position_gain=.08, arm_velocity_gain=1.0,
            master_close_max_velocity_rad_s=.5,
            source_initialization_and_motor_schedule='Same v52r2 initialization/feedback interpolation/timing; use cached action targets. No object teleport after initialization.',
            initial_settling_seconds=1.0, settling_included_in_time_steps_and_logs=True,
            collision_allowlist_unchanged=True, force_or_friction_increase=False,
            mimic_error_gate_rad=.03, action_input=relative(action_path)),
        telemetry_each_tick=[
            'All six arm and six gripper joint q,dq, measured applied motor torque, target, gains, limits and saturation.',
            'Each gear residual qA+ratio*qB, residual derivative, generalized reaction in Nm, cap saturation and time of maximum.',
            'Closed-loop anchor position/axis residual using audited link-frame/COM mapping.',
            'Left/right pad contact positions/normals/normal and tangential forces; sum force and wrench; table and forbidden contacts separately.',
            'Object pose/linear/angular velocity and support when present; absent fields null, not zero or invented.',
            'Actuator power/energy, calibration-load power/energy, and constraint work estimates with units and limitations.',
            'Mass matrix at actual measured full q, d_full/d_src per gear row; cross-check readback inertia and frame conventions.',
            'Phase, step, physical time, flags and all failures, including initial settling; no downsampling of audit data.'],
        readback_before_first_step=[
            'Package/wheel/version/source/URDF hashes; actual world dt/substeps/iterations/ERP/solver fields where exposed.',
            'getDynamicsInfo for every link/object: mass, local inertial transform, inertia, friction/damping/margins where exposed. Unsupported fields remain unknown.',
            'Constraint info for all five gears, master/follower drive settings, joint axes/limits, collision filter configuration.'],
        interpretation=[
            'At fixed dt, compare 150/300 iterations; this is the iteration-sensitivity comparison.',
            'At changed dt, fixed dimensionless ERP and position gains change ERP/dt and kp/dt. Interpret as time-discretization plus controller/constraint sensitivity, NOT pure convergence or a unique cause.',
            'N/L/C match nominal arm trajectory and initialization; L removes object contact and known object inertia/friction but intentionally introduces one external calibration load.',
            'No observed gear saturation in old snapshots does not rule out transient saturation or master torque saturation.',
            'Every combination is a single deterministic diagnostic run. No success-rate estimate, statistical significance, proof of stability or generalization.',
            'Passing any cell does not authorize engineering admission or R4; keep the original .03rad and collision/lift/hold gates for applicable contact checks.'],
        resources=dict(concurrent_workers=1, CPU_thread_limit=2, GPU_compute=False,
            new_training=0, new_model_forwards=0,
            per_run_wall_cap_s=120, total_worker_wall_cap_s=1440,
            total_session_wall_cap_s=1800, maximum_sim_time_s=222,
            maximum_step_count=79920, tree_RSS_cap_bytes=6*1024**3,
            output_cap_bytes=2*1024**3, minimum_free_disk_bytes=20*1024**3,
            meaning='Hard caps, not predicted completion times. All startup/readback/logging time charged; terminate on resource failure and preserve evidence.'),
        stop=dict(after_12_or_earlier_block=True, no_new_48_admission=True,
            no_automatic_repair_version=True, R4_72=False, formal_720=False, ablation_240=False,
            require_review_before_any_dynamics=True, require_review_after_diagnostics=True),
        bindings={relative(p):identity(p) for p in [cfg_path, source_path,
            ROOT/'scripts/sim_v52r2_episode.py', ROOT/'scripts/sim_v52_robot_assets.py',
            ROOT/cfg['robot_urdf'], ROOT/cfg['objects']['021']['urdf'],
            out/'mechanism.json', out/'observations.json', action_path]})
    # Master gain is bound to the real command helper, not inferred from a paper.
    code=(ROOT/'scripts/sim_v52_robot_assets.py').read_text(encoding='utf-8')
    if 'targetPosition=q,force=force,maxVelocity=velocity)' not in code:
        raise RuntimeError('Verify master call semantics against actual helper before publishing a plan')
    write(out/'proposed_diagnostic_plan.json',plan,once=True)
    print('Proposed 12-unit plan written. Execution remains unauthorized; no simulator loaded.')


if __name__=='__main__':
    main()
