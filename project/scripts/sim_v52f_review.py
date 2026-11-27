"""Independent v52f reviewer: AST/config/math only, or existing-log readback.

This file must never import a physics module, worker or telemetry implementation,
launch a child process, establish a simulator connection, or advance physics.
"""
import os,sys,json,math,hashlib,ast,gzip,time,collections,argparse,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='2'
ROOT=Path(__file__).resolve().parents[1]
REVIEW_DIR=ROOT/'logs/simulation_v52f/reviews'
PLAN=ROOT/'results/simulation_v52_dynamics_audit/proposed_diagnostic_plan.json'
ACTION=ROOT/'results/simulation_v52_dynamics_audit/proposed_diagnostic_action.json'
OLD=ROOT/'configs/simulation_v52r2.json'
OLD_RESULT=ROOT/'results/simulation_v52/episodes/005_slot_04_4f097762/result.json'
LOAD_PHASES={'close','lift','hold'}
PATCHED_PYTHON=ROOT/'runtime/simulation_v52e/venv/Scripts/python.exe'
PATCHED_BINARY=ROOT/'runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd'
PATCHED_ADMISSION_CONFIG=ROOT/'configs/simulation_v52e.json'
PATCHED_REFERENCE=ROOT/'results/simulation_v52e/attempts/001_ZERO001_2b433b45/afterR.json'
OLD_EXECUTION_CONFIG=ROOT/'configs/simulation_v52d.json'

def ident(path):
    p=Path(path);s=p.stat()
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=h,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sub(a,b):return [x-y for x,y in zip(a,b)]
def add(a,b):return [x+y for x,y in zip(a,b)]
def scale(a,s):return [x*s for x in a]
def norm(a):return math.sqrt(dot(a,a))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def close(a,b,atol=1e-8,rtol=1e-7):
    if isinstance(a,(list,tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(close(x,y,atol,rtol) for x,y in zip(a,b))
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b))

class Check:
    def __init__(self):self.count=0;self.failures=collections.Counter();self.examples=[]
    def test(self,condition,name,detail=None):
        self.count+=1
        if not condition:
            self.failures[name]+=1
            if len(self.examples)<80:self.examples.append(dict(check=name,detail=detail))
        return bool(condition)
    def export(self):return dict(passed=not self.failures,checks=self.count,failure_counts=dict(self.failures),examples=self.examples)

def mathematical_mocks(check):
    # Opposite order cross(F,r) has the wrong sign. Explicitly reject it.
    r=[1.,0.,0.];f=[0.,1.,0.];expected=[0.,0.,1.]
    check.test(close(cross(r,f),expected),'mock_r_cross_F_positive_z')
    check.test(not close(cross(f,r),expected),'mock_F_cross_r_negative_sign_rejected')
    left=[-.02,.003,.1];right=[.03,.003,.1];e=scale(sub(right,left),1/norm(sub(right,left)))
    forces=[scale(e,-1),e]
    check.test(close(add(*forces),[0,0,0]),'mock_equal_opposite_external_force')
    check.test(close(norm(forces[0]),1) and close(norm(forces[1]),1),'mock_one_newton_each')
    check.test(dot(forces[0],sub(left,right))>0 and dot(forces[1],sub(right,left))>0,'mock_outward_force_sign')
    check.test(close(dot([0,2,0],sub([0,.003,0],[0,.001,0])),.004),'mock_material_point_discrete_work')
    # COM offset produces a nonzero moment even with zero net external force.
    check.test(close(cross(sub([.01,0,0],[0,0,0]),[0,1,0]),[0,0,.01]),'mock_COM_lever_arm')
    check.test(close(dot([2,-1],[.3,.4]),.2),'mock_generalized_power')
    for dt,count in ((1/240,4440),(1/480,8880)):
        durations=[1,.8,2.5,.3,2.5,.3,2.5,.3,2,3,.3,3]
        check.test(sum(round(s/dt) for s in durations)==count,'mock_full_schedule_count',dt)
        check.test(close(count*dt,18.5),'mock_full_schedule_duration',dt)
    check.test(6*4440+6*8880==79920,'mock_cumulative_step_cap')
    check.test(12*18.5==222,'mock_cumulative_time_cap')
    # Nominal q=[s,-s,s,-s,-s,s] must satisfy the unchanged localtree.
    q=[.41,-.41,.41,-.41,-.41,.41]
    edges=[(0,3,1),(0,1,1),(1,2,1),(3,4,-1),(4,5,1)]
    check.test(all(close(q[a]+ratio*q[b],0) for a,b,ratio in edges),'mock_M0_five_gear_nullspace')
    bad=q.copy();bad[2]+=.1
    check.test(not all(close(bad[a]+ratio*bad[b],0) for a,b,ratio in edges),'mock_loaded_follower_drift_detected')

def source_ast(path):
    text=Path(path).read_text('utf8');tree=ast.parse(text)
    imports=[];calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module)
        elif isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else None
            if name:calls.append(dict(name=name,line=node.lineno))
    return dict(path=rel(path),identity=ident(path),imports=imports,calls=calls,text=text)

def base_preflight(check):
    plan=read(PLAN);action=read(ACTION);old=read(OLD);prior=read(OLD_RESULT)
    for p,b in plan['bindings'].items():check.test(ident(ROOT/p)==b,'original_plan_binding_unchanged',p)
    for key in ('candidate_world','goal_tool_world','requested_opening_m','open_driver_rad','estimated_contact_driver_rad'):
        check.test(action[key]==prior[key],'cached_slot4_action_exact',key)
    check.test(action['planning_joint_targets']==[dict(name=p['name'],joints=p['joints']) for p in prior['planning']],'cached_four_arm_targets_exact')
    check.test(action['source_identity']==ident(OLD_RESULT),'cached_action_source_identity')
    check.test(plan['fixed']['original_physics']==old['physics'],'plan_old_physics_exact')
    check.test(plan['fixed']['original_motion']==old['motion'],'plan_old_motion_exact')
    check.test(plan['fixed']['original_criteria']==old['criteria'],'plan_old_criteria_exact')
    check.test(len(plan['combinations'])==12,'plan_twelve_units')
    check.test({(c['scenario'],c['time_step_s'],c['solver_iterations']) for c in plan['combinations']}=={(s,d,i) for s in 'NLC' for d in (1/240,1/480) for i in (150,300)},'plan_exact_factorial')
    check.test(sum(c['maximum_step_count'] for c in plan['combinations'])==79920,'plan_step_cap')
    mathematical_mocks(check)
    self_scan=source_ast(Path(__file__))
    check.test(not any(m and ('bullet' in m or m.startswith('scripts.sim_')) for m in self_scan['imports']),'reviewer_no_physics_or_worker_import')
    check.test(not any(c['name'] in ('stepSimulation','connect','Popen','run_episode','applyExternalForce') for c in self_scan['calls']),'reviewer_cannot_start_diagnostics')
    return plan,action,old

def save_review(mode,payload):
    REVIEW_DIR.mkdir(parents=True,exist_ok=True)
    path=REVIEW_DIR/f'{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{time.time_ns()}.json'
    data=dict(at=datetime.now().astimezone().isoformat(),mode=mode,new_physics_connections=0,new_physics_steps=0,
        new_diagnostic_runs=0,source_identity=ident(Path(__file__)),**payload)
    with path.open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(review=rel(path),passed=data.get('checks',{}).get('passed'),mode=mode),ensure_ascii=False))
    return path

def preflight(config):
    check=Check();plan,action,old=base_preflight(check);cfg=read(config)
    check.test(cfg.get('schema')=='sim_v52f_execution_v1','execution_schema')
    check.test(cfg.get('execution_authorized') is True,'explicit_current_authorization')
    check.test(cfg.get('model')=='M0','M0_only')
    check.test(cfg.get('base_config')==rel(OLD),'unchanged_r2_base_config')
    check.test((ROOT/cfg.get('plan_path','')).resolve()==PLAN,'same_registered_plan')
    check.test((ROOT/cfg.get('action_path','')).resolve()==ACTION,'same_cached_action')
    check.test(cfg.get('combinations')==plan['combinations'],'exact_twelve_units_and_order')
    check.test(cfg.get('resources')==plan['resources'],'exact_original_resource_caps')
    check.test(cfg.get('criteria')==old['criteria'],'original_criteria_no_relaxation')
    check.test(cfg.get('python')==rel(PATCHED_PYTHON),'isolated_patched_interpreter_required')
    check.test(cfg.get('expected_distribution_version')=='3.2.7+v52e1','isolated_patched_distribution_required')
    check.test(cfg.get('parameter_reference_readback')==rel(PATCHED_REFERENCE),'accepted_zero_step_reference_required')
    check.test(cfg.get('parameter_targets')==dict(lateralFriction=1.,collisionMargin=.0005,restitution=0.),'exact_frozen_parameter_targets')
    patched_config=read(PATCHED_ADMISSION_CONFIG)
    check.test(ident(PATCHED_BINARY)==patched_config['bindings'][rel(PATCHED_BINARY)],'same_accepted_patched_binary')
    check.test(ident(PATCHED_REFERENCE)==read(ROOT/'results/simulation_v52e/attempts/001_ZERO001_2b433b45/result.json')['snapshots']['afterR']['identity'],'parameter_reference_identity_unchanged')
    for path in (PATCHED_BINARY,PATCHED_PYTHON,PATCHED_REFERENCE,OLD_EXECUTION_CONFIG):
        check.test(rel(path) in cfg.get('bindings',{}),'required_environment_history_binding',rel(path))
    depdir=(ROOT/cfg.get('dependency_dir','')).resolve();deppath=ROOT/cfg.get('dependency_manifest','')
    check.test(depdir==ROOT/'runtime/simulation_v52f/deps' and deppath.is_file(),'isolated_dependency_directory_and_manifest')
    if deppath.is_file():
        deps=read(deppath)
        check.test(deps.get('version')=='scipy1.15.3' and bool(deps.get('files')),'original_scipy_version_manifest')
        for p,b in deps.get('files',{}).items():
            resolved=(ROOT/p).resolve()
            check.test(resolved.is_relative_to(depdir) and ident(resolved)==b,'isolated_dependency_file_identity',p)
        check.test(not any('pybullet' in p.lower() for p in deps.get('files',{})),'isolated_dependencies_do_not_shadow_engine')
    authpath=ROOT/cfg.get('authorization_path','logs/simulation_v52f/authorization.json')
    check.test(authpath.is_file(),'authorization_record_present')
    authorization=read(authpath) if authpath.is_file() else None
    for p,b in cfg.get('bindings',{}).items():check.test((ROOT/p).is_file() and ident(ROOT/p)==b,'execution_binding_unchanged',p)
    required=['scripts/sim_v52f_episode.py','scripts/sim_v52d_telemetry.py','scripts/sim_v52f_control.py']
    source_reviews=[]
    for name in required:
        p=ROOT/name
        if not check.test(p.is_file(),'implementation_file_present',name):continue
        review=source_ast(p);body=review.pop('text');calls=review['calls']
        source_reviews.append(review)
        check.test(name in cfg.get('bindings',{}),'implementation_bound',name)
        if name.endswith('_episode.py'):
            forbidden={'calculateInverseKinematics','observed_proposal','plan_ik','resetBasePositionAndOrientation'}
            check.test(not any(c['name'] in forbidden for c in calls),'cached_action_no_new_proposal_or_teleport',name)
            check.test('simulation_v52r2' in body or 'sim_v52r2_episode' in body,'reuse_unchanged_M0_model',name)
            check.test('precharge' in body,'worker_precharge_guard_present')
            check.test('close' in body and 'lift' in body and 'hold' in body,'fixed_external_load_phases_present')
            check.test('max_force' not in body or 'gear_force' in body,'no_unexplained_gear_force_override')
            for guard in ('pad_parameter_assertions.jsonl','setup_robot_parameter_readback.json','physics_steps_returned_before','last_failed_pad_assertion'):
                check.test(guard in body,'new_actual_parameter_guard_present',guard)
            check.test(body.index('assertion=read_pad_parameter_assertion(')<body.index('assertion_stream.flush()')<body.index('client.stepSimulation()'),'actual_readback_assertion_flushed_before_step')
            check.test(body.index('if not assertion["passed"]:')<body.index('client.stepSimulation()'),'parameter_mismatch_rejected_before_physics')
        elif name.endswith('_telemetry.py'):
            check.test(not any(c['name'] in ('connect','stepSimulation','applyExternalForce','setJointMotorControl2','setJointMotorControlArray','resetJointState') for c in calls),'telemetry_is_read_only')
            check.test('getJointState' in body or 'getJointStates' in body,'telemetry_actual_joint_state')
            check.test('getConstraintState' in body,'telemetry_constraint_reaction')
            check.test('getDynamicsInfo' in body,'telemetry_runtime_dynamics_readback')
            check.test('calculateMassMatrix' in body,'telemetry_measured_mass_matrix')
        else:
            for guard in ('once_sync','precharge.json','intent','CREATE_NO_WINDOW','per_run_wall_cap_s','total_worker_wall_cap_s','total_session_wall_cap_s','tree_RSS_cap_bytes','output_cap_bytes','unknown_step_call','physics_step_calls_started','physics_step_calls_returned'):
                check.test(guard in body,'controller_required_guard_present',guard)
            check.test(body.index('append(LEDGER,intent)')<body.index('child=subprocess.Popen'),'charge_ledger_precedes_launch')
            check.test("if events(LEDGER) or (LOG/'session.json').exists()" in body,'one_shot_nonrestartable_session')
            check.test("phase='failure_or_unknown_preserved_stopped';break" in body,'failure_or_unknown_stops_remaining')
    check.test(authorization and authorization.get('authorized') is True,'authorization_true')
    check.test(authorization and authorization.get('plan')==ident(PLAN),'authorization_bound_plan')
    check.test('scripts/sim_v52f_review.py' in cfg.get('bindings',{}),'independent_reviewer_bound')
    payload=dict(review_passed=not check.failures,config_path=rel(config),config_identity=ident(config),
        physics_steps=0,checks=check.export(),authorization_record=authorization,source_reviews=source_reviews,
        boundaries=['This preflight executes no simulator or diagnostic worker.','AST presence checks complement human code review; they do not establish runtime dynamics correctness.',
            'Dt change alters ERP/dt and control-gain/dt; this plan is a sensitivity diagnostic, not pure convergence or success-rate inference.'])
    save_review('preflight',payload)
    return 0 if payload['review_passed'] else 2

def rotation(q):
    x,y,z,w=q;n=norm(q);x,y,z,w=[v/n for v in q]
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
def rotate(q,v):return [dot(row,v) for row in rotation(q)]
def phases(dt,old):
    m=old['motion'];cursor=0;out=[]
    for name,s in [('settle',1),('opening',.8),('transit',m['transit_seconds']+.3),('pregrasp',m['pregrasp_seconds']+.3),('approach',m['approach_seconds']+.3),('close',m['close_seconds']),('lift',m['lift_seconds']+.3),('hold',m['hold_seconds'])]:
        n=round(s/dt);out.append(dict(phase=name,first=cursor+1,last=cursor+n));cursor+=n
    return out
def phase_for(step,bounds):return next((b['phase'] for b in bounds if b['first']<=step<=b['last']),None)
def extreme(store,name,value,step,phase,absolute=False):
    x=abs(value) if absolute else value
    if name not in store or x>store[name]['value']:store[name]=dict(value=x,step=step,phase=phase)

def rpy_matrix(values):
    r,p,y=values;cr,sr=math.cos(r),math.sin(r);cp,sp=math.cos(p),math.sin(p);cy,sy=math.cos(y),math.sin(y)
    return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]

def audit_inertia(check,rows,urdf,label,names):
    import numpy as np
    xml=ET.parse(urdf).getroot();specs={x.get('name'):x for x in xml.findall('link')}
    for row in rows:
        name=names.get(row['link_index'])
        if name is None:continue
        node=specs[name].find('inertial')
        if node is None:continue
        mass=float(node.find('mass').get('value'));inertia=node.find('inertia').attrib
        M=np.array([[float(inertia.get('ixx',0)),float(inertia.get('ixy',0)),float(inertia.get('ixz',0))],[float(inertia.get('ixy',0)),float(inertia.get('iyy',0)),float(inertia.get('iyz',0))],[float(inertia.get('ixz',0)),float(inertia.get('iyz',0)),float(inertia.get('izz',0))]])
        origin=node.find('origin');xyz=[float(x) for x in origin.get('xyz','0 0 0').split()] if origin is not None else [0.,0.,0.]
        rpy=[float(x) for x in origin.get('rpy','0 0 0').split()] if origin is not None else [0.,0.,0.]
        R=np.array(rpy_matrix(rpy));Ractual=np.array(rotation(row['local_inertial_orientation_xyzw']))
        reconstructed=Ractual@np.diag(row['local_inertia_diagonal_kg_m2'])@Ractual.T
        check.test(close(row['mass_kg'],mass),'URDF_mass_actual',dict(asset=label,link=name))
        check.test(close(row['local_inertial_position_link_m'],xyz),'URDF_COM_actual',dict(asset=label,link=name))
        check.test(bool(np.allclose(reconstructed,R@M@R.T,atol=1e-8,rtol=2e-5)),'URDF_full_tensor_actual',dict(asset=label,link=name))

def audit_readback(check,rb,unit,old,manifest):
    check.test(rb.get('pybullet_distribution_version')=='3.2.7+v52e1','runtime_patched_distribution_version')
    check.test(Path(rb.get('python_executable','')).resolve()==PATCHED_PYTHON.resolve(),'runtime_patched_interpreter_path')
    actual_binaries=rb.get('pybullet_binary_files',[])
    patched_identity=ident(PATCHED_BINARY)
    check.test(len(actual_binaries)==1 and Path(actual_binaries[0].get('path','')).resolve()==PATCHED_BINARY.resolve()
        and actual_binaries[0].get('sha256')==patched_identity['sha256'] and actual_binaries[0].get('bytes')==patched_identity['bytes'],
        'runtime_patched_binary_identity')
    engine=rb.get('engine',{}).get('value') or {}
    check.test(rb.get('physics_steps_before_readback')==0,'readback_precedes_physics')
    check.test(close(engine.get('fixedTimeStep'),unit['time_step_s']),'actual_engine_dt')
    check.test(engine.get('numSolverIterations')==unit['solver_iterations'],'actual_engine_iterations')
    expected=dict(old['physics'],time_step=unit['time_step_s'],solver_iterations=unit['solver_iterations'])
    check.test(rb.get('commanded_physics')==expected,'unchanged_physics_except_registered_dt_iterations')
    check.test(rb.get('commanded_collision_filters')==old['allowed_self_collision_pairs'],'unchanged_collision_allowlist')
    check.test(len(rb['constraints'])==5,'exact_five_M0_gears')
    names={i['index']:i['value'][1] for i in rb['joint_info_raw']}
    link_names={i['index']:i['value'][12] for i in rb['joint_info_raw']}
    joint_index={n:i for i,n in names.items()};link_index={n:i for i,n in link_names.items()}
    for row,spec in zip(rb['constraints'],old['coupling_repair']['tree']):
        check.test(row['link_A']==joint_index[spec['a']] and row['link_B']==joint_index[spec['b']],'same_gear_joint_pair')
        check.test(close(row['gear_ratio'],spec['ratio']) and close(row['max_force'],80) and close(row['erp'],.2),'same_gear_ratio_force_ERP')
        check.test(close(row['relative_position_target'],0),'same_zero_gear_target')
    check.test(len(rb['mass_matrix_q_order'])==12,'mass_order_12')
    objects=[x for x in rb['dynamics'] if x['link_index']==-1 and close(x.get('mass_kg'),.04)]
    check.test(len(objects)==(1 if unit['scenario']=='C' else 0),'correct_object_presence_mass_readback')
    for obj in objects:
        check.test(close(obj['lateral_friction'],.6) and close(obj['restitution'],0),'actual_object_lateral_friction_restitution')
    robot_body=rb['constraints'][0]['body_A']
    base=next(x for x in rb['dynamics'] if x['body_id']==robot_body and x['link_index']==-1)
    check.test(close(base['lateral_friction'],.5,atol=1e-12,rtol=0)
        and close(base['collision_margin_m'],.001,atol=1e-12,rtol=0)
        and close(base['restitution'],0,atol=1e-12,rtol=0),'robot_base_correct_loaded_parameters_not_old_misdirection')
    pads=[link_index[n] for n in manifest['pad_links']]
    for pad in pads:
        dyn=next(x for x in rb['dynamics'] if x['body_id']==robot_body and x['link_index']==pad)
        check.test(close(dyn['mass_kg'],.005),'actual_pad_mass')
        check.test(close(dyn['lateral_friction'],1) and close(dyn['restitution'],0),'actual_pad_friction')
        check.test(close(dyn.get('collision_margin_m'),.0005,atol=1e-12,rtol=0),'actual_pad_collision_margin')
    audit_inertia(check,[x for x in rb['dynamics'] if x['body_id']==robot_body],ROOT/old['robot_urdf'],'M0 robot',dict(link_names,**{})|{-1:'world'})
    if objects:audit_inertia(check,objects,ROOT/old['objects']['021']['urdf'],'021 object',{-1:'object'})
    return robot_body,pads,link_index

def audit_parameter_records(directory,unit,result,check):
    """Reconstruct new parameter assertions without importing their producer."""
    reference=read(PATCHED_REFERENCE);ref={x['link_index']:x for x in reference['links']}
    setup_path=directory/'setup_robot_parameter_readback.json'
    setup_ok=False
    if setup_path.is_file():
        setup=read(setup_path)
        check.test(result.get('setup_parameter_readback_identity')==ident(setup_path),'setup_readback_result_identity')
        check.test(setup.get('reference_identity')==ident(PATCHED_REFERENCE) and setup.get('reference_path')==rel(PATCHED_REFERENCE),'setup_reference_binding')
        actual={x['link_index']:x for x in setup.get('links',[])};setup_ok=set(actual)==set(ref) and len(actual)==len(setup.get('links',[]))
        for i,row in actual.items():
            expected=ref.get(i)
            if expected is None:setup_ok=False;continue
            dyn_ok=row.get('dynamics_raw')==expected['dynamics_raw'] and row.get('link_name')==expected['link_name']
            ag=json.loads(json.dumps(row.get('collision_geometry_raw')));eg=json.loads(json.dumps(expected['collision_geometry_raw']))
            for geometry in (ag,eg):
                for shape in geometry:shape[0]=0
            geometry_ok=ag==eg or recursive_geometry_equal(ag,eg)
            check.test(dyn_ok,'setup_all_robot_actual_dynamics_match_accepted_reference',i)
            check.test(geometry_ok,'setup_all_robot_geometry_match_accepted_reference',i)
            setup_ok &= dyn_ok and geometry_ok
        check.test(setup.get('physics_steps')==0,'setup_checked_before_physics')
        check.test(setup.get('passed')==setup_ok,'setup_flag_reconstructed')
        check.test(result.get('setup_parameter_consistency_passed')==setup_ok,'setup_result_flag_reconstructed')
    else:
        check.test(result.get('actual_physics_steps')==0 and result.get('status')!='completed','missing_setup_requires_zero_steps_failed_result')
    log=directory/'pad_parameter_assertions.jsonl';count=passed=0;failed_row=None
    expected={'lateralFriction':1.,'collisionMargin':.0005,'restitution':0.}
    expected_pads={i:x['link_name'] for i,x in ref.items() if x['link_name'] in ('rq_left_inner_finger_pad','rq_right_inner_finger_pad')}
    bounds=phases(unit['time_step_s'],read(OLD))
    if log.is_file():
        with log.open(encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line);count+=1
                check.test(failed_row is None,'no_assertion_after_failed_parameter_check',count)
                check.test(row.get('schema')=='sim_v52f_pad_parameter_assertion_v1' and row.get('when')=='before_stepSimulation','parameter_assertion_schema',count)
                check.test(row.get('step')==count and row.get('phase')==phase_for(count,bounds) and row.get('physics_steps_returned_before')==count-1,'parameter_assertion_continuous_pre_step_order',count)
                check.test(row.get('expected')==expected,'parameter_assertion_unchanged_expected_values',count)
                actual_ok=row.get('exception') is None and row.get('readback_calls_started')==row.get('readback_calls_returned')==2
                rows=row.get('pads',[]);actual_ok &= len(rows)==2 and {x.get('link_index') for x in rows}==set(expected_pads)
                for pad in rows:
                    raw=pad.get('raw',[]);values={'lateralFriction':raw[1],'collisionMargin':raw[11],'restitution':raw[5]} if len(raw)>=12 else None
                    own_ok=values==expected and pad.get('link_name')==expected_pads.get(pad.get('link_index'))
                    check.test(pad.get('actual')==values,'parameter_actual_derived_from_raw_tuple',dict(step=count,pad=pad.get('link_index')))
                    check.test(pad.get('passed')==own_ok,'parameter_pad_pass_flag_reconstructed',dict(step=count,pad=pad.get('link_index')))
                    if own_ok:check.test(pad.get('mismatches')=={},'parameter_pad_no_hidden_mismatches',count)
                    actual_ok &= own_ok
                check.test(row.get('passed')==actual_ok,'parameter_row_pass_flag_reconstructed',count)
                if actual_ok:passed+=1
                else:failed_row=row
        check.test(result.get('pad_assertions_identity')==ident(log),'parameter_log_result_identity')
    check.test(result.get('pad_assertions_attempted')==count and result.get('pad_assertions_passed')==passed,'parameter_result_counts_match_raw_log')
    check.test(result.get('last_failed_pad_assertion')==failed_row,'parameter_failure_evidence_retained')
    steps=result.get('actual_physics_steps',0)
    if failed_row is not None:
        check.test(steps==failed_row['step']-1 and result.get('status')!='completed','parameter_failure_prevents_current_and_later_physics')
    if result.get('status')=='completed':
        check.test(setup_ok and count==passed==steps==unit['maximum_step_count'] and failed_row is None and result.get('pad_parameter_assertions_complete') is True,'every_completed_step_has_exact_actual_parameter_check')
    else:
        check.test(steps<=passed and count<=steps+1,'failed_unit_has_no_unchecked_returned_step')
    return dict(assertion_records=count,passed_assertions=passed,returned_steps=steps,setup_reconstructed_passed=setup_ok,
        failure_step=failed_row['step'] if failed_row is not None else None,new_engine_connections=0)

def recursive_geometry_equal(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12+1e-10*max(abs(a),abs(b))
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(recursive_geometry_equal(x,y) for x,y in zip(a,b))
    return a==b

def audit_unit(directory,unit,result,old,action,check):
    # NumPy is used only for recorded12x12 matrix arithmetic, never simulation.
    import numpy as np
    rb=read(directory/'telemetry_readback.json');summary=read(directory/'telemetry_summary.json')
    manifest=read(ROOT/old['robot_manifest']);mechanism=read(ROOT/'results/simulation_v52_dynamics_audit/mechanism.json')
    robot_body,pads,link_index=audit_readback(check,rb,unit,old,manifest)
    object_body=next((x['body_id'] for x in rb['dynamics'] if x['link_index']==-1 and close(x.get('mass_kg'),.04)),None)
    table_body=next(x['body_id'] for x in rb['dynamics'] if x['link_index']==-1 and x['body_id']!=robot_body and close(x.get('mass_kg'),0))
    dt=unit['time_step_s'];bounds=phases(dt,old);prev=None;count=0;extrema={};phase_extrema={};work=collections.Counter();phase_counts=collections.Counter()
    forbidden_ticks=0;post_settle_forbidden=False;formed=False;slipped=False;hold=0;loss=0;initial_pos=None
    max_lift=None;post_settle_max_mimic=0.;first_contact=None;first_violation=None;arm_starts={}
    cal=manifest['opening_calibration'];grid=[x['driver_rad'] for x in cal]
    def tcp(q):return np.array([np.interp(q,grid,[x['tcp_in_gripper_base_m'][i] for x in cal]) for i in range(3)])
    targets={x['name']:x['joints'] for x in action['planning_joint_targets']}
    shapes=ET.parse(ROOT/old['robot_urdf']).getroot()
    pad_origins={link_index[link.attrib['name']]:[float(v) for v in link.find('collision/origin').attrib.get('xyz','0 0 0').split()] for link in shapes.findall('link') if link.attrib['name'] in manifest['pad_links']}
    first_failure_count=sum(check.failures.values())
    with gzip.open(directory/'telemetry.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line);count+=1;step=row.get('step');phase=row.get('phase');phase_counts[phase]+=1
            check.test(row.get('schema')=='sim_v52d_tick_v1' and 'capture_error' not in row,'complete_tick_schema',step)
            check.test(step==count and phase==phase_for(count,bounds),'continuous_steps_phases',step)
            check.test(close(row.get('time_s'),count*dt) and close(row.get('dt_s'),dt),'tick_times',step)
            check.test(row.get('scenario')==unit['scenario'],'tick_scenario',step)
            joints={j['name']:j for j in row['joints']};links={x['index']:x for x in row['links']};object_state=row['object_state']
            check.test(len(joints)==12,'full_joint_readback',step)
            check.test((object_state is not None)==(unit['scenario']=='C'),'tick_object_presence',step)
            q={n:j['q_post_rad'] for n,j in joints.items()};preq={n:j['q_pre_rad'] for n,j in joints.items()}
            dq={n:j['dq_post_rad_s'] for n,j in joints.items()};master=q[manifest['master_joint']]
            errors={f['joint']:q[f['joint']]-(f['multiplier']*master+f['offset']) for f in manifest['mimic_joints']};err=max(map(abs,errors.values()))
            check.test(close(row['flags']['max_mimic_error_rad'],err),'recomputed_mimic',step)
            check.test(all(close(row['flags']['mimic_residual_by_joint_rad'].get(n),v) for n,v in errors.items()),'recomputed_follower_errors',step)
            extreme(extrema,'mimic_error_rad',err,step,phase)
            extreme(phase_extrema.setdefault(phase,{}),'mimic_error_rad',err,step,phase)
            if err>.03 and first_violation is None:first_violation=step
            phase_start=next(x['first'] for x in bounds if x['phase']==phase)
            if phase in targets and phase not in arm_starts:arm_starts[phase]=[preq[n] for n in manifest['arm_joint_names']]
            if phase in targets:
                movement_steps=round(old['motion'][phase+'_seconds']/dt);k=min(step-phase_start+1,movement_steps);a=k/movement_steps;a=a*a*(3-2*a)
                expected_arm=[(1-a)*x+a*y for x,y in zip(arm_starts[phase],targets[phase])]
            elif phase in ('settle','opening'):expected_arm=old['home_joints']
            elif phase=='close':expected_arm=targets['approach']
            else:expected_arm=targets['lift']
            expected_master=0. if phase=='settle' else action['open_driver_rad'] if phase in ('opening','transit','pregrasp','approach') else manifest['master_range_rad'][1]
            for n,j in joints.items():
                command=j['command'];tau=j['applied_motor_torque_Nm'];dv=j['q_post_rad']-j['q_pre_rad']
                if prev is not None:check.test(close(j['q_pre_rad'],prev['q'][n]) and close(j['dq_pre_rad_s'],prev['dq'][n]),'state_interval_continuity',dict(step=step,joint=n))
                if n in manifest['arm_joint_names']:
                    index=manifest['arm_joint_names'].index(n);expected_cap=old['physics']['arm_forces'][index];expected_gain=.1 if phase in ('settle','opening') else .08
                    check.test(close(command['position_rad'],expected_arm[index]) and close(command['position_gain'],expected_gain) and close(command['velocity_gain'],1),'unchanged_arm_trajectory_gains',dict(step=step,joint=n))
                elif n==manifest['master_joint']:
                    expected_cap=10.
                    check.test(close(command['position_rad'],expected_master) and close(command['position_gain'],.1) and close(command['velocity_gain'],1) and close(command['max_velocity_rad_s'],.5),'unchanged_master_command',step)
                else:
                    expected_cap=0.
                    check.test(command['mode']=='VELOCITY_CONTROL' and close(command['velocity_rad_s'],0),'followers_unpowered',dict(step=step,joint=n))
                check.test(close(command['force_cap_Nm'],expected_cap),'unchanged_actuator_cap',dict(step=step,joint=n))
                check.test(close(j['power_post_W'],tau*j['dq_post_rad_s']) and close(j['work_discrete_J'],tau*dv),'motor_power_and_discrete_work',dict(step=step,joint=n))
                extreme(extrema,'motor_torque_Nm',tau,step,phase,True)
            check.test(len(row['gears'])==5,'all_gears_each_tick',step)
            for gear,spec in zip(row['gears'],old['coupling_repair']['tree']):
                a,b,r=spec['a'],spec['b'],spec['ratio'];e0=preq[a]+r*preq[b];e1=q[a]+r*q[b];de=dq[a]+r*dq[b];reaction=gear['reaction_Nm']
                check.test(gear['a']==a and gear['b']==b and close(gear['ratio'],r) and close(gear['cap_Nm'],80),'same_gears_in_trace',step)
                check.test(close(gear['e_pre_rad'],e0) and close(gear['e_post_rad'],e1) and close(gear['de_post_rad_s'],de),'gear_position_velocity_residual',step)
                check.test(close(gear['applied_impulse_Nm_s'],reaction*dt) and close(gear['power_post_W'],reaction*de) and close(gear['work_discrete_J'],reaction*(e1-e0)),'gear_impulse_work',step)
                extreme(extrema,'gear_reaction_Nm',reaction,step,phase,True)
                extreme(extrema,'gear_residual_rad',e1,step,phase,True)
            for side,spec in mechanism['closure_axes'].items():
                la=links[link_index[spec['knuckle_link']]];lb=links[link_index[spec['finger_link']]]
                pa=add(la['link_frame_world_m'],rotate(la['link_frame_orientation_xyzw'],spec['anchor_in_knuckle_link_m']))
                pb=add(lb['link_frame_world_m'],rotate(lb['link_frame_orientation_xyzw'],spec['anchor_in_inner_finger_link_m']))
                observed=row['closed_loops'][side];closure=norm(sub(pa,pb))
                check.test(close(observed['knuckle_anchor_world_m'],pa) and close(observed['finger_anchor_world_m'],pb) and close(observed['position_error_m'],closure),'independent_closed_loop_positions',dict(step=step,side=side))
                aa=rotate(la['link_frame_orientation_xyzw'],spec['axis_in_each_link']);ab=rotate(lb['link_frame_orientation_xyzw'],spec['axis_in_each_link'])
                check.test(close(observed['knuckle_axis_world'],aa) and close(observed['finger_axis_world'],ab) and close(observed['axis_cross_norm'],norm(cross(aa,ab))),'independent_closed_loop_axes',dict(step=step,side=side))
                extreme(extrema,'closure_error_m',closure,step,phase)
            loads=row['external_loads'];active_load=unit['scenario']=='L' and phase in LOAD_PHASES
            check.test(len(loads)==(2 if active_load else 0),'external_load_only_registered_scenario_phases',step)
            if loads:
                bypad={x['link_index']:x for x in loads};a=bypad[pads[0]];b=bypad[pads[1]];direction=sub(b['point_pre_world_m'],a['point_pre_world_m']);direction=scale(direction,1/norm(direction))
                for sign,pad in zip((-1,1),pads):
                    load=bypad[pad];F=load['force_pre_world_N'];p0=load['point_pre_world_m'];c0=load['com_pre_world_m'];lever=sub(p0,c0)
                    check.test(close(F,scale(direction,sign)) and close(norm(F),1),'fixed_equal_opposite_outward_1N',step)
                    check.test(close(load['torque_about_com_pre_Nm'],cross(lever,F)),'external_torque_r_cross_F',step)
                    if prev is not None:
                        pl=prev['links'][pad];expected_point=add(pl['link_frame_world_m'],rotate(pl['link_frame_orientation_xyzw'],pad_origins[pad]))
                        check.test(close(p0,expected_point,atol=1e-7) and close(c0,pl['com_world_m']),'load_at_true_pad_geometry_center_pre',step)
                        v0=add(pl['com_linear_velocity_world_m_s'],cross(pl['angular_velocity_world_rad_s'],lever))
                        check.test(close(load['point_velocity_pre_world_m_s'],v0),'external_point_pre_velocity_from_COM',step)
                    nowlink=links[pad];leverpost=rotate(load['com_post_orientation_xyzw'],load['point_material_local_com_m'])
                    v1=add(nowlink['com_linear_velocity_world_m_s'],cross(nowlink['angular_velocity_world_rad_s'],leverpost))
                    check.test(close(load['point_velocity_post_world_m_s'],v1) and close(load['torque_about_com_post_Nm'],cross(leverpost,F)),'external_post_velocity_and_torque',step)
                    p1=add(load['com_post_world_m'],rotate(load['com_post_orientation_xyzw'],load['point_material_local_com_m']));displacement=sub(p1,p0)
                    check.test(close(load['same_material_point_post_world_m'],p1) and close(load['material_point_displacement_world_m'],displacement),'material_point_mapping',step)
                    check.test(close(load['work_discrete_J'],dot(F,displacement)) and close(load['power_pre_W'],dot(F,load['point_velocity_pre_world_m_s'])) and close(load['power_post_W'],dot(F,load['point_velocity_post_world_m_s'])),'external_work_power_recompute',step)
            active_pads=set();supported=False;forbidden=False;wrenches={}
            for c in row['contacts']:
                F=add(add(scale(c['normal_on_B_world'],c['normal_force_N']),scale(c['lateral_direction_1_world'],c['lateral_force_1_N'])),scale(c['lateral_direction_2_world'],c['lateral_force_2_N']))
                check.test(close(c['force_on_A_world_N'],F) and close(c['force_on_B_world_N'],scale(F,-1)),'contact_action_reaction_sign',step)
                A,B=c['body_A'],c['body_B'];la,lb=c['link_A'],c['link_B'];active=c['normal_force_N']>.2 or c['distance_m']<-.001;prohibited=False
                if A==robot_body and B==robot_body:prohibited=active
                elif robot_body in (A,B) and table_body in (A,B):prohibited=active and (la if A==robot_body else lb) not in old['base_table_allowed_link_indices']
                elif object_body is not None and robot_body in (A,B) and object_body in (A,B):
                    rl=la if A==robot_body else lb;prohibited=active and rl not in pads
                    if rl in pads and c['normal_force_N']>.2:active_pads.add(rl)
                elif object_body is not None and object_body in (A,B) and table_body in (A,B):supported|=c['normal_force_N']>.2
                check.test(c['active_by_original_gate']==active and c['forbidden']==prohibited,'unchanged_contact_gate',step);forbidden|=prohibited
                for body,index,point,force in [(A,la,c['point_A_world_m'],F),(B,lb,c['point_B_world_m'],scale(F,-1))]:
                    key=str(body)+':'+str(index);observed=row['contact_wrenches'][key]
                    center=links[index]['com_world_m'] if body==robot_body else object_state['position_world_m'] if body==object_body else observed['about_com_world_m']
                    check.test(close(observed['about_com_world_m'],center),'contact_wrench_COM',step)
                    entry=wrenches.setdefault(key,dict(force=[0,0,0],torque=[0,0,0],count=0));entry['force']=add(entry['force'],force);entry['torque']=add(entry['torque'],cross(sub(point,center),force));entry['count']+=1
                if c['category']=='robot_object':extreme(extrema,'pad_or_nonpad_object_normal_force_N',c['normal_force_N'],step,phase)
            check.test(set(wrenches)==set(row['contact_wrenches']),'all_contact_wrenches',step)
            for key,w in wrenches.items():
                observed=row['contact_wrenches'][key];check.test(close(observed['force_world_N'],w['force']) and close(observed['torque_about_com_world_Nm'],w['torque']) and observed['contact_count']==w['count'],'independent_contact_wrench_sum',step)
            bilateral=len(active_pads)==2
            check.test(row['flags']['forbidden_collision']==forbidden and row['flags']['bilateral_pad_contact']==bilateral,'contact_flags_recomputed',step)
            if forbidden:forbidden_ticks+=1
            if bilateral and first_contact is None:first_contact=step
            if phase=='settle' and object_state is not None:initial_pos=object_state['position_world_m']
            if phase!='settle':
                post_settle_max_mimic=max(post_settle_max_mimic,err);post_settle_forbidden|=forbidden
                if object_state is not None:
                    pos=object_state['position_world_m'];lift=pos[2]-initial_pos[2];max_lift=lift if max_lift is None else max(lift,max_lift)
                    formed|=phase in ('lift','hold') and bilateral and not supported and lift>=.01
                    loss=0 if bilateral else loss+1
                    ee=links[link_index[manifest['gripper_base_link']]];padcenter=add(ee['link_frame_world_m'],rotate(ee['link_frame_orientation_xyzw'],tcp(master)))
                    if formed and loss*dt>=.2 and (lift<.01 or norm(sub(pos,padcenter))>.12):slipped=True
                    valid=phase=='hold' and lift>=.1 and not supported and not post_settle_forbidden and formed and bilateral and not slipped
                    hold=hold+1 if valid else 0
            M=row['mass_matrix'];qorder=M['q_order'];mat=np.array(M['matrix'],float)
            check.test(mat.shape==(12,12) and np.isfinite(mat).all(),'full_finite_mass_matrix',step)
            check.test(close(M['q_rad'],[q[n] for n in qorder]),'mass_matrix_at_actual_q',step)
            check.test(bool(np.allclose(mat,mat.T,atol=1e-10,rtol=1e-9)),'mass_matrix_symmetry',step)
            try:np.linalg.cholesky(mat);spd=True
            except np.linalg.LinAlgError:spd=False
            check.test(spd,'mass_matrix_positive_definite',step)
            W=np.linalg.solve(mat,np.eye(12))
            for g in M['gear_rows']:
                ia,ib=qorder.index(g['a']),qorder.index(g['b']);ratio=g['ratio'];source=W[ia,ia]+ratio*ratio*W[ib,ib];full=source+ratio*(W[ia,ib]+W[ib,ia])
                check.test(close(g['d_source'],float(source)) and close(g['d_full'],float(full)) and close(g['d_full_over_d_source'],float(full/source)),'independent_mass_cross_terms',step)
            for key,rows in [('motor',row['joints']),('gear',row['gears']),('external',loads)]:
                value=sum(x['work_discrete_J'] for x in rows);check.test(close(row['power_work'][key+'_work_discrete_J'],value),'work_sum',step);work[key+'_work_discrete_J']+=value
            prev=dict(q=q,dq=dq,links=links)
    check.test(count==result['actual_physics_steps']==unit['maximum_step_count'],'full_approved_step_count')
    check.test(count==summary['sample_count'] and summary['telemetry_complete'] is True,'summary_complete')
    check.test(dict(phase_counts)==summary['phase_counts'],'summary_phase_counts')
    check.test(close(summary['max_mimic_error_rad'],extrema['mimic_error_rad']['value']),'summary_max_mimic')
    for summary_key,extreme_key in [('max_gear_residual_rad','gear_residual_rad'),('max_closure_error_m','closure_error_m'),('max_abs_motor_torque_Nm','motor_torque_Nm'),('max_abs_gear_reaction_Nm','gear_reaction_Nm')]:check.test(close(summary[summary_key],extrema[extreme_key]['value']),'summary_extrema',summary_key)
    check.test(summary['forbidden_contact_ticks']==forbidden_ticks,'summary_forbidden_ticks')
    check.test(result.get('all_tick_mimic_within_003')==(extrema['mimic_error_rad']['value']<=.03),'all_tick_mimic_gate_separate')
    check.test(result.get('original_gate_window')=='post_settle','original_score_window_unchanged')
    check.test(close(result.get('max_mimic_error_rad'),post_settle_max_mimic),'result_post_settle_mimic_window')
    check.test(result.get('forbidden_collision')==post_settle_forbidden,'result_original_collision_window')
    for key,value in work.items():check.test(close(summary[key],value,atol=1e-7),'summary_discrete_work',key)
    original_success=None
    if unit['scenario']=='C':
        original_success=hold*dt>=2 and bilateral and not slipped and not post_settle_forbidden and post_settle_max_mimic<=.03
        check.test(result['grasp_formed']==formed and result['slipped']==slipped and close(result['final_continuous_hold_seconds'],hold*dt),'original_grasp_formation_hold_slip')
        check.test(result['grasp_success']==original_success and result['original_grasp_success']==original_success,'independent_contact_success')
    return dict(unit=unit,records=count,phase_counts=dict(phase_counts),extrema=extrema,phase_extrema=phase_extrema,
        post_settle_max_mimic_rad=post_settle_max_mimic,forbidden_contact_ticks=forbidden_ticks,
        post_settle_forbidden=post_settle_forbidden,first_bilateral_contact_step=first_contact,first_mimic_violation_step=first_violation,
        recomputed_work_discrete_J=dict(work),recomputed_original_grasp_success=original_success,
        numerical_or_mechanical_gate_passed=post_settle_max_mimic<=.03 and not post_settle_forbidden,
        additional_integrity_failures=sum(check.failures.values())-first_failure_count,
        limitations='Full-tick discrete estimates; single cell does not estimate success rate or prove unique root cause. M inverse is unconstrained generalized inertia, not contact-conditioned effective inertia.')

def final(config):
    check=Check();plan,action,old=base_preflight(check);cfg=read(config)
    ledger_path=ROOT/'logs/simulation_v52f/attempts.jsonl'
    ledger=[json.loads(line) for line in ledger_path.read_text('utf8').splitlines() if line.strip()]
    intents=[x for x in ledger if x.get('event')=='intent'];returns=[x for x in ledger if x.get('event')=='returned']
    check.test(len(intents)<=12,'at_most_twelve_charged_starts')
    check.test(len({x['attempt_id'] for x in intents})==len(intents),'no_duplicate_charged_attempts')
    check.test(len({x['unit_id'] for x in intents})==len(intents),'no_unit_retries')
    check.test(len({x['attempt_id'] for x in returns})==len(returns),'no_duplicate_returns')
    check.test(all(x['attempt_id'] in {i['attempt_id'] for i in intents} for x in returns),'no_return_without_charged_intent')
    results=[];unknown=[];unknown_steps=[];stop=False;worker_wall=0.;steps=0;known_returns=0;budget_events=[]
    for index,intent in enumerate(intents):
        unit=plan['combinations'][index];check.test(not stop,'no_continuation_after_failure_or_unknown',unit['id'])
        check.test(intent['charged_index']==index+1 and intent['unit_id']==unit['id'] and intent['unit']==unit,'fixed_charged_order',unit['id'])
        check.test(intent['config']==ident(config),'intent_config_identity',unit['id'])
        directory=ROOT/intent['directory'];precharge=read(directory/'precharge.json')
        check.test(ident(directory/'precharge.json')==intent['precharge'],'precharge_identity',unit['id'])
        check.test(precharge['unit']==unit and precharge['charged_index']==index+1,'precharge_unit_and_order',unit['id'])
        check.test(precharge['config_identity']==ident(config) and precharge['worker_identity']==intent['worker']==ident(ROOT/'scripts/sim_v52f_episode.py'),'precharge_immutable_sources',unit['id'])
        returned=next((x for x in returns if x['attempt_id']==intent['attempt_id']),None)
        if returned is not None:
            worker_wall+=returned['wall_seconds']
            check.test(read(ROOT/returned['resources_path'])==returned,'resource_record_matches_return',unit['id'])
            if returned['wall_seconds']>plan['resources']['per_run_wall_cap_s'] or returned['peak_tree_rss_bytes']>plan['resources']['tree_RSS_cap_bytes']:
                budget_events.append(dict(unit=unit['id'],wall_seconds=returned['wall_seconds'],peak_sampled_RSS=returned['peak_tree_rss_bytes'],termination=returned.get('termination')))
                check.test(returned.get('termination') is not None or returned.get('exit_code')!=0,'budget_violation_requires_failed_stop',unit['id'])
        if returned is None or returned.get('result_status')!='known_result':unknown.append(unit['id']);stop=True;continue
        known_returns+=1
        result_path=ROOT/returned['result_path'];check.test(ident(result_path)==returned['result'],'returned_result_identity',unit['id'])
        result=read(result_path);steps+=result['actual_physics_steps']
        started=result.get('physics_step_calls_started');finished=result.get('physics_step_calls_returned');uncertain=result.get('unknown_step_call')
        check.test(finished==result['actual_physics_steps'] and isinstance(started,int) and started>=finished,'returned_step_count_actual',unit['id'])
        if uncertain is not None or started!=finished or returned.get('termination') is not None:unknown_steps.append(dict(unit=unit['id'],started=started,returned=finished,unknown_step_call=uncertain,termination=returned.get('termination')))
        check.test(result['config_identity']==ident(config) and result['source_identity']==intent['worker'] and result['action_identity']==ident(ACTION) and result['base_config_identity']==ident(OLD),'result_source_identities',unit['id'])
        check.test(result['unit']==unit and result['attempt_id']==intent['attempt_id'] and result['model']=='M0','worker_result_binding',unit['id'])
        check.test(result['actual_new_training']==0 and result['actual_neural_forwards']==0 and result['new_object_attachment_constraints']==0 and result['object_teleports_after_initialization']==0,'no_scope_expansion',unit['id'])
        parameter_audit=audit_parameter_records(directory,unit,result,check)
        if result['status']!='completed' or not result['completion'] or returned.get('termination') is not None or returned['exit_code']!=0 or uncertain is not None or started!=finished:
            stop=True;results.append(dict(unit=unit,status='failed_preserved',actual_physics_steps=result['actual_physics_steps'],physics_step_calls_started=started,physics_step_calls_returned=finished,unknown_step_call=uncertain,exception=result.get('exception'),parameter_audit=parameter_audit));continue
        audited=audit_unit(directory,unit,result,old,action,check);audited['parameter_audit']=parameter_audit;results.append(audited)
    check.test(worker_wall<=plan['resources']['total_worker_wall_cap_s'] or stop,'total_worker_wall_cap_or_preserved_stop')
    check.test(steps<=plan['resources']['maximum_step_count'],'total_step_cap')
    output_bytes=sum(p.stat().st_size for directory in (ROOT/'results/simulation_v52f',ROOT/'logs/simulation_v52f') for p in directory.rglob('*') if p.is_file())
    check.test(output_bytes<=plan['resources']['output_cap_bytes'],'aggregate_output_cap')
    for p,b in cfg.get('bindings',{}).items():check.test(ident(ROOT/p)==b,'final_bound_file_unchanged',p)
    save_review('final',dict(review_passed=not check.failures,config_identity=ident(config),physics_steps=0,
        checks=check.export(),charged_units=len(intents),known_returns=known_returns,unknown_units=unknown,unknown_step_calls=unknown_steps,exact_total_steps_known=not unknown and not unknown_steps,budget_events=budget_events,
        all_12_completed=len(results)==12 and not stop and all('records' in x for x in results),
        independently_recomputed_units=results,actual_recorded_diagnostic_steps=steps,worker_wall_seconds=worker_wall,
        output_bytes=output_bytes,ledger_identity=ident(ledger_path),no_automatic_M1_or_R4=True,
        interpretation=['Compare iteration settings at fixeddt; dt variation also changes dimensionless controller/constraint rates.',
            'Repaired M0 differs from v52d in actual pad friction/margin and previously misdirected base parameters. It is not a single-variable friction or margin ablation.',
            'Complete telemetry can narrow mechanisms but does not establish unique root cause or scientific method benefit.',
            'Every startup consumes its slot; unknown/failed units remain preserved and no later unit may start.']))
    return 0 if not check.failures else 2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','final'))
    parser.add_argument('--config',default='configs/simulation_v52f.json');args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Explicit required project workdir')
    config=(ROOT/args.config).resolve();config.relative_to(ROOT)
    return preflight(config) if args.mode=='preflight' else final(config)

if __name__=='__main__':raise SystemExit(main())
