"""Read-only telemetry for an externally owned, budgeted physics client.

This module never imports PyBullet, connects, steps, changes motors, or adds forces.
All positions/forces below are in world coordinates unless explicitly named local.
"""
from pathlib import Path
import collections
import copy
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_value': repr(value)}
    return value


def _write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(_json_value(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def _rotation(q):
    x, y, z, w = np.asarray(q, dtype=float)
    n = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError('Nonfinite or zero quaternion')
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _finite(value, label):
    if not np.isfinite(np.asarray(value, dtype=float)).all():
        raise ValueError('Nonfinite '+label)


def _identity(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


class Recorder:
    """pre_step -> owner's stepSimulation -> capture, with 1-based step IDs.

    targets is the worker's explicit arm/gripper/followers command dictionary.
    A command is never represented as an engine readback.
    """
    def __init__(self, client, robot, table, object_id, cfg, output_dir, scenario,
                 manifest=None):
        self.p = client
        self.robot = robot
        self.body = int(robot['body'])
        self.table = int(table)
        self.object_id = None if object_id is None else int(object_id)
        self.cfg = cfg
        self.scenario = scenario['id'] if isinstance(scenario, dict) else str(scenario)
        self.dt = float(cfg['physics']['time_step'])
        self.directory = Path(output_dir).resolve()
        if not self.directory.is_relative_to(ROOT) or self.dt <= 0:
            raise ValueError('Invalid output root or timestep')
        self.directory.mkdir(parents=True, exist_ok=True)
        if manifest is None:
            manifest = json.loads((ROOT/cfg['robot_manifest']).read_text(encoding='utf-8'))
        self.manifest = manifest
        self.num_links = client.getNumJoints(self.body)
        infos = [(i, client.getJointInfo(self.body, i)) for i in range(self.num_links)]
        movable = sorted([(i, v) for i, v in infos if v[3] >= 0], key=lambda x: x[1][3])
        if len(movable) != 12 or any(v[2] not in (0, 1) for _, v in movable):
            raise ValueError('Expected exactly twelve scalar revolute/prismatic DOFs')
        if len({v[4] for _, v in movable}) != 12:
            raise ValueError('Duplicate uIndex')
        if [i for i,_ in movable] != [i for i,_ in sorted(movable,key=lambda x:x[1][4])]:
            raise ValueError('Scalar qIndex and uIndex orders differ')
        self.indices = [i for i, _ in movable]
        self.names = [v[1].decode() if isinstance(v[1], bytes) else str(v[1]) for _, v in movable]
        self.order = {n: i for i, n in enumerate(self.names)}
        self.link_names = {i: (v[12].decode() if isinstance(v[12], bytes) else str(v[12])) for i, v in infos}
        self.gear_tree = cfg['coupling_repair']['tree']
        if len(robot['gear_ids']) != 5 or len(self.gear_tree) != 5:
            raise ValueError('Expected five frozen gear rows')
        self.J = np.zeros((5, 12))
        for k, row in enumerate(self.gear_tree):
            self.J[k, self.order[row['a']]] = 1
            self.J[k, self.order[row['b']]] = float(row['ratio'])
        mech_path = ROOT/'results/simulation_v52_dynamics_audit/mechanism.json'
        self.anchors = json.loads(mech_path.read_text(encoding='utf-8'))['closure_axes']
        self.pending = None
        self.last_step = 0
        self.failed = False
        self.closed = False
        self.summary = dict(schema='sim_v52d_telemetry_summary_v1', sample_count=0, records=0,
            last_step=0, phase_counts={}, telemetry_complete=False, max_mimic_error_rad=0.,
            max_gear_residual_rad=0., max_closure_error_m=0., forbidden_contact_ticks=0,
            first_forbidden_contact_step=None, motor_work_discrete_J=0.,
            gear_work_discrete_J=0., external_work_discrete_J=0.,
            max_abs_motor_torque_Nm=0., max_abs_gear_reaction_Nm=0.,
            last_object_state=None, collector_connect_calls=0, collector_step_calls=0)
        self.raw_file = (self.directory/'telemetry.jsonl.gz').open('xb')
        self.stream = gzip.GzipFile(filename='', fileobj=self.raw_file, mode='wb', compresslevel=1, mtime=0)
        try:
            readback = self._readback(infos)
            readback['mechanism_source'] = _identity(mech_path)
            _write_once(self.directory/'telemetry_readback.json', readback)
        except Exception:
            self.stream.close(); self.raw_file.close()
            raise

    def _optional(self, name, *args, **kwargs):
        try:
            return {'value': getattr(self.p, name)(*args, **kwargs), 'error': None}
        except Exception as exc:
            return {'value': None, 'error': repr(exc)}

    def _readback(self, infos):
        try:
            package_version = importlib.metadata.version('pybullet')
        except importlib.metadata.PackageNotFoundError:
            package_version = None
        dynamics = []
        bodies = [(self.body, list(range(-1, self.num_links))), (self.table, [-1])]
        if self.object_id is not None:
            bodies.append((self.object_id, [-1]))
        fields = ['mass_kg', 'lateral_friction', 'local_inertia_diagonal_kg_m2',
                  'local_inertial_position_link_m', 'local_inertial_orientation_xyzw',
                  'restitution', 'rolling_friction', 'spinning_friction',
                  'contact_damping', 'contact_stiffness', 'body_type', 'collision_margin_m']
        for body, links in bodies:
            for link in links:
                raw = self.p.getDynamicsInfo(body, link)
                values = {name: raw[i] if i < len(raw) else None for i, name in enumerate(fields)}
                values.update(body_id=body, link_index=link, raw=raw,
                    linear_damping_readback=None, angular_damping_readback=None,
                    unsupported_note='getDynamicsInfo does not expose linear/angular damping in this API')
                dynamics.append(values)
        constraint_fields = ['body_A', 'link_A', 'body_B', 'link_B', 'joint_type', 'axis',
            'parent_frame_position', 'child_frame_position', 'parent_frame_orientation',
            'child_frame_orientation', 'max_force', 'gear_ratio', 'gear_aux_link',
            'relative_position_target', 'erp']
        constraints = []
        for cid in self.robot['gear_ids']:
            raw = self.p.getConstraintInfo(cid)
            constraints.append(dict(id=cid, raw=raw, **{
                n: raw[i] if i < len(raw) else None for i, n in enumerate(constraint_fields)}))
        return dict(schema='sim_v52d_readback_v1', physics_steps_before_readback=0,
            python_version=platform.python_version(), python_executable=sys.executable,
            pybullet_distribution_version=package_version,
            pybullet_binary_files=[_identity(p) for p in (Path(sys.prefix)/'Lib/site-packages').glob('pybullet*.pyd')],
            robot_urdf_identity=_identity(ROOT/self.cfg['robot_urdf']),
            telemetry_source_identity=_identity(__file__),
            source_audit_identity=_identity(ROOT/'results/simulation_v52_dynamics_audit/solver_control.json'),
            engine=self._optional('getPhysicsEngineParameters'),
            api_version=self._optional('getAPIVersion'),
            joint_info_raw=[dict(index=i, value=v) for i, v in infos],
            mass_matrix_q_order=self.names, mass_matrix_joint_indices=self.indices,
            dynamics=dynamics, constraints=constraints,
            drive_settings_readback=None,
            collision_filters_readback=None,
            unsupported_fields_note='Motor gains/targets and collision filters are commands, not queryable complete readback; actual commands saved per tick',
            commanded_physics=self.cfg['physics'], commanded_collision_filters=self.cfg.get('allowed_self_collision_pairs'),
            timing_convention='pre_step reads state before integration; capture reads state after exactly one owner step; commands and Fpre apply during this interval',
            force_convention='Contact normalOnB points B to A. Estimated force on A = Fn*n + Ft1*t1 + Ft2*t2; force on B is its negative. Raw fields retained.',
            work_convention='Motor work=tau_post*(qpost-qpre); gear work=reaction_post*(e_post-e_pre); external work=Fpre dot displacement of same material point. These are discrete diagnostic estimates, not exact work.',
            mass_convention='Actual calculateMassMatrix at full measured scalar-joint q in qIndex order. Fixed-base twelve DOFs required; never use XML estimated M as readback.',
            collision_shapes={str(i): self._optional('getCollisionShapeData', self.body, i) for i in self.robot['pads']})

    def _states(self):
        raw = self.p.getJointStates(self.body, self.indices)
        self.latest_raw_joint_state = raw
        q = np.array([s[0] for s in raw]); dq = np.array([s[1] for s in raw])
        tau = np.array([s[3] for s in raw])
        _finite([q, dq, tau], 'joint state')
        return q, dq, tau, raw

    def _link(self, index, state=None):
        if index == -1:
            pos, quat = self.p.getBasePositionAndOrientation(self.body)
            vel, omega = self.p.getBaseVelocity(self.body)
            _finite([*pos,*quat,*vel,*omega], 'base link state')
            return dict(index=-1, name='BASE', com_world_m=pos, com_orientation_xyzw=quat,
                        link_frame_world_m=None, link_frame_orientation_xyzw=None,
                        com_linear_velocity_world_m_s=vel, angular_velocity_world_rad_s=omega)
        if state is None:
            state = self.p.getLinkState(self.body, index, computeLinkVelocity=1, computeForwardKinematics=1)
        _finite([*state[0],*state[1],*state[2],*state[3],*state[4],*state[5],*state[6],*state[7]], 'link state')
        return dict(index=index, name=self.link_names[index], com_world_m=state[0],
            com_orientation_xyzw=state[1], local_inertial_position_link_m=state[2],
            local_inertial_orientation_xyzw=state[3], link_frame_world_m=state[4],
            link_frame_orientation_xyzw=state[5], com_linear_velocity_world_m_s=state[6],
            angular_velocity_world_rad_s=state[7])

    def pre_step(self, step, phase, targets, external_loads):
        if self.closed or self.pending is not None or step != self.last_step + 1:
            raise ValueError('pre_step must precede exactly one next sequential physics step')
        q, dq, tau, _ = self._states()
        if 'nonfinite_value' in json.dumps(_json_value(targets)):
            raise ValueError('Nonfinite command target')
        loads = []
        for row in external_loads:
            index = int(row['link_index']); link = self._link(index)
            F = np.asarray(row['force_world_N'], dtype=float)
            point = np.asarray(row['point_world_m'], dtype=float)
            com = np.asarray(link['com_world_m']); lever = point-com
            _finite([F, point, com], 'external force')
            velocity = np.asarray(link['com_linear_velocity_world_m_s']) + np.cross(link['angular_velocity_world_rad_s'], lever)
            loads.append(dict(link_index=index, input=copy.deepcopy(row), force_pre_world_N=F,
                point_pre_world_m=point, com_pre_world_m=com, com_pre_orientation_xyzw=link['com_orientation_xyzw'],
                lever_pre_world_m=lever, torque_about_com_pre_Nm=np.cross(lever, F),
                point_material_local_com_m=_rotation(link['com_orientation_xyzw']).T@lever,
                point_velocity_pre_world_m_s=velocity, power_pre_W=float(F@velocity)))
        self.pending = dict(step=step, phase=phase, targets=copy.deepcopy(targets), q=q, dq=dq, tau=tau, loads=loads)

    def _command(self, index, targets):
        arm = targets['arm']; gripper = targets['gripper']
        if index in arm['joint_indices']:
            k = arm['joint_indices'].index(index)
            return dict(mode='POSITION_CONTROL', position_rad=arm['positions_rad'][k],
                velocity_rad_s=0., force_cap_Nm=arm['forces_Nm'][k],
                position_gain=arm['position_gains'][k], velocity_gain=arm['velocity_gains'][k],
                max_velocity_rad_s=arm.get('max_velocity_rad_s'), source='worker command, not engine setting readback')
        if index == gripper['master_joint_index']:
            return dict(mode='POSITION_CONTROL', position_rad=gripper['position_rad'],
                velocity_rad_s=0., force_cap_Nm=gripper['force_Nm'], position_gain=gripper['position_gain'],
                velocity_gain=gripper['velocity_gain'], max_velocity_rad_s=gripper['max_velocity_rad_s'],
                source='worker command, not engine setting readback')
        return dict(mode='VELOCITY_CONTROL', position_rad=None, velocity_rad_s=0.,
            force_cap_Nm=targets['followers']['force_Nm'], position_gain=None, velocity_gain=1.,
            max_velocity_rad_s=None, source='worker zero-force follower command')

    def _mass(self, q):
        M = np.asarray(self.p.calculateMassMatrix(self.body, q.tolist()), dtype=float)
        if M.shape != (12, 12):
            raise ValueError('Actual calculateMassMatrix shape does not match twelve scalar DOFs: '+str(M.shape))
        _finite(M, 'actual mass matrix')
        if not np.allclose(M, M.T, rtol=1e-9, atol=1e-10):
            raise ValueError('Actual mass matrix not symmetric')
        np.linalg.cholesky(M)
        W = np.linalg.solve(M, np.eye(12))
        rows = []
        for row, j in zip(self.gear_tree, self.J):
            a = self.order[row['a']]; b = self.order[row['b']]; ratio = float(row['ratio'])
            src = float(W[a,a]+ratio*ratio*W[b,b]); full = float(j@W@j)
            if src <= 0 or full <= 0:
                raise ValueError('Invalid gear row inverse inertia')
            rows.append(dict(a=row['a'], b=row['b'], ratio=ratio, d_source=src,
                d_full=full, d_full_over_d_source=full/src, cross_terms=full-src))
        return dict(source='calculateMassMatrix_at_measured_q', q_order=self.names,
            joint_indices=self.indices, q_rad=q.tolist(), matrix=M.tolist(), gear_rows=rows,
            limits='Loaded model generalized inertia, not an estimate of contact-constrained effective dynamics')

    def _closed_loops(self, links):
        rows = {}
        for side, spec in self.anchors.items():
            a = links[self.robot['links'][spec['knuckle_link']]]
            b = links[self.robot['links'][spec['finger_link']]]
            Ra = _rotation(a['link_frame_orientation_xyzw']); Rb = _rotation(b['link_frame_orientation_xyzw'])
            pa = np.asarray(a['link_frame_world_m']) + Ra@np.asarray(spec['anchor_in_knuckle_link_m'])
            pb = np.asarray(b['link_frame_world_m']) + Rb@np.asarray(spec['anchor_in_inner_finger_link_m'])
            aa = Ra@np.asarray(spec['axis_in_each_link']); ab = Rb@np.asarray(spec['axis_in_each_link'])
            rows[side] = dict(knuckle_anchor_world_m=pa, finger_anchor_world_m=pb,
                position_residual_world_m=pa-pb, position_error_m=float(np.linalg.norm(pa-pb)),
                knuckle_axis_world=aa, finger_axis_world=ab,
                axis_cross_norm=float(np.linalg.norm(np.cross(aa,ab))),
                unoriented_axis_angle_rad=math.acos(float(np.clip(abs(aa@ab),0,1))))
        return rows

    def _contacts(self, links, object_state):
        raw_contacts = self.p.getContactPoints()
        rows = []; wrench = {}; forbidden = []; pad_contacts = set(); support = False
        force_gate = self.cfg['criteria']['contact_force_n']; pen_gate = self.cfg['criteria']['penetration_m']
        base_allowed = self.cfg.get('base_table_allowed_link_indices', [])
        def center(body, link):
            if body == self.body: return np.asarray(links[link]['com_world_m'])
            if body == self.object_id: return np.asarray(object_state['position_world_m'])
            return np.asarray(self.p.getBasePositionAndOrientation(body)[0])
        for k, c in enumerate(raw_contacts):
            if len(c) < 14: raise ValueError('Expected full contact force tuple')
            _, A, B, la, lb, pa, pb, n, distance, fn, ft1, t1, ft2, t2 = c[:14]
            _finite([distance,fn,ft1,ft2], 'contact force')
            _finite([pa,pb,n,t1,t2], 'contact geometry')
            F = float(fn)*np.asarray(n)+float(ft1)*np.asarray(t1)+float(ft2)*np.asarray(t2)
            active = fn > force_gate or distance < -pen_gate
            category = 'other'; prohibited = False
            if A == self.body and B == self.body:
                category = 'robot_self'; prohibited = active
            elif self.body in (A,B) and self.table in (A,B):
                category = 'robot_table'; robot_link = la if A == self.body else lb
                prohibited = active and robot_link not in base_allowed
            elif self.object_id is not None and self.body in (A,B) and self.object_id in (A,B):
                category = 'robot_object'; robot_link = la if A == self.body else lb
                prohibited = active and robot_link not in self.robot['pads']
                if robot_link in self.robot['pads'] and fn > force_gate: pad_contacts.add(robot_link)
            elif self.object_id is not None and self.object_id in (A,B) and self.table in (A,B):
                category = 'object_table'
                support = support or fn > force_gate
            row = dict(contact_index=k, body_A=A, body_B=B, link_A=la, link_B=lb,
                point_A_world_m=pa, point_B_world_m=pb, normal_on_B_world=n,
                distance_m=distance, normal_force_N=fn, lateral_force_1_N=ft1,
                lateral_direction_1_world=t1, lateral_force_2_N=ft2, lateral_direction_2_world=t2,
                force_on_A_world_N=F, force_on_B_world_N=-F, category=category,
                active_by_original_gate=bool(active), forbidden=bool(prohibited), raw=c)
            rows.append(row)
            if prohibited: forbidden.append(k)
            for body, link, point, sign in [(A,la,pa,1.),(B,lb,pb,-1.)]:
                key=str(body)+':'+str(link)
                if key not in wrench: wrench[key]=dict(body_id=body,link_index=link,
                    about_com_world_m=center(body,link),force_world_N=np.zeros(3),torque_about_com_world_Nm=np.zeros(3),contact_count=0)
                w=wrench[key]; applied=sign*F
                w['force_world_N']+=applied
                w['torque_about_com_world_Nm']+=np.cross(np.asarray(point)-w['about_com_world_m'],applied)
                w['contact_count']+=1
        return rows, wrench, dict(forbidden_contact_indices=forbidden,
            forbidden_collision=bool(forbidden),bilateral_pad_contact=len(pad_contacts)==2,
            pad_links_with_normal_contact=sorted(pad_contacts),object_supported=bool(support) if self.object_id is not None else None)

    def capture(self, step, phase, targets=None, external_loads=None, elapsed_sim=None):
        pre = self.pending
        if pre is None or pre['step'] != step or pre['phase'] != phase:
            raise ValueError('capture requires matching pre_step')
        record = dict(schema='sim_v52d_tick_v1', step=int(step), phase=phase,
            time_s=step*self.dt, dt_s=self.dt, scenario=self.scenario, targets=pre['targets'])
        try:
            q, dq, tau, raw = self._states()
            joints = []
            for k, (index,name) in enumerate(zip(self.indices,self.names)):
                command = self._command(index,pre['targets']); cap=float(command['force_cap_Nm'])
                joints.append(dict(index=index,name=name,q_pre_rad=pre['q'][k],dq_pre_rad_s=pre['dq'][k],
                    applied_motor_torque_pre_Nm=pre['tau'][k],q_post_rad=q[k],dq_post_rad_s=dq[k],
                    applied_motor_torque_Nm=tau[k],joint_reaction_sensor_raw=raw[k][2],
                    reaction_sensor_enabled_by_collector=False,command=command,
                    joint_reaction_sensor_interpretation='Raw API tuple only; sensor not enabled here, not used as measured contact or actuator force',
                    observed_cap_fraction=abs(float(tau[k]))/cap if cap>0 else None,
                    at_cap_99pct=bool(abs(tau[k])>=.99*cap) if cap>0 else None,
                    power_post_W=float(tau[k]*dq[k]),work_discrete_J=float(tau[k]*(q[k]-pre['q'][k]))))
            record['joints']=joints
            link_states=self.p.getLinkStates(self.body,list(range(self.num_links)),computeLinkVelocity=1,computeForwardKinematics=1)
            links={i:self._link(i,s) for i,s in enumerate(link_states)};links[-1]=self._link(-1)
            record['links']=list(links.values())
            object_state=None
            if self.object_id is not None:
                pos,quat=self.p.getBasePositionAndOrientation(self.object_id);vel,omega=self.p.getBaseVelocity(self.object_id)
                _finite([*pos,*quat,*vel,*omega], 'object state')
                object_state=dict(body_id=self.object_id,position_world_m=pos,orientation_xyzw=quat,
                    linear_velocity_world_m_s=vel,angular_velocity_world_rad_s=omega)
            record['object_state']=object_state
            gears=[]
            for k,(spec,cid) in enumerate(zip(self.gear_tree,self.robot['gear_ids'])):
                state=self.p.getConstraintState(cid)
                if len(state)!=1: raise ValueError('Expected one gear reaction row')
                reaction=float(state[0]);_finite(reaction,'gear reaction')
                e0=float(self.J[k]@pre['q']);e1=float(self.J[k]@q);de=float(self.J[k]@dq)
                cap=float(self.cfg['physics']['gear_force'])
                gears.append(dict(id=cid,a=spec['a'],b=spec['b'],ratio=spec['ratio'],
                    e_pre_rad=e0,e_post_rad=e1,de_pre_rad_s=float(self.J[k]@pre['dq']),de_post_rad_s=de,
                    reaction_Nm=reaction,applied_impulse_Nm_s=reaction*self.dt,cap_Nm=cap,
                    at_cap_99pct=bool(abs(reaction)>=.99*cap),raw_state=state,
                    power_post_W=reaction*de,work_discrete_J=reaction*(e1-e0)))
            record['gears']=gears
            record['closed_loops']=self._closed_loops(links)
            contacts,wrenches,flags=self._contacts(links,object_state)
            record['contacts']=contacts;record['contact_wrenches']=wrenches
            loads=[]
            for row in pre['loads']:
                link=links[row['link_index']];com=np.asarray(link['com_world_m'])
                lever=_rotation(link['com_orientation_xyzw'])@row['point_material_local_com_m']
                postpoint=com+lever
                velocity=np.asarray(link['com_linear_velocity_world_m_s'])+np.cross(link['angular_velocity_world_rad_s'],lever)
                F=row['force_pre_world_N'];delta=postpoint-row['point_pre_world_m']
                loads.append(dict(**row,com_post_world_m=com,com_post_orientation_xyzw=link['com_orientation_xyzw'],
                    same_material_point_post_world_m=postpoint,lever_post_world_m=lever,
                    torque_about_com_post_Nm=np.cross(lever,F),point_velocity_post_world_m_s=velocity,
                    power_post_W=float(F@velocity),work_pre_rectangle_J=row['power_pre_W']*self.dt,
                    material_point_displacement_world_m=delta,work_discrete_J=float(F@delta)))
            record['external_loads']=loads
            record['mass_matrix']=self._mass(q)
            driver=q[self.order[self.manifest['master_joint']]]
            mimic={x['joint']:float(q[self.order[x['joint']]]-(driver*x['multiplier']+x['offset'])) for x in self.manifest['mimic_joints']}
            flags.update(mimic_residual_by_joint_rad=mimic,max_mimic_error_rad=max(abs(v) for v in mimic.values()),
                exceeds_original_mimic_gate=max(abs(v) for v in mimic.values())>self.cfg['criteria']['mimic_error_rad'])
            record['flags']=flags
            record['power_work']=dict(motor_power_post_W=sum(x['power_post_W'] for x in joints),
                motor_work_discrete_J=sum(x['work_discrete_J'] for x in joints),
                gear_power_post_W=sum(x['power_post_W'] for x in gears),gear_work_discrete_J=sum(x['work_discrete_J'] for x in gears),
                external_power_pre_W=sum(x['power_pre_W'] for x in loads),
                external_power_post_W=sum(x['power_post_W'] for x in loads),external_work_discrete_J=sum(x['work_discrete_J'] for x in loads))
            self._emit(record)
            self.last_step=step;self.pending=None
            s=self.summary;s['sample_count']+=1;s['records']+=1;s['last_step']=step
            s['phase_counts'][phase]=s['phase_counts'].get(phase,0)+1
            s['max_mimic_error_rad']=max(s['max_mimic_error_rad'],flags['max_mimic_error_rad'])
            s['max_gear_residual_rad']=max(s['max_gear_residual_rad'],max(abs(x['e_post_rad']) for x in gears))
            s['max_closure_error_m']=max(s['max_closure_error_m'],max(x['position_error_m'] for x in record['closed_loops'].values()))
            s['max_abs_motor_torque_Nm']=max(s['max_abs_motor_torque_Nm'],max(abs(x['applied_motor_torque_Nm']) for x in joints))
            s['max_abs_gear_reaction_Nm']=max(s['max_abs_gear_reaction_Nm'],max(abs(x['reaction_Nm']) for x in gears))
            if flags['forbidden_collision']:
                s['forbidden_contact_ticks']+=1
                if s['first_forbidden_contact_step'] is None:s['first_forbidden_contact_step']=step
            for key in ['motor_work_discrete_J','gear_work_discrete_J','external_work_discrete_J']:
                s[key]+=record['power_work'][key]
            s['last_object_state']=object_state
            return record
        except Exception as exc:
            self.failed=True
            record['capture_error']=repr(exc)
            record['joint_states_at_capture_error_raw']=getattr(self,'latest_raw_joint_state',None)
            self._emit(record,force_flush=True)
            raise

    def _emit(self, record, force_flush=False):
        data=json.dumps(_json_value(record),ensure_ascii=False,separators=(',',':'),allow_nan=False)
        self.stream.write((data+'\n').encode('utf-8'))
        if force_flush or record['step']%120==0:
            self.stream.flush();self.raw_file.flush()

    def finalize(self, status='completed'):
        if self.closed:return self.summary
        self.closed=True
        self.stream.close();self.raw_file.close()
        self.summary.update(status=status,telemetry_complete=not self.failed and self.pending is None,
            pending_step=None if self.pending is None else self.pending['step'],
            file=_identity(self.directory/'telemetry.jsonl.gz'),
            limits='Telemetry completeness is step continuity only; worker must also verify actual steps equal the approved unit budget. No scientific or engineering success inferred.')
        _write_once(self.directory/'telemetry_summary.json',self.summary)
        return self.summary

    def close(self):
        return self.finalize(status='closed')
