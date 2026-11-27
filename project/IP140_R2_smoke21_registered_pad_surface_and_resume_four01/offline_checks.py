"""Synthetic setter audits and complete frozen command comparison. No engine sessions."""
import copy
import hashlib
import importlib
import math
import struct
from pathlib import Path
from types import SimpleNamespace

from damping_write import BINARY, DIRECT_MODE, OBJECT_URDF, apply_damping_write, validate_damping_write

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'IP140_R1_smoke12_slot07_base_table_mount_filter01'
PHASES = (('settle', 240), ('opening', 480), ('transit', 672), ('pregrasp', 672),
          ('approach', 672), ('close', 480), ('lift', 792), ('hold', 720))


class SyntheticSetterTrace:
    """The raw row layout of ApiTrace, with deliberately artificial IDs/timestamps."""
    def __init__(self):
        self.sequence = 2
        self.module = SimpleNamespace(__file__=str(BINARY))
        self.rows = [
            [1, 'call', 1, 'connect', [DIRECT_MODE], {}], [1, 'return', 2, 3],
            [2, 'call', 3, 'loadURDF', [OBJECT_URDF], {'physicsClientId': 3}], [2, 'return', 4, 74],
        ]

    def changeDynamics(self, *args, **kwargs):
        self.sequence += 1
        self.rows.extend([[self.sequence, 'call', 5, 'changeDynamics', list(args), kwargs],
                          [self.sequence, 'return', 6, None]])


def damping_cases():
    fake = SyntheticSetterTrace()
    prototype = apply_damping_write(fake, 74, 3)
    base = fake.rows
    def result(name, edit=None, expected_reason=None):
        rows, receipt = copy.deepcopy((base, prototype))
        if edit is not None:
            edit(rows, receipt)
        problems = validate_damping_write(receipt, 74, 3, rows)
        ok = (problems == []) if expected_reason is None else expected_reason in problems
        return dict(name=name, passed=ok, expected_rejection=expected_reason, actual_violations=problems,
                    write_trace=rows, field_evidence=receipt['fields'], synthetic=True)
    yield result('same_client_single_synchronous_audited_write')
    for field in ('linearDamping', 'angularDamping'):
        yield result('missing_' + field,
                     lambda r, s, k=field: r[4][5].pop(k), 'setter_arguments_or_sequence')
        yield result('wrong_' + field,
                     lambda r, s, k=field: r[4][5].update({k: math.nextafter(.04, math.inf)}),
                     'setter_arguments_or_sequence')
    yield result('wrong_body', lambda r,s: r[4][4].__setitem__(0, 75), 'object_changeDynamics_count')
    yield result('wrong_link', lambda r,s: r[4][4].__setitem__(1, 0), 'setter_arguments_or_sequence')
    yield result('wrong_client', lambda r,s: r[4][5].update(physicsClientId=4), 'setter_arguments_or_sequence')
    yield result('setter_exception', lambda r,s: r.__setitem__(5, [3, 'error', 6, 'synthetic exception']),
                 'setter_exception_missing_or_abnormal_return')
    yield result('missing_sync_return', lambda r,s: r.pop(), 'setter_exception_missing_or_abnormal_return')
    def append_write(rows, options):
        rows.extend([[4, 'call', 7, 'changeDynamics', [74, -1], options], [4, 'return', 8, None]])
    yield result('second_identical_object_write', lambda r,s: append_write(r, r[4][5].copy()), 'object_changeDynamics_count')
    yield result('later_angular_damping_override', lambda r,s: append_write(r, {'angularDamping': .08, 'physicsClientId': 3}),
                 'object_changeDynamics_count')
    yield result('late_other_property_write', lambda r,s: append_write(r, {'lateralFriction': .5, 'physicsClientId': 3}),
                 'object_changeDynamics_count')
    yield result('config_misrepresented_as_observation', lambda r,s: s['fields']['linearDamping'].update(observed=.04),
                 'receipt_identity_parameters_observation_or_provenance')
    yield result('contactDamping_misrepresented_as_rigid_body', lambda r,s: s['fields']['angularDamping'].update(
                 verification_mode='contactDamping', observed=.04), 'receipt_identity_parameters_observation_or_provenance')
    yield result('wrong_binary_receipt', lambda r,s: s.update(binary_sha256='0'*64),
                 'receipt_identity_parameters_observation_or_provenance')
    yield result('wrong_loaded_object', lambda r,s: r[2][4].__setitem__(0, '021_other.urdf'), 'object_load_identity_or_order')


def check_imports():
    names = sorted({x.stem for x in HERE.glob('*.py')})
    for name in names:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (HERE / (name + '.py')).resolve():
            raise ValueError(f'wrong import source {name}')
    needed = {'object_episode', 'object_evaluator', 'offline_replay', 'run_once', 'zero_scene', 'damping_write',
              'episode_log', 'full_api', 'input_identity', 'trajectory', 'speed_policy',
              'base_table_mount', 'mount_offline','aggregate_support','aggregate_support_replay'}
    return dict(name='all_local_entry_points_and_dependencies_import', passed=needed.issubset(names),
                modules=names, missing=sorted(needed-set(names)))


class MotorCommandCapture:
    POSITION_CONTROL = 2  # synthetic mode marker
    def __init__(self):
        self.arm = []
        self.fingers = []
    def setJointMotorControlArray(self, *args, **kwargs):
        self.arm.append((args, kwargs))
    def setJointMotorControl2(self, *args, **kwargs):
        self.fingers.append((args, kwargs))


def check_schedule():
    import trajectory
    from object_episode import log_motor_commands
    import speed_policy
    same = {}
    # Only speed policy remains byte-identical. New authorized height requires new IK plan.
    for name in ('speed_policy.py',):
        current = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        if current != hashlib.sha256((OLD/name).read_bytes()).hexdigest():
            raise ValueError(f'frozen source differs: {name}')
        same[name] = current
    plan = trajectory.validate_plan()
    if plan['candidate_id']!=trajectory.ID:
        raise ValueError('new slot14 candidate identity')
    if trajectory.PLAN.read_bytes()==(OLD/'trajectory_plan.json').read_bytes():
        raise ValueError('new height must not reuse old joint trajectory')
    same['new_trajectory_plan']=hashlib.sha256(trajectory.PLAN.read_bytes()).hexdigest()
    same['parent_trajectory_plan']=hashlib.sha256((OLD/'trajectory_plan.json').read_bytes()).hexdigest()
    if tuple(trajectory.PHASES) != PHASES or plan['pre_open_m'] != trajectory.read(trajectory.PLAN.parent/'preregistration.json')['slots'][plan['slot']-1]['pre_open_m'] or trajectory.DT != 1/240:
        raise ValueError('frozen stage schedule')
    half_width = plan['pre_open_m']/2
    arm_trajectory = hashlib.sha256()
    fingers = hashlib.sha256()
    count = 0
    phases = []
    synthetic_arm_ids = [1, 2, 3, 4, 5, 6]
    for phase, ticks in PHASES:
        start = count+1
        speed = .039 if phase in ('close','lift','hold') else .040
        if speed_policy.phase_speed(phase) != speed or speed_policy.FINGER_OBSERVED_HARD_LIMIT_M_S != .040:
            raise ValueError('phase command speed or independent hard limit')
        for tick in range(1, ticks+1):
            q, dq, finger = trajectory.reference(plan, phase, tick)
            if phase == 'opening':
                expected = min(half_width, .040*tick*trajectory.DT)
            elif phase == 'close':
                expected = max(0., half_width-.039*tick*trajectory.DT)
            elif phase in ('transit','pregrasp','approach'):
                expected = half_width
            else:
                expected = 0.
            if finger != expected or not 0 <= finger <= .07 or not all(map(math.isfinite, (*q,*dq))):
                raise ValueError(f'command reference at {phase}/{tick}')
            capture = MotorCommandCapture()
            log_motor_commands(capture, 3, 7, synthetic_arm_ids, (13,14), q, finger, [1.]*6,
                               phase=phase, reference_speed=speed, motor_max_velocity=speed)
            if (len(capture.arm) != 1 or capture.arm[0][1]['targetPositions'] != q.tolist()
                    or [c[0][1] for c in capture.fingers] != [13,14]):
                raise ValueError(f'command count or joint mapping at {phase}/{tick}')
            for _, options in capture.fingers:
                if options['maxVelocity'] != speed or options['targetPosition'] != finger or options['force'] != 20.:
                    raise ValueError(f'finger setter mismatch {phase}/{tick}')
            arm_trajectory.update(struct.pack('>12d', *q, *dq))
            fingers.update(struct.pack('>3d', finger, expected, speed))
            count += 1
        phases.append(dict(phase=phase, first=start, last=count, ticks=ticks, finger_max_velocity_m_s=speed))
    return dict(name='4728_frozen_openloop_references_and_per_tick_command_counts', passed=count==4728,
                ticks=count, synthetic_arm_array_calls=count, synthetic_finger_setter_calls=2*count,
                phases=phases, frozen_source_sha256=same, arm_reference_q_dq_binary64_be_sha256=arm_trajectory.hexdigest(),
                finger_reference_expected_cap_binary64_be_sha256=fingers.hexdigest(),
                note='Offline command construction on synthetic joint IDs; no physics calls.')

