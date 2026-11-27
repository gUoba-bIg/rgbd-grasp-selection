"""Approved smoke09 finger speed references and independent measured-velocity limit (SI)."""
import math

FINGER_OPEN_COMMAND_SPEED_M_S = 0.040
FINGER_CLOSE_COMMAND_SPEED_M_S = 0.039
FINGER_OBSERVED_HARD_LIMIT_M_S = 0.040
MEASURED_SPEED_LIMIT_M_S = FINGER_OBSERVED_HARD_LIMIT_M_S
CLOSE_COMMAND_SPEED_M_S = FINGER_CLOSE_COMMAND_SPEED_M_S
PHASE_NAMES = ('settle','opening','transit','pregrasp','approach','close','lift','hold')


def phase_speed(phase):
    if phase not in PHASE_NAMES:
        raise ValueError('unknown_phase')
    return FINGER_CLOSE_COMMAND_SPEED_M_S if phase=='close' else FINGER_OPEN_COMMAND_SPEED_M_S


def validate_speeds(phase,reference_speed,motor_max_velocity):
    frozen=phase_speed(phase)
    if reference_speed != frozen:
        raise ValueError('invalid_reference_speed')
    if motor_max_velocity != frozen:
        raise ValueError('invalid_motor_max_velocity')


def finger_target(phase,tick,dt,pre_open_m,*,reference_speed=None):
    if not isinstance(tick,int) or tick<1 or not (0.<=pre_open_m<=.14 and math.isfinite(dt) and dt>0):
        raise ValueError('invalid_finger_command_input')
    speed=phase_speed(phase) if reference_speed is None else reference_speed
    validate_speeds(phase,speed,phase_speed(phase))
    half_width=pre_open_m/2
    if phase=='opening':
        return min(half_width,speed*tick*dt)
    if phase=='close':
        return max(0.,half_width-speed*tick*dt)
    if phase in ('transit','pregrasp','approach'):
        return half_width
    return 0.
