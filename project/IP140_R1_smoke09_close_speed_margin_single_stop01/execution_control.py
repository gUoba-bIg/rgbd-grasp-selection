"""IP140-R1 Rev02. Deterministic controller; there is no simulation launcher here.

Only encoder/robot feedback enters the action path. World/object identity is for
the independent evaluator. The caller must pass the zero-step gate before stepping.
Time is integral ticks of 1/240 s. Lengths in metres, forces in newtons.
"""
from dataclasses import dataclass, field
from math import isfinite

HZ = 240
from speed_policy import FINGER_OBSERVED_HARD_LIMIT_M_S, finger_target, phase_speed, validate_speeds
FORCE = 20.0
STABLE = 24
PHASES = (("settle", 240), ("opening", 480), ("transit", 672),
          ("pregrasp", 672), ("approach", 672), ("close", 480),
          ("lift", 792), ("hold", 720))

class Stop(Exception):
    """Reason and last samples must survive in the episode journal."""


def require_width(width_m):
    if not isfinite(width_m) or not 0 <= width_m <= .140:
        raise Stop("opening_out_of_range")
    return width_m


def gripper_command(client, body, joints, q_reference, *, phase):
    speed=phase_speed(phase)
    validate_speeds(phase,speed,speed)
    if not isfinite(q_reference) or not 0 <= q_reference <= .070:
        raise Stop("slider_reference_out_of_range")
    for joint in joints:
        client.setJointMotorControl2(body, joint, client.POSITION_CONTROL,
            targetPosition=q_reference, targetVelocity=0., positionGain=.10,
            velocityGain=1., maxVelocity=speed, force=FORCE)


def empty_errors(left_face_y, right_face_y, gaps_at_corners, width):
    if len(gaps_at_corners) != 4:
        raise Stop("four_measured_inner_face_gaps_required")
    if not all(isfinite(v) for v in (left_face_y, right_face_y, *gaps_at_corners)):
        raise Stop("nonfinite_inner_faces")
    gap = right_face_y - left_face_y
    errors = dict(gap_m=gap, signed_width_error_m=gap-width,
        left_error_m=left_face_y+width/2, right_error_m=right_face_y-width/2,
        centre_m=(left_face_y+right_face_y)/2, symmetry_error_m=abs(left_face_y+right_face_y),
        corner_spread_m=max(gaps_at_corners)-min(gaps_at_corners), min_gap_m=min(gaps_at_corners))
    good = (abs(gap-width)<=.001 and abs(errors['left_error_m'])<=.0005
        and abs(errors['right_error_m'])<=.0005 and abs(errors['centre_m'])<=.0005
        and errors['symmetry_error_m']<=.001 and errors['corner_spread_m']<=.0005
        and errors['min_gap_m']>=0)
    return errors, good


@dataclass
class ActionInput:
    actual_q: tuple
    actual_dq: tuple
    actual_motor_force: tuple
    external_pad_normal_max: tuple
    opening_in_tolerance: bool
    pose_error_m: float
    pose_error_rad: float
    endpoint_or_seam_fault: str = ""


@dataclass
class GraspCycle:
    width_m: float
    phase_index: int = 0
    tick_in_phase: int = 0
    total_ticks: int = 0
    in_tolerance: int = 0
    contact_state: tuple = (False, False)
    contact_age: int = 0
    stall_age: list = field(default_factory=lambda: [0, 0])
    closure_status: dict = field(default_factory=dict)

    def __post_init__(self):
        require_width(self.width_m)

    @property
    def phase(self):
        return PHASES[self.phase_index][0] if self.phase_index<len(PHASES) else 'complete'

    def command(self):
        """Next pre-step target; the caller logs command, then actual observations."""
        phase = self.phase
        if phase=='complete':
            raise Stop("cycle_completed")
        return finger_target(phase,self.tick_in_phase+1,1/HZ,self.width_m)

    def arm_fraction(self):
        """Arm interpolation for each preplanned waypoint, plus 0.3s settling."""
        if self.phase not in ('transit','pregrasp','approach','lift'):
            return None
        n = 720 if self.phase=='lift' else 600
        t = min(1., (self.tick_in_phase+1)/n)
        return t*t*(3.-2.*t)

    def observe(self, data, wall_seconds):
        phase = self.phase
        if phase=='complete':
            raise Stop("sample_after_completion")
        ref = self.command()
        if not all(len(x)==2 for x in (data.actual_q, data.actual_dq, data.actual_motor_force, data.external_pad_normal_max)):
            raise Stop("incomplete_bilateral_sample")
        if not all(isfinite(x) for x in (*data.actual_q,*data.actual_dq,*data.actual_motor_force,
                *data.external_pad_normal_max,data.pose_error_m,data.pose_error_rad,wall_seconds)):
            raise Stop("nonfinite_readback")
        if data.endpoint_or_seam_fault:
            raise Stop(data.endpoint_or_seam_fault)
        if wall_seconds>=120 or self.total_ticks+1>30*HZ:
            raise Stop("hard_timeout")
        if any(x < 0. or x > .070 for x in data.actual_q):
            raise Stop("prismatic_hard_limit")
        if any(abs(v)>FINGER_OBSERVED_HARD_LIMIT_M_S for v in data.actual_dq):
            raise Stop("measured_speed_over_limit")
        if any(abs(f)>FORCE+1e-9 for f in data.actual_motor_force):
            raise Stop("force_limit_disagreement")
        if phase=='opening':
            self.in_tolerance = self.in_tolerance+1 if data.opening_in_tolerance else 0
        if phase=='close':
            contact = tuple(f>.2 for f in data.external_pad_normal_max)
            self.contact_age = self.contact_age+1 if contact==self.contact_state else 1
            self.contact_state = contact
            for i,(q,v) in enumerate(zip(data.actual_q,data.actual_dq)):
                stalled = q>.0005 and q-ref>.0005 and abs(v)<.001
                self.stall_age[i] = self.stall_age[i]+1 if stalled else 0
            steady_contact = self.contact_age>=STABLE
            empty = all(abs(q)<=.0005 for q in data.actual_q) and not any(contact)
            stalled = [k>=STABLE for k in self.stall_age]
            window_expired = not (empty or (any(contact) and steady_contact) or any(stalled))
            self.closure_status = dict(stable_contact_sides=contact if steady_contact else None,
                empty_closure=empty, stall=stalled, force_saturated=[abs(f)>=19.8 for f in data.actual_motor_force],
                close_window_expired=self.tick_in_phase+1==480 and window_expired)
        self.total_ticks+=1
        self.tick_in_phase+=1
        if self.tick_in_phase<PHASES[self.phase_index][1]:
            return
        if phase=='opening' and self.in_tolerance<STABLE:
            raise Stop("preopening_timeout")
        if phase=='approach' and (data.pose_error_m>.005 or data.pose_error_rad>0.05235987755982989):
            raise Stop("approach_tracking_error")
        self.phase_index+=1
        self.tick_in_phase=0


def forbidden_self_contact(name_a, name_b, contact, close_endpoint, in_mounting_face=False):
    """in_mounting_face must be proved using both contact witnesses in CAD frames.
    Raw contact tuple fields 8/9 are signed distance (m) / simulated normal force (N).
    Never hide the whole link pair. For seams/closed pads, any negative distance or
    resolved normal impulse at the initial unloaded state is an immediate stop.
    """
    names = {name_a, name_b}
    distance, force = contact[8], contact[9]
    if not isfinite(distance) or not isfinite(force):
        return "nonfinite_contact_readback"
    pads = {"ip140_left_pad","ip140_right_pad"}
    finger_links = pads | {"ip140_left_bone","ip140_right_bone"}
    if names==pads and close_endpoint:
        if distance<0. or force>.2 or force>0. and close_endpoint=="initial":
            return "closed_pad_endpoint_artifact"
    if "ip140_palm" in names and names & finger_links:
        if distance<0. or force>0.:
            return "palm_finger_root_seam_contact"
    active = force>.2 or distance<-.001
    arm_chain=("base_link_inertia","shoulder_link","upper_arm_link","forearm_link",
               "wrist_1_link","wrist_2_link","wrist_3_link")
    adjacent={frozenset(pair) for pair in zip(arm_chain,arm_chain[1:])}
    adjacent.add(frozenset(("wrist_3_link","ip140_palm")))
    for side in ("left","right"):
        bone,pad=f"ip140_{side}_bone",f"ip140_{side}_pad"
        adjacent.update((frozenset((bone,pad)),frozenset(("ip140_palm",bone)),
                         frozenset(("ip140_palm",pad))))
    proven_seam = in_mounting_face and frozenset(names) in adjacent
    return "forbidden_self_contact" if active and not proven_seam else None
