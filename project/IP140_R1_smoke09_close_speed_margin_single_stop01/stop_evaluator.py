"""Slot 13 corrective run post-step evaluator. Its outputs have no authority over trajectory or commands."""
from collections import deque
from math import hypot, isfinite
from structural_filter import STRUCTURAL_PAIRS
from speed_policy import FINGER_OBSERVED_HARD_LIMIT_M_S

PHASE_STEPS = (('settle',240),('opening',480),('transit',672),('pregrasp',672),
               ('approach',672),('close',480))
PHASE_AT_STEP = tuple(phase for phase,count in PHASE_STEPS for _ in range(count))
DT_S = 1.0/240.
FORBIDDEN_CLASS = 'forbidden pad/non-target contact'
STABLE_STEPS = 24
LEFT_PAD = 'ip140_left_pad'
RIGHT_PAD = 'ip140_right_pad'

def validate_filter_calls(robot_body, fixture_body, link_by_name, calls):
    """Compare observed setter records to the complete ordered frozen matrix."""
    reasons=[]
    if robot_body==fixture_body: reasons.append('body_identity_conflict')
    if len(calls)!=len(STRUCTURAL_PAIRS): reasons.append('filter_count')
    for i,call in enumerate(calls):
        args=call.get('args',[])
        if fixture_body in args[:2]: reasons.append('fixture_filter')
        if (len(args)!=5 or i>=len(STRUCTURAL_PAIRS) or
            list(args)!=[robot_body,robot_body,*[link_by_name[n] for n in STRUCTURAL_PAIRS[i]],0]):
            reasons.append('filter_pair_or_order')
        if (call.get('sequence')!=i+1 or call.get('method')!='setCollisionFilterPair'
                or call.get('returned') is not True or 'return_value' not in call
                or call['return_value'] is not None):
            reasons.append('filter_call_or_return')
    return dict(passed=not reasons,violations=sorted(set(reasons)),count=len(calls))

def _finite(values):
    return all(isinstance(v,(int,float)) and isfinite(v) for v in values)

def _contact_values(row):
    if len(row)!=14 or any(not isinstance(row[k],(list,tuple)) or len(row[k])!=3 for k in (5,6,7,11,13)):
        return None
    if not _finite((*row[:5],*row[5],*row[6],*row[7],*row[8:11],*row[11],row[12],*row[13])):
        return None
    return tuple(DT_S*(row[9]*row[7][i]+row[10]*row[11][i]+row[12]*row[13][i]) for i in range(3))

class StopEvaluator:
    def __init__(self, robot_body, fixture_body, link_by_name):
        if robot_body==fixture_body or min(robot_body,fixture_body)<0:
            raise ValueError('invalid body identities')
        self.robot=robot_body
        self.fixture=fixture_body
        self.by_id={idx:name for name,idx in link_by_name.items()}
        self.left=link_by_name[LEFT_PAD]
        self.right=link_by_name[RIGHT_PAD]
        self.structural={frozenset((link_by_name[a],link_by_name[b])) for a,b in STRUCTURAL_PAIRS}
        self.observed=0
        self.failed=False
        self.streak=0
        self.longest=0
        self.tail=deque(maxlen=STABLE_STEPS)
        self.first_failure=None

    def _classify(self, row, phase, occurrence):
        item=dict(manifold_index=occurrence)
        impulse=_contact_values(row)
        if impulse is None:
            return item, ['nonfinite_or_incomplete_contact'], False
        impulse_norm=hypot(*impulse)
        item.update(body_links=[list((row[1],row[3])),list((row[2],row[4]))],signed_distance_m=row[8],
                    normal_force_N=row[9],lateral_1_N=row[10],lateral_2_N=row[12],
                    impulse_world_Ns=impulse,impulse_norm_Ns=impulse_norm,dt_s=DT_S)
        if not _finite((*impulse,impulse_norm)): return item,['nonfinite_impulse'],False
        if row[1]==row[2]==self.robot:
            pair=frozenset((row[3],row[4]))
            if pair=={self.left,self.right}:
                item['classification']='active opposing pads, independent numerical seam audit required'
                return item,[],False
            return item,['filtered_pair_manifold' if pair in self.structural else 'unregistered_self_manifold'],False
        if (row[1],row[2])==(self.robot,self.fixture):
            finger,fixture_link=row[3],row[4]
        elif (row[2],row[1])==(self.robot,self.fixture):
            finger,fixture_link=row[4],row[3]
        else:
            return item,['unregistered_external_manifold'],False
        item['robot_link_name']=self.by_id.get(finger)
        item['classification']=FORBIDDEN_CLASS if finger in (self.left,self.right) else 'forbidden robot/non-target contact'
        reasons=[]
        if fixture_link!=-1: reasons.append('wrong_fixture_link')
        if phase!='close': reasons.append('fixture_contact_outside_close')
        if finger!=self.right: reasons.append('wrong_robot_link_on_fixture')
        if row[8]<-.001: reasons.append('penetration_below_minus_1mm')
        effective=not reasons and row[9]>.2
        item['registered_negative_event_evidence']=effective
        return item,reasons,effective

    def observe(self, step_index, phase, raw_manifolds, finger_states):
        """Call exactly once after a normal step return, with raw left/right joint states."""
        self.observed+=1
        reasons=[]; contacts=[];near=[];active_indices=[]
        if step_index!=self.observed: reasons.append('step_index_discontinuity')
        if self.observed>len(PHASE_AT_STEP) or phase!=PHASE_AT_STEP[self.observed-1]:
            reasons.append('unexpected_phase_or_extra_step')
        if phase in ('lift','hold'): reasons.append('lift_or_hold_forbidden')
        if len(finger_states)!=2: reasons.append('incomplete_finger_states')
        for side,st in zip(('left','right'),finger_states):
            if (len(st)!=4 or not isinstance(st[2],(tuple,list)) or len(st[2])!=6
                    or not _finite((st[0],st[1],*st[2],st[3]))):
                reasons.append('nonfinite_or_incomplete_finger_state');continue
            if abs(st[1])>FINGER_OBSERVED_HARD_LIMIT_M_S: reasons.append('finger_speed_above_0.04_m_s')
            if abs(st[3])>20.: reasons.append('finger_motor_force_above_20_N')
            if abs(st[3])>=19.8: near.append(dict(side=side,actual_motor_force_N=st[3]))
        for i,row in enumerate(raw_manifolds):
            item,failed,effective=self._classify(row,phase,i)
            contacts.append(item);reasons.extend(failed)
            if effective: active_indices.append(i)
        valid=bool(active_indices) and not reasons
        self.streak=self.streak+1 if valid else 0
        self.longest=max(self.longest,self.streak)
        if valid:
            self.tail.append(dict(step=step_index,manifold_indices=active_indices,
                                  Fn_N=[contacts[i]['normal_force_N'] for i in active_indices]))
        else:self.tail.clear()
        if reasons and not self.failed:self.first_failure=dict(step=step_index,violations=list(dict.fromkeys(reasons)))
        self.failed=self.failed or bool(reasons)
        return dict(passed=not self.failed,violations=list(dict.fromkeys(reasons)),contacts=contacts,
                    valid_right_contact=valid,right_streak=self.streak,near_cap=near,
                    observed_return=step_index,phase=phase)

    def finish(self):
        complete=self.observed==len(PHASE_AT_STEP)
        tail=list(self.tail)
        terminal_ok=([r['step'] for r in tail]==list(range(len(PHASE_AT_STEP)-STABLE_STEPS+1,len(PHASE_AT_STEP)+1)))
        event=complete and not self.failed and terminal_ok
        reasons=[]
        if self.failed:reasons.extend(self.first_failure['violations'])
        if not complete:reasons.append('incomplete_returned_steps')
        if not terminal_ok:reasons.append('missing_final_24_consecutive')
        return dict(passed=event,slot_passed=event,expected_negative_detected=event,grasp_success=False,
                    observed_steps=self.observed,required_steps=len(PHASE_AT_STEP),end_streak=self.streak,
                    longest_streak=self.longest,terminal_interval=tail,violations=reasons,
                    first_failure=self.first_failure)
