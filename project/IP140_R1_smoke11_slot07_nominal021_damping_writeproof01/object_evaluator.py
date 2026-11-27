"""Object 021 evaluation from returned observations. Evaluation has no control output."""
from collections import deque
from copy import deepcopy
import json
from math import isfinite, hypot
from pathlib import Path
from structural_filter import STRUCTURAL_PAIRS
from speed_policy import FINGER_OBSERVED_HARD_LIMIT_M_S

ROOT=Path(__file__).resolve().parent.parent
CRITERIA=json.loads((ROOT/'configs/simulation_v52r2.json').read_bytes())['criteria']
PHASE_STEPS=(('settle',240),('opening',480),('transit',672),('pregrasp',672),
             ('approach',672),('close',480),('lift',792),('hold',720))
PHASE_AT_STEP=tuple(phase for phase,count in PHASE_STEPS for _ in range(count))
DT_S=1./240.
HOLD_STEPS=round(CRITERIA['hold_seconds']/DT_S)
LOSS_STEPS=round(CRITERIA['slip_loss_seconds']/DT_S)
LEFT_PAD='ip140_left_pad'
RIGHT_PAD='ip140_right_pad'

def _finite(values):
    return all(isinstance(v,(int,float)) and isfinite(v) for v in values)

def _contact_values(row):
    if len(row)!=14 or any(not isinstance(row[k],(list,tuple)) or len(row[k])!=3 for k in (5,6,7,11,13)):
        return None
    flat=(*row[:5],*row[5],*row[6],*row[7],*row[8:11],*row[11],row[12],*row[13])
    if not _finite(flat):return None
    vector=tuple(DT_S*sum(f*n[k] for f,n in zip((row[9],row[10],row[12]),(row[7],row[11],row[13]))) for k in range(3))
    return vector if _finite((*vector,hypot(*vector))) else None

def validate_filter_calls(robot_body,object_body,table_body,link_by_name,calls):
    reasons=[]
    if len({robot_body,object_body,table_body})!=3:reasons.append('body_identity_conflict')
    if len(calls)!=len(STRUCTURAL_PAIRS):reasons.append('filter_count')
    for i,call in enumerate(calls):
        a=call.get('args',[])
        if any(b in (object_body,table_body) for b in a[:2]):reasons.append('external_filter')
        if (len(a)!=5 or i>=len(STRUCTURAL_PAIRS) or a!=
            [robot_body,robot_body,*[link_by_name[n] for n in STRUCTURAL_PAIRS[i]],0]):
            reasons.append('filter_pair_or_order')
        if (call.get('sequence')!=i+1 or call.get('method')!='setCollisionFilterPair'
                or call.get('returned') is not True or 'return_value' not in call or call['return_value'] is not None):
            reasons.append('filter_call_or_return')
    return dict(passed=not reasons,violations=sorted(set(reasons)),count=len(calls))

class ObjectEvaluator:
    def __init__(self,robot_body,object_body,table_body,link_by_name):
        if len({robot_body,object_body,table_body})!=3 or min(robot_body,object_body,table_body)<0:
            raise ValueError('unique actual robot, object and table IDs required')
        self.robot=robot_body
        self.object=object_body
        self.table=table_body
        self.pads=(link_by_name[LEFT_PAD],link_by_name[RIGHT_PAD])
        self.by_id={i:n for n,i in link_by_name.items()}
        self.structural={frozenset((link_by_name[a],link_by_name[b])) for a,b in STRUCTURAL_PAIRS}
        self.observed=0
        self.failed=False
        self.first_failure=None
        self.initial=None
        self.grasp_formed=False
        self.slipped=False
        self.loss=0
        self.streak=0
        self.longest=0
        self.max_lift=None
        self.tail=deque(maxlen=HOLD_STEPS)
        self.observed_bilateral_steps=0
        self.terminal_counts=dict(observed=0,at_required_height=0,bilateral=0,table_supported=0,
                                  left_only=0,right_only=0,no_effective_pad=0)

    def _contact(self,row,index):
        c=dict(manifold_index=index)
        impulse=_contact_values(row)
        if impulse is None:return c,['nonfinite_or_incomplete_contact'],None,False
        c.update(body_links=[[row[1],row[3]],[row[2],row[4]]],signed_distance_m=row[8],
                 normal_force_N=row[9],lateral_1_N=row[10],lateral_2_N=row[12],
                 impulse_world_Ns=impulse,impulse_norm_Ns=hypot(*impulse),dt_s=DT_S)
        if row[1]==row[2]==self.robot:
            pair=frozenset((row[3],row[4]))
            if pair==set(self.pads):
                c['classification']='active opposing pads; independent numerical seam audit required'
                return c,[],None,False
            return c,['filtered_pair_manifold' if pair in self.structural else 'unregistered_self_manifold'],None,False
        problems=[]
        if row[8]<-CRITERIA['penetration_m']:problems.append('external_penetration_below_minus_1mm')
        roles={row[1]:row[3],row[2]:row[4]}
        if set(roles)=={self.robot,self.object}:
            link=roles[self.robot]
            if roles[self.object]!=-1:problems.append('wrong_object_link')
            if link not in self.pads:problems.append('nonpad_robot_object_contact')
            c['classification']='target_pad_contact' if link in self.pads else 'forbidden_robot_target_contact'
            c['robot_link_name']=self.by_id.get(link)
            effective=not problems and row[9]>CRITERIA['contact_force_n']
            c['effective_target_contact']=effective
            return c,problems,link if effective else None,False
        if set(roles)=={self.object,self.table}:
            if roles[self.object]!=-1 or roles[self.table]!=-1:problems.append('wrong_support_link')
            supported=not problems and row[9]>CRITERIA['contact_force_n']
            c.update(classification='target_table_support',effective_support=supported)
            return c,problems,None,supported
        c['classification']='forbidden_robot_table_contact' if set(roles)=={self.robot,self.table} else 'unknown_contact'
        problems.append('robot_table_contact' if set(roles)=={self.robot,self.table} else 'unknown_contact_identity')
        return c,problems,None,False

    def observe(self,step_index,phase,raw_manifolds,finger_states,object_state,tcp_world_position,
                *, engineering_violations=()):
        self.observed+=1
        problems=list(engineering_violations);contacts=[];pads=set();support=False;near=[]
        if step_index!=self.observed:problems.append('step_index_discontinuity')
        if self.observed>len(PHASE_AT_STEP) or phase!=PHASE_AT_STEP[self.observed-1]:problems.append('unexpected_phase_or_extra_step')
        if len(finger_states)!=2:problems.append('incomplete_finger_states')
        for side,s in zip(('left','right'),finger_states):
            if len(s)!=4 or not isinstance(s[2],(tuple,list)) or len(s[2])!=6 or not _finite((s[0],s[1],*s[2],s[3])):
                problems.append('nonfinite_or_incomplete_finger_state');continue
            if abs(s[1])>FINGER_OBSERVED_HARD_LIMIT_M_S:problems.append('finger_speed_above_0.04_m_s')
            if abs(s[3])>20.:problems.append('finger_motor_force_above_20_N')
            if abs(s[3])>=19.8:near.append(dict(side=side,actual_motor_force_N=s[3]))
        valid_state=isinstance(object_state,dict) and object_state.get('body_id')==self.object
        for key,size in (('position',3),('orientation',4),('linear_velocity',3),('angular_velocity',3)):
            a=object_state.get(key,[]) if isinstance(object_state,dict) else []
            valid_state=valid_state and isinstance(a,(tuple,list)) and len(a)==size and _finite(a)
        if not valid_state:problems.append('object_identity_or_nonfinite_state')
        valid_tcp=isinstance(tcp_world_position,(list,tuple)) and len(tcp_world_position)==3 and _finite(tcp_world_position)
        if not valid_tcp:problems.append('nonfinite_tcp')
        for i,row in enumerate(raw_manifolds):
            c,v,pad,supported=self._contact(row,i)
            contacts.append(c);problems.extend(v);support=support or supported
            if pad is not None:pads.add(pad)
        if self.observed==240 and phase=='settle' and valid_state:
            self.initial=deepcopy(object_state)
        lift=None
        separation=None
        if valid_state and self.initial is not None:
            lift=object_state['position'][2]-self.initial['position'][2]
            self.max_lift=lift if self.max_lift is None else max(self.max_lift,lift)
        if valid_state and valid_tcp:separation=hypot(*(a-b for a,b in zip(object_state['position'],tcp_world_position)))
        bilateral=set(self.pads).issubset(pads)
        if phase in ('close','lift','hold') and bilateral:
            self.observed_bilateral_steps+=1
        if len(PHASE_AT_STEP)-HOLD_STEPS<step_index<=len(PHASE_AT_STEP) and phase=='hold':
            counts=self.terminal_counts
            counts['observed']+=1
            counts['at_required_height']+=int(lift is not None and lift>=CRITERIA['lift_m'])
            counts['bilateral']+=int(bilateral)
            counts['table_supported']+=int(support)
            counts['left_only']+=int(pads=={self.pads[0]})
            counts['right_only']+=int(pads=={self.pads[1]})
            counts['no_effective_pad']+=int(not pads)
        self.loss=0 if bilateral else self.loss+1
        if phase in ('lift','hold') and bilateral and not support and lift is not None and lift>=CRITERIA['grasp_formed_lift_m']:
            self.grasp_formed=True
        if (self.grasp_formed and self.loss>=LOSS_STEPS and lift is not None and separation is not None and
                (lift<CRITERIA['grasp_formed_lift_m'] or separation>CRITERIA['slip_separation_m'])):
            self.slipped=True
        if problems and not self.failed:self.first_failure=dict(step=step_index,phase=phase,violations=list(dict.fromkeys(problems)))
        self.failed=self.failed or bool(problems)
        eligible=(phase=='hold' and lift is not None and lift>=CRITERIA['lift_m'] and
                  bilateral and not support and self.grasp_formed and not self.slipped and not self.failed)
        self.streak=self.streak+1 if eligible else 0
        self.longest=max(self.longest,self.streak)
        if eligible:
            self.tail.append(dict(step=step_index,lift_m=lift,effective_pad_links=sorted(pads),
                                  table_support=False,slipped=False))
        else:self.tail.clear()
        return dict(passed=not self.failed,engineering_passed=not self.failed,violations=list(dict.fromkeys(problems)),
                    contacts=contacts,lift_m=lift,bilateral_contact=bilateral,
                    effective_left_target_contact=self.pads[0] in pads,
                    effective_right_target_contact=self.pads[1] in pads,table_support=support,
                    grasp_formed=self.grasp_formed,slipped=self.slipped,bilateral_loss_steps=self.loss,
                    object_tcp_separation_m=separation,hold_streak=self.streak,eligible_hold_step=eligible,
                    near_cap=near,settle_baseline_frozen_this_step=self.observed==240 and self.initial is not None,
                    settle_baseline=self.initial if self.observed==240 else None)

    def finish(self):
        completed=self.observed==len(PHASE_AT_STEP)
        terminal=list(self.tail)
        valid_terminal=[r['step'] for r in terminal]==list(range(len(PHASE_AT_STEP)-HOLD_STEPS+1,len(PHASE_AT_STEP)+1))
        success=completed and not self.failed and valid_terminal and self.grasp_formed and not self.slipped
        failure=[]
        if self.failed:failure.extend(self.first_failure['violations'])
        if not completed:failure.append('incomplete_returned_steps')
        if self.initial is None:failure.append('settle_baseline_missing')
        if self.slipped:failure.append('object_slipped')
        if completed and not valid_terminal:
            failure.append('missing_final_480_valid_hold_steps')
            if self.observed_bilateral_steps==0:failure.append('no_bilateral_grasp_observed')
            if self.terminal_counts['at_required_height']<HOLD_STEPS:failure.append('terminal_height_requirement')
            if self.terminal_counts['bilateral']<HOLD_STEPS:failure.append('terminal_bilateral_contact_requirement')
            if self.terminal_counts['table_supported']:failure.append('terminal_table_support_observed')
        return dict(episode_completed=completed,engineering_gate_passed=completed and not self.failed,
                    grasp_success=success,slot_passed=success,failure_stage=self.first_failure['phase'] if self.failed else
                    ('hold' if completed and not success else None),failure_reason=failure,
                    observed_steps=self.observed,required_steps=len(PHASE_AT_STEP),settle_baseline=self.initial,
                    grasp_formed=self.grasp_formed,slipped=self.slipped,max_lift_m=self.max_lift,
                    final_hold_streak=self.streak,max_hold_streak=self.longest,terminal_interval=terminal,
                    eligible_terminal_samples_only=True,terminal_observation_counts=self.terminal_counts.copy(),
                    close_lift_hold_bilateral_steps=self.observed_bilateral_steps,
                    first_failure=self.first_failure)
