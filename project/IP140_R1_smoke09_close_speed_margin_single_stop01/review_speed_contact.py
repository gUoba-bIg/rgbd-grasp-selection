"""Reconstruct the two slider commands, measurements and final 24 steps from closed logs; no engine imports."""
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_bytes())


def stream(name):
    with gzip.open(HERE / name, "rt", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def review():
    reg = read("preregistration.json")
    spec = reg["speed_spec"]
    counts = reg["required_completion"]["phases"]
    phases = [(phase, tick) for phase,count in counts.items() for tick in range(1, count+1)]
    assert len(phases)==3216 and counts["close"]==480
    model = read("zero_readback.json")
    dt = model["physics_readback"]["fixedTimeStep"]
    robot,fixture = model["body_id"],model["fixture_body_id"]
    events = stream("episodes/13/telemetry.jsonl.gz")
    setup = next(row for row in events if row["event"]=="initial_configuration")
    all_joints=setup["joint_indices"]
    pad=setup["links"]["ip140_right_pad"]
    force_cap=reg["stop_event"]["finger_force_max_N"]
    joint_commands, raw_states, raw_contacts = {}, {}, {}
    last_setters, pending = {}, None
    step=0
    for row in stream("full_api/calls.jsonl.gz"):
        if row[1]=="call":
            pending=row
            continue
        assert pending[0]==row[0] and row[1]=="return"
        method,args,kw=pending[3:6]; val=row[3]
        if method=="setJointMotorControl2" and len(args)>=2 and args[0]==robot and args[1] in all_joints[-2:]:
            last_setters[args[1]]={"seq":row[0],"kwargs":kw}
        elif method=="stepSimulation":
            step+=1
            assert set(last_setters)==set(all_joints[-2:])
            joint_commands[step]=[last_setters[idx] for idx in all_joints[-2:]]
            last_setters={}
        elif step and method=="getJointStates" and args==[robot,all_joints]:
            raw_states[step]=val
        elif step and method=="getContactPoints" and not args and set(kw)<= {"physicsClientId"}:
            raw_contacts[step]=val
        pending=None
    phase_stats=defaultdict(lambda:dict(encoder_steps=0,returned_finger_setters=0,submitted_maxVelocity=set(),
        max_abs_dq_m_s=0.,max_abs_motor_N=0.,right_at_or_near_19_8_N_steps=0))
    last24=[];first_contact=None;first_effective=None;first_failure=None
    valid_streak=0;max_streak=0;max_fn=None;min_distance=None
    early_returns=0;activity=Counter()
    for s in events:
        if s["event"]!="state":continue
        i=s["index"];phase,tick=phases[i-1];assert (phase,tick)==(s["phase"],s["phase_tick"])
        speed=spec["close_command"] if phase=="close" else spec["opening_command"]
        t=tick*dt;half=reg["slots"][12]["pre_open_m"]/2
        if phase=="settle": expected=0.0
        elif phase=="opening": expected=min(half,speed*t)
        elif phase=="close": expected=max(0.,half-speed*t)
        else: expected=half
        for c in joint_commands[i]:
            v=c["kwargs"]
            assert v["maxVelocity"]==speed and v["force"]==force_cap
            assert v["positionGain"]==.1 and v["velocityGain"]==1. and v["targetVelocity"]==0.
            # Identical scalar slope and target; allow differing multiplication association in recomputation only.
            assert math.isclose(v["targetPosition"],expected,rel_tol=0.,abs_tol=2e-17)
            phase_stats[phase]["submitted_maxVelocity"].add(v["maxVelocity"])
            phase_stats[phase]["returned_finger_setters"]+=1
        assert s["raw_manifolds"]==raw_contacts[i] and s["actual_arm_and_finger_states"]==raw_states[i]
        f=raw_states[i][-2:]; dq=[v[1] for v in f]; effort=[v[3] for v in f]
        info=phase_stats[phase];info["encoder_steps"]+=1
        info["max_abs_dq_m_s"]=max(info["max_abs_dq_m_s"],*map(abs,dq))
        info["max_abs_motor_N"]=max(info["max_abs_motor_N"],*map(abs,effort))
        info["right_at_or_near_19_8_N_steps"]+=int(abs(effort[1])>=19.8)
        external=[p for p in raw_contacts[i] if p[1]!=robot or p[2]!=robot]
        right_rows=[]; valid_rows=[]
        for j,p in enumerate(external):
            if p[1:5] in ([robot,fixture,pad,-1],[fixture,robot,-1,pad]):
                right_rows.append(j)
                if p[9]>.2 and p[8]>=-.001:valid_rows.append(j)
                max_fn=max(max_fn,p[9]) if max_fn is not None else p[9]
                min_distance=min(min_distance,p[8]) if min_distance is not None else p[8]
        clean=not s["violations"] and all(abs(v)<=spec["observed_hard_limit"] for v in dq) and all(abs(v)<=force_cap for v in effort)
        good=(phase=="close" and clean and bool(valid_rows) and len(right_rows)==len(external))
        valid_streak=valid_streak+1 if good else 0
        max_streak=max(max_streak,valid_streak)
        row=dict(global_step=i,phase=phase,phase_tick=tick,reported_finger_motor_N=effort,measured_finger_dq_m_s=dq,
                 returned_commands=joint_commands[i],externals=external,
                 registered_right_external_indices=right_rows,effective_right_external_indices=valid_rows,
                 hard_violations=s["violations"],all_conditions_met=good)
        if external:
            activity["external_contact_steps"]+=1
            if first_contact is None:first_contact=row
        activity["steps_with_qualifying_right_point"]+=bool(valid_rows)
        if good and first_effective is None:first_effective=row
        if s["violations"] and first_failure is None:first_failure=row
        if i>=3193:last24.append(row)
        early_returns+=1
    for v in phase_stats.values():v["submitted_maxVelocity"]=sorted(v["submitted_maxVelocity"])
    result=read("episodes/13/result.json")
    assert early_returns==step==result["step_returns"]
    if result["slot_passed"]:
        assert valid_streak>=24 and len(last24)==24 and all(v["all_conditions_met"] for v in last24)
    return dict(version=HERE.name,returned_steps=step,declared_command_and_observed_limits=spec,actual_by_phase=dict(phase_stats),
                source_api="full_api/calls.jsonl.gz",source_states="episodes/13/telemetry.jsonl.gz",contacts_summary=activity,
                first_external=first_contact,first_fully_valid_right_step=first_effective,first_failure=first_failure,
                original_required_terminal_steps=[3193,3216],observed_terminal_rows=last24,
                observed_terminal_row_count=len(last24),unobserved_terminal_row_count=24-len(last24),
                fully_valid_ending_streak=valid_streak,longest_fully_valid_streak=max_streak,
                raw_max_contact_point_Fn_N=max_fn,raw_min_right_signed_distance_m=min_distance,
                required_terminal_24_passed=(len(last24)==24 and all(v["all_conditions_met"] for v in last24)),
                result={k:result[k] for k in ("slot_passed","expected_negative_detected","grasp_success")})


if __name__=="__main__":
    print(json.dumps(review(),ensure_ascii=False,indent=2,allow_nan=False))
