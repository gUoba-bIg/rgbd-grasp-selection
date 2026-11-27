"""Offline observations from the returned simulator states and their terminal detail."""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json
import math

PHASES = dict(settle=240, opening=480, transit=672, pregrasp=672, approach=672,
              close=480, lift=792, hold=720)
PLANNED = [(phase, k) for phase, count in PHASES.items() for k in range(1, count+1)]

def read(p):
    return json.loads(p.read_bytes())

def optional(p):
    return read(p) if p.is_file() else None

def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def point_delta(point, origin):
    return [a-b for a,b in zip(point, origin, strict=True)]

def observations(ep, observed, save_series):
    tape=ep/"telemetry.jsonl.gz"
    endfile=ep/"final_480_steps.json"
    if observed is not None and observed["telemetry_sha256"]!=sha(tape):
        raise ValueError("Telemetry SHA differs from episode result")
    window=optional(endfile)
    if observed is not None and (window is None or observed["terminal_observations_sha256"]!=sha(endfile)):
        raise ValueError("Recorded terminal detail mismatches episode result")
    tail={r["step"]:r for r in window["observed_rows"]} if window else {}
    if window and len(tail)!=window["actual_observed"]:
        raise ValueError("Duplicate or missing terminal detail")
    init=settled=last=None
    maxfrominit=maxfromsettle=None
    n=0; totals=Counter(); endtot=Counter(); phases=Counter()
    extent=[]; forces=[]; load_fraction=[]; lifts=[]; after_lift=[]; phase_counts={}
    limits=speeds=motors=torques=near_limit=at_limit=None
    pairs={}; first_problem=None; all_lifts=[]
    with gzip.open(tape,"rb") as raw, gzip.open(save_series,"xt",encoding="utf8",newline="\n",compresslevel=3) as out:
        for line in raw:
            row=json.loads(line)
            if row["event"]=="initial_configuration":
                if init is not None:raise ValueError("Duplicate initial observation")
                init=row["initial_object_state"];obj=row["evaluator_context"]["object_body_id"]
                table=row["evaluator_context"]["table_body_id"]; robot=row["body"]
                links={i:name for name,i in row["links"].items()}
                limits=row["arm_effort_caps_Nm"]
                speeds=[0.,0.]; motors=[0.,0.]; torques=[0.]*6;near_limit=[0,0];at_limit=[0,0]
                continue
            if row["event"]!="state":continue
            n+=1
            if init is None or row["index"]!=n or (row["phase"],row["phase_tick"])!=PLANNED[n-1]:
                raise ValueError("Returned observation index, phase or identity missing")
            position=row["object_state"]["position"]; last=row["object_state"]
            if n==240:settled=last
            delta=point_delta(position,init["position"]);dist=math.hypot(*delta)
            if maxfrominit is None or dist>maxfrominit["m"]:maxfrominit=dict(step=n,m=dist)
            stable_delta=None
            if settled is not None:
                stable_delta=point_delta(position,settled["position"])
                dist=math.hypot(*stable_delta)
                if maxfromsettle is None or dist>maxfromsettle["m"]:maxfromsettle=dict(step=n,m=dist)
            ev=row["object_evaluation"]; phases[row["phase"]]+=1
            lev=ev["effective_left_target_contact"];rev=ev["effective_right_target_contact"]
            num_pad=nonpad=0; forcepeak=None; contacts=Counter()
            for c in row["raw_manifolds"]:
                a,b,la,lb=c[1:5]
                if a==b:continue
                def name(body,link):
                    if body==obj:return f"target.{link}"
                    if body==table:return f"table.{link}"
                    if body==robot:return "robot."+links.get(link,str(link))
                    return f"unknown.{body}.{link}"
                tag=" / ".join(sorted((name(a,la),name(b,lb))))
                p=pairs.setdefault(tag,dict(points=0,min_signed_distance_m=None,max_Fn_N=None))
                p["points"]+=1;contacts[tag]+=1
                if p["min_signed_distance_m"] is None or c[8]<p["min_signed_distance_m"]:p["min_signed_distance_m"]=c[8]
                if p["max_Fn_N"] is None or c[9]>p["max_Fn_N"]:p["max_Fn_N"]=c[9]
                if {a,b}!={obj,robot}:continue
                if links.get(la if a==robot else lb) in ("ip140_right_pad","ip140_left_pad"):
                    num_pad+=1;forcepeak=c[9] if forcepeak is None else max(forcepeak,c[9])
                else:nonpad+=1
            extra=dict(left_effective=int(lev),right_effective=int(rev),bilateral=int(lev and rev),
                target_pad_manifold_steps=int(num_pad>0),nonpad_target_steps=int(nonpad>0),
                aggregate_table_support=int(ev["aggregate_table_support"]),
                pointwise_table_support=int(ev["table_support"]),table_contact_present=int(ev["table_contact_present"]),
                eligible=int(ev["eligible_hold_step"]),slipped=int(ev["slipped"]),gate_violation=int(bool(row["violations"])))
            totals.update(extra);phase_counts.setdefault(row["phase"],Counter()).update(extra)
            value=ev["lift_m"]
            if value is not None:
                if stable_delta is not None and abs(value-stable_delta[2])>1e-12:raise ValueError("Lift and state position differ")
                all_lifts.append(value)
            if row["phase"] in ("lift","hold") and value is not None:after_lift.append(value)
            if row["violations"] and first_problem is None:first_problem=dict(step=n,violations=row["violations"])
            for i,(speed,effort) in enumerate(zip(row["actual_speed_m_s"],row["sim_motor_force_N"],strict=True)):
                speeds[i]=max(speeds[i],abs(speed)); motors[i]=max(motors[i],abs(effort))
                near_limit[i]+=abs(effort)>=19.8; at_limit[i]+=abs(effort)>=20
            torques=[max(t,abs(x)) for t,x in zip(torques,row["arm_torque_Nm"],strict=True)]
            if n>=4249:
                endtot.update(extra);t=tail.get(n)
                for key in ("lift_m","effective_left_target_contact","effective_right_target_contact","aggregate_table_support",
                            "table_normal_force_sum_N","slipped"):
                    if window and (not t or t[key]!=ev[key]):raise ValueError(f"Terminal detail mismatch at {n}/{key}")
                if window and (t["object_state"]!=last or t["raw_manifolds"]!=row["raw_manifolds"] or t["eligible"]!=ev["eligible_hold_step"]):
                    raise ValueError(f"Terminal native state or contact mismatch at {n}")
                if value is not None:lifts.append(value)
                forces.append(ev["table_normal_force_sum_N"]);load_fraction.append(ev["table_load_fraction"])
            out.write(json.dumps(dict(step=n,phase=row["phase"],pose=last,delta_from_initial_xyz_m=delta,
                delta_from_settle_xyz_m=stable_delta,lift_m=value,table_sum_Fn_N=ev["table_normal_force_sum_N"],
                table_load_fraction=ev["table_load_fraction"],external_contacts=contacts,
                pad_peak_contact_Fn_N=forcepeak,forbidden_nonpad_points=nonpad,flags=extra),allow_nan=False,separators=(",",":"))+"\n")
    def limits_of(xs):return [min(xs),max(xs)] if xs else None
    if observed is not None and n!=observed["observed_steps"]:
        raise ValueError("State sequence count and episode record differ")
    if window and len(forces)!=len(tail):
        raise ValueError("Terminal state count and terminal file differ")
    spacing=None if last is None else point_delta(last["position"],init["position"])
    after=None if last is None or settled is None else point_delta(last["position"],settled["position"])
    return dict(observed_states=n,states_by_phase=phases,saved_terminal_window=len(tail) if window else None,
        initial_object_state=init,settle_240_object_state=settled,last_observed_object_state=last,
        displacement_initial_to_last_xyz_m=spacing,displacement_settle_to_last_xyz_m=after,
        max_translation_from_initial=maxfrominit,max_translation_from_settle=maxfromsettle,
        all_relative_lift_range_m=limits_of(all_lifts),lift_hold_relative_lift_range_m=limits_of(after_lift),
        full_contact_state_counts=totals if n else None,by_phase=phase_counts,external_contact_pairs=pairs if n else None,
        first_engineering_violation=first_problem,
        terminal480_counts=endtot if forces else None,terminal480_lift_range_m=limits_of(lifts),
        terminal480_table_sum_Fn_range_N=limits_of(forces),terminal480_table_load_fraction_range=limits_of(load_fraction),
        fingers_L_R=dict(peak_abs_speed_m_s=speeds,peak_abs_motor_force_N=motors,near_19p8_N_steps=near_limit,at_20_N_steps=at_limit) if n else None,
        arm_peak_abs_torque_Nm=torques if n else None,arm_torque_margin_Nm=[a-b for a,b in zip(limits,torques)] if n else None,
        telemetry_sha256=sha(tape), terminal_sha256=sha(endfile) if window else None, derived_series_sha256=sha(save_series))
