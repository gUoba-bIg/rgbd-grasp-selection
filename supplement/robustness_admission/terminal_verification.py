"""Read-only final-window reconciliation from native contact rows and object positions."""
from common_v1 import *
from evidence_ledger import method_ledger

def reconcile(directory,result):
    count=result["physical_returns"]
    if not count:
        return dict(trial=result["trial_id"],physical_returns=0,physical_outcome_observed=False)
    episode=directory/"episode"
    raw=read(episode/"result.json")
    window=read(episode/"final_480_steps.json")
    need(sha(episode/"final_480_steps.json")==raw["terminal_observations_sha256"],"terminal window link")
    rows=window["observed_rows"]
    need(window["required_interval"]==[4249,4728] and window["actual_observed"]==len(rows) and
         window["steps_returned"]==raw["step_returns"]==count, "terminal window count")
    need([r["step"] for r in rows]==list(range(4249,count+1)), "terminal native indices")
    rec=dict(trial=result["trial_id"],physical_returns=count,physical_outcome_observed=True,
             completed=raw["trial_completed"],terminal_rows=len(rows),window_sha256=raw["terminal_observations_sha256"],
             source_telemetry_sha256=raw["telemetry_sha256"],actual_success=result["grasp_success"])
    if not raw["trial_completed"] or not raw["engineering_gate_passed"]:
        need(not result["grasp_success"],"engineering stop claiming success")
        return dict(rec,replayed_all_480=False,first_stop=raw.get("failure_reason"),engineering_gate_passed=raw["engineering_gate_passed"])
    need(count==4728 and len(rows)==480 and raw["engineering_gate_passed"],"incomplete returned physics")
    setup=directory/"dynamic_setup"
    world=read(setup/"zero_step/initial_world.json")
    links=read(setup/"robot_link_readback.json")
    robot,table=world["registered_robot"],world["registered_table"]
    target=raw["settle_baseline"]["body_id"]
    need(len({robot,table,target})==3 and world["registered_object"]==target,"terminal body ownership")
    pads=[links["ip140_"+s+"_pad"]["index"] for s in ("left","right")]
    baseline=raw["settle_baseline"]["position"][2]
    n_height=n_bilateral=n_support=n_slip=0
    lifts=[]
    for row in rows:
        need(row["phase"]=="hold" and not row["engineering_violations"] and row["object_state"]["body_id"]==target,"terminal engineering receipt")
        lift=row["object_state"]["position"][2]-baseline
        need(lift==row["lift_m"],"terminal raw rise vs settle height")
        hits=set();forces=[]
        for cp in row["raw_manifolds"]:
            a,b,la,lb=cp[1:5]
            roles={a:la,b:lb}
            if {a,b}=={robot,target}:
                need(roles[target]==-1 and roles[robot] in pads and cp[8]>=-.001,"unexpected completed robot contact")
                if cp[9]>.2:hits.add(roles[robot])
            elif {a,b}=={target,table}:
                need(la==lb==-1 and cp[8]>=-.001,"unexpected completed support contact")
                forces.append(max(cp[9],0.))
        aggregate=sum(forces)>.2
        need((pads[0] in hits)==row["effective_left_target_contact"] and (pads[1] in hits)==row["effective_right_target_contact"]
             and sum(forces)==row["table_normal_force_sum_N"] and aggregate==row["aggregate_table_support"],"terminal native contacts and support")
        n_height+=int(lift>=.10);n_bilateral+=int(len(hits)==2);n_support+=aggregate;n_slip+=int(row["slipped"]);lifts.append(lift)
    success=bool(n_height==n_bilateral==480 and n_support==n_slip==0 and raw["grasp_formed"])
    need(success==raw["grasp_success"]==result["grasp_success"],"terminal adjudication vs native data")
    return dict(rec,replayed_all_480=True,height_met=n_height,bilateral=n_bilateral,aggregate_support=n_support,
                slip_states=n_slip,min_terminal_lift_m=min(lifts),max_terminal_lift_m=max(lifts),recomputed_success=success)

def run():
    root=OUT/"study"
    ledger=read(method_ledger())
    need(ledger["first_shared_fault"] is None and ledger["assigned"]==ledger["adjudicated"],"unclosed method ledger")
    rows=[reconcile(root/"trials"/tid,read(root/"trials"/tid/"result.json")) for tid in sorted(ledger["returned"])]
    write("admission/final_window_crosscheck.json",dict(utc=utc(),source="returned 4249–4728 native rows, actual object position and settle baseline",
          assigned=len(rows),with_physics=sum(r["physical_outcome_observed"] for r in rows),
          with_complete_window=sum(r.get("replayed_all_480",False) for r in rows),per_trial=rows))
