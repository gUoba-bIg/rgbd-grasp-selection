"""Choose diagnostic identities from frozen V1 evidence before any supplementary method execution."""
from collections import defaultdict,Counter
import sys
from common_v1 import *

def key(*args):
    return hashlib.sha256("|".join(map(str,(NAME,*args))).encode()).hexdigest()
def canonical(value):
    return hashlib.sha256(json.dumps(value,allow_nan=False,separators=(",",":"),sort_keys=True).encode()).hexdigest()
def freeze():
    need((OUT/"admission/v1_identity.json").is_file(),"unverified provenance")
    design=read(ROOT/R6FIRST/"preregistration/design.json");ledger=read(ROOT/R6NAME/"analysis/formal_ledger.json")
    need(len(ledger["rows"])==720 and sum(r["detail_class"]=="static_distance_uncertified" for r in ledger["rows"])==474,"cohort")
    origin=ROOT/R6NAME
    families={"uncertified":40,"success_control":10,"collision_control":10}
    populations={k:defaultdict(list) for k in families}
    earlier=("settle","opening","transit","pregrasp","approach")
    throughout={"nonpad_target_collision","simultaneous_nonpad_target_collision_at_entry","finger_independent_robot_target_collision","unregistered_self_collision","robot_table_collision"}
    for r in ledger["rows"]:
        s=r["static_first_failure"] or {}; leaves={v.split(":")[-1] for v in s.get("violations",[])}
        if r["detail_class"]=="static_distance_uncertified":group="uncertified"
        elif r["grasp_success"]:group="success_control"
        elif (r["forbidden_collision"] or bool(throughout&leaves) or
              (s.get("phase") in earlier and "pad_target_contact_before_close" in leaves)):group="collision_control"
        else:continue
        uid=r["unit_id"];i=read(origin/"trials"/r["trial_id"]/"intent.json")
        need(i["selected_p3"] and not i["r5_interface_in_chain"],"missing V1 selected input")
        sel=read(origin/"units"/uid/"selection.json")
        study=sel["eligibility"][i["selected_pool_index"]]
        need(study["eligible"] and study["candidate_id"]==i["selected_candidate_id"],"old E identity")
        pathhash=canonical(study["plan"])
        subject=key("within_unit_candidate_path",uid,i["selected_candidate_id"],i["selected_p3"]["camera_row_sha256"],pathhash,i["stable_state_sha256"])
        populations[group][subject].append(dict(trial=r["trial_id"],candidate=i["selected_candidate_id"],unit=uid,object=i["object_id"],condition=i["condition"],
          scene=i["scene_key"],method=r["method"],pose_sha256=i["selected_p3"]["camera_row_sha256"],path_sha256=pathhash,
          stable_state_sha256=i["stable_state_sha256"],intent_sha256=r["intent_sha256"],observation_sha256=r["observation_sha256"],
          old_result_sha256=r["result_sha256"],source_detail=r["detail_class"],source_phase=s.get("phase"),
          source_witness_labels=sorted(leaves&({"pad_target_contact_before_close"}|throughout)),worker_seed=i["worker_seed"]))
    seen=set();choices=[];strata=[]
    for label,count in families.items():
        ps=populations[label];need(not seen.intersection(ps),"cross-control subject overlap");seen.update(ps)
        g=defaultdict(list)
        for subject,members in ps.items():
            ordered=sorted(members,key=lambda x:key("representative",subject,x["trial"]))
            rep=ordered[0]
            strat=(rep["object"],rep["condition"],rep["method"])
            g[strat].append(dict(**rep,subject=subject,source_trials=[v["trial"] for v in ordered],
                source_methods=[v["method"] for v in ordered],draw_sha256=key("diagnostic",label,subject)))
        for rows in g.values():rows.sort(key=lambda x:x["draw_sha256"])
        strata_order=sorted(g,key=lambda s:key("stratum",label,*s))
        allocations=Counter()
        while sum(allocations.values())<count:
            before=sum(allocations.values())
            for s in strata_order:
                if allocations[s]<len(g[s]):allocations[s]+=1
                if sum(allocations.values())==count:break
            need(before<sum(allocations.values()),"insufficient distinct "+label+" controls")
        for s in strata_order:
            n=allocations[s];strata.append(dict(group=label,stratum=s,frame=len(g[s]),selected=n,
                original_trial_rows=sum(len(row["source_trials"]) for row in g[s])))
            for row in g[s][:n]:
                choices.append(dict(group=label,**row,stratum=s,inclusion_fraction=n/len(g[s])))
    choices.sort(key=lambda x:key("diagnostic_order",x["subject"]))
    for i,v in enumerate(choices,1):v["supplement_trial_id"]=f"ADM-{i:03}"
    need(len(choices)==60 and len({r["subject"] for r in choices})==60,"diagnostic unique trial count")
    write("preregistration/diagnostic_sample.json",dict(utc=utc(),population={k:len(v) for k,v in populations.items()},
          protocol_sha256=sha(WORK/"supplement/protocol.json"),source_ledger_sha256=sha(origin/"analysis/formal_ledger.json"),strata=strata,cases=choices))
    source_units={u["unit_id"]:u for u in design["units"]}
    jobs=[]
    for s in design["scenes"]:
        base=s["base_scene"];obs=origin/"scenes"/base/"observed_rgbd.npz";state=obs.parent/"settled_state.json"
        receipt=read(obs.parent/"receipt.json")
        need(sha(obs)==receipt["observation_rgbd_sha256"],"scene observation identity")
        for mode in MODES:
            original="clean" if mode=="clean" else "joint"
            copied=source_units[base+"-"+original]
            uid=base+"-"+mode
            jobs.append(dict(unit_id=uid,scene=base,condition=mode,condition_seed=s["condition_seed"][original],
                sampling_seed=s["sampling_seed"],scene_index=s["scene_index"],frozen_reference_unit=copied["unit_id"],
                worker_method_order=copied["method_order"],observation_path=obs.relative_to(ROOT).as_posix(),observation_sha256=receipt["observation_rgbd_sha256"],
                stable_state_path=state.relative_to(ROOT).as_posix(),stable_state_sha256=sha(state),
                screening_order_hash=key("robustness",uid),frozen_scene=s))
    need(len(jobs)==len({r["unit_id"] for r in jobs})==720,"six mode population")
    write("preregistration/observation_plan.json",dict(utc=utc(),protocol_sha256=sha(WORK/"supplement/protocol.json"),jobs=jobs))
    print("frozen diagnostic",dict(Counter(r["group"] for r in choices)),"and",len(jobs),"observations")
if __name__=="__main__":freeze()
