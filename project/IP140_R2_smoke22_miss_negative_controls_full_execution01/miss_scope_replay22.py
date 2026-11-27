"""Offline regression of the three named misses, the first-lift decision and static nonpad gates."""
import copy
import hashlib
import json
from pathlib import Path

from miss_scope22 import registered_miss, pre_lift_gate, enforce_nonpad_clearance
from contact_semantics import classify_contact

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"IP140_R2_smoke21_registered_pad_surface_and_resume_four01"

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def main():
    cases=[]
    def check(name,condition,**proof):
        cases.append(dict(name=name,passed=bool(condition),evidence=proof))
    def blocked(scope,roots):
        try:
            pre_lift_gate(roots,scope)
        except ValueError:
            return True
        return False
    for slot in (10,11,12):
        local=HERE/"slots"/str(slot)/"preregistration.json"
        old_dir=OLD/"slots"/str(slot)
        row=read(local)["slots"][slot-1]
        scoped=registered_miss(slot,row,sha(local))
        old=read(old_dir/"static_rejection.json")
        proof=old["rejection"]
        roots=read(old_dir/"static_preflight/left_right_contact_entry_roots.json")
        got=pre_lift_gate(roots["sides"],scoped)
        check(f"registered_miss_{slot}_old_static_rejection_only",scoped and sha(local)==sha(old_dir/"preregistration.json")
              and proof["last_static_snapshot"]==3216 and old["charged"] is False and old["step_calls"]==0
              and all(n==480 for n in proof["close_states_with_query"].values()) and len(proof["close_states_with_query"])==4
              and not roots["sides"] and got["pre_lift_bilateral_entry_absent"] and len(got["missing_entry_sides"])==2
              and proof["dual_channel_snapshots_sha256"]==sha(old_dir/"static_preflight/dual_channel_snapshots.jsonl.gz"),
              old_rejection_sha256=sha(old_dir/"static_rejection.json"), historical_evidence=proof,
              new_lift_transition=got)
        for key,bad in (("slot",slot-1),("kind","nominal"),("candidate_id","Raw"),
                        ("candidate_id","A"),("candidate_id","C0"),("not_p3",False),("object_id","unknown"),
                        ("candidate_world",row["candidate_world"][:-1]+[row["candidate_world"][-1]+1e-6])):
            other=copy.deepcopy(row);other[key]=bad
            check(f"{slot}_{key}_mutation_{str(bad)[:12]}",not registered_miss(slot,other,sha(local))
                  and blocked(registered_miss(slot,other,sha(local)),{}))
        check(f"{slot}_different_registration_denied",not registered_miss(slot,row,"0"*64))
    check("partial_real_entry_retained",pre_lift_gate({"right":{"passed":True}},True)==
          dict(registered_miss_scope=True,pre_lift_bilateral_entry_absent=True,missing_entry_sides=["left"],valid_entry_sides=["right"]))
    check("failed_existing_entry_never_excused",blocked(True,{"left":{"passed":False}}))
    check("unknown_certificate_side_never_excused",blocked(True,{"other":{"passed":True}}))
    check("nominal_cannot_use_missing_roots",blocked(False,{}))
    check("two_successful_roots_recorded",pre_lift_gate({"left":{"passed":True},"right":{"passed":True}},False)["pre_lift_bilateral_entry_absent"] is False)
    rootdir=OLD/"slots/10/static_preflight"
    links={k:v["index"] for k,v in read(rootdir/"zero_step/all_links_readback.json").items()}
    z=read(rootdir/"zero_readback.json")
    robot,target,table=[z[k] for k in ("body_id","object_body_id","table_body_id")]
    def classify(phase,source,rbody,rlink,other,d):
        row=[0,rbody,other,rlink,-1,[0.,0.,d],[0.,0.,0.],[0.,0.,1.],d,0.,0.,[1.,0.,0.],0.,[0.,1.,0.]]
        item=classify_contact(row,source=source,phase=phase,channel="proxy",
            robot=robot,target=target,table=table,links=links,entry_by_pad={})
        if phase in ("lift","hold"):enforce_nonpad_clearance(item,source)
        return item["violations"]
    for phase in ("lift","hold"):
        for link in ("ip140_palm","ip140_left_bone","upper_arm_link"):
            for source,d in (("robot_target",-1e-6),("manifold",0.)):
                errs=classify(phase,source,robot,links[link],target,d)
                check(f"{phase}_{link}_{source}_retained_hard_gate","nonpad_target_collision" in errs,violations=errs)
        check(phase+"_nearby_but_clear_bone",not classify(phase,"robot_target",robot,links["ip140_left_bone"],target,.003))
        check(phase+"_unregistered_environment_denied",bool(classify(phase,"manifold",robot,links["ip140_left_pad"],9,-.001)))
        check(phase+"_robot_table_contact_denied",bool(classify(phase,"manifold",robot,links["ip140_left_pad"],table,-.001)))
    passed=all(c["passed"] for c in cases)
    report=dict(version=HERE.name,passed=passed,case_count=len(cases),cases=cases,engine_connections=0,physical_steps=0,
                slot_starts=0,new_charges=0,source_sha256={p:sha(HERE/p) for p in (
                    "miss_scope_replay22.py","miss_scope22.py","contact_aware_path.py","static_admission_check.py","contact_semantics.py")})
    print(passed,len(cases),[c["name"] for c in cases if not c["passed"]])
    if __import__("sys").argv[1:]==["--seal"]:
        with (HERE/"miss_scope_replay22.json").open("x",encoding="utf8") as f:
            json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write("\n")
    return 0 if passed else 1

if __name__=="__main__":
    raise SystemExit(main())
