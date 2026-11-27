"""Live bridge identity is inherited from pre-registered bytes. Never select a candidate."""
import copy
import hashlib
import json
import math
import shutil
from static_identity import HERE, PREVIOUS, ORDER, read, sha, write_once

PARENT_LEDGER = HERE/"parent_smoke20_ledger.json"
BINDINGS = HERE/"pending_candidate_identity.json"
RESULT_ID_FIELDS = ("slot","candidate_id","object_id","kind","candidate_vector_sha256")

def vector_digest(values):
    if (type(values) is not list or len(values)!=17 or
        any(type(x) not in (int,float) or not math.isfinite(x) for x in values)):
        raise ValueError("Malformed 17-element candidate")
    data=json.dumps(values, separators=(",",":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()

def binding(number):
    if number not in ORDER: raise ValueError("Slot outside current authorization")
    reg = read(HERE/"candidate_registry.json")
    rows=[row for row in reg["rows"] if row["slot"]==number]
    if len(rows)!=1:raise ValueError("Registry slot not unique")
    item=rows[0]
    for obj in item["files"]:
        if sha(HERE/obj["path"])!=obj["sha256"] or sha(PREVIOUS/obj["path"])!=obj["sha256"]:
            raise ValueError("Registered source changed: "+obj["path"])
    folder=HERE/"slots"/f"{number:02d}"
    spec=read(folder/"preregistration.json")["slots"][number-1]
    plan=read(folder/"trajectory_plan.json")
    proof=read(folder/"candidate_difference_proof.json")
    cid=item["candidate_id"]
    if (not cid.startswith("ip140bridge:") or spec["candidate_id"]!=cid or plan["candidate_id"]!=cid or
        proof["new_candidate_id"]!=cid or spec["parent_id"]!=proof["parent_id"] or
        proof["parent_id"]!=item["parent_candidate_id"]):
        raise ValueError("Registered candidate identity differs")
    h=vector_digest(spec["candidate_world"])
    if h!=vector_digest(plan["candidate_world"]) or h!=vector_digest(proof["new_candidate"]):
        raise ValueError("Registered candidate vector differs")
    for row in (spec,plan,proof):
        if any(row[k]!=item[k] for k in ("slot","object_id","kind")):raise ValueError("Slot/model/kind disagreement")
    return dict(slot=number,candidate_id=cid,kind=item["kind"],object_id=item["object_id"],
                candidate_vector_sha256=h,parent_candidate_id=proof["parent_id"],
                preregistration_sha256=sha(folder/"preregistration.json"),
                trajectory_sha256=sha(folder/"frozen_trajectory_commands.jsonl.gz"),candidate_registry_sha256=sha(HERE/"candidate_registry.json"))

def assert_bound(row, identity):
    for key in RESULT_ID_FIELDS:
        if row.get(key)!=identity[key]:raise ValueError(f"{key} does not match frozen bridge identity")
    return True

def initial_from_parent(parent, ids):
    ledger=copy.deepcopy(parent)
    if ledger["starts"]!=10 or ledger["remaining_total_budget"]!=4 or ledger["existing_budget"]!=14:
        raise ValueError("Parent budget differs")
    if (ledger["slots"][7].get("grasp_success") is not True or not ledger["slots"][7].get("charged") or
        ledger["slots"][7].get("actual_step_returns")!=4728):raise ValueError("Historical slot8 changed")
    ledger.update(version=HERE.name,schema="IP140_R2_smoke21_budget",prior_starts=10,starts_this_version=0,
                  prior_step_returns=parent["physical_steps"],steps_this_version=0,physical_step_calls_this_version=0,
                  unknown_physical_step_returns_this_version=0,terminal="pending_slots_09_12",
                  prior_final_ledger=dict(path=PARENT_LEDGER.name,sha256=sha(PARENT_LEDGER)))
    for i in ORDER:
        row=ledger['slots'][i-1]; identity=ids[str(i)]
        if row["charged"] or row["candidate_id"]!=identity["candidate_id"]:raise ValueError("Wrong prior pending slot")
        assert_bound(row,identity)
        row.update(identity,state="pending",expected_result_identity={k:identity[k] for k in RESULT_ID_FIELDS})
    return ledger

def rejected_ledger_row(prior, receipt):
    e=read(receipt)
    assert_bound(e,prior)
    if e["charged"] is not False or e["step_calls"]!=0 or e["static_admitted"] is not False or not e["rejection"]:
        raise ValueError("Malformed static rejection receipt")
    row=copy.deepcopy(prior)
    row.update(state="static_inadmissible_uncharged",static_admitted=False,
               actual_step_calls=0,actual_step_returns=0,unknown_step_returns=0,
               static_rejection=receipt.relative_to(HERE).as_posix(),static_rejection_sha256=sha(receipt),
               grasp_success=None,episode_completed=False,engineering_gate_passed=None,slot_passed=None)
    return row

def prepare():
    src=PREVIOUS/"slot_ledger.json"
    with PARENT_LEDGER.open("xb") as out, src.open("rb") as inp: shutil.copyfileobj(inp,out)
    ids={str(i):binding(i) for i in ORDER}
    write_once(BINDINGS,dict(version=HERE.name,encoding="UTF-8 compact JSON array of the original seventeen numeric values",slots=ids,
               source_ledger_sha256=sha(src),pending_result_semantics="Expected identity only until an observed result exists."))
    ledger=initial_from_parent(read(src),ids)
    write_once(HERE/"ledger_initial.json",ledger);write_once(HERE/"slot_ledger.json",ledger)

if __name__=="__main__":prepare()
