"""Offline receipt, continuation, and native-accounting tests. No simulator connection."""
import copy
from unittest.mock import patch

from static_identity import HERE, ORDER, PREVIOUS, read, sha, write_once, invariant
import candidate_binding20 as cb
import run_authorized as runner
from execution_outcome20 import returned_disposition

def main():
    cases=[]
    def check(name, predicate):
        if predicate is not True:raise AssertionError(name)
        cases.append(dict(name=name,passed=True))
    def rejects(name, fun):
        try:fun()
        except (KeyError,ValueError,RuntimeError):check(name,True);return
        raise AssertionError(name)
    base=read(HERE/"ledger_initial.json")
    check("11/14 start and three exact previously uncharged IDs",
          ORDER==(10,11,12) and base["starts"]==11 and invariant()==base and runner.ledger_before()==base)
    for n in ORDER:
        prior=read(PREVIOUS/"slot_ledger.json")["slots"][n-1];now=base["slots"][n-1]
        check(f"slot{n} retains previous nonadmission without changing source charge",
              not now["charged"] and now["state"]=="pending" and
              now["smoke21_record"]["static_receipt_sha256"]==prior["static_rejection_sha256"]
              and sha(PREVIOUS/prior["static_rejection"])==prior["static_rejection_sha256"]
              and now["candidate_id"]==cb.binding(n)["candidate_id"])
    rejects("nominal017 slot9 cannot be reserved",lambda:cb.binding(9))
    corrupt=copy.deepcopy(base["slots"][9]);corrupt["candidate_id"]="v52eng"
    rejects("wrong bridge identity fails",lambda:cb.assert_bound(corrupt,cb.binding(10)))
    def result(success,viols=()):
        n=23 if viols else 4728
        return dict(kind="miss",slot=10,grasp_success=success,episode_completed=not bool(viols),
                    engineering_gate_passed=not bool(viols),slot_passed=not success and not viols,
                    step_returns=n,step_intents=n,unknown_steps=0,step_counts={"total":n},
                    violations=list(viols),failure_reason=repr(RuntimeError(f"earliest_violation_at_step_{n}:{list(viols)!r}")),
                    evaluator_terminal_result=dict(first_failure=dict(step=n,violations=list(viols))))
    for success in (True,False):
        check("complete misgrasp continues" if success else "complete ungrasped miss continues",
              returned_disposition(result(success),4728)=="engineering_complete")
    for violation in ("nonpad_robot_object_contact","robot_table_contact","unregistered_self_manifold",
                      "external_penetration_below_minus_1mm","finger_speed_above_0.04_m_s"):
        check(violation+" candidate hard failure can continue",returned_disposition(result(False,[violation]),23)=="physical_episode_terminated_on_frozen_gate")
    for violation in ("unknown_contact_identity","wrong_object_link","filtered_pair_manifold","nonfinite_or_incomplete_contact",
                      "base_table_installed_pair_returned","step_index_discontinuity"):
        rejects(violation+" requires shared investigation",lambda:returned_disposition(result(False,[violation]),23))
    rec=result(False)
    for field,bad in (("step_intents",4729),("step_returns",4700),("unknown_steps",1),("slot_passed",False)):
        rejects("reject inconsistent "+field,lambda:returned_disposition({**rec,field:bad},4728))
    # Verify the actual process controller's chronology against a single synthetic prior charge.
    live=copy.deepcopy(base);r=live["slots"][9];r.update(charged=True,state="returned_failed",actual_step_calls=23)
    live.update(starts=12,remaining_total_budget=2,charged_in_actual_execution_order=base["charged_in_actual_execution_order"]+[10])
    def look(path):
        if path.name=="ledger_initial.json":return base
        if path==runner.LEDGER:return live
        if path.name=="session_result.json":return dict(batch_can_continue=True,charged=True,starts_before=11,starts_after=12)
        raise AssertionError(path)
    with patch.object(runner,"SLOT",11),patch.object(runner,"read",side_effect=look):
        check("candidate-level failed episode still permits exactly next slot",runner.ledger_before()==live)
        live["starts"]=11
        rejects("missing an observed native charge stops",runner.ledger_before)
    write_once(HERE/"accounting_control_replay22.json",dict(version=HERE.name,scope="offline accounting and disposition",
               passed=True,cases=cases,zero_connections=True,physical_steps=0,charged=0,
               bound_sources={name:sha(HERE/name) for name in ("accounting_control_replay22.py","execution_outcome20.py","run_authorized.py","batch22.py","candidate_binding20.py","dynamic_api.py","static_identity.py")}))
    print(len(cases),"passed")
if __name__=="__main__":main()
