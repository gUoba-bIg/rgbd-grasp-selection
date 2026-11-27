"""Perform exactly one offline admission and, conditionally, one live slot 7 session."""
import json
import os
import sys
import traceback
from pathlib import Path
from input_identity import HERE, ROOT, REGISTERED_ID, digest, read, timestamp, verify, write_once


def main():
    if Path.cwd().resolve() != ROOT or sys.argv[1:] != ["--offline-zero-and-slot-7"]:
        raise SystemExit("Require the explicit project root and --offline-zero-and-slot-7")
    expected_python = ROOT / read(ROOT / "configs/simulation_v52e.json")["python"]
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise SystemExit("Use the frozen v52e1 interpreter")
    blockers = ["session_claim.json","offline_gate_replay.json","STOP.json","result.json",
                "full_api","episodes","zero_step","zero_readback.json","slot_ledger.json"]
    if any((HERE / name).exists() for name in blockers):
        raise SystemExit("Existing claim or output prevents a second run")
    initial=read(HERE / "ledger_initial.json")
    status=dict(version=HERE.name,slot=7,at=timestamp(),phase="identity",charged=False)
    write_once(HERE / "session_claim.json", dict(pid=os.getpid(), start=timestamp(),
               allowed_slot=7, input_identity_sha256=digest(HERE/"input_identity.json")))
    try:
        verify()
        status['phase'] = "offline_replay"
        from offline_replay import persist_report
        replay = persist_report()
        if (replay.get('passed') is not True or not replay.get('cases')
                or not all(r.get('passed') is True for r in replay['cases'])):
            status['stop'] = "offline_replay_failed"
            return
        verify()
        status['phase'] = "zero"
        from zero_scene import gate_valid, run_zero
        def admitted(p, client, body, links, model, registration, gate):
            persisted=read(HERE/"zero_readback.json")
            if (not gate_valid(gate) or not gate_valid(persisted) or persisted != gate
                    or gate['physics_step_calls'] != 0 or p.counts['stepSimulation'] != 0
                    or gate.get('object_body_id') is None or gate.get('table_body_id') is None
                    or len({body,gate['object_body_id'],gate['table_body_id']}) != 3
                    or registration['slots'][6]['candidate_id'] != REGISTERED_ID
                    or registration != read(HERE/'preregistration.json')):
                raise RuntimeError("persisted live admission and callback disagree")
            verify()
            dest=HERE/"episodes"/"07"
            dest.mkdir(parents=True,exist_ok=False)
            write_once(dest/"scene_state.json", dict(client=client,robot=body,object=gate['object_body_id'],
                       table=gate['table_body_id'],utc=timestamp(),zero_sha256=digest(HERE/"zero_readback.json")))
            def charge():
                if (p.counts['stepSimulation'] or p.step_authorized or status['charged']
                        or not p.protected_write_started or p.protected_object != (gate['object_body_id'], client)):
                    raise RuntimeError("first-step billing guard")
                record=dict(slot=7,candidate_id=REGISTERED_ID,not_p3=True,starts_before=initial['starts'],
                            starts_after=8,remaining_after=5,total=13,ts=timestamp(),
                            zero_sha256=digest(HERE/'zero_readback.json'),
                            offline_sha256=digest(HERE/'offline_gate_replay.json'),
                            inputs_sha256=digest(HERE/'input_identity.json'))
                write_once(dest/"precharge.json",record)
                status['charged']=True
                p.step_authorized=True
                return record
            from object_episode import run_slot
            status['phase']="episode"
            try:
                result=run_slot(p,client,body,registration['slots'][6],dest,
                    evaluator_context=dict(object_body_id=gate['object_body_id'],
                                           table_body_id=gate['table_body_id']),before_first_step=charge)
                status['stop']="slot7_returned" if status['charged'] else "setup_failed_before_first_step"
                status['episode_result_sha256']=digest(dest/"result.json")
            finally:
                p.step_authorized=False
        run_zero(on_pass=admitted)
        status.setdefault('stop',"zero_gate_blocked")
    except BaseException as exc:
        status.update(stop="exception_stop",error=repr(exc),traceback=traceback.format_exc())
    finally:
        status['finished_utc']=timestamp()
        write_once(HERE/"session_result.json",status)
        print(json.dumps(status,ensure_ascii=True))
    return


if __name__=="__main__":
    main()
