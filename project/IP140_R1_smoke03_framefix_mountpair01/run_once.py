"""One DIRECT, one loading of the robot. Allow only the two charged empty runs."""
import json
import os
import sys
import time
import traceback
from pathlib import Path
from episode_log import charge_slot, exclusive_json, timestamp
from input_identity import verify

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent


def main():
    if Path.cwd().resolve()!=PROJECT or sys.argv[1:]!=['--zero-and-empty-1-2']:
        raise SystemExit('This entry requires the explicit root and --zero-and-empty-1-2')
    inputs=verify()
    if (HERE/'STOP.json').exists() or (HERE/'episodes').exists() or (HERE/'zero_step').exists():
        raise SystemExit('Existing state found; no continuation or retry')
    exclusive_json(HERE/'session_claim.json',dict(at=timestamp(),pid=os.getpid(),
        permitted=[1,2],joint_world=True,input_bindings=inputs["new_files"]))
    from zero_gate import run_zero, gate_valid
    from empty_stroke import run_slot
    status=dict(started=timestamp(),slots=[],bench_entered=False)
    def admitted(p,client,body,links,model_manifest,registration,gate):
        if not gate_valid(gate):
            raise RuntimeError('invalid zero gate passed to bench')
        base_state=None
        status['bench_entered']=True
        try:
            for n in (1,2):
                dest=charge_slot(n,HERE/'zero_readback.json',HERE/'preregistration.json')
                status['slots'].append(n)
                try:
                    if n==1:
                        base_state=p.saveState(physicsClientId=client)
                        if base_state<0:raise RuntimeError('baseline snapshot failed')
                    else:
                        p.restoreState(stateId=base_state,physicsClientId=client)
                    exclusive_json(dest/'scene_state.json',dict(at=timestamp(),same_client=client,
                        body_id=body,snapshot_id=base_state,operation='save' if n==1 else 'restore',
                        robot_reloads=0,physics_steps_for_restore=0))
                    result=run_slot(p,client,body,registration['slots'][n-1],dest)
                    persisted=json.loads((dest/'result.json').read_text('utf-8'))
                    if result.get('slot_passed') is not True or persisted.get('slot_passed') is not True:
                        status['reason']=f'stop_on_slot_{n}'
                        break
                except BaseException as exc:
                    exclusive_json(dest/'controller_exception.json',dict(at=timestamp(),exception=repr(exc),
                        traceback=traceback.format_exc(),charged=True,retry=False))
                    status['reason']=f'stop_on_slot_{n}_exception'
                    break
            else:
                status['reason']='both_authorized_slots_passed_stop_for_review'
        finally:
            if base_state is not None and base_state>=0:
                p.removeState(base_state,physicsClientId=client)
    try:
        run_zero(on_pass=admitted)
        if not status['bench_entered']:
            status['reason']='zero_gate_not_passed'
    except BaseException as exc:
        status.update(reason='session_exception_stop',exception=repr(exc),traceback=traceback.format_exc())
    finally:
        status.update(ended=timestamp(),retry=False)
        exclusive_json(HERE/'session_result.json',status)
    print(json.dumps(status,ensure_ascii=True))
    return 0 if status['reason']=='both_authorized_slots_passed_stop_for_review' else 3


if __name__=='__main__':
    raise SystemExit(main())
