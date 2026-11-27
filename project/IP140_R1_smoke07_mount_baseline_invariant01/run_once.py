"""The only authorized continuation uses one engine session and slot 5."""
import json
import os
import sys
import traceback
from pathlib import Path
from episode_log import charge_slot, exclusive_json, timestamp
from input_identity import verify

HERE=Path(__file__).resolve().parent

def main():
    if Path.cwd().resolve()!=HERE.parent or sys.argv[1:]!=['--zero-and-slot-5']:
        raise SystemExit('require explicit project root and --zero-and-slot-5')
    inputs=verify()
    if any((HERE/s).exists() for s in ('STOP.json','episodes','zero_step','full_api','session_claim.json')):
        raise SystemExit('existing claim or results; retry forbidden')
    exclusive_json(HERE/'session_claim.json',dict(at=timestamp(),pid=os.getpid(),permitted=[5],input_bindings=inputs['new_files']))
    from zero_gate import run_zero, gate_valid
    from empty_stroke import run_slot
    status=dict(started=timestamp(),slots=[],bench_entered=False)
    def admitted(p,client,body,links,model_manifest,registration,gate):
        if not gate_valid(gate):
            raise RuntimeError('zero hard admission failed')
        dest=charge_slot(5,HERE/'zero_readback.json',HERE/'preregistration.json')
        status['bench_entered']=True
        status['slots'].append(5)
        exclusive_json(dest/'scene_state.json',dict(at=timestamp(),client=client,body_id=body,
            operation='continue in zero-gate connection; no restore, reload or reset'))
        try:
            result=run_slot(p,client,body,registration['slots'][4],dest)
            recorded=json.loads((dest/'result.json').read_text('utf-8'))
            status['reason']='slot_5_passed_review_stop' if result.get('slot_passed') is True and recorded.get('slot_passed') is True else 'slot_5_failed_stop'
        except BaseException as exc:
            exclusive_json(dest/'controller_exception.json',dict(at=timestamp(),exception=repr(exc),traceback=traceback.format_exc(),charged=True))
            status['reason']='slot_5_exception_stop'
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
    return 0 if status['reason']=='slot_5_passed_review_stop' else 3

if __name__=='__main__':
    raise SystemExit(main())
