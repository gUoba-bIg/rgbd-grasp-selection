"""Seal observed outcomes after the sole session. This module has no engine dependency."""
import difflib
from copy import deepcopy
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from input_identity import HERE, ROOT, OLD, REGISTERED_ID, digest, read, timestamp, verify, write_once

DEST = HERE / "episodes"/"14"


def optional(path):
    return read(path) if path.is_file() else {}


def seal():
    """Only seal observed slot14 evidence. No physics outcomes are invented for admission blocks."""
    session=read(HERE/"session_result.json")
    gate=optional(HERE/"zero_readback.json")
    pre_gate=optional(HERE/"path_precheck/zero_readback.json")
    path=optional(HERE/"path_precheck/result.json")
    raw=optional(HERE/"full_api/summary.json")
    pre_raw=optional(HERE/"path_precheck/full_api/summary.json")
    calls,returns=raw.get("calls",{}),raw.get("returns",{})
    pre_calls,pre_returns=pre_raw.get("calls",{}),pre_raw.get("returns",{})
    episode=optional(DEST/"result.json")
    replay=optional(HERE/"offline_gate_replay.json")
    charged=(DEST/"precharge.json").is_file()
    steps,returned=calls.get("stepSimulation",0),returns.get("stepSimulation",0)
    origin=read(HERE/"ledger_initial.json")
    if pre_calls.get('stepSimulation',0):
        raise ValueError("Forbidden precheck physics must be separately reconciled; do not conceal in dynamic budget")
    if bool(steps)!=charged:
        raise ValueError("First-step API and durable precharge disagree; reconcile without retry")
    if session['charged']!=charged or (episode and episode['step_returns']!=returned):
        raise ValueError("Budget and actual returns disagree")
    if origin['starts']!=8 or origin['existing_budget']!=14 or len(origin['slots'])!=14:
        raise ValueError("Frozen initial budget must be 8/14")
    if calls.get('connect',0) and not(path.get('passed') is True and pre_gate.get('passed') is True):
        raise ValueError("Dynamic connection must follow fully passed static path")
    stage="episode" if charged else session['phase']
    reason=(episode.get("failure_reason") if charged and episode else
        (path.get('first_failure') or gate.get('required_mismatches') or gate.get("exception")
         or pre_gate.get('required_mismatches') or pre_gate.get('exception') or session.get('error')
         or [r.get("name") for r in replay.get("cases",[]) if not r.get('passed')] or session['stop']))
    observed_outcome=charged and bool(episode)
    result=dict(version=HERE.name,slot=14,parent_slot=7,candidate_id=REGISTERED_ID,
        object_id="021",not_p3=True,development_corrective_bridge=True,independent_confirmation=False,
        status=session['stop'],charged=charged,starts_before=8,starts_after=8+int(charged),
        remaining_after=6-int(charged),total_budget=14,
        formal_offline_passed=replay.get('passed'),precheck_zero_gate_passed=pre_gate.get('passed'),
        static_path_passed=path.get('passed'),static_path_snapshots_checked=path.get('snapshots_checked'),
        zero_gate_passed=gate.get("passed"),
        precheck_connect_calls=pre_calls.get('connect',0),precheck_disconnect_returns=pre_returns.get('disconnect',0),
        precheck_physics_step_calls=pre_calls.get('stepSimulation',0),
        connect_calls=calls.get('connect',0),disconnect_returns=returns.get('disconnect',0),
        actual_step_calls=steps,actual_step_returns=returned,unknown_step_returns=steps-returned,
        episode_completed=episode.get('episode_completed') if observed_outcome else None,
        engineering_gate_passed=episode.get('engineering_gate_passed') if observed_outcome else None,
        grasp_success=episode.get('grasp_success') if observed_outcome else None,
        slot_passed=episode.get('slot_passed') if observed_outcome else None,
        physics_outcome_observed=observed_outcome,
        unknown_started_outcome=charged and not bool(episode),
        failure_stage=episode.get("failure_stage") if observed_outcome else stage,failure_reason=reason,
        parent_slot7_failure_preserved=True,slots8_to_12_frozen=True,
        utc=timestamp())
    if observed_outcome and result["slot_passed"]!=(result["engineering_gate_passed"] and result["grasp_success"]):
        raise ValueError("Slot14 terminal semantics disagree")
    write_once(HERE/'result.json',result)
    ledger=deepcopy(origin)
    ledger.update(schema="IP140_smoke13_final_ledger",starts=result['starts_after'],
        at_utc=timestamp(),remaining_total_budget=result['remaining_after'],
        starts_this_version=int(charged),steps_this_version=returned,
        # Slot14 is additional; original slots8-12 remain all five unused.
        unused_original_budget=5,additional_corrective_slot_available=not charged,
        physical_steps=origin['physical_steps']+returned,unknown_step_returns_this_version=steps-returned,
        terminal=result['status'])
    if charged:
        ledger['charged_in_actual_execution_order']=[*origin['charged_in_actual_execution_order'],14]
        ledger['returned_episodes']+=int(bool(episode))
        ledger['unknown_started_episodes']+=int(not bool(episode))
    item=ledger['slots'][13]
    item.update(charged=charged,state=("returned_passed" if result["slot_passed"] else
        ("returned_failed" if episode else "started_result_unknown")) if charged else "not_started_"+stage,
        result=('episodes/14/result.json' if charged and episode else None),smoke13_record='result.json',
        telemetry='episodes/14/telemetry.jsonl.gz' if (DEST/'telemetry.jsonl.gz').exists() else None)
    if charged:
        for name,rel in [('result','result.json'),('precharge','precharge.json'),('telemetry','telemetry.jsonl.gz')]:
            file=DEST/rel
            if file.exists():
                item[name]=file.relative_to(HERE).as_posix();item[name+'_sha256']=digest(file)
    if ledger['slots'][:13]!=origin['slots'][:13]: raise ValueError("Historical and frozen future slots changed")
    write_once(HERE/'slot_ledger.json',ledger)
    if not (DEST/'final_480_steps.json').exists():
        write_once(HERE/'final_480_steps.json',dict(version=HERE.name,requested_steps=[4249,4728],
            planned_count=480,observed_count=0,rows=[],status='not_reached' if steps else 'not_started',reason=reason,
            physics_outcome_observed=False,settle_reference=None if returned<240 else "see telemetry",
            actual_step_returns=returned))
    write_once(HERE/'STOP.json',dict(slot=14,parent_slot=7,reason=result["status"],utc=timestamp(),
        step_calls=steps,step_returns=returned,unknown_step_returns=steps-returned,
        starts=result["starts_after"],total_budget=14,remaining=result['remaining_after'],retry=False,
        slots8_to_12_started=False,result_sha256=digest(HERE/'result.json')))
    return result


def code_diff():
    chunks=[]
    nine=ROOT/'IP140_R1_smoke09_close_speed_margin_single_stop01'
    for current in sorted(HERE.glob("*.py")):
        previous=OLD/current.name
        if not previous.exists() and current.name in ("zero_scene.py","episode_log.py","run_once.py"):
            inherited="zero_gate.py" if current.name=="zero_scene.py" else current.name
            previous=nine/inherited
        prior=previous.read_text("utf-8-sig").splitlines(True) if previous.exists() else []
        if prior==current.read_text("utf-8-sig").splitlines(True):
            continue
        chunks.extend(difflib.unified_diff(prior,current.read_text("utf-8-sig").splitlines(True),
            fromfile=previous.relative_to(ROOT).as_posix() if previous.exists() else "/dev/null",
            tofile=current.relative_to(ROOT).as_posix()))
    with (HERE/'implementation.diff').open('x',encoding='utf-8',newline='\n') as stream:
        stream.writelines(chunks)


def package():
    verify()
    if not (HERE/'result.json').is_file() or not (HERE/'审查报告.md').is_file():
        raise ValueError("terminal report required before sealing")
    code_diff()
    files=[p for p in sorted(HERE.rglob("*")) if p.is_file()
           and '__pycache__' not in p.parts and p.suffix not in ('.zip','.sha256')]
    checksum=HERE/"SHA256SUMS.txt"
    hashes={p.relative_to(HERE).as_posix():digest(p) for p in files}
    with checksum.open("x",encoding="utf-8",newline="\n") as f:
        f.write("SHA256\tbytes\trelative_path\n")
        for p in files:
            rel=p.relative_to(HERE).as_posix()
            f.write(f"{hashes[rel]}\t{p.stat().st_size}\t{rel}\n")
    files.append(checksum); hashes[checksum.name]=digest(checksum)
    archive=HERE/(HERE.name+"_审查包.zip")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:
            z.write(p,p.relative_to(HERE).as_posix(),compress_type=(
                zipfile.ZIP_STORED if p.suffix in ('.gz','.png') else zipfile.ZIP_DEFLATED))
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(hashes) or len(z.namelist())!=len(files):
            raise ValueError("ZIP member set differs")
        for rel,expected in hashes.items():
            with z.open(rel) as f:
                if hashlib.file_digest(f,"sha256").hexdigest()!=expected:
                    raise ValueError("ZIP digest mismatch: "+rel)
    with Path(str(archive)+".sha256").open("x",encoding="utf-8") as f:
        f.write(digest(archive)+"  "+archive.name+"\n")
    print(json.dumps(dict(entries=len(files),bytes=archive.stat().st_size,sha256=digest(archive))))


if __name__=="__main__":
    if Path.cwd().resolve()!=ROOT:
        raise SystemExit("Explicit project root required")
    if sys.argv[1:]==["--seal-outcome"]:
        print(json.dumps(seal(),ensure_ascii=True))
    elif sys.argv[1:]==["--package"]:
        package()
    else:
        raise SystemExit("choose --seal-outcome or --package")
