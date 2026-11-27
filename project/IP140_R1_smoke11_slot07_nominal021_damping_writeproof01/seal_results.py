"""Seal observed outcomes after the sole session. This module has no engine dependency."""
import difflib
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from input_identity import HERE, ROOT, OLD, REGISTERED_ID, digest, read, timestamp, verify, write_once

DEST = HERE / "episodes"/"07"


def optional(path):
    return read(path) if path.is_file() else {}


def seal():
    session=read(HERE/"session_result.json")
    gate=optional(HERE/"zero_readback.json")
    raw=optional(HERE/"full_api"/"summary.json")
    calls, returns=raw.get("calls",{}),raw.get("returns",{})
    episode=optional(DEST/"result.json")
    replay=optional(HERE/"offline_gate_replay.json")
    charged=(DEST/"precharge.json").is_file()
    steps, returned = calls.get("stepSimulation",0), returns.get("stepSimulation",0)
    origin=read(HERE/"ledger_initial.json")
    if bool(steps) != charged:
        raise ValueError("First-step API and durable precharge disagree; reconcile without retry")
    if session['charged'] != charged or (episode and episode['step_returns']!=returned):
        raise ValueError("Budget and actual returns disagree")
    stage = "episode" if charged else ("zero_step" if gate or calls.get("connect") else session['phase'])
    reason = (episode.get("failure_reason") if charged else
        (gate.get('required_mismatches') or gate.get("exception") or session.get('error')
         or [r.get("name") for r in replay.get("cases",[]) if not r.get('passed')] or session['stop']))
    result=dict(version=HERE.name, slot=7, candidate_id=REGISTERED_ID, object_id="021",not_p3=True,
        status=session['stop'],charged=charged,starts_before=7,starts_after=7+int(charged),
        remaining_after=6-int(charged),total_budget=13,
        formal_offline_passed=replay.get('passed'),zero_gate_passed=gate.get("passed"),
        connect_calls=calls.get('connect',0), disconnect_returns=returns.get('disconnect',0),
        actual_step_calls=steps,actual_step_returns=returned, unknown_step_returns=steps-returned,
        episode_completed=episode.get('episode_completed',False),
        engineering_gate_passed=episode.get('engineering_gate_passed') if charged else None,
        grasp_success=episode.get('grasp_success') if charged else None,
        slot_passed=episode.get('slot_passed') if charged else None,
        failure_stage=episode.get("failure_stage") if charged else stage, failure_reason=reason,
        historical_smoke10_charge=0, utc=timestamp())
    if charged and result["slot_passed"] != (result["engineering_gate_passed"] and result["grasp_success"]):
        raise ValueError("Slot 7 terminal semantics disagree")
    write_once(HERE/'result.json', result)
    ledger=origin.copy()
    ledger.update(schema="IP140_smoke11_final_ledger",starts=result['starts_after'],
        at_utc=timestamp(),remaining_total_budget=result['remaining_after'],
        starts_this_version=int(charged),steps_this_version=returned,unused_original_budget=result['remaining_after'],
        physical_steps=origin['physical_steps']+returned,terminal=result['status'])
    if charged:
        ledger['charged_in_actual_execution_order']=[*origin['charged_in_actual_execution_order'],7]
        ledger['returned_episodes']+=int(bool(episode))
        ledger['unknown_started_episodes']+=int(not bool(episode))
    item=ledger['slots'][6]
    item.update(charged=charged,state=("returned_passed" if result["slot_passed"] else
        ("returned_failed" if episode else "started_result_unknown")) if charged else "not_started_"+stage,
        result=('episodes/07/result.json' if charged and episode else None),smoke11_record='result.json',
        telemetry='episodes/07/telemetry.jsonl.gz' if (DEST/'telemetry.jsonl.gz').exists() else None)
    if charged:
        for name,rel in [('result','result.json'),('precharge','precharge.json'),('telemetry','telemetry.jsonl.gz')]:
            file=DEST/rel
            if file.exists():
                item[name]=file.relative_to(HERE).as_posix()
                item[name+'_sha256']=digest(file)
    write_once(HERE/'slot_ledger.json',ledger)
    if not (DEST/'final_480_steps.json').exists():
        write_once(HERE/'final_480_steps.json',dict(version=HERE.name,requested_steps=[4249,4728],
            planned_count=480,observed_count=0,rows=[],status='not_reached',reason=reason,
            settle_reference=None if returned<240 else "see telemetry",actual_step_returns=returned))
    write_once(HERE/'STOP.json',dict(slot=7,reason=result["status"],utc=timestamp(),
         step_calls=steps,step_returns=returned,starts=result["starts_after"],retry=False,
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
