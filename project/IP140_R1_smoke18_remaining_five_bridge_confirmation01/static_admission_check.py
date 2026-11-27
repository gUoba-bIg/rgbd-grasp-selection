"""One shared, read-only proof gate for the active smoke18 slot after a fresh static admission."""
from collections import Counter
from pathlib import Path
import gzip
import json
from batch_context import VERSION as HERE, ROOT, SLOT, SLOT_DIR
from static_identity import read, sha, verify
from contact_semantics import validate_entry_certificate
from object_evaluator import validate_filter_calls

PHASES=(("settle",240),("opening",480),("transit",672),("pregrasp",672),("approach",672),("close",480),("lift",792),("hold",720))
SUCCESS="target-independent path passed; loaded target contact dynamics deferred"

def verify_static_complete():
    """No connection, physics, resets, file writes or authorization mutation."""
    identity=verify()
    base=SLOT_DIR/"static_preflight"
    paths=[base/name for name in ("zero_readback.json","zero_step/result.json","full_api/summary.json",
        "full_api/calls.jsonl.gz","path_result.json","left_right_contact_entry_roots.json",
        "dual_channel_snapshots.jsonl.gz","zero_step/all_links_readback.json")]
    if not all(p.is_file() for p in paths):raise ValueError("static admission evidence incomplete")
    zero=read(base/"zero_readback.json");final=read(base/"zero_step/result.json")
    result=read(base/"path_result.json");roots=read(base/"left_right_contact_entry_roots.json")
    summary=read(base/"full_api/summary.json")
    checks={
      "static_zero_only":zero.get("passed") is True and zero.get("connection_scope")=="contact_aware_static"
          and zero.get("eligible_slots_after_zero_gate")==[] and zero.get("dynamic_execution_authorized") is False
          and zero.get("required_mismatches")==[] and zero.get("required_complete") is True
          and len(zero.get("fields",[]))==217 and all(x.get("status")=="pass" for x in zero.get("fields",[]))
          and zero.get("physics_step_calls")==0,
      "full_path":result.get("passed") is True and result.get("status")==SUCCESS
          and result.get("snapshots_completed")==4728 and result.get("first_failure") is None
          and result.get("last_unchecked_snapshots")==0 and result.get("violations")==[]
          and result.get("checked_phase_counts")==dict(PHASES),
      "path_zero_physics_filters":result.get("physics_step_calls")==0 and result.get("filter_call_count")==14
          and result.get("charged") is False and result.get("current_slot_started") is False and result.get("physical_steps")==0,
      "disconnect_confirmed":final.get("disconnected") is True and not final.get("disconnect_exception")
          and summary.get("calls",{}).get("disconnect")==1 and summary.get("returns",{}).get("disconnect")==1,
      "two_roots_present":roots.get("both_found") is True and roots.get("both_passed") is True
          and set(roots.get("sides",{}))=={"left","right"},
      "snapshot_hash":result.get("dual_channel_sha256")==sha(base/"dual_channel_snapshots.jsonl.gz")}
    if not all(checks.values()):raise ValueError("static admission gates: "+str([k for k,v in checks.items() if not v]))
    links={k:v["index"] for k,v in read(base/"zero_step/all_links_readback.json").items()}
    robot=zero["body_id"];target=zero["object_body_id"];table=zero["table_body_id"];client=zero["actual_client"]
    for side in ("left","right"):
        entry=roots["sides"][side]
        problems=validate_entry_certificate(entry,robot=robot,target=target,pad=links["ip140_"+side+"_pad"])
        rootfile=base/"contact_entries"/side/"result.json"
        queryfile=rootfile.with_name("queries.jsonl.gz")
        if problems or not rootfile.is_file() or not queryfile.is_file() or read(rootfile)!=entry:
            raise ValueError("entry certificate or saved root mismatch: "+side+":"+str(problems))
        if (entry.get("query_log_sha256")!=sha(queryfile) or entry.get("source_sha256")!=sha(HERE/"contact_entry.py")
            or entry.get("physics_step_calls")!=0 or entry.get("filter_call_count")!=14
            or entry.get("bisections",49)>48):raise ValueError("entry source/query binding: "+side)
        paths.extend((rootfile,queryfile))
    counts=Counter();returns=Counter();filters=[];pending=None;api_lines=0
    prohibited={"stepSimulation","createConstraint","resetBasePositionAndOrientation","resetBaseVelocity","resetSimulation","removeBody"}
    with gzip.open(base/"full_api/calls.jsonl.gz","rt",encoding="utf-8") as f:
        for api_lines,line in enumerate(f,1):
            row=json.loads(line)
            if row[1]=="call":
                if pending is not None or row[0]!=sum(counts.values())+1:raise ValueError("static API unpaired/discontinuous call")
                pending=row;name,args,kw=row[3:]
                counts[name]+=1
                if name in prohibited:raise ValueError("forbidden static native API: "+name)
                if "physicsClientId" in kw and kw["physicsClientId"]!=client:raise ValueError("wrong static client")
                if name.startswith("setCollisionFilter") and name!="setCollisionFilterPair":raise ValueError("unregistered static filter method")
            elif row[1]=="return":
                if pending is None or row[0]!=pending[0]:raise ValueError("static API unmatched return")
                name,args,kw=pending[3:];returns[name]+=1
                if name=="setCollisionFilterPair":filters.append(dict(method=name,sequence=len(filters)+1,args=args,kwargs=kw,returned=True,return_value=row[3],api_seq=row[0]))
                pending=None
            else:raise ValueError("static API exceptional or unknown outcome")
    if pending is not None or dict(counts)!=summary["calls"] or dict(returns)!=summary["returns"] or summary.get("exceptions"):
        raise ValueError("actual static API does not match its summary")
    if counts["connect"]!=1 or counts["disconnect"]!=1 or counts["stepSimulation"]!=0 or counts["setCollisionFilterPair"]!=14:
        raise ValueError("static connection, step or filter budget mismatch")
    if not validate_filter_calls(robot,target,table,links,filters,require_mount=True)["passed"]:
        raise ValueError("static exact filter identities failed")
    expected=[(name,k) for name,count in PHASES for k in range(1,count+1)]
    observed=0
    with gzip.open(base/"dual_channel_snapshots.jsonl.gz","rt",encoding="utf-8") as f:
        for observed,line in enumerate(f,1):
            x=json.loads(line)
            if observed>4728 or (x.get("phase"),x.get("phase_tick"))!=expected[observed-1] or x.get("snapshot")!=observed:
                raise ValueError("snapshot coverage/order mismatch")
            if x.get("passed") is not True or x.get("violations")!=[] or x.get("physical_steps")!=0 or x.get("charged") is not False:
                raise ValueError("snapshot unproven or failed")
            for channel in ("reference_channel","proxy_channel"):
                c=x.get(channel,{})
                if c.get("passed") is not True or c.get("violations")!=[]:raise ValueError("channel failed or missing")
            if x["proxy_channel"].get("raw_manifolds_requested") is not True or not isinstance(x["proxy_channel"].get("raw_manifolds"),list):
                raise ValueError("proxy manifold evidence missing")
    if observed!=4728:raise ValueError("incomplete paired snapshot count")
    for name in ("positive_negative_replay.json","offline_gate_replay.json"):
        report=read(HERE/name)
        if report.get("passed") is not True or not report.get("cases") or not all(x.get("passed") is True for x in report["cases"]):
            raise ValueError("formal offline report failed: "+name)
        paths.append(HERE/name)
    return dict(version=HERE.name,passed=True,slot=SLOT,allowed_next_step="one fresh dynamic zero then this registered slot only",
        static_connections=1,static_physics_calls=0,paired_snapshots=4728,two_roots_passed=True,
        exact_static_filters=14,disconnect_confirmed=True,api_lines=api_lines,
        frozen_inputs_sha256=sha(HERE/"static_input_identity.json"),
        evidence_sha256={p.relative_to(HERE).as_posix():sha(p) for p in paths},checks=checks)
