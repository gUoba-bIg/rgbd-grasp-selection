"""Read-only, post-run static evidence audit.  Never imports or calls PyBullet."""
from collections import Counter
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD_NAME="IP140_R1_smoke13_slot14_legacy_height_bridge021_aggregate_support01"
ARM_NAMES=("shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
    "wrist_1_joint","wrist_2_joint","wrist_3_joint")
CHAIN=("base_link_inertia","shoulder_link","upper_arm_link","forearm_link",
    "wrist_1_link","wrist_2_link","wrist_3_link")
INTERNAL_PAIRS=(*tuple(zip(CHAIN,CHAIN[1:])),("wrist_3_link","ip140_palm"),
    ("ip140_left_bone","ip140_left_pad"),("ip140_right_bone","ip140_right_pad"),
    ("ip140_palm","ip140_left_bone"),("ip140_palm","ip140_left_pad"),
    ("ip140_palm","ip140_right_bone"),("ip140_palm","ip140_right_pad"))
PHASES=(("settle",240),("opening",480),("transit",672),("pregrasp",672),
        ("approach",672),("close",480),("lift",792),("hold",720))
PHASE_AT_SNAPSHOT=[(name,tick) for name,count in PHASES for tick in range(1,count+1)]
OBJECT_PATH="data/simulation_v52/objects/final_static_audit/021/object.urdf"
FORBIDDEN={"stepSimulation","resetSimulation","createConstraint","changeConstraint",
    "removeConstraint","applyExternalForce","applyExternalTorque","setCollisionFilterGroupMask",
    "resetBasePositionAndOrientation","resetBaseVelocity","resetJointStatesMultiDof",
    "resetJointStateMultiDof"}


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf8-sig"))


def _text(value):
    return base64.b64decode(value["bytes_b64"]).decode("utf8") if isinstance(value,dict) else value


def _finite(value):
    if isinstance(value,(list,tuple)):return all(_finite(v) for v in value)
    return isinstance(value,(int,float)) and math.isfinite(value)


def _deferred_labels(value):
    labels=[]
    if isinstance(value,str) and value in ("post_contact_static_overlap_deferred","post_contact_static_penetration_deferred"):
        labels.append(value)
    elif isinstance(value,dict):
        for key,item in value.items():
            if key in ("post_contact_static_overlap_deferred","post_contact_static_penetration_deferred") and item is True:
                labels.append(key)
            labels.extend(_deferred_labels(item))
    elif isinstance(value,list):
        for item in value:labels.extend(_deferred_labels(item))
    return labels


def audit(here=HERE,*,result_name="result.json"):
    here=Path(here).resolve()
    if here!=HERE:raise ValueError("Only this independent version may be audited")
    destination=here/"static_api_audit.json"
    if destination.exists():raise FileExistsError("Audit exists; no overwrite/repeated publication")
    problems=[];problem_counts=Counter();missing=[];evidence={}
    def issue(code,location=None,detail=None):
        problem_counts[code]+=1
        if len(problems)<1000:problems.append(dict(code=code,location=location,detail=detail))
    def optional(rel):
        path=here/rel
        if not path.exists():missing.append(rel);return None
        evidence[rel]=dict(bytes=path.stat().st_size,sha256=sha(path))
        return read(path)
    ledger=optional("slot_ledger.json")
    history=optional("history_snapshot_before.json")
    result=optional(result_name)
    zero=optional("zero_readback.json")
    summary=optional("full_api/summary.json")
    roots=optional("left_right_contact_entry_roots.json")
    sides={} if roots is None else roots.get("sides",{})
    ledger_check=dict(passed=None)
    try:
        old_rel=OLD_NAME+"/slot_ledger.json"
        frozen=history["files"][old_rel]["sha256"]
        old_hash=sha(ROOT/old_rel);new_hash=sha(here/"slot_ledger.json")
        ledger_check=dict(frozen_sha256=frozen,old_current_sha256=old_hash,
            current_copy_sha256=new_hash,passed=(frozen==old_hash==new_hash),
            starts=ledger.get("starts"),remaining=ledger.get("remaining_total_budget"),
            slot14=next(x for x in ledger["slots"] if x["slot"]==14))
        if not ledger_check["passed"]:issue("ledger_not_byte_identical_to_frozen_history")
        if (ledger.get("starts")!=8 or ledger.get("total_slots")!=14
            or ledger.get("remaining_total_budget")!=6 or ledger_check["slot14"].get("charged") is not False):
            issue("budget_or_slot14_charge_changed")
        if list((here/"episodes").glob("*/precharge.json")):
            issue("unexpected_new_precharge_file")
    except (KeyError,TypeError,StopIteration,FileNotFoundError) as exc:
        issue("ledger_evidence_incomplete",detail=repr(exc))
    raw_path=here/"full_api/calls.jsonl.gz"
    if not raw_path.exists():
        missing.append("full_api/calls.jsonl.gz")
        report=dict(version=here.name,passed=None,status="not_auditable_without_recorded_connection",
            audit_scope="Evidence integrity only; not dynamic success or authorization",
            ledger=ledger_check,missing_files=sorted(set(missing)),violations=problems,
            connection_count=None,physics_steps=None,new_charges=0 if ledger_check.get("passed") else None,
            source_sha256=sha(__file__),evidence=evidence)
        with destination.open("x",encoding="utf8",newline="\n") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
        return report
    evidence["full_api/calls.jsonl.gz"]=dict(bytes=raw_path.stat().st_size,sha256=sha(raw_path))
    calls=Counter();returns=Counter();errors=Counter();pending={};records=[]
    robot=None;target=None;table=None;client=None;links={};joints={};reset_records=[]
    object_pose_reads=[];object_pose=None;filters=[];dynamic_writes=[];last_sequence=0
    with gzip.open(raw_path,"rt",encoding="utf8") as stream:
        for lineno,line in enumerate(stream,1):
            row=json.loads(line)
            seq,kind=row[:2]
            if kind=="call":
                if seq!=last_sequence+1:issue("api_sequence_discontinuity",lineno,seq)
                last_sequence=seq;method,args,kwargs=row[3:6];calls[method]+=1
                pending[seq]=(lineno,row)
                if method in FORBIDDEN:issue("forbidden_api_call",lineno,dict(method=method,args=args,kwargs=kwargs))
            elif kind in ("return","error"):
                if seq not in pending:issue("orphan_api_return",lineno,seq);continue
                call_line,call=pending.pop(seq);method,args,kwargs=call[3:6]
                rec=dict(api_sequence=seq,call_line=call_line,return_line=lineno,method=method,args=args,
                    kwargs=kwargs,returned_normally=kind=="return",value=row[3])
                if kind=="error":errors[method]+=1
                else:returns[method]+=1
                if method in ("connect","disconnect","loadURDF","createMultiBody","changeDynamics","setCollisionFilterPair"):
                    records.append(rec)
                if method=="connect" and kind=="return":
                    if args!=[2]:issue("not_exact_direct_connection",call_line,args)
                    client=row[3]
                if method=="loadURDF" and kind=="return":
                    asset=(args[0] if args else kwargs.get("fileName","")).replace("\\","/")
                    if asset==OBJECT_PATH:target=row[3]
                    elif asset.endswith("/model/ur5e_ip140_r1.urdf"):robot=row[3]
                    else:issue("unregistered_loaded_asset",call_line,asset)
                if method=="createMultiBody" and kind=="return":
                    if table is not None:issue("multiple_table_or_body_creations",call_line)
                    table=row[3]
                    if kwargs.get("baseMass")!=0:issue("table_not_fixed",call_line)
                if method=="getJointInfo" and kind=="return" and args[0]==robot:
                    info=row[3];links[_text(info[12])]=info[0];joints[_text(info[1])]=info[0]
                if method=="resetJointState":reset_records.append(rec)
                if method=="setCollisionFilterPair":filters.append(rec)
                if method=="changeDynamics":dynamic_writes.append(rec)
                if method=="getBasePositionAndOrientation" and kind=="return" and args[0]==target:
                    value=row[3]
                    if object_pose is None:object_pose=value
                    if value!=object_pose:issue("object_pose_changed_in_raw_api",lineno,value)
                    object_pose_reads.append(dict(api_sequence=seq,return_line=lineno))
    if pending:issue("unreturned_api_calls",detail=list(pending))
    if (calls["connect"]!=1 or returns["connect"]!=1 or calls["disconnect"]!=1 or returns["disconnect"]!=1):
        issue("single_direct_session_not_confirmed")
    if calls["stepSimulation"]:issue("physical_step_count_nonzero")
    if summary is not None:
        if calls!=Counter(summary.get("calls",{})):issue("raw_call_counts_differ_from_summary")
        if returns!=Counter(summary.get("returns",{})):issue("raw_return_counts_differ_from_summary")
        if last_sequence!=summary.get("total_calls"):issue("raw_total_differs_from_summary")
    if any(x is None for x in (robot,target,table,client)) or len({robot,target,table})!=3:
        issue("actual_identity_incomplete")
    try:
        allowed_joints={joints[x] for x in ARM_NAMES}|{links["ip140_left_bone"],links["ip140_right_bone"]}
    except KeyError as exc:
        allowed_joints=set();issue("registered_joint_identity_missing",detail=str(exc))
    for rec in reset_records:
        a,k=rec["args"],rec["kwargs"]
        if (len(a)!=2 or a[0]!=robot or a[1] not in allowed_joints or k.get("physicsClientId")!=client
            or rec["returned_normally"] is not True or not _finite(k.get("targetValue"))
            or k.get("targetVelocity")!=0):
            issue("invalid_static_robot_reset",rec["call_line"],rec)
    expected=[]
    try:
        expected=[[robot,robot,links[a],links[b],0] for a,b in INTERNAL_PAIRS]
        expected.append([robot,table,links["base_link_inertia"],-1,0])
    except KeyError as exc:issue("filter_link_identity_missing",detail=str(exc))
    if len(filters)!=14:issue("filter_call_count_not_fourteen",detail=len(filters))
    for i,rec in enumerate(filters):
        if (i>=len(expected) or rec["args"]!=expected[i] or rec["kwargs"]!={"physicsClientId":client}
            or rec["returned_normally"] is not True or rec["value"] is not None):
            issue("filter_pair_order_or_return_mismatch",rec["call_line"],rec)
    object_writes=[r for r in dynamic_writes if r["args"] and r["args"][0]==target]
    desired={"lateralFriction":.6,"restitution":0.,"linearDamping":.04,"angularDamping":.04,"physicsClientId":client}
    if len(object_writes)!=1:issue("object_damping_write_count_not_one",detail=len(object_writes))
    elif (object_writes[0]["args"]!=[target,-1] or object_writes[0]["kwargs"]!=desired
        or object_writes[0]["returned_normally"] is not True or object_writes[0]["value"] is not None):
        issue("object_damping_write_wrong_path",detail=object_writes[0])
    receipt=optional("damping_write_receipt.json")
    if receipt is not None:
        for key in ("linearDamping","angularDamping"):
            if receipt.get("fields",{}).get(key)!={"verification_mode":"audited_write_path","target":.04,
                "observed":None,"public_readback_available":False,"setter_call_count":1,"setter_returned_normally":True}:
                issue("damping_falsely_readback_or_wrong_receipt",detail=key)
    root_query_audit={}
    for side,root_result in sides.items():
        qpath=here/"contact_entries"/side/"queries.jsonl.gz"
        if not qpath.exists():missing.append(str(qpath.relative_to(here)));continue
        count=0;bad=0
        with gzip.open(qpath,"rt",encoding="utf8") as stream:
            for count,line in enumerate(stream,1):
                r=json.loads(line)
                if r.get("query_index")!=count or r.get("physics_step_calls")!=0:
                    bad+=1;issue("invalid_contact_root_query_index_or_steps",f"{side}:{count}")
                if r.get("object_pose")!=object_pose:
                    bad+=1;issue("root_query_object_pose_changed",f"{side}:{count}")
                if not 0<=r.get("q_m",float("inf"))<=.070:
                    bad+=1;issue("root_query_q_outside_range",f"{side}:{count}")
                if r.get("raw_world_manifolds") is not None or r.get("raw_manifolds_requested") is not False:
                    bad+=1;issue("root_query_persistent_manifold_isolation_missing",f"{side}:{count}")
        if count!=root_result.get("queries_completed"):issue("root_query_count_mismatch",side)
        if root_result.get("query_log_sha256")!=sha(qpath):issue("root_query_digest_mismatch",side)
        root_query_audit[side]=dict(count=count,violating_queries=bad,sha256=sha(qpath),
            entry_passed=root_result.get("passed"),geometric_only=root_result.get("geometric_entry_only"))
    snapshot_path=here/"dual_channel_snapshots.jsonl.gz"
    count=0;first_failure=None;phases=Counter();deferred=Counter();proxy_checks=0
    reference_manifold_unobserved=0;proxy_manifold_observed=0
    if not snapshot_path.exists():missing.append("dual_channel_snapshots.jsonl.gz")
    else:
        evidence["dual_channel_snapshots.jsonl.gz"]=dict(bytes=snapshot_path.stat().st_size,sha256=sha(snapshot_path))
        with gzip.open(snapshot_path,"rt",encoding="utf8") as stream:
            for count,line in enumerate(stream,1):
                r=json.loads(line);snap=r.get("snapshot");phase=r.get("phase")
                phases[phase]+=1
                if snap!=count or count>4728 or (phase,r.get("phase_tick"))!=PHASE_AT_SNAPSHOT[count-1]:
                    issue("snapshot_sequence_or_phase_mismatch",count)
                if r.get("violations"):
                    if first_failure is None:first_failure=dict(snapshot=count,phase=phase,violations=r["violations"])
                elif first_failure is not None:issue("continued_after_first_static_failure",count)
                qref=r.get("q_ref_m")
                if not _finite(qref) or not 0<=qref<=.070:
                    issue("reference_q_invalid",count);continue
                for side in ("left","right"):
                    certificate=sides.get(side,{})
                    context=certificate.get("source_snapshot")
                    context=context if isinstance(context,dict) else {}
                    entry_snapshot=certificate.get("current_snapshot",context.get("current_snapshot"))
                    active=(certificate.get("passed") is True and isinstance(entry_snapshot,int)
                        and count>=entry_snapshot and phase in ("close","lift","hold"))
                    expected_q=max(qref,certificate["q_entry_m"]) if active else qref
                    actual=r.get("q_proxy_m",{}).get(side)
                    if not _finite(actual) or abs(actual-expected_q)>1e-12:
                        issue("proxy_not_max_reference_entry",f"{count}:{side}",dict(expected=expected_q,actual=actual,entry_active=active))
                    else:proxy_checks+=1
                for channel in ("reference_channel","proxy_channel"):
                    c=r.get(channel)
                    if not isinstance(c,dict):issue("missing_channel",f"{count}:{channel}");continue
                    if c.get("object_pose")!=object_pose:issue("channel_object_pose_changed",f"{count}:{channel}")
                    if not _finite(c.get("W_T_G")):issue("nonfinite_or_missing_channel_pose",f"{count}:{channel}")
                    if not _finite(c.get("actual_joint_states")):issue("nonfinite_or_missing_joint_states",f"{count}:{channel}")
                    raw_manifolds=c.get("raw_manifolds")
                    if channel=="reference_channel" and raw_manifolds is None:
                        reference_manifold_unobserved+=1
                        if c.get("raw_manifolds_requested") is not False:
                            issue("reference_null_manifolds_without_unobserved_label",count)
                    elif channel=="proxy_channel":
                        if not isinstance(raw_manifolds,list) or c.get("raw_manifolds_requested") is False:
                            issue("proxy_manifolds_not_actually_observed",count)
                        else:proxy_manifold_observed+=1
                    elif not isinstance(raw_manifolds,list):
                        issue("reference_manifold_evidence_type",count)
                    deferred.update(_deferred_labels(c))
    if count>4728:issue("too_many_snapshots")
    if count<4728 and first_failure is None and result is not None:
        # Exception/zero-gate stops are valid incomplete outcomes but must be explicit.
        if not (result.get("exception") or result.get("first_failure") or result.get("passed") is False):
            issue("incomplete_path_without_explicit_stop")
    if first_failure is not None and count!=first_failure["snapshot"]:
        issue("snapshot_after_first_failure")
    if result is not None:
        if result.get("charged") is True or result.get("physics_step_calls",0)!=0:
            issue("static_result_claims_charge_or_physics")
        for key in ("grasp_success","dynamics_validated","dynamic_success","episode_completed"):
            if result.get(key) is True:issue("static_result_claims_dynamic_success",detail=key)
        claimed=result.get("snapshots_checked",result.get("snapshots_completed"))
        if claimed is not None and claimed!=count:issue("result_coverage_count_mismatch",detail=dict(claimed=claimed,observed=count))
        if result.get("passed") is True and count!=4728:issue("incomplete_path_claims_passed")
    output=dict(version=here.name,passed=(False if problems else None if missing else True),
        status="audit_failed" if problems else "evidence_incomplete" if missing else "static_evidence_consistent",
        audit_scope="Recorded identity, accounting, API and snapshot consistency. A passing audit is not dynamic success or authorization to start slot14.",
        physics_steps=calls["stepSimulation"],new_charges=0 if ledger_check.get("passed") else None,
        connection_count=calls["connect"],disconnect_count=calls["disconnect"],
        raw_api_calls=dict(calls),raw_api_returns=dict(returns),raw_api_errors=dict(errors),
        actual_ids=dict(robot=robot,target=target,table=table,client=client),actual_links=links,
        registered_joint_resets=len(reset_records),filter_calls=filters,object_damping_writes=object_writes,
        damping_evidence_mode="audited_write_path; observed=null; never claimed actual damping readback",
        object_pose_read_count=len(object_pose_reads),initial_object_pose=object_pose,
        ledger=ledger_check,contact_root_queries=root_query_audit,
        snapshots=dict(observed=count,planned=4728,unvisited=4728-count,phases=dict(phases),first_failure=first_failure,
            proxy_equation_checks=proxy_checks,deferred_flag_counts=dict(deferred),
            reference_manifolds_explicitly_unobserved=reference_manifold_unobserved,
            proxy_manifolds_actually_observed=proxy_manifold_observed,
            loaded_state_prediction=False,dynamics_validated=False),
        static_admission_passed=None if result is None else result.get("passed"),
        missing_files=sorted(set(missing)),violations=problems,violation_counts=dict(problem_counts),
        violation_details_limited_to_first=1000,source_sha256=sha(__file__),evidence=evidence,
        limitations=["Does not repeat collision queries or physics.",
            "Does not rederive contact classification or every scalar readback; raw logs and their separate geometry audits remain necessary.",
            "Positive geometric root certificates and proxy clearance do not verify contact force, friction, grasp hold or loaded dynamics."])
    with destination.open("x",encoding="utf8",newline="\n") as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return output


if __name__=="__main__":
    result=audit()
    print(json.dumps(dict(passed=result["passed"],status=result["status"],report=str(HERE/"static_api_audit.json")),ensure_ascii=False))
