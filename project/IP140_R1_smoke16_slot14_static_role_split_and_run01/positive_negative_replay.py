"""Offline replay of smoke16's production static contact rules. No engine imports."""
from copy import deepcopy
from pathlib import Path
import gzip
import hashlib
import itertools
import json
import math
import sys

import numpy as np

from contact_semantics import audit_pad_witness, classify_contact
from object_evaluator import ObjectEvaluator
from aggregate_support import aggregate_table_contacts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / "IP140_R1_smoke15_slot14_positive_gap_manifold_fix_and_run01"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return {"nonfinite_test_literal":repr(float(value))}
    return value

def reverse(row):
    r = deepcopy(row)
    r[1],r[2],r[3],r[4] = row[2],row[1],row[4],row[3]
    r[5],r[6] = row[6],row[5]
    r[7] = [-x for x in row[7]]
    return r

def build():
    path = OLD / "static_preflight/dual_channel_snapshots.jsonl.gz"
    with gzip.open(path,"rt",encoding="utf-8") as data:
        src = json.loads(next(itertools.islice(data,3239,None)))
    assert src["snapshot"] == 3240 and src["phase"] == "lift" and src["phase_tick"] == 24 and src["passed"] is False
    channel = src["proxy_channel"]
    all_fresh = channel["raw_closest"]["robot_target"]
    robot,target,table = 0,2,1
    readback = OLD / "static_preflight/zero_step/all_links_readback.json"
    links = {k:v["index"] for k,v in json.loads(readback.read_text("utf-8-sig")).items()}
    rp = links["ip140_right_pad"]
    cache_rows = [(i,c) for i,c in enumerate(channel["raw_manifolds"]) if tuple(c[1:5])==(target,robot,-1,rp)]
    assert len(cache_rows)==2
    i, cache = cache_rows[0]
    right_fresh = [c for c in all_fresh if tuple(c[1:5])==(robot,target,rp,-1)]
    assert len(right_fresh)==1
    fresh = right_fresh[0]
    tests=[]
    def record(name,passed,**details):
        tests.append(dict(name=name,passed=bool(passed),**details))
    def witness(row,source):
        return audit_pad_witness(row,source=source,robot=robot,target=target,pad_index=rp,
            W_T_pad=channel["W_T_links"]["ip140_right_pad"],W_T_G=channel["W_T_G"],side="right",require_face=True)
    def classify(row,source="manifold",matching=None,phase="lift",which="proxy",certs=None):
        return classify_contact(row,source=source,phase=phase,channel=which,robot=robot,target=target,
            table=table,links=links,entry_by_pad=certs or {},fresh_closest=matching)
    def class_test(name,row,expect,matching=None,**kw):
        result=classify(row,matching=matching,**kw)
        record(name,(not result["violations"])==expect,expected_accepted=expect,
            raw_input=row,fresh_query=matching,result=result)
    def witness_test(name,row,source,expect):
        result=witness(row,source)
        record(name,result["passed"]==expect,expected_accepted=expect,source=source,raw_input=row,result=result)
        return result
    for index,row in cache_rows:
        class_test("actual3240_classification_manifold_"+str(index),row,True,all_fresh)
        res=witness_test("actual3240_witness_manifold_"+str(index),row,"manifold",True)
        if index==i:
            record("actual3240_tangent_preserved_and_normal_within_gate",
                abs(res["full_witness_residual_m"]-3.69887382657577e-5)<1e-12 and
                res["normal_projection_residual_m"]<=1e-9 and res["tangential_residual_m"]>3.69e-5,
                result=res)
    witness_test("actual_fresh_full_vector",fresh,"robot_target",True)
    class_test("reversed_cached_and_reversed_fresh",reverse(cache),True,[reverse(fresh)])
    witness_test("reversed_cached_witness",reverse(cache),"manifold",True)
    witness_test("reversed_fresh_witness",reverse(fresh),"robot_target",True)
    changed=deepcopy(cache);changed[8]+=2e-9
    witness_test("cached_projection_residual_2nm",changed,"manifold",False)
    n = np.asarray(fresh[7])
    axis=np.eye(3)[np.argmin(abs(n))]
    tangent = np.cross(n,axis); tangent/=np.linalg.norm(tangent)
    changed=deepcopy(fresh);changed[6]=(np.asarray(changed[6])+2.1e-9*tangent).tolist()
    witness_test("fresh_tangent_residual_2p1nm",changed,"robot_target",False)
    changed=deepcopy(cache); changed[8]+=0.0001
    changed[5]=(np.asarray(changed[5])+np.asarray(cache[7])*0.0001).tolist()
    class_test("synthetic_nonminimal_cached_distance_not_a_contact_decision",changed,True,[fresh])
    witness_test("synthetic_cached_different_distance_still_has_valid_projection",changed,"manifold",True)
    for name,matching in (("absent",None),("empty",[]),("other_pad",[c for c in all_fresh if c[3]!=rp])):
        class_test("matching_fresh_"+name,cache,False,matching)
    for idx,label in ((9,"Fn"),(10,"Ft1"),(12,"Ft2")):
        for sign in (-1.,1.):
            c=deepcopy(cache);c[idx]=sign*math.nextafter(1e-9,math.inf)
            class_test("positive_gap_force_"+label+"_"+str(sign),c,False,[fresh])
    for name,mod in (("wrong_body",(1,999)),("wrong_pad",(4,links["ip140_left_pad"])),("nonfinite_force",(9,float("nan")))):
        c=deepcopy(cache); c[mod[0]]=mod[1]
        witness_test("cached_"+name,c,"manifold",False)
    c=deepcopy(cache);c[7]=(np.asarray(cache[7])*1.1).tolist()
    witness_test("cached_nonunit_normal",c,"manifold",False)
    for source,r,padpoint in (("manifold",cache,6),("robot_target",fresh,5)):
        c=deepcopy(r); offset=np.asarray(channel["W_T_links"]["ip140_right_pad"])[:3,0]*.1
        for j in (5,6):
            c[j]=(np.asarray(c[j])+offset).tolist()
        witness_test("pad_effective_face_outside_"+source,c,source,False)
    class_test("positive_fresh_separation_approach",fresh,True,source="robot_target",phase="approach")
    for d,name in [(0.,"zero"),(-1e-6,"negative")]:
        for phase in ("approach","close"):
            c=deepcopy(fresh);c[8]=d;c[6]=(np.asarray(c[5])-d*np.asarray(c[7])).tolist()
            class_test("fresh_"+name+"_"+phase+"_no_certificate",c,False,source="robot_target",phase=phase)
    for link in ("ip140_right_bone","ip140_palm","wrist_3_link"):
        c=deepcopy(cache);c[4]=links[link]
        class_test("nonpad_cached_"+link,c,False,all_fresh)
    for k,name in ((table,"table"),(99,"unregistered_environment")):
        c=deepcopy(cache);c[1]=k
        class_test("cached_robot_"+name,c,False,all_fresh)
    ev=lambda:ObjectEvaluator(robot,target,table,links)
    base_state=dict(body_id=target,position=[0,0,0],orientation=[0,0,0,1],linear_velocity=[0]*3,angular_velocity=[0]*3)
    for d,expect in ((-.001,True),(math.nextafter(-.001,-math.inf),False)):
        c=deepcopy(fresh);c[8]=d
        result,viol,_,_=ev()._contact(c,0)
        record("dynamic_penetration_"+repr(d),(not viol)==expect,expected_accepted=expect,
            raw_input=c,result=result,violations=viol)
    for dq,expect in ((.04,True),(math.nextafter(.04,math.inf),False),(-math.nextafter(.04,math.inf),False)):
        fingers=[[0,dq,[0]*6,0],[0,0,[0]*6,0]]
        result=ev().observe(1,"settle",[],fingers,base_state,[0,0,0])
        record("dynamic_speed_"+repr(dq),result["passed"]==expect and
            ("finger_speed_above_0.04_m_s" in result["violations"])==(not expect),
            expected_accepted=expect,fingers=fingers,result=result)
    for forces,expected in (([.1,.1],False),([.12,.12],True),([.08,.1],False)):
        support=[]
        for fn in forces:
            c=deepcopy(fresh);c[1:5]=[target,table,-1,-1];c[8]=0.;c[9]=fn
            support.append(c)
        result=aggregate_table_contacts(support,target,table)
        record("dynamic_aggregate_"+str(forces),result["aggregate_table_support"] is expected and
            not result["aggregation_violations"],raw_rows=support,result=result,
            expected_aggregate_support=expected,eligible_for_terminal_no_support=not expected)
    roots_path=OLD/"static_preflight/left_right_contact_entry_roots.json"
    archived_right=json.loads(roots_path.read_text("utf-8-sig"))["sides"]["right"]
    from contact_semantics import validate_entry_certificate
    for name,changes,expect in (("historical_bracket",{},True),
            ("prior_at_zero",{"previous_signed_distance_m":0.},False),
            ("current_still_positive",{"current_signed_distance_m":1e-15},False),
            ("nonconsecutive",{"previous_snapshot":archived_right["current_snapshot"]-2},False)):
        proof=dict(archived_right,**changes)
        issues=validate_entry_certificate(proof,robot=robot,target=target,pad=rp)
        record("fresh_entry_"+name,(not issues)==expect,source_path=roots_path.relative_to(ROOT).as_posix(),
            source_sha256=sha(roots_path),modifications=changes,expected_accepted=expect,violations=issues,
            purpose="Historical certificate fixture for evaluator tests only; never a new-engine admission")
    c=deepcopy(fresh);c[8]=-.00101;c[6]=(np.asarray(c[5])-c[8]*np.asarray(c[7])).tolist()
    class_test("certified_reference_overtravel",c,True,source="robot_target",phase="close",which="reference",certs={rp:archived_right})
    class_test("certified_proxy_still_respects_minus1mm",c,False,source="robot_target",phase="close",which="proxy",certs={rp:archived_right})
    for observed,lookup in ((cache,c),(reverse(c),fresh)):
        result=classify(observed,matching=[lookup])
        record("fresh_selects_state_"+str(lookup[8]),not result["violations"] and
            result["geometric_contact"]==(lookup[8]<=0) and result["entry_trigger"] is False,
            raw_input=observed,fresh_query=lookup,result=result,category="classification only; fresh geometry evaluated independently")
    # Exercise the terminal outcome predicate with real evaluator state transitions on synthetic rows.
    from object_evaluator import PHASE_AT_STEP
    evaluator=ev()
    obj=deepcopy(base_state); bilateral=[]
    left=deepcopy(fresh);left[3]=links["ip140_left_pad"]
    for pad_row in (left,deepcopy(fresh)):
        pad_row[9]=1.;pad_row[8]=0.;bilateral.append(pad_row)
    support=[]
    for fn in (.12,.12):
        s=deepcopy(fresh);s[1:5]=[target,table,-1,-1];s[8]=0.;s[9]=fn;support.append(s)
    for step,phase in enumerate(PHASE_AT_STEP,1):
        rows=bilateral if phase in ("close","lift","hold") else []
        if phase in ("lift","hold"): obj["position"][2]=.11
        if step>len(PHASE_AT_STEP)-480:rows=bilateral+support
        output=evaluator.observe(step,phase,rows,[[0,0,[0]*6,0],[0,0,[0]*6,0]],obj,[0,0,0])
    final=evaluator.finish()
    record("terminal480_aggregate_rejects_otherwise_eligible_bilateral_hold",
        final["engineering_gate_passed"] and final["episode_completed"] and
        not final["grasp_success"] and not final["slot_passed"] and evaluator.terminal_counts["aggregate_table_supported"]==480 and
        evaluator.terminal_counts["table_supported"]==0,synthetic=True,physics_steps=0,object_state_sequence="synthetic range .0 to .11 m",
        normal_forces_N=[.12,.12],terminal_counts=evaluator.terminal_counts,final_result=final)
    historical_replay=OLD/"positive_negative_replay.json"
    historical=json.loads(historical_replay.read_text("utf-8-sig"))
    sources=("positive_negative_replay.py","contact_semantics.py","contact_entry.py","contact_aware_path.py","object_evaluator.py","aggregate_support.py")
    return {"version":HERE.name,"passed":all(c["passed"] for c in tests),"total_cases":len(tests),"cases":tests,
        "connections":0,"physics_steps":0,"new_charges":0,"charged":False,
        "source_sha256":{s:sha(HERE/s) for s in sources},
        "historical_smoke15_replay":{"path":historical_replay.relative_to(ROOT).as_posix(),"sha256":sha(historical_replay),
            "historical_test_count":len(historical["cases"]),"only_historical_under_smoke15_expectations":True,
            "superseded_expectations":["equal cached-to-global-minimum distances","cached full 3D residual hard rejection","cached point as current-state contact oracle"]},
        "sample":{"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"line":3240,"phase":src["phase"],
            "phase_tick":src["phase_tick"],"old_passed":src["passed"],"old_violations":src["violations"],
            "old_evidence_intact":True,"channel":"proxy_channel","manifolds_native":channel["raw_manifolds"],
            "fresh_native":all_fresh,"W_T_G":channel["W_T_G"],"W_T_right_pad":channel["W_T_links"]["ip140_right_pad"],
            "right_pad_link":rp,"mapping_path":readback.relative_to(ROOT).as_posix(),"mapping_sha256":sha(readback)},
        "scope":"Offline production functions on historical raw rows and named deliberate perturbations. No path admission or physical outcome."}

def persist_report():
    r=build()
    def save(name,data):
        with (HERE/name).open("x",encoding="utf-8",newline="\n") as file:
            json.dump(clean(data),file,ensure_ascii=False,indent=2,allow_nan=False)
            file.write("\n")
    save("positive_negative_replay.json",r)
    spec={"version":HERE.name,"scope":"static pad-target evidence roles",
        "persistent":{"raw_array":"preserved without substitution","hard":["identity","finite","unit normal",
            "abs(dot(pA-pB,nB)-d)<=1e-9 m","registered effective pad face","positive-gap absolute Fn/Ft1/Ft2<=1e-9 N",
            "matching same-state exact-pair fresh query"],"diagnostic_only":["tangential residual","full 3D residual","distance difference from fresh minimum"]},
        "fresh":{"hard":["identity","finite","unit normal","effective pad face for pad-target witness in scope","full 3D residual<=1e-9 m"],
            "determines_current_geometry":True,"entry":"previous d>0 then current d<=0, followed by deterministic same-side root and entry proof"},
        "remaining_gates":{"proxy_lower_distance_m":-0.001,"external_dynamic_lower_distance_m":-0.001,
            "motion_speed_limit_m_per_s":0.040,"aggregate_support":"sum(max(Fn,0)) > 0.2 N on legal object-table points",
            "proxy_nonpad_robot_table_unknown":"prohibited collisions reject",
            "mount_filters_FK_TCP":"unaltered independent gates; no physical pose is commanded by this replay"},
        "source_sha256":r["source_sha256"],"offline_report_sha256":sha(HERE/"positive_negative_replay.json")}
    save("static_gate_semantics.json",spec)
    return r

if __name__=="__main__":
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--offline-only"]:
        raise SystemExit("project root and --offline-only required")
    out=persist_report()
    print(json.dumps({k:out[k] for k in ("passed","total_cases","connections","physics_steps")}))
    raise SystemExit(0 if out["passed"] else 1)
