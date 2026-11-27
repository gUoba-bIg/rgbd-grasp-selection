"""Pure offline integration of the recorded smoke14 positive-gap failure.

Does not connect to, import, or run PyBullet.  Synthetic mutations are labeled;
the historical source bytes and verdict are never modified.
"""
from pathlib import Path
from copy import deepcopy
import ast
import gzip
import hashlib
import json
import sys
import numpy as np

from contact_semantics import classify_contact
from contact_entry import validate_contact_row,_safe
import contact_aware_path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"IP140_R1_smoke14_slot14_contact_aware_static_admission01"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    source=OLD/"dual_channel_snapshots.jsonl.gz"
    sample=None
    with gzip.open(source,"rt",encoding="utf-8") as stream:
        for line in stream:
            r=json.loads(line)
            if r["snapshot"]==2748:sample=r;break
    if sample is None:raise ValueError("historical first-failure sample missing")
    api=json.loads((OLD/"static_api_audit_r3.json").read_bytes())
    ids=api["actual_ids"];links=api["actual_links"]
    robot,target,table=ids["robot"],ids["target"],ids["table"]
    checks=[]
    def add(name,row,fresh,channel,expected_pass,*,synthetic=False,W_T_pad=None):
        data=sample[channel+"_channel"]
        c=classify_contact(row,source="manifold",phase="close",channel=channel,
            robot=robot,target=target,table=table,links=links,entry_by_pad={},fresh_closest=fresh)
        pad="ip140_left_pad"
        # A separate face audit is part of the actual path integration.
        face=validate_contact_row(row,robot=robot,target=target,pad_index=links[pad],
            W_T_pad=data["W_T_links"][pad] if W_T_pad is None else W_T_pad,
            W_T_G=data["W_T_G"],side="left",require_face=True)
        actual=not c["violations"] and face["passed"]
        checks.append(dict(name=name,synthetic_mutation=synthetic,expected_pass=expected_pass,
            actual_pass=actual,passed=actual==expected_pass,classifier=c,face_audit=face,
            raw_input=_safe(row),fresh_closest=_safe(fresh)))
    for channel in ("reference","proxy"):
        data=sample[channel+"_channel"]
        row=deepcopy(data["raw_manifolds"][0]);fresh=deepcopy(data["raw_closest"]["robot_target"])
        assert row[8]>0 and all(row[k]==0 for k in (9,10,12))
        add(channel+"_actual_positive_gap",row,fresh,channel,True)
        for index,label in ((9,"normal"),(10,"tangent1"),(12,"tangent2")):
            modified=deepcopy(row);modified[index]=2e-9
            add(channel+"_positive_gap_nonzero_"+label,modified,fresh,channel,False,synthetic=True)
        add(channel+"_missing_fresh",row,[],channel,False,synthetic=True)
        unmatched=[x for x in fresh if links["ip140_left_pad"] not in (x[3],x[4])]
        add(channel+"_wrong_fresh_pair",row,unmatched,channel,False,synthetic=True)
        divergent=deepcopy(fresh)
        for r in divergent:
            if r[3]==links["ip140_left_pad"] or r[4]==links["ip140_left_pad"]:r[8]+=2e-9
        add(channel+"_fresh_distance_mismatch",row,divergent,channel,False,synthetic=True)
        nonpad=deepcopy(row)
        if nonpad[1]==robot:nonpad[3]=links["ip140_left_bone"]
        else:nonpad[4]=links["ip140_left_bone"]
        add(channel+"_nonpad_never_exempt",nonpad,fresh,channel,False,synthetic=True)
        nonunit=deepcopy(row);nonunit[7]=[1.01*x for x in row[7]]
        add(channel+"_nonunit_normal",nonunit,fresh,channel,False,synthetic=True)
        off_face=deepcopy(row)
        delta=np.asarray(data["W_T_links"]["ip140_left_pad"])[:3,0]*.002
        for i in (5,6):off_face[i]=(np.asarray(off_face[i])+delta).tolist()
        add(channel+"_effective_face_required",off_face,fresh,channel,False,synthetic=True)
    source_code=(HERE/"contact_aware_path.py").read_text(encoding="utf-8")
    parsed=ast.parse(source_code)
    scan=next(x for x in parsed.body if isinstance(x,ast.FunctionDef) and x.name=="scan")
    calls=[n for n in ast.walk(scan) if isinstance(n,ast.Call)]
    classifier_calls=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="classify_contact"]
    integration=dict(
        output_root_keyword=any(x.arg=="output_root" for x in scan.args.kwonlyargs),
        fresh_closest_forwarded=bool(classifier_calls) and all("fresh_closest" in [k.arg for k in n.keywords] for n in classifier_calls),
        root_source_is_fresh_closest='reference_data["raw_closest"]["robot_target"]' in source_code,
        positive_distance_does_not_start_root='c[8]>0 or phase!="close":continue' in source_code,
        previous_positive_bracket_required='previous["distance_m"]<=0' in source_code,
        both_roots_required='set(roots)!={"left","right"}' in source_code,
        outputs_are_under_selected_root=all(x in source_code for x in (
            'record_path=output_root/"dual_channel_snapshots.jsonl.gz"',
            'output_dir=output_root/"contact_entries"/side',
            'write_once(output_root/"left_right_contact_entry_roots.json"',
            'write_once(output_root/"path_result.json"')),
        no_pybullet_imported="pybullet" not in sys.modules)
    unchanged={}
    for rel in ("contact_entry.py","trajectory.py","trajectory_plan.json","speed_policy.py",
                "object_episode.py","object_evaluator.py","model/ur5e_ip140_r1.urdf","model/manifest.json"):
        unchanged[rel]=dict(old_sha256=sha(OLD/rel),new_sha256=sha(HERE/rel),equal=sha(OLD/rel)==sha(HERE/rel))
    return dict(version=HERE.name,passed=all(x["passed"] for x in checks) and all(integration.values()) and all(x["equal"] for x in unchanged.values()),
        engine_connections=0,physical_steps=0,slot_starts=0,charged=False,cases=checks,case_count=len(checks),
        source_checks=integration,unchanged_assets_and_controller=unchanged,
        historical_input=dict(path=str(source.relative_to(HERE.parent)),sha256=sha(source),snapshot=2748,
            jsonl_record=2748,historical_verdict_unchanged=sample["passed"] is False),
        source_sha256={name:sha(HERE/name) for name in ("path_integration_replay.py","contact_aware_path.py","contact_semantics.py","contact_entry.py")},
        scope="Historical-sample classifier plus effective-face integration and static source checks only; does not establish 4728 new engine snapshots, entry roots or dynamic feasibility.")


if __name__=="__main__":
    result=run()
    destination=HERE/"path_integration_replay.json"
    with destination.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(_safe(result),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(passed=result["passed"],case_count=result["case_count"],report_sha256=sha(destination)),ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
