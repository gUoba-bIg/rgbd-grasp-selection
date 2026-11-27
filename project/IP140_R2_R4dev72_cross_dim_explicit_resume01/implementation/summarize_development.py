"""Summarize an ended R4 batch. Pure file reads, exclusive report creation, no simulation."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
from statistics import median

V = Path(__file__).resolve().parents[1]
METHODS = ("Raw", "A", "C0")
PAIRS = (("C0", "A"), ("A", "Raw"), ("C0", "Raw"))
FIELDS = ("trial_completed", "engineering_gate_passed", "grasp_success", "negative_control_passed")

def read(path):
    return json.loads(path.read_bytes())

def optional(path):
    return read(path) if path.is_file() else None

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def optional_seconds(value):
    if value is None:
        return None
    if type(value) not in (float,int) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid elapsed time")
    return value

def timing(values):
    valid = [v for v in values if v is not None]
    return dict(n_reported=len(valid),n_missing=len(values)-len(valid),total_sec=sum(valid) if valid else None,
                median_sec=median(valid) if valid else None)

def method_totals(rows):
    known = [r for r in rows if r["grasp_success"] is not None]
    wins = sum(r["grasp_success"] for r in known)
    complete = len(known)==len(rows)
    return dict(assigned=len(rows),counted=sum(r["counted"] for r in rows),adjudicated=len(known),successes_observed=wins,
                failures_observed=len(known)-wins,unresolved=len(rows)-len(known),
                full_assignment_success_rate=wins/len(rows) if complete and rows else None,
                forbidden_collisions_observed=sum(r["forbidden_collision"] is True for r in known),
                collision_status_missing=sum(r["forbidden_collision"] is None for r in known),
                known_failure_classes=dict(Counter(r["failure_class"] or "failure_reason_unreported" for r in known if not r["grasp_success"])),
                motion_completed=sum(r["trial_completed"] is True for r in rows),
                execution_time=timing([r["execution_sec"] for r in rows]))

def conditional_report(ledger,design,stop):
    faults = list(ledger.get("shared_integrity_faults") or [])
    if stop and stop.get("shared_integrity_fault",True):
        faults.append(stop)
    sealed=optional(V/"sealed_intents.json")
    protocol=V/"preregistration/statistical_protocol.json"
    if ledger["design_sha256"]!=sha(V/"preregistration/paired_design.json") or ledger["statistical_protocol_sha256"]!=sha(protocol):
        faults.append("design_or_statistical_protocol_digest_mismatch")
    if len(ledger["trials"])!=72 or len(design["units"])!=24 or len({r["trial_id"] for r in ledger["trials"]})!=72:
        raise ValueError("Design count/identity mismatch")
    assigned = {(t["trial_id"],t["unit_id"],t["method"]) for t in design["control_order"]}
    actual={(t["trial_id"],t["unit_id"],t["method"]) for t in ledger["trials"]}
    if assigned!=actual:
        raise ValueError("Ledger and preregistration identities differ")
    counted = sum(t["counted"] for t in ledger["trials"])
    if counted != ledger["method_trials_counted"] or ledger["remaining"] != 72-counted:
        faults.append("budget_arithmetic")
    if counted and (not sealed or sealed.get("intent_count")!=72):
        faults.append("counted_without_72_presealed_intents")
    rows=[]
    for t in ledger["trials"]:
        trial_id=t["trial_id"]; method=t["method"]; unit=t["unit_id"]
        path=V/"trials"/trial_id/"method_result.json"
        result=optional(path)
        record=dict(trial_id=trial_id,method=method,unit_id=unit,counted=t["counted"],state=t["state"],
                    result_available=result is not None,forbidden_collision=None,execution_sec=None,failure_class=None,
                    **dict.fromkeys(FIELDS))
        if result:
            mismatches=[]
            if not t["counted"] or t.get("result_sha256")!=sha(path):
                mismatches.append("result_count_or_digest")
            for field,expected in (("trial_id",trial_id),("method",method),("unit_id",unit)):
                if field in result and result[field]!=expected:
                    mismatches.append(field)
            for field in FIELDS:
                val=result.get(field)
                if field not in result or (val is not None and type(val)!=bool) or val!=t.get(field):
                    mismatches.append(field)
            if result.get("negative_control_passed") is not None:
                mismatches.append("method_trial_mislabeled_as_negative_control")
            if result.get("grasp_success") and not (result.get("engineering_gate_passed") and result.get("trial_completed")):
                mismatches.append("success_without_returned_motion_and_engineering_gate")
            if mismatches:
                faults.append(dict(trial_id=trial_id,conflicts=mismatches))
            else:
                for field in FIELDS: record[field]=result[field]
                collision=result.get("forbidden_collision")
                if collision is not None and type(collision)!=bool:
                    raise ValueError("forbidden_collision must be bool or null")
                record["forbidden_collision"]=collision
                record["failure_class"]=result.get("failure_class") or t.get("failure_class")
                exec_time=result.get("execution_sec")
                if exec_time is None:exec_time=(result.get("time_seconds") or {}).get("execution")
                record["execution_sec"]=optional_seconds(exec_time)
                record["result_sha256"]=sha(path)
        elif t.get("result_sha256") or t.get("grasp_success") is not None:
            faults.append(dict(trial_id=trial_id,missing_result=True))
        rows.append(record)
    triple=[];time_units=[];eligibility_reasons=Counter()
    for u in design["units"]:
        key=u["unit_id"]; unit=V/"units"/key
        network=optional(unit/"network.json") or {}
        orders=optional(unit/"orders.json") or {}
        selector=optional(unit/"selection.json") or {}
        ranking=orders.get("orders_by_pool_index"); selected=selector.get("selected")
        if ranking and set(ranking)!=set(METHODS):
            faults.append("incomplete rank identities "+key)
        if selected and set(selected)!=set(METHODS):
            faults.append("incomplete selection identities "+key)
        el=selector.get("eligibility")
        if el is not None:
            eligibility_reasons.update(str(e.get("reason") or "reason_unreported") for e in el if not e["eligible"])
        planning=[optional_seconds(e.get("planning_seconds")) for e in (el or [])]
        time_units.append(dict(unit_id=key,segmentation_sec=optional_seconds(network.get("segmentation_sec")),
           candidate_generation_sec=optional_seconds(network.get("candidate_generation_sec")),
           pool_sec=optional_seconds(orders.get("pool_build_sec")),scoring_sec=optional_seconds(orders.get("scoring_sec")),
           shared_candidate_planning_sec=sum(planning) if el is not None and all(v is not None for v in planning) else None))
        result={r["method"]:r for r in rows if r["unit_id"]==key}
        if set(result)!=set(METHODS):
            raise ValueError("incomplete paired unit "+key)
        triple.append(dict(unit_id=key,base_scene=u["base_scene"],object_id=u["object_id"],position_index=u["position_index"],
                           condition=u["condition"],pool_count=orders.get("pool_count"),rankings=ranking,selected_indices=selected,
                           results=result))
    totals={m:method_totals([r for r in rows if r["method"]==m]) for m in METHODS}
    full=all(t["assigned"]==t["adjudicated"]==24 for t in totals.values()) and counted==72
    pairs={}
    for first,second in PAIRS:
        won=[];lost=[];both=[];neither=[];unknown=[];agreement=[];different=[];bothnull=[];onesided=[];rankingdiff=[];rank_covered=0
        for c in triple:
            x=c["results"][first]["grasp_success"];y=c["results"][second]["grasp_success"]
            bucket=unknown if x is None or y is None else both if x and y else won if x else lost if y else neither
            bucket.append(c["unit_id"])
            sel=c["selected_indices"]
            if sel and first in sel and second in sel:
                a,b=sel[first],sel[second]
                bucket=bothnull if a is None and b is None else onesided if a is None or b is None else agreement if a==b else different
                bucket.append(c["unit_id"])
            rank=c["rankings"]
            if rank and first in rank and second in rank:
                rank_covered+=1
                if rank[first]!=rank[second]:rankingdiff.append(c["unit_id"])
        pairs[first+"-"+second]=dict(first_only=won,second_only=lost,both=both,neither=neither,unresolved=unknown,
                 success_count_difference_over_all_24=len(won)-len(lost) if not unknown else None,
                 paired_risk_difference=(len(won)-len(lost))/24 if not unknown else None,
                 ranking_units_reported=rank_covered,ranking_order_disagreement=rankingdiff,
                 same_selected=agreement,different_selected=different,both_no_executable_candidate=bothnull,
                 only_one_selected=onesided,selection_agreement_given_both_selected=len(agreement)/(len(agreement)+len(different)) if agreement or different else None)
    strata={}
    for dimension in ("condition","object_id","position_index","base_scene"):
        strata[dimension]={str(val):{m:method_totals([c["results"][m] for c in triple if c[dimension]==val]) for m in METHODS}
                           for val in sorted({c[dimension] for c in triple})}
    primary=pairs["C0-A"]
    positive_bases=sorted({c["base_scene"] for c in triple if c["unit_id"] in primary["first_only"]})
    net_positive_bases=[k for k,v in strata["base_scene"].items()
                        if all(v[m]["unresolved"]==0 for m in ("C0","A")) and v["C0"]["successes_observed"]>v["A"]["successes_observed"]]
    n={m:totals[m]["successes_observed"] for m in METHODS}
    collisions={m:totals[m]["forbidden_collisions_observed"] for m in METHODS}
    collision_ready=full and all(totals[m]["collision_status_missing"]==0 for m in METHODS)
    gates=dict(c0_minus_a_at_least_two=(n["C0"]-n["A"]>=2) if full else None,
            c0_only_minus_a_only_at_least_two=(len(primary["first_only"])-len(primary["second_only"])>=2) if full else None,
            c0_at_least_raw=(n["C0"]>=n["Raw"]) if full else None,
            c0_forbidden_collisions_not_more_than_both=(collisions["C0"]<=min(collisions["A"],collisions["Raw"])) if collision_ready else None,
            c0_only_wins_from_two_distinct_base_scenes=len(positive_bases)>=2 if full else None,
            no_shared_integrity_fault=False if faults else True if full else None)
    decision="eligible_to_request_720_separately" if all(v is True for v in gates.values()) else (
             "insufficient_development_evidence" if full and not faults and collision_ready else "incomplete_or_integrity_blocked")
    return dict(version=V.name,scope="exposed paired development",planned_method_trials=72,counted=counted,
       remaining_uncounted=ledger["remaining"],results_with_known_grasp_outcome=sum(x["adjudicated"] for x in totals.values()),
       completed_denominator_available=full,unavailable_percentages_are_null=True,shared_integrity_faults=faults,stop_record=stop,ledger_stage=ledger.get("stage"),
       frozen_intent_count=sealed["intent_count"] if sealed else 0,methods=totals,pairs=pairs,strata=strata,paired_units=triple,
       eligible_candidate_rejections=dict(eligibility_reasons),timing_per_shared_unit=time_units,
       shared_timing_summary={k:timing([u[k] for u in time_units]) for k in time_units[0] if k!="unit_id"},
       positive_base_scenes=positive_bases,net_positive_base_scenes=net_positive_bases,
       six_expansion_conditions=gates,decision=decision,automatic_expansion=False)

def publish(summary):
    d=V/"analysis";d.mkdir(exist_ok=True)
    with (d/"paired_statistics.json").open("x",encoding="utf8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=False)
    lines=["# R4 配对开发比较","","工程账本已封存，R4 开发账本单独计次。","",
        f"已计次 {summary['counted']}/72，已取得可判定抓取结果 {summary['results_with_known_grasp_outcome']}/72。","",
        "| 方法 | 预登记 | 已计次 | 可判定 | 已知成功 | 已知失败 | 未决 | 全部24次成功率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m,a in summary["methods"].items():
        p=a["full_assignment_success_rate"]
        lines.append(f"| {m} | {a['assigned']} | {a['counted']} | {a['adjudicated']} | {a['successes_observed']} | {a['failures_observed']} | {a['unresolved']} | {p:.2%} |" if p is not None else
                     f"| {m} | {a['assigned']} | {a['counted']} | {a['adjudicated']} | {a['successes_observed']} | {a['failures_observed']} | {a['unresolved']} | 待完整结果 |")
    lines+=["","| 配对差异 | 仅前者成功 | 仅后者成功 | 未决配对 | 全部24对成功率差 |","|---|---:|---:|---:|---:|"]
    for pair,p in summary["pairs"].items():
        delta="待完整结果" if p["paired_risk_difference"] is None else f"{p['paired_risk_difference']*100:+.2f} 个百分点"
        lines.append(f"| {pair} | {len(p['first_only'])} | {len(p['second_only'])} | {len(p['unresolved'])} | {delta} |")
    decision={"eligible_to_request_720_separately":"达到六项扩大申请条件，720回合仍须另行批准。",
              "insufficient_development_evidence":"开发证据不足，扩大条件未全部满足。",
              "incomplete_or_integrity_blocked":"主比较不能形成可采纳的完整结论。"}[summary["decision"]]
    lines+=["",decision,"","逐单元三组结果、分层计次、候选排序与选择分歧、失败去向、实测耗时及六项判定均在 [paired_statistics.json](paired_statistics.json)。",
            "","耗时按实际记录披露。三组共用的推理、评分及候选规划按单元计一次。"]
    with (d/"开发结果.md").open("x",encoding="utf8") as f:f.write("\n".join(lines)+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stopped",action="store_true",help="Call only when an incomplete batch has been stopped")
    parser.add_argument("--stop-file",type=Path)
    args=parser.parse_args()
    ledger=read(V/"development_ledger.json")
    if ledger["method_trials_counted"]<72 and not args.stopped:
        raise SystemExit("incomplete batch: explicit --stopped required")
    stop=optional(args.stop_file or V/"STOP.json")
    result=conditional_report(ledger,read(V/"preregistration/paired_design.json"),stop)
    publish(result)

