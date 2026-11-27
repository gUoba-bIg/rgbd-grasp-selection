"""Summarize the closed batch without importing or connecting to PyBullet."""
from pathlib import Path
import json
import sys
from native22 import inspect
from telemetry22 import PHASES, optional, observations, read, sha

V=Path(__file__).resolve().parents[1]
ROOT=V.parent
OLD=ROOT/"IP140_R2_smoke21_registered_pad_surface_and_resume_four01"
SLOTS=(10,11,12)
NAMES={"correct_negative":"正确负例","misgrasp":"误抓","engineering_failure":"工程门失败，负例无效",
       "missing_physical_observations":"未取得完整动力学结果","shared_failure":"共享故障，负例未验证"}
def save(path,data):
    with path.open("x",encoding="utf8",newline="\n") as f:
        json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2);f.write("\n")

def inspect_if_present(path,scope):
    try:return inspect(path,scope)
    except (ValueError,KeyError,FileNotFoundError,EOFError,OSError) as e:
        return dict(scope=scope,paired_without_exception=False,error=repr(e),step_calls=None,step_returns=None)

def measure_slot(i,rule,ledger):
    folder=V/"slots"/f"{i:02d}"
    static=folder/"static_preflight"; ep=folder/"episodes"/f"{i:02d}"
    session=optional(folder/"session_result.json") or {}
    path=optional(static/"path_result.json") or {}
    receipt=optional(folder/"static_admission_receipt.json") or {}
    stepfile=ep/"telemetry.jsonl.gz"; observations_file=ep/"result.json"
    outcome=optional(observations_file)
    sa=inspect_if_present(static,"static"); da=inspect_if_present(folder,"dynamic")
    actual=observations(ep,outcome,V/"postrun_analysis"/f"slot{i:02d}_object_state_and_contact.jsonl.gz") if stepfile.is_file() else None
    errors=[]
    if (rule["slot"],rule["object_id"],rule["kind"],rule["candidate_id"])!=(i,ledger["object_id"],"miss",ledger["candidate_id"]):
        errors.append("scope/candidate mismatch")
    for audit in (sa,da):
        if audit is not None and audit["paired_without_exception"] is not True:
            errors.append(f"incomplete or unexpected native {audit['scope']} API")
    if sa is not None and sa["step_calls"]!=0:errors.append("static connection had physics")
    if not da and ledger["charged"]:errors.append("missing native charge evidence")
    if da and da["step_calls"] is not None and (da["step_calls"]>0)!=ledger["charged"]:errors.append("native billing mismatch")
    if da and da["step_calls"] and not (path.get("passed") is True and path.get("snapshots_completed")==4728 and
        receipt.get("passed") is True and bool(sha(folder/"static_admission_receipt.json"))):
        errors.append("dynamic start is not bound to full static admission")
    if bool(receipt) and (path.get("pre_lift_bilateral_entry_absent")!=receipt.get("pre_lift_bilateral_entry_absent") or
        type(path.get("pre_lift_bilateral_entry_absent")) is not bool):
        errors.append("pre-lift entry declarations differ or absent")
    if outcome is not None:
        for k in ("slot","kind","object_id","candidate_id","candidate_vector_sha256"):
            if outcome.get(k)!=ledger.get(k):errors.append("result/ledger mismatch: "+k)
        if not da or not actual:errors.append("missing physical evidence of episode result")
        elif da["step_returns"]!=outcome["step_returns"] or da["step_calls"]!=outcome["step_intents"]:
            errors.append("episode and native API counts differ")
    disposition="missing_physical_observations"
    complete=bool(outcome and actual and da and outcome["episode_completed"] is True and
        outcome["unknown_steps"]==0 and outcome["step_returns"]==da["step_calls"]==da["step_returns"]==4728 and
        dict(actual["states_by_phase"])==PHASES and actual["saved_terminal_window"]==480)
    if outcome is not None:
        if outcome.get("grasp_success") is True:
            disposition="misgrasp"
        elif complete and outcome["engineering_gate_passed"] is True and outcome["grasp_success"] is False and outcome["slot_passed"] is True:
            disposition="correct_negative"
        elif outcome["engineering_gate_passed"] is False:
            disposition="engineering_failure"
    if session.get("batch_can_continue") is False or errors:
        disposition="shared_failure"
    compact=None
    if outcome is not None:
        keys=("episode_completed","engineering_gate_passed","grasp_success","slot_passed","failure_stage","failure_reason",
              "grasp_failure_reasons","outcome_class","max_lift_m","slipped","grasp_formed","unknown_steps","step_intents",
              "step_returns","terminal_observation_count","final_hold_streak","step_counts")
        compact={k:outcome.get(k) for k in keys}
    return dict(slot=i,kind=rule["kind"],object_id=rule["object_id"],candidate_id=rule["candidate_id"],
        candidate_vector_sha256=ledger["candidate_vector_sha256"],charged=ledger["charged"],classification=disposition,
        actual_motion_complete=complete,static_result=dict(passed=path.get("passed"),states=path.get("snapshots_completed"),
         first_failure=path.get("first_failure"),pre_lift_bilateral_entry_absent=path.get("pre_lift_bilateral_entry_absent"),
         missing_entry_sides=path.get("missing_entry_sides"),valid_entry_sides=path.get("valid_entry_sides"),
         path_sha256=sha(static/"path_result.json") if path else None,receipt_sha256=sha(folder/"static_admission_receipt.json") if receipt else None),
        api={"static":sa,"dynamic":da},physical=compact,measured=actual,verification_errors=errors,session_status=session,
        episode_result_sha256=sha(observations_file) if outcome else None)

def historical():
    records=read(OLD/"result.json")["outcomes"];retained=[]
    for row in records:
        if row["kind"]!="nominal":continue
        source=(OLD/row["episode_source"]["path"])
        if not source.is_file():source=ROOT/row["episode_source"]["path"]
        if sha(source)!=row["episode_source"]["sha256"]:raise ValueError("Historical nominal raw result differs")
        old=read(source)
        retained.append(dict(slot=row["slot"],object_id=row["object_id"],version=row["version"],scope=row["scope"],
          original_result_sha256=sha(source),original_result=source.relative_to(ROOT).as_posix(),grasp_success=old["grasp_success"],
          engineering_gate_passed=old["engineering_gate_passed"],native_step_returns=old["step_returns"],max_lift_m=old["max_lift_m"]))
    if len(retained)!=3:raise ValueError("Historical nominal count")
    return retained

def format_num(value,decimals=4,factor=1):
    return "—" if value is None else f"{value*factor:.{decimals}f}"

def report(j):
    start=j["initial_charged"];end=j["final_charged"]
    doc=["# IP140-R2 三个错位负对照的执行结果","",
         f"本版依次处理槽10、11、12。计费从{start}/14到{end}/14，剩余{j['remaining']}次。",
         "**"+j["decision_zh"]+"**","",
         "## 实际动力学结果","",
         "|槽 / 物体|静态完成状态|物理返回步|末480步有效双侧接触|末480步聚合台面支撑|全程最大抬升 mm|结果|",
         "|---|---:|---:|---:|---:|---:|---|"]
    for r in j["misses"]:
        m=r["measured"] or {};terminal=m.get("terminal480_counts") or {};ph=r["physical"] or {};st=r["static_result"]
        doc.append(f"|{r['slot']} / {r['object_id']}|{st['states'] if st['states'] is not None else '—'}|"
              f"{ph.get('step_returns','—')}|{terminal.get('bilateral','—')}|{terminal.get('aggregate_table_support','—')}|"
              f"{format_num(ph.get('max_lift_m'),6,1000)}|{NAMES[r['classification']]}|")
    doc+=["","抬升以第240个返回步（settle末）的物体质心高度为基线。末窗由第4249–4728个返回步构成。"
          "聚合台面支撑使用同一步合法物体—台面接触点正法向力之和大于0.2 N。",
          "","## 物体运动与接触","","|物体|末步相对settle的 Δx / Δy / Δz (mm)|"
          "相对settle末的最远平移 (mm)|全程左右有效指垫接触步数|非指垫接触物体步数|检测到滑落|",
          "|---|---|---:|---:|---:|---|"]
    for r in j["misses"]:
        m=r["measured"] or {};d=m.get("displacement_settle_to_last_xyz_m");counts=m.get("full_contact_state_counts") or {}
        tr=m.get("max_translation_from_settle") or {}; ph=r["physical"] or {}
        vec=" / ".join(format_num(x,6,1000) for x in d) if d is not None else "—"
        doc.append(f"|{r['object_id']}|{vec}|{format_num(tr.get('m'),6,1000)}|"
                 f"{counts.get('left_effective','—')} / {counts.get('right_effective','—')}|"
                 f"{counts.get('nonpad_target_steps','—')}|"+{True:"是",False:"否",None:"—"}[ph.get("slipped")]+"|")
    doc+=["","## 静态准入与历史边界","",
         "未取得双侧入口的例外仅适用于本批三个冻结的miss身份。真实跨零的几何核验、非指垫接触、机械臂/环境碰撞、力速及其他动力学门继续执行。",
         ""]
    for r in j["misses"]:
        st=r["static_result"]
        if st["pre_lift_bilateral_entry_absent"] is True:
            doc.append(f"- 槽{r['slot']}在抬升前没有双侧入口，缺失侧为{', '.join(st['missing_entry_sides'] or [])}。")
        if r["verification_errors"] or r["classification"] not in {"correct_negative","misgrasp"}:
            doc.append(f"- 槽{r['slot']}的已记录结论为{NAMES[r['classification']]}。停止原因为{r['session_status'].get('error') or r['session_status'].get('stop') or '无会话回执'}。")
    doc+=["","历史nominal分别为021的smoke17槽14（开发纠错）、014的smoke19槽8、017的smoke21槽9（开发纠错）；三者均完成4728步并抓取成功。"
          "smoke21的三个miss保留静态不准入、动力学未知的记录。本批正式结果单独登记。",
          "","## 审核依据","",
          "各槽原生API位于相应 slots/NN/static_preflight/full_api/ 与 slots/NN/full_api/；逐步遥测与原始末窗位于 slots/NN/episodes/NN/。"
          "postrun_analysis/slotNN_object_state_and_contact.jsonl.gz 保存逐步物体位姿、速度、位移、抬升、接触及支撑摘要（仅为实际有遥测的槽生成）。"
          "result.json 含六轴力矩裕量、双指峰值、接触统计、源文件SHA256及每个连接的调用/返回核对。",
          "","历史完整性校验见 history_integrity_after.json；目录与逐文件SHA256随审查包交付。"
          "三物体工程正例与错位对照的用途限于开发，不能给出论文算法效果或独立确认。","",
          "**全部任务总进度、实验总进度**均待按固定工作分解重算。",""]
    return "\n".join(doc)

def main():
    if Path.cwd().resolve()!=ROOT:raise SystemExit("Run from the project root")
    start=read(V/"ledger_initial.json"); ledger=read(V/"slot_ledger.json")
    batch=read(V/"batch_session_result.json"); history=read(V/"history_integrity_after.json")
    registry=read(V/"candidate_registry.json")["rows"]
    rows=[measure_slot(i,next(r for r in registry if r["slot"]==i),ledger["slots"][i-1]) for i in SLOTS]
    n=sum(r["charged"] for r in rows)
    accounting=(start["starts"]==11 and ledger["starts"]==11+n and ledger["remaining_total_budget"]==14-ledger["starts"]
        and batch["cumulative_starts"]==ledger["starts"] and batch["retries"]==0)
    stage_ok=(accounting and history.get("passed") is True and all(row["classification"]=="correct_negative" for row in rows))
    limited=not stage_ok
    decision=("申请审核72回合Raw/A/C0配对开发比较。本次已停止；72回合尚未启动。" if stage_ok else
        "本次未达到申请72回合开发比较的条件。已停止；平台限制及未取得的证据按槽登记。")
    summary=dict(version=V.name,initial_charged=start["starts"],new_charges=n,final_charged=ledger["starts"],remaining=ledger["remaining_total_budget"],
      accounting_consistent=accounting,history_verification=history,historical_nominals=historical(),misses=rows,
      applies_to=sorted(SLOTS),independent_confirmation=False,request_72_development_comparison=stage_ok,ran_72=False,
      frozen_protocol_limit_observed=limited,decision_zh=decision)
    save(V/"result.json",summary)
    with (V/"审查报告.md").open("x",encoding="utf8",newline="\n") as out:out.write(report(summary))
    print([(r["slot"],r["classification"],r["charged"]) for r in rows],decision)

if __name__=="__main__":
    main()
