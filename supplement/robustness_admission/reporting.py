"""Write the supplementary tables and Chinese manuscript material from verified outputs only."""
from collections import Counter
import csv
import sys
from common_v1 import *
from cluster_statistics import estimate
from history import table
from evidence_ledger import method_ledger

def textfile(path,body):
    file=OUT/path;file.parent.mkdir(parents=True,exist_ok=True)
    with file.open("x",encoding="utf8",newline="\n") as f:f.write(body.strip()+"\n")
def rate(x):
    if x["value"] is None:return "未定义"
    s=f'{100*x["value"]:.1f}%'
    if x.get("ci95"):s+=f' [{100*x["ci95"][0]:.1f}, {100*x["ci95"][1]:.1f}]'
    return s
def num(v):return "未定义" if v is None else f"{v:.3f}"
def main():
    from figures import plt,save
    need((OUT/"admission/completed_stages.json").is_file(),"terminal gate")
    from terminal_verification import run as check_hold
    check_hold()
    final_window=read(OUT/"admission/final_window_crosscheck.json")
    sample=read(OUT/"preregistration/robustness_sample.json")
    stats=read(OUT/"robustness/sampled_execution.json")
    diag=read(OUT/"diagnostic/validity.json")
    old=read(OUT/"analysis/method_rates.json");cold=read(OUT/"admission/cold_3of3.json");dst=read(OUT/"preregistration/diagnostic_sample.json")
    hist=read(OUT/"analysis/historical_stability_summary.json");six=read(OUT/"robustness/rank_summary.json")
    observation_index=read(OUT/"robustness/inference_reconciliation.json")
    common_stability=read(OUT/"analysis/six_methods_80units_rank_stability.json")
    original_pairs=read(OUT/"analysis/paired_methods.json")
    ledger=read(method_ledger())
    run_counts={"cold":{"cases":cold["complete_cases"],"P3":sum(v["model_calls"]["P3"]["returned"] for v in cold["cases"]),
                       "P5":sum(v["model_calls"]["P5"]["returned"] for v in cold["cases"]),"native_steps":sum(v["native_returns"] for v in cold["cases"])},
                "inference":observation_index,"supplemental_methods":{"assigned":ledger["assigned"],"adjudicated":ledger["adjudicated"],
                       "unit_selected_robustness":stats["assigned"],"robustness_entered_physics":stats["physical_trials"],
                       "diagnostic_assigned":sum(v["n"] for v in diag["groups"]),"diagnostic_entered_physics":sum(v["dynamics"] for v in diag["groups"]),
                       "robustness_native_steps":sum(v["steps"] for v in stats["groups"] if v["condition"]=="all"),
                       "diagnostic_native_steps":sum(v["steps"] for v in diag["groups"]),"per_trial_ledger_sha256":sha(method_ledger()),
                       "per_trial_ledger_path":method_ledger().relative_to(OUT).as_posix()},
                 "archived_R6":720,"archived_R7":240}
    write("admission/run_ledger.json",run_counts)
    oldrows=[r for r in old if r["stratum"]=="all" and r["population"]!="common_80_each"]
    with (OUT/"robustness/selection_divergence.csv").open(encoding="utf-8-sig",newline="") as f:
        diver=list(csv.DictReader(f))
    divergence=[]
    for mode in MODES:
        for pair in [("Raw","A"),("Raw","C0"),("A","C0")]:
            sub=[r for r in diver if r["condition"]==mode and (r["method1"],r["method2"])==pair]
            need(len(sub)==120,"divergence assignment")
            p=estimate(sub,lambda r:r["different"]=="True",lambda r:1)
            divergence.append(dict(mode=mode,methods=" / ".join(pair),n=120,different=int(p["numerator"]),estimate=p))
    write("robustness/divergence_summary.json",divergence)
    mode_cn={"clean":"清洁观测","depth_only":"深度高斯噪声","shift_only":"水平配准偏移","missing_only":"随机缺深","block_only":"矩形遮挡","joint":"联合退化"}
    s=["# V1 补充实验报告","《基于RGB-D实例分割与几何不确定性建模的六自由度机器人抓取选择研究》","## 身份与实验单位",
       f"冻结标签为 `{TAG}`，提交为 `{COMMIT}`。源文件、模型、配置、独立解释器和原始证据的核验记录见 `admission/v1_identity.json`。三个代表案例在新安装环境中从原始 RGB-D 重新推理，并沿用原评分、轨迹、执行器与评价器。"]
    for r in cold["cases"]:s.append(f'- {r["trial_id"]}　{r["classification"]}，返回 {r["native_returns"]} 个物理步。')
    s.extend(["R6 的 240 个配对单元来自 120 个基础场景，每种观测及每种方法共用对应稳定初态。R7 的三个机制消融各有 80 个方法试次。六种方法的横向比较另限于 R7 覆盖的同一组 80 个单元。后续单因素执行与 60 个诊断案例分别记账。",
      "## 既有正式证据的补充分析","| 组别 | 成功/全部尝试 | 全部尝试成功率及 95% 区间 | 静态准入数 | 动力学条件成功率及 95% 区间 |","|---|---:|---|---:|---|"])
    for r in oldrows:s.append(f'| {r["method"]} | {r["all_attempt_success"]["numerator"]:.0f}/{r["n"]} | {rate(r["all_attempt_success"])} | {r["static_admission"]["numerator"]:.0f} | {rate(r["conditional_success"])} |')
    s.extend(["区间为基础场景聚类重采样 20,000 次的百分位区间。每次抽中场景时共同保留该场景的观测条件与方法。分母为零时保留缺失。",
      "在两种观测下独立生成的候选池使用单元内编号。跨条件对应采用冻结 NMS 的双候选邻域判定与一对一匹配。单位编号直接相等不构成几何身份对应。Top-K 用匹配交集的 Jaccard 重合度，K 为 1、3、5、10、20。Kendall τ-b 仅在有至少两对匹配且分数存在可比较顺序的场景上定义。","| 方法 | R6 clean→joint 的 Top-1 变化/120 | Top-3 平均 Jaccard | 可计算分数 τ-b 的场景 | 平均 τ-b |","|---|---:|---:|---:|---:|"])
    for r in hist:s.append(f'| {r["method"]} | {r["top1_change"]["numerator"]:.0f} | {num(r["top3_jaccard"]["value"])} | {r["score_tau_b"]["denominator"]:.0f} | {num(r["score_tau_b"]["value"])} |')
    associations=read(OUT/"analysis/score_margin_and_outcomes.json")
    s.extend(["下表给出正式 R6 中的描述性分箱。对每种方法分别取锁定分数以及旧 E 内前两名分差的最低、最高非空分箱。相同数值留在同一箱，故箱大小可以不同。分差仅对旧 E 内至少两个候选的单元定义。",
              "| 方法 | 指标 | 分箱 | 有定义的本箱试次 | 静态准入/试次 | 抓取成功/试次 |","|---|---|---|---:|---:|---:|"])
    for m in METHODS:
        for field,label in (("observed_score","选中分数"),("margin_in_E","E 内排序分差")):
            rows=sorted((r for r in associations if r["method"]==m and r["field"]==field and r["bin"]!="undefined"),key=lambda r:int(r["bin"]))
            if not rows:continue
            bounds=[("全部",rows[0])] if len(rows)==1 else [("低",rows[0]),("高",rows[-1])]
            for edge,r in bounds:
                s.append(f'| {m} | {label} | {edge} | {r["assigned"]} | {int(r["static_admission"]["numerator"])}/{r["assigned"]} | {int(r["all_attempt_success"]["numerator"])}/{r["assigned"]} |')
    s.append("在同一组 40 个基础场景、80 个观测单元上，六种方法的完整 clean/joint 匹配、Top-K 和 Kendall 结果见 `analysis/six_methods_80units_rank_stability.json`。")
    s.append("R6 的 240 个配对单元中，锁定候选不同的次数为 "+ "、".join(f'{p["a"]} 与 {p["b"]} {p["candidate_diversion"]["numerator"]:.0f} 次' for p in original_pairs if p["population"]=="R6_240_each")+"。这些数值描述同一观测内方法间的选择分歧。上表比较同一方法跨观测的选择。")
    s.extend(["既有分数、旧 E 内第一与第二名分差、锁定候选在全池中的名次，与静态准入及最终成功的分箱对照见 `analysis/score_margin_and_outcomes.json`。单候选没有分差。风险—覆盖率曲线保留原始并列分数，并把全部分派试次作为覆盖率分母。其中的风险为已保留试次的实际失败比例。未执行候选的潜在成功率无法由该曲线识别。",
      "## 单因素鲁棒性","每个场景在六种条件下各运行一次 V1 的 P5、P3，并对共享候选池计算三种冻结排序及旧 E。单因素沿用同一场景的 joint 随机实现。网络网格上的水平偏移为 +3 像素，轴深噪声标准差 5 mm，随机缺深概率 15%，矩形为 120×160 像素。每种干预保留原 RGB。共享池与入选结果的输入、权重、种子和原始行见逐单元回执。",
      "| 条件 | 方法 | Top-1 变化比例及聚类区间 | Top-3 Jaccard | 可计算 τ-b 的场景 |","|---|---|---|---:|---:|"])
    for r in six:s.append(f'| {mode_cn[r["condition"]]} | {r["method"]} | {rate(r["top1_change"])} | {num(r["top3_jaccard"]["value"])} | {r["score_tau_b"]["denominator"]:.0f} |')
    s.append("### 分流与抽样执行")
    s.extend(["| 条件 | Raw/A 分流 | Raw/C0 分流 | A/C0 分流 |","|---|---:|---:|---:|"])
    for m in MODES:s.append("| "+mode_cn[m]+" | "+" | ".join(f'{r["different"]}/120' for r in divergence if r["mode"]==m)+" |")
    s.extend(["以下各条件与同一基础场景的 clean 配对。Top-1 栏是发生变化的场景数，每格分母为 120。Jaccard 栏依次列出 Raw、A、C0 的有效样本均值；共同为空的 Top-K 保留为缺失。",
              "| 条件 | Raw Top-1 变化 | A Top-1 变化 | C0 Top-1 变化 | 三组 Top-3 Jaccard | 可估计 τ-b 的场景数（Raw、A、C0） |",
              "|---|---:|---:|---:|---|---|"])
    for mode in MODES[1:]:
        rr=[next(x for x in six if x["method"]==m and x["condition"]==mode) for m in METHODS]
        counts=" | ".join(str(int(x["top1_change"]["numerator"])) for x in rr)
        overlap="、".join(num(x["top3_jaccard"]["value"]) for x in rr)
        taus="、".join(str(int(x["score_tau_b"]["denominator"])) for x in rr)
        s.append(f"| {mode_cn[mode]} | {counts} | {overlap} | {taus} |")
    s.extend(["对四个新增条件，以任一方法在旧 E 筛选前的 Top-1 发生几何或有无变化作为入围条件，随后按预登记的单元哈希取样。每个选中单元的三个方法各执行一次。未入围的场景没有新增动力学结果。",
      "| 条件 | Top-1 变化单元 | 抽样单元 | Raw 成功/分派 | A 成功/分派 | C0 成功/分派 |","|---|---:|---:|---:|---:|---:|"])
    for mode in MODES[1:5]:
        rs={r["method"]:r for r in stats["groups"] if r["condition"]==mode}
        s.append(f'| {mode_cn[mode]} | {len(sample["all_changed_in_hash_order"][mode])} | {len(sample["sampled"][mode])} | '+" | ".join(f'{rs[m]["success"]}/{rs[m]["n"]}' for m in METHODS)+" |")
    blind=read(OUT/"admission/outcome_blind_draw_verification.json")
    need(blind["consistent"],"sampling reproduction mismatch")
    s.append(f'首次封存抽样时，共用包装程序也打开了诊断清单，内含历史对照组别。抽样表达式只使用 Top-1 变化及单元哈希。为核对此点，独立进程仅获准读取 {blind["opened_inputs"]} 份观测计划、候选池和排序输入，并禁止读取诊断清单及试次结果。四个入围全集、哈希顺序和 24 个被抽单元均与事前封存完全一致。该复核未追加试次。')
    s.extend([f'单因素共分派 {stats["assigned"]} 个方法试次，实际进入动力学 {stats["physical_trials"]} 次。结果只适用于所抽的 Top-1 变化单元。聚类区间、配对差和失败去向见 `robustness/sampled_execution.json`。',"## 静态证书有效性",
      f'R6 有 474 条入口未认证试次。按候选和路径去重后有 {dst["population"]["uncertified"]} 个候选，其中按物体、观测和代表方法分层抽取 40 个。另抽取 10 个历史成功候选及 10 个有原始提前接触或禁止碰撞见证的候选。仅前 40 例可跳过入口证书，直接几何、碰撞、穿透、速度、控制和抬升保持判据沿用 V1。实际双侧认证状态、跳过的具体标签及原始返回分别保留。'])
    group_cn={"uncertified":"原入口未认证","success_control":"历史成功对照","collision_control":"提前接触或禁止碰撞对照"}
    for r in diag["groups"]:s.append(f'- {group_cn[r["group"]]}共 {r["n"]} 例，进入动力学 {r["dynamics"]} 例，成功 {r["success"]} 例。')
    reporting_fix=OUT/"admission/diagnostic_native_label_verification.json"
    if reporting_fix.is_file():
        fix=read(reporting_fix);ev=fix["finite_stop_fields"];steps=fix["step"]
        s.append(f'首轮外层汇总在 {fix["trial_id"]} 暂停。原执行器在闭合期第 {steps} 步记录工程失败。四角净距最小值为 {min(ev["raw_gaps_m"])*1e9:.6f} nm；原生最近点距离为 {ev["raw_closest_rows"][0][8]*1e9:.6f} nm，双指间流形数为 {ev["opposing_solver_manifold_count"]}。全部 {fix["native"]["native_calls"]} 次调用和 {steps} 个物理步均有返回，数值有限。原始结果、停批账本和轨迹保留；外层补录了这个已确定失败的细类。续接清单只包含当时尚未启动的 57 个诊断。')
    s.append(f'两批合计有 {final_window["with_complete_window"]} 个完成全程且通过工程门的动力学试次。对这些试次的末 480 步，逐步由物体实际位置与沉降末位置复算抬升，由原始接触点复算双侧有效接触和台面合力支撑，并结合冻结评价器记录的滑落与已形成抓取状态核对终判。逐次见 `admission/final_window_crosscheck.json`。')
    cs=diag["verified_sample_matrix"];r0=diag["observed_rescue_per_assigned_uncertified"];sv=cs["metrics"]
    s.append(f'40 个未认证候选中经新执行观察到的成功比例为 {rate(r0)}。在取得新动力学结局的诊断样本内，FN/(TP+FN) 为 {rate(sv["false_rejection_FN_over_TP_plus_FN"])}。二者回答不同问题，对照采用分层且按历史结局富集的抽样设计。静态停止案例的实际动力学结果保留为未知。')
    s.extend(["## 图表与适用范围",
      "图 1　R6 与 R7 的经验风险—覆盖率曲线。横轴为留选数除以全部分派数，纵轴为留选试次的失败比例。两行分别对应各 240 次的正式方法和各 80 次的消融方法。阴影为固定阈值处风险的场景聚类区间。",
      "图 2　V1 锁定候选、静态放行和最终成功的路径。三个比例在每个方法内部使用同一个全部尝试分母，误差线为场景聚类区间。",
      "图 3　各干预相对同一场景 clean 的 Top-1 变化和 Top-3 重合。高变化同时包含候选生成及排序变化。条件内有可匹配候选时另报 Kendall，见数据表。",
      "图 4　原 V1 双侧证书与重新观察的抓取结局。单独显示未进入动力学的案例。",
      "本补充使用原 V1 场景，评估结论限于这三个物体、冻结评分器及已有相机和执行配置。固定候选池内部的评分变化与跨观测的候选生成变化分别报告。该批次不增加正式 R6 的样本量。"])
    textfile("V1补充实验报告.md","\n\n".join(z if not z.startswith("|") else z for z in s).replace("|\n\n|","|\n|"))
    dr=["# 静态准入有效性","原 V1 截止时双侧证书与本轮新证据的对应情况如下。","| 原证书 | 新执行成功 | 新执行失败 | 未进入动力学 |","|---|---:|---:|---:|"]
    cc=cs["cells"]
    cells=np_array([[cc["TP"],cc["FP"],cs["no_new_physical_outcome"]["True"]],[cc["FN"],cc["TN"],cs["no_new_physical_outcome"]["False"]]])
    for n,row in zip(["有双侧证书","截至原停止点未取得双侧证书"],cells):dr.append("| "+n+" | "+" | ".join(str(int(v)) for v in row)+" |")
    dr.extend(["一次真实抓取能验证该次动作成功。安全门在物理步前停止时，其物理抓取结局仍未知。运行到禁止碰撞可以证明该次轨迹发生碰撞。普通未抓住不能证明这类几何在物理上普遍不可执行。","| 已验证样本内的预测指标 | 估计及场景聚类 95% 区间 |","|---|---|"])
    for k,v in sv.items():dr.append(f"| {k} | {rate(v)} |")
    u=next(r for r in diag["groups"] if r["group"]=="uncertified")
    dr.append(f'40 个未认证候选中，共 {u["dynamics"]} 个取得新动力学结局，{u["success"]} 个实际抓取成功。将全部 40 个作为分母，直接观察到的救回比例为 {rate(r0)}。只对实际进入动力学者计算，该比例为 {rate(diag["observed_rescue_per_executed"])}。')
    dr.append("各组第一次终止的去向见下表。未进入动力学者的真实抓取能力保留为未知。")
    dr.extend(["| 诊断组 | 归类 | 例数 |","|---|---|---:|"])
    for r in diag["groups"]:
        for reason,value in sorted(r["outcome"].items()):
            dr.append(f'| {group_cn[r["group"]]} | {reason} | {value} |')
    categories={"possible_false_rejection_observed_success":"原证书缺失，本轮实测抓取成功，支持该次可能误拒",
                "documented_violation_on_specific_path":"该条具体轨迹有明确的提前接触、禁止碰撞或违规证据",
                "insufficient_physical_evidence":"静态停止，实际受载能力仍缺乏证据",
                "observed_failed_attempt_without_general_impossibility_proof":"取得一次真实执行失败，无法推出普遍不可执行"}
    dr.append("40 个入口未认证案例的可支持解释如下。")
    for key,label in categories.items():
        dr.append(f'- {label}，{u["attribution_counts"].get(key,0)} 例。')
    dr.append("目标在静态路径检查中固定。若未取得入口证书、代理指位跟随参考命令，代理净距越限只证明该静态检查失败；物体在真实受载后是否会移动仍需动力学证据。")
    dr.append("候选层设计权重、未验证数、救回比例和保守上界见 diagnostic/validity.json。历史成功和明确碰撞被特意抽作对照，样本内的混淆矩阵不支持直接外推到 R6 全体 720 次。若各次聚类重采样得到相同结果，经验区间会退化为一个点。此时样本之外的误拒风险仍有不确定性。")
    textfile("静态准入有效性报告.md","\n\n".join(dr).replace("|\n\n|","|\n|"))
    fig,ax=plt.subplots(figsize=(7.5,3.6));ax.imshow(cells,cmap="Blues",vmin=0);ax.set_xticks(range(3),["Grasp success","Observed failure","No physical outcome"]);ax.set_yticks(range(2),["Bilateral certificate","No bilateral certificate"])
    for i in range(2):
        for j in range(3):ax.text(j,i,str(int(cells[i,j])),ha="center",va="center",color=("white" if cells[i,j]>cells.max()*.54 else "black"))
    ax.set_title("Original V1 certificate and new diagnostic outcome (60 cases)");fig.tight_layout();save(fig,"Fig4_static_certificate_verification")
    short=["# 论文增补建议","建议在实验设置中增设“配对单因素扰动与诊断抽样”小节，交代 120 个同源稳定初态、六种观测、候选对应方法、候选去重和场景聚类重采样。",
        "结果部分可在原 R6、R7 的正式表格之后独立列出补充结果。图 1、图 2 用于说明评分与执行路径，图 3 用于说明候选排序对扰动的敏感性，图 4 与验证分层说明静态证书的作用范围。",
        f'正文可报告：对 40 个原未认证候选开展一次性诊断，其中 {int(r0["numerator"])} 个在保留其余工程门时成功。该结果描述抽样候选，不修改原正式裁定。根据已观测的动力学和未验证比例解释区间，避免把证书缺失直接归因为视觉失败。',
        "讨论中将候选生成变化、排名变化、执行前硬门与物理执行分开。每种单因素的抽样执行限于最多六个发生 Top-1 变化的单元，不能用于全部场景的无偏方法优劣判断。稿件源文件保持原状。"]
    textfile("论文新增章节与图表建议.md","\n\n".join(short))
def np_array(rows):
    import numpy as np
    return np.asarray(rows,dtype=float)
if __name__=="__main__":main()
