# V1 补充实验复核说明

版本为 `V1_supplement_robustness_admission01`。科学代码使用标签 `paper_v1_r6r7_frozen`，对应提交 `e184a77ab06fe1f1c0a0f1514b04da65befcf8ed`。本目录增加数据编排、证据核验和统计。

## 核对一条结果

1. 在 `study/run/method_ledger_continued.json` 中查找试次编号及返回回执的 SHA256。`study/run/method_ledger.json` 保存首次暂停时的 74 份已登记结果。`study/method_reporting_resume01/release.json` 列出原回执、从现有数据补登记的 ADM-001 及当时尚未启动的 57 个编号。
2. 在 `study/run/jobs/<编号>/return.json` 核对结果文件哈希，再以 `study/trials/<编号>/result.json` 中记录的哈希核对同目录 `intent.json`。
3. 意图给出冻结场景、观测、候选池、排序、轨迹绑定以及原 R6 意图。单因素意图另外绑定 `preregistration/observation_fingerprint_index.json` 中该条件自身的 P3 和 P5 输入。240 个原条件的张量哈希与正式登记比较，480 个单因素的 P5 哈希与当次扰动实现的实测回执比较。
4. `static_preflight/path_result.json` 和 `static_integrity.json` 给出静态结论。进入动力学的试次另有 `episode/telemetry.jsonl.gz`、`episode/result.json`、`dynamic_api/` 和 `dynamic_integrity.json`。运行阶段的全部原始材料由本地证据索引定位。
5. 历史对照分别见 `analysis/historical_trial_paths.csv`、`preregistration/original_control_witnesses.json` 和相关 V1 来源清单。共同候选池为空或其他独立安全门先行停止时，审查者可沿同一条链核对其实际物理步数。

## 数据和统计

`protocol.json` 登记实验单位、筛选、物理预算、观测对应和置信区间。所有条件共享原基础场景。跨条件对应通过冻结 NMS 邻域和一对一位置匹配得到。诊断样本中的碰撞与历史成功对照按结局富集，适用范围见有效性报告。所有比例都可以从相应的试次行重新计数。

`freeze_population.py` 登记 120 个基础场景、720 个观测和 60 个诊断案例。`study_runner.py`、`runtime_stages.py` 与 `single_factors.py` 运行新前向和原排序。`method_seal.py` 为每种单因素按入围单元哈希取至多 6 个，并绑定三个方法的独立意图。`method_trial.py` 调用冻结执行器。`diagnostic_gate.py` 仅在登记的 40 个未认证案例中处理入口证书缺失。`final_statistics.py`、`history.py` 和 `r7_stability.py` 中的比率、缺失与配对计算可独立审查。

本次执行的原始结果目录采用一次创建。独立重现实验需要新的输出位置、原锁定依赖和相同的模型、资产及观测材料。冷启动使用正式 V1 的三套锁定环境。轻量包中的 SHA256 和审查索引可在核对证据时独立使用。
