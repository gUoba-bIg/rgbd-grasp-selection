"""Independent count-level acceptance of the fixed confirmation experiment."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.accept_segmentation_expanded_v3 import verify_partition
from src.checkpoints import atomic_json, file_sha256, fingerprint


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def near_ge(value, threshold):
    return value >= threshold or math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12)


def decide(partitions, protocol):
    """Use unrounded scene-macro endpoints; tolerate floating representation only."""
    expected = set(protocol["validation_scenes"])
    if set(partitions) != {"original", "center_mse"}:
        raise ValueError("Exactly both frozen arms are required")
    threshold = protocol["acceptance"]
    gates = {}
    for arm, partition in partitions.items():
        if len(partition["scenes"]) != len(expected) or {row["scene"] for row in partition["scenes"]} != expected:
            raise ValueError("All frozen scenes are required")
        macro = partition["scene_macro"]
        if not all(math.isfinite(macro[key]) for key in ("foreground_pixel_iou", "instance_f1_iou50")):
            raise ValueError("Metrics must be finite")
        gates[arm] = (macro["foreground_pixel_iou"] >= threshold["scene_macro_foreground_iou_min"]
                      and macro["instance_f1_iou50"] >= threshold["scene_macro_instance_f1_min"]
                      and all(row["instances"]["tp"] >= threshold["minimum_tp_per_scene"] for row in partition["scenes"]))
    control, candidate = (partitions[arm] for arm in ("original", "center_mse"))
    f1_gain = candidate["scene_macro"]["instance_f1_iou50"] - control["scene_macro"]["instance_f1_iou50"]
    fg_drop = control["scene_macro"]["foreground_pixel_iou"] - candidate["scene_macro"]["foreground_pixel_iou"]
    by_scene = {row["scene"]: row for row in control["scenes"]}
    deltas = {str(row["scene"]): row["instances"]["f1_iou50"] - by_scene[row["scene"]]["instances"]["f1_iou50"] for row in candidate["scenes"]}
    rule = protocol["candidate_adoption"]
    checks = {"candidate_absolute_gate": gates["center_mse"],
              "f1_gain": near_ge(f1_gain, rule["scene_macro_f1_gain_over_control_min"]),
              "foreground_guard": near_ge(rule["scene_macro_foreground_iou_drop_max"], fg_drop),
              "scene_consistency": sum(value > 0 for value in deltas.values()) >= rule["minimum_scenes_with_positive_f1_gain"]}
    selected = "center_mse" if all(checks.values()) else "original" if gates["original"] else None
    return {"engineering_gate_by_arm": gates, "candidate_checks": checks, "selected_arm": selected,
            "macro_f1_gain": f1_gain, "macro_foreground_drop": fg_drop, "scene_f1_gains": deltas,
            "effect_confirmed": selected is not None}


def main():
    protocol_path = ROOT / "configs/segmentation_confirmation_v6.json"
    protocol = read(protocol_path)
    report_path = ROOT / protocol["report"]
    report = read(report_path)
    recovery_path = ROOT / protocol["resume_proof"]
    recovery = read(recovery_path)
    if report["status"] != "completed" or report["frames_completed"] != 80:
        raise AssertionError("Only complete80-frame paired evaluation may be accepted")
    if recovery["status"] != "passed" or not recovery["committed_prefix_byte_identical"]:
        raise AssertionError("Committed-prefix recovery was not verified")
    if recovery["report_sha256"] != file_sha256(report_path) or recovery["protocol_sha256"] != file_sha256(protocol_path):
        raise AssertionError("Recovery evidence does not bind the current report and protocol")
    directory = ROOT / protocol["output_directory"]
    binding = read(directory / "binding.json")
    binding_hash = fingerprint(binding)
    if binding["protocol"] != protocol or binding["protocol_sha256"] != file_sha256(protocol_path) or report["binding_fingerprint"] != binding_hash:
        raise AssertionError("Protocol and frame binding differ")
    for path, checksum in binding["source_sha256"].items():
        if file_sha256(ROOT / path) != checksum:
            raise AssertionError("Evaluation dependency changed: " + path)
    if file_sha256(ROOT / protocol["data_manifest"]) != binding["data_manifest_sha256"] or file_sha256(ROOT / protocol["data_audit"]) != binding["data_audit_sha256"]:
        raise AssertionError("Frozen data manifest or audit changed")
    for path, checksum in binding["data_files_sha256"].items():
        if file_sha256(ROOT / protocol["data_root"] / path) != checksum:
            raise AssertionError("Frozen confirmation data changed: " + path)
    for arm, evidence in binding["checkpoint_evidence"].items():
        expected = protocol["arms"][arm]["checkpoint_sha256"]
        if evidence["checkpoint_sha256"] != expected or file_sha256(ROOT / evidence["checkpoint"]) != expected:
            raise AssertionError("Frozen model checkpoint changed: " + arm)
    rows_by_arm = {arm:[] for arm in ("original", "center_mse")}
    for scene in protocol["validation_scenes"]:
        for frame in protocol["frames"]:
            key = "scene_{:04d}_{:04d}".format(scene, frame)
            path = directory / "frames" / (key + ".json")
            row, receipt = read(path), read(directory / "frames" / (key + ".sha256.json"))
            expected_identity = {"split":"confirmation_validation","scene":scene,"frame":"{:04d}".format(frame),"camera":"realsense"}
            if receipt != {"binding_fingerprint":binding_hash,"identity":expected_identity,"payload_file":path.name,"payload_sha256":file_sha256(path)}:
                raise AssertionError("Paired frame receipt differs: " + key)
            if row["identity"] != expected_identity or row["binding_fingerprint"] != binding_hash or report["cache_payload_fingerprints"][key] != fingerprint(row):
                raise AssertionError("Report and paired frame evidence differ: " + key)
            for arm in rows_by_arm:
                rows_by_arm[arm].append(row["arms"][arm]["metrics"])
            if row["image"] and file_sha256(directory / row["image"]["file"]) != row["image"]["sha256"]:
                raise AssertionError("Fixed image changed: " + key)
    partitions = {arm: report["arms"][arm]["partition"] for arm in ("original", "center_mse")}
    if any(partitions[arm]["frames"] != rows_by_arm[arm] for arm in rows_by_arm):
        raise AssertionError("Aggregate frames differ from committed pairs")
    checked = {arm: verify_partition(value, protocol["validation_scenes"], protocol["frames"]) for arm, value in partitions.items()}
    decision = decide(checked, protocol)
    if decision["selected_arm"] != report["decision"]["selected_arm"]:
        raise AssertionError("Independent adoption decision differs")
    if decision["engineering_gate_by_arm"] != report["decision"]["engineering_gate_by_arm"]:
        raise AssertionError("Independent absolute gate differs")
    check_names = {"candidate_absolute_gate":"candidate_passes_absolute_gate", "f1_gain":"scene_macro_f1_gain",
                   "foreground_guard":"scene_macro_foreground_guard", "scene_consistency":"scene_consistency"}
    if {check_names[key]:value for key,value in decision["candidate_checks"].items()} != report["decision"]["candidate_adoption_checks"]:
        raise AssertionError("Independent relative-rule checks differ")
    if decision["scene_f1_gains"] != report["decision"]["scene_f1_gain_candidate_minus_control"]:
        raise AssertionError("Independent per-scene improvements differ")
    for first, second in ((decision["macro_f1_gain"],report["decision"]["scene_macro_f1_gain_candidate_minus_control"]),
                          (decision["macro_foreground_drop"],report["decision"]["scene_macro_foreground_drop_control_minus_candidate"])):
        if not math.isclose(first,second,rel_tol=0,abs_tol=1e-12):
            raise AssertionError("Independent relative metric differs")
    summary = {"轮次":"第六轮固定模型新场景工程确认", "执行状态":"完成", "效果验收":"通过预设工程确认规则" if decision["effect_confirmed"] else "未通过预设工程确认规则",
               "场景":protocol["validation_scenes"], "帧数":80, "每帧模型数":2, "本轮新增训练更新":0,
               "独立复核":decision, "指标":checked, "协议SHA256":file_sha256(protocol_path),
               "逐帧机器报告SHA256":file_sha256(report_path), "恢复报告SHA256":file_sha256(recovery_path),
               "边界":["首次使用的固定确认子集，并非官方测试集；看过结果后归入开发验证", "确定性编号选取不是代表性随机抽样", "统计单位是5场景，不能把80帧当80个独立样本", "没有新增训练、调阈值或选择中间权重", "未完成抓取AP、碰撞安全或机器人成功率评价", "其余5个验证场景、校准与官方测试未使用"]}
    target = ROOT / "results/第六轮新场景确认验收.json"
    if target.exists():
        if read(target) != summary:
            raise AssertionError("Completed acceptance differs; existing result preserved")
        print("Existing independent acceptance verified unchanged.")
    else:
        atomic_json(target, summary)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
