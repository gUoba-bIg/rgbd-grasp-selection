"""Check final pilot evidence independently and export the acceptance summary."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoints import atomic_json, file_sha256
from scripts.train_segmentation_expanded_v3 import learning_rate_for_update
from scripts.render_segmentation_expanded_v3 import run as render
import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(first, second):
    if not math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("Recorded metric disagrees with recomputation")


def verify_counts(values):
    tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
    if any(type(value) is not int or value < 0 for value in (tp, fp, fn)):
        raise AssertionError("Instance counts must be nonnegative integers")
    close(values["precision_iou50"], tp / (tp + fp) if tp + fp else 0.0)
    close(values["recall_iou50"], tp / (tp + fn) if tp + fn else 0.0)
    close(values["f1_iou50"], 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def verify_partition(partition, scenes, frames):
    rows = partition["frames"]
    identities = {(row["scene"], row["frame"]) for row in rows}
    expected = {(scene, "{:04d}".format(frame)) for scene in scenes for frame in frames}
    if identities != expected or len(rows) != len(expected):
        raise AssertionError("Evaluation must contain exactly the frozen frame identities")
    if partition["frame_count"] != len(rows) or partition["scene_count"] != len(scenes):
        raise AssertionError("Partition counts differ from its records")
    if len(partition["scenes"]) != len(scenes) or {row["scene"] for row in partition["scenes"]} != set(scenes):
        raise AssertionError("Scene summary identities differ")
    for row in rows:
        verify_counts(row["instances"])
        if not 0 <= row["foreground_intersection"] <= row["foreground_union"] <= 240 * 320:
            raise AssertionError("Invalid per-frame foreground area")
        close(row["foreground_pixel_iou"], row["foreground_intersection"] / row["foreground_union"] if row["foreground_union"] else 1.0)
    verify_counts(partition["instances"])
    totals = {key: sum(row["instances"][key] for row in rows) for key in ("tp", "fp", "fn")}
    if any(partition["instances"][key] != value for key, value in totals.items()):
        raise AssertionError("Instance counts disagree with frame totals")
    denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    close(partition["instances"]["f1_iou50"], 2 * totals["tp"] / denominator if denominator else 0.0)
    intersection = sum(row["foreground_intersection"] for row in rows)
    union = sum(row["foreground_union"] for row in rows)
    close(partition["foreground_pixel_iou"], intersection / union if union else 1.0)
    scene_iou, scene_f1 = [], []
    for scene in scenes:
        subset = [row for row in rows if row["scene"] == scene]
        numerator = sum(row["foreground_intersection"] for row in subset)
        denominator = sum(row["foreground_union"] for row in subset)
        scene_iou.append(numerator / denominator if denominator else 1.0)
        tp, fp, fn = (sum(row["instances"][key] for row in subset) for key in ("tp", "fp", "fn"))
        scene_f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        recorded = next(row for row in partition["scenes"] if row["scene"] == scene)
        verify_counts(recorded["instances"])
        if recorded["frame_count"] != len(subset) or any(recorded["instances"][key] != value for key, value in zip(("tp", "fp", "fn"), (tp, fp, fn))):
            raise AssertionError("Scene summary counts differ from its frames")
        close(recorded["foreground_pixel_iou"], scene_iou[-1])
        close(recorded["instances"]["f1_iou50"], scene_f1[-1])
    close(partition["scene_macro"]["foreground_pixel_iou"], sum(scene_iou) / len(scene_iou))
    close(partition["scene_macro"]["instance_f1_iou50"], sum(scene_f1) / len(scene_f1))
    return {key: partition[key] for key in ("frame_count", "scene_count", "foreground_pixel_iou", "instances", "scene_macro", "scenes")}


def main():
    directory = ROOT / "experiments/segmentation_expanded_v3/rgbd_original"
    aggregate_path = ROOT / "results/P2/segmentation_expanded_v3.json"
    aggregate = read(aggregate_path)
    result = read(directory / "result.json")
    if aggregate["result_sha256"] != file_sha256(directory / "result.json") or aggregate["result"] != result:
        raise AssertionError("Run result and aggregate differ")
    if any(aggregate[key] != file_sha256(directory / filename) for key, filename in
           (("history_sha256", "history.json"), ("config_sha256", "config.json"))):
        raise AssertionError("Run configuration/history differ from final aggregate")
    history = read(directory / "history.json")
    protocol = read(ROOT / "configs/segmentation_expanded_v3.json")
    if [row["step"] for row in history] != list(range(1, 1601)):
        raise AssertionError("Need exactly 1600 committed optimizer updates")
    seen = {}
    for row in history:
        if not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm_before_clip"]):
            raise AssertionError("Nonfinite training evidence")
        close(row["learning_rate_used"], learning_rate_for_update(row["step"], protocol))
        for sample in row["samples"]:
            identity = (sample["scene"], sample["frame"])
            seen[identity] = seen.get(identity, 0) + 1
    expected = {(scene, "{:04d}".format(frame)) for scene in protocol["train_scenes"] for frame in protocol["frames"]}
    if set(seen) != expected or set(seen.values()) != {20}:
        raise AssertionError("History does not show exactly 20 passes over training only")
    if not result["validation_evaluated"] or result["optimizer_steps"] != 1600 or result["status"] != "completed":
        raise AssertionError("Final evaluation is not complete")
    evaluation = result["evaluation"]
    checked = {name: verify_partition(evaluation[name], scenes, protocol["frames"]) for name, scenes in (
        ("train", protocol["train_scenes"]), ("validation", protocol["validation_scenes"]),
        ("fresh_validation", protocol["fresh_validation_scenes"]),
        ("previously_seen_validation", protocol["previously_seen_validation_scenes"]))}
    for name, scenes in (("fresh_validation", protocol["fresh_validation_scenes"]),
                         ("previously_seen_validation", protocol["previously_seen_validation_scenes"])):
        original = {(row["scene"], row["frame"]): row for row in evaluation["validation"]["frames"] if row["scene"] in scenes}
        subset = {(row["scene"], row["frame"]): row for row in evaluation[name]["frames"]}
        if original != subset:
            raise AssertionError("Validation subsets differ from the full validation frame records")
    fresh = checked["fresh_validation"]
    gate = protocol["acceptance"]
    passed = (fresh["scene_macro"]["foreground_pixel_iou"] >= gate["fresh_scene_macro_foreground_iou_min"]
              and fresh["scene_macro"]["instance_f1_iou50"] >= gate["fresh_scene_macro_instance_f1_min"]
              and all(row["instances"]["tp"] >= gate["minimum_tp_per_fresh_scene"] for row in fresh["scenes"]))
    if passed != evaluation["acceptance_gate"]["passed"]:
        raise AssertionError("Acceptance decision disagrees with frozen criteria")
    expected_decision = "engineering_continuation_gate_passed" if passed else "engineering_continuation_gate_not_met"
    if result["decision"] != expected_decision:
        raise AssertionError("Recorded decision differs from the recomputed gate")
    recovery = read(ROOT / "results/P2/segmentation_expanded_v3_resume.json")
    if recovery["status"] != "passed" or not all(recovery["bitwise_equal"].values()):
        raise AssertionError("GPU recovery verification did not pass")
    visuals = render(directory, ROOT / "results/P2/segmentation_expanded_v3_visuals", torch.device("cuda"))
    summary = {
        "轮次": "第三轮扩大样本与场景外验证", "执行状态": "完成", "decision": result["decision"],
        "工程检查": {"数据文件数": 1020, "训练帧": 160, "验证帧": 80,
                     "全项目组件测试通过数": 85, "新增验收算术测试通过数": 3,
                     "参数更新": 1600, "训练轮数": 20, "每训练帧使用次数": 20,
                     "保存间隔步数": 10, "保留完整检查点": 3,
                     "GPU中断恢复完整状态逐位一致": True, "实际拉闸断电测试": False,
                     "全部指标重新汇总核对": True, "固定验证图核对数": len(visuals["frames"])},
        "效果验收": "通过预设工程继续投入门槛" if passed else "未通过预设工程继续投入门槛",
        "门槛": gate, "指标": checked,
        "下一步": "保留本轮冻结结果，下一版可安排同预算RGB对照及误差分析；不能据单模型结果宣称优势" if passed else
                    "保留本轮结果和1600步最终模型；先定位新场景实例分割误差，再另定新版本实验，不自动增加轮数或改阈值",
        "解释边界": ["指标计算分辨率240×320，非原图分辨率",
                     "实例F1使用掩膜IoU≥0.50匹配，不是掩膜AP",
                     "四个首次验证场景在本轮评估后成为开发验证，不能继续称为未见盲测",
                     "场景13所有16帧单独列为既见验证", "只跑一个随机种子，未做RGB对照或方法消融",
                     "未使用校准集或官方测试场景", "没有抓取AP、碰撞安全或真实机器人成功率结论",
                     "数据来自固定版本公开第三方镜像，未核验官方整包字节一致性"],
        "证据": {"结果": str(aggregate_path.relative_to(ROOT)), "结果SHA256": file_sha256(aggregate_path),
                 "配置": "configs/segmentation_expanded_v3.json", "配置SHA256": file_sha256(ROOT / "configs/segmentation_expanded_v3.json"),
                 "检查点SHA256": result["checkpoint_sha256"],
                 "逐位恢复报告": "results/P2/segmentation_expanded_v3_resume.json",
                 "组件测试日志": "logs/P2/component_tests_v3.log",
                 "验收算术测试日志": "logs/P2/acceptance_tests_v3.log",
                 "对照图清单": "results/P2/segmentation_expanded_v3_visuals/manifest.json",
                 "恢复说明": "docs/第三轮扩大样本与恢复说明.txt"}}
    atomic_json(ROOT / "results/第三轮扩大样本验收.json", summary)
    print(summary["效果验收"], flush=True)


if __name__ == "__main__":
    main()
