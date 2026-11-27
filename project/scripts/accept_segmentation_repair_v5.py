"""Independently verify both arms, apply frozen decision rules, render fixed examples."""
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_segmentation_repair_v5 import load_final_model
from scripts.train_segmentation_expanded_v3 import learning_rate_for_update
from scripts.accept_segmentation_expanded_v3 import verify_partition
from scripts.render_segmentation_expanded_v3 import prepare_final_model, instance_colors
from scripts.verify_segmentation_repair_v5_resume import check_initialization, verify_parent
from scripts.verify_gpu_resume import compare_tree
from src.checkpoints import atomic_json, atomic_write, file_sha256
from src.graspnet_frames import GraspNetFrames
from src.segmentation import decode_instances, instance_counts, summarize_counts
from src.segmentation_diagnostics_v3 import selected_centers, center_matches, distribution
import numpy as np
import torch
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def engineering_gate(partition, protocol):
    threshold = protocol["acceptance"]
    return (partition["scene_macro"]["foreground_pixel_iou"] >= threshold["primary_scene_macro_foreground_iou_min"]
            and partition["scene_macro"]["instance_f1_iou50"] >= threshold["primary_scene_macro_instance_f1_min"]
            and {row["scene"] for row in partition["scenes"]} == set(protocol["primary_development_scenes"])
            and all(row["instances"]["tp"] >= threshold["minimum_tp_per_primary_scene"] for row in partition["scenes"]))


def adoption(checked, protocol):
    control, candidate = (checked[name] for name in ("original", "center_mse"))
    gates = {name: engineering_gate(value["primary_development"], protocol) for name, value in checked.items()}
    first, second = control["development_validation"], candidate["development_validation"]
    f1_gain = second["scene_macro"]["instance_f1_iou50"] - first["scene_macro"]["instance_f1_iou50"]
    fg_drop = first["scene_macro"]["foreground_pixel_iou"] - second["scene_macro"]["foreground_pixel_iou"]
    control_scenes = {row["scene"]: row for row in first["scenes"]}
    deltas = {str(row["scene"]): row["instances"]["f1_iou50"] - control_scenes[row["scene"]]["instances"]["f1_iou50"]
              for row in second["scenes"]}
    rule = protocol["candidate_adoption"]
    checks = {"candidate_passes_absolute_gate": gates["center_mse"],
              "all_development_macro_f1_gain": f1_gain >= rule["all_development_scene_macro_f1_gain_over_control_min"] or math.isclose(f1_gain,rule["all_development_scene_macro_f1_gain_over_control_min"],rel_tol=0,abs_tol=1e-12),
              "all_development_macro_foreground_guard": fg_drop <= rule["all_development_scene_macro_foreground_iou_drop_max"] or math.isclose(fg_drop,rule["all_development_scene_macro_foreground_iou_drop_max"],rel_tol=0,abs_tol=1e-12),
              "scene_consistency": sum(delta > 0 for delta in deltas.values()) >= rule["minimum_development_scenes_with_positive_f1_gain"]}
    selected = "center_mse" if all(checks.values()) else "original" if gates["original"] else None
    return {"engineering_gate_by_arm": gates, "candidate_adoption_checks": checks,
            "all_development_macro_f1_gain_candidate_minus_control": f1_gain,
            "all_development_macro_foreground_drop_control_minus_candidate": fg_drop,
            "scene_f1_gain_candidate_minus_control": deltas, "selected_arm": selected,
            "decision": "provisional_engineering_candidate" if selected else "neither_arm_accepted",
            "interpretation": "Frozen budget/engineering rule, not significance or independent generalization evidence"}


def head_diagnostics(model, manifest, evaluation, protocol):
    """Reproduce every final metric and summarize fixed-head diagnostics, no tuning."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.to("cuda").eval()
    rows = []
    for split, name in (("train", "train"), ("validation", "development_validation")):
        records = manifest["records"][split]
        dataset = GraspNetFrames(ROOT / protocol["data_root"], records, tuple(protocol["input_size"]),
                                 protocol["mode"], protocol["seed"], augment=False)
        references = {(row["scene"], row["frame"]): row for row in evaluation[name]["frames"]}
        for index, record in enumerate(records):
            inputs, targets = dataset.get(index)
            labels = targets["labels"].numpy()
            with torch.no_grad():
                outputs = model(inputs[None].to("cuda"))
                prediction = decode_instances(outputs, **protocol["decoder"])
                foreground = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= .5
                probability = outputs["centers"][0, 0].float().sigmoid().cpu()
            ref = references[(record["scene"], record["frame"])]
            counts = summarize_counts(instance_counts(prediction, labels))
            if counts != ref["instances"] or int((foreground & (labels > 0)).sum()) != ref["foreground_intersection"] or int((foreground | (labels > 0)).sum()) != ref["foreground_union"]:
                raise AssertionError("Final checkpoint failed independent per-frame re-evaluation")
            peaks = selected_centers(probability, .35, 64)
            height, width = labels.shape
            difference = outputs["offsets"][0].detach().cpu().numpy() - targets["offsets"].numpy()
            difference[0] *= width
            difference[1] *= height
            epe = np.sqrt((difference ** 2).sum(axis=0))
            centers, errors = [], []
            for label in np.unique(labels[labels > 0]):
                yy, xx = np.nonzero(labels == label)
                centers.append([int(np.round(yy.astype(np.float32).mean())), int(np.round(xx.astype(np.float32).mean()))])
                errors.append(float(epe[yy, xx].mean()))
            centers = np.asarray(centers).reshape(-1, 2)
            rows.append({"split":name,"scene":record["scene"],"frame":record["frame"],
                         "center_matching":{str(radius):center_matches(peaks,centers,radius) for radius in (5,10)},
                         "selected_center_count":len(peaks),"gt_instance_count":len(centers),
                         "offset_instance_mean_epe_pixels":errors,"model_metrics_reproduced":True})

    def summarize(members):
        matching = {}
        for radius in ("5","10"):
            totals = {key:sum(row["center_matching"][radius][key] for row in members) for key in ("tp","fp","fn")}
            tp,fp,fn = (totals[key] for key in ("tp","fp","fn"))
            matching[radius] = dict(totals,recall=tp/(tp+fn) if tp+fn else 0,
                                   precision=tp/(tp+fp) if tp+fp else 0,
                                   f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        return {"frames":len(members),"center_matching":matching,
                "selected_center_count":sum(row["selected_center_count"] for row in members),
                "gt_instance_count":sum(row["gt_instance_count"] for row in members),
                "offset_instance_mean_epe_pixels":distribution([value for row in members for value in row["offset_instance_mean_epe_pixels"]])}
    groups = {"train":[row for row in rows if row["split"]=="train"],
              "development_validation":[row for row in rows if row["split"]=="development_validation"],
              "primary_development":[row for row in rows if row["split"]=="development_validation" and row["scene"] in protocol["primary_development_scenes"]]}
    return {"groups":{name:summarize(members) for name,members in groups.items()},"frames":rows,
            "all_240_model_frame_metrics_reproduced":len(rows)==240,
            "metric_note":"Center matches use fixed decoder peaks, maximum-cardinality matching at5/10px; EPE equals mean of per-instance foreground mean distances in320x240 input pixels. Diagnostic only, not selection endpoint."}


def render(models, manifest, results, protocol):
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = [row for row in manifest["records"]["validation"] if row["scene"] in protocol["visualization_scenes"]
               and row["frame"] == protocol["visualization_frame"]]
    if len(records) != 5:
        raise AssertionError("Require five predeclared development examples")
    dataset = GraspNetFrames(ROOT / protocol["data_root"], records, tuple(protocol["input_size"]),
                             protocol["mode"], protocol["seed"], augment=False)
    directory = ROOT / "results/P2/segmentation_repair_v5_visuals"
    directory.mkdir(parents=True, exist_ok=True)
    references = {name: {(row["scene"], row["frame"]): row for row in result["frames"]}
                  for name, result in results.items()}
    for model in models.values():
        model.to(device).eval()
    images = []
    for index, record in enumerate(records):
        inputs, targets = dataset.get(index)
        labels = targets["labels"].numpy()
        with Image.open(ROOT / protocol["data_root"] / record["files"]["rgb"]) as source:
            rgb = source.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
        panels = [("RGB input", rgb), ("GT reference only", instance_colors(labels))]
        scores = {}
        for name, model in models.items():
            with torch.no_grad():
                outputs = model(inputs[None].to(device))
                predicted = decode_instances(outputs, **protocol["decoder"])
                foreground = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= .5
            counts = summarize_counts(instance_counts(predicted, labels))
            reference = references[name][(record["scene"], record["frame"])]
            intersection, union = int((foreground & (labels > 0)).sum()), int((foreground | (labels > 0)).sum())
            if counts != reference["instances"] or intersection != reference["foreground_intersection"] or union != reference["foreground_union"]:
                raise AssertionError("Visualization differs from final frozen frame metrics: " + name)
            scores[name] = counts
            panels.append((name + " | F1 {:.3f}".format(counts["f1_iou50"]), instance_colors(predicted)))
        canvas = Image.new("RGB", (1600, 340), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "DEVELOPMENT VALIDATION | scene {:04d} frame {} | predeclared example; independent label colors".format(record["scene"], record["frame"]), fill="black")
        draw.text((8, 26), "parent_v3:1600 steps; original/center_mse: same parent +1600 new steps | fixed fg .50 / center .35 / area20 / max64", fill="black")
        draw.text((8, 44), "GT is reference only; no label enters inference decoder. Example images do not replace all-scene results or independent evaluation.", fill="black")
        for column, (caption, panel) in enumerate(panels):
            draw.text((column * 320 + 8, 70), caption, fill="black")
            canvas.paste(panel, (column * 320, 94))
        path = directory / ("development_scene_{:04d}_{}.png".format(record["scene"], record["frame"]))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        atomic_write(path, lambda stream: stream.write(buffer.getvalue()))
        images.append({"scene": record["scene"], "frame": record["frame"], "file": str(path.relative_to(ROOT)),
                       "sha256": file_sha256(path), "counts": scores})
    atomic_json(directory / "manifest.json", {"frames": images, "fixed_before_training": True,
                "all_frame_metrics_match": True, "thresholds_changed": False})
    return images


def main():
    protocol_path = ROOT / "configs/segmentation_repair_v5.json"
    protocol = read(protocol_path)
    aggregate_path = ROOT / "results/P2/segmentation_repair_v5.json"
    aggregate = read(aggregate_path)
    if aggregate["status"] != "completed" or aggregate["protocol_sha256"] != file_sha256(protocol_path):
        raise AssertionError("Completed two-arm aggregate must bind this protocol")
    checked, models, histories, initializations, evidence = {}, {}, {}, {}, {}
    manifest = None
    for arm in ("original", "center_mse"):
        directory = ROOT / protocol["arms"][arm]["run_directory"]
        model, config, manifest, result, bound = load_final_model(directory, protocol)
        item = aggregate["arms"][arm]
        for key, name in (("result_sha256", "result.json"), ("config_sha256", "config.json"),
                          ("history_sha256", "history.json"), ("initialization_sha256", "initialization.json")):
            if item[key] != file_sha256(directory / name):
                raise AssertionError("Aggregate file hash mismatch")
        if item["result"] != result or config["arm"] != arm:
            raise AssertionError("Aggregate/result/arm mismatch")
        history = read(directory / "history.json")
        if [row["step"] for row in history] != list(range(1, 1601)):
            raise AssertionError("Exactly 1600 new updates required per arm")
        exposures = {}
        for row in history:
            if not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm_before_clip"]):
                raise AssertionError("Nonfinite optimization evidence")
            if not math.isclose(row["learning_rate_used"], learning_rate_for_update(row["step"], protocol), rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError("Learning rate differs from frozen schedule")
            for sample in row["samples"]:
                identity = (sample["scene"], sample["frame"])
                exposures[identity] = exposures.get(identity, 0) + 1
        expected = {(scene, "{:04d}".format(frame)) for scene in protocol["train_scenes"] for frame in protocol["frames"]}
        if set(exposures) != expected or set(exposures.values()) != {20}:
            raise AssertionError("Each training frame must have exactly20 exposures")
        partitions = (("train", protocol["train_scenes"]), ("development_validation", protocol["validation_scenes"]),
                      ("primary_development", protocol["primary_development_scenes"]),
                      ("previously_seen_validation", protocol["previously_seen_validation_scenes"]))
        checked[arm] = {name: verify_partition(result["evaluation"][name], scenes, protocol["frames"]) for name, scenes in partitions}
        full_rows = result["evaluation"]["development_validation"]["frames"]
        for name, scenes in partitions[2:]:
            if result["evaluation"][name]["frames"] != [row for row in full_rows if row["scene"] in scenes]:
                raise AssertionError("Subset differs from full development records")
        gate = engineering_gate(checked[arm]["primary_development"], protocol)
        if result["evaluation"]["acceptance_gate"]["passed"] != gate:
            raise AssertionError("Recorded absolute gate differs from recomputation")
        models[arm], histories[arm], evidence[arm] = model, history, bound
        initializations[arm] = read(directory / "initialization.json")
    if any(histories["original"][i]["samples"] != histories["center_mse"][i]["samples"] for i in range(1600)):
        raise AssertionError("The two arms used different sample orders")
    parent_initialization = verify_parent(protocol)
    for arm in ("original", "center_mse"):
        check_initialization(initializations[arm],parent_initialization,protocol,
                             aggregate["arms"][arm]["result"]["config_fingerprint"])
    starts = {arm:{key:value for key,value in record.items() if key!="run_config_fingerprint"}
              for arm,record in initializations.items()}
    if compare_tree(starts["original"],starts["center_mse"]):
        raise AssertionError("Actual research arms did not start with identical model/optimizer/scheduler/scaler/RNG state")
    decision = adoption(checked, protocol)
    heads = {arm:head_diagnostics(models[arm],manifest,aggregate["arms"][arm]["result"]["evaluation"],protocol)
             for arm in ("original","center_mse")}
    atomic_json(ROOT / "results/P2/segmentation_repair_v5_heads.json", {"arms":heads,"thresholds_changed":False,
                "primary_endpoint_unchanged":True,"checkpoints":{arm:evidence[arm] for arm in evidence}})
    parent, _, _, _, parent_result, parent_evidence = prepare_final_model(ROOT / protocol["parent_run"])
    models = {"parent_v3": parent, **models}
    result_partitions = {"parent_v3": parent_result["evaluation"]["validation"],
                         **{name: aggregate["arms"][name]["result"]["evaluation"]["development_validation"] for name in ("original", "center_mse")}}
    images = render(models, manifest, result_partitions, protocol)
    report = {"轮次":"第五轮共同权重起点的修正对照", "执行状态":"完成", "decision":decision,
              "新参数更新总数":3200, "原始模型更新数":1600, "每臂新增更新":1600,
              "同样本顺序":True, "同父模型及重置初始状态":True, "逐帧汇总独立复核":True, "解码阈值改变":False,
              "中心与偏移诊断":{arm:heads[arm]["groups"] for arm in heads}, "全部480帧独立重推理一致":True,
              "指标":checked, "原模型四场景宏指标":parent_result["evaluation"]["fresh_validation"]["scene_macro"],
              "父模型证据":parent_evidence, "运行证据":evidence, "初始化证据":initializations,
              "固定可视化":images, "配置SHA256":file_sha256(protocol_path), "总报告SHA256":file_sha256(aggregate_path),
              "解释边界":["全部80帧是开发验证，不是盲测或官方测试", "实例F1不是mask AP",
                         "两臂仅中心监督整体替换不同，不能分别归因MSE形式或200权重",
                         "共同骨干会使中心监督变化影响其他分支", "单种子工程门槛和采纳规则不代表显著性或论文录用",
                         "没有事后挑checkpoint、改阈值或追加步数", "未进行抓取AP或机械臂试验"]}
    atomic_json(ROOT / "results/第五轮修正对照验收.json", report)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
