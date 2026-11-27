"""Restartable fixed-checkpoint head interventions on training/development frames."""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _key, _relative in {"TORCH_HOME": "runtime/cache/torch", "CUDA_CACHE_PATH": "runtime/cache/cuda",
                        "TEMP": "runtime/temp", "TMP": "runtime/temp"}.items():
    os.environ[_key] = str(ROOT / _relative)

import numpy as np
import scipy
import PIL
from PIL import Image, ImageDraw
import torch

from src.checkpoints import atomic_json, atomic_write, file_sha256, fingerprint, _exclusive_lock
from src.graspnet_frames import GraspNetFrames
from src.segmentation import decode_instances, segmentation_loss
from src.segmentation_diagnostics_v3 import (HEADS, COMBINATIONS, combination_name, selected_centers,
    center_matches, decoded_metrics, verify_model_metrics, cache_key, load_cached_frame, commit_frame,
    grouped_summary, distribution)
from scripts.render_segmentation_expanded_v3 import prepare_final_model, project_path, instance_colors, FIXED_DECODER


def validate_diagnostic_protocol(protocol):
    required = {"expected_optimizer_steps": 1600, "frame_count": 240, "training_frames": 160,
        "development_validation_frames": 80, "partitions": ["train", "validation"],
        "center_match_radii_pixels": [5, 10], "center_target_window_radius_pixels": 4,
        "visualization_frame": "0000", "decoder": FIXED_DECODER, "new_training_updates": 0,
        "threshold_sweep_or_tuning": False, "intervention_heads": list(HEADS),
        "source_and_data_integrity_required": True, "model_only_must_match_original_frame_results": True,
        "per_frame_atomic_commits": True, "calibration_and_official_test_use": False}
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError("Frozen diagnostic protocol mismatch: " + key)
    if set(protocol["visualization_scenes"]) != {3, 12, 30, 31, 34, 38, 42, 43, 57, 93, 13, 18, 22, 33, 39}:
        raise ValueError("Exactly the predeclared fifteen visualization scenes are required")


def heat_image(values):
    value = np.clip(np.asarray(values), 0, 1)
    return Image.fromarray(np.stack([255 * value, 255 * np.sqrt(value),
                                    255 * (1 - value) * 0.2], axis=-1).astype(np.uint8))


def draw_frame(path, rgb_path, labels, outputs, targets, predictions, identity):
    height, width = labels.shape
    with Image.open(rgb_path) as image:
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    panes = [("RGB", rgb), ("GT reference", instance_colors(labels)),
             ("Model only (no GT)", instance_colors(predictions["model_only"])),
             ("Predicted center probability [0,1]", heat_image(outputs["centers"][0, 0].sigmoid().cpu().numpy())),
             ("GT center heatmap [0,1]", heat_image(targets["centers"][0].numpy())),
             ("Model foreground probability [0,1]", heat_image(outputs["foreground"][0, 0].sigmoid().cpu().numpy())),
             ("Oracle: GT centers only", instance_colors(predictions["oracle_centers"])),
             ("Oracle: GT centers + offsets", instance_colors(predictions["oracle_centers_offsets"])),
             ("All GT: decoder round-trip reference", instance_colors(predictions["oracle_foreground_centers_offsets"]))]
    canvas = Image.new("RGB", (width * 3, (height + 28) * 3 + 74), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "{} | scene {:04d} frame {} | frozen step 1600".format(identity["split"].upper(), identity["scene"], identity["frame"]), fill="black")
    draw.text((8, 26), "ORACLE = GT SUBSTITUTION FOR DIAGNOSIS ONLY, NOT MODEL PERFORMANCE OR A PERFORMANCE UPPER BOUND", fill="black")
    draw.text((8, 44), "Fixed decoding: fg .50 / center .35 / min area 20; independent ID colors; examples fixed before diagnosis", fill="black")
    for index, (caption, pane) in enumerate(panes):
        x, y = index % 3 * width, index // 3 * (height + 28) + 74
        draw.text((x + 5, y + 7), caption, fill="black")
        canvas.paste(pane, (x, y + 28))
    data = io.BytesIO()
    canvas.save(data, format="PNG")
    atomic_write(path, lambda stream: stream.write(data.getvalue()))


def analyze_frame(model, dataset, index, record, source_split, reference, device, decoder):
    inputs, targets = dataset.get(index)
    with torch.no_grad():
        # Keep logits on the evaluation device: moving before sigmoid may alter
        # threshold-boundary pixels relative to the frozen CUDA evaluation.
        outputs = {key: value.detach().float() for key, value in model(inputs[None].to(device)).items()}
        total_loss, components = segmentation_loss(outputs, {key: value[None].to(device) for key, value in targets.items() if key != "labels"})
    labels = targets["labels"].numpy()
    height, width = labels.shape
    oracle = {"foreground": torch.logit(targets["foreground"][None].clamp(1e-6, 1 - 1e-6)),
              "centers": torch.logit(targets["centers"][None].clamp(1e-6, 1 - 1e-6)),
              "offsets": targets["offsets"][None]}
    oracle = {key: value.to(device) for key, value in oracle.items()}
    combinations, predictions = {}, {}
    for heads in COMBINATIONS:
        name = combination_name(heads)
        intervened = {head: oracle[head] if head in heads else outputs[head] for head in HEADS}
        decoded = decode_instances(intervened, **decoder)
        predictions[name] = decoded
        combinations[name] = decoded_metrics(intervened, decoded, labels, decoder["foreground_threshold"])
        combinations[name]["oracle_heads"] = list(heads)
    verify_model_metrics(combinations["model_only"], reference)
    center_probability = outputs["centers"][0, 0].sigmoid().cpu()
    detected = selected_centers(center_probability, decoder["center_threshold"], decoder["max_centers"])
    oracle_detected = selected_centers(oracle["centers"][0, 0].sigmoid(), decoder["center_threshold"], decoder["max_centers"])
    offset_difference = outputs["offsets"][0].cpu().numpy() - targets["offsets"].numpy()
    offset_difference[0] *= width
    offset_difference[1] *= height
    epe = np.sqrt((offset_difference ** 2).sum(axis=0))
    per_instance, coordinates = [], []
    for label in np.unique(labels[labels > 0]):
        yy, xx = np.nonzero(labels == label)
        cy, cx = int(np.round(yy.astype(np.float32).mean())), int(np.round(xx.astype(np.float32).mean()))
        coordinates.append([cy, cx])
        window = center_probability[max(0, cy - 4):min(height, cy + 5), max(0, cx - 4):min(width, cx + 5)]
        distances = np.linalg.norm(detected - [cy, cx], axis=1)
        per_instance.append({"label": int(label), "resized_pixels": len(yy), "gt_center_yx": [cy, cx],
            "centroid_outside_own_mask": bool(labels[cy, cx] != label),
            "target_center_probability": float(center_probability[cy, cx]),
            "target_9x9_max_probability": float(window.max()),
            "nearest_selected_peak_distance_pixels": float(distances.min()) if len(distances) else None,
            "offset_epe_mean_pixels": float(epe[yy, xx].mean()),
            "offset_epe_median_pixels": float(np.median(epe[yy, xx])),
            "offset_epe_p90_pixels": float(np.quantile(epe[yy, xx], .9))})
    gt_centers = np.asarray(coordinates).reshape(-1, 2)
    pairs = []
    for first in range(len(gt_centers)):
        for second in range(first + 1, len(gt_centers)):
            distance = float(np.linalg.norm(gt_centers[first] - gt_centers[second]))
            if distance < 5:
                pairs.append({"labels": [per_instance[first]["label"], per_instance[second]["label"]], "distance_pixels": distance})
    with Image.open(dataset.root / record["files"]["label"]) as image:
        raw_labels = set(np.unique(np.asarray(image)).tolist()) - {0}
    resized_labels = set(np.unique(labels).tolist()) - {0}
    all_gt = combinations["oracle_foreground_centers_offsets"]
    exact = (all_gt["instances"]["fp"] == 0 and all_gt["instances"]["fn"] == 0 and
             all(value == 1.0 for value in all_gt["best_iou_per_gt_instance"]))
    identity = {"split": "train" if source_split == "train" else "development_validation",
                "scene": record["scene"], "frame": record["frame"]}
    row = {"schema": 1, "identity": identity, "source_split": source_split,
           "historical_validation_group": None if source_split == "train" else "previously_seen_scene13" if record["scene"] == 13 else "formerly_fresh_four_now_development",
           "model_only_matches_frozen_final_result": True, "combinations": combinations,
           "model_loss_no_augmentation": {"total": float(total_loss), "components": {key: float(value) for key, value in components.items()}},
           "selected_center_count": len(detected), "gt_center_count": len(gt_centers),
           "unique_gt_center_count": len(np.unique(gt_centers, axis=0)),
           "selected_center_coordinates_yx": detected.tolist(), "gt_center_coordinates_yx": gt_centers.tolist(),
           "center_probability_max": float(center_probability.max()),
           "target_9x9_max_probability_distribution": distribution([item["target_9x9_max_probability"] for item in per_instance]),
           "center_matching": {str(radius): center_matches(detected, gt_centers, radius) for radius in (5, 10)},
           "offset_instance_mean_epe_distribution": distribution([item["offset_epe_mean_pixels"] for item in per_instance]),
           "per_instance": per_instance,
           "all_oracle": {"interpretation": "Ideal label-field decoder round-trip reference, not a strict model performance upper bound",
               "exact_instance_partition": exact, "selected_center_count": len(oracle_detected),
               "small_gt_instances": [{"label": item["label"], "pixels": item["resized_pixels"]} for item in per_instance if item["resized_pixels"] < decoder["min_pixels"]],
               "coincident_gt_center_pairs": [pair for pair in pairs if pair["distance_pixels"] == 0],
               "close_gt_center_pairs_lt5px": pairs,
               "centroids_outside_own_mask_fraction": sum(item["centroid_outside_own_mask"] for item in per_instance) / len(per_instance) if per_instance else 0,
               "labels_lost_during_resize": sorted(raw_labels - resized_labels),
               "nonexact_is_not_automatically_an_error": True}}
    return row, outputs, targets, predictions


def main(config_path, max_frames=None):
    diagnostic = json.loads(config_path.read_text(encoding="utf-8"))
    validate_diagnostic_protocol(diagnostic)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be positive")
    directory = project_path(diagnostic["run_directory"])
    cache = project_path(diagnostic["output_directory"])
    report_path = project_path(diagnostic["report"])
    model, training_protocol, _, data_root, frozen_result, evidence = prepare_final_model(directory)
    manifest = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
    ordered = [(split, record) for split in ("train", "validation") for record in manifest["records"][split]]
    if len(ordered) != 240 or len(manifest["records"]["train"]) != 160 or len(manifest["records"]["validation"]) != 80:
        raise ValueError("Expected the exact frozen 160+80 records")
    sources = ["scripts/diagnose_segmentation_expanded_v3.py", "src/segmentation_diagnostics_v3.py",
               "scripts/render_segmentation_expanded_v3.py"]
    binding = {"schema": 1, "diagnostic_protocol_sha256": file_sha256(config_path), "diagnostic_protocol": diagnostic,
               "frozen_run_evidence": evidence, "diagnostic_sources_sha256": {name: file_sha256(ROOT / name) for name in sources},
               "device": diagnostic["device"], "torch": str(torch.__version__), "numpy": np.__version__,
               "scipy": scipy.__version__, "pillow": PIL.__version__}
    binding_hash = fingerprint(binding)
    cache.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(cache):
        binding_path = cache / "binding.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                raise ValueError("Existing diagnostic cache binding changed; use a new version")
        else:
            atomic_json(binding_path, binding)
        reference = {(split, row["scene"], row["frame"]): row for split in ("train", "validation")
                     for row in frozen_result["evaluation"][split]["frames"]}
        datasets = {split: GraspNetFrames(data_root, manifest["records"][split], size=tuple(training_protocol["input_size"]),
                    mode=training_protocol["mode"], seed=training_protocol["seed"], augment=False) for split in ("train", "validation")}
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(diagnostic["device"])
        moved_to_device = False
        rows, new_count, skipped, incomplete = [], 0, 0, []
        for split in ("train", "validation"):
            for index, record in enumerate(manifest["records"][split]):
                identity = {"split": "train" if split == "train" else "development_validation", "scene": record["scene"], "frame": record["frame"]}
                key = cache_key(**identity)
                expected = reference[(split, record["scene"], record["frame"])]
                needs_image = record["frame"] == diagnostic["visualization_frame"] and record["scene"] in diagnostic["visualization_scenes"]
                cached = load_cached_frame(cache, key, binding_hash, identity, expected, needs_image)
                if cached is not None:
                    rows.append(cached)
                    skipped += 1
                    continue
                if max_frames is not None and new_count >= max_frames:
                    incomplete.append(identity)
                    continue
                if shutil.disk_usage(ROOT).free < diagnostic["free_disk_reserve_gib"] * 2 ** 30 + 64 * 1024 * 1024:
                    raise RuntimeError("Cannot preserve the 60 GiB free disk reserve")
                if not moved_to_device:
                    model.to(device)
                    moved_to_device = True
                row, outputs, targets, predictions = analyze_frame(model, datasets[split], index, record, split, expected, device, diagnostic["decoder"])
                row["binding_fingerprint"] = binding_hash
                if needs_image:
                    image_path = cache / "images" / (key + ".png")
                    draw_frame(image_path, data_root / record["files"]["rgb"], targets["labels"].numpy(), outputs, targets, predictions, identity)
                    row["image"] = {"file": str(image_path.relative_to(cache)), "sha256": file_sha256(image_path)}
                commit_frame(cache, key, row)
                rows.append(row)
                new_count += 1
                print(json.dumps({"event": "frame_committed", "identity": identity, "processed_new": new_count,
                    "skipped": skipped, "model_tp": row["combinations"]["model_only"]["instances"]["tp"],
                    "all_oracle_exact": row["all_oracle"]["exact_instance_partition"]}), flush=True)
        completed = len(rows) == diagnostic["frame_count"]
        frame_references = [{"identity": row["identity"], "file": "frames/" + cache_key(**row["identity"]) + ".json",
                             "sha256": file_sha256(cache / "frames" / (cache_key(**row["identity"]) + ".json"))} for row in rows]
        report = {"schema": 1, "status": "completed" if completed else "partial", "frame_count": 240,
            "frames_completed": len(rows), "binding_fingerprint": binding_hash, "binding": binding,
            "new_training_updates": 0, "thresholds_changed": False, "checkpoint_changed": False,
            "model_only_all_completed_frames_match_frozen_final_result": all(row["model_only_matches_frozen_final_result"] for row in rows),
            "groups": grouped_summary(rows), "frames": frame_references, "incomplete_frames": incomplete,
            "visualizations": [{"identity": row["identity"], **row["image"]} for row in rows if row.get("image")],
            "metric_definitions": {"instance_matching": "Original maximum-cardinality one-to-one matching at mask IoU >= .50; F1 is not mask AP",
                "center_matching": "One-to-one maximum-cardinality matching of fixed decoder-selected peaks at 5/10 pixels; no threshold sweep",
                "center_window": "9x9 maximum around each rounded GT centroid, clipped to image; probability statistic is not detection recall",
                "gt_center_count": "Counts object instances, including coincident centroids; unique_gt_center_count is separately reported and never replaces the matching denominator",
                "offset_epe": "Euclidean endpoints after x*width,y*height, only on GT foreground; summarize per-instance means equally",
                "mean_best_gt_instance_iou": "Each GT instance selects its maximum IoU against any prediction; multiple GT instances may select one prediction. Diagnostic non-one-to-one statistic, not F1 or AP",
                "scene_macro": "Equal average of per-scene micro foreground IoU and instance F1; not an average of per-frame F1",
                "oracle_background_offsets": "GT offset arrays are supervised only on GT foreground; background zero is a code convention, not a true correspondence"},
            "limitations": ["GT head substitutions are conditional interventions on fixed outputs, not deployable model scores or attainable gains after retraining",
                "All-GT is an ideal-label-field decoder round-trip reference, not a mathematical performance upper bound",
                "Small-area filtering, centroid collisions/NMS, and labels lost on resizing can make all-GT imperfect; nonexact is not automatically a bug",
                "Predicted expanded masks can match objects removed by oracle min-area filtering; model may exceed all-GT on a metric without inconsistency",
                "GT offset zero on background is a convention; predictions marked foreground outside GT have no supervised true offset",
                "All 80 validation images have already been evaluated and are development validation; historical fresh-four name does not imply blindness now",
                "No official test/calibration data, training updates, threshold tuning, grasp AP, or robot execution"]}
        atomic_json(report_path, report)
        invocation = {"status": report["status"], "processed_new": new_count, "skipped": skipped,
                      "frames_completed": len(rows), "binding_fingerprint": binding_hash}
        atomic_json(cache / "last_invocation.json", invocation)
        print(json.dumps(dict(invocation, event="invocation_complete")), flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_expanded_v3_diagnosis.json")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum newly computed frames this invocation; validated cached frames are skipped")
    args = parser.parse_args()
    main(project_path(args.config), args.max_frames)
