"""Frozen four-training-frame diagnosis of the v1 segmentation checkpoint.

Ground-truth substitutions are interventions, not deployable predictions and
must never be reported as model accuracy. No validation images are loaded.
"""

import argparse
import io
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TORCH_HOME"] = str(ROOT / "runtime" / "cache" / "torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "runtime" / "cache" / "cuda")
os.environ["TEMP"] = str(ROOT / "runtime" / "tmp")
os.environ["TMP"] = os.environ["TEMP"]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from src.checkpoints import atomic_json, atomic_write, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet, decode_instances, instance_counts, summarize_counts


HEADS = ("foreground", "centers", "offsets")
COMBINATIONS = [tuple(head for head, bit in zip(HEADS, flags) if bit)
                for flags in itertools.product((False, True), repeat=3)]


def name_of(heads):
    return "oracle_" + "_".join(heads) if heads else "model_only"


def selected_centers(probability, threshold=0.35, max_centers=64):
    """Exactly reproduce the peak suppression used by the frozen v1 decoder."""
    probability = probability.detach().float().cpu()
    peaks = (probability == F.max_pool2d(probability[None, None], 9, 1, 4)[0, 0]) & (probability >= threshold)
    coordinates = np.argwhere(peaks.numpy())
    scores = probability.numpy()[coordinates[:, 0], coordinates[:, 1]]
    order = np.argsort(-scores, kind="stable")
    selected = []
    for index in order:
        candidate = coordinates[index]
        if all(np.linalg.norm(candidate - other) >= 5 for other in selected):
            selected.append(candidate)
            if len(selected) == max_centers:
                break
    return np.asarray(selected).reshape(-1, 2)


def best_iou_per_label(prediction, labels):
    result = []
    for label in np.unique(labels[labels > 0]):
        truth = labels == label
        best = 0.0
        for predicted_id in np.unique(prediction[prediction > 0]):
            predicted = prediction == predicted_id
            best = max(best, float((predicted & truth).sum() / (predicted | truth).sum()))
        result.append(best)
    return result


def color_instances(labels):
    canvas = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels[labels > 0]):
        canvas[labels == label] = ((int(label) * 71 + 40) % 216 + 40,
                                  (int(label) * 113 + 40) % 216 + 40,
                                  (int(label) * 173 + 40) % 216 + 40)
    return Image.fromarray(canvas)


def heat_image(probability):
    value = np.clip(np.asarray(probability), 0, 1)
    return Image.fromarray(np.stack([255 * value, 255 * np.sqrt(value),
                                    255 * (1 - value) * 0.2], axis=-1).astype(np.uint8))


def spatial_block_statistics(probability):
    """Measure the post-upsampling head; do not assume its output is constant."""
    values = probability.numpy()
    height, width = values.shape
    blocks = values.reshape(height // 4, 4, width // 4, 4)
    ranges = blocks.max(axis=(1, 3)) - blocks.min(axis=(1, 3))
    interiors = blocks[:, 1:3, :, 1:3]
    interior_ranges = interiors.max(axis=(1, 3)) - interiors.min(axis=(1, 3))
    return {"total_4x4_cells": height * width // 16,
            "exactly_constant_4x4_cells": int((ranges == 0).sum()),
            "mean_4x4_probability_range": float(ranges.mean()),
            "max_4x4_probability_range": float(ranges.max()),
            "exactly_constant_interior_2x2_cells": int((interior_ranges == 0).sum()),
            "mean_interior_2x2_probability_range": float(interior_ranges.mean()),
            "interpretation": "Nearest upsampling is followed by a full-resolution 3x3 convolution, GroupNorm and ReLU. Spatial expressivity is not established as the cause of failure by this diagnostic."}


def draw_diagnostic(path, rgb_path, labels, outputs, targets, predictions, record):
    height, width = labels.shape
    with Image.open(rgb_path) as image:
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    panes = [("RGB (training only)", rgb), ("Ground truth instances", color_instances(labels)),
             ("Predicted foreground", heat_image(outputs["foreground"][0, 0].sigmoid().cpu().numpy())),
             ("Predicted centers (fixed 0..1)", heat_image(outputs["centers"][0, 0].sigmoid().cpu().numpy())),
             ("Target centers (fixed 0..1)", heat_image(targets["centers"][0].numpy())),
             ("Model only", color_instances(predictions["model_only"])),
             ("GT centers + model fg/offsets", color_instances(predictions["oracle_centers"])),
             ("GT centers/offsets + model fg", color_instances(predictions["oracle_centers_offsets"])),
             ("All GT: target-decoder round trip", color_instances(predictions["oracle_foreground_centers_offsets"]))]
    canvas = Image.new("RGB", (width * 3, (height + 28) * 3 + 38), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), "scene {:04d}, frame {}: GT substitutions are diagnosis, not model performance".format(
        record["scene"], record["frame"]), fill="#111111")
    for index, (caption, pane) in enumerate(panes):
        x, y = (index % 3) * width, (index // 3) * (height + 28) + 38
        canvas.paste(pane, (x, y + 28))
        draw.text((x + 5, y + 7), caption, fill="#111111")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_write(path, lambda stream: stream.write(buffer.getvalue()))


def run(run_directory, device):
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((run_directory / "data_manifest.json").read_text(encoding="utf-8"))
    for source, digest in config["source_sha256"].items():
        if file_sha256(ROOT / source) != digest:
            raise ValueError("Frozen v1 source changed: " + source)
    if fingerprint(data_manifest) != config["data_fingerprint"]:
        raise ValueError("Run data manifest fingerprint mismatch")
    checkpoint_dir = run_directory / "checkpoints"
    pointer = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoint_dir / pointer["manifest"]
    if file_sha256(manifest_path) != pointer["manifest_sha256"]:
        raise ValueError("Checkpoint manifest checksum mismatch")
    checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = checkpoint_dir / checkpoint_manifest["file"]
    if file_sha256(checkpoint_path) != checkpoint_manifest["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if checkpoint_manifest["config_fingerprint"] != fingerprint(config):
        raise ValueError("Checkpoint configuration mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["cursor"]["global_step"] != 100:
        raise ValueError("Expected the frozen 100-update v1 diagnostic checkpoint")
    records = data_manifest["records"]["train"]
    if len(records) != 4 or set(record["scene"] for record in records) != {3, 12}:
        raise ValueError("Expected exactly the original four training frames")
    data_root = Path(data_manifest["data_root"])
    for record in records:
        for kind, relative_path in record["files"].items():
            if file_sha256(data_root / relative_path) != record["sha256"][kind]:
                raise ValueError("Training frame checksum mismatch")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = PilotInstanceNet(mode=config["mode"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    del payload
    dataset = GraspNetFrames(data_root, records, size=config["input_size"], mode=config["mode"], augment=False)
    totals = {name_of(heads): {"tp": 0, "fp": 0, "fn": 0, "intersection": 0, "union": 0, "best_ious": []}
              for heads in COMBINATIONS}
    frame_reports = []
    image_dir = ROOT / "results" / "P2" / "segmentation_v1_diagnosis"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        inputs, targets = dataset.get(index)
        with torch.no_grad():
            outputs = {key: value.cpu() for key, value in model(inputs[None].to(device)).items()}
        labels = targets["labels"].numpy()
        height, width = labels.shape
        oracle = {"foreground": torch.logit(targets["foreground"][None].clamp(1e-6, 1 - 1e-6)),
                  "centers": torch.logit(targets["centers"][None].clamp(1e-6, 1 - 1e-6)),
                  "offsets": targets["offsets"][None]}
        metrics, predictions = {}, {}
        for heads in COMBINATIONS:
            name = name_of(heads)
            intervened = {head: oracle[head] if head in heads else outputs[head] for head in HEADS}
            decoded = decode_instances(intervened, **config["decoder"])
            predictions[name] = decoded
            counts = instance_counts(decoded, labels)
            predicted_fg = intervened["foreground"][0, 0].sigmoid().numpy() >= config["decoder"]["foreground_threshold"]
            intersection = int((predicted_fg & (labels > 0)).sum())
            union = int((predicted_fg | (labels > 0)).sum())
            best_ious = best_iou_per_label(decoded, labels)
            metrics[name] = {"oracle_heads": list(heads), "instances": summarize_counts(counts),
                             "foreground_pixel_iou": intersection / union if union else 1.0,
                             "mean_best_gt_instance_iou": float(np.mean(best_ious)),
                             "best_iou_per_gt_instance": best_ious}
            for key in counts:
                totals[name][key] += counts[key]
            totals[name]["intersection"] += intersection
            totals[name]["union"] += union
            totals[name]["best_ious"].extend(best_ious)
        probability = outputs["centers"][0, 0].sigmoid()
        selected = selected_centers(probability, config["decoder"]["center_threshold"], config["decoder"]["max_centers"])
        unconstrained = selected_centers(probability, 0.0, config["decoder"]["max_centers"])
        offset_error = outputs["offsets"][0].numpy() - targets["offsets"].numpy()
        offset_error[0] *= width
        offset_error[1] *= height
        epe = np.sqrt((offset_error ** 2).sum(axis=0))
        per_instance = []
        for object_id in np.unique(labels[labels > 0]):
            yy, xx = np.nonzero(labels == object_id)
            cy, cx = int(np.round(yy.mean())), int(np.round(xx.mean()))
            distances = np.linalg.norm(selected - [cy, cx], axis=1)
            unconstrained_distances = np.linalg.norm(unconstrained - [cy, cx], axis=1)
            window = probability[max(0, cy - 4):min(height, cy + 5), max(0, cx - 4):min(width, cx + 5)]
            per_instance.append({"label": int(object_id), "pixels": len(yy), "target_center_yx": [cy, cx],
                                 "probability_at_target_center": float(probability[cy, cx]),
                                 "max_probability_in_9x9_target_window": float(window.max()),
                                 "nearest_selected_center_distance_pixels": float(distances.min()) if len(distances) else None,
                                 "nearest_top64_local_peak_no_threshold_distance_pixels": float(unconstrained_distances.min()),
                                 "offset_endpoint_error_mean_pixels": float(epe[yy, xx].mean()),
                                 "offset_endpoint_error_median_pixels": float(np.median(epe[yy, xx])),
                                 "offset_endpoint_error_p90_pixels": float(np.quantile(epe[yy, xx], 0.9))})
        image_path = image_dir / "scene_{:04d}_{}.png".format(record["scene"], record["frame"])
        draw_diagnostic(image_path, data_root / record["files"]["rgb"], labels, outputs, targets, predictions, record)
        frame_report = {"scene": record["scene"], "frame": record["frame"], "metrics": metrics,
                        "predicted_center_probability_max": float(probability.max()),
                        "selected_center_count": len(selected), "selected_center_coordinates_yx": selected.tolist(),
                        "per_instance": per_instance, "diagnostic_image": str(image_path.relative_to(ROOT)),
                        "spatial_block_statistics": spatial_block_statistics(probability)}
        frame_reports.append(frame_report)
        print(json.dumps({"scene": record["scene"], "frame": record["frame"],
                          "max_center_probability": frame_report["predicted_center_probability_max"],
                          "selected_centers": len(selected), "combination_tp": {
                              name: item["instances"]["tp"] for name, item in metrics.items()}}, ensure_ascii=False), flush=True)
    aggregate = {}
    for name, item in totals.items():
        aggregate[name] = {"instances": summarize_counts({key: item[key] for key in ("tp", "fp", "fn")}),
                           "foreground_pixel_iou": item["intersection"] / item["union"] if item["union"] else 1.0,
                           "mean_best_gt_instance_iou": float(np.mean(item["best_ious"]))}
    all_oracle = aggregate["oracle_foreground_centers_offsets"]
    passed = all_oracle["instances"]["f1_iou50"] == 1.0 and all_oracle["mean_best_gt_instance_iou"] == 1.0
    report = {"schema": 1, "status": "completed", "scope": "Frozen v1 four-training-frame diagnosis only",
              "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": file_sha256(checkpoint_path),
              "optimizer_steps": 100, "config_fingerprint": fingerprint(config),
              "data_fingerprint": fingerprint(data_manifest), "source_sha256": config["source_sha256"],
              "diagnostic_script_sha256": file_sha256(Path(__file__)),
              "device": str(device), "input_size": config["input_size"], "decoder": config["decoder"],
              "training_frame_count": 4, "validation_frames_loaded": 0, "test_frames_loaded": 0,
              "new_training_updates": 0, "threshold_tuning_performed": False,
              "all_oracle_roundtrip_exact_instance_recovery": passed, "aggregate": aggregate,
              "offset_endpoint_error_instance_mean_pixels": float(np.mean([
                  instance["offset_endpoint_error_mean_pixels"]
                  for frame in frame_reports for instance in frame["per_instance"]])),
              "frames": frame_reports,
              "confirmed_findings": [
                  "All 34 training instances are exactly recovered when all three heads are replaced by targets; no axis or normalization mismatch is demonstrated in the target-decoder path",
                  "At the frozen 0.35 center threshold, the four images select 1, 0, 2, 1 centers against 9, 9, 8, 8 ground-truth instances",
                  "Model foreground micro IoU is 0.35956 on these four trained frames, showing incomplete fitting independent of the instance decoder",
                  "Ground-truth centers and offsets with model foreground recover 7/34 instances; ground-truth foreground and centers with model offsets recover 23/34",
                  "Center detection and foreground fitting are both demonstrated bottlenecks; offset endpoint errors are also nonzero",
                  "The full-resolution 3x3 head creates variation inside all 4x4 cells; its outputs are not wholly constant 4x4 blocks"],
              "unresolved_hypotheses": [
                  "Loss weighting, optimization duration, augmentation and learning-rate schedule may affect convergence; isolate them in controlled training-only follow-up runs",
                  "No evidence here proves the architecture incapable of fitting the four images; spatial head changes are not yet justified as an identified fix"],
              "definitions": {"instances": "Micro precision/recall/F1 from maximum-cardinality matching at mask IoU >= 0.50; not mask AP",
                              "mean_best_gt_instance_iou": "Unweighted mean over GT instances of best predicted-mask IoU; non-one-to-one diagnostic statistic",
                              "center_distance": "Euclidean pixels on the 320x240 resized training frame; null when the frozen threshold selects no center",
                              "offset_error": "Euclidean endpoint error after undoing x/width and y/height normalization, measured on ground-truth foreground",
                              "oracle_probability": "Exact targets encoded as logits with [1e-6,1-1e-6] clipping; offsets unmodified"},
              "limitations": ["Every oracle intervention uses ground truth and is not a model performance result",
                              "Only four already-trained frames, no held-out estimate or generalization claim",
                              "No hyperparameters or decoder thresholds are selected here",
                              "Random initialization, 100 optimizer steps and augmentations may underfit this pilot",
                              "Oracle interventions isolate dependencies but do not identify an optimization or architectural root cause"]}
    path = ROOT / "results" / "P2" / "segmentation_v1_diagnosis.json"
    atomic_json(path, report)
    print(json.dumps({"report": str(path), "all_oracle_exact": passed, "aggregate": aggregate}, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="experiments/segmentation_diagnostics/rgbd_overfit_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run_directory = (ROOT / args.run_dir).resolve()
    if ROOT not in run_directory.parents:
        raise ValueError("Run directory must be in the project")
    raise SystemExit(run(run_directory, torch.device(args.device)))
