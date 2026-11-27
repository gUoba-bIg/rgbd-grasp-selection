"""Read-only diagnostic statistics and atomic per-frame cache contracts."""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from src.checkpoints import atomic_json, file_sha256
from src.segmentation import instance_counts, summarize_counts

HEADS = ("foreground", "centers", "offsets")
COMBINATIONS = [tuple(head for head, selected in zip(HEADS, flags) if selected)
                for flags in itertools.product((False, True), repeat=3)]


def combination_name(heads):
    return "oracle_" + "_".join(heads) if heads else "model_only"


def selected_centers(probability, threshold=0.35, max_centers=64):
    """Return only peaks surviving the unchanged decoder threshold and NMS."""
    probability = probability.detach().float().cpu()
    peaks = ((probability == F.max_pool2d(probability[None, None], 9, 1, 4)[0, 0]) &
             (probability >= threshold))
    candidates = np.argwhere(peaks.numpy())
    scores = probability.numpy()[candidates[:, 0], candidates[:, 1]]
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        candidate = candidates[index]
        if all(np.linalg.norm(candidate - other) >= 5 for other in selected):
            selected.append(candidate)
            if len(selected) == max_centers:
                break
    return np.asarray(selected, dtype=np.int64).reshape(-1, 2)


def center_matches(predicted, actual, radius):
    """Maximum-cardinality one-to-one center matching at a fixed pixel radius."""
    predicted, actual = np.asarray(predicted).reshape(-1, 2), np.asarray(actual).reshape(-1, 2)
    distances = np.linalg.norm(predicted[:, None] - actual[None], axis=2)
    pairs = []
    if distances.size:
        valid = distances <= radius
        # One extra valid pair dominates the entire secondary distance reward.
        scale = max(float(distances.max()), 1.0)
        reward = valid.astype(float) * (min(distances.shape) + 1) - distances / scale
        rows, columns = linear_sum_assignment(-reward)
        pairs = [{"prediction_index": int(row), "gt_index": int(column), "distance_pixels": float(distances[row, column])}
                 for row, column in zip(rows, columns) if valid[row, column]]
    counts = {"tp": len(pairs), "fp": len(predicted) - len(pairs), "fn": len(actual) - len(pairs)}
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return dict(counts, radius_pixels=radius, pairs=pairs,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"n": 0, "mean": None, "minimum": None, "p10": None, "median": None, "p90": None, "maximum": None}
    return {"n": int(values.size), "mean": float(values.mean()), "minimum": float(values.min()),
            "p10": float(np.quantile(values, 0.1)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "maximum": float(values.max())}


def best_instance_ious(predicted, labels):
    """Best IoU for each GT ID, not one-to-one matching or mask AP."""
    pred_ids = np.unique(predicted[predicted > 0])
    actual_ids = np.unique(labels[labels > 0])
    values = []
    for label in actual_ids:
        actual = labels == label
        best = 0.0
        for predicted_id in pred_ids:
            mask = predicted == predicted_id
            best = max(best, float((actual & mask).sum() / (actual | mask).sum()))
        values.append(best)
    return values


def decoded_metrics(outputs, decoded, labels, foreground_threshold):
    foreground = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= foreground_threshold
    actual = labels > 0
    intersection, union = int((foreground & actual).sum()), int((foreground | actual).sum())
    decoded_i = int(((decoded > 0) & actual).sum())
    decoded_u = int(((decoded > 0) | actual).sum())
    best = best_instance_ious(decoded, labels)
    return {"instances": summarize_counts(instance_counts(decoded, labels)),
            "foreground_intersection": intersection, "foreground_union": union,
            "foreground_pixel_iou": intersection / union if union else 1.0,
            "decoded_foreground_intersection": decoded_i, "decoded_foreground_union": decoded_u,
            "decoded_foreground_pixel_iou": decoded_i / decoded_u if decoded_u else 1.0,
            "best_iou_per_gt_instance": best,
            "mean_best_gt_instance_iou": float(np.mean(best)) if best else 1.0}


def verify_model_metrics(observed, expected):
    for key in ("instances", "foreground_intersection", "foreground_union", "foreground_pixel_iou"):
        if observed[key] != expected[key]:
            raise RuntimeError("model_only metric mismatch against frozen final result: " + key)


def cache_key(split, scene, frame):
    if split not in ("train", "development_validation") or not isinstance(scene, int) or not (len(frame) == 4 and frame.isdigit()):
        raise ValueError("Invalid frame cache identity")
    return "{}_scene_{:04d}_{}".format(split, scene, frame)


def load_cached_frame(directory, key, binding, identity, reference, image_required=False):
    """Return verified cached row, or None for an incomplete/corrupt commit.

    A changed experiment binding is never silently reused or overwritten.
    Payload/image corruption is recomputed per frame by the caller.
    """
    payload_path, receipt_path = directory / "frames" / (key + ".json"), directory / "frames" / (key + ".sha256.json")
    if not payload_path.exists() or not receipt_path.exists():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(receipt, dict):
        return None
    if receipt.get("binding_fingerprint") != binding:
        raise ValueError("Existing frame cache has a different binding; use a new diagnostic version")
    if receipt.get("payload_file") != payload_path.name or receipt.get("identity") != identity:
        raise ValueError("Frame cache identity mismatch")
    if file_sha256(payload_path) != receipt.get("payload_sha256"):
        return None
    try:
        row = json.loads(payload_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(row, dict):
        return None
    try:
        if row["binding_fingerprint"] != binding or row["identity"] != identity:
            raise ValueError("Cached payload identity mismatch")
        if set(row["combinations"]) != {combination_name(heads) for heads in COMBINATIONS}:
            return None
        verify_model_metrics(row["combinations"]["model_only"], reference)
        image = row.get("image")
        if image_required and not image:
            return None
        if image:
            path = (directory / image["file"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Cached image escapes diagnostic directory")
            if not path.is_file() or file_sha256(path) != image["sha256"]:
                return None
        return row
    except KeyError:
        return None


def commit_frame(directory, key, row):
    path = directory / "frames" / (key + ".json")
    atomic_json(path, row)
    atomic_json(path.with_name(key + ".sha256.json"), {
        "schema": 1, "identity": row["identity"], "binding_fingerprint": row["binding_fingerprint"],
        "payload_file": path.name, "payload_sha256": file_sha256(path)})


def aggregate_rows(rows):
    if not rows:
        return {"frame_count": 0, "scene_count": 0, "combinations": {}}
    result = {"frame_count": len(rows), "scene_count": len({row["identity"]["scene"] for row in rows}), "combinations": {}}
    for heads in COMBINATIONS:
        name = combination_name(heads)
        items = [row["combinations"][name] for row in rows]
        counts = {key: sum(item["instances"][key] for item in items) for key in ("tp", "fp", "fn")}
        intersection, union = (sum(item[key] for item in items) for key in ("foreground_intersection", "foreground_union"))
        values = [value for item in items for value in item["best_iou_per_gt_instance"]]
        result["combinations"][name] = {"instances": summarize_counts(counts),
            "foreground_pixel_iou": intersection / union if union else 1.0,
            "foreground_intersection": intersection, "foreground_union": union,
            "mean_best_gt_instance_iou": float(np.mean(values)) if values else 1.0}
    result["center_matching"] = {}
    for radius in (5, 10):
        totals = {key: sum(row["center_matching"][str(radius)][key] for row in rows) for key in ("tp", "fp", "fn")}
        tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
        result["center_matching"][str(radius)] = dict(totals, precision=tp / (tp + fp) if tp + fp else 0,
            recall=tp / (tp + fn) if tp + fn else 0, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    instances = [item for row in rows for item in row["per_instance"]]
    result["target_window_max_probability"] = distribution([item["target_9x9_max_probability"] for item in instances])
    result["target_center_probability"] = distribution([item["target_center_probability"] for item in instances])
    result["offset_instance_mean_epe_pixels"] = distribution([item["offset_epe_mean_pixels"] for item in instances])
    result["model_center_probability_max"] = distribution([row["center_probability_max"] for row in rows])
    result["all_oracle_exact_frames"] = sum(row["all_oracle"]["exact_instance_partition"] for row in rows)
    result["all_oracle_nonexact_frames"] = len(rows) - result["all_oracle_exact_frames"]
    result["small_instances_below_min_pixels"] = sum(len(row["all_oracle"]["small_gt_instances"]) for row in rows)
    result["coincident_gt_center_pairs"] = sum(len(row["all_oracle"]["coincident_gt_center_pairs"]) for row in rows)
    result["close_gt_center_pairs_lt5px"] = sum(len(row["all_oracle"]["close_gt_center_pairs_lt5px"]) for row in rows)
    result["labels_lost_during_resize"] = sum(len(row["all_oracle"]["labels_lost_during_resize"]) for row in rows)
    result["centroids_outside_own_mask"] = sum(item["centroid_outside_own_mask"] for item in instances)
    result["resized_instance_count"] = len(instances)
    result["centroids_outside_own_mask_fraction"] = result["centroids_outside_own_mask"] / len(instances) if instances else 0.0
    result["selected_center_count"] = sum(row["selected_center_count"] for row in rows)
    result["ground_truth_center_count"] = sum(row["gt_center_count"] for row in rows)
    result["unique_ground_truth_center_count"] = sum(row["unique_gt_center_count"] for row in rows)
    return result


def grouped_summary(rows):
    groups = {"train": [row for row in rows if row["identity"]["split"] == "train"],
              "development_validation": [row for row in rows if row["identity"]["split"] == "development_validation"],
              "historically_fresh_four_scenes_now_development": [row for row in rows if row["identity"]["scene"] in (18, 22, 33, 39)],
              "previously_seen_scene13": [row for row in rows if row["identity"]["scene"] == 13]}
    summaries = {}
    for name, members in groups.items():
        scenes = [{"scene": scene, **aggregate_rows([row for row in members if row["identity"]["scene"] == scene])}
                  for scene in sorted({row["identity"]["scene"] for row in members})]
        micro = aggregate_rows(members)
        macro = {combination_name(heads): {
            "foreground_pixel_iou": float(np.mean([scene["combinations"][combination_name(heads)]["foreground_pixel_iou"] for scene in scenes])),
            "instance_f1_iou50": float(np.mean([scene["combinations"][combination_name(heads)]["instances"]["f1_iou50"] for scene in scenes]))}
            for heads in COMBINATIONS} if scenes else {}
        scene_offset_means = [scene["offset_instance_mean_epe_pixels"]["mean"] for scene in scenes
                              if scene["offset_instance_mean_epe_pixels"]["mean"] is not None]
        diagnostic_macro = {"offset_instance_mean_epe_pixels": float(np.mean(scene_offset_means)) if scene_offset_means else None,
            "center_matching": {str(radius): {metric: float(np.mean([scene["center_matching"][str(radius)][metric] for scene in scenes]))
                                              for metric in ("precision", "recall", "f1")} for radius in (5, 10)}} if scenes else {}
        summaries[name] = {"micro": micro, "scene_macro": macro, "diagnostic_scene_macro": diagnostic_macro, "scenes": scenes}
    return summaries
