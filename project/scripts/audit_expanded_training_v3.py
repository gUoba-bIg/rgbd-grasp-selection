"""Read-only optimization audit of frozen v3 history; standard library only."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Audit paths must remain in the project")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def atomic_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Empty distribution")
    position = probability * (len(ordered) - 1)
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (position - left) * (ordered[right] - ordered[left])


def distribution(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Statistics require nonempty finite observations")
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "p10": quantile(values, 0.10),
            "p90": quantile(values, 0.90), "population_std": statistics.pstdev(values)}


def slope_per_100_updates(rows):
    xs, ys = [float(row["step"]) for row in rows], [float(row["loss"]) for row in rows]
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return 100 * sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator if denominator else 0.0


def summary(rows, clip_norm):
    components = {name: distribution(row["components"][name] for row in rows) for name in
                  ("foreground_bce", "foreground_dice_loss", "center_focal", "offset_smooth_l1")}
    weighted = {
        "foreground_bce_plus_dice": distribution(row["components"]["foreground_bce"] + row["components"]["foreground_dice_loss"] for row in rows),
        "center_focal_weight_1": components["center_focal"],
        "offset_smooth_l1_weight_5": distribution(5 * row["components"]["offset_smooth_l1"] for row in rows),
    }
    clip_count = sum(row["gradient_norm_before_clip"] > clip_norm for row in rows)
    return {"first_step": rows[0]["step"], "last_step": rows[-1]["step"], "updates": len(rows),
            "total_loss": distribution(row["loss"] for row in rows), "raw_loss_components": components,
            "weighted_loss_groups": weighted,
            "gradient_norm_before_clip": distribution(row["gradient_norm_before_clip"] for row in rows),
            "gradient_clip_threshold": clip_norm, "updates_above_clip_threshold": clip_count,
            "fraction_above_clip_threshold": clip_count / len(rows),
            "amp_retry_total": sum(row["amp_retries"] for row in rows),
            "updates_with_amp_retry": sum(row["amp_retries"] > 0 for row in rows),
            "amp_scale_values": sorted({row["amp_scale"] for row in rows}),
            "learning_rate_used": {"first": rows[0]["learning_rate_used"], "last": rows[-1]["learning_rate_used"],
                                   **distribution(row["learning_rate_used"] for row in rows)},
            "descriptive_loss_slope_per_100_updates": slope_per_100_updates(rows),
            "slope_caveat": "Descriptive only; sampled scenes and augmentations vary, no convergence significance test"}


def relative_reduction(early, late):
    return (early - late) / early * 100 if early else None


def expected_learning_rate(step, protocol):
    base, floor, warmup = protocol["learning_rate"], protocol["min_learning_rate"], protocol["warmup_steps"]
    desired = base * step / warmup if step <= warmup else floor + (base - floor) * 0.5 * (
        1 + math.cos(math.pi * (step - warmup) / (protocol["planned_steps"] - warmup)))
    return base * (desired / base)


def audit(run_directory):
    names = ("history.json", "config.json", "data_manifest.json", "result.json")
    paths = {name: run_directory / name for name in names}
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    history, config, manifest, result = (loaded[name] for name in names)
    protocol = config["protocol"]
    if not history:
        raise ValueError("No training history")
    source_checks = {name: {"recorded_sha256": expected, "current_sha256": sha256(project_path(name))}
                     for name, expected in config["source_sha256"].items()}
    for row in source_checks.values():
        row["unchanged"] = row["recorded_sha256"] == row["current_sha256"]
    planned_steps = protocol["planned_steps"]
    expected_samples = {(row["scene"], row["frame"]) for row in manifest["records"]["train"]}
    expected_epochs = protocol["equivalent_epochs"]
    exposure = Counter()
    epoch_rows = defaultdict(list)
    epoch_exposure = defaultdict(Counter)
    reconstruction_errors = []
    for row in history:
        epoch_rows[row["epoch"]].append(row)
        for sample in row["samples"]:
            identity = sample["scene"], sample["frame"]
            exposure[identity] += 1
            epoch_exposure[row["epoch"]][identity] += 1
        parts = row["components"]
        reconstruction_errors.append(abs(row["loss"] - (parts["foreground_bce"] + parts["foreground_dice_loss"] +
                                                        parts["center_focal"] + 5 * parts["offset_smooth_l1"])))
    unexpected_samples = sorted(set(exposure) - expected_samples)
    missing_samples = sorted(expected_samples - set(exposure))
    lr_errors = [abs(row["learning_rate_used"] - expected_learning_rate(row["step"], protocol)) for row in history]
    checks = {
        "history_steps_are_contiguous_and_complete": [row["step"] for row in history] == list(range(1, planned_steps + 1)),
        "result_records_completed_plan": result["status"] == "completed" and result["optimizer_steps"] == planned_steps,
        "final_cursor_is_epoch20_boundary": result["cursor"] == {"epoch": expected_epochs, "next_batch": 0, "global_step": planned_steps},
        "config_fingerprint_matches_result": fingerprint(config) == result["config_fingerprint"],
        "manifest_fingerprint_matches_config_and_result": fingerprint(manifest) == config["data_fingerprint"] == result["data_fingerprint"],
        "source_files_unchanged": all(row["unchanged"] for row in source_checks.values()),
        "no_unexpected_or_missing_training_frames": not unexpected_samples and not missing_samples,
        "every_training_frame_exposed_20_times": len(expected_samples) == 160 and all(exposure[item] == expected_epochs for item in expected_samples),
        "each_epoch_exposes_each_training_frame_once": set(epoch_exposure) == set(range(expected_epochs)) and all(
            set(values) == expected_samples and all(count == 1 for count in values.values()) for values in epoch_exposure.values()),
        "recorded_loss_reconstructs_from_weighted_components": max(reconstruction_errors) < 1e-5,
        "actual_learning_rates_match_frozen_schedule": max(lr_errors) < 1e-15,
    }
    all_history = summary(history, protocol["gradient_clip_norm"])
    epochs = [dict(epoch_index=index, epoch_number=index + 1,
                   sample_exposures=sum(epoch_exposure[index].values()), unique_training_frames=len(epoch_exposure[index]),
                   **summary(epoch_rows[index], protocol["gradient_clip_norm"])) for index in sorted(epoch_rows)]
    # A full post-warmup epoch provides all 160 examples once, avoiding the huge
    # first-step transient as the comparison baseline. This window is descriptive,
    # not a claim of stationary optimization or identical per-epoch augmentation.
    early_rows = [row for row in history if 81 <= row["step"] <= 160]
    final_rows = [row for row in history if 1521 <= row["step"] <= 1600]
    early, final = summary(early_rows, protocol["gradient_clip_norm"]), summary(final_rows, protocol["gradient_clip_norm"])
    last400_rows = [row for row in history if 1201 <= row["step"] <= 1600]
    last400 = summary(last400_rows, protocol["gradient_clip_norm"])
    last400["consecutive_100_update_blocks"] = [summary(last400_rows[index:index + 100], protocol["gradient_clip_norm"]) for index in range(0, 400, 100)]
    first_block, last_block = last400["consecutive_100_update_blocks"][0], last400["consecutive_100_update_blocks"][-1]
    last400["first100_to_last100_loss_reduction_percent"] = relative_reduction(first_block["total_loss"]["mean"], last_block["total_loss"]["mean"])
    last400["matched_complete_epoch_comparison"] = {
        "first_epoch_number": 16, "last_epoch_number": 20,
        "first_epoch_mean_loss": epochs[15]["total_loss"]["mean"], "last_epoch_mean_loss": epochs[19]["total_loss"]["mean"],
        "loss_reduction_percent": relative_reduction(epochs[15]["total_loss"]["mean"], epochs[19]["total_loss"]["mean"]),
        "weighted_component_mean_changes_last_minus_first": {
            name: epochs[19]["weighted_loss_groups"][name]["mean"] - epochs[15]["weighted_loss_groups"][name]["mean"]
            for name in epochs[15]["weighted_loss_groups"]},
        "caveat": "Each compared epoch contains each training frame once; augmentations and learning rates differ. Scalar component changes do not identify gradient dominance.",
    }
    stable_comparison = {"early_full_post_warmup_epoch_steps81_160": early, "final_full_epoch_steps1521_1600": final,
                         "total_loss_reduction_percent": relative_reduction(early["total_loss"]["mean"], final["total_loss"]["mean"]),
                         "excluded_from_baseline": "Steps 1..80 including the first-step initialization transient",
                         "interpretation": "Matched full-data-pass windows, with epoch-specific augmentation and different LR; not validation loss"}
    comparison = {}
    for name in ("train", "validation", "fresh_validation", "previously_seen_validation"):
        value = result["evaluation"][name]
        comparison[name] = {key: value[key] for key in ("frame_count", "scene_count", "foreground_pixel_iou", "instances", "scene_macro")}
    train_macro = comparison["train"]["scene_macro"]
    fresh_macro = comparison["fresh_validation"]["scene_macro"]
    comparison["train_minus_fresh_scene_macro"] = {
        "foreground_iou_percentage_points": 100 * (train_macro["foreground_pixel_iou"] - fresh_macro["foreground_pixel_iou"]),
        "instance_f1_percentage_points": 100 * (train_macro["instance_f1_iou50"] - fresh_macro["instance_f1_iou50"])}
    comparison["caveat"] = "Different scenes; this gap is descriptive and does not isolate optimization, distribution shift, architecture or decoding causes"
    report = {
        "schema": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "integrity_checks_failed",
        "audit_scope": "Read-only standard-library analysis of frozen training logs, metadata and source hashes; no model execution",
        "run_directory": str(run_directory), "checks": checks, "input_sha256": input_hashes,
        "source_integrity": source_checks, "audit_script_sha256": sha256(Path(__file__)),
        "checkpoint_weight_loading": False, "cuda_or_tensor_libraries_imported": False,
        "training_or_source_modified": False, "maximum_loss_reconstruction_abs_error": max(reconstruction_errors),
        "maximum_learning_rate_abs_error": max(lr_errors),
        "loss_definition": "total = foreground_bce + foreground_dice_loss + center_focal + 5 * offset_smooth_l1; reported raw offset is unweighted",
        "gradient_interpretation": "Only the global gradient norm before clipping was logged. Scalar loss magnitude or weighted share does not identify per-head gradient magnitude, direction, competition or dominance. No gradients were recomputed.",
        "all_updates": all_history, "by_epoch": epochs, "stable_window_comparison": stable_comparison,
        "last_400_updates": last400,
        "sample_exposure": {"unique_frames": len(exposure), "expected_unique_frames": len(expected_samples),
                            "total_training_sample_presentations": sum(exposure.values()),
                            "expected_presentations_per_frame": expected_epochs,
                            "minimum_presentations": min(exposure.values()), "maximum_presentations": max(exposure.values()),
                            "missing": missing_samples, "unexpected": unexpected_samples,
                            "per_frame": [{"scene": scene, "frame": frame, "presentations": exposure[(scene, frame)]} for scene, frame in sorted(exposure)]},
        "existing_final_model_metrics": comparison,
        "interpretation": {
            "completed_not_interrupted": "All 1600 committed optimizer updates and 20 presentations of each training frame are checked; this was not an accidentally shortened run.",
            "incomplete_instance_fitting": "Training-instance F1 is below perfect fit despite high foreground IoU. This supports an unresolved instance-head/representation/decoding limitation, not a specific proven cause.",
            "late_optimization": "Window losses and descriptive late slope quantify continuing changes; decreasing loss does not demonstrate that more updates will raise instance F1 or held-out performance.",
            "clipping": "Frequent global clipping limits aggregate gradient norm; it does not alone show harmful optimization or which head controls the update.",
            "generalization": "The train-versus-fresh instance metric gap also leaves generalization as a separate issue. More training can improve training fit without improving the evaluated scenes.",
            "need_to_restart_from_scratch": "No evidence here requires discarding the trained weights, verified data or completed stages. Do not repeat identical 1600-step training merely because the engineering gate failed.",
        },
        "minimal_next_comparison_proposal_not_executed": {
            "first": "Use the pending fixed-weight ground-truth-component substitution diagnostic to localize foreground, center, offset and decoder limitations without retraining.",
            "if_optimization_budget_is_selected": "Create a new explicitly versioned continuation experiment from the frozen step-1600 state, retaining the original control. Prespecify a small fixed additional budget (for example 400 updates at the existing 1e-5 floor), unchanged loss/architecture/decoder/data, and no automatic extension. This is a new diagnostic protocol, never a mutation of the completed v3 plan.",
            "continuation_question": "Does extra optimization improve the same training-frame instance fit? Do not infer held-out benefit from training loss alone.",
            "if_component_failure_is_selected": "Instead of extending training blindly, choose one component change indicated by substitution results; compare it against a matched unchanged control with the same data, initialization, seed and update schedule.",
            "evaluation_boundary": "The four previously fresh validation scenes have now been inspected and are development validation. Avoid repeated tuning on them while describing them as blind evidence; preserve calibration and official test partitions.",
            "authority": "Suggestions only; no new training, hyperparameter changes or experiment extension was performed.",
        },
    }
    # Detect files modified concurrently during this read, rather than reporting a mixed snapshot.
    checks["input_snapshot_unchanged_during_audit"] = all(sha256(paths[name]) == value for name, value in input_hashes.items())
    report["status"] = "passed" if all(checks.values()) else "integrity_checks_failed"
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="experiments/segmentation_expanded_v3/rgbd_original")
    parser.add_argument("--output", default="results/P2/expanded_v3_optimization_audit.json")
    args = parser.parse_args()
    run_directory, output = project_path(args.run_dir), project_path(args.output)
    if output.is_relative_to(run_directory) or output == Path(__file__).resolve():
        raise ValueError("Audit output must not overwrite experiment files or source")
    report = audit(run_directory)
    atomic_report(output, report)
    compact = {"status": report["status"], "report": str(output),
               "window_loss_early": report["stable_window_comparison"]["early_full_post_warmup_epoch_steps81_160"]["total_loss"]["mean"],
               "window_loss_final": report["stable_window_comparison"]["final_full_epoch_steps1521_1600"]["total_loss"]["mean"],
               "late400_loss_reduction_percent": report["last_400_updates"]["first100_to_last100_loss_reduction_percent"],
               "amp_retries": report["all_updates"]["amp_retry_total"],
               "clip_fraction": report["all_updates"]["fraction_above_clip_threshold"],
               "exposure_range": [report["sample_exposure"]["minimum_presentations"], report["sample_exposure"]["maximum_presentations"]]}
    print(json.dumps(compact, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
