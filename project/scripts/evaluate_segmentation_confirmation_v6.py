"""Frozen two-model confirmation with atomic, independently verifiable frame pairs."""
import argparse
import ast
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import PIL
from PIL import Image, ImageDraw
import scipy
import torch
import torchvision

from scripts.train_segmentation_repair_v5 import load_final_model
from scripts.train_segmentation_expanded_v3 import summarize_frames
from scripts.accept_segmentation_expanded_v3 import verify_partition, verify_counts
from scripts.render_segmentation_expanded_v3 import instance_colors
from src.checkpoints import atomic_json, atomic_write, file_sha256, fingerprint, _exclusive_lock
from src.graspnet_frames import GraspNetFrames
from src.segmentation import decode_instances, instance_counts, summarize_counts
from src.segmentation_diagnostics_v3 import selected_centers, center_matches, distribution

ARMS = ("original", "center_mse")
DEFAULT_CONFIG = "configs/segmentation_confirmation_v6.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("Artifact path escapes the project")
    return path


def validate_protocol(protocol):
    required = {"schema": 6, "validation_scenes": [2, 6, 7, 11, 15],
                "previously_used_validation_scenes": [13, 18, 22, 33, 39],
                "remaining_unopened_validation_scenes": [52, 59, 75, 79, 80],
                "frames": list(range(0, 256, 16)), "camera": "realsense",
                "mode": "rgbd", "input_size": [240, 320], "augment": False,
                "seed": 20261022, "new_training_updates": 0,
                "decoder": {"foreground_threshold": .5, "center_threshold": .35, "min_pixels": 20, "max_centers": 64},
                "acceptance": {"scene_macro_foreground_iou_min": .7, "scene_macro_instance_f1_min": .5, "minimum_tp_per_scene": 1},
                "candidate_adoption": {"scene_macro_f1_gain_over_control_min": .02, "scene_macro_foreground_iou_drop_max": .01,
                                       "minimum_scenes_with_positive_f1_gain": 4},
                "visualization_scenes": [2, 6, 7, 11, 15], "visualization_frame": "0000"}
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError("Frozen confirmation protocol differs: " + key)
    if set(protocol["arms"]) != set(ARMS):
        raise ValueError("Exactly the frozen two arms required")
    splits = read(project_path(protocol["split"]))["groups"]
    unseen = sorted(set(splits["validation"]) - set(protocol["previously_used_validation_scenes"]))
    if unseen[:5] != protocol["validation_scenes"] or unseen[5:] != protocol["remaining_unopened_validation_scenes"]:
        raise ValueError("Scene selection does not follow the frozen ascending-ID rule")
    if set(protocol["validation_scenes"]) & (set(splits["train"]) | set(splits["calibration"])):
        raise ValueError("Confirmation split overlaps train or calibration")


def source_hashes():
    """Resolve the transitive local import graph, including lazy imports and aliases."""
    pending, visited = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
                names.extend(node.module + "." + alias.name for alias in node.names)
        for name in names:
            if name.split(".")[0] not in ("src", "scripts"):
                continue
            candidate = ROOT / (name.replace(".", "/") + ".py")
            package = ROOT / name.replace(".", "/") / "__init__.py"
            for existing in (candidate, package):
                if existing.is_file() and existing not in visited:
                    pending.append(existing)
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(visited)}


def load_data(protocol):
    manifest_path = project_path(protocol["data_manifest"])
    manifest, audit = read(manifest_path), read(project_path(protocol["data_audit"]))
    if manifest.get("schema") != 6 or manifest.get("status") != "verified_confirmation_sample":
        raise ValueError("Confirmation data are not verified")
    if audit.get("status") != "passed" or audit.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Data audit does not bind the current completed manifest")
    if manifest["split_sha256"] != file_sha256(project_path(protocol["split"])):
        raise ValueError("Data manifest split hash changed")
    for key in ("validation_scenes", "frames", "camera"):
        if manifest["selection"][key] != protocol[key]:
            raise ValueError("Data selection differs: " + key)
    records = manifest["records"]["validation"]
    expected = [(scene, "{:04d}".format(frame)) for scene in protocol["validation_scenes"] for frame in protocol["frames"]]
    if manifest["records"].get("train") or [(r["scene"], r["frame"]) for r in records] != expected:
        raise ValueError("Require exactly80 ordered confirmation records and no training frames")
    root = project_path(protocol["data_root"])
    expected_files = set()
    for record in records:
        if record["camera"] != protocol["camera"] or set(record["files"]) != {"rgb", "depth", "label", "meta"} or set(record["sha256"]) != set(record["files"]):
            raise ValueError("Data record schema/camera differs")
        for kind, name in record["files"].items():
            suffix = ".mat" if kind == "meta" else ".png"
            wanted = "scenes/scene_{:04d}/{}/{}/{}{}".format(record["scene"], protocol["camera"], kind, record["frame"], suffix)
            if name.replace("\\", "/") != wanted:
                raise ValueError("Data frame identity differs from its path")
            expected_files.add(wanted)
            if manifest["files"][wanted]["sha256"] != record["sha256"][kind]:
                raise ValueError("Frame checksum differs from manifest file table")
    for scene in protocol["validation_scenes"]:
        prefix = "scenes/scene_{:04d}/".format(scene)
        expected_files.add(prefix + "object_id_list.txt")
        expected_files.update(prefix + protocol["camera"] + "/" + name for name in ("camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"))
    if set(manifest["files"]) != expected_files:
        raise ValueError("Expected exactly340 frozen frame and calibration files")
    for name, receipt in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != receipt["bytes"] or file_sha256(path) != receipt["sha256"]:
            raise ValueError("Confirmation source file checksum/size mismatch: " + name)
    return manifest, audit


def build_binding(config_path, protocol, manifest, evidence):
    return {"schema": 6, "protocol_sha256": file_sha256(config_path), "protocol": protocol,
            "source_sha256": source_hashes(), "data_manifest_sha256": file_sha256(project_path(protocol["data_manifest"])),
            "data_audit_sha256": file_sha256(project_path(protocol["data_audit"])),
            "data_files_sha256": {name: item["sha256"] for name, item in manifest["files"].items()},
            "checkpoint_evidence": evidence, "device": "cuda", "amp": False,
            "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                         "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__,
                         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
            "device_name": torch.cuda.get_device_name(0), "deterministic_algorithms": True,
            "tf32": False, "new_training_updates": 0}


def frame_identity(record):
    return {"split": "confirmation_validation", "scene": record["scene"], "frame": record["frame"], "camera": "realsense"}


def cache_key(identity):
    if type(identity["scene"]) is not int or identity["scene"] < 0 or not isinstance(identity["frame"], str) or len(identity["frame"]) != 4 or not identity["frame"].isdigit():
        raise ValueError("Invalid cache identity")
    return "scene_{:04d}_{}".format(identity["scene"], identity["frame"])


def integer(value, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError("Expected nonnegative bounded integer")


def finite(value, lower=0, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        raise ValueError("Invalid finite scalar")


def verify_metrics(metrics, identity):
    keys = {"scene", "frame", "foreground_intersection", "foreground_union", "foreground_pixel_iou", "instances", "predicted_instances", "actual_instances", "max_center_probability"}
    if set(metrics) != keys or metrics["scene"] != identity["scene"] or metrics["frame"] != identity["frame"]:
        raise ValueError("Frame metrics identity/schema differs")
    if set(metrics["instances"]) != {"tp", "fp", "fn", "precision_iou50", "recall_iou50", "f1_iou50"}:
        raise ValueError("Instance schema differs")
    verify_counts(metrics["instances"])
    for key in ("precision_iou50", "recall_iou50", "f1_iou50"):
        finite(metrics["instances"][key], 0, 1)
    numerator, denominator = metrics["foreground_intersection"], metrics["foreground_union"]
    integer(numerator, 76800)
    integer(denominator, 76800)
    if numerator > denominator or metrics["foreground_pixel_iou"] != (numerator / denominator if denominator else 1.0):
        raise ValueError("Foreground count/ratio inconsistency")
    finite(metrics["foreground_pixel_iou"], 0, 1)
    integer(metrics["actual_instances"], 88)
    integer(metrics["predicted_instances"], 64)
    counts = metrics["instances"]
    if metrics["actual_instances"] != counts["tp"] + counts["fn"] or metrics["predicted_instances"] != counts["tp"] + counts["fp"]:
        raise ValueError("Object totals disagree with instance counts")
    finite(metrics["max_center_probability"], 0, 1)


def verify_diagnostics(diagnostic, metrics):
    if set(diagnostic) != {"selected_center_coordinates_yx", "gt_center_coordinates_yx", "center_matching", "offset_instance_mean_epe_pixels"}:
        raise ValueError("Diagnostics schema differs")
    selected, actual = diagnostic["selected_center_coordinates_yx"], diagnostic["gt_center_coordinates_yx"]
    if not isinstance(selected, list) or not isinstance(actual, list) or len(selected) > 64 or len(actual) != metrics["actual_instances"]:
        raise ValueError("Diagnostic center counts differ")
    for point in selected + actual:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Center coordinates must have two components")
        integer(point[0], 239)
        integer(point[1], 319)
    if len({tuple(point) for point in selected}) != len(selected):
        raise ValueError("Duplicate selected centers")
    if set(diagnostic["center_matching"]) != {"5", "10"}:
        raise ValueError("Require both fixed diagnostic radii")
    for radius in (5, 10):
        # Recompute assignment from recorded coordinates, not merely count arithmetic.
        if diagnostic["center_matching"][str(radius)] != center_matches(selected, actual, radius):
            raise ValueError("Center matching cannot be reproduced from coordinates")
    errors = diagnostic["offset_instance_mean_epe_pixels"]
    if not isinstance(errors, list) or len(errors) != len(actual):
        raise ValueError("Per-instance offset error count differs")
    for value in errors:
        finite(value)


def verify_row(row, binding_hash, identity, image_required=False):
    if set(row) != {"schema", "binding_fingerprint", "identity", "arms", "image"} or row["schema"] != 6 or row["binding_fingerprint"] != binding_hash or row["identity"] != identity:
        raise ValueError("Cached frame identity/schema/binding differs")
    if set(row["arms"]) != set(ARMS):
        raise ValueError("Both arms must be committed in one row")
    for value in row["arms"].values():
        if set(value) != {"metrics", "diagnostics"}:
            raise ValueError("Cached arm schema differs")
        verify_metrics(value["metrics"], identity)
        verify_diagnostics(value["diagnostics"], value["metrics"])
    if row["arms"][ARMS[0]]["diagnostics"]["gt_center_coordinates_yx"] != row["arms"][ARMS[1]]["diagnostics"]["gt_center_coordinates_yx"]:
        raise ValueError("The two arms have different reference labels")
    if image_required != bool(row["image"]):
        raise ValueError("Visualization selection differs from frozen frame rule")
    if row["image"]:
        image = row["image"]
        if set(image) != {"file", "sha256", "counts", "metrics_recomputed"} or image["file"] != "images/" + cache_key(identity) + ".png" or image["metrics_recomputed"] is not True:
            raise ValueError("Image receipt schema differs")
        if image["counts"] != {arm: row["arms"][arm]["metrics"]["instances"] for arm in ARMS}:
            raise ValueError("Image receipt counts differ from frame")
    return row


def load_cached_frame(directory, binding_hash, identity, image_required=False, strict=False):
    key = cache_key(identity)
    path, receipt_path = directory / "frames" / (key + ".json"), directory / "frames" / (key + ".sha256.json")
    def incomplete(message):
        if strict:
            raise ValueError(message)
        return None
    if not path.exists() or not receipt_path.exists():
        return incomplete("Incomplete cached frame pair")
    try:
        receipt = read(receipt_path)
    except (ValueError, OSError):
        return incomplete("Unreadable cached frame receipt")
    if not isinstance(receipt, dict):
        return incomplete("Invalid cached frame receipt")
    if receipt.get("binding_fingerprint") != binding_hash or receipt.get("identity") != identity or receipt.get("payload_file") != path.name:
        raise ValueError("Cache receipt belongs to another binding/frame; refusing overwrite")
    if set(receipt) != {"binding_fingerprint", "identity", "payload_file", "payload_sha256"} or file_sha256(path) != receipt["payload_sha256"]:
        return incomplete("Cached frame checksum mismatch")
    try:
        row = read(path)
        verify_row(row, binding_hash, identity, image_required)
        if row["image"]:
            image_path = directory / row["image"]["file"]
            if not image_path.is_file() or file_sha256(image_path) != row["image"]["sha256"]:
                return incomplete("Cached image checksum mismatch")
        return row
    except (ValueError, KeyError, TypeError, AssertionError, OSError) as error:
        return incomplete("Cached frame validation failed: " + str(error))


def commit_frame(directory, row):
    path = directory / "frames" / (cache_key(row["identity"]) + ".json")
    atomic_json(path, row)
    atomic_json(path.with_name(path.stem + ".sha256.json"),
                {"binding_fingerprint": row["binding_fingerprint"], "identity": row["identity"],
                 "payload_file": path.name, "payload_sha256": file_sha256(path)})


def analyze_outputs(outputs, targets, record, decoder):
    labels = targets["labels"].numpy()
    prediction = decode_instances(outputs, **decoder)
    foreground = outputs["foreground"][0, 0].float().sigmoid().detach().cpu().numpy() >= decoder["foreground_threshold"]
    center_probability = outputs["centers"][0, 0].float().sigmoid().detach().cpu()
    actual = labels > 0
    intersection, union = int((foreground & actual).sum()), int((foreground | actual).sum())
    counts = summarize_counts(instance_counts(prediction, labels))
    metrics = {"scene": record["scene"], "frame": record["frame"], "foreground_intersection": intersection,
               "foreground_union": union, "foreground_pixel_iou": intersection / union if union else 1.0,
               "instances": counts, "predicted_instances": counts["tp"] + counts["fp"], "actual_instances": counts["tp"] + counts["fn"],
               "max_center_probability": float(center_probability.max())}
    peaks = selected_centers(center_probability, decoder["center_threshold"], decoder["max_centers"])
    difference = outputs["offsets"][0].detach().float().cpu().numpy() - targets["offsets"].numpy()
    difference[0] *= labels.shape[1]
    difference[1] *= labels.shape[0]
    epe = np.sqrt((difference ** 2).sum(axis=0))
    centers, errors = [], []
    for label in np.unique(labels[actual]):
        yy, xx = np.nonzero(labels == label)
        centers.append([int(np.round(yy.astype(np.float32).mean())), int(np.round(xx.astype(np.float32).mean()))])
        errors.append(float(epe[yy, xx].mean()))
    diagnostics = {"selected_center_coordinates_yx": peaks.tolist(), "gt_center_coordinates_yx": centers,
                   "center_matching": {str(radius): center_matches(peaks, centers, radius) for radius in (5, 10)},
                   "offset_instance_mean_epe_pixels": errors}
    return {"metrics": metrics, "diagnostics": diagnostics}, prediction


def draw_frame(directory, record, labels, predictions, arms, protocol):
    with Image.open(project_path(protocol["data_root"]) / record["files"]["rgb"]) as source:
        rgb = source.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
    panels = [("RGB input", rgb), ("GT reference only", instance_colors(labels))]
    counts = {}
    for arm in ARMS:
        counts[arm] = summarize_counts(instance_counts(predictions[arm], labels))
        if counts[arm] != arms[arm]["metrics"]["instances"]:
            raise AssertionError("Visualization prediction/count mismatch")
        panels.append((arm + " | F1 {:.3f}".format(counts[arm]["f1_iou50"]), instance_colors(predictions[arm])))
    canvas = Image.new("RGB", (1280, 336), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "FIXED-MODEL CONFIRMATION | scene {:04d} frame {} | predeclared example, independent instance colors".format(record["scene"], record["frame"]), fill="black")
    draw.text((8, 26), "Both checkpoints frozen after v3 1600 + v5 1600 steps; zero updates here. Fixed fg .50 / center .35 / area20 / max64.", fill="black")
    draw.text((8, 44), "GT is reference only; labels never enter inference decoder. Scene examples do not substitute for all80 frame metrics.", fill="black")
    for index, (caption, pane) in enumerate(panels):
        draw.text((index * 320 + 7, 70), caption, fill="black")
        canvas.paste(pane, (index * 320, 94))
    name = "images/" + cache_key(frame_identity(record)) + ".png"
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_write(directory / name, lambda stream: stream.write(buffer.getvalue()))
    return {"file": name, "sha256": file_sha256(directory / name), "counts": counts, "metrics_recomputed": True}


def engineering_gate(partition, protocol):
    rule = protocol["acceptance"]
    return (partition["scene_macro"]["foreground_pixel_iou"] >= rule["scene_macro_foreground_iou_min"]
            and partition["scene_macro"]["instance_f1_iou50"] >= rule["scene_macro_instance_f1_min"]
            and len(partition["scenes"]) == len(protocol["validation_scenes"])
            and {row["scene"] for row in partition["scenes"]} == set(protocol["validation_scenes"])
            and all(row["instances"]["tp"] >= rule["minimum_tp_per_scene"] for row in partition["scenes"]))


def adoption(partitions, protocol):
    control, candidate = (partitions[name] for name in ARMS)
    gates = {name: engineering_gate(partitions[name], protocol) for name in ARMS}
    gain = candidate["scene_macro"]["instance_f1_iou50"] - control["scene_macro"]["instance_f1_iou50"]
    drop = control["scene_macro"]["foreground_pixel_iou"] - candidate["scene_macro"]["foreground_pixel_iou"]
    first = {row["scene"]: row["instances"]["f1_iou50"] for row in control["scenes"]}
    deltas = {str(row["scene"]): row["instances"]["f1_iou50"] - first[row["scene"]] for row in candidate["scenes"]}
    rule = protocol["candidate_adoption"]
    # Decimal thresholds receive only machine-roundoff tolerance, matching v5.
    checks = {"candidate_passes_absolute_gate": gates["center_mse"],
              "scene_macro_f1_gain": gain >= rule["scene_macro_f1_gain_over_control_min"] or math.isclose(gain, rule["scene_macro_f1_gain_over_control_min"], rel_tol=0, abs_tol=1e-12),
              "scene_macro_foreground_guard": drop <= rule["scene_macro_foreground_iou_drop_max"] or math.isclose(drop, rule["scene_macro_foreground_iou_drop_max"], rel_tol=0, abs_tol=1e-12),
              "scene_consistency": sum(value > 0 for value in deltas.values()) >= rule["minimum_scenes_with_positive_f1_gain"]}
    selected = "center_mse" if all(checks.values()) else "original" if gates["original"] else None
    return {"engineering_gate_by_arm": gates, "candidate_adoption_checks": checks,
            "scene_macro_f1_gain_candidate_minus_control": gain,
            "scene_macro_foreground_drop_control_minus_candidate": drop,
            "scene_f1_gain_candidate_minus_control": deltas, "selected_arm": selected,
            "decision": "candidate_engineering_confirmed" if selected == "center_mse" else "control_engineering_confirmed" if selected else "neither_arm_confirmed",
            "interpretation": "Five scene-level engineering confirmation; not significance, official test, mask AP or grasp performance"}


def summarize_diagnostics(values):
    matching = {}
    for radius in ("5", "10"):
        counts = {key: sum(row["center_matching"][radius][key] for row in values) for key in ("tp", "fp", "fn")}
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        matching[radius] = dict(counts, precision=tp / (tp + fp) if tp + fp else 0,
                               recall=tp / (tp + fn) if tp + fn else 0, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {"frame_count": len(values), "center_matching": matching,
            "selected_center_count": sum(len(row["selected_center_coordinates_yx"]) for row in values),
            "gt_instance_count": sum(len(row["gt_center_coordinates_yx"]) for row in values),
            "offset_instance_mean_epe_pixels": distribution([error for row in values for error in row["offset_instance_mean_epe_pixels"]])}


def assemble_report(rows, protocol, binding):
    expected = [(scene, "{:04d}".format(frame)) for scene in protocol["validation_scenes"] for frame in protocol["frames"]]
    if [(row["identity"]["scene"], row["identity"]["frame"]) for row in rows] != expected:
        raise ValueError("No decision or report is allowed for a partial/unordered evaluation")
    bound = fingerprint(binding)
    arms, images = {}, []
    for row in rows:
        needs_image = row["identity"]["frame"] == protocol["visualization_frame"] and row["identity"]["scene"] in protocol["visualization_scenes"]
        verify_row(row, bound, row["identity"], needs_image)
        if row["image"]:
            images.append(dict(row["image"], scene=row["identity"]["scene"], frame=row["identity"]["frame"]))
    for arm in ARMS:
        partition = summarize_frames([row["arms"][arm]["metrics"] for row in rows], "confirmation scenes after first exposure become development validation")
        verify_partition(partition, protocol["validation_scenes"], protocol["frames"])
        arms[arm] = {"partition": partition,
                     "diagnostics": summarize_diagnostics([row["arms"][arm]["diagnostics"] for row in rows]),
                     "diagnostics_by_scene": {str(scene): summarize_diagnostics([row["arms"][arm]["diagnostics"] for row in rows if row["identity"]["scene"] == scene]) for scene in protocol["validation_scenes"]}}
    return {"schema": 6, "status": "completed", "frames_completed": len(rows), "binding_fingerprint": bound,
            "protocol_sha256": binding["protocol_sha256"], "new_training_updates": 0, "arms": arms,
            "decision": adoption({name: arms[name]["partition"] for name in ARMS}, protocol),
            "images": images, "frozen_checkpoints": binding["checkpoint_evidence"],
            "cache_payload_fingerprints": {cache_key(row["identity"]): fingerprint(row) for row in rows},
            "diagnostic_note": "Fixed5/10px maximum-cardinality center matching; offset error averages per-instance foreground pixels in320x240 inputs. Not a selection endpoint.",
            "exclusions": protocol["exclusions"]}


def verify_completed(directory, report_path, protocol, binding, records):
    bound = fingerprint(binding)
    if read(directory / "binding.json") != binding:
        raise ValueError("Completed cache binding differs")
    rows = []
    for record in records:
        required = record["frame"] == protocol["visualization_frame"] and record["scene"] in protocol["visualization_scenes"]
        rows.append(load_cached_frame(directory, bound, frame_identity(record), required, strict=True))
    report = read(report_path)
    if report != assemble_report(rows, protocol, binding):
        raise ValueError("Completed report differs from independently reconstructed frame evidence")
    return report


def run(config_path=DEFAULT_CONFIG, max_frames=None):
    config_path = project_path(config_path)
    protocol = read(config_path)
    validate_protocol(protocol)
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max-frames must be a positive integer")
    parent_path = project_path(protocol["parent_protocol"])
    if file_sha256(parent_path) != protocol["parent_protocol_sha256"]:
        raise ValueError("Pinned parent protocol hash changed")
    parent = read(parent_path)
    manifest, _audit = load_data(protocol)
    models, evidence = {}, {}
    for arm in ARMS:
        model, config, _parent_manifest, _result, bound = load_final_model(project_path(protocol["arms"][arm]["run_directory"]), parent)
        if config["arm"] != arm or bound["checkpoint_sha256"] != protocol["arms"][arm]["checkpoint_sha256"]:
            raise ValueError("Confirmation arm/checkpoint differs from frozen protocol")
        models[arm], evidence[arm] = model, bound
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen confirmation requires the same CUDA inference path")
    binding = build_binding(config_path, protocol, manifest, evidence)
    bound = fingerprint(binding)
    directory, report_path = project_path(protocol["output_directory"]), project_path(protocol["report"])
    directory.mkdir(parents=True, exist_ok=True)
    records = manifest["records"]["validation"]
    with _exclusive_lock(directory):
        if report_path.exists():
            report = verify_completed(directory, report_path, protocol, binding, records)
            print(json.dumps({"status": "verified_completed_read_only", "frames_completed": len(records), "decision": report["decision"]}, ensure_ascii=False), flush=True)
            return report
        binding_path = directory / "binding.json"
        if binding_path.exists():
            if read(binding_path) != binding:
                raise ValueError("Existing cache binding differs; use a new experiment version")
        elif any((directory / "frames").glob("*.json")):
            raise ValueError("Frame evidence exists without binding; refusing reconstruction")
        else:
            atomic_json(binding_path, binding)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dataset = GraspNetFrames(project_path(protocol["data_root"]), records, tuple(protocol["input_size"]), protocol["mode"], protocol["seed"], augment=False)
        rows, computed, skipped, moved = [], 0, 0, False
        for index, record in enumerate(records):
            identity = frame_identity(record)
            required = record["frame"] == protocol["visualization_frame"] and record["scene"] in protocol["visualization_scenes"]
            row = load_cached_frame(directory, bound, identity, required)
            if row is not None:
                rows.append(row)
                skipped += 1
                continue
            if max_frames is not None and computed >= max_frames:
                continue
            if shutil.disk_usage(ROOT).free < protocol["free_disk_reserve_gib"] * 2 ** 30 + 64 * 1024 * 1024:
                raise RuntimeError("Disk reserve would be violated")
            if not moved:
                for model in models.values():
                    model.to("cuda").eval()
                moved = True
            inputs, targets = dataset.get(index)
            arms, predictions = {}, {}
            with torch.no_grad():
                for arm in ARMS:
                    outputs = models[arm](inputs[None].to("cuda"))
                    arms[arm], predictions[arm] = analyze_outputs(outputs, targets, record, protocol["decoder"])
            image = draw_frame(directory, record, targets["labels"].numpy(), predictions, arms, protocol) if required else None
            row = {"schema": 6, "binding_fingerprint": bound, "identity": identity, "arms": arms, "image": image}
            verify_row(row, bound, identity, required)
            commit_frame(directory, row)
            # Verify the durable commit before including it in the completed partition.
            rows.append(load_cached_frame(directory, bound, identity, required, strict=True))
            computed += 1
            print(json.dumps({"committed": cache_key(identity), "completed": len(rows), "total": len(records)}, ensure_ascii=False), flush=True)
        invocation = {"schema": 6, "binding_fingerprint": bound, "newly_computed": computed, "skipped": skipped,
                      "total_completed": len(rows), "total_expected": len(records), "max_frames": max_frames,
                      "status": "completed" if len(rows) == len(records) else "partial", "new_training_updates": 0}
        atomic_json(directory / "last_invocation.json", invocation)
        if len(rows) != len(records):
            print(json.dumps(invocation, ensure_ascii=False), flush=True)
            return invocation
        report = assemble_report(rows, protocol, binding)
        atomic_json(report_path, report)
        verify_completed(directory, report_path, protocol, binding, records)
        print(json.dumps(report["decision"], ensure_ascii=False), flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int)
    options = parser.parse_args()
    run(options.config, options.max_frames)
