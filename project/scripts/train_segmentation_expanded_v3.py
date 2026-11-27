"""Fixed-budget RGB-D expansion; held-out images are evaluated only at completion."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _variable, _relative in {
    "TORCH_HOME": "runtime/cache/torch", "CUDA_CACHE_PATH": "runtime/cache/cuda",
    "XDG_CACHE_HOME": "runtime/cache", "TMP": "runtime/temp", "TEMP": "runtime/temp",
}.items():
    _directory = ROOT / _relative
    _directory.mkdir(parents=True, exist_ok=True)
    os.environ[_variable] = str(_directory)

import numpy as np
import scipy
import torch
import torchvision

from src.checkpoints import CheckpointStore, _exclusive_lock, atomic_json, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet, decode_instances, instance_counts, segmentation_loss, summarize_counts
from scripts.train_segmentation_pilot import prepare_run_files, restore_or_initialize_checkpoint


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All experiment paths must remain inside the project")
    return path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_expanded_v3.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="Absolute segment boundary; fixed plan is unchanged")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--crash-after-step", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Engineering recovery override, included in run fingerprint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def validate_protocol(protocol, args):
    expected = {"schema": 3, "seed": 20261021, "mode": "rgbd", "input_size": [240, 320],
                "batch_size": 1, "accumulation": 2, "planned_steps": 1600, "equivalent_epochs": 20,
                "augment": True, "learning_rate": 0.001, "weight_decay": 0.0001,
                "warmup_steps": 80, "min_learning_rate": 0.00001, "amp": True,
                "gradient_clip_norm": 1.0, "save_every_steps": 10, "keep_last": 3,
                "free_disk_reserve_gib": 60,
                "train_scenes": [3, 12, 30, 31, 34, 38, 42, 43, 57, 93],
                "validation_scenes": [13, 18, 22, 33, 39],
                "fresh_validation_scenes": [18, 22, 33, 39], "previously_seen_validation_scenes": [13],
                "frames": list(range(0, 256, 16)),
                "decoder": {"foreground_threshold": 0.5, "center_threshold": 0.35, "min_pixels": 20, "max_centers": 64},
                "acceptance": {"fresh_scene_macro_foreground_iou_min": 0.70,
                               "fresh_scene_macro_instance_f1_min": 0.50, "minimum_tp_per_fresh_scene": 1}}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError("This frozen expansion requires the declared value of " + key)
    for name in ("max_steps", "crash_after_step"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= protocol["planned_steps"]:
            raise ValueError(name + " must be within the fixed 1600-step plan")
    if args.checkpoint_interval is not None and not 1 <= args.checkpoint_interval <= protocol["save_every_steps"]:
        raise ValueError("Engineering checkpoint interval must be between 1 and 10")
    if project_path(args.run_dir) == ROOT:
        raise ValueError("Run directory must not be the project root")


def ensure_disk_reserve(reserve_gib):
    if shutil.disk_usage(ROOT).free / 2 ** 30 < reserve_gib + 1:
        raise RuntimeError("Cannot preserve the free-disk reserve plus a 1 GiB atomic-checkpoint margin")


def load_manifest(protocol):
    """Check hashes of both partitions, but do not decode any held-out image."""
    data_root = project_path(protocol["data_root"])
    manifest_path = project_path(protocol["data_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = file_sha256(manifest_path)
    if manifest.get("schema") != 3 or manifest.get("status") != "verified_expanded_sample":
        raise ValueError("Expanded data must have completed the verified_expanded_sample audit")
    if project_path(manifest["data_root"]) != data_root:
        raise ValueError("Manifest data_root differs from the declared protocol root")
    audit_path = project_path(protocol.get("data_audit", "results/P2/expanded_v3_data_audit.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_audit = {"status": "passed", "manifest_sha256": manifest_hash,
                      "split_sha256": manifest["split_sha256"], "files_verified": 1020,
                      "train_frames": 160, "validation_frames": 80,
                      "fresh_validation_frames": 64, "previously_seen_validation_frames": 16}
    if any(audit.get(key) != value for key, value in expected_audit.items()) or audit.get("errors") or audit.get("hash_failures"):
        raise ValueError("Completed data audit is missing, failed, or does not bind this exact manifest")
    if project_path(audit["data_root"]) != data_root:
        raise ValueError("Audit data_root differs from the declared protocol root")
    split_path = project_path(protocol["split"])
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest["split_sha256"] != file_sha256(split_path):
        raise ValueError("Expanded manifest split hash differs from the frozen scene split")
    if set(manifest["records"]) != {"train", "validation"}:
        raise ValueError("Expanded manifest must contain train and validation only")
    train_scenes, validation_scenes = set(protocol["train_scenes"]), set(protocol["validation_scenes"])
    if train_scenes & validation_scenes:
        raise ValueError("Training and validation scenes overlap")
    for group, scenes in (("train", train_scenes), ("validation", validation_scenes)):
        if not scenes.issubset(splits["groups"][group]) or not scenes.issubset(splits["pilot_" + group]):
            raise ValueError("Scene not in the declared " + group + " partition")
        for other, other_scenes in splits["groups"].items():
            if other != group and scenes & set(other_scenes):
                raise ValueError("Scene belongs to more than one split")
        records = manifest["records"][group]
        expected = {(scene, "{:04d}".format(frame)) for scene in scenes for frame in protocol["frames"]}
        actual = {(row["scene"], row["frame"]) for row in records}
        if len(records) != len(expected) or actual != expected:
            raise ValueError("Expanded " + group + " must contain every declared frame exactly once")
        for row in records:
            if row["camera"] != "realsense" or set(row["files"]) != {"rgb", "depth", "label", "meta"}:
                raise ValueError("Expanded records require realsense RGB, depth, label, and metadata")
            for kind, relative in row["files"].items():
                expected_relative = "scenes/scene_{:04d}/realsense/{}/{}{}".format(
                    row["scene"], kind, row["frame"], ".mat" if kind == "meta" else ".png")
                if relative.replace("\\", "/") != expected_relative:
                    raise ValueError("Record file path does not match its declared scene, frame, camera, and modality")
                path = (data_root / relative).resolve()
                if not path.is_relative_to(data_root) or not path.is_file():
                    raise ValueError("Missing or out-of-root " + group + " file")
                if file_sha256(path) != row["sha256"][kind]:
                    raise ValueError("Expanded file checksum mismatch")
    # Bind raw provenance and deterministic record order as well as all file hashes.
    return {"schema": 3, "data_root": str(data_root), "records": manifest["records"],
            "split_sha256": manifest["split_sha256"], "source_manifest_sha256": manifest_hash,
            "verified_audit_binding": expected_audit,
            "validation_policy": "Byte hashes checked on entry; images/labels decoded only after optimizer step 1600",
            "source_manifest": manifest}, data_root


def learning_rate_for_update(update, protocol):
    """One-based actual update LR: warmup 1..80, cosine 81..1600, clamped after plan."""
    if update < 1:
        raise ValueError("Update indices are one-based")
    step = min(update, protocol["planned_steps"])
    base, minimum, warmup = protocol["learning_rate"], protocol["min_learning_rate"], protocol["warmup_steps"]
    if step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / (protocol["planned_steps"] - warmup)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * progress))


def build_config(protocol, config_path, manifest, args):
    source_files = ("scripts/train_segmentation_expanded_v3.py", "scripts/train_segmentation_pilot.py",
                    "src/segmentation.py", "src/graspnet_frames.py", "src/checkpoints.py")
    return {"schema": 3, "protocol": protocol, "protocol_sha256": file_sha256(config_path),
            "data_fingerprint": fingerprint(manifest),
            "source_sha256": {name: file_sha256(ROOT / name) for name in source_files},
            "device": args.device, "effective_amp": protocol["amp"] and args.device == "cuda",
            "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU component verification",
            "save_every_optimizer_steps": args.checkpoint_interval or protocol["save_every_steps"],
            "engineering_checkpoint_override": args.checkpoint_interval,
            "amp_initial_scale": 128.0, "amp_max_overflow_retries": 16,
            "num_workers": 0, "deterministic_algorithms": True,
            "scheduler_index_rule": "LambdaLR index k describes next update k+1; final next LR clamped to minimum",
            "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                         "torchvision": str(torchvision.__version__), "numpy": np.__version__, "scipy": scipy.__version__}}


def summarize_frames(rows, scope):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation partition")
    totals = {key: sum(row["instances"][key] for row in rows) for key in ("tp", "fp", "fn")}
    intersection = sum(row["foreground_intersection"] for row in rows)
    union = sum(row["foreground_union"] for row in rows)
    scenes = []
    for scene in sorted({row["scene"] for row in rows}):
        subset = [row for row in rows if row["scene"] == scene]
        counts = {key: sum(row["instances"][key] for row in subset) for key in ("tp", "fp", "fn")}
        scene_i = sum(row["foreground_intersection"] for row in subset)
        scene_u = sum(row["foreground_union"] for row in subset)
        scenes.append({"scene": scene, "frame_count": len(subset), "foreground_intersection": scene_i,
                       "foreground_union": scene_u, "foreground_pixel_iou": scene_i / scene_u if scene_u else 1.0,
                       "instances": summarize_counts(counts)})
    return {"scope": scope, "frame_count": len(rows), "scene_count": len(scenes),
            "frames": rows, "scenes": scenes, "foreground_intersection": intersection, "foreground_union": union,
            "foreground_pixel_iou": intersection / union if union else 1.0, "instances": summarize_counts(totals),
            "scene_macro": {"foreground_pixel_iou": sum(row["foreground_pixel_iou"] for row in scenes) / len(scenes),
                            "instance_f1_iou50": sum(row["instances"]["f1_iou50"] for row in scenes) / len(scenes)},
            "metric_definition": "Mask IoU >= 0.50 maximum-cardinality one-to-one matching; micro F1 is not AP; scene macro averages scene-aggregated metrics equally"}


def evaluate(model, dataset, device, decoder, scope):
    model.eval()
    rows = []
    with torch.no_grad():
        for index, record in enumerate(dataset.records):
            inputs, targets = dataset.get(index)
            outputs = model(inputs[None].to(device))
            labels = targets["labels"].numpy()
            predicted = decode_instances(outputs, **decoder)
            counts = instance_counts(predicted, labels)
            foreground = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= decoder["foreground_threshold"]
            actual = labels > 0
            intersection, union = int((foreground & actual).sum()), int((foreground | actual).sum())
            rows.append({"scene": record["scene"], "frame": record["frame"],
                         "foreground_intersection": intersection, "foreground_union": union,
                         "foreground_pixel_iou": intersection / union if union else 1.0,
                         "instances": summarize_counts(counts), "predicted_instances": counts["tp"] + counts["fp"],
                         "actual_instances": counts["tp"] + counts["fn"],
                         "max_center_probability": float(outputs["centers"].sigmoid().max().cpu())})
    model.train()
    return summarize_frames(rows, scope)


def acceptance_gate(fresh, protocol):
    threshold = protocol["acceptance"]
    checks = {
        "fresh_scene_macro_foreground_iou": fresh["scene_macro"]["foreground_pixel_iou"] >= threshold["fresh_scene_macro_foreground_iou_min"],
        "fresh_scene_macro_instance_f1": fresh["scene_macro"]["instance_f1_iou50"] >= threshold["fresh_scene_macro_instance_f1_min"],
        "every_fresh_scene_has_true_positive": {row["scene"] for row in fresh["scenes"]} == set(protocol["fresh_validation_scenes"]) and all(
            row["instances"]["tp"] >= threshold["minimum_tp_per_fresh_scene"] for row in fresh["scenes"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": threshold,
            "interpretation": "Prespecified engineering continuation gate only, not a publication criterion or method-superiority claim"}


def final_evaluation(model, data_root, manifest, device, protocol):
    common = {"size": tuple(protocol["input_size"]), "mode": protocol["mode"], "seed": protocol["seed"], "augment": False}
    train = GraspNetFrames(data_root, manifest["records"]["train"], **common)
    validation = GraspNetFrames(data_root, manifest["records"]["validation"], **common)
    results = {"train": evaluate(model, train, device, protocol["decoder"], "160 training frames, no augmentation"),
               "validation": evaluate(model, validation, device, protocol["decoder"], "80 validation frames, fixed final checkpoint")}
    rows = results["validation"]["frames"]
    for name, scenes, scope in (
        ("fresh_validation", protocol["fresh_validation_scenes"], "64 frames from scenes not previously inspected for segmentation; after this evaluation they are development validation"),
        ("previously_seen_validation", protocol["previously_seen_validation_scenes"], "All 16 frames of scene 13; this entire scene is previously seen"),
    ):
        results[name] = summarize_frames([row for row in rows if row["scene"] in scenes], scope)
    results["acceptance_gate"] = acceptance_gate(results["fresh_validation"], protocol)
    return results


def run(args):
    config_path = project_path(args.config)
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    validate_protocol(protocol, args)
    ensure_disk_reserve(protocol["free_disk_reserve_gib"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the isolated vision interpreter")
    manifest, data_root = load_manifest(protocol)
    train_data = GraspNetFrames(data_root, manifest["records"]["train"], size=tuple(protocol["input_size"]),
                               mode=protocol["mode"], seed=protocol["seed"], augment=protocol["augment"])
    device = torch.device(args.device)
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    if args.device == "cuda":
        torch.cuda.manual_seed_all(protocol["seed"])
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = build_config(protocol, config_path, manifest, args)
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    run_directory = project_path(args.run_dir)
    run_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(run_directory):
        prepare_run_files(run_directory, config, manifest, args.resume)
        model = PilotInstanceNet(mode=protocol["mode"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed: learning_rate_for_update(completed + 1, protocol) / protocol["learning_rate"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["effective_amp"], init_scale=config["amp_initial_scale"])
        store = CheckpointStore(run_directory / "checkpoints", config_hash, data_hash, keep_last=protocol["keep_last"])
        restored = restore_or_initialize_checkpoint(run_directory, store, model, optimizer, scheduler, scaler, config)
        epoch, next_batch, global_step = (restored["cursor"][key] for key in ("epoch", "next_batch", "global_step"))
        history = restored["extra"]["history"]
        if [row["step"] for row in history] != list(range(1, global_step + 1)) or restored["extra"].get("config") != config:
            raise RuntimeError("Checkpoint history or configuration disagrees with the recorded state")
        accumulation = protocol["accumulation"]
        if not 0 <= next_batch < len(train_data) or next_batch % accumulation:
            raise RuntimeError("Checkpoint batch cursor is not an optimizer boundary")
        if global_step != epoch * math.ceil(len(train_data) / accumulation) + next_batch // accumulation:
            raise RuntimeError("Checkpoint step and epoch/frame cursor disagree")
        expected_lr = protocol["learning_rate"] * (learning_rate_for_update(global_step + 1, protocol) / protocol["learning_rate"])
        if scheduler.last_epoch != global_step or optimizer.param_groups[0]["lr"] != expected_lr or scheduler.get_last_lr()[0] != expected_lr:
            raise RuntimeError("Checkpoint scheduler disagrees with the next actual-update learning rate")
        limit = args.max_steps if args.max_steps is not None else protocol["planned_steps"]
        if global_step > limit:
            raise ValueError("Segment boundary precedes the committed checkpoint")
        start_step = global_step
        checkpoint_path = restored["checkpoint_path"]
        atomic_json(run_directory / "progress.json", {"status": "running", "step": global_step, "target_step": limit,
                    "planned_steps": protocol["planned_steps"], "config_fingerprint": config_hash, "data_fingerprint": data_hash})
        atomic_json(run_directory / "history.json", history)
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while global_step < limit:
            order = train_data.epoch_order(epoch)
            selected = order[next_batch:next_batch + accumulation]
            attempts = 0
            used_lr = optimizer.param_groups[0]["lr"]
            while True:
                accumulated_loss, parts = 0.0, {}
                for index in selected:
                    inputs, targets = train_data.get(index, epoch=epoch)
                    targets = {key: value[None].to(device) for key, value in targets.items() if key != "labels"}
                    with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=config["effective_amp"]):
                        outputs = model(inputs[None].to(device))
                        loss, components = segmentation_loss(outputs, targets)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite loss; previous committed checkpoints preserved")
                    scaler.scale(loss / len(selected)).backward()
                    accumulated_loss += float(loss.detach().cpu()) / len(selected)
                    for name, value in components.items():
                        parts[name] = parts.get(name, 0.0) + float(value.cpu()) / len(selected)
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"])
                old_scale = scaler.get_scale()
                if not config["effective_amp"] and not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("Nonfinite gradient; previous committed checkpoints preserved")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    if not bool(torch.isfinite(gradient_norm)):
                        raise RuntimeError("Accepted optimizer step has nonfinite gradient norm")
                    break
                attempts += 1
                if attempts >= config["amp_max_overflow_retries"]:
                    raise RuntimeError("Repeated AMP overflow; no skipped updates are counted")
            scheduler.step()
            global_step += 1
            finished_epoch = epoch
            next_batch += len(selected)
            if next_batch == len(order):
                epoch, next_batch = epoch + 1, 0
            row = {"step": global_step, "epoch": finished_epoch, "loss": accumulated_loss, "components": parts,
                   "learning_rate_used": used_lr, "learning_rate_next_update": scheduler.get_last_lr()[0],
                   "next_update_index": global_step + 1 if global_step < protocol["planned_steps"] else None,
                   "next_update_planned": global_step < protocol["planned_steps"],
                   "gradient_norm_before_clip": float(gradient_norm.detach().cpu()), "amp_retries": attempts,
                   "amp_scale": scaler.get_scale(), "samples": [
                       {"scene": train_data.records[index]["scene"], "frame": train_data.records[index]["frame"]} for index in selected]}
            history.append(row)
            should_save = global_step % config["save_every_optimizer_steps"] == 0 or global_step == limit
            if should_save:
                ensure_disk_reserve(protocol["free_disk_reserve_gib"])
                saved = store.save(model, optimizer, epoch, next_batch, global_step, True, scheduler=scheduler, scaler=scaler,
                                   extra={"history": history, "config": config, "data_fingerprint": data_hash})
                checkpoint_path = str(store.directory / saved["file"])
                atomic_json(run_directory / "history.json", history)
                atomic_json(run_directory / "progress.json", {"status": "running", "step": global_step, "target_step": limit,
                            "epoch": epoch, "next_batch": next_batch, "planned_steps": protocol["planned_steps"],
                            "config_fingerprint": config_hash, "data_fingerprint": data_hash})
            print(json.dumps(dict(row, checkpoint_committed=should_save)), flush=True)
            if args.crash_after_step == global_step:
                os._exit(37)
        complete = global_step == protocol["planned_steps"]
        # Never trust an existing result blindly: state integrity was checked above.
        # At a completed checkpoint deterministic evaluation may repeat after a crash.
        assessments = final_evaluation(model, data_root, manifest, device, protocol) if complete else None
        report = {"status": "completed" if complete else "step_limit_reached",
                  "config_fingerprint": config_hash, "data_fingerprint": data_hash,
                  "optimizer_steps": global_step, "planned_steps": protocol["planned_steps"],
                  "resumed_from_step": start_step, "updates_this_invocation": global_step - start_step,
                  "cursor": {"epoch": epoch, "next_batch": next_batch, "global_step": global_step},
                  "checkpoint_path": checkpoint_path, "checkpoint_sha256": file_sha256(Path(checkpoint_path)),
                  "evaluation": assessments, "validation_evaluated": complete,
                  "decision": ("engineering_continuation_gate_passed" if assessments["acceptance_gate"]["passed"] else "engineering_continuation_gate_not_met") if complete else "partial_plan_no_validation_evaluation",
                  "automatic_extension": False, "elapsed_seconds_this_invocation": time.perf_counter() - started,
                  "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
                  "latest_loss": history[-1]["loss"] if history else None,
                  "rejected_checkpoint_generations": restored.get("rejected_generations", []),
                  "latest_pointer_problem": restored.get("latest_pointer_problem"),
                  "limitations": ["One seed and four newly inspected validation scenes cannot establish method superiority",
                                  "Fresh validation becomes development validation after this evaluation",
                                  "All 16 frames of previously inspected scene 13 are reported separately",
                                  "No early stopping, best-checkpoint selection, decoder tuning, or automatic extension",
                                  "Micro F1 at mask IoU 0.50 is not mask AP",
                                  "No calibration-set, official test-set, grasp AP, or robot execution evaluation",
                                  "Process interruption verification is not a physical power-loss test"]}
        atomic_json(run_directory / "result.json", report)
        atomic_json(run_directory / "progress.json", {key: report[key] for key in (
            "status", "optimizer_steps", "planned_steps", "cursor", "config_fingerprint", "data_fingerprint", "decision")})
        print(json.dumps({"status": report["status"], "decision": report["decision"], "optimizer_steps": global_step,
                          "result": str(run_directory / "result.json")}, ensure_ascii=False), flush=True)
        return report


if __name__ == "__main__":
    try:
        run(arguments())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("Training stopped: {}".format(error), file=sys.stderr, flush=True)
        sys.exit(2)
