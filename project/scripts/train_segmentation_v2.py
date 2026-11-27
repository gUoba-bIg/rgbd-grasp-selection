"""Frozen four-frame fitting diagnostic; not a generalization experiment."""

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
for _name, _relative in {
    "TORCH_HOME": "runtime/cache/torch", "CUDA_CACHE_PATH": "runtime/cache/cuda",
    "XDG_CACHE_HOME": "runtime/cache", "TMP": "runtime/temp", "TEMP": "runtime/temp",
}.items():
    _folder = ROOT / _relative
    _folder.mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_folder)

import numpy as np
import scipy
import torch
import torchvision

from src.checkpoints import CheckpointStore, _exclusive_lock, atomic_json, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames
from src.segmentation import (PilotInstanceNet, decode_instances, instance_counts,
                              segmentation_loss as original_loss, summarize_counts)
from scripts.train_segmentation_pilot import (prepare_run_files,
                                               restore_or_initialize_checkpoint)


EXPECTED_FRAMES = {(3, "0000"), (3, "0016"), (12, "0000"), (12, "0016")}
FREE_RESERVE_GIB = 60.0
FIT_THRESHOLDS = {"foreground_pixel_iou_min": 0.90, "instance_f1_iou50_min": 0.80,
                  "true_positives_per_frame_min": 1}


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All diagnostic paths must stay within the project")
    return path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default="experiments/segmentation_diagnostics/rgbd_overfit_v1/data_manifest.json")
    parser.add_argument("--split", default="data/manifests/scene_splits.json")
    parser.add_argument("--data-root", default="data/raw/graspnet")
    parser.add_argument("--mode", choices=("rgb", "rgbd"), default="rgbd")
    parser.add_argument("--steps", type=int, default=400, help="Fixed plan; immutable on resume")
    parser.add_argument("--max-steps", type=int, default=None, help="Absolute stopping boundary, at most --steps")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--loss", choices=("original", "center_mse"), default="original")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=10)
    parser.add_argument("--crash-after-step", type=int, default=None)
    return parser.parse_args(argv)


def validate_arguments(args):
    if not 1 <= args.steps <= 10000:
        raise ValueError("steps must be in [1, 10000]; the plan is immutable on resume")
    if args.max_steps is not None and not 1 <= args.max_steps <= args.steps:
        raise ValueError("max-steps must be positive and cannot exceed the fixed step plan")
    if args.crash_after_step is not None and not 1 <= args.crash_after_step <= args.steps:
        raise ValueError("crash-after-step must be inside the fixed step plan")
    if not 1 <= args.save_every_steps <= 100:
        raise ValueError("save-every-steps must be in [1, 100]")
    if not 1 <= args.accumulation <= 4:
        raise ValueError("accumulation must be in [1, 4] for the four-frame diagnostic")
    if min(args.height, args.width) < 64 or max(args.height, args.width) > 640:
        raise ValueError("Each input dimension must be in [64, 640]")
    if args.height * args.width > 640 * 480:
        raise ValueError("Diagnostic input area cannot exceed 640 * 480")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01:
        raise ValueError("learning-rate must be finite and in (0, 0.01]")
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.amp and args.device != "cuda":
        raise ValueError("AMP requires CUDA")
    if project_path(args.run_dir) == ROOT:
        raise ValueError("run-dir must be a subdirectory, not the project root")


def ensure_disk_reserve():
    free = shutil.disk_usage(ROOT).free / 2 ** 30
    # Keep the reserve after a worst-case 1 GiB atomic checkpoint write too.
    if free < FREE_RESERVE_GIB + 1.0:
        raise RuntimeError("Insufficient disk space to preserve the 60 GiB reserve and write a checkpoint")
    return free


def load_training_manifest(manifest_path, split_path, data_root):
    """Validate frozen training identities and bytes without opening held-out files."""
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if previous["split_sha256"] != file_sha256(split_path):
        raise ValueError("Frozen manifest was produced with a different split")
    records = previous["records"]["train"]
    identities = {(record["scene"], record["frame"]) for record in records}
    if len(records) != 4 or identities != EXPECTED_FRAMES:
        raise ValueError("Expected the previously declared four training frames, without substitution")
    allowed = set(splits["groups"]["train"]) & set(splits["pilot_train"])
    forbidden = set(splits["groups"]["validation"]) | set(splits.get("pilot_validation", []))
    for name, scenes in splits["groups"].items():
        if name != "train":
            forbidden.update(scenes)
    for record in records:
        if record["scene"] not in allowed or record["scene"] in forbidden:
            raise ValueError("Training scene is not exclusively in the declared training split")
        if record["camera"] != "realsense":
            raise ValueError("The frozen four-frame diagnostic uses realsense only")
        if set(record["files"]) != {"rgb", "depth", "label", "meta"}:
            raise ValueError("Training record does not contain the four required modalities")
        for kind, relative in record["files"].items():
            path = (data_root / relative).resolve()
            if not path.is_relative_to(data_root) or not path.is_file():
                raise ValueError("Training file missing or outside data root")
            if file_sha256(path) != record["sha256"][kind]:
                raise ValueError("Training file checksum differs from the frozen manifest")
    # Validation records are provenance only. Their image/depth/label/meta paths
    # are neither opened nor hashed, and no validation dataset is instantiated.
    return {"schema": 2, "data_root": str(data_root), "records": previous["records"],
            "split_sha256": previous["split_sha256"],
            "parent_manifest_sha256": file_sha256(manifest_path),
            "training_frame_count": 4, "validation_use": "excluded from loading, training, evaluation and selection",
            "scope": "Frozen four-frame training fit diagnostic, not a held-out assessment"}


def source_hashes(loss_name):
    names = ["scripts/train_segmentation_v2.py", "scripts/train_segmentation_pilot.py",
             "src/segmentation.py", "src/graspnet_frames.py", "src/checkpoints.py"]
    if loss_name == "center_mse":
        names.append("src/segmentation_v2.py")
    elif loss_name != "original":
        raise ValueError("Unknown loss")
    return {name: file_sha256(ROOT / name) for name in names}


def select_loss(name):
    if name == "original":
        return original_loss
    if name == "center_mse":
        from src.segmentation_v2 import segmentation_loss
        return segmentation_loss
    raise ValueError("Unknown loss")


def fitting_gate(metrics):
    checks = {"foreground_iou": metrics["foreground_pixel_iou"] >= FIT_THRESHOLDS["foreground_pixel_iou_min"],
              "instance_f1": metrics["instances"]["f1_iou50"] >= FIT_THRESHOLDS["instance_f1_iou50_min"],
              "all_frames_have_true_positive": len(metrics["frames"]) == 4 and all(
                  row["instances"]["tp"] >= FIT_THRESHOLDS["true_positives_per_frame_min"] for row in metrics["frames"])}
    return {"passed": all(checks.values()), "checks": checks, "thresholds": dict(FIT_THRESHOLDS),
            "interpretation": "Training fit only; passing does not demonstrate generalization or method improvement"}


def evaluate_training(model, dataset, device, decoder):
    """One final training-set evaluation; all masks are decoded without labels."""
    model.eval()
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    intersection = union = 0
    with torch.no_grad():
        for index, record in enumerate(dataset.records):
            inputs, targets = dataset.get(index)
            outputs = model(inputs[None].to(device))
            predicted = decode_instances(outputs, **decoder)
            labels = targets["labels"].numpy()
            counts = instance_counts(predicted, labels)
            foreground = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= decoder["foreground_threshold"]
            actual = labels > 0
            frame_intersection = int((foreground & actual).sum())
            frame_union = int((foreground | actual).sum())
            intersection += frame_intersection
            union += frame_union
            for key in totals:
                totals[key] += counts[key]
            rows.append({"scene": record["scene"], "frame": record["frame"],
                         "foreground_intersection": frame_intersection, "foreground_union": frame_union,
                         "foreground_pixel_iou": frame_intersection / frame_union if frame_union else 1.0,
                         "max_center_probability": float(outputs["centers"].sigmoid().max().cpu()),
                         "predicted_instances": counts["tp"] + counts["fp"],
                         "actual_instances": counts["tp"] + counts["fn"],
                         "instances": summarize_counts(counts)})
    model.train()
    return {"frames": rows, "frame_count": len(rows), "scene_count": len({row["scene"] for row in rows}),
            "foreground_pixel_iou": intersection / union if union else 1.0,
            "foreground_intersection": intersection, "foreground_union": union,
            "instances": summarize_counts(totals),
            "metric_definition": "Micro precision/recall/F1 at mask IoU >= 0.50 using maximum-cardinality one-to-one matching; not mask AP",
            "scope": "The same four training frames, without augmentation; not held-out validation"}


def build_config(args, manifest):
    return {
        "schema": 2, "scope": "Four-frame training fitting diagnostic; not the formal training protocol",
        "architecture": "ResNet18-GroupNorm-FPN48, early fusion, foreground-center-offset heads",
        "weights": "random initialization; torchvision weights=None", "mode": args.mode,
        "input_size": [args.height, args.width], "batch_size": 1, "accumulation": args.accumulation,
        "planned_steps": args.steps, "seed": args.seed, "num_workers": 0,
        "augment": args.augment, "augmentation": "sample-seeded horizontal flip and RGB brightness [0.9,1.1]" if args.augment else "disabled",
        "depth": "meta.factor_depth to meters; clip at 2m and divide by 2; separate validity channel",
        "loss_name": args.loss,
        "loss_formula": "foreground BCE + Dice + Gaussian center focal + 5 * foreground-only SmoothL1(beta=0.05)" if args.loss == "original" else
                        "foreground BCE + Dice + 200 * mean((sigmoid(center_logits)-Gaussian_target)^2) + 5 * foreground-only SmoothL1(beta=0.05)",
        "loss_variant_scope": "Original pilot control" if args.loss == "original" else
                              "Independent Gaussian-center regression diagnostic; not a Panoptic-DeepLab reproduction",
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 0.0001,
        "schedule": args.schedule, "schedule_plan_steps": args.steps,
        "gradient_clip_norm": 1.0, "device": args.device, "amp": args.amp,
        "amp_initial_scale": 128.0, "amp_max_overflow_retries": 16,
        "deterministic_algorithms": True, "save_every_optimizer_steps": args.save_every_steps,
        "free_disk_reserve_gib": FREE_RESERVE_GIB,
        "decoder": {"foreground_threshold": 0.5, "center_threshold": 0.35, "min_pixels": 20, "max_centers": 64},
        "training_fit_thresholds": dict(FIT_THRESHOLDS),
        "selection": "Fixed final-step checkpoint only; validation excluded; no automatic plan extension",
        "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__), "numpy": np.__version__, "scipy": scipy.__version__},
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "source_sha256": source_hashes(args.loss),
        "parent_protocol_sha256": file_sha256(ROOT / "configs/training_protocol.json"),
        "data_fingerprint": fingerprint(manifest),
    }


def run(args):
    validate_arguments(args)
    ensure_disk_reserve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the project's isolated interpreter")
    run_directory = project_path(args.run_dir)
    data_root = project_path(args.data_root)
    manifest = load_training_manifest(project_path(args.manifest), project_path(args.split), data_root)
    train_data = GraspNetFrames(data_root, manifest["records"]["train"], size=(args.height, args.width),
                               mode=args.mode, seed=args.seed, augment=args.augment)
    fit_data = GraspNetFrames(data_root, manifest["records"]["train"], size=(args.height, args.width),
                             mode=args.mode, seed=args.seed, augment=False)
    for index in range(len(fit_data)):
        fit_data.get(index)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = build_config(args, manifest)
    loss_function = select_loss(args.loss)
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    run_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(run_directory):
        prepare_run_files(run_directory, config, manifest, args.resume)
        model = PilotInstanceNet(mode=args.mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
                     if args.schedule == "cosine" else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0))
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=config["amp_initial_scale"])
        store = CheckpointStore(run_directory / "checkpoints", config_hash, data_hash, keep_last=3)
        restored = restore_or_initialize_checkpoint(run_directory, store, model, optimizer, scheduler, scaler, config)
        epoch, next_batch, global_step = (restored["cursor"][key] for key in ("epoch", "next_batch", "global_step"))
        history = restored["extra"]["history"]
        if [row["step"] for row in history] != list(range(1, global_step + 1)):
            raise RuntimeError("Checkpoint history and step cursor disagree")
        if restored["extra"].get("config") != config:
            raise RuntimeError("Checkpoint configuration is incomplete or changed")
        limit = args.max_steps if args.max_steps is not None else args.steps
        if global_step > limit:
            raise ValueError("Requested stopping boundary precedes the committed optimizer step")
        if global_step != epoch * math.ceil(len(train_data) / args.accumulation) + next_batch // args.accumulation:
            raise RuntimeError("Checkpoint step count and frame cursor disagree")
        start_step = global_step
        atomic_json(run_directory / "progress.json", {"status": "running", "step": global_step,
                    "target_step": limit, "planned_steps": args.steps, "config_fingerprint": config_hash,
                    "data_fingerprint": data_hash, "resumed_from": restored["checkpoint_path"]})
        atomic_json(run_directory / "history.json", history)
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while global_step < limit:
            order = train_data.epoch_order(epoch)
            if not 0 <= next_batch < len(order) or next_batch % args.accumulation:
                raise RuntimeError("Saved batch cursor is not a valid accumulation boundary")
            selected = order[next_batch:next_batch + args.accumulation]
            attempts = 0
            learning_rate = optimizer.param_groups[0]["lr"]
            while True:
                accumulated_loss, parts = 0.0, {}
                for index in selected:
                    inputs, targets = train_data.get(index, epoch=epoch)
                    targets = {key: value[None].to(device) for key, value in targets.items() if key != "labels"}
                    with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.amp):
                        outputs = model(inputs[None].to(device))
                        loss, components = loss_function(outputs, targets)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite loss; committed checkpoint preserved")
                    scaler.scale(loss / len(selected)).backward()
                    accumulated_loss += float(loss.detach().cpu()) / len(selected)
                    for key, value in components.items():
                        parts[key] = parts.get(key, 0.0) + float(value.cpu()) / len(selected)
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                old_scale = scaler.get_scale()
                if not args.amp and not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("Nonfinite gradients; committed checkpoint preserved")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    if not bool(torch.isfinite(gradient_norm)):
                        raise RuntimeError("Nonfinite gradient norm despite an accepted optimizer step")
                    break
                attempts += 1
                if attempts >= config["amp_max_overflow_retries"]:
                    raise RuntimeError("Repeated AMP overflow; no skipped update is counted")
            scheduler.step()
            global_step += 1
            finished_epoch = epoch
            next_batch += len(selected)
            if next_batch == len(order):
                epoch, next_batch = epoch + 1, 0
            row = {"step": global_step, "epoch": finished_epoch, "loss": accumulated_loss,
                   "components": parts, "learning_rate_before_step": learning_rate,
                   "learning_rate_after_step": scheduler.get_last_lr()[0],
                   "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                   "samples": [{"scene": train_data.records[index]["scene"], "frame": train_data.records[index]["frame"]} for index in selected],
                   "amp_retries": attempts, "amp_scale": scaler.get_scale()}
            history.append(row)
            should_save = global_step % args.save_every_steps == 0 or global_step == limit
            if should_save:
                ensure_disk_reserve()
                store.save(model, optimizer, epoch, next_batch, global_step, True, scheduler=scheduler, scaler=scaler,
                           extra={"history": history, "config": config, "data_fingerprint": data_hash})
                atomic_json(run_directory / "history.json", history)
                atomic_json(run_directory / "progress.json", {"status": "running", "step": global_step,
                            "target_step": limit, "planned_steps": args.steps, "epoch": epoch, "next_batch": next_batch,
                            "config_fingerprint": config_hash, "data_fingerprint": data_hash})
            print(json.dumps(dict(row, checkpoint_committed=should_save)), flush=True)
            if args.crash_after_step == global_step:
                os._exit(37)
        training_fit = evaluate_training(model, fit_data, device, config["decoder"])
        gate = fitting_gate(training_fit)
        complete = global_step == args.steps
        report = {"status": "completed" if complete else "step_limit_reached",
                  "config_fingerprint": config_hash, "data_fingerprint": data_hash,
                  "optimizer_steps": global_step, "planned_steps": args.steps,
                  "resumed_from_step": start_step, "updates_this_invocation": global_step - start_step,
                  "cursor": {"epoch": epoch, "next_batch": next_batch, "global_step": global_step},
                  "training_fit": training_fit, "fitting_gate": gate,
                  "decision": ("training_fit_gate_passed" if gate["passed"] else "training_fit_gate_not_met") if complete else "partial_plan_not_final_acceptance",
                  "validation_evaluated": False, "automatic_extension": False,
                  "elapsed_seconds_this_invocation": time.perf_counter() - started,
                  "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
                  "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "latest_loss": history[-1]["loss"] if history else None,
                  "rejected_checkpoint_generations": restored.get("rejected_generations", []),
                  "latest_pointer_problem": restored.get("latest_pointer_problem"),
                  "limitations": ["Four-frame training fitting only; no generalization or method-improvement claim",
                                  "Constant LR and disabled augmentation are diagnostic settings, not the formal training protocol",
                                  "No validation or test files loaded, no selection using held-out labels",
                                  "Micro F1 at mask IoU >= 0.50 is not mask AP",
                                  "No grasp AP or robot execution evaluated",
                                  "Process crash recovery does not simulate electrical power loss"]}
        atomic_json(run_directory / "result.json", report)
        atomic_json(run_directory / "progress.json", {key: report[key] for key in (
            "status", "optimizer_steps", "planned_steps", "cursor", "config_fingerprint", "data_fingerprint", "decision")})
        print(json.dumps({"status": report["status"], "decision": report["decision"],
                          "result": str(run_directory / "result.json"), "optimizer_steps": global_step}, ensure_ascii=False), flush=True)
        return report


if __name__ == "__main__":
    try:
        run(arguments())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("Training stopped: {}".format(error), file=sys.stderr, flush=True)
        sys.exit(2)
