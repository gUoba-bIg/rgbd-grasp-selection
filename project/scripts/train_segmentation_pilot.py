"""Run a restartable real-frame segmentation pilot; no grasp gate is implied."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

# Required before any CUDA context is created when deterministic GEMM is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import torch
import torchvision

from src.checkpoints import (CheckpointStore, _exclusive_lock, atomic_json,
                             file_sha256, fingerprint)
from src.graspnet_frames import GraspNetFrames, discover_pilot_frames
from src.segmentation import (PilotInstanceNet, decode_instances, instance_counts,
                              segmentation_loss, summarize_counts)


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def prepare_run_files(directory, config, manifest, resume):
    """Validate durable initialization metadata before allowing any resume."""
    config_path = directory / "config.json"
    manifest_path = directory / "data_manifest.json"
    if config.get("data_fingerprint") != fingerprint(manifest):
        raise ValueError("Configuration does not bind the current data manifest")
    if config_path.exists():
        if not resume:
            raise ValueError("Existing run requires --resume; use a new run directory")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if fingerprint(previous) != fingerprint(config):
            raise ValueError("Run configuration, source files or data changed; use a new run directory")
        if manifest_path.exists():
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if fingerprint(previous_manifest) != fingerprint(manifest):
                raise ValueError("Recorded data manifest differs; no initialization recovery is allowed")
        else:
            # The config already durably binds this exact manifest hash. Recovery
            # is allowed only if no checkpoint or training evidence exists.
            assert_no_training_evidence(directory)
            atomic_json(manifest_path, manifest)
    elif resume:
        raise ValueError("No previous configuration exists in this run directory")
    else:
        entries = [path for path in directory.iterdir() if path.name != ".checkpoint.lock"]
        if entries:
            raise ValueError("New run directory contains previous files; refusing to overwrite")
        atomic_json(config_path, config)
        atomic_json(manifest_path, manifest)


def assert_no_training_evidence(directory):
    """Never interpret damaged or missing training state as a fresh run."""
    checkpoint_directory = directory / "checkpoints"
    if checkpoint_directory.exists():
        artifacts = [path for path in checkpoint_directory.iterdir() if path.name != ".checkpoint.lock"]
        if artifacts:
            raise RuntimeError("Checkpoint artifacts exist but cannot be reset to step zero")
    allowed = {"config.json", "data_manifest.json", "history.json", "progress.json", ".checkpoint.lock", "checkpoints"}
    unexpected = [path.name for path in directory.iterdir() if path.name not in allowed
                  and not path.name.startswith(("config.json.tmp-", "data_manifest.json.tmp-"))]
    if unexpected:
        raise RuntimeError("Unrecognized run artifacts prevent initialization recovery: " + ", ".join(unexpected))
    history_path = directory / "history.json"
    if history_path.exists():
        if json.loads(history_path.read_text(encoding="utf-8")) != []:
            raise RuntimeError("Training history exists without an intact checkpoint; refusing step-zero reset")
    progress_path = directory / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(progress, dict) or progress.get("status") != "running":
            raise RuntimeError("Progress is not an untouched initialization record")
        for name in ("step", "optimizer_steps", "epoch", "next_batch"):
            if progress.get(name, 0) != 0:
                raise RuntimeError("Training progress exists without an intact checkpoint; refusing step-zero reset")
        cursor = progress.get("cursor", {})
        if not isinstance(cursor, dict) or any(cursor.get(name, 0) != 0 for name in ("epoch", "next_batch", "global_step")):
            raise RuntimeError("Saved cursor shows prior training; refusing step-zero reset")


def restore_or_initialize_checkpoint(directory, store, model, optimizer, scheduler, scaler, config):
    """Commit step zero before the first update, or restore existing evidence.

    An initialization-only directory can be completed deterministically. Any
    checkpoint artifact (including a corrupt pointer or unpublished payload)
    forces the ordinary integrity-checked recovery path, never silent reset.
    """
    artifacts = [path for path in store.directory.iterdir() if path.name != ".checkpoint.lock"]
    if artifacts:
        return store.load_latest(model, optimizer, scheduler=scheduler, scaler=scaler)
    assert_no_training_evidence(directory)
    extra = {"history": [], "config": config, "data_fingerprint": store.data_fingerprint}
    manifest = store.save(model, optimizer, 0, 0, 0, True,
                          scheduler=scheduler, scaler=scaler, extra=extra)
    return {"cursor": {"epoch": 0, "next_batch": 0, "global_step": 0}, "extra": extra,
            "checkpoint_path": str(store.directory / manifest["file"]),
            "generation": manifest["generation"], "initialized_step_zero": True,
            "rejected_generations": [], "latest_pointer_problem": None}


def evaluate(model, dataset, device, decoder):
    model.eval()
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    intersection = union = 0
    with torch.no_grad():
        for index, record in enumerate(dataset.records):
            inputs, targets = dataset.get(index)
            outputs = model(inputs[None].to(device))
            predicted_instances = decode_instances(outputs, **decoder)
            labels = targets["labels"].numpy()
            counts = instance_counts(predicted_instances, labels)
            pred_fg = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= decoder["foreground_threshold"]
            actual_fg = labels > 0
            frame_intersection = int((pred_fg & actual_fg).sum())
            frame_union = int((pred_fg | actual_fg).sum())
            intersection += frame_intersection
            union += frame_union
            for key in totals:
                totals[key] += counts[key]
            rows.append({"scene": record["scene"], "frame": record["frame"],
                         "foreground_intersection": frame_intersection, "foreground_union": frame_union,
                         "foreground_pixel_iou": frame_intersection / frame_union if frame_union else 1.0,
                         "instances": summarize_counts(counts)})
    model.train()
    return {"frames": rows, "frame_count": len(rows),
            "scene_count": len({row["scene"] for row in rows}),
            "foreground_pixel_iou": intersection / union if union else 1.0,
            "foreground_intersection": intersection, "foreground_union": union,
            "instances": summarize_counts(totals),
            "metric_definition": "Micro precision/recall/F1 at mask IoU >= 0.50 using one-to-one Hungarian matching; not mask AP",
            "scope": "Declared pilot validation frames only; no test-set or grasp result"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/graspnet")
    parser.add_argument("--split", default="data/manifests/scene_splits.json")
    parser.add_argument("--mode", choices=("rgb", "rgbd"), default="rgbd")
    parser.add_argument("--camera", choices=("realsense", "kinect"), default="realsense")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Absolute optimizer-step stopping limit; may be increased on resume")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--max-frames-per-scene", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA float16 mixed precision")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=1)
    parser.add_argument("--crash-after-step", type=int, default=None,
                        help="Exit 37 after this optimizer step; restore the last committed checkpoint")
    return parser.parse_args()


def run(args):
    for name in ("epochs", "height", "width", "accumulation", "max_frames_per_scene", "save_every_steps"):
        if getattr(args, name) < 1:
            raise ValueError(name + " must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.height < 64 or args.width < 64 or args.learning_rate <= 0:
        raise ValueError("Require input dimensions >=64 and positive learning rate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter; use the isolated CUDA environment")
    if args.amp and args.device != "cuda":
        raise ValueError("AMP is supported only for CUDA in this pilot")
    run_directory = project_path(args.run_dir)
    if os.path.commonpath([str(run_directory), str(ROOT)]) != str(ROOT) or run_directory == ROOT:
        raise ValueError("run-dir must be a subdirectory of the project")
    data_root = project_path(args.data_root)
    # Validate real data before writing a configuration or an experiment result.
    manifest = discover_pilot_frames(data_root, project_path(args.split), args.camera, args.max_frames_per_scene)
    train_data = GraspNetFrames(data_root, manifest["records"]["train"], size=(args.height, args.width),
                               mode=args.mode, seed=args.seed, augment=True)
    validation_data = GraspNetFrames(data_root, manifest["records"]["validation"], size=(args.height, args.width),
                                    mode=args.mode, seed=args.seed, augment=False)
    for dataset in (train_data, validation_data):
        for index in range(len(dataset)):
            dataset.get(index)
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
    config = {
        "schema": 1, "scope": "Real-frame segmentation engineering pilot; P3 grasp baseline is not evaluated",
        "architecture": "ResNet18-GroupNorm-FPN48, early fusion, foreground-center-offset heads",
        "weights": "random initialization; torchvision weights=None", "mode": args.mode,
        "input_size": [args.height, args.width], "batch_size": 1, "accumulation": args.accumulation,
        "epochs": args.epochs, "seed": args.seed, "num_workers": 0,
        "augmentation": "sample-seeded horizontal flip and RGB brightness [0.9,1.1]",
        "depth": "meta.factor_depth to meters; clip at 2m and divide by 2; separate validity channel",
        "loss": "foreground BCE + Dice + Gaussian center focal + 5 * foreground-only SmoothL1(beta=0.05)",
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 0.0001,
        "scheduler": "CosineAnnealingLR over planned optimizer steps; no warmup in this pilot",
        "gradient_clip_norm": 1.0, "device": args.device, "amp": args.amp,
        "amp_initial_scale": 128.0, "amp_max_overflow_retries": 16,
        "deterministic_algorithms": True, "save_every_optimizer_steps": args.save_every_steps,
        "decoder": {"foreground_threshold": 0.5, "center_threshold": 0.35, "min_pixels": 20, "max_centers": 64},
        "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__), "numpy": np.__version__, "scipy": scipy.__version__},
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "source_sha256": {name: file_sha256(ROOT / name) for name in
                          ("scripts/train_segmentation_pilot.py", "src/segmentation.py", "src/graspnet_frames.py", "src/checkpoints.py")},
        "parent_protocol_sha256": file_sha256(ROOT / "configs" / "training_protocol.json"),
        "data_fingerprint": fingerprint(manifest),
    }
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    run_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(run_directory):
        prepare_run_files(run_directory, config, manifest, args.resume)
        model = PilotInstanceNet(mode=args.mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        planned_steps = args.epochs * math.ceil(len(train_data) / args.accumulation)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=planned_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=config["amp_initial_scale"])
        store = CheckpointStore(run_directory / "checkpoints", config_hash, data_hash, keep_last=3)
        epoch = next_batch = global_step = 0
        history = []
        restored = restore_or_initialize_checkpoint(run_directory, store, model, optimizer, scheduler, scaler, config)
        epoch = restored["cursor"]["epoch"]
        next_batch = restored["cursor"]["next_batch"]
        global_step = restored["cursor"]["global_step"]
        history = restored["extra"]["history"]
        if [row["step"] for row in history] != list(range(1, global_step + 1)):
            raise RuntimeError("Checkpoint history and step cursor disagree")
        if restored["extra"].get("config") != config:
            raise RuntimeError("Checkpoint configuration is incomplete or changed")
        start_step = global_step
        limit = min(args.max_steps, planned_steps) if args.max_steps is not None else planned_steps
        atomic_json(run_directory / "progress.json", {
            "status": "running", "step": global_step, "target_step": limit,
            "config_fingerprint": config_hash, "data_fingerprint": data_hash,
            "resumed_from": restored["checkpoint_path"] if restored else None,
        })
        atomic_json(run_directory / "history.json", history)
        model.train()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        while epoch < args.epochs and global_step < limit:
            order = train_data.epoch_order(epoch)
            if next_batch >= len(order) or next_batch % args.accumulation:
                raise RuntimeError("Saved batch cursor is not a valid accumulation boundary")
            selected = order[next_batch:next_batch + args.accumulation]
            attempts = 0
            while True:
                accumulated_loss = 0.0
                parts = {}
                for index in selected:
                    inputs, targets = train_data.get(index, epoch=epoch)
                    targets = {key: value[None].to(device) for key, value in targets.items() if key != "labels"}
                    with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.amp):
                        outputs = model(inputs[None].to(device))
                        loss, components = segmentation_loss(outputs, targets)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite training loss; latest committed checkpoint is preserved")
                    scaler.scale(loss / len(selected)).backward()
                    accumulated_loss += float(loss.detach().cpu()) / len(selected)
                    for key, value in components.items():
                        parts[key] = parts.get(key, 0.0) + float(value.cpu()) / len(selected)
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                if not args.amp and not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("Nonfinite gradients; latest committed checkpoint is preserved")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    break
                attempts += 1
                if attempts >= config["amp_max_overflow_retries"]:
                    raise RuntimeError("Repeated mixed precision overflow; restart a new run without --amp")
                # Retry the same deterministic micro-batches; no skipped update is counted.
            scheduler.step()
            global_step += 1
            finished_epoch = epoch
            next_batch += len(selected)
            if next_batch == len(order):
                epoch += 1
                next_batch = 0
            row = {"step": global_step, "epoch": finished_epoch, "loss": accumulated_loss,
                   "components": parts, "learning_rate_after_step": scheduler.get_last_lr()[0],
                   "samples": [{"scene": train_data.records[index]["scene"], "frame": train_data.records[index]["frame"]} for index in selected],
                   "amp_retries": attempts}
            history.append(row)
            should_save = global_step % args.save_every_steps == 0 or next_batch == 0 or global_step == limit
            if should_save:
                store.save(model, optimizer, epoch, next_batch, global_step, True,
                           scheduler=scheduler, scaler=scaler,
                           extra={"history": history, "config": config, "data_fingerprint": data_hash})
                atomic_json(run_directory / "history.json", history)
                atomic_json(run_directory / "progress.json", {
                    "status": "running", "step": global_step, "target_step": limit,
                    "epoch": epoch, "next_batch": next_batch,
                    "config_fingerprint": config_hash, "data_fingerprint": data_hash})
            print(json.dumps(dict(row, checkpoint_committed=should_save)), flush=True)
            if args.crash_after_step == global_step:
                os._exit(37)
        # Evaluation is reproducible and may safely repeat after an interrupted evaluation.
        validation = evaluate(model, validation_data, device, config["decoder"])
        report = {"status": "completed" if global_step >= planned_steps else "step_limit_reached",
                  "config_fingerprint": config_hash, "data_fingerprint": data_hash,
                  "optimizer_steps": global_step, "planned_steps": planned_steps,
                  "resumed_from_step": start_step, "updates_this_invocation": global_step - start_step,
                  "cursor": {"epoch": epoch, "next_batch": next_batch, "global_step": global_step},
                  "validation": validation, "elapsed_seconds_this_invocation": time.perf_counter() - started,
                  "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
                  "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "latest_loss": history[-1]["loss"] if history else None,
                  "limitations": ["Small-frame engineering pilot from random initialization; not evidence of method improvement",
                                  "Early RGB-D fusion pilot differs from the proposed quality-aware dual encoder",
                                  "Instance precision and recall at IoU 0.50 are not COCO mask AP",
                                  "No grasp candidate inference, grasp AP or robot execution was evaluated",
                                  "Crash recovery is process-level verification, not an electrical power-loss test"]}
        atomic_json(run_directory / "result.json", report)
        atomic_json(run_directory / "progress.json", {key: report[key] for key in
                    ("status", "optimizer_steps", "planned_steps", "cursor", "config_fingerprint", "data_fingerprint")})
        print(json.dumps({"status": report["status"], "result": str(run_directory / "result.json"),
                          "optimizer_steps": global_step, "updates_this_invocation": global_step - start_step}, ensure_ascii=False), flush=True)
        return report


if __name__ == "__main__":
    try:
        run(arguments())
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        print("Training stopped: {}".format(error), file=sys.stderr, flush=True)
        sys.exit(2)
