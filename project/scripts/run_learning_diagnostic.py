"""Bounded real-frame learning diagnostic; training-set fitting is not generalization."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.train_segmentation_pilot import evaluate
from src.checkpoints import CheckpointStore, atomic_json, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames, discover_pilot_frames
from src.segmentation import PilotInstanceNet


RUN_DIRECTORY = ROOT / "experiments" / "segmentation_diagnostics" / "rgbd_overfit_v1"
REPORT = ROOT / "results" / "P2" / "learning_diagnostic.json"
LOG = ROOT / "results" / "P2" / "learning_diagnostic.log"


def center_diagnostics(model, dataset, device, threshold):
    """Describe fixed-threshold center behavior without selecting any threshold."""
    rows = []
    model.eval()
    with torch.no_grad():
        for index, record in enumerate(dataset.records):
            inputs, _ = dataset.get(index)
            outputs = model(inputs[None].to(device))
            centers = outputs["centers"].float().sigmoid()[0, 0]
            rows.append({"scene": record["scene"], "frame": record["frame"],
                         "max_center_probability": float(centers.max().cpu()),
                         "pixels_at_or_above_fixed_center_threshold": int((centers >= threshold).sum().cpu())})
    return rows


def main():
    started = time.perf_counter()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "status": "running", "kind": "bounded_real_frame_learning_diagnostic",
        "run_directory": str(RUN_DIRECTORY), "log": str(LOG),
        "planned_epochs": 50, "maximum_optimizer_steps": 100,
        "learning_rate": 0.001, "amp": True, "frames_per_scene": 2,
        "scope": "Four real training frames from two scenes; separate validation scenes; training fitting is not a generalization result",
        "decoder_policy": "Use the existing frozen decoder unchanged; do not tune thresholds on validation results",
        "limits": ["This bounded diagnostic does not establish method improvement or publishable performance",
                   "Training-set metrics describe fitting only and must not be reported as held-out performance",
                   "Validation covers a small selected subset, not the official benchmark test set",
                   "No grasp inference, grasp AP or robot execution is assessed",
                   "Stop after 100 optimizer steps regardless of the outcome; no automatic further training"],
    }
    atomic_json(REPORT, base)
    command = [sys.executable, str(ROOT / "scripts" / "train_segmentation_pilot.py"),
               "--data-root", str(ROOT / "data" / "raw" / "graspnet"),
               "--mode", "rgbd", "--epochs", "50", "--max-steps", "100",
               "--max-frames-per-scene", "2", "--learning-rate", "0.001", "--amp",
               "--device", "cuda", "--accumulation", "2", "--save-every-steps", "1",
               "--run-dir", str(RUN_DIRECTORY)]
    if (RUN_DIRECTORY / "config.json").exists():
        command.append("--resume")
    try:
        before_training = discover_pilot_frames(ROOT / "data" / "raw" / "graspnet",
                                               ROOT / "data" / "manifests" / "scene_splits.json", max_frames_per_scene=2)
        if len(before_training["records"]["train"]) != 4:
            raise RuntimeError("The fixed diagnostic requires exactly 4 training frames before it can start")
        print("Running bounded RGB-D learning diagnostic, maximum 100 optimizer steps.", flush=True)
        with LOG.open("a", encoding="utf-8") as stream:
            stream.write("\nDiagnostic invocation: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            stream.flush()
            process = subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError("Training returned exit {}; see {}".format(process.returncode, LOG))
        config = json.loads((RUN_DIRECTORY / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((RUN_DIRECTORY / "data_manifest.json").read_text(encoding="utf-8"))
        current_manifest = discover_pilot_frames(ROOT / "data" / "raw" / "graspnet",
                                               ROOT / "data" / "manifests" / "scene_splits.json", max_frames_per_scene=2)
        if fingerprint(manifest) != fingerprint(current_manifest):
            raise RuntimeError("Input data changed after training")
        if len(manifest["records"]["train"]) != 4:
            raise RuntimeError("This fixed diagnostic requires exactly 4 training frames")
        for name, digest in config["source_sha256"].items():
            if file_sha256(ROOT / name) != digest:
                raise RuntimeError("Source changed after training: " + name)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        device = torch.device("cuda")
        model = PilotInstanceNet(mode=config["mode"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] * math.ceil(4 / config["accumulation"]))
        scaler = torch.amp.GradScaler("cuda", enabled=config["amp"], init_scale=config["amp_initial_scale"])
        store = CheckpointStore(RUN_DIRECTORY / "checkpoints", fingerprint(config), fingerprint(manifest), keep_last=3)
        restored = store.load_latest(model, optimizer, scheduler=scheduler, scaler=scaler)
        if restored["cursor"]["global_step"] != 100:
            raise RuntimeError("Diagnostic did not finish exactly 100 optimizer steps")
        datasets = {name: GraspNetFrames(manifest["data_root"], manifest["records"][name],
                                         size=tuple(config["input_size"]), mode="rgbd", seed=config["seed"], augment=False)
                    for name in ("train", "validation")}
        metrics = {name: evaluate(model, dataset, device, config["decoder"]) for name, dataset in datasets.items()}
        metrics["train"]["scope"] = "Training frames without augmentation; fitting diagnostic only, not generalization"
        centers = {name: center_diagnostics(model, dataset, device, config["decoder"]["center_threshold"])
                   for name, dataset in datasets.items()}
        history = json.loads((RUN_DIRECTORY / "history.json").read_text(encoding="utf-8"))
        observations = {}
        for name, summary in metrics.items():
            counts = summary["instances"]
            observations[name] = {
                "foreground_pixel_iou": summary["foreground_pixel_iou"],
                "predicted_instances": counts["tp"] + counts["fp"],
                "matched_instances_iou50": counts["tp"],
                "ground_truth_instances": counts["tp"] + counts["fn"],
                "produces_instances": counts["tp"] + counts["fp"] > 0,
                "has_matched_instances_iou50": counts["tp"] > 0,
                "instance_f1_iou50": counts["f1_iou50"],
            }
        if not observations["train"]["produces_instances"]:
            interpretation = "After the fixed training budget, the unchanged decoder still emits no training-set instances. Foreground fitting alone does not validate the instance segmentation pipeline."
        elif not observations["train"]["has_matched_instances_iou50"]:
            interpretation = "The unchanged decoder emits training-set instances, but none matches ground truth at IoU 0.50. Instance fitting remains insufficient under this budget."
        else:
            interpretation = "The unchanged decoder produces training-set instances with IoU-0.50 matches. The reported training metrics measure fitting only; validation metrics remain separate and do not establish full benchmark generalization."
        base.update({"status": "completed", "optimizer_steps": restored["cursor"]["global_step"],
                     "configuration_fingerprint": fingerprint(config), "data_fingerprint": fingerprint(manifest),
                     "checkpoint_generation": restored["generation"], "decoder": config["decoder"],
                     "training_evaluation_augmentation": False, "metrics": metrics,
                     "center_diagnostics": centers, "observations": observations,
                     "loss_first_step": history[0]["loss"], "loss_last_step": history[-1]["loss"],
                     "first_loss_components": history[0]["components"], "last_loss_components": history[-1]["components"],
                     "interpretation": interpretation, "further_training_started": False})
        print(json.dumps({"status": "completed", "report": str(REPORT), "observations": observations}, ensure_ascii=False), flush=True)
    except Exception as error:
        base.update({"status": "failed", "error": "{}: {}".format(type(error).__name__, error)})
        raise
    finally:
        base["elapsed_seconds_this_invocation"] = time.perf_counter() - started
        atomic_json(REPORT, base)


if __name__ == "__main__":
    main()
