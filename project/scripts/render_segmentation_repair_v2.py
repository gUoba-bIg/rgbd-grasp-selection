"""Render the final, fixed-threshold two-arm four-training-frame fitting check.

Importing this module does not load data, weights, or initialize a GPU. The run
entry point requires both arms to have completed the frozen 400-update plan.
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _name, _relative in {
        "TORCH_HOME": "runtime/cache/torch", "CUDA_CACHE_PATH": "runtime/cache/cuda",
        "XDG_CACHE_HOME": "runtime/cache", "TMP": "runtime/temp", "TEMP": "runtime/temp"}.items():
    os.environ[_name] = str(ROOT / _relative)

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.checkpoints import atomic_json, atomic_write, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet, decode_instances, instance_counts, summarize_counts


ARMS = ("original", "center_mse")
EXPECTED_FRAMES = {(3, "0000"), (3, "0016"), (12, "0000"), (12, "0016")}
FIXED_DECODER = {"foreground_threshold": 0.5, "center_threshold": 0.35,
                 "min_pixels": 20, "max_centers": 64}


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if ROOT not in path.parents:
        raise ValueError("Artifact paths must be inside the project")
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_arm(name, experiment_directory):
    directory = experiment_directory / name
    config, manifest, result = (read_json(directory / filename) for filename in
                                 ("config.json", "data_manifest.json", "result.json"))
    if config["loss_name"] != name or config["planned_steps"] != 400:
        raise ValueError("Arm does not match the frozen 400-update plan: " + name)
    if result["status"] != "completed" or result["optimizer_steps"] != 400:
        raise ValueError("Both final arms must complete before rendering: " + name)
    if result["validation_evaluated"] is not False or config["decoder"] != FIXED_DECODER:
        raise ValueError("Arm violates the training-only fixed-threshold scope")
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    if config["data_fingerprint"] != data_hash or result["data_fingerprint"] != data_hash:
        raise ValueError("Data fingerprint mismatch: " + name)
    if result["config_fingerprint"] != config_hash:
        raise ValueError("Configuration fingerprint mismatch: " + name)
    for relative, digest in config["source_sha256"].items():
        if file_sha256(project_path(relative)) != digest:
            raise ValueError("Frozen source changed: " + relative)
    records = manifest["records"]["train"]
    if len(records) != 4 or {(record["scene"], record["frame"]) for record in records} != EXPECTED_FRAMES:
        raise ValueError("Only the frozen four training frames may be rendered")
    data_root = project_path(manifest["data_root"])
    for record in records:
        for kind, relative in record["files"].items():
            path = (data_root / relative).resolve()
            if not path.is_relative_to(data_root) or file_sha256(path) != record["sha256"][kind]:
                raise ValueError("Training file path or checksum mismatch")
    checkpoint_directory = directory / "checkpoints"
    pointer = read_json(checkpoint_directory / "latest.json")
    meta_path = (checkpoint_directory / pointer["manifest"]).resolve()
    if not meta_path.is_relative_to(checkpoint_directory) or file_sha256(meta_path) != pointer["manifest_sha256"]:
        raise ValueError("Checkpoint manifest integrity failure")
    metadata = read_json(meta_path)
    checkpoint = (checkpoint_directory / metadata["file"]).resolve()
    if not checkpoint.is_relative_to(checkpoint_directory) or file_sha256(checkpoint) != metadata["sha256"]:
        raise ValueError("Checkpoint payload integrity failure")
    if metadata["global_step"] != 400 or metadata["config_fingerprint"] != config_hash or metadata["data_fingerprint"] != data_hash:
        raise ValueError("Checkpoint must be the exact final frozen arm")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["cursor"]["global_step"] != 400 or payload["config_fingerprint"] != config_hash or payload["data_fingerprint"] != data_hash:
        raise ValueError("Checkpoint payload does not match final plan")
    model = PilotInstanceNet(mode=config["mode"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return {"model": model, "config": config, "manifest": manifest,
            "evidence": {"arm": name, "optimizer_steps": 400,
                         "config_fingerprint": config_hash, "data_fingerprint": data_hash,
                         "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": metadata["sha256"],
                         "recorded_training_fit": result["training_fit"]}}


def instance_colors(labels):
    """Independent deterministic ID colors; no ground-truth matching is used."""
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels[labels > 0]):
        image[labels == label] = ((int(label) * 71 + 40) % 216 + 40,
                                 (int(label) * 113 + 40) % 216 + 40,
                                 (int(label) * 173 + 40) % 216 + 40)
    return Image.fromarray(image)


def render_frame(path, rgb, truth, predictions, metrics, record):
    height, width = truth.shape
    captions = ["RGB", "Ground truth", "Original center loss", "Center MSE loss"]
    panels = [rgb, instance_colors(truth), instance_colors(predictions["original"]),
              instance_colors(predictions["center_mse"])]
    canvas = Image.new("RGB", (width * 4, height + 137), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "TRAINING FIT ONLY | scene {:04d} frame {} | final step 400".format(
        record["scene"], record["frame"]), fill="#111111")
    draw.text((8, 27), "Fixed thresholds: foreground 0.50, center 0.35, min area 20 px, max centers 64", fill="#111111")
    for index, (caption, panel) in enumerate(zip(captions, panels)):
        x = index * width
        draw.text((x + 8, 53), caption, fill="#111111")
        canvas.paste(panel, (x, 75))
        if index >= 2:
            arm = ARMS[index - 2]
            score = metrics[arm]
            draw.text((x + 8, height + 84), "TP/FP/FN: {tp}/{fp}/{fn} | F1: {f1_iou50:.3f}".format(
                **score["instances"]), fill="#111111")
            draw.text((x + 8, height + 101), "Foreground IoU: {:.3f}".format(score["foreground_pixel_iou"]), fill="#111111")
    draw.text((8, height + 120), "Independent ID colors across panels; GT used only for reference and metrics. Model decoding never uses GT.", fill="#111111")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_write(path, lambda stream: stream.write(buffer.getvalue()))


def run(experiment_directory, output_directory, device):
    arms = {name: load_final_arm(name, experiment_directory) for name in ARMS}
    reference = arms["original"]
    if reference["manifest"] != arms["center_mse"]["manifest"]:
        raise ValueError("Arms do not share exactly the same training manifest")
    for field in ("input_size", "mode", "seed", "decoder"):
        if reference["config"][field] != arms["center_mse"]["config"][field]:
            raise ValueError("Shared comparison setting differs: " + field)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for item in arms.values():
        item["model"].to(device)
    dataset = GraspNetFrames(reference["manifest"]["data_root"], reference["manifest"]["records"]["train"],
                            size=reference["config"]["input_size"], mode=reference["config"]["mode"], augment=False)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_reports = []
    for index, record in enumerate(dataset.records):
        inputs, targets = dataset.get(index)
        truth = targets["labels"].numpy()
        predictions, metrics = {}, {}
        with torch.no_grad():
            for arm in ARMS:
                outputs = arms[arm]["model"](inputs[None].to(device))
                # Deliberately no labels, instance count or oracle heads enter decoding.
                predicted = decode_instances(outputs, **FIXED_DECODER)
                predictions[arm] = predicted
                foreground = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= 0.5
                intersection = int((foreground & (truth > 0)).sum())
                union = int((foreground | (truth > 0)).sum())
                metrics[arm] = {"instances": summarize_counts(instance_counts(predicted, truth)),
                                "foreground_pixel_iou": intersection / union if union else 1.0}
        rgb_path = Path(reference["manifest"]["data_root"]) / record["files"]["rgb"]
        with Image.open(rgb_path) as source:
            rgb = source.convert("RGB").resize((truth.shape[1], truth.shape[0]), Image.Resampling.BILINEAR)
        image_path = output_directory / "scene_{:04d}_{}.png".format(record["scene"], record["frame"])
        render_frame(image_path, rgb, truth, predictions, metrics, record)
        frame_reports.append({"scene": record["scene"], "frame": record["frame"],
                              "image": str(image_path.relative_to(ROOT)), "sha256": file_sha256(image_path),
                              "metrics": metrics})
    report = {"schema": 1, "status": "completed", "scope": "TRAINING FIT ONLY",
              "selection": "Both frozen final 400-update checkpoints, no checkpoint or threshold selection",
              "fixed_decoder": FIXED_DECODER, "validation_frames_loaded": 0, "test_frames_loaded": 0,
              "model_decoding_uses_ground_truth": False,
              "colors": "Independent deterministic colors from each panel's label IDs; no cross-panel GT matching",
              "ground_truth_usage": "Reference panel and evaluation metrics only; never input to the model or decoder",
              "device": str(device), "source_sha256": file_sha256(Path(__file__)),
              "arms": {arm: arms[arm]["evidence"] for arm in ARMS}, "frames": frame_reports,
              "limitations": ["These four images were used to train both models; visuals do not demonstrate generalization",
                              "Training precision/recall/F1 at mask IoU >= 0.50 is not mask AP",
                              "Different colors across panels do not imply object identity mismatch"]}
    atomic_json(output_directory / "manifest.json", report)
    print(json.dumps({"status": "completed", "images": len(frame_reports),
                      "manifest": str(output_directory / "manifest.json")}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="experiments/segmentation_repair_v2")
    parser.add_argument("--output-dir", default="results/P2/segmentation_repair_v2_visuals")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(project_path(args.experiment_dir), project_path(args.output_dir), torch.device(args.device))
