"""Render four predeclared validation frames from the frozen final v3 checkpoint.

The sample identities below are fixed before final evaluation. No image is
selected by its score, and no threshold or checkpoint is selected here.
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
for _name, _relative in {"TORCH_HOME": "runtime/cache/torch", "CUDA_CACHE_PATH": "runtime/cache/cuda",
                         "TEMP": "runtime/temp", "TMP": "runtime/temp"}.items():
    os.environ[_name] = str(ROOT / _relative)

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.checkpoints import atomic_json, atomic_write, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet, decode_instances, instance_counts, summarize_counts

FIXED_FRAMES = ((18, "0000"), (22, "0000"), (33, "0000"), (39, "0000"))
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


def prepare_final_model(directory):
    config, manifest, result = (read_json(directory / name) for name in
                                ("config.json", "data_manifest.json", "result.json"))
    protocol = config["protocol"]
    if protocol["planned_steps"] != 1600 or protocol["decoder"] != FIXED_DECODER:
        raise ValueError("Require the frozen 1600-update plan and original fixed thresholds")
    if result["status"] != "completed" or result["optimizer_steps"] != 1600 or not result["validation_evaluated"]:
        raise ValueError("Final 1600-step model evaluation must exist before rendering")
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    if config["data_fingerprint"] != data_hash or result["data_fingerprint"] != data_hash:
        raise ValueError("Data fingerprint mismatch")
    if result["config_fingerprint"] != config_hash:
        raise ValueError("Run/result configuration mismatch")
    if file_sha256(ROOT / "configs/segmentation_expanded_v3.json") != config["protocol_sha256"]:
        raise ValueError("Frozen protocol file changed")
    for source, digest in config["source_sha256"].items():
        if file_sha256(project_path(source)) != digest:
            raise ValueError("Frozen source changed: " + source)
    source_manifest_path = project_path(protocol["data_manifest"])
    if file_sha256(source_manifest_path) != manifest["source_manifest_sha256"]:
        raise ValueError("Completed source data manifest changed")
    source_manifest = read_json(source_manifest_path)
    if source_manifest["status"] != "verified_expanded_sample" or source_manifest != manifest["source_manifest"]:
        raise ValueError("Expanded data provenance is not the frozen verified source")
    if source_manifest["split_sha256"] != file_sha256(project_path(protocol["split"])):
        raise ValueError("Frozen scene split changed")
    if manifest["records"] != source_manifest["records"]:
        raise ValueError("Run record identities differ from completed data source")
    data_root = project_path(manifest["data_root"])
    checked_paths = set()
    # Recheck byte hashes for all declared train/validation frame files, without
    # decoding images other than the four predeclared validation examples.
    for split in ("train", "validation"):
        for record in manifest["records"][split]:
            for kind, relative in record["files"].items():
                path = (data_root / relative).resolve()
                if not path.is_relative_to(data_root) or file_sha256(path) != record["sha256"][kind]:
                    raise ValueError("Data path/checksum mismatch: " + relative)
                checked_paths.add(str(path))
    train_scenes = {record["scene"] for record in manifest["records"]["train"]}
    if train_scenes.intersection(scene for scene, frame in FIXED_FRAMES):
        raise ValueError("Fixed visualization frame occurs in training")
    validation = {(record["scene"], record["frame"]): record for record in manifest["records"]["validation"]}
    records = [validation[identity] for identity in FIXED_FRAMES]
    if any(record.get("validation_exposure") != "fresh" for record in records):
        raise ValueError("Expected the four initially fresh validation scenes")
    checkpoint_directory = directory / "checkpoints"
    pointer = read_json(checkpoint_directory / "latest.json")
    meta_path = (checkpoint_directory / pointer["manifest"]).resolve()
    if not meta_path.is_relative_to(checkpoint_directory) or file_sha256(meta_path) != pointer["manifest_sha256"]:
        raise ValueError("Checkpoint manifest integrity failure")
    metadata = read_json(meta_path)
    checkpoint = (checkpoint_directory / metadata["file"]).resolve()
    if not checkpoint.is_relative_to(checkpoint_directory) or file_sha256(checkpoint) != metadata["sha256"]:
        raise ValueError("Checkpoint payload integrity failure")
    if checkpoint != project_path(result["checkpoint_path"]) or metadata["sha256"] != result["checkpoint_sha256"]:
        raise ValueError("Result and checkpoint identities disagree")
    if metadata["global_step"] != 1600 or metadata["config_fingerprint"] != config_hash or metadata["data_fingerprint"] != data_hash:
        raise ValueError("Checkpoint does not match the exact final run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["cursor"]["global_step"] != 1600 or payload["config_fingerprint"] != config_hash or payload["data_fingerprint"] != data_hash:
        raise ValueError("Checkpoint payload disagrees with the frozen final run")
    if payload["extra"]["config"] != config:
        raise ValueError("Checkpoint embedded configuration differs")
    model = PilotInstanceNet(mode=protocol["mode"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    evidence = {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": metadata["sha256"],
                "config_fingerprint": config_hash, "data_fingerprint": data_hash,
                "result_sha256": file_sha256(directory / "result.json"),
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "frame_files_hash_checked": len(checked_paths), "optimizer_steps": 1600}
    return model, protocol, records, data_root, result, evidence


def instance_colors(labels):
    pixels = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels[labels > 0]):
        pixels[labels == label] = ((int(label) * 71 + 40) % 216 + 40,
                                  (int(label) * 113 + 40) % 216 + 40,
                                  (int(label) * 173 + 40) % 216 + 40)
    return Image.fromarray(pixels)


def draw_frame(path, rgb, labels, predicted, record, metrics):
    height, width = labels.shape
    canvas = Image.new("RGB", (width * 3, height + 150), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), "VALIDATION (NOT TRAINING FIT) | scene {:04d} frame {} | final step 1600".format(
        record["scene"], record["frame"]), fill="#111111")
    draw.text((8, 27), "Predeclared example: frame 0000 in each fresh validation scene; never selected by score", fill="#111111")
    draw.text((8, 45), "Fixed thresholds: foreground 0.50, center 0.35, min area 20 px, max centers 64", fill="#111111")
    panels = [("RGB validation input", rgb), ("Ground truth reference", instance_colors(labels)),
              ("Model prediction (no GT in decoder)", instance_colors(predicted))]
    for index, (caption, image) in enumerate(panels):
        x = index * width
        draw.text((x + 8, 66), caption, fill="#111111")
        canvas.paste(image, (x, 87))
    draw.text((8, height + 95), "This frame: foreground IoU {:.4f} | TP/FP/FN {}/{}/{} | instance F1@IoU0.50 {:.4f}".format(
        metrics["foreground_pixel_iou"], metrics["instances"]["tp"], metrics["instances"]["fp"],
        metrics["instances"]["fn"], metrics["instances"]["f1_iou50"]), fill="#111111")
    draw.text((8, height + 113), "Independent ID colors; GT only reference/metrics. These validation scenes are development data after this evaluation.", fill="#111111")
    draw.text((8, height + 131), "Single seed; four examples are not the full validation summary, an official test result, or grasp success.", fill="#111111")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_write(path, lambda stream: stream.write(buffer.getvalue()))


def run(directory, output_directory, device):
    model, protocol, records, data_root, result, evidence = prepare_final_model(directory)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.to(device)
    dataset = GraspNetFrames(data_root, records, size=tuple(protocol["input_size"]),
                            mode=protocol["mode"], seed=protocol["seed"], augment=False)
    reference = {(row["scene"], row["frame"]): row for row in result["evaluation"]["validation"]["frames"]}
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_reports = []
    for index, record in enumerate(records):
        inputs, targets = dataset.get(index)
        with torch.no_grad():
            outputs = model(inputs[None].to(device))
            predicted = decode_instances(outputs, **FIXED_DECODER)
            foreground = outputs["foreground"][0, 0].sigmoid().cpu().numpy() >= 0.5
        labels = targets["labels"].numpy()
        intersection = int((foreground & (labels > 0)).sum())
        union = int((foreground | (labels > 0)).sum())
        metrics = {"instances": summarize_counts(instance_counts(predicted, labels)),
                   "foreground_intersection": intersection, "foreground_union": union,
                   "foreground_pixel_iou": intersection / union if union else 1.0}
        recorded = reference[(record["scene"], record["frame"])]
        for key, value in metrics.items():
            if value != recorded[key]:
                raise RuntimeError("Rendered frame metrics differ from final evaluation: {} {} {}".format(record["scene"], record["frame"], key))
        with Image.open(data_root / record["files"]["rgb"]) as source:
            rgb = source.convert("RGB").resize((labels.shape[1], labels.shape[0]), Image.Resampling.BILINEAR)
        image_path = output_directory / "scene_{:04d}_{}.png".format(record["scene"], record["frame"])
        draw_frame(image_path, rgb, labels, predicted, record, metrics)
        frame_reports.append({"scene": record["scene"], "frame": record["frame"], "split": "validation",
                              "validation_exposure_at_start": "fresh", "metrics": metrics,
                              "exact_match_to_recorded_final_evaluation": True,
                              "image": str(image_path.relative_to(ROOT)), "sha256": file_sha256(image_path)})
    report = {"schema": 3, "status": "completed", "scope": "Four predeclared validation illustrations, not training fit",
              "selection_rule": "Frame 0000 for scenes 18,22,33,39, fixed before final evaluation; no score-based choice",
              "fixed_frames": [{"scene": scene, "frame": frame} for scene, frame in FIXED_FRAMES],
              "fixed_decoder": FIXED_DECODER, "checkpoint_selection": "Frozen final step 1600 only",
              "ground_truth_usage": "Reference image and evaluation metrics only; no GT enters model or decoder",
              "colors": "Independent deterministic instance-ID colors in each panel; no GT-matching recoloring",
              "train_frames_decoded": 0, "validation_frames_decoded": 4, "official_test_frames_decoded": 0,
              "device": str(device), "source_sha256": file_sha256(Path(__file__)), "evidence": evidence,
              "frames": frame_reports,
              "limitations": ["No parameter or threshold update; exact metric equality with the recorded final evaluation is required",
                              "Initially fresh validation scenes become development data after their first evaluation",
                              "Four predetermined examples do not replace the 64-frame/four-scene fresh-validation summary",
                              "Single-seed model; no method-superiority, official-test, grasp-AP or real-robot success claim"]}
    atomic_json(output_directory / "manifest.json", report)
    print(json.dumps({"status": "completed", "images": len(frame_reports),
                      "manifest": str(output_directory / "manifest.json")}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="experiments/segmentation_expanded_v3/rgbd_original")
    parser.add_argument("--output-dir", default="results/P2/segmentation_expanded_v3_visuals")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(project_path(args.run_dir), project_path(args.output_dir), torch.device(args.device))
