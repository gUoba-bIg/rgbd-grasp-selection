"""Read verified GraspNet RGB, depth, object-label and calibration frames."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch

from src.checkpoints import file_sha256


def discover_pilot_frames(data_root, split_path, camera="realsense", max_frames_per_scene=4):
    """Use only the declared pilot scenes; incomplete frames are never fabricated."""
    data_root = Path(data_root).resolve()
    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as stream:
        splits = json.load(stream)
    train_scenes = set(splits["pilot_train"])
    validation_scenes = set(splits["pilot_validation"])
    if train_scenes & validation_scenes:
        raise ValueError("Pilot train and validation scenes overlap")
    if not train_scenes.issubset(set(splits["groups"]["train"])):
        raise ValueError("Pilot training scenes are outside the declared training split")
    if not validation_scenes.issubset(set(splits["groups"]["validation"])):
        raise ValueError("Pilot validation scenes are outside the declared validation split")
    frame_step = int(splits.get("pilot_frame_step", 16))
    if frame_step < 1 or max_frames_per_scene < 1:
        raise ValueError("Frame sampling parameters must be positive")
    records = {"train": [], "validation": []}
    missing = []
    for name, scenes in (("train", train_scenes), ("validation", validation_scenes)):
        for scene in sorted(scenes):
            scene_name = "scene_{:04d}".format(scene)
            candidates = [data_root / "scenes" / scene_name / camera,
                          data_root / scene_name / camera]
            folder = next((path for path in candidates if path.is_dir()), candidates[0])
            accepted = 0
            for rgb in sorted((folder / "rgb").glob("*.png")):
                if not rgb.stem.isdigit() or int(rgb.stem) % frame_step:
                    continue
                paths = {"rgb": rgb, "depth": folder / "depth" / rgb.name,
                         "label": folder / "label" / rgb.name,
                         "meta": folder / "meta" / (rgb.stem + ".mat")}
                absent = [key for key, path in paths.items() if not path.is_file()]
                if absent:
                    missing.append({"scene": scene, "frame": rgb.stem, "missing": absent})
                    continue
                records[name].append({
                    "scene": scene, "frame": rgb.stem, "camera": camera,
                    "files": {key: str(path.relative_to(data_root)) for key, path in paths.items()},
                    "sha256": {key: file_sha256(path) for key, path in paths.items()},
                })
                accepted += 1
                if accepted >= max_frames_per_scene:
                    break
    if len({row["scene"] for row in records["train"]}) < 2 or len(records["train"]) < 4:
        raise ValueError("Need at least 4 complete training frames from 2 declared pilot training scenes")
    if not records["validation"]:
        raise ValueError("Need complete frames from at least 1 declared pilot validation scene")
    return {"data_root": str(data_root), "split_sha256": file_sha256(split_path),
            "camera": camera, "frame_step": frame_step, "max_frames_per_scene": max_frames_per_scene,
            "records": records, "incomplete_frames": missing,
            "scope": "Small real-frame segmentation pilot; not full-dataset or grasp evaluation"}


def instance_targets(labels, sigma=3.0):
    """One center per visible object label, offsets normalized by image dimensions."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or labels.min() < 0 or labels.max() > 88:
        raise ValueError("GraspNet labels must be a 2D array with background 0 and object IDs 1..88")
    height, width = labels.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    foreground = (labels > 0).astype(np.float32)
    centers = np.zeros((height, width), dtype=np.float32)
    offsets = np.zeros((2, height, width), dtype=np.float32)
    for object_id in np.unique(labels):
        if object_id == 0:
            continue
        mask = labels == object_id
        cy, cx = int(np.round(yy[mask].mean())), int(np.round(xx[mask].mean()))
        gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        centers = np.maximum(centers, gaussian)
        offsets[0, mask] = (cx - xx[mask]) / width
        offsets[1, mask] = (cy - yy[mask]) / height
    return {"foreground": torch.from_numpy(foreground[None]),
            "centers": torch.from_numpy(centers[None]), "offsets": torch.from_numpy(offsets),
            "labels": torch.from_numpy(labels.copy())}


class GraspNetFrames:
    """Explicit per-sample augmentation seeds do not consume global RNG state."""

    def __init__(self, data_root, records, size=(240, 320), mode="rgbd", seed=20261021,
                 augment=False, depth_clip_m=2.0):
        if mode not in ("rgb", "rgbd"):
            raise ValueError("Unsupported input mode")
        self.root = Path(data_root)
        self.records = list(records)
        self.size = tuple(size)
        self.mode = mode
        self.seed = seed
        self.augment = augment
        self.depth_clip_m = depth_clip_m

    def __len__(self):
        return len(self.records)

    def get(self, index, epoch=0):
        record = self.records[index]
        paths = {key: self.root / value for key, value in record["files"].items()}
        target_size = (self.size[1], self.size[0])
        with Image.open(str(paths["rgb"])) as image:
            original_size = image.size
            rgb = np.asarray(image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        with Image.open(str(paths["depth"])) as image:
            if image.size != original_size:
                raise ValueError("RGB and depth dimensions differ")
            depth = np.asarray(image.resize(target_size, Image.Resampling.NEAREST), dtype=np.float32)
        with Image.open(str(paths["label"])) as image:
            if image.size != original_size:
                raise ValueError("RGB and label dimensions differ")
            original_labels = np.asarray(image)
            if original_labels.ndim != 2 or original_labels.min() < 0 or original_labels.max() > 88:
                raise ValueError("Invalid source label values")
            labels = np.asarray(image.resize(target_size, Image.Resampling.NEAREST), dtype=np.int64)
        metadata = loadmat(str(paths["meta"]))
        factor = float(np.asarray(metadata["factor_depth"]).reshape(-1)[0])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid depth calibration factor")
        depth = depth / factor
        valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, np.minimum(depth, self.depth_clip_m) / self.depth_clip_m, 0).astype(np.float32)
        if self.augment:
            identity = "{}:{}:{}:{}".format(self.seed, epoch, record["scene"], record["frame"])
            sample_seed = int(hashlib.sha256(identity.encode("ascii")).hexdigest()[:8], 16)
            rng = np.random.RandomState(sample_seed)
            if rng.rand() < 0.5:
                rgb, depth, valid, labels = rgb[:, ::-1], depth[:, ::-1], valid[:, ::-1], labels[:, ::-1]
            rgb = np.clip(rgb * rng.uniform(0.9, 1.1), 0, 1)
        rgb = (rgb - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        channels = [rgb.transpose(2, 0, 1)]
        if self.mode == "rgbd":
            channels.extend([depth[None], valid.astype(np.float32)[None]])
        inputs = torch.from_numpy(np.ascontiguousarray(np.concatenate(channels, axis=0), dtype=np.float32))
        target = instance_targets(labels)
        return inputs, target

    def epoch_order(self, epoch):
        return np.random.RandomState((self.seed + epoch) % (2 ** 32)).permutation(len(self)).tolist()
