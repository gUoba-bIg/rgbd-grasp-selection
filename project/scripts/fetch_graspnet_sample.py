"""Fetch fixed-scene GraspNet image samples without downloading complete ZIPs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.remote_zip import RemoteZip, atomic_write, atomic_json, safe_destination

REVISION = "9073d1d4cb6c92272b6aa072545def5101e4fecb"
BASE_URL = "https://hf-mirror.com/saic3d/graspnet/resolve/" + REVISION
OFFICIAL = "https://graspnet.net/datasets.html"
MANIFEST = ROOT / "data/manifests/download_manifest.json"
DATA_ROOT = ROOT / "data/raw/graspnet"


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(minimum=False):
    protocol_path = ROOT / "data/manifests/scene_splits.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    train = protocol["pilot_train"][:2] if minimum else protocol["pilot_train"]
    validation = protocol["pilot_validation"][:1] if minimum else protocol["pilot_validation"]
    scenes = train + validation
    frames = list(range(0, 64 if minimum else 256, protocol["pilot_frame_step"]))
    if set(train) & set(validation) or not set(train).issubset(protocol["groups"]["train"]) or not set(validation).issubset(protocol["groups"]["validation"]):
        raise ValueError("Scene split inconsistency")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"version": 1, "files": {}, "archives": {}}
    manifest.update({"updated_at": now(), "status": "downloading", "dataset": "GraspNet-1Billion", "source_kind": "third_party_public_mirror", "mirror_repository": "https://huggingface.co/saic3d/graspnet", "transport_mirror": "https://hf-mirror.com", "mirror_revision": REVISION, "official_dataset_page": OFFICIAL, "license_source": OFFICIAL + "#license", "license_statement": "Official dataset page states free non-commercial use under BY-NC-SA; retain attribution to GraspNet team, MVIG, SJTU.", "integrity_scope": "Each selected ZIP member verified by archive CRC32 and local SHA256. No official full-archive hash comparison is available; mirror authenticity is not established by CRC.", "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "selection": {"train_scenes": train, "validation_scenes": validation, "frames": frames, "camera": "realsense"}, "raw_root": str(DATA_ROOT)})
    atomic_json(MANIFEST, manifest)
    needed = set()
    for scene in scenes:
        prefix = f"scenes/scene_{scene:04d}/"
        needed.add(prefix + "object_id_list.txt")
        for name in ("camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"):
            needed.add(prefix + "realsense/" + name)
        for frame in frames:
            for kind, suffix in (("rgb", "png"), ("depth", "png"), ("label", "png"), ("meta", "mat")):
                needed.add(prefix + f"realsense/{kind}/{frame:04d}.{suffix}")
    remaining = set(needed)
    try:
        for number in range(1, 5):
            archive_name = f"train_{number}.zip"
            url = BASE_URL + "/" + archive_name
            print("Index", archive_name, flush=True)
            archive = RemoteZip(url, ROOT / "data/downloads/zip_ranges")
            members = {(name if name.startswith("scenes/") else "scenes/" + name): name for name in archive.entries}
            found = sorted(remaining & set(members))
            manifest["archives"][archive_name] = {"url": url, "total_bytes": archive.reader.size, "etag": archive.reader.etag, "entry_count": len(archive.entries), "selected_members": len(found), "range_supported": True}
            atomic_json(MANIFEST, manifest)
            print("Selected", len(found), "of", len(archive.entries), "entries", flush=True)

            def one(name):
                member = members[name]
                info = archive.entries[member]
                destination = safe_destination(DATA_ROOT, name)
                old = manifest["files"].get(name)
                if destination.exists():
                    data = destination.read_bytes()
                    checksum = hashlib.sha256(data).hexdigest()
                    if len(data) == info.file_size and (zlib.crc32(data) & 0xFFFFFFFF) == info.CRC and (old is None or old["sha256"] == checksum):
                        return name, {"bytes": len(data), "sha256": checksum, "crc32": f"{info.CRC:08x}", "archive": archive_name, "source_member": member, "verified_at": now()}, False
                data = archive.read(member)
                atomic_write(destination, data)
                return name, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "crc32": f"{info.CRC:08x}", "archive": archive_name, "source_member": member, "verified_at": now()}, True

            # Modest parallelism; each finished file is committed independently.
            with ThreadPoolExecutor(max_workers=4) as pool:
                for index, (name, record, downloaded) in enumerate(pool.map(one, found), 1):
                    manifest["files"][name] = record
                    remaining.discard(name)
                    manifest["updated_at"] = now()
                    atomic_json(MANIFEST, manifest)
                    if index % 16 == 0 or index == len(found):
                        print(f"{archive_name}: {index}/{len(found)} checked; {len(remaining)} remaining", flush=True)
            if not remaining:
                break
        if remaining:
            raise RuntimeError("Selected members absent from archives: " + str(sorted(remaining)[:10]))
        manifest["status"] = "download_complete_pending_content_audit"
        manifest["updated_at"] = now()
        atomic_json(MANIFEST, manifest)
        print("Sample download complete", len(needed), "files", flush=True)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        manifest["remaining_files"] = len(remaining)
        manifest["updated_at"] = now()
        atomic_json(MANIFEST, manifest)
        raise


def audit():
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    rows = []
    errors = []
    for split in ("train", "validation"):
        for scene in selection[split + "_scenes"]:
            base = DATA_ROOT / f"scenes/scene_{scene:04d}/realsense"
            intrinsics = np.load(base / "camK.npy", allow_pickle=False)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                errors.append(f"Invalid camera intrinsics: {scene}")
            for frame in selection["frames"]:
                arrays = {kind: np.asarray(Image.open(base / kind / f"{frame:04d}.png")) for kind in ("rgb", "depth", "label")}
                rgb, depth, label = (arrays[kind] for kind in ("rgb", "depth", "label"))
                meta = loadmat(base / "meta" / f"{frame:04d}.mat")
                factor = float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
                labels = np.unique(label).tolist()
                objects = (np.asarray(meta["cls_indexes"]).reshape(-1)).tolist()
                if rgb.shape != (720, 1280, 3) or depth.shape != (720, 1280) or label.shape != depth.shape:
                    errors.append(f"Unexpected image shapes: {scene}/{frame}")
                if not np.isfinite(factor) or factor <= 0 or label.max() > 88 or label.min() < 0 or not set(labels).difference({0}).issubset(set(objects)):
                    errors.append(f"Invalid depth/labels/objects: {scene}/{frame}")
                if "intrinsic_matrix" not in meta or not np.allclose(intrinsics, meta["intrinsic_matrix"], rtol=1e-4, atol=1e-3):
                    errors.append(f"Camera intrinsics mismatch: {scene}/{frame}")
                if not np.any(depth > 0) or not np.any(label > 0):
                    errors.append(f"Empty depth/labels: {scene}/{frame}")
                rows.append({"split": split, "scene": scene, "frame": frame, "image_shape": list(rgb.shape), "depth_dtype": str(depth.dtype), "factor_depth": factor, "visible_labels": labels, "meta_objects": objects, "valid_depth_fraction": float(np.mean(depth > 0))})
    hash_failures = []
    for name, entry in manifest["files"].items():
        data = safe_destination(DATA_ROOT, name).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != entry["crc32"]:
            hash_failures.append(name)
    result = {"created_at": now(), "status": "passed" if not errors and not hash_failures else "failed", "scope": "Engineering pilot sample only; no grasp AP or real robot success claim", "source_kind": manifest["source_kind"], "integrity_scope": manifest["integrity_scope"], "raw_root": str(DATA_ROOT), "camera": selection["camera"], "train_scenes": selection["train_scenes"], "validation_scenes": selection["validation_scenes"], "frames_per_scene": len(selection["frames"]), "train_frames": sum(row["split"] == "train" for row in rows), "validation_frames": sum(row["split"] == "validation" for row in rows), "files_verified": len(manifest["files"]), "bytes_verified": sum(entry["bytes"] for entry in manifest["files"].values()), "errors": errors, "hash_failures": hash_failures, "frames": rows}
    atomic_json(ROOT / "results/P2/data_audit.json", result)
    if errors or hash_failures:
        raise RuntimeError("Content audit failed: " + str(errors + hash_failures))
    manifest["status"] = "verified_engineering_sample"
    manifest["updated_at"] = now()
    manifest.pop("error", None)
    manifest.pop("remaining_files", None)
    atomic_json(MANIFEST, manifest)
    print("Audit passed:", result["train_frames"], "train /", result["validation_frames"], "validation frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum", action="store_true", help="2 train + 1 validation scenes, 4 frames each")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        fetch(args.minimum)
    if not args.skip_audit:
        audit()
