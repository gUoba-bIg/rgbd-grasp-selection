"""Fetch and audit frozen expanded GraspNet frames using bounded ZIP ranges only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from src.remote_zip import RemoteZip, atomic_json, atomic_write, safe_destination

REVISION = "9073d1d4cb6c92272b6aa072545def5101e4fecb"
BASE_URL = "https://hf-mirror.com/saic3d/graspnet/resolve/" + REVISION
OFFICIAL = "https://graspnet.net/datasets.html"
MANIFEST = ROOT / "data/manifests/expanded_v3.json"
DATA_ROOT = ROOT / "data/raw/graspnet_expanded_v3"
OLD_ROOT = ROOT / "data/raw/graspnet"
CACHE = ROOT / "data/downloads/zip_ranges"
AUDIT = ROOT / "results/P2/expanded_v3_data_audit.json"
SPLIT = ROOT / "data/manifests/scene_splits.json"
EXPECTED_TRAIN = [3, 12, 30, 31, 34, 38, 42, 43, 57, 93]
EXPECTED_VALIDATION = [13, 18, 22, 33, 39]
FRAME_IDS = list(range(0, 256, 16))
KINDS = {"rgb": "png", "depth": "png", "label": "png", "meta": "mat"}
RESERVE_BYTES = 60 * 2 ** 30


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_from_split(protocol):
    if protocol["pilot_train"] != EXPECTED_TRAIN or protocol["pilot_validation"] != EXPECTED_VALIDATION:
        raise ValueError("Expanded selection must use the exact previously declared pilot scenes")
    if protocol["pilot_frame_step"] != 16:
        raise ValueError("Frozen frame step changed")
    train, validation = set(EXPECTED_TRAIN), set(EXPECTED_VALIDATION)
    if train & validation or not train.issubset(protocol["groups"]["train"]) or not validation.issubset(protocol["groups"]["validation"]):
        raise ValueError("Scene split overlap or membership mismatch")
    for group, scenes in protocol["groups"].items():
        if group != "train" and train & set(scenes):
            raise ValueError("Training scene appears in another split")
        if group != "validation" and validation & set(scenes):
            raise ValueError("Validation scene appears in another split")
    return {"train_scenes": list(EXPECTED_TRAIN), "validation_scenes": list(EXPECTED_VALIDATION),
            "frames": list(FRAME_IDS), "camera": "realsense",
            "validation_groups": {"previously_seen": [13], "fresh": [18, 22, 33, 39]}}


def selected_paths(selection):
    paths = set()
    for scene in selection["train_scenes"] + selection["validation_scenes"]:
        base = f"scenes/scene_{scene:04d}"
        paths.add(base + "/object_id_list.txt")
        for name in ("camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"):
            paths.add(base + "/realsense/" + name)
        for frame in selection["frames"]:
            for kind, suffix in KINDS.items():
                paths.add(base + f"/realsense/{kind}/{frame:04d}.{suffix}")
    return paths


def build_records(selection, files):
    records = {"train": [], "validation": []}
    for split in records:
        for scene in selection[split + "_scenes"]:
            for frame in selection["frames"]:
                paths = {kind: f"scenes/scene_{scene:04d}/realsense/{kind}/{frame:04d}.{suffix}"
                         for kind, suffix in KINDS.items()}
                if not all(path in files for path in paths.values()):
                    continue
                record = {"scene": scene, "frame": f"{frame:04d}", "camera": "realsense",
                          "files": paths, "sha256": {kind: files[path]["sha256"] for kind, path in paths.items()}}
                if split == "validation":
                    record["validation_exposure"] = "previously_seen" if scene == 13 else "fresh"
                records[split].append(record)
    return records


def disk_guard(root=ROOT, additional_bytes=64 * 1024 * 1024):
    # Additional margin covers concurrent selected member writes and manifest commits.
    if shutil.disk_usage(root).free < RESERVE_BYTES + additional_bytes:
        raise RuntimeError("Insufficient disk to preserve the 60 GiB reserve")


@contextmanager
def process_lock():
    directory = ROOT / "runtime/expanded_v3"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def initialize_manifest(selection, split_sha256):
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key, expected in (("selection", selection), ("split_sha256", split_sha256),
                              ("mirror_revision", REVISION), ("data_root", str(DATA_ROOT))):
            if manifest.get(key) != expected:
                raise ValueError("Existing expanded manifest differs: " + key)
        return manifest
    return {"schema": 3, "dataset": "GraspNet-1Billion", "status": "initialized", "created_at": now(),
            "source_kind": "third_party_public_mirror", "mirror_revision": REVISION,
            "mirror_repository": "https://huggingface.co/saic3d/graspnet", "transport_mirror": "https://hf-mirror.com",
            "official_dataset_page": OFFICIAL, "license_source": OFFICIAL + "#license",
            "license_statement": "Official dataset page states free non-commercial use under BY-NC-SA; retain attribution to GraspNet team, MVIG, SJTU.",
            "integrity_scope": "Selected members checked against ZIP CRC32 and recorded SHA256; no official full-archive hash comparison, so mirror authenticity is not established by these checks.",
            "data_root": str(DATA_ROOT), "split_sha256": split_sha256, "selection": selection,
            "records": {"train": [], "validation": []}, "files": {}, "archives": {},
            "scope": "Expanded development subset, not an official test-set evaluation",
            "validation_exposure_note": "Scene 13 was previously evaluated in engineering pilots; scenes 18,22,33,39 have not been used for model evaluation before this expansion.",
            "reserve_gib": 60, "full_archive_fallback_allowed": False}


def verified_bytes(data, info, recorded_sha=None):
    return (len(data) == info.file_size and zlib.crc32(data) & 0xFFFFFFFF == info.CRC and
            (recorded_sha is None or hashlib.sha256(data).hexdigest() == recorded_sha))


def fetch_member(name, member, archive_name, archive, previous, old_files):
    info = archive.entries[member]
    destination = safe_destination(DATA_ROOT, name)
    old_path = safe_destination(OLD_ROOT, name)
    origin = "downloaded_range"
    data = None
    if destination.exists():
        candidate = destination.read_bytes()
        if verified_bytes(candidate, info, previous.get("sha256") if previous else None):
            data, origin = candidate, "existing_verified"
    if data is None and old_path.exists():
        candidate = old_path.read_bytes()
        old = old_files.get(name)
        if verified_bytes(candidate, info, old.get("sha256") if old else None):
            data, origin = candidate, "copied_previous_sample"
    if data is None:
        disk_guard()
        data = archive.read(member)
        if not verified_bytes(data, info):
            raise ValueError("Remote member failed size/CRC validation")
    if origin != "existing_verified":
        disk_guard(additional_bytes=64 * 1024 * 1024 + len(data))
        atomic_write(destination, data)
    # Verify committed bytes, not only the source buffer.
    checksum = hashlib.sha256(data).hexdigest()
    if sha256(destination) != checksum:
        raise ValueError("Committed file checksum mismatch")
    return name, {"bytes": len(data), "compressed_bytes": info.compress_size,
                  "sha256": checksum, "crc32": f"{info.CRC:08x}",
                  "archive": archive_name, "source_member": member,
                  "origin": previous.get("origin", origin) if previous and origin == "existing_verified" else origin,
                  "verified_at": now()}


def commit_manifest(manifest):
    manifest["records"] = build_records(manifest["selection"], manifest["files"])
    manifest["updated_at"] = now()
    atomic_json(MANIFEST, manifest)


def fetch(workers=4):
    protocol = json.loads(SPLIT.read_text(encoding="utf-8-sig"))
    selection = selection_from_split(protocol)
    manifest = initialize_manifest(selection, sha256(SPLIT))
    if manifest["status"] == "verified_expanded_sample":
        # The completed manifest is an immutable training input. Recheck every
        # local byte and content field without rewriting timestamps or hashes.
        audit()
        print("Expanded sample already verified; completed manifest preserved byte-for-byte", flush=True)
        return manifest
    manifest["status"] = "downloading"
    manifest.pop("error", None)
    commit_manifest(manifest)
    needed, remaining = selected_paths(selection), selected_paths(selection)
    old_manifest = ROOT / "data/manifests/download_manifest.json"
    old_files = json.loads(old_manifest.read_text(encoding="utf-8"))["files"] if old_manifest.exists() else {}
    try:
        # Index first, providing an exact selected-byte estimate before transfer.
        indexed = []
        for number in range(1, 5):
            disk_guard()
            archive_name = f"train_{number}.zip"
            print("Index", archive_name, flush=True)
            archive = RemoteZip(BASE_URL + "/" + archive_name, CACHE)
            members = {(name if name.startswith("scenes/") else "scenes/" + name): name for name in archive.entries}
            found = sorted(remaining & members.keys())
            remaining.difference_update(found)
            manifest["archives"][archive_name] = {"url": BASE_URL + "/" + archive_name,
                "total_archive_bytes": archive.reader.size, "etag": archive.reader.etag,
                "entry_count": len(archive.entries), "selected_members": len(found), "range_supported": True,
                "selected_uncompressed_bytes": sum(archive.entries[members[name]].file_size for name in found),
                "selected_compressed_bytes": sum(archive.entries[members[name]].compress_size for name in found)}
            indexed.append((archive_name, archive, members, found))
            commit_manifest(manifest)
            print(archive_name, "selected", len(found), "members", flush=True)
        if remaining:
            raise RuntimeError("Selected paths absent from fixed source: " + str(sorted(remaining)[:10]))
        manifest["estimated_selected_bytes"] = sum(item["selected_uncompressed_bytes"] for item in manifest["archives"].values())
        manifest["estimated_selected_compressed_bytes"] = sum(item["selected_compressed_bytes"] for item in manifest["archives"].values())
        commit_manifest(manifest)
        print("Selected totals:", len(needed), "files;", manifest["estimated_selected_bytes"], "raw bytes;",
              manifest["estimated_selected_compressed_bytes"], "compressed bytes", flush=True)
        for archive_name, archive, members, found in indexed:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [pool.submit(fetch_member, name, members[name], archive_name, archive,
                                    manifest["files"].get(name), old_files) for name in found]
                for future in as_completed(jobs):
                    name, record = future.result()
                    manifest["files"][name] = record
                    commit_manifest(manifest)
                    count = len(manifest["files"])
                    if count % 16 == 0 or count == len(needed):
                        print(f"Committed {count}/{len(needed)} files; frames train={len(manifest['records']['train'])}, validation={len(manifest['records']['validation'])}", flush=True)
            manifest["archives"][archive_name]["network_bytes_this_invocation"] = archive.reader.network_bytes
            commit_manifest(manifest)
        if set(manifest["files"]) != needed:
            raise RuntimeError("Committed file set differs from frozen selection")
        manifest["status"] = "download_complete_pending_content_audit"
        commit_manifest(manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        commit_manifest(manifest)
        raise
    return manifest


def inspect_frame(rgb, depth, label, meta, intrinsics):
    import numpy as np
    errors = []
    if rgb.shape != (720, 1280, 3) or depth.shape != (720, 1280) or label.shape != (720, 1280):
        errors.append("unexpected_image_shapes")
    if rgb.dtype != np.uint8 or not np.issubdtype(depth.dtype, np.integer) or not np.issubdtype(label.dtype, np.integer):
        errors.append("unexpected_image_dtypes")
    factor = float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
    labels = np.unique(label).tolist()
    objects = np.asarray(meta["cls_indexes"]).reshape(-1).tolist()
    if not np.isfinite(factor) or factor <= 0:
        errors.append("invalid_depth_factor")
    if depth.min() < 0 or not np.any(depth > 0):
        errors.append("invalid_or_empty_depth")
    if label.min() < 0 or label.max() > 88 or not np.any(label > 0):
        errors.append("invalid_or_empty_labels")
    if not set(labels).difference({0}).issubset(objects):
        errors.append("labels_not_in_metadata_objects")
    if "intrinsic_matrix" not in meta or np.asarray(meta["intrinsic_matrix"]).shape != (3, 3) or not np.allclose(intrinsics, meta["intrinsic_matrix"], rtol=1e-4, atol=1e-3):
        errors.append("camera_intrinsics_mismatch")
    row = {"image_shape": list(rgb.shape), "depth_dtype": str(depth.dtype), "factor_depth": factor,
           "visible_labels": labels, "meta_objects": objects, "valid_depth_fraction": float(np.mean(depth > 0))}
    return errors, row


def audit():
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selection = selection_from_split(json.loads(SPLIT.read_text(encoding="utf-8-sig")))
    if manifest["selection"] != selection or manifest["split_sha256"] != sha256(SPLIT):
        raise ValueError("Split or selection changed before content audit")
    if manifest["mirror_revision"] != REVISION or manifest["data_root"] != str(DATA_ROOT):
        raise ValueError("Unexpected expanded source or root")
    if set(manifest["files"]) != selected_paths(selection):
        raise ValueError("Cannot audit incomplete or extra selected files")
    errors, rows, hash_failures = [], [], []
    for name, info in manifest["files"].items():
        data = safe_destination(DATA_ROOT, name).read_bytes()
        if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"] or f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != info["crc32"]:
            hash_failures.append(name)
    for split in ("train", "validation"):
        for scene in selection[split + "_scenes"]:
            base = DATA_ROOT / f"scenes/scene_{scene:04d}/realsense"
            intrinsics = np.load(base / "camK.npy", allow_pickle=False)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                errors.append(f"Invalid scene intrinsics: {scene}")
            for frame in selection["frames"]:
                try:
                    arrays = {}
                    for kind in ("rgb", "depth", "label"):
                        with Image.open(base / kind / f"{frame:04d}.png") as image:
                            arrays[kind] = np.asarray(image)
                    meta = loadmat(base / "meta" / f"{frame:04d}.mat")
                    found, row = inspect_frame(arrays["rgb"], arrays["depth"], arrays["label"], meta, intrinsics)
                    errors.extend(f"{scene}/{frame}: {message}" for message in found)
                    row.update({"split": split, "scene": scene, "frame": f"{frame:04d}"})
                    if split == "validation":
                        row["validation_exposure"] = "previously_seen" if scene == 13 else "fresh"
                    rows.append(row)
                except (ValueError, KeyError, OSError) as exc:
                    errors.append(f"{scene}/{frame}: {type(exc).__name__}: {exc}")
            print("Content audited", split, scene, flush=True)
    records = build_records(selection, manifest["files"])
    if len(records["train"]) != 160 or len(records["validation"]) != 80 or len(rows) != 240:
        errors.append("Incomplete expanded frame counts")
    was_completed = manifest["status"] == "verified_expanded_sample"
    if was_completed:
        if errors or hash_failures or manifest["records"] != records:
            raise RuntimeError("Completed expanded sample integrity failed; frozen manifest/data were not changed. Use a new version for changes: " + str((errors + hash_failures)[:10]))
        if AUDIT.exists():
            previous_audit = json.loads(AUDIT.read_text(encoding="utf-8"))
            if previous_audit.get("status") != "passed" or previous_audit.get("manifest_sha256") != sha256(MANIFEST):
                raise RuntimeError("Completed manifest has an inconsistent audit; no records were overwritten")
            print("Completed expanded sample rechecked; manifest and audit preserved byte-for-byte", flush=True)
            return previous_audit
    else:
        manifest["records"] = records
        manifest["status"] = "content_audit_failed" if errors or hash_failures else "verified_expanded_sample"
        if not errors and not hash_failures:
            manifest.pop("error", None)
        commit_manifest(manifest)
    report = {"schema": 3, "created_at": now(), "status": "failed" if errors or hash_failures else "passed",
              "data_root": str(DATA_ROOT), "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha256(MANIFEST),
              "split_sha256": manifest["split_sha256"], "source_kind": manifest["source_kind"],
              "mirror_revision": REVISION, "integrity_scope": manifest["integrity_scope"],
              "selection": selection, "train_frames": len(records["train"]), "validation_frames": len(records["validation"]),
              "fresh_validation_frames": 64, "previously_seen_validation_frames": 16,
              "files_verified": len(manifest["files"]), "bytes_verified": sum(info["bytes"] for info in manifest["files"].values()),
              "free_disk_gib": shutil.disk_usage(ROOT).free / 2 ** 30,
              "errors": errors, "hash_failures": hash_failures, "frames": rows,
              "scope": "Data transfer and content checks only; no training or model inference performed",
              "code_sha256": {"scripts/fetch_graspnet_expanded_v3.py": sha256(Path(__file__)), "src/remote_zip.py": sha256(ROOT / "src/remote_zip.py")}}
    atomic_json(AUDIT, report)
    if errors or hash_failures:
        raise RuntimeError("Expanded content audit failed: " + str((errors + hash_failures)[:10]))
    print("Expanded audit passed: 160 train, 80 validation (16 previously seen + 64 fresh), 1020 files", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.audit_only and args.skip_audit:
        parser.error("Cannot combine --audit-only and --skip-audit")
    for name in ("TEMP", "TMP"):
        os.environ[name] = str(ROOT / "runtime/temp")
    with process_lock():
        disk_guard()
        if not args.audit_only:
            fetch(args.workers)
        if not args.skip_audit:
            audit()
