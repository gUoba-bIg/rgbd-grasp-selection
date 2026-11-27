"""Fetch the first five preregistered remaining validation scenes with ZIP ranges."""
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
MANIFEST = ROOT / "data/manifests/confirmation_v6.json"
DATA_ROOT = ROOT / "data/raw/graspnet_confirmation_v6"
CACHE = ROOT / "data/downloads/zip_ranges"
AUDIT = ROOT / "results/P2/confirmation_v6_data_audit.json"
SPLIT = ROOT / "data/manifests/scene_splits.json"
RUNTIME = ROOT / "runtime/confirmation_v6"
RECEIPT = RUNTIME / "audit_commit.json"
EXPECTED_VALIDATION = [2, 6, 7, 11, 15]
PREVIOUSLY_USED = [13, 18, 22, 33, 39]
REMAINING_UNOPENED = [52, 59, 75, 79, 80]
FRAME_IDS = list(range(0, 256, 16))
KINDS = {"rgb": "png", "depth": "png", "label": "png", "meta": "mat"}
RESERVE_BYTES = 60 * 2 ** 30
COMPLETE = "verified_confirmation_sample"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def source_hashes():
    return {"scripts/fetch_graspnet_confirmation_v6.py": sha256(Path(__file__)),
            "src/remote_zip.py": sha256(ROOT / "src/remote_zip.py")}


def selection_from_split(protocol):
    groups = protocol["groups"]
    validation = groups["validation"]
    if len(validation) != len(set(validation)) or sorted(protocol["pilot_validation"]) != PREVIOUSLY_USED:
        raise ValueError("Fixed validation membership or previously used scenes changed")
    remaining = sorted(set(validation) - set(PREVIOUSLY_USED))
    if remaining[:5] != EXPECTED_VALIDATION or remaining[5:] != REMAINING_UNOPENED:
        raise ValueError("Ascending unused validation selection differs from the frozen scene list")
    if not set(PREVIOUSLY_USED).issubset(validation):
        raise ValueError("Previously used validation scenes missing from split")
    if protocol["pilot_frame_step"] != 16 or protocol["camera_primary"] != "realsense":
        raise ValueError("Frozen frame step or primary camera changed")
    for group, scenes in groups.items():
        if group != "validation" and set(validation) & set(scenes):
            raise ValueError("Validation scene appears in another split: " + group)
    return {"train_scenes": [], "validation_scenes": list(EXPECTED_VALIDATION),
            "frames": list(FRAME_IDS), "camera": "realsense",
            "previously_used_validation_scenes": list(PREVIOUSLY_USED),
            "remaining_unopened_validation_scenes": list(REMAINING_UNOPENED),
            "selection_rule": "ascending fixed validation minus previously used; first five"}


def selected_paths(selection):
    paths = set()
    for scene in selection["validation_scenes"]:
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
    for scene in selection["validation_scenes"]:
        for frame in selection["frames"]:
            paths = {kind: f"scenes/scene_{scene:04d}/realsense/{kind}/{frame:04d}.{suffix}"
                     for kind, suffix in KINDS.items()}
            if all(path in files for path in paths.values()):
                records["validation"].append({"scene": scene, "frame": f"{frame:04d}",
                    "camera": "realsense", "files": paths,
                    "sha256": {kind: files[path]["sha256"] for kind, path in paths.items()},
                    "validation_exposure": "not_used_for_model_selection_before_round6"})
    return records


def disk_guard(root=ROOT, additional_bytes=64 * 1024 * 1024):
    if shutil.disk_usage(root).free < RESERVE_BYTES + additional_bytes:
        raise RuntimeError("Insufficient disk to preserve the 60 GiB reserve")


@contextmanager
def process_lock():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / ".download.lock").open("a+b") as stream:
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
                              ("mirror_revision", REVISION), ("data_root", str(DATA_ROOT)),
                              ("code_sha256", source_hashes())):
            if manifest.get(key) != expected:
                raise ValueError("Existing confirmation manifest differs: " + key)
        if not set(manifest["files"]).issubset(selected_paths(selection)):
            raise ValueError("Manifest contains paths outside the frozen selection")
        return manifest
    return {"schema": 6, "dataset": "GraspNet-1Billion", "status": "initialized", "created_at": now(),
            "source_kind": "third_party_public_mirror", "mirror_revision": REVISION,
            "mirror_repository": "https://huggingface.co/saic3d/graspnet", "transport_mirror": "https://hf-mirror.com",
            "official_dataset_page": OFFICIAL, "license_source": OFFICIAL + "#license",
            "license_statement": "Official dataset page states free non-commercial use under BY-NC-SA; retain attribution to GraspNet team, MVIG, SJTU.",
            "integrity_scope": "Selected members checked against ZIP CRC32 and recorded SHA256; no official full-archive hash comparison, so mirror authenticity is not established by these checks.",
            "data_root": str(DATA_ROOT), "split_sha256": split_sha256, "selection": selection,
            "records": {"train": [], "validation": []}, "files": {}, "archives": {},
            "scope": "Fixed-model internal validation confirmation only, not official test evaluation",
            "validation_exposure_note": "These scenes were not used for model selection before round6. After first inference they are development validation, not permanently blind data.",
            "reserve_gib": 60, "full_archive_fallback_allowed": False, "code_sha256": source_hashes()}


def verified_bytes(data, info, recorded_sha=None):
    return (len(data) == info.file_size and zlib.crc32(data) & 0xFFFFFFFF == info.CRC and
            (recorded_sha is None or hashlib.sha256(data).hexdigest() == recorded_sha))


def fetch_member(name, member, archive_name, archive, previous=None):
    info = archive.entries[member]
    destination = safe_destination(DATA_ROOT, name)
    expected_sha = previous.get("sha256") if previous else None
    if destination.exists() and verified_bytes(destination.read_bytes(), info, expected_sha):
        data = destination.read_bytes()
        origin = previous.get("origin", "existing_verified") if previous else "existing_verified"
    else:
        disk_guard()
        data = archive.read(member)
        if not verified_bytes(data, info, expected_sha):
            raise ValueError("Remote member failed size/CRC/SHA validation")
        disk_guard(additional_bytes=64 * 1024 * 1024 + len(data))
        atomic_write(destination, data)
        origin = "downloaded_range"
    checksum = hashlib.sha256(data).hexdigest()
    if sha256(destination) != checksum:
        raise ValueError("Committed file checksum mismatch")
    return name, {"bytes": len(data), "compressed_bytes": info.compress_size, "sha256": checksum,
                  "crc32": f"{info.CRC:08x}", "archive": archive_name, "source_member": member,
                  "origin": origin, "verified_at": now()}


def commit_manifest(manifest):
    manifest["records"] = build_records(manifest["selection"], manifest["files"])
    manifest["updated_at"] = now()
    atomic_json(MANIFEST, manifest)


def fetch(workers=4):
    if workers not in range(1, 5):
        raise ValueError("Workers must be between one and four")
    selection = selection_from_split(json.loads(SPLIT.read_text(encoding="utf-8-sig")))
    manifest = initialize_manifest(selection, sha256(SPLIT))
    if manifest["status"] == COMPLETE:
        audit()
        print("Confirmation sample already verified; frozen manifest/audit preserved", flush=True)
        return manifest
    needed = selected_paths(selection)
    if manifest["status"] == "download_complete_pending_content_audit" and set(manifest["files"]) == needed:
        return manifest
    manifest["status"] = "downloading"
    manifest.pop("error", None)
    commit_manifest(manifest)
    remaining = set(needed)
    try:
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
            if not remaining:
                break
        if remaining:
            raise RuntimeError("Selected paths absent from fixed source: " + str(sorted(remaining)[:10]))
        manifest["estimated_selected_bytes"] = sum(item["selected_uncompressed_bytes"] for item in manifest["archives"].values())
        manifest["estimated_selected_compressed_bytes"] = sum(item["selected_compressed_bytes"] for item in manifest["archives"].values())
        disk_guard(additional_bytes=manifest["estimated_selected_bytes"] + 64 * 1024 * 1024)
        commit_manifest(manifest)
        print("Selected totals:", len(needed), "files;", manifest["estimated_selected_bytes"], "raw bytes;",
              manifest["estimated_selected_compressed_bytes"], "compressed bytes", flush=True)
        for archive_name, archive, members, found in indexed:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [pool.submit(fetch_member, name, members[name], archive_name, archive,
                                    manifest["files"].get(name)) for name in found]
                for future in as_completed(jobs):
                    name, record = future.result()
                    manifest["files"][name] = record
                    commit_manifest(manifest)
                    count = len(manifest["files"])
                    if count % 16 == 0 or count == len(needed):
                        print(f"Committed {count}/{len(needed)} files; frames={len(manifest['records']['validation'])}", flush=True)
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


def inspect_auxiliary(intrinsics, camera_poses, table_pose, object_ids):
    import numpy as np
    errors = []
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or
            intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        errors.append("invalid_scene_intrinsics")
    if (camera_poses.shape != (256, 4, 4) or not np.isfinite(camera_poses).all() or
            not np.allclose(camera_poses[:, 3, :], [0, 0, 0, 1], atol=1e-5)):
        errors.append("invalid_camera_poses")
    if (table_pose.shape != (4, 4) or not np.isfinite(table_pose).all() or
            not np.allclose(table_pose[3], [0, 0, 0, 1], atol=1e-5)):
        errors.append("invalid_table_pose")
    if not object_ids or len(set(object_ids)) != len(object_ids) or min(object_ids) < 0 or max(object_ids) > 87:
        errors.append("invalid_object_id_list")
    return errors


def inspect_frame(rgb, depth, label, meta, intrinsics, object_ids):
    import numpy as np
    errors = []
    if rgb.shape != (720, 1280, 3) or depth.shape != (720, 1280) or label.shape != (720, 1280):
        errors.append("unexpected_image_shapes")
    if rgb.dtype != np.uint8 or not np.issubdtype(depth.dtype, np.integer) or not np.issubdtype(label.dtype, np.integer):
        errors.append("unexpected_image_dtypes")
    factor_array = np.asarray(meta["factor_depth"])
    factor = float(factor_array.reshape(-1)[0])
    labels = np.unique(label).tolist()
    object_array = np.asarray(meta["cls_indexes"])
    objects = object_array.reshape(-1).tolist()
    if factor_array.size != 1 or not np.isfinite(factor) or factor <= 0:
        errors.append("invalid_depth_factor")
    if depth.size == 0 or depth.min() < 0 or not np.any(depth > 0):
        errors.append("invalid_or_empty_depth")
    if label.size == 0 or label.min() < 0 or label.max() > 88 or not np.any(label > 0):
        errors.append("invalid_or_empty_labels")
    if (not np.issubdtype(object_array.dtype, np.integer) or not objects or
            len(set(objects)) != len(objects) or not set(objects).issubset({item + 1 for item in object_ids})):
        errors.append("invalid_metadata_objects")
    if not set(labels).difference({0}).issubset(objects):
        errors.append("labels_not_in_metadata_objects")
    matrix = np.asarray(meta["intrinsic_matrix"])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(intrinsics, matrix, rtol=1e-4, atol=1e-3):
        errors.append("camera_intrinsics_mismatch")
    poses = np.asarray(meta["poses"])
    if poses.shape != (3, 4, len(objects)) or not np.isfinite(poses).all():
        errors.append("invalid_metadata_poses")
    row = {"image_shape": list(rgb.shape), "depth_dtype": str(depth.dtype), "factor_depth": factor,
           "visible_labels": labels, "meta_objects": objects, "valid_depth_fraction": float(np.mean(depth > 0))}
    return errors, row


def inspect_contents(manifest):
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat
    errors, rows, hash_failures = [], [], []
    for name, info in manifest["files"].items():
        try:
            data = safe_destination(DATA_ROOT, name).read_bytes()
            if (len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"] or
                    f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != info["crc32"]):
                hash_failures.append(name)
        except OSError:
            hash_failures.append(name)
    for scene in manifest["selection"]["validation_scenes"]:
        scene_root = DATA_ROOT / f"scenes/scene_{scene:04d}"
        base = scene_root / "realsense"
        try:
            intrinsics = np.load(base / "camK.npy", allow_pickle=False)
            poses = np.load(base / "camera_poses.npy", allow_pickle=False)
            table = np.load(base / "cam0_wrt_table.npy", allow_pickle=False)
            object_ids = [int(value) for value in (scene_root / "object_id_list.txt").read_text(encoding="utf-8").split()]
            errors.extend(f"{scene}: {message}" for message in inspect_auxiliary(intrinsics, poses, table, object_ids))
        except (ValueError, KeyError, OSError, IndexError, TypeError, EOFError) as exc:
            errors.append(f"{scene}: invalid_auxiliary: {type(exc).__name__}: {exc}")
            continue
        for frame in manifest["selection"]["frames"]:
            try:
                arrays = {}
                for kind in ("rgb", "depth", "label"):
                    with Image.open(base / kind / f"{frame:04d}.png") as image:
                        arrays[kind] = np.asarray(image)
                meta = loadmat(base / "meta" / f"{frame:04d}.mat")
                found, row = inspect_frame(arrays["rgb"], arrays["depth"], arrays["label"], meta, intrinsics, object_ids)
                errors.extend(f"{scene}/{frame}: {message}" for message in found)
                row.update({"split": "validation", "scene": scene, "frame": f"{frame:04d}"})
                rows.append(row)
            except (ValueError, KeyError, OSError, IndexError, TypeError, EOFError) as exc:
                errors.append(f"{scene}/{frame}: {type(exc).__name__}: {exc}")
        print("Content audited validation", scene, flush=True)
    return errors, rows, hash_failures


def load_audit_receipt(manifest_sha):
    """Validate the journal binding before either replay or completed reentry."""
    if not RECEIPT.exists():
        raise RuntimeError("Completed manifest lacks a consistent audit and recovery receipt")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    report = receipt["report"]
    if (receipt.get("manifest_sha256") != manifest_sha or report.get("manifest_sha256") != manifest_sha or
            report.get("status") != "passed" or report.get("code_sha256") != source_hashes() or
            hashlib.sha256(json_bytes(report)).hexdigest() != receipt.get("report_sha256")):
        raise RuntimeError("Audit recovery receipt does not bind the completed manifest/report")
    return receipt


def recover_audit_receipt(manifest_sha):
    """Repair only the interrupted final manifest -> audit commit, never data."""
    receipt = load_audit_receipt(manifest_sha)
    report = receipt["report"]
    if AUDIT.exists() and json.loads(AUDIT.read_text(encoding="utf-8")).get("status") == "passed":
        raise RuntimeError("Refusing to overwrite a different completed audit")
    atomic_json(AUDIT, report)
    return report


def audit():
    selection = selection_from_split(json.loads(SPLIT.read_text(encoding="utf-8-sig")))
    manifest = initialize_manifest(selection, sha256(SPLIT))
    if set(manifest["files"]) != selected_paths(selection):
        raise ValueError("Cannot audit incomplete or extra selected files")
    errors, rows, hash_failures = inspect_contents(manifest)
    records = build_records(selection, manifest["files"])
    if records["train"] or len(records["validation"]) != 80 or len(rows) != 80:
        errors.append("Incomplete confirmation frame counts")
    was_completed = manifest["status"] == COMPLETE
    if was_completed:
        if errors or hash_failures or manifest["records"] != records:
            raise RuntimeError("Completed sample integrity failed; frozen manifest/data were not changed: " + str((errors + hash_failures)[:10]))
        if AUDIT.exists():
            previous = json.loads(AUDIT.read_text(encoding="utf-8"))
            if previous.get("status") == "passed" and previous.get("manifest_sha256") == sha256(MANIFEST):
                if previous.get("code_sha256") != source_hashes():
                    raise RuntimeError("Completed audit code provenance differs")
                receipt = load_audit_receipt(sha256(MANIFEST))
                if sha256(AUDIT) != receipt["report_sha256"]:
                    raise RuntimeError("Completed audit bytes differ from their committed receipt")
                print("Completed confirmation sample rechecked; manifest/audit bytes preserved", flush=True)
                return previous
        return recover_audit_receipt(sha256(MANIFEST))
    manifest["records"] = records
    manifest["status"] = "content_audit_failed" if errors or hash_failures else COMPLETE
    manifest["updated_at"] = now()
    manifest.pop("error", None)
    manifest_sha = hashlib.sha256(json_bytes(manifest)).hexdigest()
    report = {"schema": 6, "created_at": now(), "status": "failed" if errors or hash_failures else "passed",
              "data_root": str(DATA_ROOT), "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": manifest_sha,
              "split_sha256": manifest["split_sha256"], "source_kind": manifest["source_kind"],
              "mirror_revision": REVISION, "integrity_scope": manifest["integrity_scope"], "selection": selection,
              "train_frames": 0, "validation_frames": len(records["validation"]), "scene_count": 5,
              "files_verified": len(manifest["files"]) - len(hash_failures),
              "bytes_selected": sum(info["bytes"] for info in manifest["files"].values()),
              "bytes_verified": sum(info["bytes"] for name, info in manifest["files"].items() if name not in hash_failures),
              "free_disk_gib": shutil.disk_usage(ROOT).free / 2 ** 30, "errors": errors, "hash_failures": hash_failures,
              "frames": rows, "scope": "Transfer and content checks only; no training or inference performed",
              "code_sha256": source_hashes()}
    # Durable journal bridges the two atomic files. A restart after the manifest
    # commit revalidates every file before replaying this exact report.
    if report["status"] == "passed":
        atomic_json(RECEIPT, {"schema": 6, "manifest_sha256": manifest_sha, "report": report,
                             "report_sha256": hashlib.sha256(json_bytes(report)).hexdigest()})
    atomic_json(MANIFEST, manifest)
    atomic_json(AUDIT, report)
    if errors or hash_failures:
        raise RuntimeError("Confirmation content audit failed: " + str((errors + hash_failures)[:10]))
    print("Confirmation audit passed: 0 train, 80 validation, 5 scenes, 340 files", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.audit_only and args.skip_audit:
        parser.error("Cannot combine --audit-only and --skip-audit")
    with process_lock():
        disk_guard()
        if not args.audit_only:
            fetch(args.workers)
        if not args.skip_audit:
            audit()
