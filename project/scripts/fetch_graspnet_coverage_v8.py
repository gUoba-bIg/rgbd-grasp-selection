"""Build the frozen coverage snapshot by verified copies and bounded ZIP ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import fetch_graspnet_confirmation_v6 as previous
from src.checkpoints import _exclusive_lock
from src.remote_zip import RemoteZip, safe_destination

CONFIG = ROOT / "configs/segmentation_coverage_v8.json"
MANIFEST = ROOT / "data/manifests/coverage_v8.json"
DATA_ROOT = ROOT / "data/raw/graspnet_coverage_v8"
AUDIT = ROOT / "results/P2/coverage_v8_data_audit.json"
RUNTIME = ROOT / "runtime/coverage_v8"
RECEIPT = RUNTIME / "audit_commit.json"
CACHE = ROOT / "data/downloads/zip_ranges"
REVISION, BASE_URL = previous.REVISION, previous.BASE_URL
RESERVE_BYTES = 60 * 2 ** 30
COMPLETE = "verified_coverage_sample"
CONTROL = [3, 12, 30, 31, 34, 38, 42, 43, 57, 93]
NEW_TRAIN = [4, 5, 8, 9, 10, 14, 16, 17, 19, 20]
VALIDATION = [2, 6, 7, 11, 13, 15, 18, 22, 33, 39]
UNOPENED = [52, 59, 75, 79, 80]
FRAMES = list(range(0, 256, 16))
KINDS = previous.KINDS
sha256, now = previous.sha256, previous.now


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError("Path must remain inside the project")
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path, payload):
    # Antivirus/indexer sharing locks can briefly prevent Windows replacement.
    # Retry only this explicit transient class, without ignoring a final failure.
    for attempt in range(5):
        try:
            previous.atomic_write(path, payload)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.1 * 2 ** attempt)


def atomic_json(path, value):
    atomic_write(path, json_bytes(value))


def source_hashes():
    names = ["scripts/fetch_graspnet_coverage_v8.py", "scripts/fetch_graspnet_confirmation_v6.py",
             "src/remote_zip.py", "src/checkpoints.py"]
    return {name: sha256(ROOT / name) for name in names}


def selection_from_protocol(protocol, split):
    expanded = sorted(CONTROL + NEW_TRAIN)
    if (protocol.get("schema") != 8 or protocol["arms"]["coverage_control"]["train_scenes"] != CONTROL or
            protocol["arms"]["coverage_expanded"]["train_scenes"] != expanded or
            protocol["new_train_scenes"] != NEW_TRAIN or protocol["validation_scenes"] != VALIDATION or
            protocol["remaining_unopened_validation_scenes"] != UNOPENED or
            protocol["camera"] != "realsense" or protocol["frames"] != FRAMES or
            protocol["free_disk_reserve_gib"] != 60):
        raise ValueError("Frozen coverage data protocol differs")
    groups = split["groups"]
    if split["pilot_train"] != CONTROL or sorted(set(groups["train"]) - set(CONTROL))[:10] != NEW_TRAIN:
        raise ValueError("New training scenes do not follow ascending fixed training IDs")
    if set(groups["validation"]) != set(VALIDATION + UNOPENED):
        raise ValueError("Validation membership changed")
    for group, chosen in (("train", expanded), ("validation", VALIDATION)):
        if not set(chosen).issubset(groups[group]):
            raise ValueError("Chosen scenes outside their fixed split")
        for other, scenes in groups.items():
            if other != group and set(chosen) & set(scenes):
                raise ValueError("Chosen scene appears in another split: " + other)
    return {"train_scenes": expanded, "control_train_scenes": list(CONTROL), "new_train_scenes": list(NEW_TRAIN),
            "validation_scenes": list(VALIDATION), "remaining_unopened_validation_scenes": list(UNOPENED),
            "frames": list(FRAMES), "camera": "realsense"}


def paths_for_scenes(scenes, frames=FRAMES):
    return previous.selected_paths({"validation_scenes": scenes, "frames": frames})


def selected_paths(selection):
    return paths_for_scenes(selection["train_scenes"] + selection["validation_scenes"], selection["frames"])


def build_records(selection, files):
    result = {"train": [], "validation": []}
    for split in result:
        for scene in selection[split + "_scenes"]:
            for frame in selection["frames"]:
                paths = {kind: f"scenes/scene_{scene:04d}/realsense/{kind}/{frame:04d}.{suffix}" for kind, suffix in KINDS.items()}
                if all(name in files for name in paths.values()):
                    result[split].append({"scene": scene, "frame": f"{frame:04d}", "camera": "realsense", "files": paths,
                        "sha256": {kind: files[name]["sha256"] for kind, name in paths.items()}})
    return result


def disk_guard(additional_bytes=64 * 1024 * 1024):
    if shutil.disk_usage(ROOT).free < RESERVE_BYTES + additional_bytes:
        raise RuntimeError("Insufficient disk to preserve the 60 GiB reserve")


def load_context():
    protocol = read(CONFIG)
    split_path = project_path(protocol["split"])
    selection = selection_from_protocol(protocol, read(split_path))
    for field, path in (("data_root", DATA_ROOT), ("data_manifest", MANIFEST), ("data_audit", AUDIT)):
        if project_path(protocol[field]) != path.resolve():
            raise ValueError("Frozen artifact location differs: " + field)
    local_paths = selected_paths(selection) - paths_for_scenes(NEW_TRAIN)
    sources, snapshots = {}, []
    if len(protocol["prior_snapshots"]) != 2:
        raise ValueError("Exactly two pinned prior snapshots required")
    for snapshot in protocol["prior_snapshots"]:
        path = project_path(snapshot["manifest"])
        if sha256(path) != snapshot["sha256"]:
            raise ValueError("Pinned prior snapshot hash changed: " + snapshot["manifest"])
        old, root = read(path), project_path(snapshot["data_root"])
        if (old["status"] not in ("verified_expanded_sample", "verified_confirmation_sample") or
                old["mirror_revision"] != REVISION or Path(old["data_root"]).resolve() != root or
                old["split_sha256"] != sha256(split_path)):
            raise ValueError("Prior snapshot source/root/split/status differs")
        for name in sorted(local_paths & old["files"].keys()):
            if name in sources:
                raise ValueError("Ambiguous duplicate prior snapshot member")
            sources[name] = {"root": root, "info": old["files"][name], "manifest": snapshot["manifest"],
                             "manifest_sha256": snapshot["sha256"]}
        snapshots.append(dict(snapshot, verified_status=old["status"]))
    if set(sources) != local_paths or len(sources) != 1360:
        raise ValueError("Prior snapshots must supply exactly the 1360 old members")
    binding = {"protocol_sha256": sha256(CONFIG), "split_sha256": sha256(split_path),
               "prior_snapshots": snapshots, "code_sha256": source_hashes(), "selection": selection,
               "mirror_revision": REVISION, "data_root": str(DATA_ROOT)}
    return protocol, binding, sources


def initialize_manifest(binding):
    if MANIFEST.exists():
        manifest = read(MANIFEST)
        for key, value in binding.items():
            if manifest.get(key) != value:
                raise ValueError("Existing coverage manifest binding differs: " + key)
        if not set(manifest["files"]).issubset(selected_paths(binding["selection"])):
            raise ValueError("Manifest includes paths outside frozen selection")
        return manifest
    return dict(binding, schema=8, dataset="GraspNet-1Billion", status="initialized", created_at=now(),
                source_kind="third_party_public_mirror_with_verified_local_snapshot_copies",
                mirror_repository="https://huggingface.co/saic3d/graspnet", transport_mirror="https://hf-mirror.com",
                official_dataset_page=previous.OFFICIAL, license_source=previous.OFFICIAL + "#license",
                license_statement="Official dataset page states free non-commercial use under BY-NC-SA; retain attribution to GraspNet team, MVIG, SJTU.",
                integrity_scope="Selected members checked with ZIP CRC32/SHA256; no official full-archive hash match. Local reuse is independent byte copy, never hard link.",
                scope="Frozen training-coverage strategy and already-exposed development validation; not independent confirmation or official test",
                reserve_gib=60, full_archive_fallback_allowed=False, files={}, archives={}, records={"train": [], "validation": []})


def commit_manifest(manifest):
    manifest["records"] = build_records(manifest["selection"], manifest["files"])
    manifest["updated_at"] = now()
    atomic_json(MANIFEST, manifest)


def bytes_match(payload, info):
    return (len(payload) == info["bytes"] and hashlib.sha256(payload).hexdigest() == info["sha256"] and
            f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}" == info["crc32"])


def copy_member(name, source, previous_record=None):
    source_path = safe_destination(source["root"], name)
    destination = safe_destination(DATA_ROOT, name)
    if destination.exists() and os.path.samefile(source_path, destination):
        raise ValueError("Prior and new snapshots must not share a hard link or target")
    payload = source_path.read_bytes()
    if not bytes_match(payload, source["info"]):
        raise ValueError("Pinned source member bytes/CRC/SHA changed: " + name)
    if previous_record and previous_record["sha256"] != source["info"]["sha256"]:
        raise ValueError("Previously committed copy SHA differs")
    if not destination.exists() or not bytes_match(destination.read_bytes(), source["info"]):
        disk_guard(additional_bytes=64 * 1024 * 1024 + len(payload))
        atomic_write(destination, payload)
    if sha256(destination) != source["info"]["sha256"] or os.path.samefile(source_path, destination):
        raise ValueError("Committed copy failed independent identity/checksum check")
    record = dict(source["info"], origin="copied_verified_snapshot", source_manifest=source["manifest"],
                  source_manifest_sha256=source["manifest_sha256"], verified_at=now())
    return name, record


def fetch_member(name, member, archive_name, archive, previous_record=None):
    info = archive.entries[member]
    destination = safe_destination(DATA_ROOT, name)
    expected_sha = previous_record.get("sha256") if previous_record else None
    payload = destination.read_bytes() if destination.exists() else None
    if payload is None or not previous.verified_bytes(payload, info, expected_sha):
        disk_guard()
        payload = archive.read(member)
        if not previous.verified_bytes(payload, info, expected_sha):
            raise ValueError("Remote member failed size/CRC/SHA validation")
        disk_guard(additional_bytes=64 * 1024 * 1024 + len(payload))
        atomic_write(destination, payload)
    checksum = hashlib.sha256(payload).hexdigest()
    if sha256(destination) != checksum:
        raise ValueError("Committed download checksum mismatch")
    return name, {"bytes": len(payload), "compressed_bytes": info.compress_size, "sha256": checksum,
                  "crc32": f"{info.CRC:08x}", "archive": archive_name, "source_member": member,
                  "origin": "downloaded_range", "verified_at": now()}


def fetch(workers=4):
    if workers not in range(1, 5):
        raise ValueError("Workers must be between one and four")
    _protocol, binding, sources = load_context()
    manifest = initialize_manifest(binding)
    if manifest["status"] == COMPLETE:
        audit()
        return manifest
    needed = selected_paths(binding["selection"])
    if manifest["status"] == "download_complete_pending_content_audit" and set(manifest["files"]) == needed:
        return manifest
    manifest["status"] = "downloading"
    manifest.pop("error", None)
    commit_manifest(manifest)

    def commit_results(jobs):
        for job in as_completed(jobs):
            name, item = job.result()
            manifest["files"][name] = item
            commit_manifest(manifest)
            count = len(manifest["files"])
            if count % 32 == 0 or count == len(needed):
                print(f"Committed {count}/{len(needed)} members", flush=True)

    try:
        disk_guard(additional_bytes=sum(source["info"]["bytes"] for source in sources.values()) + 64 * 1024 * 1024)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            commit_results([pool.submit(copy_member, name, source, manifest["files"].get(name)) for name, source in sources.items()])
        remaining, indexed = paths_for_scenes(NEW_TRAIN), []
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
            if not remaining:
                break
        if remaining:
            raise RuntimeError("Selected new training members absent from pinned source")
        manifest["copied_member_count"] = len(sources)
        manifest["remote_member_count"] = sum(len(item[3]) for item in indexed)
        manifest["estimated_selected_bytes"] = sum(source["info"]["bytes"] for source in sources.values()) + sum(
            item["selected_uncompressed_bytes"] for item in manifest["archives"].values())
        if manifest["remote_member_count"] != 680:
            raise RuntimeError("Exactly680 new training members expected")
        disk_guard(additional_bytes=manifest["estimated_selected_bytes"] + 64 * 1024 * 1024)
        commit_manifest(manifest)
        for archive_name, archive, members, found in indexed:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                commit_results([pool.submit(fetch_member, name, members[name], archive_name, archive,
                                            manifest["files"].get(name)) for name in found])
            manifest["archives"][archive_name]["network_bytes_this_invocation"] = archive.reader.network_bytes
            commit_manifest(manifest)
        if set(manifest["files"]) != needed:
            raise RuntimeError("Committed members differ from frozen selection")
        manifest["status"] = "download_complete_pending_content_audit"
        commit_manifest(manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        commit_manifest(manifest)
        raise
    return manifest


def inspect_contents(manifest):
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat
    errors, rows, hash_failures = [], [], []
    for name, info in manifest["files"].items():
        try:
            if not bytes_match(safe_destination(DATA_ROOT, name).read_bytes(), info):
                hash_failures.append(name)
        except OSError:
            hash_failures.append(name)
    for split in ("train", "validation"):
        for scene in manifest["selection"][split + "_scenes"]:
            scene_root = DATA_ROOT / f"scenes/scene_{scene:04d}"
            base = scene_root / "realsense"
            try:
                intrinsic = np.load(base / "camK.npy", allow_pickle=False)
                poses = np.load(base / "camera_poses.npy", allow_pickle=False)
                table = np.load(base / "cam0_wrt_table.npy", allow_pickle=False)
                objects = [int(value) for value in (scene_root / "object_id_list.txt").read_text(encoding="utf-8").split()]
                errors.extend(f"{scene}: {item}" for item in previous.inspect_auxiliary(intrinsic, poses, table, objects))
            except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as exc:
                errors.append(f"{scene}: invalid_auxiliary: {exc}")
                continue
            for frame in manifest["selection"]["frames"]:
                try:
                    arrays = {}
                    resized_labels = None
                    for kind in ("rgb", "depth", "label"):
                        with Image.open(base / kind / f"{frame:04d}.png") as image:
                            arrays[kind] = np.asarray(image)
                            if kind == "label":
                                resized_labels = np.unique(np.asarray(image.resize((320, 240), Image.Resampling.NEAREST))).tolist()
                    meta = loadmat(base / "meta" / f"{frame:04d}.mat")
                    found, row = previous.inspect_frame(arrays["rgb"], arrays["depth"], arrays["label"], meta, intrinsic, objects)
                    errors.extend(f"{scene}/{frame}: {item}" for item in found)
                    row.update(split=split, scene=scene, frame=f"{frame:04d}", visible_labels_input_240x320=resized_labels)
                    rows.append(row)
                except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as exc:
                    errors.append(f"{scene}/{frame}: {exc}")
            print("Content audited", split, scene, flush=True)
    return errors, rows, hash_failures


def population(rows, field):
    labels, combinations, per_scene, pairs = {}, set(), {}, set()
    for row in rows:
        visible = sorted(set(row[field]) - {0})
        combinations.add(tuple(visible))
        per_scene.setdefault(row["scene"], set()).update(visible)
        pairs.update(itertools.combinations(visible, 2))
        for label in visible:
            item = labels.setdefault(label, {"frame_count": 0, "scenes": set()})
            item["frame_count"] += 1
            item["scenes"].add(row["scene"])
    return {"frame_count": len(rows), "scene_count": len(per_scene), "label_ids_one_based": sorted(labels),
            "object_ids_zero_based": [label - 1 for label in sorted(labels)],
            "visible_object_frame_occurrences": sum(item["frame_count"] for item in labels.values()),
            "labels": {str(label): {"frame_count": item["frame_count"], "scene_count": len(item["scenes"]),
                                    "scenes": sorted(item["scenes"])} for label, item in sorted(labels.items())},
            "distinct_visible_frame_combinations": [list(item) for item in sorted(combinations)],
            "scene_visible_label_unions": {str(scene): sorted(values) for scene, values in sorted(per_scene.items())},
            "co_visible_label_pairs": [list(item) for item in sorted(pairs)]}


def coverage_statistics(rows, selection):
    result = {"scope": "Descriptive visible-label coverage only, before training. Label presence does not prove pose/view/appearance coverage or cause of model errors.",
              "label_convention": "PNG labels1..88 map to object_id_list0..87 by subtracting1; background excluded", "resolutions": {}}
    for resolution, field in (("original_720x1280", "visible_labels"), ("input_240x320", "visible_labels_input_240x320")):
        validation_rows = [row for row in rows if row["split"] == "validation"]
        validation = population(validation_rows, field)
        arms = {}
        for arm, scenes in (("coverage_control", selection["control_train_scenes"]), ("coverage_expanded", selection["train_scenes"])):
            training = population([row for row in rows if row["split"] == "train" and row["scene"] in scenes], field)
            observed = set(training["label_ids_one_based"])
            val_labels = set(validation["label_ids_one_based"])
            combinations = {tuple(values) for values in training["distinct_visible_frame_combinations"]}
            scene_combinations = {tuple(values) for values in training["scene_visible_label_unions"].values()}
            covered_occurrences = sum(item["frame_count"] for label, item in validation["labels"].items() if int(label) in observed)
            total = validation["visible_object_frame_occurrences"]
            training["validation_coverage"] = {"seen_label_ids": sorted(val_labels & observed),
                "unseen_label_ids": sorted(val_labels - observed), "covered_object_frame_occurrences": covered_occurrences,
                "total_object_frame_occurrences": total, "covered_occurrence_fraction": covered_occurrences / total if total else None,
                "frames_with_exact_training_visible_label_combination": sum(tuple(sorted(set(row[field]) - {0})) in combinations for row in validation_rows),
                "scenes_with_exact_training_visible_label_union": [int(scene) for scene, values in validation["scene_visible_label_unions"].items() if tuple(values) in scene_combinations],
                "unseen_co_visible_label_pairs": [pair for pair in validation["co_visible_label_pairs"] if pair not in training["co_visible_label_pairs"]]}
            arms[arm] = training
        result["resolutions"][resolution] = {"arms": arms, "validation": validation,
            "new_label_ids_expanded_vs_control": sorted(set(arms["coverage_expanded"]["label_ids_one_based"]) - set(arms["coverage_control"]["label_ids_one_based"]))}
    return result


def load_receipt(manifest_sha):
    if not RECEIPT.exists():
        raise RuntimeError("Completed manifest has no matching audit recovery receipt")
    receipt = read(RECEIPT)
    report = receipt["report"]
    if (receipt["manifest_sha256"] != manifest_sha or report["manifest_sha256"] != manifest_sha or report["status"] != "passed" or
            report["code_sha256"] != source_hashes() or report["protocol_sha256"] != sha256(CONFIG) or
            receipt["report_sha256"] != hashlib.sha256(json_bytes(report)).hexdigest()):
        raise RuntimeError("Audit recovery receipt binding differs")
    return receipt


def audit():
    _protocol, binding, sources = load_context()
    manifest = initialize_manifest(binding)
    if set(manifest["files"]) != selected_paths(binding["selection"]):
        raise ValueError("Cannot audit incomplete or extra coverage members")
    errors, rows, hash_failures = inspect_contents(manifest)
    for name, source in sources.items():
        info = manifest["files"][name]
        if (any(info[key] != source["info"][key] for key in ("bytes", "sha256", "crc32")) or
                info.get("origin") != "copied_verified_snapshot" or info.get("source_manifest") != source["manifest"] or
                info.get("source_manifest_sha256") != source["manifest_sha256"]):
            errors.append("Copied member provenance differs: " + name)
    for name in paths_for_scenes(NEW_TRAIN):
        if manifest["files"][name].get("origin") != "downloaded_range":
            errors.append("New training member origin differs: " + name)
    records = build_records(binding["selection"], manifest["files"])
    if len(records["train"]) != 320 or len(records["validation"]) != 160 or len(rows) != 480:
        errors.append("Incomplete coverage frame counts")
    coverage = coverage_statistics(rows, binding["selection"])
    if manifest["status"] == COMPLETE:
        if errors or hash_failures or manifest["records"] != records:
            raise RuntimeError("Completed coverage integrity failed; frozen manifest/data were not changed")
        receipt = load_receipt(sha256(MANIFEST))
        if receipt["report"]["coverage"] != coverage:
            raise RuntimeError("Completed descriptive coverage statistics differ")
        if AUDIT.exists():
            old = read(AUDIT)
            if old.get("status") == "passed":
                if sha256(AUDIT) != receipt["report_sha256"]:
                    raise RuntimeError("Refusing to overwrite a different completed audit")
                print("Completed coverage audit rechecked; all frozen bytes preserved", flush=True)
                return old
        atomic_json(AUDIT, receipt["report"])
        return receipt["report"]
    manifest["records"] = records
    manifest["status"] = "content_audit_failed" if errors or hash_failures else COMPLETE
    manifest["updated_at"] = now()
    manifest.pop("error", None)
    digest = hashlib.sha256(json_bytes(manifest)).hexdigest()
    report = {"schema": 8, "created_at": now(), "status": "failed" if errors or hash_failures else "passed",
              "manifest_sha256": digest, "protocol_sha256": binding["protocol_sha256"], "split_sha256": binding["split_sha256"],
              "code_sha256": binding["code_sha256"], "prior_snapshots": binding["prior_snapshots"],
              "data_root": str(DATA_ROOT), "selection": binding["selection"], "mirror_revision": REVISION,
              "train_frames": len(records["train"]), "validation_frames": len(records["validation"]),
              "files_verified": len(manifest["files"]) - len(hash_failures),
              "bytes_verified": sum(info["bytes"] for name, info in manifest["files"].items() if name not in hash_failures),
              "copied_members": sum(info["origin"] == "copied_verified_snapshot" for info in manifest["files"].values()),
              "range_download_members": sum(info["origin"] == "downloaded_range" for info in manifest["files"].values()),
              "errors": errors, "hash_failures": hash_failures, "frames": rows, "coverage": coverage,
              "scope": "Data transfer/content/descriptive coverage checks only; no model inference, training or new validation scene exposure",
              "free_disk_gib": shutil.disk_usage(ROOT).free / 2 ** 30}
    if report["status"] == "passed":
        atomic_json(RECEIPT, {"schema": 8, "manifest_sha256": digest, "report": report,
                             "report_sha256": hashlib.sha256(json_bytes(report)).hexdigest()})
    atomic_json(MANIFEST, manifest)
    atomic_json(AUDIT, report)
    if errors or hash_failures:
        raise RuntimeError("Coverage content audit failed: " + str((errors + hash_failures)[:10]))
    print("Coverage audit passed: 320 train / 160 development validation frames, 2040 members", flush=True)
    print(json.dumps({name: value["new_label_ids_expanded_vs_control"] for name, value in coverage["resolutions"].items()}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.audit_only and args.skip_audit:
        parser.error("Cannot combine --audit-only and --skip-audit")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(RUNTIME):
        disk_guard()
        if not args.audit_only:
            fetch(args.workers)
        if not args.skip_audit:
            audit()
