"""Bounded, resumable acquisition and structural audit of one training frame's geometry."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from src.remote_zip import RemoteZip, atomic_json, safe_destination

CONFIG = ROOT / "configs/grasp_geometry_v10.json"
DATA_ROOT = ROOT / "data/raw/grasp_geometry_v10"
MANIFEST = ROOT / "data/manifests/grasp_geometry_v10.json"
AUDIT = ROOT / "results/P3/grasp_geometry_assets_v10.json"
RUNTIME = ROOT / "runtime/grasp_geometry_v10"
CACHE = ROOT / "data/downloads/zip_ranges"
REVISION = "9073d1d4cb6c92272b6aa072545def5101e4fecb"
BASE_URL = "https://hf-mirror.com/saic3d/graspnet/resolve/" + REVISION
OBJECTS = [2, 21, 22, 41, 43, 44, 52, 60, 62]
COMPLETE = "verified_geometry_assets"
ARCHIVES = {
    "train_1.zip": {"size": 20411802782, "etag": '"e28d2b498c9a631f2dad3f6ca36f8c74096c79663d71ded824966970bc5b2c08"'},
    "models.zip": {"size": 4599338858, "etag": '"7bf30437d9d6ccf5745b546bdfd4ba3f2ef5c549b74fef7d0c3754512e508e96"'},
}
# Frozen central-directory observations, before fetching any selected payload.
# Values: compressed bytes, original bytes, CRC32, local-header offset.
MODEL_ENTRIES = {
    "062/nontextured.ply": (7257052, 13173880, 2177862384, 518590184),
    "062/textured.sdf": (3420841, 9869142, 2297502785, 532044547),
    "062/textured.obj": (11905745, 40997464, 2094330204, 535465469),
    "052/nontextured.ply": (10164430, 19066420, 1222275119, 621705959),
    "052/textured.sdf": (3425663, 10038622, 695257118, 636105531),
    "052/textured.obj": (17529509, 60143632, 2696323474, 639531275),
    "021/nontextured.ply": (9183108, 19399178, 1017975442, 1114214121),
    "021/textured.sdf": (3756781, 11064028, 2970773675, 1129601125),
    "021/textured.obj": (13861544, 61197282, 1357185388, 1133357987),
    "044/nontextured.ply": (9313765, 16371320, 2818928500, 1738581988),
    "044/textured.sdf": (3525980, 10004096, 2845382149, 1753454637),
    "044/textured.obj": (15731944, 51429923, 3951384704, 1756980698),
    "002/nontextured.ply": (9163451, 19398666, 2906494483, 2006990771),
    "002/textured.sdf": (3652463, 10520025, 1557334398, 2030473161),
    "002/textured.obj": (10212448, 51217685, 3050640359, 2034125934),
    "022/nontextured.ply": (81978390, 131324467, 2015833002, 2044338747),
    "022/textured.sdf": (3740309, 10783594, 697896586, 2136991339),
    "022/textured.obj": (78355586, 255546568, 935022465, 2140731729),
    "060/nontextured.ply": (14732243, 27049742, 4128211556, 3072274288),
    "060/textured.sdf": (3495249, 10235868, 2398554464, 3090269765),
    "060/textured.obj": (26004887, 86529892, 2671881289, 3093765095),
    "043/nontextured.ply": (4502394, 8071110, 1226856507, 3161213249),
    "043/textured.sdf": (3190503, 9943237, 3071900633, 3167524861),
    "043/textured.obj": (7424974, 24315177, 3700731464, 3170715445),
    "041/nontextured.ply": (3962849, 7083333, 2772201890, 3258471142),
    "041/textured.sdf": (3282264, 9902554, 4010996099, 3265856510),
    "041/textured.obj": (6531829, 21169102, 3325144047, 3269138855),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_digest(path):
    digest, crc, size = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size, "crc32": crc & 0xffffffff}


def sha256(path):
    return file_digest(path)["sha256"]


def entries():
    result = {}
    for tail, (compressed, size, crc, offset) in MODEL_ENTRIES.items():
        name = "models/" + tail
        result[name] = dict(archive="models.zip", source_member=name, compressed_bytes=compressed,
                            bytes=size, crc32=crc, header_offset=offset, compress_type=8)
    result["scenes/scene_0003/realsense/annotations/0000.xml"] = dict(
        archive="train_1.zip", source_member="scene_0003/realsense/annotations/0000.xml",
        compressed_bytes=540, bytes=2497, crc32=1703878368, header_offset=2137713022, compress_type=8)
    return result


def source_hashes():
    return {p: sha256(ROOT / p) for p in ["scripts/fetch_grasp_geometry_assets_v10.py", "src/remote_zip.py"]}


def load_context(config=CONFIG):
    protocol = read(config)
    if source_hashes()["src/remote_zip.py"] != protocol["source_hashes"]["src/remote_zip.py"]:
        raise ValueError("Frozen shared Range implementation changed")
    for key, value in {"schema": 10, "scene_id": 3, "frame_id": 0, "camera": "realsense",
                       "object_ids": OBJECTS, "data_root": DATA_ROOT.relative_to(ROOT).as_posix(),
                       "data_manifest": MANIFEST.relative_to(ROOT).as_posix(),
                       "data_audit": AUDIT.relative_to(ROOT).as_posix()}.items():
        if protocol.get(key) != value:
            raise ValueError("Protocol selection/path changed: " + key)
    if any(protocol["source"].get(k) != v for k, v in {"revision": REVISION, "base_url": BASE_URL}.items()):
        raise ValueError("Mirror source changed")
    lock_path = safe_destination(ROOT, protocol["asset_lock"]["path"])
    if sha256(lock_path) != protocol["asset_lock"]["sha256"]:
        raise ValueError("Asset lock changed")
    lock = read(lock_path)
    if lock["entries"] != entries() or lock["archives"] != ARCHIVES or lock["revision"] != REVISION:
        raise ValueError("Asset lock differs from discovered metadata")
    limits = protocol["asset_limits"]
    if (limits["total_bytes"] > 2 * 2 ** 30 or limits["member_bytes"] > 256 * 2 ** 20 or
            limits["reserve_bytes"] < 60 * 2 ** 30 or not 1 <= limits["chunk_bytes"] <= 4 * 2 ** 20):
        raise ValueError("Resource limits expanded")
    if sum(x["bytes"] for x in entries().values()) > limits["total_bytes"]:
        raise ValueError("Expected assets exceed resource budget")
    for relative, expected in protocol["prior_inputs"].items():
        path = safe_destination(ROOT, relative)
        if sha256(path) != expected:
            raise ValueError("Frozen prior input changed: " + relative)
    selected = entries()
    expected_local = {"scenes/scene_0003/realsense/" + n for n in ("camera_poses.npy", "cam0_wrt_table.npy", "meta/0000.mat")}
    if len(protocol["local_copies"]) != 3 or {x["destination"] for x in protocol["local_copies"]} != expected_local:
        raise ValueError("Local-copy selection changed")
    for item in protocol["local_copies"]:
        source = safe_destination(ROOT, item["source"])
        safe_destination(DATA_ROOT, item["destination"])
        found = file_digest(source)
        if found["sha256"] != item["sha256"] or protocol["prior_inputs"].get(item["source"]) != item["sha256"]:
            raise ValueError("Local-copy source not pinned")
        selected[item["destination"]] = dict(found, origin="copied_verified_snapshot", source=item["source"])
    return protocol, {"protocol_sha256": sha256(config), "source_sha256": source_hashes(),
                      "asset_lock_sha256": sha256(lock_path), "prior_inputs": protocol["prior_inputs"],
                      "entries": selected, "archives": ARCHIVES}


@contextmanager
def process_lock():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / ".assets.lock").open("a+b") as stream:
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


def disk_guard(limits, additional=0):
    if shutil.disk_usage(ROOT).free < limits["reserve_bytes"] + additional:
        raise RuntimeError("Insufficient free space for 60 GiB reserve")
    used = sum(p.stat().st_size for p in DATA_ROOT.rglob("*") if p.is_file()) if DATA_ROOT.exists() else 0
    # Only metadata is cached; compressed payloads are never retained.
    if used + additional + 10 * 2 ** 20 > limits["total_bytes"]:
        raise RuntimeError("Assets plus bounded metadata exceed 2 GiB budget")


def verify_file(path, metadata, recorded=None):
    found = file_digest(path)
    if found["bytes"] != metadata["bytes"] or found["crc32"] != metadata["crc32"]:
        raise ValueError("Asset size/CRC mismatch: " + str(path))
    if "sha256" in metadata and found["sha256"] != metadata["sha256"]:
        raise ValueError("Asset differs from frozen source SHA: " + str(path))
    if recorded is not None and found != {k: recorded[k] for k in found}:
        raise ValueError("Asset SHA/receipt mismatch: " + str(path))
    return found


def check_tree(manifest, complete=False):
    selected = manifest["binding"]["entries"]
    allowed = set(selected)
    if not set(manifest["files"]).issubset(allowed):
        raise ValueError("Unknown manifest asset")
    for name in allowed:
        safe_destination(DATA_ROOT, name)
    if DATA_ROOT.exists():
        for path in DATA_ROOT.rglob("*"):
            if path.is_file():
                name = path.relative_to(DATA_ROOT).as_posix()
                if name not in allowed and (complete or not name.endswith(".part") or name[:-5] not in allowed):
                    raise ValueError("Unexpected asset file: " + name)
    for name, record in manifest["files"].items():
        if any(record.get(k) != v for k, v in selected[name].items()):
            raise ValueError("Recorded asset metadata differs from selection")
        verify_file(safe_destination(DATA_ROOT, name), selected[name], record)
    if complete and set(manifest["files"]) != allowed:
        raise ValueError("Completed manifest missing selected assets")


def stream_member(archive, metadata, target, limits):
    """Use <=4 MiB HTTP ranges and bounded inflate output; never buffer whole member."""
    name = metadata["source_member"]
    info = archive.entries[name]
    observed = (info.compress_size, info.file_size, info.CRC, info.header_offset, info.compress_type)
    expected = tuple(metadata[k] for k in ["compressed_bytes", "bytes", "crc32", "header_offset", "compress_type"])
    if observed != expected or info.flag_bits & 1 or info.file_size > limits["member_bytes"]:
        raise ValueError("ZIP central metadata differs from frozen selection")
    raw = archive.reader.range(info.header_offset, 30, cache=False)
    header = struct.unpack("<4s5H3I2H", raw)
    if header[0] != b"PK\x03\x04" or header[2] != info.flag_bits or header[3] != info.compress_type:
        raise ValueError("ZIP local header mismatch")
    tail = archive.reader.range(info.header_offset + 30, header[-2] + header[-1], cache=False)
    if tail[:header[-2]].decode("utf-8" if info.flag_bits & 2048 else "cp437") != name:
        raise ValueError("ZIP local member name mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    disk_guard(limits, info.file_size)
    digest, crc, written = hashlib.sha256(), 0, 0
    decoder = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    if info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
        raise ValueError("Unsupported compression")
    offset = info.header_offset + 30 + len(tail)
    with temporary.open("wb") as stream:
        def emit(block):
            nonlocal crc, written
            written += len(block)
            if written > info.file_size:
                raise ValueError("Inflated asset exceeds frozen size")
            stream.write(block)
            digest.update(block)
            crc = zlib.crc32(block, crc)
        for start in range(0, info.compress_size, limits["chunk_bytes"]):
            block = archive.reader.range(offset + start, min(limits["chunk_bytes"], info.compress_size - start), cache=False)
            if decoder is None:
                emit(block)
            else:
                while block:
                    emit(decoder.decompress(block, limits["chunk_bytes"]))
                    block = decoder.unconsumed_tail
        if decoder is not None:
            emit(decoder.flush())
            if not decoder.eof or decoder.unused_data:
                raise ValueError("Incomplete or trailing DEFLATE stream")
        if written != info.file_size or crc & 0xffffffff != info.CRC:
            raise ValueError("Downloaded member size/CRC mismatch")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    found = verify_file(target, metadata)
    if found["sha256"] != digest.hexdigest():
        raise ValueError("Committed asset changed")
    return dict(metadata, **found)


def copy_snapshot(metadata, target, limits):
    source = safe_destination(ROOT, metadata["source"])
    verify_file(source, metadata, metadata)
    disk_guard(limits, metadata["bytes"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    with source.open("rb") as original, temporary.open("wb") as output:
        shutil.copyfileobj(original, output, limits["chunk_bytes"])
        output.flush()
        os.fsync(output.fileno())
    verify_file(temporary, metadata, metadata)
    os.replace(temporary, target)
    verify_file(source, metadata, metadata)
    return dict(metadata, **verify_file(target, metadata, metadata))




def finish(manifest, audit):
    receipt = RUNTIME / "assets_commit.json"
    complete_manifest = dict(manifest, status=COMPLETE)
    encoded = (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    complete_manifest["audit_sha256"] = hashlib.sha256(encoded).hexdigest()
    transaction = {"manifest": complete_manifest, "audit": audit}
    if receipt.exists() and read(receipt) != transaction:
        raise ValueError("Existing final transaction differs")
    if not receipt.exists():
        atomic_json(receipt, transaction)
    if AUDIT.exists():
        if sha256(AUDIT) != complete_manifest["audit_sha256"]:
            raise ValueError("Existing audit differs from final transaction")
    else:
        atomic_json(AUDIT, audit)
    if not MANIFEST.exists() or read(MANIFEST) != complete_manifest:
        atomic_json(MANIFEST, complete_manifest)
    return complete_manifest


def fetch(config=CONFIG, max_new_files=None, check_only=False):
    if max_new_files is not None and max_new_files < 1:
        raise ValueError("max-new-files must be positive")
    protocol, binding = load_context(config)
    manifest = read(MANIFEST) if MANIFEST.exists() else {"schema": 10, "status": "downloading", "binding": binding, "files": {}}
    if manifest.get("binding") != binding:
        raise ValueError("Existing manifest binding differs")
    if manifest.get("status") not in ("downloading", COMPLETE):
        raise ValueError("Unknown manifest state")
    check_tree(manifest, manifest["status"] == COMPLETE)
    if manifest["status"] == COMPLETE:
        if sha256(AUDIT) != manifest["audit_sha256"]:
            raise ValueError("Completed audit changed")
        transaction = read(RUNTIME / "assets_commit.json")
        if transaction["manifest"] != manifest or transaction["audit"] != read(AUDIT):
            raise ValueError("Completed transaction changed")
        print("verified_geometry_assets: complete, read-only verification", flush=True)
        return manifest
    if check_only:
        raise ValueError("Check-only requires an already completed asset manifest")
    receipt = RUNTIME / "assets_commit.json"
    if receipt.exists():
        transaction = read(receipt)
        if transaction["manifest"]["binding"] != binding or transaction["manifest"]["files"] != manifest["files"]:
            raise ValueError("Pending transaction changed")
        return finish(manifest, transaction["audit"])
    if not MANIFEST.exists():
        atomic_json(MANIFEST, manifest)
    archives = {}
    metadata_network_bytes = 0
    new_files = 0
    for name, metadata in sorted(binding["entries"].items()):
        if name in manifest["files"]:
            continue
        target = safe_destination(DATA_ROOT, name)
        if target.exists():
            # Crash between atomic file commit and manifest commit: CRC/size must still match.
            record = dict(metadata, **verify_file(target, metadata))
        elif metadata.get("origin") == "copied_verified_snapshot":
            record = copy_snapshot(metadata, target, protocol["asset_limits"])
        else:
            archive_name = metadata["archive"]
            if archive_name not in archives:
                archive = RemoteZip(BASE_URL + "/" + archive_name, CACHE)
                if {"size": archive.reader.size, "etag": archive.reader.etag} != ARCHIVES[archive_name]:
                    raise ValueError("Remote archive identity changed")
                metadata_network_bytes += archive.reader.network_bytes
                if metadata_network_bytes > protocol["asset_limits"].get("metadata_network_budget_bytes", 10 * 2 ** 20):
                    raise ValueError("ZIP metadata transfer exceeded budget")
                archives[archive_name] = archive
            record = stream_member(archives[archive_name], metadata, target, protocol["asset_limits"])
        manifest["files"][name] = record
        atomic_json(MANIFEST, manifest)
        new_files += 1
        print("asset committed", len(manifest["files"]), "/", len(binding["entries"]), name, flush=True)
        if max_new_files is not None and new_files >= max_new_files:
            return manifest
    check_tree(manifest, complete=True)
    # Rehash frozen prior inputs before committing the completed structural audit.
    _, after_binding = load_context(config)
    if after_binding != binding:
        raise ValueError("Inputs changed during acquisition")
    audit = {"schema": 10, "status": "passed_integrity_only", "binding": binding, "files": manifest["files"],
             "selected_file_count": len(manifest["files"]), "remote_file_count": len(entries()),
             "copied_local_file_count": 3, "total_asset_bytes": sum(x["bytes"] for x in manifest["files"].values()),
             "scope": "Selected file transfer and byte integrity only; geometry structure/poses/scoring require separate audit",
             "license_source": "https://graspnet.net/datasets.html#license",
             "license": "GraspNet team, MVIG, SJTU: non-commercial, attribution and share-alike (BY-NC-SA as stated by official page)",
             "integrity_scope": "Fixed third-party mirror revision plus archive identity/CRC/local SHA; not official whole-archive authenticity verification",
             "pickle_loaded": False, "unopened_validation_calibration_test_accessed": False}
    return finish(manifest, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--max-new-files", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        if args.max_new_files is not None:
            parser.error("--check-only and --max-new-files are mutually exclusive")
        fetch(args.config, check_only=True)
    else:
        with process_lock():
            fetch(args.config, args.max_new_files)


if __name__ == "__main__":
    main()
