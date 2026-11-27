"""Durable local checkpoints for a single experiment (Python 3.7+).

Call save only after optimizer.step(), scaler.update() when used,
scheduler.step(), and zero_grad().
The caller owns deterministic data ordering and records the next batch, not the
last completed batch. Data-loader prefetch queues and partially accumulated
gradients are deliberately not serialized. Never load untrusted checkpoint files.
"""

import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import time
import uuid

import numpy as np
import torch


SCHEMA_VERSION = 1
GENERATION_PATTERN = re.compile(r"^checkpoint-[0-9]{19,20}-[0-9a-f]{12}$")


class CheckpointError(RuntimeError):
    pass


class FingerprintMismatch(CheckpointError):
    pass


def fingerprint(value):
    """Hash JSON-serializable settings or a previously assembled data manifest."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(directory):
    # Windows does not expose directory fsync through Python's normal file API.
    if os.name != "nt":
        descriptor = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write(path, writer):
    """Write a sibling temporary file, flush it, then replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with open(str(temporary), "xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(payload))


@contextlib.contextmanager
def _exclusive_lock(directory):
    """OS-backed lock: released by the OS even when a process is terminated."""
    path = Path(directory) / ".checkpoint.lock"
    with open(str(path), "a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise CheckpointError("Checkpoint directory is busy: {}".format(directory)) from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise CheckpointError("Checkpoint directory is busy: {}".format(directory)) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _component_type(component):
    if component is None:
        return None
    return type(component).__module__ + "." + type(component).__name__


def _capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _validate_rng(rng):
    if not isinstance(rng, dict) or not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng):
        raise CheckpointError("Checkpoint has incomplete random state")
    saved_count = len(rng["torch_cuda"])
    current_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_count != current_count:
        raise CheckpointError("CUDA device count changed; exact resume is unavailable")


def _restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


class CheckpointStore:
    """Immutable checkpoint generations, integrity checks, and recovery.

    keep_last must be at least two. An interrupted temporary write is ignored.
    New generations are published before old verified generations are pruned.
    One directory must correspond to exactly one configuration and data manifest.
    Actual power-loss durability also depends on the filesystem and storage device.
    """

    def __init__(self, directory, config_fingerprint, data_fingerprint, keep_last=3):
        if not isinstance(keep_last, int) or isinstance(keep_last, bool) or keep_last < 2:
            raise ValueError("At least two complete generations must be retained")
        for value in (config_fingerprint, data_fingerprint):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Fingerprints must be lowercase SHA-256 hex strings")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config_fingerprint = config_fingerprint
        self.data_fingerprint = data_fingerprint
        self.keep_last = keep_last

    def _check_fingerprints(self, value):
        if value.get("config_fingerprint") != self.config_fingerprint:
            raise FingerprintMismatch("Configuration fingerprint differs; use a new run directory")
        if value.get("data_fingerprint") != self.data_fingerprint:
            raise FingerprintMismatch("Data fingerprint differs; use a new run directory")

    def _manifest_paths(self):
        return sorted(self.directory.glob("checkpoint-*.json"), reverse=True)

    def _read_manifest(self, path):
        generation = path.stem
        if not GENERATION_PATTERN.fullmatch(generation):
            raise CheckpointError("Invalid generation name")
        with open(str(path), "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointError("Unsupported checkpoint schema")
        if manifest.get("generation") != generation or manifest.get("file") != generation + ".pt":
            raise CheckpointError("Manifest filename mismatch")
        self._check_fingerprints(manifest)
        checkpoint = self.directory / manifest["file"]
        if checkpoint.stat().st_size != manifest.get("size_bytes"):
            raise CheckpointError("Checkpoint length mismatch")
        if file_sha256(checkpoint) != manifest.get("sha256"):
            raise CheckpointError("Checkpoint SHA-256 mismatch")
        return manifest, checkpoint

    def save(self, model, optimizer, epoch, next_batch, global_step,
             at_optimizer_boundary, scheduler=None, scaler=None, extra=None,
             pending_micro_batches=0):
        if at_optimizer_boundary is not True or pending_micro_batches != 0:
            raise CheckpointError("Save only at a completed optimizer-step boundary")
        for value in (epoch, next_batch, global_step):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Resume cursors must be nonnegative integers")
        with _exclusive_lock(self.directory):
            # Refuse accidental reuse even before saving the first new generation.
            for path in self._manifest_paths():
                try:
                    self._read_manifest(path)
                except FingerprintMismatch:
                    raise
                except (OSError, ValueError, KeyError, TypeError, AttributeError, CheckpointError):
                    continue
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation = "checkpoint-{}-{}".format(time.time_ns(), uuid.uuid4().hex[:12])
            payload = {
                "schema_version": SCHEMA_VERSION,
                "config_fingerprint": self.config_fingerprint,
                "data_fingerprint": self.data_fingerprint,
                "cursor": {"epoch": epoch, "next_batch": next_batch, "global_step": global_step},
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "component_types": {"model": _component_type(model), "optimizer": _component_type(optimizer),
                                    "scheduler": _component_type(scheduler), "scaler": _component_type(scaler)},
                "rng": _capture_rng(),
                "extra": extra if extra is not None else {},
            }
            checkpoint = self.directory / (generation + ".pt")
            atomic_write(checkpoint, lambda stream: torch.save(payload, stream))
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "generation": generation,
                "file": checkpoint.name,
                "sha256": file_sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "global_step": global_step,
                "config_fingerprint": self.config_fingerprint,
                "data_fingerprint": self.data_fingerprint,
            }
            manifest_path = self.directory / (generation + ".json")
            atomic_json(manifest_path, manifest)
            atomic_json(self.directory / "latest.json", {
                "schema_version": SCHEMA_VERSION,
                "manifest": manifest_path.name,
                "manifest_sha256": file_sha256(manifest_path),
            })
            self._prune_verified()
            return manifest

    def _prune_verified(self):
        verified = []
        for path in self._manifest_paths():
            try:
                manifest, checkpoint = self._read_manifest(path)
            except FingerprintMismatch:
                raise
            except (OSError, ValueError, KeyError, TypeError, AttributeError, CheckpointError):
                continue
            verified.append((path, checkpoint))
        for manifest_path, checkpoint in verified[self.keep_last:]:
            # Removing metadata first prevents an interrupted prune advertising a missing file.
            manifest_path.unlink()
            checkpoint.unlink()
        _sync_directory(self.directory)

    def load_latest(self, model, optimizer, scheduler=None, scaler=None):
        """Restore newest verified generation and return its cursor and recovery notes.

        Raises if nothing can be restored. A fingerprint mismatch is never silently
        bypassed. Loss of latest.json does not lose committed generation manifests.
        """
        with _exclusive_lock(self.directory):
            rejected = []
            latest_problem = None
            try:
                with open(str(self.directory / "latest.json"), "r", encoding="utf-8") as stream:
                    pointer = json.load(stream)
                name = pointer["manifest"]
                if Path(name).name != name or not GENERATION_PATTERN.fullmatch(Path(name).stem):
                    raise CheckpointError("Invalid latest pointer")
                if file_sha256(self.directory / name) != pointer["manifest_sha256"]:
                    raise CheckpointError("Latest pointer digest mismatch")
            except (OSError, ValueError, KeyError, TypeError, AttributeError, CheckpointError) as error:
                latest_problem = "{}: {}".format(type(error).__name__, error)
            for path in self._manifest_paths():
                try:
                    manifest, checkpoint = self._read_manifest(path)
                    options = {"map_location": "cpu"}
                    if "weights_only" in inspect.signature(torch.load).parameters:
                        options["weights_only"] = False
                    payload = torch.load(str(checkpoint), **options)
                    self._check_fingerprints(payload)
                    if payload["schema_version"] != SCHEMA_VERSION:
                        raise CheckpointError("Unsupported payload schema")
                    if payload["cursor"]["global_step"] != manifest["global_step"]:
                        raise CheckpointError("Manifest and payload step differ")
                    for key in ("model", "optimizer", "scheduler", "scaler", "component_types", "rng", "extra"):
                        if key not in payload:
                            raise CheckpointError("Incomplete checkpoint payload")
                except FingerprintMismatch:
                    raise
                except Exception as error:
                    rejected.append({"manifest": path.name, "reason": "{}: {}".format(type(error).__name__, error)})
                    continue
                _validate_rng(payload["rng"])
                components = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
                for name, component in components.items():
                    if payload["component_types"].get(name) != _component_type(component):
                        raise CheckpointError("Component type changed: " + name)
                model.load_state_dict(payload["model"], strict=True)
                optimizer.load_state_dict(payload["optimizer"])
                if scheduler is not None:
                    scheduler.load_state_dict(payload["scheduler"])
                if scaler is not None:
                    scaler.load_state_dict(payload["scaler"])
                _restore_rng(payload["rng"])
                return {
                    "cursor": payload["cursor"], "extra": payload["extra"],
                    "checkpoint_path": str(checkpoint), "generation": manifest["generation"],
                    "rejected_generations": rejected, "latest_pointer_problem": latest_problem,
                }
            raise CheckpointError("No intact checkpoint available; rejected generations: {}".format(rejected))
