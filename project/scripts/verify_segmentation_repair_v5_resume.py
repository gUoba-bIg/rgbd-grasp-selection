"""Verify both warm-start arms recover exactly across epoch/warmup boundaries.

These short engineering trajectories are not research results. The parent model
is inherited; all local optimization and random-stream state must start afresh.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch

from scripts.pipeline import project_environment
from scripts.verify_gpu_resume import compare_tree, latest_payload
from src.checkpoints import atomic_json, file_sha256, fingerprint


ARMS = ("original", "center_mse")
STATE_COMPONENTS = ("model", "optimizer", "scheduler", "scaler", "cursor", "rng", "extra")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if ROOT not in path.parents:
        raise ValueError("Verification paths must remain inside the project")
    return path


def tensor_state_fingerprint(state):
    """Independent, tensor-byte-sensitive digest of the model-only inheritance."""
    descriptions = {}
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError("Expected model state tensors only")
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        descriptions[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                              "sha256": hashlib.sha256(raw).hexdigest()}
    return fingerprint(descriptions)


def verify_parent(protocol):
    parent = project_path(protocol["parent_run"])
    result = read_json(parent / "result.json")
    if result["status"] != "completed" or result["optimizer_steps"] != 1600:
        raise AssertionError("Frozen parent must have completed its original 1600 updates")
    checkpoint = project_path(result["checkpoint_path"])
    if file_sha256(checkpoint) != protocol["parent_checkpoint_sha256"] or result["checkpoint_sha256"] != protocol["parent_checkpoint_sha256"]:
        raise AssertionError("Warm-start parent checkpoint bytes differ from the frozen protocol")
    latest = read_json(parent / "checkpoints/latest.json")
    latest_meta = read_json(parent / "checkpoints" / latest["manifest"])
    if latest_meta["sha256"] != protocol["parent_checkpoint_sha256"] or (parent / "checkpoints" / latest_meta["file"]).resolve() != checkpoint:
        raise AssertionError("Latest pointer and frozen parent result refer to different checkpoint bytes")
    payload = latest_payload(parent)
    if payload["cursor"]["global_step"] != 1600:
        raise AssertionError("Latest parent checkpoint is not the final frozen step")
    for key in ("config_fingerprint", "data_fingerprint"):
        if payload[key] != protocol["parent_" + key]:
            raise AssertionError("Parent fingerprint mismatch: " + key)
    parent_config = read_json(parent / "config.json")
    parent_manifest = read_json(parent / "data_manifest.json")
    if fingerprint(parent_config) != payload["config_fingerprint"] or fingerprint(parent_manifest) != payload["data_fingerprint"]:
        raise AssertionError("Parent configuration/manifest bytes do not match the checkpoint")
    for relative, digest in parent_config["source_sha256"].items():
        if file_sha256(project_path(relative)) != digest:
            raise AssertionError("Frozen parent source changed: " + relative)
    evidence = {"parent_checkpoint_sha256": protocol["parent_checkpoint_sha256"],
                "parent_model_tensor_sha256": tensor_state_fingerprint(payload["model"]),
                "parent_config_fingerprint": payload["config_fingerprint"],
                "parent_data_fingerprint": payload["data_fingerprint"], "parent_optimizer_steps": 1600}
    del payload
    return evidence


def check_initialization(record, parent, protocol, run_config_fingerprint):
    expected = {"parent_checkpoint_sha256": parent["parent_checkpoint_sha256"],
                "parent_model_tensor_sha256": parent["parent_model_tensor_sha256"],
                "initial_model_tensor_sha256": parent["parent_model_tensor_sha256"],
                "model_only_warm_start": True, "optimizer_history_empty": True,
                "optimizer_state_entries": 0, "scheduler_last_epoch": 0,
                "history_length": 0,
                "cursor": {"epoch": 0, "next_batch": 0, "global_step": 0},
                "rng_seed": protocol["seed"], "parent_config_fingerprint": parent["parent_config_fingerprint"],
                "parent_data_fingerprint": parent["parent_data_fingerprint"],
                "run_config_fingerprint": run_config_fingerprint}
    for key, value in expected.items():
        if key not in record or compare_tree(record[key], value):
            raise AssertionError("Invalid step-zero initialization evidence: " + key)
    expected_lr = protocol["learning_rate"] / protocol["warmup_steps"]
    if abs(record["initial_used_lr"] - expected_lr) > 1e-18:
        raise AssertionError("Step-zero optimizer does not use the reset warmup learning rate")
    scaler = record["scaler_state"]
    expected_scaler = {"scale": 128.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                       "growth_interval": 2000, "_growth_tracker": 0}
    if compare_tree(scaler, expected_scaler):
        raise AssertionError("Gradient scaler was not freshly initialized")
    scheduler = record["scheduler_state"]
    if scheduler.get("last_epoch") != 0 or scheduler.get("_step_count") != 1:
        raise AssertionError("Full scheduler evidence does not show a fresh initial state")
    if scheduler.get("base_lrs") != [protocol["learning_rate"]] or abs(scheduler["_last_lr"][0] - record["initial_used_lr"]) > 1e-18:
        raise AssertionError("Initial scheduler and optimizer learning rates disagree")
    rng_digest = record["initial_rng_fingerprint"]
    if not isinstance(rng_digest, str) or len(rng_digest) != 64 or any(character not in "0123456789abcdef" for character in rng_digest):
        raise AssertionError("Missing actual initial random-stream fingerprint")
    return {"parent_model_byte_identity": True, "optimizer_history_empty": True,
            "scheduler_reset": True, "scaler_reset": True, "local_cursor_reset": True,
            "rng_seed_reset": True, "initial_rng_fingerprint": rng_digest,
            "initial_model_tensor_sha256": record["initial_model_tensor_sha256"],
            "initial_used_lr": record["initial_used_lr"], "scaler_state": scaler}


def check_partial_result(result, resumed=False):
    if result["status"] != "step_limit_reached" or result["optimizer_steps"] != 82 or result["planned_steps"] != 1600:
        raise AssertionError("Recovery test must stop at local step 82 of the frozen 1600-update plan")
    if result.get("validation_evaluated") is not False or result.get("evaluation") is not None:
        raise AssertionError("Partial engineering run must not perform final training/validation assessment")
    if resumed and (result["resumed_from_step"] != 80 or result["updates_this_invocation"] != 2):
        raise AssertionError("Resume must replay only updates 81 and 82 from durable step 80")


def invoke(command, log_path, expected):
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        process = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                 stdout=stream, stderr=subprocess.STDOUT)
        stream.flush()
        os.fsync(stream.fileno())
    if process.returncode != expected:
        raise RuntimeError("Return code {} instead of {}; see {}".format(process.returncode, expected, log_path))
    return {"exit_code": process.returncode, "log": str(log_path.relative_to(ROOT)), "log_sha256": file_sha256(log_path)}


def verify(config_path):
    protocol = read_json(config_path)
    required = {"schema": 5, "planned_steps": 1600, "seed": 20261022, "learning_rate": 0.0003,
                "warmup_steps": 80, "amp": True, "accumulation": 2}
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError("Unexpected frozen protocol value: " + key)
    if set(protocol["arms"]) != set(ARMS):
        raise ValueError("Both and only the original/center_mse arms are required")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA execution is required for this engineering verification")
    if shutil.disk_usage(ROOT).free < (protocol["free_disk_reserve_gib"] + 2) * 2 ** 30:
        raise RuntimeError("Cannot preserve the free disk reserve and temporary verification checkpoints")
    output = ROOT / "results/P2/segmentation_repair_v5_resume.json"
    logs = ROOT / "logs/P2/segmentation_repair_v5_resume"
    scratch = ROOT / "runtime/verification"
    logs.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    parent = verify_parent(protocol)
    from scripts.train_segmentation_repair_v5 import source_hashes
    source_names = sorted(set(source_hashes("original")) | set(source_hashes("center_mse")) | {
        "scripts/verify_segmentation_repair_v5_resume.py", "scripts/verify_gpu_resume.py"})
    report = {"schema": 5, "status": "running", "arms": {}, "parent": parent,
              "scope": "Actual two-arm CUDA/AMP warm-start recovery across local epoch and warmup boundaries",
              "excluded_from_research_results": True, "real_power_cut_tested": False,
              "planned_steps_per_research_arm": 1600, "final_engineering_step": 82,
              "forced_exit_after_step": 81, "durable_step_before_resume": 80, "checkpoint_interval": 2,
              "validation_evaluated": False, "config_sha256": file_sha256(config_path),
              "data_manifest_sha256": file_sha256(project_path(protocol["data_manifest"])),
              "source_sha256": {name: file_sha256(ROOT / name) for name in source_names},
              "device": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
              "temporary_checkpoints_cleaned": False}
    atomic_json(output, report)
    cross_arm = {}
    try:
        for arm in ARMS:
            with tempfile.TemporaryDirectory(prefix="repair_v5_" + arm + "_", dir=str(scratch)) as temporary:
                base = Path(temporary)
                common = [sys.executable, "-B", str(ROOT / "scripts/train_segmentation_repair_v5.py"),
                          "--config", str(config_path), "--arm", arm, "--max-steps", "82",
                          "--checkpoint-interval", "2", "--device", "cuda"]
                processes = {}
                for phase, extra, expected in (("continuous", [], 0),
                        ("interrupted", ["--crash-after-step", "81"], 37), ("resumed", ["--resume"], 0)):
                    directory = base / ("continuous" if phase == "continuous" else "interrupted")
                    print("Warm-start recovery:", arm, phase, flush=True)
                    processes[phase] = invoke(common + ["--run-dir", str(directory)] + extra,
                                             logs / (arm + "_" + phase + ".log"), expected)
                    if phase == "interrupted":
                        state = latest_payload(directory)
                        if state["cursor"] != {"epoch": 1, "next_batch": 0, "global_step": 80}:
                            raise AssertionError("Expected committed epoch-one/local-step-80 boundary before forced exit at 81")
                        del state
                reference = latest_payload(base / "continuous")
                recovered = latest_payload(base / "interrupted")
                differences = {key: compare_tree(reference[key], recovered[key]) for key in STATE_COMPONENTS}
                checks = {key: not paths for key, paths in differences.items()}
                if recovered["cursor"] != {"epoch": 1, "next_batch": 4, "global_step": 82} or not all(checks.values()):
                    raise AssertionError("Full-state resume mismatch: " + str({key: paths[:12] for key, paths in differences.items() if paths}))
                continuous_result = read_json(base / "continuous/result.json")
                resumed_result = read_json(base / "interrupted/result.json")
                check_partial_result(continuous_result)
                check_partial_result(resumed_result, resumed=True)
                init_records = {phase: read_json(base / phase / "initialization.json") for phase in ("continuous", "interrupted")}
                init_checks = {phase: check_initialization(record, parent, protocol, reference["config_fingerprint"])
                               for phase, record in init_records.items()}
                if compare_tree(init_records["continuous"], init_records["interrupted"]):
                    raise AssertionError("Continuous/interrupted runs did not have identical initialization evidence")
                history = reference["extra"]["history"]
                if [row["step"] for row in history] != list(range(1, 83)):
                    raise AssertionError("Recovery history is not the exact 82 committed updates")
                cross_arm[arm] = {"initial_model_tensor_sha256": init_records["continuous"]["initial_model_tensor_sha256"],
                                  "initial_rng_fingerprint": init_records["continuous"]["initial_rng_fingerprint"],
                                  "reset_state": {key: init_records["continuous"][key] for key in (
                                      "optimizer_history_empty", "optimizer_state_entries", "scheduler_state",
                                      "scaler_state", "cursor", "history_length", "rng_seed")},
                                  "samples": [row["samples"] for row in history],
                                  "learning_rate_used": [row["learning_rate_used"] for row in history]}
                report["arms"][arm] = {"status": "passed", "bitwise_equal": checks, "processes": processes,
                    "initialization_checks": init_checks, "initialization": init_records["continuous"],
                    "initialization_sha256": {phase: file_sha256(base / phase / "initialization.json") for phase in init_records},
                    "config_fingerprint": reference["config_fingerprint"], "data_fingerprint": reference["data_fingerprint"],
                    "final_cursor": recovered["cursor"], "resumed_from_step": resumed_result["resumed_from_step"],
                    "updates_on_resume": resumed_result["updates_this_invocation"], "validation_evaluated": False,
                    "loss_history_sha256": fingerprint(history), "loss_history_exactly_equal": True,
                    "scaler_state_at82": recovered["scaler"]}
                del reference, recovered
            atomic_json(output, report)
        same_model = cross_arm["original"]["initial_model_tensor_sha256"] == cross_arm["center_mse"]["initial_model_tensor_sha256"]
        same_samples = not compare_tree(cross_arm["original"]["samples"], cross_arm["center_mse"]["samples"])
        same_schedule = not compare_tree(cross_arm["original"]["learning_rate_used"], cross_arm["center_mse"]["learning_rate_used"])
        same_rng = cross_arm["original"]["initial_rng_fingerprint"] == cross_arm["center_mse"]["initial_rng_fingerprint"]
        same_reset = not compare_tree(cross_arm["original"]["reset_state"], cross_arm["center_mse"]["reset_state"])
        if not all((same_model, same_samples, same_schedule, same_rng, same_reset)):
            raise AssertionError("Two-arm starting model, reset states, sample order, or actual learning-rate schedule differs")
        report.update({"status": "passed", "cross_arm_checks": {"same_parent_model_tensors_at_step0": same_model,
                       "same_actual_random_streams_at_step0": same_rng,
                       "same_reset_optimizer_scheduler_scaler_cursor": same_reset,
                       "same_sample_order_through_step82": same_samples, "same_actual_learning_rate_through_step82": same_schedule},
                       "temporary_checkpoints_cleaned": True})
    except Exception as error:
        report.update({"status": "failed", "error": type(error).__name__ + ": " + str(error),
                       "temporary_checkpoints_cleaned": True})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output, report)
    print(json.dumps({"status": report["status"], "arms": list(report["arms"]),
                      "report": str(output), "all_states_bitwise_equal": True}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_repair_v5.json")
    args = parser.parse_args()
    verify(project_path(args.config))
