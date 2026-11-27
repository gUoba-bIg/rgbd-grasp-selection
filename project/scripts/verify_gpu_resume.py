"""Verify CUDA+AMP pilot resume using explicitly synthetic engineering fixtures."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from scipy.io import savemat
import torch

from src.checkpoints import atomic_json, file_sha256


def make_engineering_fixtures(directory):
    """Synthetic shapes exercise the trainer; they are never research data."""
    data = directory / "synthetic_engineering_fixtures"
    data.mkdir(parents=True)
    atomic_json(data / "purpose.json", {
        "kind": "synthetic_engineering_fixture",
        "purpose": "Verify interruption and deterministic GPU continuation only",
        "excluded_from_research_results": True,
    })
    split = directory / "synthetic_scene_splits.json"
    atomic_json(split, {"pilot_train": [3, 12], "pilot_validation": [13],
                        "groups": {"train": [3, 12], "validation": [13]},
                        "pilot_frame_step": 16, "kind": "synthetic_engineering_fixture"})
    rng = np.random.RandomState(7721)
    yy, xx = np.mgrid[:240, :320]
    for scene in (3, 12, 13):
        for frame in ((0, 16) if scene != 13 else (0,)):
            base = data / "scenes" / "scene_{:04d}".format(scene) / "realsense"
            for name in ("rgb", "depth", "label", "meta"):
                (base / name).mkdir(parents=True, exist_ok=True)
            rgb = rng.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            label = np.zeros((240, 320), dtype=np.uint8)
            label[40 + frame:105 + frame, 35:100] = 3
            label[(yy - 150) ** 2 + (xx - 225) ** 2 < 35 ** 2] = 7
            rgb[label == 3] = [190, 45, 70]
            rgb[label == 7] = [20, 170, 80]
            depth = np.full((240, 320), 1500, dtype=np.uint16)
            depth[label > 0] = 900 + scene
            depth[:8, :15] = 0
            name = "{:04d}".format(frame)
            Image.fromarray(rgb).save(str(base / "rgb" / (name + ".png")))
            Image.fromarray(depth).save(str(base / "depth" / (name + ".png")))
            Image.fromarray(label).save(str(base / "label" / (name + ".png")))
            savemat(str(base / "meta" / (name + ".mat")), {"factor_depth": np.array([[1000.0]])})
    return data, split


def invoke(command, log_path, expected_code):
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != expected_code:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError("Expected exit {}, got {}. {}".format(expected_code, process.returncode, tail))
    return process.returncode


def latest_payload(run_directory):
    checkpoints = run_directory / "checkpoints"
    pointer = json.loads((checkpoints / "latest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoints / pointer["manifest"]
    if file_sha256(manifest_path) != pointer["manifest_sha256"]:
        raise AssertionError("Checkpoint manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = checkpoints / manifest["file"]
    if file_sha256(payload_path) != manifest["sha256"]:
        raise AssertionError("Checkpoint payload hash changed")
    return torch.load(str(payload_path), map_location="cpu", weights_only=False)


def compare_tree(first, second, path="root", mismatches=None):
    """Compare tensor/array bytes as well as all nested non-tensor state."""
    if mismatches is None:
        mismatches = []
    if torch.is_tensor(first):
        equal = torch.is_tensor(second) and first.dtype == second.dtype and first.shape == second.shape
        if equal:
            equal = torch.equal(first.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                second.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(first, np.ndarray):
        equal = isinstance(second, np.ndarray) and first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()
    elif isinstance(first, dict):
        if not isinstance(second, dict) or set(first) != set(second):
            equal = False
        else:
            for key in first:
                compare_tree(first[key], second[key], path + "." + str(key), mismatches)
            return mismatches
    elif isinstance(first, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            equal = False
        else:
            for index, (left, right) in enumerate(zip(first, second)):
                compare_tree(left, right, path + "[{}]".format(index), mismatches)
            return mismatches
    elif isinstance(first, float):
        # float.hex distinguishes negative zero as well as different finite values.
        equal = isinstance(second, float) and first.hex() == second.hex()
    else:
        equal = type(first) is type(second) and first == second
    if not equal:
        mismatches.append(path)
    return mismatches


def verify(output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this verification")
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = ROOT / "runtime" / "verification"
    scratch_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "running", "kind": "synthetic_engineering_verification",
              "excluded_from_research_results": True, "device": torch.cuda.get_device_name(0),
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "python": sys.version.split()[0], "amp": True, "input_size": [240, 320],
              "model": "Actual ResNet18-GroupNorm-FPN48 pilot training code",
              "planned_optimizer_steps": 4, "crash_after_step": 3, "checkpoint_interval": 2,
              "limits": ["Synthetic shapes and depth arrays are engineering fixtures only",
                         "No research dataset training or evaluation result is established by this check",
                         "Forced process termination was tested; physical electrical power loss was not simulated",
                         "Equality applies to this recorded interpreter, GPU, precision and configuration"]}
    atomic_json(output, report)
    with tempfile.TemporaryDirectory(prefix="gpu_resume_", dir=str(scratch_root)) as temporary:
        directory = Path(temporary)
        data, split = make_engineering_fixtures(directory)
        common = [sys.executable, str(ROOT / "scripts" / "train_segmentation_pilot.py"),
                  "--data-root", str(data), "--split", str(split), "--mode", "rgbd",
                  "--epochs", "2", "--max-steps", "4", "--height", "240", "--width", "320",
                  "--accumulation", "2", "--max-frames-per-scene", "2",
                  "--device", "cuda", "--amp", "--save-every-steps", "2"]
        uninterrupted = directory / "uninterrupted"
        interrupted = directory / "interrupted"
        log_directory = output.parent / "gpu_resume_logs"
        log_directory.mkdir(exist_ok=True)
        try:
            print("Running continuous CUDA+AMP reference (synthetic engineering fixtures).", flush=True)
            invoke(common + ["--run-dir", str(uninterrupted)], log_directory / "continuous.log", 0)
            print("Running CUDA+AMP worker, forced exit after uncommitted step 3.", flush=True)
            code = invoke(common + ["--run-dir", str(interrupted), "--crash-after-step", "3"],
                          log_directory / "interrupted.log", 37)
            interrupted_payload = latest_payload(interrupted)
            saved_step = interrupted_payload["cursor"]["global_step"]
            if saved_step != 2:
                raise AssertionError("Expected durable step 2 before recovery")
            del interrupted_payload
            print("Resuming durable step 2 and comparing full training state.", flush=True)
            invoke(common + ["--run-dir", str(interrupted), "--resume"], log_directory / "resumed.log", 0)
            reference = latest_payload(uninterrupted)
            recovered = latest_payload(interrupted)
            comparisons = {}
            mismatches = {}
            for name in ("model", "optimizer", "scheduler", "scaler", "cursor", "rng", "extra"):
                differences = compare_tree(reference[name], recovered[name], name)
                comparisons[name + "_bitwise_equal"] = not differences
                if differences:
                    mismatches[name] = differences[:20]
            history_equal = not compare_tree(reference["extra"]["history"], recovered["extra"]["history"])
            resumed_result = json.loads((interrupted / "result.json").read_text(encoding="utf-8"))
            report.update({"status": "passed" if all(comparisons.values()) else "failed",
                           "comparisons": comparisons, "mismatches": mismatches,
                           "loss_history_exactly_equal": history_equal,
                           "loss_history": [row["loss"] for row in reference["extra"]["history"]],
                           "forced_exit_code": code, "durable_step_before_resume": saved_step,
                           "resumed_from_step": resumed_result["resumed_from_step"],
                           "updates_on_resume": resumed_result["updates_this_invocation"],
                           "final_cursor": recovered["cursor"], "scaler_state": recovered["scaler"],
                           "config_fingerprint": reference["config_fingerprint"],
                           "data_fingerprint": reference["data_fingerprint"],
                           "configuration": reference["extra"]["config"],
                           "logs": {name: str(log_directory / (name + ".log")) for name in ("continuous", "interrupted", "resumed")},
                           "temporary_fixture_and_checkpoint_cleanup": True})
            if not all(comparisons.values()) or not history_equal:
                raise AssertionError("GPU resume state differs: {}".format(mismatches))
            if resumed_result["resumed_from_step"] != 2 or resumed_result["updates_this_invocation"] != 2:
                raise AssertionError("Previously committed optimizer steps were not skipped correctly")
        except Exception as error:
            report.update({"status": "failed", "error": "{}: {}".format(type(error).__name__, error)})
            raise
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(output, report)
    print(json.dumps({"status": report["status"], "report": str(output),
                      "loss_history_exactly_equal": report["loss_history_exactly_equal"],
                      "all_states_bitwise_equal": all(report["comparisons"].values())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "results" / "P2" / "gpu_resume_verification.json"))
    args = parser.parse_args()
    verify(Path(args.output).resolve())
