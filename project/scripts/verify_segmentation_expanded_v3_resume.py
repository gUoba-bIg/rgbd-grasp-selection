"""Verify actual-model recovery across an epoch and warmup boundary."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from scripts.verify_gpu_resume import latest_payload, compare_tree
from src.checkpoints import atomic_json, file_sha256


def main():
    output = ROOT / "results/P2/segmentation_expanded_v3_resume.json"
    logs = ROOT / "logs/P2/segmentation_expanded_v3_resume"
    logs.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "runtime/verification"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="expanded_v3_resume_", dir=str(scratch)) as temporary:
        base = Path(temporary)
        common = [sys.executable, "-B", str(ROOT / "scripts/train_segmentation_expanded_v3.py"),
                  "--config", "configs/segmentation_expanded_v3.json", "--max-steps", "82",
                  "--checkpoint-interval", "2"]
        for name, extra, expected in (("continuous", [], 0),
                                     ("interrupted", ["--crash-after-step", "81"], 37),
                                     ("resumed", ["--resume"], 0)):
            directory = base / ("continuous" if name == "continuous" else "interrupted")
            print("Recovery check:", name, flush=True)
            with (logs / (name + ".log")).open("w", encoding="utf-8") as stream:
                process = subprocess.run(common + ["--run-dir", str(directory)] + extra,
                                         cwd=ROOT, env=project_environment(ROOT),
                                         stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode != expected:
                raise RuntimeError("{} returned {}; expected {}. Inspect {}".format(
                    name, process.returncode, expected, logs / (name + ".log")))
            if name == "interrupted":
                payload = latest_payload(directory)
                if payload["cursor"]["global_step"] != 80:
                    raise AssertionError("Expected committed step 80 before forced exit at 81")
                del payload
        reference = latest_payload(base / "continuous")
        recovered = latest_payload(base / "interrupted")
        checks = {key: not compare_tree(reference[key], recovered[key]) for key in
                  ("model", "optimizer", "scheduler", "scaler", "cursor", "rng", "extra")}
        if recovered["cursor"]["global_step"] != 82 or not all(checks.values()):
            raise AssertionError("Actual training state mismatch: " + str(checks))
        result = json.loads((base / "interrupted/result.json").read_text(encoding="utf-8"))
        if result.get("validation_evaluated", False):
            raise AssertionError("Partial engineering run must not evaluate validation data")
        report = {"status": "passed", "bitwise_equal": checks,
                  "scope": "Actual RGBD model with augmentation, CUDA AMP, epoch order and warmup/cosine boundary; engineering check only",
                  "excluded_from_research_results": True, "real_power_cut_tested": False,
                  "planned_training_steps": 1600, "durable_step_before_resume": 80,
                  "forced_exit_after_step": 81, "final_step": 82,
                  "validation_evaluated": False, "temporary_checkpoints_cleaned": True,
                  "config_sha256": file_sha256(ROOT / "configs/segmentation_expanded_v3.json"),
                  "trainer_sha256": file_sha256(ROOT / "scripts/train_segmentation_expanded_v3.py"),
                  "data_manifest_sha256": file_sha256(ROOT / "data/manifests/expanded_v3.json"),
                  "elapsed_seconds": time.perf_counter() - started}
        del reference, recovered
    atomic_json(output, report)
    print("Expanded trainer: all states bitwise equal after interrupted update 81.", flush=True)


if __name__ == "__main__":
    main()
