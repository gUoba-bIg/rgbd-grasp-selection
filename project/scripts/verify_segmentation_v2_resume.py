"""Verify both diagnostic losses resume exactly; no scientific score is claimed."""
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
from src.checkpoints import atomic_json


def main():
    output = ROOT / "results/P2/segmentation_v2_resume.json"
    log_directory = ROOT / "logs/P2/segmentation_v2_resume"
    log_directory.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "runtime/verification"
    scratch.mkdir(parents=True, exist_ok=True)
    results = {}
    started = time.perf_counter()
    for loss in ("original", "center_mse"):
        with tempfile.TemporaryDirectory(prefix="v2_resume_", dir=str(scratch)) as temporary:
            base = Path(temporary)
            common = [sys.executable, str(ROOT / "scripts/train_segmentation_v2.py"),
                      "--steps", "6", "--max-steps", "6", "--loss", loss,
                      "--schedule", "constant", "--learning-rate", "0.001", "--amp",
                      "--save-every-steps", "2"]
            for name, extra, code in (("continuous", [], 0), ("interrupted", ["--crash-after-step", "3"], 37),
                                      ("resumed", ["--resume"], 0)):
                folder = base / ("continuous" if name == "continuous" else "interrupted")
                with (log_directory / (loss + "_" + name + ".log")).open("w", encoding="utf-8") as stream:
                    process = subprocess.run(common + ["--run-dir", str(folder)] + extra,
                                             cwd=ROOT, env=project_environment(ROOT), stdout=stream, stderr=subprocess.STDOUT)
                if process.returncode != code:
                    raise RuntimeError("{} {} returned {}, expected {}".format(loss, name, process.returncode, code))
                if name == "interrupted":
                    state = latest_payload(folder)
                    if state["cursor"]["global_step"] != 2:
                        raise AssertionError("Expected committed step 2 before the crash")
                    del state
            first = latest_payload(base / "continuous")
            second = latest_payload(base / "interrupted")
            checks = {key: not compare_tree(first[key], second[key]) for key in
                      ("model", "optimizer", "scheduler", "scaler", "cursor", "rng", "extra")}
            if not all(checks.values()):
                raise AssertionError("Full state mismatch: " + str(checks))
            results[loss] = {"status": "passed", "bitwise_equal": checks,
                             "durable_step_before_resume": 2, "final_step": second["cursor"]["global_step"]}
            print(loss + " full-state resume equivalence passed", flush=True)
            del first, second
    atomic_json(output, {"status": "passed", "scope": "Engineering recovery check on declared training frames, excluded from research results",
                         "real_power_cut_tested": False, "temporary_checkpoints_cleaned": True,
                         "losses": results, "elapsed_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
