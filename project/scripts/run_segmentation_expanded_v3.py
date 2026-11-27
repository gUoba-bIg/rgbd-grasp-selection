"""Run one frozen expanded-data pilot in recoverable bounded segments."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json, file_sha256


def main():
    config_path = ROOT / "configs/segmentation_expanded_v3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    directory = ROOT / "experiments/segmentation_expanded_v3/rgbd_original"
    log_directory = ROOT / "logs/P2/segmentation_expanded_v3"
    log_directory.mkdir(parents=True, exist_ok=True)
    planned = config["planned_steps"]
    limits = (planned,) if (directory / "config.json").exists() else (200, 800, planned)
    for limit in limits:
        command = [sys.executable, "-B", str(ROOT / "scripts/train_segmentation_expanded_v3.py"),
                   "--config", str(config_path), "--run-dir", str(directory), "--max-steps", str(limit)]
        if (directory / "config.json").exists():
            command.append("--resume")
        logfile = log_directory / ("through_" + str(limit) + ".log")
        print("RGBD expanded pilot through step", limit, "of", planned, flush=True)
        with logfile.open("ab", buffering=0) as stream:
            stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
            process = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                     stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        if process.returncode:
            raise RuntimeError("Training stopped with code {}; checkpoints preserved. See {}".format(
                process.returncode, logfile))
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if [row["step"] for row in history] != list(range(1, planned + 1)):
        raise RuntimeError("Training history must contain exactly the frozen optimizer steps")
    if result["status"] != "completed" or result["optimizer_steps"] != planned or not result["validation_evaluated"]:
        raise RuntimeError("Final training/evaluation is incomplete")
    atomic_json(ROOT / "results/P2/segmentation_expanded_v3.json", {
        "status": "completed", "decision": result["decision"],
        "scope": "Single-seed expanded RGBD pilot, not a method comparison or official test-set result",
        "protocol_sha256": file_sha256(config_path),
        "run_directory": str(directory.relative_to(ROOT)),
        "history_sha256": file_sha256(directory / "history.json"),
        "config_sha256": file_sha256(directory / "config.json"),
        "result_sha256": file_sha256(directory / "result.json"), "result": result,
        "validation_notice": "Scenes 18/22/33/39 were not used for previous model diagnostics. After this assessment they are development validation, not reusable blind tests.",
        "official_test_and_calibration_scenes_used": False})
    print("Fixed 1600-step plan completed. Inspect the predefined validation gate.", flush=True)


if __name__ == "__main__":
    main()
