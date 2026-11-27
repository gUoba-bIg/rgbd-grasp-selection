"""Execute the frozen two-arm training-only diagnostic with resumable segments."""
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
    protocol_path = ROOT / "configs/segmentation_repair_v2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    results = {}
    log_dir = ROOT / "logs/P2/segmentation_repair_v2"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "results/P2/segmentation_repair_v2.json"
    common = protocol["common"]
    evidence = []
    for arm in protocol["arms"]:
        directory = ROOT / arm["directory"]
        # A short committed segment checks stability before the bounded remainder.
        # A resumed run can already be beyond the short probe; never request a
        # stopping cursor earlier than its committed checkpoint.
        limits = (common["optimizer_steps_per_arm"],) if (directory / "config.json").exists() else (100, common["optimizer_steps_per_arm"])
        for limit in limits:
            command = [sys.executable, str(ROOT / "scripts/train_segmentation_v2.py"),
                       "--run-dir", str(directory), "--loss", arm["name"], "--amp",
                       "--schedule", common["schedule"], "--learning-rate", str(common["learning_rate"]),
                       "--steps", str(common["optimizer_steps_per_arm"]), "--max-steps", str(limit),
                       "--save-every-steps", str(common["checkpoint_every_steps"])]
            if (directory / "config.json").exists():
                command.append("--resume")
            logfile = log_dir / (arm["name"] + "_" + str(limit) + ".log")
            print("Running", arm["name"], "to", limit, "updates", flush=True)
            with logfile.open("ab", buffering=0) as stream:
                stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
                process = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                         stdout=stream, stderr=subprocess.STDOUT)
                os.fsync(stream.fileno())
            if process.returncode:
                raise RuntimeError("Training stopped with code {}; existing checkpoints preserved. See {}".format(process.returncode, logfile))
            evidence.append({"arm": arm["name"], "limit": limit, "log": str(logfile.relative_to(ROOT))})
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        if [row["step"] for row in history] != list(range(1, common["optimizer_steps_per_arm"] + 1)):
            raise RuntimeError("History does not contain exactly the planned updates")
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        results[arm["name"]] = {"directory": arm["directory"], "result": result,
                                "history_sha256": file_sha256(directory / "history.json"),
                                "config_sha256": file_sha256(directory / "config.json")}
        atomic_json(report_path, {"status": "running" if len(results) < len(protocol["arms"]) else "completed",
                                  "protocol_sha256": file_sha256(protocol_path), "arms": results,
                                  "segments": evidence, "validation_used_for_selection": False,
                                  "scope": "Training-only fitting diagnosis, not held-out performance or a formal comparison"})
    print("Both bounded diagnostic arms finished. Inspect fitting gates, not only loss.", flush=True)


if __name__ == "__main__":
    main()
