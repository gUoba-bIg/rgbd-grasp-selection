"""Run two bounded, resumable real-data engineering pilots and record evidence."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoints import atomic_json, file_sha256
from scripts.pipeline import project_environment


def main():
    audit = json.loads((ROOT / "results/P2/data_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["train_frames"] < 8 or audit["validation_frames"] < 4:
        raise RuntimeError("A verified real-data sample is required")
    if shutil.disk_usage(ROOT).free < 60 * 2 ** 30:
        raise RuntimeError("The 60 GiB free-disk reserve is not available")
    log_dir = ROOT / "logs/P2/segmentation_pilot"
    log_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    evidence = []
    # Each run first stops after four updates, then resumes to the planned 20.
    # Re-entry after completion performs no additional optimizer updates.
    for mode in ("rgb", "rgbd"):
        directory = ROOT / "experiments/segmentation_pilot" / (mode + "_v1")
        for limit in (4, 20):
            command = [sys.executable, str(ROOT / "scripts/train_segmentation_pilot.py"),
                       "--data-root", "data/raw/graspnet", "--mode", mode,
                       "--epochs", "5", "--max-steps", str(limit), "--amp",
                       "--run-dir", str(directory), "--device", "cuda"]
            if (directory / "config.json").exists():
                command.append("--resume")
            filename = log_dir / (mode + "_limit_" + str(limit) + ".log")
            print("Running", mode, "to", limit, "updates; log:", filename, flush=True)
            with filename.open("ab", buffering=0) as stream:
                stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
                result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                        stdout=stream, stderr=subprocess.STDOUT, check=False)
                os.fsync(stream.fileno())
            if result.returncode:
                raise RuntimeError("Pilot stopped with code {}. Inspect {}. Existing checkpoints are preserved.".format(result.returncode, filename))
            report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if report["gpu_peak_allocated_gib"] > 13.5:
                raise RuntimeError("Pilot exceeds the declared 13.5 GiB memory budget")
            evidence.append({"mode": mode, "limit": limit,
                             "resumed_from_step": report["resumed_from_step"],
                             "updates_this_invocation": report["updates_this_invocation"],
                             "completed_step": report["optimizer_steps"],
                             "log": filename.relative_to(ROOT).as_posix()})
        if report["status"] != "completed" or report["optimizer_steps"] != report["planned_steps"]:
            raise RuntimeError("Pilot did not finish its own planned updates")
        reports[mode] = {"directory": directory.relative_to(ROOT).as_posix(), "result": report,
                         "config_sha256": file_sha256(directory / "config.json"),
                         "manifest_sha256": file_sha256(directory / "data_manifest.json")}
    output = {"status": "engineering_pilot_completed", "decision": "仅验收数据到GPU训练和续训链路，不作为方法有效性结论",
              "data_audit_sha256": file_sha256(ROOT / "results/P2/data_audit.json"),
              "runs": reports, "resume_evidence": evidence,
              "limitations": ["Only 8 training and 4 validation frames; random initialization and 5 epochs",
                              "No grasp inference, grasp AP, held-out test score or robot success measured",
                              "RGB and RGB-D early-fusion pilots are not the proposed full study",
                              "Do not infer RGB-D superiority from these small-sample scores"]}
    atomic_json(ROOT / "results/P2/segmentation_pilot.json", output)
    print(json.dumps({"status": output["status"], "steps": {key: value["result"]["optimizer_steps"] for key, value in reports.items()}}), flush=True)


if __name__ == "__main__":
    main()
