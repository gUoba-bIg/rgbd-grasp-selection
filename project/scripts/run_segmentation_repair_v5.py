"""Execute two prespecified warm-start arms in durable, bounded segments."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from scripts.train_segmentation_repair_v5 import load_final_model
from src.checkpoints import atomic_json, file_sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    config_path = ROOT / "configs/segmentation_repair_v5.json"
    protocol = read(config_path)
    recovery = read(ROOT / "results/P2/segmentation_repair_v5_resume.json")
    if recovery["status"] != "passed" or recovery["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("The exact frozen two-arm recovery check must pass first")
    if recovery["data_manifest_sha256"] != file_sha256(ROOT / protocol["data_manifest"]):
        raise RuntimeError("The verified data manifest changed")
    for name, digest in recovery["source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise RuntimeError("Recovery-verified source changed: " + name)
    if not all(recovery["cross_arm_checks"].values()) or any(
            recovery["arms"][arm]["status"] != "passed" or not all(recovery["arms"][arm]["bitwise_equal"].values())
            for arm in ("original", "center_mse")):
        raise RuntimeError("Both arms must pass all full-state and shared-start recovery checks")
    arms = {}
    for arm in ("original", "center_mse"):
        directory = ROOT / protocol["arms"][arm]["run_directory"]
        existing = read(directory / "result.json") if (directory / "result.json").exists() else {}
        if existing.get("status") == "completed":
            _, config, _, result, _ = load_final_model(directory, protocol)
            if config["arm"] != arm:
                raise RuntimeError("Completed arm identity mismatch")
            print(arm, "SKIP: verified completed model", flush=True)
        else:
            limits = (protocol["planned_steps"],) if (directory / "config.json").exists() else protocol["segment_boundaries"]
            for limit in limits:
                command = [sys.executable, "-B", str(ROOT / "scripts/train_segmentation_repair_v5.py"),
                           "--config", str(config_path), "--arm", arm, "--run-dir", str(directory),
                           "--max-steps", str(limit)]
                if (directory / "config.json").exists():
                    command.append("--resume")
                logs = ROOT / "logs/P2/segmentation_repair_v5" / arm
                logs.mkdir(parents=True, exist_ok=True)
                log = logs / ("through_" + str(limit) + ".log")
                print(arm, "through", limit, "of", protocol["planned_steps"], flush=True)
                with log.open("ab", buffering=0) as stream:
                    stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
                    process = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                             stdout=stream, stderr=subprocess.STDOUT)
                    os.fsync(stream.fileno())
                if process.returncode:
                    raise RuntimeError("{} stopped with code {}; checkpoints preserved. {}".format(arm, process.returncode, log))
            _, config, _, result, _ = load_final_model(directory, protocol)
        history = read(directory / "history.json")
        if [row["step"] for row in history] != list(range(1, protocol["planned_steps"] + 1)):
            raise RuntimeError("Each arm requires exactly the prespecified optimizer updates")
        arms[arm] = {"run_directory": str(directory.relative_to(ROOT)), "result": result,
                     "result_sha256": file_sha256(directory / "result.json"),
                     "config_sha256": file_sha256(directory / "config.json"),
                     "history_sha256": file_sha256(directory / "history.json"),
                     "initialization_sha256": file_sha256(directory / "initialization.json")}
    atomic_json(ROOT / "results/P2/segmentation_repair_v5.json", {
        "status": "completed", "protocol_sha256": file_sha256(config_path), "arms": arms,
        "new_research_updates": 2 * protocol["planned_steps"],
        "scope": "Two matched warm-start arms; all validation scenes are development data",
        "automatic_extension": False, "thresholds_changed": False,
        "acceptance_pending_independent_check": True})
    print("Both fixed-budget arms complete; proceed to independent arithmetic and adoption checks.", flush=True)


if __name__ == "__main__":
    main()
