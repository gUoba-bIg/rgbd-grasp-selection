"""Execute frozen-weight diagnosis with a committed-prefix recovery check."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json, file_sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    config_path = ROOT / "configs/segmentation_expanded_v3_diagnosis.json"
    config = read(config_path)
    cache = ROOT / config["output_directory"]
    proof_path = ROOT / "results/P2/segmentation_expanded_v3_diagnosis_resume.json"
    logs = ROOT / "logs/P2/segmentation_diagnosis_v4"
    logs.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, "-B", str(ROOT / "scripts/diagnose_segmentation_expanded_v3.py"),
              "--config", str(config_path)]
    names = ["train_scene_0003_0000", "train_scene_0003_0016"]
    prefix = [cache / "frames" / (name + suffix) for name in names for suffix in (".json", ".sha256.json")]

    def invoke(name, extra):
        logfile = logs / (name + ".log")
        command = common + extra
        print("Frozen-model diagnosis:", name, flush=True)
        with logfile.open("ab", buffering=0) as stream:
            stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
            process = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                     stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        if process.returncode:
            raise RuntimeError("Diagnosis stopped with code {}; committed frames retained. See {}".format(process.returncode, logfile))
        return read(cache / "last_invocation.json")

    probe = None
    if not all(path.exists() for path in prefix):
        probe = invoke("two_frame_probe", ["--max-frames", "2"])
    if not all(path.exists() for path in prefix):
        raise RuntimeError("The first two frozen training frames were not committed")
    before = {str(path.relative_to(ROOT)): file_sha256(path) for path in prefix}
    proof = {"status": "prefix_committed", "protocol_sha256": file_sha256(config_path),
             "probe": probe, "prefix_sha256_before_resume": before,
             "scope": "Segmented analysis cache recovery, not training and not a physical power-cut test"}
    if proof_path.exists():
        previous = read(proof_path)
        if previous["protocol_sha256"] != proof["protocol_sha256"]:
            raise RuntimeError("Diagnosis protocol changed; previous recovery evidence is preserved")
        proof["first_probe"] = previous.get("first_probe", previous.get("probe"))
    else:
        proof["first_probe"] = probe
    atomic_json(proof_path, proof)
    invocation = invoke("complete_from_cache", [])
    after = {str(path.relative_to(ROOT)): file_sha256(path) for path in prefix}
    final = read(ROOT / config["report"])
    if final["status"] != "completed" or final["frames_completed"] != 240:
        raise AssertionError("The frozen 240-frame diagnostic is incomplete")
    if before != after or invocation["skipped"] < 2:
        raise AssertionError("Previously committed diagnostic frames were not preserved and skipped")
    proof.update({"status": "passed", "prefix_sha256_after_resume": after,
                  "committed_prefix_byte_identical": True, "final_invocation": invocation,
                  "frames_completed": 240, "new_training_updates": 0,
                  "report_sha256": file_sha256(ROOT / config["report"])})
    atomic_json(proof_path, proof)
    print("240-frame diagnosis complete; the committed prefix was skipped unchanged.", flush=True)


if __name__ == "__main__":
    main()
