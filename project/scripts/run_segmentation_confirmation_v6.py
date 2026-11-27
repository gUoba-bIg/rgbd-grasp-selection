"""Commit two paired frames, resume all80, and preserve verifiable recovery proof."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json, file_sha256, _exclusive_lock


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    config_path = ROOT / "configs/segmentation_confirmation_v6.json"
    config = read(config_path)
    cache = ROOT / config["output_directory"]
    proof_path = ROOT / config["resume_proof"]
    report_path = ROOT / config["report"]
    log_dir = ROOT / "logs/P2/segmentation_confirmation_v6"
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = [cache / "frames" / (name + suffix) for name in ("scene_0002_0000", "scene_0002_0016") for suffix in (".json", ".sha256.json")]
    common = [sys.executable, "-B", str(ROOT / "scripts/evaluate_segmentation_confirmation_v6.py"), "--config", str(config_path)]

    def hashes():
        return {str(path.relative_to(ROOT)): file_sha256(path) for path in prefix}

    def invoke(name, extra):
        path = log_dir / (name + ".log")
        command = common + extra
        print("Fixed confirmation:", name, flush=True)
        with path.open("ab", buffering=0) as stream:
            stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
            result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT), stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        if result.returncode:
            raise RuntimeError("Confirmation interrupted; committed frames preserved. See " + str(path))
        return read(cache / "last_invocation.json")

    prior = read(proof_path) if proof_path.exists() else None
    binding = {"protocol_sha256": file_sha256(config_path), "wrapper_sha256": file_sha256(Path(__file__))}
    if prior:
        if any(prior.get(key) != value for key, value in binding.items()):
            raise AssertionError("Recovery proof belongs to different code/protocol")
        if hashes() != prior["prefix_sha256_before_resume"]:
            raise AssertionError("Committed prefix changed since recovery proof was saved")
        if prior["status"] == "passed":
            if prior["report_sha256"] != file_sha256(report_path):
                raise AssertionError("Completed confirmation report changed")
            invoke("read_only_reentry", [])
            print("Completed confirmation and recovery verified unchanged.", flush=True)
            return
    probe = None
    if not all(path.exists() for path in prefix):
        probe = invoke("two_frame_probe", ["--max-frames", "2"])
    if not all(path.exists() for path in prefix):
        raise AssertionError("First two paired frames were not committed")
    before = hashes()
    proof = dict(binding, status="prefix_committed", prefix_sha256_before_resume=before,
                 first_probe=prior.get("first_probe") if prior else probe,
                 scope="Two-paired-frame committed-prefix process recovery; not a physical power-cut test")
    atomic_json(proof_path, proof)
    invocation = invoke("complete_from_cache", [])
    report = read(report_path)
    after = hashes()
    if report["status"] != "completed" or report["frames_completed"] != 80:
        raise AssertionError("Expected the entire predeclared80-frame paired report")
    if before != after or invocation["skipped"] < 2:
        raise AssertionError("Committed prefix was not skipped unchanged")
    proof.update(status="passed", prefix_sha256_after_resume=after, committed_prefix_byte_identical=True,
                 final_invocation=invocation, frames_completed=80, model_frame_evaluations=160,
                 new_training_updates=0, report_sha256=file_sha256(report_path))
    atomic_json(proof_path, proof)
    print("All80 paired frames complete; first2 committed frames retained byte-identically.", flush=True)


if __name__ == "__main__":
    lock_directory = ROOT / "runtime/confirmation_v6/wrapper_lock"
    lock_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(lock_directory):
        main()
