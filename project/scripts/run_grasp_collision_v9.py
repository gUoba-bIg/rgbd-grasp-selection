"""Resume observed-point collision checks after two committed candidate blocks.

This verifies process-level cache recovery, not physical power loss, grasp
success, or a complete object-geometry collision test.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json, file_sha256, _exclusive_lock

EXPECTED_BLOCKS = 30


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(directory):
    """Include cache contents and membership, excluding the OS lock file."""
    return {str(path.relative_to(directory)): file_sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != ".checkpoint.lock"}


def run_locked():
    protocol = ROOT / "configs/grasp_observed_collision_v9.json"
    script = ROOT / "scripts/verify_grasp_observed_collision_v9.py"
    executable = ROOT / "runtime/envs/grasp_eval/Scripts/python.exe"
    directory = ROOT / "results/P3/grasp_observed_collision_v9"
    proof_path = ROOT / "results/P3/grasp_observed_collision_v9_resume.json"
    completion = directory / "completion.json"
    invocation_path = directory / "last_invocation.json"
    logs = ROOT / "logs/P3/grasp_observed_collision_v9_wrapper"
    if not executable.is_file():
        raise FileNotFoundError("Isolated grasp evaluation interpreter is missing")
    logs.mkdir(parents=True, exist_ok=True)
    prefix = [directory / "blocks" / (name + suffix)
              for name in ("block_0000", "block_0001")
              for suffix in (".json", ".sha256.json")]
    binding = {
        "protocol_sha256": file_sha256(protocol),
        "wrapper_sha256": file_sha256(Path(__file__)),
        "collision_source_sha256": file_sha256(script),
        "interpreter_sha256": file_sha256(executable),
        "helper_sha256": {
            name: file_sha256(ROOT / name)
            for name in ("scripts/pipeline.py", "src/checkpoints.py")},
    }

    def hashes():
        return {str(path.relative_to(ROOT)): file_sha256(path) for path in prefix}

    def invoke(name, extra):
        command = [str(executable), "-B", str(script)] + extra
        log = logs / (name + ".log")
        print("Observed-point collision recovery:", name, flush=True)
        with log.open("ab", buffering=0) as stream:
            stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False)
                          + "\n").encode("utf-8"))
            process = subprocess.run(
                command, cwd=ROOT, env=project_environment(ROOT),
                stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        if process.returncode:
            raise RuntimeError("Collision verification stopped; committed blocks "
                               "preserved. See " + str(log))
        return read(invocation_path)

    def validate_completion(last):
        finished = read(completion)
        if (finished.get("status") != "completed"
                or finished.get("expected_blocks") != EXPECTED_BLOCKS
                or last.get("status") != "completed"
                or last.get("expected_blocks") != EXPECTED_BLOCKS
                or last.get("committed_blocks") != EXPECTED_BLOCKS):
            raise AssertionError("Expected all30 frozen candidate blocks")
        if (last.get("reused_blocks", -1) < 2
                or last.get("new_blocks", -1) < 0
                or last["new_blocks"] + last["reused_blocks"] != EXPECTED_BLOCKS):
            raise AssertionError("The two committed blocks were not reused")
        if finished.get("last_invocation_sha256") != file_sha256(invocation_path):
            raise AssertionError("Completed invocation receipt does not match")

    def check_read_only(name):
        before = snapshot(directory)
        invoke(name, ["--check-only"])
        if snapshot(directory) != before:
            raise AssertionError("Read-only verification changed committed cache")

    prior = read(proof_path) if proof_path.exists() else None
    if prior is not None:
        if any(prior.get(key) != value for key, value in binding.items()):
            raise AssertionError("Recovery code/protocol/interpreter binding changed")
        if prior.get("status") not in ("prefix_committed", "passed"):
            raise AssertionError("Unknown recovery proof status; preserve evidence")
        if hashes() != prior.get("prefix_sha256_before_resume"):
            raise AssertionError("Committed prefix changed")
        if prior["status"] == "passed":
            if file_sha256(completion) != prior.get("completion_sha256"):
                raise AssertionError("Completed collision record changed")
            validate_completion(read(invocation_path))
            check_read_only("read_only_reentry")
            print("Completed30-block recovery proof verified unchanged.", flush=True)
            return

    first_probe = prior.get("first_probe") if prior else None
    if not all(path.exists() for path in prefix):
        first_probe = invoke("two_block_probe", ["--max-new-blocks", "2"])
    if not all(path.exists() for path in prefix):
        raise AssertionError("Initial two collision blocks were not committed")
    before = hashes()
    proof = dict(binding, status="prefix_committed",
                 prefix_sha256_before_resume=before, first_probe=first_probe,
                 expected_blocks=EXPECTED_BLOCKS,
                 scope="Committed-prefix process recovery only; no physical "
                       "power-cut, training, graspAP or robot success claim")
    if prior is None:
        atomic_json(proof_path, proof)
    last = invoke("complete_from_cache", [])
    validate_completion(last)
    after = hashes()
    if before != after:
        raise AssertionError("Committed prefix changed during resume")
    check_read_only("final_read_only_check")
    proof.update(status="passed", prefix_sha256_after_resume=after,
                 committed_prefix_byte_identical=True,
                 final_invocation=last, read_only_check_passed=True,
                 blocks_completed=EXPECTED_BLOCKS, new_training_updates=0,
                 completion_sha256=file_sha256(completion))
    atomic_json(proof_path, proof)
    print("All30 collision blocks complete; first2 retained byte-identically.", flush=True)


def main():
    lock_directory = ROOT / "runtime/grasp_collision_v9/wrapper_lock"
    lock_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(lock_directory):
        run_locked()


if __name__ == "__main__":
    main()
