"""Commit two geometry scores, resume the remainder, then verify read-only reentry."""
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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_state(path):
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns}


def snapshot(directory, report):
    result = {str(p): file_state(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    if report.exists():
        result[str(report)] = file_state(report)
    return result


def validate_completion(completion_path, invocation_path, report_path):
    completion, last, report = read(completion_path), read(invocation_path), read(report_path)
    count = completion.get("expected_grasps")
    require(type(count) is int and count >= 2, "Insufficient selected grasps for two-record recovery probe")
    require(completion.get("status") == "completed" and last.get("status") == "completed",
            "Geometry scoring did not complete")
    require(last.get("expected_grasps") == count and last.get("committed_grasps") == count,
            "Incomplete selected candidate set")
    require(last.get("reused_grasps", -1) >= 2 and last.get("new_grasps", -1) >= 0 and
            last["new_grasps"] + last["reused_grasps"] == count, "Committed prefix was not reused")
    require(report.get("status") == "accepted_limited_single_frame_geometry_engineering" and
            report.get("selected_count") == count and report.get("official_oracle_exact_match") is True,
            "Official single-frame reference equality not accepted")
    require(report.get("dataset_ap_computed") is False and report.get("robot_success_rate_computed") is False,
            "Single-frame result scope changed")
    return count, last


def run_locked():
    config = ROOT / "configs/grasp_geometry_v10.json"
    script = ROOT / "scripts/verify_grasp_geometry_v10.py"
    interpreter = ROOT / "runtime/envs/grasp_eval/Scripts/python.exe"
    directory = ROOT / "results/P3/grasp_geometry_v10"
    report_path = ROOT / "results/P3/grasp_geometry_v10.json"
    proof_path = ROOT / "results/P3/grasp_geometry_v10_resume.json"
    invocation = directory / "last_invocation.json"
    completion = directory / "completion.json"
    logs = ROOT / "logs/P3/grasp_geometry_v10_wrapper"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = [directory / "grasps" / (name + suffix)
              for name in ("grasp_0000", "grasp_0001") for suffix in (".json", ".sha256.json")]
    binding = {"protocol_sha256": file_sha256(config), "scoring_script_sha256": file_sha256(script),
               "wrapper_sha256": file_sha256(Path(__file__)), "interpreter_sha256": file_sha256(interpreter),
               "helper_sha256": {name: file_sha256(ROOT / name) for name in
                                 ("scripts/pipeline.py", "src/checkpoints.py", "scripts/verify_grasp_observed_collision_v9.py")}}

    def prefix_state():
        return {p.relative_to(ROOT).as_posix(): file_state(p) for p in prefix}

    def invoke(name, extra):
        command = [str(interpreter), "-B", str(script), "--config", str(config)] + extra
        print("Single-frame geometry recovery:", name, flush=True)
        with (logs / (name + ".log")).open("ab", buffering=0) as stream:
            stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
            result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                    stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        require(result.returncode == 0, "Geometry verification stopped; committed candidates preserved. See " + str(logs))
        return read(invocation)

    def read_only(name):
        before = snapshot(directory, report_path)
        invoke(name, ["--check-only"])
        require(snapshot(directory, report_path) == before, "Read-only verification changed committed geometry output")

    prior = read(proof_path) if proof_path.exists() else None
    if prior is not None:
        require(all(prior.get(k) == v for k, v in binding.items()), "Geometry recovery code binding changed")
        require(prior.get("status") in ("prefix_committed", "passed"), "Unknown geometry recovery proof state")
        require(prefix_state() == prior["prefix_before"], "Committed candidate prefix changed")
        if prior["status"] == "passed":
            require(file_sha256(completion) == prior["completion_sha256"], "Completed geometry receipt changed")
            validate_completion(completion, invocation, report_path)
            read_only("completed_read_only_reentry")
            print("Completed single-frame geometry recovery proof verified unchanged.", flush=True)
            return

    probe = prior.get("first_probe") if prior else None
    if not all(path.exists() for path in prefix):
        probe = invoke("two_candidate_probe", ["--max-new-grasps", "2"])
    require(all(path.exists() for path in prefix), "First two scores were not committed")
    before = prefix_state()
    if prior is None:
        prior = dict(binding, status="prefix_committed", prefix_before=before, first_probe=probe,
                     scope="Two committed scores followed by resumed completion and exact official reference; not physical power loss")
        atomic_json(proof_path, prior)
    invoke("complete_from_prefix", [])
    count, last = validate_completion(completion, invocation, report_path)
    after = prefix_state()
    require(after == before, "Committed candidate prefix rewritten during recovery")
    read_only("final_read_only_check")
    proof = dict(prior, status="passed", prefix_after=after, byte_and_mtime_identical=True,
                 read_only_check_passed=True, selected_grasps_completed=count, final_invocation=last,
                 new_training_updates=0, completion_sha256=file_sha256(completion),
                 report_sha256=file_sha256(report_path))
    atomic_json(proof_path, proof)
    print("All%d selected grasps complete; first2 preserved and official reference matched." % count, flush=True)


def main():
    directory = ROOT / "runtime/grasp_geometry_v10/score_wrapper_lock"
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(directory):
        run_locked()


if __name__ == "__main__":
    main()
