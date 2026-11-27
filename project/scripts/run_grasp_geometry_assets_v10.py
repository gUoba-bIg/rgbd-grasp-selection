"""Verify two-file committed-prefix recovery for the bounded V10 asset snapshot."""
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


def asset_snapshot(root, manifest_path, audit_path, commit_path):
    result = {str(path): file_state(path) for path in sorted(root.rglob("*")) if path.is_file()}
    result.update({str(path): file_state(path) for path in (manifest_path, audit_path, commit_path)})
    return result


def verify_complete(manifest_path, audit_path, commit_path):
    manifest, audit, commit = read(manifest_path), read(audit_path), read(commit_path)
    require(manifest.get("status") == "verified_geometry_assets", "Asset transfer did not complete")
    require(len(manifest["files"]) == 31 and set(manifest["files"]) == set(manifest["binding"]["entries"]),
            "Expected exactly31 frozen assets")
    require(audit.get("status") == "passed_integrity_only" and audit.get("selected_file_count") == 31,
            "Integrity-only audit missing")
    require(audit.get("remote_file_count") == 28 and audit.get("copied_local_file_count") == 3,
            "Incorrect remote/local asset split")
    require(audit["files"] == manifest["files"] and audit["binding"] == manifest["binding"],
            "Asset audit binding mismatch")
    require(manifest["audit_sha256"] == file_sha256(audit_path), "Audit hash mismatch")
    require(commit == {"manifest": manifest, "audit": audit}, "Final transaction mismatch")
    return manifest


def run_locked():
    config = ROOT / "configs/grasp_geometry_v10.json"
    protocol = read(config)
    script = ROOT / "scripts/fetch_grasp_geometry_assets_v10.py"
    interpreter = ROOT / "runtime/envs/grasp_eval/Scripts/python.exe"
    data = ROOT / protocol["data_root"]
    manifest_path = ROOT / protocol["data_manifest"]
    audit_path = ROOT / protocol["data_audit"]
    commit_path = ROOT / "runtime/grasp_geometry_v10/assets_commit.json"
    proof_path = ROOT / "results/P3/grasp_geometry_assets_v10_resume.json"
    logs = ROOT / "logs/P3/grasp_geometry_assets_v10_wrapper"
    logs.mkdir(parents=True, exist_ok=True)
    binding = {"protocol_sha256": file_sha256(config), "script_sha256": file_sha256(script),
               "wrapper_sha256": file_sha256(Path(__file__)), "interpreter_sha256": file_sha256(interpreter),
               "helpers": {name: file_sha256(ROOT / name) for name in
                           ("scripts/pipeline.py", "src/checkpoints.py", "src/remote_zip.py")}}

    def invoke(name, extra):
        command = [str(interpreter), "-B", str(script), "--config", str(config)] + extra
        print("Geometry asset recovery:", name, flush=True)
        with (logs / (name + ".log")).open("ab", buffering=0) as stream:
            stream.write(("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
            result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT),
                                    stdout=stream, stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        require(result.returncode == 0, "Asset operation stopped; committed files preserved. See " + str(logs))

    def prefix_state(names):
        return {name: file_state(data / name) for name in names}

    def check_read_only(name):
        before = asset_snapshot(data, manifest_path, audit_path, commit_path)
        invoke(name, ["--check-only"])
        require(asset_snapshot(data, manifest_path, audit_path, commit_path) == before,
                "Completed asset verification changed bytes, membership or timestamps")

    prior = read(proof_path) if proof_path.exists() else None
    if prior is not None:
        require(all(prior.get(key) == value for key, value in binding.items()), "Recovery binding changed")
        require(prior.get("status") in ("prefix_committed", "passed"), "Unknown recovery proof state")
        require(prefix_state(prior["prefix_names"]) == prior["prefix_before"], "Committed file prefix changed")
        if prior["status"] == "passed":
            verify_complete(manifest_path, audit_path, commit_path)
            require(file_sha256(commit_path) == prior["commit_sha256"], "Completed transaction changed")
            check_read_only("read_only_reentry")
            print("Completed31-file asset recovery proof verified unchanged.", flush=True)
            return

    if prior is None:
        existing = read(manifest_path) if manifest_path.exists() else {"files": {}}
        probe_started_with = len(existing["files"])
        if probe_started_with < 2:
            invoke("two_file_probe", ["--max-new-files", str(2 - probe_started_with)])
        prefix = read(manifest_path)
        require(len(prefix["files"]) >= 2, "Two committed assets are required for recovery")
        names = sorted(prefix["files"])[:2]
        before = prefix_state(names)
        require(all(before[name]["sha256"] == prefix["files"][name]["sha256"] for name in names),
                "Prefix does not match manifest")
        prior = dict(binding, status="prefix_committed", prefix_names=names, prefix_before=before,
                     probe_started_with_files=probe_started_with, probe_committed_files=len(prefix["files"]),
                     scope="Committed-file process recovery, not physical power loss or geometry scoring")
        atomic_json(proof_path, prior)

    invoke("complete_from_prefix", [])
    manifest = verify_complete(manifest_path, audit_path, commit_path)
    after = prefix_state(prior["prefix_names"])
    require(after == prior["prefix_before"], "Committed prefix rewritten during recovery")
    check_read_only("final_read_only_check")
    proof = dict(prior, status="passed", prefix_after=after, byte_and_mtime_identical=True,
                 files_completed=31, remote_files=28, local_copies=3, read_only_check_passed=True,
                 manifest_sha256=file_sha256(manifest_path), audit_sha256=file_sha256(audit_path),
                 commit_sha256=file_sha256(commit_path), total_asset_bytes=sum(x["bytes"] for x in manifest["files"].values()))
    atomic_json(proof_path, proof)
    print("All31 geometry assets verified; committed2-file prefix retained unchanged.", flush=True)


def main():
    directory = ROOT / "runtime/grasp_geometry_v10/asset_wrapper_lock"
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(directory):
        run_locked()


if __name__ == "__main__":
    main()
