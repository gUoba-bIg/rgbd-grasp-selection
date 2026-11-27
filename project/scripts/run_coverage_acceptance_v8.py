"""Verify two-frame-to800-frame independent acceptance cache recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json,file_sha256,_exclusive_lock


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    config = ROOT / "configs/segmentation_coverage_v8.json"
    directory = ROOT / "results/P2/segmentation_coverage_v8_acceptance"
    proof_path = ROOT / "results/P2/segmentation_coverage_v8_acceptance_resume.json"
    completion = directory / "completion.json"
    logs = ROOT / "logs/P2/coverage_acceptance_v8"
    logs.mkdir(parents=True,exist_ok=True)
    prefix = [directory/"frames"/(name+suffix)
              for name in ("coverage_control_train_scene_0003_0000","coverage_control_train_scene_0003_0016")
              for suffix in (".json",".sha256.json")]
    binding = {"protocol_sha256":file_sha256(config),"wrapper_sha256":file_sha256(Path(__file__)),
               "acceptance_source_sha256":file_sha256(ROOT/"scripts/accept_segmentation_coverage_v8.py")}

    def hashes():
        return {str(path.relative_to(ROOT)):file_sha256(path) for path in prefix}

    def invoke(name,extra):
        command = [sys.executable,"-B",str(ROOT/"scripts/accept_segmentation_coverage_v8.py")]+extra
        log = logs/(name+".log")
        print("Independent coverage acceptance:",name,flush=True)
        with log.open("ab",buffering=0) as stream:
            stream.write(("\nCOMMAND "+json.dumps(command,ensure_ascii=False)+"\n").encode("utf-8"))
            process = subprocess.run(command,cwd=ROOT,env=project_environment(ROOT),stdout=stream,stderr=subprocess.STDOUT)
            os.fsync(stream.fileno())
        if process.returncode:
            raise RuntimeError("Acceptance stopped; committed inference frames retained. "+str(log))
        return read(directory/"last_invocation.json")

    prior = read(proof_path) if proof_path.exists() else None
    if prior:
        if any(prior.get(key) != value for key,value in binding.items()) or hashes() != prior["prefix_sha256_before_resume"]:
            raise AssertionError("Acceptance prefix/code/protocol changed")
        if prior["status"] == "passed":
            if file_sha256(completion) != prior["completion_sha256"]:
                raise AssertionError("Completed acceptance changed")
            invoke("read_only_reentry",[])
            print("800-frame independent acceptance and recovery verified unchanged.",flush=True)
            return
    first_probe = None
    if not all(path.exists() for path in prefix):
        first_probe = invoke("two_frame_probe",["--max-new-frames","2"])
    if not all(path.exists() for path in prefix):
        raise AssertionError("Initial two acceptance frames were not committed")
    before = hashes()
    proof = dict(binding,status="prefix_committed",prefix_sha256_before_resume=before,
                 first_probe=prior.get("first_probe") if prior else first_probe,
                 scope="Independent inference cache process recovery, not training or physical power-cut testing")
    atomic_json(proof_path,proof)
    last = invoke("complete_from_cache",[])
    finished = read(completion)
    after = hashes()
    if finished["status"] != "completed" or finished["expected_model_frames"] != 800 or last["committed_frames"] != 800:
        raise AssertionError("Independent800-model-frame acceptance is incomplete")
    if before != after or last["reused_frames"] < 2:
        raise AssertionError("Committed acceptance prefix was not reused unchanged")
    proof.update(status="passed",prefix_sha256_after_resume=after,committed_prefix_byte_identical=True,
                 final_invocation=last,model_frame_evaluations=800,new_training_updates=0,
                 completion_sha256=file_sha256(completion))
    atomic_json(proof_path,proof)
    print("800 independent model-frame checks complete; committed prefix preserved.",flush=True)


if __name__ == "__main__":
    lock = ROOT / "runtime/coverage_v8/acceptance_wrapper_lock"
    lock.mkdir(parents=True,exist_ok=True)
    with _exclusive_lock(lock):
        main()
