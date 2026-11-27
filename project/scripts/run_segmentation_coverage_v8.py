"""Run the frozen training-coverage comparison in resumable bounded segments."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_environment
from src.checkpoints import atomic_json,file_sha256

ARMS = ("coverage_control","coverage_expanded")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    from scripts.train_segmentation_coverage_v8 import load_final_model
    import torch, torchvision, numpy as np, scipy, PIL
    path = ROOT / "configs/segmentation_coverage_v8.json"
    protocol = read(path)
    proof = read(ROOT / "results/P2/segmentation_coverage_v8_resume.json")
    if proof["status"] != "passed" or proof["config_sha256"] != file_sha256(path) or proof["data_manifest_sha256"] != file_sha256(ROOT / protocol["data_manifest"]):
        raise RuntimeError("Exact coverage data/protocol GPU recovery must pass first")
    if not all(proof["cross_arm_checks"].values()) or any(not all(proof["arms"][arm]["bitwise_equal"].values()) for arm in ARMS):
        raise RuntimeError("Both coverage arms require shared-start and full-state recovery proof")
    actual_runtime = {"python":sys.version.split()[0],"torch":str(torch.__version__),"torchvision":str(torchvision.__version__),
                      "numpy":np.__version__,"scipy":scipy.__version__,"pillow":PIL.__version__,
                      "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"gpu":torch.cuda.get_device_name(0)}
    if proof["runtime_identity"] != actual_runtime:
        raise RuntimeError("GPU recovery was verified under a different runtime")
    for name,checksum in proof["source_sha256"].items():
        if file_sha256(ROOT / name) != checksum:
            raise RuntimeError("Recovery-verified source changed: "+name)
    arms = {}
    for arm in ARMS:
        directory = ROOT / protocol["arms"][arm]["run_directory"]
        existing = read(directory/"result.json") if (directory/"result.json").exists() else {}
        if existing.get("status") == "completed":
            _,config,_,result,_ = load_final_model(directory,protocol)
            print(arm,"SKIP: verified completed final checkpoint",flush=True)
        else:
            # Preserve all not-yet-reached segment boundaries even after restart.
            current = existing.get("optimizer_steps",0)
            limits = [value for value in protocol["segment_boundaries"] if value > current]
            if not limits:
                limits = [protocol["planned_steps"]]
            for limit in limits:
                command = [sys.executable,"-B",str(ROOT/"scripts/train_segmentation_coverage_v8.py"),
                           "--config",str(path),"--arm",arm,"--run-dir",str(directory),"--max-steps",str(limit)]
                if (directory/"config.json").exists():
                    command.append("--resume")
                log_dir = ROOT / "logs/P2/segmentation_coverage_v8" / arm
                log_dir.mkdir(parents=True,exist_ok=True)
                logfile = log_dir / ("through_"+str(limit)+".log")
                print(arm,"through",limit,"of",protocol["planned_steps"],flush=True)
                with logfile.open("ab",buffering=0) as stream:
                    stream.write(("\nCOMMAND "+json.dumps(command,ensure_ascii=False)+"\n").encode("utf-8"))
                    process = subprocess.run(command,cwd=ROOT,env=project_environment(ROOT),stdout=stream,stderr=subprocess.STDOUT)
                    os.fsync(stream.fileno())
                if process.returncode:
                    raise RuntimeError("Coverage training stopped; checkpoint retained. "+str(logfile))
            _,config,_,result,_ = load_final_model(directory,protocol)
        if config["arm"] != arm:
            raise AssertionError("Completed coverage run has wrong arm identity")
        history = read(directory/"history.json")
        if [row["step"] for row in history] != list(range(1,1601)):
            raise AssertionError("Fixed1600-update history is incomplete")
        arms[arm] = {"run_directory":str(directory.relative_to(ROOT)),"result":result,
                     "result_sha256":file_sha256(directory/"result.json"),"config_sha256":file_sha256(directory/"config.json"),
                     "history_sha256":file_sha256(directory/"history.json"),"initialization_sha256":file_sha256(directory/"initialization.json")}
    report = {"schema":8,"status":"completed","protocol_sha256":file_sha256(path),"arms":arms,
              "new_research_updates":3200,"automatic_extension":False,"thresholds_changed":False,
              "acceptance_pending_independent_check":True,
              "scope":"Fixed update budget coverage strategy comparison; all10 validation scenes are development only"}
    target = ROOT / "results/P2/segmentation_coverage_v8.json"
    if target.exists():
        if read(target) != report:
            raise AssertionError("Completed aggregate differs; old result preserved")
    else:
        atomic_json(target,report)
    print("Both fixed coverage arms complete; independent effect acceptance follows.",flush=True)


if __name__ == "__main__":
    main()
