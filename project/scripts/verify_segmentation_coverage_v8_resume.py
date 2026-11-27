"""Actual CUDA/AMP state recovery across each coverage arm's epoch boundary."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import torch
import torchvision
import numpy as np
import scipy
import PIL
from scripts.verify_gpu_resume import compare_tree, latest_payload
from scripts.verify_segmentation_repair_v5_resume import verify_parent, check_initialization, invoke
from src.checkpoints import atomic_json, file_sha256, fingerprint

ARMS = ("coverage_control", "coverage_expanded")
COMPONENTS = ("model", "optimizer", "scheduler", "scaler", "cursor", "rng", "extra")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def boundary_for(protocol, arm):
    count = protocol["arms"][arm]["train_frames"]
    if count not in (160,320) or protocol["accumulation"] != 2:
        raise ValueError("Unexpected coverage arm size or accumulation")
    return count // 2


def check_partial(result, boundary, resumed=False):
    if result["status"] != "step_limit_reached" or result["optimizer_steps"] != boundary + 2 or result["planned_steps"] != 1600:
        raise AssertionError("Engineering trajectory ended at the wrong step")
    if result["validation_evaluated"] or result.get("evaluation") is not None:
        raise AssertionError("Recovery probes must not evaluate development data")
    if resumed and (result["resumed_from_step"] != boundary or result["updates_this_invocation"] != 2):
        raise AssertionError("Only two lost/new updates should execute after recovery")


def verify(config_path):
    from scripts.train_segmentation_coverage_v8 import source_hashes
    protocol = read(config_path)
    if protocol["schema"] != 8 or set(protocol["arms"]) != set(ARMS) or protocol["planned_steps"] != 1600:
        raise ValueError("Unexpected frozen coverage protocol")
    if not torch.cuda.is_available() or shutil.disk_usage(ROOT).free < (protocol["free_disk_reserve_gib"] + 2) * 2 ** 30:
        raise RuntimeError("CUDA and disk reserve plus temporary checkpoints are required")
    output = ROOT / "results/P2/segmentation_coverage_v8_resume.json"
    logs = ROOT / "logs/P2/segmentation_coverage_v8_resume"
    scratch = ROOT / "runtime/verification"
    logs.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    names = sorted(set(source_hashes()) | {"scripts/verify_segmentation_coverage_v8_resume.py",
        "scripts/verify_segmentation_repair_v5_resume.py", "scripts/verify_gpu_resume.py", "scripts/pipeline.py"})
    binding = {"config_sha256":file_sha256(config_path),
               "data_manifest_sha256":file_sha256(ROOT / protocol["data_manifest"]),
               "source_sha256":{name:file_sha256(ROOT / name) for name in names},
               "runtime_identity":{"python":sys.version.split()[0],"torch":str(torch.__version__),
                   "torchvision":str(torchvision.__version__),"numpy":np.__version__,"scipy":scipy.__version__,
                   "pillow":PIL.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
                   "gpu":torch.cuda.get_device_name(0)}}
    previous = read(output) if output.exists() else None
    if previous:
        if any(previous.get(key) != value for key,value in binding.items()):
            raise RuntimeError("Coverage recovery binding changed; previous evidence preserved")
        if previous["status"] == "passed":
            verify_parent(protocol)
            print("Completed coverage GPU recovery verified unchanged.", flush=True)
            return previous
    parent = verify_parent(protocol)
    report = dict(binding, schema=8, status="running", arms={}, parent=parent,
                  parent_total_optimizer_steps=3200, excluded_from_research_results=True,
                  real_power_cut_tested=False, validation_evaluated=False, checkpoint_interval=2,
                  scope="Actual per-arm CUDA/AMP recovery at epoch boundaries:control80,expanded160; both cross warmup80",
                  device=torch.cuda.get_device_name(0), torch=str(torch.__version__))
    if previous:
        report["arms"] = previous.get("arms",{})
    atomic_json(output, report)
    started = time.perf_counter()
    try:
        for arm in ARMS:
            if report["arms"].get(arm,{}).get("status") == "passed":
                print(arm,"SKIP: completed bound engineering recovery proof",flush=True)
                continue
            boundary = boundary_for(protocol,arm)
            with tempfile.TemporaryDirectory(prefix="coverage_v8_"+arm+"_", dir=str(scratch)) as temporary:
                base = Path(temporary)
                common = [sys.executable,"-B",str(ROOT / "scripts/train_segmentation_coverage_v8.py"),
                          "--config",str(config_path),"--arm",arm,"--max-steps",str(boundary+2),
                          "--checkpoint-interval","2","--device","cuda"]
                processes = {}
                for phase, extra, expected in (("continuous",[],0),
                        ("interrupted",["--crash-after-step",str(boundary+1)],37),("resumed",["--resume"],0)):
                    directory = base / ("continuous" if phase == "continuous" else "interrupted")
                    print("Coverage recovery:",arm,phase,"through",boundary+2,flush=True)
                    processes[phase] = invoke(common+["--run-dir",str(directory)]+extra,logs/(arm+"_"+phase+".log"),expected)
                    if phase == "interrupted":
                        durable = latest_payload(directory)
                        if durable["cursor"] != {"epoch":1,"next_batch":0,"global_step":boundary}:
                            raise AssertionError("Expected saved epoch boundary before forced exit")
                        del durable
                reference, recovered = latest_payload(base/"continuous"),latest_payload(base/"interrupted")
                differences = {key:compare_tree(reference[key],recovered[key]) for key in COMPONENTS}
                checks = {key:not differences[key] for key in COMPONENTS}
                if recovered["cursor"] != {"epoch":1,"next_batch":4,"global_step":boundary+2} or not all(checks.values()):
                    raise AssertionError("Coverage recovery is not bitwise identical: "+str(differences))
                first,last = read(base/"continuous/result.json"),read(base/"interrupted/result.json")
                check_partial(first,boundary)
                check_partial(last,boundary,True)
                initial = {phase:read(base/phase/"initialization.json") for phase in ("continuous","interrupted")}
                initial_checks = {phase:check_initialization(record,parent,protocol,reference["config_fingerprint"]) for phase,record in initial.items()}
                if compare_tree(initial["continuous"],initial["interrupted"]):
                    raise AssertionError("Continuous and interrupted initial states differ")
                history = reference["extra"]["history"]
                if [row["step"] for row in history] != list(range(1,boundary+3)):
                    raise AssertionError("Engineering update history has a gap")
                report["arms"][arm] = {"status":"passed","bitwise_equal":checks,"processes":processes,
                    "initialization":initial["continuous"],"initialization_checks":initial_checks,
                    "config_fingerprint":reference["config_fingerprint"],"data_fingerprint":reference["data_fingerprint"],
                    "final_cursor":recovered["cursor"],"resumed_from_step":boundary,"updates_on_resume":2,
                    "validation_evaluated":False,"loss_history_sha256":fingerprint(history),
                    "learning_rates_first82":[row["learning_rate_used"] for row in history[:82]],
                    "samples_first82_sha256":fingerprint([row["samples"] for row in history[:82]]),
                    "scaler_state_at_final_engineering_step":recovered["scaler"]}
                del reference,recovered
            atomic_json(output,report)
        starts = {arm:{key:value for key,value in report["arms"][arm]["initialization"].items() if key != "run_config_fingerprint"} for arm in ARMS}
        checks = {"same_model_optimizer_scheduler_scaler_rng_cursor_at_step0":not compare_tree(starts[ARMS[0]],starts[ARMS[1]]),
                  "same_learning_rate_first82":not compare_tree(report["arms"][ARMS[0]]["learning_rates_first82"],report["arms"][ARMS[1]]["learning_rates_first82"]),
                  "each_arm_all_state_bitwise_recovery":all(all(report["arms"][arm]["bitwise_equal"].values()) for arm in ARMS)}
        if not all(checks.values()):
            raise AssertionError("Coverage arms did not share initialization/schedule or recover exactly")
        report.update(status="passed",cross_arm_checks=checks,temporary_checkpoints_cleaned=True,
                      sample_order_note="Per-arm order differs intentionally because data coverage differs; only within-arm resume order must match")
    except Exception as error:
        report.update(status="failed",error=type(error).__name__+": "+str(error))
        raise
    finally:
        report["elapsed_seconds_this_invocation"] = time.perf_counter()-started
        atomic_json(output,report)
    print("Both coverage arms passed full-state GPU recovery; shared model and reset state verified.",flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/segmentation_coverage_v8.json")
    args = parser.parse_args()
    verify((ROOT / args.config).resolve())
