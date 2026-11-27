"""Exercise interrupted/resumed CPU optimization; no research data are used."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.checkpoints import CheckpointStore, FingerprintMismatch, atomic_json, fingerprint


CONFIG = {"model": "tiny_dropout_regression", "seed": 251, "steps": 12,
          "batch_size": 4, "batches_per_epoch": 6, "lr": 0.03, "device": "cpu"}
DATA_SPEC = {"kind": "synthetic_component_verification", "generator_seed": 712, "rows": 24, "columns": 3}


def build_components():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Tanh(),
                                torch.nn.Dropout(0.2), torch.nn.Linear(7, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["lr"], momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    return model, optimizer, scheduler, scaler


def get_store(directory, config_fingerprint=None, data_fingerprint=None):
    return CheckpointStore(directory, config_fingerprint or fingerprint(CONFIG),
                           data_fingerprint or fingerprint(DATA_SPEC), keep_last=3)


def optimize(directory, resume=False, crash=False):
    model, optimizer, scheduler, scaler = build_components()
    store = get_store(directory)
    start = 0
    losses = []
    resumed = None
    if resume:
        resumed = store.load_latest(model, optimizer, scheduler=scheduler, scaler=scaler)
        start = resumed["cursor"]["global_step"]
        if start != resumed["cursor"]["epoch"] * 6 + resumed["cursor"]["next_batch"]:
            raise AssertionError("Batch cursor differs from optimizer progress")
        losses = resumed["extra"]["losses"]
    data_generator = np.random.RandomState(DATA_SPEC["generator_seed"])
    inputs = torch.tensor(data_generator.normal(size=(24, 3)), dtype=torch.float32)
    targets = inputs.sum(dim=1, keepdim=True) * 0.25
    for step in range(start, CONFIG["steps"]):
        epoch, batch = divmod(step, 6)
        # Reconstruct the permutation without consuming the saved global RNG.
        indices = np.random.RandomState(900 + epoch).permutation(24)[batch * 4:(batch + 1) * 4]
        jitter = random.random() * 0.005 + float(np.random.normal(scale=0.002))
        prediction = model(inputs[indices] + jitter)
        loss = torch.nn.functional.mse_loss(prediction, targets[indices])
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        completed = step + 1
        if completed % 2 == 0:
            next_epoch, next_batch = divmod(completed, 6)
            store.save(model, optimizer, next_epoch, next_batch, completed, True,
                       scheduler=scheduler, scaler=scaler, extra={"losses": list(losses)})
        if crash and completed == 5:
            # No finalizer or exception handler may publish the fifth update.
            os._exit(37)
    return {"parameters": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "losses": losses, "scheduler": scheduler.state_dict(), "start_step": start,
            "resume": resumed}


def verify(work_directory):
    work_directory = Path(work_directory)
    reference = optimize(work_directory / "reference")
    script = str(Path(__file__).resolve())
    child = subprocess.run([sys.executable, script, "--crash-worker", str(work_directory / "interrupted")],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if child.returncode != 37:
        raise AssertionError("Interruption worker exit was {}: {}".format(child.returncode, child.stderr))
    recovered = optimize(work_directory / "interrupted", resume=True)
    equal_weights = all(torch.equal(value, recovered["parameters"][key])
                        for key, value in reference["parameters"].items())
    equal_losses = reference["losses"] == recovered["losses"]
    equal_schedule = reference["scheduler"] == recovered["scheduler"]
    if not (equal_weights and equal_losses and equal_schedule and recovered["start_step"] == 4):
        raise AssertionError("Uninterrupted and resumed optimization differ")

    directory = work_directory / "interrupted"
    # Corrupt only test artifacts. A broken pointer must still locate the newest manifest.
    with open(str(directory / "latest.json"), "wb") as stream:
        stream.write(b"interrupted pointer")
    components = build_components()
    pointer_restore = get_store(directory).load_latest(components[0], components[1],
                                                      scheduler=components[2], scaler=components[3])
    if pointer_restore["cursor"]["global_step"] != 12 or not pointer_restore["latest_pointer_problem"]:
        raise AssertionError("Broken latest pointer recovery did not work")
    with open(pointer_restore["checkpoint_path"], "r+b") as stream:
        stream.seek(0)
        stream.write(b"BROKEN")
    components = build_components()
    fallback = get_store(directory).load_latest(components[0], components[1],
                                               scheduler=components[2], scaler=components[3])
    if fallback["cursor"]["global_step"] != 10 or len(fallback["rejected_generations"]) != 1:
        raise AssertionError("Corrupt generation did not fall back to the previous checkpoint")
    mismatches = {}
    for key in ("config", "data"):
        components = build_components()
        kwargs = {key + "_fingerprint": fingerprint({"changed": key})}
        try:
            get_store(directory, **kwargs).load_latest(components[0], components[1],
                                                     scheduler=components[2], scaler=components[3])
        except FingerprintMismatch:
            mismatches[key] = True
        else:
            raise AssertionError("Changed {} fingerprint was accepted".format(key))
    return {
        "status": "passed",
        "purpose": "Checkpoint component verification only; no dataset experiment was performed.",
        "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
        "optimization_device": "cpu", "cuda_rng_captured": torch.cuda.is_available(),
        "steps": CONFIG["steps"], "forced_process_exit_code": child.returncode,
        "crashed_after_step": 5, "resumed_from_step": recovered["start_step"],
        "parameters_bitwise_equal": equal_weights, "loss_history_exactly_equal": equal_losses,
        "scheduler_state_equal": equal_schedule,
        "loss_history": reference["losses"],
        "broken_pointer_recovered_step": pointer_restore["cursor"]["global_step"],
        "corrupt_generation_fallback_step": fallback["cursor"]["global_step"],
        "fingerprint_mismatch_rejected": mismatches,
        "limits": ["Actual electrical power loss was not simulated.",
                   "GPU optimization and mixed precision equivalence were not tested here.",
                   "Exact continuation requires deterministic batch order and transforms.",
                   "Only completed optimizer steps may be checkpointed.",
                   "Storage hardware failure requires a separate backup device."],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash-worker")
    parser.add_argument("--output", default=str(ROOT / "results" / "P0" / "resume_verification.json"))
    arguments = parser.parse_args()
    if arguments.crash_worker:
        optimize(Path(arguments.crash_worker), crash=True)
        raise RuntimeError("Crash worker unexpectedly completed")
    scratch_root = ROOT / "results" / "P0"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resume_check_", dir=str(scratch_root)) as directory:
        report = verify(directory)
    atomic_json(Path(arguments.output), report)
    print(json.dumps({"status": report["status"], "report": str(Path(arguments.output).resolve()),
                      "parameters_bitwise_equal": report["parameters_bitwise_equal"],
                      "loss_history_exactly_equal": report["loss_history_exactly_equal"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
