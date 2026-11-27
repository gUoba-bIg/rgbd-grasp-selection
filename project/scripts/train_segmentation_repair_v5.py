"""Two bounded model-only warm-start arms with new, identical optimization state."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import PIL
import scipy
import torch
import torchvision

from scripts import train_segmentation_expanded_v3 as expanded
from scripts.render_segmentation_expanded_v3 import prepare_final_model
from scripts.train_segmentation_pilot import prepare_run_files, restore_or_initialize_checkpoint
from src.checkpoints import CheckpointStore, _exclusive_lock, atomic_json, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet, segmentation_loss as original_loss

PARENT_RUN = "experiments/segmentation_expanded_v3/rgbd_original"


def project_path(value):
    return expanded.project_path(value)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_repair_v5.json")
    parser.add_argument("--arm", required=True, choices=("original", "center_mse"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--crash-after-step", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def validate_protocol(protocol, args):
    expected = {"schema": 5, "parent_run": PARENT_RUN, "parent_optimizer_steps": 1600,
                "seed": 20261022, "mode": "rgbd", "input_size": [240, 320], "batch_size": 1,
                "accumulation": 2, "planned_steps": 1600, "equivalent_epochs": 20,
                "total_research_update_budget": 3200, "segment_boundaries": [200, 800, 1600],
                "augment": True, "learning_rate": 0.0003, "weight_decay": 0.0001,
                "warmup_steps": 80, "min_learning_rate": 0.00001, "amp": True,
                "gradient_clip_norm": 1.0, "save_every_steps": 10, "keep_last": 3,
                "free_disk_reserve_gib": 60, "train_scenes": [3, 12, 30, 31, 34, 38, 42, 43, 57, 93],
                "validation_scenes": [13, 18, 22, 33, 39], "primary_development_scenes": [18, 22, 33, 39],
                "previously_seen_validation_scenes": [13], "frames": list(range(0, 256, 16)),
                "decoder": {"foreground_threshold": 0.5, "center_threshold": 0.35, "min_pixels": 20, "max_centers": 64},
                "acceptance": {"primary_scene_macro_foreground_iou_min": 0.70,
                               "primary_scene_macro_instance_f1_min": 0.50, "minimum_tp_per_primary_scene": 1}}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError("Frozen repair protocol differs at " + key)
    if set(protocol["arms"]) != {"original", "center_mse"} or protocol["arms"]["original"]["loss"] != "src.segmentation.segmentation_loss" or protocol["arms"]["center_mse"]["loss"] != "src.segmentation_v2.segmentation_loss" or protocol["arms"]["center_mse"]["center_mse_weight"] != 200:
        raise ValueError("Only the two frozen loss arms are permitted")
    for name in ("parent_checkpoint_sha256", "parent_config_fingerprint", "parent_data_fingerprint"):
        if len(protocol[name]) != 64 or any(character not in "0123456789abcdef" for character in protocol[name]):
            raise ValueError("Invalid parent identity digest: " + name)
    for name in ("max_steps", "crash_after_step"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 1600:
            raise ValueError(name + " must be inside the fixed local plan")
    if args.checkpoint_interval is not None and not 1 <= args.checkpoint_interval <= 10:
        raise ValueError("Checkpoint override must be within 1..10")
    ensure_run_target(project_path(args.run_dir), args.arm)


def ensure_run_target(directory, arm):
    parent = project_path(PARENT_RUN)
    if directory == ROOT or directory == parent or directory.is_relative_to(parent) or parent.is_relative_to(directory):
        raise ValueError("Repair output must not be the parent experiment or one of its ancestors/descendants")
    for ancestor in directory.parents:
        if ancestor == ROOT:
            break
        if (ancestor / "config.json").is_file():
            raise ValueError("A repair run cannot be nested inside another experiment")
    if (directory / "config.json").exists():
        existing = read_json(directory / "config.json")
        if existing.get("schema") != 5 or existing.get("arm") != arm:
            raise ValueError("Existing directory belongs to another experiment or arm")


def model_tensor_fingerprint(state_dict):
    records = {}
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        records[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                         "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return fingerprint(records)


def rng_fingerprint():
    numpy_state = np.random.get_state()
    return fingerprint({"python": random.getstate(),
                        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                        "torch_cpu": model_tensor_fingerprint({"state": torch.get_rng_state()}),
                        "torch_cuda": [model_tensor_fingerprint({"state": state}) for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []})


def load_parent(protocol):
    directory = project_path(protocol["parent_run"])
    model, parent_protocol, _, _, result, evidence = prepare_final_model(directory)
    for field, actual in (("parent_checkpoint_sha256", evidence["checkpoint_sha256"]),
                          ("parent_config_fingerprint", evidence["config_fingerprint"]),
                          ("parent_data_fingerprint", evidence["data_fingerprint"])):
        if protocol[field] != actual:
            raise ValueError("Pinned parent identity mismatch: " + field)
    if evidence["optimizer_steps"] != 1600 or parent_protocol["mode"] != "rgbd":
        raise ValueError("Require the declared completed RGB-D parent")
    manifest, data_root = expanded.load_manifest(protocol)
    if fingerprint(manifest) != protocol["parent_data_fingerprint"]:
        raise ValueError("Repair data does not exactly match the frozen parent data")
    evidence = dict(evidence, parent_run=PARENT_RUN, config_sha256=file_sha256(directory / "config.json"),
                    data_manifest_sha256=file_sha256(directory / "data_manifest.json"),
                    parent_model_tensor_sha256=model_tensor_fingerprint(model.state_dict()),
                    inherited_components=["model"], reset_components=["optimizer", "scheduler", "scaler", "cursor", "random_streams"])
    return model, manifest, data_root, evidence


def source_hashes(arm):
    names = ["scripts/train_segmentation_repair_v5.py", "scripts/train_segmentation_expanded_v3.py",
             "scripts/train_segmentation_pilot.py", "scripts/render_segmentation_expanded_v3.py",
             "src/checkpoints.py", "src/segmentation.py", "src/graspnet_frames.py"]
    if arm == "center_mse":
        names.append("src/segmentation_v2.py")
    return {name: file_sha256(project_path(name)) for name in names}


def select_loss(arm):
    if arm == "original":
        return original_loss
    from src.segmentation_v2 import segmentation_loss
    return segmentation_loss


def build_config(protocol, config_path, manifest, evidence, args):
    return {"schema": 5, "arm": args.arm, "protocol": protocol, "protocol_sha256": file_sha256(config_path),
            "protocol_file": str(config_path.relative_to(ROOT)), "data_fingerprint": fingerprint(manifest),
            "parent_evidence": evidence, "source_sha256": source_hashes(args.arm),
            "device": args.device, "effective_amp": protocol["amp"] and args.device == "cuda",
            "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU component verification",
            "save_every_optimizer_steps": args.checkpoint_interval or protocol["save_every_steps"],
            "engineering_checkpoint_override": args.checkpoint_interval,
            "amp_initial_scale": 128.0, "amp_max_overflow_retries": 16, "num_workers": 0,
            "deterministic_algorithms": True,
            "initialization_policy": "Model-only warm start; all optimization state and RNG reset; local step zero",
            "scheduler_index_rule": "LambdaLR index k sets next update k+1; internal LR clamps at final floor",
            "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                         "torchvision": str(torchvision.__version__), "numpy": np.__version__,
                         "scipy": scipy.__version__, "Pillow": PIL.__version__}}


def initialization_evidence(model, optimizer, scheduler, scaler, config):
    evidence, protocol = config["parent_evidence"], config["protocol"]
    actual = model_tensor_fingerprint(model.state_dict())
    if actual != evidence["parent_model_tensor_sha256"] or optimizer.state_dict()["state"] or scheduler.last_epoch != 0:
        raise RuntimeError("Model-only initialization did not preserve weights and reset optimization")
    return {"schema": 5, "parent_checkpoint_sha256": evidence["checkpoint_sha256"],
            "parent_config_fingerprint": evidence["config_fingerprint"], "parent_data_fingerprint": evidence["data_fingerprint"],
            "parent_model_tensor_sha256": evidence["parent_model_tensor_sha256"], "initial_model_tensor_sha256": actual,
            "model_only_warm_start": True, "optimizer_history_empty": True, "optimizer_state_entries": 0,
            "scheduler_last_epoch": 0, "scheduler_state": scheduler.state_dict(),
            "initial_used_lr": optimizer.param_groups[0]["lr"], "scaler_state": scaler.state_dict(),
            "cursor": {"epoch": 0, "next_batch": 0, "global_step": 0}, "history_length": 0,
            "rng_seed": protocol["seed"], "initial_rng_fingerprint": rng_fingerprint(),
            "run_config_fingerprint": fingerprint(config)}


def final_evaluation(model, data_root, manifest, device, protocol):
    common = {"size": tuple(protocol["input_size"]), "mode": protocol["mode"], "seed": protocol["seed"], "augment": False}
    training = GraspNetFrames(data_root, manifest["records"]["train"], **common)
    development = GraspNetFrames(data_root, manifest["records"]["validation"], **common)
    results = {"train": expanded.evaluate(model, training, device, protocol["decoder"], "160 training frames without augmentation"),
               "development_validation": expanded.evaluate(model, development, device, protocol["decoder"], "All 80 previously inspected development-validation frames; final checkpoint only")}
    rows = results["development_validation"]["frames"]
    for name, scenes, scope in (
        ("primary_development", protocol["primary_development_scenes"], "Historical 64-frame/four-scene subgroup, now development validation"),
        ("previously_seen_validation", protocol["previously_seen_validation_scenes"], "All 16 development-validation frames of scene 13"),
    ):
        results[name] = expanded.summarize_frames([row for row in rows if row["scene"] in scenes], scope)
    primary, threshold = results["primary_development"], protocol["acceptance"]
    checks = {"primary_scene_macro_foreground_iou": primary["scene_macro"]["foreground_pixel_iou"] >= threshold["primary_scene_macro_foreground_iou_min"],
              "primary_scene_macro_instance_f1": primary["scene_macro"]["instance_f1_iou50"] >= threshold["primary_scene_macro_instance_f1_min"],
              "every_primary_scene_has_true_positive": {row["scene"] for row in primary["scenes"]} == set(protocol["primary_development_scenes"]) and all(
                  row["instances"]["tp"] >= threshold["minimum_tp_per_primary_scene"] for row in primary["scenes"])}
    results["acceptance_gate"] = {"passed": all(checks.values()), "checks": checks, "thresholds": threshold,
                                  "interpretation": "Engineering gate only; arm adoption is a separate paired-control decision"}
    return results


def reset_random_streams(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run(args):
    config_path = project_path(args.config)
    protocol = read_json(config_path)
    validate_protocol(protocol, args)
    expanded.ensure_disk_reserve(protocol["free_disk_reserve_gib"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model, manifest, data_root, parent_evidence = load_parent(protocol)
    # Parent loader constructs and loads a CPU model; random streams are reset
    # after that work, and no parent optimizer/scaler/RNG is ever imported.
    reset_random_streams(protocol["seed"], args.device)
    device = torch.device(args.device)
    model.to(device)
    config = build_config(protocol, config_path, manifest, parent_evidence, args)
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    run_directory = project_path(args.run_dir)
    run_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(run_directory):
        prepare_run_files(run_directory, config, manifest, args.resume)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed: expanded.learning_rate_for_update(completed + 1, protocol) / protocol["learning_rate"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["effective_amp"], init_scale=config["amp_initial_scale"])
        initialization = initialization_evidence(model, optimizer, scheduler, scaler, config)
        store = CheckpointStore(run_directory / "checkpoints", config_hash, data_hash, keep_last=protocol["keep_last"])
        restored = restore_or_initialize_checkpoint(run_directory, store, model, optimizer, scheduler, scaler, config)
        epoch, next_batch, global_step = (restored["cursor"][key] for key in ("epoch", "next_batch", "global_step"))
        history = restored["extra"]["history"]
        initialization_path = run_directory / "initialization.json"
        if initialization_path.exists():
            if read_json(initialization_path) != initialization:
                raise RuntimeError("Initialization evidence differs from the pinned model-only start")
        elif global_step:
            raise RuntimeError("Training evidence exists but initialization evidence is missing; refusing reconstruction")
        else:
            atomic_json(initialization_path, initialization)
        if global_step == 0 and (model_tensor_fingerprint(model.state_dict()) != initialization["initial_model_tensor_sha256"] or optimizer.state_dict()["state"] or rng_fingerprint() != initialization["initial_rng_fingerprint"]):
            raise RuntimeError("Committed step-zero state differs from the actual parent-only initialization")
        if [row["step"] for row in history] != list(range(1, global_step + 1)) or restored["extra"].get("config") != config:
            raise RuntimeError("Checkpoint history or configuration mismatch")
        if global_step and restored["extra"].get("initialization") != initialization:
            raise RuntimeError("Checkpoint initialization lineage mismatch")
        train_data = GraspNetFrames(data_root, manifest["records"]["train"], size=tuple(protocol["input_size"]),
                                   mode=protocol["mode"], seed=protocol["seed"], augment=protocol["augment"])
        accumulation = protocol["accumulation"]
        if not 0 <= next_batch < len(train_data) or next_batch % accumulation or global_step != epoch * math.ceil(len(train_data) / accumulation) + next_batch // accumulation:
            raise RuntimeError("Checkpoint epoch/frame cursor mismatch")
        expected_lr = protocol["learning_rate"] * (expanded.learning_rate_for_update(global_step + 1, protocol) / protocol["learning_rate"])
        if scheduler.last_epoch != global_step or optimizer.param_groups[0]["lr"] != expected_lr or scheduler.get_last_lr()[0] != expected_lr:
            raise RuntimeError("Checkpoint learning-rate state mismatch")
        limit = args.max_steps if args.max_steps is not None else protocol["planned_steps"]
        if global_step > limit:
            raise ValueError("Stopping boundary precedes the committed checkpoint")
        start_step, checkpoint_path = global_step, restored["checkpoint_path"]
        loss_function = select_loss(args.arm)
        atomic_json(run_directory / "history.json", history)
        atomic_json(run_directory / "progress.json", {"status": "running", "step": global_step, "target_step": limit,
                    "arm": args.arm, "planned_steps": 1600, "config_fingerprint": config_hash, "data_fingerprint": data_hash})
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while global_step < limit:
            order = train_data.epoch_order(epoch)
            selected = order[next_batch:next_batch + accumulation]
            attempts, used_lr = 0, optimizer.param_groups[0]["lr"]
            while True:
                accumulated_loss, parts = 0.0, {}
                for index in selected:
                    inputs, targets = train_data.get(index, epoch=epoch)
                    targets = {key: value[None].to(device) for key, value in targets.items() if key != "labels"}
                    with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=config["effective_amp"]):
                        outputs = model(inputs[None].to(device))
                        loss, components = loss_function(outputs, targets)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite loss; committed state preserved")
                    scaler.scale(loss / len(selected)).backward()
                    accumulated_loss += float(loss.detach().cpu()) / len(selected)
                    for name, value in components.items():
                        parts[name] = parts.get(name, 0.0) + float(value.cpu()) / len(selected)
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"])
                old_scale = scaler.get_scale()
                if not config["effective_amp"] and not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("Nonfinite gradient; committed state preserved")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    if not bool(torch.isfinite(gradient_norm)):
                        raise RuntimeError("Accepted update has nonfinite gradient norm")
                    break
                attempts += 1
                if attempts >= config["amp_max_overflow_retries"]:
                    raise RuntimeError("Repeated AMP overflow; no skipped updates counted")
            scheduler.step()
            global_step += 1
            finished_epoch = epoch
            next_batch += len(selected)
            if next_batch == len(order):
                epoch, next_batch = epoch + 1, 0
            row = {"step": global_step, "epoch": finished_epoch, "loss": accumulated_loss, "components": parts,
                   "learning_rate_used": used_lr, "learning_rate_next_update": scheduler.get_last_lr()[0],
                   "next_update_index": global_step + 1 if global_step < 1600 else None, "next_update_planned": global_step < 1600,
                   "gradient_norm_before_clip": float(gradient_norm.detach().cpu()), "amp_retries": attempts,
                   "amp_scale": scaler.get_scale(), "samples": [{"scene": train_data.records[index]["scene"],
                       "frame": train_data.records[index]["frame"]} for index in selected]}
            history.append(row)
            should_save = global_step % config["save_every_optimizer_steps"] == 0 or global_step == limit
            if should_save:
                expanded.ensure_disk_reserve(protocol["free_disk_reserve_gib"])
                saved = store.save(model, optimizer, epoch, next_batch, global_step, True, scheduler=scheduler, scaler=scaler,
                                   extra={"history": history, "config": config, "data_fingerprint": data_hash, "initialization": initialization})
                checkpoint_path = str(store.directory / saved["file"])
                atomic_json(run_directory / "history.json", history)
                atomic_json(run_directory / "progress.json", {"status": "running", "step": global_step, "target_step": limit,
                            "arm": args.arm, "epoch": epoch, "next_batch": next_batch, "planned_steps": 1600,
                            "config_fingerprint": config_hash, "data_fingerprint": data_hash})
            print(json.dumps(dict(row, checkpoint_committed=should_save)), flush=True)
            if args.crash_after_step == global_step:
                os._exit(37)
        complete = global_step == 1600
        assessments = final_evaluation(model, data_root, manifest, device, protocol) if complete else None
        report = {"status": "completed" if complete else "step_limit_reached", "arm": args.arm,
                  "config_fingerprint": config_hash, "data_fingerprint": data_hash, "parent_evidence": parent_evidence,
                  "optimizer_steps": global_step, "planned_steps": 1600, "parent_optimizer_steps": 1600,
                  "resumed_from_step": start_step, "updates_this_invocation": global_step - start_step,
                  "cursor": {"epoch": epoch, "next_batch": next_batch, "global_step": global_step},
                  "checkpoint_path": checkpoint_path, "checkpoint_sha256": file_sha256(Path(checkpoint_path)),
                  "evaluation": assessments, "validation_evaluated": complete,
                  "decision": ("engineering_gate_passed" if assessments["acceptance_gate"]["passed"] else "engineering_gate_not_met") if complete else "partial_plan_no_validation_evaluation",
                  "automatic_extension": False, "elapsed_seconds_this_invocation": time.perf_counter() - started,
                  "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
                  "latest_loss": history[-1]["loss"] if history else None,
                  "rejected_checkpoint_generations": restored.get("rejected_generations", []),
                  "latest_pointer_problem": restored.get("latest_pointer_problem"),
                  "limitations": ["Both arms are new optimizer-restart experiments from a shared frozen model, not restoration of the parent optimizer",
                                  "All evaluated nontraining scenes are development validation already inspected",
                                  "One seed and small development data cannot establish method superiority or independent generalization",
                                  "No early stopping, threshold tuning, best-checkpoint selection or automatic budget extension",
                                  "Micro instance F1 at IoU0.50 is not mask AP; no grasp or robot success evaluation"]}
        atomic_json(run_directory / "result.json", report)
        atomic_json(run_directory / "progress.json", {key: report[key] for key in (
            "status", "arm", "optimizer_steps", "planned_steps", "cursor", "config_fingerprint", "data_fingerprint", "decision")})
        print(json.dumps({"status": report["status"], "arm": args.arm, "decision": report["decision"],
                          "optimizer_steps": global_step, "result": str(run_directory / "result.json")}, ensure_ascii=False), flush=True)
        return report


def load_final_model(directory, protocol):
    """Return (CPU model, run config, manifest, result, evidence), fully rechecked."""
    directory = project_path(directory)
    config, manifest, result = (read_json(directory / name) for name in ("config.json", "data_manifest.json", "result.json"))
    if config.get("schema") != 5 or config["protocol"] != protocol or config["arm"] not in protocol["arms"]:
        raise ValueError("Final repair configuration mismatch")
    args = arguments(["--config", config["protocol_file"], "--arm", config["arm"], "--run-dir", str(directory), "--device", config["device"]])
    args.checkpoint_interval = config["engineering_checkpoint_override"]
    validate_protocol(protocol, args)
    _, current_manifest, _, parent_evidence = load_parent(protocol)
    if config != build_config(protocol, project_path(config["protocol_file"]), current_manifest, parent_evidence, args) or manifest != current_manifest:
        raise ValueError("Final config/data/source/parent binding changed")
    config_hash, data_hash = fingerprint(config), fingerprint(manifest)
    if result["status"] != "completed" or result["optimizer_steps"] != 1600 or result["planned_steps"] != 1600 or not result["validation_evaluated"] or result["arm"] != config["arm"]:
        raise ValueError("Final repair checkpoint is not a completed 1600-update arm")
    if result["config_fingerprint"] != config_hash or result["data_fingerprint"] != data_hash or result["parent_evidence"] != parent_evidence:
        raise ValueError("Final result identity mismatch")
    checkpoint_directory = directory / "checkpoints"
    pointer = read_json(checkpoint_directory / "latest.json")
    metadata_path = (checkpoint_directory / pointer["manifest"]).resolve()
    if not metadata_path.is_relative_to(checkpoint_directory) or file_sha256(metadata_path) != pointer["manifest_sha256"]:
        raise ValueError("Final checkpoint receipt mismatch")
    metadata = read_json(metadata_path)
    checkpoint_path = (checkpoint_directory / metadata["file"]).resolve()
    if not checkpoint_path.is_relative_to(checkpoint_directory) or file_sha256(checkpoint_path) != metadata["sha256"] or checkpoint_path.stat().st_size != metadata["size_bytes"]:
        raise ValueError("Final checkpoint checksum/size mismatch")
    if checkpoint_path != project_path(result["checkpoint_path"]) or metadata["sha256"] != result["checkpoint_sha256"]:
        raise ValueError("Final result and weight file disagree")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for value in (metadata, payload):
        if value["config_fingerprint"] != config_hash or value["data_fingerprint"] != data_hash:
            raise ValueError("Final checkpoint binding mismatch")
    if metadata["global_step"] != 1600 or payload["cursor"] != {"epoch": 20, "next_batch": 0, "global_step": 1600} or payload["extra"]["config"] != config:
        raise ValueError("Final checkpoint cursor/config mismatch")
    if result["cursor"] != payload["cursor"] or payload["extra"]["data_fingerprint"] != data_hash:
        raise ValueError("Final result cursor or embedded data identity mismatch")
    history = read_json(directory / "history.json")
    if history != payload["extra"]["history"] or [row["step"] for row in history] != list(range(1, 1601)):
        raise ValueError("Final history differs from the complete checkpoint-embedded history")
    initialization = read_json(directory / "initialization.json")
    if payload["extra"]["initialization"] != initialization or initialization["run_config_fingerprint"] != config_hash or initialization["initial_model_tensor_sha256"] != parent_evidence["parent_model_tensor_sha256"]:
        raise ValueError("Final initialization evidence mismatch")
    model = PilotInstanceNet(mode=protocol["mode"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    evidence = {"arm": config["arm"], "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": metadata["sha256"], "config_fingerprint": config_hash, "data_fingerprint": data_hash,
                "result_sha256": file_sha256(directory / "result.json"), "initialization_sha256": file_sha256(directory / "initialization.json"),
                "parent_evidence": parent_evidence, "local_optimizer_steps": 1600}
    return model, config, manifest, result, evidence


if __name__ == "__main__":
    try:
        run(arguments())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("Repair training stopped: {}".format(error), file=sys.stderr, flush=True)
        sys.exit(2)
