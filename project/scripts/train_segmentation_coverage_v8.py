"""Same-checkpoint, fixed-update comparison of ten versus twenty training scenes."""

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
from scripts import train_segmentation_repair_v5 as parent_trainer
from scripts.train_segmentation_pilot import prepare_run_files, restore_or_initialize_checkpoint
from src.checkpoints import CheckpointStore, _exclusive_lock, atomic_json, file_sha256, fingerprint
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet
from src.segmentation_v2 import segmentation_loss

PARENT_RUN = "experiments/segmentation_repair_v5/center_mse"
ARMS = ("coverage_control", "coverage_expanded")
OLD_TRAIN = [3, 12, 30, 31, 34, 38, 42, 43, 57, 93]
NEW_TRAIN = [4, 5, 8, 9, 10, 14, 16, 17, 19, 20]


def project_path(value):
    return expanded.project_path(value)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_coverage_v8.json")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--crash-after-step", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def validate_protocol(protocol, args):
    expected = {"schema": 8, "parent_run": PARENT_RUN, "parent_total_optimizer_steps": 3200,
                "seed": 20261022, "mode": "rgbd", "input_size": [240, 320], "batch_size": 1,
                "accumulation": 2, "planned_steps": 1600, "total_research_update_budget": 3200,
                "segment_boundaries": [200, 800, 1600], "augment": True, "learning_rate": .0003,
                "weight_decay": .0001, "warmup_steps": 80, "min_learning_rate": .00001, "amp": True,
                "gradient_clip_norm": 1., "save_every_steps": 10, "keep_last": 3, "free_disk_reserve_gib": 60,
                "loss": "src.segmentation_v2.segmentation_loss", "center_mse_weight": 200,
                "new_train_scenes": NEW_TRAIN, "camera": "realsense",
                "validation_scenes": [2, 6, 7, 11, 13, 15, 18, 22, 33, 39],
                "historical_validation_scenes": [13, 18, 22, 33, 39], "round6_validation_scenes": [2, 6, 7, 11, 15],
                "remaining_unopened_validation_scenes": [52, 59, 75, 79, 80], "frames": list(range(0, 256, 16)),
                "decoder": {"foreground_threshold": .5, "center_threshold": .35, "min_pixels": 20, "max_centers": 64},
                "acceptance": {"scene_macro_foreground_iou_min": .70, "scene_macro_instance_f1_min": .50, "minimum_tp_per_scene": 1},
                "candidate_adoption": {"scene_macro_f1_gain_over_control_min": .02, "scene_macro_foreground_iou_drop_max": .01,
                                       "minimum_scenes_with_positive_f1_gain": 8}}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError("Frozen coverage protocol differs at " + key)
    if set(protocol["arms"]) != set(ARMS) or args.arm not in ARMS:
        raise ValueError("Require the two frozen coverage arms")
    for arm, scenes, frames, epochs in ((ARMS[0], OLD_TRAIN, 160, 20), (ARMS[1], sorted(OLD_TRAIN + NEW_TRAIN), 320, 10)):
        actual = protocol["arms"][arm]
        if actual["train_scenes"] != scenes or actual["train_frames"] != frames or actual["equivalent_epochs"] != epochs:
            raise ValueError("Coverage arm selection/exposure budget differs")
    split = read_json(project_path(protocol["split"]))["groups"]
    if sorted(set(split["train"]) - set(OLD_TRAIN))[:10] != NEW_TRAIN:
        raise ValueError("New scenes violate the frozen training-only ascending rule")
    if not set(OLD_TRAIN + NEW_TRAIN).issubset(split["train"]) or not set(protocol["validation_scenes"]).issubset(split["validation"]):
        raise ValueError("Declared data do not stay within their split")
    if set(protocol["validation_scenes"]) & set(OLD_TRAIN + NEW_TRAIN):
        raise ValueError("Training/validation overlap")
    for name in ("parent_checkpoint_sha256", "parent_config_fingerprint", "parent_data_fingerprint", "parent_protocol_sha256"):
        if len(protocol[name]) != 64 or any(character not in "0123456789abcdef" for character in protocol[name]):
            raise ValueError("Invalid parent digest")
    for name in ("max_steps", "crash_after_step"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 1600:
            raise ValueError(name + " must remain within the fixed update budget")
    if args.checkpoint_interval is not None and not 1 <= args.checkpoint_interval <= 10:
        raise ValueError("Checkpoint interval must stay within1..10")
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
        if existing.get("schema") != 8 or existing.get("arm") != arm:
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


def load_manifest(protocol, arm):
    """Verify the entire2040-file snapshot, then select this arm's training records."""
    data_root = project_path(protocol["data_root"])
    manifest_path, audit_path = project_path(protocol["data_manifest"]), project_path(protocol["data_audit"])
    manifest, audit = read_json(manifest_path), read_json(audit_path)
    if manifest.get("schema") != 8 or manifest.get("status") != "verified_coverage_sample":
        raise ValueError("Coverage data are not fully verified")
    if project_path(manifest["data_root"]) != data_root or manifest["split_sha256"] != file_sha256(project_path(protocol["split"])):
        raise ValueError("Coverage root/split binding differs")
    expected_audit = {"status": "passed", "manifest_sha256": file_sha256(manifest_path),
                      "files_verified": 2040, "train_frames": 320, "validation_frames": 160}
    if any(audit.get(key) != value for key, value in expected_audit.items()) or audit.get("errors") or audit.get("hash_failures"):
        raise ValueError("Coverage audit is missing, failed or has a different manifest")
    protocol_hash = file_sha256(project_path("configs/segmentation_coverage_v8.json"))
    if manifest["protocol_sha256"] != protocol_hash or audit["protocol_sha256"] != protocol_hash:
        raise ValueError("Coverage data do not bind the frozen v8 protocol")
    if audit["split_sha256"] != manifest["split_sha256"] or audit["code_sha256"] != manifest["code_sha256"]:
        raise ValueError("Coverage data audit/source binding differs")
    for relative, digest in manifest["code_sha256"].items():
        if file_sha256(project_path(relative)) != digest:
            raise ValueError("Frozen coverage data code changed: " + relative)
    for key in ("frames", "camera", "new_train_scenes", "remaining_unopened_validation_scenes"):
        if manifest["selection"][key] != protocol[key]:
            raise ValueError("Coverage selection differs: " + key)
    if manifest["selection"]["control_train_scenes"] != OLD_TRAIN:
        raise ValueError("Control training selection differs")
    for prior in protocol["prior_snapshots"]:
        if file_sha256(project_path(prior["manifest"])) != prior["sha256"]:
            raise ValueError("Prior immutable snapshot manifest changed")
    if set(manifest["records"]) != {"train", "validation"}:
        raise ValueError("Only training and development validation allowed")
    expected_files = set()
    for group, scenes in (("train", protocol["arms"][ARMS[1]]["train_scenes"]), ("validation", protocol["validation_scenes"])):
        if manifest["selection"][group + "_scenes"] != scenes:
            raise ValueError("Manifest scene selection differs")
        records = manifest["records"][group]
        expected = [(scene, "{:04d}".format(frame)) for scene in scenes for frame in protocol["frames"]]
        if [(row["scene"], row["frame"]) for row in records] != expected:
            raise ValueError("Require every frozen frame once, in declared order")
        for row in records:
            if row["camera"] != "realsense" or set(row["files"]) != {"rgb", "depth", "label", "meta"} or set(row["sha256"]) != set(row["files"]):
                raise ValueError("Invalid coverage record modalities/camera")
            for kind, relative in row["files"].items():
                wanted = "scenes/scene_{:04d}/realsense/{}/{}{}".format(row["scene"], kind, row["frame"], ".mat" if kind == "meta" else ".png")
                if relative.replace("\\", "/") != wanted or manifest["files"][wanted]["sha256"] != row["sha256"][kind]:
                    raise ValueError("Coverage record path/checksum identity differs")
                expected_files.add(wanted)
        for scene in scenes:
            prefix = "scenes/scene_{:04d}/".format(scene)
            expected_files.add(prefix + "object_id_list.txt")
            expected_files.update(prefix + "realsense/" + name for name in ("camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"))
    if set(manifest["files"]) != expected_files or len(expected_files) != 2040:
        raise ValueError("Require exactly2040 frozen data files")
    for relative, receipt in manifest["files"].items():
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root) or not path.is_file() or path.stat().st_size != receipt["bytes"] or file_sha256(path) != receipt["sha256"]:
            raise ValueError("Coverage data bytes changed: " + relative)
    training = [row for row in manifest["records"]["train"] if row["scene"] in protocol["arms"][arm]["train_scenes"]]
    if len(training) != protocol["arms"][arm]["train_frames"]:
        raise ValueError("Wrong per-arm training subset")
    return {"schema": 8, "arm": arm, "data_root": str(data_root),
            "records": {"train": training, "validation": manifest["records"]["validation"]},
            "source_manifest": manifest, "source_manifest_sha256": file_sha256(manifest_path),
            "verified_audit_sha256": file_sha256(audit_path), "split_sha256": manifest["split_sha256"],
            "validation_policy": "Only hashes read before final1600 updates; all10 scenes are already development validation"}, data_root


def load_parent(protocol, arm):
    directory = project_path(protocol["parent_run"])
    parent_path = project_path(protocol["parent_protocol"])
    if file_sha256(parent_path) != protocol["parent_protocol_sha256"]:
        raise ValueError("Pinned parent protocol changed")
    model, config, _old_manifest, result, evidence = parent_trainer.load_final_model(directory, read_json(parent_path))
    for field, actual in (("parent_checkpoint_sha256", evidence["checkpoint_sha256"]),
                          ("parent_config_fingerprint", evidence["config_fingerprint"]),
                          ("parent_data_fingerprint", evidence["data_fingerprint"])):
        if protocol[field] != actual:
            raise ValueError("Pinned parent identity mismatch: " + field)
    if evidence["local_optimizer_steps"] != 1600 or config["arm"] != "center_mse" or config["protocol"]["mode"] != "rgbd":
        raise ValueError("Require the completed v5 center-MSE parent")
    manifest, data_root = load_manifest(protocol, arm)
    evidence = dict(evidence, parent_run=PARENT_RUN, parent_total_optimizer_steps=3200,
                    config_sha256=file_sha256(directory / "config.json"),
                    data_manifest_sha256=file_sha256(directory / "data_manifest.json"),
                    parent_model_tensor_sha256=model_tensor_fingerprint(model.state_dict()),
                    inherited_components=["model"], reset_components=["optimizer", "scheduler", "scaler", "cursor", "random_streams"])
    return model, manifest, data_root, evidence


def source_hashes(arm=None):
    names = set(parent_trainer.source_hashes("center_mse")) | {"scripts/train_segmentation_coverage_v8.py", "scripts/accept_segmentation_expanded_v3.py"}
    return {name: file_sha256(project_path(name)) for name in sorted(names)}


def select_loss(arm):
    if arm not in ARMS:
        raise ValueError("Unknown coverage arm")
    return segmentation_loss


def build_config(protocol, config_path, manifest, evidence, args):
    return {"schema": 8, "arm": args.arm, "protocol": protocol, "protocol_sha256": file_sha256(config_path),
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
    return {"schema": 8, "parent_checkpoint_sha256": evidence["checkpoint_sha256"],
            "parent_config_fingerprint": evidence["config_fingerprint"], "parent_data_fingerprint": evidence["data_fingerprint"],
            "parent_model_tensor_sha256": evidence["parent_model_tensor_sha256"], "initial_model_tensor_sha256": actual,
            "model_only_warm_start": True, "optimizer_history_empty": True, "optimizer_state_entries": 0,
            "scheduler_last_epoch": 0, "scheduler_state": scheduler.state_dict(),
            "initial_used_lr": optimizer.param_groups[0]["lr"], "scaler_state": scaler.state_dict(),
            "cursor": {"epoch": 0, "next_batch": 0, "global_step": 0}, "history_length": 0,
            "rng_seed": protocol["seed"], "initial_rng_fingerprint": rng_fingerprint(),
            "run_config_fingerprint": fingerprint(config)}


def acceptance_gate(partition, protocol):
    threshold = protocol["acceptance"]
    checks = {"scene_macro_foreground_iou": partition["scene_macro"]["foreground_pixel_iou"] >= threshold["scene_macro_foreground_iou_min"],
              "scene_macro_instance_f1": partition["scene_macro"]["instance_f1_iou50"] >= threshold["scene_macro_instance_f1_min"],
              "every_scene_has_true_positive": {row["scene"] for row in partition["scenes"]} == set(protocol["validation_scenes"]) and all(
                  row["instances"]["tp"] >= threshold["minimum_tp_per_scene"] for row in partition["scenes"])}
    return {"passed": all(checks.values()), "checks": checks, "thresholds": threshold,
            "interpretation": "Development-only engineering gate; paired adoption is separately assessed"}


def final_evaluation(model, data_root, manifest, device, protocol, arm):
    common = {"size": tuple(protocol["input_size"]), "mode": protocol["mode"], "seed": protocol["seed"], "augment": False}
    training = GraspNetFrames(data_root, manifest["records"]["train"], **common)
    development = GraspNetFrames(data_root, manifest["records"]["validation"], **common)
    results = {"train": expanded.evaluate(model, training, device, protocol["decoder"], str(len(training)) + " arm-specific training frames without augmentation"),
               "development_validation": expanded.evaluate(model, development, device, protocol["decoder"], "All160 development frames; final checkpoint only")}
    rows = results["development_validation"]["frames"]
    for name, scenes in (("historical_validation", protocol["historical_validation_scenes"]), ("round6_validation", protocol["round6_validation_scenes"])):
        results[name] = expanded.summarize_frames([row for row in rows if row["scene"] in scenes], "Previously inspected development subset")
    results["acceptance_gate"] = acceptance_gate(results["development_validation"], protocol)
    return results


def verify_history(history, manifest, protocol, arm, final=False):
    steps = len(history)
    if [row["step"] for row in history] != list(range(1, steps + 1)) or (final and steps != 1600):
        raise ValueError("History is not a contiguous committed update trajectory")
    records = manifest["records"]["train"]
    data = GraspNetFrames(project_path(protocol["data_root"]), records, tuple(protocol["input_size"]), protocol["mode"], protocol["seed"], augment=protocol["augment"])
    updates_per_epoch = len(records) // protocol["accumulation"]
    exposures = {}
    epoch_orders = {}
    for step, row in enumerate(history, 1):
        epoch, batch = (step - 1) // updates_per_epoch, (step - 1) % updates_per_epoch * protocol["accumulation"]
        if epoch not in epoch_orders:
            epoch_orders[epoch] = data.epoch_order(epoch)
        expected = [{"scene": records[i]["scene"], "frame": records[i]["frame"]} for i in epoch_orders[epoch][batch:batch + protocol["accumulation"]]]
        if row["epoch"] != epoch or row["samples"] != expected:
            raise ValueError("History sample order/epoch differs from this arm's fixed dataset")
        if not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm_before_clip"]):
            raise ValueError("Nonfinite committed history")
        if not math.isclose(row["learning_rate_used"], expanded.learning_rate_for_update(step, protocol), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("History learning rate differs from frozen schedule")
        for sample in expected:
            key = (sample["scene"], sample["frame"])
            exposures[key] = exposures.get(key, 0) + 1
    if final:
        expected_ids = {(row["scene"], row["frame"]) for row in records}
        if set(exposures) != expected_ids or set(exposures.values()) != {protocol["arms"][arm]["equivalent_epochs"]}:
            raise ValueError("Full history per-frame exposure count differs")
    return exposures


def verify_evaluation(evaluation, protocol, arm):
    from scripts.accept_segmentation_expanded_v3 import verify_partition
    groups = {"train": protocol["arms"][arm]["train_scenes"], "development_validation": protocol["validation_scenes"],
              "historical_validation": protocol["historical_validation_scenes"], "round6_validation": protocol["round6_validation_scenes"]}
    if not isinstance(evaluation, dict) or set(evaluation) != set(groups) | {"acceptance_gate"}:
        raise ValueError("Final evaluation schema differs")
    for name, scenes in groups.items():
        verify_partition(evaluation[name], scenes, protocol["frames"])
    rows = evaluation["development_validation"]["frames"]
    for name in ("historical_validation", "round6_validation"):
        if evaluation[name]["frames"] != [row for row in rows if row["scene"] in groups[name]]:
            raise ValueError("Final evaluation subgroup differs from development records")
    if evaluation["acceptance_gate"] != acceptance_gate(evaluation["development_validation"], protocol):
        raise ValueError("Final evaluation gate differs from frozen arithmetic")


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
    completed_path = project_path(args.run_dir) / "result.json"
    if args.resume and completed_path.exists() and read_json(completed_path).get("status") == "completed":
        _model, _config, _manifest, result, _evidence = load_final_model(project_path(args.run_dir), protocol)
        print(json.dumps({"status": "verified_completed_read_only", "arm": args.arm}), flush=True)
        return result
    expanded.ensure_disk_reserve(protocol["free_disk_reserve_gib"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model, manifest, data_root, parent_evidence = load_parent(protocol, args.arm)
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
        verify_history(history, manifest, protocol, args.arm, final=complete)
        assessments = final_evaluation(model, data_root, manifest, device, protocol, args.arm) if complete else None
        report = {"status": "completed" if complete else "step_limit_reached", "arm": args.arm,
                  "config_fingerprint": config_hash, "data_fingerprint": data_hash, "parent_evidence": parent_evidence,
                  "optimizer_steps": global_step, "planned_steps": 1600, "parent_total_optimizer_steps": 3200,
                  "arm_training_frames": len(train_data), "equivalent_epochs": protocol["arms"][args.arm]["equivalent_epochs"],
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
    if config.get("schema") != 8 or config["protocol"] != protocol or config["arm"] not in protocol["arms"]:
        raise ValueError("Final repair configuration mismatch")
    args = arguments(["--config", config["protocol_file"], "--arm", config["arm"], "--run-dir", str(directory), "--device", config["device"]])
    args.checkpoint_interval = config["engineering_checkpoint_override"]
    validate_protocol(protocol, args)
    _, current_manifest, _, parent_evidence = load_parent(protocol, config["arm"])
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
    if metadata["global_step"] != 1600 or payload["cursor"] != {"epoch": protocol["arms"][config["arm"]]["equivalent_epochs"], "next_batch": 0, "global_step": 1600} or payload["extra"]["config"] != config:
        raise ValueError("Final checkpoint cursor/config mismatch")
    if result["cursor"] != payload["cursor"] or payload["extra"]["data_fingerprint"] != data_hash:
        raise ValueError("Final result cursor or embedded data identity mismatch")
    if result["arm_training_frames"] != protocol["arms"][config["arm"]]["train_frames"] or result["equivalent_epochs"] != protocol["arms"][config["arm"]]["equivalent_epochs"] or result["parent_total_optimizer_steps"] != 3200:
        raise ValueError("Final arm dataset/exposure/lineage budget differs")
    history = read_json(directory / "history.json")
    if history != payload["extra"]["history"] or [row["step"] for row in history] != list(range(1, 1601)):
        raise ValueError("Final history differs from the complete checkpoint-embedded history")
    verify_history(history, manifest, protocol, config["arm"], final=True)
    initialization = read_json(directory / "initialization.json")
    if payload["extra"]["initialization"] != initialization or initialization["run_config_fingerprint"] != config_hash or initialization["initial_model_tensor_sha256"] != parent_evidence["parent_model_tensor_sha256"]:
        raise ValueError("Final initialization evidence mismatch")
    verify_evaluation(result["evaluation"], protocol, config["arm"])
    expected_decision = "engineering_gate_passed" if result["evaluation"]["acceptance_gate"]["passed"] else "engineering_gate_not_met"
    if result["decision"] != expected_decision:
        raise ValueError("Final decision differs from verified evaluation")
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
        print("Coverage training stopped: {}".format(error), file=sys.stderr, flush=True)
        sys.exit(2)


