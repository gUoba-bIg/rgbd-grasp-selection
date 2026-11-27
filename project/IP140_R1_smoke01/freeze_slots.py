"""Materialize an immutable smoke manifest by reading existing observations. No physics API."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(path):
    return path.read_text(encoding="utf-8-sig")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    return {"project_relative": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path), "bytes": path.stat().st_size}


def read_json(path):
    return json.loads(load(path))


def archival_math():
    spec = importlib.util.spec_from_file_location("archive_geometry", ROOT / "scripts/sim_v52_math.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates():
    math = archival_math()
    original = ROOT / "configs/simulation_v52.json"
    repair = ROOT / "configs/simulation_v52r2.json"
    configs = {sha(f): f for f in (original, repair)}
    settings = read_json(repair)
    records = []
    for kind in ("nominal", "miss"):
        for object_id in ("021", "014", "017"):
            # Order in the archived manifest, without reading any outcome for selection.
            slot = next(s for s in settings["slots"] if s["object_id"] == object_id and s["kind"] == kind)
            number = slot["slot_id"]
            episodes = list((ROOT / "results/simulation_v52/episodes").glob(f"*_slot_{number:02d}_*"))
            if len(episodes) != 1:
                raise ValueError(f"Expected one original attempt for archived slot {number}")
            episode = episodes[0]
            intent_file = episode / "intent.json"
            result_file = episode / "result.json"
            policy_file = episode / "policy_input.json"
            depth_file = episode / "observed_rgbd.npz"
            intent, result, policy = (read_json(p) for p in (intent_file, result_file, policy_file))
            data = policy["candidate_world"]
            config_path = configs[intent["config"]["sha256"]]
            cfg = read_json(config_path)
            assert result["slot"] == slot == cfg["slots"][number] and intent["slot_id"] == number
            assert result["config_identity"]["sha256"] == sha(config_path)
            assert result["candidate_world"] == data and len(data) == 17 and np.isfinite(data).all()
            exe_source = ROOT / intent["command"][2]
            assert sha(exe_source) == result["source_identity"]["sha256"]
            assert sha(ROOT / "scripts/sim_v52_math.py") == cfg["bindings"]["scripts/sim_v52_math.py"]["sha256"]
            with np.load(depth_file, allow_pickle=False) as archived:
                points = math.unproject(archived["depth_buffer"], archived["view"], archived["projection"])
            nominal, obs = math.observed_proposal(points, slot["yaw"], cfg["table_z"], cfg["workspace"])
            assert obs["point_count"] == policy["observation"]["point_count"]
            assert abs(obs["estimated_object_width_m"] - policy["observation"]["estimated_object_width_m"]) < 1e-12
            correction = result.get("engineering_proposal_height_adjustment_m")
            if correction is not None:
                nominal[15] += correction  # Preserve the signed, logged legacy height adjustment.
            else:
                assert config_path == original
            if kind == "miss":
                nominal[13] += cfg["negative_controls"]["miss_x_m"]
            max_diff = float(np.max(np.abs(nominal - data)))
            if max_diff > 1e-12:
                raise ValueError(f"Stored observation and original policy disagree: {episode.name} / {max_diff}")
            R = np.array(data[4:13]).reshape(3, 3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array(data[13:16]) + data[3] * R[:, 0]
            asset = cfg["objects"][object_id]
            row = {"slot": len(records) + 7, "candidate_id": f"v52eng:slot{number:02d}:{intent['attempt_id']}",
                   "kind": kind, "object_id": object_id, "candidate_world": data, "T_W_G_m": T.tolist(),
                   "pre_open_m": data[1], "original_slot": slot, "original_attempt_id": intent["attempt_id"],
                   "not_p3": True, "height_adjustment_m_as_recorded": correction,
                   "height_adjustment_source": "legacy result" if correction is not None else "no adjustment field in v52 original",
                   "check_stored_candidate_max_abs_error": max_diff,
                   "source_config": source(config_path),
                   "source_files": [source(p) for p in (intent_file, policy_file, result_file, depth_file, exe_source)],
                   "scene_setup_only": {"urdf": source(ROOT / asset["urdf"]),
                                        "position_world_m": [*slot["xy"], cfg["table_z"] + asset["height_m"] / 2 + .003],
                                        "quaternion_xyzw": [0, 0, 0, 1], "lateral_friction": .6, "restitution": 0,
                                        "linear_damping": .04, "angular_damping": .04, "fixed_base": False},
                   "accept": "success by frozen lift/hold/collision criteria" if kind == "nominal"
                             else "motion executed; valid miss with no successful grasp; exceptions/other failure are not a correct negative"}
            records.append(row)
    assert [r["original_slot"]["slot_id"] for r in records] == [0, 16, 32, 2, 18, 34]
    return records


def benches():
    rows = []
    for i in (1, 2):
        rows.append({"slot": i, "candidate_id": f"IP140-BENCH-{i:02d}:empty-stroke",
                     "candidate_world": None, "kind": "empty_stroke", "fixture": None,
                     "arm_pose": "historical home joints, held throughout",
                     "opening_sequence_mm": [0,35,70,105,140,105,70,35,0],
                     "seconds_each": 2.0, "tail_s": .100, "q_initial_m": [0,0],
                     "accept": "every sample of every settled tail meets frozen gap, midline, single-finger, symmetry and parallelism gates; no unexpected contact/impulse or penetration"})
    T = np.array([[0.,0.,1.,.50],[0.,1.,0.,-.06],[-1.,0.,0.,.25],[0.,0.,0.,1.]])
    rotation = T[:3,:3]
    grasp = [1.,.100,.020,.020,*rotation.flatten().tolist(),*(T[:3,3] - .020*rotation[:,0]).tolist(),-1.]
    configs = [
        ("center", 0., "both pads in sustained independent external contact in final 100ms"),
        ("offset_positive_y", .010, "right pad first, strictly earlier step than left; both sustained during final 100ms"),
        ("offset_negative_y", -.010, "left pad first, strictly earlier step than right; both sustained during final 100ms"),
        ("single_sided_stop", .030, "right pad only; classifier reports a forbidden pad/non-target contact, left pad no active external contact, no other forbidden event")]
    for i,(kind,y_m,expect) in enumerate(configs, start=3):
        size = [.020, .010 if i==6 else .040, .018]
        box_center = [-.030, y_m, 0.]
        block = {"full_sides_xyz_in_G_m": size, "center_in_goal_G_m": box_center,
                 "center_world_m": (T[:3,3]+rotation@box_center).tolist(), "R_W_fixture": rotation.tolist(),
                 "fixed_to_world": True, "base_mass_kg": 0.0, "lateral_friction": .6,
                 "restitution": 0., "rolling_friction": 0., "spinning_friction": 0., "margin_m": .0005,
                 "evaluation_class": "non_target_stop" if i==6 else "target_test_block"}
        rows.append({"slot": i,"kind":kind, "candidate_id": f"IP140-BENCH-{i:02d}:{kind}",
                     "candidate_world": grasp, "T_W_G_m": T.tolist(), "pre_open_m": .100, "fixture": block,
                     "arm_pose": "from home; common pre-opening and approach; hold goal through close; end after 2s close",
                     "accept": expect, "unexpected_fixture_collisions_fail": True})
    return rows


def freeze():
    dest = HERE / "preregistration.json"
    rows = benches() + candidates()
    assert [r["slot"] for r in rows] == list(range(1, 13))
    assert len({r["candidate_id"] for r in rows}) == 12
    data = {"version": "IP140-R1-smoke01", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "max_starts": 12, "retry": False, "start_requires": "complete all actual zero-step readbacks",
            "no_score_export_or_comparison": True,
            "policy": "manifest world candidate poses and held encoder references only; no object truth for any action, fixture identity only for post hoc evaluation",
            "bench_scope": "bench 3-6 fixed rigid fixtures initialized in scene before any motion, never attached to gripper or moved during execution; no lifting credited",
            "bench_contacts": "per-pad earliest positive contact means one external point normal force >0.2 N. 100ms means consecutive recorded actual substeps. Slot6 expected classification is not a whitelist exception.",
            "stop": "initial endpoint or seam false active contact, penetration or spurious impulse: stop immediately and preserve; bench failure blocks slots 7-12; first object positive failure, false-success negative or unexpected/unknown event stops subsequent starts",
            "pair_021_caveat": "Archived nominal slot00 used v52 without later proposal height correction. Its paired miss slot02 used v52r2. Retained exact old poses, including logged vertical discrepancy; do not reconstruct a pure-X pair.",
            "ids": "bench IDs are new preregistered protocols (slots1-2 no 6D candidate); v52eng IDs name existing observation-based archived candidates, not historical network IDs",
            "references": [source(p) for p in (Path(__file__).resolve(), ROOT/"scripts/sim_v52_math.py",
                                ROOT/"IP140-R1_执行端替换两页协议_小修02.pdf", ROOT/"IP140-R1_参数与证据附表_小修02.pdf")],
            "slots": rows}
    with dest.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
    print(f"Static preregistration: {len(rows)} slots; immutable SHA256 {sha(dest)}")


if __name__ == "__main__":
    freeze()
