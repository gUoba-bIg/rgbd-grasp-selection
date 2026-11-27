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


def archive_episode(slot):
    path = ROOT / "results/simulation_v52/episodes"
    episodes = sorted(path.glob(f"*_slot_{slot:02d}_*"))
    if len(episodes) != 1:
        raise ValueError(f"Original slot {slot} has {len(episodes)} archived attempts")
    return episodes[0]


def select_pairs(settings, repair_hash):
    # Use only the schedule and pre-run intent, never the result/outcome for selection.
    expected = {"021": (4, 6), "014": (16, 18), "017": (32, 34)}
    chosen, ancestry = {}, []
    for oid in expected:
        for nominal in settings["slots"]:
            a = nominal["slot_id"]
            if nominal["object_id"] != oid or nominal["kind"] != "nominal" or a % 4 != 0:
                continue
            b = a + 2
            miss = settings["slots"][b]
            if not (miss["slot_id"] == b and miss["object_id"] == oid and miss["kind"] == "miss"
                    and miss["xy"] == nominal["xy"] and miss["yaw"] == nominal["yaw"]):
                raise ValueError(f"Invalid paired schedule for {oid}: {a}/{b}")
            intents = [archive_episode(i)/"intent.json" for i in (a, b)]
            hashes = [read_json(p)["config"]["sha256"] for p in intents]
            valid = hashes == [repair_hash, repair_hash]
            ancestry.append({"object":oid, "pair":[a,b], "intent_config_hashes":hashes,
                             "intents":[source(p) for p in intents], "both_v52r2":valid})
            if valid:
                chosen[oid] = (a, b)
                break
        if chosen.get(oid) != expected[oid]:
            raise ValueError(f"Historical lineage does not reproduce authorized pair for {oid}")
    return chosen, ancestry


def candidates():
    math = archival_math()
    repair = ROOT / "configs/simulation_v52r2.json"
    settings = read_json(repair)
    selected, ancestry = select_pairs(settings, sha(repair))
    records = []
    for kind in ("nominal","miss"):
        for object_id in ("021","014","017"):
            number = selected[object_id][0 if kind == "nominal" else 1]
            slot = settings["slots"][number]
            episode = archive_episode(number)
            intent_file = episode / "intent.json"
            result_file = episode / "result.json"
            policy_file = episode / "policy_input.json"
            depth_file = episode / "observed_rgbd.npz"
            intent, result, policy = (read_json(p) for p in (intent_file, result_file, policy_file))
            data = policy["candidate_world"]
            config_path = repair
            assert intent["config"]["sha256"] == sha(repair)
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
                raise ValueError("Expected archived v52r2 height adjustment")
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
    assert [r["original_slot"]["slot_id"] for r in records] == [4, 16, 32, 6, 18, 34]
    return records, ancestry


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


def write_once(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


def freeze():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Explicit project root required")
    old = ROOT / "IP140_R1_smoke01"
    if read_json(old/"slot_ledger.json")["starts"] != 0 or read_json(old/"STOP.json")["smoke_starts"] != 0:
        raise ValueError("Prior budget state differs")
    xml_name = "ur5e_ip140_r1.urdf"
    model = HERE/"model"
    model.mkdir(exist_ok=True)
    for name, expected in [(xml_name,"368a6e7d830b2e3fad85c335fab010728c3e8a058ce3988ca1ff931693915c8b"),
                           ("manifest.json","f1b63142ab2639863e3292468be188d17baa11559c8ba19846c20bce4bef873d")]:
        src, target = old/"model"/name, model/name
        if sha(src) != expected:
            raise ValueError(f"Model changed: {name}")
        if not target.exists():
            with target.open("xb") as f:
                f.write(src.read_bytes())
        if sha(target) != expected:
            raise ValueError(f"Model copy differs: {name}")
    assets = read_json(model/"manifest.json")["reused_arm_asset_sha256"]
    for path, expected in assets.items():
        if sha(ROOT/path) != expected:
            raise ValueError(f"Referenced mesh or material changed: {path}")
    selected, ancestry = candidates()
    rows = benches() + selected
    baseline = read_json(old/"preregistration.json")
    assert rows[:6] == baseline["slots"][:6]
    assert [r["slot"] for r in rows] == list(range(1,13))
    assert len({r["candidate_id"] for r in rows}) == 12
    for i in (7,8,10,11):
        assert rows[i] == baseline["slots"][i]
    stamp = datetime.now(timezone.utc).isoformat()
    changes = [dict(object_id="021", kind=kind, old_slot=a, new_slot=b)
               for kind,a,b in (("nominal",0,4),("miss",2,6))]
    meta = dict(version=HERE.name, created_at_utc=stamp, prior_zero="Windows path lookup failure, not physics/model failure",
                unchanged_model_identity=[source(model/xml_name),source(model/"manifest.json")],
                prior_seal=[source(old/name) for name in ("STOP.json","slot_ledger.json","preregistration.json","freeze_slots.py")],
                model_manifest_note="Manifest retains the byte-identical smoke01 version string intentionally; this sidecar identifies the new execution version.")
    write_once(HERE/"model_source_version.json",meta)
    data = dict(version=HERE.name, frozen_at_utc=stamp, max_starts=12, retry=False,
                prior_physical_starts=0, eligible_slots_after_zero_gate=[1,2],
                start_requires="all readable hard gates pass; only approved deferred and first-step watchdog sets",
                stop="first hard gate mismatch or first failed/unknown start; unconditional pause after slot 2",
                correction="Pre-first-start historical-lineage correction, no budget expansion; no IP140 outcome used",
                historical_seed_rule="First nominal (block+0) and miss (block+2) with identical XY/yaw/object in a 4-slot historical block, both intent.config.sha256 equal v52r2. Selected using schedule/intent before opening any results.",
                changed_slots=changes, selection_evidence=ancestry,
                references=[source(p) for p in (Path(__file__).resolve(), ROOT/"scripts/sim_v52_math.py", HERE/"model_source_version.json",
                             ROOT/"IP140-R1_执行端替换两页协议_小修02.pdf",ROOT/"IP140-R1_参数与证据附表_小修02.pdf",old/"preregistration.json")], slots=rows)
    for key in ("no_score_export_or_comparison","policy","bench_scope","bench_contacts","ids"):
        data[key] = baseline[key]
    dest = HERE/"preregistration.json"
    write_once(dest,data)
    manifest_hash = sha(dest)
    write_once(HERE/"ledger_initial.json",dict(schema="IP140.smoke02.ledger.initial", at_utc=stamp,
               version=HERE.name, total_slots=12, starts=0, returned_episodes=0, unknown_started_episodes=0,
               retries=0, physical_steps=0, existing_budget=12, budget_added=0, preregistration_sha256=manifest_hash,
               slots=[dict(slot=r["slot"],candidate_id=r["candidate_id"],kind=r["kind"],charged=False,
                           state="waiting_zero_gate" if r["slot"]<3 else "withheld_pending_review",
                           result=None, telemetry=None) for r in rows]))
    print(manifest_hash)


if __name__ == "__main__":
    freeze()
