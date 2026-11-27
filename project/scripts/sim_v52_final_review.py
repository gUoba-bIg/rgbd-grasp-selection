"""Independent, read-only final R3 review. Run only after all 48 attempts return.

No simulator, model, training, scorer, report generator, or recovery entry is called.
The sole output is a new review artifact; existing scientific results stay untouched.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/simulation_v52"
REGISTRY = {
    "52": ("configs/simulation_v52.json", "logs/simulation_v52/freeze.json",
           "scripts/sim_v52_control.py", "scripts/sim_v52_episode.py"),
    "52r1": ("configs/simulation_v52r1.json", "logs/simulation_v52/recovery_v52r1/freeze.json",
             "scripts/sim_v52r1_control.py", "scripts/sim_v52r1_episode.py"),
    "52r2": ("configs/simulation_v52r2.json", "logs/simulation_v52/recovery_v52r2/freeze.json",
             "scripts/sim_v52r2_control.py", "scripts/sim_v52r2_episode.py"),
}


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    stat = path.stat()
    return {"sha256": h.hexdigest(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def checked_path(name):
    path = (ROOT / name).resolve()
    path.relative_to(ROOT.resolve())
    return path


def now():
    return dt.datetime.now().astimezone().isoformat()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected):
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def review():
    require(Path.cwd().resolve() == ROOT.resolve(), "Explicit project root working directory required")
    out = LOG / "final_review.json"
    require(not out.exists(), "Do not overwrite an existing final review")
    sources = {}
    verified = {}

    def load(name):
        path = checked_path(name)
        before = identity(path)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        require(identity(path) == before, "Input changed while reading: " + name)
        sources[name] = before
        return value

    def bind(name, expected):
        path = checked_path(name)
        if name in verified:
            require(verified[name] == expected, "Inconsistent frozen identity: " + name)
        else:
            require(path.is_file() and identity(path) == expected, "Frozen artifact changed: " + name)
            verified[name] = expected

    configs = {}
    for version, (config_name, freeze_name, control_name, episode_name) in REGISTRY.items():
        config = load(config_name)
        receipt = load(freeze_name)
        require(sources[config_name] == receipt["config"], "Config differs from freeze: " + version)
        require(config["R4_authorized"] is False and config["automatic_expansion"] is False,
                "Unapproved expansion flag: " + version)
        for name, expected in config["bindings"].items():
            bind(name, expected)
        require(control_name in config["bindings"] and episode_name in config["bindings"],
                "Execution sources absent from freeze: " + version)
        configs[version] = config
    cfg = configs["52r2"]
    for key in ("slots", "criteria", "physics", "limits", "motion", "negative_controls", "objects", "home_joints"):
        require(all(config[key] == configs["52"][key] for config in configs.values()),
                "Original protocol field changed: " + key)
    require(cfg["limits"]["attempts"] == 48 and len(cfg["slots"]) == 48, "Wrong cumulative budget")
    require(collections.Counter(s["kind"] for s in cfg["slots"]) == {"nominal": 24, "miss": 12, "table_collision": 12},
            "Wrong fixed case allocation")

    historical = load("logs/simulation_v52/historical_snapshot.json")["protected"]
    for name, expected in historical.items():
        bind(name, expected)

    ledger_name = "logs/simulation_v52/episodes.jsonl"
    ledger_path = checked_path(ledger_name)
    sources[ledger_name] = identity(ledger_path)
    events = [json.loads(s) for s in ledger_path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    require(identity(ledger_path) == sources[ledger_name], "Ledger changed during review")
    require(all(e["event"] in ("intent", "returned") for e in events), "Unexpected ledger event")
    intents = [e for e in events if e["event"] == "intent"]
    returned = [e for e in events if e["event"] == "returned"]
    require(len(intents) == 48 and len(returned) == 48, "Wait for exactly 48 intents and 48 returns")
    require(len({e["attempt_id"] for e in intents}) == 48, "Duplicate attempt intent")
    require(len({e["attempt_id"] for e in returned}) == 48, "Duplicate return")
    require({e["attempt_id"] for e in intents} == {e["attempt_id"] for e in returned}, "Missing/orphan return")
    require([e["slot_id"] for e in intents] == list(range(48)), "Repeated, missing or reordered slot")
    returns = {e["attempt_id"]: e for e in returned}
    rows = []
    resources = []
    episode_versions = collections.Counter()
    success_checks = []
    for intent in intents:
        slot = intent["slot_id"]
        version = "52" if slot == 0 else "52r1" if slot == 1 else "52r2"
        config_name, _, _, episode_name = REGISTRY[version]
        config = configs[version]
        ret = returns[intent["attempt_id"]]
        require(ret["slot_id"] == slot and ret.get("exit_code") == 0 and ret.get("termination") is None,
                "Terminated, nonzero or mismatched return for slot " + str(slot))
        require(ret.get("result_status") == "known_result" and ret.get("result") is not None,
                "Unknown result for slot " + str(slot))
        require(intent["config"] == sources[config_name], "Wrong version identity for slot " + str(slot))
        require(episode_name in intent["command"], "Wrong worker command for slot " + str(slot))
        require(load(intent["directory"] + "/intent.json") == intent, "Intent copy mismatch")
        require(load(intent["directory"] + "/resources.json") == ret, "Resource receipt mismatch")
        result_name = intent["directory"] + "/result.json"
        result = load(result_name)
        require(sources[result_name] == ret["result"], "Result receipt changed for slot " + str(slot))
        require(result["slot"] == cfg["slots"][slot], "Result fixed slot mismatch")
        require(result["config_identity"] == intent["config"], "Result config mismatch")
        require(result["source_identity"] == config["bindings"][episode_name], "Result source mismatch")
        require(result.get("actual_neural_forwards") == 0 and result.get("new_object_attachment_constraints") == 0
                and result.get("object_teleports_after_initialization") == 0, "Forbidden operation declared")
        collisions = load(intent["directory"] + "/collision_events.json")
        trace = load(intent["directory"] + "/trace.json")
        require(result["collision_events_count"] == len(collisions), "Collision event count mismatch")
        require(bool(collisions) == result["forbidden_collision"], "Collision flag mismatch")
        require(isinstance(trace, list), "Invalid trace")
        require(close(result["simulation_seconds"], result["physics_steps"] * cfg["physics"]["time_step"]),
                "Physics step accounting mismatch")
        require(result["simulation_seconds"] <= cfg["limits"]["episode_sim_seconds"] + cfg["physics"]["time_step"],
                "Episode exceeds fixed simulation budget")
        require(ret["peak_tree_rss_bytes"] <= cfg["limits"]["tree_rss_bytes"], "Memory budget exceeded")
        criteria = cfg["criteria"]
        coupling_ok = result.get("max_mimic_error_rad", math.inf) <= criteria["mimic_error_rad"]
        if "dynamic_coupling_passed" in result:
            require(result["dynamic_coupling_passed"] == coupling_ok, "Coupling gate mismatch")
        necessary = (result.get("final_continuous_hold_seconds", 0) >= criteria["hold_seconds"]
                     and result.get("final_bilateral_contact", False) and not result["slipped"]
                     and not result["forbidden_collision"] and result.get("grasp_formed", False)
                     and coupling_ok and result.get("dynamic_coupling_passed", False))
        if result["success"]:
            require(necessary and result["outcome"] == "success", "Success bypasses frozen conditions")
        if result["outcome"] == "success":
            require(result["success"], "Success outcome inconsistent")
        if not coupling_ok:
            require(not result["success"], "Lift/hold bypassed failed coupling")
        if result["outcome"] == "dynamic_coupling_failed":
            require(not coupling_ok and not result["success"], "Invalid coupling failure classification")
        success_checks.append({"slot_id": slot, "version": version, "success": result["success"],
            "outcome": result["outcome"], "max_lift_m": result.get("max_lift_m", 0),
            "final_hold_seconds": result.get("final_continuous_hold_seconds", 0),
            "max_hold_seconds": result.get("max_continuous_hold_seconds", 0),
            "mimic_error_rad": result.get("max_mimic_error_rad"), "forbidden_collision": result["forbidden_collision"]})
        rows.append(result)
        resources.append(ret)
        episode_versions[version] += 1
    require(rows[0]["outcome"] == "planned_table_collision" and rows[1]["outcome"] == "dynamic_coupling_failed",
            "Historical failures were replaced")
    require(episode_versions == {"52": 1, "52r1": 1, "52r2": 46}, "Wrong revision coverage")

    nominal = [r for r in rows if r["slot"]["kind"] == "nominal"]
    miss = [r for r in rows if r["slot"]["kind"] == "miss"]
    table = [r for r in rows if r["slot"]["kind"] == "table_collision"]
    positive = sum(r["success"] for r in nominal)
    per_object = {o: sum(r["success"] for r in nominal if r["slot"]["object_id"] == o) for o in cfg["objects"]}
    miss_correct = sum(r["outcome"] == "grasp_or_lift_failed" and r.get("dynamic_coupling_passed", False)
                       and not r["forbidden_collision"] for r in miss)
    table_correct = sum(r["outcome"] in ("planned_table_collision", "target_contact_below_known_table") for r in table)
    integrity = not any(r["outcome"] in ("engineering_exception", "dynamic_coupling_failed") for r in rows)
    passed = (integrity and positive >= cfg["criteria"]["positive_min"]
              and min(per_object.values()) >= cfg["criteria"]["per_object_positive_min"]
              and miss_correct == 12 and table_correct == 12)
    formed = sum(r["grasp_formed"] for r in rows)
    slipped = sum(r["slipped"] for r in rows)
    expected = {
        "attempts_started": 48, "attempts_with_results": 48, "attempts_cap": 48,
        "unknown_episode_results": [], "planned_slots": 48, "complete": True,
        "execution_integrity_passed": integrity, "engineering_passed": passed,
        "R4_started": False, "review_required": True, "nominal_successes": positive,
        "nominal_attempts": 24, "positive_successes_per_object": per_object,
        "miss_controls_correct": miss_correct, "miss_controls_attempted": 12,
        "table_controls_correct": table_correct, "table_controls_attempted": 12,
        "all_attempt_success_rate": sum(r["success"] for r in rows) / 48,
        "forbidden_collision_rate": sum(r["forbidden_collision"] for r in rows) / 48,
        "infeasible_rate": sum(r.get("infeasible", False) for r in rows) / 48,
        "slip_count": slipped, "grasp_formed_count": formed, "slip_rate_all": slipped / 48,
        "slip_rate_formed": slipped / formed if formed else None,
        "total_worker_wall_seconds": sum(r["wall_seconds"] for r in resources),
        "total_worker_cpu_seconds": sum(r["cpu_seconds"] for r in resources),
        "peak_tree_rss_bytes": max(r["peak_tree_rss_bytes"] for r in resources),
        "actual_new_neural_forwards": 0, "actual_new_training": 0,
        "old_P5_P6_scientific_passed": False, "historical_unchanged": len(historical),
    }
    require(expected["total_worker_wall_seconds"] <= cfg["limits"]["cumulative_wall_seconds"], "Cumulative wall budget exceeded")
    state = load("runtime/pipeline/simulation_v52.json")
    require(state["phase"] == "completed_stopped_for_review" and state["R3_complete"] is True,
            "Worker must finish and stop before final review")
    require(state["R4_authorized"] is False and state["automatic_expansion"] is False,
            "R4/expansion must remain disabled")
    require(state["engineering_attempts_started"] == 48 and state["engineering_passed"] == passed, "State inconsistent")
    summary = load(state["report"])
    for key, value in expected.items():
        require(close(summary.get(key), value), "Final summary differs: " + key)
    require(len(summary["rows"]) == 48, "Summary omitted failures")
    for actual, intent, result in zip(summary["rows"], intents, rows):
        require(actual["attempt_id"] == intent["attempt_id"] and actual["slot_id"] == intent["slot_id"]
                and actual["directory"] == intent["directory"] and actual["result"] == result,
                "Summary result row differs from preserved evidence")
    # There is no R4 implementation in the admitted controller. Do not mistake this
    # bounded ledger/source review for monitoring unrelated processes on the host.
    require(not any("R4" in " ".join(i["command"]) for i in intents), "R4 command in engineering ledger")
    for name, expected_id in {**verified, **sources}.items():
        require(identity(checked_path(name)) == expected_id, "Artifact changed during final review: " + name)
    zero_success = not any(r["success"] for r in rows)
    review_data = {
        "at": now(), "reviewer": "independent_final_review", "review_script": identity(__file__),
        "evidence_review_passed": True, "engineering_admission_passed": passed,
        "all_episode_protocol_successes_false": zero_success,
        "conclusion": ("All 48 attempts are fully accounted for, with zero protocol successes; successful lifting or holding cannot bypass failed mimic/collision gates."
                       if zero_success else "All 48 attempts and their protocol outcomes are independently accounted for; retain all historical failures in the aggregate decision."),
        "actual_attempts": 48, "actual_returned_results": 48, "unknown_attempts": [],
        "revision_counts": dict(episode_versions), "outcome_counts": dict(collections.Counter(r["outcome"] for r in rows)),
        "recomputed_summary": expected, "reported_summary": state["report"],
        "per_episode_protocol_checks": success_checks,
        "R4_started": False, "R4_evidence_scope": "Frozen controller flags, stopped state, all recorded worker commands; root must separately confirm no live project process",
        "historical_protected_files_verified": len(historical), "frozen_and_result_files_verified": len(verified) + len(sources),
        "artifact_identities": sources,
        "unchanged_scientific_boundary": "Old P5/P6 failure conclusions remain unchanged; engineering accounting review is not algorithm efficacy or permission for R4.",
        "reviewer_physics_steps": 0, "reviewer_model_forwards": 0, "reviewer_official_scoring_calls": 0,
        "source_result_modifications": 0,
    }
    with out.open("x", encoding="utf-8") as stream:
        json.dump(review_data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"review": str(out), "evidence_review_passed": True,
                      "engineering_admission_passed": passed, "zero_protocol_successes": zero_success}))


if __name__ == "__main__":
    review()
