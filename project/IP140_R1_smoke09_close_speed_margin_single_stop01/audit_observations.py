"""Read the closed API stream and telemetry without importing the physics engine."""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
PHASES = (("settle", 240), ("opening", 480), ("transit", 672), ("pregrasp", 672),
          ("approach", 672), ("close", 480))
SCHEDULE = [name for name, count in PHASES for _ in range(count)]


def read(path):
    return json.loads((BASE / path).read_bytes())


def digest(path):
    with (BASE / path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with gzip.open(BASE / path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def audit():
    gate = read("zero_readback.json")
    report = {"version": BASE.name, "zero_passed": gate["passed"],
              "zero_stepSimulation": gate["physics_step_calls"],
              "zero_mismatches": gate["required_mismatches"]}
    session = read("session_result.json")
    charged = (BASE / "episodes/13/precharge.json").exists()
    robot = gate.get("body_id")
    fixture = gate.get("fixture_body_id")
    right = left = None
    before = BASE / "zero_step/filter_pre.json"
    if before.exists():
        pairs = read("zero_step/filter_pre.json")["pairs"]
        right = pairs["ip140_right_bone|ip140_right_pad"]["link_ids"][-1]
        left = pairs["ip140_left_bone|ip140_left_pad"]["link_ids"][-1]
    calls, returns = Counter(), Counter()
    filters, outcomes, pending = [], [], None
    observations = {}
    total = 0
    for row in rows("full_api/calls.jsonl.gz"):
        seq, kind = row[:2]
        if kind == "call":
            assert pending is None and seq == total + 1
            total = seq
            pending = row
            calls[row[3]] += 1
            continue
        assert pending is not None and pending[0] == seq
        method, args, kwargs = pending[3:6]
        pending = None
        if kind != "return":
            outcomes.append([seq, method, kind])
            continue
        value = row[3]
        returns[method] += 1
        if method.startswith("setCollisionFilter"):
            filters.append({"seq": seq, "method": method, "args": args, "kwargs": kwargs, "return": value})
        if method == "getContactPoints" and returns["stepSimulation"] and not args and set(kwargs) <= {"physicsClientId"}:
            step = returns["stepSimulation"]
            if step in observations:
                assert value == observations[step]
            observations[step] = value
    assert pending is None
    api = read("full_api/summary.json")
    assert dict(calls) == api["calls"] and dict(returns) == api["returns"]
    for h, size in api["lossless_raw_blobs_bytes"].items():
        with gzip.open(BASE / "full_api" / (h + ".bin.gz"), "rb") as stream:
            blob = stream.read()
        assert len(blob) == size and hashlib.sha256(blob).hexdigest() == h
    report.update(api_calls=total, api_exceptions=outcomes, physics_calls=calls["stepSimulation"],
                  physics_returns=returns["stepSimulation"], filter_calls=filters,
                  connection_returns=returns["connect"], disconnect_returns=returns["disconnect"],
                  robot_loads=returns["loadURDF"], fixture_creations=returns["createMultiBody"],
                  fixture_id=fixture, robot_id=robot, right_pad=right,left_pad=left, charged=charged)
    if gate["passed"]:
        assert len(filters) == 13 and gate["structural_filter_setter_count"] == 13
        for f,key in zip(filters,read("preregistration.json")["approved_filter_order"],strict=True):
            assert f["method"] == "setCollisionFilterPair" and f["args"] == [robot,robot,*pairs[key]["link_ids"],0]
            assert f["return"] is None and fixture not in f["args"][:2]
    if not charged:
        assert calls["stepSimulation"] == 0
        report["slot_result"] = None
        return report

    indices = {"step_intent": [], "step_return": [], "state": []}
    states_by_phase = Counter()
    telem = "episodes/13/telemetry.jsonl.gz"
    for row in rows(telem):
        event = row["event"]
        if event in indices:
            index = row["index"]
            indices[event].append(index)
            if event != "step_return":
                assert SCHEDULE[index-1] == row["phase"]
            if event == "state":
                states_by_phase[row["phase"]] += 1
                if "raw_manifolds" in row: assert row["raw_manifolds"] == observations[index]
    for series in indices.values():
        assert series == list(range(1, len(series)+1))
    assert len(indices["step_intent"]) == calls["stepSimulation"]
    assert len(indices["step_return"]) == returns["stepSimulation"]
    assert set(observations) == set(indices["state"])

    streak = longest = 0
    valid_count = total_external = near_force_count = 0
    events = []
    for step, contacts in observations.items():
        phase = SCHEDULE[step-1]
        valid_points, other = [], []
        for j,c in enumerate(contacts):
            if c[1]==c[2]==robot:
                if set(c[3:5])!={left,right}:other.append(j)
                continue
            total_external += 1
            if (c[1:5]==[robot,fixture,right,-1] or c[1:5]==[fixture,robot,-1,right]) and phase=="close":
                vals = [*c[:5], *c[5], *c[6], *c[7], *c[8:11], *c[11], c[12], *c[13]]
                if not all(math.isfinite(v) for v in vals) or c[8]<-.001:
                    other.append(j)
                elif c[9]>.2:
                    valid_points.append(j)
                else:
                    near_force_count += 1
            else:
                other.append(j)
        if contacts:
            events.append(dict(step=step,phase=phase,raw_points=contacts,
                               qualifying_right_indices=valid_points,unexpected_indices=other))
        good=bool(valid_points) and not other
        streak=streak+1 if good else 0
        longest=max(streak,longest)
        valid_count+=int(good)
    res = read("episodes/13/result.json")
    assert res["telemetry_sha256"] == digest(telem)
    assert res["step_intents"]==calls["stepSimulation"] and res["step_returns"]==returns["stepSimulation"]
    assert session["slots"]==[13] and not session["retry"] and res["grasp_success"] is False
    if res["slot_passed"]:
        assert all(len(v)==3216 for v in indices.values()) and streak>=24
        assert not any(e["unexpected_indices"] for e in events) and res["expected_negative_detected"] is True
    report.update(slot_result=res, states_per_phase=dict(states_by_phase),
                  observed_raw_world_manifold_steps=len(events),total_external_points=total_external,
                  raw_world_contact_records=events, subthreshold_registered_right_points=near_force_count,
                  qualifying_right_contact_steps=valid_count, longest_continuous_right_steps=longest,
                  terminal_continuous_right_steps=streak,all_3216_returned=returns["stepSimulation"]==3216,
                  terminal_24_raw_contact_requirement_met=returns["stepSimulation"]==3216 and streak>=24,
                  unknown_steps=calls["stepSimulation"]-returns["stepSimulation"])
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False, allow_nan=False))

