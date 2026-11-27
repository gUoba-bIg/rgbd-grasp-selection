"""Rebuild the actual input tensor identities from saved, pristine V1 observations.

The native forward already returned exactly once. This verification has no model or physics calls.
"""
from common_v1 import *
import sys

P = C / "p"
sys.path[:0] = [str(P / R4 / "implementation"), str(P)]
import numpy as np
from policy_network import prepare
from src.risk_observations_v31 import points

def fingerprint(array):
    a = np.ascontiguousarray(array)
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape), "dtype": a.dtype.str}

def for_unit(spec, original, rgb_camera):
    uid = spec["unit_id"]
    folder = OUT / "study/units" / uid
    proof = read(folder / "network.json")
    entry = read(folder / "forward_intent.json")
    need(proof["observation_sha256"] == spec["observation_sha256"] == entry["source"], "fingerprint observation identity " + uid)
    need(proof["network_output_sha256"] == sha(folder / "network.npz"), "fresh evidence identity " + uid)
    need(entry["point_sampling_seed"] == spec["sampling_seed"], "P3 sampling seed " + uid)
    need(proof["networks_returned"] == {"P3": 1, "P5_fixed": 1} and proof["physics_steps"]==0, "model receipt " + uid)
    x,k,size,tf = rgb_camera
    with np.load(folder / "network.npz", allow_pickle=False) as n:
        need(np.array_equal(n["K"], k) and np.array_equal(n["original_size"], size) and np.array_equal(n["world_T_camera"],tf), "input camera " + uid)
        depth,valid = n["depth"],n["valid"]
        x=x.copy()
        x[3]=depth/2.
        x[4]=valid.astype(np.float32)
    cloud,_ = points(depth,valid,k,size)
    need(len(cloud)==proof["visible_cloud_count"] and np.isfinite(x).all(), "input shape or values " + uid)
    rng=np.random.RandomState(spec["sampling_seed"])
    idx=rng.choice(len(cloud),20000,replace=False)
    tensors={"P5":fingerprint(x[None]),"P3":fingerprint(cloud[idx][None])}
    anchor=spec["condition"] in ("clean","joint")
    if anchor:
        need(tensors==original[uid]["model_input_fingerprints"],"original forward tensor divergence "+uid)
    else:
        need(tensors["P5"]["sha256"]==proof["corruption"]["input_sha256"], "inference corruption receipt "+uid)
    return dict(model_input_fingerprints=tensors,sampling_seed=spec["sampling_seed"],sampled_index_sha256=fingerprint(idx)["sha256"],
                original_tensor_check=anchor,native_transform_tensor_check=not anchor,network_receipt_sha256=sha(folder/"network.json"),
                network_output_sha256=proof["network_output_sha256"],observation_sha256=spec["observation_sha256"])

def build(jobs):
    original=read(ROOT/R6FIRST/"preregistration/actual_unit_inputs.json")["units"]
    records={}
    last=None; cached=None
    for j in jobs:
        if j["scene"]!=last:
            source=ROOT/j["observation_path"]
            need(sha(source)==j["observation_sha256"],"frozen source RGB-D")
            with np.load(source,allow_pickle=False) as obs:
                need(set(obs.files)=={"rgb","depth_buffer","view","projection"},"unexpected observation fields")
                cached=prepare(obs["view"],obs["projection"],obs["rgb"],obs["depth_buffer"])
            last=j["scene"]
        need(j["unit_id"] not in records,"duplicate observation")
        records[j["unit_id"]]=for_unit(j,original,cached)
    need(len(records)==720 and sum(r["original_tensor_check"] for r in records.values())==240,"input identity coverage")
    rel="preregistration/observation_fingerprint_index.json"
    h=write(rel,dict(units=records,utc=utc(),verification_mode="deterministic_reconstruction_from_each_fresh_observation_and_forward_return",
                     model_calls_in_verification=0,physics_calls_in_verification=0,original_tensor_pairs_verified=240,
                     actual_single_factor_P5_transform_receipts_verified=480,source_sha256=sha(Path(__file__))))
    return records,h
