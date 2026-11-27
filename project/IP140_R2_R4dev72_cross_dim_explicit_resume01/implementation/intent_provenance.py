"""Provenance fields added to R4 method intent; no methods are executed here."""
from pathlib import Path
import hashlib
import json
import numpy as np

V=Path(__file__).resolve().parents[1];ROOT=V.parent
def sh(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def rd(p):return json.loads(p.read_bytes())
def once(path,v):
    with path.open("x",encoding="utf8") as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)

def dependencies():
    reg=V/"audit/input_files.json"
    protected=rd(reg)["files"]
    files={name:m["sha256"] for name,m in protected.items()}
    for folder in ("implementation","preregistration"):
        for p in (V/folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:files[p.relative_to(ROOT).as_posix()]=sh(p)
    for k,digest in files.items():
        if sh(ROOT/k)!=digest:raise RuntimeError("project input identity conflict "+k)
    catalog=V/"audit/intent_dependencies.json"
    once(catalog,dict(version=V.name,origin_registry_sha256=sh(reg),project_relative_sha256=files))
    return sh(catalog)

def extras(unit,d,planned,method,poolidx,manifest_hash,order_idx):
    w=rd(d/"network_intent.json"); network=rd(d/"network.json")
    data=rd(d/"forward_audit/row_comparison.json")["native_forward_inputs"]
    key=unit["unit_id"]
    if set(data)!={"P3","P5"} or network["condition"]!=unit["condition"]:raise ValueError("network input identity")
    seen=V/"scenes"/unit["base_scene"]/"observed_rgbd.npz"
    cfg=rd(V/"preregistration/input_contract.json")
    entry=None
    if poolidx is not None:
        with np.load(d/"pool.npz",allow_pickle=False) as p:
            row=p["candidate_camera"][poolidx]; idx=int(p["source_indices"][poolidx])
        detail=rd(d/"orders.json")["candidate_details"][poolidx]
        if idx!=detail["source_index"]:raise ValueError("native candidate row mismatch")
        entry=dict(native_index=idx,pool_index=poolidx,coordinate_frame="camera_optical_x_right_y_down_z_forward",
            camera_17=row.tolist(),original_dtype=row.dtype.str,raw_bytes_sha256=hashlib.sha256(row.tobytes()).hexdigest(),
            camera_row_bytes_hex=row.tobytes().hex(),world_17=detail["candidate_world"],world_T_G=detail["world_T_G"],
            pre_open_m=detail["pre_open_m"],legacy_height_adjustment_m=0.)
    cloudseed=cfg["p3_base_seed"]+w["scene_key"]*1000+w["frame"]
    selected_receipt=rd(d/"selection.json")
    return dict(execution_order_1based=order_idx,within_unit_order=planned["within_unit_index"],
        unit_order_1based=unit["unit_order"],seed_policy=dict(order=cfg["order_seed"],p3_base=cfg["p3_base_seed"],
        p3_sampling=cloudseed,scene_key=w["scene_key"],frame=w["frame"],condition_recipe=network["corruption"]),
        stable_observation_path=seen.relative_to(V).as_posix(),stable_observation_sha256=sh(seen),
        actual_input_hashes_path=(d/"forward_audit/row_comparison.json").relative_to(V).as_posix(),
        actual_input_hashes_sha256=sh(d/"forward_audit/row_comparison.json"),actual_input_fingerprints=data,
        network_sha256=sh(d/"network.npz"),formal_p3_candidate=entry,
        no_candidate=poolidx is None,empty_common_pool=len(selected_receipt["eligibility"])==0,
        dependencies_manifest_path="audit/intent_dependencies.json",dependencies_manifest_sha256=manifest_hash,
        weights_sha256=network["weights"],cross_source_sha256=rd(V/"audit/cross_source/site_audit.json")["after_sha256"])

