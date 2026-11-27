"""Build the pre-connection schema from frozen URDFs, candidate registrations and reviewed logical interface."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/"IP140_R1_smoke18_remaining_five_bridge_confirmation01"
OLD=ROOT/"IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01"
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def field_format(v):
    if isinstance(v,list):return [field_format(x) for x in v]
    if type(v) in (int,float):return "number"
    if type(v) is bool:return "bool"
    if type(v) is str:return "text"
    raise ValueError("Unsupported field value type")
def build(output=None):
    reviewed=BASE/"postrun_analysis/zero_schema_diagnosis.json"
    prior=read(reviewed)
    common={r["logical_field"]:r["smoke18_slot8"]["atol"] for r in prior["shared_fields"]}
    if len(common)!=210:raise ValueError("Reviewed common field manifest is incomplete")
    assets={}
    slots={}
    frozen=read(BASE/"external_assets18.json")["files"]
    mesh_fields=["one_collision_shape","mesh_identity","mesh_scale","mesh_pose_xyz","mesh_pose_xyzw","collision_hull_geometry","mesh_filename"]
    box_fields=["compound_collision_shape_count","compound_registered_identity_and_box_types","compound_all_dims","compound_all_pos","compound_all_quat","compound_collision_geometry"]
    branches={"named_mesh":mesh_fields,"anonymous_mesh":mesh_fields[:-1],"compound_boxes":box_fields}
    for i,reg in [(i,HERE/"slots"/f"{i:02d}"/"preregistration.json") for i in range(8,13)]+[(14,OLD/"preregistration.json")]:
        rr=read(reg)["slots"][i-1]
        assert rr["slot"]==i
        spec=rr["scene_setup_only"]; u=spec["urdf"]; rel=u["project_relative"]; path=ROOT/rel
        if frozen[rel]!=u["sha256"] or sha(path)!=u["sha256"]:raise ValueError("URDF registry mismatch")
        parts=ET.parse(path).findall("./link/collision")
        asset=dict(urdf_relative=rel,urdf_sha256=u["sha256"],link_count=len(ET.parse(path).findall("./link")),primitive_count=len(parts))
        if len(parts)==1 and parts[0].find("geometry/mesh") is not None:
            mesh=parts[0].find("geometry/mesh")
            mesh_path=path.parent/mesh.get("filename")
            obj=mesh_path.relative_to(ROOT).as_posix()
            if obj not in frozen or frozen[obj]!=sha(mesh_path):raise ValueError("OBJ external registration mismatch")
            # Name availability is predeclared for this registered URI/pinned loader.
            # No live return selects the anonymous branch. A different asset would need its own pre-registration.
            asset.update(mode="named_mesh",obj_relative=obj,obj_sha256=frozen[obj],aabb_expansion_m=0.003)
        elif parts and all(x.find("geometry/box") is not None for x in parts):
            asset.update(mode="compound_boxes",aabb_expansion_m=0.001)
        else:raise ValueError("Unregistered URDF representation")
        assets[rel]=asset
        slots[str(i)]=dict(candidate_id=rr["candidate_id"],object_id=rr["object_id"],reg_sha256=sha(reg),urdf_relative=rel,
                           source="historical_regression" if i==14 else "active_slot")
    doc=dict(common_required=common,branches=branches,assets=assets,registrations=slots,active_slots=list(range(8,13)),
             field_atol_source={"mesh":{x["mesh"]["field"].split(".")[-1]:x["mesh"]["atol"] for x in prior["asset_specific_comparison"]},
                                "compound":{x["boxes"]["field"].split(".")[-1]:x["boxes"]["atol"] for x in prior["asset_specific_comparison"] if x["boxes"]}},
             interface_source=reviewed.relative_to(ROOT).as_posix(),interface_source_sha256=sha(reviewed),
             naming="target. and slot. are canonical display names. Check exact concrete field names; the legacy mesh-margin spelling is selected by registered geometry.",
             registered_anonymous_mesh_count=0)
    formats_common={r["logical_field"]:field_format(r["smoke18_slot8"]["expected"]) for r in prior["shared_fields"]}
    formats_specific={"mesh":{},"compound":{}}
    for variant in prior["asset_specific_comparison"]:
        for key,label in (("mesh","mesh"),("boxes","compound")):
            if variant[key]:
                row=variant[key];formats_specific[label][row["field"].split(".")[-1]]=field_format(row["expected"])
    doc.update(common_value_format=formats_common,specific_value_format=formats_specific)
    with (output or HERE/"asset_typed_schema.json").open("x",encoding="utf-8") as f:json.dump(doc,f,indent=2,allow_nan=False);f.write("\n")
if __name__=="__main__":build()
