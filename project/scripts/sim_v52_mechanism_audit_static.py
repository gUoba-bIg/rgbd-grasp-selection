"""Pure XML/NumPy kinematic audit. No simulator imports or physical execution."""
from __future__ import annotations
import hashlib
import json
import math
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/simulation_v52_dynamics_audit/mechanism.json"
URDF = ROOT / "data/simulation_v52/robot/ur5e_robotiq_2f140.urdf"
CONFIG = ROOT / "configs/simulation_v52r2.json"
SOURCE = ROOT / "third_party/simulation_v52/robotiq/grippers/robotiq_description/urdf/robotiq_2f_140.xacro"
COMMIT = "27ae16b7d2c7d7e9011544521d644a0f38c4b501"
REMOTE = "https://raw.githubusercontent.com/robotiq/isaacsim_assets/" + COMMIT + "/"
PAYLOAD = "grippers/Robotiq_2F_140/payloads/"
BASE_LINK = "rq_robotiq_140_base_link"
ORDER = [
    "rq_finger_joint", "rq_left_inner_knuckle_joint", "rq_left_inner_finger_joint",
    "rq_right_outer_knuckle_joint", "rq_right_inner_knuckle_joint", "rq_right_inner_finger_joint",
]
M = np.array([1., -1., 1., -1., -1., 1.])
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def vec(s):
    return np.fromstring(s, sep=" ")
def skew(a):
    x,y,z = a
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def rot(axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = skew(axis)
    return np.eye(3) + math.sin(angle)*k + (1.-math.cos(angle))*(k@k)
def origin(e):
    if e is None:
        return np.eye(4)
    a = vec(e.get("rpy", "0 0 0"))
    T = np.eye(4)
    T[:3,:3] = rot(np.array([0.,0.,1.]),a[2]) @ rot(np.array([0.,1.,0.]),a[1]) @ rot(np.array([1.,0.,0.]),a[0])
    T[:3,3] = vec(e.get("xyz", "0 0 0"))
    return T
def rank(a, tol=1e-8):
    return int(np.linalg.matrix_rank(a,tol=tol))

raw = ET.parse(URDF).getroot()
joints = {}
for element in raw.findall("joint"):
    name = element.get("name")
    if not name.startswith("rq_"):
        continue
    a = element.find("axis")
    lim = element.find("limit")
    mim = element.find("mimic")
    joints[name] = {
        "name":name, "type":element.get("type"),
        "parent":element.find("parent").get("link"), "child":element.find("child").get("link"),
        "T":origin(element.find("origin")), "axis":vec(a.get("xyz")) if a is not None else np.array([1.,0.,0.]),
        "limits":dict(lim.attrib) if lim is not None else None,
        "mimic":dict(mim.attrib) if mim is not None else None,
    }
links = {e.get("name"): e for e in raw.findall("link")}
def fk(q, extra=None):
    values = dict(zip(ORDER,q))
    if extra:
        values.update(extra)
    poses = {BASE_LINK:np.eye(4)}
    pending = [j for j in joints.values() if j["parent"] != "tool0" and j["child"] != BASE_LINK and j["child"] != "rq_robotiq_base_link"]
    while pending:
        changed = False
        for j in list(pending):
            if j["parent"] not in poses:
                continue
            motion = np.eye(4)
            if j["type"] == "revolute" or j["name"] in (extra or {}):
                motion[:3,:3] = rot(j["axis"],values.get(j["name"],0.))
            poses[j["child"]] = poses[j["parent"]] @ j["T"] @ motion
            pending.remove(j)
            changed = True
        if not changed:
            raise RuntimeError("Non-tree or missing-root XML")
    return poses

inputs = {str(p.relative_to(ROOT)):digest(p) for p in [URDF,CONFIG,SOURCE]}
cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
C = []
for i,name in enumerate(ORDER[1:],1):
    row = np.zeros(6); row[i] = 1.; row[0] = -float(joints[name]["mimic"]["multiplier"])
    C.append(row)
C = np.array(C)
tree = []
for relation in cfg["coupling_repair"]["tree"]:
    row = np.zeros(6); row[ORDER.index(relation["a"])] = 1.; row[ORDER.index(relation["b"])] = relation["ratio"]
    tree.append(row)
tree = np.array(tree)

# Fixed-commit official numerical descriptions are read in memory, not installed.
remote_sources = {}
remote_texts = {}
for name in ["Robotiq_2F_140_kinematics_physics.usda","Robotiq_2F_140_parallel_grip_physics.usda",
             "Robotiq_2F_140_compliant_physics.usda","Robotiq_2F_140_newton_fourbar_physics.usda","base.usda"]:
    rel = PAYLOAD + name
    request = urllib.request.Request(REMOTE+rel,headers={"User-Agent":"Research-static-mechanism-audit"})
    with urllib.request.urlopen(request,timeout=30) as response:
        data = response.read(150000)
    text = data.decode("utf-8")
    remote_texts[name] = text
    remote_sources[name] = {
        "url":REMOTE+rel, "commit":COMMIT, "bytes":len(data),
        "sha256":hashlib.sha256(data).hexdigest(),
        "snapshot_kind":"selected_structured_mechanical_fields_and_line_references_not_full_USD_composition",
    }

def joint_blocks(text):
    result = {}
    for match in re.finditer(r'def PhysicsRevoluteJoint "([^"]+)"',text):
        start = text.index("{",match.end())
        depth=1; pos=start+1
        while depth:
            depth += (text[pos]=="{")-(text[pos]=="}")
            pos+=1
        body = text[start:pos]
        row={"line":text[:match.start()].count("\n")+1}
        for key in ["localPos0","localPos1","localRot0","localRot1"]:
            m=re.search(r"physics:"+key+r"\s*=\s*\(([^)]+)\)",body)
            if m:row[key]=[float(x) for x in m.group(1).split(",")]
        for key in ["body0","body1"]:
            m=re.search(r"physics:"+key+r"\s*=\s*<([^>]+)>",body)
            if m:row[key]=m.group(1)
        for key in ["lowerLimit","upperLimit"]:
            m=re.search(r"physics:"+key+r"\s*=\s*([-0-9.]+)",body)
            if m:row[key]=float(m.group(1))
        result[match.group(1)]=row
    return result
official = joint_blocks(remote_texts["Robotiq_2F_140_kinematics_physics.usda"])

closures = {}
qfit = np.linspace(0.,.7,31)
qgrid = np.unique(np.append(np.linspace(0.,.7,1401),.695))
for side in ["left","right"]:
    outername = "rq_finger_joint" if side=="left" else "rq_right_outer_knuckle_joint"
    outer=joints[outername]
    k=joints["rq_"+side+"_inner_knuckle_joint"]
    f=joints["rq_"+side+"_inner_finger_joint"]
    fixed=joints["rq_"+side+"_outer_finger_joint"]
    pO=outer["T"][:3,3]
    pK=k["T"][:3,3]
    a=fixed["T"][:3,3]+f["T"][:3,3]
    Rf=fk(np.zeros(6))["rq_"+side+"_inner_finger"][:3,:3]
    b=Rf.T@(pK-pO)
    b[0]=0. # The x coordinate selects a representative on the same hinge axis.
    rows=[]; rhs=[]
    for q0 in qfit:
        poses=fk(q0*M); tk=poses["rq_"+side+"_inner_knuckle"]; tf=poses["rq_"+side+"_inner_finger"]
        rows.append(np.column_stack([tk[:3,:3],-tf[:3,:3]])); rhs.append(tf[:3,3]-tk[:3,3])
    A=np.vstack(rows); y=np.concatenate(rhs)
    fit,_,_,_=np.linalg.lstsq(A,y,rcond=None)
    _,singular,Vt=np.linalg.svd(A,full_matrices=False)
    residuals=[]; axis_errors=[]; orientation_errors=[]; gap=[]
    atzero=fk(np.zeros(6))
    for q0 in qgrid:
        poses=fk(q0*M); tk=poses["rq_"+side+"_inner_knuckle"]; tf=poses["rq_"+side+"_inner_finger"]
        residuals.append(np.linalg.norm(tk[:3,:3]@a+tk[:3,3]-tf[:3,:3]@b-tf[:3,3]))
        axis_errors.append(np.linalg.norm(np.cross(tk[:3,0],tf[:3,0])))
        orientation_errors.append(np.linalg.norm(tf[:3,:3]-atzero["rq_"+side+"_inner_finger"][:3,:3]))
    close0=atzero["rq_"+side+"_inner_knuckle"][:3,:3]@a+atzero["rq_"+side+"_inner_knuckle"][:3,3]
    remotejoint=official[side+"_inner_finger_knuckle_joint"]
    formal=np.array(remotejoint["localPos0"])
    def to_com(link,point):
        inertial=links[link].find("inertial")
        frame=origin(inertial.find("origin") if inertial is not None else None)
        return (frame[:3,:3].T@(point-frame[:3,3])).tolist()
    closures[side]={
        "knuckle_link":"rq_"+side+"_inner_knuckle", "finger_link":"rq_"+side+"_inner_finger",
        "anchor_in_knuckle_link_m":a.tolist(), "anchor_in_inner_finger_link_m":b.tolist(),
        "axis_in_each_link":[1.,0.,0.], "point_on_axis_gauge":"x=0; a unique axis line, not a uniquely identified bearing midpoint",
        "knuckle_com_anchor_m":to_com("rq_"+side+"_inner_knuckle",a),
        "inner_finger_com_anchor_m":to_com("rq_"+side+"_inner_finger",b),
        "derived_base_anchor_at_zero_m":close0.tolist(),
        "official_140_base_anchor_at_zero_m":formal.tolist(),
        "official_anchor_distance_m":float(np.linalg.norm(close0-formal)),
        "long_link_length_m":float(np.linalg.norm(a)), "ground_and_distal_length_m":float(np.linalg.norm(pK-pO)),
        "fit_rank":rank(A),"fit_singular_values":singular.tolist(),"fit_anchor_vector":fit.tolist(),
        "fit_max_residual_m":float(np.max(np.abs(A@fit-y))),
        "fit_nullspace_vector":Vt[-1].tolist(),
        "full_interval_continuous_proof":{
            "variable":"q in [0,0.7] (also any q within this ideal nominal branch)",
            "identity":"pK+R(q)*a = pO+R(q)*v+Rf*b, where a=v and Rf*b=pK-pO",
            "constant_rotation_identity_residual":float(np.linalg.norm(outer["T"][:3,:3]@f["T"][:3,:3]-Rf)),
            "constant_translation_identity_residual_m":float(np.linalg.norm(Rf@b-(pK-pO))),
            "fit_is_not_the_proof":"The coefficient identity proves closure for all q; the 1402-point sweep only checks numerical implementation.",
        },
        "sweep":{"range_rad":[0.,.7],"points":int(len(qgrid)),"includes_protocol_endpoint_rad":.695,
                 "max_closure_error_m":float(max(residuals)),"max_axis_cross_product":float(max(axis_errors)),
                 "max_inner_finger_rotation_matrix_change":float(max(orientation_errors))}
    }

def closed_positions(q,extra=None):
    poses=fk(q,extra)
    residual=[]
    for side in ["left","right"]:
        row=closures[side]
        a=np.array(row["anchor_in_knuckle_link_m"]); b=np.array(row["anchor_in_inner_finger_link_m"])
        tk=poses[row["knuckle_link"]]; tf=poses[row["finger_link"]]
        residual.extend(tk[:3,:3]@a+tk[:3,3]-tf[:3,:3]@b-tf[:3,3])
    return np.array(residual)
def jacobian(fun,x):
    columns=[]
    h=1e-6
    for i in range(len(x)):
        d=np.zeros(len(x));d[i]=h
        columns.append((fun(x+d)-fun(x-d))/(2*h))
    return np.column_stack(columns)
sync=np.zeros(6);sync[0]=sync[3]=1.
jac_rows=[]
for q0 in np.unique(np.append(np.linspace(0.,.7,141),.695)):
    q=q0*M
    J=jacobian(closed_positions,q)
    combined=np.vstack([J,sync])
    singular=np.linalg.svd(combined,compute_uv=False)
    q8=np.concatenate([q,[0.,0.]])
    def full_fivebar(x):
        return closed_positions(x[:6],{"rq_left_outer_finger_joint":x[6],"rq_right_outer_finger_joint":x[7]})
    J8=jacobian(full_fivebar,q8)
    sync8=np.concatenate([sync,[0.,0.]])
    jac_rows.append({"q_rad":float(q0),"loop_rank":rank(J),"combined_rank":rank(combined),
                     "rank_stacked_with_original":rank(np.vstack([combined,C])),
                     "minimum_nonzero_singular_value":float(singular[4]),
                     "axial_row_max_abs":float(np.max(np.abs(J[[0,3]]))),
                     "fivebar_loop_rank":rank(J8),"fivebar_plus_sync_rank":rank(np.vstack([J8,sync8]))})
widths={}
for q0 in [0.,.34680420592893596,.695,.7]:
    poses=fk(q0*M)
    centers=[]
    for side in ["left","right"]:
        name="rq_"+side+"_inner_finger_pad";collision=links[name].find("collision")
        T=poses[name]@origin(collision.find("origin"))
        centers.append(T[:3,3])
    widths[str(q0)]={"pad_gap_m":float(centers[1][1]-centers[0][1]-.0075),
                    "center_z_m":float((centers[0][2]+centers[1][2])/2)}
# Explicit topology and source line references; no model instantiation.
remote_lines={}
for name,text in remote_texts.items():
    tokens = ["outer_finger_joint","inner_finger_knuckle_joint","excludeFromArticulation","stiffness","spherical","active = false"]
    remote_lines[name]=[{"line":i+1,"text":line.strip()} for i,line in enumerate(text.splitlines()) if any(token in line for token in tokens)]
after={str(p.relative_to(ROOT)):digest(p) for p in [URDF,CONFIG,SOURCE]}
result={
    "schema":"v52_mechanism_static_audit_v1",
    "at":datetime.now(timezone.utc).isoformat(),
    "scope":"pure XML and NumPy; no simulator import, no physics, no old-file mutation",
    "execution":{"physics_connections":0,"physics_steps":0,"grasp_attempts":0,"tests_run":0,"bytecode_writes":0},
    "inputs":inputs,"inputs_unchanged":inputs==after,
    "script_sha256":digest(Path(__file__)),
    "joint_order":ORDER,"nominal_multipliers":M.tolist(),
    "joint_roles":[{k:v for k,v in joints[name].items() if k not in ["T","axis"]}|
                   {"origin_matrix":joints[name]["T"].tolist(),"axis":joints[name]["axis"].tolist()} for name in ORDER],
    "fixed_outer_finger_joints":[joints["rq_"+s+"_outer_finger_joint"]["type"] for s in ["left","right"]],
    "original_mimic_matrix":C.tolist(),"v52r2_local_tree_matrix":tree.tolist(),
    "rank_original":rank(C),"rank_tree":rank(tree),"rank_stacked_original_tree":rank(np.vstack([C,tree])),
    "nominal_nullspace_residual_original":float(np.max(np.abs(C@M))),
    "nominal_nullspace_residual_tree":float(np.max(np.abs(tree@M))),
    "source_commit_official_140":COMMIT,"official_sources":remote_sources,
    "official_140_joint_fields":official,"official_source_line_references":remote_lines,
    "closure_axes":closures,"closure_jacobian_grid":jac_rows,
    "jacobian_summary":{
        "grid_points":len(jac_rows),"range_rad":[0.,.7],
        "loop_rank_set":sorted(set(x["loop_rank"] for x in jac_rows)),
        "loop_plus_sync_rank_set":sorted(set(x["combined_rank"] for x in jac_rows)),
        "stacked_old_new_rank_set":sorted(set(x["rank_stacked_with_original"] for x in jac_rows)),
        "max_axial_row_abs":max(x["axial_row_max_abs"] for x in jac_rows),
        "minimum_nonzero_singular_value":min(x["minimum_nonzero_singular_value"] for x in jac_rows),
        "fivebar_loop_rank_set":sorted(set(x["fivebar_loop_rank"] for x in jac_rows)),
        "fivebar_plus_sync_rank_set":sorted(set(x["fivebar_plus_sync_rank"] for x in jac_rows)),
    },
    "degrees_of_freedom":{
        "urdf_without_mimic":6,"urdf_with_rank5_mimic":1,
        "parallel_fourbar_per_side":1,"both_fourbars_before_side_sync":2,"parallel_with_side_sync":1,
        "compliant_fivebar_per_side":2,"both_fivebars_before_side_sync":4,"compliant_with_side_sync":3,
        "full_fivebar_count_interpretation":"One actuated and two passive configuration DOFs before stops/contact; not three motors.",
    },
    "pad_gap_endpoint_observations":widths,
    "interpretation":{
        "original_five_gears_are_not_redundant":True,
        "nominal_mimic_is_not_adaptive_mechanical_model":True,
        "actual_spring_rates_gear_train_ratios_and_friction_identified":False,
        "official_simulator_tuning_is_not_real_hardware_calibration":True,
        "closure_axis_unique_with_axial_gauge":True,
        "full_compliant_contact_branch_validated":False,
        "p2p_three_rows_have_only_two_independent_rows_per_planar_side":True,
        "future_representation_change_requires_review_and_fresh_protocol":True,
        "existing_constraints_deleted":False,
    },
}
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open("x",encoding="utf-8",newline="\n") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
    f.write("\n")
print(json.dumps({"output":str(OUT.relative_to(ROOT)),"rank_original":rank(C),"rank_tree":rank(tree),
                  "jacobian_summary":result["jacobian_summary"],"widths":widths,"old_inputs_unchanged":inputs==after,
                  "physical_steps":0,"script_imported_simulator":any(n=="pybullet" or n.startswith("pybullet.") for n in sys.modules)},indent=2))
