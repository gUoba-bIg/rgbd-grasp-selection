"""Produce IP140-R1 model files without importing a simulator. Existing files are immutable."""
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

VERSION = Path(__file__).resolve().parent
ROOT = VERSION.parent
MODEL = VERSION / "model"
HISTORICAL = ROOT / "data/simulation_v52/robot/ur5e_robotiq_2f140.urdf"
PROTOCOL = {
    "IP140-R1_执行端替换两页协议_小修02.pdf": "3a1b30adc529a7a8536b03e6aba4d200a9545d6083da5c245ab624b2a1e4579e",
    "IP140-R1_参数与证据附表_小修02.pdf": "ebda80ada9f7dac264bef1a4cc60bb4c3807429788784cf1e2ab7015abb02389",
    HISTORICAL.relative_to(ROOT).as_posix(): "8bb0507bf59236e1914e3c8885c7cf8c16a19c12958279bd091a7818f0034950",
}
F_T_G = [[1, 0, 0, .070], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
MOTOR = {"controlMode": "POSITION_CONTROL", "positionGain": .1, "velocityGain": 1.,
         "targetVelocity": 0., "force_N_per_slider": 20., "maxVelocity_m_s_per_slider": .040,
         "reference_slew_m_s_per_slider": .040, "jointDamping_Ns_m": 1., "jointDryFriction_N": 0.}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyz(vals):
    return " ".join(format(v, ".15g") for v in vals)


def origin(parent, translation, rpy=(0, 0, 0)):
    ET.SubElement(parent, "origin", xyz=xyz(translation), rpy=xyz(rpy))


def add_box(robot, components, name, size, com, com_at_zero_G, mass, inertia, mu, rgba):
    link = ET.SubElement(robot, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    origin(inertial, com)
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]),
                  izz=str(inertia[2]), ixy="0", ixz="0", iyz="0")
    # Single primitive, external collision faces coincide with the approved box.
    for tag in ("collision", "visual"):
        part = ET.SubElement(link, tag, name=name+"_"+tag)
        origin(part, com)
        ET.SubElement(ET.SubElement(part, "geometry"), "box", size=xyz(size))
        if tag == "visual":
            ET.SubElement(ET.SubElement(part, "material", name=name+"_color"), "color", rgba=rgba)
    contact = ET.SubElement(link, "contact")
    for key, value in (("lateral_friction", mu), ("rolling_friction", 0.),
                       ("spinning_friction", 0.), ("restitution", 0.)):
        ET.SubElement(contact, key, value=str(value))
    theoretical = [mass*(size[(k+1)%3]**2+size[(k+2)%3]**2)/12 for k in range(3)]
    if not all(math.isclose(a,b,rel_tol=1e-11) for a,b in zip(inertia,theoretical)):
        raise ValueError(f"Inconsistent cuboid inertia: {name}")
    components[name] = dict(shape="GEOM_BOX", box_size_m=size, center_in_link_m=com,
        com_at_zero_in_G_m=com_at_zero_G, orientation_xyzw=[0,0,0,1], mass_kg=mass,
        inertia_at_com_kg_m2=inertia, inertia_offdiagonal=[0,0,0], lateralFriction=mu,
        rollingFriction=0., spinningFriction=0., restitution=0., collisionMargin_m=.0005,
        linearDamping=0., angularDamping=0.)


def add_joint(robot, joints, name, parent, child, translation, rpy=(0,0,0), axis=None):
    kind = "fixed" if axis is None else "prismatic"
    joint = ET.SubElement(robot, "joint", name=name, type=kind)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    origin(joint, translation, rpy)
    row = dict(type=kind, parent=parent, child=child, xyz_m=translation, rpy_rad=rpy)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=xyz(axis))
        ET.SubElement(joint, "limit", lower="0", upper=".07", effort="20", velocity=".04")
        ET.SubElement(joint, "dynamics", damping="1.0", friction="0")
        row.update(axis_in_joint_frame=axis, limits_m=[0,.07], open_at_positive_q=True,
                   zero_position_m=0., motor=MOTOR)
    joints[name] = row


def write_new_or_identical(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing content differs; cannot overwrite: {path}")
        return
    with path.open("xb") as dest:
        dest.write(data)


def build():
    for rel, expected in PROTOCOL.items():
        if digest(ROOT/rel) != expected:
            raise ValueError(f"Historical source digest changed: {rel}")
    robot = ET.parse(HISTORICAL).getroot()
    robot.set("name", "ur5e_ip140_r1_smoke01")
    for node in list(robot):
        if node.get("name","").startswith("rq_"):
            robot.remove(node)
    assets = {}
    for mesh in robot.findall(".//mesh"):
        actual = (HISTORICAL.parent / mesh.attrib["filename"]).resolve(strict=True)
        assets[actual.relative_to(ROOT).as_posix()] = digest(actual)
        mesh.set("filename", Path(os.path.relpath(actual, MODEL)).as_posix())
    # Legacy OBJ visual files reference the sibling material.mtl.
    material = HISTORICAL.parent / "meshes/material.mtl"
    assets[material.relative_to(ROOT).as_posix()] = digest(material)

    parts, joints = {}, {}
    add_box(robot,parts,"ip140_palm",[.010,.160,.020],[-.065,0,0],[-.065,0,0],
            .086400,[1.872e-4,3.6e-6,1.8504e-4],.50,"0.25 0.31 0.37 1")
    add_joint(robot,joints,"ip140_mount","flange","ip140_palm",[.070,0,0],[math.pi/2,0,0])
    for side,sign in (("left",-1),("right",1)):
        bone, pad = "ip140_"+side+"_bone", "ip140_"+side+"_pad"
        add_box(robot,parts,bone,[.060,.008,.020],[.030,sign*.006,0],[-.030,sign*.006,0],
                .025920,[1.00224e-6,8.64e-6,7.91424e-6],.50,"0.70 0.74 0.79 1")
        add_joint(robot,joints,"ip140_"+side+"_slide","ip140_palm",bone,[-.060,0,0],
                  axis=[0,sign,0])
        add_box(robot,parts,pad,[.060,.002,.020],[.030,sign*.001,0],[-.030,sign*.001,0],
                .002640,[8.888e-8,8.800e-7,7.9288e-7],1.0,"0.12 0.17 0.19 1")
        add_joint(robot,joints,"ip140_"+side+"_pad_mount",bone,pad,[0,0,0])
    if robot.findall(".//mimic") or len(robot.findall("./joint[@type='prismatic']"))!=2:
        raise ValueError("Unexpected joint topology")
    linknames = [link.attrib["name"] for link in robot.findall("link")]
    childnames = [j.find("child").get("link") for j in robot.findall("joint")]
    if len(linknames)!=len(set(linknames)) or sorted(childnames)!=sorted(set(linknames)-{"world"}):
        raise ValueError("Joint/link mapping inconsistent")
    ET.indent(robot, space="  ")
    raw = ET.tostring(robot,encoding="utf-8",xml_declaration=True) + b"\n"
    filename = "ur5e_ip140_r1.urdf"
    manifest = dict(version="IP140-R1/smoke01", model=filename, model_sha256=hashlib.sha256(raw).hexdigest(),
        purpose="Engineering smoke only; design targets and provenance, not engine readback",
        provenance=PROTOCOL, reused_arm_asset_sha256=assets, flange_link="flange",
        G_link="ip140_palm", F_T_G=F_T_G, units="SI: m, kg, s, rad, N, kg*m^2",
        model_type="single width command, two symmetric independent prismatic sliders",
        components=parts, joints=joints, required_distinct_pad_links=["ip140_left_pad","ip140_right_pad"],
        merge_fixed_links=False, gear_mimic_or_loop_count=0, gripper_mass_kg=.14352,
        load_contract=dict(useFixedBase=True, retain_file_inertias=True, self_collision=True,
            include_parent_collisions=True, fixed_link_merging=False, wholesale_pair_exclusions_for_new_geometry=False),
        readback_contract="Margin/damping and motor properties require runtime setup and engine verification; URDF values and setters are not acceptance.",
        pad_inner_faces_G="yL=-q_left, yR=+q_right; x in [-.060,0], z in [-.010,.010]; no negative endpoint command.",
        design_assumptions="Gripper masses, inertias, gains, friction and control settings are simulation design choices, not measured hardware values.")
    MODEL.mkdir(exist_ok=True)
    write_new_or_identical(MODEL/filename,raw)
    write_new_or_identical(MODEL/"manifest.json",(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))
    print(filename, manifest["model_sha256"])
    print("manifest.json",digest(MODEL/"manifest.json"))


if __name__ == "__main__":
    build()
