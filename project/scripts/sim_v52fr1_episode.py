"""v52f: repaired-parameter M0, original twelve units; new isolated engine only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROCESS_STARTED = time.monotonic()
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    st = p.stat()
    return dict(sha256=h.hexdigest(), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def project_path(value):
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def require_identity(path, expected):
    actual = identity(path)
    if actual != expected:
        raise ValueError("identity_mismatch:" + str(Path(path).relative_to(ROOT)))
    return actual


def phase_schedule(dt, base):
    motion = base["motion"]
    rows = [("settle", 1.0), ("opening", .8),
            ("transit", motion["transit_seconds"] + .3),
            ("pregrasp", motion["pregrasp_seconds"] + .3),
            ("approach", motion["approach_seconds"] + .3),
            ("close", motion["close_seconds"]),
            ("lift", motion["lift_seconds"] + .3),
            ("hold", motion["hold_seconds"])]
    return [{"phase":name, "seconds":seconds, "steps":round(seconds/dt)} for name,seconds in rows]


def validate_launch(args):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Explicit project workdir required")
    config_path = project_path(args.config)
    directory = project_path(args.attempt_dir)
    if directory == ROOT or not directory.is_dir():
        raise ValueError("Controller must create an exclusive attempt directory")
    token = os.environ.get("SIM_V52F_CONTROLLER_TOKEN")
    if not token:
        raise RuntimeError("Use the precharging diagnostic controller")
    cfg = read(config_path)
    if cfg.get("schema") != "sim_v52f_execution_v1" or cfg.get("execution_authorized") is not True or cfg.get("model") != "M0":
        raise ValueError("Only explicitly authorized M0 diagnostics are permitted")
    base_path = project_path(cfg["base_config"])
    if base_path != (ROOT / "configs/simulation_v52r2.json").resolve():
        raise ValueError("M0 must use the frozen r2 base config")
    plan_path = project_path(cfg["plan_path"])
    action_path = project_path(cfg["action_path"])
    plan, action, base = read(plan_path), read(action_path), read(base_path)
    if cfg.get("resources", plan["resources"]) != plan["resources"]:
        raise ValueError("Resource caps may not change")
    if cfg.get("criteria", plan["fixed"]["original_criteria"]) != plan["fixed"]["original_criteria"]:
        raise ValueError("Acceptance gates may not change")
    if base["physics"] != plan["fixed"]["original_physics"] or base["motion"] != plan["fixed"]["original_motion"] or base["criteria"] != plan["fixed"]["original_criteria"]:
        raise ValueError("Frozen M0 physics/motion/criteria mismatch")
    if cfg.get("combinations", plan["combinations"]) != plan["combinations"]:
        raise ValueError("Only the twelve original single-use combinations are permitted")
    matches = [row for row in plan["combinations"] if row["id"] == args.unit]
    if len(matches) != 1:
        raise ValueError("Unknown diagnostic unit")
    unit = matches[0]
    precharge = read(directory / "precharge.json")
    if precharge.get("schema") != "sim_v52f_precharge_v1" or precharge.get("unit_id") != args.unit or precharge.get("unit") != unit:
        raise ValueError("Missing or mismatched precharge")
    expected_index = next(i+1 for i,row in enumerate(plan["combinations"]) if row["id"] == args.unit)
    if precharge.get("charged_index") != expected_index:
        raise ValueError("Precharge is not in the fixed order")
    if precharge.get("token_hash") != hashlib.sha256(token.encode("utf-8")).hexdigest():
        raise ValueError("Controller token mismatch")
    config_id = require_identity(config_path, precharge["config_identity"])
    source_id = require_identity(Path(__file__), precharge["worker_identity"])
    for relative, bound in plan["bindings"].items():
        require_identity(project_path(relative), bound)
    for relative, bound in cfg.get("bindings", {}).items():
        require_identity(project_path(relative), bound)
    expected_python=project_path("runtime/simulation_v52e/venv/Scripts/python.exe")
    if project_path(cfg["python"])!=expected_python or Path(sys.executable).resolve()!=expected_python:
        raise RuntimeError("Only the independently repaired v52e interpreter is authorized")
    if cfg.get("expected_distribution_version")!="3.2.7+v52e1":
        raise RuntimeError("Unexpected patched engine version")
    spec=importlib.util.find_spec("pybullet")
    if spec is None or Path(spec.origin).resolve()!=project_path(cfg["pybullet_extension"]):
        raise RuntimeError("Engine resolution does not match isolated bound binary")
    if importlib.metadata.version("pybullet")!=cfg["expected_distribution_version"]:
        raise RuntimeError("Installed engine metadata version mismatch")
    mandatory={cfg["python"],cfg["pybullet_extension"],cfg["parameter_reference_readback"],cfg["dependency_manifest"],"scripts/sim_v52fr1_episode.py"}
    if not mandatory.issubset(cfg["bindings"]):
        raise RuntimeError("Unbound engine, worker, reference or dependency provenance")
    dependency_root=project_path(cfg["dependency_dir"])
    dependencies=read(project_path(cfg["dependency_manifest"]))
    if not dependencies.get("files"):
        raise RuntimeError("Empty dependency manifest")
    for relative,bound in dependencies["files"].items():
        path=project_path(relative);path.relative_to(dependency_root);require_identity(path,bound)
    scipy_spec=importlib.util.find_spec("scipy")
    if scipy_spec is None or not Path(scipy_spec.origin).resolve().is_relative_to(dependency_root):
        raise RuntimeError("SciPy must resolve within isolated frozen v52f dependencies")
    if importlib.metadata.version("scipy")!="1.15.3":
        raise RuntimeError("Only the original SciPy1.15.3 dependency is authorized")
    if (directory / "result.json").exists():
        raise FileExistsError("No result overwrite or retry")
    if (directory / "worker_claim.json").exists():
        raise FileExistsError("A worker already claimed this charged unit")
    effective = copy.deepcopy(base)
    effective["physics"]["time_step"] = unit["time_step_s"]
    effective["physics"]["solver_iterations"] = unit["solver_iterations"]
    effective["limits"]["episode_wall_seconds"] = plan["resources"]["per_run_wall_cap_s"]
    effective["limits"]["episode_sim_seconds"] = unit["maximum_sim_seconds"]
    schedule = phase_schedule(unit["time_step_s"], effective)
    if sum(row["steps"] for row in schedule) != unit["maximum_step_count"]:
        raise ValueError("Fixed schedule does not match step budget")
    if not math.isclose(sum(row["seconds"] for row in schedule),18.5,abs_tol=1e-12):
        raise ValueError("Unexpected physical duration")
    if [row["name"] for row in action["planning_joint_targets"]] != ["transit","pregrasp","approach","lift"]:
        raise ValueError("Cached action ordering changed")
    if shutil.disk_usage(ROOT).free < plan["resources"]["minimum_free_disk_bytes"]:
        raise RuntimeError("Insufficient free disk for fixed diagnostic budget")
    write_once(directory / "worker_claim.json", dict(
        attempt_id=precharge["attempt_id"], unit_id=args.unit, pid=os.getpid(),
        config_identity=config_id, source_identity=source_id,
        expected_physics_steps=unit["maximum_step_count"], no_retry=True))
    return cfg, effective, plan, action, unit, precharge, directory, config_path



def normalize(value):
    # Match the preserved v52e lossless encoding of raw C-binding bytes.
    def encode(item):
        if isinstance(item, bytes):return {"bytes_hex":item.hex()}
        if isinstance(item, (tuple,list)):return [encode(x) for x in item]
        if isinstance(item, dict):return {str(k):encode(v) for k,v in item.items()}
        return item
    return json.loads(json.dumps(encode(value),ensure_ascii=False,allow_nan=False))


def geometry_equal(a,b):
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12+1e-10*max(abs(a),abs(b))
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(geometry_equal(x,y) for x,y in zip(a,b))
    return a==b


def read_setup_robot_parameters(client,robot,reference):
    names={index:name for name,index in robot["links"].items()};names[-1]="BASE"
    old={x["link_index"]:x for x in reference["links"]};rows=[];failures=[]
    for index in sorted(names):
        expected=old.get(index);raw=normalize(client.getDynamicsInfo(robot["body"],index))
        geometry=normalize(client.getCollisionShapeData(robot["body"],index))
        row=dict(link_index=index,link_name=names[index],dynamics_raw=raw,collision_geometry_raw=geometry)
        rows.append(row)
        if expected is None or expected["link_name"]!=names[index]:
            failures.append(dict(link=index,field="link_identity"));continue
        if raw!=expected["dynamics_raw"]:
            failures.append(dict(link=index,field="dynamics_raw",expected=expected["dynamics_raw"],actual=raw))
        old_geometry=normalize(expected["collision_geometry_raw"])
        # The table is loaded first in diagnostics, so body IDs differ from v52e.
        geometry_comparable=normalize(geometry)
        for item in geometry_comparable:item[0]=0
        for item in old_geometry:item[0]=0
        if not geometry_equal(geometry_comparable,old_geometry):
            failures.append(dict(link=index,field="collision_geometry",expected=old_geometry,actual=geometry))
    if set(names)!=set(old):failures.append(dict(field="full_link_set",expected=sorted(old),actual=sorted(names)))
    return dict(schema="sim_v52f_setup_readback_v1",physics_steps=0,reference_label=reference["label"],
        expected_base_lateral_friction=.5,expected_base_margin_m=.001,links=rows,passed=not failures,failures=failures,
        comparison="Full raw dynamics equality; geometry only body ID normalization and frozen1e-12/1e-10 numerical tolerance; no pose comparison after original home reset")


def read_pad_parameter_assertion(client,robot,manifest,step,phase,previous_steps):
    expected={"lateralFriction":1.,"collisionMargin":.0005,"restitution":0.}
    record=dict(schema="sim_v52f_pad_parameter_assertion_v1",step=step,phase=phase,
        physics_steps_returned_before=previous_steps,when="before_stepSimulation",expected=expected,pads=[],passed=False,
        exception=None,readback_calls_started=0,readback_calls_returned=0,wall_seconds=time.monotonic()-PROCESS_STARTED)
    try:
        for name,index in zip(manifest["pad_links"],robot["pads"]):
            record["readback_calls_started"]+=1
            raw_value=client.getDynamicsInfo(robot["body"],index)
            record["readback_calls_returned"]+=1
            try:raw=normalize(raw_value)
            except BaseException:
                record["invalid_returned_raw_repr"]=repr(raw_value)
                raise
            if len(raw)<12:raise RuntimeError("Full dynamics tuple with collision margin required")
            actual={"lateralFriction":raw[1],"collisionMargin":raw[11],"restitution":raw[5]}
            mismatches={key:dict(expected=value,actual=actual[key]) for key,value in expected.items()
                if not isinstance(actual[key],(int,float)) or not math.isfinite(actual[key]) or actual[key]!=value}
            record["pads"].append(dict(link_index=index,link_name=name,raw=raw,actual=actual,passed=not mismatches,mismatches=mismatches))
        record["passed"]=len(record["pads"])==2 and all(x["passed"] for x in record["pads"])
    except BaseException as exc:
        record["exception"]=repr(exc);record["passed"]=False
    return record


def run_unit(diag, cfg, plan, action, unit, charge, directory, config_path):
    result = dict(
        status="failed", unit_id=unit["id"], unit=unit, scenario=unit["scenario"],
        attempt_id=charge["attempt_id"], model="M0", completion=False,
        actual_physics_steps=0, expected_physics_steps=unit["maximum_step_count"],
        pad_assertions_attempted=0, pad_assertions_passed=0, last_failed_pad_assertion=None, parameter_assertions_passed=False,
        parameter_reference_identity=identity(project_path(diag["parameter_reference_readback"])),
        physical_condition_note="Corrected pad1/.0005/rest0 and base/non-target v52e baseline differ from erroneous v52d actual physics; not a structural ablation.",
        physics_step_calls_started=0, physics_step_calls_returned=0, unknown_step_call=None,
        exception=None, config_identity=identity(config_path), source_identity=identity(__file__),
        base_config_identity=identity(project_path(diag["base_config"])),
        action_identity=identity(project_path(diag["action_path"])),
        actual_new_training=0, actual_neural_forwards=0,
        new_object_attachment_constraints=0, object_teleports_after_initialization=0,
        telemetry_complete=False, summary={}, original_gate_window="post_settle",
        grasp_success=None if unit["scenario"] != "C" else False,
        original_grasp_success=None if unit["scenario"] != "C" else False,
        diagnostic_pass_is_not_engineering_admission=True,
        forbidden_collision=False, grasp_formed=None if unit["scenario"] != "C" else False,
        slipped=None if unit["scenario"] != "C" else False,
        cached_action=action, phase_schedule=phase_schedule(unit["time_step_s"], cfg),
    )
    client = recorder = assertion_stream = None
    collision_events = []
    cleanup_errors = []
    phase = "initialization"
    try:
        import numpy as np
        import psutil
        import pybullet
        import scipy
        if Path(pybullet.__file__).resolve()!=project_path(diag["pybullet_extension"]):
            raise RuntimeError("Loaded engine path differs from bound isolated engine")
        require_identity(Path(pybullet.__file__),diag["bindings"][diag["pybullet_extension"]])
        if importlib.metadata.version("pybullet")!=diag["expected_distribution_version"]:
            raise RuntimeError("Loaded engine distribution version mismatch")
        if not Path(scipy.__file__).resolve().is_relative_to(project_path(diag["dependency_dir"])):
            raise RuntimeError("Loaded SciPy source is outside isolated dependencies")
        result["engine_provenance"]=dict(python=str(Path(sys.executable).resolve()),python_identity=identity(sys.executable),
            pybullet_extension=str(Path(pybullet.__file__).resolve()),pybullet_identity=identity(pybullet.__file__),
            distribution_version=importlib.metadata.version("pybullet"),scipy_path=str(Path(scipy.__file__).resolve()),scipy_version=scipy.__version__,
            dependency_manifest_identity=identity(project_path(diag["dependency_manifest"])))
        from pybullet_utils.bullet_client import BulletClient
        from scipy.spatial.transform import Rotation
        from scripts.sim_v52r2_episode import load_robot, make_table, tool_state, ExecutionRejected
        from scripts.sim_v52_robot_assets import command_gripper, tcp_offset_for_driver
        from scripts.sim_v52_math import angle_error
        from scripts.sim_v52d_telemetry import Recorder

        dt = unit["time_step_s"]
        scenario = unit["scenario"]
        resources = plan["resources"]
        manifest = read(ROOT / cfg["robot_manifest"])
        result["reused_source_identities"] = {
            path:identity(ROOT/path) for path in [
                "scripts/sim_v52r2_episode.py","scripts/sim_v52_robot_assets.py",
                "scripts/sim_v52_math.py","scripts/sim_v52d_telemetry.py"]}
        client = BulletClient(connection_mode=pybullet.DIRECT)
        client.setTimeStep(dt)
        client.setGravity(0,0,-9.81)
        client.setPhysicsEngineParameter(numSolverIterations=unit["solver_iterations"],deterministicOverlappingPairs=1)
        table = make_table(client,cfg)
        robot = load_robot(client,cfg,manifest,True)
        obj = None
        if scenario == "C":
            selected = next(row for row in plan["scenarios"] if row["id"] == "C")
            asset = cfg["objects"][selected["object_id"]]
            if selected["object_id"] != "021" or selected["scene_xy_m"] != [.48,.06] or selected["object_urdf"] != asset["urdf"]:
                raise ValueError("Only the frozen slot-4 ball contact scene is authorized")
            obj = client.loadURDF(asset["urdf"],basePosition=[*selected["scene_xy_m"],cfg["table_z"]+asset["height_m"]/2+.003],
                                  useFixedBase=False,flags=client.URDF_USE_INERTIA_FROM_FILE)
            client.changeDynamics(obj,-1,lateralFriction=cfg["physics"]["object_friction"],restitution=0.,linearDamping=.04,angularDamping=.04)
            if not math.isclose(client.getDynamicsInfo(obj,-1)[0],selected["object_mass_config_kg"],rel_tol=0.,abs_tol=1e-12):
                raise ValueError("Frozen contact object mass mismatch")
        result["objects_loaded"] = 1 if obj is not None else 0
        result["object_id"] = "021" if obj is not None else None
        if scenario not in ("N","L","C"):
            raise ValueError("Unexpected scenario")
        # The centers are from the frozen collision-box origin in each link frame.
        robot_xml = ET.parse(ROOT/cfg["robot_urdf"]).getroot()
        centers = {}
        for name in manifest["pad_links"]:
            link = next(x for x in robot_xml.findall("link") if x.get("name") == name)
            collisions = link.findall("collision")
            if len(collisions) != 1 or collisions[0].find("geometry/box") is None:
                raise ValueError("Pad must retain its single frozen collision box")
            origin_element = collisions[0].find("origin")
            xyz = np.fromstring(origin_element.get("xyz","0 0 0") if origin_element is not None else "0 0 0",sep=" ")
            centers[robot["links"][name]] = xyz
        master = robot["joints"][manifest["master_joint"]]
        targets = {
            "arm":{"joint_indices":list(robot["arm"]), "positions_rad":list(cfg["home_joints"]),
                   "forces_Nm":list(cfg["physics"]["arm_forces"]),
                   "position_gains":[.1]*6, "velocity_gains":[1.]*6,"max_velocity_rad_s":None},
            "gripper":{"master_joint_index":master,"position_rad":0.,
                       "force_Nm":cfg["physics"]["gripper_torque"],"position_gain":.1,"velocity_gain":1.,
                       "max_velocity_rad_s":.5},
            "followers":{"mode":"VELOCITY_CONTROL","force_Nm":0.},
        }
        setup=read_setup_robot_parameters(client,robot,read(project_path(diag["parameter_reference_readback"])))
        setup["reference_path"]=diag["parameter_reference_readback"]
        setup["reference_identity"]=identity(project_path(diag["parameter_reference_readback"]))
        write_once(directory/"setup_robot_parameter_readback.json",setup)
        result["setup_parameter_consistency_passed"]=setup["passed"]
        result["setup_parameter_readback_identity"]=identity(directory/"setup_robot_parameter_readback.json")
        if not setup["passed"]:
            raise RuntimeError("Full robot parameter readback differs from accepted v52e afterR; no physics permitted")
        assertion_stream=(directory/"pad_parameter_assertions.jsonl").open("x",encoding="utf8",newline="\n")
        recorder = Recorder(client,robot,table,obj,cfg,directory,scenario,manifest=manifest)
        hold_streak = bilateral_loss = 0
        initial_eval = None
        pad_z = float(tcp_offset_for_driver(action["estimated_contact_driver_rad"],manifest)[2])
        proc = psutil.Process()
        peak_rss = 0

        def limits():
            nonlocal peak_rss
            if time.monotonic()-PROCESS_STARTED > resources["per_run_wall_cap_s"]:
                raise TimeoutError("diagnostic_wall_limit")
            if result["actual_physics_steps"] >= unit["maximum_step_count"]:
                raise RuntimeError("diagnostic_step_limit")
            if result["actual_physics_steps"] % 120 == 0:
                # The controller additionally monitors tree RSS every .25 s.
                # Avoid Windows process enumeration on every physical tick.
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_rss = max(peak_rss,rss)
                if rss > resources["tree_RSS_cap_bytes"]:
                    raise MemoryError("diagnostic_RSS_limit")
                used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                if used > resources["output_cap_bytes"]:
                    raise RuntimeError("diagnostic_output_limit")
            result["peak_tree_rss_bytes"] = peak_rss
            result["worker_resource_poll_interval_steps"] = 120

        def known_loads():
            if scenario != "L" or phase not in ("close","lift","hold"):
                return []
            rows = []
            for index in robot["pads"]:
                state = client.getLinkState(robot["body"],index,computeForwardKinematics=True,computeLinkVelocity=True)
                center = np.asarray(state[4])+Rotation.from_quat(state[5]).as_matrix()@centers[index]
                rows.append((index,center,np.asarray(state[0])))
            vector = rows[1][1]-rows[0][1]
            length = float(np.linalg.norm(vector))
            if not math.isfinite(length) or length <= 1e-12:
                raise ValueError("Undefined outward-pad load direction")
            direction = vector/length
            loads = []
            for sign,(index,point,com) in zip((-1.,1.),rows):
                force = sign*direction  # exactly 1 N per pad; no added torque actuation
                loads.append(dict(link_index=index, force_world_N=force.tolist(),
                                  point_world_m=point.tolist(), direction_world=(sign*direction).tolist(),
                                  com_world_m=com.tolist(), torque_about_com_Nm=np.cross(point-com,force).tolist(),
                                  collision_origin_link_m=centers[index].tolist(),
                                  load_type="calibration_outward_1N_per_pad",
                                  physics_frame="WORLD_FRAME"))
            return loads

        def original_evaluation():
            nonlocal hold_streak,bilateral_loss
            driver = client.getJointState(robot["body"],master)[0]
            error = max(abs(client.getJointState(robot["body"],robot["joints"][row["joint"]])[0]-
                            (driver*row["multiplier"]+row["offset"])) for row in manifest["mimic_joints"])
            result["max_mimic_error_rad"] = max(result.get("max_mimic_error_rad",0.),error)
            result["actual_tcp_shift_from_nominal_max_m"] = max(
                result.get("actual_tcp_shift_from_nominal_max_m",0.),
                abs(float(tcp_offset_for_driver(driver,manifest)[2])-pad_z))
            pads = set()
            if obj is not None:
                for contact in client.getContactPoints(bodyA=robot["body"],bodyB=obj):
                    active = contact[9]>cfg["criteria"]["contact_force_n"] or contact[8]<-cfg["criteria"]["penetration_m"]
                    if contact[3] in robot["pads"]:
                        if contact[9]>cfg["criteria"]["contact_force_n"]:
                            pads.add(contact[3])
                    elif active:
                        collision_events.append(dict(step=result["actual_physics_steps"],phase=phase,kind="nonpad_object",
                                                     link=contact[3],force=contact[9],distance=contact[8]))
            for contact in client.getContactPoints(bodyA=robot["body"],bodyB=table):
                if contact[3] not in cfg["base_table_allowed_link_indices"] and (
                    contact[9]>cfg["criteria"]["contact_force_n"] or contact[8]<-cfg["criteria"]["penetration_m"]):
                    collision_events.append(dict(step=result["actual_physics_steps"],phase=phase,kind="robot_table",
                                                 link=contact[3],force=contact[9],distance=contact[8]))
            for contact in client.getContactPoints(bodyA=robot["body"],bodyB=robot["body"]):
                if contact[9]>cfg["criteria"]["contact_force_n"] or contact[8]<-cfg["criteria"]["penetration_m"]:
                    collision_events.append(dict(step=result["actual_physics_steps"],phase=phase,kind="robot_self",
                                                 links=[contact[3],contact[4]],force=contact[9],distance=contact[8]))
            if collision_events:
                result["forbidden_collision"] = True
            if obj is None:
                return
            position = np.asarray(client.getBasePositionAndOrientation(obj)[0])
            delta = float(position[2]-initial_eval[2])
            bilateral = len(pads)==2
            support = any(c[9]>cfg["criteria"]["contact_force_n"] for c in client.getContactPoints(bodyA=obj,bodyB=table))
            if phase in ("lift","hold") and bilateral and not support and delta>=cfg["criteria"]["grasp_formed_lift_m"]:
                result["grasp_formed"] = True
            bilateral_loss = 0 if bilateral else bilateral_loss+1
            ee_pos,ee_rot = tool_state(client,robot)
            pad_center = ee_pos+ee_rot@tcp_offset_for_driver(driver,manifest)
            separation = float(np.linalg.norm(position-pad_center))
            if result["grasp_formed"] and bilateral_loss*dt>=cfg["criteria"]["slip_loss_seconds"] and (
                delta<cfg["criteria"]["grasp_formed_lift_m"] or separation>cfg["criteria"]["slip_separation_m"]):
                result["slipped"] = True
            valid_hold = phase=="hold" and delta>=cfg["criteria"]["lift_m"] and not support and not result["forbidden_collision"] and result["grasp_formed"] and bilateral and not result["slipped"]
            hold_streak = hold_streak+1 if valid_hold else 0
            result["final_continuous_hold_seconds"] = hold_streak*dt
            result["final_bilateral_contact"] = bilateral
            result["max_lift_m"] = max(result.get("max_lift_m",0.),delta)
            result["max_continuous_hold_seconds"] = max(result.get("max_continuous_hold_seconds",0.),hold_streak*dt)

        def tick():
            limits()
            step = result["actual_physics_steps"]+1
            assertion=read_pad_parameter_assertion(client,robot,manifest,step,phase,result["actual_physics_steps"])
            assertion_stream.write(json.dumps(assertion,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
            assertion_stream.flush()
            # Keep every full pre-step readback; bound storage sync overhead only.
            # Failed assertions are durable before raising; finally also fsyncs.
            if step%120==0 or not assertion["passed"]:
                os.fsync(assertion_stream.fileno())
            result["pad_assertions_attempted"]+=1
            if not assertion["passed"]:
                result["last_failed_pad_assertion"]=assertion
                raise RuntimeError("Pre-step actual pad parameter assertion failed; no current or subsequent physics step")
            result["pad_assertions_passed"]+=1
            loads = known_loads()
            recorder.pre_step(step,phase,targets,loads)
            for load in loads:
                client.applyExternalForce(robot["body"],load["link_index"],load["force_world_N"],
                                          load["point_world_m"],client.WORLD_FRAME)
            result["physics_step_calls_started"] += 1
            result["unknown_step_call"] = dict(step=step,phase=phase,reason="started_without_return")
            client.stepSimulation()
            result["physics_step_calls_returned"] += 1
            result["unknown_step_call"] = None
            result["actual_physics_steps"] = step
            recorder.capture(step,phase,targets,loads)
            # Preserve the old scoring window while the audit telemetry also covers settle.
            if phase != "settle":
                original_evaluation()

        def move(target,seconds):
            start = np.array([client.getJointState(robot["body"],j)[0] for j in robot["arm"]])
            steps = round(seconds/dt)
            for k in range(steps):
                alpha = (k+1)/steps
                alpha = alpha*alpha*(3-2*alpha)
                q = (1-alpha)*start+alpha*np.array(target)
                client.setJointMotorControlArray(robot["body"],robot["arm"],client.POSITION_CONTROL,targetPositions=q.tolist(),
                    forces=cfg["physics"]["arm_forces"],positionGains=[.08]*6,velocityGains=[1.]*6)
                targets["arm"]["positions_rad"] = q.tolist()
                targets["arm"]["position_gains"] = [.08]*6
                targets["arm"]["velocity_gains"] = [1.]*6
                tick()
            for _ in range(round(.3/dt)):
                tick()

        phase = "settle"
        for _ in range(round(1./dt)):
            tick()
        if obj is not None:
            initial_eval = np.asarray(client.getBasePositionAndOrientation(obj)[0])
            result["evaluation_initial_position"] = initial_eval.tolist()
        else:
            result["evaluation_initial_position"] = None
        phase = "opening"
        command_gripper(client,robot["body"],action["open_driver_rad"],force=cfg["physics"]["gripper_torque"],manifest=manifest)
        targets["gripper"]["position_rad"] = action["open_driver_rad"]
        for _ in range(round(.8/dt)):
            tick()
        paths = action["planning_joint_targets"]
        phase = "transit"; move(paths[0]["joints"],cfg["motion"]["transit_seconds"])
        phase = "pregrasp"; move(paths[1]["joints"],cfg["motion"]["pregrasp_seconds"])
        phase = "approach"; move(paths[2]["joints"],cfg["motion"]["approach_seconds"])
        goal = np.asarray(action["goal_tool_world"])
        actual_pos,actual_rot = tool_state(client,robot)
        approach_error = dict(position_m=float(np.linalg.norm(actual_pos-goal[:3,3])),angle_rad=angle_error(actual_rot,goal[:3,:3]))
        result["executed_approach_error"] = approach_error
        if approach_error["position_m"]>cfg["criteria"]["ik_position_m"] or approach_error["angle_rad"]>cfg["criteria"]["ik_angle_rad"]:
            raise ExecutionRejected("executed_pose_tracking_failed")
        phase = "close"
        command_gripper(client,robot["body"],manifest["master_range_rad"][1],force=cfg["physics"]["gripper_torque"],velocity=.5,manifest=manifest)
        targets["gripper"]["position_rad"] = manifest["master_range_rad"][1]
        for _ in range(round(cfg["motion"]["close_seconds"]/dt)):
            tick()
        phase = "lift"; move(paths[3]["joints"],cfg["motion"]["lift_seconds"])
        phase = "hold"
        for _ in range(round(cfg["motion"]["hold_seconds"]/dt)):
            tick()
        result["dynamic_coupling_passed"] = result.get("max_mimic_error_rad",1e10)<=cfg["criteria"]["mimic_error_rad"]
        if obj is not None:
            passed = result.get("final_continuous_hold_seconds",0.)>=cfg["criteria"]["hold_seconds"] and result.get("final_bilateral_contact",False) and not result["slipped"] and not result["forbidden_collision"] and result["dynamic_coupling_passed"]
            result["grasp_success"] = result["original_grasp_success"] = bool(passed)
            outcome = "success" if passed else ("forbidden_collision" if result["forbidden_collision"] else ("slip" if result["slipped"] else "grasp_or_lift_failed"))
            if not result["dynamic_coupling_passed"]:
                outcome = "dynamic_coupling_failed"
            result["original_outcome"] = outcome
        else:
            result["original_outcome"] = None
        if result["actual_physics_steps"] != result["expected_physics_steps"]:
            raise RuntimeError("Incomplete physical schedule")
        result["status"] = "completed"
        result["completion"] = True
    except BaseException as exc:
        result["status"] = "failed"
        result["completion"] = False
        result["exception"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["failure_phase"] = phase
    finally:
        if assertion_stream is not None:
            try:
                assertion_stream.flush();os.fsync(assertion_stream.fileno());assertion_stream.close()
                result["pad_assertions_identity"]=identity(directory/"pad_parameter_assertions.jsonl")
            except BaseException as exc:
                cleanup_errors.append("pad_assertions_finalize:"+repr(exc));result["status"]="failed";result["completion"]=False
        if recorder is not None:
            try:
                summary = recorder.finalize(status=result["status"])
                result["summary"] = summary
                result["all_tick_diagnostics"] = dict(summary)
                result["all_tick_mimic_within_003"] = summary.get("max_mimic_error_rad",float("inf")) <= cfg["criteria"]["mimic_error_rad"]
                result["gate_window_note"] = "Original grasp gates use post_settle as frozen r2; all_tick_diagnostics also includes settling and does not overwrite original outcome. N/L grasp success is not applicable."
                count = summary.get("sample_count",summary.get("records"))
                result["telemetry_complete"] = bool(summary.get("telemetry_complete",False) and count==result["actual_physics_steps"])
                result["pad_parameter_assertions_complete"]=bool(result["pad_assertions_attempted"]==result["pad_assertions_passed"]==result["actual_physics_steps"] and result["last_failed_pad_assertion"] is None)
                result["parameter_assertions_passed"]=bool(result["pad_parameter_assertions_complete"] and result.get("setup_parameter_consistency_passed",False) and result["actual_physics_steps"]==result["expected_physics_steps"])
                if result["status"]=="completed" and not result["pad_parameter_assertions_complete"]:
                    result["status"]="failed";result["completion"]=False;result["exception"]="Pre-step parameter assertion coverage mismatch"
                if result["status"]=="completed" and not result["telemetry_complete"]:
                    result["status"]="failed"; result["completion"]=False
                    result["exception"]="Telemetry completeness mismatch"
            except BaseException as exc:
                cleanup_errors.append("telemetry_finalize:"+repr(exc))
                result["status"]="failed"; result["completion"]=False
                result["telemetry_complete"]=False
        if client is not None:
            try:
                client.disconnect()
            except BaseException as exc:
                cleanup_errors.append("disconnect:"+repr(exc))
                result["status"]="failed"; result["completion"]=False
        result["wall_seconds"] = time.monotonic()-PROCESS_STARTED
        result["simulation_seconds"] = result["actual_physics_steps"]*unit["time_step_s"]
        if result["wall_seconds"] > plan["resources"]["per_run_wall_cap_s"]:
            result["status"]="failed"; result["completion"]=False
            if result["exception"] is None:
                result["exception"]="Total worker wall limit including setup/finalization"
        result["collision_events_count"] = len(collision_events)
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors and result["exception"] is None:
            result["exception"] = "; ".join(cleanup_errors)
        write_once(directory/"collision_events.json",collision_events)
        write_once(directory/"result.json",result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit",required=True)
    parser.add_argument("--attempt-dir",required=True)
    parser.add_argument("--config",required=True)
    args = parser.parse_args()
    launch = validate_launch(args)
    result = run_unit(*launch)
    print(json.dumps({key:result.get(key) for key in ["unit_id","status","actual_physics_steps","expected_physics_steps",
                                                    "telemetry_complete","grasp_success","exception","wall_seconds"]}))
    return 0 if result["status"]=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
