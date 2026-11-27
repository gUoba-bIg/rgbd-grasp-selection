"""Common IP140-R2 scene construction; no connection or motion on import.

Reads only engineering geometry/controller/asset definitions. The selection world contains
no target body. Object identity and state in full worlds belong to setup/evaluation, never the ranker.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from platform_imports import PLATFORM, ROOT, VERSION, bind_frozen_libraries

CONFIG = ROOT / "configs/simulation_v52r2.json"
REGISTRY = PLATFORM / "asset_typed_schema.json"
DESIGN = VERSION / "preregistration/paired_design.json"


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(v):
    if isinstance(v, bytes):
        return v.decode("utf8", errors="backslashreplace")
    if hasattr(v, "tolist"):
        return v.tolist()
    raise TypeError(type(v).__name__)


def write(path, payload):
    path=Path(path)
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open("x",encoding="utf8",newline="\n") as f:
        json.dump(payload,f,ensure_ascii=False,default=native,allow_nan=False,indent=2)
        f.write("\n")


def canonical(x):
    return json.dumps(x,default=native,sort_keys=True,allow_nan=False,separators=(",",":"))


from common import scene_binding as registered_scene, workspace_ok

def bound_damping(oid,model_path):
    # Keep each setup independent. Only the two identity bindings of this native-write validator
    # depend on the scene; all source/binary checks, writes and numeric targets are inherited.
    path=PLATFORM/"damping_write.py"
    spec=importlib.util.spec_from_file_location("r4_frozen_damping_"+oid,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OBJECT_ID=oid
    mod.OBJECT_URDF=model_path
    return mod


def setup_robot(p,client,evidence_dir):
    bind_frozen_libraries()
    from structural_filter import apply_and_verify
    from speed_policy import phase_speed
    if not workspace_ok() or p.counts["stepSimulation"]!=0:
        raise RuntimeError("new unstepped project-root client required")
    directory=Path(evidence_dir)
    directory.mkdir(exist_ok=True,parents=True)
    cfg=read(CONFIG)
    mf=read(PLATFORM/"model/manifest.json")
    model=PLATFORM/"model"/mf["model"]
    if sha(model)!=mf["model_sha256"] or any(sha(ROOT/f)!=dig for f,dig in mf["reused_arm_asset_sha256"].items()):
        raise ValueError("frozen robot asset hash mismatch")
    kw=dict(physicsClientId=client)
    p.setPhysicsEngineParameter(fixedTimeStep=cfg["physics"]["time_step"],numSolverIterations=cfg["physics"]["solver_iterations"],**kw)
    p.setGravity(0,0,-9.81,**kw)
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
    robot=p.loadURDF(model.relative_to(ROOT).as_posix(),useFixedBase=True,flags=flags,**kw)
    if robot<0:raise RuntimeError("robot load")
    cnt=p.getNumJoints(robot,**kw)
    ji=[p.getJointInfo(robot,i,**kw) for i in range(cnt)]
    links={row[12].decode():idx for idx,row in enumerate(ji)}
    joints={row[1].decode():idx for idx,row in enumerate(ji)}
    arm=[joints[n] for n in read(ROOT/cfg["robot_manifest"])["arm_joint_names"]]
    for idx,q in zip(arm,cfg["home_joints"],strict=True):
        if ji[idx][2]!=p.JOINT_REVOLUTE:raise ValueError("arm joint classification")
        p.resetJointState(robot,idx,float(q),targetVelocity=0.,**kw)
    for name,comp in mf["components"].items():
        data={k:comp[k] for k in ("lateralFriction","restitution","spinningFriction","rollingFriction")}
        p.changeDynamics(robot,links[name],**data,collisionMargin=comp["collisionMargin_m"],**kw)
    fingers=[links["ip140_"+side+"_bone"] for side in ("left","right")]
    for idx in fingers:
        p.setJointMotorControl2(robot,idx,p.POSITION_CONTROL,targetPosition=0.,targetVelocity=0.,
                                positionGain=.10,velocityGain=1.,force=20.,maxVelocity=phase_speed("settle"),**kw)
    raw={}
    for idx in range(-1,cnt):
        row=ji[idx] if idx>=0 else None
        name=row[12].decode() if row is not None else "BASE"
        raw[name]=dict(index=idx,joint_info=row,dynamics=p.getDynamicsInfo(robot,idx,**kw),
                collision=p.getCollisionShapeData(robot,idx,**kw),
                state=p.getJointState(robot,idx,**kw) if idx>=0 else None,
                pose=p.getLinkState(robot,idx,computeForwardKinematics=1,**kw) if idx>=0 else p.getBasePositionAndOrientation(robot,**kw))
    template=PLATFORM/"slots/10/zero_step/all_links_readback.json"
    write(directory/"robot_link_readback.json",raw)
    prior=read(template)
    errors=[name for name in sorted(set(raw)|set(prior)) if canonical(raw.get(name))!=canonical(prior.get(name))]
    eng=p.getPhysicsEngineParameters(**kw)
    if errors or eng["fixedTimeStep"]!=cfg["physics"]["time_step"] or eng["numSolverIterations"]!=cfg["physics"]["solver_iterations"]:
        raise ValueError(f"robot or engine readback differs from frozen setup: {errors}, {eng}")
    proof=apply_and_verify(p,robot,links,client,persist=lambda name,payload:write(directory/"zero_step"/name,payload))
    if not proof["passed"] or p.counts["setCollisionFilterPair"]!=13: raise ValueError(f"internal filter admission: {proof['violations']}")
    return SimpleNamespace(p=p,client=client,robot=robot,links=links,arm=arm,fingers=fingers,
        home=np.asarray(cfg["home_joints"]),arm_efforts=cfg["physics"]["arm_forces"],target=None,table=None,
        evidence_dir=directory,engine=eng,zero_step_identity=dict(frozen_robot_readback_sha256=sha(template),model_sha256=sha(model),
                                             registry_sha256=sha(REGISTRY),configuration_sha256=sha(CONFIG)))


def mount_table(ctx):
    from base_table_mount import apply_mount_filter
    p=ctx.p
    proof=apply_mount_filter(p,ctx.client,ctx.robot,ctx.table,ctx.links,persist=lambda name,payload:write(ctx.evidence_dir/"zero_step"/name,payload))
    if not proof["passed"] or p.counts["setCollisionFilterPair"]!=14 or p.returns["setCollisionFilterPair"]!=14 or p.counts["stepSimulation"]:
        raise ValueError(f"table mount or full filter count admission: {proof['violations']}")


def load_eligibility_world(p,client,evidence_dir):
    if p.connection_scope!="eligibility": raise ValueError("target-free eligibility connection required")
    ctx=setup_robot(p,client,evidence_dir)
    kw=dict(physicsClientId=client)
    col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.6,.6,.025],**kw)
    vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[.6,.6,.025],rgbaColor=[.65,.65,.65,1],**kw)
    ctx.table=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,basePosition=[.45,0,-.025],**kw)
    mount_table(ctx)
    if p.getNumBodies(**kw)!=2 or ctx.target is not None: raise RuntimeError("selection world must have exactly robot and fixed table")
    write(ctx.evidence_dir/"eligibility_setup.json",dict(**ctx.zero_step_identity,objects=0,body_count=2,zero_physics_steps=0,filter_returns=14))
    return ctx


def setup_world(p,client,scene_key,evidence_dir,*,settled_state=None):
    scene=registered_scene(scene_key)
    if p.connection_scope not in {"observation","dynamic","contact_aware_static"}:
        raise ValueError("unexpected scope for full-scene setup")
    if settled_state is not None and p.connection_scope!="contact_aware_static":
        raise ValueError("settled-pose object loading permitted only in static certification; dynamic methods repeat full initial settle")
    ctx=setup_robot(p,client,evidence_dir)
    from scene_asset import audit_initial_world, read_scene
    oid=ctx.object_id=scene["object_id"]
    ctx.object_urdf_path=ROOT/scene["asset"]["urdf_relative"]
    ctx.scene_key=scene["base_scene"]
    src=scene["asset"]
    oi=scene["spawn_position"],scene["spawn_quaternion"]
    if settled_state is not None:
        ev=read(settled_state) if isinstance(settled_state,(str,Path)) else settled_state
        if ev["scene_key"]!=ctx.scene_key or ev["object_id"]!=oid or ev["returned_settle_steps"]!=240:raise ValueError("wrong sealed initial state")
        oi=ev["object_pose"]
    target_binding=dict(urdf=dict(project_relative=src["urdf_relative"],sha256=src["urdf_sha256"]),
        position_world_m=oi[0],quaternion_xyzw=oi[1],lateral_friction=.6,restitution=0.,linear_damping=.04,angular_damping=.04,fixed_base=False)
    checks={}
    def check(name,expected,actual,atol=1e-10):
        if name in checks:raise ValueError("duplicate return field "+name)
        try:
            a,b=np.asarray(expected),np.asarray(actual)
            ok=(a.shape==b.shape and
                (np.isfinite(b).all() and np.allclose(a,b,rtol=0.,atol=atol) if a.dtype.kind in "biufc" and b.dtype.kind in "biufc" else canonical(expected)==canonical(actual)))
        except (ValueError,TypeError):
            ok=False
        checks[name]=dict(expected=expected,readback=actual,atol=atol,passed=bool(ok))
    ctx.target,ctx.table=read_scene(p,client,ctx.robot,dict(object_id=oid,scene_setup_only=target_binding),write,check,{},
                                    output_root=ctx.evidence_dir,damping=bound_damping(oid,src["urdf_relative"]))
    write(ctx.evidence_dir/"zero_step/asset_fields.json",checks)
    errors=[k for k,v in checks.items() if not v["passed"]]
    if errors:raise ValueError("asset zero readback: "+repr(errors))
    ctx.object_mass_kg=float(checks["object"+oid+".mass"]["readback"])
    mount_table(ctx)
    world=audit_initial_world(p,client,ctx.robot,ctx.target,ctx.table,ctx.links,oid)
    write(ctx.evidence_dir/"zero_step/initial_world.json",world)
    if not world["passed"] or p.exceptions or p.counts["stepSimulation"]:
        raise ValueError(f"initial external collision admission: {world['failures']}")
    write(ctx.evidence_dir/"setup.json",dict(scene=scene,scope=p.connection_scope,actual_load_pose=oi,
              **ctx.zero_step_identity,object_mass_kg=ctx.object_mass_kg,filtered=14,steps=0,asset_checks=len(checks),damping="audited_write_path"))
    return ctx


def settle_state(ctx):
    from structural_filter import current_frames
    p=ctx.p
    kw=dict(physicsClientId=ctx.client)
    returned=p.returns["stepSimulation"]
    if returned!=240 or p.counts["stepSimulation"]!=returned:raise ValueError("snapshot must be taken at exact tick 240")
    return json.loads(canonical(dict(scene_key=ctx.scene_key,object_id=ctx.object_id,returned_settle_steps=returned,
             object_pose=p.getBasePositionAndOrientation(ctx.target,**kw),object_velocity=p.getBaseVelocity(ctx.target,**kw),
             robot_joint_states=p.getJointStates(ctx.robot,ctx.arm+ctx.fingers,**kw),raw_manifolds=p.getContactPoints(**kw),
             link_frames=current_frames(p,ctx.robot,ctx.links,ctx.client))))


def confirm_same_settle(expected,actual):
    expected=read(expected) if isinstance(expected,(str,Path)) else expected
    actual=read(actual) if isinstance(actual,(str,Path)) else actual
    if canonical(expected)!=canonical(actual):
        keys=sorted(k for k in set(expected)|set(actual) if canonical(expected.get(k))!=canonical(actual.get(k)))
        raise ValueError("paired tick-240 state differs: "+repr(keys))
    return True


def render_observation(ctx):
    p=ctx.p
    if ctx.p.connection_scope!="observation" or p.returns["stepSimulation"]!=240:
        raise RuntimeError("render the registered settled full scene only")
    cam=read(CONFIG)["camera"]
    width,height=cam["width"],cam["height"]
    view=p.computeViewMatrix(cam["eye"],cam["target"],cam["up"])
    proj=p.computeProjectionMatrixFOV(cam["fov"],width/height,cam["near"],cam["far"])
    rw,rh,rgba,depth,_=p.getCameraImage(width,height,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER,
                                       flags=p.ER_NO_SEGMENTATION_MASK,physicsClientId=ctx.client)
    if (rw,rh)!=(width,height):raise RuntimeError("native rendering dimensions")
    # No fifth (segmentation) camera return is retained in the policy observation.
    return dict(rgb=np.asarray(rgba,dtype=np.uint8).reshape(height,width,4)[:,:,:3].copy(),
                depth_buffer=np.asarray(depth,dtype=np.float32).reshape(height,width),
                view=np.asarray(view,dtype=np.float64),projection=np.asarray(proj,dtype=np.float64))
