"""One exclusive zero-integration gate with same-client handoff. Sets only historical arm home coordinates; no physics step."""
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import sys
import traceback
import zlib

from full_api import ApiTrace
from structural_filter import STRUCTURAL_PAIRS, INTERNAL_SEAMS, apply_and_verify, audit_seams
from mountpair import fixed_mount, audit_mount
from seam_geometry import current_frames

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "zero_step"
SRC = Path("runtime/simulation_v52e/src/pybullet-3.2.7")
PROOFS = {
    "shape_query": ("examples/SharedMemory/PhysicsServerCommandProcessor.cpp", "14327,14360-14382"),
    "shape_import": ("examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp","1528"),
    "python_fields": ("examples/pybullet/pybullet.c", "1573-1653, 5208-5306, 5339-5444, 12961-12963"),
    "live_motor_setters_and_damping_scope": ("examples/SharedMemory/PhysicsServerCommandProcessor.cpp", "7305-7354, 10711-10717, 11257-11258, 11310-11311"),
    "motor_data": ("src/BulletDynamics/Featherstone/btMultiBodyJointMotor.h","27-32, 46-58"),
    "model_export": ("src/BulletDynamics/Featherstone/btMultiBody.cpp","120-121, 2295-2383"),
    "world_export": ("src/BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.cpp","834-875"),
    "parent_flags": ("examples/Importers/ImportURDFDemo/URDF2Bullet.cpp","718-728")
}

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def js(v):
    if isinstance(v, bytes):
        return v.decode("utf8", errors="backslashreplace")
    if hasattr(v, "tolist"):
        return v.tolist()
    raise TypeError(type(v).__name__)

def write(p,v):
    with p.open("x",encoding="utf8") as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False,default=js)
        f.flush()
        os.fsync(f.fileno())

def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def image_png(path, rgba):
    h,w,_=rgba.shape
    def chunk(t,b):
        return struct.pack(">I",len(b))+t+b+struct.pack(">I",zlib.crc32(t+b)&0xffffffff)
    pixels=b"".join(b"\0"+r[:,:3].tobytes() for r in rgba)
    content=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">2I5B",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(pixels))+chunk(b"IEND",b"")
    with path.open("xb") as f:
        f.write(content)

DEFERRED = {
    "live_motor.mode": "No getter. Preserve command and observe trajectory.",
    "live_motor.Kp_Kd": "No live gains getter. Preserve setter and track errors.",
    "live_motor.reference": "Command logged, actual q/dq independently observed.",
    "live_motor.maxVelocity_m_s": "URDF metadata only. Check actual speed every step.",
    "live_motor.per_finger_max_force_N": "Motor clamp not exposed. Check reported force every step.",
    "effective_collision_pair_policy": "Effective filter cannot be completely queried. Retain flags, observed pair identities and witnesses.",
}
WATCHDOG = {
    "filtered_structural_pairs": "All nine exact filtered pairs remain absent from the solver manifold on each step.",
    "arm_home_q_dq": "Each arm joint: absolute home error <=0.010 rad and speed <=0.010 rad/s at every step.",
    "tcp_pose_drift": "Observed TCP G drift from zero pose <=0.005 m and <=3 degrees; retain full poses.",
    "arm_motor_saturation": "Read all six arm motor torques; stop on saturation against the submitted finite ceilings.",
    "dynamic_slide_limit_band": "Retain actual q; stop outside [-1e-5,0.07001] m. Command limits remain [0,0.070] m.",
    "initial_contact_impulse": "First actual integrated step, residual impulse/contact must be examined.",
    "q0_pad_pad_contact": "Opposing pads at the closed endpoint: keep raw witnesses; only finite registered tangent-band, zero-force/zero-impulse rows may qualify under the active-seam thresholds.",
    "palm_finger_root_contact": "Observe local root-seam contact, penetration and abnormal impulse.",
    "fixed_pad_mounts": "Audit fixed left/right pad mounts before filtering, preserve closest geometry and require empty solver manifolds after filtering.",
    "registered_active_seams": "A manifold on one of five active internal pairs requires signed distance, face witnesses, unit normal, force and impulse checks.",
}
DEFERRED_IDS=tuple(sorted(DEFERRED))
WATCHDOG_IDS=tuple(sorted(WATCHDOG))
APPROVED_SHAPES=("ip140_palm","ip140_left_bone","ip140_left_pad","ip140_right_bone","ip140_right_pad")


def gate_valid(gate):
    return (gate.get("passed") is True and gate.get("status")=="passed_with_deferred_runtime_checks"
        and gate.get("version")==HERE.name and gate.get("structural_filter_passed") is True
        and gate.get("structural_filter_setter_count")==9 and gate.get("wrist_static_hard_gates_passed") is True
        and gate.get("all_nine_post_filter_closest_and_empty_manifolds") is True and gate.get("post_filter_seams_passed") is True
        and gate.get("required_mismatches")==[] and gate.get("required_complete") is True
        and gate.get("deferred_no_getter")==list(DEFERRED_IDS)
        and gate.get("first_step_watchdogs")==list(WATCHDOG_IDS)
        and gate.get("actual_physics_steps")==0 and gate.get("physics_step_calls")==0)


def run_zero(on_pass=None):
    if Path.cwd().resolve()!=ROOT:
        raise RuntimeError("Explicit project root workdir required")
    cfg=load(ROOT/"configs/simulation_v52e.json")
    if Path(sys.executable).resolve()!=(ROOT/cfg["python"]).resolve():
        raise SystemExit("Expected isolated v52e1 interpreter")
    mpath=HERE/"model/manifest.json"
    mf=load(mpath)
    model=mpath.parent/mf["model"]
    spec=HERE/"preregistration.json"
    model_arg = model.relative_to(ROOT).as_posix()
    if not model_arg.isascii() or Path(model_arg).is_absolute():
        raise ValueError("Model entry point must be project-relative ASCII")
    assert sha(model)==mf["model_sha256"]
    assert set(mf["components"])==set(APPROVED_SHAPES) and len(mf["joints"])==5
    assert sorted(mf["required_distinct_pad_links"])==["ip140_left_pad","ip140_right_pad"]
    registration=load(spec)
    assert registration["max_starts"]==12 and len(registration["slots"])==12
    for name,expected in {**mf["provenance"],**mf["reused_arm_asset_sha256"]}.items():
        assert sha(ROOT/name)==expected, name
    import numpy as np
    import pybullet as p
    binary=Path(p.__file__).resolve()
    version=importlib.metadata.version("pybullet")
    assert version=="3.2.7+v52e1" and binary==(ROOT/cfg["pybullet_extension"]).resolve()
    assert sha(binary)==cfg["bindings"][cfg["pybullet_extension"]]["sha256"]

    OUT.mkdir(exist_ok=False)
    trace=ApiTrace(p,HERE/"full_api")
    p=trace.api
    start=utc()
    write(OUT/"start.json", {"utc":start,"pid":os.getpid(),"max_connections":1,"max_steps":0})
    result=dict(version=HERE.name,started_at=start,passed=False,status="unknown",actual_physics_steps=0,physics_step_calls=0,
                connection_starts=0,connection_returns=0,disconnected=False,preregistration_sha256=sha(spec),
                model_sha256=sha(model),manifest_sha256=sha(mpath),worker_sha256=sha(Path(__file__)),
                engine={"version":version,"path":binary.relative_to(ROOT).as_posix(),"sha256":sha(binary)},
                fields=[],required_mismatches=[],required_complete=False,sources={},cwd=str(Path.cwd()),
                model_arg=model_arg,physics_step_call_count_scope="zero readback phase only",
                deferred_no_getter=list(DEFERRED_IDS),first_step_watchdogs=list(WATCHDOG_IDS),
                deferred_evidence_plan=DEFERRED,watchdog_checks=WATCHDOG,backend_limits={
                    "rigid_linear_and_angular_damping_per_new_link":{
                       "actual_readback":None,"per_link_setter_applied":False,
                       "reason":"Rigid linear/angular damping live at multibody scope. No global change."}},
                execution_hashes={f.name:sha(f) for f in sorted(HERE.glob("*.py"))},
                collision_transform_policy="W_T_link @ link_T_inertial @ inertial_T_shape; cross-check against W_T_COM @ inertial_T_shape",
                pure_matrix_zero_tolerance_m=1e-9,registered_internal_seam_numeric_zero_m=1e-9,
                wrist_static_hard_gates_passed=False,all_nine_post_filter_closest_and_empty_manifolds=False,
                structural_filter_passed=False,structural_filter_setter_count=0,post_filter_seams_passed=False,
                load_calls_started=0,load_calls_returned=0)
    for name,(rel,lines) in PROOFS.items():
        path=SRC/rel
        result["sources"][name]={"path":path.as_posix(),"lines":lines,"sha256":sha(ROOT/path)}
    client=None
    serial=0
    events=(OUT/"calls.jsonl").open("x",encoding="utf8",buffering=1)
    def event(**data):
        events.write(json.dumps(dict(utc=utc(),**data),ensure_ascii=False,default=js)+"\n")
        events.flush()
    def call(f,*args,**kw):
        nonlocal serial
        serial+=1
        key=serial
        event(seq=key,kind="start",call=f.__name__,args=args,kwargs=kw)
        try:
            val=f(*args,**kw)
        except BaseException as exc:
            event(seq=key,kind="raised",call=f.__name__,exception=repr(exc))
            raise
        event(seq=key,kind="returned",call=f.__name__)
        return val
    def check(name,expected,actual,atol=1.e-10):
        same=bool(np.allclose(expected,actual,atol=atol,rtol=0)) if not isinstance(expected,(str,bool)) else expected==actual
        result["fields"].append({"field":name,"expected":expected,"readback":actual,"status":"pass" if same else "mismatch","atol":atol})
        if not same: result["required_mismatches"].append(name)
    def fail(name,why):
        result["fields"].append(dict(field=name,status="hard_failure",reason=why))
        result["required_mismatches"].append(name)
    def frame(pos,quat):
        a=np.eye(4);a[:3,:3]=np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3);a[:3,3]=pos
        return a
    try:
        result["connection_starts"]=1
        client=call(p.connect,p.DIRECT)
        if client<0: raise RuntimeError("DIRECT connection failed")
        result["connection_returns"]=1
        flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION|p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        result["load_flags"]={"mask":flags,"self_collision":True,"include_parent":True,"merge_fixed_links":False,"source":"submitted load flags; not a filter-state getter"}
        call(p.setPhysicsEngineParameter,fixedTimeStep=1/240,numSolverIterations=150,physicsClientId=client)
        call(p.setGravity,0,0,-9.81,physicsClientId=client)
        result["load_calls_started"]=1
        body=call(p.loadURDF,model_arg,useFixedBase=True,flags=flags,physicsClientId=client)
        result["load_calls_returned"]=1
        result["body_id"]=body
        if body<0: raise ValueError("Missing engine body ID")
        def read(f,*args,**kwargs):
            return call(f,*args,**kwargs,physicsClientId=client)
        cnt=read(p.getNumJoints,body)
        ji={idx:read(p.getJointInfo,body,idx) for idx in range(cnt)}
        link={row[12].decode():idx for idx,row in ji.items()}
        if not set(mf["components"]).issubset(link):
            raise RuntimeError("Missing independent model links: "+str(set(mf["components"])-set(link)))
        hist_config=ROOT/"configs/simulation_v52r2.json"
        hc=load(hist_config)
        arm_manifest=ROOT/hc["robot_manifest"]
        arm_names=load(arm_manifest)["arm_joint_names"]
        home=hc["home_joints"]
        by_joint={v[1].decode():i for i,v in ji.items()}
        if len(arm_names)!=6 or len(home)!=6:
            raise ValueError("Six historic arm home targets required")
        for name,q in zip(arm_names,home):
            idx=by_joint[name]
            if ji[idx][2]!=p.JOINT_REVOLUTE:
                raise ValueError("Unexpected home joint type")
            read(p.resetJointState,body,idx,float(q),targetVelocity=0.)
        result["arm_home"]={"joint_names":arm_names,"positions":home,"config_sha256":sha(hist_config),"arm_manifest_sha256":sha(arm_manifest)}
        # Set only new collision links, never multibody-global arm damping.
        for name,expected in mf["components"].items():
            kw={k:expected[k] for k in ("lateralFriction","restitution","spinningFriction","rollingFriction")}
            kw["collisionMargin"]=expected["collisionMargin_m"]
            read(p.changeDynamics,body,link[name],**kw)
        for side in ("left","right"):
            idx=link["ip140_"+side+"_bone"]
            read(p.setJointMotorControl2,body,idx,p.POSITION_CONTROL,targetPosition=0.,targetVelocity=0.,
                 positionGain=.10,velocityGain=1.,force=20.,maxVelocity=.040)
        raw={}
        for idx in range(-1,cnt):
            row=ji.get(idx)
            name=row[12].decode() if row is not None else "BASE"
            raw[name]={"index":idx,"joint_info":row,"dynamics":read(p.getDynamicsInfo,body,idx),
                       "collision":read(p.getCollisionShapeData,body,idx),
                       "state":read(p.getJointState,body,idx) if idx>=0 else None,
                       "pose":read(p.getLinkState,body,idx,computeForwardKinematics=1) if idx>=0 else read(p.getBasePositionAndOrientation,body)}
        write(OUT/"all_links_readback.json",raw)
        # Prefer double precision COM poses composed with inverse local inertial frames.
        def wf(name):
            a=raw[name]["pose"];return frame(a[0],a[1])@np.linalg.inv(frame(a[2],a[3]))
        G=wf(mf["G_link"])
        Ginv=np.linalg.inv(G)
        fg=np.linalg.inv(wf(mf["flange_link"]))@G
        check("flange_T_G",mf["F_T_G"],fg,1e-9)
        check("separate_pad_and_bone_link_IDs",4,len({link["ip140_"+side+"_"+part] for side in ("left","right") for part in ("pad","bone")}))
        check("observed_gear_or_loop_constraints",0,read(p.getNumConstraints))
        check("arm_home_positions",home,[read(p.getJointState,body,by_joint[n])[0] for n in arm_names])
        geom={}
        frame_diag={}
        for name,t in mf["components"].items():
            r=raw[name];v=r["dynamics"];sh=r["collision"]
            for key,j in [("mass_kg",0),("lateralFriction",1),("inertia_at_com_kg_m2",2),("center_in_link_m",3),
                          ("orientation_xyzw",4),("restitution",5),("rollingFriction",6),("spinningFriction",7),("collisionMargin_m",11)]:
                check(name+"."+key,t[key],v[j])
            check(name+".collision_count",1,len(sh))
            if len(sh)!=1: continue
            first=sh[0]
            check(name+".box.type",p.GEOM_BOX,first[2])
            check(name+".box.size",t["box_size_m"],first[3],1.e-9)
            check(name+".shape_body_id",body,first[0])
            # For compound children the tuple's linkIndex field is examined below;
            # the enclosing lookup remains explicitly keyed by link index.
            check(name+".shape_link_index",link[name],first[1])
            # The shape-query tuple is relative to the collider's inertial frame.
            # BulletUrdfImporter.cpp:1528, PhysicsServerCommandProcessor.cpp:14327,14360-14382.
            inertial_T_shape=frame(first[5], first[6])
            link_T_inertial=frame(v[3], v[4])
            local=link_T_inertial @ inertial_T_shape
            wt=wf(name) @ local
            direct_world=frame(r["pose"][0],r["pose"][1]) @ inertial_T_shape
            check(name+".box.world_route_equivalence",direct_world,wt,1.e-9)
            check(name+".box.link_origin",t["center_in_link_m"],local[:3,3],1e-9)
            check(name+".box.link_rotation",np.eye(3),local[:3,:3],1e-9)
            check(name+".box.G_origin",t["com_at_zero_in_G_m"],(Ginv@wt)[:3,3],1e-9)
            geom[name]=(Ginv@wt,np.asarray(first[3]))
            frame_diag[name]=dict(inertial_T_shape=inertial_T_shape,link_T_inertial=link_T_inertial,
                expected_link_offset_m=t["center_in_link_m"],link_T_shape=local,
                W_T_link=wf(name),W_T_shape=wt,composed_from_world_COM=direct_world)
        write(OUT/"collision_frame_diagnostics.json",dict(convention=result["collision_transform_policy"],
              units_translation="m",components=frame_diag))
        for jname,t in mf["joints"].items():
            idx=link[t["child"]];row=ji[idx]
            check(jname+".name",jname,row[1].decode())
            check(jname+".parent",link[t["parent"]],row[16])
            check(jname+".type",p.JOINT_FIXED if t["type"]=="fixed" else p.JOINT_PRISMATIC,row[2])
            if t["type"]=="prismatic":
                for key,x,y in [("opening_axis",t["axis_in_joint_frame"],row[13]),("limits_m",t["limits_m"],row[8:10]),
                               ("q_loaded_zero",0,raw[t["child"]]["state"][0]),("URDF_damping",1.,row[6]),
                               ("URDF_dry_friction",0.,row[7]),("URDF_effort_rating",20.,row[10]),("URDF_speed_rating",.04,row[11])]:
                    check(jname+"."+key,x,y)
        # Measured inner-face gaps at four corners. Do not round a negative bound into a pass.
        faces=[]
        for side,sign in [("left",1),("right",-1)]:
            mat,size=geom["ip140_"+side+"_pad"]
            faces.append(np.array([(mat@np.array([x*size[0]/2,sign*size[1]/2,z*size[2]/2,1]))[:3]
                                   for x in (-1,1) for z in (-1,1)]))
        gaps=faces[1][:,1]-faces[0][:,1]
        result["opening_q0"]={"corners_left_G_m":faces[0],"corners_right_G_m":faces[1],"signed_gaps_m":gaps,
                              "raw_net_m":gaps.mean(),"midline_m":(faces[0][:,1].mean()+faces[1][:,1].mean())/2,
                              "raw_min_m":min(gaps),"qualification":"floating-point transformed witnesses; comparison not clipped"}
        check("zero_width_within_1mm",True,bool(abs(gaps.mean())<=.001))
        check("zero_midline_within_0.5mm",True,bool(abs(result["opening_q0"]["midline_m"])<=.0005))
        check("zero_parallel_spread_within_0.5mm",True,bool(np.ptp(gaps)<=.0005))
        check("zero_min_corner_geometric_floating_bound",True,bool(min(gaps)>=-1e-9))
        check("zero_single_face_target_0.5mm",True,bool(max(abs(face[:,1].mean()) for face in faces)<=.0005))
        check("zero_symmetry_1mm",True,bool(abs(faces[0][:,1].mean()+faces[1][:,1].mean())<=.001))
        distal_midpoint=np.mean([face[2:,:].mean(axis=0) for face in faces],axis=0)
        check("G_T_tip_midline_origin_from_distal_faces_m",[0,0,0],distal_midpoint,1e-9)
        # Five active tangent seams remain monitored; two fixed pad/bone installations have separate proof.
        read(p.performCollisionDetection)
        pre_manifolds=read(p.getContactPoints)
        # No movement or filtering is allowed between these manifolds and their frame.
        topology=fixed_mount(p,body,link,client)
        face=audit_mount(p,body,link,client,manifolds=pre_manifolds)
        write(OUT/"pre_filter_wrist_fixed_chain.json",topology)
        transform=dict(stage="pre_filter_zero_step",actual_T_wrist_G=topology["actual_T_wrist_G"],
                       rotation_max_error=topology["rotation_max_error"],translation_max_error_m=topology["translation_max_error_m"],
                       passed=topology["relative_transform_passed"],registration_sha256=topology["registration_sha256"])
        write(OUT/"pre_filter_wrist_relative_transform.json",transform)
        write(OUT/"pre_filter_wrist_mount_face_witnesses.json",face)
        check("wrist3_flange_palm.fixed_chain_topology",True,topology["fixed_chain_passed"])
        check("wrist3_palm.zero_relative_transform",True,topology["relative_transform_passed"])
        check("wrist3_palm.mounting_face_witnesses_pre_filter",True,face["passed"])
        check("wrist3_palm.fixed_chain_transform_and_collision_margins",True,topology["passed"])
        result["wrist_static_hard_gates_passed"]=bool(topology["passed"] and face["passed"])
        pre_frames=current_frames(p,body,link,client)
        pre_seams=audit_seams(p,body,link,client,pre_manifolds,require_pad_witness=True,
                               manifold_frames=pre_frames,current_frames_w=pre_frames,dt=1/240)
        write(OUT/"seams_before_filter.json",pre_seams)
        check("five_active_seams.pre_signed_distance_face_force_and_impulse",True,pre_seams["passed"])
        structural_ids={frozenset((link[a],link[b])) for a,b in STRUCTURAL_PAIRS}
        active_ids={frozenset((link[a],link[b])) for a,b in INTERNAL_SEAMS}
        check("distinct_nine_structural_pairs",9,len(structural_ids))
        check("distinct_five_active_pairs",5,len(active_ids))
        check("active_seams_not_filtered",True,not bool(active_ids & structural_ids))
        def check_manifolds(rows,phase,allowed_self):
            for n,row in enumerate(rows):
                pair=frozenset((row[3],row[4]))
                tag=f"{phase}_manifold_{n}:{row[1]},{row[3]}:{row[2]},{row[4]}"
                if not all(math.isfinite(row[j]) for j in (8,9,10,12)):
                    fail(tag,"Nonfinite distance or contact force in engine manifold.")
                elif row[1]==body and row[2]==body:
                    if pair not in allowed_self:
                        fail(tag,"Self-manifold outside the allowed structural set for this phase.")
                elif row[9]>.2 or row[8]<-.001:
                    fail(tag,"External contact exceeds the unchanged 0.2 N or 1 mm threshold.")
        # The seam auditor above is a separate hard gate for every row on an active pair.
        check_manifolds(pre_manifolds,"pre_filter",structural_ids | active_ids)
        # Round trip actual registered six world-frame candidates through measured mount.
        roundtrips=[]
        for slot in registration["slots"]:
            if slot["slot"]<7:
                continue
            row=slot.get("candidate_world",slot.get("candidate_row_world"))
            vrow=np.asarray(row,dtype=float).reshape(17)
            T=np.eye(4);T[:3,:3]=vrow[4:13].reshape(3,3);T[:3,3]=vrow[13:16]+vrow[3]*T[:3,0]
            F=T@np.linalg.inv(fg);back=F@fg
            recovered=back[:3,3]-vrow[3]*back[:3,0]
            roundtrips.append({"candidate":slot.get("candidate_id"),"W_T_F_goal":F,"W_T_G":T,"roundtrip_W_T_G":back,
                               "max_origin_roundtrip_error_m":float(np.max(np.abs(recovered-vrow[13:16]))),
                               "max_rotation_error":float(np.max(np.abs(back[:3,:3]-T[:3,:3])))})
        write(OUT/"candidate_roundtrip.json",roundtrips)
        check("registered_pose_roundtrips_count",6,len(roundtrips))
        if roundtrips:
            check("registered_roundtrip_residual",True,all(v["max_origin_roundtrip_error_m"]<=1e-9 and v["max_rotation_error"]<=1e-9 for v in roundtrips))
        try:
            look=G[:3,3].tolist()
            view=p.computeViewMatrixFromYawPitchRoll(look,.35,46,-25,0,2)
            proj=p.computeProjectionMatrixFOV(47,1.4,.01,3)
            im=read(p.getCameraImage,560,400,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
            image_png(OUT/"zero_assembly.png",np.asarray(im[2],dtype=np.uint8).reshape(400,560,4))
        except Exception as exc:
            result["optional_image_unavailable"]=repr(exc)
        result["physics_readback"]=read(p.getPhysicsEngineParameters)
        check("fixed_substep_s",1/240,result["physics_readback"]["fixedTimeStep"],1e-12)
        check("solver_iterations",150,result["physics_readback"]["numSolverIterations"])
        # Mutate collision routing only after all other observable zero gates pass.
        if not result["required_mismatches"]:
            filtered=apply_and_verify(p,body,link,client,persist=lambda name,data:write(OUT/name,data))
            write(OUT/"structural_filter_readback.json",filtered)
            result["structural_filter_passed"]=filtered["passed"]
            result["structural_filter_setter_count"]=len(filtered["setter_calls"])
            check("structural_filter.exact_list_normal_returns_and_post_manifolds",True,filtered["passed"])
            check("structural_filter.setter_count",9,result["structural_filter_setter_count"])
            result["structural_filter_violations"]=filtered["violations"]
            if filtered["post"] is not None:
                post_by_pair=filtered["post"]["pairs"]
                nine_detail={}
                for pair in STRUCTURAL_PAIRS:
                    name="|".join(pair)
                    indexes=(link[pair[0]],link[pair[1]])
                    record=post_by_pair[name]
                    near=record["closest"]
                    clean=(bool(near) and not record["manifolds"] and all(
                        row[1]==body and row[2]==body and (row[3],row[4]) in (indexes,indexes[::-1])
                        and all(math.isfinite(x) for x in (*row[5],*row[6],*row[7],row[8],row[9],row[10],row[12])) for row in near))
                    nine_detail[name]=dict(passed=clean,link_indices=indexes,raw_closest=near,raw_manifolds=record["manifolds"])
                    check("post_filter.closest_present_and_manifold_absent."+name,True,clean)
                result["all_nine_post_filter_closest_and_empty_manifolds"]=all(x["passed"] for x in nine_detail.values()) and len(nine_detail)==9
                write(OUT/"post_filter_nine_pairs_closest_and_manifolds.json",nine_detail)
                for phase in ("pre","post"):
                    for side in ("left","right"):
                        check(f"{phase}.fixed_pad_mount.{side}",True,filtered[phase]["pad_mounts"][side]["passed"])
                post=filtered["post"]["raw_manifolds"]
                check_manifolds(post,"post_filter",active_ids)
                post_frames=current_frames(p,body,link,client)
                post_seams=audit_seams(p,body,link,client,post,require_pad_witness=True,
                                       manifold_frames=post_frames,current_frames_w=post_frames,dt=1/240)
                write(OUT/"seams_after_filter.json",post_seams)
                result["post_filter_seams_passed"]=post_seams["passed"]
                check("five_active_seams.post_signed_distance_face_force_and_impulse",True,post_seams["passed"])
                write(OUT/"zero_contacts.json",dict(physics_steps=0,pre_filter_manifolds=pre_manifolds,
                      post_filter_manifolds=post,post_filter_closest_witnesses=post_seams["pairs"],
                      impulse_scope="No integration in the zero phase. First-step solver impulses are independently monitored."))
        counts=dict(trace.counts)
        result["complete_api_counts_at_zero_handoff"]=counts
        result["physics_step_calls"]=counts.get("stepSimulation",0)
        check("actual_zero_step_calls_at_handoff",0,result["physics_step_calls"])
        check("actual_connect_calls_at_handoff",1,counts.get("connect",0))
        check("actual_load_calls_at_handoff",1,counts.get("loadURDF",0))
        actual_masks={name:int(n) for name,n in counts.items() if "collisionfilter" in name.lower() and name.startswith("set")}
        result["observed_collision_filter_api_counts"]=actual_masks
        check("only_nine_filter_writes_in_full_API_history",True,actual_masks=={"setCollisionFilterPair":9})
        result["other_collision_monitoring_evidence"]={
            "load_flags":result["load_flags"],"exact_pair_record":"zero_step/structural_filter_readback.json",
            "registered_gripper_seams_pre":"zero_step/seams_before_filter.json",
            "post_seams_observed":result["post_filter_seams_passed"],
            "full_api_filter_setter_counts":actual_masks,
            "physical_scope":"Only nine named structural pairs changed. No filter-state getter; pre/post detection and each-step raw contact monitoring supply observational evidence. No external object is present during this zero-step model check."}
        result["required_complete"]=not result["required_mismatches"] and result["structural_filter_passed"] and result["post_filter_seams_passed"]
    except BaseException as exc:
        fail("incomplete_required_readback",repr(exc))
        result["exception"]=repr(exc)
        result["traceback"]=traceback.format_exc()

    # Freeze the gate before its result is used to charge or begin physical movement.
    try:
        result["passed"]=result["required_complete"] and not result["required_mismatches"]
        result["status"]="passed_with_deferred_runtime_checks" if result["passed"] else "blocked_required_mismatch"
        result["zero_sample_finished_at"]=utc()
        result["artifacts"]={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.is_file() and f.name!="calls.jsonl"}
        result["zero_api_call_prefix"]={"bytes":(OUT/"calls.jsonl").stat().st_size,"sha256":sha(OUT/"calls.jsonl")}
        write(HERE/"zero_readback.json",result)
        if gate_valid(result) and on_pass is not None:
            on_pass(p,client,body,link,mf,registration,load(HERE/"zero_readback.json"))
    finally:
        if client is not None and client>=0:
            try:
                call(p.disconnect,physicsClientId=client)
                result["disconnected"]=True
            except Exception as exc:
                result["disconnect_exception"]=repr(exc)
        events.close()
        trace.close()
        result["ended_at"]=utc()
        result["artifacts"]={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.is_file()}
        write(OUT/"result.json",result)
    return result

if __name__=="__main__":
    if sys.argv[1:]!=["--one-zero-readback"]:
        raise SystemExit("Use authorized integrated runner; standalone accepts --one-zero-readback only.")
    result=run_zero()
    print(json.dumps({k:result[k] for k in ("status","passed","actual_physics_steps","required_mismatches")}))
    sys.exit(0 if result["passed"] else 3)
