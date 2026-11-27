"""Single zero-integration readback. Does not contain a step, reset, retry or smoke launcher."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "zero_step"
SRC = Path("runtime/simulation_v52e/src/pybullet-3.2.7")
PROOFS = {
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

def main():
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=["--one-zero-readback"]:
        raise SystemExit("Run once from the research root with --one-zero-readback")
    cfg=load(ROOT/"configs/simulation_v52e.json")
    if Path(sys.executable).resolve()!=(ROOT/cfg["python"]).resolve():
        raise SystemExit("Expected isolated v52e1 interpreter")
    mpath=HERE/"model/manifest.json"
    mf=load(mpath)
    model=mpath.parent/mf["model"]
    spec=HERE/"preregistration.json"
    assert sha(model)==mf["model_sha256"]
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
    start=utc()
    write(OUT/"start.json", {"utc":start,"pid":os.getpid(),"max_connections":1,"max_steps":0})
    result=dict(started_at=start,passed=False,status="unknown",actual_physics_steps=0,physics_step_calls=0,
                connection_starts=0,connection_returns=0,disconnected=False,preregistration_sha256=sha(spec),
                model_sha256=sha(model),manifest_sha256=sha(mpath),worker_sha256=sha(Path(__file__)),
                engine={"version":version,"path":binary.relative_to(ROOT).as_posix(),"sha256":sha(binary)},
                fields=[],unverified=[],mismatches=[],sources={})
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
        val=f(*args,**kw)
        event(seq=key,kind="returned",call=f.__name__)
        return val
    def check(name,expected,actual,atol=1.e-10):
        same=bool(np.allclose(expected,actual,atol=atol,rtol=0)) if not isinstance(expected,(str,bool)) else expected==actual
        result["fields"].append({"field":name,"expected":expected,"readback":actual,"status":"pass" if same else "mismatch","atol":atol})
        if not same: result["mismatches"].append(name)
    def unknown(name,expected,why):
        result["fields"].append(dict(field=name,expected=expected,readback=None,status="not_verified",reason=why))
        result["unverified"].append(name)
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
        body=call(p.loadURDF,str(model),useFixedBase=True,flags=flags,physicsClientId=client)
        def read(f,*args,**kwargs):
            return call(f,*args,**kwargs,physicsClientId=client)
        cnt=read(p.getNumJoints,body)
        ji={idx:read(p.getJointInfo,body,idx) for idx in range(cnt)}
        link={row[12].decode():idx for idx,row in ji.items()}
        if not set(mf["components"]).issubset(link):
            raise RuntimeError("Missing independent model links: "+str(set(mf["components"])-set(link)))
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
        check("G_T_tip_midline_origin_design",[0,0,0],(Ginv@G)[:3,3])
        geom={}
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
            wt=frame(r["pose"][0],r["pose"][1])@frame(first[5],first[6])
            local=np.linalg.inv(wf(name))@wt
            check(name+".box.link_origin",t["center_in_link_m"],local[:3,3],1e-9)
            check(name+".box.link_rotation",np.eye(3),local[:3,:3],1e-9)
            check(name+".box.G_origin",t["com_at_zero_in_G_m"],(Ginv@wt)[:3,3],1e-9)
            geom[name]=(Ginv@wt,np.asarray(first[3]))
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
        if min(gaps)<0:
            unknown("zero_nonnegative_gap_geometric_certificate",">=0 at each corner","Unrounded transformed face gap <0. Distinguish numerical uncertainty from measurable penetration; see closest-point witnesses.")
        queries={}
        pairs=[("ip140_left_pad","ip140_right_pad")]+[("ip140_palm","ip140_"+s+"_"+part) for s in ("left","right") for part in ("pad","bone")]
        for a,b in pairs:
            pts=read(p.getClosestPoints,body,body,.002,linkIndexA=link[a],linkIndexB=link[b])
            key=a+"|"+b
            queries[key]=dict(raw_points=pts,min_signed_distance_m=min((v[8] for v in pts),default=None))
            if not pts:
                unknown("proximity."+key,"signed witness near assembly face","no witness within 2mm; not a measured zero")
            elif min(v[8] for v in pts)<0:
                result["mismatches"].append("negative_closest_point."+key)
        read(p.performCollisionDetection)
        ct=read(p.getContactPoints)
        write(OUT/"zero_contacts.json",dict(physics_steps=0,closest_points=queries,manifolds=ct,
              distance_is_signed=True,normal_force_note="no solver integration or physical impulse measurement; do not read zero-step force as loaded validation"))
        result["endpoint_min_signed_m"]=queries[pairs[0][0]+"|"+pairs[0][1]]["min_signed_distance_m"]
        result["seam_min_signed_m"]={k:v["min_signed_distance_m"] for k,v in list(queries.items())[1:]}
        for v in ct:
            if {v[3],v[4]} in [{link[a],link[b]} for a,b in pairs]:
                if v[9]>.2 or v[8]<0: result["mismatches"].append(f"endpoint_or_seam_manifold:{v[3]}:{v[4]}")
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
            roundtrips.append({"candidate":slot.get("candidate_id"),"flange_T_goal_world":F,
                               "max_origin_roundtrip_error_m":float(np.max(np.abs(recovered-vrow[13:16]))),
                               "max_rotation_error":float(np.max(np.abs(back[:3,:3]-T[:3,:3])))})
        write(OUT/"candidate_roundtrip.json",roundtrips)
        check("registered_pose_roundtrips_count",6,len(roundtrips))
        if roundtrips:
            check("registered_roundtrip_residual",True,all(v["max_origin_roundtrip_error_m"]<1e-8 and v["max_rotation_error"]<1e-8 for v in roundtrips))
        try:
            look=G[:3,3].tolist()
            view=p.computeViewMatrixFromYawPitchRoll(look,.35,46,-25,0,2)
            proj=p.computeProjectionMatrixFOV(47,1.4,.01,3)
            im=read(p.getCameraImage,560,400,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
            image_png(OUT/"zero_assembly.png",np.asarray(im[2],dtype=np.uint8).reshape(400,560,4))
        except Exception as exc:
            result["optional_image_unavailable"]=repr(exc)
        for field,expected,why in [
            ("live_motor.mode","POSITION_CONTROL","Internal joint motor has no public getter for current command mode."),
            ("live_motor.Kp_Kd",[.10,1.],"Position/velocity feedback gains are write-only in this binding."),
            ("live_motor.reference",[0.,0.],"JointState q/dq are actual joint state, not active target q/dq."),
            ("live_motor.maxVelocity_m_s",.04,"URDF maxVelocity reads metadata; actual motor RHS speed clamp has no Python readback."),
            ("live_motor.per_finger_max_force_N",20.,"URDF effort rating differs from current motor impulse bound; no internal motor maximum impulse readback."),
            ("rigid_linear_and_angular_damping_per_new_link",[0.,0.],"C API exposes a whole-multibody value; Python tuple omits it. Per-new-link override would affect historical arm. Not applied."),
            ("effective_collision_pair_policy","registered face-level policy and actual enabled pairs","No public Python pair-mask getter. Closest-point queries do not prove the filter. Manifolds observe only current contacts."),
            ("initial_dynamics_impulse","no false impulse","Zero-step readback supplies no solved first-step impulse. This is a smoke-step watchdog if other evidence passes; not measured here.")]:
            if field=="initial_dynamics_impulse":
                result["deferred_first_step_watchdog"]=why
            else: unknown(field,expected,why)
        result["physics_readback"]=read(p.getPhysicsEngineParameters)
        result["status"]="blocked"
    except BaseException as exc:
        result["status"]="blocked_exception"
        result["exception"]=repr(exc)
        result["traceback"]=traceback.format_exc()
    finally:
        if client is not None and client>=0:
            try:
                call(p.disconnect,physicsClientId=client)
                result["disconnected"]=True
            except Exception as exc: result["disconnect_exception"]=repr(exc)
        events.close()
        result["ended_at"]=utc()
        result["artifacts"]={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.is_file()}
        write(OUT/"result.json",result)
    print(json.dumps({k:result[k] for k in ("status","passed","actual_physics_steps","mismatches","unverified")}))
    return 3

if __name__=="__main__":
    sys.exit(main())

