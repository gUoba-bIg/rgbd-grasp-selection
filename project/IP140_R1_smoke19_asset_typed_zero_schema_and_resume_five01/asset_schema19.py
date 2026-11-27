"""Pre-registered typed zero-receipt verifier. No engine imports, writes or physical calls."""
from collections import Counter
from pathlib import Path
import base64
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DEFINITION=HERE/"asset_typed_schema.json"
GEOM_ATOL=1e-10

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition, label):
    if not condition:raise ValueError("asset_typed_zero_schema: "+label)

def registered_mode(urdf_relative, urdf_sha256):
    a=read(DEFINITION)["assets"].get(str(urdf_relative))
    require(a is not None and a["urdf_sha256"]==urdf_sha256,"unregistered asset identity")
    return a["mode"]

def registration(registration_path, slot):
    defs=read(DEFINITION)
    binding=defs["registrations"].get(str(slot))
    require(binding is not None and binding["reg_sha256"]==sha(registration_path),"registration hash")
    slots=[s for s in read(registration_path)["slots"] if s["slot"]==slot]
    require(len(slots)==1,"unique slot identity")
    s=slots[0];scene=s["scene_setup_only"];u=scene['urdf']
    require((s["candidate_id"],s["object_id"],u["project_relative"])==(binding["candidate_id"],binding["object_id"],binding["urdf_relative"]),"candidate scene ownership")
    a=defs["assets"][binding["urdf_relative"]]
    require(a["urdf_sha256"]==u["sha256"]==sha(ROOT/a["urdf_relative"]),"URDF content")
    return defs,binding,a,scene

def decode(x):
    if isinstance(x,dict) and "bytes_b64" in x:
        return base64.b64decode(x["bytes_b64"],validate=True).decode("utf8","backslashreplace")
    if isinstance(x,list):return [decode(y) for y in x]
    return x

def native_prefix(output_dir, zero, asset):
    trace=zero["zero_api_call_prefix"];length=trace["bytes"]
    require(type(length) is int and 0<length<=128*1024*1024,"native prefix size")
    with (Path(output_dir)/"zero_step/calls.jsonl").open("rb") as f:packed=f.read(length)
    require(len(packed)==length and hashlib.sha256(packed).hexdigest()==trace["sha256"],"wrapper zero prefix hash")
    if "native_api_zero_prefix" in zero:
        native_stamp=zero["native_api_zero_prefix"]
        nbytes=native_stamp["bytes"]
        require(type(nbytes) is int and 0<nbytes<=128*1024*1024,"native gzip prefix size")
        with (Path(output_dir)/"full_api/calls.jsonl.gz").open("rb") as f:
            compressed=f.read(nbytes)
        require(len(compressed)==nbytes and hashlib.sha256(compressed).hexdigest()==native_stamp["sha256"],"native gzip prefix hash")
    count=sum(zero["complete_api_counts_at_zero_handoff"].values())
    require(type(count) is int and 0<count<=10000,"bounded native prefix count")
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
    raw=bytearray()
    with (Path(output_dir)/"full_api/calls.jsonl.gz").open("rb") as src:
        while raw.count(10)<2*count:
            chunk=src.read(32768)
            require(bool(chunk),"incomplete flushed native prefix")
            raw.extend(decoder.decompress(chunk))
            require(len(raw)<=128*1024*1024,"native prefix size bound")
    lines=raw.split(b"\n",2*count)[:2*count]
    obj=zero["object_body_id"];client=zero["actual_client"]
    native={}
    methods={"getCollisionShapeData","getAABB","getDynamicsInfo","getMeshData"}
    for i in range(0,len(lines),2):
        call=json.loads(lines[i]);ret=json.loads(lines[i+1]);number=i//2+1
        require(call[:2]==[number,"call"] and ret[:2]==[number,"return"],"native sequence pair")
        method,args,kw=call[3:];require(method!="stepSimulation","zero prefix physical step")
        if (method in methods and args==[obj,-1]) or (method=="loadURDF" and args==[asset["urdf_relative"]]):
            require(method not in native and kw["physicsClientId"]==client,"duplicate or wrong owner: "+method)
            native[method]=dict(seq=number,args=args,kwargs=kw,**{"return":decode(ret[3])})
    return native

def load_receipt(output_dir,registration_path,slot):
    _,b,asset,_=registration(registration_path,slot)
    output_dir=Path(output_dir)
    z=read(output_dir/"zero_readback.json")
    rd=read(output_dir/"zero_step"/("object"+b["object_id"]+"_readback.json"))
    return z,rd,native_prefix(output_dir,z,asset)

def exact_zero_value(expected,observed,atol):
    if isinstance(expected,list):
        return isinstance(observed,list) and len(expected)==len(observed) and all(exact_zero_value(x,y,atol) for x,y in zip(expected,observed))
    if type(expected) in (bool,str):
        return type(observed) is type(expected) and expected==observed
    return type(expected) in (int,float) and type(observed) in (int,float) and math.isfinite(expected) and math.isfinite(observed) and abs(expected-observed)<=atol

def named_fields(defs,slot,obj,mode):
    def concretize(k):
        if k=="target.collision_margin_m" and mode!="compound_boxes": k="target.mesh_margin_m"
        return k.replace("target.","object"+obj+".").replace("slot.",f"slot{slot:02d}.")
    expected={concretize(k):v for k,v in defs["common_required"].items()}
    kind="compound" if mode=="compound_boxes" else "mesh"
    for n in defs["branches"][mode]:expected["object"+obj+"."+n]=defs["field_atol_source"][kind][n]
    return expected

def field_format(value):
    if isinstance(value,list):return [field_format(each) for each in value]
    if type(value) in (float,int):return "number"
    if type(value) is str:return "text"
    if type(value) is bool:return "bool"
    raise ValueError("Invalid zero-readback field shape/type")

def verify_fields(rows,expected,formats):
    require(isinstance(rows,list) and all(isinstance(r,dict) and set(r)=={"field","expected","readback","status","atol"} for r in rows),"row syntax")
    counts=Counter(row['field'] for row in rows)
    require(set(counts)==set(expected),"missing="+str(sorted(set(expected)-set(counts)))+" unexpected="+str(sorted(set(counts)-set(expected))))
    for r in rows:
        k=r["field"];tol=r["atol"]
        require(counts[k]==1 and r["status"]=="pass","duplicate or status: "+k)
        require(type(tol) in (float,int) and math.isfinite(tol) and 0<=tol==expected[k],"atol: "+k)
        require(field_format(r["expected"])==formats[k] and field_format(r["readback"])==formats[k],"field type or shape: "+k)
        require(exact_zero_value(r["expected"],r["readback"],tol),"invalid expected/readback: "+k)

def near(actual,expected,tol,label):
    x=np.asarray(actual,dtype=float);y=np.asarray(expected,dtype=float)
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.abs(x-y)<=tol),label)

def validate_receipt(output_dir,registration_path,slot,*,zero=None,readback=None,native=None):
    defs,b,asset,scene=registration(registration_path,slot)
    z=zero if zero is not None else read(Path(output_dir)/"zero_readback.json")
    require(z["preregistration_sha256"]==b["reg_sha256"],"receipt registration SHA")
    if b["source"]!="historical_regression":
        require((z.get("slot"),z.get("object_id"),z.get("candidate_id"))==(slot,b["object_id"],b["candidate_id"]),"receipt slot candidate object")
    else:
        require(z.get("static_candidate_slot")==slot and z.get("candidate_id",b["candidate_id"])==b["candidate_id"],"bound historical regression identity")
    mode=asset["mode"];names=named_fields(defs,slot,b["object_id"],mode)
    formats=named_fields({**defs,"common_required":defs["common_value_format"],"field_atol_source":defs["specific_value_format"]},slot,b["object_id"],mode)
    verify_fields(z["fields"],names,formats)
    rd=readback if readback is not None else read(Path(output_dir)/"zero_step"/("object"+b["object_id"]+"_readback.json"))
    nt=native if native is not None else native_prefix(output_dir,z,asset)
    methods={"loadURDF","getCollisionShapeData","getAABB","getDynamicsInfo"}
    if mode!="compound_boxes":methods.add("getMeshData")
    require(set(nt)==methods,"native method identities")
    obj=z["object_body_id"];client=z["actual_client"]
    require(all(type(x) is int and x>=0 for x in (obj,client,z["robot_body_id"],z["table_body_id"])) and len({obj,z["robot_body_id"],z["table_body_id"]})==3,"body & client identities")
    require((rd["body_id"],rd["base_link_id"],rd["physicsClientId"])==(obj,-1,client),"readback target identity")
    load=nt["loadURDF"]
    require(load["return"]==obj and load["args"]==[asset["urdf_relative"]] and load["kwargs"]==dict(basePosition=scene["position_world_m"],baseOrientation=scene["quaternion_xyzw"],useFixedBase=False,flags=2,physicsClientId=client),"loadURDF ownership and frozen arguments")
    seq=[nt[m]["seq"] for m in methods];require(all(type(x) is int and x>0 for x in seq) and len(set(seq))==len(seq) and load["seq"]==min(seq),"query sequence identity")
    for name,key in (("getCollisionShapeData","collision_shapes"),("getAABB","actual_AABB"),("getDynamicsInfo","dynamics"),("getMeshData","mesh")):
        if name not in methods:continue
        n=nt[name];require(n["args"]==[obj,-1] and n["kwargs"]=={"physicsClientId":client} and n["return"]==rd[key],"native return and ownership: "+name)
    urdf=ROOT/asset["urdf_relative"];xml=ET.parse(urdf);parts=xml.findall("./link/collision");rows=rd["collision_shapes"]
    require(len(parts)==asset["primitive_count"]==len(rows) and all(len(r)==7 for r in rows) and rows[0][:2]==[obj,-1],"primitive count/first owner")
    pos=scene["position_world_m"];near(rd["base_pose"][0],pos,GEOM_ATOL,"initial translation")
    near(rd["base_pose"][1],scene["quaternion_xyzw"],GEOM_ATOL,"initial rotation")
    near(scene["quaternion_xyzw"],[0,0,0,1],0,"frozen initial orientation")
    if mode=="compound_boxes":
        lows=[];highs=[]
        for part,row in zip(parts,rows):
            box=part.find("geometry/box");origin=part.find("origin")
            require(box is not None and row[2]==3 and origin.get("rpy")=="0 0 0","registered box type or orientation")
            xyz=np.asarray(list(map(float,origin.get("xyz").split())))
            dims=np.asarray(list(map(float,box.get("size").split())))
            near(row[3],dims,GEOM_ATOL,"ordered box dimensions")
            near(row[5],xyz,GEOM_ATOL,"ordered box local translation")
            near(row[6],[0,0,0,1],GEOM_ATOL,"ordered box quaternion")
            lows.append(xyz-dims/2);highs.append(xyz+dims/2)
        low=np.min(lows,axis=0);high=np.max(highs,axis=0)
    else:
        part=parts[0];mesh=part.find("geometry/mesh");row=rows[0];meshfile=ROOT/asset["obj_relative"]
        require(mesh is not None and (urdf.parent/mesh.get("filename")).resolve()==meshfile.resolve() and sha(meshfile)==asset["obj_sha256"],"registered OBJ path/hash")
        require(row[:3]==[obj,-1,5],"GEOM_MESH identity")
        if mode=="named_mesh":
            require(isinstance(row[4],str) and bool(row[4]) and row[4]!="unknown_file" and meshfile.resolve() in {(ROOT/row[4]).resolve(),(urdf.parent/row[4]).resolve()},"named mesh requires exact OBJ URI")
        elif mode=="anonymous_mesh":
            require(row[4] in ("","unknown_file"),"anonymous mesh cannot return named URI")
        else:raise ValueError("Unregistered mesh branch")
        near(row[3],list(map(float,mesh.get("scale").split())),GEOM_ATOL,"mesh scale")
        near(row[5],list(map(float,part.find("origin").get("xyz").split())),GEOM_ATOL,"mesh origin")
        near(row[6],[0,0,0,1],GEOM_ATOL,"mesh quaternion")
        proof=mesh_proof(rd["mesh"],meshfile)
        saved=rd["mesh_proof"]
        require(proof["passed"] and saved["passed"] is True and saved["source"]==proof["source"]
                and saved["raw_collision_mesh"]==rd["mesh"] and saved["source_sha256"]==proof["source_sha256"]==asset["obj_sha256"],"native hull, stored proof, and registered OBJ")
        hull=np.asarray(rd["mesh"][1]);low=hull.min(0);high=hull.max(0)
    expansion=asset["aabb_expansion_m"]
    near(rd["actual_AABB"],[low+pos-expansion,high+pos+expansion],GEOM_ATOL,"registered world AABB")
    return dict(passed=True,slot=slot,candidate_id=b["candidate_id"],object_id=b["object_id"],asset_type=mode,
                urdf_relative=asset["urdf_relative"],urdf_sha256=asset["urdf_sha256"],obj_sha256=asset.get("obj_sha256"),
                expected_named_fields=sorted(names),common_count=len(defs["common_required"]),asset_specific_count=len(defs["branches"][mode]),
                native_sequences={k:nt[k]["seq"] for k in sorted(nt)},native_client_id=client,body_id=obj,link_index=-1,
                primitive_count=len(rows),schema_sha256=sha(DEFINITION),geometric_tolerance_m=GEOM_ATOL)

def mesh_proof(rows, filename):
    """Verify the engine's convex hull in its local frame, using the importer's grid."""
    texts = filename.read_text(encoding="ascii").splitlines()
    source = np.array([list(map(float, l.split()[1:4])) for l in texts if l.startswith("v ")])
    indices = np.array([[int(i.split("/")[0])-1 for i in l.split()[1:4]]
                        for l in texts if l.startswith("f ")], dtype=int)
    count, vertices = rows[:2]
    live = np.asarray(vertices, dtype=float)
    if count < 4 or live.shape != (count, 3) or not np.isfinite(live).all():
        return dict(passed=False, reason="invalid engine hull vertices", raw=rows)
    # BulletUrdfImporter.cpp createConvexHullFromShapes -> optimizeConvexHull;
    # btConvexHullComputer.cpp scales each extent by 1/10216 and truncates toward zero.
    # OBJ coordinates first pass through tinyobj's float storage.
    cell = np.ptp(source.astype(np.float32).astype(float), axis=0) / 10216.
    parse_eps = np.spacing(np.float32(np.max(np.abs(source)))).item()
    error = cell + parse_eps
    distance_in_cells = np.max(np.abs(live[:, None, :] - source[None, :, :]) / error, axis=2)
    normal = np.cross(source[indices[:, 1]]-source[indices[:, 0]],
                      source[indices[:, 2]]-source[indices[:, 0]])
    nondegenerate = np.linalg.norm(normal, axis=1) > 0
    normal = normal[nondegenerate] / np.linalg.norm(normal[nondegenerate], axis=1)[:, None]
    axes = np.vstack([normal, -normal, np.eye(3), -np.eye(3)])
    support_error = np.abs((source @ axes.T).max(axis=0) - (live @ axes.T).max(axis=0))
    normalized = support_error / (np.abs(axes) @ error)
    max_vertex_error = float(np.max(np.min(distance_in_cells,axis=1)))
    max_support_error = float(np.max(normalized))
    return dict(passed=max_vertex_error <= 1 and max_support_error <= 1,
                source="collision.obj", source_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                source_vertices=len(source), observed_hull_vertices=count, raw_collision_mesh=rows,
                importer_grid_m=cell, float_parse_bound_m=parse_eps,
                max_live_vertex_to_source_normalized=max_vertex_error,
                max_support_error_normalized=max_support_error,
                support_directions_tested=len(axes), support_directions=axes, support_abs_errors_m=support_error)
