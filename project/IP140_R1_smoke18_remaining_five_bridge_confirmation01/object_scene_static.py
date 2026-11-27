"""Read-only scene observations, except for creation and the single audited object-base write."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
from batch_context import OBJECT_ID
import numpy as np
from damping_write import apply_damping_write, validate_damping_write
from object_evaluator import _contact_values
from structural_filter import STRUCTURAL_PAIRS
from base_table_mount import check_row

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


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


def read_scene(p, client, robot, slot, write, check, result, *, output_root=None):
    output_root=HERE if output_root is None else Path(output_root)
    source = slot["scene_setup_only"]
    rel = source["urdf"]["project_relative"]
    urdf=ROOT/rel
    if hashlib.sha256(urdf.read_bytes()).hexdigest()!=source["urdf"]["sha256"] or slot["object_id"]!=OBJECT_ID:
        raise ValueError("Unregistered target asset")
    xml=ET.parse(urdf).getroot()
    link=xml.find("link")
    collisions=link.findall("collision")
    is_mesh=OBJECT_ID=="021"
    meshfile=urdf.parent/"collision.obj"
    inertia=link.find("inertial/inertia")
    if (any(float(inertia.get(k))!=0 for k in ("ixy","ixz","iyz"))
            or link.find("inertial/origin").get("rpy")!="0 0 0"
            or link.find("inertial/origin").get("xyz")!="0 0 0"):
        raise ValueError("Registered principal axes do not match frozen zero-frame convention")
    if is_mesh:
        if len(collisions)!=1 or collisions[0].find("geometry/mesh").get("filename")!="collision.obj":
            raise ValueError("Frozen mesh geometry identity mismatch")
    elif not collisions or any(c.find("geometry/box") is None for c in collisions):
        raise ValueError("Frozen compound target requires all voxel boxes")
    tag="object"+OBJECT_ID
    kw = dict(physicsClientId=client)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6,.6,.025], **kw)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[.6,.6,.025], rgbaColor=[.65,.65,.65,1], **kw)
    table = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                              basePosition=[.45,0,-.025], **kw)
    obj = p.loadURDF(rel, basePosition=source["position_world_m"],
                     baseOrientation=source["quaternion_xyzw"], useFixedBase=False,
                     flags=p.URDF_USE_INERTIA_FROM_FILE, **kw)
    if obj < 0 or len({robot, table, obj}) != 3:
        raise ValueError("Scene IDs conflict or object load failed")
    result.update(object_body_id=obj, table_body_id=table, object_link_index=-1,
                  robot_body_id=robot, actual_client=client)
    p.protect_object(obj,client)
    receipt = apply_damping_write(p,obj,client)
    write(output_root/"damping_write_receipt.json",receipt)
    issues = validate_damping_write(receipt,obj,client,p.audit_rows)
    audit = dict(passed=not issues,violations=issues)
    write(output_root/"zero_step"/"damping_write_audit.json",audit)
    check("object_base_damping_audited_write",True,audit["passed"])
    def read(f,*args,**kwargs):
        return f(*args,**kwargs,**kw)
    rd = {"body_info":read(p.getBodyInfo,obj),"num_joints":read(p.getNumJoints,obj),
          "dynamics":read(p.getDynamicsInfo,obj,-1),"base_pose":read(p.getBasePositionAndOrientation,obj),
          "velocity":read(p.getBaseVelocity,obj),"collision_shapes":read(p.getCollisionShapeData,obj,-1),
          "actual_AABB":read(p.getAABB,obj,-1),
          "base_link_id":-1,"body_id":obj,"physicsClientId":client}
    table_rd = {"dynamics":read(p.getDynamicsInfo,table,-1),"pose":read(p.getBasePositionAndOrientation,table),
                "shapes":read(p.getCollisionShapeData,table,-1),"AABB":read(p.getAABB,table,-1),
                "body_id":table,"link_id":-1,"client_id":client}
    if is_mesh:
        rd["mesh"]=read(p.getMeshData,obj,-1)
        rd["mesh_proof"]=mesh_proof(rd["mesh"],meshfile)
    else:
        expected_boxes=[]
        for part in collisions:
            if part.find("origin").get("rpy")!="0 0 0":
                raise ValueError("Unexpected rotated registered voxel primitive")
            expected_boxes.append(dict(name=part.get("name"),dims=list(map(float,part.find("geometry/box").get("size").split())),
                pos=list(map(float,part.find("origin").get("xyz").split())),quat=[0.,0.,0.,1.]))
        # The pinned server fills body and link only for the first output entry of a queried compound
        # (processRequestCollisionShapeInfoCommand). Native child geometry is serialized in URDF order.
        # Preserve every header. For later rows use the single body/link request, sequence, type and
        # full slot-specific ordered geometry/poses to establish ownership.
        rows=rd["collision_shapes"]
        expected_count=len(expected_boxes)
        same_count=len(rows)==expected_count
        types=same_count and all(r[2]==p.GEOM_BOX for r in rows)
        identity=bool(rows and tuple(rows[0][:2])==(obj,-1) and types)
        def max_box_error(index,key):
            if not same_count:return None
            return float(np.max(np.abs(np.array([r[index] for r in rows])-np.array([e[key] for e in expected_boxes]))))
        errors={key:max_box_error(idx,key) for idx,key in ((3,"dims"),(5,"pos"),(6,"quat"))}
        matches={k:(v is not None and np.isfinite(v) and v<=1e-10) for k,v in errors.items()}
        if source["quaternion_xyzw"] != [0,0,0,1]:
            raise ValueError("Compound AABB formula requires registered identity orientation")
        centres=np.asarray([b["pos"] for b in expected_boxes])
        half_sizes=np.asarray([b["dims"] for b in expected_boxes])*.5
        # btCompoundShape::getAabb adds the compound's own margin to its children's union.
        goal_origin=np.asarray(source["position_world_m"])
        expected_world_aabb=np.asarray([(centres-half_sizes).min(axis=0)-.001,
                                       (centres+half_sizes).max(axis=0)+.001])+goal_origin
        aabb_error=float(np.max(np.abs(np.asarray(rd["actual_AABB"])-expected_world_aabb)))
        rd["compound_box_proof"]=dict(passed=bool(identity and all(matches.values()) and aabb_error<=1e-10),
            expected_world_AABB=expected_world_aabb.tolist(),max_abs_AABB_error_m=aabb_error,request_body=obj,request_link=-1,
            expected_count=expected_count,observed_count=len(rows),expected_boxes=expected_boxes,
            all_expected_box_types=bool(types),queried_target_identity_and_types=bool(identity),
            maximum_abs_errors=errors,geometry_tolerance_m=1e-10,geometry_matches=matches,
            identity_provenance="Native same-body/base-link request; first row header is populated by the frozen C++ server. All child geometries, poses, ordering and count verified against frozen URDF; all raw rows preserved.")
    write(output_root/"zero_step"/f"{tag}_readback.json",rd)
    write(output_root/"zero_step"/"table_readback.json",table_rd)
    check(f"{tag}.body_info_names",True,list(rd["body_info"])==[link.get("name").encode(),xml.get("name").encode()])
    check(f"{tag}.num_joints",0,rd["num_joints"])
    check(f"{tag}.initial_position",source["position_world_m"],rd["base_pose"][0])
    check(f"{tag}.initial_orientation",source["quaternion_xyzw"],rd["base_pose"][1])
    check(f"{tag}.initial_twist",np.zeros((2,3)),rd["velocity"])
    # Based on frozen URDF and actual historical loader plus BulletUrdfImporter default margin.
    for label, idx, expected in [
        ("mass",0,float(link.find("inertial/mass").get("value"))),("lateral_friction",1,.6),("principal_inertia",2,[float(inertia.get(axis)) for axis in ("ixx","iyy","izz")]),
        ("local_inertial_origin",3,[0,0,0]),("local_inertial_orientation",4,[0,0,0,1]),("restitution",5,0),
        ("rolling_friction",6,0),("spinning_friction",7,0),("collision_margin_m",11,.001)]:
        check(f"{tag}."+label,expected,rd["dynamics"][idx],1e-12)
    shapes=rd["collision_shapes"]
    if is_mesh:
        check(f"{tag}.one_collision_shape",1,len(shapes))
        if len(shapes)==1:
            row=shapes[0]
            check(f"{tag}.mesh_identity",[obj,-1,p.GEOM_MESH],row[:3])
            check(f"{tag}.mesh_scale",[1,1,1],row[3])
            check(f"{tag}.mesh_pose_xyz",[0,0,0],row[5])
            check(f"{tag}.mesh_pose_xyzw",[0,0,0,1],row[6])
            name=row[4].decode("utf-8","strict")
            if name != "unknown_file":
                path=Path(name)
                expected=meshfile.resolve()
                resolved={path.resolve(), (meshfile.parent/path).resolve()}
                check(f"{tag}.mesh_filename",True,expected in resolved)
        check(f"{tag}.collision_hull_geometry",True,rd["mesh_proof"]["passed"])
    else:
        proof=rd["compound_box_proof"]
        check(f"{tag}.compound_collision_shape_count",len(collisions),len(shapes))
        check(f"{tag}.compound_registered_identity_and_box_types",True,proof["queried_target_identity_and_types"])
        for k in ("dims","pos","quat"):
            check(f"{tag}.compound_all_{k}",True,proof["geometry_matches"][k])
        check(f"{tag}.compound_collision_geometry",True,proof["passed"])
    check("table.mass",0,table_rd["dynamics"][0])
    check("table.position",[.45,0,-.025],table_rd["pose"][0])
    check("table.orientation",[0,0,0,1],table_rd["pose"][1])
    check("table.single_collision_shape",1,len(table_rd["shapes"]))
    if len(table_rd["shapes"])==1:
        row=table_rd["shapes"][0]
        check("table.shape_identity",[table,-1,p.GEOM_BOX],row[:3])
        check("table.box_dimensions",[1.2,1.2,.05],row[3])
        check("table.shape_local_translation",[0,0,0],row[5])
        check("table.shape_local_quaternion",[0,0,0,1],row[6])
    actual_bodies={read(p.getBodyUniqueId,i) for i in range(read(p.getNumBodies))}
    check("registered_world_bodies_only",True,actual_bodies=={obj,robot,table})
    return obj,table


def audit_initial_world(p, client, robot, obj, table, links):
    kw=dict(physicsClientId=client)
    p.performCollisionDetection(**kw)
    manifolds=p.getContactPoints(**kw)
    by_name={v:k for k,v in links.items()}
    by_name[-1]="BASE"
    opposite=frozenset([links["ip140_left_pad"],links["ip140_right_pad"]])
    structural={frozenset(links[n] for n in pair):"|".join(pair) for pair in STRUCTURAL_PAIRS}
    counts=dict.fromkeys(structural.values(),0)
    table_support=[]
    failures=[]
    def contact_label(row):
        return [by_name.get(i) if b==robot else (f"object{OBJECT_ID}" if b==obj else "table" if b==table else "unknown")
                for b,i in ((row[1],row[3]),(row[2],row[4]))]
    for i,row in enumerate(manifolds):
        if _contact_values(row) is None:
            failures.append(dict(query="getContactPoints",index=i,reason="nonfinite_or_incomplete"))
            continue
        bodies={row[1],row[2]}
        ids=frozenset([row[3],row[4]])
        if bodies=={robot}:
            if ids in structural:
                counts[structural[ids]]+=1
            if ids==opposite: continue
            reason="forbidden_self_contact"
        elif bodies=={obj,table} and row[3]==row[4]==-1:
            table_support.append(i)
            if row[8]>=-.001: continue
            reason="registered_object_table_penetration_below_minus_1mm"
        else:
            reason="initial_robot_external_manifold" if robot in bodies else "unregistered_contact"
        failures.append(dict(query="getContactPoints",index=i,reason=reason,links=contact_label(row),distance_m=row[8]))
    closest={}
    mount_closest_checks=[]
    for name, a,b in [("robot_table",robot,table),(f"robot_object{OBJECT_ID}",robot,obj),(f"object{OBJECT_ID}_table",obj,table)]:
        points=p.getClosestPoints(a,b,0.,**kw)
        closest[name]=points
        for i,row in enumerate(points):
            why=None
            if _contact_values(row) is None: why="nonfinite_or_incomplete"
            elif {row[1],row[2]}!={a,b}: why="unexpected_query_identity"
            elif a==robot and b==table and {row[1]:row[3],row[2]:row[4]}=={robot:links["base_link_inertia"],table:-1}:
                mount_check=check_row(row,robot,table,links["base_link_inertia"])
                mount_closest_checks.append(dict(group=name,index=i,raw=row,check=mount_check))
                if not mount_check["passed"]:
                    why="registered_mount_closest_invariant"
            elif (a==robot and row[8]<0.) or (a==obj and row[8]<-.001):
                why="initial_external_penetration"
            if why:
                failures.append(dict(query="getClosestPoints",group=name,index=i,reason=why,
                                     links=contact_label(row),distance_m=row[8]))
    if not mount_closest_checks:
        failures.append(dict(query="getClosestPoints",group="robot_table",reason="registered_mount_witness_missing"))
    return dict(passed=not failures,registered_robot=robot,registered_table=table,registered_object=obj,
                filtered_pair_counts=counts,object_table_support_manifold_indices=table_support,
                world_manifolds=manifolds,direct_closest=closest,
                registered_mount_closest_checks=mount_closest_checks,failures=failures)
