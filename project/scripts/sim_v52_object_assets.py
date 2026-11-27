"""Create isolated, colored real GraspNet objects for v52 engineering admission.

Pure geometry conversion and static checks only: no physics client or episode.
Original models and historical research files are read-only.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,time,traceback,math,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from scipy.spatial import ConvexHull,HalfspaceIntersection,cKDTree
import trimesh
import open3d as o3d
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/simulation_v52/objects/final_static_audit'
REPORT=ROOT/'results/simulation_v52/object_assets.json'
SOURCE=ROOT/'data/raw/prerequisites_v26'
IDS=(14,17,21)
MASSES={14:.10,17:.10,21:.04}
LIMITS=dict(visual_faces=8000,visual_vertex_surface_error_m=.001,
    visual_extent_error_m=.001,visual_volume_relative_error=.05,
    collision_vertex_surface_error_m=.004,collision_volume_relative_error=.20,
    collision_extent_error_m=.001,collision_vertices=600,
    horizontal_width_m=.120,standing_height_min_m=.04,standing_height_max_m=.12)

def require(value,message):
    if not value:raise RuntimeError(message)

def ident(path):
    path=Path(path);st=path.stat()
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=digest,bytes=st.st_size,mtime_ns=st.st_mtime_ns)

def rel(path):return Path(path).relative_to(ROOT).as_posix()
def now():return datetime.now().astimezone().isoformat()

def once(path,value):
    path=Path(path);require(path.resolve().is_relative_to(ROOT),'Output outside project')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def directions():
    # Fixed 162-direction spherical cover plus exact six axis supports.
    d=np.vstack((trimesh.creation.icosphere(subdivisions=2,radius=1).vertices,np.eye(3),-np.eye(3)))
    return np.unique(np.round(d,12),axis=0)

def closest_distances(vertices,mesh):
    s=o3d.t.geometry.RaycastingScene()
    om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(om))
    result=[]
    for start in range(0,len(vertices),32768):
        result.append(s.compute_distance(o3d.core.Tensor(np.asarray(vertices[start:start+32768],dtype=np.float32))).numpy())
    values=np.concatenate(result)
    return dict(max_m=float(values.max(initial=0)),p99_m=float(np.quantile(values,.99)),mean_m=float(values.mean()),vertices=len(vertices))

def surface_probe(mesh):
    # All vertices and every face centroid. This is not a continuous Hausdorff certificate.
    return np.vstack((mesh.vertices,np.asarray(mesh.triangles).mean(axis=1)))

def audit_geometry(source,derived):
    a=closest_distances(surface_probe(source),derived)
    b=closest_distances(surface_probe(derived),source)
    volume=float(abs(source.volume));dv=float(abs(derived.volume))
    return dict(source_to_derived=a,derived_to_source=b,
        max_vertex_and_face_centroid_surface_error_m=max(a['max_m'],b['max_m']),
        continuous_hausdorff_certified=False,
        max_extent_error_m=float(np.abs(source.extents-derived.extents).max()),
        source_volume_m3=volume,derived_volume_m3=dv,
        relative_volume_error=abs(dv-volume)/volume,
        derived_watertight=bool(derived.is_watertight))

def colored_visual(source,directory):
    om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(source.vertices),o3d.utility.Vector3iVector(source.faces))
    om.vertex_colors=o3d.utility.Vector3dVector(np.asarray(source.visual.vertex_colors[:,:3],float)/255.)
    simple=om.simplify_quadric_decimation(target_number_of_triangles=LIMITS['visual_faces'])
    mesh=trimesh.Trimesh(np.asarray(simple.vertices),np.asarray(simple.triangles),process=False)
    _,indices=cKDTree(source.vertices).query(mesh.vertices,k=1)
    colors=np.asarray(source.visual.vertex_colors[indices,:3],dtype=np.uint8)
    # Per-triangle atlas: preserves original PLY vertex-color variation without
    # requiring missing GraspNet texture files. UVs are independent per face.
    tile=16;columns=math.ceil(math.sqrt(len(mesh.faces)));side=columns*tile
    texture=np.zeros((side,side,3),dtype=np.uint8);uv=[]
    yy,xx=np.mgrid[0:tile,0:tile];u=np.clip((xx-2)/(tile-5),0,1);v=np.clip((yy-2)/(tile-5),0,1)
    weights=np.stack((np.maximum(1-u-v,0),u,v),axis=-1);weights/=weights.sum(axis=-1,keepdims=True)
    for n,face in enumerate(mesh.faces):
        row,col=divmod(n,columns);oy,ox=row*tile,col*tile
        texture[oy:oy+tile,ox:ox+tile]=np.clip(weights@colors[face].astype(float),0,255).astype(np.uint8)
        for x,y in ((2,2),(tile-3,2),(2,tile-3)):
            uv.append(((ox+x+.5)/side,1-(oy+y+.5)/side))
    Image.fromarray(texture).save(directory/'visual.png')
    (directory/'visual.mtl').write_text('newmtl original_vertex_colors\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nmap_Kd visual.png\n',encoding='ascii')
    with (directory/'visual.obj').open('x',encoding='ascii') as f:
        f.write('# Real GraspNet PLY-derived mesh; metric scale unchanged\nmtllib visual.mtl\nusemtl original_vertex_colors\n')
        for xyz in mesh.vertices:f.write('v %.10g %.10g %.10g\n'%tuple(xyz))
        for pair in uv:f.write('vt %.10g %.10g\n'%pair)
        for n,face in enumerate(mesh.faces):f.write('f '+' '.join(f'{int(j)+1}/{3*n+k+1}' for k,j in enumerate(face))+'\n')
    return mesh,dict(original_colors='PLY vertex RGBA, RGB used',transfer='nearest original vertex after geometry-only quadric decimation',
        atlas='per-triangle barycentric RGB interpolation',texture_size=[side,side],color_ground_truth_claim=False)

def outer_collision(source):
    ds=directions();support=np.array([np.max(source.vertices@d) for d in ds])
    center=source.vertices.mean(axis=0)
    halfspaces=np.column_stack((ds,-support))
    require(np.max(ds@center-support)<-1e-6,'Convex intersection needs a strict interior point')
    vertices=HalfspaceIntersection(halfspaces,center).intersections
    hull=trimesh.convex.convex_hull(vertices)
    # The construction contains every source vertex, proved against each fixed
    # support halfspace; it may fill concavities, measured separately below.
    excess=max(float(np.max(source.vertices@d)-s) for d,s in zip(ds,support))
    return hull,dict(method='intersection of fixed directional support halfspaces; convex outer approximation of real surface',
        support_directions=ds.tolist(),support_offsets_m=support.tolist(),support_count=len(ds),
        every_source_vertex_halfspace_max_violation_m=excess,contains_all_source_vertices=excess<=1e-10,
        convex=True,not_a_primitive_box=True)

def source_occurrences():
    config=json.loads((ROOT/'configs/risk_calibration_v31.json').read_text('utf-8-sig'))
    occurrences={i:[] for i in IDS}
    for scene in config['groups']['train']:
        path=SOURCE/f'scenes/scene_{scene:04d}/realsense/annotations/0064.xml'
        for obj in ET.parse(path).findall('obj'):
            oid=int(obj.findtext('obj_id'))
            if oid in occurrences:occurrences[oid].append(dict(scene=scene,name=obj.findtext('obj_name'),annotation=rel(path),identity=ident(path)))
    require(all(occurrences[i] for i in IDS),'Each engineering object must occur in the original70 training scenes')
    return occurrences

def write_urdf(directory,oid,mass,extents):
    # A documented engineering assumption: center at AABB center; uniform
    # same-size bounding-box inertia, not measured manufacturer properties.
    x,y,z=extents;ix,iy,iz=mass/12*np.array([y*y+z*z,x*x+z*z,x*x+y*y])
    text=f'''<?xml version="1.0"?>
<robot name="graspnet_{oid:03d}_engineering_v52">
  <link name="object">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.10g}"/><inertia ixx="{ix:.12g}" ixy="0" ixz="0" iyy="{iy:.12g}" iyz="0" izz="{iz:.12g}"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="visual.obj" scale="1 1 1"/></geometry></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="collision.obj" scale="1 1 1"/></geometry></collision>
  </link>
</robot>
'''
    (directory/'object.urdf').write_text(text,encoding='utf8')
    return dict(mass_kg=mass,mass_is_assumed=True,inertia_kg_m2=[float(ix),float(iy),float(iz)],
        inertia_model='uniform same-extents bounding-box inertia at AABB center, engineering assumption, not measured object inertia',
        center_of_mass_asset_m=[0.,0.,0.],friction=dict(lateral=.6,rolling=.001,spinning=.001,restitution=0.),
        friction_is_assumed=True,friction_range_for_future_sensitivity=[.3,.9],
        plausible_mass_range_for_future_sensitivity_kg=[.02,.08] if oid==21 else [.05,.5],
        randomization_in_this_engineering_asset_build=False)

def run():
    start=time.monotonic();require(Path.cwd().resolve()==ROOT,'Explicit project workdir required')
    require(not REPORT.exists() and not (DEST/'selection.json').exists(),'Existing assets retained; no overwrite/resume')
    DEST.mkdir(parents=True,exist_ok=True)
    selection=dict(at=now(),object_ids=list(IDS),selection_basis='Real GraspNet objects in original candidate-training scenes; compact can/round geometries, both horizontal extents<120mm, original-z height40..120mm; no outcomes read',
        rejected_geometry=[dict(object_id=41,reason='Original181.757mm long dimension prevents both yaw0/yaw90 horizontal grasp-width<120mm; upright height would exceed120mm',extents_m=[.18175699561834335,.051601000130176544,.037449998781085014]),dict(object_id=4,reason='No occurrence in any original0..99 scene frame0064 XML; training/development occurrence not established',evidence='data/simulation_v52/objects/preparation_history/source_occurrence_failure_001/failure.json')],
        limits=LIMITS,scaling=1.,mass_kg=MASSES,physics_episodes=0,model_forwards=0,
        historical_source_mutation=False,source_code=ident(Path(__file__)),source_occurrences=source_occurrences())
    selection['required_usable_objects']=3
    selection['final_bounded_selection']=True
    selection['earlier_object_002_rejected']={'reason':'Deep can-base concavity exceeds unchanged4mm collision surface limit; single hull, VHACD and spatial partition preparations preserved, no physical trial.', 'evidence':'data/simulation_v52/objects/collision_failure_static_detail.json'}
    once(DEST/'selection.json',selection);objects=[];rejected=[]
    try:
        for oid in IDS:
            directory=DEST/f'{oid:03d}';directory.mkdir(exist_ok=False)
            src=SOURCE/f'models/{oid:03d}/nontextured.ply';obj=src.with_name('textured.obj');bindings={rel(p):ident(p) for p in (src,obj)}
            raw=trimesh.load(src,process=False);require(isinstance(raw,trimesh.Trimesh) and raw.is_watertight,'Expected watertight original triangulated object')
            require(raw.visual.kind=='vertex','Original PLY colors missing')
            extents=raw.extents;require(max(extents[:2])<LIMITS['horizontal_width_m'] and LIMITS['standing_height_min_m']<=extents[2]<=LIMITS['standing_height_max_m'],'Object violates preregistered geometry scope')
            center=raw.bounds.mean(axis=0);transform=np.eye(4);transform[:3,3]=-center
            source=raw.copy();source.apply_transform(transform)
            collision,cp=outer_collision(source);collision.export(directory/'collision.obj');ca=audit_geometry(source,collision)
            collision_passed=(len(collision.vertices)<=LIMITS['collision_vertices'] and ca['max_vertex_and_face_centroid_surface_error_m']<=LIMITS['collision_vertex_surface_error_m'] and ca['max_extent_error_m']<=LIMITS['collision_extent_error_m'] and ca['relative_volume_error']<=LIMITS['collision_volume_relative_error'])
            if not collision_passed:
                for name,value in bindings.items():require(ident(ROOT/name)==value,'Original geometry changed')
                refusal=dict(object_id=oid,name=selection['source_occurrences'][oid][0]['name'],source_bindings=bindings,extents_m=extents.tolist(),source_to_asset=transform.tolist(),collision_audit=ca,collision_construction=cp,collision_vertices=len(collision.vertices),status='rejected_static_geometry',reason='Convex collision proxy exceeds unchanged4mm surface error limit',physics_episodes=0,source_originals_unchanged=True,files={rel(directory/'collision.obj'):ident(directory/'collision.obj')})
                once(directory/'rejection.json',refusal);rejected.append(refusal)
                print(json.dumps(dict(object_id=oid,passed=False,collision_error_m=ca['max_vertex_and_face_centroid_surface_error_m'])),flush=True)
                continue
            visual,color=colored_visual(source,directory);va=audit_geometry(source,visual)
            require(va['max_vertex_and_face_centroid_surface_error_m']<=LIMITS['visual_vertex_surface_error_m'] and va['max_extent_error_m']<=LIMITS['visual_extent_error_m'] and va['relative_volume_error']<=LIMITS['visual_volume_relative_error'],'Visual proxy exceeds fixed geometry limits')
            physical=write_urdf(directory,oid,MASSES[oid],extents)
            files={rel(p):ident(p) for p in directory.iterdir() if p.is_file()}
            for name,value in bindings.items():require(ident(ROOT/name)==value,'Original geometry changed')
            record=dict(object_id=oid,name=selection['source_occurrences'][oid][0]['name'],source_bindings=bindings,
                source_to_asset=transform.tolist(),asset_to_source=np.linalg.inv(transform).tolist(),
                source_bounds_m=raw.bounds.tolist(),source_bbox_center_m=center.tolist(),asset_bounds_m=source.bounds.tolist(),
                extents_m=extents.tolist(),bottom_z_m=float(source.bounds[0,2]),top_z_m=float(source.bounds[1,2]),
                standing_pose='Original local z up; no mesh rotation or rescaling. Asset x/y AABB center at0. Place asset origin z=table_z-bottom_z+clearance.',
                candidate_axis_hint_geometry_only=dict(approach_asset=[0,0,-1],closing_axes_asset=[[1,0,0],[0,1,0]],both_widths_under_120mm=True,not_an_executable_grasp_or_positive_label=True),
                source_vertices=len(source.vertices),source_faces=len(source.faces),visual_vertices=len(visual.vertices),visual_faces=len(visual.faces),collision_vertices=len(collision.vertices),collision_faces=len(collision.faces),
                visual_audit=va,collision_audit=ca,collision_construction=cp,color=color,physical_assumptions=physical,
                urdf=rel(directory/'object.urdf'),files=files,passed_static_checks=True,physics_verified=False)
            once(directory/'metadata.json',record);objects.append(record)
            print(json.dumps(dict(object_id=oid,extents_m=record['extents_m'],visual_error_m=va['max_vertex_and_face_centroid_surface_error_m'],collision_error_m=ca['max_vertex_and_face_centroid_surface_error_m'],collision_vertices=len(collision.vertices),passed=True)),flush=True)
        once(REPORT,dict(at=now(),passed=len(objects)==3,scope='R1/R2 static real-object assets only; zero grasping or physics episodes',objects=objects,rejected_objects=rejected,
            required_usable_object_count=3,usable_object_count=len(objects),r3_48_episode_prerequisite_satisfied=len(objects)==3,
            status='blocked_static_collision_geometry' if len(objects)<3 else 'static_assets_ready',
            selection=ident(DEST/'selection.json'),source_code=ident(Path(__file__)),seconds=time.monotonic()-start,
            environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,trimesh=trimesh.__version__,open3d=o3d.__version__),
            actual_physics_episodes=0,actual_neural_forwards=0,historical_assets_unchanged=True,
            limitations=['Physical masses/friction/inertia are declared assumptions, not GraspNet measured parameters.','Convex collision assets may fill concavities; errors quantified on all vertices and all face centroids, not a continuous Hausdorff proof.','Visual color uses original PLY vertex colors; missing high-resolution texture photographs are not invented.','No grasp, stable-settling or simulator-import success claimed by static conversion.']))
        once(REPORT.with_suffix('.json.receipt.json'),ident(REPORT))
    except BaseException as exc:
        once(DEST/'failure.json',dict(at=now(),error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,completed_object_ids=[r['object_id'] for r in objects],actual_physics_episodes=0));raise

if __name__=='__main__':run()
