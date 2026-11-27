"""One bounded, geometry-only voxel repair for preselected GraspNet014/017.

No physics import, connection, step, model inference or grasp episode.
Preserve historical failures. Use original watertight mesh center occupancy;
never fill holes, dilate, erode, sweep settings or change scientific thresholds.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,time,threading,traceback,sys
from pathlib import Path
import numpy as np
import trimesh
import open3d as o3d
import sim_v52_object_assets as base

ROOT=base.ROOT
DEST=ROOT/'data/simulation_v52/objects/voxel_repair'
REPORT=ROOT/'results/simulation_v52/object_assets_voxel_repair.json'
IDS=(14,17)
PITCH_M=.0015
MAX_BOXES=1024
BUDGET_SECONDS=900
LIMITS=dict(collision_surface_error_m=.004,collision_extent_error_m=.001,
            collision_relative_volume_error=.20,visual_surface_error_m=.001,
            visual_extent_error_m=.001,visual_relative_volume_error=.05)
START=time.monotonic()

def deadline():
    base.require(time.monotonic()-START<BUDGET_SECONDS,'Fixed900s geometry budget exhausted')

def timeout():
    base.once(DEST/'budget_exceeded.json',dict(at=base.now(),budget_seconds=BUDGET_SECONDS,
        elapsed_seconds=time.monotonic()-START,physics_episodes=0,completed=False))
    os._exit(124)

def occupancy_grid(mesh):
    bounds=mesh.bounds.copy()
    shape=np.ceil(mesh.extents/PITCH_M).astype(int)
    spacing=mesh.extents/shape
    axes=[bounds[0,i]+(np.arange(shape[i])+.5)*spacing[i] for i in range(3)]
    centers=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    ray=o3d.t.geometry.RaycastingScene()
    om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
    ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(om))
    values=[]
    for s in range(0,len(centers),32768):
        deadline()
        values.append(ray.compute_occupancy(o3d.core.Tensor(centers[s:s+32768].astype(np.float32)),nsamples=5).numpy())
    occupied=np.concatenate(values).reshape(tuple(shape))>.5
    base.require(occupied.any(),'Empty occupancy')
    return occupied,spacing,bounds[0]

def merge_boxes(occupied):
    # Fixed scan x,y,z; grow alongx, theny, thenz, no post-hoc strategy search.
    remaining=occupied.copy();n=occupied.shape;boxes=[]
    for x in range(n[0]):
        deadline()
        for y in range(n[1]):
            for z in range(n[2]):
                if not remaining[x,y,z]:continue
                xe=x+1
                while xe<n[0] and remaining[xe,y,z]:xe+=1
                ye=y+1
                while ye<n[1] and remaining[x:xe,ye,z].all():ye+=1
                ze=z+1
                while ze<n[2] and remaining[x:xe,y:ye,ze].all():ze+=1
                boxes.append([x,xe,y,ye,z,ze]);remaining[x:xe,y:ye,z:ze]=False
                base.require(len(boxes)<=MAX_BOXES,'Fixed1024box representation budget exceeded')
    rebuilt=np.zeros_like(occupied,dtype=np.uint16)
    for x,xe,y,ye,z,ze in boxes:rebuilt[x:xe,y:ye,z:ze]+=1
    base.require(np.array_equal(rebuilt,occupied.astype(np.uint16)),'Box compound changes occupancy or overlaps')
    return boxes

def exposed_union_mesh(occupied,spacing,origin):
    # Include exactly voxel faces whose neighbor is empty/outside. Internal
    # box seams and occupied-occupied voxel faces never enter the audit mesh.
    specs=[(0,-1,[(0,0,0),(0,0,1),(0,1,1),(0,1,0)]),
           (0, 1,[(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
           (1,-1,[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
           (1, 1,[(0,1,0),(0,1,1),(1,1,1),(1,1,0)]),
           (2,-1,[(0,0,0),(0,1,0),(1,1,0),(1,0,0)]),
           (2, 1,[(0,0,1),(1,0,1),(1,1,1),(0,1,1)])]
    vertices=[];faces=[];count=0
    for axis,sign,corners in specs:
        neighbor=np.roll(occupied,-sign,axis=axis)
        edge=[slice(None)]*3;edge[axis]=-1 if sign>0 else 0;neighbor[tuple(edge)]=False
        cells=np.argwhere(occupied & ~neighbor)
        q=(cells[:,None,:]+np.array(corners)[None,:,:])*spacing+origin
        vertices.append(q.reshape(-1,3));ix=np.arange(len(cells))*4+count
        faces.append(np.column_stack((ix,ix+1,ix+2)));faces.append(np.column_stack((ix,ix+2,ix+3)))
        count+=len(cells)*4
    mesh=trimesh.Trimesh(vertices=np.vstack(vertices),faces=np.vstack(faces),process=False)
    mesh.merge_vertices()
    volume=occupied.sum()*np.prod(spacing)
    base.require(abs(mesh.volume-volume)<1e-10,'Exposed union mesh volume differs from occupied voxel volume')
    return mesh

def urdf(directory,oid,extents,spacing,origin,boxes):
    mass=base.MASSES[oid];x,y,z=extents
    inertia=mass/12*np.array([y*y+z*z,x*x+z*z,x*x+y*y])
    collisions=[];box_info=[]
    for n,(x,xe,y,ye,z,ze) in enumerate(boxes):
        low=origin+spacing*np.array([x,y,z]);high=origin+spacing*np.array([xe,ye,ze])
        center=(low+high)/2;size=high-low
        center_text=' '.join(f'{v:.12g}' for v in center);size_text=' '.join(f'{v:.12g}' for v in size)
        collisions.append(f'    <collision name="voxel_box_{n:04d}"><origin xyz="{center_text}" rpy="0 0 0"/><geometry><box size="{size_text}"/></geometry></collision>')
        box_info.append(dict(grid_interval=[x,xe,y,ye,z,ze],center_m=center.tolist(),size_m=size.tolist()))
    ix,iy,iz=inertia
    text=f'''<?xml version="1.0"?>
<robot name="graspnet_{oid:03d}_voxel_repair_v52">
  <link name="object">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.10g}"/><inertia ixx="{ix:.12g}" ixy="0" ixz="0" iyy="{iy:.12g}" iyz="0" izz="{iz:.12g}"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="visual.obj" scale="1 1 1"/></geometry></visual>
{chr(10).join(collisions)}
  </link>
</robot>
'''
    (directory/'object.urdf').write_text(text,encoding='utf8')
    base.once(directory/'collision_boxes.json',dict(boxes=box_info,disjoint=True,exact_occupied_union=True))
    return dict(mass_kg=mass,mass_is_assumed=True,inertia_kg_m2=inertia.tolist(),
        inertia_model='uniform original-AABB inertia, engineering assumption; not measured object inertia',
        center_of_mass_asset_m=[0,0,0],friction=dict(lateral=.6,rolling=.001,spinning=.001,restitution=0),
        friction_is_assumed=True,friction_set_by_simulation_entrypoint=True,
        plausible_mass_range_for_future_sensitivity_kg=[.05,.5],friction_range_for_future_sensitivity=[.3,.9])

def run():
    base.require(Path.cwd().resolve()==ROOT,'Explicit project workdir required')
    base.require(not DEST.exists() and not REPORT.exists(),'No overwrite or automatic retry')
    DEST.mkdir(parents=True)
    watchdog=threading.Timer(BUDGET_SECONDS,timeout);watchdog.daemon=True;watchdog.start()
    records=[]
    try:
        selection=dict(at=base.now(),object_ids=list(IDS),fixed_target_pitch_m=PITCH_M,
            actual_pitch_rule='Each original-AABB axis split into ceil(length/1.5mm) equal cells; spacing<=1.5mm',
            occupancy='original watertight source mesh, cell-center5ray Open3D occupancy; no filling/dilation/erosion',
            merge='fixed x,y,zscan; contiguousgrowx theny thenz; exact disjoint occupied-set equality checked',
            collision_representation_budget=dict(max_boxes_per_object=MAX_BOXES,geometry_elapsed_budget_seconds=BUDGET_SECONDS),
            representation_change='Explicit new resource cap1024boxes replaces attempted32convex-pieces. Geometry/scientific thresholds unchanged; R3 not frozen and no physical outcomes exist.',
            limits=LIMITS,source_code=base.ident(Path(__file__)),helper_code=base.ident(Path(base.__file__)),
            physics_connections=0,physics_steps=0,grasp_episodes=0,parameter_sweep=False)
        base.once(DEST/'selection.json',selection)
        occurrences=base.source_occurrences()
        for oid in IDS:
            deadline();directory=DEST/f'{oid:03d}';directory.mkdir()
            src=base.SOURCE/f'models/{oid:03d}/nontextured.ply';textured=src.with_name('textured.obj')
            bindings={base.rel(p):base.ident(p) for p in (src,textured)}
            raw=trimesh.load(src,process=False)
            base.require(raw.is_watertight and raw.visual.kind=='vertex','Watertight colored original source required')
            center=raw.bounds.mean(axis=0);transform=np.eye(4);transform[:3,3]=-center
            source=raw.copy();source.apply_transform(transform)
            occupied,spacing,origin=occupancy_grid(source);boxes=merge_boxes(occupied)
            boundary=exposed_union_mesh(occupied,spacing,origin)
            np.savez_compressed(directory/'occupancy.npz',occupied=occupied,spacing_m=spacing,origin_m=origin)
            boundary.export(directory/'collision_union_boundary.obj')
            ca=base.audit_geometry(source,boundary)
            ca.update(audit_surface='Only exposed faces of voxel occupied union; no internal compound seams',
                occupied_voxels=int(occupied.sum()),grid_shape=list(occupied.shape),spacing_m=spacing.tolist(),
                grid_origin_m=origin.tolist(),box_count=len(boxes),boxes_disjoint_and_exact_occupancy=True,
                fill_holes=False,morphology_used=False,original_scale=1.)
            base.once(directory/'collision_audit.json',ca)
            base.require(ca['max_vertex_and_face_centroid_surface_error_m']<=LIMITS['collision_surface_error_m'] and ca['max_extent_error_m']<=LIMITS['collision_extent_error_m'] and ca['relative_volume_error']<=LIMITS['collision_relative_volume_error'],'Fixed collision geometry gate failed; stop without sweep')
            visual,color=base.colored_visual(source,directory);va=base.audit_geometry(source,visual)
            base.require(va['max_vertex_and_face_centroid_surface_error_m']<=LIMITS['visual_surface_error_m'] and va['max_extent_error_m']<=LIMITS['visual_extent_error_m'] and va['relative_volume_error']<=LIMITS['visual_relative_volume_error'],'Fixed visual geometry gate failed; stop')
            physical=urdf(directory,oid,source.extents,spacing,origin,boxes)
            for p,b in bindings.items():base.require(base.ident(ROOT/p)==b,'Historical source mutation')
            record=dict(object_id=oid,name=occurrences[oid][0]['name'],source_occurrences=occurrences[oid],source_bindings=bindings,
                source_to_asset=transform.tolist(),asset_to_source=np.linalg.inv(transform).tolist(),
                source_bounds_m=raw.bounds.tolist(),source_bbox_center_m=center.tolist(),asset_bounds_m=source.bounds.tolist(),
                extents_m=source.extents.tolist(),bottom_z_m=float(source.bounds[0,2]),top_z_m=float(source.bounds[1,2]),
                standing_pose='Original localz up; no rotation/rescaling; origin z=table_z-bottom_z+clearance.',
                urdf=base.rel(directory/'object.urdf'),visual_audit=va,collision_audit=ca,color=color,
                physical_assumptions=physical,passed_static_checks=True,physics_verified=False,physics_episodes=0,
                collision_representation='disjoint merged voxel boxes from original source occupancy',
                files={base.rel(p):base.ident(p) for p in directory.iterdir() if p.is_file()})
            base.once(directory/'metadata.json',record);records.append(record)
            print(json.dumps(dict(object_id=oid,boxes=len(boxes),pitch_m=spacing.tolist(),collision_error_m=ca['max_vertex_and_face_centroid_surface_error_m'],extent_error_m=ca['max_extent_error_m'],relative_volume_error=ca['relative_volume_error'],passed=True)),flush=True)
        prior=ROOT/'results/simulation_v52/object_assets.json'
        prior_record=json.loads(prior.read_text('utf8'))['objects'][0]
        base.require(prior_record['object_id']==21 and prior_record['passed_static_checks'],'Expected original021 admitted asset')
        for p,b in prior_record['files'].items():base.require(base.ident(ROOT/p)==b,'Original021 derived asset changed')
        records.append(prior_record)
        base.once(REPORT,dict(at=base.now(),passed=True,objects=records,usable_object_count=3,
            scope='Bounded R1/R2 static collision-asset repair only; not physics or grasp validation',
            r3_three_object_static_prerequisite_satisfied=True,r3_physics_verified=False,
            prior_report=base.rel(prior),prior_report_identity=base.ident(prior),selection=base.rel(DEST/'selection.json'),
            selection_identity=base.ident(DEST/'selection.json'),source_code=base.ident(Path(__file__)),
            seconds=time.monotonic()-START,actual_physics_episodes=0,actual_physics_connections=0,
            actual_neural_forwards=0,historical_assets_unchanged=True,
            limitations=['Finite vertex/face-centroid geometry audit is not a continuous Hausdorff proof.',
                'Features below fixed1.5mm target voxel spacing may be missed; no hole-filling operation used.',
                'Box union is an approximate occupied shape of the real mesh, not replacement with one ideal primitive.',
                'Mass/inertia/friction assumptions remain unmeasured; dynamics and stable settling untested.']))
        base.once(REPORT.with_suffix('.json.receipt.json'),base.ident(REPORT))
    except BaseException as exc:
        failure=dict(at=base.now(),passed=False,error=repr(exc),traceback=traceback.format_exc(),
            seconds=time.monotonic()-START,completed_object_ids=[r['object_id'] for r in records],
            physics_connections=0,physics_steps=0,grasp_episodes=0,no_retry_or_parameter_sweep=True)
        base.once(DEST/'failure.json',failure)
        if not REPORT.exists():base.once(REPORT,failure)
        raise
    finally:watchdog.cancel()

if __name__=='__main__':run()
