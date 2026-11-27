"""Training-side geometry references only; never an inference feature source.

Open3D is imported lazily so inventory/array audits can run in the vision env.
No function starts background work. Real rendering is called only by the root
authorized geometry CLI. Original assets are read, never rewritten or aligned.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import binary_dilation

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'logs/P6/correspondence_v40/data_audit.json'
MODEL_ROOT=ROOT/'data/raw/prerequisites_v26/models'
REFERENCE_KEYS=((3,'0000'),(3,'0128'),(4,'0000'),(4,'0128'))
LIMITS=dict(render_frames=4,seconds=600,rss_bytes=3*1024**3,output_bytes=128*1024**2,threads=2)
QUALITY_THRESHOLDS=dict(foreground_iou_min=.90,interior_median_abs_depth_m_max=.005,interior_p95_abs_depth_m_max=.020)


def need(value,message):
    if not bool(value): raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def identity(path):
    p=Path(path); st=p.stat()
    return dict(sha256=sha(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)


def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def reference_records():
    a=json.loads(AUDIT.read_text(encoding='utf-8')); need(a['passed'],'Parent data audit did not pass')
    indexed={(r['scene'],r['frame']):r for r in a['records']['fit']}
    records=[]
    for key in REFERENCE_KEYS:
        need(key in indexed,'Reference frame is not in frozen training partition')
        r=dict(indexed[key]); r['paths']={k:relative(ROOT/a['data_root']/v) for k,v in r['files'].items()}
        records.append(r)
    return records


def inventory():
    records=reference_records(); protected={relative(AUDIT):identity(AUDIT)}; objects=set()
    for r in records:
        for kind,path in r['paths'].items():
            info=identity(ROOT/path); need(info['sha256']==r['sha256'][kind],'Parent raw input changed: '+path); protected[path]=info
        m=loadmat(ROOT/r['paths']['meta'],variable_names=['cls_indexes','poses','intrinsic_matrix','factor_depth'])
        ids=np.asarray(m['cls_indexes']).reshape(-1).astype(int)
        need(m['poses'].shape==(3,4,len(ids)),'Pose/object order mismatch')
        r['object_ids_zero_based']=(ids-1).tolist(); objects.update(r['object_ids_zero_based'])
    for oid in sorted(objects):
        p=MODEL_ROOT/f'{oid:03d}'/'nontextured.ply'; need(p.is_file(),'Missing triangle mesh: '+str(p))
        protected[relative(p)]=identity(p)
    # Camera trajectories are audit/provenance assets, not additional inference inputs.
    for scene in sorted({s for s,_ in REFERENCE_KEYS}):
        for name in ('camera_poses.npy','cam0_wrt_table.npy'):
            p=ROOT/'data/raw/prerequisites_v26/scenes'/f'scene_{scene:04d}'/'realsense'/name
            protected[relative(p)]=identity(p)
    return dict(records=records,protected=protected,mesh_root=relative(MODEL_ROOT),mesh_objects=sorted(objects),limits=LIMITS,
        reference_resolution=[640,480],raw_asset_decodes=0,real_render_calls=0,
        role='Training-side reference/stratification only; exact synthetic source-ID remains primary controlled ground truth.',
        trusted_geometry_thresholds=QUALITY_THRESHOLDS,
        scope='Per-frame object meshes/poses in camera coordinates; first hit includes inter-object occlusion. Unmodeled table/background is unknown, never deleted from prediction.',
        no_registration_refit=True,no_pose_adjustment=True)


def source_pixel_coordinates(original_size,target_size=(640,480)):
    W,H=map(int,original_size); w,h=map(int,target_size)
    v,u=np.indices((h,w),dtype=np.int64)
    return ((2*u+1)*W)//(2*w),((2*v+1)*H)//(2*h)


def camera_rays(K,original_size,target_size=(640,480)):
    K=np.asarray(K,np.float64); need(K.shape==(3,3) and np.isfinite(K).all(),'Invalid K')
    need(np.allclose(K[2],[0,0,1],atol=1e-12) and K[0,0]>0 and K[1,1]>0,'Invalid pinhole camera')
    ox,oy=source_pixel_coordinates(original_size,target_size)
    directions=np.stack((ox,oy,np.ones_like(ox)),axis=-1)@np.linalg.inv(K).T
    need(np.allclose(directions[...,2],1.,atol=1e-12),'Unnormalized ray z must equal one')
    origins=np.zeros_like(directions)
    return np.concatenate((origins,directions),axis=-1).astype(np.float32)


def pose_and_mesh_check(vertices,triangles,pose):
    v=np.asarray(vertices,np.float64); t=np.asarray(triangles); T=np.asarray(pose,np.float64)
    need(v.ndim==2 and v.shape[1]==3 and len(v)>2 and np.isfinite(v).all(),'Invalid mesh vertices')
    need(t.ndim==2 and t.shape[1]==3 and len(t)>0 and t.min()>=0 and t.max()<len(v),'Invalid triangle indices')
    need(T.shape==(3,4) and np.isfinite(T).all(),'Invalid object-to-camera pose')
    R=T[:,:3]
    need(np.allclose(R.T@R,np.eye(3),atol=1e-4) and abs(np.linalg.det(R)-1)<1e-4,'Nonrigid object pose')
    diagonal=float(np.linalg.norm(v.max(axis=0)-v.min(axis=0)))
    need(.005<=diagonal<=1.,'Mesh scale outside prior0.005..1m bounds; no rescaling allowed')
    return dict(vertices=len(v),triangles=len(t),diagonal_m=diagonal,pose=T.tolist(),
        orthogonality_max=float(np.max(np.abs(R.T@R-np.eye(3)))),rotation_determinant=float(np.linalg.det(R)),
        unit_convention='mesh and translation interpreted as supplied meters; range sanity is not independent proof of metrology')


def cast_scene(meshes,K,original_size,target_size=(640,480),check_scale=True):
    """meshes: [{object_id (label convention1..88),vertices,triangles,pose}]."""
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene(nthreads=LIMITS['threads'])
    geometry_objects={}; triangle_counts={}; transformed={}; metadata=[]
    for entry in meshes:
        v=np.asarray(entry['vertices'],np.float64); t=np.asarray(entry['triangles'],np.int64); T=np.asarray(entry['pose'],np.float64)
        if check_scale: info=pose_and_mesh_check(v,t,T)
        else: info=dict(vertices=len(v),triangles=len(t),pose=T.tolist())
        cam=v@T[:,:3].T+T[:,3]
        tensor_mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(cam.astype(np.float32)),o3d.core.Tensor(t.astype(np.int64)))
        gid=int(scene.add_triangles(tensor_mesh)); geometry_objects[gid]=int(entry['object_id']); triangle_counts[gid]=len(t)
        transformed[gid]=(cam,t)
        metadata.append(dict(object_id=int(entry['object_id']),geometry_id=gid,**info))
    rays=camera_rays(K,original_size,target_size)
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=LIMITS['threads'])
    depth=hit['t_hit'].numpy(); gids=hit['geometry_ids'].numpy(); tids=hit['primitive_ids'].numpy()
    bary=hit['primitive_uvs'].numpy(); normals=hit['primitive_normals'].numpy()
    valid=np.isfinite(depth)&(depth>0)
    object_id=np.zeros(depth.shape,np.int16); geometry_id=np.full(depth.shape,-1,np.int16); triangle_id=np.full(depth.shape,-1,np.int32)
    for gid,oid in geometry_objects.items():
        mask=valid&(gids==gid); object_id[mask]=oid; geometry_id[mask]=gid; triangle_id[mask]=tids[mask].astype(np.int32)
        need(np.all(tids[mask]<triangle_counts[gid]),'Returned primitive index outside actual model')
    need(np.all(object_id[valid]>0),'Hit with unknown geometry identity')
    z=np.where(valid,depth,0).astype(np.float32); bary=np.where(valid[...,None],bary,0).astype(np.float32)
    normals=np.where(valid[...,None],normals,0).astype(np.float32)
    # Fixed64 face/barycentric/ray anchors independently establish t_hit=z convention.
    ids=np.flatnonzero(valid); count=min(64,len(ids)); selected=ids[np.arange(count)*len(ids)//count] if count else np.empty(0,int)
    anchors=[]
    for i in selected:
        y,x=divmod(int(i),z.shape[1]); gid=int(geometry_id[y,x]); tid=int(triangle_id[y,x]); cam,t=transformed[gid]
        points=cam[t[tid]]; u,v=map(float,bary[y,x]); xyz=(1-u-v)*points[0]+u*points[1]+v*points[2]
        ray_xyz=rays[y,x,3:].astype(float)*float(z[y,x])
        error=float(np.linalg.norm(xyz-ray_xyz))
        need(error<=2e-5,'Triangle barycentric point differs from first-hit ray geometry')
        anchors.append(dict(index=int(i),object_id=int(object_id[y,x]),geometry_id=gid,triangle_id=tid,
            camera_vertices=points.tolist(),barycentric_uv=[u,v],ray_xyz=ray_xyz.tolist(),reconstruction_error_m=error))
    arrays=dict(reference_depth_m=z,reference_object_id=object_id,reference_geometry_id=geometry_id,reference_triangle_id=triangle_id,
        reference_barycentric_uv=bary,reference_normal_camera=normals,reference_valid=valid)
    return arrays,dict(meshes=metadata,anchors=anchors,visible_pixels=int(valid.sum()),missing_pixels=int((~valid).sum()),
        primitive_scope='Triangle id is local to geometry/model. Adjacent triangle inequality is not a different physical surface.',
        ray_scope='Original nearest-neighbor sampled source pixel with original K, nonunit ray direction z=1: t_hit is optical-axis depth in meters.',
        unmodeled_background='No hit is unknown reference, not free space; complete observed cloud remains untouched.',open3d_version=o3d.__version__)


def render_reference(scene,frame):
    import open3d as o3d
    records={(r['scene'],r['frame']):r for r in reference_records()}; key=(int(scene),str(frame))
    need(key in records,'Rendering outside fixed four training frames is forbidden')
    r=records[key]; m=loadmat(ROOT/r['paths']['meta'],variable_names=['poses','cls_indexes','intrinsic_matrix','factor_depth'])
    ids=np.asarray(m['cls_indexes']).reshape(-1).astype(int); poses=np.asarray(m['poses'],float); K=np.asarray(m['intrinsic_matrix'],float)
    factor=float(m['factor_depth'].reshape(-1)[0]); need(np.isfinite(factor) and factor>0,'Invalid depth units')
    with Image.open(ROOT/r['paths']['depth']) as im:
        original_size=im.size; observed=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32)/factor
    with Image.open(ROOT/r['paths']['label']) as im:
        need(im.size==original_size,'Label/depth dimensions differ'); labels=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.int16)
    meshes=[]
    for i,label in enumerate(ids):
        path=MODEL_ROOT/f'{label-1:03d}'/'nontextured.ply'
        mesh=o3d.io.read_triangle_mesh(str(path),enable_post_processing=False)
        meshes.append(dict(object_id=int(label),vertices=np.asarray(mesh.vertices).copy(),triangles=np.asarray(mesh.triangles).copy(),pose=poses[:,:,i]))
    arrays,metadata=cast_scene(meshes,K,original_size)
    arrays['observed_depth_m']=observed; arrays['label_object_id']=labels; arrays['K']=K
    arrays['original_size']=np.asarray(original_size,np.int64); arrays['factor_depth']=np.asarray(factor)
    metadata.update(scene=int(scene),frame=str(frame),original_size=list(original_size),K=K.tolist(),factor_depth=factor,
        mesh_reads=len(meshes),raw_rgb_decodes=0,raw_depth_decodes=1,raw_label_decodes=1,neural_calls=0)
    return arrays,metadata


def reference_metrics(arrays):
    ref=np.asarray(arrays['reference_depth_m']); oid=np.asarray(arrays['reference_object_id']); valid=np.asarray(arrays['reference_valid'],bool)
    obs=np.asarray(arrays['observed_depth_m']); labels=np.asarray(arrays['label_object_id'])
    need(ref.shape==obs.shape==labels.shape==oid.shape==valid.shape,'Reference shape mismatch')
    need(np.array_equal(valid,(ref>0)&np.isfinite(ref)) and np.all(oid[valid]>0) and np.all(ref[~valid]==0),'Invalid visible/missing reference encoding')
    foreground=labels>0; union=foreground|valid; intersection=foreground&valid
    obs_valid=np.isfinite(obs)&(obs>0)&(obs<=2.)
    b=np.zeros(ref.shape,bool)
    b[1:]|=labels[1:]!=labels[:-1]; b[:-1]|=labels[:-1]!=labels[1:]
    b[:,1:]|=labels[:,1:]!=labels[:,:-1]; b[:,:-1]|=labels[:,:-1]!=labels[:,1:]
    boundary=binary_dilation(b,iterations=2)
    interior=foreground&valid&obs_valid&~boundary&(labels==oid)
    pairs=foreground&valid&obs_valid
    residual=np.abs(obs-ref)
    values=residual[interior]
    iou=float(intersection.sum()/union.sum()) if union.any() else 0.
    median=float(np.median(values)) if len(values) else None
    p95=float(np.quantile(values,.95)) if len(values) else None
    identities=int(np.sum((labels==oid)&intersection))
    metrics=dict(foreground_iou=iou,interior_median_abs_depth_m=median,interior_p95_abs_depth_m=p95,
        foreground_label_pixels=int(foreground.sum()),rendered_object_pixels=int(valid.sum()),intersection_pixels=int(intersection.sum()),union_pixels=int(union.sum()),
        matched_object_identity_pixels=identities,identity_agreement_on_intersection=float(identities/intersection.sum()) if intersection.any() else None,
        labeled_foreground_no_reference=int((foreground&~valid).sum()),rendered_object_without_foreground_label=int((valid&~foreground).sum()),
        foreground_invalid_observed_depth=int((foreground&~obs_valid).sum()),eligible_depth_pair_pixels=int(pairs.sum()),
        interior_same_object_pair_pixels=int(interior.sum()),boundary_pair_pixels=int((pairs&boundary).sum()),
        pair_object_identity_mismatches=int((pairs&(labels!=oid)).sum()),
        all_pair_median_abs_depth_m=float(np.median(residual[pairs])) if pairs.any() else None,
        boundary_median_abs_depth_m=float(np.median(residual[pairs&boundary])) if np.any(pairs&boundary) else None,
        raw_depth_over2m=int(np.sum(np.isfinite(obs)&(obs>2.))),no_hit_total=int((~valid).sum()),
        sampling='Full640x480 nearest-source grid;2px GT-label boundary excluded only from interior depth metric and fully reported separately.')
    gates=dict(foreground_iou=iou>=.90,interior_depth_median=median is not None and median<=.005,
        interior_depth_p95=p95 is not None and p95<=.020)
    return metrics,gates


def audit_cached(arrays,metadata):
    """Read-only numerical audit; zero ray casting and zero image/mesh reload."""
    K=np.asarray(arrays['K']); original_size=tuple(np.asarray(arrays['original_size']).tolist()); ref=arrays['reference_depth_m']
    rays=camera_rays(K,original_size,(ref.shape[1],ref.shape[0])); valid=arrays['reference_valid']
    normal=np.asarray(arrays['reference_normal_camera']); norms=np.linalg.norm(normal[valid],axis=1)
    need(np.isfinite(norms).all() and np.allclose(norms,1.,atol=1e-4),'Invalid visible primitive normals')
    counts={m['geometry_id']:m['triangles'] for m in metadata['meshes']}
    objects={m['geometry_id']:m['object_id'] for m in metadata['meshes']}
    for gid,count in counts.items():
        mask=valid&(arrays['reference_geometry_id']==gid)
        need(np.all((arrays['reference_triangle_id'][mask]>=0)&(arrays['reference_triangle_id'][mask]<count)),'Primitive id outside mesh')
        need(np.all(arrays['reference_object_id'][mask]==objects[gid]),'Geometry-object identity mismatch')
    maxerror=0.
    for anchor in metadata['anchors']:
        y,x=divmod(anchor['index'],ref.shape[1]); vv=np.asarray(anchor['camera_vertices']); a,b=anchor['barycentric_uv']
        xyz=vv[0]*(1-a-b)+vv[1]*a+vv[2]*b; ray=rays[y,x,3:].astype(float)*float(ref[y,x])
        error=float(np.linalg.norm(xyz-ray)); need(error<=2e-5,'Cached barycentric-ray mismatch'); maxerror=max(maxerror,error)
        need(int(arrays['reference_object_id'][y,x])==anchor['object_id'] and int(arrays['reference_triangle_id'][y,x])==anchor['triangle_id'],'Cached anchor identity mismatch')
    metrics,gates=reference_metrics(arrays)
    return dict(passed=True,anchor_count=len(metadata['anchors']),max_anchor_error_m=maxerror,visible_pixels=int(valid.sum()),
        metrics=metrics,quality_gates=gates,geometry_trusted=all(gates.values()),new_render_calls=0,
        limitation='This checks cached identities/units/first-hit encoding and recomputes metrics, not an independent second renderer or real sensor ground truth.')


def candidate_support(target_xyz,candidate_xyz,candidate_valid,sample_valid,tolerance_m):
    """Evaluation-only finite-center support, not probability/continuous-tail coverage."""
    target=np.asarray(target_xyz,float); candidate=np.asarray(candidate_xyz,float); mask=np.asarray(candidate_valid,bool); eligible=np.asarray(sample_valid,bool)
    need(target.ndim==2 and target.shape[1]==3 and candidate.ndim==3 and candidate.shape[0]==len(target) and candidate.shape[2]==3 and mask.shape==candidate.shape[:2] and eligible.shape==(len(target),),'Invalid support arrays')
    tolerance=np.broadcast_to(np.asarray(tolerance_m,float),(len(target),))
    need(np.isfinite(tolerance).all() and np.all(tolerance>=0),'Invalid finite support tolerance')
    mask=mask&np.isfinite(candidate).all(axis=2); finite_target=np.isfinite(target).all(axis=1)
    distances=np.linalg.norm(candidate-target[:,None,:],axis=2); distances[~mask]=np.inf
    nearest=distances.min(axis=1); supported=eligible&finite_target&(nearest<=tolerance)
    denominator=int(eligible.sum()); missing=int(np.sum(eligible&(~finite_target|~mask.any(axis=1))))
    return dict(eligible_count=denominator,supported_count=int(supported.sum()),unsupported_count=denominator-int(supported.sum()),
        missing_target_or_no_valid_candidate=missing,support_fraction=float(supported.sum()/denominator) if denominator else None,
        max_candidates_per_point=candidate.shape[1],finite_tolerance_min_m=float(tolerance.min()) if len(target) else None,
        finite_tolerance_max_m=float(tolerance.max()) if len(target) else None,
        scope='Oracle finite candidate-center inclusion only. Invalid/unsupported remain denominator; not a fitted-method result or calibration.',
        nearest_distance_m=np.where(np.isfinite(nearest),nearest,np.nan),supported=supported)
