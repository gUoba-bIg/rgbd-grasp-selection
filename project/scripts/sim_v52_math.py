"""Observation-only geometric interface. No labels or simulator body poses are inputs."""
import numpy as np

def transform(rotation=None, translation=None):
    value = np.eye(4)
    if rotation is not None:
        value[:3, :3] = rotation
    if translation is not None:
        value[:3, 3] = translation
    return value

def grasp_to_tool(T_world_camera, candidate, pad_center_z):
    """^A T_B maps B coordinates to A. GraspNet contact=t+R[:,0]*depth."""
    candidate = np.asarray(candidate, dtype=float)
    T_camera_grasp = transform(candidate[4:13].reshape(3, 3), candidate[13:16])
    # Gripper +z is approach; +y is closing. Its +x is GraspNet -z.
    R_grasp_tool = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    T_grasp_tool = transform(R_grasp_tool, [candidate[3]-pad_center_z, 0., 0.])
    return np.asarray(T_world_camera) @ T_camera_grasp @ T_grasp_tool

def unproject(depth_buffer, view, projection):
    depth_buffer = np.asarray(depth_buffer, dtype=float)
    height, width = depth_buffer.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    clip = np.stack((2*(xx+.5)/width-1, 1-2*(yy+.5)/height,
                     2*depth_buffer-1, np.ones_like(depth_buffer)), axis=-1)
    inverse = np.linalg.inv(np.asarray(projection).reshape(4,4, order='F') @
                            np.asarray(view).reshape(4,4, order='F'))
    homogeneous = clip.reshape(-1,4) @ inverse.T
    return (homogeneous[:,:3]/homogeneous[:,3:4]).reshape(height,width,3)

def observed_proposal(points_world, yaw, table_z, workspace):
    cloud = np.asarray(points_world).reshape(-1,3)
    lo, hi = np.asarray(workspace[0]), np.asarray(workspace[1])
    selected = cloud[np.isfinite(cloud).all(axis=1) & (cloud>lo).all(axis=1) & (cloud<hi).all(axis=1)]
    if len(selected)<30:
        raise ValueError('empty_observation')
    bmin, bmax = np.quantile(selected, [.01,.99], axis=0)
    closing = np.array([np.cos(yaw), np.sin(yaw), 0.])
    approach = np.array([0.,0.,-1.])
    rotation = np.column_stack((approach, closing, np.cross(approach, closing)))
    width = float(np.ptp(selected @ closing))
    center = (bmin+bmax)/2
    center[2] = table_z + (bmax[2]-table_z)*.55
    depth = .02
    candidate = np.r_[1., width+.015, .02, depth, rotation.ravel(), center-approach*depth, -1.]
    return candidate, dict(point_count=len(selected), bbox_min=bmin.tolist(), bbox_max=bmax.tolist(),
                           estimated_object_width_m=width, contact_center_world=center.tolist(),
                           policy_inputs='observed depth point cloud, fixed table/workspace, fixed requested yaw; no body pose or instance/physics labels')

def angle_error(rotation_a, rotation_b):
    return float(np.arccos(np.clip((np.trace(np.asarray(rotation_a).T@np.asarray(rotation_b))-1)/2,-1,1)))
