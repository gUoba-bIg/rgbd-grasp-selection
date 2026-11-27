"""Observation-only conversion from frozen TinyRenderer camera to the P7 input grid.

No object, ground-truth mask, body state or segmentation ID is accepted.
Coordinates use the archived v52 pixel-centre convention, then the published
P7 RGB/depth interpolation, normalization and P5v23 corruption routine.
"""
import numpy as np

def camera_geometry(depth_buffer, view_matrix, projection_matrix):
    image=np.asarray(depth_buffer)
    if image.ndim!=2 or not np.isfinite(image).all():
        raise ValueError("Invalid depth buffer")
    height,width=image.shape
    view=np.asarray(view_matrix,dtype=np.float64).reshape(4,4,order="F")
    proj=np.asarray(projection_matrix,dtype=np.float64).reshape(4,4,order="F")
    if not np.isfinite(view).all() or not np.isfinite(proj).all() or not np.array_equal(proj[3],[0.,0.,-1.,0.]):
        raise ValueError("Registered OpenGL perspective required")
    if np.any(image<0) or np.any(image>1):
        raise ValueError("Invalid depth buffer interval")
    K=np.array([[proj[0,0]*width/2,0.,(1-proj[0,2])*width/2-.5],
                [0.,proj[1,1]*height/2,(1+proj[1,2])*height/2-.5],[0.,0.,1.]])
    T_world_camera=np.linalg.inv(view)@np.diag([1.,-1.,-1.,1.])
    z=proj[2,3]/(2*image.astype(np.float64)-1+proj[2,2])
    valid=(image<1)&(z>0)&np.isfinite(z)
    return K,T_world_camera,z,valid

def image_tensor(rgb_uint8,depth_m,valid,*,scene_key,frame,condition,quality_screen_epoch):
    from PIL import Image
    import torch
    from src.p5_quality_v23 import corrupt
    if condition not in ("clean","joint"):
        raise ValueError("Unsupported observation condition")
    if rgb_uint8.dtype!=np.uint8 or rgb_uint8.shape!=(*depth_m.shape,3) or valid.shape!=depth_m.shape:
        raise ValueError("RGB and depth must share a pixel grid")
    rgb=np.array(Image.fromarray(rgb_uint8).resize((640,480),Image.Resampling.BILINEAR),np.float32)/255.
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    z=np.where(valid,depth_m,0).astype(np.float32)
    depth=np.asarray(Image.fromarray(z).resize((640,480),Image.Resampling.NEAREST),dtype=np.float32)
    mask=np.isfinite(depth)&(depth>0)
    if not np.all(np.isfinite(rgb)):
        raise ValueError("Invalid RGB")
    inp=np.concatenate((rgb.transpose(2,0,1),np.where(mask,np.minimum(depth,2.)/2,0)[None],mask[None].astype(np.float32)),axis=0)
    changed,_unused_targets,description=corrupt(torch.from_numpy(inp),dict(scene=scene_key,frame=f'{frame:04d}'),quality_screen_epoch,condition)
    return changed.numpy(),description
