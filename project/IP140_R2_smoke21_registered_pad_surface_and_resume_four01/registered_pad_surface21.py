"""Static R2 load-bearing surfaces derived only from registered pad collision geometry."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISTANCE_TOL_M = 1e-9
NORMAL_VECTOR_TOL = 1e-9

@lru_cache(maxsize=2)
def pad_geometry(side):
    if side not in ('left', 'right'):
        raise ValueError('unregistered pad side')
    manifest = json.loads((HERE/'model/manifest.json').read_bytes())
    model = HERE/'model'/manifest['model']
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest['model_sha256']:
        raise ValueError('pad URDF identity differs')
    name = 'ip140_' + side + '_pad'
    component = manifest['components'][name]
    collisions = ET.parse(model).findall(f"./link[@name='{name}']/collision")
    if len(collisions) != 1 or component['shape'] != 'GEOM_BOX':
        raise ValueError('registered pad box required')
    dims = tuple(map(float,collisions[0].find('geometry/box').attrib['size'].split()))
    origin = tuple(map(float,collisions[0].find('origin').attrib['xyz'].split()))
    rpy = tuple(map(float,collisions[0].find('origin').attrib['rpy'].split()))
    if (list(dims) != component['box_size_m'] or list(origin) != component['center_in_link_m']
            or rpy != (0.,0.,0.) or component['orientation_xyzw'] != [0,0,0,1]
            or dims != (.060,.002,.020) or component['collisionMargin_m'] != .0005):
        raise ValueError('registered pad dimensions, frame or radius differs')
    sign = 1 if side == 'left' else -1
    lo = tuple(c-s/2 for c,s in zip(origin,dims))
    hi = tuple(c+s/2 for c,s in zip(origin,dims))
    y_inner = hi[1] if sign==1 else lo[1]
    if y_inner != 0 or sign*origin[1] >= 0:
        raise ValueError('pad inner plane / handedness differs')
    return lo,hi,y_inner,sign,component['collisionMargin_m']

def registered_surface(side, point_pad_m, normal_pad_to_target_pad):
    """Classify a measured point without projecting it onto a legal surface.

    The core edge axis is parallel to z. x increases toward the tip; side sign
    identifies the inner direction. The exposed cylinder ends at the two z
    core planes. Its tip-facing tangent is excluded. The flat/arc tangent is
    assigned to the flat face. A zero radius or wrong-side witness cannot pass.
    """
    lo,hi,inner,sign,r = pad_geometry(side)
    x,y,z = point_pad_m
    nx,ny,nz = normal_pad_to_target_pad
    if not all(math.isfinite(v) for v in (x,y,z,nx,ny,nz)):
        raise ValueError('invalid local witness or normal')
    x0=hi[0]-r
    y0=inner-sign*r
    dx,dy=x-x0,y-y0
    rho=math.hypot(dx,dy)
    axis_z_lo,axis_z_hi=lo[2]+r,hi[2]-r
    tol=DISTANCE_TOL_M
    axial=axis_z_lo <= z <= axis_z_hi
    flat=abs(y-inner)<=tol and lo[0]+r-tol <= x <= x0+tol and axial
    # Reject the x end face. It is tangent to the cylinder but has no inner-facing component.
    arc_sector=dx>0 and sign*dy>0 and axial
    radial_residual=abs(rho-r)
    normal=[dx/rho,dy/rho,0.] if rho>0 else None
    error=math.dist((nx,ny,nz),normal) if normal else None
    flat_normal=ny*sign>0
    arc=arc_sector and radial_residual<=tol
    valid_unit=abs(math.sqrt(nx*nx+ny*ny+nz*nz)-1)<=NORMAL_VECTOR_TOL
    analytic_normal_ok=error is not None and error<=NORMAL_VECTOR_TOL
    surface_type='flat_inner_face' if flat else 'inner_tip_quarter_cylinder' if arc else None
    accepted=valid_unit and ((flat and flat_normal) or (arc and analytic_normal_ok))
    return dict(accepted=bool(accepted),surface_type=surface_type,
        geometric_surface_member=bool(flat or arc),unit_normal=bool(valid_unit),
        analytic_axis_xy_m=[x0,y0],registered_radius_m=r,axis_z_limits_m=[axis_z_lo,axis_z_hi],
        in_quarter_sector=bool(arc_sector), radial_length_m=rho,radial_residual_m=radial_residual,
        analytic_arc_outward_normal=normal,normal_vector_error=error,
        distance_tolerance_m=tol,normal_vector_tolerance=NORMAL_VECTOR_TOL,
        flat_plane_residual_m=abs(y-inner),inner_normal_alignment=ny*sign)
