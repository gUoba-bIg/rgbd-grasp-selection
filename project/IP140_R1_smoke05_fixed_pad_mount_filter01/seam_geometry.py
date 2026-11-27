"""Frozen IP140-R1 box faces. All coordinates are metres in the owning link frame."""
from math import hypot, isfinite
import numpy as np

EPS = 1e-9
FORCE_EPS_N = 1e-9
IMPULSE_EPS_NS = 1e-12
PALM = 'ip140_palm'
PADS = ('ip140_left_pad', 'ip140_right_pad')
BONES = ('ip140_left_bone', 'ip140_right_bone')
PAD_MOUNTS = tuple(zip(BONES, PADS))
ACTIVE = (PADS,) + tuple((PALM, name) for names in zip(PADS, BONES) for name in names)
POSE_LINKS = (PALM, *BONES, *PADS)

# Bounds: local x/y/z. A zero-thickness interval identifies the registered plane.
def face_boxes(pair):
    def face(x,y,z=(-.010,.010)): return (x,y,z)
    if pair == PADS: return (face((0.,.060),(0.,0.)),)*2
    if pair in PAD_MOUNTS:
        y = -.002 if pair == PAD_MOUNTS[0] else .002
        return (face((0.,.060),(y,y)),)*2
    if pair not in ACTIVE: raise ValueError('unregistered face')
    part=pair[1]
    y=(-.002,0.) if part==PADS[0] else (0.,.002) if part==PADS[1] else (
       (-.010,-.002) if part==BONES[0] else (.002,.010))
    return face((-.060,-.060),(-.080,.080)),face((0.,0.),y)

def rigid(t,q):
    x,y,z,w = q
    s = 2./(x*x+y*y+z*z+w*w)
    return np.array(((1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w),t[0]),
                     (s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w),t[1]),
                     (s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y),t[2]),
                     (0.,0.,0.,1.)))
def inverse(T):
    inv=np.eye(4); inv[:3,:3]=T[:3,:3].T; inv[:3,3]=-inv[:3,:3]@T[:3,3]
    return inv

def current_frames(p,body,links,cid):
    names=(*POSE_LINKS,'tool0') if 'tool0' in links else POSE_LINKS
    rows=p.getLinkStates(body,[links[n] for n in names],computeForwardKinematics=1,physicsClientId=cid)
    return {n:rigid(row[0],row[1])@inverse(rigid(row[2],row[3])) for n,row in zip(names,rows)}

def local_xyz(W_T_link,point):
    return W_T_link[:3,:3].T @ (np.asarray(point)-W_T_link[:3,3])

def check_row(pair,ids,row,world,body,dt):
    issues=[]
    if len(row)!=14: return dict(passed=False,violations=['contact_record_shape'])
    # Contact flag, body and link IDs, positions/directions and all signed scalar forces.
    flat=(*row[:5],*row[5],*row[6],*row[7],*row[8:11],*row[11],row[12],*row[13])
    if not all(isfinite(v) for v in flat): issues.append('nonfinite_field')
    if (row[1],row[2])!=(body,body) or (row[3],row[4]) not in (ids,ids[::-1]):
        issues.append('pair_identity')
    if issues: return dict(passed=False,violations=issues)
    ij=(5,6) if (row[3],row[4])==ids else (6,5)
    local=[]; face_error=[]
    for name,k,box in zip(pair,ij,face_boxes(pair)):
        pt=local_xyz(world[name],row[k]); local.append(pt.tolist())
        e=max(max(lo-v,v-hi,0.) for v,(lo,hi) in zip(pt,box));face_error.append(float(e))
        if not np.isfinite(pt).all() or e>EPS: issues.append('outside_registered_face:'+name)
    delta=[row[5][k]-row[6][k]-row[7][k]*row[8] for k in range(3)]
    residual=hypot(*delta); unit_error=abs(hypot(*row[7])-1.)
    if unit_error>EPS:issues.append('nonunit_normal')
    if residual>EPS:issues.append('witness_residual')
    if row[8]<-EPS:issues.append('penetration_below_numeric_band')
    signed=(row[9],row[10],row[12])
    if any(abs(f)>FORCE_EPS_N for f in signed):issues.append('reported_force')
    impulse=[float(dt)*sum(f*v[k] for f,v in zip(signed,(row[7],row[11],row[13]))) for k in range(3)]
    magnitude=hypot(*impulse)
    if not isfinite(magnitude) or magnitude>IMPULSE_EPS_NS:issues.append('vector_impulse')
    return dict(passed=not issues,violations=issues,witnesses_local_m=local,face_outside_m=face_error,
                normal_unit_error=unit_error,witness_residual_m=residual,
                signed_force_components_N=signed,vector_impulse_world_Ns=impulse,impulse_norm_Ns=magnitude)
