"""No physics: check all preserved P3 candidates and camera/depth/TCP conventions."""
import json
import math
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pybullet as p
from scripts.sim_v52_common import ROOT,OUT,read,write,identity,now
from scripts.sim_v52_math import grasp_to_tool,transform,unproject,observed_proposal

def run():
    target=OUT/'coordinate_checks.json'
    if target.exists():raise FileExistsError('No overwrite')
    maxima=dict(roundtrip_m=0.,contact_center_m=0.,axis_error=0.,rotation_orthogonality=0.)
    count=0;camera_files={}
    for path in sorted((OUT/'audit/policy_inputs').glob('scene_*.json')):
        if path.name.endswith('.receipt.json'):
            continue
        data=read(path);scene=data['case']['scene'];frame=data['case']['frame']
        directory=ROOT/f'data/raw/prerequisites_v26/scenes/scene_{scene:04d}/realsense'
        camera_to_table=np.load(directory/'cam0_wrt_table.npy')
        camera_poses=np.load(directory/'camera_poses.npy')
        T=camera_to_table@camera_poses[frame]
        for name in ('cam0_wrt_table.npy','camera_poses.npy'):
            camera_files[str((directory/name).relative_to(ROOT))]=identity(directory/name)
        for row in data['candidate_rows']:
            row=np.asarray(row);R=row[4:13].reshape(3,3);pad=.19
            goal=grasp_to_tool(T,row,pad)
            expected=T@np.r_[row[13:16]+R[:,0]*row[3],1.]
            actual=goal@np.array([0,0,pad,1.])
            maxima['contact_center_m']=max(maxima['contact_center_m'],float(np.linalg.norm(actual-expected)))
            maxima['axis_error']=max(maxima['axis_error'],float(np.linalg.norm(goal[:3,2]-T[:3,:3]@R[:,0])),float(np.linalg.norm(goal[:3,1]-T[:3,:3]@R[:,1])))
            roundtrip=np.linalg.inv(T)@(T@np.r_[row[13:16],1.])
            maxima['roundtrip_m']=max(maxima['roundtrip_m'],float(np.linalg.norm(roundtrip[:3]-row[13:16])))
            maxima['rotation_orthogonality']=max(maxima['rotation_orthogonality'],float(np.max(np.abs(R.T@R-np.eye(3)))))
            assert np.linalg.det(R)>.999
            count+=1
    assert count==1000
    assert maxima['contact_center_m']<1e-5 and maxima['axis_error']<1e-6 and maxima['roundtrip_m']<1e-6
    assert maxima['rotation_orthogonality']<1e-4
    width,height=320,240
    view=p.computeViewMatrix([.53,0,.9],[.53,0,0],[0,1,0])
    proj=p.computeProjectionMatrixFOV(50,width/height,.05,2.)
    # An analytically flat table plane at camera optical depth .9m.
    buffer=(2.-.05*2./.9)/(2.-.05)
    cloud=unproject(np.full((height,width),buffer),view,proj)
    max_table_error=float(np.max(np.abs(cloud[:,:,2])))
    assert max_table_error<1e-6
    # Observable point cloud with a fixed cuboid; no body or label API involved.
    x,y=np.meshgrid(np.linspace(.46,.54,21),np.linspace(-.025,.025,21))
    surface=np.stack((x,y,np.full_like(x,.06)),axis=-1)
    proposal,obs=observed_proposal(surface,0,0,[[.35,-.18,.006],[.72,.18,.16]])
    assert abs(obs['estimated_object_width_m']-.08)<1e-8 and proposal[1]>.08
    data=dict(at=now(),passed=True,preserved_candidates_checked=count,max_errors=maxima,
              camera_plane_unprojection_max_error_m=max_table_error,source_cameras=camera_files,
              physical_steps=0,grasp_episodes=0,new_neural_forwards=0,
              boundary='Mathematical/static adapter validation only; no P3 candidate has been physically executed',
              source={str(pth.relative_to(ROOT)):identity(pth) for pth in [Path(__file__),ROOT/'scripts/sim_v52_math.py']})
    write(target,data,once=True)
    print(json.dumps({'passed':True,'candidates':count,'max_errors':maxima,'table_unproject_m':max_table_error}))

if __name__=='__main__':run()
