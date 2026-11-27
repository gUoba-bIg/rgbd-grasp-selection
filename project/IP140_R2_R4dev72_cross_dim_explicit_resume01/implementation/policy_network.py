"""Observation-only P3 and fixed P5 evaluation. No engine imports or state access."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
P3='experiments/candidate_v30/epoch_18_final.pt'
P5='experiments/p5_fusion_v38/fixed/final.pt'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(view,projection,rgb,buffer):
    """OpenGL pixel centres to positive-z optical frame, RGB x right/y down."""
    h,w=buffer.shape
    if rgb.shape!=(h,w,3) or rgb.dtype!=np.uint8 or not np.isfinite(buffer).all() or np.min(buffer)<0 or np.max(buffer)>1.0000003:raise ValueError('invalid image/GL depth')
    proj=np.asarray(projection).reshape(4,4,order='F')
    if not np.allclose(proj[3],[0,0,-1,0],rtol=0,atol=1e-9):raise ValueError('nonperspective camera')
    cv_to_world=np.linalg.inv(np.asarray(view).reshape(4,4,order='F'))@np.diag([1.,-1.,-1.,1.])
    k=np.array([[w*proj[0,0]/2,0,(w-1-w*proj[0,2])/2],[0,h*proj[1,1]/2,(h-1+h*proj[1,2])/2],[0,0,1.]])
    valid=(buffer<1)&(buffer>=0)
    z=np.zeros(buffer.shape,np.float64)
    z[valid]=proj[2,3]/(2*buffer[valid].astype(float)-1+proj[2,2])
    if np.any(z[valid]<=0) or not np.isfinite(z).all():raise ValueError('depth conversion')
    colour=np.asarray(Image.fromarray(rgb).resize((640,480),Image.Resampling.BILINEAR),np.float32)/255.
    colour=(colour-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
    z=np.asarray(Image.fromarray(z.astype(np.float32)).resize((640,480),Image.Resampling.NEAREST))
    inp=np.concatenate((colour.transpose(2,0,1),np.minimum(z,2.)[None]/2,(z>0)[None])).astype(np.float32)
    return inp,k,np.array([w,h]),cv_to_world

def generate(observation,folder,scene,frame,condition):
    import torch
    from src.grasp_prerequisite_data_v27 import SOURCE
    from graspnet import GraspNet,pred_decode
    from src.modal_fusion_v38 import ModalFusionV38
    from src.p5_quality_v23 import corrupt
    from src.risk_observations_v31 import points
    from scripts.extract_mask_scores_v17 import decode_with_scores
    cfg=json.loads((ROOT/'configs/risk_calibration_v31.json').read_bytes())
    scope=json.loads((ROOT/'configs/p56_joint_v51r2.json').read_bytes())['protected']
    hashes={s:sha(ROOT/s) for s in [P3,P5]}
    for s,d in hashes.items():
        if scope[s]['sha256']!=d:raise ValueError('checkpoint sha mismatch '+s)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(observation,allow_pickle=False) as o:
        if set(o.files)!=set('rgb depth_buffer view projection'.split()):raise ValueError('policy consumes only four raw observation arrays')
        X,K,size,T=prepare(o['view'],o['projection'],o['rgb'],o['depth_buffer'])
    X,_,description=corrupt(torch.from_numpy(X),dict(scene=scene,frame=f'{frame:04d}'),cfg['quality_screen_epoch'],condition)
    X=X.numpy()
    cloud,source_pixels=points(X[3]*2,X[4].astype(bool),K,size)
    if len(cloud)<20000:raise ValueError('frozen P3 needs >=20000 observed points')
    rng=np.random.RandomState(cfg['seed']+scene*1000+frame)
    samples=rng.choice(len(cloud),20000,replace=False)
    seg=ModalFusionV38('fixed').cuda().eval()
    seg.load_state_dict(torch.load(ROOT/P5,map_location='cpu',weights_only=False)['model'],strict=True)
    grasp=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04],is_training=False).cuda().eval()
    grasp.load_state_dict(torch.load(ROOT/P3,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    start=time.perf_counter()
    with torch.inference_mode(): h=seg(torch.from_numpy(X)[None].cuda())
    heads=dict(foreground_probability=h['foreground'][0,0].sigmoid().cpu().numpy(),center_probability=h['centers'][0,0].sigmoid().cpu().numpy(),offsets=h['offsets'][0].cpu().numpy())
    instances,unused,scores,_=decode_with_scores(heads,**cfg['decoder'])
    torch.cuda.synchronize();seg_seconds=time.perf_counter()-start
    start=time.perf_counter()
    with torch.inference_mode():
        e=grasp({'point_clouds':torch.from_numpy(cloud[samples])[None].cuda()});cand=pred_decode(e)[0].cpu().numpy().copy()
        where=torch.nonzero(torch.argmax(e['objectness_score'][0].float(),0)==1).reshape(-1)
        idx=e['sa1_inds'][0].long()[e['sa2_inds'][0].long()][where].cpu().numpy()
    torch.cuda.synchronize();grasp_seconds=time.perf_counter()-start
    pixels=source_pixels[samples[idx]]
    if cand.shape!=(len(pixels),17) or not np.isfinite(cand).all():raise ValueError('P3 return')
    with (folder/'network.npz').open('xb') as out:
        np.savez_compressed(out,candidates=cand,pixels=pixels,depth=X[3]*2,valid=X[4].astype(bool),instances=instances,foreground_probability=heads['foreground_probability'],K=K,original_size=size,world_T_camera=T)
    with (folder/'network.json').open('x',encoding='utf8') as out:
        json.dump(dict(observation_sha256=sha(observation),weights=hashes,networks_returned=dict(P3=1,P5_fixed=1),torch=torch.__version__,native_grasps=len(cand),instance_count=int(instances.max()),sampled_count=20000,visible_cloud_count=len(cloud),condition=condition,corruption=description,segmentation_sec=seg_seconds,candidate_generation_sec=grasp_seconds,source_sha256=sha(__file__),network_output_sha256=sha(folder/'network.npz'),policy_input_fields=['rgb','depth_buffer','view','projection'],pose_compensation_m=0.,physics_steps=0),out,indent=2,allow_nan=False)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('observation',type=Path);q.add_argument('folder',type=Path);q.add_argument('scene',type=int);q.add_argument('frame',type=int);q.add_argument('condition',choices=['clean','joint'])
    a=q.parse_args();generate(a.observation,a.folder,a.scene,a.frame,a.condition)

