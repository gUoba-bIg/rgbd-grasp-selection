"""Deterministically reconstruct the input array fingerprints, without additional inference."""
import sys
import numpy as np
from common import *
sys.path[:0]=[str(R4/'implementation'),str(ROOT)]
from policy_network import prepare
from src.p5_quality_v23 import corrupt
from src.risk_observations_v31 import points
import torch
def digest(a):
    a=np.ascontiguousarray(a)
    return dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(),shape=list(a.shape),dtype=a.dtype.str)
def main():
    d=read(HERE/'preregistration/design.json');s={x['base_scene']:x for x in d['scenes']}
    cfg=read(ROOT/'configs/risk_calibration_v31.json');torch.set_num_threads(1)
    complete=HERE/'audit/tensor_inputs';complete.mkdir(exist_ok=False)
    for u in d['units']:
        sc=s[u['scene']];mode=u['condition'];source=HERE/'scenes'/u['scene']/'observed_rgbd.npz'
        with np.load(source,allow_pickle=False) as o:X,K,size,T=prepare(o['view'],o['projection'],o['rgb'],o['depth_buffer'])
        X,_,recipe=corrupt(torch.from_numpy(X),dict(scene=sc['scene_index'],frame='0000'),cfg['quality_screen_epoch'],mode);X=X.numpy()
        cloud,pixels=points(X[3]*2,X[4].astype(bool),K,size)
        rng=np.random.RandomState(sc['sampling_seed']);idx=rng.choice(len(cloud),20000,replace=False)
        require(np.isfinite(X).all() and np.isfinite(cloud[idx]).all() and (recipe.get('seed',sc['condition_seed'][mode])==sc['condition_seed'][mode]),'model input integrity')
        put(complete/(u['unit_id']+'.json'),dict(unit=u['unit_id'],mode=mode,observation_sha256=sha(source),fingerprints=dict(P5=digest(X[None]),P3=digest(cloud[idx][None])),source='Independent deterministic CPU reconstruction using frozen encoder, corruption, camera point projection and seeded sampler. No second model forward.',depth=digest(X[3]*2),valid=digest(X[4].astype(bool)),K=digest(K),world_T_camera=digest(T),native_size=digest(size),candidate_pixel_membership_reference_sha256=digest(pixels[idx])['sha256'],noise=recipe,seed=sc['sampling_seed']))
    put(HERE/'audit/tensor_inputs_complete.json',dict(count=len(d['units']),utc=now(),network_forwards=0))
if __name__=='__main__':main()
