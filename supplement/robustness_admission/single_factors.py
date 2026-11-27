"""Single-component interventions using exactly the field draws of V1's joint transform."""
import hashlib
import numpy as np
from common_v1 import need
FACTORS=("depth_only","shift_only","missing_only","block_only")

def project(inputs,record,epoch,condition,native):
    import torch
    if condition in ("clean","joint"):
        return native(inputs,record,epoch,condition)
    need(condition in FACTORS,"unknown supplementary corruption")
    reference=inputs[3].numpy().copy()*2.
    valid=inputs[4].numpy().astype(bool)
    h,w=valid.shape
    joint,_,metadata=native(inputs,record,epoch,"joint")
    seed=metadata["seed"]
    need(metadata["shift_px"]==3 and metadata["noise_sigma_m"]==.005 and metadata["missing_rate"]==.15,"frozen V1 joint amplitudes")
    rng=np.random.RandomState(seed)
    noise=rng.normal(0,.005,reference.shape).astype(np.float32)
    keep=rng.uniform(size=reference.shape)>=.15
    hh,ww=h//4,w//4
    y=int(rng.randint(0,h-hh+1));x=int(rng.randint(0,w-ww+1))
    need(metadata["rectangle"]==[y,x,hh,ww],"V1 joint field identity")
    def render(parts):
        z,v=reference.copy(),valid.copy()
        if "shift_only" in parts:
            z[:,3:]=reference[:,:-3];z[:,:3]=0
            v[:,3:]=valid[:,:-3];v[:,:3]=False
        if "depth_only" in parts:
            z=np.where(v,np.clip(z+noise,0,2),0).astype(np.float32);v &= z>0
        if "missing_only" in parts:v &= keep
        if "block_only" in parts:v[y:y+hh,x:x+ww]=False
        z[~v]=0.
        ans=inputs.clone()
        ans[3]=torch.from_numpy(z/2.);ans[4]=torch.from_numpy(v.astype(np.float32))
        return ans
    need(torch.equal(joint,render(FACTORS)),"joint not reproduced byte-for-byte")
    new=render((condition,))
    need(torch.equal(new[:3],inputs[:3]),"RGB changed")
    def digest(arr):return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    receipt=dict(condition=condition,seed=seed,joint_recomposition_exact=True,field_provenance="original joint random draws, same epoch/scene/frame",
        shift_px=3 if condition=="shift_only" else 0,noise_sigma_m=.005 if condition=="depth_only" else 0.,
        missing_rate=.15 if condition=="missing_only" else 0.,rectangle=[y,x,hh,ww] if condition=="block_only" else None,
        joint_rectangle=[y,x,hh,ww],source_rgb_sha256=digest(inputs[:3].numpy()),source_depth_sha256=digest(reference),
        iid_mask_sha256=digest(keep),noise_field_sha256=digest(noise),joint_input_sha256=digest(joint.numpy()),input_sha256=digest(new.numpy()))
    # The V1 inference entry explicitly ignores the training-target return.
    return new,{},receipt
