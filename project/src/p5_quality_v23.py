"""Observable-only quality model and paired early-fusion gate for P5 v23."""
import hashlib, math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.segmentation import PilotInstanceNet
from src.graspnet_frames import GraspNetFrames,instance_targets
SEED=20261023
CORE=('clean','noise5','missing15','shift3','joint')
GRADED=('noise1','noise3','missing5','missing30','shift1','shift5','block')

def corrupt(inputs,record,epoch,condition):
    assert inputs.device.type=='cpu' and inputs.shape[0]==5 and inputs.dtype==torch.float32
    assert condition in CORE+GRADED
    reference=inputs[3].numpy().copy()*2.;valid=inputs[4].numpy().astype(bool);h,w=valid.shape
    identity=f"P5v23|{SEED}|{epoch}|{record['scene']}|{record['frame']}|{condition}"
    seed=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16);rng=np.random.RandomState(seed)
    z=reference.copy();v=valid.copy()
    shift=int(condition[5:]) if condition.startswith('shift') else (3 if condition=='joint' else 0)
    if shift:
        z[:,shift:]=reference[:,:-shift];z[:,:shift]=0;v[:,shift:]=valid[:,:-shift];v[:,:shift]=False
    noise=float(condition[5:])/1000 if condition.startswith('noise') else (.005 if condition=='joint' else 0.)
    if noise:z=np.where(v,np.clip(z+rng.normal(0,noise,z.shape).astype(np.float32),0,2),0).astype(np.float32);v &= z>0
    rate=float(condition[7:])/100 if condition.startswith('missing') else (.15 if condition=='joint' else 0.)
    if rate:v &= rng.uniform(size=z.shape)>=rate
    block=condition in ('missing15','joint','block')
    rectangle=None
    if block:
        hh,ww=h//4,w//4;y=int(rng.randint(0,h-hh+1));x=int(rng.randint(0,w-ww+1));v[y:y+hh,x:x+ww]=False;rectangle=[y,x,hh,ww]
    z[~v]=0.;changed=inputs.clone();changed[3]=torch.from_numpy(z/2.);changed[4]=torch.from_numpy(v.astype(np.float32))
    residual=z-reference;supervised=valid&v
    boundary=np.zeros((h,w),bool)
    jump=(np.abs(reference[1:]-reference[:-1])>.02)&valid[1:]&valid[:-1];boundary[1:]|=jump;boundary[:-1]|=jump
    jump=(np.abs(reference[:,1:]-reference[:,:-1])>.02)&valid[:,1:]&valid[:,:-1];boundary[:,1:]|=jump;boundary[:,:-1]|=jump
    anomaly=supervised&boundary&(np.abs(residual)>.01)
    targets=dict(q_valid=torch.from_numpy(v.astype(np.float32)[None]),q_residual=torch.from_numpy(residual.astype(np.float32)[None]),q_supervised=torch.from_numpy(supervised.astype(np.float32)[None]),q_anomaly=torch.from_numpy(anomaly.astype(np.float32)[None]))
    meta=dict(condition=condition,seed=seed,shift_px=shift,noise_sigma_m=noise,missing_rate=rate,rectangle=rectangle,original_valid=int(valid.sum()),supervised=int(supervised.sum()))
    return changed,targets,meta

class Frames(GraspNetFrames):
    def __init__(self,*args,condition=None,**kwargs):super().__init__(*args,**kwargs);self.condition=condition
    def get(self,index,epoch=0):
        clean,t=super().get(index,epoch);t['centers']=instance_targets(t['labels'].numpy(),sigma=6.)['centers']
        condition=self.condition or CORE[(index+epoch)%len(CORE)]
        changed,q,meta=corrupt(clean,self.records[index],epoch,condition);t.update(q);return changed,t

class QualityNet(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(9,16,5,padding=2),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,3,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):
        z=x[:,3:4].float()*2;v=x[:,4:5].float();fraction=F.avg_pool2d(v,5,1,2);mean=F.avg_pool2d(z*v,5,1,2)/fraction.clamp_min(1e-6);second=F.avg_pool2d(z.square()*v,5,1,2)/fraction.clamp_min(1e-6)
        std=(second-mean.square()).clamp_min(0).sqrt();residual=((z-mean)/.01).clamp(-5,5);spread=(std/.01).clamp(0,5)
        gradient=F.max_pool2d(z,3,1,1)+F.max_pool2d(-z,3,1,1)
        features=torch.cat((x[:,:5].float(),residual,spread,fraction,(gradient/.02).clamp(0,5)),1)
        raw=self.layers(features);log_sigma=(raw[:,1:2].float()+math.log(.003)).clamp(math.log(.0001),math.log(.05))
        return dict(valid_logits=raw[:,0:1].float(),log_sigma=log_sigma,anomaly_logits=raw[:,2:3].float())

def quality_loss(outputs,t):
    mask=t['q_supervised'].float();den=mask.sum().clamp_min(1);r=t['q_residual'].float();ls=outputs['log_sigma']
    nll=((.5*(r/ls.exp()).square()+ls-math.log(.005))*mask).sum()/den
    valid=F.binary_cross_entropy_with_logits(outputs['valid_logits'],t['q_valid'].float())
    target=t['q_anomaly'].float();positive=target.sum();weight=((target.numel()-positive)/positive.clamp_min(1)).clamp(1,64)
    anomaly=F.binary_cross_entropy_with_logits(outputs['anomaly_logits'],target,pos_weight=weight)
    return .1*nll+valid+.5*anomaly,dict(q_nll=nll.detach(),q_valid_bce=valid.detach(),q_anomaly_bce=anomaly.detach())

class GatedModel(nn.Module):
    def __init__(self,quality,arm):
        super().__init__();assert arm in ('fixed','quality');self.backbone=PilotInstanceNet('rgbd');self.quality=quality;self.arm=arm;self.strength=0.;self.ablation='learned'
        for p in self.quality.parameters():p.requires_grad_(False)
    def forward(self,x):
        with torch.no_grad():q=self.quality(x)
        reliability=q['valid_logits'].sigmoid()*torch.exp(-q['log_sigma'].exp()/.005)*(1-.5*q['anomaly_logits'].sigmoid())
        strength=self.strength if self.arm=='quality' else 0.
        gate=1-strength*(1-reliability)
        if self.ablation=='unit':gate=torch.ones_like(gate)
        elif self.ablation=='mean':
            valid=x[:,4:5];mean=(gate*valid).sum((2,3),keepdim=True)/valid.sum((2,3),keepdim=True).clamp_min(1);gate=mean.expand_as(gate)
        elif self.ablation!='learned':raise ValueError('Unknown gate ablation')
        modified=torch.cat((x[:,:3],x[:,3:]*gate.to(x.dtype)),1)
        heads=self.backbone(modified);return dict(**heads,quality=q,effective_gate=gate)

def strength(step):return .5*min(1.,max(0.,(step-80)/160.))
