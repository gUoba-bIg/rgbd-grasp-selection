"""v38 genuinely separate RGB and depth/validity feature paths."""
import torch
from torch import nn
from torch.nn import functional as F
from src.segmentation import PilotInstanceNet
from src.p5_quality_v23 import QualityNet


class ModalFusionV38(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ('fixed', 'quality'):
            raise ValueError('Unknown paired arm')
        self.arm = arm
        self.ablation = 'learned'
        self.record_diagnostics = False
        self.backbone = PilotInstanceNet('rgb')
        self.depth = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv2d(48, 48, 1))
        nn.init.zeros_(self.depth[-1].weight)
        nn.init.zeros_(self.depth[-1].bias)
        self.quality = QualityNet()
        self.backbone.requires_grad_(False)
        self.quality.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.quality.eval()
        return self

    @staticmethod
    def fuse(rgb_features, depth_features, reliability):
        return rgb_features + reliability * depth_features

    def forward(self, inputs):
        # Only the separately trained depth path sees depth and validity.
        # Freeze RGB feature extraction, but NOT the post-fusion head's graph:
        # its derivative is necessary to train the newly introduced depth path.
        with torch.no_grad():
            hidden = self.backbone.stem(inputs[:, :3])
            features = []
            for stage in self.backbone.stages:
                hidden = stage(hidden)
                features.append(hidden)
            rgb = self.backbone.laterals[-1](features[-1])
            for index in range(2, -1, -1):
                rgb = self.backbone.laterals[index](features[index]) + F.interpolate(
                    rgb, size=features[index].shape[-2:], mode='nearest')
            q = self.quality(inputs)
            reliability = (q['valid_logits'].sigmoid()
                           * torch.exp(-q['log_sigma'].exp() / .005)
                           * (1 - .5 * q['anomaly_logits'].sigmoid()))
        if self.arm == 'fixed' or self.ablation == 'unit':
            reliability = torch.ones_like(reliability)
        elif self.ablation == 'zero':
            reliability = torch.zeros_like(reliability)
        elif self.ablation == 'mean':
            valid = inputs[:, 4:5]
            mean = (reliability * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp_min(1)
            reliability = mean.expand_as(reliability)
        elif self.ablation != 'learned':
            raise ValueError('Unknown reliability ablation')
        depth = self.depth(inputs[:, 3:5])
        if depth.shape[-2:] != rgb.shape[-2:]:
            raise RuntimeError('Independent paths have incompatible spatial resolution')
        gate = F.interpolate(reliability, size=rgb.shape[-2:], mode='area').to(rgb.dtype)
        fused = self.fuse(rgb, depth, gate)
        hidden = self.backbone.head(F.interpolate(self.backbone.refine(fused), size=inputs.shape[-2:], mode='nearest'))
        result = dict(foreground=self.backbone.foreground(hidden), centers=self.backbone.centers(hidden),
                      offsets=self.backbone.offsets(hidden), quality=q, effective_gate=reliability)
        if self.record_diagnostics:
            correction = (gate * depth).detach().float()
            result['fusion_diagnostics'] = dict(
                correction_to_rgb_norm=(correction.norm()/rgb.detach().float().norm().clamp_min(1e-12)),
                depth_norm=depth.detach().float().norm(),
                gate_quantiles=torch.quantile(gate.detach().float().flatten(), torch.tensor([0.,.1,.5,.9,1.],device=gate.device)))
        return result
